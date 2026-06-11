# DocPilot 本地引擎复杂表格还原升级设计（RapidTable / SLANet-plus 集成）

- 责任人：（待填）
- 更新日期：2026-06-08
- 状态：设计待评审
- 适用范围：`POST /api/v1/parse` 中 `parser_engine=deepdoc`（本地 ONNX 引擎）的 **表格结构识别** 子模块
- 约束基线：纯本地 CPU / ONNX，不引入 GPU、不引入 torch/paddle 等重依赖，完全离线可用

## 1. 背景与目标

### 1.1 背景

DocPilot Standalone 已是成熟的多引擎文档解析服务，PDF 引擎包含本地 `deepdoc`（ONNX 流水线）、远程 `paddleocr_vl` / `mineru`、本地 `markitdown`、`plain` 等（详见 `docs/PARSER_ENGINE_STRATEGY.md`）。

其中本地默认引擎 `deepdoc` 的**复杂表格还原**是已知短板：合并单元格、无线框表、跨页表、密集表的结构还原质量有限。由于业务（合同 / 报表 / 政务 / 技术报告）中表格高频出现，且表格质量直接影响结构化解析和 chunk 可用性，提升本地引擎的表格还原质量是高价值优化点。

远程引擎（`paddleocr_vl` / `mineru`）的表格能力较强，但属于远程 API；内网 / 离线场景只能使用本地 `deepdoc`。因此本设计聚焦：**在纯本地 CPU/ONNX 约束下，提升 `deepdoc` 引擎的复杂表格还原质量**。

### 1.2 目标

1. 显著提升 `deepdoc` 引擎在复杂表格（合并单元格、无线框表、跨页表）上的结构还原质量，以公开基准 TEDS 衡量。
2. 全程纯本地 CPU / ONNX，离线可用，与现有 vision 层架构一致。
3. 改动局部、可灰度、可随时回退到现有实现，默认行为不变。

### 1.3 非目标（YAGNI）

- 不引入 GPU / VLM / torch 路径（如 UniTable、PaddleOCR-VL 本地版）。
- 不改造远程引擎（`paddleocr_vl` / `mineru` / `tcadp` / `docling`）。
- 不做引擎自动路由、质量择优、多引擎 fallback 框架（可后续单独立项）。
- 不提升公式 / 印章 / 扫描 OCR / 阅读顺序等其他元素（本次仅表格）。

## 2. 现状分析

### 2.1 当前表格解析流程

本地 `deepdoc` 引擎的表格处理是当前 DeepDoc 经典的「检测 → 方向校正 → TSR 结构识别 → 几何规则拼装 → 生成 HTML」流水线：

| 步骤 | 代码位置 | 说明 |
|---|---|---|
| 表格区域检测 | layout 识别，`type=="table"` | 版面模型给出表格框 |
| 裁图 + 方向校正 | `pdf_parser.py:_evaluate_table_orientation`、`_table_transformer_job:337-429` | 裁出表格图、评估并校正旋转角度 |
| TSR 结构识别 | `pdf_parser.py:436` `self.tbl_det(imgs)` | 调 `TableStructureRecognizer`（`tsr.onnx`） |
| 旋转表重新 OCR | `pdf_parser.py:_ocr_rotated_tables:515` | 对旋转后的表重新 OCR |
| 给文本框打标签 | `pdf_parser.py:463-513` | 用 TSR 组件给 `self.boxes` 打 R/H/C/SP 标签 |
| 跨页表合并 | `pdf_parser.py:_extract_table_figure:1448`，`nearest:1527` | 在 box 层合并同一逻辑表 |
| 生成 HTML | `pdf_parser.py:1676` `self.tbl_det.construct_table(...)` | 几何规则拼单元格 + 输出 HTML |

TSR 模型采用 Microsoft Table Transformer（TATR）标签体系（`table_structure_recognizer.py:30-37`）：`table / table column / table row / table column header / table projected row header / table spanning cell`。单元格归属、行列聚类、rowspan/colspan 全部由 `construct_table`（`table_structure_recognizer.py:140`）和 `__cal_spans`（`484-563`）中的几何启发式规则推断。

### 2.2 短板根因

1. **TATR 模型偏弱**：对无线框表、密集表、复杂合并单元格的结构预测能力有限。在 PubTabNet / FinTabNet / SynthTabNet 上，TATR 弱于 SLANet 系与 UniTable，尤其在大表 / 复杂表上明显掉队。
2. **拼装依赖几何启发式**：单元格归属与 span 合并基于 bbox 中点重叠推断（`__cal_spans:484-563`），密集 / 不规整表容易串行串列。
3. **跨页表**：依赖 X 坐标排序硬拼，续表表头重复、跨页合并是硬伤。
4. **强依赖上游**：版面框检测错或 OCR 漏字会直接导致表格还原失败。

## 3. 技术选型

### 3.1 选型结论

在「纯本地 CPU / ONNX」约束下，选用 **RapidTable 封装的 SLANet-plus（ONNX）模型 + 有线/无线表分类** 替换当前 TATR 表格识别路径。

### 3.2 候选对比

| 方案 | 推理后端 | 公开基准表格质量 | 是否满足约束 | 结论 |
|---|---|---|---|---|
| Table Transformer（TATR，现状） | ONNX/CPU | 较弱（基准里最低档） | 满足 | 现状，待替换 |
| SLANet-plus（RapidTable） | **ONNX/CPU** | OmniDocBench 表格 TEDS ≈ 82.5（RapidTable 管线） | **满足** | **选用** |
| UniTable | PyTorch/**GPU** | 最高（PubTabNet TEDS ≈ 96.5） | 不满足（需 torch+GPU） | 排除 |
| VLM 类（PaddleOCR-VL / MinerU2.5 等） | **GPU/VLM** | 最高（TEDS ≈ 0.92–0.95） | 不满足（需 GPU/VLM） | 排除 |

### 3.3 选型依据

- **SLANet-plus 走 onnxruntime、CPU 可跑**（RapidTable 基础安装自带），与现有全 ONNX 的 vision 层完全契合；UniTable 需 PyTorch + GPU，被约束排除。
- RapidTable 姊妹库 TableStructureRec **内置「有线/无线表」分类模型**先分流再识别，CPU 约 1–7 秒/表，正好对症「无线框表」短板。
- 公开基准上，从 TATR 换到 SLANet 系会有确定性的 TEDS 提升；纯 CPU/ONNX 的现实天花板 ≈ RapidTable 的 OmniDocBench 表格 TEDS ≈ 82.5。

参考资料见第 12 节。

## 4. 方案设计

### 4.1 核心思想

把「表格区域 → HTML」这一步抽象成**可插拔引擎**，RapidTable 作为新引擎与现有 TATR 并列，由环境变量切换，默认回退 TATR、可灰度上线。

### 4.2 复用 vs 新增

**完全复用、不改动：**
- 表格区域检测（layout `type=="table"`）
- 表格裁图 + 方向自动校正（`_evaluate_table_orientation`、`_table_transformer_job` 的预处理部分）
- 表格区域内的 OCR 文本框（`self.boxes`，已带 `text` + 坐标）
- 跨页「逻辑表」归组（`_extract_table_figure` 的 `nearest` / 合并逻辑）
- 表格产出链（`tbls` → `results_to_markdown`）

**新增 / 改造：**
1. **新增 `deepdoc/vision/rapid_table_recognizer.py`**：封装 RapidTable（SLANet-plus ONNX + 有线/无线分类）。
   - 接口：`__call__(table_image, ocr_boxes) -> html`，把已有 OCR 结果喂入，**不让 RapidTable 内部再跑 OCR**（省一次推理，复用已调优的中文 OCR）。
   - 模型经 `common/model_store.py` 加载，与 `Recognizer` 基类的加载范式保持一致。
2. **表格产出处按 `DEEPDOC_TABLE_ENGINE` 分流**：
   - `tatr`（默认）：现有 `tbl_det` 打标签 + `construct_table` 几何拼装，**代码路径完全不变**。
   - `rapidtable`：跳过几何拼装，对每个表格区域（方向校正后的图 + 区域内 OCR boxes）调新引擎输出 HTML。
3. **跨页表 HTML 缝合**：RapidTable 为单表模型，在「逻辑表」归组后，跨页部分按**表头对齐缝合 `<tr>`**（在 HTML 层缝合，而非几何拼 box）。

### 4.3 数据流（rapidtable 引擎）

```
page_layout（table 框）
  → 裁图 + 方向校正                 [复用现有]
  → 收集该区域 OCR boxes            [复用 self.boxes]
  → RapidTableRecognizer(img, ocr_boxes) → 单表 HTML + 单元格框
  → 跨页「逻辑表」按表头对齐缝合 HTML
  → (img, html) 进入 tbls → results_to_markdown   [复用现有产出链]
```

### 4.4 精确集成点

- `pdf_parser.py:_table_transformer_job`：该方法当前将「裁图 + 方向校正 + TSR 推理 + 打 R/H/C/SP 标签」耦合在一起。需做**轻量拆分**，使「裁图 + 方向校正 + 收集区域 OCR boxes」成为引擎无关的预处理；`rapidtable` 模式复用预处理、跳过 TSR 打标签，`tatr` 模式行为完全不变。
- `pdf_parser.py:_extract_table_figure:1676`（`construct_table(...)` 调用处）：按引擎分流，`rapidtable` 走新路径输出 HTML。

> 注：保持 `_extract_table_figure` 的对外返回结构（`tbls` 的 `(img, html)` 形态）不变，确保下游 `results_to_markdown` 无感。

### 4.5 跨页表缝合策略

1. 沿用现有「逻辑表」归组，识别跨页同属一张表的分片。
2. 每个分片单独经 RapidTable 得到 HTML。
3. 缝合：以首片的表头行（`<thead>` / 首个 `<tr>`）为基准，后续分片去除重复表头，按列对齐追加 `<tr>`。
4. 列数不一致或表头无法对齐时，**降级为分片各自独立输出**（不强行缝合产生错乱），并记录日志。

## 5. 模型分发（`common/model_store.py`）

- 在 `MODEL_GROUP_FILES`（`model_store.py:12-31`）新增一个 group，建议名 `table_v2`：
  - SLANet-plus 结构识别 ONNX
  - 有线/无线表分类 ONNX
  - 必要的字典 / 配置附属文件
- 托管：复用现有 HF repo `qwqqwq/deepdoc-standalone`，新增 `table/` 子目录；用 `tools/publish_models_to_hf.py` 推送。
- 下载：`python download_models.py table_v2`，或容器启动按 `DEEPDOC_DOWNLOAD_GROUPS` 自动补齐——与现有机制一致。
- License：实现前核验 RapidTable / SLANet-plus 模型权重 license 允许再分发（预期 Apache-2.0 系）。

## 6. 配置项（环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPDOC_TABLE_ENGINE` | `tatr` | 表格识别引擎：`tatr`（现状）/ `rapidtable`（新）。默认不改变现有行为。 |
| `DEEPDOC_TABLE_RAPID_MODEL` | `slanet_plus` | 预留，未来可扩展其他 RapidTable 模型类型。 |

遵循 `AGENTS.md` 约定：不静默变更默认值；行为变更需更新 `docs/API.md`。

## 7. 错误处理与回退

- rapidtable 引擎在以下情况**自动回退 TATR**（不抛错）：
  - `rapid-table` 依赖缺失
  - 模型文件缺失
  - 单表识别异常（裁图损坏 / 区域 OCR boxes 为空 / RapidTable 抛错）
- 回退粒度为**单表级**：某张表失败仅该表回退，不影响整篇文档。
- 用 `common.logger`（仓库自定义 logger）记录回退原因，便于观测。
- 设计不变量：**最坏情况 = 现状，绝不比现在差。**

## 8. 基准验收

- 主指标：**TEDS / S-TEDS**（表格识别标准指标，HTML 结构相似度）。
- 评测集：
  - **OmniDocBench 表格子集**（整页表格 TEDS，对标 RapidTable ≈ 82.5）
  - **PubTabNet** 抽样（经典 TEDS）
  - 业务样本抽查（目视）
- 新增 `tools/eval_table.py`：同一批样本分别用 `tatr` 与 `rapidtable` 跑，输出 TEDS 对比表与逐项明细。
- 建议验收门槛：
  1. `rapidtable` 在 OmniDocBench 表格子集 TEDS **明显高于** `tatr`（目标向 ~82 看齐）。
  2. **无线框表 / 合并单元格子项不退化**。
  3. 业务样本抽查目视通过。
- 性能记录：CPU 单表耗时（预期 1–7s）及对整体解析时延的影响，作为是否启用 batch / 是否仅对复杂表启用的决策依据。

## 9. 分阶段实施

| 阶段 | 内容 | 验证方式 |
|---|---|---|
| 阶段 0（go/no-go，~0.5 天） | 离线脚本验证 RapidTable v3 API + SLANet-plus ONNX 可用、纯 CPU、能接收自有 OCR 结果、依赖不拖入 torch | 独立脚本跑通单图 → HTML |
| 阶段 1 | `rapid_table_recognizer.py` 封装 + `model_store` group + 下载脚本 | 单测：表格图 + OCR boxes → HTML |
| 阶段 2 | `pdf_parser` 按 `DEEPDOC_TABLE_ENGINE` 分流（先不做跨页）+ 单表回退 | 端到端单页表格 PDF 跑通 |
| 阶段 3 | 跨页逻辑表的表头对齐缝合 | 跨页表样本验证 |
| 阶段 4 | `eval_table.py` 出 TEDS 对比，达标后文档化 | 更新 `docs/API.md`、`PARSER_ENGINE_STRATEGY.md` |

## 10. 测试策略（契合「无 pytest」现状）

- 脚本级：`tools/eval_table.py`（基准对比）+ `deepdoc/vision/t_table.py` 冒烟（单图 → HTML）。
- API 冒烟：`parser_engine=deepdoc` + `DEEPDOC_TABLE_ENGINE=rapidtable` 跑样本 PDF。
- 回归：确保默认 `DEEPDOC_TABLE_ENGINE=tatr` 的输出与改造前一致。

## 11. 风险与未决项（实现前处理）

1. **RapidTable v3.0.0 接口**：v3 改过返回值结构，阶段 0 须对实际安装版本核验类名、调用签名、返回结构。
2. **SLANet-plus ONNX 版本**：历史上某版 paddlex-SLANet-plus 因 paddle2onnx 问题无法 ONNX 化，须选用可正常 ONNX 推理的版本。
3. **依赖链**：确认 `pip install rapid-table` 基础版为纯 onnxruntime，不拖入 torch；若强依赖 `rapidocr`，则仅使用其表格结构模块并传入自有 OCR 结果。
4. **OCR 结果格式对接**：本仓库 box（dict：`text/x0/x1/top/bottom`）→ RapidTable 期望格式（通常 `[[四点框], text, score]`）的转换函数需单独实现并测试。
5. **CPU 时延**：表格密集文档可能显著变慢，对策：使用 v3 batch 推理，或仅对复杂表格启用 rapidtable。
6. **License**：核验模型权重可再分发。

## 12. 参考资料

- RapidTable：<https://github.com/RapidAI/RapidTable>
- TableStructureRec（有线/无线分类、ONNX、CPU 1–7s/表）：<https://github.com/RapidAI/TableStructureRec>
- OmniDocBench（CVPR 2025，RapidTable 表格 TEDS ≈ 82.5）：<https://github.com/opendatalab/OmniDocBench> ，论文：<https://arxiv.org/html/2412.07626v1>
- SLANet-1M（CPU 友好、逼近 UniTable）：<https://aclanthology.org/2025.swisstext-1.9.pdf>
- UniTable（torch/GPU，精度最高）：<https://arxiv.org/html/2403.04822v2>
- PaddleOCR-VL（VLM 类表格 TEDS ≈ 0.92，需 GPU）：<https://arxiv.org/pdf/2510.14528>

## 13. 附：相关代码位置索引

- 表格主流程：`deepdoc/parser/pdf_parser.py:_table_transformer_job:337`、`_extract_table_figure:1448`、`construct_table` 调用 `:1676`、跨页 `nearest:1527`
- TSR 模型与拼装：`deepdoc/vision/table_structure_recognizer.py`（labels `:30-37`、`construct_table:140`、`__cal_spans:484-563`）
- ONNX 推理基类：`deepdoc/vision/recognizer.py:Recognizer:30`
- 模型分组：`common/model_store.py:MODEL_GROUP_FILES:12-31`
- 引擎路由与 PDF 解析调用：`main.py:_parse_pdf_from_tmp:509-542`（deepdoc 分支）

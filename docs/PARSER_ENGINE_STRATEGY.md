# 多引擎文档解析策略与 PDF 印章识别说明

- 责任人：陆佳欢
- 更新日期：2026-05-27
- 适用范围：`POST /api/v1/parse` 的 PDF 解析引擎选型，以及政务、合同类 PDF 的印章数量校验。

## 1. 背景与目标

DocPilot 当前支持 `docpilot`、`paddleocr_vl`、`mineru`、`plain` 多种 PDF 解析路径；其中 `docpilot` 兼容旧别名 `deepdoc`。同时支持 `markitdown` 作为非 PDF Markdown 转换器。不同引擎在本地化部署、扫描件识别、复杂版面还原、论文公式表格处理、印章识别等方面侧重点不同。`paddleocr` 仍作为兼容别名保留，但对外建议统一使用 `paddleocr_vl`。

本文档用于沉淀初步解析选型规则：

1. 明确 DocPilot 本地引擎、PaddleOCR、MinerU 在合同、论文、技术报告等文档场景下的适用边界。
2. 为 API 调用方提供稳定、可解释的 `parser_engine` 选择建议。
3. 说明 PDF 印章识别能力及 `seal_count` 返回规则，用于政务与合同场景的基础校验。

## 2. 引擎能力边界

| 引擎 | 当前定位 | 优先适用场景 | 不建议优先使用的场景 | 关键依赖 |
|---|---|---|---|---|
| `docpilot`（兼容 `deepdoc`） | 默认本地解析引擎，基于 OCR、版面识别、表格识别等本地模型 | 常规 PDF、文本层较清晰的合同/报告、需要离线或内网解析的场景 | 大量扫描页、强视觉理解、必须返回 `seal_count` 的场景 | 本地 ONNX 模型，`DEEPDOC_MODEL_PATH` |
| `paddleocr_vl` | 远程 PaddleOCR-VL 解析引擎，视觉 OCR 与版面块识别能力更强 | 扫描版 PDF、政务材料、盖章合同、需要印章数量 `seal_count` 的场景 | 无可用 PaddleOCR 服务、强离线要求、网络不稳定场景 | `PADDLEOCR_API_URL` 或 `PADDLEOCR_GPU_API_URL` |
| `markitdown` | 本地 Markdown 转换引擎，基于 Microsoft MarkItDown | 快速转换 Office、CSV/XML/EPUB/ZIP 等非 PDF 文件为 Markdown | 需要印章计数、复杂版面定位框、强视觉结构恢复的场景 | 本地 Python 依赖 `markitdown` |
| `mineru` | 远程 MinerU 解析引擎，偏复杂学术/技术文档结构恢复 | 论文、技术报告、公式/表格/代码块较多的 PDF | 以印章校验为核心的合同/政务材料 | `MINERU_APISERVER` 或 `MINERU_GPU_API_URL` |
| `plain` | PDF 文本层快速提取 | 文本层完整、只需要快速抽取正文的轻量场景 | 扫描件、复杂版面、表格/图片/印章识别 | 无额外模型 |

### 2.1 deepdoc 本地 pipeline 开关（可回退）

`deepdoc` 本地引擎的 CPU pipeline 升级通过环境变量灰度，不改变默认解析行为：

| 变量 | 默认值 | 可选值 | 作用 | 模型准备 |
|---|---|---|---|---|
| `DEEPDOC_OCR_VERSION` | `v4` | `v4` / `v5` | 切换 OCR 检测与识别模型 | `v5` 需 `python download_models.py core_v5` |
| `DEEPDOC_REC_IMAGE_SHAPE` | 空 | `C,H,W` | 通用 OCR 识别输入尺寸覆盖，空值保持默认 `3,48,320` | 与当前 OCR 模型匹配 |
| `DEEPDOC_OCR_V4_REC_IMAGE_SHAPE` | 空 | `C,H,W` | v4 OCR 识别输入尺寸覆盖，优先于 `DEEPDOC_REC_IMAGE_SHAPE` | 与 `core` 中 `rec.onnx` 匹配 |
| `DEEPDOC_OCR_V5_REC_IMAGE_SHAPE` | 空 | `C,H,W` | v5 OCR 识别输入尺寸覆盖，优先于 `DEEPDOC_REC_IMAGE_SHAPE` | 与 `core_v5` 中 `rec_v5.onnx` 匹配 |
| `DEEPDOC_LAYOUT_ENGINE` | `legacy` | `legacy` / `ppdoclayout` | 切换 PDF OCR + Layout 路径的版面后端 | `ppdoclayout` 需 `python download_models.py layout_v2` |
| `DEEPDOC_FORMULA_MODE` | `rapidlatex` | `rapidlatex` / `pp_formula_net_s` | 切换公式识别模式 | `pp_formula_net_s` 需 `python download_models.py formula_v2` |
| `DEEPDOC_READING_ORDER_STRATEGY` | `legacy` | `legacy` / `rules` | 切换阅读顺序和跨页段落规则；`rules` 启用多栏排序、重复页眉页脚去重、caption 绑定和标题/缩进误合并拦截 | 无额外模型 |

`DEEPDOC_LAYOUT_ENGINE=ppdoclayout` 使用 PP-DocLayout-plus / RT-DETR 后处理，类别会映射回 DocPilot 现有 `text/title/table/figure/equation/header/footer/reference` 等类型，保证下游 structured/chunks/assets 产物边界不变。

`DEEPDOC_READING_ORDER_STRATEGY=rules` 只作用于 `deepdoc` 本地 PDF 解析后的结构整理阶段，默认仍为 `legacy`。开启后会在文本合并后执行规则排序，减少多栏内容按 Y 轴交错、重复页眉页脚进入 chunks、caption 与图片/表格脱钩，以及跨页段落误接到新标题的问题。

`python download_models.py manifest` 会输出 `ocr_dictionaries` 和 `ocr_recognition_alignments`，用于在启用 `core_v5` 前检查 `ocr.res`/`ocr_v5.res` 字典完整性，以及 `rec.onnx`/`rec_v5.onnx` 输出类别数与字典行数、空格类、CTC blank 是否对齐。

`DEEPDOC_FORMULA_MODE=pp_formula_net_s` 通过 PaddleX PP-FormulaNet-S 适配器接入公式识别，需安装 `pip install -e ".[formula-v2]"` 并预置 `formula_v2` 模型组。生产公式识别默认继续保持 `DEEPDOC_FORMULA_MODE=rapidlatex`，切默认前必须先补真实公式子集 A/B 指标和 CPU 耗时报告。

这些开关只作用于本地文档解析 pipeline，不改变 `parser_engine`、`compute_device`、`return_images`、`strict_text` 等 API 参数默认值。

`POST /api/v1/parse`、`/api/v1/parse/stream` 和 `/api/v1/parse/async` 现已锁定两个 deepdoc 相关契约参数：

- `deepdoc_pdf_mode=auto/native/ocr/hybrid`
- `execution_profile=auto/cpu/gpu`

当前版本里，`execution_profile` 只做参数归一化与透传，默认 `auto`；`deepdoc_pdf_mode=hybrid` 只做前向兼容占位，尚未启用新的 page/block 混合路由，当前行为与 `auto` 一致。

### 2.2 deepdoc 表格识别引擎（可选 rapidtable）

`deepdoc` 引擎的表格识别支持两种实现，由环境变量 `DEEPDOC_TABLE_ENGINE` 切换，默认 `tatr`、可随时回退：

| 取值 | 实现 | 适用 | 备注 |
|---|---|---|---|
| `tatr`（默认） | Table Transformer + 几何拼装 | 线框规整、结构简单的表格 | 行为与历史版本一致 |
| `rapidtable` | RapidTable / SLANet-plus（纯 ONNX/CPU） | 复杂表格：合并单元格、无线框表、密集表 | 需 `pip install -e ".[table]"` 并 `python download_models.py table_v2`；单表识别失败自动回退 `tatr` |

`rapidtable` 仅作用于 `parser_engine=deepdoc` 且返回 HTML 的解析；当前跨页表的坐标映射需按样本校验。可用 `tools/eval_table.py` 在 OmniDocBench / PubTabNet 上做 TEDS 对比择优。

## 3. 场景选型规则

### 3.1 合同类文档

推荐顺序：

1. **扫描版合同、盖章合同、政务合同**：优先 `paddleocr_vl`。该场景通常需要识别红章、签章页、扫描噪声和视觉版面块，`paddleocr_vl` 可返回 `seal_count`，便于做“是否盖章”“盖章数量是否符合预期”的基础校验。
2. **文本层清晰、主要关注条款抽取**：优先 `deepdoc`。如需本地印章文字补充，可使用 `enable_seal=true`，但当前 API 级 `seal_count` 以 `paddleocr_vl` 返回为准。
3. **只需快速抽正文且不关心版面**：可使用 `plain`。
4. **需要快速把 Office/CSV/XML/EPUB 转成 Markdown，且不依赖本地 OCR 版面框**：可使用 `markitdown`。

建议参数：

```bash
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "parser_engine=paddleocr_vl" \
  -F "compute_device=gpu" \
  -F "return_images=false" \
  -F "file=@/path/to/contract.pdf"
```

### 3.2 政务材料

推荐优先使用 `paddleocr_vl`，尤其是包含公章、骑缝章、扫描页、表单、批复件、红头文件的 PDF。政务材料通常要求解析结果具备可校验字段，`seal_count` 可作为流程校验信号。

注意：`seal_count` 只表示解析引擎识别到的印章/签章块数量，不代表印章真伪、主体合法性或盖章有效性。

### 3.3 学术论文

推荐顺序：

1. **公式、表格、图片说明、参考文献较多**：优先 `mineru`。
2. **普通论文或页数受控的 PDF**：可使用 `deepdoc`，并按需要设置 `DEEPDOC_LAYOUT_MODEL=paper`。
3. **只提取文本层内容**：可使用 `plain`。
4. **只需尽快转为 Markdown 初稿，接受结构保真度低于 DocPilot/MinerU**：可使用 `markitdown`。

`mineru` 更适合作为论文结构恢复引擎；若论文同时包含扫描页或印章校验要求，需要结合 `paddleocr_vl` 做补充解析。

### 3.4 技术报告

技术报告的选型取决于结构复杂度：

1. **普通报告、文本和表格为主**：优先 `deepdoc`。
2. **公式、代码块、复杂表格、图文混排较多**：优先 `mineru`。
3. **扫描版报告或截图式报告**：优先 `paddleocr_vl`。

若报告有合规签章页，建议对签章页或整份 PDF 使用 `paddleocr_vl`，读取 `seal_count` 做基础校验。

## 4. PDF 印章识别与 `seal_count`

### 4.1 对外返回规则

当前 API 在以下条件满足时返回 `seal_count`：

- 文件类型为 `pdf`
- `parser_engine=paddleocr_vl`
- PaddleOCR 服务返回的版面块中包含 `seal` 或 `stamp` 类型块

响应示例：

```json
{
  "results": [
    {
      "filename": "contract.pdf",
      "type": "pdf",
      "markdown": "# ...",
      "seal_count": 2
    }
  ]
}
```

字段含义：

- `seal_count=0`：未识别到印章块，或上游服务未启用/未返回印章识别结果。
- `seal_count>0`：识别到对应数量的印章/签章块。
- 非 PDF 文件、非 `paddleocr_vl` 引擎默认不返回该字段。

### 4.2 PaddleOCR 印章识别配置

API 侧通过环境变量控制 PaddleOCR 的印章识别开关：

```bash
export PADDLEOCR_USE_SEAL_RECOGNITION=true
export PADDLEOCR_GPU_API_URL=http://<paddleocr-host>:<port>
```

调用时选择 `parser_engine=paddleocr_vl`：

```bash
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "parser_engine=paddleocr_vl" \
  -F "compute_device=gpu" \
  -F "file=@/path/to/government_contract.pdf"
```

如果 PaddleOCR 服务版本不支持印章块标签，或未开启上游印章识别能力，API 仍可能返回 `seal_count: 0`。

### 4.3 DocPilot 本地印章能力

`deepdoc` 引擎支持本地印章检测增强：

```bash
python download_models.py seal

curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "parser_engine=deepdoc" \
  -F "enable_seal=true" \
  -F "file=@/path/to/contract.pdf"
```

该能力使用 `resources/models/seal/seal_det.onnx` 检测印章区域，并将识别结果以 Markdown 文本插入，例如：

```markdown
[印章: 某某有限公司]
```

当前该路径主要用于补充 Markdown 内容，不作为 API 级 `seal_count` 的返回来源。需要稳定读取印章数量时，应优先选择 `paddleocr_vl`。

## 5. 推荐决策表

| 文档特征 | 推荐引擎 | 原因 |
|---|---|---|
| 扫描版合同、盖章合同、政务审批件 | `paddleocr_vl` | OCR/视觉块识别更适合扫描材料，并支持 `seal_count` |
| 文本层清晰的合同、制度文件、普通报告 | `deepdoc` | 本地可控，默认链路稳定，适合常规版面 |
| Office 文档、CSV/XML/EPUB、快速 Markdown 转换 | `markitdown` | 本地调用 MarkItDown，接入轻，覆盖格式广 |
| 论文、专利说明、复杂技术报告 | `mineru` | 对公式、表格、图文混排结构恢复更友好 |
| 只需要 PDF 文本层快速抽取 | `plain` | 成本低、速度快，但不处理 OCR 和复杂版面 |
| 内网离线、不能依赖远程服务 | `deepdoc` | 本地模型链路，不依赖 PaddleOCR/MinerU 服务 |
| 必须校验印章数量 | `paddleocr_vl` | 当前 `seal_count` 由 PaddleOCR PDF 解析链路返回 |

## 6. 质量与风险边界

1. `seal_count` 是识别计数字段，不是法律意义上的验章或真伪鉴定。
2. 模糊扫描、低分辨率、印章缺失、印章与正文重叠、电子签章样式差异，可能导致漏检或误检。
3. 不同 PaddleOCR 服务版本、模型配置、阈值策略可能影响 `seal_count`。
4. 多页合同存在骑缝章时，建议以业务规则定义期望数量，例如“至少 1 个”“每份合同至少封面/落款页各 1 个”，不要简单要求固定值。
5. `return_images=false` 可降低响应体体积；如需人工复核印章位置，可设置 `return_images=true` 获取更多图片内容。

## 7. 验收建议

建议使用以下样本做回归验证：

| 样本 | 推荐引擎 | 预期 |
|---|---|---|
| 无印章文本合同 PDF | `paddleocr_vl` | `seal_count=0` |
| 单印章扫描合同 PDF | `paddleocr_vl` | `seal_count>=1` |
| 多印章政务材料 PDF | `paddleocr_vl` | `seal_count` 与人工标注数量基本一致 |
| 普通论文 PDF | `mineru` 或 `deepdoc` | Markdown 结构完整，无需 `seal_count` |
| 文本层完整的普通报告 | `deepdoc` 或 `plain` | 正文提取稳定 |

上线前至少验证：

1. `parser_engine=paddleocr_vl`（或兼容别名 `paddleocr`）的 PDF 响应包含整数型 `seal_count`。
2. 非 PDF 或非 `paddleocr_vl` 引擎不依赖 `seal_count` 做业务判断。
3. PaddleOCR 服务不可用时，接口返回可解释错误，而不是空结果或裸异常。
4. 对政务/合同业务侧，明确 `seal_count` 的阈值规则和人工复核策略。

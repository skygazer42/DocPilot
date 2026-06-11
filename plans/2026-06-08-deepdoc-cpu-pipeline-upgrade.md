# DocPilot CPU Pipeline 现代化:对标评估与升级实施计划

> - 责任人:待指派
> - 日期:2026-06-08
> - 状态:draft v2（已按技术评审校准）
> - 范围:`deepdoc` 本地解析引擎（`deepdoc/vision/*` + `deepdoc/parser/pdf_parser.py`）的解析能力升级，不改变对外 API 契约
> - 关联文档:`plans/optimization-roadmap.md`、`docs/PARSER_ENGINE_STRATEGY.md`、`plans/2026-06-06-unified-model-hub-and-parser-cleanup.md`

## 修订记录

- **v2（2026-06-08，按技术评审校准）**:
  1. 版面模型由 DocLayout-YOLO（AGPL-3.0，与本项目 Apache-2.0 不兼容）改为 **PP-DocLayout-plus（Apache 2.0，RT-DETR）**；删除来源存疑的 `0.91 mAP`，速度表述改为"实时级、远快于 transformer"。
  2. 公式模型由 UniMERNet/texify 改为 **PP-FormulaNet-S / plus-S（Apache 2.0，CPU 友好）** 作为 CPU 默认候选；UniMERNet 仅作高精度/离线模式，texify 因 deprecated + GPL-3.0 排除。
  3. 所有"收益数字"从"模型官方 headline"改为"以 DocPilot 端到端指标验证"。
  4. 难度普遍上调（OCR 低→低-中，版面 中→中-高，阅读顺序 中→中-高）。
  5. 性能优化顺序改为 **profiling → 批处理/线程治理/缓存 → 按模块 INT8 校准**，删除"INT8 抵消 v5 略慢"的未验证结论。
  6. 新增 §2.6 许可证合规，作为 Sprint 0 前置门禁。
  7. 关键收敛:OCR + 版面 + 表格 + 公式 四环节全部落到 **PaddleOCR 全家桶（全 Apache 2.0、全 CPU 友好）**。

---

## 当前实现状态（2026-06-08）

本轮已把 CPU pipeline 升级的**脚手架已落地**，默认行为仍保持旧链路，不引入 RAG、向量化、问答或回答生成。

| Sprint | 已落地 | 仍需完成 |
|---|---|---|
| Sprint 0 | `tools/eval_omnidocbench.py`、license gate、空数据集失败保护、dataset contract 校验和 CPU pipeline readiness 门禁已落地；`--validate-dataset` 已输出 `2026-06-08.cpu-pipeline-dataset-contract.v1`，正式评测会先预检 dataset contract 且报告内写入 `dataset_contract`；正式评测报告会写入 `2026-06-08.cpu-pipeline-model-manifest.v1` 的 `model_manifest`，记录评测时模型目录、模型组、声明模型文件的存在性/大小/sha256；下载/发布模型 manifest 已写入 `model_group_provenance`，记录模型组来源/许可证、默认状态、开关和 readiness gate，`tools/ci/verify_hf_models.py` 会校验远端 manifest 的 `model_group_provenance` 与本地声明一致；license gate 的升级模型组候选会从 `MODEL_GROUP_PROVENANCE` 派生，避免模型来源/许可证声明和 license gate 手工清单漂移；正式评测报告的 `model_manifest` 会按 `pipeline_config` 记录相关模型组，readiness 会按 A/B 报告的 `pipeline_config` 校验对应模型组快照是否与当前 `--model-root` 一致，避免模型换版后复用旧报告；readiness 会写入 `dataset_contract` 并用 `dataset_contract_failed` 拦截坏标注；评测工具已支持递归发现评测 PDF，子目录样本使用 dataset-relative 样本名；readiness 已要求 `2026-06-08.cpu-pipeline-license-gate.v1` license gate 报告且状态为 `passed`，并要求 A/B 报告声明 `engine=deepdoc`、samples 明细中声明的 `engine` 也必须是 `deepdoc`、A/B 报告内嵌 `license_gate`、A/B 报告内嵌 `dataset_contract`、A/B 报告内嵌 `model_manifest`、数据集匹配本次 readiness `--dataset`、baseline/candidate 同数据集且同 `sample_count`，且 `sample_count` 必须是正整数并等于 dataset contract 识别到的样本数；readiness 会校验 A/B 报告内嵌 `license_gate` 的 allowed/blocked 候选覆盖及 `license/status` 字段一致性，也会校验 A/B 报告内嵌 `dataset_contract` 的 schema/status/sample_count/samples 是否与当前 readiness 数据集一致；A/B 报告必须包含 `samples` 明细，readiness 也会校验 `summary.sample_count` 必须是正整数并等于 `len(samples)`，summary 中由 samples 明细产生的均值指标必须在每个 sample 中都有对应指标值，summary 和 samples 指标都必须是数值且有限，JSON boolean 不能作为数值字段，且 summary 均值必须按全量 samples 明细计算并一致，samples 明细里的样本名和声明的 `pdf_path` 必须匹配 dataset contract；评测工具已覆盖 CER/WER、block F1、TEDS/cell F1、阅读顺序、跨页合并、`chunk_text_coverage`、`business_field_location_hit_rate`；layout readiness 已要求跨页合并准确率、chunk 覆盖率和业务字段命中率 candidate 不低于 baseline | 真实业务 PDF 基线仍需数据集后运行，并输出全量指标 |
| Sprint 1 | `core_v5` 模型组、`DEEPDOC_OCR_VERSION=v4|v5`、v4 默认与 v5 路径切换、OCR v4/v5 字典 manifest/CI 多字典校验已落地、OCR rec 输出类别数与字典行数对齐校验已落地、OCR rec_image_shape 版本化配置已落地 | v5 权重/A-B 指标待真实模型与数据集验证，字典维度仍需用真实权重核验 |
| Sprint 2 | `layout_v2` 模型组、`PPDocLayoutRecognizer`、`DEEPDOC_LAYOUT_ENGINE=legacy|ppdoclayout`、RT-DETR 后处理脚手架已落地；`table_v2` 模型组、`DEEPDOC_TABLE_ENGINE=tatr|rapidtable`、RapidTable/SLANet-plus 可切换后端和 table A/B readiness 门禁已落地，readiness 已要求 `mean_table_teds` / `mean_table_cell_f1` candidate 不低于 baseline | PP-DocLayout 真实权重、23 类顺序、block F1 和阅读顺序收益待验证；表格默认化仍需真实表格评测后才能切默认 |
| Sprint 3 | `DEEPDOC_READING_ORDER_STRATEGY=legacy|rules`、阅读顺序 rules 策略已落地；默认仍为 legacy；rules 已覆盖多栏排序、重复页眉页脚去重、caption 绑定、标题/缩进跨页误合并拦截，并已接入主解析路径、`eval_omnidocbench.py`、`profile_pipeline.py` 和 readiness 的 `reading_order_strategy` 门禁 | 仍需在真实业务 PDF 上跑 baseline/candidate A/B，验证阅读顺序编辑距离、跨页合并准确率、chunk 覆盖率和业务字段定位命中率后，才能考虑默认切换 |
| Sprint 4 | `formula_v2` 模型组、`DEEPDOC_FORMULA_MODE=rapidlatex|pp_formula_net_s` 路径解析已落地；PP-FormulaNet-S 已通过 PaddleX `create_model(...).predict(...)` 适配器接入，默认仍为 `rapidlatex`；`eval_omnidocbench.py` 已支持 `.gt.formulas.json` 的公式文本级 NED/exact-match 中间指标，readiness 已要求 formula A/B 报告包含 `mean_formula_normalized_edit_distance`、`mean_formula_exact_match_rate` 和 `mean_elapsed_seconds` | 真实 PP-FormulaNet-S 权重/PaddleX 环境联调、真实 OmniDocBench CDM/CPU 耗时 A/B 待真实论文子集验证；默认仍不能切换 |
| Sprint 5 | `tools/profile_pipeline.py` 已落地，可输出 7 个阶段耗时 JSON（`rasterize_ocr`、`layout`、`table`、`text_merge`、`cross_page_text`、`reading_order`、`extract_assets`）；profile 报告已统一记录含 `formula_mode` 的 pipeline_config、`dataset`、`sample_name`、`model_manifest`、`license_gate`、`dataset_contract` 和 `stage_summary` 瓶颈摘要，readiness 会校验 profile 顶层配置、顶层 `dataset`、阶段求和、必选阶段覆盖、额外阶段、最慢阶段与 `pipeline_config`/`stage_summary` 一致，也会校验 profile `model_manifest` 与当前 `--model-root` 一致；profile 报告自身也会内嵌 `dataset_contract`，并且会内嵌 `license_gate`，readiness 会校验 profile 内嵌 `license_gate` 的 allowed/blocked 候选覆盖及 `license/status` 字段一致性，也会校验 profile 内嵌 `dataset_contract` 的 schema/status/sample_count/samples，且 profile 顶层 `dataset` 必须匹配 readiness 当前 `--dataset`，profile `sample_name`/`pdf_path` 必须对应 readiness 当前 dataset contract 中的同一条 PDF 样本；`tools/quantize_models.py --model-dir` 已支持递归扫描模型组子目录，且默认跳过 `rec.onnx`、`rec_v5.onnx`、`formula/decoder.onnx` 等高风险序列/decoder 模型，需逐模块校准后用 `--include-risky-sequence-models` 显式开启 | 真实 profiling 报告、批处理/线程/缓存优化和逐模块 INT8 校准仍需样本与模型后推进 |

> 补充: profile 顶层 `dataset` 必须存在并匹配本次 readiness 的 `--dataset`；profile 顶层 `sample_name` 与 `pdf_path` 必须指向 dataset contract 的同一样本；profile 内嵌 `dataset_contract.samples[*].pdf_path` 若有声明，也必须与当前 readiness 数据集中的同名 PDF 一致，避免复用旧数据集路径的 profile 报告。
> 内嵌 `dataset_contract.samples` 必须是数组，且每一项都必须是 JSON object，避免非结构化评测集合同绕过样本名和路径校验。
> 正式 A/B 报告的 `samples[*].pdf_path` 和内嵌 `dataset_contract.samples[*].pdf_path` 都是必填项，缺失时 readiness 会失败，避免路径不明的旧报告混过门禁。
> 正式 A/B 报告的 `samples` 必须是数组，且每一项都必须是 JSON object，避免非结构化明细绕过样本名、路径、引擎和指标校验。
> 正式 A/B 报告中，只要 `summary` 声明了由 samples 明细产生的均值指标，每个 sample 都必须提供对应指标值；readiness 会按全量 samples 明细重算均值，缺失、非数值/非有限数值、JSON boolean 或摘要/明细不一致都会失败。
> 未传 `--dataset` 的 profile 报告也会写入 `status=failed` 的 `dataset_contract`，显式记录未绑定评测集，不能用于通过 readiness。
> profile 报告中的 `total_elapsed_seconds`、`stages[*].elapsed_seconds`、`stage_summary.slowest_stage_elapsed_seconds`、`stage_summary.slowest_stage_share`、`stage_summary.stages_by_elapsed_seconds[*].elapsed_seconds` 和 `stage_summary.stages_by_elapsed_seconds[*].share` 都必须是数值且有限，`stage_summary.stages_by_elapsed_seconds[*]` 的 `share` 必须匹配对应阶段耗时占比，JSON boolean 不能作为耗时或占比字段，避免 `NaN` / `Infinity` 或 `true` / `false` 耗时报告混过 readiness。

同时，解析侧 tokenizer 已去掉 `rag_tokenizer` 适配路径，保留为本地文档解析 tokenizer，避免核心解析代码继续依赖 RAG 命名实现。

因此，本计划当前状态是“可回退开关 + 模型组 + 评测/profiling 工具初版 + readiness 指标门禁已实现”，不是“CPU pipeline 全量升级完成”。后续每次默认值切换都必须先补真实指标和 API 文档。

---

## 0. TL;DR

1. **结论先行**:在「CPU 为主、资源有限、可离线」约束下，deepdoc 的 **pipeline 架构（ONNX、模块化）路线是对的**——Docling、MinerU-pipeline 后端、PP-StructureV3、RapidOCR 走的都是同一条路。VLM 端到端方案在纯 CPU 上不可用，不在范围。
2. **差距在「各环节模型代次老」**，不在路线。
3. **升级策略**:保留 pipeline，逐环节替换为 CPU 友好的现代轻量模型，**优先收敛到 PaddleOCR 全家桶（Apache 2.0）**:OCR→PP-OCRv5、版面→PP-DocLayout-plus、表格→SLANet-plus、公式→PP-FormulaNet-S。全程用真实 PDF 评测集量化收益，所有替换可 env 回退。
4. **收益必须以 DocPilot 端到端指标验证**（CER/WER、block F1、TEDS/cell F1、阅读顺序编辑距离、跨页合并准确率、chunk 覆盖率、业务字段定位命中率），不引用模型官方 headline 数字。
5. **许可证是硬门禁**:DocLayout-YOLO（AGPL）、Marker/Surya（GPL+收入限制）不可用于本 Apache-2.0 项目；优先 PaddleOCR 全家桶 + Docling（MIT）。
6. **脚手架已就绪一半**:`table_v2`（SLANet-plus）后端、`model_store` 模型组机制、`tools/eval_table.py` 已存在。

---

## 1. 背景、约束与目标

### 1.1 deepdoc 现状（技术栈）

`deepdoc` 本地引擎沿用上游 deepdoc 的技术路线，纯 ONNX、CPU 可跑的传统 CV pipeline:

| 环节 | 实现（源码可确认部分） | 关键文件 |
|---|---|---|
| 文字检测 | DB 类检测（`det.onnx`）+ `DBPostProcess` | `deepdoc/vision/ocr.py::TextDetector` |
| 文字识别 | CTC 类识别（`rec.onnx`）+ 字典 `ocr.res` + `CTCLabelDecode`（具体网络需看 ONNX 图） | `deepdoc/vision/ocr.py::TextRecognizer` |
| 版面分析 | YOLO 类 layout（`layout.{manual,paper,laws}.onnx`） | `deepdoc/vision/layout_recognizer.py` |
| 表格结构 | Table Transformer（`tsr.onnx`）+ **已接入** RapidTable/SLANet-plus（可切换，失败回退 tatr） | `table_structure_recognizer.py`、`rapid_table_recognizer.py` |
| 段落跨行合并 | XGBoost（`updown_concat_xgb.model`） | `deepdoc/parser/pdf_parser.py` |
| 公式 | RapidLaTeXOCR（`formula/*`，默认关闭） | `deepdoc/vision/formula_recognizer.py` |
| 印章 | seal det（`seal/seal_det.onnx`，默认关闭） | `deepdoc/vision/seal_recognizer.py` |

模型由 `common/model_store.py::MODEL_GROUP_FILES`（core/formula/seal/table_v2）管理，从 HF 仓库下载。外部引擎（`mineru`、`paddleocr_vl`、`docling`、`tcadp`）作为远程/适配器并存。工程层（异步、artifact、ingest、监控、限流、多租户）已成熟。

### 1.2 本次硬约束:CPU 为主、资源有限、可离线

- 目标部署以 **CPU** 为主，无稳定 GPU。
- 关注**轻量、低成本、可离线**，核心能力不依赖公网/远程 API。
- 该约束**直接排除 VLM 端到端方案**作为本地主链路（§2.2）。

### 1.3 目标与非目标

**目标:** ① 基于 2025–2026 主流方案的对标评估;② 分 Sprint、可执行、可回退的升级计划;③ 建立**真实 PDF 评测基线**让收益可量化。

**非目标（YAGNI）:** 不引入 GPU VLM 作主链路;不重写工程层;不做问答、向量化或回答生成;不改 `parser_engine`/`compute_device`/`return_images`/`strict_text` 默认行为;不引入 AGPL/GPL 组件;不合并对象存储/数据库等无关改造。

---

## 2. 对标评估（2025–2026 主流开闭源）

### 2.1 核心判断:CPU 场景路线正确，差距在模型代次

deepdoc 与 Docling / MinerU-pipeline / PP-StructureV3 / RapidOCR 属于**同一条 pipeline 赛道**，落后的是各环节模型版本，而非架构。

### 2.2 为什么 CPU 不能上 VLM（证据）

| 证据 | 数据 | 来源 |
|---|---|---|
| dots.ocr 即便 GPU 也极慢 | RTX 4000 Ada 单页发票 **26 秒**；A100 仅 **0.35 页/秒** | [dots.ocr issue #103](https://github.com/rednote-hilab/dots.ocr/issues/103) |
| MinerU 把 pipeline 作 CPU 兜底 | `vlm-engine` 必须 GPU；`pipeline backend` 支持纯 CPU | [MinerU GitHub](https://github.com/opendatalab/mineru) |
| 同类 pipeline 的 CPU 基准 | Docling x86 CPU **3.1 秒/页** | [Docling arXiv:2501.17887](https://arxiv.org/html/2501.17887v1) |

**结论:CPU 场景下，pipeline 是唯一现实路线。**

### 2.3 评测锚点（OmniDocBench）

[OmniDocBench](https://github.com/opendatalab/OmniDocBench)（CVPR 2025）指标:文本 Normalized Edit Distance、表格 TEDS、公式 CDM、阅读顺序 Edit Distance。

| 模型 | 类型 | Text Edit↓ | 备注 |
|---|---|---|---|
| MinerU2.5 | VLM 1.2B | 0.047 | 需 GPU（SOTA 参照） |
| **PP-StructureV3** | **pipeline** | **0.073** | **CPU 可跑，deepdoc 同类天花板** |

> 来源:[MinerU2.5 arXiv:2509.22186](https://arxiv.org/pdf/2509.22186)；PaddleOCR 3.0 技术报告称 PP-StructureV3 为 OmniDocBench 上 pipeline 类 SOTA。**这些是 SOTA 差距参照，不等于 deepdoc 端到端可达值——以本仓库评测集为准。**

### 2.4 各环节对标（校准后）

| 环节 | 合理性 | deepdoc 现状 | CPU 友好替代（首选） | 校准后表述 | 难度 |
|---|---|---|---|---|---|
| 检测+识别 | 高 | DB 检测 + CTC 识别 | **PP-OCRv5**（Apache，~16MB mobile，0.07B） | 多语种/繁中/日文/竖排/手写/低质扫描收益明显；**以端到端 CER/WER 验证**，不写死 +13% | 低-中 |
| 版面分析 | 中-高 | YOLO 类 layout | **PP-DocLayout-plus**（Apache，RT-DETR，23 类） | 跨域版面鲁棒性提升（论文/报告/复杂扫描/多栏/试卷/竖排）；**实时级、远快于 transformer**；不写死 mAP | 中-高 |
| 表格结构 | 中 | TableTransformer + 已接入 SLANet-plus | SLANet-plus（Apache，Intel CPU 优化） | 后端**已 pluggable**；先在真实表格集评 TEDS/cell F1 优于 tatr 再设默认；不写死 95-96% | 低-中 |
| 阅读顺序/跨页 | **高** | XGBoost 拼接（弱项） | 规则 + 轻量排序（column/caption/header-footer/跨页） | **高 ROI 瓶颈**；规则优先于重模型 | 中-高 |
| 公式 | 中 | RapidLaTeXOCR | **PP-FormulaNet-S / plus-S**（Apache，CPU 友好） | RapidLaTeXOCR vs PP-FormulaNet-S 做 CPU A/B；UniMERNet 仅高精度/GPU 模式；texify 排除（deprecated+GPL） | 中 |
| 整体编排 | 中-高 | 同步串行 | profiling → 批/线程/缓存 → INT8 | 先 profiling 再优化；INT8 逐模块校准 | 中 |

来源:[PP-OCRv5 多语种](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md)、[PP-DocLayout-L (HF, apache-2.0)](https://huggingface.co/PaddlePaddle/PP-DocLayout-L)、[PP-StructureV3 (PaddleX)](https://paddlepaddle.github.io/PaddleX/3.5/en/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html)、[RapidTable](https://github.com/RapidAI/RapidTable)、[公式识别 (PaddleX)](https://paddlepaddle.github.io/PaddleX/3.1/en/module_usage/tutorials/ocr_modules/formula_recognition.html)、[RapidLaTeXOCR](https://github.com/RapidAI/RapidLatexOCR)、[DocLayout-YOLO arXiv:2410.12628](https://arxiv.org/html/2410.12628v1)、[ONNX Runtime 量化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。

### 2.5 闭源云作为「难文档兜底」（非主链路）

| 方案 | 价格（/1000 页） | 定位 |
|---|---|---|
| Mistral OCR 3 | $1（批量）~$2 | 最便宜 |
| Azure Document Intelligence | $0.53~$10 | 企业承诺量 |
| 腾讯云 TCADP | 国内 | 已接入适配器 |

### 2.6 许可证合规（硬门禁，核实于 2026-06）

本项目 **deepdoc-standalone 自身是 Apache 2.0**，集成组件必须许可证兼容。

| | 方案 | 许可证 | 用于本项目 |
|---|---|---|---|
| 🟢 | Docling | MIT | 可（代码+逻辑） |
| 🟢 | PaddleOCR 全家桶（PP-OCRv5 / PP-DocLayout / SLANet-plus / PP-FormulaNet） | Apache 2.0 | 可（权重+代码） |
| 🟢 | RapidOCR / RapidLaTeXOCR | Apache 2.0（建议二次确认） | 可 |
| 🟡 | MinerU | 2026.4 称 AGPL→Apache（信息冲突） | 用前核实 |
| 🔴 | DocLayout-YOLO | AGPL-3.0 | **不可**（与 Apache 项目不兼容 + 网络条款） |
| 🔴 | Marker / Surya / texify | GPL-3.0（+ 收入受限权重） | **不可** |

---

## 3. 升级总体策略

### 3.1 原则

1. **保留 pipeline**，逐环节替换，单点可独立验证/回退。
2. **评测驱动**:先建真实 PDF 评测基线（Sprint 0），每步给「升级前 vs 升级后 vs 同类参照」的量化对比;**只认 DocPilot 端到端指标，不认模型官方 headline**。
3. **许可证门禁**:任何拟集成组件先过 §2.6 红绿灯。
4. **不破坏 API**:替换通过 env 切换，默认保留旧行为直到验证;变更默认值时更新 `docs/API.md`、`docs/PARSER_ENGINE_STRATEGY.md`。
5. **CPU 优先**:替换必须在目标 CPU 跑出可接受 sec/页。

### 3.2 现有可复用资产

| 资产 | 位置 | 意义 |
|---|---|---|
| 模型组机制 | `common/model_store.py::MODEL_GROUP_FILES` + `MODEL_GROUP_PROVENANCE` + `ensure_groups` | 加新组需同时加文件清单和模型组来源/许可证声明 |
| 下载入口 | `download_models.py`（含 `core_v5`/`layout_v2`/`table_v2`/`formula_v2`，`download_models.py all` 覆盖全部声明模型组） | 加组名即可；manifest 输出 `model_group_provenance` |
| 表格后端 | `table_v2` 组 + `rapid_table_recognizer.py`（已 pluggable，失败回退 tatr） | 表格升级=默认化决策 + 跨页合并 |
| 版面框架 | `layout_recognizer.py`（含 `LayoutRecognizer4YOLOv10`、`AscendLayoutRecognizer`） | RT-DETR（PP-DocLayout）需**新写后处理**，不复用 YOLOv10 分支 |
| OCR 批处理/CPU 调优 | `ocr.py::recognize_batch`、`OCR_INTRA_OP_NUM_THREADS` | 批处理与线程控制已部分就绪 |
| 评测 | `tools/eval_table.py` | OmniDocBench 评测管线起点 |
| ONNX 加载 | `ocr.py::load_model`（CPU/GPU 自适应、缓存） | 量化加载在此扩展 |

---

## 4. 分 Sprint 实施计划

> 每个 Sprint:目标 / 改动文件 / 做法 / 验证 / 回退 / 验收。验证遵循 `AGENTS.md`（无 pytest，用脚本级 + API 冒烟 + `uv run ruff check .`）。

### Sprint 0 — 评测基线 + 许可证审查（前置，P0）

**目标:** 建真实 PDF 评测集 + 量化 deepdoc 现状 + 锁定可合法集成的组件。

**改动文件:** 新增 `tools/eval_omnidocbench.py`（复用 `eval_table.py` 的 TEDS，补 CER/WER、阅读顺序编辑距离、跨页合并准确率）;`tools/eval_datasets/`（不提交大文件）;`docs/` 评测与许可证说明。

**做法:**
1. 备 **100–300 页真实业务 PDF**，分类:扫描件、论文、财报/年报、法律/合同、表格密集、多栏、多语种、公式密集。
2. 指标体系:OCR CER/WER、layout block F1、表格 TEDS/cell F1、阅读顺序编辑距离、跨页合并准确率、chunk 覆盖率、业务字段定位命中率。
3. 用 `deepdoc` 跑全量出基线;`plain`/远程引擎做横向对照。
4. 按 §2.6 确认各候选组件许可证，输出可集成清单。

**验证:** 先跑 `python tools/eval_omnidocbench.py --validate-dataset --dataset tools/eval_datasets/biz_mini --out ./eval_out/dataset-contract.json`，再跑 `python tools/eval_omnidocbench.py --engine deepdoc --dataset tools/eval_datasets/biz_mini --out ./eval_out/baseline`；正式评测报告会内嵌 dataset contract、license gate 和 model manifest；readiness 会把 dataset contract、license gate、model manifest、页数下限和 A/B/profile 报告一起作为门禁。

**回退:** 纯新增，无需回退。

**验收:** 产出 deepdoc 各指标基线 + 各文档类分项 + 许可证清单。

---

### Sprint 1 — OCR 升级 PP-OCRv5（P0）

**目标:** PP-OCRv5（mobile ONNX）替换检测+识别，对标多语种/繁中/日文/竖排/手写收益。

**改动文件:** `common/model_store.py`（新增 `core_v5` 组或 env 切换）;HF 仓库（优先取 RapidOCR 已转 PP-OCRv5 ONNX + 字典）;`deepdoc/vision/ocr.py`（校验 `rec_image_shape`、`CTCLabelDecode`、字典对齐、det 后处理参数）;`tools/ci/verify_hf_models.py`。

**做法:** 加 `DEEPDOC_OCR_VERSION=v4|v5`（默认 v4，验证后切）;**重点校验字典对齐**（v5 字典更大，rec 输出维度须匹配）;在 Sprint 0 集对比 v4 vs v5 的 CER/WER 与每页耗时。

**验证:** `t_ocr.py` + `eval_omnidocbench.py --ocr-version v5` + `/api/v1/ocr` 冒烟。

**回退:** `DEEPDOC_OCR_VERSION=v4`。

**验收:** v5 端到端 CER/WER 优于 v4;CPU 每页耗时可接受;字典校验通过。难度:**低-中**。

---

### Sprint 2 — 版面 PP-DocLayout + 表格默认化（P1）

**目标:** 版面换 PP-DocLayout-plus（Apache，RT-DETR，23 类）;表格按评测决定是否默认 SLANet-plus + 跨页合并。

**改动文件:** `deepdoc/vision/layout_recognizer.py`（**新写 RT-DETR 后处理识别器**，对齐 23 类→deepdoc type 体系，保 `garbage_layouts` 等下游逻辑）;`common/model_store.py`（`layout_v2` 组）;HF 仓库;`deepdoc/parser/pdf_parser.py`（版面选择 env + 跨页表合并）;`table_structure_recognizer.py`/`rapid_table_recognizer.py`(默认决策);`docs/*`。

**做法:** 加 `DEEPDOC_LAYOUT_ENGINE=legacy|ppdoclayout`（默认 legacy）;**类别映射表**是关键（23 类含印章/页眉页脚/参考文献/旁注/公式编号）;表格先评测 TEDS/cell F1 稳定优于 tatr 再切默认（失败回退已有）;跨页表按列宽/列数/表头重复合并。

**验证:** `t_recognizer.py --mode layout` + `eval_omnidocbench.py --layout-engine ppdoclayout` + `eval_table.py`。

**回退:** `DEEPDOC_LAYOUT_ENGINE=legacy` / `DEEPDOC_TABLE_ENGINE=tatr`。

**验收:** 版面 block F1 / 阅读顺序改善;表格 TEDS/cell F1 提升无回归;默认变更已更新文档。难度:**中-高**。

---

### Sprint 3 — 阅读顺序 / 跨页合并（P1，高 ROI）

**目标:** 修复 XGBoost 拼接弱项，规则 + 轻量排序提升长文档结构保真度。

**改动文件:** `deepdoc/parser/pdf_parser.py`（排序与合并逻辑）。

**做法:** column clustering（多栏）;header/footer 跨页去重;标题层级参与排序;figure/table/caption 绑定;跨页段落用字号/缩进/行距/标点/hyphen/页码位置判断。规则优先于重模型;必要时引入轻量 LayoutReader 思路。当前 `DEEPDOC_READING_ORDER_STRATEGY=rules` 已实现多栏排序、重复页眉页脚去重、caption 绑定、标题/缩进跨页误合并拦截，并在文本合并后执行，避免被默认 Y 轴排序覆盖。

**验证:** `eval_omnidocbench.py` 的 block order edit distance + 跨页段落合并准确率 + chunk 覆盖率/业务字段定位命中率。layout candidate 报告必须记录 `reading_order_strategy=rules`；profile 报告必须记录 `reading_order_strategy`。

**回退:** `DEEPDOC_READING_ORDER_STRATEGY=legacy`。

**验收:** 阅读顺序编辑距离、跨页合并准确率、chunk 覆盖率和业务字段定位命中率改善。难度:**中-高**。

---

### Sprint 4 — 公式模块重选（P2）

**目标:** CPU 默认用 PP-FormulaNet-S，公式密集/高精度场景可选 UniMERNet。

**改动文件:** `deepdoc/vision/formula_recognizer.py`;`common/model_store.py`（`formula` 组更新）;HF 仓库。

**做法:** RapidLaTeXOCR vs PP-FormulaNet-S / plus-S 在论文子集做 CPU A/B（CDM + 每页耗时 + 模型体积）;UniMERNet（1530MB/CPU~8288ms）仅 `DEEPDOC_FORMULA_MODE=high_accuracy`;texify 排除（deprecated+GPL）。

**验证:** 当前 readiness 先要求公式文本级 NED/exact-match + CPU 耗时作为中间门禁；正式默认切换仍需论文子集 CDM 对比 + CPU 耗时。

**回退:** `enable_formula` 默认关闭不变;模型 env 切换。

**验收:** CPU 默认公式模型在 CDM/耗时上达平衡。难度:**中**。

---

### Sprint 5 — 性能:profiling → 批处理/线程/缓存 → INT8（P1/P2）

**目标:** 在不牺牲精度前提下提升 CPU 吞吐。**顺序不可颠倒。**

**改动文件:** 新增 `tools/profile_pipeline.py`、`tools/quantize_models.py`;`deepdoc/vision/ocr.py::load_model`（INT8 加载）;`deepdoc/parser/pdf_parser.py`（栅格化缓存、页级队列、crop batching）。

**做法:**
1. **先 profiling**:定位 OCR/layout/TSR/公式各自瓶颈。
2. **再批处理/线程**:PDF 栅格化缓存、页级任务队列、OCR crop batching、ORT/OpenVINO 线程池治理（避免多模型 oversubscription）。
3. **最后 INT8**:按模块**静态量化校准**。检测模型通常收益明显;**序列识别/解码模型（rec、公式 decoder）INT8 可能精度回退，必须逐模块校准并比对精度**，不达标则不量化该模块。当前 `tools/quantize_models.py --model-dir` 默认跳过 `rec.onnx`、`rec_v5.onnx`、`formula/decoder.onnx`，确认校准达标后才用 `--include-risky-sequence-models` 或 `--model` 单独量化。

**验证:** `profile_pipeline.py` 出瓶颈报告，报告必须记录含 `formula_mode` 的 `pipeline_config`、`model_manifest`、`license_gate`、`dataset_contract`、`reading_order_strategy`、`sample_name`、7 个阶段耗时（`rasterize_ocr`、`layout`、`table`、`text_merge`、`cross_page_text`、`reading_order`、`extract_assets`）和 `stage_summary`；profile 阶段列表只能包含这 7 个阶段，缺失或额外阶段都会失败；profile 耗时字段和 `stage_summary` 耗时/占比字段都必须是数值且有限，JSON boolean 不能作为耗时或占比字段；profile 报告自身也必须包含 `dataset_contract` 和 `license_gate`，readiness 会校验 profile 内嵌 `license_gate` 的 allowed/blocked 候选覆盖及 `license/status` 字段一致性，也会校验 profile 内嵌 `dataset_contract` 的 schema/status/sample_count/samples；`stage_summary` 需包含最慢阶段、最慢阶段耗时/占比和按耗时排序的阶段列表，按耗时排序的阶段列表必须包含每个阶段的耗时和占比，且 readiness 会校验顶层配置、阶段求和、必选阶段覆盖、额外阶段、最慢阶段摘要、每个排序阶段的耗时/占比和模型快照一致，并要求 profile 报告需对应 readiness 当前 dataset 中的 PDF。`tools/quantize_models.py --model-dir resources/models` 会递归扫描模型组子目录，为默认安全范围内的非 INT8 `.onnx` 生成 `.int8.onnx`;量化前后在 Sprint 0 集对比精度与每页耗时，序列/decoder 模型必须单独校准后再显式纳入。

profile 内嵌 `dataset_contract.samples[*].pdf_path` 若有声明，也必须与当前 readiness 数据集中的同名 PDF 一致。
正式 A/B 报告的 `samples[*].pdf_path` 和内嵌 `dataset_contract.samples[*].pdf_path` 都是必填项，缺失时 readiness 会失败，避免路径不明的旧报告混过门禁。
未传 `--dataset` 的 profile 报告也会写入 `status=failed` 的 `dataset_contract`，显式记录未绑定评测集，不能用于通过 readiness。

**回退:** `DEEPDOC_QUANT` 默认关闭;各项 env 开关。

**验收:** 吞吐提升且各模块精度损失在阈值内。难度:**中**。

---

## 5. 风险与缓解

| 风险 | 缓解 |
|---|---|
| PP-OCRv5 字典/维度不对齐 | Sprint 1 强制字典校验;env 回退 v4 |
| PP-DocLayout 23 类与下游 type 不匹配 | 显式类别映射表;legacy 默认 + 灰度 |
| SLANet-plus 未必优于 tatr | 先评测 TEDS/cell F1 再设默认;失败回退已有 |
| INT8 致序列模型精度回退 | 逐模块校准 + 精度比对，不达标不量化 |
| 评测集不代表真实分布 | 100–300 页多类真实业务 PDF |
| 引入 AGPL/GPL 组件 | §2.6 许可证门禁 |
| 默认值变更影响调用方 | env 控制 + 文档同步 |

## 6. 不做什么

- 不引入 GPU VLM 作主链路。
- 不引入 AGPL（DocLayout-YOLO）/ GPL（Marker/Surya/texify）组件。
- 不重写工程层。
- 不静默变更 API 默认行为。

## 7. 总体验收标准

1. Sprint 0 真实 PDF 评测基线可复现，含各文档类分项 + 许可证清单。
2. 各环节升级均以 **DocPilot 端到端指标**（非模型 headline）证明收益。
3. 所有替换可 env 回退;API 默认行为不变;变更处文档已同步。
4. INT8 仅在逐模块精度达标后启用。
5. 每个 Sprint 报告区分「已验证（数字/demo）」与「未验证（原因）」。

## 8. 来源与参考

- OmniDocBench: https://github.com/opendatalab/OmniDocBench
- MinerU / MinerU2.5: https://github.com/opendatalab/mineru ・ https://arxiv.org/pdf/2509.22186
- Docling: https://arxiv.org/html/2501.17887v1 ・ https://github.com/docling-project/docling
- PP-OCRv5: https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md
- PP-DocLayout-L (apache-2.0): https://huggingface.co/PaddlePaddle/PP-DocLayout-L
- PP-StructureV3: https://paddlepaddle.github.io/PaddleX/3.5/en/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html
- 公式识别 / PP-FormulaNet (PaddleX): https://paddlepaddle.github.io/PaddleX/3.1/en/module_usage/tutorials/ocr_modules/formula_recognition.html
- UniMERNet: https://arxiv.org/html/2404.15254v2
- RapidTable: https://github.com/RapidAI/RapidTable ・ RapidLaTeXOCR: https://github.com/RapidAI/RapidLatexOCR
- DocLayout-YOLO（AGPL，指标口径参考）: https://arxiv.org/html/2410.12628v1 ・ https://github.com/opendatalab/DocLayout-YOLO
- texify（deprecated, GPL）: https://github.com/VikParuchuri/texify
- ONNX Runtime 量化: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- dots.ocr 速度: https://github.com/rednote-hilab/dots.ocr/issues/103
- Mistral OCR 3: https://venturebeat.com/technology/mistral-launches-ocr-3-to-digitize-enterprise-documents-touts-74-win-rate

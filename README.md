# DocPilot Standalone

DocPilot Standalone 是一个文档解析服务，目标边界很清楚：把 PDF、Office、HTML、Markdown、TXT 等文件解析成可直接使用的 Markdown、结构化 JSON、切块结果和图片/表格/印章/公式资产。

它不是问答系统，也不负责模型回答生成。解析完成后的结构和切块可以交给上游业务系统继续处理。

## Features

- **OCR 与版面理解**：识别文字、坐标、标题、页眉页脚、表格、图片、公式、印章和二维码/条形码等结构。
- **多格式解析**：支持 PDF、DOCX、XLSX、XLS、PPTX、PPT、HTML、JSON、Markdown、TXT、CSV、RTF、ODT、EML、MSG、XML、ZIP、EPUB、CAJ。
- **多 PDF 引擎**：内置 `deepdoc`，可按需接入 `paddleocr_vl`、`mineru`、`plain`，非 PDF 可显式使用 `markitdown`。
- **EPUB 原生解析**：默认读取 OPF spine 和 XHTML 章节，按章节顺序输出标题、正文、列表、表格 blocks 和结构化 chunks；显式 `parser_engine=markitdown` 时可使用 MarkItDown fallback。
- **RTF/ODT 原生解析**：RTF 提取段落结构，ODT 直接读取 `content.xml`，输出标题、正文、列表、表格 blocks 和结构化 chunks。
- **邮件解析**：EML 走标准 MIME 解析，MSG 支持可选 `extract_msg`、OLE 属性和文本 fallback；轻量文本附件会递归展开进 blocks/chunks。
- **CAJ 转 PDF 解析**：CAJ 通过可配置 `caj2pdf` 外部转换器转成 PDF，再复用现有 PDF 解析链输出结构化产物和 chunks。
- **手写体回退**：可选开启低置信度 OCR 行的 `rec_handwriting.onnx` 回退识别，默认关闭并懒加载。
- **图片解析增强**：图片直传 `/parse` 时会输出 OCR blocks、原图 asset，并识别二维码/条形码为 `barcode` asset/block/chunk。
- **结构化产物**：可持久化 `manifest.json`、`markdown.md`、`structured.json`、`chunks.jsonl`、`ingest.jsonl` 和 `assets/`。
- **切块输出**：按结构生成 chunk，保留页码、标题路径、直接/上下文资产引用，方便业务侧复用。
- **可选发布**：解析后可把文档、chunk、资产和关联关系发布到 file、HTTP 或 PostgreSQL。
- **异步任务**：支持异步解析、任务事件流、回调投递、失败重试、取消和清理。
- **生产运维**：提供 `/health`、`/ready`、`/metrics`、`/api/v1/build-info`、self-check 和 retention janitor；Docker 部署只保留解析 API 容器。

## Installation

### Prerequisites

- Python 3.10-3.12
- Java Runtime，用于部分 Tika 解析路径
- 系统库：`libgl1`、`libglib2.0-0`

### Setup

```bash
conda create -n deepdoc python=3.10
conda activate deepdoc

pip install -e .

# Gradio 控制台
pip install -e ".[gradio]"

# S3 / MinIO artifact backend
pip install -e ".[artifact-s3]"

# PostgreSQL ingest backend
pip install -e ".[ingest-postgres]"
```

## Models

```bash
# 默认下载到 ./resources/models
python download_models.py published

# 最小 deepdoc 默认解析模型组
python download_models.py core

# 可选能力模型
python download_models.py formula
python download_models.py seal

# CPU pipeline staged upgrade groups
python download_models.py core_v5
python download_models.py layout_v2
python download_models.py table_v2
python download_models.py formula_v2

# 全部声明模型组；包含未发布的 handwriting 时需要自行提供 rec_handwriting.onnx
python download_models.py all

# 查看模型 manifest、缺失文件、OCR 字典和 rec/dict 对齐校验
python download_models.py manifest
```

模型目录默认是 `resources/models`。容器内统一使用 `/app/resources/models`。模型仓库默认是 `qwqqwq/deepdoc-standalone`；`published` 会下载该仓库中已发布的模型组：`core,core_v5,layout_v2,formula,formula_v2,seal,table_v2`。`manifest` 输出中的 `model_group_provenance` 会记录模型组来源/许可证、是否默认启用、关联开关和 readiness gate；`ocr_dictionary` 保留旧字段，报告 `ocr.res` 的 sha256、行数、唯一字符数、重复/空行和基础中文/数字/英文字符覆盖状态；`ocr_dictionaries` 会同时报告 `ocr.res` 和 `ocr_v5.res` 等已声明 OCR 字典；`ocr_recognition_alignments` 会检查 `rec.onnx`/`rec_v5.onnx` 的输出类别数是否匹配对应字典行数、空格类和 CTC blank。

CPU pipeline 现代化开关均默认保持旧行为，开启前需先下载对应模型组：

| 变量 | 默认值 | 可选值 | 说明 |
|---|---|---|---|
| `DEEPDOC_OCR_VERSION` | `v4` | `v4` / `v5` | `v5` 使用 `core_v5` 中的 `det_v5.onnx`、`rec_v5.onnx` 和 `ocr_v5.res` |
| `DEEPDOC_REC_IMAGE_SHAPE` | 空 | `C,H,W` | 通用 OCR 识别输入尺寸覆盖，空值保持默认 `3,48,320` |
| `DEEPDOC_OCR_V4_REC_IMAGE_SHAPE` | 空 | `C,H,W` | v4 OCR 识别输入尺寸覆盖，优先于 `DEEPDOC_REC_IMAGE_SHAPE` |
| `DEEPDOC_OCR_V5_REC_IMAGE_SHAPE` | 空 | `C,H,W` | v5 OCR 识别输入尺寸覆盖，优先于 `DEEPDOC_REC_IMAGE_SHAPE` |
| `DEEPDOC_LAYOUT_ENGINE` | `legacy` | `legacy` / `ppdoclayout` | `ppdoclayout` 使用 `layout_v2` 中的 PP-DocLayout-plus RT-DETR 后端 |
| `DEEPDOC_TABLE_ENGINE` | `tatr` | `tatr` / `rapidtable` | `rapidtable` 使用 `table_v2` 中的 RapidTable / SLANet-plus 后端；单表失败回退 `tatr` |
| `DEEPDOC_FORMULA_MODE` | `rapidlatex` | `rapidlatex` / `pp_formula_net_s` | `pp_formula_net_s` 使用 PaddleX PP-FormulaNet-S 适配器，需安装 `.[formula-v2]` 并预置 `formula_v2` 模型组 |
| `DEEPDOC_READING_ORDER_STRATEGY` | `legacy` | `legacy` / `rules` | `rules` 启用多栏排序、重复页眉页脚去重、caption 绑定和更严格跨页段落误合并拦截 |

示例：

```bash
python download_models.py core_v5
DEEPDOC_OCR_VERSION=v5 python main.py

python download_models.py layout_v2
DEEPDOC_LAYOUT_ENGINE=ppdoclayout python main.py

python download_models.py table_v2
DEEPDOC_TABLE_ENGINE=rapidtable python main.py
```

评测与 profiling 工具：

```bash
python tools/eval_omnidocbench.py --license-gate --out eval_out/license-gate.json
python tools/eval_omnidocbench.py --validate-dataset --dataset tools/eval_datasets/biz_mini --out eval_out/dataset-contract.json
python tools/eval_omnidocbench.py --engine deepdoc --dataset tools/eval_datasets/biz_mini --out ./eval_out/baseline
python tools/eval_omnidocbench.py --engine deepdoc --dataset tools/eval_datasets/biz_mini --layout-engine ppdoclayout --reading-order-strategy rules --out ./eval_out/layout-ppdoclayout-rules.json
python tools/check_cpu_pipeline_readiness.py \
  --dataset tools/eval_datasets/biz_mini \
  --min-pages 100 \
  --license-gate-report eval_out/license-gate.json \
  --ocr-baseline-report eval_out/ocr-v4.json \
  --ocr-candidate-report eval_out/ocr-v5.json \
  --layout-baseline-report eval_out/layout-legacy.json \
  --layout-candidate-report eval_out/layout-ppdoclayout-rules.json \
  --table-baseline-report eval_out/table-tatr.json \
  --table-candidate-report eval_out/table-rapidtable.json \
  --formula-baseline-report eval_out/formula-rapidlatex.json \
  --formula-candidate-report eval_out/formula-pp_formula_net_s.json \
  --profile-report eval_out/profile.json
python tools/profile_pipeline.py tools/eval_datasets/biz_mini/contracts/sample.pdf --dataset tools/eval_datasets/biz_mini --layout-engine legacy --ocr-version v4 --formula-mode rapidlatex --reading-order-strategy legacy --out eval_out/profile.json
```

`eval_omnidocbench.py --validate-dataset` 会输出 `2026-06-08.cpu-pipeline-dataset-contract.v1` 报告，用于在真实 A/B 前校验评测集标注合同：PDF 必须存在且非空，标注文件必须能匹配同名 PDF，blocks/chunks/fields/formulas/table HTML 等 ground truth 必须具备可评测字段。该工具会递归发现 `--dataset` 下的 PDF；子目录样本名使用 dataset-relative 路径，例如 `contracts/sample`，避免不同类别目录中的同名 PDF 冲突。正式 `--engine deepdoc --dataset ...` 评测也会先执行 dataset contract 预检，失败时不会进入 PDF 解析；成功时评测报告会包含 `dataset_contract`。`check_cpu_pipeline_readiness.py` 会把这份同源校验结果写入 `dataset_contract`，若状态失败会追加 `dataset_contract_failed` 门禁。

`check_cpu_pipeline_readiness.py` 会检查 `core_v5`、`layout_v2`、`table_v2`、`formula_v2` 模型组、dataset contract、license gate 报告、真实 PDF 页数下限，以及 OCR/layout/table/formula 的 baseline/candidate 成对 A/B 报告和 profile 报告是否齐全。license gate 报告必须来自 `python tools/eval_omnidocbench.py --license-gate --out eval_out/license-gate.json`，schema 为 `2026-06-08.cpu-pipeline-license-gate.v1`，且状态为 `passed`；该门禁用于阻止 AGPL/GPL 候选组件进入本地解析主链路。升级模型组候选会从 `MODEL_GROUP_PROVENANCE` 派生，避免模型来源/许可证声明和 license gate 手工清单漂移。A/B 报告必须声明 `engine=deepdoc`，若 `samples` 明细中声明了 `engine` 也必须是 `deepdoc`，避免 plain 或外部引擎报告混入本地 DocPilot CPU pipeline 门禁。A/B 报告自身也必须包含 `license_gate`，readiness 会校验 A/B 报告内嵌 `license_gate` 的 allowed/blocked 候选覆盖及 `license/status` 字段一致性，避免旧评测报告缺少许可证上下文或字段漂移却误过门禁。A/B 报告自身也必须包含 `dataset_contract`，readiness 会校验 A/B 报告内嵌 `dataset_contract` 的 schema/status/sample_count/samples，确保报告自带的评测集合同与当前 readiness 数据集一致。正式评测报告和 profile 报告还会包含 `model_manifest`（schema `2026-06-08.cpu-pipeline-model-manifest.v1`），记录评测或 profiling 时的模型目录、模型组、每个声明模型文件的存在性、大小和 sha256；model_manifest 会按报告的 `pipeline_config` 只记录相关模型组，readiness 会按报告的 `pipeline_config` 校验对应模型组快照是否与当前 `--model-root` 一致，例如 OCR baseline 对 `core`，OCR candidate 对 `core_v5`，layout candidate/profile 对 `layout_v2`，table candidate/profile 对 `table_v2`，formula candidate/profile 对 `formula_v2`，避免模型文件换版后复用旧 A/B 报告或旧 profiling 报告。A/B 报告必须与本次 readiness 的 `--dataset` 一致，baseline/candidate 也必须使用同一数据集和相同 `sample_count`，且 `sample_count` 必须是正整数并等于 dataset contract 识别到的样本数；A/B 报告必须包含 `samples` 明细，`summary.sample_count` 还必须是正整数并等于 `len(samples)`，summary 中由 samples 明细产生的均值指标必须在每个 sample 中都有对应指标值，summary 和 samples 指标都必须是数值且有限，JSON boolean 不能作为数值字段，且 summary 均值必须按全量 samples 明细计算并一致，samples 明细里的样本名和声明的 `pdf_path` 也必须匹配 dataset contract，避免混用旧评测报告、只用子集报告或摘要/明细不一致的报告误过门禁。报告里的 `pipeline_config` 也会被校验，例如 OCR baseline 必须是 `ocr_version=v4`，candidate 必须是 `ocr_version=v5`；layout baseline 必须是 `layout_engine=legacy` 且 `reading_order_strategy=legacy`，candidate 必须是 `layout_engine=ppdoclayout` 且 `reading_order_strategy=rules`；table baseline 必须是 `table_engine=tatr`，candidate 必须是 `table_engine=rapidtable`。layout A/B 报告还必须包含 `mean_cross_page_merge_accuracy`、`mean_chunk_text_coverage` 和 `mean_business_field_location_hit_rate`，且 candidate 不得低于 baseline；table A/B 报告必须包含 `mean_table_teds` 和 `mean_table_cell_f1`，且 candidate 不得低于 baseline；formula A/B 报告必须包含 `mean_formula_normalized_edit_distance`、`mean_formula_exact_match_rate` 和 `mean_elapsed_seconds`，且 candidate 不得低于 baseline 质量或高于 baseline 耗时。profile 报告必须记录 `pipeline_config`（含 `formula_mode`）、`model_manifest`、`license_gate`、`reading_order_strategy`、7 个阶段耗时（`rasterize_ocr`、`layout`、`table`、`text_merge`、`cross_page_text`、`reading_order`、`extract_assets`）和 `stage_summary` 瓶颈摘要；profile 阶段列表只能包含这 7 个阶段，缺失或额外阶段都会失败；profile 报告自身也必须包含 `dataset_contract`，并且必须包含 `license_gate`；readiness 会校验 profile 内嵌 `license_gate` 的 allowed/blocked 候选覆盖及 `license/status` 字段一致性，也会校验 profile 内嵌 `dataset_contract` 的 schema/status/sample_count/samples；profile 顶层 `dataset` 必须匹配本次 readiness 的 `--dataset`，其 `sample_name` 和 `pdf_path` 必须对照 dataset contract 指向同一条 PDF 样本，避免拿旧数据集、临时单页样本或错配样本的耗时报告混过门禁；`stage_summary` 包含 `slowest_stage`、最慢阶段耗时/占比和按耗时排序的阶段列表，按耗时排序的阶段列表必须包含每个阶段的耗时和占比，readiness 会校验顶层配置、总耗时、阶段求和、必选阶段覆盖、额外阶段、最慢阶段摘要、每个排序阶段的耗时/占比和模型快照一致。当前环境缺真实权重、license gate 报告或真实评测集时应返回 `failed`，避免把脚手架误判为可切默认。

profile 内嵌 `dataset_contract.samples[*].pdf_path` 若有声明，也必须与当前 readiness 数据集中的同名 PDF 一致，避免复用旧数据集路径的 profile 报告。
readiness 会校验 profile 内嵌 `dataset_contract` 的 schema/status/sample_count/samples。
内嵌 `dataset_contract.samples` 必须是数组，且每一项都必须是 JSON object，避免非结构化评测集合同绕过样本名和路径校验。
profile 报告也必须对应本次 readiness 的 `--dataset`，且顶层 `dataset` 字段缺失或不匹配时 readiness 会失败；profile 顶层 `sample_name` 和 `pdf_path` 也必须指向 dataset contract 的同一样本。
正式 A/B 报告的 `samples[*].pdf_path` 和内嵌 `dataset_contract.samples[*].pdf_path` 都是必填项，缺失时 readiness 会失败，避免路径不明的旧报告混过门禁。
正式 A/B 报告的 `samples` 必须是数组，且每一项都必须是 JSON object，避免非结构化明细绕过样本名、路径、引擎和指标校验。
正式 A/B 报告中，只要 `summary` 声明了由 samples 明细产生的均值指标，每个 sample 都必须提供对应指标值；readiness 会按全量 samples 明细重算均值，缺失、非数值/非有限数值、JSON boolean 或摘要/明细不一致都会失败。
未传 `--dataset` 的 profile 报告也会写入 `status=failed` 的 `dataset_contract`，显式记录未绑定评测集，不能用于通过 readiness。
profile 报告中的 `total_elapsed_seconds`、`stages[*].elapsed_seconds`、`stage_summary.slowest_stage_elapsed_seconds`、`stage_summary.slowest_stage_share`、`stage_summary.stages_by_elapsed_seconds[*].elapsed_seconds` 和 `stage_summary.stages_by_elapsed_seconds[*].share` 都必须是数值且有限，`stage_summary.stages_by_elapsed_seconds[*]` 的 `share` 必须匹配对应阶段耗时占比，JSON boolean 不能作为耗时或占比字段，避免 `NaN` / `Infinity` 或 `true` / `false` 耗时报告混过 readiness。

`eval_omnidocbench.py` 的样本目录约定：

| 文件 | 用途 | 输出指标 |
|---|---|---|
| `<name>.pdf` | 必需，待评测 PDF | `elapsed_seconds`、`text_length` |
| `<name>.gt.txt` | 可选，全文文本标注 | `character_error_rate`、`word_error_rate`、`text_normalized_edit_distance` |
| `<name>.gt.blocks.json` | 可选，结构块标注；可为 block 数组或含 `blocks` 字段的 structured JSON | `block_type_f1`、`reading_order_normalized_edit_distance`、`cross_page_merge_accuracy` |
| `<name>.gt.tables.html` 或 `<name>.gt.html` | 可选，表格 HTML 标注 | `mean_table_teds`、`mean_table_cell_f1` |
| `<name>.gt.formulas.json` | 可选，公式 LaTeX 标注；可为数组或含 `formulas` / `equations` / `expected_formulas` 字段的 JSON | `mean_formula_normalized_edit_distance`、`mean_formula_exact_match_rate` |
| `<name>.gt.chunks.json` | 可选，业务可复用 chunk 文本标注；可为数组或含 `chunks` 字段的 JSON | `chunk_text_coverage` |
| `<name>.gt.fields.json` | 可选，业务字段标注；可为数组或含 `fields` 字段的 JSON，字段项支持 `name`、`value`、`page_numbers` | `business_field_location_hit_rate` |

这些工具只覆盖文档解析 pipeline 的质量、许可证和耗时检查，不做问答、向量化或回答生成。

需要启用 ONNX INT8 动态量化时，可先生成量化模型：

```bash
python tools/quantize_models.py --model-dir resources/models
```

该脚本会递归扫描 `resources/models` 及 `layout/`、`formula/`、`table/` 等模型组子目录，为非 INT8 `.onnx` 模型生成同名 `.int8.onnx` 文件，不覆盖原始模型。默认批量扫描会跳过高风险序列/decoder 模型（如 `rec.onnx`、`rec_v5.onnx`、`formula/decoder.onnx`），这些模型需要先做逐模块精度校准；确认达标后再显式传 `--include-risky-sequence-models`，或用 `--model path/to/model.onnx` 单独量化。服务默认 `DEEPDOC_QUANT=fp32`，继续加载原始模型；显式设置 `DEEPDOC_QUANT=int8` 后加载 `.int8.onnx`。如果量化模型不存在，服务会直接报错而不是静默回退。INT8 动态量化路径当前固定使用 CPU Execution Provider，适合 CPU 推理或 GPU 不稳定时的轻量化部署。

## Run

### API Service

```bash
export DEEPDOC_MODEL_PATH=./resources/models
python main.py
```

开发调试也可以使用：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Gradio Console

```bash
python gradio_app.py
```

默认监听 `0.0.0.0:7860`。Gradio 只作为本地 Python 调试入口，不进入默认 Docker 部署。

## API Examples

### OCR

```bash
curl -X POST "http://localhost:8000/api/v1/ocr" \
  -F "file=@/path/to/image.png"
```

### Synchronous Parse

```bash
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "file=@/path/to/document.pdf"
```

### Synchronous Parse Stream

```bash
curl -N -X POST "http://localhost:8000/api/v1/parse/stream" \
  -F "file=@/path/to/document.pdf" \
  -F "return_structured=true" \
  -F "include_chunks=true"
```

该端点返回 `text/event-stream`，事件包括 `start`、`file_started`、`file_completed` 和 `done`，用于同步解析时获取文件级进度和最终解析结果。

### Structured Parse With Chunks

```bash
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "file=@/path/to/document.pdf" \
  -F "return_structured=true" \
  -F "persist_artifacts=true" \
  -F "include_chunks=true"
```

返回结果里会包含 `parse_id`、`document_id`、`artifact_urls`、`asset_count`、`chunk_count`，并在持久化目录生成：

- `manifest.json`
- `markdown.md`
- `structured.json`
- `chunks.jsonl`
- `ingest.jsonl`
- `publish-events.jsonl`
- `assets/`

### PDF Engine Selection

```bash
# DocPilot auto mode: native text PDFs use text layer, scanned PDFs use OCR + Layout
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "parser_engine=deepdoc" \
  -F "deepdoc_pdf_mode=auto" \
  -F "return_structured=true" \
  -F "file=@/path/to/document.pdf"

# DocPilot force OCR + Layout
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "parser_engine=deepdoc" \
  -F "deepdoc_pdf_mode=ocr" \
  -F "file=@/path/to/document.pdf"

# PaddleOCR-VL
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "parser_engine=paddleocr_vl" \
  -F "compute_device=gpu" \
  -F "file=@/path/to/document.pdf"

# MinerU
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "parser_engine=mineru" \
  -F "compute_device=cpu" \
  -F "return_images=false" \
  -F "strict_text=true" \
  -F "file=@/path/to/document.pdf"

# Plain PDF text layer
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "parser_engine=plain" \
  -F "file=@/path/to/document.pdf"
```

### Async Parse

```bash
curl -X POST "http://localhost:8000/api/v1/parse/async" \
  -F "file=@/path/to/document.pdf" \
  -F "parser_engine=deepdoc" \
  -F "return_structured=true" \
  -F "persist_artifacts=true" \
  -F "include_chunks=true" \
  -F "publish_ingest=true" \
  -F "callback_url=http://your-app.internal/deepdoc/callback" \
  -F "callback_events=terminal" \
  -F "callback_secret=replace-me"
```

任务查询与运维：

```bash
curl "http://localhost:8000/api/v1/tasks/<task_id>"
curl "http://localhost:8000/api/v1/tasks/<task_id>/events"
curl -N "http://localhost:8000/api/v1/tasks/<task_id>/stream?include_callback_events=true"
curl "http://localhost:8000/api/v1/tasks/<task_id>/callback-events"
curl -X POST "http://localhost:8000/api/v1/tasks/<task_id>/cancel"
curl -X POST "http://localhost:8000/api/v1/tasks/<task_id>/retry" \
  -H "Content-Type: application/json" \
  -d '{"force":true,"copy_callback":true,"requested_by":"ops"}'
curl -X POST "http://localhost:8000/api/v1/tasks/cleanup" \
  -H "Content-Type: application/json" \
  -d '{"dry_run":true,"older_than_days":7,"keep_latest":100,"status":"succeeded"}'
```

### Artifact Access

```bash
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/manifest"
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/structured"
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/markdown"
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/chunks?asset_url_mode=proxy"
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/ingest?asset_url_mode=proxy"
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/publish-events"
curl "http://localhost:8000/api/v1/artifacts?limit=20"
```

Artifact 清理与发布重试：

```bash
curl -X POST "http://localhost:8000/api/v1/artifacts/<parse_id>/publish" \
  -H "Content-Type: application/json" \
  -d '{"force":true,"requested_by":"retry","asset_url_mode":"proxy"}'

curl -X POST "http://localhost:8000/api/v1/artifacts/publish-retry" \
  -H "Content-Type: application/json" \
  -d '{"limit":50,"scan_limit":250,"only_due":true,"requested_by":"retry-batch"}'

curl -X POST "http://localhost:8000/api/v1/artifacts/cleanup" \
  -H "Content-Type: application/json" \
  -d '{"dry_run":true,"older_than_days":30,"keep_latest":100}'
```

### Ingest Query

PostgreSQL ingest backend 保存的是解析结果、chunk、资产和 chunk-asset 关系。查询接口用于运维排查和业务侧读取解析结构。

```bash
curl "http://localhost:8000/api/v1/ingest/stats?include_breakdown=true"
curl "http://localhost:8000/api/v1/ingest/documents?limit=20"
curl "http://localhost:8000/api/v1/ingest/documents/<parse_id>"
curl "http://localhost:8000/api/v1/ingest/records?q=architecture&mode=text"
curl "http://localhost:8000/api/v1/ingest/assets?parse_id=<parse_id>&limit=20"
curl "http://localhost:8000/api/v1/ingest/chunks?parse_id=<parse_id>&limit=20"
curl "http://localhost:8000/api/v1/ingest/chunk-asset-links?parse_id=<parse_id>&relation_type=direct"
```

`/api/v1/ingest/records` 只接受 `mode=text`。服务边界是文档解析和结构化输出，不提供问答检索或回答生成接口。

### OpenAPI

```bash
curl "http://localhost:8000/openapi.json"
open http://localhost:8000/docs
```

## Request Parameters

常用表单参数：

- `parser_engine`: `deepdoc`、`paddleocr_vl`、`mineru`、`plain`；非 PDF 可显式传 `markitdown`
- `compute_device`: `gpu` 或 `cpu`，用于远程解析引擎
- `return_images`: 是否在 Markdown 中保留图片内容
- `strict_text`: 是否剥离 HTML 标签并输出更严格的纯文本 Markdown
- `enable_formula`: deepdoc 引擎公式识别，默认关闭
- `enable_seal`: deepdoc 引擎印章识别，默认关闭
- `return_structured`: 响应中返回结构化 JSON
- `persist_artifacts`: 持久化解析产物
- `persist_source`: 持久化原始上传文件
- `include_chunks`: 生成 chunk；`return_structured` 或 `persist_artifacts` 启用时通常一起启用
- `chunk_max_tokens`: chunk 最大 token 数，默认 `800`
- `chunk_overlap_tokens`: chunk 重叠 token 数，默认 `120`
- `publish_ingest`: 解析完成后发布 ingest 记录；启用时会强制持久化 artifact
- `reuse_artifacts`: 按 `artifact_key` 复用已有解析产物
- `tenant_id`: 租户 ID；JWT 模式下一般由 token claim 提供，admin token 可显式覆盖

## Authentication And Tenant Scope

静态 key：

```bash
export SECRET_ACCESS_KEY=change-me
```

JWT HS256：

```bash
export DEEPDOC_AUTH_MODE=jwt_hs256
export DEEPDOC_AUTH_JWT_SECRET=change-me-too
export DEEPDOC_AUTH_ADMIN_SCOPES=admin,artifacts:admin
export DEEPDOC_DEFAULT_TENANT_ID=default
```

JWT payload 中的 `tenant_id` 会传递到 parse artifact metadata、artifact 访问控制、异步任务、审计事件、PostgreSQL ingest 查询、request guard 配额分桶和 JSON 日志。API key 或匿名模式可用 `X-Tenant-ID` 请求头指定租户，也可通过 `tenant_id` query/form/body 参数传入；JWT 模式下一般由 token claim 提供租户，非 admin 不允许覆盖 token tenant，admin scope 可以显式覆盖 `tenant_id`。

## Artifact Backend

默认 backend 为 local，目录是 `resources/artifacts`。

S3 / MinIO / compatible object storage：

```bash
pip install -e ".[artifact-s3]"
export DEEPDOC_ARTIFACT_BACKEND=s3
export DEEPDOC_ARTIFACT_BUCKET=deepdoc-artifacts
export DEEPDOC_ARTIFACT_PREFIX=prod
export DEEPDOC_ARTIFACT_ENDPOINT_URL=http://minio:9000
export DEEPDOC_ARTIFACT_REGION=us-east-1
export DEEPDOC_ARTIFACT_ACCESS_KEY_ID=minioadmin
export DEEPDOC_ARTIFACT_SECRET_ACCESS_KEY=minioadmin
export DEEPDOC_ARTIFACT_ADDRESSING_STYLE=path
export DEEPDOC_ARTIFACT_PUBLIC_BASE_URL=https://cdn.example.com/deepdoc-artifacts
```

## Ingest Publishing

File sink：

```bash
export DEEPDOC_INGEST_PUBLISHER=file
export DEEPDOC_INGEST_FILE_PATH=/data/deepdoc/published-ingest.jsonl
```

HTTP sink：

```bash
export DEEPDOC_INGEST_PUBLISHER=http
export DEEPDOC_INGEST_HTTP_URL=https://ingest.example.com/v1/documents
export DEEPDOC_INGEST_HTTP_AUTH_HEADER=Authorization
export DEEPDOC_INGEST_HTTP_AUTH_TOKEN='Bearer xxxxx'
export DEEPDOC_INGEST_HTTP_MAX_ATTEMPTS=3
export DEEPDOC_INGEST_HTTP_BACKOFF_SECONDS=1
export DEEPDOC_INGEST_HTTP_MAX_BACKOFF_SECONDS=8
export DEEPDOC_INGEST_HTTP_RETRY_STATUS_CODES=429,500,502,503,504
export DEEPDOC_INGEST_RETRY_BASE_DELAY_SECONDS=60
export DEEPDOC_INGEST_RETRY_MAX_DELAY_SECONDS=3600
export DEEPDOC_INGEST_RETRY_MAX_FAILURES=5
```

PostgreSQL sink：

```bash
pip install -e ".[ingest-postgres]"
export DEEPDOC_INGEST_PUBLISHER=postgres
export DEEPDOC_INGEST_PG_DSN=postgresql://user:password@postgres:5432/deepdoc
export DEEPDOC_INGEST_PG_SCHEMA=deepdoc_ingest
export DEEPDOC_INGEST_PG_CONNECT_TIMEOUT=10
```

PostgreSQL schema 包含：

- `documents`: 文档级 manifest 和发布状态
- `records`: 扁平文本记录
- `chunks`: 结构化 chunk
- `assets`: 图片、表格、印章、公式等资产
- `chunk_asset_links`: chunk 与资产的直接/上下文关系
- `parse_aliases`: artifact 复用时的新旧 `parse_id` 映射

## Operations

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/ready"
curl "http://localhost:8000/api/v1/build-info"
curl "http://localhost:8000/metrics"
```

外部响应默认会脱敏内部文件路径、本地 URL、callback 目标地址、DSN、队列名和内部任务存储路径。公开业务字段如 `parse_id`、`task_id`、`structured_url`、`assets_url_prefix`、`asset_refs`、`asset_urls` 会保留。

需要完整内部诊断信息时，只能在 admin 鉴权模式下使用 `include_internal=1`，或显式设置：

```bash
export DEEPDOC_API_EXPOSE_INTERNALS=1
```

### Self-Check

```bash
curl -X POST "http://localhost:8000/api/v1/self-checks/run" \
  -H "Content-Type: application/json" \
  -d '{"suite":"core","force_reparse":false,"force_republish":true}'
curl "http://localhost:8000/api/v1/self-checks?limit=5"
curl "http://localhost:8000/api/v1/self-checks/<check_id>"
curl -X POST "http://localhost:8000/api/v1/self-checks/cleanup" \
  -H "Content-Type: application/json" \
  -d '{"dry_run":true,"older_than_days":7,"keep_latest":20,"status":"passed"}'
```

### Retention Janitor

后台 janitor 可清理：

- async tasks
- persisted artifacts
- ops audit events
- production self-check results

常用配置：

- `DEEPDOC_RETENTION_JANITOR_ENABLED`
- `DEEPDOC_RETENTION_JANITOR_REQUIRED_FOR_READY`
- `DEEPDOC_RETENTION_JANITOR_POLL_SECONDS`
- `DEEPDOC_RETENTION_JANITOR_TASKS_*`
- `DEEPDOC_RETENTION_JANITOR_ARTIFACTS_*`
- `DEEPDOC_RETENTION_JANITOR_AUDIT_EVENTS_*`
- `DEEPDOC_RETENTION_JANITOR_SELF_CHECKS_*`

### Request Protection

- `DEEPDOC_RATE_LIMIT_ENABLED`
- `DEEPDOC_RATE_LIMIT_BACKEND`
- `DEEPDOC_RATE_LIMIT_GENERAL`
- `DEEPDOC_RATE_LIMIT_PARSE`
- `DEEPDOC_RATE_LIMIT_PARSE_BYTES`
- `DEEPDOC_RATE_LIMIT_ARTIFACT`
- `DEEPDOC_RATE_LIMIT_INGEST`
- `DEEPDOC_MAX_INFLIGHT_PARSE`
- `DEEPDOC_MAX_INFLIGHT_ARTIFACT`
- `DEEPDOC_MAX_INFLIGHT_INGEST`

### Structured Logs

默认保留文本日志。生产环境可设置：

```bash
export DEEPDOC_LOG_FORMAT=json
```

启用后 console 和 `resources/logs/` 文件日志输出 JSONL，包含 `timestamp`、`level`、`message`、source、trace 字段；请求内日志会带 `request_id`，解析文件时会带 `file_sha` 和 `engine`。

### Tracing

```bash
export DEEPDOC_TRACING_ENABLED=1
export DEEPDOC_TRACING_EXPORTER=otlp
export DEEPDOC_TRACING_OTLP_ENDPOINT=http://observability.example.com:4318/v1/traces
export DEEPDOC_TRACING_SERVICE_NAME=deepdoc-standalone
export DEEPDOC_TRACING_SAMPLE_RATIO=1.0
```

本地无 collector 时可输出到 console：

```bash
export DEEPDOC_TRACING_ENABLED=1
export DEEPDOC_TRACING_EXPORTER=console
```

## Docker Deployment

单服务：

```bash
docker build -f Dockerfile.cpu -t deepdoc-service:v1 .
docker run -d \
  -p 8000:8000 \
  -e DEEPDOC_MODEL_PATH=/app/resources/models \
  -e DEEPDOC_AUTO_DOWNLOAD=1 \
  -e DEEPDOC_DOWNLOAD_GROUPS=published \
  -v ./resources/models:/app/resources/models \
  deepdoc-service:v1
```

Compose：

```bash
docker compose up -d
```

`docker-compose.yml` 只启动 `deepdoc` API 服务，并使用 `Dockerfile.cpu`。

需要 GPU 镜像时只切换 Dockerfile 和 provider：

```bash
DEEPDOC_DOCKERFILE=Dockerfile.gpu DEEPDOC_ONNX_PROVIDER=auto docker compose up -d --build
```

也可以直接构建两个明确入口：

```bash
docker build -f Dockerfile.cpu -t deepdoc-standalone-cpu:0.1 .
docker build -f Dockerfile.gpu -t deepdoc-standalone-gpu:0.1 .
```

本地 compose 只包含：

- `deepdoc`: API service

不会启动 Gradio、数据库、队列、worker 或观测平台容器。异步任务、外部 ingest、对象存储等仍是代码层可选能力，但不属于默认 Docker 部署。

生产环境也复用同一个 `.env` 和同一个镜像入口。直接运行已构建的 CPU 镜像：

```bash
docker run -d \
  --name deepdoc \
  --env-file .env \
  -p 8000:8000 \
  -e DEEPDOC_MODEL_PATH=/app/resources/models \
  -v "$(pwd)/resources/models:/app/resources/models" \
  -v "$(pwd)/resources/artifacts:/app/resources/artifacts" \
  -v "$(pwd)/resources/logs:/app/resources/logs" \
  -v "$(pwd)/resources/temp:/app/resources/temp" \
  deepdoc-standalone-cpu:0.1
```

GPU 主机只换 GPU 镜像并显式打开 GPU provider：

```bash
docker run -d \
  --name deepdoc \
  --gpus all \
  --env-file .env \
  -p 8000:8000 \
  -e DEEPDOC_MODEL_PATH=/app/resources/models \
  -e DEEPDOC_ONNX_PROVIDER=auto \
  -v "$(pwd)/resources/models:/app/resources/models" \
  -v "$(pwd)/resources/artifacts:/app/resources/artifacts" \
  -v "$(pwd)/resources/logs:/app/resources/logs" \
  -v "$(pwd)/resources/temp:/app/resources/temp" \
  deepdoc-standalone-gpu:0.1
```

如果模型仓库需要 token：

```bash
export HF_TOKEN=...
docker compose up -d --build
```

也可以先在宿主机预置模型后关闭自动下载：

```bash
python download_models.py published
export DEEPDOC_AUTO_DOWNLOAD=0
docker compose up -d
```

## Publish Model Pack To Hugging Face

目标模型仓库默认是：

- `qwqqwq/deepdoc-standalone`

生成 manifest：

```bash
python tools/publish_models_to_hf.py --groups core --write-manifest-only
```

发布模型包：

```bash
export HF_TOKEN=<your_token>
python tools/publish_models_to_hf.py --groups all
```

校验远端文件：

```bash
export HF_TOKEN=<your_token>
python tools/ci/verify_hf_models.py --groups all
```

该校验会比对远端 manifest、文件大小、本地 sha256，并要求远端 manifest 的 `model_group_provenance` 与本地模型组来源/许可证声明一致；本地模型存在时还会输出 `ocr_dictionary` 字典覆盖报告。默认模型仓库是受限仓库时必须提供 `HF_TOKEN`。

## Notes

1. 本项目的服务边界是文档解析、结构化产物、切块、资产和可选发布。
2. 默认模型目录是 `resources/models`，容器内是 `/app/resources/models`。
3. GPU 镜像在 x86_64 Linux 上安装 `onnxruntime-gpu`，宿主机仍需可用 CUDA 驱动。
4. 部分非 PDF 解析路径依赖 Tika，请确保 `java` 在系统 PATH 中。
5. 当前仓库没有标准 pytest 套件，推荐用 `python -m unittest`、`python -m py_compile`、compose config 和 API 冒烟测试验证。

## License

Apache 2.0

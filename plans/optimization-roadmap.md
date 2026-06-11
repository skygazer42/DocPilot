# DocPilot Standalone 文档解析优化路线图

## Context

DocPilot Standalone 是一个文档解析与结构化产物服务，当前边界是：

- 把 PDF、Office、HTML、Markdown、TXT 等文件解析为 Markdown、结构化 JSON、chunks 和资产引用。
- 保留本地 `deepdoc`、远程 `paddleocr_vl` / `mineru`、本地 `plain` 等解析引擎。
- 支持异步任务、artifact 持久化、ingest 发布、metrics、tracing、health/ready 等代码层运维能力。

本路线图只围绕“文档解析完成后的结构和切块可直接使用”展开。服务不规划问答、向量化、回答生成或外部模型对话链路。

## 现状关键发现

| 维度 | 现状 | 缺口 |
|---|---|---|
| OCR | DBNet + CRNN，`ocr.res` 已有 sha256 和基础字符覆盖校验；公式、印章、二维码和低置信度手写体回退已作为可选解析增强接入 | 复杂低质扫描仍需按业务样本持续评测 |
| 推理 | 默认 CPU `onnxruntime`，GPU 镜像使用 `onnxruntime-gpu`；OCR/Layout ONNX 模型已支持可选 INT8 动态量化加载、显式 TensorRT EP、OCR rec width bucket 动态聚批，Layout 可合并同批多页推理 | TensorRT 仍需宿主 CUDA/TensorRT provider 验证，不作为 Docker 镜像入口 |
| 解析 | deepdoc PDF 已可自动判别原生文本层/扫描件；结果缓存、同步文件级 SSE、OCR/Layout 重组件共享和跨页段落/表格合并已接入 | 复杂 PDF 的解析策略还可继续增强 |
| 格式 | pdf/docx/xlsx/pptx/html/md/txt/json/csv/tsv/rtf/odt/eml/msg/epub/caj/png/jpg/jpeg/bmp/tiff/webp | 长尾格式 CAJ 已通过外部转换器接入；图片已接入 `/parse` OCR+Layout 链 |
| 结构化产物 | 已有 blocks / assets / chunks / ingest records；chunk 策略、资产上下文、本地规则资产摘要和跨页合并 metadata 已可消费 | 长尾格式的结构化一致性还可继续增强 |
| 基础设施 | gevent 单进程 + 文本/JSON 日志；Prometheus metrics、health/ready、Swagger UI、OpenTelemetry tracing、request guard、限流/配额、审计日志、异步 worker、S3 artifact backend 和 CORS 白名单已接入 | 仍需持续强化错误码/i18n、模型性能和长尾格式 |

---

## 2026-06-08 已落地进展

- 问答、向量化和回答生成相关入口已从当前产品边界移除，当前服务边界收敛为“文档解析 → 结构化产物 → chunks / ingest records / assets”。
- `chunk_strategy` 已支持从 API 参数或 `DEEPDOC_CHUNK_STRATEGY` 配置，当前策略为 `structure_aware`、`page_aware`、`asset_aware`。
- `asset_aware` 已让 table / figure / equation / seal 等资产块独立成 chunk；`page_aware` 已避免已知页码的 block 跨页混合。
- 资产上下文已写入 `assets[].metadata`，包括直接 block、邻近正文 block、相关 chunk 和上下文文本，业务系统可从 assets 或 chunks 双向追溯。
- 资产本地规则摘要已写入 `assets[].metadata`，包括 `asset_summary`、`asset_summary_source=local_rules` 和 `asset_summary_facts`，覆盖表格行列数、图片尺寸、页码和文本长度等基础 facts。
- OCR 字典校验已接入模型 manifest 和 HF 模型校验工具，`ocr.res` 会输出 sha256、行数、唯一字符数、重复/空行和基础中文/数字/英文字符覆盖报告。
- deepdoc PDF 解析已支持可选公式识别和印章识别：`enable_formula=true` 会在 Equation 区域输出 LaTeX Markdown，`enable_seal=true` 会调用本地 seal_det 并把识别文本写回解析 boxes。
- OCR 手写体回退已接入且默认关闭：`DEEPDOC_HANDWRITING_FALLBACK=1` 时，低于阈值的 rec 结果会懒加载 `rec_handwriting.onnx` 复识别，只有手写分数更高且文本非空时才替换主识别结果。
- deepdoc PDF 默认 `deepdoc_pdf_mode=auto`，会用 `pdfplumber.page.chars` 的字符数量、字体覆盖率和有文本页比例判别原生文本层；原生文本 PDF 可走 native text 快路径输出 `pdf_native_text` blocks/chunks，扫描件或视觉能力请求继续走 OCR + Layout。
- 结果缓存已通过 `artifact_key = sha256(file)+artifact_profile` 和 `reuse_artifacts=true` 接入；同一文件、同一解析配置命中时复用已持久化 artifact，不重复跑解析流程。
- `chunks.jsonl` 和 `ingest.jsonl` 已写入导出 schema 版本、实际 chunk 策略和 chunk schema 版本，方便业务系统直接消费。
- Prometheus metrics 已接入 `/metrics`，覆盖 HTTP 请求、解析结果、解析源文件大小、asset/chunk 数、artifact/ingest、async task、self-check、retention janitor、request guard 和 build provenance。
- `/health` 与 `/ready` 已分离：health 返回基础运行状态，ready 校验必需模型组、artifact backend、ingest store、request protection、async/self-check/retention 等依赖状态。
- Swagger UI 已挂载 `/docs`，并复用 `/openapi.json` / `/docs/openapi.json`。
- 请求保护已接入 `before_request`：支持内存/Redis 固定窗口限流、parse bytes 配额、admin bypass、fail-open 配置，并通过 `DEEPDOC_MAX_INFLIGHT_PARSE/ARTIFACT/INGEST` 做请求级 admission/in-flight 准入控制。
- `/api/v1/parse` 已支持多文件批量上传；同步 `/api/v1/parse/stream` 已通过 SSE 输出 `start`、`file_started`、`file_completed`、`done` 文件级进度和最终解析结果；`/api/v1/parse/async` 的异步任务会写入 `progress.current/total` 和 `file_completed` 事件，`/api/v1/tasks/<task_id>/stream` 可通过 SSE 订阅任务进度。
- deepdoc PDF parser 已通过模块级缓存共享 OCR、LayoutRecognizer、TableStructureRecognizer 和 `updown_concat_xgb.model` 等重组件；parser 实例仍保留独立的 `boxes`、`page_images`、`page_layout` 等单次解析状态。
- EPUB 已接入原生 parser：直接读取 `META-INF/container.xml`、OPF manifest/spine 和 XHTML 章节，按 spine 顺序输出 Markdown、结构化 blocks、表格 asset 和 asset-aware chunks；显式 `parser_engine=markitdown` 仍可使用 MarkItDown fallback。
- RTF/ODT 已接入原生轻量 parser：RTF 提取段落并输出 title/text blocks，ODT 直接读取 `content.xml` 并输出 title/text/list/table blocks、表格 asset 和 asset-aware chunks。
- EML/MSG 邮件已接入原生 parser：EML 使用标准 MIME 解析，MSG 支持可选 `extract_msg`、OLE 属性和文本 fallback；轻量文本附件会递归展开为附件正文 block 并进入 chunks。
- CAJ 已接入外部转换器包装：默认调用 `caj2pdf {input} {output}` 转 PDF，再复用现有 PDF 解析链输出 Markdown、structured blocks、assets、chunks 和转换元数据。
- 图片二维码/条形码识别已接入：OpenCV QRCodeDetector + 可选 pyzbar 识别 payload、类型和坐标，结果写入 `barcode` asset/block，并可通过 asset-aware chunk 直接消费。
- 结构化 JSON 日志已通过 `DEEPDOC_LOG_FORMAT=json` 接入；字段包含 timestamp、level、message、source、trace_id/span_id、`request_id`，解析文件时补充 `file_sha` 和 `engine`。
- OpenTelemetry tracing 已接入 Flask、requests、botocore/S3、psycopg 和解析内部阶段，覆盖 parse、structured artifact build、ingest publish 等关键 span。
- 异步任务队列已接入 Redis broker、独立 parse worker、callback redrive、任务取消/重试/批量重试/清理、任务事件和 SSE。
- S3/MinIO artifact backend 已接入，支持 structured/markdown/chunks/ingest/assets/source/manifest 写读、artifact_key 索引、直链/签名 URL 和远端清理。
- Ops audit 已接入本地 JSONL/PostgreSQL backend，记录 parse、artifact、ingest、async、self-check、retention 等运维事件，并提供 `/api/v1/audit/events` 查询、查看和清理接口。
- CORS 已通过 `DEEPDOC_CORS_*` 收紧 origin/header/method/expose header，并在严格生产配置中禁止 allow-all。
- 统一错误码和 i18n 错误响应已接入：直接 API 错误响应和批量解析单文件错误项都会保留兼容字段 `error`，并补充 `error_code`、英文/中文 message、locale 和结构化 details。
- ONNX INT8 动态量化已接入：`tools/quantize_models.py` 可为 `resources/models/*.onnx` 生成 `.int8.onnx`，默认 `DEEPDOC_QUANT=fp32` 不改变现有行为，显式 `DEEPDOC_QUANT=int8` 时加载量化模型且缺失量化模型会明确报错。
- TensorRT EP 仍是代码层显式运行时选项，但不再作为 Docker 镜像入口或默认部署路径。
- Docker 镜像收敛为两个入口：`Dockerfile.cpu` 默认 `onnxruntime` + CPU provider，`Dockerfile.gpu` 默认 `onnxruntime-gpu` + auto provider；Compose 默认只启动解析 API，不含 Gradio、数据库、队列、worker 或观测平台容器。
- 动态批处理已接入：OCR rec 在动态宽度模型上按目标宽度 bucket 分组，减少极宽文本行造成的无效 padding；Layout 识别会把同一 `batch_size` 内可合并的多页输入合并成一次 ONNX Runtime 调用，不能合并时回退逐页推理。
- OCR/Layout 启动预热已接入：API 服务和异步解析 worker 加载模型后默认跑一张白图触发 ONNX Runtime/CUDA kernel 首次执行，可用 `DEEPDOC_MODEL_WARMUP=0` 关闭。
- OCR 模型 session 缓存已接入 LRU 和 idle TTL 释放：`DEEPDOC_MODEL_CACHE_MAX_SIZE` 控制单进程缓存上限，`DEEPDOC_MODEL_CACHE_IDLE_TTL_SECONDS` 控制空闲释放时间，降低多 worker / 多设备运行时长期占用显存或内存的风险。
- deepdoc PDF 跨页合并已接入：同列、横向重叠、页底到页顶延续且无硬句末标点的段落会合并为跨页 text block；连续页表格会合并为同一 table asset/block，并在 metadata 写入 `cross_page`、`merged_page_numbers` 和 `merge_reason`。
- 已新增 artifact 导出 smoke、artifact 质量报告工具和边界测试，用于持续验证解析产物可用性。

---

## A. OCR 与小模型能力增强

| # | 优化点 | 方案要点 | 改动文件 | 优先级 | 工作量 |
|---|---|---|---|---|---|
| A1 | **公式识别(LaTeX OCR)** | 已落地：deepdoc 引擎 `enable_formula=true` 时对 layout `Equation` 区域调用本地 RapidLaTeXOCR，Markdown 输出 `$$...$$`，默认关闭 + 懒加载 | `deepdoc/vision/formula_recognizer.py`、`deepdoc/parser/pdf_parser.py`、`main.py` | P0 | Done |
| A2 | **印章识别** | 已落地：deepdoc 引擎 `enable_seal=true` 时调用本地 ONNX seal_det，极坐标展开后复用 OCR rec 写回印章文本，默认关闭 + 懒加载 | `deepdoc/vision/seal_recognizer.py`、`deepdoc/parser/pdf_parser.py`、`main.py` | P1 | Done |
| A3 | **二维码/条形码** | 已落地：OpenCV QRCodeDetector + 可选 pyzbar 识别图片中的二维码/条形码 payload、类型和坐标，结果写入 `barcode` asset/block，并进入 asset-aware chunks | `deepdoc/vision/barcode.py`、`common/parse_builders.py`、`common/parse_artifacts.py`、`main.py`、`tests/test_image_parse.py` | P2 | Done |
| A4 | **手写体识别** | 已落地：`DEEPDOC_HANDWRITING_FALLBACK=1` 时对主 rec 低置信度文本行懒加载 `rec_handwriting.onnx` 回退识别，支持阈值 `DEEPDOC_HANDWRITING_FALLBACK_THRESHOLD`、最小分差 `DEEPDOC_HANDWRITING_MIN_SCORE_DELTA` 和模型名 `DEEPDOC_HANDWRITING_MODEL_NAME`，默认关闭 | `deepdoc/vision/ocr.py::TextRecognizer`、`common/model_store.py`、`download_models.py`、`tests/test_handwriting_fallback.py` | P2 | Done |
| A5 | **图像/表格资产摘要** | 已落地：在解析产物中生成本地规则摘要，写入 `assets[].metadata.asset_summary` / `asset_summary_facts`，不调用外部对话模型 | `common/parse_artifacts.py`、`docs/API.md`、`tools/ci/artifact_quality_report.py` | P1 | Done |
| A6 | **OCR 中文字典验证** | 已落地：`download_models.py manifest` 和 `tools/ci/verify_hf_models.py` 输出 `ocr.res` sha256、行数、唯一字符数、重复/空行和基础字符覆盖报告 | `common/model_store.py`、`download_models.py`、`tools/ci/verify_hf_models.py` | P1 | Done |

## B. 推理性能优化

| # | 优化点 | 方案要点 | 改动文件 | 优先级 | 工作量 |
|---|---|---|---|---|---|
| B1 | **ONNX INT8 动态量化** | 已落地：`tools/quantize_models.py` 使用 `onnxruntime.quantization.quantize_dynamic` 生成 `.int8.onnx`；默认 `DEEPDOC_QUANT=fp32` 保持原模型，显式 `DEEPDOC_QUANT=int8` 时加载量化模型并固定 CPU EP，缺失量化模型时明确报错 | `tools/quantize_models.py`、`deepdoc/vision/ocr.py::load_model`、`tests/test_quantization.py` | P0 | Done |
| B2 | **TensorRT EP** | 已落地为代码层显式运行时选项：`DEEPDOC_ONNX_PROVIDER=tensorrt` 时校验 ORT TensorRT/CUDA provider 和 CUDA 设备，按 TensorRT → CUDA → CPU 顺序创建 session；不再作为 Docker 默认镜像入口 | `deepdoc/vision/ocr.py::load_model`、`tests/test_tensorrt_provider.py` | P1 | Done |
| B3 | **动态批处理** | 已落地：OCR rec 在动态宽度模型上按 width bucket 聚批，支持 `DEEPDOC_REC_DYNAMIC_BATCHING` 和 `DEEPDOC_REC_WIDTH_BUCKET_STEP`；Layout 底层 `Recognizer` 会将同一 batch 内可合并的多页输入一次喂入 ONNX Runtime，并保留逐页 fallback | `deepdoc/vision/ocr.py::TextRecognizer.__call__`、`deepdoc/vision/recognizer.py`、`tests/test_rec_dynamic_batching.py`、`tests/test_layout_dynamic_batching.py` | P0 | Done |
| B4 | **显存生命周期管理** | 已落地：OCR 模型 session 缓存支持 LRU 容量淘汰和 idle TTL 空闲释放，缓存命中刷新最近访问时间，可通过 `DEEPDOC_MODEL_CACHE_MAX_SIZE` / `DEEPDOC_MODEL_CACHE_IDLE_TTL_SECONDS` 调整 | `deepdoc/vision/ocr.py::loaded_models`、`tests/test_model_cache.py` | P1 | Done |
| B5 | **模型预热** | 已落地：API 服务启动加载 OCR/Layout 后默认跑一张白图触发 ONNX Runtime/CUDA kernel 首次执行，支持 `DEEPDOC_MODEL_WARMUP=0` 关闭和 `DEEPDOC_MODEL_WARMUP_IMAGE_SIZE` 调整尺寸 | `main.py`、`tests/test_model_warmup.py` | P1 | Done |

## C. 解析流程优化

| # | 优化点 | 方案要点 | 改动文件 | 优先级 | 工作量 |
|---|---|---|---|---|---|
| C1 | **扫描 vs. 原生 PDF 自动判别** | 已落地：`deepdoc_pdf_mode=auto/native/ocr/hybrid` 与 `execution_profile=auto/cpu/gpu` 契约已锁定；当前 auto 仍根据 `pdfplumber.page.chars` 字符数量、字体覆盖率和有文本页比例选择 native text 或 OCR/Layout，视觉资产请求继续走 OCR/Layout；`hybrid` 当前先按 `auto` 处理，`execution_profile` 先做参数透传，后续接 GPU 定向的 page/block 混合路由 | `deepdoc/parser/pdf_parser.py`、`main.py`、`common/parse_builders.py` | P0 | Done |
| C2 | **结果缓存** | 已落地：`reuse_artifacts=true` 时按 `sha256(file)+artifact_profile` 生成 `artifact_key`，命中已持久化 manifest/structured/chunks 后直接返回 `cache_hit=true` | `main.py`、`common/parse_artifacts.py`、`tests/test_csv_parser.py` | P0 | Done |
| C3 | **流式输出 SSE** | 已落地：`POST /api/v1/parse/stream` 返回同步解析文件级进度和最终结果，事件包括 `start`、`file_started`、`file_completed`、`done`；底层 PDF 页内阶段生成器仍可作为后续精细化优化 | `main.py`、`openapi.json` | P1 | Done |
| C4 | **跨页表格/段落合并** | 已落地：同列、页底到页顶延续且无硬句末标点的段落合并为跨页 text block；连续页表格合并为同一 table asset/block，并在结构化产物 metadata 写入 `cross_page`、`merged_page_numbers` 和 `merge_reason` | `deepdoc/parser/pdf_parser.py`、`common/parse_builders.py`、`main.py`、`tests/test_pdf_cross_page_merge.py` | P1 | Done |
| C5 | **并发控制** | 已落地：请求级 admission/in-flight 准入控制，按 parse/artifact/ingest pool 限制并发；`ProcessPoolExecutor` / GPU semaphore 作为后续推理性能优化项，不作为当前解析边界的必要依赖 | `common/ratelimit.py`、`main.py::enforce_request_protection` | P0 | Done |
| C6 | **OCR/Layout 全局单例** | 已落地：模块级缓存共享 OCR、LayoutRecognizer/AscendLayoutRecognizer、TableStructureRecognizer 和 `updown_concat_xgb.model`，按模型目录、layout backend/domain、并行设备数和 TSR backend 分 key；parser 实例状态不共享 | `deepdoc/parser/pdf_parser.py`、`tests/test_pdf_parser_singletons.py` | P1 | Done |

## D. 格式扩展

| # | 优化点 | 方案要点 | 改动文件 | 优先级 | 工作量 |
|---|---|---|---|---|---|
| D1 | **EPUB** | 已落地：无新增依赖，直接读取 EPUB `container.xml`、OPF manifest/spine 和 XHTML 章节，按 spine 输出 Markdown、结构化 title/text/list/table blocks、表格 asset 和 chunks；显式 `parser_engine=markitdown` 保留 fallback | `deepdoc/parser/epub_parser.py`、`common/parse_builders.py`、`main.py`、`tests/test_epub_parser.py` | P1 | Done |
| D2 | **CSV/TSV** | 已落地：专用 CSV/TSV parser 自动嗅探分隔符，输出 Markdown table、`table` block、`table` asset、asset-aware chunk 和 ingest records | `deepdoc/parser/csv_parser.py`、`common/parse_builders.py`、`main.py` | P0 | Done |
| D3 | **RTF/ODT** | 已落地：RTF 优先使用 `striprtf` 提取段落结构并保留 fallback，ODT 无新增依赖读取 `content.xml`，输出 Markdown、结构化 title/text/list/table blocks、表格 asset 和 chunks | `deepdoc/parser/rtf_parser.py`、`deepdoc/parser/odt_parser.py`、`common/parse_builders.py`、`main.py`、`tests/test_rtf_odt_parser.py` | P2 | Done |
| D4 | **CAJ** | 已落地：`DeepDocCajParser` 通过可配置 `DEEPDOC_CAJ2PDF_COMMAND_TEMPLATE` 调用外部 `caj2pdf` 转 PDF，主解析路径复用现有 PDF parser、structured artifact 和 chunk 策略，并在 document metadata 写入 `source_file_type=caj`、`converted_file_type=pdf` 和转换信息 | `deepdoc/parser/caj_parser.py`、`main.py`、`docs/API.md`、`openapi.json`、`tests/test_caj_parser.py` | P2 | Done |
| D5 | **邮件 eml/msg** | 已落地：EML 使用标准 MIME 解析邮件头、正文和附件清单；MSG 支持可选 `extract_msg`、OLE 属性读取和文本 fallback；文本/HTML/CSV/JSON/XML/Markdown 等轻量附件递归展开为附件正文 block 并进入 chunks，PDF/Office 等重附件保留清单供业务侧单独解析 | `deepdoc/parser/email_parser.py`、`common/parse_builders.py`、`main.py`、`tests/test_email_parser.py` | P2 | Done |
| D6 | **图片直传 `/parse`** | 已落地：`/api/v1/parse` 接收 png/jpg/jpeg/bmp/tiff/webp，复用 OCR+Layout 输出 OCR Markdown、文本 blocks、原图 `image` asset、asset-aware chunk 和 ingest records | `main.py`、`common/parse_builders.py` | P0 | Done |

## E. 结构化产物与 Chunk 增强

| # | 优化点 | 方案要点 | 改动文件 | 优先级 | 工作量 |
|---|---|---|---|---|---|
| E1 | **Chunk 策略配置化** | 已落地 `structure_aware` / `page_aware` / `asset_aware`；保留 token 上限与 overlap | `common/parse_artifacts.py`、`common/parse_builders.py`、`main.py`、`docs/API.md`、`openapi.json` | P0 | Done |
| E2 | **表格友好切块** | 已落地：`asset_aware` 让表格等资产 block 独立成 chunk，保留页码和资产引用 | `common/parse_artifacts.py` | P0 | Done |
| E3 | **资产上下文增强** | 已落地：figure/table/seal/equation 与直接 block、邻近正文、相关 chunk 双向关联，写入 `assets[].metadata` 和 chunk asset view | `common/parse_builders.py`、`common/parse_artifacts.py`、`tools/ci/artifact_quality_report.py` | P1 | Done |
| E4 | **Ingest 导出稳定化** | 已落地：`chunks.jsonl` 和 `ingest.jsonl` 写入字段版本、chunk 策略和 chunk schema 版本，方便业务系统消费 | `common/parse_artifacts.py`、`docs/API.md`、`tools/ci/artifact_quality_report.py` | P0 | Done |

## F. 可观测性

| # | 优化点 | 方案要点 | 改动文件 | 优先级 | 工作量 |
|---|---|---|---|---|---|
| F1 | **Prometheus 指标** | 已落地：`/metrics` 暴露 Prometheus 文本，覆盖 HTTP 请求/耗时、解析结果、源文件大小、asset/chunk 数、artifact/ingest、async/self-check/retention、request guard 和 build provenance | `common/metrics.py`、`main.py` | P0 | Done |
| F2 | **OpenTelemetry tracing** | 已落地：`DEEPDOC_TRACING_*` 配置驱动，支持 OTLP/console exporter，instrument Flask、requests、botocore/S3、psycopg，并在 parse、structured artifact build、ingest publish 等内部阶段创建 span | `common/tracing.py`、`main.py` | P1 | Done |
| F3 | **结构化 JSON 日志** | 已落地：`DEEPDOC_LOG_FORMAT=json` 时 console/file handler 输出 JSONL，包含 `request_id`、`file_sha`、`engine`、source 和 trace 字段；默认仍保留文本日志 | `common/log.py`、`main.py` | P0 | Done |
| F4 | **审计日志** | 已落地：本地 JSONL/PostgreSQL ops audit store，事件带 tenant、actor、request_id、trace/span、action/resource/status/payload/metadata；提供 `/api/v1/audit/events` 查询、查看、清理接口 | `common/audit_log.py`、`main.py`、`openapi.json` | P2 | Done |

## G. 部署运维

| # | 优化点 | 方案要点 | 改动文件 | 优先级 | 工作量 |
|---|---|---|---|---|---|
| G1 | **Dockerfile CPU/GPU 分离** | 已收敛：只保留 `Dockerfile.cpu` / `Dockerfile.gpu` 两个明确入口；Compose 默认使用 CPU 镜像并只启动解析 API；暂不保留 `.github` workflow 和 `.ci` 运行目录 | `Dockerfile.cpu`、`Dockerfile.gpu`、`docker-compose.yml`、`tests/test_docker_image_variants.py` | P0 | Done |
| G2 | **异步任务队列** | 代码层能力保留：支持 local task store、任务状态/事件/SSE、取消、单任务重试、批量重试和清理接口；默认 Docker 部署不再附带 Redis/worker 容器 | `common/async_tasks.py`、`main.py` | P1 | Done |
| G3 | **S3/MinIO 对象存储抽象** | 已落地：`DEEPDOC_ARTIFACT_BACKEND=s3` 支持 S3/MinIO compatible backend，写读 manifest/structured/markdown/chunks/ingest/source/assets，支持 artifact_key 索引、direct/signed URL 和删除 | `common/parse_artifacts.py`、`pyproject.toml`、`docs/API.md` | P1 | Done |
| G4 | **限流 + 配额** | 已落地：支持内存/Redis backend、general/parse/artifact/ingest/admin 固定窗口规则、parse bytes quota、admin bypass、fail-open，并在 health/ready 暴露 request protection 状态 | `common/ratelimit.py`、`main.py::enforce_request_protection` | P1 | Done |
| G5 | **多租户** | 已落地：JWT claim、`X-Tenant-ID`、query/form/body `tenant_id` 和 `DEEPDOC_DEFAULT_TENANT_ID` 统一进入租户上下文；artifact/task/ingest/audit 查询按租户隔离，request guard 优先按 tenant 分桶，JSON 日志自动带 tenant/auth 字段 | `main.py`、`common/log.py`、`common/ratelimit.py`、`tests/test_multitenancy.py` | P2 | Done |
| G6 | **健康检查增强** | 已落地：`/health` 与 `/ready` 分离，readiness 校验模型组、artifact backend、ingest store、request protection、async/self-check/retention 和 API 文档入口 | `main.py::health_check`、`ready_check` | P0 | Done |

## H. API 增强

| # | 优化点 | 方案要点 | 改动文件 | 优先级 | 工作量 |
|---|---|---|---|---|---|
| H1 | **Swagger UI** | 已落地：`flask-swagger-ui` 挂 `/docs`，复用 `/openapi.json` 和 `/docs/openapi.json` | `main.py` | P0 | Done |
| H2 | **SSE 流式解析端点** | 已落地：同步 `POST /api/v1/parse/stream` 输出 `text/event-stream`，复用现有 parse 参数、鉴权、租户上下文和审计日志 | `main.py`、`openapi.json`、`docs/API.md` | P1 | Done |
| H3 | **异步任务接口** | 已落地：`/api/v1/parse/async`、`/api/v1/tasks`、任务详情/事件/SSE/callback-events、取消、重试、批量重试和清理接口 | `main.py`、`openapi.json` | P1 | Done |
| H4 | **批量上传 + 异步进度** | 已落地：同步 `/api/v1/parse` 支持多文件上传；同步 `/api/v1/parse/stream` 输出文件级进度；异步任务写入 `progress.current/total`、`file_completed` 事件，并通过 `/api/v1/tasks/<task_id>/stream` SSE 输出 | `main.py::parse_endpoint`、`parse_stream_endpoint`、`run_async_parse_task`、`stream_async_task_events` | P1 | Done |
| H5 | **统一错误码 + i18n** | 已落地：新增 `ErrorCode` 枚举和中英双语 message，直接 API 错误响应与批量解析单文件错误项统一输出 `error_code`、`message`、`message_zh`、`locale`、`details`，并保留旧 `error` 字段兼容 | `common/errors.py`、`main.py`、`docs/API.md`、`openapi.json`、`tests/test_error_response.py` | P2 | Done |
| H6 | **CORS 收紧** | 已落地：`DEEPDOC_CORS_*` 控制 enabled、allowed origins、origin regex、headers、methods、exposed headers、credentials、max_age；strict production 禁止 allow-all 和暴露内部详情 | `main.py`、`docker-compose.yml` | P1 | Done |

---

## 三阶段实施计划

### Sprint 1(约 2 周）— 快速见效：可观测 + 性能 + 关键格式

1. **F1 Prometheus**、**F2 OpenTelemetry tracing**、**F3 JSON 日志**、**F4 审计日志** 与 **G6 健康检查** 已完成。
2. **B1 INT8 量化** + **B3 动态批处理**。
3. **C1 扫描/原生自动判别** 与 **C2 结果缓存** 已完成。
4. **D2 CSV** + **D6 图片直传 `/parse`**。
5. **G1 Dockerfile CPU/GPU 双入口**与**H1 Swagger UI**已完成。

### Sprint 2(约 3 周）— 能力补齐：模型 + 流式 + 结构化产物

1. **A1 公式识别** + **A2 印章本地化**。
2. **A6 OCR 字典校验** + **A5 资产摘要**。
3. **C5 请求级并发控制**、**C3 SSE 流式解析端点** 与 **C6 全局单例**已完成。
4. **E1 Chunk 策略配置化** + **E2 表格友好切块**。
5. **C4 跨页合并** + **E3 资产上下文增强**已完成。

### Sprint 3(约 3-4 周）— 生产化：异步队列 + 多租户 + 长尾格式

1. **G2 异步队列** + **H3 异步接口**已完成。
2. **G3 S3/MinIO 抽象**、**G4 限流配额**与 **G5 多租户**已完成。
3. **B2 TensorRT EP**已完成。
4. **E4 Ingest 导出稳定化**。
5. **D1 EPUB**、**D3 RTF/ODT**、**D4 CAJ** 与 **D5 邮件**已完成。
6. **H5 错误码 i18n**。

---

## 关键复用资源

- `common/file_utils.py`：文件验证（扩展名、大小、图片尺寸）。
- `common/markdown_utils.py`：表格转 Markdown、HTML 清理。
- `common/log.py::Log`：单例 logger。
- `common/misc_utils.py::thread_pool_exec`：线程池抽象。
- `common/parse_artifacts.py`：structured/chunks/ingest/artifacts 的核心模型与导出。
- `common/parse_builders.py`：非 PDF 解析结果到结构化产物的转换。
- `common/nlp/tokenizer.py`：解析内部使用的分词工具。
- `deepdoc/vision/operators.py::load_model`：统一 ONNX session 创建。
- `main.py::PARSER_IMPORTS` / `PDF_PARSER_OVERRIDES`：解析器注册表。
- 已装依赖：`tika`、`huggingface-hub`、`opencv-python-headless`、`pdfplumber`、`pypdf`、`tencentcloud-sdk-python`。

## 关键修改文件路径

- `/data/temp49/kd-brain/deepdoc-standalone/main.py`
- `/data/temp49/kd-brain/deepdoc-standalone/deepdoc/parser/pdf_parser.py`
- `/data/temp49/kd-brain/deepdoc-standalone/deepdoc/vision/ocr.py`
- `/data/temp49/kd-brain/deepdoc-standalone/deepdoc/vision/layout_recognizer.py`
- `/data/temp49/kd-brain/deepdoc-standalone/deepdoc/vision/table_structure_recognizer.py`
- `/data/temp49/kd-brain/deepdoc-standalone/common/{log.py, setting.py, markdown_utils.py, misc_utils.py, parse_artifacts.py, parse_builders.py}`
- `/data/temp49/kd-brain/deepdoc-standalone/Dockerfile.cpu`
- `/data/temp49/kd-brain/deepdoc-standalone/Dockerfile.gpu`
- `/data/temp49/kd-brain/deepdoc-standalone/docker-compose.yml`
- `/data/temp49/kd-brain/deepdoc-standalone/download_models.py`
- `/data/temp49/kd-brain/deepdoc-standalone/pyproject.toml`

## 验证方式

```bash
time curl -X POST "http://localhost:8001/api/v1/parse" \
     -F "file=@sample.pdf" -F "parser_engine=deepdoc"

curl -X POST "http://localhost:8001/api/v1/parse" -F "file=@data.csv"
curl -X POST "http://localhost:8001/api/v1/parse" -F "file=@scan.png"

curl -N -X POST "http://localhost:8001/api/v1/parse/stream" \
     -F "file=@sample.pdf" -F "parser_engine=deepdoc"

TID=$(curl -s -X POST "http://localhost:8001/api/v1/parse/async" \
       -F "file=@huge.pdf" | jq -r .task_id)
curl "http://localhost:8001/api/v1/tasks/$TID"
curl -N "http://localhost:8001/api/v1/tasks/$TID/stream"

curl "http://localhost:8001/health"
curl "http://localhost:8001/ready"
curl "http://localhost:8001/metrics"

python deepdoc/vision/t_ocr.py --inputs sample.png --output_dir ./debug_ocr
python deepdoc/vision/t_recognizer.py --inputs sample.pdf --mode layout

uv run ruff check .
```

每个 Sprint 结束需明确报告：

- 已验证项：性能数字、新功能 demo、错误响应示例。
- 未验证项：标注原因，如“需 GPU 实测”或“需外部解析引擎”。
- 是否破坏现有 API 行为：`parser_engine` / `compute_device` / `return_images` / `strict_text` 默认值不得变更。

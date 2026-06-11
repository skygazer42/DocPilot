# DocPilot 文档解析接口对接说明

本文档说明 DocPilot Standalone 的当前服务边界：文档解析、结构化产物、切块、资产访问、异步任务、可选 ingest 发布和运维接口。

服务不负责问答、回答生成或外部大模型调用。解析完成后的 `structured.json`、`chunks.jsonl`、`ingest.jsonl` 和资产引用就是可交付给业务系统使用的数据。

## 基础信息

- Base URL: `http://localhost:8000`
- 同步解析 Content-Type: `multipart/form-data`
- JSON 管理接口 Content-Type: `application/json`
- OpenAPI: `GET /openapi.json`
- Swagger UI: `GET /docs`

鉴权可选：

- `Authorization: Bearer <token>`
- `X-API-Key: <token>`

多租户模式建议使用：

```bash
export DEEPDOC_AUTH_MODE=jwt_hs256
export DEEPDOC_AUTH_JWT_SECRET=change-me
export DEEPDOC_AUTH_ADMIN_SCOPES=admin,artifacts:admin
export DEEPDOC_DEFAULT_TENANT_ID=default
```

JWT payload 中的 `tenant_id` 会传递到解析元数据、artifact 访问控制、异步任务、审计事件、ingest 查询、request guard 配额分桶和 JSON 日志。API key 或匿名模式可用 `X-Tenant-ID` 请求头指定租户，也可通过 `tenant_id` query/form/body 参数传入；JWT 模式下一般由 token claim 提供租户，非 admin 不允许覆盖 token tenant，admin scope 可以显式覆盖 `tenant_id`。

## 生产脱敏

除 `/health` 和 `/ready` 外，其它外部 API 响应也默认做生产级脱敏：

- 内部文件系统路径
- 本地或内网 URL
- callback 目标地址
- sink destination / DSN
- 队列名与内部任务存储路径

公开业务字段会保留：

- `parse_id`
- `task_id`
- `structured_url`
- `assets_url_prefix`
- `asset_refs`
- `asset_urls`

需要完整内部诊断信息时，只能在启用鉴权且具备 admin scope 时请求 `include_internal=1`，或显式配置：

```bash
export DEEPDOC_API_EXPOSE_INTERNALS=1
```

## 健康与运维

### `GET /health`

返回服务基础健康信息，包含：

- `auth_mode`
- `default_tenant_id`
- `api_docs`
- `build`
- `cors`
- `runtime_config`
- `artifact_backend`
- `ingest_publisher`
- `self_checks`
- `tracing`
- `ingest_query_backend`
- `ingest_query_status`
- `request_protection`
- `retention_janitor`

### `GET /ready`

生产就绪探针。它会校验：

- 必需模型组是否齐全
- artifact backend 是否可访问
- ingest backend 是否可访问
- request protection backend 和 inflight admission 是否处于可服务状态
- 异步任务平面是否可服务
- callback redrive worker heartbeat 是否新鲜
- self-check 平面是否可服务
- retention janitor worker heartbeat 是否新鲜
- API 文档入口是否暴露
- 当前进程是否启用严格生产配置检查

关键依赖不可用时返回 `503`。

### `GET /api/v1/build-info`

返回当前运行镜像的构建来源、版本、依赖指纹和 runtime 摘要，供生产验收、灰度排查和镜像比对使用。

### `GET /metrics`

Prometheus metrics。当前覆盖：

- parse 结果、耗时、资产数和 chunk 数
- artifact 访问与发布
- ingest 发布和查询
- async task、callback redrive、self-check、retention janitor
- request guard、rate limit、inflight admission
- build provenance

## 文档解析

### `POST /api/v1/ocr`

对上传图片执行 OCR。

```bash
curl -X POST "http://localhost:8000/api/v1/ocr" \
  -F "file=@/path/to/image.png"
```

### `POST /api/v1/parse`

同步解析一个或多个文件。

```bash
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "file=@/path/to/document.pdf"
```

结构化解析并持久化：

```bash
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "file=@/path/to/document.pdf" \
  -F "return_structured=true" \
  -F "persist_artifacts=true" \
  -F "include_chunks=true" \
  -F "chunk_strategy=asset_aware"
```

### `POST /api/v1/parse/stream`

同步流式解析一个或多个文件：

```bash
curl -N -X POST "http://localhost:8000/api/v1/parse/stream" \
  -F "file=@/path/to/document.pdf" \
  -F "return_structured=true" \
  -F "persist_artifacts=true" \
  -F "include_chunks=true"
```

返回 `text/event-stream`。事件包括 `start`、`file_started`、`file_completed`、`done`，其中 `file_completed` 会携带单文件解析结果，`done` 会携带本次请求的最终 `results`、`file_count`、`result_count` 和 `error_count`。该端点提供文件级进度，解析产物字段与 `/api/v1/parse` 保持一致。

常用表单参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---:|---:|---|
| `file` | file/list | 必填 | 支持 pdf/docx/xlsx/xls/pptx/ppt/html/json/md/txt/csv/tsv/rtf/odt/eml/msg/xml/zip/epub/caj/png/jpg/jpeg/bmp/tiff/webp |
| `parser_engine` | string | `deepdoc` | PDF 引擎：`deepdoc`、`paddleocr_vl`、`mineru`、`plain`；非 PDF 可显式传 `markitdown` |
| `compute_device` | string | `gpu` | 远程解析引擎设备模式：`gpu` 或 `cpu` |
| `execution_profile` | string | `auto` | 执行画像：`auto` 保持当前默认调度；`cpu` 为保守 CPU 路径；`gpu` 为后续 GPU 页级并发/混合路由保留，当前仅做参数归一化与透传 |
| `deepdoc_pdf_mode` | string | `auto` | deepdoc PDF 模式：`auto` 自动判别原生文本层/扫描件；`native` 强制原生文本层；`ocr` 强制 OCR + Layout；`hybrid` 为后续 page/block 混合路由保留，当前行为与 `auto` 一致 |
| `return_images` | bool | `false` | Markdown 中是否保留图片内容 |
| `strict_text` | bool | `false` | 是否剥离 HTML 标签并返回更严格的纯文本 Markdown |
| `enable_formula` | bool | `false` | deepdoc 引擎公式识别 |
| `enable_seal` | bool | `false` | deepdoc 引擎印章识别 |
| `return_structured` | bool | `false` | 响应中是否返回结构化解析产物 |
| `persist_artifacts` | bool | `false` | 是否持久化 markdown、structured、chunks、ingest 和 assets |
| `persist_source` | bool | `false` | 持久化时是否保存原始上传文件 |
| `include_chunks` | bool | 自动 | 是否生成 chunks；`return_structured` 或 `persist_artifacts` 开启时通常一起开启 |
| `chunk_max_tokens` | int | `800` | chunk 最大 token 数 |
| `chunk_overlap_tokens` | int | `120` | chunk 间重叠 token 数 |
| `chunk_strategy` | string | `structure_aware` | 切块策略：`structure_aware` 按结构顺序合并；`page_aware` 避免跨页合并；`asset_aware` 让 table/figure/equation/seal/barcode 等资产块独立成 chunk |
| `publish_ingest` | bool | `false` | 解析完成后发布 ingest 记录；开启时会强制持久化 artifact |
| `reuse_artifacts` | bool | `false` | 按 `artifact_key` 复用已有解析产物 |
| `tenant_id` | string | - | 租户 ID；JWT 模式下一般由 token claim 提供 |

错误响应已统一为兼容旧字段的结构化格式。所有直接 API 错误响应会保留 `error`，同时提供稳定 `error_code`、英文 `message`、中文 `message_zh`、归一化 `locale` 和结构化 `details`；批量解析中单文件失败的 `results[]` 项也会带同样字段。客户端应优先按 `error_code` 分支处理，`error` 继续用于兼容旧调用方和日志展示。

CSV/TSV 文件会走专用结构化表格解析，输出 Markdown table、`table` block、`table` asset、asset-aware chunk 和对应 ingest records。

EPUB 文件默认走原生解析器，读取 `container.xml`、OPF manifest/spine 和 XHTML 章节，按 spine 顺序输出标题、正文、列表、表格 blocks、表格 asset 和结构化 chunks。需要兼容 MarkItDown 行为时，可显式传 `parser_engine=markitdown`。

RTF/ODT 文件默认走原生轻量解析器：RTF 提取段落并输出 title/text blocks，ODT 读取 `content.xml` 并输出 title/text/list/table blocks；表格会生成 `table` asset、asset-aware chunk 和对应 ingest records。

eml/msg 邮件文件默认走原生解析器：EML 使用 Python 标准库解析 MIME 头、正文和附件清单；MSG 优先使用可选 `extract_msg`，缺依赖时尝试 OLE 属性读取，最后兼容文本型 `.msg` fallback。文本、HTML、CSV、JSON、XML、Markdown 等轻量附件会递归展开为附件正文 block，并进入 chunks；PDF/Office 等重附件先作为附件清单保留，可由业务侧单独调用 `/parse` 解析。

CAJ 文件会先调用外部 `caj2pdf` 转换为 PDF，再复用现有 PDF 解析链输出 Markdown、structured blocks、assets 和 chunks。默认命令模板为 `caj2pdf {input} {output}`，可通过 `DEEPDOC_CAJ2PDF_COMMAND_TEMPLATE` 覆盖；转换超时由 `DEEPDOC_CAJ2PDF_TIMEOUT` 控制。CAJ 结构化产物的 document metadata 会保留 `source_file_type=caj`、`converted_file_type=pdf` 和 `caj_conversion`，用于追踪转换来源。

图片文件会走 OCR + Layout 解析链，输出 OCR Markdown、文本 blocks、原图 `image` asset、asset-aware chunk 和对应 ingest records。图片中的二维码/条形码会通过 OpenCV QRCodeDetector 和可选 pyzbar 识别为 `barcode` asset/block，并进入 asset-aware chunks；识别结果包含条码类型、文本 payload 和位置坐标。

deepdoc PDF 默认 `deepdoc_pdf_mode=auto`。服务会先用 `pdfplumber.page.chars` 判断原生文本层质量：字符数量、字体覆盖率和有文本页比例达标时，且未请求 `return_images` / `enable_formula` / `enable_seal` 等视觉能力，会走 native text 快路径并输出 `pdf_native_text` blocks/chunks；扫描件、图片型 PDF 或需要视觉资产时继续走 OCR + Layout。需要强制行为时可传 `deepdoc_pdf_mode=native` 或 `deepdoc_pdf_mode=ocr`。`deepdoc_pdf_mode=hybrid` 与 `execution_profile=auto/cpu/gpu` 当前先作为前向兼容契约保留并写入解析参数，其中 `hybrid` 现阶段按 `auto` 处理，后续版本再接入 GPU 定向的 page/block 混合路由。

deepdoc OCR + Layout 路径会做保守的跨页合并：同列、横向重叠、第一页页底到下一页页顶延续、且上一段没有硬句末标点的段落，会合并成一个跨页 text block。合并后的 block 会在 `metadata` 中写入 `merge_reason=text_cross_page_continuation`、`merged_page_numbers`、`source_box_count` 和 `source_layoutnos`。连续页表格会合并为同一个 table asset/block；跨页 table 的 asset 和 block 会带 `metadata.cross_page=true`、`metadata.merged_page_numbers` 和 `metadata.merge_reason=table_cross_page_continuation`，方便业务侧在 chunks 或 assets 中保留跨页关系。

`chunks.jsonl` 和 `ingest.jsonl` 会在每条记录的 `metadata.schema_version` 中写入导出 schema 版本，并在 `metadata.chunk_strategy` 中写入实际切块策略。结构化产物和 chunk 导出的 `assets[].metadata` 会写入：

- `asset_context_schema_version`、`direct_block_refs`、`context_block_refs`、`direct_chunk_refs`、`context_chunk_refs`、`chunk_refs` 和 `context_texts`，用于从表格、图片、公式、印章、二维码/条形码等资产反查直接 block、邻近正文和相关 chunk。
- `asset_summary_schema_version`、`asset_summary_source=local_rules`、`asset_summary` 和 `asset_summary_facts`，用于给表格、图片、公式、印章、二维码/条形码等资产提供本地规则摘要和基础 facts，例如页码、尺寸、文本长度、表格行列数、条码类型。

这些字段都是解析产物导出格式，不包含问答生成、向量化或外部模型调用。

PDF 引擎示例：

```bash
# DocPilot 自动判别：原生文本层 PDF 走 native text，扫描件走 OCR + Layout
curl -X POST "http://localhost:8000/api/v1/parse" \
  -F "parser_engine=deepdoc" \
  -F "deepdoc_pdf_mode=auto" \
  -F "return_structured=true" \
  -F "file=@/path/to/document.pdf"

# DocPilot 强制 OCR + Layout
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

`download=true` 且只上传一个文件时，可直接下载 Markdown：

```bash
curl -X POST "http://localhost:8000/api/v1/parse?download=true" \
  -F "file=@/path/to/document.pdf" \
  -o parsed.md
```

### 同步响应

默认 JSON 响应包含：

- `status`
- `results`
- `errors`

单个 result 常见字段：

- `filename`
- `type`
- `content`
- `parser_engine`
- `document_id`
- `parse_id`
- `artifact_urls`
- `asset_count`
- `chunk_count`
- `structured`
- `ingest_publish`

`return_structured=true` 时，`structured` 中包含：

- `document`: 文档级元数据
- `blocks`: 标题、段落、表格、图片、页眉页脚等结构块
- `assets`: 抽取的表格、图片、公式、印章等资产
- `chunks`: 面向业务复用的结构化切块

## 异步任务

### `POST /api/v1/parse/async`

提交异步解析任务：

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

返回：

- `task_id`
- `status`
- `created_at`
- `poll_url`
- `events_url`
- `stream_url`

异步接口和 `/api/v1/parse/stream` 都复用 `/api/v1/parse` 的表单参数，包括 `deepdoc_pdf_mode` 和 `execution_profile`。

任务查询：

```bash
curl "http://localhost:8000/api/v1/tasks/<task_id>"
curl "http://localhost:8000/api/v1/tasks/<task_id>/events"
curl -N "http://localhost:8000/api/v1/tasks/<task_id>/stream?include_callback_events=true"
curl "http://localhost:8000/api/v1/tasks/<task_id>/callback-events"
```

任务操作：

```bash
curl -X POST "http://localhost:8000/api/v1/tasks/<task_id>/cancel"
curl -X POST "http://localhost:8000/api/v1/tasks/<task_id>/retry" \
  -H "Content-Type: application/json" \
  -d '{"force":true,"copy_callback":true,"requested_by":"ops"}'
curl -X POST "http://localhost:8000/api/v1/tasks/retry" \
  -H "Content-Type: application/json" \
  -d '{"dry_run":true,"task_statuses":["failed","cancelled"],"limit":50}'
curl -X POST "http://localhost:8000/api/v1/tasks/<task_id>/callback/retry" \
  -H "Content-Type: application/json" \
  -d '{"force":true,"requested_by":"ops"}'
curl -X POST "http://localhost:8000/api/v1/tasks/callbacks/retry" \
  -H "Content-Type: application/json" \
  -d '{"dry_run":true,"callback_statuses":["dead_lettered","failed"],"limit":50}'
curl -X POST "http://localhost:8000/api/v1/tasks/cleanup" \
  -H "Content-Type: application/json" \
  -d '{"dry_run":true,"older_than_days":7,"keep_latest":100,"status":"succeeded"}'
```

## Artifact

当 `persist_artifacts=true` 时，服务会保存解析产物。local backend 默认目录为 `resources/artifacts`。

每个 parse 目录包含：

- `manifest.json`: parse 级摘要、URL、artifact key、发布状态
- `markdown.md`: Markdown 输出
- `structured.json`: 完整结构化解析结果
- `chunks.jsonl`: 结构化切块
- `ingest.jsonl`: 扁平发布记录
- `publish-events.jsonl`: append-only 发布事件
- `assets/`: 表格、图片、公式、印章等资产文件

### Artifact 读取

```bash
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/manifest"
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/structured"
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/markdown"
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/chunks?asset_url_mode=proxy"
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/ingest?asset_url_mode=proxy"
curl "http://localhost:8000/api/v1/artifacts/<parse_id>/publish-events"
curl "http://localhost:8000/api/v1/artifacts?limit=20"
```

`asset_url_mode` 支持：

- `proxy`: 返回 API 代理下载路径
- `direct`: 返回对象存储直连 URL
- `signed`: 返回带过期时间的签名 URL

`expires_in` 控制签名 URL 秒级有效期，默认 `3600`。

### Artifact 发布重试

```bash
curl -X POST "http://localhost:8000/api/v1/artifacts/<parse_id>/publish" \
  -H "Content-Type: application/json" \
  -d '{"force":true,"requested_by":"retry","asset_url_mode":"proxy"}'

curl -X POST "http://localhost:8000/api/v1/artifacts/publish-retry" \
  -H "Content-Type: application/json" \
  -d '{"limit":50,"scan_limit":250,"only_due":true,"requested_by":"retry-batch"}'
```

### Artifact 清理

```bash
curl -X POST "http://localhost:8000/api/v1/artifacts/cleanup" \
  -H "Content-Type: application/json" \
  -d '{"dry_run":true,"older_than_days":30,"keep_latest":100}'
```

## Ingest

Ingest 是解析产物的发布与结构化落库层，用来保存文档、文本记录、chunk、资产和 chunk-asset 关系。

### 发布配置

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
export DEEPDOC_INGEST_PUBLISHER=postgres
export DEEPDOC_INGEST_PG_DSN=postgresql://user:password@postgres:5432/deepdoc
export DEEPDOC_INGEST_PG_SCHEMA=deepdoc_ingest
export DEEPDOC_INGEST_PG_CONNECT_TIMEOUT=10
```

PostgreSQL schema 包含：

- `documents`
- `records`
- `chunks`
- `assets`
- `chunk_asset_links`
- `parse_aliases`

### 查询接口

```bash
curl "http://localhost:8000/api/v1/ingest/stats?include_breakdown=true"
curl "http://localhost:8000/api/v1/ingest/documents?limit=20"
curl "http://localhost:8000/api/v1/ingest/documents/<parse_id>"
curl "http://localhost:8000/api/v1/ingest/records?q=architecture&mode=text"
curl "http://localhost:8000/api/v1/ingest/assets?parse_id=<parse_id>&limit=20"
curl "http://localhost:8000/api/v1/ingest/assets/<parse_id>/<asset_id>"
curl "http://localhost:8000/api/v1/ingest/chunks?parse_id=<parse_id>&limit=20"
curl "http://localhost:8000/api/v1/ingest/chunk-asset-links?parse_id=<parse_id>&relation_type=direct"
```

`/api/v1/ingest/records` 参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---:|---:|---|
| `q` | string | - | 文本关键词 |
| `mode` | string | `text` | 当前只支持 `text` |
| `parse_id` | string | - | 按解析任务过滤 |
| `document_id` | string | - | 按文档过滤 |
| `tenant_id` | string | - | admin token 可显式覆盖 |
| `limit` | int | `20` | 最大 `200` |
| `offset` | int | `0` | 偏移量 |

## 统一运维审计

审计事件用于追踪解析、artifact、ingest、异步任务、self-check 和清理操作。

常用接口：

```bash
curl "http://localhost:8000/api/v1/audit/events?limit=50"
curl "http://localhost:8000/api/v1/audit/events/<event_id>"
curl -X POST "http://localhost:8000/api/v1/audit/events/cleanup" \
  -H "Content-Type: application/json" \
  -d '{"dry_run":true,"older_than_days":14,"keep_latest":5000}'
```

常用过滤：

- `limit`
- `offset`
- `status`
- `action`
- `resource_type`
- `resource_id`
- `tenant_id`

## Self-Check

### `POST /api/v1/self-checks/run`

```bash
curl -X POST "http://localhost:8000/api/v1/self-checks/run" \
  -H "Content-Type: application/json" \
  -d '{"suite":"core","force_reparse":false,"force_republish":true}'
```

### 查询与清理

```bash
curl "http://localhost:8000/api/v1/self-checks?limit=5"
curl "http://localhost:8000/api/v1/self-checks/<check_id>"
curl -X POST "http://localhost:8000/api/v1/self-checks/cleanup" \
  -H "Content-Type: application/json" \
  -d '{"dry_run":true,"older_than_days":7,"keep_latest":20,"status":"passed"}'
```

后台 worker 配置：

- `DEEPDOC_SELF_CHECK_AUTO_ENABLED`
- `DEEPDOC_SELF_CHECK_REQUIRED_FOR_READY`
- `DEEPDOC_SELF_CHECK_LAST_RUN_MAX_AGE_SECONDS`
- `DEEPDOC_SELF_CHECK_HEARTBEAT_FILE`
- `DEEPDOC_SELF_CHECK_HEALTH_MAX_AGE_SECONDS`
- `DEEPDOC_SELF_CHECK_POLL_SECONDS`
- `DEEPDOC_SELF_CHECK_HEARTBEAT_SECONDS`
- `DEEPDOC_SELF_CHECK_RUN_ON_START`
- `DEEPDOC_SELF_CHECK_FORCE_REPARSE`
- `DEEPDOC_SELF_CHECK_FORCE_REPUBLISH`
- `DEEPDOC_SELF_CHECK_TENANT_ID`

## Retention Janitor

后台 janitor 周期性清理：

- async tasks
- persisted artifacts
- ops audit events
- production self-check results

相关配置：

- `DEEPDOC_RETENTION_JANITOR_ENABLED`
- `DEEPDOC_RETENTION_JANITOR_REQUIRED_FOR_READY`
- `DEEPDOC_RETENTION_JANITOR_POLL_SECONDS`
- `DEEPDOC_RETENTION_JANITOR_HEARTBEAT_SECONDS`
- `DEEPDOC_RETENTION_JANITOR_HEARTBEAT_FILE`
- `DEEPDOC_RETENTION_JANITOR_HISTORY_LIMIT`
- `DEEPDOC_RETENTION_JANITOR_HEALTH_MAX_AGE_SECONDS`
- `DEEPDOC_RETENTION_JANITOR_LAST_RUN_MAX_AGE_SECONDS`
- `DEEPDOC_RETENTION_JANITOR_TASKS_*`
- `DEEPDOC_RETENTION_JANITOR_ARTIFACTS_*`
- `DEEPDOC_RETENTION_JANITOR_AUDIT_EVENTS_*`
- `DEEPDOC_RETENTION_JANITOR_SELF_CHECKS_*`

## Artifact 对象存储

默认 backend 是 local。S3 / MinIO / compatible object storage：

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

## Docker / Compose

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

- `deepdoc`

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

# 最小 deepdoc 默认解析模型组
python download_models.py core
export DEEPDOC_AUTO_DOWNLOAD=0
docker compose up -d
```

模型本地校验：

```bash
python download_models.py manifest
```

输出中的 `model_group_provenance` 会记录模型组来源/许可证、是否默认启用、关联开关和 readiness gate；`ocr_dictionary` 保留旧字段，会报告 `resources/models/ocr.res` 的 sha256、行数、唯一字符数、重复/空行和基础中文/数字/英文字符覆盖状态；`ocr_dictionaries` 会同时报告 `ocr.res` 和 `ocr_v5.res` 等已声明 OCR 字典；`ocr_recognition_alignments` 会检查 `rec.onnx`/`rec_v5.onnx` 的输出类别数是否匹配对应字典行数、空格类和 CTC blank。发布模型包后，`tools/ci/verify_hf_models.py` 会校验远端 manifest 中的 `model_group_provenance` 与本地声明一致，避免缺少模型组来源/许可证的候选权重进入后续评测。

CPU pipeline 现代化开关默认保持历史行为，不会静默改变解析结果：

| 变量 | 默认值 | 可选值 | 作用 | 需要的模型组 |
|---|---|---|---|---|
| `DEEPDOC_OCR_VERSION` | `v4` | `v4` / `v5` | 切换 deepdoc 本地 OCR 检测/识别模型 | `core` / `core_v5` |
| `DEEPDOC_REC_IMAGE_SHAPE` | 空 | `C,H,W` | 通用 OCR 识别输入尺寸覆盖，空值保持默认 `3,48,320` | `core` / `core_v5` |
| `DEEPDOC_OCR_V4_REC_IMAGE_SHAPE` | 空 | `C,H,W` | v4 OCR 识别输入尺寸覆盖，优先于 `DEEPDOC_REC_IMAGE_SHAPE` | `core` |
| `DEEPDOC_OCR_V5_REC_IMAGE_SHAPE` | 空 | `C,H,W` | v5 OCR 识别输入尺寸覆盖，优先于 `DEEPDOC_REC_IMAGE_SHAPE` | `core_v5` |
| `DEEPDOC_LAYOUT_ENGINE` | `legacy` | `legacy` / `ppdoclayout` | 切换 deepdoc PDF OCR + Layout 路径的版面识别后端 | `core` / `layout_v2` |
| `DEEPDOC_TABLE_ENGINE` | `tatr` | `tatr` / `rapidtable` | 切换 deepdoc PDF 表格识别后端；`rapidtable` 单表失败回退 `tatr` | `core` / `table_v2` |
| `DEEPDOC_FORMULA_MODE` | `rapidlatex` | `rapidlatex` / `pp_formula_net_s` | 切换公式识别模型模式 | `formula` / `formula_v2` |
| `DEEPDOC_READING_ORDER_STRATEGY` | `legacy` | `legacy` / `rules` | 切换阅读顺序和跨页段落规则；`rules` 启用多栏排序、重复页眉页脚去重、caption 绑定和标题/缩进误合并拦截 | 无 |

启用新模型前先下载或预置对应模型组：

```bash
python download_models.py core_v5
python download_models.py layout_v2
python download_models.py table_v2
python download_models.py formula_v2

# 或一次下载已发布模型组
python download_models.py published

# 全部声明模型组；包含未发布的 handwriting 时需要自行提供 rec_handwriting.onnx
python download_models.py all
```

`DEEPDOC_FORMULA_MODE=pp_formula_net_s` 会通过 PaddleX PP-FormulaNet-S 适配器执行公式识别，需安装 `pip install -e ".[formula-v2]"` 并预置 `formula_v2` 模型组。默认仍为 `DEEPDOC_FORMULA_MODE=rapidlatex`；切换生产默认前必须先补真实业务/论文公式子集 A/B 指标和 CPU 耗时报告。

端到端评测与性能 profiling：

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

`check_cpu_pipeline_readiness.py` 会检查 `core_v5`、`layout_v2`、`table_v2`、`formula_v2` 模型组、dataset contract、license gate 报告、真实 PDF 页数下限，以及 OCR/layout/table/formula 的 baseline/candidate 成对 A/B 报告和 profile 报告是否齐全。license gate 报告必须来自 `python tools/eval_omnidocbench.py --license-gate --out eval_out/license-gate.json`，schema 为 `2026-06-08.cpu-pipeline-license-gate.v1`，且状态为 `passed`；该门禁用于阻止 AGPL/GPL 候选组件进入本地解析主链路。升级模型组候选会从 `MODEL_GROUP_PROVENANCE` 派生，避免模型来源/许可证声明和 license gate 手工清单漂移。A/B 报告必须声明 `engine=deepdoc`，若 `samples` 明细中声明了 `engine` 也必须是 `deepdoc`，避免 plain 或外部引擎报告混入本地 DocPilot CPU pipeline 门禁。A/B 报告自身也必须包含 `license_gate`，readiness 会校验 A/B 报告内嵌 `license_gate` 的 allowed/blocked 候选覆盖及 `license/status` 字段一致性，避免旧评测报告缺少许可证上下文或字段漂移却误过门禁。A/B 报告自身也必须包含 `dataset_contract`，readiness 会校验 A/B 报告内嵌 `dataset_contract` 的 schema/status/sample_count/samples，确保报告自带的评测集合同与当前 readiness 数据集一致。正式评测报告和 profile 报告还会包含 `model_manifest`（schema `2026-06-08.cpu-pipeline-model-manifest.v1`），记录评测或 profiling 时的模型目录、模型组、每个声明模型文件的存在性、大小和 sha256；model_manifest 会按报告的 `pipeline_config` 只记录相关模型组，readiness 会按报告的 `pipeline_config` 校验对应模型组快照是否与当前 `--model-root` 一致，例如 OCR baseline 对 `core`，OCR candidate 对 `core_v5`，layout candidate/profile 对 `layout_v2`，table candidate/profile 对 `table_v2`，formula candidate/profile 对 `formula_v2`，避免模型文件换版后复用旧 A/B 报告或旧 profiling 报告。A/B 报告必须与本次 readiness 的 `--dataset` 一致，baseline/candidate 也必须使用同一数据集和相同 `sample_count`，且 `sample_count` 必须是正整数并等于 dataset contract 识别到的样本数；A/B 报告必须包含 `samples` 明细，`summary.sample_count` 还必须是正整数并等于 `len(samples)`，summary 中由 samples 明细产生的均值指标必须在每个 sample 中都有对应指标值，summary 和 samples 指标都必须是数值且有限，JSON boolean 不能作为数值字段，且 summary 均值必须按全量 samples 明细计算并一致，samples 明细里的样本名和声明的 `pdf_path` 也必须匹配 dataset contract，避免混用旧评测报告、只用子集报告或摘要/明细不一致的报告误过门禁。报告里的 `pipeline_config` 也会被校验，例如 OCR baseline 必须是 `ocr_version=v4`，candidate 必须是 `ocr_version=v5`；layout baseline 必须是 `layout_engine=legacy` 且 `reading_order_strategy=legacy`，candidate 必须是 `layout_engine=ppdoclayout` 且 `reading_order_strategy=rules`；table baseline 必须是 `table_engine=tatr`，candidate 必须是 `table_engine=rapidtable`。layout A/B 报告还必须包含 `mean_cross_page_merge_accuracy`、`mean_chunk_text_coverage` 和 `mean_business_field_location_hit_rate`，且 candidate 不得低于 baseline；table A/B 报告必须包含 `mean_table_teds` 和 `mean_table_cell_f1`，且 candidate 不得低于 baseline；formula A/B 报告必须包含 `mean_formula_normalized_edit_distance`、`mean_formula_exact_match_rate` 和 `mean_elapsed_seconds`，且 candidate 不得低于 baseline 质量或高于 baseline 耗时。profile 报告必须记录 `pipeline_config`（含 `formula_mode`）、`model_manifest`、`license_gate`、`reading_order_strategy`、7 个阶段耗时（`rasterize_ocr`、`layout`、`table`、`text_merge`、`cross_page_text`、`reading_order`、`extract_assets`）和 `stage_summary` 瓶颈摘要；profile 阶段列表只能包含这 7 个阶段，缺失或额外阶段都会失败；profile 报告自身也必须包含 `dataset_contract`，并且必须包含 `license_gate`；readiness 会校验 profile 内嵌 `license_gate` 的 allowed/blocked 候选覆盖及 `license/status` 字段一致性，也会校验 profile 内嵌 `dataset_contract` 的 schema/status/sample_count/samples；profile 顶层 `dataset` 必须匹配本次 readiness 的 `--dataset`，其 `sample_name` 和 `pdf_path` 必须对照 dataset contract 指向同一条 PDF 样本，避免拿旧数据集、临时单页样本或错配样本的耗时报告混过门禁；`stage_summary` 包含 `slowest_stage`、最慢阶段耗时/占比和按耗时排序的阶段列表，按耗时排序的阶段列表必须包含每个阶段的耗时和占比，readiness 会校验顶层配置、总耗时、阶段求和、必选阶段覆盖、额外阶段、最慢阶段摘要、每个排序阶段的耗时/占比和模型快照一致。当前环境缺真实权重、license gate 报告或真实评测集时应返回 `failed`，避免把无数据的脚手架状态误判为可切默认。

profile 内嵌 `dataset_contract.samples[*].pdf_path` 若有声明，也必须与当前 readiness 数据集中的同名 PDF 一致，避免复用旧数据集路径的 profile 报告。
readiness 会校验 profile 内嵌 `dataset_contract` 的 schema/status/sample_count/samples。
内嵌 `dataset_contract.samples` 必须是数组，且每一项都必须是 JSON object，避免非结构化评测集合同绕过样本名和路径校验。
profile 报告也必须对应本次 readiness 的 `--dataset`，且顶层 `dataset` 字段缺失或不匹配时 readiness 会失败；profile 顶层 `sample_name` 和 `pdf_path` 也必须指向 dataset contract 的同一样本。
正式 A/B 报告的 `samples[*].pdf_path` 和内嵌 `dataset_contract.samples[*].pdf_path` 都是必填项，缺失时 readiness 会失败，避免路径不明的旧报告混过门禁。
正式 A/B 报告的 `samples` 必须是数组，且每一项都必须是 JSON object，避免非结构化明细绕过样本名、路径、引擎和指标校验。
正式 A/B 报告中，只要 `summary` 声明了由 samples 明细产生的均值指标，每个 sample 都必须提供对应指标值；readiness 会按全量 samples 明细重算均值，缺失、非数值/非有限数值、JSON boolean 或摘要/明细不一致都会失败。
未传 `--dataset` 的 profile 报告也会写入 `status=failed` 的 `dataset_contract`，显式记录未绑定评测集，不能用于通过 readiness。
profile 报告中的 `total_elapsed_seconds`、`stages[*].elapsed_seconds`、`stage_summary.slowest_stage_elapsed_seconds`、`stage_summary.slowest_stage_share`、`stage_summary.stages_by_elapsed_seconds[*].elapsed_seconds` 和 `stage_summary.stages_by_elapsed_seconds[*].share` 都必须是数值且有限，`stage_summary.stages_by_elapsed_seconds[*]` 的 `share` 必须匹配对应阶段耗时占比，JSON boolean 不能作为耗时或占比字段，避免 `NaN` / `Infinity` 或 `true` / `false` 耗时报告混过 readiness。

`eval_omnidocbench.py` 需要真实标注样本目录才能产出基线指标；空目录会直接失败，避免把无数据的评测误报为通过。

样本目录按同名文件配对：

| 文件 | 用途 | 输出指标 |
|---|---|---|
| `<name>.pdf` | 必需，待评测 PDF | `elapsed_seconds`、`text_length` |
| `<name>.gt.txt` | 可选，全文文本标注 | `character_error_rate`、`word_error_rate`、`text_normalized_edit_distance` |
| `<name>.gt.blocks.json` | 可选，结构块标注；可为 block 数组或含 `blocks` 字段的 structured JSON | `block_type_f1`、`reading_order_normalized_edit_distance`、`cross_page_merge_accuracy` |
| `<name>.gt.tables.html` 或 `<name>.gt.html` | 可选，表格 HTML 标注 | `mean_table_teds`、`mean_table_cell_f1` |
| `<name>.gt.formulas.json` | 可选，公式 LaTeX 标注；可为数组或含 `formulas` / `equations` / `expected_formulas` 字段的 JSON | `mean_formula_normalized_edit_distance`、`mean_formula_exact_match_rate` |
| `<name>.gt.chunks.json` | 可选，业务可复用 chunk 文本标注；可为数组或含 `chunks` 字段的 JSON | `chunk_text_coverage` |
| `<name>.gt.fields.json` | 可选，业务字段标注；可为数组或含 `fields` 字段的 JSON，字段项支持 `name`、`value`、`page_numbers` | `business_field_location_hit_rate` |

需要启用 ONNX INT8 动态量化时，先生成量化模型：

```bash
python tools/quantize_models.py --model-dir resources/models
```

该脚本会递归扫描 `resources/models` 及 `layout/`、`formula/`、`table/` 等模型组子目录，为非 INT8 `.onnx` 模型生成同名 `.int8.onnx` 文件，不覆盖原始模型。默认批量扫描会跳过高风险序列/decoder 模型（如 `rec.onnx`、`rec_v5.onnx`、`formula/decoder.onnx`），这些模型需要先做逐模块精度校准；确认达标后再显式传 `--include-risky-sequence-models`，或用 `--model path/to/model.onnx` 单独量化。服务默认 `DEEPDOC_QUANT=fp32`，继续加载原始模型；显式设置 `DEEPDOC_QUANT=int8` 后加载 `.int8.onnx`。如果量化模型不存在，服务会直接报错而不是静默回退。INT8 动态量化路径当前固定使用 CPU Execution Provider，适合 CPU 推理或 GPU 不稳定时的轻量化部署。

ONNX Runtime provider 默认 `DEEPDOC_ONNX_PROVIDER=auto`，会按现有逻辑在 CUDA 可用时使用 `CUDAExecutionProvider`，否则回退 `CPUExecutionProvider`。需要启用 TensorRT 时显式设置 `DEEPDOC_ONNX_PROVIDER=tensorrt`；服务会要求 ONNX Runtime 同时提供 `TensorrtExecutionProvider` 和 `CUDAExecutionProvider`，并使用 TensorRT → CUDA → CPU 的 provider fallback 顺序。TensorRT engine cache 默认写入 `DEEPDOC_TENSORRT_CACHE_DIR=/app/resources/temp/tensorrt_engine_cache`，可用 `DEEPDOC_TENSORRT_FP16=1` 开启 FP16，并通过 `DEEPDOC_TENSORRT_MAX_WORKSPACE_SIZE` 控制 workspace size。默认不会自动切换到 TensorRT，避免普通 GPU 镜像在缺 TensorRT provider 时启动失败。

OCR 识别阶段默认启用动态批处理：当 rec 模型输入宽度为动态维度时，会按文本行目标宽度 bucket 分组，减少同批次被极宽文本行拖高的 padding 宽度；固定宽度模型保持原有固定 batch 行为。可通过 `DEEPDOC_REC_DYNAMIC_BATCHING=0` 关闭；`DEEPDOC_REC_WIDTH_BUCKET_STEP` 控制 bucket 粒度，默认 `64`。Layout 识别会把同一个 `batch_size` 内可合并的多页输入合并成一次 ONNX Runtime 调用，不能合并时自动回退到逐页推理。

手写体回退默认关闭。设置 `DEEPDOC_HANDWRITING_FALLBACK=1` 后，OCR rec 主模型分数低于 `DEEPDOC_HANDWRITING_FALLBACK_THRESHOLD` 的文本行会懒加载手写识别模型重跑；默认模型名为 `DEEPDOC_HANDWRITING_MODEL_NAME=rec_handwriting`，对应模型文件 `rec_handwriting.onnx`。只有手写结果分数至少高出 `DEEPDOC_HANDWRITING_MIN_SCORE_DELTA` 且文本非空时才替换主识别结果。该能力需要先下载 `handwriting` 模型组：

```bash
python download_models.py handwriting
```

API 服务和异步解析 worker 启动时默认执行一次 OCR/Layout 预热：加载模型后跑一张白图触发 ONNX Runtime/CUDA kernel 首次执行，降低首个真实解析请求的冷启动抖动。可通过 `DEEPDOC_MODEL_WARMUP=0` 关闭；白图尺寸可用 `DEEPDOC_MODEL_WARMUP_IMAGE_SIZE` 调整，默认 `64`。

OCR 模型 session 使用 LRU 缓存并支持空闲释放，避免多 worker 或多设备配置下长期持有过多显存/内存。`DEEPDOC_MODEL_CACHE_MAX_SIZE` 控制单进程最大缓存模型数，默认 `8`；`DEEPDOC_MODEL_CACHE_IDLE_TTL_SECONDS` 控制空闲 TTL，默认 `3600` 秒。缓存命中会刷新最近访问时间；超过容量时淘汰最久未使用模型。

## Structured Logs

默认日志格式为文本。需要容器/日志平台直接采集结构化字段时设置：

```bash
export DEEPDOC_LOG_FORMAT=json
```

启用后 console 与 `resources/logs/` 文件日志输出 JSONL。公共字段包括 `timestamp`、`level`、`message`、source、`trace_id`、`span_id`、`trace_sampled`；HTTP 请求内日志会带 `request_id`，解析单文件时会补充 `file_sha` 和 `engine`。

也可以先预置模型 volume，再关闭自动下载：

```bash
python download_models.py published
DEEPDOC_AUTO_DOWNLOAD=0 docker compose up -d
```

## Tracing

```bash
export DEEPDOC_TRACING_ENABLED=1
export DEEPDOC_TRACING_EXPORTER=otlp
export DEEPDOC_TRACING_OTLP_ENDPOINT=http://observability.example.com:4318/v1/traces
export DEEPDOC_TRACING_SERVICE_NAME=deepdoc-standalone
export DEEPDOC_TRACING_SAMPLE_RATIO=1.0
```

本地输出：

```bash
export DEEPDOC_TRACING_ENABLED=1
export DEEPDOC_TRACING_EXPORTER=console
```

当前 tracing 覆盖：

- Flask request spans
- outbound `requests` calls
- PostgreSQL `psycopg` queries
- `botocore` / S3 object storage requests
- parse、structured artifact build、ingest publish 等关键内部阶段

## 下载模式

```bash
curl -X POST "http://localhost:8000/api/v1/parse?download=true" \
  -F "file=@/path/to/document.pdf" \
  -o result.md
```

## 状态码

| 状态码 | 说明 |
|---:|---|
| `200` | 请求成功 |
| `202` | 异步任务已提交 |
| `400` | 参数错误或全部文件解析失败 |
| `401` | 未授权 |
| `403` | 权限不足 |
| `404` | 资源不存在 |
| `409` | 当前状态不允许操作 |
| `413` | 上传体超过限制 |
| `429` | 触发限流 |
| `500` | 服务内部错误 |
| `503` | 依赖不可用或服务未就绪 |

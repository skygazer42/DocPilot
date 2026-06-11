# DocPilot 文档解析接口对接说明

本文档说明 DocPilot Standalone 的接口边界：文档解析、结构化产物、切块、资产访问、异步任务和 ingest 查询。

服务不负责问答、回答生成或外部大模型调用。解析完成后的 `structured.json`、`chunks.jsonl`、`ingest.jsonl` 和资产引用就是可交付给业务系统使用的数据。

相关文档：

- 部署与运行参数：[DEPLOYMENT.md](DEPLOYMENT.md)
- 健康检查、审计、自检、日志与 tracing：[OPERATIONS.md](OPERATIONS.md)

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
发布配置、对象存储和后端连接参数见 [DEPLOYMENT.md](DEPLOYMENT.md)。

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

## 运维与部署

以下内容已拆分到专门文档：

- 生产脱敏、`/health`、`/ready`、`/metrics`、审计、自检、janitor、结构化日志、tracing：
  [OPERATIONS.md](OPERATIONS.md)
- Docker / Compose、对象存储、ingest 发布配置、模型下载、量化、provider 和 profiling：
  [DEPLOYMENT.md](DEPLOYMENT.md)

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

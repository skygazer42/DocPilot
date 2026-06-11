# Operations Guide

本文档覆盖运行态与运维平面：脱敏策略、健康检查、构建信息、审计、自检、清理、结构化日志和 tracing。解析接口本身见 [API.md](API.md)，部署与模型配置见 [DEPLOYMENT.md](DEPLOYMENT.md)。

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

## 健康与运维接口

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

## Structured Logs

默认日志格式为文本。需要容器/日志平台直接采集结构化字段时设置：

```bash
export DEEPDOC_LOG_FORMAT=json
```

启用后 console 与 `resources/logs/` 文件日志输出 JSONL。公共字段包括 `timestamp`、`level`、`message`、source、`trace_id`、`span_id`、`trace_sampled`；HTTP 请求内日志会带 `request_id`，解析单文件时会补充 `file_sha` 和 `engine`。

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

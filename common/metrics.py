from __future__ import annotations

from typing import Any

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest

    METRICS_ENABLED = True
except Exception:  # pragma: no cover - optional runtime fallback
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    METRICS_ENABLED = False

    class _NoOpMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            return None

        def dec(self, *args, **kwargs):
            return None

        def observe(self, *args, **kwargs):
            return None

        def set(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

    def Counter(*args, **kwargs):  # type: ignore[misc]
        return _NoOpMetric()

    def Gauge(*args, **kwargs):  # type: ignore[misc]
        return _NoOpMetric()

    def Histogram(*args, **kwargs):  # type: ignore[misc]
        return _NoOpMetric()

    def Info(*args, **kwargs):  # type: ignore[misc]
        return _NoOpMetric()

    def generate_latest(*args, **kwargs):  # type: ignore[misc]
        return b"# metrics disabled\n"


HTTP_REQUESTS_TOTAL = Counter(
    "deepdoc_http_requests_total",
    "Total HTTP requests handled by DeepDoc.",
    ["method", "route", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "deepdoc_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
HTTP_INFLIGHT_REQUESTS = Gauge(
    "deepdoc_http_inflight_requests",
    "Current in-flight HTTP requests.",
)
HTTP_EXCEPTIONS_TOTAL = Counter(
    "deepdoc_http_exceptions_total",
    "Unhandled HTTP exceptions by route.",
    ["method", "route", "exception_type"],
)
PARSE_RESULTS_TOTAL = Counter(
    "deepdoc_parse_results_total",
    "Parse result outcomes by parser engine and file type.",
    ["parser_engine", "file_type", "status", "cache_hit"],
)
PARSE_SOURCE_BYTES = Histogram(
    "deepdoc_parse_source_bytes",
    "Uploaded source file sizes in bytes.",
    ["parser_engine", "file_type"],
    buckets=(1024, 10 * 1024, 100 * 1024, 1024 * 1024, 5 * 1024 * 1024, 10 * 1024 * 1024, 50 * 1024 * 1024),
)
PARSE_ASSET_COUNT = Histogram(
    "deepdoc_parse_asset_count",
    "Asset counts emitted per parse result.",
    ["parser_engine", "file_type"],
    buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128),
)
PARSE_CHUNK_COUNT = Histogram(
    "deepdoc_parse_chunk_count",
    "Chunk counts emitted per parse result.",
    ["parser_engine", "file_type"],
    buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
)
INGEST_PUBLISH_TOTAL = Counter(
    "deepdoc_ingest_publish_total",
    "Ingest publish outcomes.",
    ["sink_type", "status"],
)
INGEST_PUBLISH_RECORD_COUNT = Histogram(
    "deepdoc_ingest_publish_record_count",
    "Record counts sent to ingest publishers.",
    ["sink_type", "status"],
    buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
)
BACKEND_INFO = Info(
    "deepdoc_backend",
    "DeepDoc backend runtime configuration.",
)
BUILD_INFO = Info(
    "deepdoc_build",
    "DeepDoc build provenance for the running image.",
)
BUILD_TIMESTAMP_SECONDS = Gauge(
    "deepdoc_build_timestamp_seconds",
    "Build timestamp of the running image as Unix epoch seconds.",
)
BUILD_AGE_SECONDS = Gauge(
    "deepdoc_build_age_seconds",
    "Age in seconds of the running image build.",
)
OCR_LOADED = Gauge(
    "deepdoc_ocr_loaded",
    "Whether local OCR models are loaded.",
)
TRACING_ENABLED = Gauge(
    "deepdoc_tracing_enabled",
    "Whether OpenTelemetry tracing is enabled.",
)
SELF_CHECK_RUNTIME_STATUS = Gauge(
    "deepdoc_self_check_runtime_status",
    "Current production self-check plane status as a one-hot gauge.",
    ["status"],
)
SELF_CHECK_WORKER_RUNTIME_STATUS = Gauge(
    "deepdoc_self_check_worker_runtime_status",
    "Current production self-check worker status as a one-hot gauge.",
    ["status"],
)
SELF_CHECK_AUTO_ENABLED = Gauge(
    "deepdoc_self_check_auto_enabled",
    "Whether the automatic production self-check worker is enabled.",
)
SELF_CHECK_REQUIRED_FOR_READY = Gauge(
    "deepdoc_self_check_required_for_ready",
    "Whether readiness requires a fresh successful production self-check.",
)
SELF_CHECK_LAST_RUN_AGE_SECONDS = Gauge(
    "deepdoc_self_check_last_run_age_seconds",
    "Age in seconds of the latest production self-check run.",
)
SELF_CHECK_LAST_RUN_MAX_AGE_SECONDS = Gauge(
    "deepdoc_self_check_last_run_max_age_seconds",
    "Maximum allowed age in seconds for the latest production self-check run when readiness gating is enabled.",
)
SELF_CHECK_WORKER_HEARTBEAT_AGE_SECONDS = Gauge(
    "deepdoc_self_check_worker_heartbeat_age_seconds",
    "Age in seconds of the latest production self-check worker heartbeat.",
)
SELF_CHECK_LATEST_RUN_DURATION_SECONDS = Gauge(
    "deepdoc_self_check_latest_run_duration_seconds",
    "Duration in seconds of the latest production self-check run.",
)
SELF_CHECK_LATEST_RUN_STATUS = Gauge(
    "deepdoc_self_check_latest_run_status",
    "Latest production self-check run status for the current suite as a one-hot gauge.",
    ["suite", "status"],
)
SELF_CHECK_LATEST_RUN_INFO = Info(
    "deepdoc_self_check_latest_run",
    "Latest production self-check run summary.",
)
RETENTION_JANITOR_RUNTIME_STATUS = Gauge(
    "deepdoc_retention_janitor_runtime_status",
    "Current retention janitor plane status as a one-hot gauge.",
    ["status"],
)
RETENTION_JANITOR_ENABLED = Gauge(
    "deepdoc_retention_janitor_enabled",
    "Whether the automatic retention janitor worker is enabled.",
)
RETENTION_JANITOR_REQUIRED_FOR_READY = Gauge(
    "deepdoc_retention_janitor_required_for_ready",
    "Whether readiness requires a fresh successful retention janitor run.",
)
RETENTION_JANITOR_LAST_RUN_AGE_SECONDS = Gauge(
    "deepdoc_retention_janitor_last_run_age_seconds",
    "Age in seconds of the latest retention janitor run.",
)
RETENTION_JANITOR_LAST_RUN_MAX_AGE_SECONDS = Gauge(
    "deepdoc_retention_janitor_last_run_max_age_seconds",
    "Maximum allowed age in seconds for the latest retention janitor run when readiness gating is enabled.",
)
RETENTION_JANITOR_WORKER_HEARTBEAT_AGE_SECONDS = Gauge(
    "deepdoc_retention_janitor_worker_heartbeat_age_seconds",
    "Age in seconds of the latest retention janitor worker heartbeat.",
)
RETENTION_JANITOR_LATEST_RUN_DURATION_SECONDS = Gauge(
    "deepdoc_retention_janitor_latest_run_duration_seconds",
    "Duration in seconds of the latest retention janitor run.",
)
RETENTION_JANITOR_LATEST_RUN_STATUS = Gauge(
    "deepdoc_retention_janitor_latest_run_status",
    "Latest retention janitor run status as a one-hot gauge.",
    ["status"],
)
RETENTION_JANITOR_LATEST_RUN_INFO = Info(
    "deepdoc_retention_janitor_latest_run",
    "Latest retention janitor run summary.",
)
RETENTION_JANITOR_RUNS_TOTAL = Counter(
    "deepdoc_retention_janitor_runs_total",
    "Retention janitor run outcomes.",
    ["status"],
)
RETENTION_JANITOR_DURATION_SECONDS = Histogram(
    "deepdoc_retention_janitor_duration_seconds",
    "Retention janitor run duration in seconds.",
    ["status"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 900),
)
RETENTION_JANITOR_PLANE_RUNS_TOTAL = Counter(
    "deepdoc_retention_janitor_plane_runs_total",
    "Retention janitor per-plane outcomes.",
    ["plane", "status"],
)
RETENTION_JANITOR_PLANE_DELETED = Histogram(
    "deepdoc_retention_janitor_plane_deleted",
    "Deleted object counts per retention janitor plane run.",
    ["plane", "status"],
    buckets=(0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000),
)
ASYNC_TASK_RUNTIME_STATUS = Gauge(
    "deepdoc_async_task_runtime_status",
    "Current async task plane status as a one-hot gauge.",
    ["status"],
)
ASYNC_TASK_WORKER_RUNTIME_STATUS = Gauge(
    "deepdoc_async_task_worker_runtime_status",
    "Current async task worker status as a one-hot gauge.",
    ["status"],
)
ASYNC_TASK_CALLBACK_REDRIVE_WORKER_RUNTIME_STATUS = Gauge(
    "deepdoc_async_task_callback_redrive_worker_runtime_status",
    "Current async task callback redrive worker status as a one-hot gauge.",
    ["status"],
)
ASYNC_TASK_ENABLED = Gauge(
    "deepdoc_async_task_enabled",
    "Whether async task processing is enabled.",
)
ASYNC_TASK_QUEUE_DEPTH = Gauge(
    "deepdoc_async_task_queue_depth",
    "Current async task broker queue depth by queue type.",
    ["queue"],
)
REQUEST_GUARD_DECISIONS_TOTAL = Counter(
    "deepdoc_request_guard_decisions_total",
    "Request protection decisions by component and outcome.",
    ["component", "scope", "status", "backend"],
)
ADMISSION_INFLIGHT = Gauge(
    "deepdoc_admission_inflight",
    "Current in-flight request counts for guarded pools.",
    ["pool"],
)
ASYNC_TASK_CALLBACK_DELIVERIES_TOTAL = Counter(
    "deepdoc_async_task_callback_deliveries_total",
    "Async task callback delivery outcomes.",
    ["event_type", "status"],
)
ASYNC_TASK_CALLBACK_DURATION_SECONDS = Histogram(
    "deepdoc_async_task_callback_duration_seconds",
    "Async task callback delivery latency in seconds.",
    ["event_type", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
ASYNC_TASK_CALLBACK_REDRIVE_RUNS_TOTAL = Counter(
    "deepdoc_async_task_callback_redrive_runs_total",
    "Async task callback redrive sweep outcomes.",
    ["source", "status"],
)
ASYNC_TASK_CALLBACK_REDRIVE_CANDIDATES = Histogram(
    "deepdoc_async_task_callback_redrive_candidates",
    "Candidate counts scanned by async task callback redrive sweeps.",
    ["source", "status"],
    buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
)
ASYNC_TASK_CALLBACK_REDRIVE_DELIVERED = Histogram(
    "deepdoc_async_task_callback_redrive_delivered",
    "Delivered callback counts produced by async task callback redrive sweeps.",
    ["source", "status"],
    buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
)
ASYNC_TASK_RETRY_RUNS_TOTAL = Counter(
    "deepdoc_async_task_retry_runs_total",
    "Async task retry operation outcomes.",
    ["source", "status"],
)
ASYNC_TASK_RETRY_CANDIDATES = Histogram(
    "deepdoc_async_task_retry_candidates",
    "Candidate counts considered by async task retry operations.",
    ["source", "status"],
    buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
)
ASYNC_TASK_RETRY_ENQUEUED = Histogram(
    "deepdoc_async_task_retry_enqueued",
    "Retry task counts enqueued by async task retry operations.",
    ["source", "status"],
    buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
)
GPU_PAGE_POOL_JOBS_TOTAL = Counter(
    "deepdoc_gpu_page_pool_jobs_total",
    "GPU page pool jobs planned by route and OCR scope.",
    ["route", "ocr_scope"],
)
GPU_PAGE_POOL_DEVICE_JOBS = Gauge(
    "deepdoc_gpu_page_pool_device_jobs",
    "Current planned GPU page pool jobs per visible worker device.",
    ["device_id"],
)
SELF_CHECK_RUNS_TOTAL = Counter(
    "deepdoc_self_check_runs_total",
    "Production self-check run outcomes.",
    ["suite", "status"],
)
SELF_CHECK_DURATION_SECONDS = Histogram(
    "deepdoc_self_check_duration_seconds",
    "Production self-check duration in seconds.",
    ["suite", "status"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)
SELF_CHECK_WORKER_RUNS_TOTAL = Counter(
    "deepdoc_self_check_worker_runs_total",
    "Automatic production self-check worker run outcomes.",
    ["suite", "status"],
)
SELF_CHECK_WORKER_DURATION_SECONDS = Histogram(
    "deepdoc_self_check_worker_duration_seconds",
    "Automatic production self-check worker run duration in seconds.",
    ["suite", "status"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 900),
)
_SELF_CHECK_RUNTIME_STATUSES = ("ok", "disabled", "error")
_SELF_CHECK_RUN_STATUSES = ("running", "passed", "failed")
_last_self_check_latest_run_label: tuple[str, str] | None = None
_RETENTION_JANITOR_RUNTIME_STATUSES = ("ok", "disabled", "error")
_RETENTION_JANITOR_RUN_STATUSES = ("ok", "error")
_last_retention_janitor_latest_run_status: str | None = None
_ASYNC_TASK_RUNTIME_STATUSES = ("ok", "disabled", "error")
_replayed_retention_janitor_run_ids: set[str] = set()
_replayed_self_check_run_ids: set[str] = set()
_replayed_callback_redrive_run_ids: set[str] = set()


def _set_one_hot_gauge(metric, *, allowed_statuses: tuple[str, ...], current_status: str) -> None:
    normalized_current = (current_status or "unknown").strip() or "unknown"
    for status in allowed_statuses:
        metric.labels(status=status).set(1 if normalized_current == status else 0)


def _set_optional_gauge(metric, value: Any) -> None:
    try:
        if value is None:
            metric.set(float("nan"))
            return
        metric.set(float(value))
    except Exception:
        metric.set(float("nan"))


def observe_http_request(method: str, route: str, status_code: int, duration_seconds: float) -> None:
    HTTP_REQUESTS_TOTAL.labels(method=method, route=route, status_code=str(int(status_code))).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, route=route).observe(max(0.0, float(duration_seconds)))


def observe_http_exception(method: str, route: str, exception_type: str) -> None:
    HTTP_EXCEPTIONS_TOTAL.labels(
        method=method,
        route=route,
        exception_type=exception_type or "Exception",
    ).inc()


def inc_http_inflight() -> None:
    HTTP_INFLIGHT_REQUESTS.inc()


def dec_http_inflight() -> None:
    HTTP_INFLIGHT_REQUESTS.dec()


def observe_parse_result(
    *,
    parser_engine: str,
    file_type: str,
    status: str,
    cache_hit: bool,
    source_bytes: int | None = None,
    asset_count: int | None = None,
    chunk_count: int | None = None,
) -> None:
    engine = (parser_engine or "unknown").strip() or "unknown"
    normalized_file_type = (file_type or "unknown").strip() or "unknown"
    PARSE_RESULTS_TOTAL.labels(
        parser_engine=engine,
        file_type=normalized_file_type,
        status=(status or "unknown").strip() or "unknown",
        cache_hit="true" if cache_hit else "false",
    ).inc()
    if source_bytes is not None:
        PARSE_SOURCE_BYTES.labels(parser_engine=engine, file_type=normalized_file_type).observe(max(0, int(source_bytes)))
    if asset_count is not None:
        PARSE_ASSET_COUNT.labels(parser_engine=engine, file_type=normalized_file_type).observe(max(0, int(asset_count)))
    if chunk_count is not None:
        PARSE_CHUNK_COUNT.labels(parser_engine=engine, file_type=normalized_file_type).observe(max(0, int(chunk_count)))


def observe_ingest_publish(*, sink_type: str, status: str, record_count: int) -> None:
    normalized_sink = (sink_type or "unknown").strip() or "unknown"
    normalized_status = (status or "unknown").strip() or "unknown"
    INGEST_PUBLISH_TOTAL.labels(sink_type=normalized_sink, status=normalized_status).inc()
    INGEST_PUBLISH_RECORD_COUNT.labels(sink_type=normalized_sink, status=normalized_status).observe(max(0, int(record_count)))


def update_backend_metrics(
    *,
    artifact_backend: str,
    ingest_publisher: str,
    ingest_query_backend: str | None,
    auth_mode: str,
    ocr_loaded: bool,
    tracing_enabled: bool,
    tracing_exporters: str | None,
) -> None:
    BACKEND_INFO.info(
        {
            "artifact_backend": artifact_backend or "unknown",
            "ingest_publisher": ingest_publisher or "unknown",
            "ingest_query_backend": ingest_query_backend or "none",
            "auth_mode": auth_mode or "unknown",
            "tracing_exporters": tracing_exporters or "none",
        }
    )
    OCR_LOADED.set(1 if ocr_loaded else 0)
    TRACING_ENABLED.set(1 if tracing_enabled else 0)


def update_build_metrics(build_info: dict[str, Any] | None) -> None:
    payload = build_info if isinstance(build_info, dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}

    def _text(value: Any) -> str:
        return str(value or "").strip()

    def _short(value: Any) -> str:
        text = _text(value)
        return text[:12] if text else ""

    BUILD_INFO.info(
        {
            "status": _text(payload.get("status")) or "unknown",
            "package_name": _text(payload.get("package_name")) or "deepdoc",
            "package_version": _text(payload.get("package_version")) or "0.0.0",
            "build_source": _text(payload.get("build_source")) or "unknown",
            "image_tag": _text(payload.get("image_tag")) or "none",
            "vcs_ref_short": (
                _text(payload.get("vcs_ref_short"))
                or _text(summary.get("vcs_ref_short"))
                or _short(payload.get("vcs_ref"))
                or "none"
            ),
            "source_tree_sha12": (
                _text(payload.get("source_tree_sha12"))
                or _text(summary.get("source_tree_sha12"))
                or _short(payload.get("source_tree_sha256"))
                or "none"
            ),
            "requirements_sha12": (
                _text(payload.get("requirements_sha12"))
                or _text(summary.get("requirements_sha12"))
                or _short(payload.get("requirements_sha256"))
                or "none"
            ),
            "pyproject_sha12": (
                _text(payload.get("pyproject_sha12"))
                or _text(summary.get("pyproject_sha12"))
                or _short(payload.get("pyproject_sha256"))
                or "none"
            ),
            "openapi_sha12": (
                _text(payload.get("openapi_sha12"))
                or _text(summary.get("openapi_sha12"))
                or _short(payload.get("openapi_sha256"))
                or "none"
            ),
        }
    )
    _set_optional_gauge(BUILD_TIMESTAMP_SECONDS, payload.get("build_timestamp_epoch_seconds"))
    _set_optional_gauge(BUILD_AGE_SECONDS, payload.get("build_age_seconds"))


def update_self_check_metrics(state: dict[str, Any] | None) -> None:
    global _last_self_check_latest_run_label

    payload = state if isinstance(state, dict) else {}
    top_status = str(payload.get("status") or "error").strip() or "error"
    worker = payload.get("worker") if isinstance(payload.get("worker"), dict) else {}
    latest_run = payload.get("latest_run") if isinstance(payload.get("latest_run"), dict) else {}

    _set_one_hot_gauge(
        SELF_CHECK_RUNTIME_STATUS,
        allowed_statuses=_SELF_CHECK_RUNTIME_STATUSES,
        current_status=top_status,
    )
    _set_one_hot_gauge(
        SELF_CHECK_WORKER_RUNTIME_STATUS,
        allowed_statuses=_SELF_CHECK_RUNTIME_STATUSES,
        current_status=str(worker.get("status") or "disabled").strip() or "disabled",
    )

    SELF_CHECK_AUTO_ENABLED.set(1 if bool(payload.get("auto_enabled")) else 0)
    SELF_CHECK_REQUIRED_FOR_READY.set(1 if bool(payload.get("required_for_ready")) else 0)
    _set_optional_gauge(SELF_CHECK_LAST_RUN_AGE_SECONDS, payload.get("last_run_age_seconds"))
    _set_optional_gauge(SELF_CHECK_LAST_RUN_MAX_AGE_SECONDS, payload.get("last_run_max_age_seconds"))
    _set_optional_gauge(SELF_CHECK_WORKER_HEARTBEAT_AGE_SECONDS, worker.get("age_seconds"))

    suite = str(latest_run.get("suite") or "").strip()
    run_status = str(latest_run.get("status") or "").strip()
    if _last_self_check_latest_run_label is not None:
        previous_suite, previous_status = _last_self_check_latest_run_label
        if previous_suite and previous_status:
            SELF_CHECK_LATEST_RUN_STATUS.labels(suite=previous_suite, status=previous_status).set(0)
    if suite and run_status:
        for status in _SELF_CHECK_RUN_STATUSES:
            SELF_CHECK_LATEST_RUN_STATUS.labels(suite=suite, status=status).set(1 if run_status == status else 0)
        _last_self_check_latest_run_label = (suite, run_status)
    else:
        _last_self_check_latest_run_label = None

    duration_ms = latest_run.get("duration_ms")
    _set_optional_gauge(
        SELF_CHECK_LATEST_RUN_DURATION_SECONDS,
        None if duration_ms is None else max(0.0, float(duration_ms) / 1000.0),
    )
    SELF_CHECK_LATEST_RUN_INFO.info(
        {
            "check_id": str(latest_run.get("check_id") or ""),
            "suite": suite or "",
            "status": run_status or "",
            "finished_at": str(latest_run.get("finished_at") or ""),
        }
    )

    history_payload = payload.get("history")
    history = [item for item in history_payload if isinstance(item, dict)] if isinstance(history_payload, list) else []
    if not history:
        worker_history_payload = worker.get("history")
        history = (
            [item for item in worker_history_payload if isinstance(item, dict)]
            if isinstance(worker_history_payload, list)
            else []
        )
    for run in history:
        check_id = str(run.get("check_id") or "").strip()
        if not check_id or check_id in _replayed_self_check_run_ids:
            continue
        run_suite = str(run.get("suite") or "unknown").strip() or "unknown"
        run_status = str(run.get("status") or "unknown").strip() or "unknown"
        if run_status not in {"passed", "failed", "error"}:
            continue
        duration_ms = run.get("duration_ms")
        observe_self_check_run(suite=run_suite, status=run_status, duration_ms=duration_ms)
        if bool(run.get("auto_run")) or str(run.get("requested_by") or "").strip() == "self-check-worker":
            observe_self_check_worker_run(suite=run_suite, status=run_status, duration_ms=duration_ms)
        _replayed_self_check_run_ids.add(check_id)


def update_retention_janitor_metrics(state: dict[str, Any] | None) -> None:
    global _last_retention_janitor_latest_run_status

    payload = state if isinstance(state, dict) else {}
    top_status = str(payload.get("status") or "disabled").strip() or "disabled"
    worker = payload.get("worker") if isinstance(payload.get("worker"), dict) else {}
    latest_run = payload.get("latest_run") if isinstance(payload.get("latest_run"), dict) else {}
    run_status = str(latest_run.get("status") or "").strip()
    history_payload = worker.get("history")
    history = [item for item in history_payload if isinstance(item, dict)] if isinstance(history_payload, list) else []

    _set_one_hot_gauge(
        RETENTION_JANITOR_RUNTIME_STATUS,
        allowed_statuses=_RETENTION_JANITOR_RUNTIME_STATUSES,
        current_status=top_status,
    )
    RETENTION_JANITOR_ENABLED.set(1 if bool(payload.get("enabled")) else 0)
    RETENTION_JANITOR_REQUIRED_FOR_READY.set(1 if bool(payload.get("required_for_ready")) else 0)
    _set_optional_gauge(RETENTION_JANITOR_LAST_RUN_AGE_SECONDS, payload.get("last_run_age_seconds"))
    _set_optional_gauge(RETENTION_JANITOR_LAST_RUN_MAX_AGE_SECONDS, payload.get("last_run_max_age_seconds"))
    _set_optional_gauge(RETENTION_JANITOR_WORKER_HEARTBEAT_AGE_SECONDS, worker.get("age_seconds"))
    _set_optional_gauge(
        RETENTION_JANITOR_LATEST_RUN_DURATION_SECONDS,
        None if latest_run.get("duration_ms") is None else max(0.0, float(latest_run.get("duration_ms")) / 1000.0),
    )

    if _last_retention_janitor_latest_run_status is not None:
        RETENTION_JANITOR_LATEST_RUN_STATUS.labels(status=_last_retention_janitor_latest_run_status).set(0)
    if run_status:
        for allowed_status in _RETENTION_JANITOR_RUN_STATUSES:
            RETENTION_JANITOR_LATEST_RUN_STATUS.labels(status=allowed_status).set(1 if run_status == allowed_status else 0)
        _last_retention_janitor_latest_run_status = run_status
    else:
        _last_retention_janitor_latest_run_status = None

    RETENTION_JANITOR_LATEST_RUN_INFO.info(
        {
            "run_id": str(latest_run.get("run_id") or ""),
            "status": run_status or "",
            "finished_at": str(latest_run.get("finished_at") or ""),
        }
    )

    for run in history:
        run_id = str(run.get("run_id") or "").strip()
        if not run_id or run_id in _replayed_retention_janitor_run_ids:
            continue
        observe_retention_janitor_run(
            status=str(run.get("status") or "unknown").strip() or "unknown",
            duration_ms=run.get("duration_ms"),
            plane_results=run.get("planes") if isinstance(run.get("planes"), dict) else None,
        )
        _replayed_retention_janitor_run_ids.add(run_id)


def update_async_task_metrics(state: dict[str, Any] | None) -> None:
    payload = state if isinstance(state, dict) else {}
    enabled = bool(payload.get("enabled"))
    store = payload.get("store") if isinstance(payload.get("store"), dict) else {}
    broker = payload.get("broker") if isinstance(payload.get("broker"), dict) else {}
    worker = payload.get("worker") if isinstance(payload.get("worker"), dict) else {}
    callback_redrive = payload.get("callback_redrive_worker") if isinstance(payload.get("callback_redrive_worker"), dict) else {}

    top_status = "disabled"
    if enabled:
        top_status = "ok"
        if str(store.get("status") or "ok") != "ok":
            top_status = "error"
        elif str(broker.get("status") or "disabled") not in {"ok", "disabled"}:
            top_status = "error"
        elif str(worker.get("status") or "disabled") not in {"ok", "disabled"}:
            top_status = "error"
        elif str(callback_redrive.get("status") or "disabled") not in {"ok", "disabled"}:
            top_status = "error"

    _set_one_hot_gauge(
        ASYNC_TASK_RUNTIME_STATUS,
        allowed_statuses=_ASYNC_TASK_RUNTIME_STATUSES,
        current_status=top_status,
    )
    _set_one_hot_gauge(
        ASYNC_TASK_WORKER_RUNTIME_STATUS,
        allowed_statuses=_ASYNC_TASK_RUNTIME_STATUSES,
        current_status=str(worker.get("status") or "disabled").strip() or "disabled",
    )
    _set_one_hot_gauge(
        ASYNC_TASK_CALLBACK_REDRIVE_WORKER_RUNTIME_STATUS,
        allowed_statuses=_ASYNC_TASK_RUNTIME_STATUSES,
        current_status=str(callback_redrive.get("status") or "disabled").strip() or "disabled",
    )
    ASYNC_TASK_ENABLED.set(1 if enabled else 0)

    queues = broker.get("queues") if isinstance(broker.get("queues"), dict) else {}
    _set_optional_gauge(ASYNC_TASK_QUEUE_DEPTH.labels(queue="queued"), queues.get("queued"))
    _set_optional_gauge(ASYNC_TASK_QUEUE_DEPTH.labels(queue="processing"), queues.get("processing"))

    history_payload = callback_redrive.get("history")
    history = [item for item in history_payload if isinstance(item, dict)] if isinstance(history_payload, list) else []
    for run in history:
        run_id = str(run.get("run_id") or "").strip()
        if not run_id or run_id in _replayed_callback_redrive_run_ids:
            continue
        observe_async_task_callback_redrive_run(
            source=str(run.get("metrics_source") or "worker").strip() or "worker",
            status=str(run.get("status") or "unknown").strip() or "unknown",
            candidate_count=max(0, int(run.get("candidate_count") or 0)),
            delivered_count=max(0, int(run.get("delivered") or 0)),
        )
        _replayed_callback_redrive_run_ids.add(run_id)


def observe_request_guard_decision(
    *,
    component: str,
    scope: str,
    status: str,
    backend: str,
) -> None:
    REQUEST_GUARD_DECISIONS_TOTAL.labels(
        component=(component or "unknown").strip() or "unknown",
        scope=(scope or "unknown").strip() or "unknown",
        status=(status or "unknown").strip() or "unknown",
        backend=(backend or "unknown").strip() or "unknown",
    ).inc()


def observe_retention_janitor_run(
    *,
    status: str,
    duration_ms: int | None,
    plane_results: dict[str, dict[str, Any]] | None = None,
) -> None:
    normalized_status = (status or "unknown").strip() or "unknown"
    RETENTION_JANITOR_RUNS_TOTAL.labels(status=normalized_status).inc()
    if duration_ms is not None:
        RETENTION_JANITOR_DURATION_SECONDS.labels(status=normalized_status).observe(max(0.0, float(duration_ms) / 1000.0))
    for plane_name, result in (plane_results or {}).items():
        if not isinstance(result, dict):
            continue
        plane_status = str(result.get("status") or "unknown").strip() or "unknown"
        RETENTION_JANITOR_PLANE_RUNS_TOTAL.labels(plane=plane_name, status=plane_status).inc()
        deleted_count_raw = result.get("deleted_count")
        if deleted_count_raw is None:
            deleted_count_raw = result.get("deleted")
            if isinstance(deleted_count_raw, list):
                deleted_count_raw = len(deleted_count_raw)
        try:
            deleted_count = max(0, int(deleted_count_raw or 0))
        except Exception:
            deleted_count = 0
        RETENTION_JANITOR_PLANE_DELETED.labels(plane=plane_name, status=plane_status).observe(deleted_count)


def inc_admission_inflight(pool: str) -> None:
    ADMISSION_INFLIGHT.labels(pool=(pool or "unknown").strip() or "unknown").inc()


def dec_admission_inflight(pool: str) -> None:
    ADMISSION_INFLIGHT.labels(pool=(pool or "unknown").strip() or "unknown").dec()


def observe_async_task_callback_delivery(*, event_type: str, status: str, duration_ms: int | None = None) -> None:
    normalized_event_type = (event_type or "unknown").strip() or "unknown"
    normalized_status = (status or "unknown").strip() or "unknown"
    ASYNC_TASK_CALLBACK_DELIVERIES_TOTAL.labels(
        event_type=normalized_event_type,
        status=normalized_status,
    ).inc()
    if duration_ms is not None:
        ASYNC_TASK_CALLBACK_DURATION_SECONDS.labels(
            event_type=normalized_event_type,
            status=normalized_status,
        ).observe(max(0.0, float(duration_ms) / 1000.0))


def observe_async_task_callback_redrive_run(
    *,
    source: str,
    status: str,
    candidate_count: int,
    delivered_count: int,
) -> None:
    normalized_source = (source or "unknown").strip() or "unknown"
    normalized_status = (status or "unknown").strip() or "unknown"
    ASYNC_TASK_CALLBACK_REDRIVE_RUNS_TOTAL.labels(
        source=normalized_source,
        status=normalized_status,
    ).inc()
    ASYNC_TASK_CALLBACK_REDRIVE_CANDIDATES.labels(
        source=normalized_source,
        status=normalized_status,
    ).observe(max(0, int(candidate_count)))
    ASYNC_TASK_CALLBACK_REDRIVE_DELIVERED.labels(
        source=normalized_source,
        status=normalized_status,
    ).observe(max(0, int(delivered_count)))


def observe_async_task_retry_run(
    *,
    source: str,
    status: str,
    candidate_count: int,
    enqueued_count: int,
) -> None:
    normalized_source = (source or "unknown").strip() or "unknown"
    normalized_status = (status or "unknown").strip() or "unknown"
    ASYNC_TASK_RETRY_RUNS_TOTAL.labels(
        source=normalized_source,
        status=normalized_status,
    ).inc()
    ASYNC_TASK_RETRY_CANDIDATES.labels(
        source=normalized_source,
        status=normalized_status,
    ).observe(max(0, int(candidate_count)))
    ASYNC_TASK_RETRY_ENQUEUED.labels(
        source=normalized_source,
        status=normalized_status,
    ).observe(max(0, int(enqueued_count)))


def observe_gpu_page_pool_dispatch(
    *,
    page_jobs: list[dict[str, Any]] | None,
    device_job_counts: dict[int, int] | None,
) -> None:
    for job in page_jobs or []:
        GPU_PAGE_POOL_JOBS_TOTAL.labels(
            route=str(job.get("route") or "unknown").strip() or "unknown",
            ocr_scope=str(job.get("ocr_scope") or "unknown").strip() or "unknown",
        ).inc()
    for device_id, count in (device_job_counts or {}).items():
        GPU_PAGE_POOL_DEVICE_JOBS.labels(device_id=str(int(device_id))).set(max(0, int(count)))


def observe_self_check_run(*, suite: str, status: str, duration_ms: int | None = None) -> None:
    normalized_suite = (suite or "unknown").strip() or "unknown"
    normalized_status = (status or "unknown").strip() or "unknown"
    SELF_CHECK_RUNS_TOTAL.labels(suite=normalized_suite, status=normalized_status).inc()
    if duration_ms is not None:
        SELF_CHECK_DURATION_SECONDS.labels(
            suite=normalized_suite,
            status=normalized_status,
        ).observe(max(0.0, float(duration_ms) / 1000.0))


def observe_self_check_worker_run(*, suite: str, status: str, duration_ms: int | None = None) -> None:
    normalized_suite = (suite or "unknown").strip() or "unknown"
    normalized_status = (status or "unknown").strip() or "unknown"
    SELF_CHECK_WORKER_RUNS_TOTAL.labels(
        suite=normalized_suite,
        status=normalized_status,
    ).inc()
    if duration_ms is not None:
        SELF_CHECK_WORKER_DURATION_SECONDS.labels(
            suite=normalized_suite,
            status=normalized_status,
        ).observe(max(0.0, float(duration_ms) / 1000.0))


def render_metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

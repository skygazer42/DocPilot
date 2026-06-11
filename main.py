# ruff: noqa: E402

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Number of distinct clusters")
import base64
import hashlib
import hmac
import inspect
import json
import os
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv(override=False)
from common import setting, logger

os.environ["TIKTOKEN_CACHE_DIR"] = setting.TIKTOKEN_CACHE_DIR
import time
import tempfile
import traceback
import re
import shutil
from importlib import import_module
from urllib.parse import quote
from urllib.parse import urlparse
from flask import Flask, request, jsonify, Response, g, has_request_context, has_app_context, stream_with_context
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from gevent.pywsgi import WSGIServer
import numpy as np
import fitz
from PIL import Image, ImageDraw
from common.file_utils import validate_file
from common.audit_log import AuditEvent, create_audit_log_store
from common.branding import API_DISPLAY_NAME, PRODUCT_NAME, PRODUCT_SLUG, SELF_CHECK_AUTHOR, SELF_CHECK_TITLE
from common.build_info import get_build_info, summarize_build_info
from common.errors import ErrorCode, build_error_payload, enrich_error_payload, normalize_error_locale
from common.markdown_utils import post_process_markdown, results_to_markdown
from common.metrics import (
    dec_admission_inflight,
    dec_http_inflight,
    inc_admission_inflight,
    inc_http_inflight,
    observe_gpu_page_pool_dispatch,
    observe_request_guard_decision,
    observe_http_exception,
    observe_http_request,
    observe_ingest_publish,
    observe_parse_result,
    observe_retention_janitor_run,
    observe_async_task_callback_redrive_run,
    observe_async_task_retry_run,
    observe_self_check_run,
    render_metrics_payload,
    update_async_task_metrics,
    update_build_metrics,
    update_retention_janitor_metrics,
    update_self_check_metrics,
    update_backend_metrics,
)
from common.tracing import (
    add_span_event,
    get_tracing_state,
    initialize_tracing,
    set_current_span_attributes,
    trace_operation,
)
from common.parse_artifacts import (
    ARTIFACT_PROFILE_VERSION,
    DEFAULT_CHUNK_MAX_TOKENS,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_STRATEGY,
    ParseArtifact,
    ParseAsset,
    ParseBlock,
    ParseChunk,
    ParseManifest,
    build_artifact_key,
    build_parse_manifest,
    build_ingest_export_records,
    build_chunk_export_records,
    build_document,
    count_tokens,
    create_artifact_store,
    normalize_chunk_strategy,
    parse_manifest_payload,
)
from common.ingest_publisher import IngestPublishError, build_ingest_publish_state, create_ingest_publisher
from common.ingest_postgres import PostgresIngestStore, load_postgres_ingest_config_from_env
from common.async_tasks import (
    AsyncTask,
    AsyncTaskCallbackConfig,
    AsyncTaskInput,
    LocalUploadedFile,
    build_async_task,
    build_async_retry_task,
    create_async_task_broker,
    create_async_task_store,
    task_access_payload,
)
from common.model_store import get_download_groups_from_env, list_missing_files
from common.parse_builders import build_csv_artifact, build_deepdoc_artifact, build_epub_artifact, build_generic_artifact
from common.parse_builders import build_image_artifact, build_mineru_artifact, build_paddleocr_artifact
from common.parse_builders import build_native_pdf_artifact, build_rich_text_artifact
from common.ratelimit import create_inflight_admission_controller, create_request_rate_limiter
from common.retention_janitor import load_retention_janitor_config
from common.self_check import SelfCheckRun, SelfCheckStep, SelfCheckStore, new_self_check_run, summarize_self_check_run
from common.task_callbacks import deliver_async_task_callback

os.environ.setdefault("DEEPDOC_MODEL_PATH", setting.MODELS_DIR)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
TRACING_STATE = initialize_tracing(app)

api_key = os.environ.get("SECRET_ACCESS_KEY")
auth_header = os.environ.get("DEEPDOC_AUTH_HEADER", "Authorization")
DEFAULT_AUTH_EXEMPT_PATHS = "/health,/ready,/metrics,/docs,/docs/,/docs/openapi.json,/openapi.json,/api/v1/openapi.json,/api/v1/build-info"
DEFAULT_CORS_ALLOWED_HEADERS = "Authorization,Content-Type,X-API-Key,X-Request-ID,X-Tenant-ID"
DEFAULT_CORS_EXPOSE_HEADERS = (
    "X-Request-ID,"
    "X-RateLimit-General-Limit,X-RateLimit-General-Remaining,X-RateLimit-General-Reset,X-RateLimit-General-Policy,"
    "X-RateLimit-Parse-Limit,X-RateLimit-Parse-Remaining,X-RateLimit-Parse-Reset,X-RateLimit-Parse-Policy,"
    "X-RateLimit-Ingest-Limit,X-RateLimit-Ingest-Remaining,X-RateLimit-Ingest-Reset,X-RateLimit-Ingest-Policy,"
    "X-RateLimit-Artifact-Limit,X-RateLimit-Artifact-Remaining,X-RateLimit-Artifact-Reset,X-RateLimit-Artifact-Policy,"
    "X-Admission-Pool,X-Admission-Limit,X-Admission-InFlight,X-Admission-Queue,X-Admission-Policy"
)
DEFAULT_CORS_METHODS = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
DEFAULT_CORS_RESOURCE_PATTERNS = (
    r"/api/.*",
    r"/openapi\.json",
    r"/api/v1/openapi\.json",
    r"/docs/openapi\.json",
)
DEFAULT_TENANT_ID = (os.environ.get("DEEPDOC_DEFAULT_TENANT_ID") or "").strip() or None
INSECURE_PLACEHOLDER_VALUES = {
    "",
    "change-me",
    "changeme",
    "change_me",
    "replace-me",
    "replace_me",
    "your_token",
    "<your_token>",
    "<your-secret>",
    "secret",
    "password",
    "deepdoc",
    "admin",
    "minioadmin",
}


def _parse_csv_list(value: str | None) -> list[str]:
    result: list[str] = []
    for part in (value or "").replace("\n", ",").split(","):
        normalized = part.strip()
        if normalized:
            result.append(normalized)
    return result


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_csv_set(value: str | None) -> set[str]:
    result: set[str] = set()
    result.update(_parse_csv_list((value or "").replace(" ", ",")))
    return result


AUTH_ADMIN_SCOPES = _parse_csv_set(os.environ.get("DEEPDOC_AUTH_ADMIN_SCOPES")) or {"admin"}
_auth_exempt_entries = _parse_csv_list(os.environ.get("DEEPDOC_AUTH_EXEMPT_PATHS", DEFAULT_AUTH_EXEMPT_PATHS))
auth_exempt_paths = {entry for entry in _auth_exempt_entries if not entry.endswith("/") and not entry.endswith("*")}
auth_exempt_prefixes = {
    entry[:-1] if entry.endswith("*") else entry
    for entry in _auth_exempt_entries
    if entry.endswith("/") or entry.endswith("*")
}


def _is_auth_exempt_path(path: str | None) -> bool:
    normalized = (path or "").strip()
    if not normalized:
        return False
    if normalized in auth_exempt_paths:
        return True
    return any(normalized.startswith(prefix) for prefix in auth_exempt_prefixes)


def _is_preflight_request() -> bool:
    return bool(has_request_context() and request.method == "OPTIONS" and request.headers.get("Origin"))


def _request_public_base_url() -> str:
    if not has_request_context():
        return ""
    forwarded_proto = (request.headers.get("X-Forwarded-Proto") or request.scheme or "http").split(",")[0].strip()
    forwarded_host = (
        request.headers.get("X-Forwarded-Host") or request.headers.get("Host") or request.host or ""
    ).split(",")[0].strip()
    forwarded_prefix = (request.headers.get("X-Forwarded-Prefix") or "").split(",")[0].strip()
    if forwarded_prefix:
        if not forwarded_prefix.startswith("/"):
            forwarded_prefix = f"/{forwarded_prefix}"
        forwarded_prefix = forwarded_prefix.rstrip("/")
    else:
        forwarded_prefix = (request.script_root or "").rstrip("/")
    base_url = f"{forwarded_proto}://{forwarded_host}".rstrip("/")
    if forwarded_prefix:
        base_url = f"{base_url}{forwarded_prefix}"
    return base_url


def _api_docs_state() -> dict[str, str]:
    base_url = _request_public_base_url().rstrip("/")
    return {
        "docs_url": f"{base_url}/docs".rstrip("/"),
        "openapi_url": f"{base_url}/openapi.json".rstrip("/"),
        "build_info_url": f"{base_url}/api/v1/build-info".rstrip("/"),
    }


def _health_internal_details_requested() -> bool:
    if not has_request_context():
        return False
    return _parse_bool(request.args.get("include_internal"), default=False)


def _health_internal_details_exposed() -> bool:
    if _parse_bool(os.environ.get("DEEPDOC_HEALTH_EXPOSE_INTERNALS"), default=False):
        return True
    if not _health_internal_details_requested():
        return False
    auth_context = _current_auth_context()
    return bool(auth_context.get("is_admin"))


def _api_internal_details_requested() -> bool:
    if not has_request_context():
        return False
    if _parse_bool(request.args.get("include_internal"), default=False):
        return True
    if _parse_bool(request.form.get("include_internal"), default=False):
        return True
    if request.is_json:
        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            return _parse_bool(payload.get("include_internal"), default=False)
    return False


def _api_internal_details_exposed() -> bool:
    if _parse_bool(os.environ.get("DEEPDOC_API_EXPOSE_INTERNALS"), default=False):
        return True
    if not _api_internal_details_requested():
        return False
    auth_context = _current_auth_context()
    return bool(auth_context.get("is_admin"))


_REDACTABLE_API_URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"()]+")
_REDACTABLE_API_HOST_RE = re.compile(r"(host=')([^']+)(')")
_REDACTABLE_API_FS_PATH_RE = re.compile(r"(?:(?<=\s)|^)(/(?:app|data|home|root|run|srv|tmp|usr|var|work)[^\s'\",)]*)")


def _sanitize_api_sensitive_text(text: str) -> str:
    sanitized = _REDACTABLE_API_URL_RE.sub("<redacted>", str(text))
    sanitized = _REDACTABLE_API_HOST_RE.sub(r"\1<redacted>\3", sanitized)
    sanitized = _REDACTABLE_API_FS_PATH_RE.sub("<redacted>", sanitized)
    return sanitized


def _sanitize_api_payload(value, *, path: tuple[str, ...] = (), expose_internal: bool = False):
    if expose_internal:
        return deepcopy(value)
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            sanitized[str(key)] = _sanitize_api_payload(item, path=path + (str(key),), expose_internal=expose_internal)
        return sanitized
    if isinstance(value, list):
        return [
            _sanitize_api_payload(item, path=path + (str(index),), expose_internal=expose_internal)
            for index, item in enumerate(value)
        ]

    key = path[-1] if path else ""
    parent = path[-2] if len(path) >= 2 else ""
    if key in {
        "destination",
        "dsn",
        "redis_url",
        "endpoint",
        "callback_url",
        "request_url",
        "source_path",
        "absolute_path",
        "storage_absolute_path",
        "root_dir",
        "queue_name",
        "processing_queue_name",
        "events_path",
        "source_dir",
        "task_dir",
        "task_path",
        "result_path",
        "callback_events_path",
        "build_info_path",
        "paddle_api_url",
        "mineru_api",
        "mineru_server_url",
        "source_url",
        "storage_source_url",
    }:
        return "<redacted>"
    if key == "url" and parent == "callback":
        return "<redacted>"
    if isinstance(value, str) and key in {"last_error", "error", "detail", "message", "response_body_snippet"}:
        return _sanitize_api_sensitive_text(value)
    return value


def _external_response_payload(payload):
    return _sanitize_api_payload(
        enrich_error_payload(payload, locale=_request_error_locale()),
        expose_internal=_api_internal_details_exposed(),
    )


def _jsonify_external(payload, status_code: int = 200):
    response = jsonify(_external_response_payload(payload))
    response.status_code = status_code
    return response


def _request_error_locale() -> str:
    if not has_request_context():
        return "en-US"
    return normalize_error_locale(request.headers.get("Accept-Language"))


def _jsonify_error(
    code: ErrorCode | str,
    *,
    status_code: int,
    message: str | None = None,
    details: dict[str, object] | None = None,
):
    response = jsonify(build_error_payload(code, message=message, locale=_request_error_locale(), details=details))
    response.status_code = status_code
    return response


def _apply_unified_error_response(response: Response) -> Response:
    if response.status_code < 400 or not response.is_json or response.is_streamed:
        return response
    raw_body = response.get_data()
    if b'"error"' not in raw_body:
        return response
    payload = response.get_json(silent=True)
    if payload is None:
        return response
    enriched = enrich_error_payload(payload, status_code=response.status_code, locale=_request_error_locale())
    response.set_data(json.dumps(enriched, ensure_ascii=False))
    response.content_type = "application/json; charset=utf-8"
    return response


def _sanitize_health_payload(value, *, path: tuple[str, ...] = (), expose_internal: bool = False):
    if expose_internal:
        return deepcopy(value)
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            child_path = path + (str(key),)
            if child_path == ("async_tasks", "callback_redrive_worker", "run", "results"):
                continue
            sanitized[str(key)] = _sanitize_health_payload(item, path=child_path, expose_internal=expose_internal)
        return sanitized
    if isinstance(value, list):
        return [
            _sanitize_health_payload(item, path=path + (str(index),), expose_internal=expose_internal)
            for index, item in enumerate(value)
        ]

    if path[:1] in {("api_docs",), ("cors",)}:
        return value

    key = path[-1] if path else ""
    if key in {
        "url",
        "api_url",
        "redis_url",
        "endpoint",
        "callback_url",
        "request_url",
        "dsn",
        "trace_id",
        "span_id",
        "request_id",
        "task_id",
        "delivery_id",
        "last_delivery_id",
        "default_tenant_id",
        "queue_name",
        "processing_queue_name",
    }:
        return "<redacted>"
    if key in {
        "path",
        "root_dir",
        "events_path",
        "source_dir",
        "task_dir",
        "task_path",
        "result_path",
        "callback_events_path",
        "build_info_path",
        "model_path",
    }:
        return "<redacted>"
    return value


@lru_cache(maxsize=1)
def _load_openapi_template() -> dict[str, object]:
    openapi_path = Path(setting.BASE_DIR) / "openapi.json"
    return json.loads(openapi_path.read_text(encoding="utf-8"))


def _build_openapi_payload() -> dict[str, object]:
    payload = deepcopy(_load_openapi_template())
    public_base_url = _request_public_base_url()
    if public_base_url:
        original_servers = payload.get("servers")
        if isinstance(original_servers, list):
            payload["x-original-servers"] = deepcopy(original_servers)
        payload["servers"] = [{"url": public_base_url, "description": "Current request origin"}]
    return payload


def _cors_health_state() -> dict[str, object]:
    enabled = _parse_bool(os.environ.get("DEEPDOC_CORS_ENABLED"), True)
    allowed_origins = _parse_csv_list(os.environ.get("DEEPDOC_CORS_ALLOWED_ORIGINS"))
    allowed_origin_regex = (os.environ.get("DEEPDOC_CORS_ALLOWED_ORIGIN_REGEX") or "").strip() or None
    allow_all = _parse_bool(
        os.environ.get("DEEPDOC_CORS_ALLOW_ALL"),
        default=not allowed_origins and not allowed_origin_regex,
    )
    return {
        "enabled": enabled,
        "allow_all": allow_all,
        "allowed_origins": allowed_origins,
        "allowed_origin_regex": allowed_origin_regex,
        "resources": list(DEFAULT_CORS_RESOURCE_PATTERNS),
        "methods": _parse_csv_list(os.environ.get("DEEPDOC_CORS_ALLOWED_METHODS", DEFAULT_CORS_METHODS)),
        "allowed_headers": _parse_csv_list(os.environ.get("DEEPDOC_CORS_ALLOWED_HEADERS", DEFAULT_CORS_ALLOWED_HEADERS)),
        "exposed_headers": _parse_csv_list(os.environ.get("DEEPDOC_CORS_EXPOSE_HEADERS", DEFAULT_CORS_EXPOSE_HEADERS)),
        "supports_credentials": _parse_bool(os.environ.get("DEEPDOC_CORS_SUPPORTS_CREDENTIALS"), False),
        "max_age_seconds": int(os.environ.get("DEEPDOC_CORS_MAX_AGE_SECONDS", "600")),
    }


def _is_strict_config_enabled() -> bool:
    environment = (os.environ.get("DEEPDOC_ENVIRONMENT") or os.environ.get("APP_ENV") or "").strip().lower()
    default = environment in {"prod", "production"}
    return _parse_bool(os.environ.get("DEEPDOC_CONFIG_STRICT"), default=default)


def _looks_insecure_secret(value: str | None) -> bool:
    normalized = (value or "").strip()
    return normalized.lower() in INSECURE_PLACEHOLDER_VALUES or len(normalized) < 16


def _dsn_contains_placeholder_secret(value: str | None) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    parsed = urlparse(text)
    password = parsed.password or ""
    username = parsed.username or ""
    if _looks_insecure_secret(password):
        return True
    return username == "deepdoc" and password == "deepdoc"


def _validate_runtime_config() -> dict[str, object]:
    strict = _is_strict_config_enabled()
    issues: list[str] = []
    warnings: list[str] = []
    auth_mode = (os.environ.get("DEEPDOC_AUTH_MODE") or "").strip().lower()
    jwt_enabled = auth_mode == "jwt_hs256"
    api_key_enabled = bool((api_key or "").strip()) or auth_mode == "api_key"

    if strict:
        if jwt_enabled:
            if _looks_insecure_secret(os.environ.get("DEEPDOC_AUTH_JWT_SECRET")):
                issues.append("DEEPDOC_AUTH_JWT_SECRET must be set to a strong non-placeholder value")
        elif api_key_enabled:
            if _looks_insecure_secret(api_key):
                issues.append("SECRET_ACCESS_KEY must be set to a strong non-placeholder value")
        else:
            issues.append("Authentication must be enabled with SECRET_ACCESS_KEY or DEEPDOC_AUTH_MODE=jwt_hs256")

        for env_name in ("DEEPDOC_INGEST_PG_DSN", "DEEPDOC_AUDIT_PG_DSN"):
            if _dsn_contains_placeholder_secret(os.environ.get(env_name)):
                issues.append(f"{env_name} contains a placeholder or weak database password")
        postgres_password = (os.environ.get("DEEPDOC_POSTGRES_PASSWORD") or "").strip()
        if postgres_password and _looks_insecure_secret(postgres_password):
            issues.append("DEEPDOC_POSTGRES_PASSWORD must be set to a strong non-placeholder value")

        cors_state = _cors_health_state()
        if cors_state.get("enabled") and cors_state.get("allow_all"):
            issues.append("DEEPDOC_CORS_ALLOW_ALL must be disabled in strict production configuration")
        if _parse_bool(os.environ.get("DEEPDOC_API_EXPOSE_INTERNALS"), default=False):
            issues.append("DEEPDOC_API_EXPOSE_INTERNALS must be disabled in strict production configuration")
        if _parse_bool(os.environ.get("DEEPDOC_HEALTH_EXPOSE_INTERNALS"), default=False):
            issues.append("DEEPDOC_HEALTH_EXPOSE_INTERNALS must be disabled in strict production configuration")

    if issues:
        raise RuntimeError("Invalid strict DeepDoc runtime configuration: " + "; ".join(issues))
    if warnings:
        logger.warning("Runtime configuration warnings: %s", "; ".join(warnings))
    return {
        "strict": strict,
        "environment": (os.environ.get("DEEPDOC_ENVIRONMENT") or "").strip() or None,
        "warnings": warnings,
    }


def _configure_cors(flask_app: Flask) -> dict[str, object]:
    state = _cors_health_state()
    if not bool(state.get("enabled")):
        logger.info("CORS is disabled by configuration")
        return state

    supports_credentials = bool(state.get("supports_credentials"))
    allow_all = bool(state.get("allow_all"))
    if supports_credentials and allow_all:
        raise RuntimeError("DEEPDOC_CORS_SUPPORTS_CREDENTIALS=1 cannot be combined with DEEPDOC_CORS_ALLOW_ALL=1")

    cors_options = {
        "methods": state["methods"],
        "allow_headers": state["allowed_headers"],
        "expose_headers": state["exposed_headers"],
        "supports_credentials": supports_credentials,
        "max_age": int(state["max_age_seconds"]),
        "vary_header": True,
    }
    allowed_origin_regex = state.get("allowed_origin_regex")
    if allow_all:
        cors_options["origins"] = "*"
        cors_options["send_wildcard"] = True
    else:
        allowed_origins = list(state.get("allowed_origins") or [])
        if not allowed_origins and not allowed_origin_regex:
            logger.warning("CORS enabled without explicit origins; cross-origin access will be denied")
            cors_options["origins"] = []
        else:
            cors_options["origins"] = allowed_origins
            if allowed_origin_regex:
                cors_options["origins"].append(allowed_origin_regex)

    CORS(flask_app, resources={pattern: cors_options for pattern in DEFAULT_CORS_RESOURCE_PATTERNS})
    return state


RUNTIME_CONFIG_STATE = _validate_runtime_config()
CORS_STATE = _configure_cors(app)


@app.route("/openapi.json", methods=["GET"])
@app.route("/api/v1/openapi.json", methods=["GET"])
@app.route("/docs/openapi.json", methods=["GET"])
def openapi_spec():
    response = jsonify(_build_openapi_payload())
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/v1/build-info", methods=["GET"])
def get_build_info_endpoint():
    payload = get_build_info()
    response = _jsonify_external(payload)
    response.headers["Cache-Control"] = "no-store"
    return response


SWAGGER_URL = "/docs"
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    "openapi.json",
    config={
        "app_name": API_DISPLAY_NAME,
        "displayRequestDuration": True,
        "docExpansion": "list",
        "deepLinking": True,
        "persistAuthorization": True,
        "tryItOutEnabled": True,
    },
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


def _auth_mode() -> str:
    configured = (os.environ.get("DEEPDOC_AUTH_MODE") or "").strip().lower()
    if configured in {"none", "disabled", ""}:
        if api_key:
            return "api_key"
        return "none"
    if configured in {"api_key", "jwt_hs256"}:
        return configured
    raise RuntimeError(f"Unsupported DEEPDOC_AUTH_MODE: {configured}")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _json_bytes_loads(payload: bytes) -> dict[str, object]:
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JWT segment payload must be a JSON object")
    return data


def _extract_scope_set(payload: dict[str, object]) -> set[str]:
    scopes: set[str] = set()
    raw_scope = payload.get("scope")
    if isinstance(raw_scope, str):
        scopes.update(_parse_csv_set(raw_scope.replace(" ", ",")))
    raw_scopes = payload.get("scopes")
    if isinstance(raw_scopes, list):
        scopes.update(str(item).strip() for item in raw_scopes if str(item).strip())
    elif isinstance(raw_scopes, str):
        scopes.update(_parse_csv_set(raw_scopes.replace(" ", ",")))
    return {scope for scope in scopes if scope}


def _extract_requested_tenant_id(*, include_body: bool = True) -> str | None:
    if not has_request_context():
        return None
    header_tenant = (request.headers.get("X-Tenant-ID") or "").strip()
    if header_tenant:
        return header_tenant
    query_tenant = (request.args.get("tenant_id") or "").strip()
    if query_tenant:
        return query_tenant
    if not include_body:
        return None
    form_tenant = (request.form.get("tenant_id") or "").strip()
    if form_tenant:
        return form_tenant
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        body_tenant = str(payload.get("tenant_id") or "").strip()
        if body_tenant:
            return body_tenant
    return None


def _verify_jwt_hs256(token: str) -> dict[str, object]:
    secret = (os.environ.get("DEEPDOC_AUTH_JWT_SECRET") or "").strip()
    if not secret:
        raise ValueError("DEEPDOC_AUTH_JWT_SECRET is required when DEEPDOC_AUTH_MODE=jwt_hs256")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT token format")
    header_segment, payload_segment, signature_segment = parts
    header = _json_bytes_loads(_base64url_decode(header_segment))
    payload = _json_bytes_loads(_base64url_decode(payload_segment))
    algorithm = str(header.get("alg") or "").strip()
    if algorithm != "HS256":
        raise ValueError("Only HS256 JWT tokens are supported")
    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    expected_signature = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    ).decode("utf-8").rstrip("=")
    if not hmac.compare_digest(expected_signature, signature_segment):
        raise ValueError("Invalid JWT signature")
    now_ts = int(datetime.now(timezone.utc).timestamp())
    exp = payload.get("exp")
    if exp is not None and now_ts >= int(exp):
        raise ValueError("JWT token expired")
    nbf = payload.get("nbf")
    if nbf is not None and now_ts < int(nbf):
        raise ValueError("JWT token not active yet")
    issuer = (os.environ.get("DEEPDOC_AUTH_JWT_ISSUER") or "").strip()
    if issuer and str(payload.get("iss") or "").strip() != issuer:
        raise ValueError("Invalid JWT issuer")
    audience = (os.environ.get("DEEPDOC_AUTH_JWT_AUDIENCE") or "").strip()
    if audience:
        raw_aud = payload.get("aud")
        if isinstance(raw_aud, list):
            audiences = {str(item).strip() for item in raw_aud if str(item).strip()}
        else:
            audiences = {str(raw_aud or "").strip()} if raw_aud is not None else set()
        if audience not in audiences:
            raise ValueError("Invalid JWT audience")
    return payload


def _build_auth_context(*, mode: str, subject: str | None, tenant_id: str | None, scopes: set[str], claims: dict[str, object] | None = None) -> dict[str, object]:
    normalized_tenant = (tenant_id or "").strip() or None
    normalized_scopes = sorted(scope for scope in scopes if scope)
    return {
        "mode": mode,
        "subject": subject,
        "tenant_id": normalized_tenant,
        "scopes": normalized_scopes,
        "is_admin": any(scope in AUTH_ADMIN_SCOPES for scope in normalized_scopes),
        "claims": claims or {},
    }


def _default_auth_context() -> dict[str, object]:
    return _build_auth_context(mode=_auth_mode(), subject=None, tenant_id=DEFAULT_TENANT_ID, scopes=set())


def _resolve_auth_context_from_request(*, allow_anonymous: bool) -> dict[str, object]:
    mode = _auth_mode()
    token = _get_auth_token()
    if mode == "none":
        requested_tenant = _extract_requested_tenant_id(include_body=False)
        return _build_auth_context(mode=mode, subject=None, tenant_id=requested_tenant or DEFAULT_TENANT_ID, scopes=set())
    if allow_anonymous and not token:
        requested_tenant = _extract_requested_tenant_id(include_body=False)
        return _build_auth_context(mode=mode, subject=None, tenant_id=requested_tenant or DEFAULT_TENANT_ID, scopes=set())
    if not token:
        raise PermissionError("Missing authentication token")
    if mode == "api_key":
        if not api_key or token != api_key:
            raise PermissionError("Invalid API key")
        requested_tenant = _extract_requested_tenant_id(include_body=False)
        default_scopes = _parse_csv_set(os.environ.get("DEEPDOC_AUTH_DEFAULT_SCOPES"))
        return _build_auth_context(
            mode=mode,
            subject="api_key",
            tenant_id=requested_tenant or DEFAULT_TENANT_ID,
            scopes=default_scopes,
        )
    if mode == "jwt_hs256":
        claims = _verify_jwt_hs256(token)
        scopes = _extract_scope_set(claims)
        tenant_id = str(claims.get("tenant_id") or claims.get("tid") or "").strip() or None
        if not tenant_id:
            tenant_id = DEFAULT_TENANT_ID
        return _build_auth_context(
            mode=mode,
            subject=str(claims.get("sub") or "").strip() or None,
            tenant_id=tenant_id,
            scopes=scopes,
            claims=claims,
        )
    raise PermissionError(f"Unsupported auth mode: {mode}")


def _current_auth_context() -> dict[str, object]:
    if not has_app_context():
        return _default_auth_context()
    context = getattr(g, "auth_context", None)
    if isinstance(context, dict):
        return context
    context = _default_auth_context()
    g.auth_context = context
    return context


def _resolve_effective_tenant_id(*, allow_admin_override: bool = True) -> str | None:
    auth_context = _current_auth_context()
    requested_tenant = _extract_requested_tenant_id()
    current_tenant = str(auth_context.get("tenant_id") or "").strip() or None
    if current_tenant:
        if requested_tenant and requested_tenant != current_tenant:
            if bool(auth_context.get("is_admin")) and allow_admin_override:
                return requested_tenant
            raise PermissionError("tenant_id override is not allowed")
        return current_tenant
    return requested_tenant or DEFAULT_TENANT_ID


def _manifest_tenant_id(manifest: ParseManifest | dict[str, object] | None) -> str | None:
    metadata = manifest.metadata if isinstance(manifest, ParseManifest) else (manifest or {}).get("metadata")
    if isinstance(metadata, dict):
        tenant_id = str(metadata.get("tenant_id") or "").strip()
        if tenant_id:
            return tenant_id
    return None


def _ensure_manifest_access(manifest: ParseManifest) -> None:
    auth_context = _current_auth_context()
    if bool(auth_context.get("is_admin")):
        return
    context_tenant = str(auth_context.get("tenant_id") or "").strip() or None
    manifest_tenant = _manifest_tenant_id(manifest)
    if context_tenant and manifest_tenant and context_tenant != manifest_tenant:
        raise PermissionError("artifact does not belong to current tenant")
    requested_tenant = _extract_requested_tenant_id()
    if context_tenant and requested_tenant and requested_tenant != context_tenant:
        raise PermissionError("tenant_id override is not allowed")


def _require_admin_capability(feature: str) -> None:
    if _auth_mode() == "none":
        return
    auth_context = _current_auth_context()
    if bool(auth_context.get("is_admin")):
        return
    raise PermissionError(f"{feature} requires admin scope")

# Global instances
ocr_engine = None
layout_engine = None
UPLOAD_TMP_DIR = Path(setting.WORK_DIR)
UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_STORE = create_artifact_store()
INGEST_PUBLISHER = create_ingest_publisher()
REQUEST_RATE_LIMITER = create_request_rate_limiter()
INFLIGHT_ADMISSION = create_inflight_admission_controller()
ASYNC_TASK_STORE = create_async_task_store()
ASYNC_TASK_BROKER = create_async_task_broker()
AUDIT_LOG_STORE = create_audit_log_store()
SELF_CHECK_STORE = SelfCheckStore()


def _create_ingest_query_store():
    sink_type = (os.environ.get("DEEPDOC_INGEST_PUBLISHER", "none") or "none").strip().lower()
    if sink_type != "postgres":
        return None
    try:
        return PostgresIngestStore(load_postgres_ingest_config_from_env())
    except Exception:
        logger.exception("Failed to initialize PostgreSQL ingest query store")
        return None


INGEST_QUERY_STORE = _create_ingest_query_store()


def _current_trace_identifiers() -> tuple[str | None, str | None]:
    try:
        from opentelemetry import trace as otel_trace

        span = otel_trace.get_current_span()
        if span is None:
            return None, None
        context = span.get_span_context()
        if context is None or not context.is_valid:
            return None, None
        return format(context.trace_id, "032x"), format(context.span_id, "016x")
    except Exception:
        return None, None


def _append_ops_audit_event(
    action: str,
    *,
    resource_type: str,
    resource_id: str | None = None,
    status: str = "ok",
    payload: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    tenant_id: str | None = None,
    auth_context: dict[str, object] | None = None,
    request_id: str | None = None,
) -> str | None:
    if AUDIT_LOG_STORE is None:
        return None
    try:
        context = auth_context if isinstance(auth_context, dict) else _current_auth_context()
        trace_id, span_id = _current_trace_identifiers()
        event_metadata = dict(metadata or {})
        if has_request_context():
            forwarded_for = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
            event_metadata.setdefault("http_method", request.method)
            event_metadata.setdefault("http_path", request.path)
            event_metadata.setdefault("remote_addr", forwarded_for or str(request.remote_addr or "").strip() or None)
            user_agent = (request.headers.get("User-Agent") or "").strip()
            if user_agent:
                event_metadata.setdefault("user_agent", user_agent)
        event = AuditEvent(
            tenant_id=tenant_id or str(context.get("tenant_id") or "").strip() or None,
            actor_subject=str(context.get("subject") or "").strip() or None,
            actor_mode=str(context.get("mode") or "").strip() or None,
            actor_is_admin=bool(context.get("is_admin")),
            actor_scopes=[str(item).strip() for item in (context.get("scopes") or []) if str(item).strip()],
            request_id=request_id or ((request.headers.get("X-Request-ID") or "").strip() if has_request_context() else None) or None,
            trace_id=trace_id,
            span_id=span_id,
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id or "").strip() or None,
            status=str(status or "ok").strip() or "ok",
            payload=dict(payload or {}),
            metadata=event_metadata,
        )
        AUDIT_LOG_STORE.append_event(event)
        return event.event_id
    except Exception:
        logger.exception("Failed to append ops audit event action=%s resource_type=%s", action, resource_type)
        return None


def _get_auth_token() -> str | None:
    header_value = request.headers.get(auth_header, "")
    if header_value:
        if header_value.startswith("Bearer "):
            return header_value.removeprefix("Bearer ").strip()
        return header_value.strip()
    return request.headers.get("X-API-Key", "").strip() or None


def _internal_auth_headers() -> dict[str, str]:
    if not api_key:
        return {}
    header_name = str(auth_header or "Authorization").strip() or "Authorization"
    header_value = f"Bearer {api_key}" if header_name.lower() == "authorization" else api_key
    return {header_name: header_value}


@app.before_request
def require_api_key():
    try:
        allow_anonymous = _is_preflight_request() or _is_auth_exempt_path(request.path)
        g.auth_context = _resolve_auth_context_from_request(allow_anonymous=allow_anonymous)
    except PermissionError:
        return _jsonify_error(ErrorCode.UNAUTHORIZED, status_code=401, message="unauthorized")
    except Exception as exc:
        logger.exception("Failed to resolve request auth context")
        return _jsonify_error(ErrorCode.INTERNAL_ERROR, status_code=500, message=str(exc))
    auth_context = g.auth_context if isinstance(g.auth_context, dict) else {}
    set_current_span_attributes(
        {
            "deepdoc.auth_mode": auth_context.get("mode"),
            "deepdoc.auth_subject": auth_context.get("subject"),
            "deepdoc.tenant_id": auth_context.get("tenant_id"),
            "deepdoc.is_admin": bool(auth_context.get("is_admin")),
            "http.request_id": request.headers.get("X-Request-ID"),
        }
    )
    return None


@app.before_request
def start_request_metrics():
    g._metrics_started_at = time.perf_counter()
    g._metrics_finalized = False
    g._metrics_inflight = True
    inc_http_inflight()


def _request_guard_scope() -> str | None:
    path = (request.path or "").strip()
    if not path or _is_preflight_request() or _is_auth_exempt_path(path):
        return None
    if path.startswith("/api/v1/parse"):
        return "parse"
    if path.startswith("/api/v1/artifacts"):
        return "artifact"
    if path.startswith("/api/v1/ingest"):
        return "ingest"
    return "general"


def _request_guard_identity() -> str:
    auth_context = _current_auth_context()
    subject = str(auth_context.get("subject") or "").strip()
    tenant_id = str(auth_context.get("tenant_id") or "").strip()
    forwarded_for = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    remote_addr = forwarded_for or str(request.remote_addr or "").strip() or "unknown"
    if tenant_id:
        return f"tenant:{tenant_id}"
    if subject:
        return f"subject:{subject}"
    return f"ip:{remote_addr}"


def _apply_request_guard_headers(response: Response) -> Response:
    headers = getattr(g, "_request_guard_headers", None)
    if isinstance(headers, dict):
        for key, value in headers.items():
            response.headers[key] = value
    return response


def _release_admission_lease() -> None:
    if bool(getattr(g, "_admission_released", False)):
        return
    lease = getattr(g, "_admission_lease", None)
    if lease is None:
        g._admission_released = True
        return
    try:
        INFLIGHT_ADMISSION.release(lease)
        dec_admission_inflight(getattr(lease, "pool", "unknown"))
    finally:
        g._admission_lease = None
        g._admission_released = True


@app.before_request
def enforce_request_protection():
    scope = _request_guard_scope()
    if scope is None:
        return None
    auth_context = _current_auth_context()
    is_admin = bool(auth_context.get("is_admin"))
    identity = _request_guard_identity()
    request_bytes = max(0, int(request.content_length or 0))
    try:
        rate_limit = REQUEST_RATE_LIMITER.evaluate(
            scope=scope,
            identity=identity,
            request_bytes=request_bytes,
            is_admin=is_admin,
        )
    except Exception:
        logger.exception("Request guard rate-limit failure scope=%s", scope)
        observe_request_guard_decision(
            component="rate_limit",
            scope=scope,
            status="error",
            backend=REQUEST_RATE_LIMITER.backend_name,
        )
        if REQUEST_RATE_LIMITER.fail_open:
            rate_limit = None
        else:
            response = jsonify({"error": "request protection backend unavailable", "scope": scope})
            response.status_code = 503
            return _apply_request_guard_headers(response)
    if rate_limit is not None:
        g._request_guard_headers = rate_limit.headers
        decision_status = "allowed" if rate_limit.allowed else "rejected"
        observe_request_guard_decision(
            component="rate_limit",
            scope=scope,
            status=decision_status,
            backend=REQUEST_RATE_LIMITER.backend_name,
        )
        if not rate_limit.allowed and rate_limit.denied_decision is not None:
            response = jsonify(rate_limit.error_payload())
            response.status_code = rate_limit.denied_decision.rule.status_code
            return _apply_request_guard_headers(response)
    if scope not in {"parse", "artifact", "ingest"}:
        return None
    admission = INFLIGHT_ADMISSION.acquire(scope)
    observe_request_guard_decision(
        component="admission",
        scope=scope,
        status="allowed" if admission.allowed else "rejected",
        backend="local",
    )
    if admission.limit > 0 or not admission.allowed:
        g._request_guard_headers = {**getattr(g, "_request_guard_headers", {}), **admission.headers()}
    if not admission.allowed:
        response = jsonify(admission.error_payload())
        response.status_code = 503
        return _apply_request_guard_headers(response)
    g._admission_lease = admission.lease
    g._admission_released = False
    if admission.lease is not None:
        inc_admission_inflight(scope)
    return None


def _route_label() -> str:
    if request.url_rule is not None and getattr(request.url_rule, "rule", None):
        return str(request.url_rule.rule)
    if request.endpoint:
        return str(request.endpoint)
    return "unmatched"


def _finalize_request_metrics(*, status_code: int, exception: Exception | None = None) -> None:
    if bool(getattr(g, "_metrics_finalized", False)):
        return
    method = (request.method or "UNKNOWN").upper()
    route = _route_label()
    started_at = getattr(g, "_metrics_started_at", None)
    duration_seconds = 0.0
    if started_at is not None:
        duration_seconds = max(0.0, time.perf_counter() - float(started_at))
    observe_http_request(method, route, status_code, duration_seconds)
    if exception is not None:
        observe_http_exception(method, route, exception.__class__.__name__)
    if bool(getattr(g, "_metrics_inflight", False)):
        dec_http_inflight()
        g._metrics_inflight = False
    g._metrics_finalized = True


@app.after_request
def record_request_metrics(response: Response):
    response = _apply_unified_error_response(response)
    _finalize_request_metrics(status_code=response.status_code)
    response = _apply_request_guard_headers(response)
    _release_admission_lease()
    return response


@app.teardown_request
def teardown_request_metrics(exception: Exception | None):
    if exception is None:
        return
    _finalize_request_metrics(status_code=500, exception=exception)
    _release_admission_lease()


def _artifact_backend_health() -> dict[str, object]:
    backend = ARTIFACT_STORE.__class__.__name__.removesuffix("ArtifactStore").lower() or "local"
    try:
        ARTIFACT_STORE.list_manifests(limit=1)
        return {"status": "ok", "backend": backend}
    except Exception as exc:
        logger.exception("Artifact backend health check failed")
        return {"status": "error", "backend": backend, "error": str(exc)}


def _request_protection_health() -> dict[str, object]:
    rate_limit: dict[str, object]
    try:
        rate_limit = REQUEST_RATE_LIMITER.check_health()
    except Exception as exc:
        logger.exception("Request rate limiter health check failed")
        rate_limit = {
            "status": "error",
            "backend": REQUEST_RATE_LIMITER.backend_name,
            "enabled": REQUEST_RATE_LIMITER.enabled,
            "fail_open": REQUEST_RATE_LIMITER.fail_open,
            "error": str(exc),
        }
    admission = INFLIGHT_ADMISSION.snapshot()
    return {"rate_limit": rate_limit, "admission": admission}


def _readiness_model_groups() -> list[str]:
    configured = (os.environ.get("DEEPDOC_READINESS_MODEL_GROUPS") or "").strip()
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    return get_download_groups_from_env(default="published")


def _read_worker_heartbeat_state(
    *,
    enabled: bool,
    heartbeat_file_env: str,
    default_path: str,
    max_age_env: str,
    default_max_age_seconds: int,
    missing_error: str,
    stale_error: str,
    include_history: bool = False,
) -> dict[str, object]:
    heartbeat_file = Path(os.environ.get(heartbeat_file_env, default_path))
    if not enabled:
        return {"status": "disabled"}
    if not heartbeat_file.exists():
        return {"status": "error", "error": missing_error, "path": str(heartbeat_file)}
    try:
        payload = json.loads(heartbeat_file.read_text(encoding="utf-8"))
        updated_at = datetime.fromisoformat(str(payload.get("updated_at") or ""))
        max_age = max(10, int(os.environ.get(max_age_env, str(default_max_age_seconds))))
        age_seconds = max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds())
        state = {
            "status": "ok" if age_seconds <= max_age else "error",
            "path": str(heartbeat_file),
            "state": str(payload.get("state") or "unknown"),
            "updated_at": str(payload.get("updated_at") or ""),
            "age_seconds": age_seconds,
            "max_age_seconds": max_age,
        }
        if payload.get("task_id") is not None:
            state["task_id"] = payload.get("task_id")
        if payload.get("run") is not None and isinstance(payload.get("run"), dict):
            state["run"] = payload.get("run")
        if include_history and payload.get("history") is not None and isinstance(payload.get("history"), list):
            state["history"] = [item for item in payload.get("history") if isinstance(item, dict)]
        if age_seconds > max_age:
            state["error"] = stale_error
        return state
    except Exception as exc:
        logger.exception("Worker heartbeat health check failed path=%s", heartbeat_file)
        return {"status": "error", "path": str(heartbeat_file), "error": str(exc)}


def _callback_redrive_enabled() -> bool:
    broker_enabled = getattr(ASYNC_TASK_BROKER, "backend_name", "none") != "none"
    return _parse_bool(os.environ.get("DEEPDOC_ASYNC_CALLBACK_REDRIVE_ENABLED"), default=broker_enabled)


def _self_check_worker_enabled() -> bool:
    return _parse_bool(os.environ.get("DEEPDOC_SELF_CHECK_AUTO_ENABLED"), default=False)


def _self_check_history_limit() -> int:
    try:
        return max(1, int((os.environ.get("DEEPDOC_SELF_CHECK_HISTORY_LIMIT") or "64").strip()))
    except Exception:
        return 64


def _self_check_required_for_ready() -> bool:
    return _parse_bool(os.environ.get("DEEPDOC_SELF_CHECK_REQUIRED_FOR_READY"), default=False)


def _self_check_last_run_max_age_seconds() -> int:
    try:
        return max(60, int(os.environ.get("DEEPDOC_SELF_CHECK_LAST_RUN_MAX_AGE_SECONDS", "86400")))
    except (TypeError, ValueError):
        return 86400


def _self_check_worker_health(*, include_history: bool = False) -> dict[str, object]:
    return _read_worker_heartbeat_state(
        enabled=_self_check_worker_enabled(),
        heartbeat_file_env="DEEPDOC_SELF_CHECK_HEARTBEAT_FILE",
        default_path=os.path.join(setting.TASKS_DIR, "self-check-heartbeat.json"),
        max_age_env="DEEPDOC_SELF_CHECK_HEALTH_MAX_AGE_SECONDS",
        default_max_age_seconds=120,
        missing_error="self-check worker heartbeat file missing",
        stale_error="self-check worker heartbeat stale",
        include_history=include_history,
    )


def _self_checks_health(*, include_history: bool = False) -> dict[str, object]:
    store_state = SELF_CHECK_STORE.check_health()
    worker_state = _self_check_worker_health(include_history=include_history)
    latest_run = SELF_CHECK_STORE.latest_run()
    state: dict[str, object] = {
        "status": "ok",
        "auto_enabled": _self_check_worker_enabled(),
        "required_for_ready": _self_check_required_for_ready(),
        "last_run_max_age_seconds": _self_check_last_run_max_age_seconds(),
        "store": store_state,
        "worker": worker_state,
        "latest_run": summarize_self_check_run(latest_run),
    }
    if include_history:
        history: list[dict[str, object]] = []
        for run in SELF_CHECK_STORE.list_runs(limit=_self_check_history_limit()):
            if str(run.status or "").strip().lower() not in {"passed", "failed"}:
                continue
            summary = summarize_self_check_run(run)
            if summary is None:
                continue
            history.append(
                {
                    **summary,
                    "requested_by": str(run.metadata.get("requested_by") or "").strip() or "manual",
                    "auto_run": bool(run.metadata.get("auto_run")),
                }
            )
        state["history"] = history
    if str(store_state.get("status") or "") != "ok":
        state["status"] = "error"
        state["error"] = "self-check store unavailable"
        return state
    if _self_check_worker_enabled() and str(worker_state.get("status") or "") != "ok":
        state["status"] = "error"
        state["error"] = "self-check worker unhealthy"
        return state
    if latest_run is None:
        if _self_check_required_for_ready():
            state["status"] = "error"
            state["error"] = "no production self-check result is available"
        return state
    finished_at = str(latest_run.finished_at or latest_run.created_at or "").strip()
    if finished_at:
        try:
            age_seconds = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(finished_at)).total_seconds())
            state["last_run_age_seconds"] = age_seconds
            if _self_check_required_for_ready() and age_seconds > _self_check_last_run_max_age_seconds():
                state["status"] = "error"
                state["error"] = "latest production self-check is stale"
                return state
        except Exception:
            if _self_check_required_for_ready():
                state["status"] = "error"
                state["error"] = "latest production self-check missing completion timestamp"
                return state
    if str(latest_run.status or "").strip().lower() == "failed":
        state["status"] = "error"
        state["error"] = "latest production self-check failed"
        return state
    return state


def _retention_janitor_config():
    return load_retention_janitor_config()


def _retention_janitor_enabled() -> bool:
    return bool(_retention_janitor_config().enabled)


def _retention_janitor_required_for_ready() -> bool:
    return bool(_retention_janitor_config().required_for_ready)


def _retention_janitor_worker_health(*, include_history: bool = False) -> dict[str, object]:
    config = _retention_janitor_config()
    return _read_worker_heartbeat_state(
        enabled=bool(config.enabled),
        heartbeat_file_env="DEEPDOC_RETENTION_JANITOR_HEARTBEAT_FILE",
        default_path=config.heartbeat_file,
        max_age_env="DEEPDOC_RETENTION_JANITOR_HEALTH_MAX_AGE_SECONDS",
        default_max_age_seconds=int(config.health_max_age_seconds),
        missing_error="retention janitor heartbeat file missing",
        stale_error="retention janitor heartbeat stale",
        include_history=include_history,
    )


def _retention_janitor_health(*, include_history: bool = False) -> dict[str, object]:
    config = _retention_janitor_config()
    worker_state = _retention_janitor_worker_health(include_history=include_history)
    history = worker_state.get("history") if include_history and isinstance(worker_state.get("history"), list) else None
    latest_run = worker_state.get("run") if isinstance(worker_state.get("run"), dict) else None
    state: dict[str, object] = {
        "status": "disabled" if not config.enabled else "ok",
        "enabled": bool(config.enabled),
        "required_for_ready": bool(config.required_for_ready),
        "last_run_max_age_seconds": int(config.last_run_max_age_seconds),
        "worker": worker_state,
        "rules": config.summarized_rules(),
        "latest_run": latest_run,
    }
    if include_history and history:
        state["history"] = history
    if not config.enabled:
        return state
    if str(worker_state.get("status") or "") != "ok":
        state["status"] = "error"
        state["error"] = "retention janitor worker unhealthy"
        return state
    if latest_run is None:
        if config.required_for_ready:
            state["status"] = "error"
            state["error"] = "no retention janitor run is available"
        return state
    finished_at = str(latest_run.get("finished_at") or latest_run.get("started_at") or "").strip()
    if finished_at:
        try:
            age_seconds = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(finished_at)).total_seconds())
            state["last_run_age_seconds"] = age_seconds
            if config.required_for_ready and age_seconds > int(config.last_run_max_age_seconds):
                state["status"] = "error"
                state["error"] = "latest retention janitor run is stale"
                return state
        except Exception:
            if config.required_for_ready:
                state["status"] = "error"
                state["error"] = "latest retention janitor run missing completion timestamp"
                return state
    if str(latest_run.get("status") or "").strip().lower() == "error":
        state["status"] = "error"
        state["error"] = "latest retention janitor run failed"
        return state
    return state


def _async_tasks_health(*, include_history: bool = False) -> dict[str, object]:
    store_state: dict[str, object]
    broker_state: dict[str, object]
    worker_state: dict[str, object]
    callback_redrive_state: dict[str, object]
    try:
        store_state = ASYNC_TASK_STORE.check_health()
    except Exception as exc:
        logger.exception("Async task store health check failed")
        store_state = {"status": "error", "error": str(exc)}
    try:
        broker_state = ASYNC_TASK_BROKER.check_health()
    except Exception as exc:
        logger.exception("Async task broker health check failed")
        broker_state = {"status": "error", "backend": getattr(ASYNC_TASK_BROKER, "backend_name", "unknown"), "error": str(exc)}
    worker_enabled = getattr(ASYNC_TASK_BROKER, "backend_name", "none") != "none"
    worker_state = _read_worker_heartbeat_state(
        enabled=worker_enabled,
        heartbeat_file_env="DEEPDOC_ASYNC_WORKER_HEARTBEAT_FILE",
        default_path=os.path.join(setting.TASKS_DIR, "worker-heartbeat.json"),
        max_age_env="DEEPDOC_ASYNC_WORKER_HEALTH_MAX_AGE_SECONDS",
        default_max_age_seconds=60,
        missing_error="worker heartbeat file missing",
        stale_error="worker heartbeat stale",
        include_history=include_history,
    )
    callback_redrive_state = _read_worker_heartbeat_state(
        enabled=_callback_redrive_enabled(),
        heartbeat_file_env="DEEPDOC_ASYNC_CALLBACK_REDRIVE_HEARTBEAT_FILE",
        default_path=os.path.join(setting.TASKS_DIR, "callback-redrive-heartbeat.json"),
        max_age_env="DEEPDOC_ASYNC_CALLBACK_REDRIVE_HEALTH_MAX_AGE_SECONDS",
        default_max_age_seconds=120,
        missing_error="callback redrive heartbeat file missing",
        stale_error="callback redrive heartbeat stale",
        include_history=include_history,
    )
    return {
        "enabled": worker_enabled,
        "store": store_state,
        "broker": broker_state,
        "worker": worker_state,
        "callback_redrive_worker": callback_redrive_state,
    }




def _load_accessible_async_task(task_id: str) -> AsyncTask:
    task = ASYNC_TASK_STORE.load_task(task_id)
    auth_context = _current_auth_context()
    if bool(auth_context.get("is_admin")):
        return task
    current_tenant = str(auth_context.get("tenant_id") or "").strip() or None
    task_tenant = str(task.tenant_id or "").strip() or None
    if current_tenant and task_tenant and current_tenant != task_tenant:
        raise PermissionError("task does not belong to current tenant")
    requested_tenant = _extract_requested_tenant_id()
    if current_tenant and requested_tenant and requested_tenant != current_tenant:
        raise PermissionError("tenant_id override is not allowed")
    return task


PARSER_IMPORTS = {
    "pdf": ("deepdoc.parser.pdf_parser", "DeepDocPdfParser"),
    "docx": ("deepdoc.parser.docx_parser", "DeepDocDocxParser"),
    "xlsx": ("deepdoc.parser.excel_parser", "DeepDocExcelParser"),
    "excel": ("deepdoc.parser.excel_parser", "DeepDocExcelParser"),
    "xls": ("deepdoc.parser.markitdown_parser", "MarkItDownParser"),
    "pptx": ("deepdoc.parser.ppt_parser", "DeepDocPptParser"),
    "ppt": ("deepdoc.parser.ppt_parser", "DeepDocPptParser"),
    "html": ("deepdoc.parser.html_parser", "DeepDocHtmlParser"),
    "json": ("deepdoc.parser.json_parser", "DeepDocJsonParser"),
    "md": ("deepdoc.parser.markdown_parser", "DeepDocMarkdownParser"),
    "markdown": ("deepdoc.parser.markdown_parser", "DeepDocMarkdownParser"),
    "txt": ("deepdoc.parser.txt_parser", "DeepDocTxtParser"),
    "csv": ("deepdoc.parser.csv_parser", "DeepDocCsvParser"),
    "tsv": ("deepdoc.parser.csv_parser", "DeepDocCsvParser"),
    "rtf": ("deepdoc.parser.rtf_parser", "DeepDocRtfParser"),
    "odt": ("deepdoc.parser.odt_parser", "DeepDocOdtParser"),
    "eml": ("deepdoc.parser.email_parser", "DeepDocEmailParser"),
    "msg": ("deepdoc.parser.email_parser", "DeepDocEmailParser"),
    "caj": ("deepdoc.parser.caj_parser", "DeepDocCajParser"),
    "xml": ("deepdoc.parser.markitdown_parser", "MarkItDownParser"),
    "zip": ("deepdoc.parser.markitdown_parser", "MarkItDownParser"),
    "epub": ("deepdoc.parser.epub_parser", "DeepDocEpubParser"),
    "mineru": ("deepdoc.parser.mineru_parser", "MinerUParser"),
    "docling": ("deepdoc.parser.docling_parser", "DoclingParser"),
    "tcadp": ("deepdoc.parser.tcadp_parser", "TCADPParser"),
}
IMAGE_FILE_TYPES = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

PDF_PARSER_OVERRIDES = {
    "plain": ("deepdoc.parser.pdf_parser", "PlainParser"),
    "paddleocr_vl": ("deepdoc.parser.paddleocr_parser", "PaddleOCRParser"),
    "mineru": ("deepdoc.parser.mineru_parser", "MinerUParser"),
}

PADDLEOCR_VL_ENGINE = "paddleocr_vl"
MARKITDOWN_ENGINE = "markitdown"
MARKITDOWN_IMPORT_SPEC = ("deepdoc.parser.markitdown_parser", "MarkItDownParser")
MARKITDOWN_FILE_TYPES = {
    "docx",
    "xlsx",
    "xls",
    "pptx",
    "ppt",
    "html",
    "json",
    "md",
    "markdown",
    "txt",
    "csv",
    "tsv",
    "xml",
    "zip",
    "epub",
}
PARSER_ENGINE_ALIASES = {
    "docpilot": "deepdoc",
    "doc-pilot": "deepdoc",
    "paddleocr": PADDLEOCR_VL_ENGINE,
    "paddleocr-vl": PADDLEOCR_VL_ENGINE,
    "paddleocrvl": PADDLEOCR_VL_ENGINE,
}
SUPPORTED_PDF_ENGINES = {"deepdoc", PADDLEOCR_VL_ENGINE, "mineru", "plain"}
SUPPORTED_PARSER_ENGINES = SUPPORTED_PDF_ENGINES | {MARKITDOWN_ENGINE}
SUPPORTED_COMPUTE_DEVICE = {"gpu", "cpu"}
SUPPORTED_EXECUTION_PROFILES = {"auto", "cpu", "gpu"}
SUPPORTED_DEEPDOC_PDF_MODES = {"auto", "native", "ocr", "hybrid"}


def _normalize_parser_engine(value: str | None) -> str:
    mode = (value or "deepdoc").strip().lower()
    mode = PARSER_ENGINE_ALIASES.get(mode, mode)
    if mode not in SUPPORTED_PARSER_ENGINES:
        return "deepdoc"
    return mode


def _normalize_compute_device(value: str | None) -> str:
    device = (value or "gpu").strip().lower()
    if device not in SUPPORTED_COMPUTE_DEVICE:
        return "gpu"
    return device


def _normalize_execution_profile(value: str | None) -> str:
    profile = (value or "auto").strip().lower()
    if profile not in SUPPORTED_EXECUTION_PROFILES:
        return "auto"
    return profile


def _normalize_deepdoc_pdf_mode(value: str | None) -> str:
    mode = (value or "auto").strip().lower()
    aliases = {
        "native_text": "native",
        "text": "native",
        "plain_text": "native",
        "layout": "ocr",
        "vision": "ocr",
    }
    mode = aliases.get(mode, mode)
    if mode not in SUPPORTED_DEEPDOC_PDF_MODES:
        return "auto"
    return mode


def _is_valid_http_url(url: str) -> bool:
    value = (url or "").strip()
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _cleanup_enabled() -> bool:
    raw_value = os.environ.get("DEEPDOC_CLEANUP_OUTPUT")
    if raw_value is None:
        raw_value = os.environ.get("MINERU_DELETE_OUTPUT", "1")
    return _parse_bool(raw_value, default=True)


def _get_request_timeout() -> int:
    return max(1, int(os.environ.get("DEEPDOC_REQUEST_TIMEOUT", "600")))


def _build_parse_options() -> dict[str, object]:
    parser_engine = _normalize_parser_engine(request.form.get("parser_engine"))
    compute_device = _normalize_compute_device(request.form.get("compute_device"))
    execution_profile = _normalize_execution_profile(
        request.form.get("execution_profile") or os.environ.get("DEEPDOC_EXECUTION_PROFILE", "auto")
    )
    return_images = _parse_bool(request.form.get("return_images"), default=False)
    strict_text = _parse_bool(request.form.get("strict_text"), default=False)
    enable_formula = _parse_bool(request.form.get("enable_formula"), default=False)
    enable_seal = _parse_bool(request.form.get("enable_seal"), default=False)
    return_structured = _parse_bool(request.form.get("return_structured"), default=False)
    persist_artifacts = _parse_bool(
        request.form.get("persist_artifacts"),
        default=return_structured,
    )
    publish_ingest = _parse_bool(
        request.form.get("publish_ingest"),
        default=_parse_bool(os.environ.get("DEEPDOC_AUTO_PUBLISH_INGEST"), default=False),
    )
    reuse_artifacts = _parse_bool(
        request.form.get("reuse_artifacts"),
        default=_parse_bool(os.environ.get("DEEPDOC_REUSE_ARTIFACTS"), default=False),
    )
    if publish_ingest and not persist_artifacts:
        logger.info("publish_ingest=true forces persist_artifacts=true for stable asset URLs")
        persist_artifacts = True
    persist_source = _parse_bool(
        request.form.get("persist_source"),
        default=persist_artifacts,
    )
    include_chunks = _parse_bool(
        request.form.get("include_chunks"),
        default=(return_structured or persist_artifacts),
    )
    chunk_max_tokens = max(
        64,
        int(request.form.get("chunk_max_tokens") or os.environ.get("DEEPDOC_CHUNK_MAX_TOKENS", DEFAULT_CHUNK_MAX_TOKENS)),
    )
    chunk_overlap_tokens = max(
        0,
        int(
            request.form.get("chunk_overlap_tokens")
            or os.environ.get("DEEPDOC_CHUNK_OVERLAP_TOKENS", DEFAULT_CHUNK_OVERLAP_TOKENS)
        ),
    )
    chunk_strategy = normalize_chunk_strategy(
        request.form.get("chunk_strategy") or os.environ.get("DEEPDOC_CHUNK_STRATEGY", DEFAULT_CHUNK_STRATEGY)
    )
    deepdoc_layout_model = (
        request.form.get("deepdoc_layout_model")
        or request.form.get("layout_model")
        or os.environ.get("DEEPDOC_LAYOUT_MODEL", "manual")
        or "manual"
    ).strip().lower()
    if deepdoc_layout_model not in {"manual", "paper", "laws", "general"}:
        deepdoc_layout_model = "manual"
    deepdoc_max_pages = max(
        1,
        int(request.form.get("deepdoc_max_pages") or os.environ.get("DEEPDOC_PDF_MAX_PAGES", "10")),
    )
    deepdoc_pdf_mode = _normalize_deepdoc_pdf_mode(
        request.form.get("deepdoc_pdf_mode") or os.environ.get("DEEPDOC_PDF_MODE", "auto")
    )
    options: dict[str, object] = {
        "parser_engine": parser_engine,
        "compute_device": compute_device,
        "execution_profile": execution_profile,
        "return_images": return_images,
        "strict_text": strict_text,
        "enable_formula": enable_formula,
        "enable_seal": enable_seal,
        "return_structured": return_structured,
        "persist_artifacts": persist_artifacts,
        "publish_ingest": publish_ingest,
        "reuse_artifacts": reuse_artifacts,
        "persist_source": persist_source,
        "include_chunks": include_chunks,
        "chunk_max_tokens": chunk_max_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "chunk_strategy": chunk_strategy,
        "deepdoc_layout_model": deepdoc_layout_model,
        "deepdoc_max_pages": deepdoc_max_pages,
        "deepdoc_pdf_mode": deepdoc_pdf_mode,
        "error_locale": _request_error_locale(),
    }
    if enable_formula and parser_engine != "deepdoc":
        logger.info(
            "enable_formula=true ignored: only supported with parser_engine=deepdoc (got %s)",
            parser_engine,
        )
    if enable_seal and parser_engine != "deepdoc":
        logger.info(
            "enable_seal=true ignored: only supported with parser_engine=deepdoc (got %s)",
            parser_engine,
        )
    if parser_engine == PADDLEOCR_VL_ENGINE:
        source_key = (
            "PADDLEOCR_GPU_API_URL" if compute_device == "gpu" else "PADDLEOCR_API_URL"
        )
        paddle_api = os.environ.get(source_key)
        if not _is_valid_http_url(paddle_api):
            raise ValueError(
                f"[PaddleOCR] missing or invalid API URL: {source_key}, expected http(s)://host:port"
            )
        options["paddle_api_url"] = paddle_api
        paddle_formula_enable = _parse_bool(
            request.form.get("paddle_use_formula_recognition"),
            default=_parse_bool(os.environ.get("PADDLEOCR_USE_FORMULA_RECOGNITION"), default=False),
        )
        paddle_seal_enable = _parse_bool(
            request.form.get("paddle_seal_enable"),
            default=_parse_bool(os.environ.get("PADDLEOCR_USE_SEAL_RECOGNITION"), default=False),
        )
        paddle_algorithm_config: dict[str, object] = {
            "use_formula_recognition": paddle_formula_enable,
            "use_seal_recognition": paddle_seal_enable,
            "merge_tables": _parse_bool(request.form.get("paddle_table_enable"), default=True),
        }
        options["paddle_prettify_markdown"] = _parse_bool(
            request.form.get("paddle_prettify_markdown"),
            default=True,
        )
        options["paddle_show_formula_number"] = _parse_bool(
            request.form.get("paddle_show_formula_number"),
            default=False,
        )
        if not return_images:
            paddle_algorithm_config.update(
                {
                    "use_ocr_for_image_block": False,
                    "markdown_ignore_labels": ["image", "figure"],
                }
            )
        options["paddle_algorithm_config"] = paddle_algorithm_config
    elif parser_engine == "mineru":
        source_key = (
            "MINERU_GPU_API_URL" if compute_device == "gpu" else "MINERU_APISERVER"
        )
        mineru_api = (os.environ.get(source_key) or "").strip()
        if not _is_valid_http_url(mineru_api):
            raise ValueError(
                f"[MinerU] missing or invalid API URL: {source_key}, expected http(s)://host:port"
            )

        default_backend = (
            "hybrid-auto-engine" if compute_device == "gpu" else "pipeline"
        )
        options["mineru_api"] = mineru_api
        options["mineru_server_url"] = (
            os.environ.get("MINERU_SERVER_URL")
            or os.environ.get("MINERU_VL_SERVER")
            or ""
        ).strip()
        options["mineru_backend"] = (
            os.environ.get("MINERU_BACKEND", default_backend) or default_backend
        ).strip()
        mineru_max_pages = max(
            1,
            int(request.form.get("mineru_max_pages") or os.environ.get("DEEPDOC_PDF_MAX_PAGES", "10")),
        )
        options["mineru_lang"] = (
            request.form.get("mineru_language") or os.environ.get("MINERU_LANG", "ch") or "ch"
        ).strip() or "ch"
        options["mineru_parse_method"] = "ocr" if _parse_bool(request.form.get("mineru_is_ocr"), default=False) else "auto"
        options["mineru_formula_enable"] = _parse_bool(
            request.form.get("mineru_formula_enable"),
            default=True,
        )
        options["mineru_table_enable"] = _parse_bool(
            request.form.get("mineru_table_enable"),
            default=True,
        )
        options["mineru_start_page_id"] = 0
        options["mineru_end_page_id"] = max(0, mineru_max_pages - 1)

    return options


def _augment_parse_options_with_request_context(parse_options: dict[str, object]) -> dict[str, object]:
    enriched = dict(parse_options)
    auth_context = _current_auth_context()
    tenant_id = _resolve_effective_tenant_id()
    enriched["tenant_id"] = tenant_id
    enriched["auth_subject"] = str(auth_context.get("subject") or "").strip() or None
    enriched["auth_scopes"] = list(auth_context.get("scopes") or [])
    enriched["auth_mode"] = str(auth_context.get("mode") or "none")
    return enriched


def _build_async_task_callback_config() -> AsyncTaskCallbackConfig | None:
    callback_url = (
        request.form.get("callback_url")
        or request.form.get("webhook_url")
        or ""
    ).strip()
    if not callback_url:
        return None
    if not _is_valid_http_url(callback_url):
        raise ValueError("callback_url must be a valid http(s) URL")

    raw_event_types = (
        request.form.get("callback_events")
        or request.form.get("webhook_events")
        or "terminal"
    )
    event_types = sorted(
        {
            str(item).strip().lower()
            for item in str(raw_event_types).replace(" ", ",").split(",")
            if str(item).strip()
        }
    )
    if not event_types:
        event_types = ["terminal"]
    allowed_event_types = {"terminal", "task.succeeded", "task.failed", "task.cancelled"}
    invalid_event_types = sorted(event_type for event_type in event_types if event_type not in allowed_event_types)
    if invalid_event_types:
        raise ValueError(
            f"callback_events contains unsupported values: {', '.join(invalid_event_types)}"
        )

    callback_timeout_seconds = max(
        1,
        min(
            int(
                request.form.get("callback_timeout_seconds")
                or os.environ.get("DEEPDOC_ASYNC_CALLBACK_TIMEOUT_SECONDS", "10")
            ),
            120,
        ),
    )
    callback_max_attempts = max(
        1,
        min(
            int(
                request.form.get("callback_max_attempts")
                or os.environ.get("DEEPDOC_ASYNC_CALLBACK_MAX_ATTEMPTS", "3")
            ),
            10,
        ),
    )
    callback_backoff_seconds = max(
        0.0,
        min(
            float(
                request.form.get("callback_backoff_seconds")
                or os.environ.get("DEEPDOC_ASYNC_CALLBACK_BACKOFF_SECONDS", "1")
            ),
            300.0,
        ),
    )
    callback_max_backoff_seconds = max(
        callback_backoff_seconds,
        min(
            float(
                request.form.get("callback_max_backoff_seconds")
                or os.environ.get("DEEPDOC_ASYNC_CALLBACK_MAX_BACKOFF_SECONDS", "10")
            ),
            600.0,
        ),
    )
    callback_secret = (request.form.get("callback_secret") or "").strip() or None
    callback_include_result = _parse_bool(request.form.get("callback_include_result"), default=True)

    return AsyncTaskCallbackConfig(
        url=callback_url,
        event_types=event_types,
        include_result=callback_include_result,
        timeout_seconds=callback_timeout_seconds,
        max_attempts=callback_max_attempts,
        backoff_seconds=callback_backoff_seconds,
        max_backoff_seconds=callback_max_backoff_seconds,
        secret=callback_secret,
    )


def _parser_import_spec(
    file_type: str, parser_engine: str | None = None
) -> tuple[str, str] | None:
    key = file_type.lower()
    parser_mode = _normalize_parser_engine(parser_engine)
    if key == "pdf":
        if parser_mode in PDF_PARSER_OVERRIDES:
            return PDF_PARSER_OVERRIDES[parser_mode]
    if parser_mode == MARKITDOWN_ENGINE and key in MARKITDOWN_FILE_TYPES:
        return MARKITDOWN_IMPORT_SPEC
    return PARSER_IMPORTS.get(key)


def _load_parser(file_type: str, parser_engine: str | None = None):
    spec = _parser_import_spec(file_type, parser_engine=parser_engine)
    if not spec:
        return None
    module_name, attr_name = spec
    module = import_module(module_name)
    return getattr(module, attr_name)


def load_models():
    global ocr_engine, layout_engine
    logger.info(
        f"Loading OCR and Layout models from {os.environ['DEEPDOC_MODEL_PATH']}..."
    )
    try:
        from deepdoc.vision import OCR

        ocr_engine = OCR()
        logger.info("OCR models loaded successfully.")
    except Exception as e:
        logger.error(f"Warning: Failed to load OCR models: {e}")

    try:
        from deepdoc.vision import LayoutRecognizer

        layout_engine = LayoutRecognizer("layout")
        logger.info("Layout models loaded successfully.")
    except Exception as e:
        logger.error(f"Warning: Failed to load layout models: {e}")


def warmup_models(
    *,
    image_size: int | None = None,
    enabled: bool | None = None,
    load_if_needed: bool = True,
) -> dict[str, object]:
    global ocr_engine, layout_engine
    warmup_enabled = _parse_bool(os.environ.get("DEEPDOC_MODEL_WARMUP"), default=True) if enabled is None else enabled
    if not warmup_enabled:
        logger.info("Skipping OCR/layout model warmup (DEEPDOC_MODEL_WARMUP=0).")
        return {"status": "skipped", "reason": "disabled", "source": "model_warmup"}

    if load_if_needed and (not ocr_engine or not layout_engine):
        load_models()
    if not ocr_engine or not layout_engine:
        logger.warning("Skipping OCR/layout model warmup: models are not initialized.")
        return {"status": "skipped", "reason": "models_not_initialized", "source": "model_warmup"}

    size = max(16, int(image_size or os.environ.get("DEEPDOC_MODEL_WARMUP_IMAGE_SIZE", "64")))
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    started_at = time.perf_counter()
    try:
        ocr_lines = ocr_engine(image)
        ocr_boxes = _ocr_lines_to_boxes(ocr_lines)
        layout_engine([image], [ocr_boxes], scale_factor=1)
    except Exception as exc:
        logger.exception("OCR/layout model warmup failed")
        return {"status": "error", "error": str(exc), "source": "model_warmup"}

    duration_seconds = max(0.0, time.perf_counter() - started_at)
    logger.info(
        "OCR/layout model warmup completed in %.3fs (image_size=%s, ocr_box_count=%s)",
        duration_seconds,
        size,
        len(ocr_boxes),
    )
    return {
        "status": "ok",
        "source": "model_warmup",
        "image_size": size,
        "ocr_box_count": len(ocr_boxes),
        "duration_seconds": duration_seconds,
    }


def _ocr_lines_to_boxes(ocr_lines) -> list[dict[str, object]]:
    if not ocr_lines:
        return []

    boxes: list[dict[str, object]] = []
    for line in ocr_lines:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        points, rec = line
        if not isinstance(points, (list, tuple)) or len(points) < 4:
            continue
        if not isinstance(rec, (list, tuple)) or len(rec) < 1:
            text = ""
            score = 0.0
        else:
            text = str(rec[0] or "")
            score = float(rec[1]) if len(rec) > 1 and rec[1] is not None else 0.0

        if not text.strip():
            continue

        try:
            x0 = float(points[0][0])
            x1 = float(points[1][0])
            top = float(points[0][1])
            bottom = float(points[-1][1])
        except Exception:
            continue

        if x0 > x1 or top > bottom:
            continue

        boxes.append(
            {
                "text": text,
                "score": score,
                "x0": x0,
                "x1": x1,
                "top": top,
                "bottom": bottom,
                "page_number": 0,
            }
        )
    return boxes


def _parse_image_from_tmp(tmp_path: str, parse_options: dict[str, object]):
    global ocr_engine, layout_engine
    if not ocr_engine or not layout_engine:
        load_models()
    if not ocr_engine or not layout_engine:
        raise RuntimeError("Models not initialized")

    import cv2

    file_bytes = Path(tmp_path).read_bytes()
    if not file_bytes:
        raise ValueError("Empty image file")
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file or format not supported")

    ocr_lines = ocr_engine(img)
    ocr_boxes = _ocr_lines_to_boxes(ocr_lines)
    ocr_res, page_layout = layout_engine([img], [ocr_boxes], scale_factor=1)
    boxes = ocr_res if isinstance(ocr_res, list) else ocr_boxes
    if boxes and not isinstance(boxes[0], dict):
        boxes = ocr_boxes

    try:
        pil_image = Image.open(tmp_path).convert("RGB")
    except Exception:
        logger.exception("Failed to open uploaded image with PIL")
        pil_image = Image.fromarray(img[:, :, ::-1]).convert("RGB")
    try:
        from deepdoc.vision.barcode import detect_barcodes

        barcodes = detect_barcodes(pil_image)
    except Exception:
        logger.exception("Failed to detect barcodes from uploaded image")
        barcodes = []

    parse_meta = {
        "page_count": 1,
        "total_page_count": 1,
        "structured_source": {
            "engine": "image",
            "image": pil_image,
            "boxes": deepcopy(boxes),
            "barcodes": barcodes,
            "page_layout": page_layout,
        },
    }
    return boxes, [], parse_meta


@app.route("/health", methods=["GET"])
def health_check():
    expose_internal = _health_internal_details_exposed()
    artifact_backend = ARTIFACT_STORE.__class__.__name__.removesuffix("ArtifactStore").lower() or "local"
    ingest_publisher_type = getattr(INGEST_PUBLISHER, "sink_type", "unknown")
    build_info = get_build_info()
    self_checks = _self_checks_health()
    retention_janitor = _retention_janitor_health()
    tracing_state = get_tracing_state()
    request_protection = _request_protection_health()
    async_tasks = _async_tasks_health()
    audit_log_status: dict[str, object] | None = None
    if AUDIT_LOG_STORE is not None:
        try:
            audit_log_status = AUDIT_LOG_STORE.check_health()
        except Exception as exc:
            audit_log_status = {"status": "error", "error": str(exc)}
    ingest_query_backend = None
    ingest_query_status: dict[str, object] | None = None
    if INGEST_QUERY_STORE is not None:
        ingest_query_backend = INGEST_QUERY_STORE.__class__.__name__.removesuffix("IngestStore").lower() or "postgres"
        try:
            ingest_query_status = INGEST_QUERY_STORE.check_health()
        except Exception as exc:
            ingest_query_status = {"status": "error", "error": str(exc)}
    update_backend_metrics(
        artifact_backend=artifact_backend,
        ingest_publisher=ingest_publisher_type,
        ingest_query_backend=ingest_query_backend,
        auth_mode=_auth_mode(),
        ocr_loaded=ocr_engine is not None,
        tracing_enabled=bool(tracing_state.enabled and tracing_state.instrumented),
        tracing_exporters=",".join(tracing_state.exporters),
    )
    update_build_metrics(build_info)
    update_async_task_metrics(async_tasks)
    update_self_check_metrics(self_checks)
    update_retention_janitor_metrics(retention_janitor)
    payload = {
        "status": "ok",
        "ocr_loaded": ocr_engine is not None,
        "model_path": os.environ.get("DEEPDOC_MODEL_PATH"),
        "auth_mode": _auth_mode(),
        "default_tenant_id": DEFAULT_TENANT_ID,
        "api_docs": _api_docs_state(),
        "cors": CORS_STATE,
        "runtime_config": RUNTIME_CONFIG_STATE,
        "build": summarize_build_info(build_info),
        "artifact_backend": artifact_backend,
        "ingest_publisher": ingest_publisher_type,
        "self_checks": self_checks,
        "retention_janitor": retention_janitor,
        "tracing": tracing_state.to_dict(),
        "ingest_query_backend": ingest_query_backend,
        "ingest_query_status": ingest_query_status,
        "audit_log": audit_log_status,
        "request_protection": request_protection,
        "async_tasks": async_tasks,
        "internal_details_exposed": expose_internal,
    }
    return jsonify(_sanitize_health_payload(payload, expose_internal=expose_internal))


@app.route("/ready", methods=["GET"])
def readiness_check():
    expose_internal = _health_internal_details_exposed()
    required_groups = _readiness_model_groups()
    missing_models = list_missing_files(required_groups)
    build_info = get_build_info()
    artifact_backend_state = _artifact_backend_health()
    request_protection = _request_protection_health()
    async_tasks = _async_tasks_health()
    self_checks = _self_checks_health()
    retention_janitor = _retention_janitor_health()
    audit_log_state: dict[str, object]
    if AUDIT_LOG_STORE is None:
        audit_log_state = {"status": "disabled"}
    else:
        try:
            audit_log_state = AUDIT_LOG_STORE.check_health()
        except Exception as exc:
            logger.exception("Readiness audit log health check failed")
            audit_log_state = {"status": "error", "error": str(exc)}
    ingest_state: dict[str, object]
    if INGEST_QUERY_STORE is None and getattr(INGEST_PUBLISHER, "sink_type", "none") == "postgres":
        ingest_state = {"status": "error", "error": "postgres ingest backend is configured but unavailable"}
    elif INGEST_QUERY_STORE is None:
        ingest_state = {"status": "disabled"}
    else:
        try:
            ingest_state = INGEST_QUERY_STORE.check_health()
        except Exception as exc:
            logger.exception("Readiness ingest health check failed")
            ingest_state = {"status": "error", "error": str(exc)}

    rate_limit_state = request_protection.get("rate_limit") if isinstance(request_protection, dict) else {}
    rate_limit_required = bool(rate_limit_state.get("enabled")) and not bool(rate_limit_state.get("fail_open", True))
    ready = (
        not missing_models
        and str(artifact_backend_state.get("status") or "") == "ok"
        and str(ingest_state.get("status") or "ok") in {"ok", "disabled"}
        and str(((async_tasks.get("store") or {}).get("status") or "ok")) == "ok"
        and str(((async_tasks.get("broker") or {}).get("status") or "disabled")) in {"ok", "disabled"}
        and str(((async_tasks.get("worker") or {}).get("status") or "disabled")) in {"ok", "disabled"}
        and str(((async_tasks.get("callback_redrive_worker") or {}).get("status") or "disabled")) in {"ok", "disabled"}
        and str((audit_log_state.get("status") or "ok")) in {"ok", "disabled"}
        and str((self_checks.get("status") or "ok")) in {"ok", "disabled"}
        and str((retention_janitor.get("status") or "ok")) in {"ok", "disabled"}
        and (
            str((rate_limit_state or {}).get("status") or "ok") == "ok"
            or not rate_limit_required
        )
    )
    status_code = 200 if ready else 503
    update_build_metrics(build_info)
    update_async_task_metrics(async_tasks)
    update_self_check_metrics(self_checks)
    update_retention_janitor_metrics(retention_janitor)
    payload = {
        "status": "ready" if ready else "not_ready",
        "required_model_groups": required_groups,
        "missing_model_files": missing_models,
        "api_docs": _api_docs_state(),
        "cors": CORS_STATE,
        "runtime_config": RUNTIME_CONFIG_STATE,
        "build": summarize_build_info(build_info),
        "artifact_backend": artifact_backend_state,
        "self_checks": self_checks,
        "retention_janitor": retention_janitor,
        "ingest_store": ingest_state,
        "audit_log": audit_log_state,
        "request_protection": request_protection,
        "async_tasks": async_tasks,
        "internal_details_exposed": expose_internal,
    }
    return jsonify(_sanitize_health_payload(payload, expose_internal=expose_internal)), status_code


@app.route("/metrics", methods=["GET"])
def metrics():
    artifact_backend = ARTIFACT_STORE.__class__.__name__.removesuffix("ArtifactStore").lower() or "local"
    ingest_publisher_type = getattr(INGEST_PUBLISHER, "sink_type", "unknown")
    build_info = get_build_info()
    tracing_state = get_tracing_state()
    ingest_query_backend = None
    if INGEST_QUERY_STORE is not None:
        ingest_query_backend = INGEST_QUERY_STORE.__class__.__name__.removesuffix("IngestStore").lower() or "postgres"
    update_backend_metrics(
        artifact_backend=artifact_backend,
        ingest_publisher=ingest_publisher_type,
        ingest_query_backend=ingest_query_backend,
        auth_mode=_auth_mode(),
        ocr_loaded=ocr_engine is not None,
        tracing_enabled=bool(tracing_state.enabled and tracing_state.instrumented),
        tracing_exporters=",".join(tracing_state.exporters),
    )
    update_build_metrics(build_info)
    update_async_task_metrics(_async_tasks_health(include_history=True))
    update_self_check_metrics(_self_checks_health(include_history=True))
    update_retention_janitor_metrics(_retention_janitor_health(include_history=True))
    payload, content_type = render_metrics_payload()
    return Response(payload, mimetype=content_type)




def _artifact_parse_dir(parse_id: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f-]{12,64}", (parse_id or "").strip()):
        return None
    return Path(parse_id)


def _parse_asset_url_mode() -> str:
    mode = (request.args.get("asset_url_mode") or "proxy").strip().lower()
    if mode not in {"proxy", "direct", "signed"}:
        raise ValueError("asset_url_mode must be one of: proxy, direct, signed")
    return mode


def _parse_asset_url_ttl() -> int:
    raw_value = (request.args.get("expires_in") or "").strip()
    if not raw_value:
        return int(os.environ.get("DEEPDOC_ARTIFACT_SIGNED_URL_TTL", "3600"))
    ttl = int(raw_value)
    if ttl < 60 or ttl > 604800:
        raise ValueError("expires_in must be between 60 and 604800 seconds")
    return ttl


def _async_queue_name() -> str:
    return str(getattr(ASYNC_TASK_BROKER, "queue_name", "deepdoc:async:parse"))


def _async_task_result_summary(results: list[dict[str, object]]) -> dict[str, object]:
    error_count = sum(1 for item in results if "error" in item)
    success_count = len(results) - error_count
    parse_ids = [str(item.get("parse_id") or "").strip() for item in results if str(item.get("parse_id") or "").strip()]
    return {
        "file_count": len(results),
        "success_count": success_count,
        "error_count": error_count,
        "parse_ids": parse_ids,
    }


def _accumulate_gpu_page_pool_summary(
    existing: dict[str, object] | None,
    planned: dict[str, object],
    *,
    filename: str,
) -> dict[str, object]:
    summary = deepcopy(existing) if isinstance(existing, dict) else {}
    planned_files = list(summary.get("planned_files") or [])
    planned_files.append(
        {
            "filename": filename,
            "page_count": int(planned.get("page_count") or 0),
            "submitted_job_count": int(planned.get("submitted_job_count") or 0),
            "worker_device_ids": [int(device_id) for device_id in planned.get("worker_device_ids") or []],
        }
    )
    summary["planned_files"] = planned_files
    summary["submitted_job_count"] = int(summary.get("submitted_job_count") or 0) + int(planned.get("submitted_job_count") or 0)
    summary["page_count"] = int(summary.get("page_count") or 0) + int(planned.get("page_count") or 0)

    worker_device_ids = {int(device_id) for device_id in summary.get("worker_device_ids") or [] if int(device_id) >= 0}
    worker_device_ids.update(int(device_id) for device_id in planned.get("worker_device_ids") or [] if int(device_id) >= 0)
    summary["worker_device_ids"] = sorted(worker_device_ids)

    device_job_counts: dict[int, int] = {}
    for key, value in (summary.get("device_job_counts") or {}).items():
        try:
            device_job_counts[int(key)] = max(0, int(value))
        except Exception:
            continue
    for key, value in (planned.get("device_job_counts") or {}).items():
        try:
            device_id = int(key)
            device_job_counts[device_id] = device_job_counts.get(device_id, 0) + max(0, int(value))
        except Exception:
            continue
    summary["device_job_counts"] = device_job_counts
    return summary


def _merge_async_task_result_summary(
    base_summary: dict[str, object],
    *,
    existing_summary: dict[str, object] | None = None,
    progress: dict[str, int] | None = None,
    gpu_page_pool: dict[str, object] | None = None,
) -> dict[str, object]:
    merged = dict(base_summary)
    if isinstance(existing_summary, dict):
        existing_gpu_page_pool = existing_summary.get("gpu_page_pool")
        if isinstance(existing_gpu_page_pool, dict):
            merged["gpu_page_pool"] = deepcopy(existing_gpu_page_pool)
    if isinstance(gpu_page_pool, dict):
        merged["gpu_page_pool"] = deepcopy(gpu_page_pool)
    if progress is not None:
        merged["progress"] = progress
    return merged


def _resolve_gpu_page_pool_devices() -> list[int]:
    try:
        from deepdoc.vision.ocr import ensure_parallel_devices_configured

        detected = max(0, int(ensure_parallel_devices_configured() or 0))
    except Exception:
        detected = max(0, int(os.environ.get("PARALLEL_DEVICES", "0") or 0))
    if detected <= 0:
        detected = 1
    return list(range(detected))


def _plan_gpu_page_pool_dispatch(input_file: AsyncTaskInput, parse_options: dict[str, object], *, task_id: str) -> dict[str, object] | None:
    parser_engine = str(parse_options.get("parser_engine") or "").strip().lower()
    file_type = str(input_file.file_type or Path(input_file.filename).suffix.lstrip(".")).strip().lower()
    if parser_engine != "deepdoc" or file_type != "pdf":
        return None
    if _normalize_deepdoc_pdf_mode(str(parse_options.get("deepdoc_pdf_mode") or "auto")) != "hybrid":
        return None
    if _normalize_execution_profile(str(parse_options.get("execution_profile") or "auto")) != "gpu":
        return None

    from common.gpu_page_pool import dispatch_gpu_page_jobs
    from deepdoc.parser.pdf_hybrid_router import build_pdf_hybrid_plan

    max_pdf_pages = max(1, int(parse_options.get("deepdoc_max_pages") or os.environ.get("DEEPDOC_PDF_MAX_PAGES", "10")))
    hybrid_plan = build_pdf_hybrid_plan(input_file.source_path, page_from=0, page_to=max_pdf_pages)
    page_jobs = [
        {
            "job_id": f"{task_id}:page-{int(page.get('page_number') or 0)}",
            "page_number": int(page.get("page_number") or 0),
            "route": str(page.get("route") or "unknown").strip() or "unknown",
            "ocr_scope": str(page.get("ocr_scope") or "unknown").strip() or "unknown",
            "reasons": [str(reason) for reason in page.get("reasons") or [] if str(reason).strip()],
        }
        for page in hybrid_plan.get("pages") or []
        if int(page.get("page_number") or 0) > 0
    ]
    dispatch_summary = dispatch_gpu_page_jobs(
        task_id=task_id,
        page_jobs=page_jobs,
        devices=_resolve_gpu_page_pool_devices(),
    )
    return {
        **dispatch_summary,
        "filename": input_file.filename,
        "page_count": int(hybrid_plan.get("page_count") or len(page_jobs)),
        "hybrid_route_summary": deepcopy(hybrid_plan.get("route_summary") or {}),
        "ocr_page_numbers": [int(page_number) for page_number in hybrid_plan.get("ocr_page_numbers") or [] if int(page_number) > 0],
        "complex_block_page_numbers": [
            int(page_number) for page_number in hybrid_plan.get("complex_block_page_numbers") or [] if int(page_number) > 0
        ],
    }


def _stage_async_uploads(files) -> tuple[str, list[AsyncTaskInput]]:
    staged: list[AsyncTaskInput] = []
    max_files = max(1, int(os.environ.get("DEEPDOC_ASYNC_MAX_FILES", "32")))
    if len(files) > max_files:
        raise ValueError(f"Too many files for async parse: {len(files)} > {max_files}")

    temp_task_id = uuid4().hex
    paths = ASYNC_TASK_STORE.get_paths(temp_task_id)
    try:
        for index, file in enumerate(files):
            filename = file.filename or ""
            if not filename:
                raise ValueError("No selected file")
            file_type = _infer_file_type(filename)
            if not file_type:
                raise ValueError(f"Unsupported file extension: {filename}")
            validation_error = validate_file(file, allowed_extensions=list(PARSER_IMPORTS.keys()))
            if validation_error:
                raise ValueError(validation_error)
            safe_name = f"{index:02d}-{uuid4().hex[:8]}-{Path(filename).name}"
            target_path = Path(paths.source_dir) / safe_name
            file.save(target_path)
            payload = target_path.read_bytes()
            staged.append(
                AsyncTaskInput(
                    filename=filename,
                    file_type=file_type,
                    size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    source_path=str(target_path),
                )
            )
        return temp_task_id, staged
    except Exception:
        shutil.rmtree(paths.task_dir, ignore_errors=True)
        raise


def _promote_async_staging_task(temp_task_id: str, task: AsyncTask) -> list[AsyncTaskInput]:
    temp_paths = ASYNC_TASK_STORE.get_paths(temp_task_id)
    final_paths = ASYNC_TASK_STORE.get_paths(task.task_id)
    final_source_dir = Path(final_paths.source_dir)
    final_inputs: list[AsyncTaskInput] = []
    for input_file in task.input_files:
        source_path = Path(input_file.source_path)
        final_path = final_source_dir / source_path.name
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(final_path))
        final_inputs.append(input_file.model_copy(update={"source_path": str(final_path)}))
    shutil.rmtree(temp_paths.task_dir, ignore_errors=True)
    return final_inputs


def _render_chunk_jsonl(parse_id: str, asset_url_mode: str, expires_in: int) -> Response:
    payload, _ = ARTIFACT_STORE.read_file(parse_id, "structured.json", "application/json")
    artifact = ParseArtifact.model_validate(json.loads(payload.decode("utf-8")))
    records = build_chunk_export_records(
        artifact,
        store=ARTIFACT_STORE,
        asset_url_mode=asset_url_mode,
        signed_url_ttl=expires_in,
    )
    body = b"\n".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
        for record in records
    ) + (b"\n" if records else b"")
    return Response(body, mimetype="application/x-ndjson")


def _render_ingest_jsonl(parse_id: str, asset_url_mode: str, expires_in: int) -> Response:
    payload, _ = ARTIFACT_STORE.read_file(parse_id, "structured.json", "application/json")
    artifact = ParseArtifact.model_validate(json.loads(payload.decode("utf-8")))
    chunk_records = build_chunk_export_records(
        artifact,
        store=ARTIFACT_STORE,
        asset_url_mode=asset_url_mode,
        signed_url_ttl=expires_in,
    )
    records = build_ingest_export_records(chunk_records)
    body = b"\n".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
        for record in records
    ) + (b"\n" if records else b"")
    return Response(body, mimetype="application/x-ndjson")




def _sse_event(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _collect_sse_response_events(response: Response) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    current_event: str | None = None
    current_data_lines: list[str] = []

    def flush() -> None:
        nonlocal current_event, current_data_lines
        if not current_event:
            current_data_lines = []
            return
        payload_text = "\n".join(current_data_lines).strip()
        payload: object = None
        if payload_text:
            try:
                payload = json.loads(payload_text)
            except Exception:
                payload = payload_text
        events.append({"event": current_event, "data": payload})
        current_event = None
        current_data_lines = []

    for chunk in response.response:
        text = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        for line in text.splitlines():
            if not line.strip():
                flush()
                continue
            if line.startswith("event: "):
                current_event = line[len("event: ") :].strip()
                continue
            if line.startswith("data: "):
                current_data_lines.append(line[len("data: ") :])
    flush()
    return events


def _cleanup_candidates_from_manifests(
    manifests: list[ParseManifest],
    *,
    older_than_days: int | None,
    keep_latest: int | None,
    parser_engine: str | None,
    file_type: str | None,
) -> list[ParseManifest]:
    filtered = manifests
    if parser_engine:
        filtered = [manifest for manifest in filtered if manifest.parser_engine == parser_engine]
    if file_type:
        filtered = [manifest for manifest in filtered if manifest.file_type == file_type]
    if older_than_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, older_than_days))
        filtered = [
            manifest
            for manifest in filtered
            if datetime.fromisoformat(manifest.created_at) <= cutoff
        ]
    if keep_latest and keep_latest > 0:
        kept_ids = {manifest.parse_id for manifest in manifests[:keep_latest]}
        filtered = [manifest for manifest in filtered if manifest.parse_id not in kept_ids]
    return filtered


def _select_publish_retry_candidates(
    manifests: list[ParseManifest],
    *,
    limit: int,
    parser_engine: str | None,
    file_type: str | None,
    only_due: bool,
    include_dead_lettered: bool,
) -> list[ParseManifest]:
    now = datetime.now(timezone.utc)
    selected: list[ParseManifest] = []
    for manifest in manifests:
        if parser_engine and manifest.parser_engine != parser_engine:
            continue
        if file_type and manifest.file_type != file_type:
            continue
        publish_state = manifest.metadata.get("ingest_publish") if isinstance(manifest.metadata, dict) else None
        if not isinstance(publish_state, dict):
            continue
        if str(publish_state.get("status") or "").strip().lower() != "failed":
            continue
        if not include_dead_lettered and bool(publish_state.get("dead_lettered", False)):
            continue
        if only_due:
            if not bool(publish_state.get("retryable", False)):
                continue
            next_retry_at = str(publish_state.get("next_retry_at") or "").strip()
            if next_retry_at:
                try:
                    if datetime.fromisoformat(next_retry_at) > now:
                        continue
                except ValueError:
                    logger.warning("Invalid next_retry_at in manifest %s", manifest.parse_id)
        selected.append(manifest)
        if len(selected) >= limit:
            break
    return selected


def _require_ingest_query_store():
    if INGEST_QUERY_STORE is None:
        return None, (jsonify({"error": "ingest query backend unavailable"}), 503)
    return INGEST_QUERY_STORE, None


def _self_check_environment_snapshot() -> dict[str, object]:
    ingest_state = INGEST_QUERY_STORE.check_health() if INGEST_QUERY_STORE is not None else {"status": "disabled"}
    return {
        "artifact_backend": getattr(ARTIFACT_STORE, "backend", ARTIFACT_STORE.__class__.__name__),
        "ingest_publisher": getattr(INGEST_PUBLISHER, "sink_type", "unknown"),
        "auth_mode": _auth_mode(),
        "ingest_store": ingest_state,
        "tracing": get_tracing_state().to_dict(),
    }


def _finalize_self_check_step(
    *,
    name: str,
    started_at: float,
    started_iso: str,
    status: str,
    summary: str,
    details: dict[str, object] | None = None,
) -> SelfCheckStep:
    duration_ms = int(max(0.0, (time.time() - started_at) * 1000.0))
    return SelfCheckStep(
        step_id=uuid4().hex,
        name=name,
        status=status,
        started_at=started_iso,
        finished_at=datetime.now(timezone.utc).isoformat(),
        duration_ms=duration_ms,
        summary=summary,
        details=details or {},
    )


def _execute_self_check_step(name: str, runner) -> SelfCheckStep:
    started_at = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    try:
        summary, details = runner()
        return _finalize_self_check_step(
            name=name,
            started_at=started_at,
            started_iso=started_iso,
            status="passed",
            summary=summary,
            details=details,
        )
    except Exception as exc:
        logger.exception("Self-check step failed: %s", name)
        return _finalize_self_check_step(
            name=name,
            started_at=started_at,
            started_iso=started_iso,
            status="failed",
            summary=str(exc),
            details={"error": str(exc), "traceback": traceback.format_exc()},
        )


def _skipped_self_check_step(name: str, summary: str, details: dict[str, object] | None = None) -> SelfCheckStep:
    now_iso = datetime.now(timezone.utc).isoformat()
    return SelfCheckStep(
        step_id=uuid4().hex,
        name=name,
        status="skipped",
        started_at=now_iso,
        finished_at=now_iso,
        duration_ms=0,
        summary=summary,
        details=details or {},
    )


def _build_self_check_asset_visual() -> Image.Image:
    image = Image.new("RGB", (480, 260), "#eef4ff")
    canvas = ImageDraw.Draw(image)
    canvas.rounded_rectangle((20, 20, 460, 240), radius=18, outline="#2557d6", width=3, fill="#f7faff")
    canvas.rectangle((45, 72, 180, 210), outline="#2557d6", width=2)
    canvas.rectangle((205, 50, 430, 110), outline="#2557d6", width=2)
    canvas.rectangle((205, 132, 430, 210), outline="#2557d6", width=2)
    canvas.line((180, 141, 205, 141), fill="#2557d6", width=3)
    canvas.text((55, 42), PRODUCT_NAME, fill="#12306b")
    canvas.text((225, 70), "Chunks", fill="#12306b")
    canvas.text((225, 152), "Assets + Links", fill="#12306b")
    canvas.text((58, 112), "Parser", fill="#12306b")
    return image


def _build_real_pdf_self_check_visual() -> Image.Image:
    image = Image.new("RGB", (360, 220), "white")
    canvas = ImageDraw.Draw(image)
    canvas.rounded_rectangle((20, 20, 340, 200), radius=16, outline="navy", width=4)
    canvas.rectangle((50, 70, 140, 150), outline="darkgreen", width=3)
    canvas.rectangle((220, 70, 310, 150), outline="firebrick", width=3)
    canvas.line((140, 110, 220, 110), fill="black", width=3)
    canvas.text((95, 50), "Parser", fill="black")
    canvas.text((245, 50), "Assets", fill="black")
    return image


def _write_real_pdf_self_check(path: str) -> None:
    image_path = None
    document = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=str(UPLOAD_TMP_DIR)) as tmp:
            image_path = tmp.name
        _build_real_pdf_self_check_visual().save(image_path)
        document = fitz.open()
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 72), SELF_CHECK_TITLE, fontsize=18)
        page.insert_text(
            (72, 110),
            "Figure 1 shows the parser and asset storage flow. This paragraph should recall the diagram.",
            fontsize=11,
        )
        page.insert_image(fitz.Rect(100, 160, 480, 420), filename=image_path)
        page.insert_text((180, 440), "Figure 1: parser and asset storage flow", fontsize=11)
        document.set_metadata(
            {
                "title": SELF_CHECK_TITLE,
                "author": SELF_CHECK_AUTHOR,
                "creator": SELF_CHECK_AUTHOR,
                "producer": SELF_CHECK_AUTHOR,
                "subject": "production self-check",
                "keywords": f"{PRODUCT_SLUG},self-check,pdf,structured-assets",
            }
        )
        document.save(path, garbage=4, deflate=True, pretty=False)
    finally:
        if document is not None:
            document.close()
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


def _self_check_asset_profile(tenant_id: str | None) -> dict[str, object]:
    return {
        "version": "2026-06-07-self-check-v1",
        "suite": "asset_export",
        "tenant_id": tenant_id or "",
        "profile_version": ARTIFACT_PROFILE_VERSION,
    }


def _ensure_self_check_asset_artifact(*, tenant_id: str | None) -> tuple[ParseArtifact, object, ParseManifest, bool]:
    artifact_profile = _self_check_asset_profile(tenant_id)
    file_bytes = b"deepdoc:self-check:asset-export:v1"
    artifact_key = build_artifact_key(hashlib.sha256(file_bytes).hexdigest(), artifact_profile)
    existing_manifest = ARTIFACT_STORE.find_manifest_by_artifact_key(artifact_key, tenant_id=tenant_id)
    if existing_manifest is not None:
        return (
            _load_parse_artifact(existing_manifest.parse_id),
            ARTIFACT_STORE.get_paths(existing_manifest.parse_id, existing_manifest.filename),
            existing_manifest,
            True,
        )

    document = build_document(
        filename="self-check-asset-export.pdf",
        file_type="pdf",
        parser_engine="deepdoc",
        file_bytes=file_bytes,
        page_count=1,
        total_page_count=1,
        metadata={"tenant_id": tenant_id, "self_check": True},
    )
    artifact_paths = ARTIFACT_STORE.get_paths(document.parse_id, document.filename)
    asset_id = "asset-self-check-figure-1"
    image = _build_self_check_asset_visual()
    storage = ARTIFACT_STORE.save_image_asset(paths=artifact_paths, image=image, asset_id=asset_id)
    asset = ParseAsset(
        asset_id=asset_id,
        asset_type="figure",
        title="Self-check architecture diagram",
        text="Synthetic architecture diagram for production self-check validation",
        page_numbers=[1],
        positions=[],
        width=image.width,
        height=image.height,
        sha256=None,
        storage=storage,
        metadata={"self_check": True},
    )
    blocks = [
        ParseBlock(
            block_id="block-self-check-title-1",
            block_type="title",
            text="1 Self Check Overview",
            page_numbers=[1],
            token_count=count_tokens("1 Self Check Overview"),
            metadata={"self_check": True},
        ),
        ParseBlock(
            block_id="block-self-check-text-1",
            block_type="text",
            text=(
                "architecture architecture architecture architecture architecture "
                "architecture architecture architecture architecture architecture "
                "Related assets: [Figure] Self-check architecture diagram"
            ),
            page_numbers=[1],
            token_count=count_tokens(
                "architecture architecture architecture architecture architecture "
                "architecture architecture architecture architecture architecture "
                "Related assets: [Figure] Self-check architecture diagram"
            ),
            metadata={"self_check": True},
        ),
        ParseBlock(
            block_id="block-self-check-figure-1",
            block_type="figure",
            text="Self-check architecture diagram",
            page_numbers=[1],
            token_count=count_tokens("Self-check architecture diagram"),
            asset_refs=[asset_id],
            metadata={"self_check": True},
        ),
        ParseBlock(
            block_id="block-self-check-text-2",
            block_type="text",
            text=(
                "runtime runtime runtime runtime runtime runtime runtime runtime runtime runtime "
                "runtime runtime runtime runtime runtime runtime runtime runtime runtime runtime"
            ),
            page_numbers=[1],
            token_count=count_tokens(
                "runtime runtime runtime runtime runtime runtime runtime runtime runtime runtime "
                "runtime runtime runtime runtime runtime runtime runtime runtime runtime runtime"
            ),
            metadata={"self_check": True},
        ),
    ]
    chunks = [
        ParseChunk(
            chunk_id="chunk-self-check-context-1",
            text=(
                "# 1 Self Check Overview\n\n"
                "architecture architecture architecture architecture architecture architecture "
                "architecture architecture architecture architecture\n\n"
                "Related assets: [Figure] Self-check architecture diagram"
            ),
            token_count=count_tokens(
                "# 1 Self Check Overview\n\n"
                "architecture architecture architecture architecture architecture architecture "
                "architecture architecture architecture architecture\n\n"
                "Related assets: [Figure] Self-check architecture diagram"
            ),
            page_numbers=[1],
            block_refs=["block-self-check-title-1", "block-self-check-text-1"],
            asset_refs=[asset_id],
            metadata={
                "title_path": ["1 Self Check Overview"],
                "direct_asset_refs": [],
                "context_asset_refs": [asset_id],
                "chunk_strategy": "structure_aware_v2",
                "self_check": True,
            },
        ),
        ParseChunk(
            chunk_id="chunk-self-check-direct-1",
            text=(
                "Section: 1 Self Check Overview\n\n"
                "[Figure]\nSelf-check architecture diagram\n\n"
                "runtime runtime runtime runtime runtime runtime runtime runtime runtime runtime"
            ),
            token_count=count_tokens(
                "Section: 1 Self Check Overview\n\n"
                "[Figure]\nSelf-check architecture diagram\n\n"
                "runtime runtime runtime runtime runtime runtime runtime runtime runtime runtime"
            ),
            page_numbers=[1],
            block_refs=["block-self-check-figure-1", "block-self-check-text-2"],
            asset_refs=[asset_id],
            metadata={
                "title_path": ["1 Self Check Overview"],
                "direct_asset_refs": [asset_id],
                "context_asset_refs": [],
                "chunk_strategy": "structure_aware_v2",
                "self_check": True,
            },
        ),
    ]
    markdown = (
        "# 1 Self Check Overview\n\n"
        "Related assets: [Figure] Self-check architecture diagram\n\n"
        "[Figure]\nSelf-check architecture diagram"
    )
    artifact = ParseArtifact(
        document=document,
        markdown=markdown,
        assets=[asset],
        blocks=blocks,
        chunks=chunks,
        metadata={"parser_engine": "deepdoc", "tenant_id": tenant_id, "self_check": True},
    )
    manifest = build_parse_manifest(
        artifact,
        artifact_paths,
        storage_backend=ARTIFACT_STORE.__class__.__name__.removesuffix("ArtifactStore").lower() or "local",
        artifact_key=artifact_key,
        extra_metadata={"artifact_profile": artifact_profile, "self_check": True},
    )
    ARTIFACT_STORE.write_source(artifact_paths, file_bytes)
    ARTIFACT_STORE.write_markdown(artifact_paths, markdown)
    ARTIFACT_STORE.write_manifest(artifact_paths, manifest)
    ARTIFACT_STORE.write_structured(artifact_paths, artifact)
    ARTIFACT_STORE.write_chunks(artifact_paths, artifact)
    ARTIFACT_STORE.write_ingest(artifact_paths, artifact)
    return artifact, artifact_paths, manifest, False


def _run_text_parse_export_self_check(*, tenant_id: str | None, force_reparse: bool) -> tuple[str, dict[str, object]]:
    content = (
        f"{PRODUCT_NAME} production self-check validates parsed markdown, structured artifacts, and chunk exports.\n"
        "The parser stores chunks with stable identifiers so downstream systems can consume them directly."
    ).encode("utf-8")
    temp_path = None
    upload = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", dir=str(UPLOAD_TMP_DIR)) as tmp:
            temp_path = tmp.name
            tmp.write(content)
        upload = LocalUploadedFile(temp_path, "self-check-parse-export.txt")
        parse_options = {
            "parser_engine": "markitdown",
            "return_structured": True,
            "persist_artifacts": True,
            "persist_source": True,
            "publish_ingest": True,
            "publish_requested_by": "self-check",
            "include_chunks": True,
            "reuse_artifacts": not bool(force_reparse),
            "return_images": False,
            "strict_text": False,
            "tenant_id": tenant_id,
        }
        result = _parse_single_file(upload, parse_options)
        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        parse_id = str(result.get("parse_id") or "").strip()
        document_id = str(result.get("document_id") or "").strip()
        if not parse_id or not document_id:
            raise RuntimeError("text parse self-check returned empty parse identifiers")
        if int(result.get("chunk_count") or 0) <= 0:
            raise RuntimeError("text parse self-check returned no chunks")
        artifact = _load_parse_artifact(parse_id)
        if not artifact.markdown.strip():
            raise RuntimeError("text parse self-check returned empty markdown")
        if not artifact.chunks:
            raise RuntimeError("text parse self-check structured artifact has no chunks")
        return (
            "Text parse, structured artifact persistence, and chunk export succeeded.",
            {
                "parse_id": parse_id,
                "document_id": document_id,
                "cache_hit": bool(result.get("cache_hit")),
                "ingest_publish": result.get("ingest_publish") or {},
                "chunk_count": len(artifact.chunks),
                "block_count": len(artifact.blocks),
                "asset_count": len(artifact.assets),
            },
        )
    finally:
        if upload is not None:
            upload.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _run_asset_export_self_check(*, tenant_id: str | None, force_republish: bool) -> tuple[str, dict[str, object]]:
    artifact, artifact_paths, manifest, reused = _ensure_self_check_asset_artifact(tenant_id=tenant_id)
    publish_options = {
        "publish_ingest": True,
        "publish_requested_by": "self-check",
        "publish_asset_url_mode": "proxy",
    }
    current_publish = manifest.metadata.get("ingest_publish") if isinstance(manifest.metadata, dict) else None
    if force_republish or not isinstance(current_publish, dict) or current_publish.get("status") != "published":
        publish_result, manifest = _publish_ingest_records(
            artifact=artifact,
            manifest=manifest,
            artifact_paths=artifact_paths,
            parse_options=publish_options,
        )
    else:
        publish_result = current_publish
    chunk_records = build_chunk_export_records(artifact, store=ARTIFACT_STORE, asset_url_mode="proxy", signed_url_ttl=3600)
    ingest_records = build_ingest_export_records(chunk_records)
    if not chunk_records:
        raise RuntimeError("asset self-check returned no chunk records")
    if not ingest_records:
        raise RuntimeError("asset self-check returned no export records")
    asset_linked_chunks = [record for record in chunk_records if record.asset_refs or record.assets]
    if not asset_linked_chunks:
        raise RuntimeError("asset self-check returned no asset-linked chunks")
    if INGEST_QUERY_STORE is not None:
        asset_rows = INGEST_QUERY_STORE.list_assets(parse_id=manifest.parse_id, tenant_id=tenant_id, limit=10)
        chunk_rows = INGEST_QUERY_STORE.list_chunks(parse_id=manifest.parse_id, tenant_id=tenant_id, limit=10)
        link_rows = INGEST_QUERY_STORE.list_chunk_asset_links(parse_id=manifest.parse_id, tenant_id=tenant_id, limit=20)
    else:
        asset_rows = []
        chunk_rows = []
        link_rows = []
    return (
        "Synthetic asset artifact, chunk export, and ingest record export succeeded.",
        {
            "parse_id": manifest.parse_id,
            "document_id": manifest.document_id,
            "artifact_key": manifest.artifact_key,
            "artifact_reused": reused,
            "ingest_publish": publish_result,
            "chunk_record_count": len(chunk_records),
            "ingest_record_count": len(ingest_records),
            "asset_linked_chunk_count": len(asset_linked_chunks),
            "asset_rows": len(asset_rows),
            "chunk_rows": len(chunk_rows),
            "link_rows": len(link_rows),
        },
    )


def _run_real_pdf_asset_export_self_check(*, tenant_id: str | None) -> tuple[str, dict[str, object]]:
    temp_path = None
    upload = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=str(UPLOAD_TMP_DIR)) as tmp:
            temp_path = tmp.name
        _write_real_pdf_self_check(temp_path)
        upload = LocalUploadedFile(temp_path, "self-check-real-asset-export.pdf")
        parse_options = {
            "parser_engine": "deepdoc",
            "return_structured": True,
            "persist_artifacts": True,
            "persist_source": True,
            "publish_ingest": True,
            "publish_requested_by": "self-check",
            "include_chunks": True,
            "reuse_artifacts": False,
            "return_images": True,
            "strict_text": False,
            "deepdoc_layout_model": "general",
            "deepdoc_max_pages": 2,
            "tenant_id": tenant_id,
        }
        result = _parse_single_file(upload, parse_options)
        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        parse_id = str(result.get("parse_id") or "").strip()
        document_id = str(result.get("document_id") or "").strip()
        if not parse_id or not document_id:
            raise RuntimeError("real pdf self-check returned empty parse identifiers")
        if int(result.get("asset_count") or 0) <= 0:
            raise RuntimeError("real pdf self-check returned no extracted assets")
        if int(result.get("chunk_count") or 0) <= 0:
            raise RuntimeError("real pdf self-check returned no chunks")
        artifact = _load_parse_artifact(parse_id)
        chunk_records = build_chunk_export_records(artifact, store=ARTIFACT_STORE, asset_url_mode="proxy", signed_url_ttl=3600)
        asset_linked_chunks = [record for record in chunk_records if record.asset_refs or record.assets]
        if not asset_linked_chunks:
            raise RuntimeError("real pdf self-check returned no asset-linked chunks")
        if INGEST_QUERY_STORE is not None:
            asset_rows = INGEST_QUERY_STORE.list_assets(parse_id=parse_id, tenant_id=tenant_id, limit=10)
            chunk_rows = INGEST_QUERY_STORE.list_chunks(parse_id=parse_id, tenant_id=tenant_id, limit=10)
            link_rows = INGEST_QUERY_STORE.list_chunk_asset_links(parse_id=parse_id, tenant_id=tenant_id, limit=20)
        else:
            asset_rows = []
            chunk_rows = []
            link_rows = []
        return (
            "Real PDF parse, asset extraction, structured artifact, and asset-linked chunk export succeeded.",
            {
                "parse_id": parse_id,
                "document_id": document_id,
                "asset_count": len(artifact.assets),
                "chunk_count": len(artifact.chunks),
                "asset_linked_chunk_count": len(asset_linked_chunks),
                "ingest_publish": result.get("ingest_publish") or {},
                "asset_rows": len(asset_rows),
                "chunk_rows": len(chunk_rows),
                "link_rows": len(link_rows),
            },
        )
    finally:
        if upload is not None:
            upload.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)




def _run_core_self_check(
    *,
    tenant_id: str | None,
    force_reparse: bool,
    force_republish: bool,
    requested_by: str | None = None,
    auto_run: bool = False,
) -> SelfCheckRun:
    run = new_self_check_run(
        suite="core",
        environment=_self_check_environment_snapshot(),
        metadata={
            "tenant_id": tenant_id,
            "force_reparse": bool(force_reparse),
            "force_republish": bool(force_republish),
            "requested_by": str(requested_by or "").strip() or "manual",
            "auto_run": bool(auto_run),
        },
    )
    SELF_CHECK_STORE.write_run(run)
    steps = [
        _execute_self_check_step(
            "text_parse_export",
            lambda: _run_text_parse_export_self_check(tenant_id=tenant_id, force_reparse=force_reparse),
        ),
        _execute_self_check_step(
            "asset_export",
            lambda: _run_asset_export_self_check(tenant_id=tenant_id, force_republish=force_republish),
        ),
        _execute_self_check_step(
            "real_pdf_asset_export",
            lambda: _run_real_pdf_asset_export_self_check(tenant_id=tenant_id),
        ),
    ]
    failed_steps = [step for step in steps if step.status == "failed"]
    skipped_steps = [step for step in steps if step.status == "skipped"]
    finished_at = datetime.now(timezone.utc).isoformat()
    duration_ms = int(
        max(
            0.0,
            (
                datetime.fromisoformat(finished_at) - datetime.fromisoformat(run.created_at)
            ).total_seconds()
            * 1000.0,
        )
    )
    status = "failed" if failed_steps else "passed"
    if not failed_steps:
        if skipped_steps:
            summary = (
                "Production self-check passed, with non-blocking skipped steps: "
                + ", ".join(step.name for step in skipped_steps)
                + "."
            )
        else:
            summary = "Production self-check passed for parse, structured artifact, chunk export, and asset export."
    else:
        summary = f"Production self-check failed in {', '.join(step.name for step in failed_steps)}."
    completed = run.model_copy(
        update={
            "status": status,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "steps": steps,
            "summary": summary,
        }
    )
    SELF_CHECK_STORE.write_run(completed)
    observe_self_check_run(suite=completed.suite, status=completed.status, duration_ms=completed.duration_ms)
    _append_ops_audit_event(
        "self_check.run",
        resource_type="self_check",
        resource_id=completed.check_id,
        status=completed.status,
        payload=completed.model_dump(mode="json"),
        metadata={
            "suite": completed.suite,
            "requested_by": str(completed.metadata.get("requested_by") or "").strip() or "manual",
            "auto_run": bool(completed.metadata.get("auto_run")),
        },
        tenant_id=tenant_id,
    )
    return completed


def run_self_check_worker_once(
    *,
    tenant_id: str | None,
    force_reparse: bool,
    force_republish: bool,
    requested_by: str = "self-check-worker",
) -> SelfCheckRun:
    return _run_core_self_check(
        tenant_id=tenant_id,
        force_reparse=force_reparse,
        force_republish=force_republish,
        requested_by=requested_by,
        auto_run=True,
    )


def _retention_janitor_auth_context() -> dict[str, object]:
    return {
        "mode": "system",
        "subject": "retention-janitor",
        "tenant_id": None,
        "is_admin": True,
        "scopes": ["admin"],
    }


def _retention_janitor_artifact_cleanup(*, rule, requested_by: str) -> dict[str, object]:
    manifests = ARTIFACT_STORE.list_manifests(limit=max(1, int(rule.limit or 1000)), tenant_id=None)
    candidates = _cleanup_candidates_from_manifests(
        manifests,
        older_than_days=rule.older_than_days,
        keep_latest=rule.keep_latest,
        parser_engine=None,
        file_type=None,
    )
    deleted = 0
    for manifest in candidates:
        ARTIFACT_STORE.delete_artifact(manifest)
        deleted += 1
    result = {
        "status": "ok",
        "scanned": len(manifests),
        "candidate_count": len(candidates),
        "deleted_count": deleted,
        "older_than_days": rule.older_than_days,
        "keep_latest": rule.keep_latest,
        "limit": rule.limit,
    }
    _append_ops_audit_event(
        "artifacts.cleanup",
        resource_type="artifact",
        status="ok",
        payload=result,
        metadata={"requested_by": requested_by, "automated": True},
        tenant_id=None,
        auth_context=_retention_janitor_auth_context(),
    )
    return result


def _retention_janitor_task_cleanup(*, rule, requested_by: str) -> dict[str, object]:
    raw_result = ASYNC_TASK_STORE.cleanup_tasks(
        limit=max(1, int(rule.limit or 1000)),
        tenant_id=None,
        older_than_days=rule.older_than_days,
        keep_latest=rule.keep_latest,
        statuses=list(rule.statuses),
        include_active=bool(rule.include_active),
        dry_run=False,
    )
    result = {
        **raw_result,
        "status": "ok",
    }
    _append_ops_audit_event(
        "tasks.cleanup",
        resource_type="async_task",
        status="ok",
        payload=result,
        metadata={"requested_by": requested_by, "automated": True},
        tenant_id=None,
        auth_context=_retention_janitor_auth_context(),
    )
    return result




def _retention_janitor_audit_cleanup(*, rule, requested_by: str) -> dict[str, object]:
    audit_store = _get_ops_audit_store()
    if audit_store is None:
        raise RuntimeError("ops audit backend unavailable")
    raw_result = audit_store.cleanup_events(
        tenant_id=None,
        older_than_days=rule.older_than_days,
        keep_latest=max(0, int(rule.keep_latest or 0)),
        action=None,
        resource_type=None,
        status=None,
        dry_run=False,
    )
    result = {
        **raw_result,
        "status_filter": raw_result.get("status"),
        "status": "ok",
    }
    _append_ops_audit_event(
        "audit.cleanup",
        resource_type="audit_event",
        status="ok",
        payload=result,
        metadata={"requested_by": requested_by, "automated": True},
        tenant_id=None,
        auth_context=_retention_janitor_auth_context(),
    )
    return result


def _retention_janitor_self_check_cleanup(*, rule, requested_by: str) -> dict[str, object]:
    raw_result = SELF_CHECK_STORE.cleanup_runs(
        older_than_days=rule.older_than_days,
        keep_latest=rule.keep_latest,
        limit=max(1, int(rule.limit or 1000)),
        dry_run=False,
        status=None,
    )
    result = {
        **raw_result,
        "status": "ok",
    }
    _append_ops_audit_event(
        "self_check.cleanup",
        resource_type="self_check",
        status="ok",
        payload=result,
        metadata={"requested_by": requested_by, "automated": True},
        tenant_id=None,
        auth_context=_retention_janitor_auth_context(),
    )
    return result


def run_retention_janitor_once(*, requested_by: str = "retention-janitor") -> dict[str, object]:
    config = _retention_janitor_config()
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    plane_results: dict[str, dict[str, object]] = {}
    status = "ok"
    run_id = uuid4().hex
    plane_handlers = {
        "tasks": _retention_janitor_task_cleanup,
        "artifacts": _retention_janitor_artifact_cleanup,
        "audit_events": _retention_janitor_audit_cleanup,
        "self_checks": _retention_janitor_self_check_cleanup,
    }
    for plane_name, rule in config.rules.items():
        if not bool(rule.enabled):
            plane_results[plane_name] = {"status": "disabled"}
            continue
        try:
            plane_results[plane_name] = plane_handlers[plane_name](rule=rule, requested_by=requested_by)
        except Exception as exc:
            logger.exception("Retention janitor plane failed plane=%s", plane_name)
            plane_results[plane_name] = {
                "status": "error",
                "error": str(exc),
            }
            status = "error"
    duration_ms = max(0, int((time.monotonic() - started_monotonic) * 1000))
    finished_at = datetime.now(timezone.utc)
    run = {
        "run_id": run_id,
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_ms": duration_ms,
        "requested_by": requested_by,
        "planes": plane_results,
    }
    observe_retention_janitor_run(status=status, duration_ms=duration_ms, plane_results=plane_results)
    _append_ops_audit_event(
        "retention_janitor.run",
        resource_type="retention_janitor",
        resource_id=run_id,
        status=status,
        payload=run,
        metadata={"requested_by": requested_by, "automated": True},
        tenant_id=None,
        auth_context=_retention_janitor_auth_context(),
    )
    return run




def _get_ops_audit_store():
    store = AUDIT_LOG_STORE
    if store is None:
        return None
    required_methods = ("append_event", "get_event", "list_events", "cleanup_events")
    if all(hasattr(store, name) for name in required_methods):
        return store
    return None


def _resolve_audit_tenant_filter(*, payload: dict[str, object] | None = None) -> str | None:
    auth_context = _current_auth_context()
    requested_tenant = None
    if isinstance(payload, dict):
        requested_tenant = str(payload.get("tenant_id") or "").strip() or None
    if requested_tenant is None:
        requested_tenant = (request.args.get("tenant_id") or "").strip() or None
    if _auth_mode() == "none":
        return requested_tenant
    if bool(auth_context.get("is_admin")):
        return requested_tenant
    current_tenant = str(auth_context.get("tenant_id") or "").strip() or None
    if requested_tenant and requested_tenant != current_tenant:
        raise PermissionError("tenant_id override is not allowed")
    return current_tenant


@app.route("/api/v1/audit/events", methods=["GET"])
def list_audit_events():
    try:
        _require_admin_capability("audit event listing")
        tenant_filter = _resolve_audit_tenant_filter()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    audit_store = _get_ops_audit_store()
    if audit_store is None:
        return jsonify({"error": "ops audit backend unavailable"}), 503
    try:
        limit = max(1, min(int((request.args.get("limit") or "20").strip()), 200))
        offset = max(0, int((request.args.get("offset") or "0").strip()))
    except ValueError:
        return jsonify({"error": "limit and offset must be integers"}), 400
    resource_id = (request.args.get("resource_id") or "").strip() or None
    try:
        events = audit_store.list_events(
            tenant_id=tenant_filter,
            limit=1000 if resource_id else limit,
            offset=0 if resource_id else offset,
            action=(request.args.get("action") or "").strip() or None,
            resource_type=(request.args.get("resource_type") or "").strip() or None,
            status=(request.args.get("status") or "").strip() or None,
            request_id=(request.args.get("request_id") or "").strip() or None,
            actor_subject=(request.args.get("actor_subject") or "").strip() or None,
        )
    except Exception as exc:
        logger.exception("Failed to list audit events")
        return jsonify({"error": str(exc)}), 500
    if resource_id:
        events = [event for event in events if str(event.get("resource_id") or "").strip() == resource_id]
        events = events[offset : offset + limit]
    return _jsonify_external(
        {
            "results": events,
            "limit": limit,
            "offset": offset,
            "tenant_id": tenant_filter,
        }
    )


@app.route("/api/v1/audit/events/<event_id>", methods=["GET"])
def get_audit_event(event_id: str):
    try:
        _require_admin_capability("audit event inspection")
        tenant_filter = _resolve_audit_tenant_filter()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    audit_store = _get_ops_audit_store()
    if audit_store is None:
        return jsonify({"error": "ops audit backend unavailable"}), 503
    try:
        event = audit_store.get_event(event_id, tenant_id=tenant_filter)
    except Exception as exc:
        logger.exception("Failed to load audit event event_id=%s", event_id)
        return jsonify({"error": str(exc)}), 500
    if event is None:
        return jsonify({"error": "audit event not found"}), 404
    return _jsonify_external({"event": event})


@app.route("/api/v1/audit/events/cleanup", methods=["POST"])
def cleanup_audit_events():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object"}), 400
    try:
        _require_admin_capability("audit event cleanup")
        tenant_filter = _resolve_audit_tenant_filter(payload=payload)
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    audit_store = _get_ops_audit_store()
    if audit_store is None:
        return jsonify({"error": "ops audit backend unavailable"}), 503
    try:
        older_than_days = payload.get("older_than_days")
        older_than_days = int(older_than_days) if older_than_days is not None else None
        if older_than_days is not None:
            older_than_days = max(0, older_than_days)
        keep_latest = max(0, int(payload.get("keep_latest", 0) or 0))
        dry_run = _parse_bool(payload.get("dry_run"), default=True)
        result = audit_store.cleanup_events(
            tenant_id=tenant_filter,
            older_than_days=older_than_days,
            keep_latest=keep_latest,
            action=str(payload.get("action") or "").strip() or None,
            resource_type=str(payload.get("resource_type") or "").strip() or None,
            status=str(payload.get("status") or "").strip() or None,
            dry_run=dry_run,
        )
    except (TypeError, ValueError):
        return jsonify({"error": "invalid cleanup payload"}), 400
    except Exception as exc:
        logger.exception("Failed to cleanup audit events")
        return jsonify({"error": str(exc)}), 500
    _append_ops_audit_event(
        "audit.cleanup",
        resource_type="audit_event",
        status="ok",
        payload=result,
        metadata={"requested_by": "api"},
        tenant_id=tenant_filter,
    )
    return _jsonify_external(result)




def _load_accessible_manifest(parse_id: str) -> ParseManifest:
    payload, _ = ARTIFACT_STORE.read_file(parse_id, "manifest.json", "application/json")
    manifest = parse_manifest_payload(json.loads(payload.decode("utf-8")))
    _ensure_manifest_access(manifest)
    return manifest


@app.route("/api/v1/artifacts", methods=["GET"])
def list_artifact_manifests():
    limit_raw = (request.args.get("limit") or "20").strip()
    publish_status = (request.args.get("publish_status") or "").strip().lower() or None
    publish_retryable_raw = (request.args.get("publish_retryable") or "").strip().lower()
    try:
        limit = max(1, min(int(limit_raw), 200))
    except ValueError:
        return jsonify({"error": "limit must be an integer between 1 and 200"}), 400
    publish_retryable = None
    if publish_retryable_raw:
        if publish_retryable_raw not in {"true", "false", "1", "0", "yes", "no"}:
            return jsonify({"error": "publish_retryable must be a boolean string"}), 400
        publish_retryable = _parse_bool(publish_retryable_raw, default=False)
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = (request.args.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    manifests = ARTIFACT_STORE.list_manifests(limit=limit, tenant_id=tenant_filter)
    if publish_status:
        manifests = [
            manifest
            for manifest in manifests
            if str(((manifest.metadata.get("ingest_publish") or {}).get("status") or "")).strip().lower() == publish_status
        ]
    if publish_retryable is not None:
        manifests = [
            manifest
            for manifest in manifests
            if bool((manifest.metadata.get("ingest_publish") or {}).get("retryable", False)) is publish_retryable
        ]
    manifests = [manifest.model_dump(mode="json") for manifest in manifests]
    return _jsonify_external({"results": manifests})


@app.route("/api/v1/artifacts/cleanup", methods=["POST"])
def cleanup_artifacts():
    payload = request.get_json(silent=True) or {}
    try:
        limit = max(1, min(int(payload.get("limit", 1000)), 5000))
        keep_latest = payload.get("keep_latest")
        if keep_latest is not None:
            keep_latest = max(0, int(keep_latest))
        older_than_days = payload.get("older_than_days")
        if older_than_days is not None:
            older_than_days = max(0, int(older_than_days))
        dry_run = bool(payload.get("dry_run", True))
        parser_engine = str(payload.get("parser_engine") or "").strip() or None
        file_type = str(payload.get("file_type") or "").strip() or None
    except (TypeError, ValueError):
        return jsonify({"error": "invalid cleanup payload"}), 400
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(payload.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id

    manifests = ARTIFACT_STORE.list_manifests(limit=limit, tenant_id=tenant_filter)
    candidates = _cleanup_candidates_from_manifests(
        manifests,
        older_than_days=older_than_days,
        keep_latest=keep_latest,
        parser_engine=parser_engine,
        file_type=file_type,
    )
    deleted: list[dict[str, object]] = []
    if not dry_run:
        for manifest in candidates:
            deleted.append(ARTIFACT_STORE.delete_artifact(manifest))
    result = {
        "dry_run": dry_run,
        "total_scanned": len(manifests),
        "candidate_count": len(candidates),
        "candidates": [manifest.model_dump(mode="json") for manifest in candidates],
        "deleted": deleted,
    }
    _append_ops_audit_event(
        "artifacts.cleanup",
        resource_type="artifact",
        status="ok",
        payload=result,
        metadata={"parser_engine": parser_engine, "file_type": file_type},
        tenant_id=tenant_filter,
    )
    return _jsonify_external(result)


@app.route("/api/v1/artifacts/<parse_id>/manifest", methods=["GET"])
def get_manifest_artifact(parse_id: str):
    if _artifact_parse_dir(parse_id) is None:
        return jsonify({"error": "invalid artifact id"}), 404
    try:
        manifest = _load_accessible_manifest(parse_id)
    except FileNotFoundError:
        return jsonify({"error": "artifact not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    return _jsonify_external(manifest.model_dump(mode="json"))


@app.route("/api/v1/artifacts/<parse_id>/publish-events", methods=["GET"])
def get_publish_events_artifact(parse_id: str):
    if _artifact_parse_dir(parse_id) is None:
        return jsonify({"error": "invalid artifact id"}), 404
    try:
        _load_accessible_manifest(parse_id)
        payload, media_type = ARTIFACT_STORE.read_file(parse_id, "publish-events.jsonl", "application/x-ndjson")
    except FileNotFoundError:
        return Response(b"", mimetype="application/x-ndjson")
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    lines: list[bytes] = []
    for raw_line in payload.decode("utf-8").splitlines():
        if not raw_line.strip():
            continue
        sanitized_line = _external_response_payload(json.loads(raw_line))
        lines.append(json.dumps(sanitized_line, ensure_ascii=False).encode("utf-8"))
    body = b"\n".join(lines) + (b"\n" if lines else b"")
    return Response(body, mimetype=media_type)


@app.route("/api/v1/artifacts/<parse_id>/publish", methods=["POST"])
def publish_artifact_ingest(parse_id: str):
    if _artifact_parse_dir(parse_id) is None:
        return jsonify({"error": "invalid artifact id"}), 404
    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force", False))
    requested_by = str(payload.get("requested_by") or "retry").strip() or "retry"
    asset_url_mode = str(payload.get("asset_url_mode") or "").strip().lower() or None
    expires_in = payload.get("expires_in")
    if asset_url_mode and asset_url_mode not in {"proxy", "direct", "signed"}:
        return jsonify({"error": "asset_url_mode must be one of: proxy, direct, signed"}), 400
    if expires_in is not None:
        try:
            expires_in = int(expires_in)
        except (TypeError, ValueError):
            return jsonify({"error": "expires_in must be an integer"}), 400
        if expires_in < 60 or expires_in > 604800:
            return jsonify({"error": "expires_in must be between 60 and 604800 seconds"}), 400

    try:
        manifest = _load_accessible_manifest(parse_id)
        artifact = _load_parse_artifact(parse_id)
    except FileNotFoundError:
        return jsonify({"error": "artifact not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403

    publish_status = manifest.metadata.get("ingest_publish") if isinstance(manifest.metadata, dict) else None
    if not force and isinstance(publish_status, dict) and publish_status.get("status") == "published":
        return _jsonify_external(
            {
                "parse_id": manifest.parse_id,
                "artifact_key": manifest.artifact_key,
                "skipped": True,
                "ingest_publish": publish_status,
            }
        )

    publish_options: dict[str, object] = {
        "publish_ingest": True,
        "publish_requested_by": requested_by,
    }
    if asset_url_mode:
        publish_options["publish_asset_url_mode"] = asset_url_mode
    if expires_in is not None:
        publish_options["publish_signed_url_ttl"] = expires_in
    publish_result, updated_manifest = _publish_ingest_records(
        artifact=artifact,
        manifest=manifest,
        artifact_paths=ARTIFACT_STORE.get_paths(manifest.parse_id, manifest.filename),
        parse_options=publish_options,
    )
    result = {
        "parse_id": (updated_manifest or manifest).parse_id,
        "artifact_key": (updated_manifest or manifest).artifact_key,
        "skipped": False,
        "ingest_publish": publish_result,
    }
    _append_ops_audit_event(
        "artifact.publish",
        resource_type="artifact",
        resource_id=parse_id,
        status=str((publish_result or {}).get("status") or "unknown"),
        payload=result,
        metadata={"requested_by": requested_by, "force": force},
        tenant_id=_manifest_tenant_id(updated_manifest or manifest),
    )
    return _jsonify_external(result)


@app.route("/api/v1/artifacts/publish-retry", methods=["POST"])
def retry_failed_artifact_ingest():
    payload = request.get_json(silent=True) or {}
    try:
        limit = max(1, min(int(payload.get("limit", 50)), 500))
        scan_limit = max(limit, min(int(payload.get("scan_limit", max(limit * 5, 100))), 5000))
        only_due = bool(payload.get("only_due", True))
        include_dead_lettered = bool(payload.get("include_dead_lettered", False))
        force = bool(payload.get("force", False))
        parser_engine = str(payload.get("parser_engine") or "").strip() or None
        file_type = str(payload.get("file_type") or "").strip() or None
        requested_by = str(payload.get("requested_by") or "retry-batch").strip() or "retry-batch"
        requested_tenant = str(payload.get("tenant_id") or "").strip() or None
        asset_url_mode = str(payload.get("asset_url_mode") or "").strip().lower() or None
        expires_in = payload.get("expires_in")
        if asset_url_mode and asset_url_mode not in {"proxy", "direct", "signed"}:
            raise ValueError("asset_url_mode must be one of: proxy, direct, signed")
        if expires_in is not None:
            expires_in = int(expires_in)
            if expires_in < 60 or expires_in > 604800:
                raise ValueError("expires_in must be between 60 and 604800 seconds")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "invalid retry payload"}), 400

    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    manifests = ARTIFACT_STORE.list_manifests(limit=scan_limit, tenant_id=tenant_filter)
    candidates = _select_publish_retry_candidates(
        manifests,
        limit=limit,
        parser_engine=parser_engine,
        file_type=file_type,
        only_due=only_due and not force,
        include_dead_lettered=include_dead_lettered or force,
    )

    results: list[dict[str, object]] = []
    succeeded = 0
    failed = 0
    skipped = 0
    for manifest in candidates:
        try:
            artifact = _load_parse_artifact(manifest.parse_id)
        except FileNotFoundError:
            skipped += 1
            results.append(
                {
                    "parse_id": manifest.parse_id,
                    "artifact_key": manifest.artifact_key,
                    "status": "skipped",
                    "reason": "structured artifact missing",
                }
            )
            continue
        publish_options: dict[str, object] = {
            "publish_ingest": True,
            "publish_requested_by": requested_by,
        }
        if asset_url_mode:
            publish_options["publish_asset_url_mode"] = asset_url_mode
        if expires_in is not None:
            publish_options["publish_signed_url_ttl"] = expires_in
        publish_result, updated_manifest = _publish_ingest_records(
            artifact=artifact,
            manifest=manifest,
            artifact_paths=ARTIFACT_STORE.get_paths(manifest.parse_id, manifest.filename),
            parse_options=publish_options,
        )
        status = str((publish_result or {}).get("status") or "unknown")
        if status == "published":
            succeeded += 1
        elif status == "failed":
            failed += 1
        else:
            skipped += 1
        results.append(
            {
                "parse_id": (updated_manifest or manifest).parse_id,
                "artifact_key": (updated_manifest or manifest).artifact_key,
                "status": status,
                "ingest_publish": publish_result,
            }
        )

    response_payload = {
        "scanned": len(manifests),
        "candidate_count": len(candidates),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }
    _append_ops_audit_event(
        "artifacts.publish_retry",
        resource_type="artifact",
        status="ok" if failed == 0 else "partial",
        payload=response_payload,
        metadata={
            "requested_by": requested_by,
            "only_due": only_due and not force,
            "include_dead_lettered": include_dead_lettered or force,
            "parser_engine": parser_engine,
            "file_type": file_type,
        },
        tenant_id=tenant_filter,
    )
    return _jsonify_external(response_payload)


@app.route("/api/v1/ingest/documents", methods=["GET"])
def list_ingest_documents():
    store, error_response = _require_ingest_query_store()
    if error_response:
        return error_response
    try:
        limit = max(1, min(int((request.args.get("limit") or "20").strip()), 200))
        offset = max(0, int((request.args.get("offset") or "0").strip()))
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400
    parser_engine = str(request.args.get("parser_engine") or "").strip() or None
    file_type = str(request.args.get("file_type") or "").strip() or None
    publish_status = str(request.args.get("publish_status") or "").strip() or None
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(request.args.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    try:
        results = store.list_documents(
            limit=limit,
            offset=offset,
            tenant_id=tenant_filter,
            parser_engine=parser_engine,
            file_type=file_type,
            publish_status=publish_status,
        )
    except Exception as exc:
        logger.exception("Failed to list ingest documents")
        return jsonify({"error": str(exc)}), 500
    return _jsonify_external({"results": results})


@app.route("/api/v1/ingest/stats", methods=["GET"])
def get_ingest_stats():
    store, error_response = _require_ingest_query_store()
    if error_response:
        return error_response
    parse_id = str(request.args.get("parse_id") or "").strip() or None
    document_id = str(request.args.get("document_id") or "").strip() or None
    parser_engine = str(request.args.get("parser_engine") or "").strip() or None
    file_type = str(request.args.get("file_type") or "").strip() or None
    include_breakdown = _parse_bool(request.args.get("include_breakdown"), default=True)
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(request.args.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    try:
        result = store.get_stats(
            tenant_id=tenant_filter,
            parse_id=parse_id,
            document_id=document_id,
            parser_engine=parser_engine,
            file_type=file_type,
            include_breakdown=include_breakdown,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to load ingest stats")
        return jsonify({"error": str(exc)}), 500
    return _jsonify_external(result)


@app.route("/api/v1/ingest/documents/<parse_id>", methods=["GET"])
def get_ingest_document(parse_id: str):
    store, error_response = _require_ingest_query_store()
    if error_response:
        return error_response
    if _artifact_parse_dir(parse_id) is None:
        return jsonify({"error": "invalid parse_id"}), 404
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(request.args.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    try:
        result = store.get_document(parse_id, tenant_id=tenant_filter)
    except Exception as exc:
        logger.exception("Failed to load ingest document parse_id=%s", parse_id)
        return jsonify({"error": str(exc)}), 500
    if result is None:
        return jsonify({"error": "ingest document not found"}), 404
    return _jsonify_external(result)


@app.route("/api/v1/ingest/assets", methods=["GET"])
def list_ingest_assets():
    store, error_response = _require_ingest_query_store()
    if error_response:
        return error_response
    if not hasattr(store, "list_assets"):
        return jsonify({"error": "ingest asset queries are not supported by the active backend"}), 501
    try:
        limit = max(1, min(int((request.args.get("limit") or "20").strip()), 500))
        offset = max(0, int((request.args.get("offset") or "0").strip()))
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400
    parse_id = str(request.args.get("parse_id") or "").strip() or None
    document_id = str(request.args.get("document_id") or "").strip() or None
    asset_type = str(request.args.get("asset_type") or "").strip() or None
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(request.args.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    try:
        results = store.list_assets(
            limit=limit,
            offset=offset,
            tenant_id=tenant_filter,
            parse_id=parse_id,
            document_id=document_id,
            asset_type=asset_type,
        )
    except Exception as exc:
        logger.exception("Failed to list ingest assets")
        return jsonify({"error": str(exc)}), 500
    return _jsonify_external({"results": results})


@app.route("/api/v1/ingest/assets/<parse_id>/<asset_id>", methods=["GET"])
def get_ingest_asset(parse_id: str, asset_id: str):
    store, error_response = _require_ingest_query_store()
    if error_response:
        return error_response
    if not hasattr(store, "get_asset"):
        return jsonify({"error": "ingest asset queries are not supported by the active backend"}), 501
    if _artifact_parse_dir(parse_id) is None:
        return jsonify({"error": "invalid parse_id"}), 404
    if not asset_id.strip():
        return jsonify({"error": "invalid asset_id"}), 404
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(request.args.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    try:
        result = store.get_asset(parse_id, asset_id, tenant_id=tenant_filter)
    except Exception as exc:
        logger.exception("Failed to load ingest asset parse_id=%s asset_id=%s", parse_id, asset_id)
        return jsonify({"error": str(exc)}), 500
    if result is None:
        return jsonify({"error": "ingest asset not found"}), 404
    return _jsonify_external(result)


@app.route("/api/v1/ingest/chunks", methods=["GET"])
def list_ingest_chunks():
    store, error_response = _require_ingest_query_store()
    if error_response:
        return error_response
    if not hasattr(store, "list_chunks"):
        return jsonify({"error": "ingest chunk queries are not supported by the active backend"}), 501
    try:
        limit = max(1, min(int((request.args.get("limit") or "20").strip()), 500))
        offset = max(0, int((request.args.get("offset") or "0").strip()))
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400
    parse_id = str(request.args.get("parse_id") or "").strip() or None
    document_id = str(request.args.get("document_id") or "").strip() or None
    chunk_id = str(request.args.get("chunk_id") or "").strip() or None
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(request.args.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    try:
        results = store.list_chunks(
            limit=limit,
            offset=offset,
            tenant_id=tenant_filter,
            parse_id=parse_id,
            document_id=document_id,
            chunk_id=chunk_id,
        )
    except Exception as exc:
        logger.exception("Failed to list ingest chunks")
        return jsonify({"error": str(exc)}), 500
    return _jsonify_external({"results": results})


@app.route("/api/v1/ingest/chunk-asset-links", methods=["GET"])
def list_ingest_chunk_asset_links():
    store, error_response = _require_ingest_query_store()
    if error_response:
        return error_response
    if not hasattr(store, "list_chunk_asset_links"):
        return jsonify({"error": "ingest chunk-asset links are not supported by the active backend"}), 501
    try:
        limit = max(1, min(int((request.args.get("limit") or "50").strip()), 1000))
        offset = max(0, int((request.args.get("offset") or "0").strip()))
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400
    parse_id = str(request.args.get("parse_id") or "").strip() or None
    document_id = str(request.args.get("document_id") or "").strip() or None
    chunk_id = str(request.args.get("chunk_id") or "").strip() or None
    asset_id = str(request.args.get("asset_id") or "").strip() or None
    relation_type = str(request.args.get("relation_type") or "").strip().lower() or None
    if relation_type and relation_type not in {"direct", "context"}:
        return jsonify({"error": "relation_type must be direct or context"}), 400
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(request.args.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    try:
        results = store.list_chunk_asset_links(
            limit=limit,
            offset=offset,
            tenant_id=tenant_filter,
            parse_id=parse_id,
            document_id=document_id,
            chunk_id=chunk_id,
            asset_id=asset_id,
            relation_type=relation_type,
        )
    except Exception as exc:
        logger.exception("Failed to list ingest chunk asset links")
        return jsonify({"error": str(exc)}), 500
    return _jsonify_external({"results": results})


@app.route("/api/v1/ingest/records", methods=["GET"])
def search_ingest_records():
    store, error_response = _require_ingest_query_store()
    if error_response:
        return error_response
    try:
        limit = max(1, min(int((request.args.get("limit") or "20").strip()), 200))
        offset = max(0, int((request.args.get("offset") or "0").strip()))
    except ValueError:
        return jsonify({"error": "limit/offset must be integers"}), 400
    query = str(request.args.get("q") or "").strip() or None
    parse_id = str(request.args.get("parse_id") or "").strip() or None
    document_id = str(request.args.get("document_id") or "").strip() or None
    mode = str(request.args.get("mode") or "text").strip().lower() or "text"
    if mode != "text":
        return jsonify({"error": "mode must be text"}), 400
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(request.args.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    try:
        results = store.search_records(
            query=query,
            tenant_id=tenant_filter,
            parse_id=parse_id,
            document_id=document_id,
            limit=limit,
            offset=offset,
            mode=mode,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to search ingest records")
        return jsonify({"error": str(exc)}), 500
    return _jsonify_external({"results": results})




@app.route("/api/v1/self-checks", methods=["GET"])
def list_self_checks():
    try:
        _require_admin_capability("self-check listing")
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    try:
        limit = max(1, min(int((request.args.get("limit") or "20").strip()), 200))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    status_filter = str(request.args.get("status") or "").strip().lower() or None
    if status_filter and status_filter not in {"running", "passed", "failed"}:
        return jsonify({"error": "status must be one of: running, passed, failed"}), 400
    results = [
        run.model_dump(mode="json")
        for run in SELF_CHECK_STORE.list_runs(limit=limit, status=status_filter)
    ]
    return _jsonify_external({"results": results})


@app.route("/api/v1/self-checks/<check_id>", methods=["GET"])
def get_self_check(check_id: str):
    try:
        _require_admin_capability("self-check inspection")
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    try:
        run = SELF_CHECK_STORE.load_run(check_id)
    except FileNotFoundError:
        return jsonify({"error": "self-check not found"}), 404
    except Exception as exc:
        logger.exception("Failed to load self-check run")
        return jsonify({"error": str(exc)}), 500
    return _jsonify_external(run.model_dump(mode="json"))


@app.route("/api/v1/self-checks/run", methods=["POST"])
def run_self_check():
    try:
        _require_admin_capability("self-check execution")
        tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object"}), 400
    suite = str(payload.get("suite") or "core").strip().lower() or "core"
    if suite != "core":
        return jsonify({"error": "unsupported self-check suite"}), 400
    force_reparse = _parse_bool(payload.get("force_reparse"), default=False)
    force_republish = _parse_bool(payload.get("force_republish"), default=False)
    run = _run_core_self_check(
        tenant_id=tenant_id,
        force_reparse=force_reparse,
        force_republish=force_republish,
        requested_by="api",
        auto_run=False,
    )
    status_code = 200 if run.status == "passed" else 500
    return _jsonify_external(run.model_dump(mode="json"), status_code=status_code)


@app.route("/api/v1/self-checks/cleanup", methods=["POST"])
def cleanup_self_checks():
    try:
        _require_admin_capability("self-check cleanup")
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object"}), 400
    try:
        limit = max(1, min(int(payload.get("limit", 1000)), 5000))
        keep_latest = payload.get("keep_latest")
        keep_latest = int(keep_latest) if keep_latest is not None else None
        if keep_latest is not None:
            keep_latest = max(0, keep_latest)
        older_than_days = payload.get("older_than_days")
        older_than_days = int(older_than_days) if older_than_days is not None else None
        if older_than_days is not None:
            older_than_days = max(0, older_than_days)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid cleanup payload"}), 400
    dry_run = _parse_bool(payload.get("dry_run"), default=True)
    status_filter = str(payload.get("status") or "").strip().lower() or None
    if status_filter and status_filter not in {"running", "passed", "failed"}:
        return jsonify({"error": "status must be one of: running, passed, failed"}), 400
    result = SELF_CHECK_STORE.cleanup_runs(
        older_than_days=older_than_days,
        keep_latest=keep_latest,
        limit=limit,
        dry_run=dry_run,
        status=status_filter,
    )
    _append_ops_audit_event(
        "self_check.cleanup",
        resource_type="self_check",
        status="ok",
        payload=result,
        metadata={
            "dry_run": dry_run,
            "older_than_days": older_than_days,
            "keep_latest": keep_latest,
            "status": status_filter,
        },
        tenant_id=None,
    )
    return _jsonify_external(result)




@app.route("/api/v1/artifacts/<parse_id>/structured", methods=["GET"])
def get_structured_artifact(parse_id: str):
    if _artifact_parse_dir(parse_id) is None:
        return jsonify({"error": "invalid artifact id"}), 404
    try:
        _load_accessible_manifest(parse_id)
        payload, _ = ARTIFACT_STORE.read_file(parse_id, "structured.json", "application/json")
    except FileNotFoundError:
        return jsonify({"error": "artifact not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    return _jsonify_external(json.loads(payload.decode("utf-8")))


@app.route("/api/v1/artifacts/<parse_id>/markdown", methods=["GET"])
def get_markdown_artifact(parse_id: str):
    if _artifact_parse_dir(parse_id) is None:
        return jsonify({"error": "invalid artifact id"}), 404
    try:
        _load_accessible_manifest(parse_id)
        payload, media_type = ARTIFACT_STORE.read_file(parse_id, "markdown.md", "text/markdown")
    except FileNotFoundError:
        return jsonify({"error": "artifact not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    return Response(payload, mimetype=media_type)


@app.route("/api/v1/artifacts/<parse_id>/chunks", methods=["GET"])
def get_chunk_artifact(parse_id: str):
    if _artifact_parse_dir(parse_id) is None:
        return jsonify({"error": "invalid artifact id"}), 404
    try:
        asset_url_mode = _parse_asset_url_mode()
        expires_in = _parse_asset_url_ttl()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if asset_url_mode != "proxy" or request.args.get("refresh") == "true":
        try:
            _load_accessible_manifest(parse_id)
            return _render_chunk_jsonl(parse_id, asset_url_mode, expires_in)
        except FileNotFoundError:
            return jsonify({"error": "artifact not found"}), 404
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
    try:
        _load_accessible_manifest(parse_id)
        payload, media_type = ARTIFACT_STORE.read_file(parse_id, "chunks.jsonl", "application/x-ndjson")
    except FileNotFoundError:
        return jsonify({"error": "artifact not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    return Response(payload, mimetype=media_type)


@app.route("/api/v1/artifacts/<parse_id>/ingest", methods=["GET"])
def get_ingest_artifact(parse_id: str):
    if _artifact_parse_dir(parse_id) is None:
        return jsonify({"error": "invalid artifact id"}), 404
    try:
        asset_url_mode = _parse_asset_url_mode()
        expires_in = _parse_asset_url_ttl()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if asset_url_mode != "proxy" or request.args.get("refresh") == "true":
        try:
            _load_accessible_manifest(parse_id)
            return _render_ingest_jsonl(parse_id, asset_url_mode, expires_in)
        except FileNotFoundError:
            return jsonify({"error": "artifact not found"}), 404
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 403
    try:
        _load_accessible_manifest(parse_id)
        payload, media_type = ARTIFACT_STORE.read_file(parse_id, "ingest.jsonl", "application/x-ndjson")
    except FileNotFoundError:
        return jsonify({"error": "artifact not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    return Response(payload, mimetype=media_type)


@app.route("/api/v1/artifacts/<parse_id>/assets/<path:filename>", methods=["GET"])
def get_asset_artifact(parse_id: str, filename: str):
    if _artifact_parse_dir(parse_id) is None:
        return jsonify({"error": "invalid artifact id"}), 404
    asset_name = Path(filename).name
    try:
        _load_accessible_manifest(parse_id)
        payload, media_type = ARTIFACT_STORE.read_file(parse_id, f"assets/{asset_name}")
    except FileNotFoundError:
        return jsonify({"error": "asset not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    return Response(payload, mimetype=media_type)


@app.route("/api/v1/ocr", methods=["POST"])
def ocr_endpoint():
    global ocr_engine, layout_engine
    if not ocr_engine or not layout_engine:
        load_models()
    if not ocr_engine or not layout_engine:
        return _jsonify_error(ErrorCode.MODELS_NOT_INITIALIZED, status_code=503)

    if "file" not in request.files:
        return _jsonify_error(ErrorCode.NO_FILE_PART, status_code=400, details={"field": "file"})

    file = request.files["file"]
    validation_error = validate_file(
        file,
        allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        check_image=True,
    )
    if validation_error:
        return _jsonify_error(ErrorCode.VALIDATION_ERROR, status_code=400, message=validation_error)

    filename = file.filename or ""
    if filename == "":
        return _jsonify_error(ErrorCode.NO_SELECTED_FILE, status_code=400)

    try:
        import cv2

        file_bytes = file.read()
        if not file_bytes:
            return _jsonify_error(ErrorCode.EMPTY_IMAGE_FILE, status_code=400)

        np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return _jsonify_error(ErrorCode.INVALID_IMAGE_FILE, status_code=400)

        start = time.time()
        ocr_lines = ocr_engine(img)
        ocr_boxes = _ocr_lines_to_boxes(ocr_lines)
        ocr_res, page_layout = layout_engine([img], [ocr_boxes], scale_factor=1)

        def clean(obj):
            if isinstance(obj, list):
                return [clean(x) for x in obj]
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        return jsonify(
            {
                "cost_seconds": time.time() - start,
                "ocr_result": clean(ocr_res),
                "layout_result": clean(page_layout),
            }
        )
    except Exception as e:
        traceback.print_exc()
        return _jsonify_error(ErrorCode.INTERNAL_ERROR, status_code=500, message=str(e))


@app.route("/api/v1/parse", methods=["POST"])
def parse_endpoint():
    files = request.files.getlist("file")
    if not files:
        return _jsonify_error(ErrorCode.NO_FILE_PART, status_code=400, details={"field": "file"})

    try:
        parse_options = _build_parse_options()
    except ValueError as e:
        return _jsonify_error(ErrorCode.BAD_REQUEST, status_code=400, message=str(e))
    try:
        parse_options = _augment_parse_options_with_request_context(parse_options)
    except PermissionError as exc:
        return _jsonify_error(ErrorCode.FORBIDDEN, status_code=403, message=str(exc))

    results = []
    for file in files:
        result = _parse_single_file(file, parse_options)
        results.append(result)

    _append_ops_audit_event(
        "parse.sync",
        resource_type="parse_request",
        status="failed" if all("error" in result for result in results) else "ok",
        payload={
            "file_count": len(results),
            "filenames": [str(result.get("filename") or "") for result in results],
            "parser_engine": str(parse_options.get("parser_engine") or "deepdoc"),
            "result_count": len(results),
            "error_count": sum(1 for result in results if "error" in result),
            "parse_ids": [str(result.get("parse_id") or "") for result in results if str(result.get("parse_id") or "")],
        },
        metadata={
            "return_structured": bool(parse_options.get("return_structured")),
            "persist_artifacts": bool(parse_options.get("persist_artifacts")),
            "publish_ingest": bool(parse_options.get("publish_ingest")),
        },
        tenant_id=str(parse_options.get("tenant_id") or "").strip() or None,
    )

    errors = [r for r in results if "error" in r]
    if len(errors) == len(results):
        return _jsonify_external({"results": results}, status_code=400)

    if request.args.get("download") == "true" and len(results) == 1:
        safe_filename = quote(f"{results[0].get('filename', 'output')}.md")
        return Response(
            results[0].get("markdown", ""),
            mimetype="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"
            },
        )

    return _jsonify_external({"results": results})


@app.route("/api/v1/parse/stream", methods=["POST"])
def parse_stream_endpoint():
    files = request.files.getlist("file")
    if not files:
        return _jsonify_error(ErrorCode.NO_FILE_PART, status_code=400, details={"field": "file"})

    try:
        parse_options = _build_parse_options()
    except ValueError as e:
        return _jsonify_error(ErrorCode.BAD_REQUEST, status_code=400, message=str(e))
    try:
        parse_options = _augment_parse_options_with_request_context(parse_options)
    except PermissionError as exc:
        return _jsonify_error(ErrorCode.FORBIDDEN, status_code=403, message=str(exc))

    parser_engine_label = str(parse_options.get("parser_engine") or "deepdoc")
    stream_stage_dir = Path(tempfile.mkdtemp(prefix="deepdoc-stream-", dir=str(UPLOAD_TMP_DIR)))
    staged_files: list[LocalUploadedFile] = []
    try:
        for index, file in enumerate(files):
            filename = file.filename or ""
            safe_name = f"{index:02d}-{uuid4().hex[:8]}-{Path(filename or 'upload').name}"
            target_path = stream_stage_dir / safe_name
            file.save(target_path)
            staged_files.append(LocalUploadedFile(target_path, filename))
    except Exception as exc:
        shutil.rmtree(stream_stage_dir, ignore_errors=True)
        logger.exception("Failed to stage stream parse uploads")
        return _jsonify_error(ErrorCode.UPLOAD_SAVE_FAILED, status_code=500, message=f"Failed to save upload: {exc}")
    total_files = len(staged_files)

    def _event_stream():
        results: list[dict[str, object]] = []

        def emit(event_type: str, payload: dict[str, object]) -> str:
            return _sse_event(event_type, _external_response_payload(payload))

        try:
            yield emit("start", {"file_count": total_files, "parser_engine": parser_engine_label})
            for index, file in enumerate(staged_files, start=1):
                filename = file.filename or ""
                yield emit("file_started", {"index": index, "total": total_files, "filename": filename})
                try:
                    result = _parse_single_file(file, parse_options)
                except Exception as exc:
                    logger.exception("Stream parse failed for file=%s", filename)
                    result = {"filename": filename, "error": str(exc), "parser_engine": parser_engine_label}
                results.append(result)
                has_error = "error" in result
                yield emit(
                    "file_completed",
                    {
                        "index": index,
                        "total": total_files,
                        "filename": filename,
                        "has_error": has_error,
                        "parse_id": str(result.get("parse_id") or "").strip() or None,
                        "result": result,
                    },
                )

            error_count = sum(1 for result in results if "error" in result)
            status = "failed" if error_count == len(results) else "ok"
            parse_ids = [
                str(result.get("parse_id") or "")
                for result in results
                if str(result.get("parse_id") or "").strip()
            ]
            _append_ops_audit_event(
                "parse.stream",
                resource_type="parse_request",
                status=status,
                payload={
                    "file_count": total_files,
                    "filenames": [str(result.get("filename") or "") for result in results],
                    "parser_engine": parser_engine_label,
                    "result_count": len(results),
                    "error_count": error_count,
                    "parse_ids": parse_ids,
                },
                metadata={
                    "return_structured": bool(parse_options.get("return_structured")),
                    "persist_artifacts": bool(parse_options.get("persist_artifacts")),
                    "publish_ingest": bool(parse_options.get("publish_ingest")),
                },
                tenant_id=str(parse_options.get("tenant_id") or "").strip() or None,
            )
            yield emit(
                "done",
                {
                    "status": status,
                    "file_count": total_files,
                    "result_count": len(results),
                    "error_count": error_count,
                    "results": results,
                },
            )
        finally:
            for staged_file in staged_files:
                staged_file.close()
            shutil.rmtree(stream_stage_dir, ignore_errors=True)

    response = Response(stream_with_context(_event_stream()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/v1/parse/async", methods=["POST"])
def parse_async_endpoint():
    if getattr(ASYNC_TASK_BROKER, "backend_name", "none") == "none":
        return _jsonify_error(ErrorCode.ASYNC_PARSE_DISABLED, status_code=503)
    files = request.files.getlist("file")
    if not files:
        return _jsonify_error(ErrorCode.NO_FILE_PART, status_code=400, details={"field": "file"})

    try:
        parse_options = _build_parse_options()
    except ValueError as e:
        return _jsonify_error(ErrorCode.BAD_REQUEST, status_code=400, message=str(e))
    try:
        parse_options = _augment_parse_options_with_request_context(parse_options)
    except PermissionError as exc:
        return _jsonify_error(ErrorCode.FORBIDDEN, status_code=403, message=str(exc))

    # Async tasks always persist artifacts so status and artifact APIs can resolve stable parse outputs.
    parse_options["persist_artifacts"] = True
    parse_options["publish_requested_by"] = "async_task"
    try:
        callback_config = _build_async_task_callback_config()
    except ValueError as exc:
        return _jsonify_error(ErrorCode.BAD_REQUEST, status_code=400, message=str(exc))

    try:
        temp_task_id, staged_inputs = _stage_async_uploads(files)
        task = build_async_task(
            queue_name=_async_queue_name(),
            parser_engine=str(parse_options.get("parser_engine") or "deepdoc"),
            parse_options=parse_options,
            input_files=staged_inputs,
            tenant_id=str(parse_options.get("tenant_id") or "").strip() or None,
            requested_by="api",
            auth_subject=str(parse_options.get("auth_subject") or "").strip() or None,
            callback=callback_config,
        )
        promoted_inputs = _promote_async_staging_task(temp_task_id, task)
        task = task.model_copy(update={"input_files": promoted_inputs})
        ASYNC_TASK_STORE.create_task(task)
        ASYNC_TASK_BROKER.enqueue(task.task_id)
    except ValueError as exc:
        return _jsonify_error(ErrorCode.BAD_REQUEST, status_code=400, message=str(exc))
    except Exception as exc:
        logger.exception("Failed to enqueue async parse task")
        return _jsonify_error(ErrorCode.INTERNAL_ERROR, status_code=500, message=str(exc))

    _append_ops_audit_event(
        "parse.async.submit",
        resource_type="async_task",
        resource_id=task.task_id,
        status="queued",
        payload={
            "file_count": len(task.input_files),
            "filenames": [input_file.filename for input_file in task.input_files],
            "parser_engine": task.parser_engine,
            "queue_name": task.queue_name,
        },
        metadata={
            "persist_artifacts": bool(parse_options.get("persist_artifacts")),
            "publish_ingest": bool(parse_options.get("publish_ingest")),
            "callback_configured": callback_config is not None,
            "callback_url": callback_config.url if callback_config is not None else None,
        },
        tenant_id=task.tenant_id,
        auth_context={
            "subject": task.auth_subject,
            "tenant_id": task.tenant_id,
            "mode": str(_current_auth_context().get("mode") or ""),
            "is_admin": bool(_current_auth_context().get("is_admin")),
            "scopes": list(_current_auth_context().get("scopes") or []),
        },
    )

    return _jsonify_external(task_access_payload(task, include_result=False), status_code=202)


@app.route("/api/v1/tasks", methods=["GET"])
def list_async_tasks():
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = (request.args.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    limit = max(1, min(int(request.args.get("limit", "20") or "20"), 200))
    status_filter = (request.args.get("status") or "").strip().lower() or None
    callback_statuses = _normalize_async_callback_status_values(request.args.get("callback_status"))
    callback_configured_raw = request.args.get("callback_configured")
    callback_configured = _parse_bool(callback_configured_raw, default=False) if callback_configured_raw is not None else None
    tasks = ASYNC_TASK_STORE.list_tasks(limit=limit, tenant_id=tenant_filter)
    if status_filter:
        tasks = [task for task in tasks if task.status == status_filter]
    if callback_statuses or callback_configured is not None:
        tasks = [
            task
            for task in tasks
            if _task_matches_callback_filters(
                task,
                callback_statuses=callback_statuses,
                callback_configured=callback_configured,
            )
        ]
    return _jsonify_external({"tasks": [task_access_payload(task, include_result=False) for task in tasks]})


@app.route("/api/v1/tasks/cleanup", methods=["POST"])
def cleanup_async_tasks():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object"}), 400
    try:
        limit = max(1, min(int(payload.get("limit", 1000)), 5000))
        keep_latest = payload.get("keep_latest")
        if keep_latest is not None:
            keep_latest = max(0, int(keep_latest))
        older_than_days = payload.get("older_than_days")
        if older_than_days is not None:
            older_than_days = max(0, int(older_than_days))
        include_active = _parse_bool(payload.get("include_active"), default=False)
        dry_run = _parse_bool(payload.get("dry_run"), default=True)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid cleanup payload"}), 400
    status_values: list[str] = []
    raw_status = payload.get("status")
    raw_statuses = payload.get("statuses")
    if raw_statuses is not None:
        if isinstance(raw_statuses, str):
            status_values.extend(part.strip().lower() for part in raw_statuses.split(",") if part.strip())
        elif isinstance(raw_statuses, list):
            status_values.extend(str(part).strip().lower() for part in raw_statuses if str(part).strip())
        else:
            return jsonify({"error": "statuses must be a string or list"}), 400
    if raw_status is not None:
        status_value = str(raw_status).strip().lower()
        if status_value:
            status_values.append(status_value)
    valid_statuses = {"queued", "running", "succeeded", "failed", "cancel_requested", "cancelled"}
    status_values = sorted({value for value in status_values if value})
    if any(value not in valid_statuses for value in status_values):
        return jsonify({"error": "invalid task status filter"}), 400
    if older_than_days is None and (keep_latest is None or keep_latest <= 0) and not status_values:
        return jsonify({"error": "older_than_days, keep_latest, or status filter is required"}), 400
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(payload.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    try:
        result = ASYNC_TASK_STORE.cleanup_tasks(
            limit=limit,
            tenant_id=tenant_filter,
            older_than_days=older_than_days,
            keep_latest=keep_latest,
            statuses=status_values,
            include_active=include_active,
            dry_run=dry_run,
        )
    except Exception as exc:
        logger.exception("Failed to cleanup async tasks")
        return jsonify({"error": str(exc)}), 500
    _append_ops_audit_event(
        "tasks.cleanup",
        resource_type="async_task",
        status="ok",
        payload=result,
        metadata={
            "include_active": include_active,
            "statuses": status_values,
        },
        tenant_id=tenant_filter,
    )
    return _jsonify_external(result)


@app.route("/api/v1/tasks/<task_id>", methods=["GET"])
def get_async_task(task_id: str):
    include_result = _parse_bool(request.args.get("include_result"), default=True)
    try:
        task = _load_accessible_async_task(task_id)
    except FileNotFoundError:
        return jsonify({"error": "task not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    result = ASYNC_TASK_STORE.load_result(task_id) if include_result else None
    return _jsonify_external(task_access_payload(task, result=result, include_result=include_result))


@app.route("/api/v1/tasks/<task_id>/events", methods=["GET"])
def get_async_task_events(task_id: str):
    try:
        _load_accessible_async_task(task_id)
    except FileNotFoundError:
        return jsonify({"error": "task not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    events = ASYNC_TASK_STORE.read_events(task_id)
    return _jsonify_external({"task_id": task_id, "events": [event.model_dump(mode="json") for event in events]})


@app.route("/api/v1/tasks/<task_id>/stream", methods=["GET"])
def stream_async_task_events(task_id: str):
    include_result = _parse_bool(request.args.get("include_result"), default=True)
    include_callback_events = _parse_bool(request.args.get("include_callback_events"), default=True)
    replay_existing = _parse_bool(request.args.get("replay_existing"), default=True)
    poll_seconds = max(1, min(int(request.args.get("poll_seconds", "1") or "1"), 30))
    heartbeat_seconds = max(2, min(int(request.args.get("heartbeat_seconds", "10") or "10"), 120))
    timeout_seconds = max(10, min(int(request.args.get("timeout_seconds", "300") or "300"), 3600))
    try:
        _load_accessible_async_task(task_id)
    except FileNotFoundError:
        return jsonify({"error": "task not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403

    def _event_stream():
        seen_task_event_ids: set[str] = set()
        seen_callback_event_ids: set[str] = set()
        deadline = time.monotonic() + timeout_seconds
        last_heartbeat_at = 0.0

        def emit(event_type: str, payload: dict[str, object]) -> str:
            return _sse_event(event_type, _external_response_payload(payload))

        if replay_existing:
            snapshot_task = _load_accessible_async_task(task_id)
            snapshot_result = ASYNC_TASK_STORE.load_result(task_id) if include_result else None
            yield emit(
                "task_snapshot",
                {"task": task_access_payload(snapshot_task, result=snapshot_result, include_result=include_result)},
            )
            for event in ASYNC_TASK_STORE.read_events(task_id):
                seen_task_event_ids.add(event.event_id)
                yield emit("task_event", event.model_dump(mode="json"))
            if include_callback_events:
                for callback_event in ASYNC_TASK_STORE.read_callback_events(task_id):
                    seen_callback_event_ids.add(callback_event.callback_event_id)
                    yield emit("task_callback_event", callback_event.model_dump(mode="json"))

        while True:
            current_task = _load_accessible_async_task(task_id)
            current_result = ASYNC_TASK_STORE.load_result(task_id) if include_result else None
            for event in ASYNC_TASK_STORE.read_events(task_id):
                if event.event_id in seen_task_event_ids:
                    continue
                seen_task_event_ids.add(event.event_id)
                yield emit("task_event", event.model_dump(mode="json"))
            if include_callback_events:
                for callback_event in ASYNC_TASK_STORE.read_callback_events(task_id):
                    if callback_event.callback_event_id in seen_callback_event_ids:
                        continue
                    seen_callback_event_ids.add(callback_event.callback_event_id)
                    yield emit("task_callback_event", callback_event.model_dump(mode="json"))

            now_monotonic = time.monotonic()
            if now_monotonic - last_heartbeat_at >= heartbeat_seconds:
                yield emit(
                    "heartbeat",
                    {
                        "task_id": task_id,
                        "status": current_task.status,
                        "callback_status": (
                            str((current_task.callback_state.status if current_task.callback_state is not None else "none") or "none")
                        ),
                    },
                )
                last_heartbeat_at = now_monotonic

            callback_status = str((current_task.callback_state.status if current_task.callback_state is not None else "none") or "none")
            callback_pending = callback_status in {"pending", "delivering"}
            if current_task.status in {"succeeded", "failed", "cancelled"} and not callback_pending:
                yield emit(
                    "task_final",
                    {"task": task_access_payload(current_task, result=current_result, include_result=include_result)},
                )
                yield emit(
                    "done",
                    {
                        "task_id": task_id,
                        "status": current_task.status,
                        "callback_status": callback_status,
                    },
                )
                return

            if now_monotonic >= deadline:
                yield emit(
                    "timeout",
                    {
                        "task_id": task_id,
                        "status": current_task.status,
                        "callback_status": callback_status,
                    },
                )
                return
            time.sleep(poll_seconds)

    response = Response(stream_with_context(_event_stream()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/v1/tasks/<task_id>/callback-events", methods=["GET"])
def get_async_task_callback_events(task_id: str):
    try:
        task = _load_accessible_async_task(task_id)
    except FileNotFoundError:
        return jsonify({"error": "task not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    if task.callback is None:
        return jsonify({"error": "task callback is not configured"}), 404
    events = ASYNC_TASK_STORE.read_callback_events(task_id)
    return _jsonify_external({"task_id": task_id, "events": [event.model_dump(mode="json") for event in events]})


@app.route("/api/v1/tasks/retry", methods=["POST"])
def retry_async_tasks_batch():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object"}), 400
    try:
        limit = max(1, min(int(payload.get("limit", 50)), 500))
        scan_limit = max(limit, min(int(payload.get("scan_limit", max(limit * 5, 200))), 5000))
        dry_run = _parse_bool(payload.get("dry_run"), default=True)
        force = _parse_bool(payload.get("force"), default=False)
        copy_callback = _parse_bool(payload.get("copy_callback"), default=True)
        requested_by = str(payload.get("requested_by") or "api-batch-retry").strip() or "api-batch-retry"
    except (TypeError, ValueError):
        return jsonify({"error": "invalid retry payload"}), 400
    task_status_values = _normalize_async_task_status_values(payload.get("task_statuses"))
    if payload.get("task_status") is not None:
        task_status_values = sorted(set(task_status_values).union(_normalize_async_task_status_values(payload.get("task_status"))))
    if not task_status_values:
        task_status_values = ["failed", "cancelled"]
    if "succeeded" in task_status_values and not force:
        return jsonify({"error": "retrying succeeded tasks requires force=true"}), 400
    parser_engine = str(payload.get("parser_engine") or "").strip().lower() or None
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(payload.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    response_payload = run_async_task_retry_sweep(
        limit=limit,
        scan_limit=scan_limit,
        task_statuses=task_status_values,
        force=force,
        requested_by=requested_by,
        dry_run=dry_run,
        tenant_filter=tenant_filter,
        copy_callback=copy_callback,
        parser_engine=parser_engine,
        audit_action="tasks.retry",
        emit_audit_when_empty=True,
        metrics_source="api",
    )
    return jsonify(response_payload)


@app.route("/api/v1/tasks/callbacks/retry", methods=["POST"])
def retry_async_task_callbacks_batch():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object"}), 400
    try:
        limit = max(1, min(int(payload.get("limit", 50)), 500))
        scan_limit = max(limit, min(int(payload.get("scan_limit", max(limit * 5, 200))), 5000))
        dry_run = _parse_bool(payload.get("dry_run"), default=True)
        force = _parse_bool(payload.get("force"), default=True)
        requested_by = str(payload.get("requested_by") or "api-batch-retry").strip() or "api-batch-retry"
    except (TypeError, ValueError):
        return jsonify({"error": "invalid retry payload"}), 400
    callback_statuses = _normalize_async_callback_status_values(payload.get("callback_statuses") or payload.get("callback_status"))
    if not callback_statuses:
        callback_statuses = ["dead_lettered", "failed"]
    task_status_values = _normalize_async_task_status_values(payload.get("task_statuses"))
    if payload.get("task_status") is not None:
        task_status_values = sorted(set(task_status_values).union(_normalize_async_task_status_values(payload.get("task_status"))))
    if payload.get("task_status") is not None and not task_status_values and str(payload.get("task_status")).strip():
        return jsonify({"error": "invalid task status filter"}), 400
    try:
        effective_tenant_id = _resolve_effective_tenant_id()
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    auth_context = _current_auth_context()
    requested_tenant = str(payload.get("tenant_id") or "").strip() or None
    if requested_tenant and not bool(auth_context.get("is_admin")) and requested_tenant != effective_tenant_id:
        return jsonify({"error": "tenant_id override is not allowed"}), 403
    tenant_filter = requested_tenant if bool(auth_context.get("is_admin")) else effective_tenant_id
    response_payload = run_async_task_callback_redrive_sweep(
        limit=limit,
        scan_limit=scan_limit,
        callback_statuses=callback_statuses,
        task_statuses=task_status_values,
        force=force,
        requested_by=requested_by,
        dry_run=dry_run,
        tenant_filter=tenant_filter,
        audit_action="tasks.callbacks.retry",
        emit_audit_when_empty=True,
        metrics_source="api",
    )
    return jsonify(response_payload)


@app.route("/api/v1/tasks/<task_id>/retry", methods=["POST"])
def retry_async_task(task_id: str):
    try:
        task = _load_accessible_async_task(task_id)
    except FileNotFoundError:
        observe_async_task_retry_run(source="api_single", status="not_found", candidate_count=1, enqueued_count=0)
        return jsonify({"error": "task not found"}), 404
    except PermissionError as exc:
        observe_async_task_retry_run(source="api_single", status="forbidden", candidate_count=1, enqueued_count=0)
        return jsonify({"error": str(exc)}), 403
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        observe_async_task_retry_run(source="api_single", status="bad_request", candidate_count=1, enqueued_count=0)
        return jsonify({"error": "JSON body must be an object"}), 400
    force = _parse_bool(payload.get("force"), default=False)
    copy_callback = _parse_bool(payload.get("copy_callback"), default=True)
    requested_by = str(payload.get("requested_by") or "api-retry").strip() or "api-retry"
    validation_error = _validate_async_task_retry_source(task, force=force)
    if validation_error:
        observe_async_task_retry_run(source="api_single", status="conflict", candidate_count=1, enqueued_count=0)
        return jsonify({"error": validation_error}), 409
    try:
        retry_task, updated_source_task = _enqueue_async_task_retry(
            task,
            requested_by=requested_by,
            auth_subject=str(_current_auth_context().get("subject") or "").strip() or None,
            force=force,
            copy_callback=copy_callback,
        )
    except FileNotFoundError as exc:
        observe_async_task_retry_run(source="api_single", status="source_missing", candidate_count=1, enqueued_count=0)
        return jsonify({"error": str(exc)}), 409
    except ValueError as exc:
        observe_async_task_retry_run(source="api_single", status="conflict", candidate_count=1, enqueued_count=0)
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:
        logger.exception("Failed to retry async task task_id=%s", task_id)
        observe_async_task_retry_run(source="api_single", status="failed", candidate_count=1, enqueued_count=0)
        return jsonify({"error": str(exc)}), 500
    observe_async_task_retry_run(source="api_single", status="ok", candidate_count=1, enqueued_count=1)
    _append_ops_audit_event(
        "task.retry",
        resource_type="async_task",
        resource_id=task_id,
        status="queued",
        payload={
            "source_task_id": task_id,
            "retry_task_id": retry_task.task_id,
            "source_status": task.status,
            "copy_callback": copy_callback,
        },
        metadata={
            "requested_by": requested_by,
            "force": force,
        },
        tenant_id=retry_task.tenant_id,
    )
    return _jsonify_external(
        {
            "source_task": task_access_payload(updated_source_task, result=ASYNC_TASK_STORE.load_result(task_id), include_result=True),
            "retry_task": task_access_payload(retry_task, include_result=False),
        },
        status_code=202,
    )


@app.route("/api/v1/tasks/<task_id>/callback/retry", methods=["POST"])
def retry_async_task_callback(task_id: str):
    try:
        task = _load_accessible_async_task(task_id)
    except FileNotFoundError:
        return jsonify({"error": "task not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    if task.callback is None:
        return jsonify({"error": "task callback is not configured"}), 404
    if task.status not in {"succeeded", "failed", "cancelled"}:
        return jsonify({"error": "task callback can only be retried after terminal completion"}), 409
    payload = request.get_json(silent=True) or {}
    force = _parse_bool(payload.get("force"), default=True)
    requested_by = str(payload.get("requested_by") or "api-retry").strip() or "api-retry"
    result = ASYNC_TASK_STORE.load_result(task_id)
    try:
        delivery_result = _deliver_terminal_async_task_callback(task_id, requested_by=requested_by, force=force)
    except Exception as exc:
        logger.exception("Failed to retry async task callback task_id=%s", task_id)
        return jsonify({"error": str(exc)}), 500
    refreshed_task = ASYNC_TASK_STORE.load_task(task_id)
    _append_ops_audit_event(
        "task.callback.retry",
        resource_type="async_task",
        resource_id=task_id,
        status=str((delivery_result or {}).get("status") or "skipped"),
        payload=delivery_result or {"configured": False},
        metadata={"requested_by": requested_by, "force": force},
        tenant_id=refreshed_task.tenant_id,
    )
    return _jsonify_external(
        {
            "task": task_access_payload(refreshed_task, result=result, include_result=True),
            "delivery": delivery_result,
        }
    )


@app.route("/api/v1/tasks/<task_id>/cancel", methods=["POST"])
def cancel_async_task(task_id: str):
    try:
        task = _load_accessible_async_task(task_id)
    except FileNotFoundError:
        return jsonify({"error": "task not found"}), 404
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403

    if task.status in {"succeeded", "failed", "cancelled"}:
        result = ASYNC_TASK_STORE.load_result(task_id)
        return _jsonify_external(task_access_payload(task, result=result, include_result=True))

    updated = task.model_copy(
        update={
            "status": "cancel_requested",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    ASYNC_TASK_STORE.write_task(updated)
    ASYNC_TASK_STORE.append_event(
        task_id,
        "cancel_requested",
        {"requested_by": str(_current_auth_context().get("subject") or "api")},
    )
    _append_ops_audit_event(
        "task.cancel",
        resource_type="async_task",
        resource_id=task_id,
        status="cancel_requested",
        payload={"previous_status": task.status},
        tenant_id=task.tenant_id,
    )
    return _jsonify_external(task_access_payload(updated, include_result=False))


def _infer_file_type(filename: str) -> str | None:
    """从文件名推断解析器类型。"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext if ext in PARSER_IMPORTS or ext in IMAGE_FILE_TYPES else None


def _parse_datetime_safe(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_async_callback_status_values(values: object) -> list[str]:
    normalized: set[str] = set()
    if values is None:
        return []
    items: list[str] = []
    if isinstance(values, list):
        items = [str(item).strip().lower() for item in values if str(item).strip()]
    else:
        items = [part.strip().lower() for part in str(values).split(",") if part.strip()]
    valid = {"pending", "delivering", "delivered", "failed", "dead_lettered", "disabled", "none"}
    for item in items:
        if item in valid:
            normalized.add(item)
    return sorted(normalized)


def _task_matches_callback_filters(
    task: AsyncTask,
    *,
    callback_statuses: list[str] | None = None,
    callback_configured: bool | None = None,
) -> bool:
    callback = task.callback
    callback_state = task.callback_state
    configured = callback is not None
    if callback_configured is not None and configured != bool(callback_configured):
        return False
    normalized_statuses = set(callback_statuses or [])
    if not normalized_statuses:
        return True
    if not configured:
        return "none" in normalized_statuses
    current_status = str(getattr(callback_state, "status", "") or "pending").strip().lower() or "pending"
    return current_status in normalized_statuses


def _normalize_async_task_status_values(values: object) -> list[str]:
    normalized: set[str] = set()
    if values is None:
        return []
    if isinstance(values, list):
        items = [str(item).strip().lower() for item in values if str(item).strip()]
    else:
        items = [part.strip().lower() for part in str(values).split(",") if part.strip()]
    valid = {"queued", "running", "succeeded", "failed", "cancel_requested", "cancelled"}
    for item in items:
        if item in valid:
            normalized.add(item)
    return sorted(normalized)


def _async_task_retry_cooldown_seconds() -> int:
    return max(0, int(os.environ.get("DEEPDOC_ASYNC_RETRY_COOLDOWN_SECONDS", "60") or "60"))


def _validate_async_task_retry_source(task: AsyncTask, *, force: bool, cooldown_seconds: int | None = None) -> str | None:
    if task.status not in {"succeeded", "failed", "cancelled"}:
        return "task is not in a terminal state"
    if task.status == "succeeded" and not force:
        return "successful tasks require force=true to retry"
    if not task.input_files:
        return "task has no source files to retry"
    for input_file in task.input_files:
        source_path = Path(input_file.source_path)
        if not source_path.exists():
            return f"task source file missing: {source_path}"
    if force:
        return None
    latest_retry_task_id = str((task.metadata or {}).get("latest_retry_task_id") or "").strip()
    if latest_retry_task_id:
        try:
            latest_retry_task = ASYNC_TASK_STORE.load_task(latest_retry_task_id)
        except FileNotFoundError:
            latest_retry_task = None
        except Exception:
            logger.exception("Failed to load latest retry task task_id=%s latest_retry_task_id=%s", task.task_id, latest_retry_task_id)
            latest_retry_task = None
        if latest_retry_task is not None and latest_retry_task.status in {"queued", "running", "cancel_requested"}:
            return f"latest retry task is still active: {latest_retry_task_id}"
    cooldown_seconds = _async_task_retry_cooldown_seconds() if cooldown_seconds is None else max(0, int(cooldown_seconds))
    if cooldown_seconds <= 0:
        return None
    last_retry_at = _parse_datetime_safe((task.metadata or {}).get("last_retry_enqueued_at"))
    if last_retry_at is None:
        return None
    elapsed = (datetime.now(timezone.utc) - last_retry_at).total_seconds()
    if elapsed < cooldown_seconds:
        return f"retry cooldown active ({int(max(0, cooldown_seconds - elapsed))}s remaining)"
    return None


def _enqueue_async_task_retry(
    task: AsyncTask,
    *,
    requested_by: str,
    auth_subject: str | None,
    force: bool,
    copy_callback: bool,
) -> tuple[AsyncTask, AsyncTask]:
    validation_error = _validate_async_task_retry_source(task, force=force)
    if validation_error:
        raise ValueError(validation_error)
    callback_config = task.callback.model_copy(deep=True) if copy_callback and task.callback is not None else None
    retry_task = build_async_retry_task(
        task,
        queue_name=_async_queue_name(),
        requested_by=requested_by,
        auth_subject=auth_subject,
        callback=callback_config,
    )
    cloned_inputs = ASYNC_TASK_STORE.clone_inputs_for_retry(task, target_task_id=retry_task.task_id)
    retry_metadata = dict(retry_task.metadata or {})
    retry_metadata["file_count"] = len(cloned_inputs)
    retry_task = retry_task.model_copy(update={"input_files": cloned_inputs, "metadata": retry_metadata})
    ASYNC_TASK_STORE.create_task(retry_task)
    ASYNC_TASK_STORE.append_event(
        retry_task.task_id,
        "retried_from",
        {
            "source_task_id": task.task_id,
            "source_status": task.status,
            "requested_by": requested_by,
            "copy_callback": bool(copy_callback and task.callback is not None),
        },
    )
    source_metadata = dict(task.metadata or {})
    source_metadata.update(
        {
            "latest_retry_task_id": retry_task.task_id,
            "last_retry_enqueued_at": datetime.now(timezone.utc).isoformat(),
            "last_retry_requested_by": requested_by,
        }
    )
    source_task = task.model_copy(update={"updated_at": datetime.now(timezone.utc).isoformat(), "metadata": source_metadata})
    ASYNC_TASK_STORE.write_task(source_task)
    ASYNC_TASK_STORE.append_event(
        task.task_id,
        "retry_enqueued",
        {
            "retry_task_id": retry_task.task_id,
            "requested_by": requested_by,
            "copy_callback": bool(copy_callback and task.callback is not None),
            "force": force,
        },
    )
    ASYNC_TASK_BROKER.enqueue(retry_task.task_id)
    return retry_task, source_task


def _callback_redrive_cooldown_seconds() -> int:
    return max(0, int(os.environ.get("DEEPDOC_ASYNC_CALLBACK_REDRIVE_COOLDOWN_SECONDS", "60") or "60"))


def _eligible_for_callback_redrive(task: AsyncTask, *, cooldown_seconds: int) -> bool:
    callback = task.callback
    callback_state = task.callback_state
    if callback is None:
        return False
    if task.status not in {"succeeded", "failed", "cancelled"}:
        return False
    if callback_state is None:
        return True
    current_status = str(callback_state.status or "").strip().lower() or "pending"
    if current_status in {"pending", "delivering"}:
        return False
    next_retry_at = _parse_datetime_safe(callback_state.next_retry_at)
    now = datetime.now(timezone.utc)
    if next_retry_at is not None and next_retry_at > now:
        return False
    if cooldown_seconds <= 0:
        return True
    reference_time = _parse_datetime_safe(callback_state.last_attempt_at) or _parse_datetime_safe(task.updated_at)
    if reference_time is None:
        return True
    return (now - reference_time).total_seconds() >= cooldown_seconds


def run_async_task_callback_redrive_sweep(
    *,
    limit: int,
    scan_limit: int,
    callback_statuses: list[str],
    task_statuses: list[str],
    force: bool,
    requested_by: str,
    dry_run: bool,
    tenant_filter: str | None = None,
    cooldown_seconds: int | None = None,
    audit_action: str = "tasks.callbacks.retry",
    emit_audit_when_empty: bool = True,
    metrics_source: str = "api",
) -> dict[str, object]:
    cooldown_seconds = _callback_redrive_cooldown_seconds() if cooldown_seconds is None else max(0, int(cooldown_seconds))
    tasks = ASYNC_TASK_STORE.list_tasks(limit=scan_limit, tenant_id=tenant_filter)
    candidates: list[AsyncTask] = []
    for task in tasks:
        if task_statuses and task.status not in task_statuses:
            continue
        if not _task_matches_callback_filters(task, callback_statuses=callback_statuses):
            continue
        if not _eligible_for_callback_redrive(task, cooldown_seconds=cooldown_seconds):
            continue
        candidates.append(task)
        if len(candidates) >= limit:
            break

    results: list[dict[str, object]] = []
    delivered = 0
    failed = 0
    skipped = 0
    for task in candidates:
        if dry_run:
            results.append(
                {
                    "task_id": task.task_id,
                    "status": task.status,
                    "callback_status": str((task.callback_state.status if task.callback_state is not None else "none") or "none"),
                    "callback_url": task.callback.url if task.callback is not None else None,
                }
            )
            continue
        delivery = _deliver_terminal_async_task_callback(task.task_id, requested_by=requested_by, force=force)
        if delivery is None:
            skipped += 1
            results.append({"task_id": task.task_id, "status": task.status, "delivery": None, "skipped": True})
            continue
        normalized_status = str(delivery.get("status") or "").strip().lower()
        if normalized_status == "delivered":
            delivered += 1
        elif delivery.get("skipped"):
            skipped += 1
        else:
            failed += 1
        refreshed_task = ASYNC_TASK_STORE.load_task(task.task_id)
        results.append(
            {
                "task_id": task.task_id,
                "status": task.status,
                "delivery": delivery,
                "callback_state": (
                    refreshed_task.callback_state.model_dump(mode="json")
                    if refreshed_task.callback_state is not None
                    else None
                ),
            }
        )

    outcome_status = "noop"
    if candidates:
        if failed and delivered:
            outcome_status = "partial_failure"
        elif failed:
            outcome_status = "failed"
        else:
            outcome_status = "ok"
    observe_async_task_callback_redrive_run(
        source=metrics_source,
        status=outcome_status,
        candidate_count=len(candidates),
        delivered_count=delivered,
    )
    payload: dict[str, object] = {
        "run_id": uuid4().hex,
        "scanned": len(tasks),
        "candidate_count": len(candidates),
        "delivered": delivered,
        "failed": failed,
        "skipped": skipped,
        "dry_run": dry_run,
        "results": results,
        "cooldown_seconds": cooldown_seconds,
        "status": outcome_status,
        "metrics_source": metrics_source,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if emit_audit_when_empty or len(candidates) > 0 or failed > 0 or delivered > 0:
        _append_ops_audit_event(
            audit_action,
            resource_type="async_task",
            status="ok" if failed == 0 else "partial_failure",
            payload=payload,
            metadata={
                "callback_statuses": callback_statuses,
                "task_statuses": task_statuses,
                "force": force,
                "requested_by": requested_by,
                "cooldown_seconds": cooldown_seconds,
            },
            tenant_id=tenant_filter,
        )
    return payload


def run_async_task_retry_sweep(
    *,
    limit: int,
    scan_limit: int,
    task_statuses: list[str],
    force: bool,
    requested_by: str,
    dry_run: bool,
    tenant_filter: str | None = None,
    copy_callback: bool = True,
    parser_engine: str | None = None,
    cooldown_seconds: int | None = None,
    audit_action: str = "tasks.retry",
    emit_audit_when_empty: bool = True,
    metrics_source: str = "api",
) -> dict[str, object]:
    normalized_parser_engine = str(parser_engine or "").strip().lower() or None
    cooldown_seconds = _async_task_retry_cooldown_seconds() if cooldown_seconds is None else max(0, int(cooldown_seconds))
    tasks = ASYNC_TASK_STORE.list_tasks(limit=scan_limit, tenant_id=tenant_filter)
    candidates: list[tuple[AsyncTask, str | None]] = []
    for task in tasks:
        if task_statuses and task.status not in task_statuses:
            continue
        if normalized_parser_engine and str(task.parser_engine or "").strip().lower() != normalized_parser_engine:
            continue
        validation_error = _validate_async_task_retry_source(task, force=force, cooldown_seconds=cooldown_seconds)
        if validation_error:
            continue
        candidates.append((task, None))
        if len(candidates) >= limit:
            break

    results: list[dict[str, object]] = []
    enqueued = 0
    failed = 0
    skipped = 0
    auth_subject = str((_current_auth_context().get("subject") or "")).strip() or None
    for task, _ in candidates:
        if dry_run:
            results.append(
                {
                    "task_id": task.task_id,
                    "status": task.status,
                    "parser_engine": task.parser_engine,
                    "callback_configured": task.callback is not None,
                    "retry_attempt": int((task.metadata or {}).get("retry_attempt") or 0) + 1,
                }
            )
            continue
        try:
            retry_task, _updated_source = _enqueue_async_task_retry(
                task,
                requested_by=requested_by,
                auth_subject=auth_subject,
                force=force,
                copy_callback=copy_callback,
            )
            enqueued += 1
            results.append(
                {
                    "task_id": task.task_id,
                    "status": task.status,
                    "retry_task_id": retry_task.task_id,
                    "retry_attempt": int((retry_task.metadata or {}).get("retry_attempt") or 0),
                    "callback_configured": retry_task.callback is not None,
                }
            )
        except Exception as exc:
            failed += 1
            logger.exception("Failed to retry async task task_id=%s", task.task_id)
            results.append(
                {
                    "task_id": task.task_id,
                    "status": task.status,
                    "error": str(exc),
                }
            )

    outcome_status = "noop"
    if candidates:
        if failed and enqueued:
            outcome_status = "partial_failure"
        elif failed:
            outcome_status = "failed"
        else:
            outcome_status = "ok"
    observe_async_task_retry_run(
        source=metrics_source,
        status=outcome_status,
        candidate_count=len(candidates),
        enqueued_count=enqueued,
    )
    payload: dict[str, object] = {
        "scanned": len(tasks),
        "candidate_count": len(candidates),
        "enqueued": enqueued,
        "failed": failed,
        "skipped": skipped,
        "dry_run": dry_run,
        "results": results,
        "copy_callback": copy_callback,
        "cooldown_seconds": cooldown_seconds,
        "parser_engine": normalized_parser_engine,
    }
    if emit_audit_when_empty or len(candidates) > 0 or failed > 0 or enqueued > 0:
        _append_ops_audit_event(
            audit_action,
            resource_type="async_task",
            status="ok" if failed == 0 else "partial_failure",
            payload=payload,
            metadata={
                "task_statuses": task_statuses,
                "force": force,
                "requested_by": requested_by,
                "copy_callback": copy_callback,
                "cooldown_seconds": cooldown_seconds,
                "parser_engine": normalized_parser_engine,
            },
            tenant_id=tenant_filter,
        )
    return payload


def _parse_pdf_from_tmp(parser_cls, tmp_path: str, parse_options: dict[str, object]):
    parser_mode = parse_options.get("parser_engine", "deepdoc")
    need_structured = bool(parse_options.get("return_structured", False)) or bool(
        parse_options.get("persist_artifacts", False)
    )

    if parser_mode == "plain":
        parser = parser_cls()
        return parser(tmp_path)
    if parser_mode == MARKITDOWN_ENGINE:
        parser = parser_cls()
        return parser(tmp_path)
    if parser_mode == PADDLEOCR_VL_ENGINE:
        parser = parser_cls(request_timeout=_get_request_timeout())
        result = parser.parse_pdf(
            filepath=tmp_path,
            parse_method="raw",
            api_url=parse_options.get("paddle_api_url"),
            algorithm_config=parse_options.get("paddle_algorithm_config"),
            prettify_markdown=bool(parse_options.get("paddle_prettify_markdown", True)),
            show_formula_number=bool(parse_options.get("paddle_show_formula_number", False)),
            request_timeout=_get_request_timeout(),
        )
        if (
            need_structured
            and isinstance(result, tuple)
            and len(result) >= 3
            and isinstance(result[2], dict)
        ):
            result[2]["structured_source"] = {
                "engine": "paddleocr_vl",
                "parser": parser,
                "raw_result": result[2].get("raw_result"),
            }
        return result
    if parser_mode == "mineru":
        mineru_api = parse_options.get("mineru_api", "")
        mineru_server_url = parse_options.get("mineru_server_url", "")
        mineru_backend = parse_options.get("mineru_backend", "pipeline")
        cleanup_output = _cleanup_enabled()
        mineru_output_dir = os.environ.get("MINERU_OUTPUT_DIR") or setting.WORK_DIR
        mineru_lang = str(parse_options.get("mineru_lang") or os.environ.get("MINERU_LANG", "ch"))
        mineru_parse_method = str(parse_options.get("mineru_parse_method") or os.environ.get("MINERU_PARSE_METHOD", "auto"))
        mineru_start_page_id = int(parse_options.get("mineru_start_page_id") or os.environ.get("MINERU_START_PAGE_ID", "0"))
        mineru_end_page_id = int(parse_options.get("mineru_end_page_id") or os.environ.get("MINERU_END_PAGE_ID", "99999"))
        mineru_formula_enable = bool(parse_options.get("mineru_formula_enable", True))
        mineru_table_enable = bool(parse_options.get("mineru_table_enable", True))

        parser = parser_cls(
            mineru_api=mineru_api,
            mineru_server_url=mineru_server_url,
        )
        if need_structured:
            cleanup_output = False

        result = parser.parse_pdf(
            filepath=tmp_path,
            binary=None,
            parse_method="raw",
            backend=mineru_backend,
            server_url=mineru_server_url or None,
            output_dir=mineru_output_dir,
            delete_output=cleanup_output,
            request_timeout=_get_request_timeout(),
            parser_config={
                "mineru_return_images": parse_options.get("return_images", False),
                "mineru_lang": mineru_lang,
                "mineru_parse_method": mineru_parse_method,
                "mineru_formula_enable": mineru_formula_enable,
                "mineru_table_enable": mineru_table_enable,
                "mineru_start_page_id": mineru_start_page_id,
                "mineru_end_page_id": mineru_end_page_id,
            },
        )
        if (
            need_structured
            and isinstance(result, tuple)
            and len(result) >= 3
            and isinstance(result[2], dict)
        ):
            result[2]["structured_source"] = {
                "engine": "mineru",
                "parser": parser,
                "raw_outputs": result[2].get("raw_outputs", []),
            }
            result[2]["cleanup_dir"] = result[2].get("output_dir")
        return result

    zoomin = 3
    max_pdf_pages = max(1, int(parse_options.get("deepdoc_max_pages") or os.environ.get("DEEPDOC_PDF_MAX_PAGES", "10")))
    deepdoc_pdf_mode = _normalize_deepdoc_pdf_mode(str(parse_options.get("deepdoc_pdf_mode") or "auto"))
    execution_profile = _normalize_execution_profile(str(parse_options.get("execution_profile") or "auto"))
    pdf_text_layer = None
    hybrid_plan = None
    stage_timings: dict[str, float] = {}
    visual_parse_requested = any(
        bool(parse_options.get(key, False))
        for key in ("return_images", "enable_formula", "enable_seal")
    )
    if deepdoc_pdf_mode == "hybrid" and execution_profile == "gpu":
        from deepdoc.parser.pdf_hybrid_router import (
            build_pdf_hybrid_plan,
            build_pdf_text_layer_report_from_hybrid_plan,
        )
        from deepdoc.parser.pdf_parser import extract_native_pdf_text

        hybrid_plan = build_pdf_hybrid_plan(tmp_path, page_from=0, page_to=max_pdf_pages)
        pdf_text_layer = build_pdf_text_layer_report_from_hybrid_plan(hybrid_plan)
        if hybrid_plan.get("all_pages_digital_clean") and not visual_parse_requested:
            boxes = deepcopy(hybrid_plan.get("native_boxes") or [])
            for box in boxes:
                box["text"] = re.sub(r"([\t \u3000]|\u3000){2,}", " ", str(box.get("text") or "").strip())
            rows = [(box["text"], "") for box in boxes if str(box.get("text") or "").strip()]
            parse_meta = {
                "page_count": hybrid_plan.get("page_count"),
                "total_page_count": hybrid_plan.get("total_page_count"),
                "pdf_parse_mode": "hybrid",
                "deepdoc_pdf_mode": deepdoc_pdf_mode,
                "execution_profile": execution_profile,
                "pdf_text_layer": pdf_text_layer,
                "page_routes": deepcopy(hybrid_plan.get("pages") or []),
                "hybrid_route_summary": deepcopy(hybrid_plan.get("route_summary") or {}),
            }
            if need_structured:
                structured_boxes = deepcopy(boxes)
                if structured_boxes and "chars" not in structured_boxes[0]:
                    structured_boxes, _native_meta = extract_native_pdf_text(
                        tmp_path,
                        page_from=0,
                        page_to=max_pdf_pages,
                        preserve_geometry=True,
                    )
                parse_meta["structured_source"] = {
                    "engine": "native_pdf",
                    "boxes": structured_boxes,
                    "pdf_text_layer": pdf_text_layer,
                    "page_routes": deepcopy(hybrid_plan.get("pages") or []),
                    "hybrid_route_summary": deepcopy(hybrid_plan.get("route_summary") or {}),
                }
            return rows, [], parse_meta
    elif deepdoc_pdf_mode in {"auto", "native", "hybrid"}:
        from deepdoc.parser.pdf_parser import detect_pdf_text_layer, extract_native_pdf_text

        pdf_text_layer = detect_pdf_text_layer(tmp_path, page_from=0, page_to=max_pdf_pages)
        use_native_text = deepdoc_pdf_mode == "native" or (
            deepdoc_pdf_mode == "auto"
            and not visual_parse_requested
            and pdf_text_layer.get("recommended_mode") == "native_text"
        ) or (
            deepdoc_pdf_mode == "hybrid"
            and execution_profile != "gpu"
            and not visual_parse_requested
            and pdf_text_layer.get("recommended_mode") == "native_text"
        )
        if use_native_text:
            boxes, native_meta = extract_native_pdf_text(tmp_path, page_from=0, page_to=max_pdf_pages)
            for box in boxes:
                box["text"] = re.sub(r"([\t \u3000]|\u3000){2,}", " ", str(box.get("text") or "").strip())
            rows = [(box["text"], "") for box in boxes if str(box.get("text") or "").strip()]
            parse_meta = {
                "page_count": native_meta.get("page_count"),
                "total_page_count": native_meta.get("total_page_count"),
                "pdf_parse_mode": "native_text",
                "deepdoc_pdf_mode": deepdoc_pdf_mode,
                "execution_profile": execution_profile,
                "pdf_text_layer": pdf_text_layer,
            }
            if need_structured:
                parse_meta["structured_source"] = {
                    "engine": "native_pdf",
                    "boxes": deepcopy(boxes),
                    "pdf_text_layer": pdf_text_layer,
                }
            return rows, [], parse_meta

    layout_model = str(parse_options.get("deepdoc_layout_model") or os.environ.get("DEEPDOC_LAYOUT_MODEL", "manual")).strip().lower()
    valid_models = {"manual", "paper", "laws", "general"}
    if layout_model not in valid_models:
        layout_model = "manual"

    parser = parser_cls()
    if layout_model != "general":
        parser.model_speciess = layout_model
        from deepdoc.parser.pdf_parser import get_shared_pdf_parser_components

        layout_components = get_shared_pdf_parser_components(model_speciess=layout_model)
        parser.layouter = layout_components["layouter"]
        parser.layouters = list(layout_components.get("layouters") or [parser.layouter])

    use_hybrid_gpu_pages = deepdoc_pdf_mode == "hybrid" and execution_profile == "gpu" and hybrid_plan is not None
    layout_page_numbers = None
    image_page_numbers = None
    if use_hybrid_gpu_pages:
        hybrid_pages = {
            int(page.get("page_number") or 0): page
            for page in hybrid_plan.get("pages") or []
            if int(page.get("page_number") or 0) > 0
        }
        seed_layouts_by_page = {
            int(page_number): deepcopy(page_layouts)
            for page_number, page_layouts in (hybrid_plan.get("seed_layouts_by_page") or {}).items()
            if int(page_number) > 0 and page_layouts
        }
        ocr_page_numbers = {int(page_number) for page_number in hybrid_plan.get("ocr_page_numbers") or [] if int(page_number) > 0}
        char_page_numbers = {
            page_number
            for page_number in ocr_page_numbers
            if str((hybrid_pages.get(page_number) or {}).get("route") or "") != "scanned"
        }
        if not visual_parse_requested:
            complex_block_page_numbers = {
                int(page_number)
                for page_number in hybrid_plan.get("complex_block_page_numbers") or []
                if int(page_number) > 0
            }
            layout_page_numbers = ocr_page_numbers | (complex_block_page_numbers - set(seed_layouts_by_page))
            image_page_numbers = set(layout_page_numbers)
            image_page_numbers.update(seed_layouts_by_page)
        stage_started_at = time.perf_counter()
        parser.prepare_pages(
            tmp_path,
            zoomin=zoomin,
            page_from=0,
            page_to=max_pdf_pages,
            char_page_numbers=char_page_numbers,
            image_page_numbers=image_page_numbers,
            load_outlines=not use_hybrid_gpu_pages,
        )
        stage_timings["prepare_pages"] = round(time.perf_counter() - stage_started_at, 6)
        native_seed_page_boxes = {
            int(page["page_number"]): deepcopy(hybrid_plan.get("native_boxes_by_page", {}).get(int(page["page_number"]), []))
            for page in hybrid_plan.get("pages", [])
            if str(page.get("ocr_scope") or "") != "full_page"
        }
        if native_seed_page_boxes:
            parser.seed_page_boxes(native_seed_page_boxes)
        if seed_layouts_by_page and hasattr(parser, "seed_page_layouts"):
            parser.seed_page_layouts(seed_layouts_by_page)
        parser.hybrid_clean_pages = {
            int(page.get("page_number") or 0)
            for page in hybrid_plan.get("pages", [])
            if page.get("route") == "digital_clean" and int(page.get("page_number") or 0) > 0
        }
        parser.complex_block_only_pages = {
            int(page_number)
            for page_number in hybrid_plan.get("complex_block_page_numbers") or []
            if int(page_number) > 0
        }
        stage_started_at = time.perf_counter()
        parser.run_page_ocr(page_numbers=ocr_page_numbers, zoomin=zoomin)
        stage_timings["run_page_ocr"] = round(time.perf_counter() - stage_started_at, 6)
        stage_started_at = time.perf_counter()
        parser.finalize_page_boxes()
        stage_timings["finalize_page_boxes"] = round(time.perf_counter() - stage_started_at, 6)
    else:
        stage_started_at = time.perf_counter()
        parser.__images__(tmp_path, zoomin, page_from=0, page_to=max_pdf_pages)
        stage_timings["images_pipeline"] = round(time.perf_counter() - stage_started_at, 6)
    if getattr(parser, "total_page", 0) > max_pdf_pages:
        logger.info(
            f"PDF page limit active: parsing first {max_pdf_pages}/"
            f"{parser.total_page} pages (DEEPDOC_PDF_MAX_PAGES)."
        )
    stage_started_at = time.perf_counter()
    parser._layouts_rec(zoomin, page_numbers=layout_page_numbers)
    stage_timings["layouts"] = round(time.perf_counter() - stage_started_at, 6)
    stage_started_at = time.perf_counter()
    table_auto_rotate = bool(parse_options.get("deepdoc_table_auto_rotate", not use_hybrid_gpu_pages))
    need_image = bool(parse_options.get("return_images", False))
    need_table_structure = not (
        use_hybrid_gpu_pages
        and not need_structured
        and not need_image
    )
    table_transformer_kwargs = {"auto_rotate": table_auto_rotate}
    try:
        if "need_table_structure" in inspect.signature(parser._table_transformer_job).parameters:
            table_transformer_kwargs["need_table_structure"] = need_table_structure
    except (TypeError, ValueError):
        pass
    parser._table_transformer_job(zoomin, **table_transformer_kwargs)
    stage_timings["table_transformer"] = round(time.perf_counter() - stage_started_at, 6)
    stage_started_at = time.perf_counter()
    parser._text_merge()
    stage_timings["text_merge"] = round(time.perf_counter() - stage_started_at, 6)
    if parse_options.get("enable_seal"):
        stage_started_at = time.perf_counter()
        parser._recognize_seals(zoomin)
        stage_timings["recognize_seals"] = round(time.perf_counter() - stage_started_at, 6)
    if parse_options.get("enable_formula"):
        stage_started_at = time.perf_counter()
        parser._recognize_formulas(zoomin)
        stage_timings["recognize_formulas"] = round(time.perf_counter() - stage_started_at, 6)
    merge_cross_page_text = getattr(parser, "_merge_cross_page_text", None)
    if callable(merge_cross_page_text):
        stage_started_at = time.perf_counter()
        merge_cross_page_text()
        stage_timings["merge_cross_page_text"] = round(time.perf_counter() - stage_started_at, 6)
    if need_structured:
        stage_started_at = time.perf_counter()
        tables_with_positions, figures_with_positions = parser._extract_table_figure(
            True,
            zoomin,
            True,
            True,
            True,
        )
        stage_timings["extract_table_figure"] = round(time.perf_counter() - stage_started_at, 6)
        if need_image:
            tbls = [res for res, _ in figures_with_positions] + [res for res, _ in tables_with_positions]
        else:
            tbls = [res for res, _ in tables_with_positions]
    else:
        tables_with_positions = []
        figures_with_positions = []
        stage_started_at = time.perf_counter()
        tbls = parser._extract_table_figure(need_image, zoomin, True, True)
        stage_timings["extract_table_figure"] = round(time.perf_counter() - stage_started_at, 6)
    stage_started_at = time.perf_counter()
    parser._concat_downward()
    stage_timings["concat_downward"] = round(time.perf_counter() - stage_started_at, 6)
    stage_started_at = time.perf_counter()
    parser._filter_forpages()
    stage_timings["filter_forpages"] = round(time.perf_counter() - stage_started_at, 6)
    for box in parser.boxes:
        box["text"] = re.sub(r"([\t \u3000]|\u3000){2,}", " ", box["text"].strip())
    parse_meta = {
        "page_count": len(getattr(parser, "page_images", []) or []),
        "total_page_count": getattr(parser, "total_page", None),
        "pdf_parse_mode": "hybrid" if use_hybrid_gpu_pages else "ocr",
        "deepdoc_pdf_mode": deepdoc_pdf_mode,
        "execution_profile": execution_profile,
        "table_auto_rotate": table_auto_rotate,
        "stage_timings": stage_timings,
        "ocr_block_count": int(getattr(parser, "_last_selective_ocr_block_count", 0) or 0),
        "complex_block_counts": deepcopy(getattr(parser, "_last_complex_block_counts", {}) or {}),
    }
    if pdf_text_layer is not None:
        parse_meta["pdf_text_layer"] = pdf_text_layer
    if hybrid_plan is not None:
        parse_meta["page_routes"] = deepcopy(hybrid_plan.get("pages") or [])
        parse_meta["hybrid_route_summary"] = deepcopy(hybrid_plan.get("route_summary") or {})
    if need_structured:
        parse_meta["structured_source"] = {
            "engine": "deepdoc",
            "parser": parser,
            "boxes": deepcopy(parser.boxes),
            "tables_with_positions": tables_with_positions,
            "figures_with_positions": figures_with_positions,
            "zoomin": zoomin,
            "pdf_text_layer": pdf_text_layer,
            "page_routes": deepcopy(hybrid_plan.get("pages") or []) if hybrid_plan is not None else [],
            "hybrid_route_summary": deepcopy(hybrid_plan.get("route_summary") or {}) if hybrid_plan is not None else {},
            "table_auto_rotate": table_auto_rotate,
            "stage_timings": deepcopy(stage_timings),
            "ocr_block_count": int(getattr(parser, "_last_selective_ocr_block_count", 0) or 0),
            "complex_block_counts": deepcopy(getattr(parser, "_last_complex_block_counts", {}) or {}),
        }
    return parser.boxes, tbls, parse_meta


def _parse_caj_from_tmp(parser_cls, tmp_path: str, parse_options: dict[str, object]):
    parser = parser_cls()
    output_dir = tempfile.mkdtemp(prefix="deepdoc-caj-", dir=str(UPLOAD_TMP_DIR))
    conversion = parser.convert_to_pdf(
        tmp_path,
        output_dir=output_dir,
        request_timeout=_get_request_timeout(),
    )
    pdf_parser_cls = _load_parser("pdf", parser_engine=parse_options.get("parser_engine"))
    if pdf_parser_cls is None:
        raise RuntimeError("PDF parser is unavailable after CAJ conversion")
    result = _parse_pdf_from_tmp(pdf_parser_cls, conversion.pdf_path, parse_options)
    if isinstance(result, tuple) and len(result) >= 3 and isinstance(result[2], dict):
        result[2]["source_file_type"] = "caj"
        result[2]["converted_file_type"] = "pdf"
        result[2]["caj_conversion"] = conversion.model_dump()
        result[2]["cleanup_dir"] = conversion.output_dir
    return result


def _artifact_response_payload(artifact_paths) -> dict[str, object]:
    if not artifact_paths:
        return {}
    return {
        "markdown_url": artifact_paths.markdown_url,
        "manifest_url": artifact_paths.manifest_url,
        "publish_events_url": artifact_paths.publish_events_url,
        "structured_url": artifact_paths.structured_url,
        "chunks_url": artifact_paths.chunks_url,
        "ingest_url": artifact_paths.ingest_url,
        "assets_url_prefix": artifact_paths.assets_url_prefix,
        "root_dir": artifact_paths.root_dir,
        "storage_backend": ARTIFACT_STORE.__class__.__name__.removesuffix("ArtifactStore").lower() or "local",
    }


def _manifest_artifact_urls(manifest: ParseManifest) -> dict[str, object]:
    return {
        "markdown_url": manifest.markdown_url,
        "manifest_url": getattr(manifest, "manifest_url", f"/api/v1/artifacts/{manifest.parse_id}/manifest"),
        "publish_events_url": getattr(
            manifest,
            "publish_events_url",
            f"/api/v1/artifacts/{manifest.parse_id}/publish-events",
        ),
        "structured_url": manifest.structured_url,
        "chunks_url": manifest.chunks_url,
        "ingest_url": manifest.ingest_url,
        "assets_url_prefix": manifest.assets_url_prefix,
        "root_dir": manifest.root_dir,
        "storage_backend": manifest.storage_backend,
    }


def _build_artifact_profile(file_type: str, parse_options: dict[str, object]) -> dict[str, object]:
    parser_engine = str(parse_options.get("parser_engine", "deepdoc"))
    profile: dict[str, object] = {
        "artifact_profile_version": ARTIFACT_PROFILE_VERSION,
        "file_type": file_type,
        "tenant_id": str(parse_options.get("tenant_id") or "").strip(),
        "parser_engine": parser_engine,
        "compute_device": str(parse_options.get("compute_device", "gpu")),
        "return_images": bool(parse_options.get("return_images", False)),
        "strict_text": bool(parse_options.get("strict_text", False)),
        "enable_formula": bool(parse_options.get("enable_formula", False)),
        "enable_seal": bool(parse_options.get("enable_seal", False)),
        "include_chunks": bool(parse_options.get("include_chunks", False)),
        "chunk_max_tokens": int(parse_options.get("chunk_max_tokens") or DEFAULT_CHUNK_MAX_TOKENS),
        "chunk_overlap_tokens": int(parse_options.get("chunk_overlap_tokens") or DEFAULT_CHUNK_OVERLAP_TOKENS),
        "chunk_strategy": str(parse_options.get("chunk_strategy") or DEFAULT_CHUNK_STRATEGY),
    }
    if file_type == "pdf":
        if parser_engine == "deepdoc":
            profile.update(
                {
                    "layout_model": str(parse_options.get("deepdoc_layout_model") or "manual").strip().lower(),
                    "max_pdf_pages": int(parse_options.get("deepdoc_max_pages") or os.environ.get("DEEPDOC_PDF_MAX_PAGES", "10")),
                    "pdf_mode": str(parse_options.get("deepdoc_pdf_mode") or "auto").strip().lower(),
                }
            )
        elif parser_engine == PADDLEOCR_VL_ENGINE:
            profile.update(
                {
                    "paddle_api_url": str(parse_options.get("paddle_api_url") or ""),
                    "paddle_algorithm_config": parse_options.get("paddle_algorithm_config") or {},
                    "paddle_prettify_markdown": bool(parse_options.get("paddle_prettify_markdown", True)),
                    "paddle_show_formula_number": bool(parse_options.get("paddle_show_formula_number", False)),
                }
            )
        elif parser_engine == "mineru":
            profile.update(
                {
                    "mineru_api": str(parse_options.get("mineru_api") or ""),
                    "mineru_server_url": str(parse_options.get("mineru_server_url") or ""),
                    "mineru_backend": str(parse_options.get("mineru_backend") or ""),
                    "mineru_lang": str(parse_options.get("mineru_lang") or os.environ.get("MINERU_LANG", "ch")),
                    "mineru_parse_method": str(parse_options.get("mineru_parse_method") or os.environ.get("MINERU_PARSE_METHOD", "auto")),
                    "mineru_formula_enable": bool(parse_options.get("mineru_formula_enable", True)),
                    "mineru_table_enable": bool(parse_options.get("mineru_table_enable", True)),
                    "mineru_start_page_id": int(parse_options.get("mineru_start_page_id") or os.environ.get("MINERU_START_PAGE_ID", "0")),
                    "mineru_end_page_id": int(parse_options.get("mineru_end_page_id") or os.environ.get("MINERU_END_PAGE_ID", "99999")),
                }
            )
    return profile


def _load_parse_artifact(parse_id: str) -> ParseArtifact:
    payload, _ = ARTIFACT_STORE.read_file(parse_id, "structured.json", "application/json")
    return ParseArtifact.model_validate(json.loads(payload.decode("utf-8")))


def _maybe_load_cached_result(
    *,
    filename: str,
    parse_options: dict[str, object],
    artifact_key: str,
) -> dict[str, object] | None:
    if not bool(parse_options.get("reuse_artifacts", False)):
        return None
    manifest = ARTIFACT_STORE.find_manifest_by_artifact_key(
        artifact_key,
        tenant_id=str(parse_options.get("tenant_id") or "").strip() or None,
    )
    if manifest is None:
        return None
    try:
        _ensure_manifest_access(manifest)
    except PermissionError:
        logger.warning("Artifact cache hit rejected by tenant policy parse_id=%s", manifest.parse_id)
        return None

    try:
        markdown_payload, _ = ARTIFACT_STORE.read_file(manifest.parse_id, "markdown.md", "text/markdown")
    except FileNotFoundError:
        logger.warning("Cached manifest found but markdown missing for parse_id=%s", manifest.parse_id)
        return None

    result: dict[str, object] = {
        "filename": filename,
        "type": manifest.file_type,
        "markdown": markdown_payload.decode("utf-8"),
        "parser_engine": manifest.parser_engine,
        "document_id": manifest.document_id,
        "parse_id": manifest.parse_id,
        "tenant_id": _manifest_tenant_id(manifest),
        "artifact_urls": _manifest_artifact_urls(manifest),
        "asset_count": manifest.asset_count,
        "chunk_count": manifest.chunk_count,
        "cache_hit": True,
    }
    if manifest.parser_engine == PADDLEOCR_VL_ENGINE and manifest.metadata.get("seal_count") is not None:
        result["seal_count"] = int(manifest.metadata.get("seal_count") or 0)
    ingest_publish = manifest.metadata.get("ingest_publish")
    if ingest_publish:
        result["ingest_publish"] = ingest_publish
    should_publish = bool(parse_options.get("publish_ingest", False)) and (
        not isinstance(ingest_publish, dict) or ingest_publish.get("status") != "published"
    )
    artifact = None
    if parse_options.get("return_structured"):
        try:
            artifact = _load_parse_artifact(manifest.parse_id)
        except FileNotFoundError:
            logger.warning("Cached manifest found but structured artifact missing for parse_id=%s", manifest.parse_id)
            return None
        result["structured"] = artifact.model_dump(mode="json")
    if should_publish:
        if artifact is None:
            try:
                artifact = _load_parse_artifact(manifest.parse_id)
            except FileNotFoundError:
                logger.warning("Cached manifest found but structured artifact missing for publish parse_id=%s", manifest.parse_id)
                return result
        publish_result, _ = _publish_ingest_records(
            artifact=artifact,
            manifest=manifest,
            artifact_paths=ARTIFACT_STORE.get_paths(manifest.parse_id, manifest.filename),
            parse_options=parse_options,
        )
        if publish_result is not None:
            result["ingest_publish"] = publish_result
    return result


def _publish_ingest_records(
    *,
    artifact: ParseArtifact,
    manifest: ParseManifest | None,
    artifact_paths,
    parse_options: dict[str, object],
):
    with trace_operation(
        "deepdoc.ingest.publish",
        attributes={
            "deepdoc.parse_id": artifact.document.parse_id,
            "deepdoc.document_id": artifact.document.document_id,
            "deepdoc.artifact_key": manifest.artifact_key if manifest is not None else None,
            "deepdoc.ingest.sink_type": getattr(INGEST_PUBLISHER, "sink_type", "unknown"),
        },
    ):
        if not bool(parse_options.get("publish_ingest", False)):
            return None, manifest
        if manifest is None or artifact_paths is None:
            raise RuntimeError("publish_ingest requires persisted artifacts and manifest")
        strict_publish = _parse_bool(os.environ.get("DEEPDOC_INGEST_PUBLISH_STRICT"), default=False)
        requested_by = str(parse_options.get("publish_requested_by") or "parse").strip() or "parse"

        asset_url_mode = str(
            parse_options.get("publish_asset_url_mode")
            or os.environ.get("DEEPDOC_INGEST_PUBLISH_ASSET_URL_MODE")
            or "proxy"
        ).strip().lower()
        if asset_url_mode not in {"proxy", "direct", "signed"}:
            asset_url_mode = "proxy"
        signed_url_ttl = int(
            parse_options.get("publish_signed_url_ttl") or os.environ.get("DEEPDOC_INGEST_PUBLISH_SIGNED_URL_TTL", "3600")
        )
        chunk_records = build_chunk_export_records(
            artifact,
            store=ARTIFACT_STORE,
            asset_url_mode=asset_url_mode,
            signed_url_ttl=signed_url_ttl,
        )
        ingest_records = build_ingest_export_records(chunk_records)
        set_current_span_attributes(
            {
                "deepdoc.ingest.chunk_record_count": len(chunk_records),
                "deepdoc.ingest.record_count": len(ingest_records),
                "deepdoc.ingest.asset_url_mode": asset_url_mode,
            }
        )
        publish_error = None
        retry_base_delay_seconds = float(os.environ.get("DEEPDOC_INGEST_RETRY_BASE_DELAY_SECONDS", "60"))
        retry_max_delay_seconds = float(os.environ.get("DEEPDOC_INGEST_RETRY_MAX_DELAY_SECONDS", "3600"))
        max_failure_count = int(os.environ.get("DEEPDOC_INGEST_RETRY_MAX_FAILURES", "5"))
        try:
            publish_result = INGEST_PUBLISHER.publish(
                manifest,
                ingest_records,
                artifact=artifact,
                chunk_records=chunk_records,
            )
        except Exception as exc:
            publish_error = exc
            logger.exception("Ingest publish failed for parse_id=%s", artifact.document.parse_id)
            if isinstance(exc, IngestPublishError):
                error_metadata = dict(exc.metadata or {})
                error_metadata.setdefault("error", str(exc))
                sink_type = exc.sink_type
                destination = exc.destination
                response_code = exc.response_code
            else:
                error_metadata = {"error": str(exc)}
                sink_type = getattr(INGEST_PUBLISHER, "sink_type", "unknown")
                destination = None
                response_code = None
            add_span_event("deepdoc.ingest.publish.failed", {"error": str(exc)})
            publish_result = {
                "enabled": True,
                "sink_type": sink_type,
                "status": "failed",
                "record_count": len(ingest_records),
                "destination": destination,
                "published_at": None,
                "response_code": response_code,
                "metadata": error_metadata,
            }
        publish_state, publish_attempt = build_ingest_publish_state(
            previous_state=manifest.metadata.get("ingest_publish"),
            result=publish_result,
            requested_by=requested_by,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_max_delay_seconds=retry_max_delay_seconds,
            max_failure_count=max_failure_count,
        )
        try:
            ARTIFACT_STORE.append_publish_event(artifact_paths, publish_attempt.model_dump(mode="json"))
        except Exception:
            logger.exception("Failed to append publish event for parse_id=%s", artifact.document.parse_id)

        updated_manifest = manifest.model_copy(
            update={
                "metadata": {
                    **manifest.metadata,
                    "ingest_publish": publish_state.model_dump(mode="json"),
                }
            }
        )
        ARTIFACT_STORE.write_manifest(artifact_paths, updated_manifest)
        publish_sink_type = getattr(publish_result, "sink_type", None)
        publish_status = getattr(publish_result, "status", None)
        if isinstance(publish_result, dict):
            publish_sink_type = publish_sink_type or publish_result.get("sink_type")
            publish_status = publish_status or publish_result.get("status")
        observe_ingest_publish(
            sink_type=str(publish_sink_type or getattr(INGEST_PUBLISHER, "sink_type", "unknown")),
            status=str(publish_status or "unknown"),
            record_count=len(ingest_records),
        )
        set_current_span_attributes(
            {
                "deepdoc.ingest.status": publish_status or "unknown",
                "deepdoc.ingest.retryable": publish_state.retryable,
                "deepdoc.ingest.dead_lettered": publish_state.dead_lettered,
            }
        )
        if publish_error is not None and strict_publish:
            raise publish_error
        return publish_state.model_dump(mode="json"), updated_manifest


def _build_structured_artifact(
    *,
    filename: str,
    file_type: str,
    file_bytes: bytes,
    markdown_content: str,
    parse_options: dict[str, object],
    parse_meta: dict[str, object],
    artifact_profile: dict[str, object],
    artifact_key: str,
):
    parser_engine = str(parse_options.get("parser_engine", "deepdoc"))
    with trace_operation(
        "deepdoc.build_structured_artifact",
        attributes={
            "deepdoc.filename": filename,
            "deepdoc.file_type": file_type,
            "deepdoc.parser_engine": parser_engine,
            "deepdoc.artifact_key": artifact_key,
            "deepdoc.persist_artifacts": bool(parse_options.get("persist_artifacts", False)),
            "deepdoc.include_chunks": bool(parse_options.get("include_chunks", False)),
        },
    ):
        document = build_document(
            filename=filename,
            file_type=file_type,
            parser_engine=parser_engine,
            file_bytes=file_bytes,
            page_count=parse_meta.get("page_count"),
            total_page_count=parse_meta.get("total_page_count"),
            metadata={
                "artifact_key": artifact_key,
                "tenant_id": str(parse_options.get("tenant_id") or "").strip() or None,
                "auth_subject": str(parse_options.get("auth_subject") or "").strip() or None,
                "auth_mode": str(parse_options.get("auth_mode") or "none"),
                "return_images": bool(parse_options.get("return_images", False)),
                "strict_text": bool(parse_options.get("strict_text", False)),
                "enable_formula": bool(parse_options.get("enable_formula", False)),
                "enable_seal": bool(parse_options.get("enable_seal", False)),
                "deepdoc_pdf_mode": parse_meta.get("deepdoc_pdf_mode") or parse_options.get("deepdoc_pdf_mode"),
                "pdf_parse_mode": parse_meta.get("pdf_parse_mode"),
                "pdf_text_layer": parse_meta.get("pdf_text_layer"),
                "page_routes": parse_meta.get("page_routes"),
                "source_file_type": parse_meta.get("source_file_type"),
                "converted_file_type": parse_meta.get("converted_file_type"),
                "caj_conversion": parse_meta.get("caj_conversion"),
            },
        )
        set_current_span_attributes(
            {
                "deepdoc.parse_id": document.parse_id,
                "deepdoc.document_id": document.document_id,
                "deepdoc.tenant_id": str(parse_options.get("tenant_id") or "").strip() or None,
            }
        )
        persist_artifacts = bool(parse_options.get("persist_artifacts", False))
        artifact_paths = ARTIFACT_STORE.get_paths(document.parse_id, filename) if persist_artifacts else None
        store = ARTIFACT_STORE if persist_artifacts else None
        structured_source = parse_meta.get("structured_source")
        chunk_max_tokens = int(parse_options.get("chunk_max_tokens") or DEFAULT_CHUNK_MAX_TOKENS)
        chunk_overlap_tokens = int(parse_options.get("chunk_overlap_tokens") or DEFAULT_CHUNK_OVERLAP_TOKENS)
        chunk_strategy = normalize_chunk_strategy(str(parse_options.get("chunk_strategy") or DEFAULT_CHUNK_STRATEGY))

        if (
            file_type in {"pdf", "caj"}
            and isinstance(structured_source, dict)
            and structured_source.get("engine") == "deepdoc"
        ):
            artifact = build_deepdoc_artifact(
                document=document,
                markdown=markdown_content,
                parser=structured_source["parser"],
                boxes=structured_source["boxes"],
                tables_with_positions=structured_source["tables_with_positions"],
                figures_with_positions=structured_source["figures_with_positions"],
                zoomin=int(structured_source.get("zoomin", 3)),
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chunk_strategy=chunk_strategy,
                store=store,
                artifact_paths=artifact_paths,
                metadata={
                    "parser_engine": parser_engine,
                    "chunk_strategy": chunk_strategy,
                    "source": "deepdoc_ocr_layout",
                    "source_file_type": parse_meta.get("source_file_type"),
                    "converted_file_type": parse_meta.get("converted_file_type"),
                    "pdf_parse_mode": parse_meta.get("pdf_parse_mode"),
                    "pdf_text_layer": parse_meta.get("pdf_text_layer"),
                    "page_routes": parse_meta.get("page_routes"),
                },
            )
        elif (
            file_type in {"pdf", "caj"}
            and isinstance(structured_source, dict)
            and structured_source.get("engine") == "native_pdf"
        ):
            artifact = build_native_pdf_artifact(
                document=document,
                markdown=markdown_content,
                boxes=structured_source.get("boxes") if isinstance(structured_source.get("boxes"), list) else [],
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chunk_strategy=chunk_strategy,
                metadata={
                    "parser_engine": parser_engine,
                    "chunk_strategy": chunk_strategy,
                    "source_file_type": parse_meta.get("source_file_type"),
                    "converted_file_type": parse_meta.get("converted_file_type"),
                    "pdf_parse_mode": parse_meta.get("pdf_parse_mode"),
                    "pdf_text_layer": parse_meta.get("pdf_text_layer"),
                    "page_routes": parse_meta.get("page_routes"),
                },
            )
        elif (
            file_type in {"pdf", "caj"}
            and isinstance(structured_source, dict)
            and structured_source.get("engine") == "paddleocr_vl"
        ):
            artifact = build_paddleocr_artifact(
                document=document,
                markdown=markdown_content,
                raw_result=structured_source.get("raw_result"),
                parser=structured_source.get("parser"),
                store=store,
                artifact_paths=artifact_paths,
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chunk_strategy=chunk_strategy,
                metadata={
                    "parser_engine": parser_engine,
                    "chunk_strategy": chunk_strategy,
                    "source_file_type": parse_meta.get("source_file_type"),
                    "converted_file_type": parse_meta.get("converted_file_type"),
                },
            )
        elif (
            file_type in {"pdf", "caj"}
            and isinstance(structured_source, dict)
            and structured_source.get("engine") == "mineru"
        ):
            artifact = build_mineru_artifact(
                document=document,
                markdown=markdown_content,
                raw_outputs=structured_source.get("raw_outputs"),
                parser=structured_source.get("parser"),
                store=store,
                artifact_paths=artifact_paths,
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chunk_strategy=chunk_strategy,
                metadata={
                    "parser_engine": parser_engine,
                    "chunk_strategy": chunk_strategy,
                    "source_file_type": parse_meta.get("source_file_type"),
                    "converted_file_type": parse_meta.get("converted_file_type"),
                },
            )
        elif (
            file_type in {"csv", "tsv"}
            and isinstance(structured_source, dict)
            and structured_source.get("engine") == "csv"
        ):
            artifact = build_csv_artifact(
                document=document,
                markdown=markdown_content,
                rows=structured_source.get("rows") if isinstance(structured_source.get("rows"), list) else [],
                delimiter=str(structured_source.get("delimiter") or ("\t" if file_type == "tsv" else ",")),
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chunk_strategy=chunk_strategy,
                metadata={"parser_engine": parser_engine, "chunk_strategy": chunk_strategy},
            )
        elif (
            file_type == "epub"
            and isinstance(structured_source, dict)
            and structured_source.get("engine") == "epub"
        ):
            artifact = build_epub_artifact(
                document=document,
                markdown=markdown_content,
                blocks=structured_source.get("blocks") if isinstance(structured_source.get("blocks"), list) else [],
                epub_metadata=structured_source.get("metadata") if isinstance(structured_source.get("metadata"), dict) else {},
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chunk_strategy=chunk_strategy,
                metadata={"parser_engine": parser_engine, "chunk_strategy": chunk_strategy},
            )
        elif (
            file_type in {"rtf", "odt"}
            and isinstance(structured_source, dict)
            and structured_source.get("engine") == file_type
        ):
            artifact = build_rich_text_artifact(
                document=document,
                markdown=markdown_content,
                blocks=structured_source.get("blocks") if isinstance(structured_source.get("blocks"), list) else [],
                source=file_type,
                source_metadata=structured_source.get("metadata") if isinstance(structured_source.get("metadata"), dict) else {},
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chunk_strategy=chunk_strategy,
                metadata={"parser_engine": parser_engine, "chunk_strategy": chunk_strategy},
            )
        elif (
            file_type in {"eml", "msg"}
            and isinstance(structured_source, dict)
            and structured_source.get("engine") == "email"
        ):
            artifact = build_rich_text_artifact(
                document=document,
                markdown=markdown_content,
                blocks=structured_source.get("blocks") if isinstance(structured_source.get("blocks"), list) else [],
                source="email",
                source_metadata=structured_source.get("metadata") if isinstance(structured_source.get("metadata"), dict) else {},
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chunk_strategy=chunk_strategy,
                metadata={"parser_engine": parser_engine, "chunk_strategy": chunk_strategy},
            )
        elif (
            file_type in IMAGE_FILE_TYPES
            and isinstance(structured_source, dict)
            and structured_source.get("engine") == "image"
        ):
            artifact = build_image_artifact(
                document=document,
                markdown=markdown_content,
                image=structured_source["image"],
                boxes=structured_source.get("boxes") if isinstance(structured_source.get("boxes"), list) else [],
                barcodes=structured_source.get("barcodes")
                if isinstance(structured_source.get("barcodes"), list)
                else [],
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chunk_strategy=chunk_strategy,
                store=store,
                artifact_paths=artifact_paths,
                metadata={"parser_engine": parser_engine, "chunk_strategy": chunk_strategy},
            )
        else:
            artifact = build_generic_artifact(
                document=document,
                markdown=markdown_content,
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chunk_strategy=chunk_strategy,
                metadata={"parser_engine": parser_engine, "chunk_strategy": chunk_strategy},
            )

        if not bool(parse_options.get("include_chunks", False)):
            artifact = artifact.model_copy(update={"chunks": []})

        set_current_span_attributes(
            {
                "deepdoc.artifact.asset_count": len(artifact.assets),
                "deepdoc.artifact.block_count": len(artifact.blocks),
                "deepdoc.artifact.chunk_count": len(artifact.chunks),
            }
        )

        manifest = None
        if artifact_paths:
            manifest_metadata = {"artifact_profile": artifact_profile}
            if parse_meta.get("seal_count") is not None:
                manifest_metadata["seal_count"] = int(parse_meta.get("seal_count") or 0)
            manifest = build_parse_manifest(
                artifact,
                artifact_paths,
                storage_backend=ARTIFACT_STORE.__class__.__name__.removesuffix("ArtifactStore").lower() or "local",
                artifact_key=artifact_key,
                extra_metadata=manifest_metadata,
            )

        if artifact_paths:
            if bool(parse_options.get("persist_source", False)):
                ARTIFACT_STORE.write_source(artifact_paths, file_bytes)
            ARTIFACT_STORE.write_markdown(artifact_paths, markdown_content)
            if manifest is not None:
                ARTIFACT_STORE.write_manifest(artifact_paths, manifest)
            ARTIFACT_STORE.write_structured(artifact_paths, artifact)
            ARTIFACT_STORE.write_chunks(artifact_paths, artifact)
            ARTIFACT_STORE.write_ingest(artifact_paths, artifact)
            add_span_event(
                "deepdoc.artifact.persisted",
                {
                    "deepdoc.parse_id": document.parse_id,
                    "deepdoc.asset_count": len(artifact.assets),
                    "deepdoc.chunk_count": len(artifact.chunks),
                },
            )

        return artifact, artifact_paths, manifest


def _parse_single_file(file, parse_options: dict[str, object]) -> dict:
    """解析单个文件，返回结果字典。"""
    filename = file.filename or ""
    parser_engine_label = str(parse_options.get("parser_engine") or "deepdoc").strip() or "deepdoc"
    with trace_operation(
        "deepdoc.parse_single_file",
        attributes={
            "deepdoc.filename": filename,
            "deepdoc.parser_engine": parser_engine_label,
            "deepdoc.tenant_id": str(parse_options.get("tenant_id") or "").strip() or None,
        },
    ):
        if not filename:
            observe_parse_result(
                parser_engine=parser_engine_label,
                file_type="unknown",
                status="error",
                cache_hit=False,
            )
            return {
                "filename": filename,
                **build_error_payload(ErrorCode.NO_SELECTED_FILE, locale=str(parse_options.get("error_locale") or "")),
            }

        file_type = _infer_file_type(filename)
        set_current_span_attributes({"deepdoc.file_type": file_type or "unknown"})
        if not file_type:
            observe_parse_result(
                parser_engine=parser_engine_label,
                file_type="unknown",
                status="error",
                cache_hit=False,
            )
            return {
                "filename": filename,
                **build_error_payload(
                    ErrorCode.UNSUPPORTED_FILE_EXTENSION,
                    message=f"Unsupported file extension: {filename}",
                    locale=str(parse_options.get("error_locale") or ""),
                    details={"filename": filename},
                ),
            }

        validation_error = validate_file(
            file,
            allowed_extensions=list(PARSER_IMPORTS.keys()) + sorted(IMAGE_FILE_TYPES),
            check_image=file_type in IMAGE_FILE_TYPES,
        )
        if validation_error:
            observe_parse_result(
                parser_engine=parser_engine_label,
                file_type=file_type,
                status="error",
                cache_hit=False,
            )
            return {
                "filename": filename,
                **build_error_payload(
                    ErrorCode.VALIDATION_ERROR,
                    message=validation_error,
                    locale=str(parse_options.get("error_locale") or ""),
                ),
            }

        parser_cls = None
        if file_type not in IMAGE_FILE_TYPES:
            try:
                parser_cls = _load_parser(
                    file_type, parser_engine=parse_options.get("parser_engine")
                )
            except ModuleNotFoundError as e:
                observe_parse_result(
                    parser_engine=parser_engine_label,
                    file_type=file_type,
                    status="error",
                    cache_hit=False,
                )
                return {
                    "filename": filename,
                    **build_error_payload(
                        ErrorCode.PARSER_DEPENDENCY_MISSING,
                        message=f"Parser dependency missing: {e}",
                        locale=str(parse_options.get("error_locale") or ""),
                    ),
                }

            if not parser_cls:
                observe_parse_result(
                    parser_engine=parser_engine_label,
                    file_type=file_type,
                    status="error",
                    cache_hit=False,
                )
                return {
                    "filename": filename,
                    **build_error_payload(
                        ErrorCode.UNSUPPORTED_FILE_TYPE,
                        message=f"Unsupported file type: {file_type}",
                        locale=str(parse_options.get("error_locale") or ""),
                        details={"file_type": file_type},
                    ),
                }

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_type}", dir=str(UPLOAD_TMP_DIR)
            ) as tmp:
                tmp_path = tmp.name
                file.save(tmp_path)
        except Exception as e:
            traceback.print_exc()
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            observe_parse_result(
                parser_engine=parser_engine_label,
                file_type=file_type,
                status="error",
                cache_hit=False,
            )
            return {
                "filename": filename,
                **build_error_payload(
                    ErrorCode.UPLOAD_SAVE_FAILED,
                    message=f"Failed to save upload: {e}",
                    locale=str(parse_options.get("error_locale") or ""),
                ),
            }

        try:
            file_bytes = Path(tmp_path).read_bytes()
            source_sha = hashlib.sha256(file_bytes).hexdigest()
            with logger.context(file_sha=source_sha, engine=parser_engine_label):
                artifact_profile = _build_artifact_profile(file_type, parse_options)
                artifact_key = build_artifact_key(source_sha, artifact_profile)
                set_current_span_attributes(
                    {
                        "deepdoc.source_bytes": len(file_bytes),
                        "deepdoc.artifact_key": artifact_key,
                    }
                )
                cached_result = _maybe_load_cached_result(
                    filename=filename,
                    parse_options=parse_options,
                    artifact_key=artifact_key,
                )
                if cached_result is not None:
                    add_span_event(
                        "deepdoc.artifact.cache_hit",
                        {
                            "deepdoc.artifact_key": artifact_key,
                            "deepdoc.parse_id": cached_result.get("parse_id"),
                        },
                    )
                    observe_parse_result(
                        parser_engine=parser_engine_label,
                        file_type=file_type,
                        status="success",
                        cache_hit=True,
                        source_bytes=len(file_bytes),
                        asset_count=int(cached_result.get("asset_count") or 0),
                        chunk_count=int(cached_result.get("chunk_count") or 0),
                    )
                    return cached_result

                parse_meta: dict[str, object] = {}
                if file_type in IMAGE_FILE_TYPES:
                    with trace_operation(
                        "deepdoc.parse_image",
                        attributes={"deepdoc.filename": filename, "deepdoc.parser_engine": parser_engine_label},
                    ):
                        parse_results = _parse_image_from_tmp(
                            tmp_path=tmp_path,
                            parse_options=parse_options,
                        )
                    if (
                        isinstance(parse_results, tuple)
                        and len(parse_results) >= 3
                        and isinstance(parse_results[2], dict)
                    ):
                        parse_meta = parse_results[2]
                        parse_results = parse_results[:2]
                elif file_type == "pdf":
                    with trace_operation(
                        "deepdoc.parse_pdf",
                        attributes={"deepdoc.filename": filename, "deepdoc.parser_engine": parser_engine_label},
                    ):
                        parse_results = _parse_pdf_from_tmp(
                            parser_cls=parser_cls,
                            tmp_path=tmp_path,
                            parse_options=parse_options,
                        )
                    if (
                        isinstance(parse_results, tuple)
                        and len(parse_results) >= 3
                        and isinstance(parse_results[2], dict)
                    ):
                        parse_meta = parse_results[2]
                        parse_results = parse_results[:2]
                elif file_type == "caj":
                    with trace_operation(
                        "deepdoc.parse_caj",
                        attributes={"deepdoc.filename": filename, "deepdoc.parser_engine": parser_engine_label},
                    ):
                        parse_results = _parse_caj_from_tmp(
                            parser_cls=parser_cls,
                            tmp_path=tmp_path,
                            parse_options=parse_options,
                        )
                    if (
                        isinstance(parse_results, tuple)
                        and len(parse_results) >= 3
                        and isinstance(parse_results[2], dict)
                    ):
                        parse_meta = parse_results[2]
                        parse_results = parse_results[:2]
                else:
                    with trace_operation(
                        "deepdoc.parse_non_pdf",
                        attributes={"deepdoc.filename": filename, "deepdoc.parser_engine": parser_engine_label},
                    ):
                        parser = parser_cls()
                        parse_results = parser(tmp_path)
                    if (
                        isinstance(parse_results, tuple)
                        and len(parse_results) >= 3
                        and isinstance(parse_results[2], dict)
                    ):
                        parse_meta = parse_results[2]
                        parse_results = parse_results[:2]

                markdown_content = post_process_markdown(
                    results_to_markdown(parse_results),
                    return_images=bool(parse_options.get("return_images", False)),
                    strict_text=bool(parse_options.get("strict_text", False)),
                )
                result: dict[str, object] = {
                    "filename": filename,
                    "type": file_type,
                    "markdown": markdown_content,
                    "parser_engine": parser_engine_label,
                    "tenant_id": str(parse_options.get("tenant_id") or "").strip() or None,
                    "cache_hit": False,
                }
                if file_type == "pdf" and parse_options.get("parser_engine") == PADDLEOCR_VL_ENGINE:
                    result["seal_count"] = int(parse_meta.get("seal_count") or 0)

                if parse_options.get("return_structured") or parse_options.get("persist_artifacts"):
                    artifact, artifact_paths, manifest = _build_structured_artifact(
                        filename=filename,
                        file_type=file_type,
                        file_bytes=file_bytes,
                        markdown_content=markdown_content,
                        parse_options=parse_options,
                        parse_meta=parse_meta,
                        artifact_profile=artifact_profile,
                        artifact_key=artifact_key,
                    )
                    publish_result, manifest = _publish_ingest_records(
                        artifact=artifact,
                        manifest=manifest,
                        artifact_paths=artifact_paths,
                        parse_options=parse_options,
                    )
                    result["document_id"] = artifact.document.document_id
                    result["parse_id"] = artifact.document.parse_id
                    result["artifact_urls"] = _artifact_response_payload(artifact_paths)
                    result["asset_count"] = len(artifact.assets)
                    result["chunk_count"] = len(artifact.chunks)
                    if publish_result is not None:
                        result["ingest_publish"] = publish_result
                    if parse_options.get("return_structured"):
                        result["structured"] = artifact.model_dump(mode="json")

                    observe_parse_result(
                        parser_engine=artifact.document.parser_engine,
                        file_type=file_type,
                        status="success",
                        cache_hit=False,
                        source_bytes=len(file_bytes),
                        asset_count=len(artifact.assets),
                        chunk_count=len(artifact.chunks),
                    )
                else:
                    observe_parse_result(
                        parser_engine=parser_engine_label,
                        file_type=file_type,
                        status="success",
                        cache_hit=False,
                        source_bytes=len(file_bytes),
                    )

                return result

        except Exception as e:
            traceback.print_exc()
            observe_parse_result(
                parser_engine=parser_engine_label,
                file_type=file_type,
                status="error",
                cache_hit=False,
                source_bytes=len(file_bytes) if "file_bytes" in locals() else None,
            )
            return {
                "filename": filename,
                **build_error_payload(
                    ErrorCode.INTERNAL_ERROR,
                    message=str(e),
                    locale=str(parse_options.get("error_locale") or ""),
                ),
            }
        finally:
            cleanup_dir = parse_meta.get("cleanup_dir") if "parse_meta" in locals() else None
            if cleanup_dir and _cleanup_enabled():
                cleanup_path = Path(str(cleanup_dir))
                if cleanup_path.exists():
                    import shutil

                    shutil.rmtree(cleanup_path, ignore_errors=True)
        if os.path.exists(tmp_path) and _cleanup_enabled():
            os.remove(tmp_path)


def _deliver_terminal_async_task_callback(
    task_id: str,
    *,
    requested_by: str = "worker",
    force: bool = False,
) -> dict[str, object] | None:
    try:
        task = ASYNC_TASK_STORE.load_task(task_id)
    except Exception:
        logger.exception("Failed to load async task for callback delivery task_id=%s", task_id)
        return None
    if task.callback is None or task.status not in {"succeeded", "failed", "cancelled"}:
        return None
    result = ASYNC_TASK_STORE.load_result(task_id)
    try:
        return deliver_async_task_callback(
            task_store=ASYNC_TASK_STORE,
            task=task,
            result=result,
            force=force,
            requested_by=requested_by,
            audit_hook=_append_ops_audit_event,
        )
    except Exception as exc:
        logger.exception("Async task callback delivery failed task_id=%s", task_id)
        ASYNC_TASK_STORE.append_event(
            task_id,
            "callback_error",
            {"error": str(exc), "requested_by": requested_by},
        )
        _append_ops_audit_event(
            "task.callback.delivery",
            resource_type="async_task",
            resource_id=task_id,
            status="error",
            payload={"error": str(exc)},
            metadata={"requested_by": requested_by},
            tenant_id=task.tenant_id,
        )
        return {"configured": True, "status": "error", "error": str(exc)}


def run_async_parse_task(task_id: str) -> dict[str, object]:
    task = ASYNC_TASK_STORE.load_task(task_id)
    task_auth_context = {
        "subject": task.auth_subject,
        "tenant_id": task.tenant_id,
        "mode": "async_task",
        "is_admin": False,
        "scopes": [],
    }
    if task.status == "cancel_requested":
        cancelled = task.model_copy(
            update={
                "status": "cancelled",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        ASYNC_TASK_STORE.write_task(cancelled)
        ASYNC_TASK_STORE.append_event(task_id, "cancelled", {"reason": "cancel_requested_before_start"})
        _append_ops_audit_event(
            "parse.async.complete",
            resource_type="async_task",
            resource_id=task_id,
            status="cancelled",
            payload={"reason": "cancel_requested_before_start", "file_count": len(task.input_files)},
            tenant_id=task.tenant_id,
            auth_context=task_auth_context,
        )
        _deliver_terminal_async_task_callback(task_id)
        return {"status": "cancelled", "results": []}

    started_at = datetime.now(timezone.utc).isoformat()
    running = task.model_copy(
        update={
            "status": "running",
            "started_at": started_at,
            "updated_at": started_at,
            "heartbeat_at": started_at,
            "last_error": None,
        }
    )
    ASYNC_TASK_STORE.write_task(running)
    ASYNC_TASK_STORE.append_event(task_id, "running", {"file_count": len(running.input_files)})

    results: list[dict[str, object]] = []
    progress_total = max(1, len(running.input_files))

    for index, input_file in enumerate(running.input_files, start=1):
        latest = ASYNC_TASK_STORE.load_task(task_id)
        if latest.status == "cancel_requested":
            cancelled_at = datetime.now(timezone.utc).isoformat()
            cancelled_summary = _merge_async_task_result_summary(
                _async_task_result_summary(results),
                existing_summary=latest.result_summary,
                gpu_page_pool=((latest.metadata or {}).get("gpu_page_pool") or None),
            )
            cancelled = latest.model_copy(
                update={
                    "status": "cancelled",
                    "updated_at": cancelled_at,
                    "finished_at": cancelled_at,
                    "result_available": bool(results),
                    "result_summary": cancelled_summary if results or cancelled_summary else {},
                }
            )
            if results:
                ASYNC_TASK_STORE.save_result(task_id, {"results": results})
            ASYNC_TASK_STORE.write_task(cancelled)
            ASYNC_TASK_STORE.append_event(
                task_id,
                "cancelled",
                {"reason": "cancel_requested_during_run", "completed_files": len(results)},
            )
            _append_ops_audit_event(
                "parse.async.complete",
                resource_type="async_task",
                resource_id=task_id,
                status="cancelled",
                payload={"reason": "cancel_requested_during_run", "completed_files": len(results)},
                tenant_id=task.tenant_id,
                auth_context=task_auth_context,
            )
            _deliver_terminal_async_task_callback(task_id)
            return {"status": "cancelled", "results": results}

        dispatch_summary = None
        try:
            dispatch_summary = _plan_gpu_page_pool_dispatch(input_file, dict(latest.parse_options), task_id=task_id)
        except Exception as exc:
            logger.exception("Failed to plan GPU page pool dispatch task_id=%s filename=%s", task_id, input_file.filename)
            ASYNC_TASK_STORE.append_event(
                task_id,
                "gpu_page_jobs_plan_error",
                {"filename": input_file.filename, "error": str(exc)},
            )
        if dispatch_summary is not None:
            observe_gpu_page_pool_dispatch(
                page_jobs=dispatch_summary.get("page_jobs"),
                device_job_counts=dispatch_summary.get("device_job_counts"),
            )
            planned_task = ASYNC_TASK_STORE.load_task(task_id)
            gpu_page_pool_summary = _accumulate_gpu_page_pool_summary(
                (planned_task.metadata or {}).get("gpu_page_pool") if isinstance(planned_task.metadata, dict) else None,
                dispatch_summary,
                filename=input_file.filename,
            )
            ASYNC_TASK_STORE.write_task(
                planned_task.model_copy(
                    update={
                        "metadata": {
                            **(planned_task.metadata or {}),
                            "gpu_page_pool": gpu_page_pool_summary,
                        }
                    }
                )
            )
            ASYNC_TASK_STORE.append_event(
                task_id,
                "gpu_page_jobs_planned",
                {
                    "filename": input_file.filename,
                    **deepcopy(dispatch_summary),
                },
            )

        upload = LocalUploadedFile(input_file.source_path, input_file.filename)
        try:
            result = _parse_single_file(upload, dict(latest.parse_options))
        finally:
            upload.close()
        results.append(result)
        updated_at = datetime.now(timezone.utc).isoformat()
        current_task = ASYNC_TASK_STORE.load_task(task_id)
        current = current_task.model_copy(
            update={
                "updated_at": updated_at,
                "heartbeat_at": updated_at,
                "result_summary": _merge_async_task_result_summary(
                    _async_task_result_summary(results),
                    existing_summary=current_task.result_summary,
                    progress={"current": index, "total": progress_total},
                    gpu_page_pool=((current_task.metadata or {}).get("gpu_page_pool") or None),
                ),
            }
        )
        ASYNC_TASK_STORE.write_task(current)
        ASYNC_TASK_STORE.append_event(
            task_id,
            "file_completed",
            {
                "index": index,
                "total": progress_total,
                "filename": input_file.filename,
                "has_error": "error" in result,
                "parse_id": result.get("parse_id"),
            },
        )
        if current.status == "cancel_requested":
            cancelled_at = datetime.now(timezone.utc).isoformat()
            summary = _merge_async_task_result_summary(
                _async_task_result_summary(results),
                existing_summary=current.result_summary,
                gpu_page_pool=((current.metadata or {}).get("gpu_page_pool") or None),
            )
            cancelled = current.model_copy(
                update={
                    "status": "cancelled",
                    "updated_at": cancelled_at,
                    "finished_at": cancelled_at,
                    "heartbeat_at": cancelled_at,
                    "result_available": bool(results),
                    "result_summary": summary,
                }
            )
            if results:
                ASYNC_TASK_STORE.save_result(task_id, {"results": results})
            ASYNC_TASK_STORE.write_task(cancelled)
            ASYNC_TASK_STORE.append_event(
                task_id,
                "cancelled",
                {"reason": "cancel_requested_after_file", "completed_files": len(results)},
            )
            _append_ops_audit_event(
                "parse.async.complete",
                resource_type="async_task",
                resource_id=task_id,
                status="cancelled",
                payload={"reason": "cancel_requested_after_file", "completed_files": len(results)},
                tenant_id=task.tenant_id,
                auth_context=task_auth_context,
            )
            _deliver_terminal_async_task_callback(task_id)
            return {"status": "cancelled", "results": results}

    finished_at = datetime.now(timezone.utc).isoformat()
    latest_task = ASYNC_TASK_STORE.load_task(task_id)
    summary = _merge_async_task_result_summary(
        _async_task_result_summary(results),
        existing_summary=latest_task.result_summary,
        gpu_page_pool=((latest_task.metadata or {}).get("gpu_page_pool") or None),
    )
    terminal_status = "failed" if summary["error_count"] == summary["file_count"] else "succeeded"
    completed = latest_task.model_copy(
        update={
            "status": terminal_status,
            "updated_at": finished_at,
            "finished_at": finished_at,
            "heartbeat_at": finished_at,
            "result_available": True,
            "result_summary": summary,
            "last_error": None if terminal_status == "succeeded" else "all files failed",
        }
    )
    ASYNC_TASK_STORE.save_result(task_id, {"results": results})
    ASYNC_TASK_STORE.write_task(completed)
    ASYNC_TASK_STORE.append_event(
        task_id,
        terminal_status,
        {
            "file_count": summary["file_count"],
            "success_count": summary["success_count"],
            "error_count": summary["error_count"],
        },
    )
    _append_ops_audit_event(
        "parse.async.complete",
        resource_type="async_task",
        resource_id=task_id,
        status=terminal_status,
        payload={
            "file_count": summary["file_count"],
            "success_count": summary["success_count"],
            "error_count": summary["error_count"],
            "parse_ids": [str(result.get("parse_id") or "") for result in results if str(result.get("parse_id") or "")],
        },
        tenant_id=task.tenant_id,
        auth_context=task_auth_context,
    )
    _deliver_terminal_async_task_callback(task_id)
    return {"status": terminal_status, "results": results}


if __name__ == "__main__":
    if os.environ.get("DEEPDOC_PDF_PARSER", "").strip().lower() == "plain":
        logger.info("Skipping OCR/layout model load (DEEPDOC_PDF_PARSER=plain).")
    else:
        load_models()
        warmup_models()
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8000"))
    logger.info("Starting Flask server on %s:%s", host, port)
    http_server = WSGIServer((host, port), app)
    http_server.serve_forever()

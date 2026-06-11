from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterator
from urllib.parse import urlparse

from common import logger


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_csv(value: str | None) -> list[str]:
    items: list[str] = []
    for raw_item in str(value or "").split(","):
        normalized = raw_item.strip()
        if normalized:
            items.append(normalized)
    return items


def _clamp_sample_ratio(value: str | None, default: float = 1.0) -> float:
    try:
        ratio = float(value or default)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, ratio))


def _package_version() -> str | None:
    try:
        return version("deepdoc")
    except PackageNotFoundError:
        return None


def _normalize_otlp_endpoint(value: str | None) -> str | None:
    endpoint = str(value or "").strip()
    if not endpoint:
        return None
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return endpoint
    if parsed.path and parsed.path != "/":
        return endpoint
    suffix = "/v1/traces"
    return endpoint.rstrip("/") + suffix


def _parse_otlp_headers(value: str | None) -> dict[str, str] | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    if raw_value.startswith("{"):
        try:
            payload = json.loads(raw_value)
        except Exception:
            logger.exception("Invalid DEEPDOC_TRACING_OTLP_HEADERS, expected JSON object or key=value pairs")
            return None
        if not isinstance(payload, dict):
            logger.warning("Invalid DEEPDOC_TRACING_OTLP_HEADERS type: expected object")
            return None
        headers = {str(key).strip(): str(val).strip() for key, val in payload.items() if str(key).strip()}
        return headers or None
    headers: dict[str, str] = {}
    for item in raw_value.split(","):
        if "=" not in item:
            continue
        key, raw_header_value = item.split("=", 1)
        header_key = key.strip()
        header_value = raw_header_value.strip()
        if header_key and header_value:
            headers[header_key] = header_value
    return headers or None


def _normalize_attribute_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        normalized_items = [
            item
            for item in (_normalize_attribute_value(item) for item in value)
            if isinstance(item, (bool, int, float, str))
        ]
        return normalized_items if normalized_items else None
    return str(value)


@dataclass(frozen=True)
class TracingState:
    enabled: bool
    exporters: tuple[str, ...]
    service_name: str
    endpoint: str | None
    sample_ratio: float
    excluded_urls: str | None
    instrumented: bool
    log_correlation: bool
    error: str | None = None
    provider: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "exporters": list(self.exporters),
            "service_name": self.service_name,
            "endpoint": self.endpoint,
            "sample_ratio": self.sample_ratio,
            "excluded_urls": self.excluded_urls,
            "instrumented": self.instrumented,
            "log_correlation": self.log_correlation,
            "error": self.error,
        }


_TRACING_STATE = TracingState(
    enabled=False,
    exporters=tuple(),
    service_name="deepdoc-standalone",
    endpoint=None,
    sample_ratio=1.0,
    excluded_urls=None,
    instrumented=False,
    log_correlation=True,
)
_TRACING_APP_IDS: set[int] = set()
_REQUESTS_INSTRUMENTED = False
_BOTOCORE_INSTRUMENTED = False
_PSYCOPG_INSTRUMENTED = False


def get_tracing_state() -> TracingState:
    return _TRACING_STATE


def tracing_enabled() -> bool:
    return bool(_TRACING_STATE.enabled and _TRACING_STATE.instrumented)


def _build_resource_attributes(service_name: str) -> dict[str, str]:
    attributes = {
        "service.name": service_name,
    }
    service_version = (os.environ.get("DEEPDOC_TRACING_SERVICE_VERSION") or "").strip() or _package_version()
    if service_version:
        attributes["service.version"] = service_version
    deployment_environment = (
        os.environ.get("DEEPDOC_TRACING_DEPLOYMENT_ENVIRONMENT")
        or os.environ.get("DEEPDOC_ENVIRONMENT")
        or ""
    ).strip()
    if deployment_environment:
        attributes["deployment.environment"] = deployment_environment
    service_namespace = (os.environ.get("DEEPDOC_TRACING_SERVICE_NAMESPACE") or "").strip()
    if service_namespace:
        attributes["service.namespace"] = service_namespace
    return attributes


def initialize_tracing(app=None) -> TracingState:
    global _TRACING_STATE, _REQUESTS_INSTRUMENTED, _BOTOCORE_INSTRUMENTED, _PSYCOPG_INSTRUMENTED

    requested_exporters = tuple(dict.fromkeys(_parse_csv(os.environ.get("DEEPDOC_TRACING_EXPORTER") or "none")))
    endpoint = _normalize_otlp_endpoint(os.environ.get("DEEPDOC_TRACING_OTLP_ENDPOINT"))
    if not requested_exporters:
        requested_exporters = ("otlp",) if endpoint else ("none",)
    enabled = _parse_bool(
        os.environ.get("DEEPDOC_TRACING_ENABLED"),
        default=any(exporter != "none" for exporter in requested_exporters),
    )
    service_name = (os.environ.get("DEEPDOC_TRACING_SERVICE_NAME") or "deepdoc-standalone").strip() or "deepdoc-standalone"
    sample_ratio = _clamp_sample_ratio(os.environ.get("DEEPDOC_TRACING_SAMPLE_RATIO"), default=1.0)
    excluded_urls = (os.environ.get("DEEPDOC_TRACING_EXCLUDED_URLS") or "/health,/metrics").strip() or None

    if not enabled or all(exporter == "none" for exporter in requested_exporters):
        _TRACING_STATE = TracingState(
            enabled=False,
            exporters=tuple(),
            service_name=service_name,
            endpoint=endpoint,
            sample_ratio=sample_ratio,
            excluded_urls=excluded_urls,
            instrumented=False,
            log_correlation=True,
        )
        return _TRACING_STATE

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
        from opentelemetry.instrumentation.flask import FlaskInstrumentor
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except Exception as exc:
        logger.exception("Tracing requested but OpenTelemetry dependencies are unavailable")
        _TRACING_STATE = TracingState(
            enabled=False,
            exporters=requested_exporters,
            service_name=service_name,
            endpoint=endpoint,
            sample_ratio=sample_ratio,
            excluded_urls=excluded_urls,
            instrumented=False,
            log_correlation=True,
            error=str(exc),
            provider="opentelemetry",
        )
        return _TRACING_STATE

    try:
        provider = trace.get_tracer_provider()
        provider_type = provider.__class__.__module__ + "." + provider.__class__.__name__
        if provider_type == "opentelemetry.trace.ProxyTracerProvider":
            provider = TracerProvider(
                sampler=ParentBased(TraceIdRatioBased(sample_ratio)),
                resource=Resource.create(_build_resource_attributes(service_name)),
            )
            for exporter in requested_exporters:
                normalized_exporter = exporter.strip().lower()
                if normalized_exporter == "none":
                    continue
                if normalized_exporter == "console":
                    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
                    continue
                if normalized_exporter == "otlp":
                    provider.add_span_processor(
                        BatchSpanProcessor(
                            OTLPSpanExporter(
                                endpoint=endpoint,
                                headers=_parse_otlp_headers(os.environ.get("DEEPDOC_TRACING_OTLP_HEADERS")),
                                timeout=max(1, int(os.environ.get("DEEPDOC_TRACING_OTLP_TIMEOUT", "10"))),
                            )
                        )
                    )
                    continue
                raise ValueError(f"Unsupported DEEPDOC_TRACING_EXPORTER: {exporter}")
            trace.set_tracer_provider(provider)

        if app is not None and id(app) not in _TRACING_APP_IDS:
            FlaskInstrumentor().instrument_app(
                app,
                excluded_urls=excluded_urls,
                request_hook=_flask_request_hook,
            )
            _TRACING_APP_IDS.add(id(app))
        if not _REQUESTS_INSTRUMENTED:
            RequestsInstrumentor().instrument()
            _REQUESTS_INSTRUMENTED = True
        if not _BOTOCORE_INSTRUMENTED:
            try:
                BotocoreInstrumentor().instrument()
                _BOTOCORE_INSTRUMENTED = True
            except Exception:
                logger.exception("Failed to instrument botocore tracing")
        if not _PSYCOPG_INSTRUMENTED:
            try:
                PsycopgInstrumentor().instrument()
                _PSYCOPG_INSTRUMENTED = True
            except Exception:
                logger.exception("Failed to instrument psycopg tracing")
    except Exception as exc:
        logger.exception("Failed to initialize tracing")
        _TRACING_STATE = TracingState(
            enabled=False,
            exporters=requested_exporters,
            service_name=service_name,
            endpoint=endpoint,
            sample_ratio=sample_ratio,
            excluded_urls=excluded_urls,
            instrumented=False,
            log_correlation=True,
            error=str(exc),
            provider="opentelemetry",
        )
        return _TRACING_STATE

    _TRACING_STATE = TracingState(
        enabled=True,
        exporters=tuple(exporter for exporter in requested_exporters if exporter != "none"),
        service_name=service_name,
        endpoint=endpoint,
        sample_ratio=sample_ratio,
        excluded_urls=excluded_urls,
        instrumented=True,
        log_correlation=True,
        provider="opentelemetry",
    )
    logger.info(
        "Tracing initialized exporters=%s service_name=%s endpoint=%s sample_ratio=%s",
        ",".join(_TRACING_STATE.exporters) or "none",
        _TRACING_STATE.service_name,
        _TRACING_STATE.endpoint or "-",
        _TRACING_STATE.sample_ratio,
    )
    return _TRACING_STATE


def _flask_request_hook(span, environ) -> None:
    if span is None or not span.is_recording():
        return
    request_id = str(environ.get("HTTP_X_REQUEST_ID") or "").strip()
    if request_id:
        span.set_attribute("deepdoc.request_id", request_id)
    tenant_id = str(environ.get("HTTP_X_TENANT_ID") or "").strip()
    if tenant_id:
        span.set_attribute("deepdoc.tenant_id", tenant_id)


def _set_span_attributes(span, attributes: dict[str, Any] | None) -> None:
    if span is None or attributes is None or not span.is_recording():
        return
    for key, raw_value in attributes.items():
        normalized = _normalize_attribute_value(raw_value)
        if normalized is None:
            continue
        span.set_attribute(str(key), normalized)


@contextmanager
def trace_operation(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    kind=None,
) -> Iterator[Any]:
    if not tracing_enabled():
        yield None
        return
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    tracer = trace.get_tracer("deepdoc")
    span_kwargs = {"kind": kind} if kind is not None else {}
    with tracer.start_as_current_span(name, **span_kwargs) as span:
        _set_span_attributes(span, attributes)
        try:
            yield span
        except Exception as exc:
            if span is not None and span.is_recording():
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    if not tracing_enabled():
        return
    from opentelemetry import trace

    span = trace.get_current_span()
    if span is None or not span.is_recording():
        return
    span.add_event(name, {key: _normalize_attribute_value(val) for key, val in (attributes or {}).items() if _normalize_attribute_value(val) is not None})


def set_current_span_attributes(attributes: dict[str, Any] | None = None) -> None:
    if not tracing_enabled():
        return
    from opentelemetry import trace

    span = trace.get_current_span()
    _set_span_attributes(span, attributes)

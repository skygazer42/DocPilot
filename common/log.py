#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import os
import logging
import threading
import time
import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from common import setting

LOG_DIR = setting.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("deepdoc_log_context", default={})


def _log_format() -> str:
    return (os.environ.get("DEEPDOC_LOG_FORMAT") or "text").strip().lower()


def _request_id_from_flask_context() -> str | None:
    try:
        from flask import has_request_context, request

        if has_request_context():
            return (request.headers.get("X-Request-ID") or "").strip() or None
    except Exception:
        pass
    return None


def _auth_context_from_flask_context() -> dict[str, Any]:
    try:
        from flask import g, has_app_context

        if has_app_context():
            context = getattr(g, "auth_context", None)
            if isinstance(context, dict):
                return context
    except Exception:
        pass
    return {}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "trace_id"):
            _TraceContextFilter().filter(record)
        message = record.getMessage()
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
            "source_file": record.filename,
            "source_line": record.lineno,
            "source_function": record.funcName,
            "trace_id": getattr(record, "trace_id", "-"),
            "span_id": getattr(record, "span_id", "-"),
            "trace_sampled": getattr(record, "trace_sampled", "false"),
        }
        context = dict(_LOG_CONTEXT.get({}) or {})
        request_id = context.pop("request_id", None) or _request_id_from_flask_context()
        if request_id:
            payload["request_id"] = str(request_id)
        auth_context = _auth_context_from_flask_context()
        auth_fields = {
            "tenant_id": auth_context.get("tenant_id"),
            "auth_subject": auth_context.get("subject"),
            "auth_mode": auth_context.get("mode"),
            "auth_scopes": auth_context.get("scopes"),
            "auth_is_admin": auth_context.get("is_admin"),
        }
        for key, value in auth_fields.items():
            if value in (None, "", []):
                continue
            payload[key] = value
        for key, value in context.items():
            if value is None:
                continue
            payload[str(key)] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class Log:
    """
    单例日志类
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.logger = logging.getLogger("app_logger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        trace_context_filter = _TraceContextFilter()

        text_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-7s - [trace_id=%(trace_id)s span_id=%(span_id)s sampled=%(trace_sampled)s] "
            "%(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        formatter = JsonLogFormatter() if _log_format() == "json" else text_formatter

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(trace_context_filter)
        self.logger.addHandler(console_handler)
        self.logger.addFilter(trace_context_filter)

        # 文件输出
        self.current_log_file = None
        self.file_handler = None
        self.formatter = formatter
        self.file_logging_disabled = False
        self.update_file_handler()

    def update_file_handler(self):
        if self.file_logging_disabled:
            return
        folder_name = os.path.join(LOG_DIR, time.strftime("%Y-%m-%d"))
        log_file_path = os.path.join(folder_name, f"{time.strftime('%H')}.log")

        try:
            os.makedirs(folder_name, exist_ok=True)
            if self.current_log_file != log_file_path:
                if self.file_handler:
                    self.logger.removeHandler(self.file_handler)
                    self.file_handler.close()

                fh = logging.FileHandler(log_file_path, encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(self.formatter)
                fh.addFilter(_TraceContextFilter())
                self.logger.addHandler(fh)

                self.file_handler = fh
                self.current_log_file = log_file_path
        except OSError as exc:
            if self.file_handler:
                self.logger.removeHandler(self.file_handler)
                self.file_handler.close()
                self.file_handler = None
            self.current_log_file = None
            self.file_logging_disabled = True
            self.logger.warning("File logging disabled: %s", exc)

    def _write(self, level, message, *args, **kwargs):
        self.update_file_handler()
        stacklevel = kwargs.pop("stacklevel", 3)
        getattr(self.logger, level)(message, *args, stacklevel=stacklevel, **kwargs)

    @contextmanager
    def context(self, **fields: Any):
        current = dict(_LOG_CONTEXT.get({}) or {})
        merged = {**current, **{key: value for key, value in fields.items() if value is not None}}
        token = _LOG_CONTEXT.set(merged)
        try:
            yield
        finally:
            _LOG_CONTEXT.reset(token)

    def debug(self, message, *args, **kwargs):
        self._write("debug", message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._write("info", message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._write("warning", message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._write("error", message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self._write("exception", message, *args, **kwargs)


class _TraceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        trace_id = "-"
        span_id = "-"
        trace_sampled = "false"
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            if span is not None:
                context = span.get_span_context()
                if context is not None and context.is_valid:
                    trace_id = format(context.trace_id, "032x")
                    span_id = format(context.span_id, "016x")
                    trace_sampled = "true" if context.trace_flags.sampled else "false"
        except Exception:
            pass
        record.trace_id = trace_id
        record.span_id = span_id
        record.trace_sampled = trace_sampled
        return True

from __future__ import annotations

import re
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    BAD_REQUEST = "BAD_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    NO_FILE_PART = "NO_FILE_PART"
    NO_SELECTED_FILE = "NO_SELECTED_FILE"
    EMPTY_IMAGE_FILE = "EMPTY_IMAGE_FILE"
    INVALID_IMAGE_FILE = "INVALID_IMAGE_FILE"
    MODELS_NOT_INITIALIZED = "MODELS_NOT_INITIALIZED"
    UNSUPPORTED_FILE_EXTENSION = "UNSUPPORTED_FILE_EXTENSION"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PARSER_DEPENDENCY_MISSING = "PARSER_DEPENDENCY_MISSING"
    UPLOAD_SAVE_FAILED = "UPLOAD_SAVE_FAILED"
    ASYNC_PARSE_DISABLED = "ASYNC_PARSE_DISABLED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    USAGE_QUOTA_EXCEEDED = "USAGE_QUOTA_EXCEEDED"
    SERVER_BUSY = "SERVER_BUSY"
    TENANT_OVERRIDE_FORBIDDEN = "TENANT_OVERRIDE_FORBIDDEN"
    JSON_BODY_REQUIRED = "JSON_BODY_REQUIRED"
    INVALID_PAYLOAD = "INVALID_PAYLOAD"
    INVALID_ARTIFACT_ID = "INVALID_ARTIFACT_ID"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    ASSET_NOT_FOUND = "ASSET_NOT_FOUND"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    CALLBACK_NOT_CONFIGURED = "CALLBACK_NOT_CONFIGURED"
    SELF_CHECK_NOT_FOUND = "SELF_CHECK_NOT_FOUND"
    INGEST_NOT_FOUND = "INGEST_NOT_FOUND"


ERROR_MESSAGES: dict[ErrorCode, tuple[str, str]] = {
    ErrorCode.BAD_REQUEST: ("Bad request", "请求参数错误"),
    ErrorCode.UNAUTHORIZED: ("Unauthorized", "未授权"),
    ErrorCode.FORBIDDEN: ("Forbidden", "无权限访问"),
    ErrorCode.NOT_FOUND: ("Not found", "资源不存在"),
    ErrorCode.CONFLICT: ("Conflict", "请求冲突"),
    ErrorCode.INTERNAL_ERROR: ("Internal server error", "服务器内部错误"),
    ErrorCode.SERVICE_UNAVAILABLE: ("Service unavailable", "服务不可用"),
    ErrorCode.NOT_IMPLEMENTED: ("Not implemented", "当前后端不支持该能力"),
    ErrorCode.NO_FILE_PART: ("No file part", "缺少文件字段"),
    ErrorCode.NO_SELECTED_FILE: ("No selected file", "未选择文件"),
    ErrorCode.EMPTY_IMAGE_FILE: ("Empty image file", "图片文件为空"),
    ErrorCode.INVALID_IMAGE_FILE: ("Invalid image file or format not supported", "图片无效或格式不支持"),
    ErrorCode.MODELS_NOT_INITIALIZED: ("Models not initialized", "模型未初始化"),
    ErrorCode.UNSUPPORTED_FILE_EXTENSION: ("Unsupported file extension", "不支持的文件扩展名"),
    ErrorCode.UNSUPPORTED_FILE_TYPE: ("Unsupported file type", "不支持的文件类型"),
    ErrorCode.VALIDATION_ERROR: ("File validation failed", "文件校验失败"),
    ErrorCode.PARSER_DEPENDENCY_MISSING: ("Parser dependency missing", "解析器依赖缺失"),
    ErrorCode.UPLOAD_SAVE_FAILED: ("Failed to save upload", "上传文件保存失败"),
    ErrorCode.ASYNC_PARSE_DISABLED: ("Async parse is disabled", "异步解析未启用"),
    ErrorCode.RATE_LIMIT_EXCEEDED: ("Rate limit exceeded", "请求频率超限"),
    ErrorCode.USAGE_QUOTA_EXCEEDED: ("Usage quota exceeded", "用量配额超限"),
    ErrorCode.SERVER_BUSY: ("Server busy", "服务繁忙"),
    ErrorCode.TENANT_OVERRIDE_FORBIDDEN: ("Tenant override is not allowed", "不允许覆盖租户"),
    ErrorCode.JSON_BODY_REQUIRED: ("JSON body must be an object", "JSON 请求体必须是对象"),
    ErrorCode.INVALID_PAYLOAD: ("Invalid payload", "请求体无效"),
    ErrorCode.INVALID_ARTIFACT_ID: ("Invalid artifact id", "artifact id 无效"),
    ErrorCode.ARTIFACT_NOT_FOUND: ("Artifact not found", "解析产物不存在"),
    ErrorCode.ASSET_NOT_FOUND: ("Asset not found", "资产不存在"),
    ErrorCode.TASK_NOT_FOUND: ("Task not found", "任务不存在"),
    ErrorCode.CALLBACK_NOT_CONFIGURED: ("Task callback is not configured", "任务回调未配置"),
    ErrorCode.SELF_CHECK_NOT_FOUND: ("Self-check not found", "自检记录不存在"),
    ErrorCode.INGEST_NOT_FOUND: ("Ingest record not found", "ingest 记录不存在"),
}


def normalize_error_locale(locale: str | None) -> str:
    raw = str(locale or "").strip()
    if not raw:
        return "en-US"
    first = raw.split(",", 1)[0].strip()
    normalized = first.replace("_", "-").lower()
    if normalized.startswith("zh"):
        return "zh-CN"
    return "en-US"


def infer_error_code(message: str | None, *, status_code: int | None = None) -> ErrorCode:
    text = str(message or "").strip()
    normalized = text.lower()
    if normalized == "unauthorized":
        return ErrorCode.UNAUTHORIZED
    if normalized == "no file part":
        return ErrorCode.NO_FILE_PART
    if normalized == "no selected file":
        return ErrorCode.NO_SELECTED_FILE
    if normalized == "empty image file":
        return ErrorCode.EMPTY_IMAGE_FILE
    if normalized == "invalid image file or format not supported":
        return ErrorCode.INVALID_IMAGE_FILE
    if normalized == "models not initialized":
        return ErrorCode.MODELS_NOT_INITIALIZED
    if normalized == "async parse is disabled":
        return ErrorCode.ASYNC_PARSE_DISABLED
    if normalized == "rate limit exceeded":
        return ErrorCode.RATE_LIMIT_EXCEEDED
    if normalized == "usage quota exceeded":
        return ErrorCode.USAGE_QUOTA_EXCEEDED
    if normalized == "server busy":
        return ErrorCode.SERVER_BUSY
    if normalized == "tenant_id override is not allowed":
        return ErrorCode.TENANT_OVERRIDE_FORBIDDEN
    if normalized == "json body must be an object":
        return ErrorCode.JSON_BODY_REQUIRED
    if normalized == "invalid artifact id":
        return ErrorCode.INVALID_ARTIFACT_ID
    if normalized == "artifact not found":
        return ErrorCode.ARTIFACT_NOT_FOUND
    if normalized == "asset not found":
        return ErrorCode.ASSET_NOT_FOUND
    if normalized == "task not found":
        return ErrorCode.TASK_NOT_FOUND
    if normalized == "task callback is not configured":
        return ErrorCode.CALLBACK_NOT_CONFIGURED
    if normalized == "self-check not found":
        return ErrorCode.SELF_CHECK_NOT_FOUND
    if normalized.startswith("unsupported file extension:"):
        return ErrorCode.UNSUPPORTED_FILE_EXTENSION
    if normalized.startswith("unsupported file type:"):
        return ErrorCode.UNSUPPORTED_FILE_TYPE
    if normalized.startswith("parser dependency missing:"):
        return ErrorCode.PARSER_DEPENDENCY_MISSING
    if normalized.startswith("failed to save upload:"):
        return ErrorCode.UPLOAD_SAVE_FAILED
    if "not supported by the active backend" in normalized:
        return ErrorCode.NOT_IMPLEMENTED
    if normalized.startswith("invalid ") or " invalid " in normalized:
        return ErrorCode.INVALID_PAYLOAD
    if status_code == 401:
        return ErrorCode.UNAUTHORIZED
    if status_code == 403:
        return ErrorCode.FORBIDDEN
    if status_code == 404:
        return ErrorCode.NOT_FOUND
    if status_code == 409:
        return ErrorCode.CONFLICT
    if status_code == 501:
        return ErrorCode.NOT_IMPLEMENTED
    if status_code == 503:
        return ErrorCode.SERVICE_UNAVAILABLE
    if status_code and status_code >= 500:
        return ErrorCode.INTERNAL_ERROR
    return ErrorCode.BAD_REQUEST


def _infer_details(message: str, code: ErrorCode) -> dict[str, Any]:
    if code == ErrorCode.NO_FILE_PART:
        return {"field": "file"}
    if code == ErrorCode.UNSUPPORTED_FILE_EXTENSION:
        match = re.match(r"Unsupported file extension:\s*(?P<filename>.+)$", message)
        return {"filename": match.group("filename").strip()} if match else {}
    if code == ErrorCode.UNSUPPORTED_FILE_TYPE:
        match = re.match(r"Unsupported file type:\s*(?P<file_type>.+)$", message)
        return {"file_type": match.group("file_type").strip()} if match else {}
    return {}


def build_error_payload(
    code: ErrorCode | str,
    *,
    message: str | None = None,
    locale: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error_code = code if isinstance(code, ErrorCode) else ErrorCode(str(code))
    message_en, message_zh = ERROR_MESSAGES[error_code]
    error_message = message or message_en
    return {
        "error": error_message,
        "error_code": error_code.value,
        "message": message_en,
        "message_zh": message_zh,
        "locale": normalize_error_locale(locale),
        "details": details or {},
    }


def enrich_error_payload(value: Any, *, status_code: int | None = None, locale: str | None = None) -> Any:
    if isinstance(value, list):
        return [enrich_error_payload(item, status_code=status_code, locale=locale) for item in value]
    if not isinstance(value, dict):
        return value

    enriched = {
        key: enrich_error_payload(item, status_code=status_code, locale=locale)
        for key, item in value.items()
    }
    if "error" not in enriched or enriched.get("error_code"):
        return enriched

    error_text = str(enriched.get("error") or "")
    code = infer_error_code(error_text, status_code=status_code)
    details = enriched.get("details") if isinstance(enriched.get("details"), dict) else _infer_details(error_text, code)
    error_fields = build_error_payload(code, message=error_text, locale=locale, details=details)
    return {**enriched, **error_fields}

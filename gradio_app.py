# ruff: noqa: E402

import json
import os
import re
import time
import inspect
import mimetypes
import gradio as gr
import requests

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from common import setting
from common import logger
from common.branding import PARSER_CONSOLE_TITLE, PRODUCT_NAME
from common.gradio_temp import cleanup_temp_on_startup, prepare_gradio_temp_dir

load_dotenv(override=False)

GRADIO_TMP_DIR = prepare_gradio_temp_dir(setting.WORK_DIR)
removed_items = cleanup_temp_on_startup(GRADIO_TMP_DIR)
if removed_items > 0:
    logger.info("[Startup] cleaned temp dir=%s removed_items=%d", GRADIO_TMP_DIR, removed_items)
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TMP_DIR))

from gradio_pdf import PDF as GradioPDF

_gradio_major_version = int(gr.__version__.split(".")[0])
IS_GRADIO_6 = _gradio_major_version >= 6

STATUS_TIMER_INTERVAL_SECONDS = 0.25
STATUS_UPDATE_STEP_SECONDS = 0.5
ASYNC_TASK_POLL_SECONDS = max(0.25, float(os.environ.get("GRADIO_ASYNC_TASK_POLL_SECONDS", "1.0")))
PADDLEOCR_VL_ENGINE = "paddleocr_vl"
MARKITDOWN_ENGINE = "markitdown"


STATUS_BOX_AUTOSCROLL_JS = """
(value) => {
    const scrollToBottom = () => {
        const textarea = document.querySelector(".convert-status-box textarea");
        if (!textarea) {
            return;
        }
        textarea.scrollTop = textarea.scrollHeight;
    };
    requestAnimationFrame(() => {
        scrollToBottom();
        requestAnimationFrame(scrollToBottom);
    });
    return [];
}
"""

MINERU_LANG_CHOICES = [
    ("ch (Chinese, English, Chinese Traditional)", "ch"),
    ("en (English)", "en"),
    ("korean (Korean)", "korean"),
    ("japan (Japanese)", "japan"),
    ("chinese_cht (Chinese Traditional)", "chinese_cht"),
]

DEEPDOC_LAYOUT_MODELS = ["manual", "paper", "laws", "general"]
TASK_STATUS_FILTER_CHOICES = [
    ("全部", "all"),
    ("queued", "queued"),
    ("running", "running"),
    ("cancel_requested", "cancel_requested"),
    ("succeeded", "succeeded"),
    ("failed", "failed"),
    ("cancelled", "cancelled"),
]
ARTIFACT_PUBLISH_STATUS_FILTER_CHOICES = [
    ("全部", "all"),
    ("published", "published"),
    ("failed", "failed"),
    ("disabled", "disabled"),
]
SELF_CHECK_STATUS_FILTER_CHOICES = [
    ("全部", "all"),
    ("running", "running"),
    ("passed", "passed"),
    ("failed", "failed"),
]
AUDIT_EVENT_STATUS_FILTER_CHOICES = [
    ("全部", "all"),
    ("ok", "ok"),
    ("partial", "partial"),
    ("queued", "queued"),
    ("succeeded", "succeeded"),
    ("failed", "failed"),
    ("cancel_requested", "cancel_requested"),
    ("cancelled", "cancelled"),
]


def _is_valid_http_url(url: str) -> bool:
    value = (url or "").strip()
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _clean_text(text: str) -> str:
    text = re.sub(r"@@\d+(?:-\d+)?\t[-\d.]+\t[-\d.]+\t[-\d.]+\t[-\d.]+##", "", text)
    return text.strip()


def _results_to_markdown(parse_results) -> str:
    if isinstance(parse_results, tuple) and len(parse_results) >= 1:
        text_content = parse_results[0]
        tables = parse_results[1] if len(parse_results) > 1 else []
    else:
        text_content = parse_results
        tables = []

    parts = []

    if isinstance(text_content, list):
        for box in text_content:
            if isinstance(box, dict) and box.get("text", "").strip():
                parts.append(_clean_text(box["text"]))
            elif isinstance(box, dict) and box.get("block_content", "").strip():
                parts.append(_clean_text(box["block_content"]))
            elif isinstance(box, str) and box.strip():
                parts.append(_clean_text(box))
            elif (
                isinstance(box, (list, tuple))
                and len(box) > 0
                and isinstance(box[0], str)
                and box[0].strip()
            ):
                parts.append(_clean_text(box[0]))
    elif isinstance(text_content, str) and text_content.strip():
        parts.append(_clean_text(text_content))

    if isinstance(tables, list) and tables:
        parts.append("\n\n### Tables\n")
        for table in tables:
            if isinstance(table, str) and table.strip():
                parts.append(table)

    return "\n\n".join(p for p in parts if p)


def _status(*lines: str) -> str:
    return "\n".join([line for line in lines if line])


def _get_request_timeout() -> int:
    return max(1, int(os.environ.get("DEEPDOC_REQUEST_TIMEOUT", "600")))


def _resolve_paddle_api_url(use_gpu: bool) -> str:
    if use_gpu:
        return (os.environ.get("PADDLEOCR_GPU_API_URL", "") or "").strip()
    return (os.environ.get("PADDLEOCR_API_URL", "") or "").strip()


def _resolve_mineru_api_url(use_gpu: bool) -> str:
    if use_gpu:
        return (os.environ.get("MINERU_GPU_API_URL", "") or "").strip()
    return (
        os.environ.get("MINERU_APISERVER") or os.environ.get("MINERU_API_URI", "")
    ).strip()


def _deepdoc_api_base() -> str:
    return (os.environ.get("DEEPDOC_API_BASE") or "http://127.0.0.1:8000").strip().rstrip("/")


def _build_api_headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    bearer_token = (
        os.environ.get("DEEPDOC_GRADIO_AUTH_BEARER_TOKEN")
        or os.environ.get("DEEPDOC_AUTH_BEARER_TOKEN")
        or ""
    ).strip()
    api_key = (
        os.environ.get("DEEPDOC_GRADIO_API_KEY")
        or os.environ.get("SECRET_ACCESS_KEY")
        or ""
    ).strip()
    if bearer_token:
        headers["Authorization"] = (
            bearer_token
            if bearer_token.lower().startswith("bearer ")
            else f"Bearer {bearer_token}"
        )
    elif api_key:
        headers["X-API-Key"] = api_key
    return headers


def _api_timeout() -> tuple[int, int]:
    total = _get_request_timeout()
    connect = min(10, max(3, total))
    read = max(10, total)
    return (connect, read)


def _bool_form(value: bool) -> str:
    return "true" if bool(value) else "false"


def _task_api_url(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return _deepdoc_api_base() + path


def _write_markdown_file(file_path: str, markdown_content: str) -> str:
    base_name = Path(file_path).stem or "output"
    safe_base_name = re.sub(r"[^\w\-.]+", "_", base_name).strip("._") or "output"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_path = GRADIO_TMP_DIR / f"{safe_base_name}_{timestamp}.md"
    output_path.write_text(markdown_content, encoding="utf-8")
    return str(output_path)


def _cleanup_runtime_tmp_files(
    tmp_dir: Path,
    min_age_seconds: int = 600,
    keep_paths: set[str] | None = None,
) -> int:
    now = time.time()
    removed = 0
    keep_resolved: set[str] = set()
    if keep_paths:
        for item in keep_paths:
            if not item:
                continue
            try:
                keep_resolved.add(str(Path(item).resolve()))
            except Exception:
                continue
    if not tmp_dir.exists():
        return 0
    for file_path in tmp_dir.iterdir():
        if not file_path.is_file() or not file_path.name.startswith("tmp"):
            continue
        try:
            resolved = str(file_path.resolve())
        except Exception:
            resolved = str(file_path)
        if resolved in keep_resolved:
            continue
        try:
            stat = file_path.stat()
        except FileNotFoundError:
            continue
        if stat.st_size != 0 or (now - stat.st_mtime) < min_age_seconds:
            continue
        try:
            file_path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def _prepare_pdf_preview_path(file_path) -> str | None:
    candidate_path = file_path
    if isinstance(file_path, dict):
        candidate_path = (
            file_path.get("path")
            or file_path.get("name")
            or file_path.get("orig_name")
            or ""
        )
    elif hasattr(file_path, "path"):
        candidate_path = getattr(file_path, "path", "")

    candidate_text = str(candidate_path or "").strip()
    if not candidate_text:
        logger.info("[Preview] skip: empty upload path")
        return None

    src = Path(candidate_text)
    if not src.exists():
        logger.warning(
            "[Preview] invalid source path=%s suffix=%s exists=%s",
            str(src),
            src.suffix.lower(),
            src.exists(),
        )
        return None

    is_pdf = src.suffix.lower() == ".pdf"
    if not is_pdf:
        try:
            with src.open("rb") as f:
                is_pdf = f.read(5).startswith(b"%PDF")
        except Exception:
            is_pdf = False

    if not is_pdf:
        logger.warning(
            "[Preview] invalid pdf source path=%s suffix=%s",
            str(src),
            src.suffix.lower(),
        )
        return None

    logger.info(
        "[Preview] source accepted path=%s size=%s",
        str(src),
        src.stat().st_size,
    )
    return str(src)


def _update_pdf_preview(file_path: str):
    logger.info("[Preview] update requested upload_path=%s", file_path)
    preview_path = _prepare_pdf_preview_path(file_path)
    if not preview_path:
        logger.warning("[Preview] update empty value upload_path=%s", file_path)
        return gr.update(value=None, visible=True)
    logger.info("[Preview] update value preview_path=%s", preview_path)
    return gr.update(value=preview_path, visible=True)


def _build_async_parse_form(
    parser_mode: str,
    deepdoc_layout_model: str,
    deepdoc_max_pages: int,
    paddle_use_gpu: bool,
    paddle_prettify_markdown: bool,
    paddle_show_formula_number: bool,
    paddle_use_formula_recognition: bool,
    paddle_table_enable: bool,
    paddle_seal_enable: bool,
    mineru_max_pages: int,
    mineru_use_gpu: bool,
    mineru_language: str,
    mineru_is_ocr: bool,
    mineru_formula_enable: bool,
    mineru_table_enable: bool,
) -> dict[str, str]:
    form: dict[str, str] = {
        "parser_engine": parser_mode,
        "compute_device": "gpu" if (paddle_use_gpu or mineru_use_gpu) else "cpu",
        "persist_artifacts": "true",
        "include_chunks": "true",
        "publish_ingest": "true",
        "reuse_artifacts": "true",
        "return_structured": "false",
    }
    if parser_mode == "deepdoc":
        form["deepdoc_layout_model"] = str(deepdoc_layout_model or "general")
        form["deepdoc_max_pages"] = str(max(1, int(deepdoc_max_pages)))
    elif parser_mode == PADDLEOCR_VL_ENGINE:
        form["compute_device"] = "gpu" if bool(paddle_use_gpu) else "cpu"
        form["paddle_prettify_markdown"] = _bool_form(paddle_prettify_markdown)
        form["paddle_show_formula_number"] = _bool_form(paddle_show_formula_number)
        form["paddle_use_formula_recognition"] = _bool_form(paddle_use_formula_recognition)
        form["paddle_table_enable"] = _bool_form(paddle_table_enable)
        form["paddle_seal_enable"] = _bool_form(paddle_seal_enable)
    elif parser_mode == "mineru":
        form["compute_device"] = "gpu" if bool(mineru_use_gpu) else "cpu"
        form["mineru_max_pages"] = str(max(1, int(mineru_max_pages)))
        form["mineru_language"] = str(mineru_language or "ch")
        form["mineru_is_ocr"] = _bool_form(mineru_is_ocr)
        form["mineru_formula_enable"] = _bool_form(mineru_formula_enable)
        form["mineru_table_enable"] = _bool_form(mineru_table_enable)
    return form


def _submit_async_parse(
    file_path: str,
    parser_mode: str,
    deepdoc_layout_model: str,
    deepdoc_max_pages: int,
    paddle_use_gpu: bool,
    paddle_prettify_markdown: bool,
    paddle_show_formula_number: bool,
    paddle_use_formula_recognition: bool,
    paddle_table_enable: bool,
    paddle_seal_enable: bool,
    mineru_max_pages: int,
    mineru_use_gpu: bool,
    mineru_language: str,
    mineru_is_ocr: bool,
    mineru_formula_enable: bool,
    mineru_table_enable: bool,
) -> dict:
    url = _task_api_url("/api/v1/parse/async")
    form = _build_async_parse_form(
        parser_mode,
        deepdoc_layout_model,
        deepdoc_max_pages,
        paddle_use_gpu,
        paddle_prettify_markdown,
        paddle_show_formula_number,
        paddle_use_formula_recognition,
        paddle_table_enable,
        paddle_seal_enable,
        mineru_max_pages,
        mineru_use_gpu,
        mineru_language,
        mineru_is_ocr,
        mineru_formula_enable,
        mineru_table_enable,
    )
    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    with Path(file_path).open("rb") as handle:
        response = requests.post(
            url,
            data=form,
            files={"file": (Path(file_path).name, handle, mime_type)},
            headers=_build_api_headers(),
            timeout=_api_timeout(),
        )
    response.raise_for_status()
    return response.json()


def _fetch_async_task(task_id: str) -> dict:
    response = requests.get(
        _task_api_url(f"/api/v1/tasks/{task_id}"),
        params={"include_result": "true"},
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    return response.json()


def _fetch_async_task_events(task_id: str) -> list[dict]:
    response = requests.get(
        _task_api_url(f"/api/v1/tasks/{task_id}/events"),
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    events = payload.get("events")
    return events if isinstance(events, list) else []


def _fetch_async_task_callback_events(task_id: str) -> list[dict]:
    response = requests.get(
        _task_api_url(f"/api/v1/tasks/{task_id}/callback-events"),
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    events = payload.get("events")
    return events if isinstance(events, list) else []


def _safe_fetch_async_task_events(task_id: str) -> list[dict]:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return []
    try:
        return _fetch_async_task_events(normalized_task_id)
    except Exception:
        logger.exception("Failed to fetch async task events task_id=%s", normalized_task_id)
        return []


def _safe_fetch_async_task_callback_events(task_id: str) -> list[dict]:
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return []
    try:
        return _fetch_async_task_callback_events(normalized_task_id)
    except Exception:
        logger.exception("Failed to fetch async task callback events task_id=%s", normalized_task_id)
        return []


def _cancel_async_task_request(task_id: str) -> dict:
    response = requests.post(
        _task_api_url(f"/api/v1/tasks/{task_id}/cancel"),
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    return response.json()


def _retry_async_task_callback_request(task_id: str, *, force: bool = True) -> dict:
    response = requests.post(
        _task_api_url(f"/api/v1/tasks/{task_id}/callback/retry"),
        json={"force": bool(force), "requested_by": "gradio"},
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    return response.json()


def _retry_async_task_request(task_id: str, *, force: bool = True, copy_callback: bool = True) -> dict:
    response = requests.post(
        _task_api_url(f"/api/v1/tasks/{task_id}/retry"),
        json={
            "force": bool(force),
            "copy_callback": bool(copy_callback),
            "requested_by": "gradio",
        },
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    return response.json()


def _task_progress_line(task_payload: dict) -> str | None:
    summary = task_payload.get("result_summary") or {}
    progress = summary.get("progress") or {}
    current = progress.get("current")
    total = progress.get("total")
    if isinstance(current, int) and isinstance(total, int) and total > 0:
        return f"进度: {current}/{total}"
    return None


def _task_primary_result(task_payload: dict) -> dict[str, object] | None:
    result_payload = task_payload.get("result") or {}
    if not isinstance(result_payload, dict):
        return None
    results = result_payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    primary = results[0]
    return primary if isinstance(primary, dict) else None


def _event_payload_summary(payload: dict) -> str:
    if not isinstance(payload, dict) or not payload:
        return ""
    pairs = []
    for key in ("reason", "filename", "parse_id", "requested_by", "file_count", "success_count", "error_count"):
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        pairs.append(f"{key}={value}")
    if not pairs:
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(payload)
    return " ".join(pairs)


def _render_task_events(events: list[dict], *, limit: int = 20) -> str:
    lines: list[str] = []
    for event in events[-max(1, int(limit)):]:
        created_at = str(event.get("created_at") or "")
        event_type = str(event.get("event_type") or "unknown")
        payload_summary = _event_payload_summary(event.get("payload") or {})
        line = f"{created_at} {event_type}"
        if payload_summary:
            line += f" {payload_summary}"
        lines.append(line)
    return "\n".join(lines)


def _render_task_callback_events(events: list[dict], *, limit: int = 20) -> str:
    lines: list[str] = []
    for event in events[-max(1, int(limit)):]:
        created_at = str(event.get("created_at") or "")
        event_type = str(event.get("event_type") or "callback")
        attempt_no = event.get("attempt_no")
        status = str(event.get("status") or "unknown")
        response_status = event.get("response_status")
        error = str(event.get("error") or "").strip()
        line = f"{created_at} {event_type} attempt={attempt_no} status={status}"
        if response_status is not None:
            line += f" response={response_status}"
        if error:
            line += f" error={error}"
        lines.append(line)
    return "\n".join(lines)


def _restore_latest_task_events() -> str:
    tasks = _fetch_async_tasks(limit=1, status_filter="all")
    if not tasks:
        return ""
    latest_task_id = str(tasks[0].get("task_id") or "").strip()
    if not latest_task_id:
        return ""
    return _render_task_events(_safe_fetch_async_task_events(latest_task_id))


def _absolute_task_url(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return _task_api_url(text)


def _render_artifact_summary(task_payload: dict, primary: dict[str, object] | None) -> str:
    if not primary:
        return ""
    artifact_urls = primary.get("artifact_urls") or {}
    lines = []
    task_id = str(task_payload.get("task_id") or "").strip()
    parse_id = str(primary.get("parse_id") or "").strip()
    document_id = str(primary.get("document_id") or "").strip()
    if task_id:
        lines.append(f"- Task: `{task_id}`")
    if parse_id:
        lines.append(f"- Parse ID: `{parse_id}`")
    if document_id:
        lines.append(f"- Document ID: `{document_id}`")
    if primary.get("chunk_count") is not None:
        lines.append(f"- Chunks: `{primary.get('chunk_count')}`")
    if primary.get("asset_count") is not None:
        lines.append(f"- Assets: `{primary.get('asset_count')}`")
    if isinstance(artifact_urls, dict):
        for label, key in (
            ("Manifest", "manifest_url"),
            ("Markdown", "markdown_url"),
            ("Structured", "structured_url"),
            ("Chunks", "chunks_url"),
            ("Ingest", "ingest_url"),
            ("Assets", "assets_url_prefix"),
            ("Publish Events", "publish_events_url"),
        ):
            absolute = _absolute_task_url(artifact_urls.get(key))
            if absolute:
                lines.append(f"- [{label}]({absolute})")
    ingest_publish = primary.get("ingest_publish") or {}
    if isinstance(ingest_publish, dict) and ingest_publish:
        status = str(ingest_publish.get("status") or "").strip()
        sink_type = str(ingest_publish.get("sink_type") or "").strip()
        if status or sink_type:
            lines.append(f"- Ingest: `{status or 'unknown'}` via `{sink_type or 'unknown'}`")
    return "\n".join(lines)


def _fetch_async_tasks(limit: int = 20, status_filter: str | None = None) -> list[dict]:
    params: dict[str, object] = {"limit": max(1, min(int(limit), 200))}
    normalized_status = str(status_filter or "").strip().lower()
    if normalized_status and normalized_status not in {"all", "*"}:
        params["status"] = normalized_status
    response = requests.get(
        _task_api_url("/api/v1/tasks"),
        params=params,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    tasks = payload.get("tasks")
    return tasks if isinstance(tasks, list) else []


def _format_task_timestamp(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%m-%d %H:%M:%S")
    except Exception:
        return text


def _task_primary_filename(task_payload: dict) -> str:
    input_files = task_payload.get("input_files")
    if isinstance(input_files, list) and input_files:
        first = input_files[0]
        if isinstance(first, dict):
            return str(first.get("filename") or "").strip()
    return ""


def _task_parse_ids(task_payload: dict) -> list[str]:
    summary = task_payload.get("result_summary") or {}
    parse_ids = summary.get("parse_ids")
    if not isinstance(parse_ids, list):
        return []
    return [str(item).strip() for item in parse_ids if str(item).strip()]


def _task_display_label(task_payload: dict) -> str:
    task_id = str(task_payload.get("task_id") or "").strip()
    status = str(task_payload.get("status") or "unknown").strip()
    parser_engine = str(task_payload.get("parser_engine") or "-").strip()
    filename = _task_primary_filename(task_payload) or "-"
    created_at = _format_task_timestamp(task_payload.get("created_at"))
    short_task_id = task_id[:8] if task_id else "-"
    return f"{created_at} | {status:<16} | {parser_engine:<12} | {filename} | {short_task_id}"


def _render_task_list_summary(tasks: list[dict]) -> str:
    if not tasks:
        return "暂无任务。"
    lines = []
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        status = str(task.get("status") or "unknown").strip()
        parser_engine = str(task.get("parser_engine") or "-").strip()
        filename = _task_primary_filename(task) or "-"
        created_at = _format_task_timestamp(task.get("created_at"))
        parse_ids = _task_parse_ids(task)
        parse_suffix = f" parse={parse_ids[0]}" if parse_ids else ""
        lines.append(
            f"- `{created_at}` `{status}` `{parser_engine}` `{filename}` `{task_id}`{parse_suffix}"
        )
    return "\n".join(lines)


def _render_task_detail_summary(task_payload: dict) -> str:
    if not task_payload:
        return "未选择任务。"
    lines = []
    task_id = str(task_payload.get("task_id") or "").strip()
    if task_id:
        lines.append(f"- Task: `{task_id}`")
    status = str(task_payload.get("status") or "").strip()
    if status:
        lines.append(f"- Status: `{status}`")
    parser_engine = str(task_payload.get("parser_engine") or "").strip()
    if parser_engine:
        lines.append(f"- Parser: `{parser_engine}`")
    filename = _task_primary_filename(task_payload)
    if filename:
        lines.append(f"- File: `{filename}`")
    lines.append(f"- Created: `{_format_task_timestamp(task_payload.get('created_at'))}`")
    if task_payload.get("started_at"):
        lines.append(f"- Started: `{_format_task_timestamp(task_payload.get('started_at'))}`")
    if task_payload.get("finished_at"):
        lines.append(f"- Finished: `{_format_task_timestamp(task_payload.get('finished_at'))}`")
    result_summary = task_payload.get("result_summary") or {}
    if isinstance(result_summary, dict):
        file_count = result_summary.get("file_count")
        success_count = result_summary.get("success_count")
        error_count = result_summary.get("error_count")
        if file_count is not None:
            lines.append(f"- Files: `{file_count}`")
        if success_count is not None or error_count is not None:
            lines.append(f"- Result: success=`{success_count or 0}` error=`{error_count or 0}`")
    parse_ids = _task_parse_ids(task_payload)
    if parse_ids:
        lines.append(f"- Parse IDs: `{', '.join(parse_ids)}`")
    metadata = task_payload.get("metadata") or {}
    if isinstance(metadata, dict):
        retry_attempt = metadata.get("retry_attempt")
        if retry_attempt is not None:
            lines.append(f"- Retry Attempt: `{retry_attempt}`")
        original_task_id = str(metadata.get("original_task_id") or "").strip()
        if original_task_id:
            lines.append(f"- Original Task: `{original_task_id}`")
        retried_from_task_id = str(metadata.get("retried_from_task_id") or "").strip()
        if retried_from_task_id:
            lines.append(f"- Retried From: `{retried_from_task_id}`")
        latest_retry_task_id = str(metadata.get("latest_retry_task_id") or "").strip()
        if latest_retry_task_id:
            lines.append(f"- Latest Retry Task: `{latest_retry_task_id}`")
    last_error = str(task_payload.get("last_error") or "").strip()
    if last_error:
        lines.append(f"- Last Error: `{last_error}`")
    callback_state = task_payload.get("callback_state") or {}
    if isinstance(callback_state, dict) and callback_state:
        lines.append(
            "- Callback: "
            + f"status=`{callback_state.get('status') or 'unknown'}` "
            + f"deliveries=`{callback_state.get('delivery_count') or 0}` "
            + f"success=`{callback_state.get('success_count') or 0}` "
            + f"failure=`{callback_state.get('failure_count') or 0}`"
        )
        if callback_state.get("last_response_status") is not None:
            lines.append(f"- Callback Response: `{callback_state.get('last_response_status')}`")
        if callback_state.get("last_error"):
            lines.append(f"- Callback Error: `{callback_state.get('last_error')}`")
    callback_config = task_payload.get("callback") or {}
    if isinstance(callback_config, dict) and callback_config:
        callback_url = str(callback_config.get("url") or "").strip()
        callback_events = callback_config.get("event_types") or []
        lines.append(
            "- Callback Config: "
            + f"url=`{callback_url or '-'}` "
            + f"events=`{', '.join(str(item) for item in callback_events) or '-'}`"
        )
    for label, key in (
        ("Status JSON", "status_url"),
        ("Events", "events_url"),
        ("Stream", "stream_url"),
        ("Cancel", "cancel_url"),
        ("Retry", "retry_url"),
        ("Callback Events", "callback_events_url"),
        ("Callback Retry", "callback_retry_url"),
    ):
        absolute = _absolute_task_url(task_payload.get(key))
        if absolute:
            lines.append(f"- [{label}]({absolute})")
    return "\n".join(lines)


def _task_center_detail_outputs(task_payload: dict | None) -> tuple[str, str, str, str, object, object, object]:
    if not task_payload:
        return (
            "未选择任务。",
            "",
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    events = _safe_fetch_async_task_events(str(task_payload.get("task_id") or ""))
    callback_events = _safe_fetch_async_task_callback_events(str(task_payload.get("task_id") or ""))
    primary = _task_primary_result(task_payload)
    status = str(task_payload.get("status") or "").strip().lower()
    callback_enabled = bool(task_payload.get("callback"))
    return (
        _render_task_detail_summary(task_payload),
        _render_artifact_summary(task_payload, primary),
        _render_task_events(events),
        _render_task_callback_events(callback_events),
        gr.update(interactive=status in {"queued", "running", "cancel_requested"}),
        gr.update(interactive=callback_enabled and status in {"succeeded", "failed", "cancelled"}),
        gr.update(interactive=status in {"succeeded", "failed", "cancelled"}),
    )


def _refresh_task_center(limit: float | int, status_filter: str, selected_task_id: str):
    tasks = _fetch_async_tasks(limit=int(limit or 20), status_filter=status_filter)
    selected = str(selected_task_id or "").strip()
    valid_ids = {str(task.get("task_id") or "").strip() for task in tasks}
    if not selected or selected not in valid_ids:
        selected = str(tasks[0].get("task_id") or "").strip() if tasks else ""
    choices = [(_task_display_label(task), str(task.get("task_id") or "").strip()) for task in tasks]
    task_payload = _fetch_async_task(selected) if selected else None
    detail_summary, artifact_summary, events_text, callback_events_text, cancel_update, callback_retry_update, retry_update = _task_center_detail_outputs(task_payload)
    return (
        _render_task_list_summary(tasks),
        gr.update(choices=choices, value=selected or None),
        detail_summary,
        artifact_summary,
        events_text,
        callback_events_text,
        cancel_update,
        callback_retry_update,
        retry_update,
        selected,
    )


def _inspect_task_center_task(task_id: str):
    selected = str(task_id or "").strip()
    if not selected:
        return "未选择任务。", "", "", "", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), ""
    task_payload = _fetch_async_task(selected)
    detail_summary, artifact_summary, events_text, callback_events_text, cancel_update, callback_retry_update, retry_update = _task_center_detail_outputs(task_payload)
    return detail_summary, artifact_summary, events_text, callback_events_text, cancel_update, callback_retry_update, retry_update, selected


def _cleanup_task_center_tasks(
    statuses: list[str] | None,
    keep_latest: float | int | None,
    older_than_days: float | int | None,
    include_active: bool,
    dry_run: bool,
    cleanup_limit: float | int,
    task_limit: float | int,
    task_status_filter: str,
    selected_task_id: str,
):
    payload: dict[str, object] = {
        "dry_run": bool(dry_run),
        "include_active": bool(include_active),
        "limit": max(1, min(int(cleanup_limit or 100), 5000)),
    }
    normalized_statuses = [str(item).strip().lower() for item in (statuses or []) if str(item).strip()]
    if normalized_statuses:
        payload["statuses"] = normalized_statuses
    if keep_latest is not None and str(keep_latest).strip():
        payload["keep_latest"] = max(0, int(keep_latest))
    if older_than_days is not None and str(older_than_days).strip():
        payload["older_than_days"] = max(0, int(older_than_days))
    response = requests.post(
        _task_api_url("/api/v1/tasks/cleanup"),
        json=payload,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    cleanup_result = response.json()
    matched = cleanup_result.get("matched")
    deleted = cleanup_result.get("deleted")
    mode = "dry-run" if cleanup_result.get("dry_run") else "applied"
    summary_lines = [
        f"任务清理 `{mode}`",
        f"matched=`{matched}` deleted=`{deleted}` scanned=`{cleanup_result.get('scanned')}`",
    ]
    if cleanup_result.get("tasks"):
        for item in cleanup_result.get("tasks")[:10]:
            if not isinstance(item, dict):
                continue
            summary_lines.append(
                f"- `{item.get('status')}` `{item.get('task_id')}` finished=`{_format_task_timestamp(item.get('finished_at'))}`"
            )
    (
        task_list_md,
        task_selector_update,
        detail_summary,
        artifact_summary,
        events_text,
        callback_events_text,
        cancel_update,
        callback_retry_update,
        retry_update,
        next_selected_task_id,
    ) = _refresh_task_center(task_limit, task_status_filter, selected_task_id)
    return (
        _status(*summary_lines),
        task_list_md,
        task_selector_update,
        detail_summary,
        artifact_summary,
        events_text,
        callback_events_text,
        cancel_update,
        callback_retry_update,
        retry_update,
        next_selected_task_id,
    )


def _retry_task_center_callback(task_id: str):
    selected = str(task_id or "").strip()
    if not selected:
        return _status("未选择任务"), "", gr.update(interactive=False), ""
    payload = _retry_async_task_callback_request(selected, force=True)
    task_payload = payload.get("task") if isinstance(payload, dict) else None
    if not isinstance(task_payload, dict):
        task_payload = _fetch_async_task(selected)
    callback_state = task_payload.get("callback_state") or {}
    summary_lines = [
        "已重发任务回调",
        f"任务: {selected}",
        f"状态: {callback_state.get('status') or '-'}",
    ]
    delivery = payload.get("delivery") if isinstance(payload, dict) else None
    if isinstance(delivery, dict):
        if delivery.get("delivery_id"):
            summary_lines.append(f"Delivery: {delivery.get('delivery_id')}")
        if delivery.get("response_status") is not None:
            summary_lines.append(f"HTTP: {delivery.get('response_status')}")
        if delivery.get("last_error"):
            summary_lines.append(f"Error: {delivery.get('last_error')}")
    detail_summary, _artifact_summary, _events_text, callback_events_text, _cancel_update, callback_retry_update, _retry_update = _task_center_detail_outputs(
        task_payload
    )
    return _status(*summary_lines), callback_events_text, callback_retry_update, detail_summary


def _retry_task_center_task(task_id: str):
    selected = str(task_id or "").strip()
    if not selected:
        return _status("未选择任务"), "", gr.update(interactive=False), ""
    payload = _retry_async_task_request(selected, force=True, copy_callback=True)
    source_task = payload.get("source_task") if isinstance(payload, dict) else None
    retry_task = payload.get("retry_task") if isinstance(payload, dict) else None
    if not isinstance(source_task, dict):
        source_task = _fetch_async_task(selected)
    retry_task_id = str((retry_task or {}).get("task_id") or "").strip()
    summary_lines = [
        "已提交任务重试",
        f"源任务: {selected}",
        f"重试任务: {retry_task_id or '-'}",
    ]
    detail_summary, _artifact_summary, events_text, _callback_events_text, _cancel_update, _callback_retry_update, retry_update = _task_center_detail_outputs(
        source_task
    )
    return _status(*summary_lines), events_text, retry_update, detail_summary


def _fetch_artifact_manifests(limit: int = 20, publish_status_filter: str | None = None) -> list[dict]:
    params: dict[str, object] = {"limit": max(1, min(int(limit), 200))}
    normalized_status = str(publish_status_filter or "").strip().lower()
    if normalized_status and normalized_status not in {"all", "*"}:
        params["publish_status"] = normalized_status
    response = requests.get(
        _task_api_url("/api/v1/artifacts"),
        params=params,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    return results if isinstance(results, list) else []


def _fetch_artifact_manifest(parse_id: str) -> dict:
    response = requests.get(
        _task_api_url(f"/api/v1/artifacts/{parse_id}/manifest"),
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _fetch_artifact_publish_events(parse_id: str) -> str:
    response = requests.get(
        _task_api_url(f"/api/v1/artifacts/{parse_id}/publish-events"),
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    return response.text.strip()


def _safe_fetch_artifact_publish_events(parse_id: str) -> str:
    normalized_parse_id = str(parse_id or "").strip()
    if not normalized_parse_id:
        return ""
    try:
        return _fetch_artifact_publish_events(normalized_parse_id)
    except Exception:
        logger.exception("Failed to fetch artifact publish events parse_id=%s", normalized_parse_id)
        return ""


def _artifact_publish_state(manifest: dict) -> str:
    metadata = manifest.get("metadata") or {}
    if not isinstance(metadata, dict):
        return "-"
    publish_state = metadata.get("ingest_publish") or {}
    if not isinstance(publish_state, dict):
        return "-"
    return str(publish_state.get("status") or "disabled").strip() or "disabled"


def _artifact_display_label(manifest: dict) -> str:
    parse_id = str(manifest.get("parse_id") or "").strip()
    created_at = _format_task_timestamp(manifest.get("created_at"))
    parser_engine = str(manifest.get("parser_engine") or "-").strip()
    filename = str(manifest.get("filename") or "-").strip()
    publish_state = _artifact_publish_state(manifest)
    return f"{created_at} | {publish_state:<10} | {parser_engine:<12} | {filename} | {parse_id[:12] if parse_id else '-'}"


def _render_artifact_list_summary(manifests: list[dict]) -> str:
    if not manifests:
        return "暂无产物。"
    lines = []
    for manifest in manifests:
        parse_id = str(manifest.get("parse_id") or "").strip()
        filename = str(manifest.get("filename") or "-").strip()
        parser_engine = str(manifest.get("parser_engine") or "-").strip()
        publish_state = _artifact_publish_state(manifest)
        created_at = _format_task_timestamp(manifest.get("created_at"))
        lines.append(
            f"- `{created_at}` `{publish_state}` `{parser_engine}` `{filename}` `{parse_id}`"
        )
    return "\n".join(lines)


def _render_artifact_manifest_detail(manifest: dict) -> str:
    if not manifest:
        return "未选择产物。"
    lines = []
    for label, key in (
        ("Parse ID", "parse_id"),
        ("Document ID", "document_id"),
        ("Filename", "filename"),
        ("Parser", "parser_engine"),
        ("File Type", "file_type"),
    ):
        value = str(manifest.get(key) or "").strip()
        if value:
            lines.append(f"- {label}: `{value}`")
    lines.append(f"- Created: `{_format_task_timestamp(manifest.get('created_at'))}`")
    if manifest.get("chunk_count") is not None:
        lines.append(f"- Chunks: `{manifest.get('chunk_count')}`")
    if manifest.get("asset_count") is not None:
        lines.append(f"- Assets: `{manifest.get('asset_count')}`")
    publish_state = _artifact_publish_state(manifest)
    lines.append(f"- Publish: `{publish_state}`")
    metadata = manifest.get("metadata") or {}
    if isinstance(metadata, dict):
        publish_meta = metadata.get("ingest_publish") or {}
        if isinstance(publish_meta, dict):
            sink = str(publish_meta.get("sink_type") or "").strip()
            last_error = str(publish_meta.get("last_error") or "").strip()
            if sink:
                lines.append(f"- Sink: `{sink}`")
            if last_error:
                lines.append(f"- Last Error: `{last_error}`")
    for label, key in (
        ("Manifest", "manifest_url"),
        ("Markdown", "markdown_url"),
        ("Structured", "structured_url"),
        ("Chunks", "chunks_url"),
        ("Ingest", "ingest_url"),
        ("Assets", "assets_url_prefix"),
        ("Publish Events", "publish_events_url"),
    ):
        absolute = _absolute_task_url(manifest.get(key))
        if absolute:
            lines.append(f"- [{label}]({absolute})")
    return "\n".join(lines)


def _artifact_center_detail_outputs(manifest: dict | None) -> tuple[str, str, object]:
    if not manifest:
        return "未选择产物。", "", gr.update(interactive=False)
    publish_events = _safe_fetch_artifact_publish_events(str(manifest.get("parse_id") or ""))
    return (
        _render_artifact_manifest_detail(manifest),
        publish_events,
        gr.update(interactive=True),
    )


def _refresh_artifact_center(limit: float | int, publish_status_filter: str, selected_parse_id: str):
    manifests = _fetch_artifact_manifests(limit=int(limit or 20), publish_status_filter=publish_status_filter)
    selected = str(selected_parse_id or "").strip()
    valid_ids = {str(manifest.get("parse_id") or "").strip() for manifest in manifests}
    if not selected or selected not in valid_ids:
        selected = str(manifests[0].get("parse_id") or "").strip() if manifests else ""
    choices = [(_artifact_display_label(manifest), str(manifest.get("parse_id") or "").strip()) for manifest in manifests]
    manifest = _fetch_artifact_manifest(selected) if selected else None
    detail_summary, publish_events, republish_update = _artifact_center_detail_outputs(manifest)
    return (
        _render_artifact_list_summary(manifests),
        gr.update(choices=choices, value=selected or None),
        detail_summary,
        publish_events,
        republish_update,
        selected,
    )


def _inspect_artifact_center(parse_id: str):
    selected = str(parse_id or "").strip()
    if not selected:
        return "未选择产物。", "", gr.update(interactive=False), ""
    manifest = _fetch_artifact_manifest(selected)
    detail_summary, publish_events, republish_update = _artifact_center_detail_outputs(manifest)
    return detail_summary, publish_events, republish_update, selected


def _republish_artifact_center(
    parse_id: str,
    artifact_limit: float | int,
    artifact_publish_status_filter: str,
    selected_parse_id: str,
):
    normalized_parse_id = str(parse_id or "").strip()
    if not normalized_parse_id:
        return (
            "未选择产物。",
            *(_refresh_artifact_center(artifact_limit, artifact_publish_status_filter, selected_parse_id))
        )
    response = requests.post(
        _task_api_url(f"/api/v1/artifacts/{normalized_parse_id}/publish"),
        json={"force": True, "requested_by": "gradio-ops"},
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    status_lines = [
        "已重发选中产物",
        f"parse_id={payload.get('parse_id')}",
    ]
    ingest_publish = payload.get("ingest_publish") or {}
    if isinstance(ingest_publish, dict):
        status_lines.append(f"status={ingest_publish.get('status')}")
        status_lines.append(f"sink={ingest_publish.get('sink_type')}")
    (
        artifact_list_md,
        artifact_selector_update,
        artifact_detail_summary,
        artifact_publish_events,
        artifact_republish_btn_update,
        next_selected_parse_id,
    ) = _refresh_artifact_center(artifact_limit, artifact_publish_status_filter, normalized_parse_id)
    return (
        _status(*status_lines),
        artifact_list_md,
        artifact_selector_update,
        artifact_detail_summary,
        artifact_publish_events,
        artifact_republish_btn_update,
        next_selected_parse_id,
    )


def _cleanup_artifact_center(
    keep_latest: float | int | None,
    older_than_days: float | int | None,
    dry_run: bool,
    cleanup_limit: float | int,
    artifact_limit: float | int,
    artifact_publish_status_filter: str,
    selected_parse_id: str,
):
    payload: dict[str, object] = {
        "dry_run": bool(dry_run),
        "limit": max(1, min(int(cleanup_limit or 200), 5000)),
    }
    if keep_latest is not None and str(keep_latest).strip():
        payload["keep_latest"] = max(0, int(keep_latest))
    if older_than_days is not None and str(older_than_days).strip():
        payload["older_than_days"] = max(0, int(older_than_days))
    response = requests.post(
        _task_api_url("/api/v1/artifacts/cleanup"),
        json=payload,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    summary_lines = [
        f"产物清理 `{'dry-run' if payload.get('dry_run') else 'applied'}`",
        f"candidates=`{payload.get('candidate_count')}` deleted=`{len(payload.get('deleted') or [])}` scanned=`{payload.get('total_scanned')}`",
    ]
    for item in (payload.get("candidates") or [])[:10]:
        if not isinstance(item, dict):
            continue
        summary_lines.append(
            f"- `{item.get('parser_engine')}` `{item.get('filename')}` `{item.get('parse_id')}`"
        )
    (
        artifact_list_md,
        artifact_selector_update,
        artifact_detail_summary,
        artifact_publish_events,
        artifact_republish_btn_update,
        next_selected_parse_id,
    ) = _refresh_artifact_center(artifact_limit, artifact_publish_status_filter, selected_parse_id)
    return (
        _status(*summary_lines),
        artifact_list_md,
        artifact_selector_update,
        artifact_detail_summary,
        artifact_publish_events,
        artifact_republish_btn_update,
        next_selected_parse_id,
    )


def _fetch_ops_audit_events(limit: int = 20, status_filter: str | None = None, action_filter: str | None = None) -> list[dict]:
    params: dict[str, object] = {"limit": max(1, min(int(limit), 200))}
    normalized_status = str(status_filter or "").strip().lower()
    if normalized_status and normalized_status not in {"all", "*"}:
        params["status"] = normalized_status
    normalized_action = str(action_filter or "").strip()
    if normalized_action:
        params["action"] = normalized_action
    response = requests.get(
        _task_api_url("/api/v1/audit/events"),
        params=params,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    return results if isinstance(results, list) else []


def _fetch_ops_audit_event(event_id: str) -> dict:
    response = requests.get(
        _task_api_url(f"/api/v1/audit/events/{event_id}"),
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _ops_audit_display_label(result: dict) -> str:
    event_id = str(result.get("event_id") or "").strip()
    created_at = _format_task_timestamp(result.get("created_at"))
    status = str(result.get("status") or "-").strip()
    action = str(result.get("action") or "-").strip()
    resource_type = str(result.get("resource_type") or "-").strip()
    resource_id = str(result.get("resource_id") or "").strip()
    if len(resource_id) > 18:
        resource_id = resource_id[:15] + "..."
    return f"{created_at} | {status:<10} | {action:<24} | {resource_type}:{resource_id or '-'} | {event_id[:12] if event_id else '-'}"


def _render_ops_audit_list_summary(results: list[dict]) -> str:
    if not results:
        return "暂无审计事件。"
    lines = []
    for result in results:
        event_id = str(result.get("event_id") or "").strip()
        created_at = _format_task_timestamp(result.get("created_at"))
        status = str(result.get("status") or "-").strip()
        action = str(result.get("action") or "-").strip()
        resource_type = str(result.get("resource_type") or "-").strip()
        resource_id = str(result.get("resource_id") or "").strip()
        lines.append(f"- `{created_at}` `{status}` `{action}` `{resource_type}` `{resource_id or '-'} `{event_id}")
    return "\n".join(lines)


def _render_ops_audit_detail_summary(result: dict) -> str:
    if not result:
        return "未选择审计事件。"
    lines = []
    for label, key in (
        ("Event ID", "event_id"),
        ("Action", "action"),
        ("Status", "status"),
        ("Resource Type", "resource_type"),
        ("Resource ID", "resource_id"),
        ("Request ID", "request_id"),
        ("Actor", "actor_subject"),
        ("Actor Mode", "actor_mode"),
        ("Tenant", "tenant_id"),
        ("Trace ID", "trace_id"),
    ):
        value = str(result.get(key) or "").strip()
        if value:
            lines.append(f"- {label}: `{value}`")
    lines.append(f"- Created: `{_format_task_timestamp(result.get('created_at'))}`")
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        for label, key in (("HTTP Method", "http_method"), ("Path", "http_path"), ("Remote Addr", "remote_addr")):
            value = str(metadata.get(key) or "").strip()
            if value:
                lines.append(f"- {label}: `{value}`")
    return "\n".join(lines)


def _refresh_ops_audit_center(limit: float | int, status_filter: str, action_filter: str, selected_event_id: str):
    results = _fetch_ops_audit_events(limit=int(limit or 20), status_filter=status_filter, action_filter=action_filter)
    selected = str(selected_event_id or "").strip()
    valid_ids = {str(result.get("event_id") or "").strip() for result in results}
    if not selected or selected not in valid_ids:
        selected = str(results[0].get("event_id") or "").strip() if results else ""
    choices = [(_ops_audit_display_label(result), str(result.get("event_id") or "").strip()) for result in results]
    detail = _fetch_ops_audit_event(selected) if selected else None
    return (
        _render_ops_audit_list_summary(results),
        gr.update(choices=choices, value=selected or None),
        _render_ops_audit_detail_summary(detail),
        _pretty_json(detail or {}),
        selected,
    )


def _inspect_ops_audit_center(event_id: str):
    selected = str(event_id or "").strip()
    if not selected:
        return "未选择审计事件。", "", ""
    detail = _fetch_ops_audit_event(selected)
    return _render_ops_audit_detail_summary(detail), _pretty_json(detail), selected


def _cleanup_ops_audit_center(
    status_filter: str,
    action_filter: str,
    keep_latest: float | int | None,
    older_than_days: float | int | None,
    dry_run: bool,
    cleanup_limit: float | int,
    audit_limit: float | int,
    audit_status_filter: str,
    audit_action_filter: str,
    selected_event_id: str,
):
    payload: dict[str, object] = {
        "dry_run": bool(dry_run),
        "limit": max(1, min(int(cleanup_limit or 200), 5000)),
    }
    normalized_status = str(status_filter or "").strip().lower()
    if normalized_status and normalized_status not in {"all", "*"}:
        payload["status"] = normalized_status
    normalized_action = str(action_filter or "").strip()
    if normalized_action:
        payload["action"] = normalized_action
    if keep_latest is not None and str(keep_latest).strip():
        payload["keep_latest"] = max(0, int(keep_latest))
    if older_than_days is not None and str(older_than_days).strip():
        payload["older_than_days"] = max(1, int(older_than_days))
    response = requests.post(
        _task_api_url("/api/v1/audit/events/cleanup"),
        json=payload,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    summary_lines = [
        f"审计清理 `{'dry-run' if payload.get('dry_run') else 'applied'}`",
        f"event_ids=`{len(payload.get('event_ids') or [])}` deleted=`{payload.get('deleted_count')}` scanned=`{payload.get('scanned')}`",
    ]
    for event_id in (payload.get("event_ids") or [])[:10]:
        summary_lines.append(f"- `{event_id}`")
    (
        audit_list_md,
        audit_selector_update,
        audit_detail_summary,
        audit_raw_json,
        next_selected_event_id,
    ) = _refresh_ops_audit_center(audit_limit, audit_status_filter, audit_action_filter, selected_event_id)
    return (
        _status(*summary_lines),
        audit_list_md,
        audit_selector_update,
        audit_detail_summary,
        audit_raw_json,
        next_selected_event_id,
    )


def _fetch_self_checks(limit: int = 20, status_filter: str | None = None) -> list[dict]:
    params: dict[str, object] = {"limit": max(1, min(int(limit), 200))}
    normalized_status = str(status_filter or "").strip().lower()
    if normalized_status and normalized_status not in {"all", "*"}:
        params["status"] = normalized_status
    response = requests.get(
        _task_api_url("/api/v1/self-checks"),
        params=params,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    return results if isinstance(results, list) else []


def _fetch_self_check(check_id: str) -> dict:
    response = requests.get(
        _task_api_url(f"/api/v1/self-checks/{check_id}"),
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _self_check_display_label(result: dict) -> str:
    check_id = str(result.get("check_id") or "").strip()
    created_at = _format_task_timestamp(result.get("created_at"))
    status = str(result.get("status") or "-").strip()
    suite = str(result.get("suite") or "core").strip()
    duration_ms = result.get("duration_ms")
    duration_text = f"{duration_ms}ms" if duration_ms is not None else "-"
    return f"{created_at} | {status:<8} | {suite:<8} | {duration_text:<8} | {check_id[:12] if check_id else '-'}"


def _render_self_check_list_summary(results: list[dict]) -> str:
    if not results:
        return "暂无生产自检结果。"
    lines: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        check_id = str(result.get("check_id") or "").strip()
        created_at = _format_task_timestamp(result.get("created_at"))
        status = str(result.get("status") or "-").strip()
        suite = str(result.get("suite") or "core").strip()
        summary = str(result.get("summary") or "").strip()
        if len(summary) > 96:
            summary = summary[:93] + "..."
        lines.append(f"- `{created_at}` `{status}` `{suite}` `{check_id}` {summary}")
    return "\n".join(lines)


def _render_self_check_detail_summary(result: dict) -> str:
    if not result:
        return "未选择生产自检。"
    lines: list[str] = []
    for label, key in (
        ("Check ID", "check_id"),
        ("Suite", "suite"),
        ("Status", "status"),
    ):
        value = str(result.get(key) or "").strip()
        if value:
            lines.append(f"- {label}: `{value}`")
    lines.append(f"- Created: `{_format_task_timestamp(result.get('created_at'))}`")
    if result.get("finished_at"):
        lines.append(f"- Finished: `{_format_task_timestamp(result.get('finished_at'))}`")
    if result.get("duration_ms") is not None:
        lines.append(f"- Duration: `{result.get('duration_ms')}ms`")
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        force_reparse = metadata.get("force_reparse")
        force_republish = metadata.get("force_republish")
        tenant_id = str(metadata.get("tenant_id") or "").strip()
        if force_reparse is not None:
            lines.append(f"- Force Reparse: `{bool(force_reparse)}`")
        if force_republish is not None:
            lines.append(f"- Force Republish: `{bool(force_republish)}`")
        if tenant_id:
            lines.append(f"- Tenant: `{tenant_id}`")
    summary = str(result.get("summary") or "").strip()
    if summary:
        lines.append(f"- Summary: {summary}")
    environment = result.get("environment") or {}
    if isinstance(environment, dict):
        ingest_publisher = str(environment.get("ingest_publisher") or "").strip()
        if ingest_publisher:
            lines.append(f"- Ingest Publisher: `{ingest_publisher}`")
    return "\n".join(lines)


def _render_self_check_steps(result: dict) -> str:
    steps = result.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return "暂无自检步骤。"
    lines: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        lines.append(
            f"- `{step.get('name') or '-'}` status=`{step.get('status') or '-'}` duration=`{step.get('duration_ms') or '-'}ms`"
        )
        summary = str(step.get("summary") or "").strip()
        if summary:
            lines.append(f"  {summary}")
        details = step.get("details") or {}
        if isinstance(details, dict):
            detail_parts: list[str] = []
            for key in (
                "cache_hit",
                "artifact_reused",
                "result_count",
                "asset_count",
                "asset_rows",
                "chunk_rows",
                "link_rows",
                "answer_source",
                "answer_asset_count",
                "parse_id",
                "document_id",
            ):
                if key in details and details.get(key) not in (None, "", []):
                    detail_parts.append(f"{key}=`{details.get(key)}`")
            if detail_parts:
                lines.append("  " + " ".join(detail_parts))
    return "\n".join(lines)


def _self_check_center_detail_outputs(result: dict | None) -> tuple[str, str]:
    if not result:
        return "未选择生产自检。", ""
    return (_render_self_check_detail_summary(result), _render_self_check_steps(result))


def _refresh_self_check_center(limit: float | int, status_filter: str, selected_check_id: str):
    results = _fetch_self_checks(limit=int(limit or 20), status_filter=status_filter)
    selected = str(selected_check_id or "").strip()
    valid_ids = {str(item.get("check_id") or "").strip() for item in results}
    if not selected or selected not in valid_ids:
        selected = str(results[0].get("check_id") or "").strip() if results else ""
    choices = [(_self_check_display_label(item), str(item.get("check_id") or "").strip()) for item in results]
    result = _fetch_self_check(selected) if selected else None
    detail_summary, steps_text = _self_check_center_detail_outputs(result)
    return (
        _render_self_check_list_summary(results),
        gr.update(choices=choices, value=selected or None),
        detail_summary,
        steps_text,
        _pretty_json(result or {}),
        selected,
    )


def _inspect_self_check_center(check_id: str):
    selected = str(check_id or "").strip()
    if not selected:
        return "未选择生产自检。", "", "", ""
    result = _fetch_self_check(selected)
    detail_summary, steps_text = _self_check_center_detail_outputs(result)
    return detail_summary, steps_text, _pretty_json(result), selected


def _run_self_check_console(
    force_reparse: bool,
    force_republish: bool,
    self_check_limit: float | int,
    self_check_status_filter: str,
    selected_check_id: str,
):
    response = requests.post(
        _task_api_url("/api/v1/self-checks/run"),
        json={
            "suite": "core",
            "force_reparse": bool(force_reparse),
            "force_republish": bool(force_republish),
        },
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    payload: dict = {}
    try:
        raw_payload = response.json()
        payload = raw_payload if isinstance(raw_payload, dict) else {}
    except Exception:
        payload = {}
    if response.status_code not in {200, 500} or not payload.get("check_id"):
        response.raise_for_status()
    summary_lines = [
        "已执行生产自检",
        f"check_id={payload.get('check_id')}",
        f"status={payload.get('status')}",
        f"suite={payload.get('suite') or 'core'}",
    ]
    if payload.get("duration_ms") is not None:
        summary_lines.append(f"duration={payload.get('duration_ms')}ms")
    (
        list_md,
        selector_update,
        detail_summary,
        steps_text,
        raw_json,
        next_selected_check_id,
    ) = _refresh_self_check_center(self_check_limit, self_check_status_filter, str(payload.get("check_id") or ""))
    return (
        _status(*summary_lines),
        list_md,
        selector_update,
        detail_summary,
        steps_text,
        raw_json,
        next_selected_check_id,
    )


def _cleanup_self_check_center(
    status_filter: str,
    keep_latest: float | int | None,
    older_than_days: float | int | None,
    dry_run: bool,
    cleanup_limit: float | int,
    self_check_limit: float | int,
    self_check_status_filter: str,
    selected_check_id: str,
):
    payload: dict[str, object] = {
        "dry_run": bool(dry_run),
        "limit": max(1, min(int(cleanup_limit or 200), 5000)),
    }
    normalized_status = str(status_filter or "").strip().lower()
    if normalized_status and normalized_status not in {"all", "*"}:
        payload["status"] = normalized_status
    if keep_latest is not None and str(keep_latest).strip():
        payload["keep_latest"] = max(0, int(keep_latest))
    if older_than_days is not None and str(older_than_days).strip():
        payload["older_than_days"] = max(0, int(older_than_days))
    response = requests.post(
        _task_api_url("/api/v1/self-checks/cleanup"),
        json=payload,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    result = response.json()
    summary_lines = [
        f"Self-check 清理 `{'dry-run' if result.get('dry_run') else 'applied'}`",
        f"candidates=`{result.get('candidate_count', 0)}` deleted=`{result.get('deleted_count', 0)}`",
    ]
    for item in (result.get("candidates") or [])[:10]:
        if not isinstance(item, dict):
            continue
        summary_lines.append(
            f"- `{item.get('status')}` `{item.get('suite')}` `{item.get('check_id')}`"
        )
    (
        list_md,
        selector_update,
        detail_summary,
        steps_text,
        raw_json,
        next_selected_check_id,
    ) = _refresh_self_check_center(self_check_limit, self_check_status_filter, selected_check_id)
    return (
        _status(*summary_lines),
        list_md,
        selector_update,
        detail_summary,
        steps_text,
        raw_json,
        next_selected_check_id,
    )


def _fetch_health_payload() -> dict:
    response = requests.get(
        _task_api_url("/health"),
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _fetch_ingest_stats_payload() -> dict:
    response = requests.get(
        _task_api_url("/api/v1/ingest/stats"),
        params={"include_breakdown": "true"},
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _fetch_metrics_payload() -> str:
    response = requests.get(
        _task_api_url("/metrics"),
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    return response.text


def _metric_value(metrics_text: str, metric_name: str, *, label_filters: dict[str, str] | None = None) -> float | None:
    label_filters = label_filters or {}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        if label_filters:
            if not line.startswith(f"{metric_name}{{"):
                continue
            labels_part, _, value_part = line.partition("}")
            for key, expected in label_filters.items():
                if f'{key}="{expected}"' not in labels_part:
                    break
            else:
                try:
                    return float(value_part.strip())
                except Exception:
                    return None
        else:
            if not re.match(rf"^{re.escape(metric_name)}(?:\{{.*\}})?\s", line):
                continue
            try:
                return float(line.rsplit(" ", 1)[-1].strip())
            except Exception:
                return None
    return None


def _render_ops_health_summary(health: dict) -> str:
    if not health:
        return "无法获取系统状态。"
    lines = [
        f"- API Status: `{health.get('status')}`",
        f"- OCR Loaded: `{bool(health.get('ocr_loaded'))}`",
        f"- Artifact Backend: `{health.get('artifact_backend')}`",
        f"- Ingest Publisher: `{health.get('ingest_publisher')}`",
    ]
    build = health.get("build") or {}
    if isinstance(build, dict) and build:
        lines.append(
            f"- Build: version=`{build.get('package_version') or '-'}` status=`{build.get('status') or '-'}` built=`{_format_task_timestamp(build.get('build_timestamp'))}`"
        )
        if build.get("vcs_ref_short") or build.get("source_tree_sha12"):
            lines.append(
                f"- Build Fingerprint: vcs=`{build.get('vcs_ref_short') or '-'}` source=`{build.get('source_tree_sha12') or '-'}` reqs=`{build.get('requirements_sha12') or '-'}`"
            )
        if build.get("image_tag") or build.get("build_source"):
            lines.append(
                f"- Build Source: `{build.get('build_source') or '-'}` image=`{build.get('image_tag') or '-'}`"
            )
    api_docs = health.get("api_docs") or {}
    if isinstance(api_docs, dict):
        build_info_url = str(api_docs.get("build_info_url") or "").strip()
        if build_info_url:
            lines.append(f"- Build Info API: [{build_info_url}]({build_info_url})")
    ingest_query_status = health.get("ingest_query_status") or {}
    if isinstance(ingest_query_status, dict):
        lines.append(f"- Ingest Query Store: `{ingest_query_status.get('status')}`")
    tracing = health.get("tracing") or {}
    if isinstance(tracing, dict):
        lines.append(
            f"- Tracing: enabled=`{bool(tracing.get('enabled'))}` exporter=`{','.join(tracing.get('exporters') or []) or '-'}`"
        )
    request_protection = health.get("request_protection") or {}
    if isinstance(request_protection, dict):
        rate_limit = request_protection.get("rate_limit") or {}
        admission = request_protection.get("admission") or {}
        if isinstance(rate_limit, dict):
            lines.append(f"- Rate Limit Backend: `{rate_limit.get('backend')}` status=`{rate_limit.get('status')}`")
        if isinstance(admission, dict):
            inflight = admission.get("inflight") or {}
            if isinstance(inflight, dict):
                lines.append(
                    f"- Admission Inflight: parse=`{inflight.get('parse', 0)}` artifact=`{inflight.get('artifact', 0)}` ingest=`{inflight.get('ingest', 0)}`"
                )
    async_tasks = health.get("async_tasks") or {}
    if isinstance(async_tasks, dict):
        worker = async_tasks.get("worker") or {}
        broker = async_tasks.get("broker") or {}
        if isinstance(worker, dict):
            lines.append(
                f"- Async Worker: `{worker.get('status')}` state=`{worker.get('state')}` age=`{worker.get('age_seconds')}`s"
            )
        if isinstance(broker, dict):
            queues = broker.get("queues") or {}
            if isinstance(queues, dict):
                lines.append(
                    f"- Async Queue: queued=`{queues.get('queued', 0)}` processing=`{queues.get('processing', 0)}`"
                )
    self_checks = health.get("self_checks") or {}
    if isinstance(self_checks, dict):
        lines.append(
            f"- Self Checks: `{self_checks.get('status')}` auto=`{bool(self_checks.get('auto_enabled'))}` required=`{bool(self_checks.get('required_for_ready'))}`"
        )
        worker = self_checks.get("worker") or {}
        if isinstance(worker, dict) and worker:
            lines.append(
                f"- Self-check Worker: `{worker.get('status')}` state=`{worker.get('state')}` age=`{worker.get('age_seconds')}`s"
            )
        latest_run = self_checks.get("latest_run") or {}
        if isinstance(latest_run, dict) and latest_run:
            lines.append(
                f"- Latest Self-check: `{latest_run.get('status')}` `{latest_run.get('check_id')}` finished=`{_format_task_timestamp(latest_run.get('finished_at') or latest_run.get('created_at'))}`"
            )
    return "\n".join(lines)


def _render_ops_ingest_summary(stats: dict) -> str:
    if not stats:
        return "无法获取 ingest 状态。"
    documents = stats.get("documents") or {}
    records = stats.get("records") or {}
    breakdown = stats.get("breakdown") or {}
    lines = [
        f"- Ingest Status: `{stats.get('status')}`",
        f"- Schema: `{stats.get('schema')}`",
    ]
    if isinstance(documents, dict):
        lines.append(
            f"- Documents: total=`{documents.get('total', 0)}` published=`{documents.get('published', 0)}` failed=`{documents.get('failed', 0)}` disabled=`{documents.get('disabled', 0)}`"
        )
    if isinstance(records, dict):
        lines.append(
            f"- Records: total=`{records.get('total', 0)}` with_assets=`{records.get('with_assets', 0)}` with_asset_urls=`{records.get('with_asset_urls', 0)}`"
        )
    if isinstance(breakdown, dict):
        parser_engines = breakdown.get("parser_engines") or []
        if isinstance(parser_engines, list) and parser_engines:
            engine_parts = []
            for item in parser_engines[:5]:
                if not isinstance(item, dict):
                    continue
                engine_parts.append(
                    f"{item.get('parser_engine')} docs={item.get('document_count', 0)} recs={item.get('record_count', 0)} assets={item.get('records_with_assets', 0)}"
                )
            if engine_parts:
                lines.append(f"- By Engine: {'; '.join(engine_parts)}")
        file_types = breakdown.get("file_types") or []
        if isinstance(file_types, list) and file_types:
            type_parts = []
            for item in file_types[:5]:
                if not isinstance(item, dict):
                    continue
                type_parts.append(
                    f"{item.get('file_type')} docs={item.get('document_count', 0)} recs={item.get('record_count', 0)}"
                )
            if type_parts:
                lines.append(f"- By File Type: {'; '.join(type_parts)}")
    return "\n".join(lines)


def _render_ops_activity_summary(tasks: list[dict], manifests: list[dict], self_checks: list[dict]) -> str:
    def _count_by(items: list[dict], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            value = str(item.get(key) or "unknown").strip() or "unknown"
            counts[value] = counts.get(value, 0) + 1
        return counts

    task_counts = _count_by(tasks, "status")
    artifact_counts = _count_by(manifests, "parser_engine")
    self_check_counts = _count_by(self_checks, "status")
    lines = [
        f"- Recent Tasks: total=`{len(tasks)}` {', '.join(f'{k}={v}' for k, v in sorted(task_counts.items())) or '-'}",
        f"- Recent Artifacts: total=`{len(manifests)}` {', '.join(f'{k}={v}' for k, v in sorted(artifact_counts.items())) or '-'}",
        f"- Recent Self Checks: total=`{len(self_checks)}` {', '.join(f'{k}={v}' for k, v in sorted(self_check_counts.items())) or '-'}",
    ]
    if tasks:
        latest_task = tasks[0]
        lines.append(
            f"- Latest Task: `{latest_task.get('status')}` `{_task_primary_filename(latest_task) or '-'}` `{latest_task.get('task_id')}`"
        )
    if manifests:
        latest_manifest = manifests[0]
        lines.append(
            f"- Latest Artifact: `{latest_manifest.get('parser_engine')}` `{latest_manifest.get('filename')}` `{latest_manifest.get('parse_id')}`"
        )
    if self_checks:
        latest_check = self_checks[0]
        lines.append(
            f"- Latest Self Check: `{latest_check.get('status')}` `{latest_check.get('check_id')}` `{latest_check.get('suite') or 'core'}`"
        )
    return "\n".join(lines)


def _render_ops_alerts_summary(
    health: dict,
    stats: dict,
    metrics_text: str,
    failed_tasks: list[dict],
    failed_manifests: list[dict],
    failed_self_checks: list[dict],
) -> str:
    alerts: list[str] = []
    status = str(health.get("status") or "").strip().lower()
    if status not in {"ok", "ready"}:
        alerts.append(f"- `critical` API health=`{health.get('status')}`")
    if not bool(health.get("ocr_loaded")):
        alerts.append("- `critical` OCR models not loaded")
    build = health.get("build") or {}
    if isinstance(build, dict):
        build_status = str(build.get("status") or "").strip().lower()
        if build_status not in {"ok"}:
            alerts.append(f"- `warning` build metadata status=`{build.get('status')}`")

    self_checks = health.get("self_checks") or {}
    if isinstance(self_checks, dict):
        self_check_status = str(self_checks.get("status") or "").strip().lower()
        if self_check_status not in {"ok", "disabled"}:
            alerts.append(f"- `critical` self-check plane status=`{self_checks.get('status')}`")
        self_check_worker = self_checks.get("worker") or {}
        if (
            bool(self_checks.get("auto_enabled"))
            and isinstance(self_check_worker, dict)
            and str(self_check_worker.get("status") or "").strip().lower() != "ok"
        ):
            alerts.append(
                f"- `warning` self-check worker status=`{self_check_worker.get('status')}` state=`{self_check_worker.get('state')}`"
            )

    ingest_query_status = health.get("ingest_query_status") or {}
    if isinstance(ingest_query_status, dict):
        if str(ingest_query_status.get("status") or "").strip().lower() != "ok":
            alerts.append(f"- `critical` ingest query backend=`{ingest_query_status.get('status')}`")

    async_tasks = health.get("async_tasks") or {}
    if isinstance(async_tasks, dict):
        worker = async_tasks.get("worker") or {}
        if isinstance(worker, dict):
            worker_status = str(worker.get("status") or "").strip().lower()
            if worker_status not in {"ok"}:
                alerts.append(f"- `critical` async worker status=`{worker.get('status')}` state=`{worker.get('state')}`")
        broker = async_tasks.get("broker") or {}
        if isinstance(broker, dict):
            queues = broker.get("queues") or {}
            if isinstance(queues, dict):
                queued = int(queues.get("queued") or 0)
                processing = int(queues.get("processing") or 0)
                if queued > 20:
                    alerts.append(f"- `warning` async queue backlog queued=`{queued}` processing=`{processing}`")

    request_protection = health.get("request_protection") or {}
    if isinstance(request_protection, dict):
        rate_limit = request_protection.get("rate_limit") or {}
        if isinstance(rate_limit, dict) and str(rate_limit.get("status") or "").strip().lower() != "ok":
            alerts.append(f"- `warning` rate-limit backend status=`{rate_limit.get('status')}`")

    documents = stats.get("documents") or {}
    if isinstance(documents, dict):
        failed_docs = int(documents.get("failed") or 0)
        if failed_docs > 0:
            alerts.append(f"- `warning` ingest documents failed=`{failed_docs}`")
    ingest_rejected = _metric_value(
        metrics_text,
        "deepdoc_request_guard_decisions_total",
        label_filters={"component": "rate_limit", "scope": "ingest", "status": "rejected"},
    ) or 0.0
    if ingest_rejected > 0:
        alerts.append(f"- `warning` ingest rate-limit rejections=`{int(ingest_rejected)}`")

    if failed_tasks:
        alerts.append(f"- `warning` recent failed tasks=`{len(failed_tasks)}`")
    if failed_manifests:
        alerts.append(f"- `warning` recent failed artifact publishes=`{len(failed_manifests)}`")
    if failed_self_checks:
        alerts.append(f"- `warning` recent failed self-checks=`{len(failed_self_checks)}`")

    if not alerts:
        return "- `ok` no active alerts detected from current health, ingest, and recent failure surfaces"
    return "\n".join(alerts)


def _render_ops_failures_summary(
    failed_tasks: list[dict],
    failed_manifests: list[dict],
    failed_self_checks: list[dict],
) -> str:
    lines: list[str] = []
    if failed_tasks:
        lines.append("Recent Failed Tasks:")
        for task in failed_tasks[:5]:
            if not isinstance(task, dict):
                continue
            filename = _task_primary_filename(task) or "-"
            last_error = str(task.get("last_error") or "").strip()
            if len(last_error) > 120:
                last_error = last_error[:117] + "..."
            lines.append(
                f"- `{_format_task_timestamp(task.get('finished_at') or task.get('updated_at'))}` `{filename}` `{task.get('task_id')}` error=`{last_error or '-'} `"
            )
    if failed_manifests:
        lines.append("Recent Failed Artifact Publishes:")
        for manifest in failed_manifests[:5]:
            if not isinstance(manifest, dict):
                continue
            publish_state = (manifest.get("metadata") or {}).get("ingest_publish") or {}
            last_error = str(publish_state.get("last_error") or "").strip()
            if len(last_error) > 120:
                last_error = last_error[:117] + "..."
            lines.append(
                f"- `{_format_task_timestamp(manifest.get('created_at'))}` `{manifest.get('filename')}` `{manifest.get('parse_id')}` error=`{last_error or '-'} `"
            )
    if failed_self_checks:
        lines.append("Recent Failed Self Checks:")
        for result in failed_self_checks[:5]:
            if not isinstance(result, dict):
                continue
            summary = str(result.get("summary") or "").strip()
            if len(summary) > 120:
                summary = summary[:117] + "..."
            lines.append(
                f"- `{_format_task_timestamp(result.get('finished_at') or result.get('created_at'))}` `{result.get('check_id')}` suite=`{result.get('suite') or 'core'}` summary=`{summary or '-'} `"
            )
    if not lines:
        return "No recent failures."
    return "\n".join(lines)


def _render_ops_metrics_summary(metrics_text: str) -> str:
    published = _metric_value(
        metrics_text,
        "deepdoc_ingest_publish_total",
        label_filters={"sink_type": "postgres", "status": "published"},
    )
    build_age_seconds = _metric_value(metrics_text, "deepdoc_build_age_seconds")
    tracing_enabled = _metric_value(metrics_text, "deepdoc_tracing_enabled")
    artifact_rejected = _metric_value(
        metrics_text,
        "deepdoc_request_guard_decisions_total",
        label_filters={"component": "rate_limit", "scope": "artifact", "status": "rejected"},
    ) or 0.0
    parse_inflight = _metric_value(
        metrics_text,
        "deepdoc_admission_inflight",
        label_filters={"pool": "parse"},
    ) or 0.0
    artifact_inflight = _metric_value(
        metrics_text,
        "deepdoc_admission_inflight",
        label_filters={"pool": "artifact"},
    ) or 0.0
    ingest_inflight = _metric_value(
        metrics_text,
        "deepdoc_admission_inflight",
        label_filters={"pool": "ingest"},
    ) or 0.0
    lines = [
        f"- Build Age Seconds: `{int(build_age_seconds or 0)}`",
        f"- Tracing Enabled Metric: `{int(tracing_enabled or 0)}`",
        f"- Ingest Published Total: `{int(published or 0)}`",
        f"- Rate Limit Rejections: artifact=`{int(artifact_rejected)}`",
        f"- Admission Inflight: parse=`{int(parse_inflight)}` artifact=`{int(artifact_inflight)}` ingest=`{int(ingest_inflight)}`",
    ]
    return "\n".join(lines)


def _refresh_ops_overview():
    health = _fetch_health_payload()
    ingest_stats = _fetch_ingest_stats_payload()
    metrics_text = _fetch_metrics_payload()
    tasks = _fetch_async_tasks(limit=10, status_filter="all")
    manifests = _fetch_artifact_manifests(limit=10, publish_status_filter="all")
    self_checks = _fetch_self_checks(limit=10, status_filter="all")
    failed_tasks = _fetch_async_tasks(limit=5, status_filter="failed")
    failed_manifests = _fetch_artifact_manifests(limit=5, publish_status_filter="failed")
    failed_self_checks = _fetch_self_checks(limit=5, status_filter="failed")
    return (
        _render_ops_health_summary(health),
        _render_ops_ingest_summary(ingest_stats),
        _render_ops_alerts_summary(health, ingest_stats, metrics_text, failed_tasks, failed_manifests, failed_self_checks),
        _render_ops_failures_summary(failed_tasks, failed_manifests, failed_self_checks),
        _render_ops_activity_summary(tasks, manifests, self_checks),
        _render_ops_metrics_summary(metrics_text),
    )


def _normalize_optional_text(value: object) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _pretty_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(value)


def _prefill_ingest_target_from_task(task_id: str):
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return "", ""
    try:
        task_payload = _fetch_async_task(normalized_task_id)
    except Exception:
        logger.exception("Failed to prefill ingest target from task task_id=%s", normalized_task_id)
        return "", ""
    primary = _task_primary_result(task_payload) or {}
    parse_id = str(primary.get("parse_id") or "").strip()
    document_id = str(primary.get("document_id") or "").strip()
    if not parse_id:
        parse_ids = _task_parse_ids(task_payload)
        parse_id = parse_ids[0] if parse_ids else ""
    return parse_id, document_id


def _prefill_ingest_target_from_artifact(parse_id: str):
    normalized_parse_id = str(parse_id or "").strip()
    if not normalized_parse_id:
        return "", ""
    try:
        manifest = _fetch_artifact_manifest(normalized_parse_id)
    except Exception:
        logger.exception("Failed to prefill ingest target from artifact parse_id=%s", normalized_parse_id)
        return "", ""
    return (
        str(manifest.get("parse_id") or "").strip(),
        str(manifest.get("document_id") or "").strip(),
    )


def _fetch_ingest_documents_payload(
    limit: int = 20,
    *,
    tenant_id: str | None = None,
    parser_engine: str | None = None,
    file_type: str | None = None,
    publish_status: str | None = None,
):
    params: dict[str, object] = {"limit": max(1, min(int(limit), 200))}
    if tenant_id := _normalize_optional_text(tenant_id):
        params["tenant_id"] = tenant_id
    if parser_engine := _normalize_optional_text(parser_engine):
        params["parser_engine"] = parser_engine
    if file_type := _normalize_optional_text(file_type):
        params["file_type"] = file_type
    if publish_status := _normalize_optional_text(publish_status):
        params["publish_status"] = publish_status
    response = requests.get(
        _task_api_url("/api/v1/ingest/documents"),
        params=params,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    return results if isinstance(results, list) else []


def _render_ingest_stats_detail(stats: dict) -> str:
    if not stats:
        return "暂无 ingest stats。"
    documents = stats.get("documents") or {}
    records = stats.get("records") or {}
    assets = stats.get("assets") or {}
    chunk_asset_links = stats.get("chunk_asset_links") or {}
    breakdown = stats.get("breakdown") or {}
    parser_breakdown = breakdown.get("parser_engines") or {}
    file_type_breakdown = breakdown.get("file_types") or {}
    if isinstance(parser_breakdown, dict):
        parser_pairs = [f"{key}={value}" for key, value in list(parser_breakdown.items())[:8]]
    elif isinstance(parser_breakdown, list):
        parser_pairs = [
            f"{item.get('parser_engine')} docs={item.get('document_count')} records={item.get('record_count')}"
            for item in parser_breakdown[:8]
            if isinstance(item, dict)
        ]
    else:
        parser_pairs = []
    if isinstance(file_type_breakdown, dict):
        file_type_pairs = [f"{key}={value}" for key, value in list(file_type_breakdown.items())[:8]]
    elif isinstance(file_type_breakdown, list):
        file_type_pairs = [
            f"{item.get('file_type')} docs={item.get('document_count')} records={item.get('record_count')}"
            for item in file_type_breakdown[:8]
            if isinstance(item, dict)
        ]
    else:
        file_type_pairs = []
    lines = [
        f"- Status: `{stats.get('status') or '-'}`",
        f"- Schema: `{stats.get('schema') or '-'}`",
        f"- Documents: total=`{documents.get('total', 0)}` published=`{documents.get('published', 0)}` failed=`{documents.get('failed', 0)}` disabled=`{documents.get('disabled', 0)}`",
        f"- Records: total=`{records.get('total', 0)}` with_assets=`{records.get('with_assets', 0)}` with_urls=`{records.get('with_asset_urls', 0)}`",
        f"- Assets: total=`{assets.get('total', 0)}` materialized=`{assets.get('materialized', 0)}` with_urls=`{assets.get('with_urls', 0)}`",
        f"- Chunk Asset Links: total=`{chunk_asset_links.get('total', 0)}` direct=`{chunk_asset_links.get('direct', 0)}` context=`{chunk_asset_links.get('context', 0)}`",
    ]
    if parser_pairs:
        lines.append("- Parser Breakdown: " + " ".join(f"`{pair}`" for pair in parser_pairs))
    if file_type_pairs:
        lines.append("- File Type Breakdown: " + " ".join(f"`{pair}`" for pair in file_type_pairs))
    return "\n".join(lines)


def _render_ingest_documents_list(results: list[dict]) -> str:
    if not results:
        return "暂无 ingest 文档。"
    lines: list[str] = []
    for item in results[:20]:
        if not isinstance(item, dict):
            continue
        parse_id = str(item.get("parse_id") or "").strip()
        filename = str(item.get("filename") or "-").strip()
        parser_engine = str(item.get("parser_engine") or "-").strip()
        file_type = str(item.get("file_type") or "-").strip()
        publish_status = str(item.get("publish_status") or "-").strip()
        chunk_count = item.get("chunk_count")
        asset_count = item.get("asset_count")
        created_at = _format_task_timestamp(item.get("created_at"))
        lines.append(
            f"- `{created_at}` `{publish_status}` `{parser_engine}` `{file_type}` `{filename}` `{parse_id}` chunks=`{chunk_count}` assets=`{asset_count}`"
        )
    return "\n".join(lines)


def _refresh_ingest_console(
    parse_id: str,
    document_id: str,
    parser_engine: str,
    file_type: str,
    tenant_id: str,
    document_limit: float | int,
):
    params: dict[str, str] = {}
    if parse_id := _normalize_optional_text(parse_id):
        params["parse_id"] = parse_id
    if document_id := _normalize_optional_text(document_id):
        params["document_id"] = document_id
    if parser_engine := _normalize_optional_text(parser_engine):
        params["parser_engine"] = parser_engine
    if file_type := _normalize_optional_text(file_type):
        params["file_type"] = file_type
    if tenant_id := _normalize_optional_text(tenant_id):
        params["tenant_id"] = tenant_id
    response = requests.get(
        _task_api_url("/api/v1/ingest/stats"),
        params={**params, "include_breakdown": "true"},
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    stats_payload = response.json()
    documents = _fetch_ingest_documents_payload(
        int(document_limit or 20),
        tenant_id=tenant_id,
        parser_engine=parser_engine,
        file_type=file_type,
    )
    return (
        _render_ingest_stats_detail(stats_payload if isinstance(stats_payload, dict) else {}),
        _render_ingest_documents_list(documents),
    )


def _query_ingest_console_records(
    query_text: str,
    parse_id: str,
    document_id: str,
    tenant_id: str,
    record_limit: float | int,
):
    params: dict[str, object] = {
        "limit": max(1, min(int(record_limit or 10), 200)),
        "mode": "text",
    }
    if query_text := _normalize_optional_text(query_text):
        params["q"] = query_text
    if parse_id := _normalize_optional_text(parse_id):
        params["parse_id"] = parse_id
    if document_id := _normalize_optional_text(document_id):
        params["document_id"] = document_id
    if tenant_id := _normalize_optional_text(tenant_id):
        params["tenant_id"] = tenant_id
    response = requests.get(
        _task_api_url("/api/v1/ingest/records"),
        params=params,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list):
        results = []
    lines = [
        f"- Query: `{params.get('q') or '-'}`",
        f"- Mode: `{params.get('mode')}`",
        f"- Result Count: `{len(results)}`",
    ]
    for item in results[:10]:
        if not isinstance(item, dict):
            continue
        title_path = item.get("metadata", {}).get("title_path") or []
        title_path_text = " / ".join(str(part) for part in title_path if str(part).strip())
        lines.append(
            f"- chunk=`{item.get('chunk_id') or '-'}` parse=`{item.get('parse_id') or '-'}` score=`{item.get('score')}` assets=`{len(item.get('asset_refs') or [])}`"
        )
        if title_path_text:
            lines.append(f"  title_path: `{title_path_text}`")
        excerpt = re.sub(r"\s+", " ", str(item.get("text") or "").strip())
        if len(excerpt) > 280:
            excerpt = excerpt[:277] + "..."
        if excerpt:
            lines.append(f"  text: {excerpt}")
    return ("\n".join(lines), _pretty_json(payload))


def _search_ingest_console_assets(
    parse_id: str,
    document_id: str,
    tenant_id: str,
    asset_type: str,
    asset_limit: float | int,
):
    params: dict[str, object] = {
        "limit": max(1, min(int(asset_limit or 10), 200)),
    }
    if parse_id := _normalize_optional_text(parse_id):
        params["parse_id"] = parse_id
    if document_id := _normalize_optional_text(document_id):
        params["document_id"] = document_id
    if tenant_id := _normalize_optional_text(tenant_id):
        params["tenant_id"] = tenant_id
    if asset_type := _normalize_optional_text(asset_type):
        params["asset_type"] = asset_type
    response = requests.get(
        _task_api_url("/api/v1/ingest/assets"),
        params=params,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list):
        results = []
    lines = [
        f"- Result Count: `{len(results)}`",
        f"- Asset Type: `{params.get('asset_type') or '-'}`",
    ]
    for item in results[:10]:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- asset=`{item.get('asset_id') or '-'}` type=`{item.get('type') or '-'}` parse=`{item.get('parse_id') or '-'}` page=`{item.get('page')}` size=`{item.get('width') or '-'}x{item.get('height') or '-'}`"
        )
        title = re.sub(r"\s+", " ", str(item.get("title") or item.get("text") or "").strip())
        if len(title) > 180:
            title = title[:177] + "..."
        if title:
            lines.append(f"  title: {title}")
        resolved_url = str(item.get("resolved_url") or item.get("download_path") or "").strip()
        if resolved_url:
            lines.append(f"  url: `{resolved_url}`")
    return ("\n".join(lines), _pretty_json(payload))


def _search_ingest_console_chunks(
    parse_id: str,
    document_id: str,
    tenant_id: str,
    chunk_id: str,
    chunk_limit: float | int,
):
    params: dict[str, object] = {
        "limit": max(1, min(int(chunk_limit or 10), 200)),
    }
    if parse_id := _normalize_optional_text(parse_id):
        params["parse_id"] = parse_id
    if document_id := _normalize_optional_text(document_id):
        params["document_id"] = document_id
    if tenant_id := _normalize_optional_text(tenant_id):
        params["tenant_id"] = tenant_id
    if chunk_id := _normalize_optional_text(chunk_id):
        params["chunk_id"] = chunk_id
    response = requests.get(
        _task_api_url("/api/v1/ingest/chunks"),
        params=params,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list):
        results = []
    lines = [
        f"- Result Count: `{len(results)}`",
        f"- Chunk Filter: `{params.get('chunk_id') or '-'}`",
    ]
    for item in results[:10]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        title_path = metadata.get("title_path") or []
        title_path_text = " / ".join(str(part) for part in title_path if str(part).strip())
        direct_refs = metadata.get("direct_asset_refs") or []
        context_refs = metadata.get("context_asset_refs") or []
        lines.append(
            f"- chunk=`{item.get('chunk_id') or '-'}` parse=`{item.get('parse_id') or '-'}` tokens=`{item.get('token_count')}` pages=`{','.join(str(p) for p in (item.get('page_numbers') or [])) or '-'}`"
        )
        if title_path_text:
            lines.append(f"  title_path: `{title_path_text}`")
        if direct_refs or context_refs:
            lines.append(
                f"  assets: direct=`{','.join(str(ref) for ref in direct_refs) or '-'}` context=`{','.join(str(ref) for ref in context_refs) or '-'}`"
            )
        excerpt = re.sub(r"\s+", " ", str(item.get("text") or "").strip())
        if len(excerpt) > 220:
            excerpt = excerpt[:217] + "..."
        if excerpt:
            lines.append(f"  text: {excerpt}")
    return ("\n".join(lines), _pretty_json(payload))


def _search_ingest_console_chunk_asset_links(
    parse_id: str,
    document_id: str,
    tenant_id: str,
    chunk_id: str,
    asset_id: str,
    relation_type: str,
    link_limit: float | int,
):
    params: dict[str, object] = {
        "limit": max(1, min(int(link_limit or 20), 500)),
    }
    if parse_id := _normalize_optional_text(parse_id):
        params["parse_id"] = parse_id
    if document_id := _normalize_optional_text(document_id):
        params["document_id"] = document_id
    if tenant_id := _normalize_optional_text(tenant_id):
        params["tenant_id"] = tenant_id
    if chunk_id := _normalize_optional_text(chunk_id):
        params["chunk_id"] = chunk_id
    if asset_id := _normalize_optional_text(asset_id):
        params["asset_id"] = asset_id
    relation_type = _normalize_optional_text(relation_type)
    if relation_type and relation_type != "all":
        params["relation_type"] = relation_type
    response = requests.get(
        _task_api_url("/api/v1/ingest/chunk-asset-links"),
        params=params,
        headers=_build_api_headers(),
        timeout=_api_timeout(),
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list):
        results = []
    lines = [
        f"- Result Count: `{len(results)}`",
        f"- Relation Type: `{params.get('relation_type') or 'all'}`",
    ]
    for item in results[:12]:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- chunk=`{item.get('chunk_id') or '-'}` asset=`{item.get('asset_id') or '-'}` relation=`{item.get('relation_type') or '-'}` ordinal=`{item.get('ordinal')}`"
        )
        asset_title = re.sub(r"\s+", " ", str(item.get("asset_title") or "").strip())
        if asset_title:
            lines.append(f"  title: {asset_title}")
        resolved_url = str(item.get("resolved_url") or item.get("download_path") or "").strip()
        if resolved_url:
            lines.append(f"  url: `{resolved_url}`")
    return ("\n".join(lines), _pretty_json(payload))


def _finalize_task_outputs(file_path: str, task_payload: dict, total_elapsed: float):
    primary = _task_primary_result(task_payload)
    if not primary:
        return (
            _status("失败", "任务没有返回解析结果", f"总耗时: {total_elapsed:.1f}s"),
            None,
            "",
            "",
            "",
            "",
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            "",
        )

    if primary.get("error") and not primary.get("markdown"):
        return (
            _status(
                "失败" if task_payload.get("status") != "cancelled" else "已取消",
                f"任务: {task_payload.get('task_id')}",
                str(primary.get("error")),
                f"总耗时: {total_elapsed:.1f}s",
            ),
            None,
            "",
            "",
            "",
            "",
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            "",
        )

    markdown_content = str(primary.get("markdown") or "")
    output_file_path = _write_markdown_file(file_path, markdown_content) if markdown_content else None
    artifact_urls = primary.get("artifact_urls") or {}
    summary_lines = [
        "已取消" if task_payload.get("status") == "cancelled" else "完成",
        f"解析器: {task_payload.get('parser_engine')}",
        f"文件: {Path(file_path).name}",
        f"任务: {task_payload.get('task_id')}",
        f"总耗时: {total_elapsed:.1f}s",
    ]
    if primary.get("parse_id"):
        summary_lines.append(f"Parse ID: {primary.get('parse_id')}")
    if primary.get("chunk_count") is not None:
        summary_lines.append(f"Chunks: {primary.get('chunk_count')}")
    if primary.get("asset_count") is not None:
        summary_lines.append(f"Assets: {primary.get('asset_count')}")
    if isinstance(artifact_urls, dict) and artifact_urls.get("manifest_url"):
        summary_lines.append(f"Manifest: {artifact_urls.get('manifest_url')}")
    artifact_summary = _render_artifact_summary(task_payload, primary)
    events_text = _render_task_events(_safe_fetch_async_task_events(str(task_payload.get("task_id") or "")))
    return (
        _status(*summary_lines),
        output_file_path,
        markdown_content,
        markdown_content,
        artifact_summary,
        events_text,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        "",
    )


def run_parse(
    file_path: str,
    parser_mode: str,
    deepdoc_layout_model: str,
    deepdoc_max_pages: int,
    paddle_use_gpu: bool,
    paddle_prettify_markdown: bool,
    paddle_show_formula_number: bool,
    paddle_use_formula_recognition: bool,
    paddle_table_enable: bool,
    paddle_seal_enable: bool,
    mineru_max_pages: int,
    mineru_use_gpu: bool,
    mineru_language: str,
    mineru_is_ocr: bool,
    mineru_formula_enable: bool,
    mineru_table_enable: bool,
):
    if not file_path:
        yield (
            _status("请先上传文件"),
            None,
            "",
            "",
            "",
            "",
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            "",
        )
        return

    start_ts = time.perf_counter()
    task_id = ""
    try:
        yield (
            _status("提交中...", f"解析器: {parser_mode}", f"文件: {Path(file_path).name}"),
            None,
            "",
            "",
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
        )
        submitted = _submit_async_parse(
            file_path,
            parser_mode,
            deepdoc_layout_model,
            deepdoc_max_pages,
            bool(paddle_use_gpu),
            paddle_prettify_markdown,
            paddle_show_formula_number,
            paddle_use_formula_recognition,
            paddle_table_enable,
            paddle_seal_enable,
            mineru_max_pages,
            mineru_use_gpu,
            mineru_language,
            mineru_is_ocr,
            mineru_formula_enable,
            mineru_table_enable,
        )
        task_id = str(submitted.get("task_id") or "").strip()
        if not task_id:
            raise RuntimeError("异步任务提交成功，但没有返回 task_id")
        yield (
            _status("已提交", f"任务: {task_id}", f"解析器: {parser_mode}"),
            None,
            "",
            "",
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            task_id,
        )

        last_status = ""
        while True:
            task_payload = _fetch_async_task(task_id)
            events = _safe_fetch_async_task_events(task_id)
            status = str(task_payload.get("status") or "queued")
            elapsed = time.perf_counter() - start_ts
            progress_line = _task_progress_line(task_payload)
            primary = _task_primary_result(task_payload)
            status_lines = [
                "处理中..." if status in {"queued", "running", "cancel_requested"} else status,
                f"任务: {task_id}",
                f"状态: {status}",
                f"已处理: {elapsed:.1f}s",
            ]
            if progress_line:
                status_lines.append(progress_line)
            status_text = _status(*status_lines)
            artifact_summary = _render_artifact_summary(task_payload, primary)
            events_text = _render_task_events(events)
            if status_text != last_status:
                yield (
                    status_text,
                    None,
                    "",
                    "",
                    artifact_summary,
                    events_text,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=status in {"queued", "running", "cancel_requested"}),
                    task_id,
                )
                last_status = status_text

            if status in {"succeeded", "failed", "cancelled"}:
                total_elapsed = time.perf_counter() - start_ts
                yield _finalize_task_outputs(file_path, task_payload, total_elapsed)
                break
            time.sleep(ASYNC_TASK_POLL_SECONDS)
    except Exception as exc:
        logger.exception("run_parse failed via async backend")
        yield (
            _status("失败", str(exc)),
            None,
            "",
            "",
            "",
            "",
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            "",
        )
        return

    total_elapsed = time.perf_counter() - start_ts
    cleanup_age_raw = (
        os.environ.get("GRADIO_TMP_CLEANUP_MIN_AGE_SECONDS", "600") or "600"
    ).strip()
    try:
        cleanup_age = max(60, int(cleanup_age_raw))
    except Exception:
        cleanup_age = 600
    removed = _cleanup_runtime_tmp_files(
        GRADIO_TMP_DIR,
        min_age_seconds=cleanup_age,
        keep_paths={file_path or ""},
    )
    if removed > 0:
        logger.info("[Cleanup] removed tmp files=%d min_age=%ds", removed, cleanup_age)


def cancel_active_task(task_id: str):
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return _status("没有活动任务"), "", gr.update(interactive=False)
    try:
        payload = _cancel_async_task_request(normalized_task_id)
        events_text = _render_task_events(_safe_fetch_async_task_events(normalized_task_id))
        return (
            _status("已请求取消", f"任务: {normalized_task_id}", f"状态: {payload.get('status')}"),
            events_text,
            gr.update(interactive=False),
        )
    except Exception as exc:
        logger.exception("cancel_active_task failed task_id=%s", normalized_task_id)
        return _status("取消失败", str(exc)), "", gr.update(interactive=True)


def prepare_run_ui():
    return (
        "处理中...",
        None,
        "",
        "",
        "",
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        "",
    )


def _prepare_preview_update(file_path: str):
    logger.info("[Preview] stage-1 file changed upload_path=%s", file_path)
    return gr.update(visible=True)


def _reset_ui_outputs():
    return (
        None,
        "",
        None,
        "",
        "",
        "",
        "",
        gr.update(value=None, visible=True),
        gr.update(interactive=False),
        "",
    )


def build_app():
    with gr.Blocks(
        title=PARSER_CONSOLE_TITLE,
    ) as demo:
        gr.HTML(f"<h3>{PARSER_CONSOLE_TITLE}</h3>")

        parser_mode_state = gr.State("deepdoc")
        task_id_state = gr.State("")
        task_center_selected_state = gr.State("")
        artifact_center_selected_state = gr.State("")
        self_check_center_selected_state = gr.State("")
        audit_center_selected_state = gr.State("")
        ops_auto_refresh_timer = gr.Timer(20.0, active=True)

        with gr.Row():
            with gr.Column(variant="panel", scale=5):
                with gr.Row():
                    file_input = gr.File(
                        label="请选择要上传的文件",
                        file_types=[
                            ".pdf",
                            ".docx",
                            ".xlsx",
                            ".xls",
                            ".pptx",
                            ".ppt",
                            ".html",
                            ".json",
                            ".md",
                            ".txt",
                            ".csv",
                            ".xml",
                            ".zip",
                            ".epub",
                        ],
                        type="filepath",
                    )

                with gr.Tabs():
                    with gr.Tab(PRODUCT_NAME) as deepdoc_tab:
                        deepdoc_max_pages = gr.Slider(
                            1,
                            200,
                            int(os.environ.get("DEEPDOC_PDF_MAX_PAGES", "10")),
                            step=1,
                            label="最大转换页数",
                        )
                        deepdoc_layout_model = gr.Dropdown(
                            DEEPDOC_LAYOUT_MODELS,
                            label="版面模型",
                            value="general",
                            visible=False,
                        )
                    with gr.Tab("PaddleOCR-VL") as paddle_tab:
                        with gr.Group():
                            with gr.Row():
                                paddle_gpu_api_url = (
                                    os.environ.get("PADDLEOCR_GPU_API_URL", "") or ""
                                ).strip()
                                paddle_gpu_available = _is_valid_http_url(
                                    paddle_gpu_api_url
                                )
                                paddle_use_gpu = gr.Checkbox(
                                    label="启用 GPU 加速",
                                    value=paddle_gpu_available,
                                    interactive=paddle_gpu_available,
                                    info=(
                                        "使用 PADDLEOCR_GPU_API_URL"
                                        if paddle_gpu_available
                                        else "未配置 PADDLEOCR_GPU_API_URL，GPU 已禁用"
                                    ),
                                )
                            with gr.Row(equal_height=True):
                                with gr.Column():
                                    gr.Markdown("**识别选项：**")
                                    paddle_use_formula_recognition = gr.Checkbox(
                                        label="启用公式识别", value=True
                                    )
                                    paddle_seal_enable = gr.Checkbox(
                                        label="启用印章识别", value=True
                                    )
                            paddle_prettify_markdown = gr.Checkbox(
                                label="prettify_markdown",
                                value=True,
                                visible=False,
                            )
                            paddle_show_formula_number = gr.Checkbox(
                                label="启用公式编号（公式）",
                                value=False,
                                visible=False,
                            )
                            paddle_table_enable = gr.Checkbox(
                                label="跨页表格合并（mergeTables）",
                                value=True,
                                visible=False,
                            )
                    with gr.Tab("MarkItDown") as markitdown_tab:
                        gr.Markdown(
                            "使用本地 MarkItDown 转换为 Markdown，适合快速处理 Office、CSV/XML/EPUB/ZIP 等非 PDF 文件。"
                        )
                    with gr.Tab("MinerU") as mineru_tab:
                        with gr.Group():
                            with gr.Row():
                                mineru_max_pages = gr.Slider(
                                    1,
                                    200,
                                    10,
                                    step=1,
                                    label="最大转换页数",
                                )
                            with gr.Row():
                                mineru_gpu_api_url = (
                                    os.environ.get("MINERU_GPU_API_URL", "") or ""
                                ).strip()
                                mineru_gpu_available = _is_valid_http_url(
                                    mineru_gpu_api_url
                                )
                                mineru_use_gpu = gr.Checkbox(
                                    label="启用 GPU 加速",
                                    value=mineru_gpu_available,
                                    interactive=mineru_gpu_available,
                                    info=(
                                        "使用 MINERU_GPU_API_URL"
                                        if mineru_gpu_available
                                        else "未配置 MINERU_GPU_API_URL，GPU 已禁用"
                                    ),
                                )
                            with gr.Row(equal_height=True):
                                with gr.Column():
                                    gr.Markdown("**识别选项：**")
                                    mineru_table_enable = gr.Checkbox(
                                        label="启用表格识别",
                                        value=True,
                                        info="禁用后，表格将显示为图片。",
                                    )
                                    mineru_formula_enable = gr.Checkbox(
                                        label="启用行内公式识别",
                                        value=True,
                                        info="禁用后，行内公式将不会被检测或解析。",
                                    )
                                with gr.Column():
                                    mineru_language = gr.Dropdown(
                                        MINERU_LANG_CHOICES,
                                        label="OCR 语言",
                                        value="ch",
                                        info="为扫描版 PDF 选择 OCR 语言。",
                                    )
                                    mineru_is_ocr = gr.Checkbox(
                                        label="强制启用 OCR",
                                        value=False,
                                        info="仅在识别效果极差时启用，需选择正确的 OCR 语言。",
                                    )

                with gr.Row():
                    run_btn = gr.Button(
                        "转换",
                        variant="primary",
                    )
                    cancel_btn = gr.Button(
                        "取消",
                        variant="stop",
                        interactive=False,
                    )
                    clear_btn = gr.ClearButton(value="清除")
                doc_preview = GradioPDF(
                    label="文档预览",
                    interactive=False,
                    visible=True,
                    height=800,
                )

            with gr.Column(variant="panel", scale=5):
                status_box = gr.TextArea(
                    label="转换状态",
                    value="",
                    lines=4,
                    max_lines=4,
                    interactive=False,
                    autoscroll=True,
                    elem_classes=["convert-status-box"],
                )
                with gr.Accordion("运行概览", open=True):
                    ops_refresh_btn = gr.Button("刷新概览", variant="secondary")
                    gr.Markdown("自动刷新：`20s`")
                    ops_health_md = gr.Markdown(value="加载中...")
                    ops_ingest_md = gr.Markdown(value="")
                    ops_alerts_md = gr.Markdown(value="")
                    ops_failures_md = gr.Markdown(value="")
                    ops_activity_md = gr.Markdown(value="")
                    ops_metrics_md = gr.Markdown(value="")
                with gr.Accordion("生产自检", open=False):
                    with gr.Row():
                        self_check_refresh_btn = gr.Button("刷新自检", variant="secondary")
                        self_check_run_btn = gr.Button("执行自检", variant="primary")
                        self_check_limit = gr.Slider(
                            minimum=5,
                            maximum=50,
                            step=5,
                            value=20,
                            label="最近自检数",
                        )
                        self_check_status_filter = gr.Dropdown(
                            SELF_CHECK_STATUS_FILTER_CHOICES,
                            label="自检状态",
                            value="all",
                        )
                    with gr.Row():
                        self_check_force_reparse = gr.Checkbox(
                            label="强制重跑文本解析链",
                            value=False,
                        )
                        self_check_force_republish = gr.Checkbox(
                            label="强制重发 ingest",
                            value=True,
                        )
                    self_check_action_result = gr.TextArea(
                        label="自检操作结果",
                        value="",
                        lines=4,
                        max_lines=8,
                        interactive=False,
                        autoscroll=True,
                    )
                    self_check_list_md = gr.Markdown(value="暂无生产自检结果。")
                    self_check_selector = gr.Dropdown(
                        label="选择生产自检",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    self_check_summary = gr.Markdown(value="未选择生产自检。")
                    self_check_steps_box = gr.TextArea(
                        label="自检步骤",
                        value="",
                        lines=8,
                        max_lines=14,
                        interactive=False,
                        autoscroll=True,
                    )
                    self_check_raw_box = gr.TextArea(
                        label="自检原始 JSON",
                        value="",
                        lines=10,
                        max_lines=16,
                        interactive=False,
                        autoscroll=True,
                    )
                    with gr.Accordion("自检清理", open=False):
                        with gr.Row():
                            self_check_cleanup_status = gr.Dropdown(
                                SELF_CHECK_STATUS_FILTER_CHOICES,
                                label="清理状态",
                                value="passed",
                            )
                            self_check_cleanup_keep_latest = gr.Number(
                                value=20,
                                minimum=0,
                                precision=0,
                                label="保留最近自检数",
                            )
                            self_check_cleanup_older_than_days = gr.Number(
                                value=7,
                                minimum=0,
                                precision=0,
                                label="清理多少天前",
                            )
                        with gr.Row():
                            self_check_cleanup_dry_run = gr.Checkbox(
                                label="仅预演",
                                value=True,
                            )
                            self_check_cleanup_limit = gr.Number(
                                value=200,
                                minimum=1,
                                precision=0,
                                label="最多扫描",
                            )
                        self_check_cleanup_btn = gr.Button("执行自检清理", variant="secondary")
                        self_check_cleanup_result = gr.TextArea(
                            label="自检清理结果",
                            value="",
                            lines=6,
                            max_lines=10,
                            interactive=False,
                            autoscroll=True,
                        )
                artifact_info = gr.Markdown(label="任务产物", value="")
                task_events_box = gr.TextArea(
                    label="任务事件",
                    value="",
                    lines=8,
                    max_lines=12,
                    interactive=False,
                    autoscroll=True,
                )
                with gr.Accordion("任务中心", open=False):
                    with gr.Row():
                        task_center_refresh_btn = gr.Button("刷新任务", variant="secondary")
                        task_center_limit = gr.Slider(
                            minimum=5,
                            maximum=50,
                            step=5,
                            value=20,
                            label="最近任务数",
                        )
                        task_center_status_filter = gr.Dropdown(
                            TASK_STATUS_FILTER_CHOICES,
                            label="状态过滤",
                            value="all",
                        )
                    recent_tasks_md = gr.Markdown(value="暂无任务。")
                    task_center_selector = gr.Dropdown(
                        label="选择任务",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    task_center_summary = gr.Markdown(value="未选择任务。")
                    task_center_artifact = gr.Markdown(value="")
                    task_center_events = gr.TextArea(
                        label="选中任务事件",
                        value="",
                        lines=10,
                        max_lines=14,
                        interactive=False,
                        autoscroll=True,
                    )
                    task_center_callback_events = gr.TextArea(
                        label="选中任务回调事件",
                        value="",
                        lines=8,
                        max_lines=12,
                        interactive=False,
                        autoscroll=True,
                    )
                    with gr.Row():
                        task_center_inspect_btn = gr.Button("刷新详情", variant="secondary")
                        task_center_cancel_btn = gr.Button(
                            "取消选中任务",
                            variant="stop",
                            interactive=False,
                        )
                        task_center_retry_btn = gr.Button(
                            "重试选中任务",
                            variant="secondary",
                            interactive=False,
                        )
                        task_center_callback_retry_btn = gr.Button(
                            "重发任务回调",
                            variant="secondary",
                            interactive=False,
                        )
                    with gr.Accordion("任务清理", open=False):
                        task_cleanup_statuses = gr.CheckboxGroup(
                            choices=["succeeded", "failed", "cancelled"],
                            value=["succeeded"],
                            label="清理状态",
                        )
                        with gr.Row():
                            task_cleanup_keep_latest = gr.Number(
                                value=20,
                                minimum=0,
                                precision=0,
                                label="保留最近任务数",
                            )
                            task_cleanup_older_than_days = gr.Number(
                                value=7,
                                minimum=0,
                                precision=0,
                                label="清理多少天前",
                            )
                        with gr.Row():
                            task_cleanup_include_active = gr.Checkbox(
                                label="包含运行中任务",
                                value=False,
                            )
                            task_cleanup_dry_run = gr.Checkbox(
                                label="仅预演",
                                value=True,
                            )
                            task_cleanup_limit = gr.Number(
                                value=200,
                                minimum=1,
                                precision=0,
                                label="最多扫描",
                            )
                        task_cleanup_btn = gr.Button("执行清理", variant="secondary")
                        task_cleanup_result = gr.TextArea(
                            label="清理结果",
                            value="",
                            lines=6,
                            max_lines=10,
                            interactive=False,
                            autoscroll=True,
                        )
                with gr.Accordion("产物中心", open=False):
                    with gr.Row():
                        artifact_center_refresh_btn = gr.Button("刷新产物", variant="secondary")
                        artifact_center_limit = gr.Slider(
                            minimum=5,
                            maximum=50,
                            step=5,
                            value=20,
                            label="最近产物数",
                        )
                        artifact_center_publish_status_filter = gr.Dropdown(
                            ARTIFACT_PUBLISH_STATUS_FILTER_CHOICES,
                            label="发布状态",
                            value="all",
                        )
                    artifact_list_md = gr.Markdown(value="暂无产物。")
                    artifact_selector = gr.Dropdown(
                        label="选择产物",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    artifact_center_summary = gr.Markdown(value="未选择产物。")
                    artifact_publish_events_box = gr.TextArea(
                        label="选中产物发布事件",
                        value="",
                        lines=8,
                        max_lines=12,
                        interactive=False,
                        autoscroll=True,
                    )
                    with gr.Row():
                        artifact_center_inspect_btn = gr.Button("刷新产物详情", variant="secondary")
                        artifact_center_republish_btn = gr.Button(
                            "重发选中产物",
                            variant="secondary",
                            interactive=False,
                        )
                    with gr.Accordion("产物清理", open=False):
                        with gr.Row():
                            artifact_cleanup_keep_latest = gr.Number(
                                value=20,
                                minimum=0,
                                precision=0,
                                label="保留最近产物数",
                            )
                            artifact_cleanup_older_than_days = gr.Number(
                                value=7,
                                minimum=0,
                                precision=0,
                                label="清理多少天前",
                            )
                        with gr.Row():
                            artifact_cleanup_dry_run = gr.Checkbox(
                                label="仅预演",
                                value=True,
                            )
                            artifact_cleanup_limit = gr.Number(
                                value=200,
                                minimum=1,
                                precision=0,
                                label="最多扫描",
                            )
                        artifact_cleanup_btn = gr.Button("执行产物清理", variant="secondary")
                        artifact_action_result = gr.TextArea(
                            label="产物操作结果",
                            value="",
                            lines=6,
                            max_lines=10,
                            interactive=False,
                            autoscroll=True,
                        )
                with gr.Accordion("Ingest 运维", open=False):
                    with gr.Row():
                        ingest_refresh_btn = gr.Button("刷新 Ingest", variant="secondary")
                        ingest_fill_from_task_btn = gr.Button("带入选中任务", variant="secondary")
                        ingest_fill_from_artifact_btn = gr.Button("带入选中产物", variant="secondary")
                    with gr.Row():
                        ingest_parse_id = gr.Textbox(
                            label="Parse ID 过滤",
                            value="",
                            placeholder="可选，限定某次解析",
                        )
                        ingest_document_id = gr.Textbox(
                            label="Document ID 过滤",
                            value="",
                            placeholder="可选，限定某个文档",
                        )
                    with gr.Row():
                        ingest_parser_engine = gr.Textbox(
                            label="Parser Engine 过滤",
                            value="",
                            placeholder="如 deepdoc / markitdown / mineru",
                        )
                        ingest_file_type = gr.Textbox(
                            label="File Type 过滤",
                            value="",
                            placeholder="如 pdf / txt / docx",
                        )
                        ingest_tenant_id = gr.Textbox(
                            label="Tenant ID",
                            value="",
                            placeholder="仅 admin 跨租户时使用",
                        )
                    with gr.Row():
                        ingest_document_limit = gr.Slider(
                            minimum=5,
                            maximum=50,
                            step=5,
                            value=20,
                            label="最近 Ingest 文档数",
                        )
                    ingest_stats_md = gr.Markdown(value="暂无 ingest stats。")
                    ingest_documents_md = gr.Markdown(value="暂无 ingest 文档。")
                    with gr.Accordion("Record 查询", open=False):
                        with gr.Row():
                            ingest_record_query = gr.Textbox(
                                label="Record 文本过滤",
                                value="",
                                placeholder="如 architecture diagram",
                            )
                            ingest_record_limit = gr.Number(
                                value=10,
                                minimum=1,
                                precision=0,
                                label="返回条数",
                            )
                        ingest_record_search_btn = gr.Button("查询 Record", variant="secondary")
                        ingest_record_summary_md = gr.Markdown(value="尚未执行 ingest record 查询。")
                        ingest_record_raw_box = gr.TextArea(
                            label="Record 查询原始 JSON",
                            value="",
                            lines=10,
                            max_lines=16,
                            interactive=False,
                            autoscroll=True,
                        )
                    with gr.Accordion("Assets / Chunks", open=False):
                        with gr.Row():
                            ingest_asset_type = gr.Textbox(
                                label="Asset Type 过滤",
                                value="",
                                placeholder="如 figure / table / seal / equation",
                            )
                            ingest_chunk_id = gr.Textbox(
                                label="Chunk ID 过滤",
                                value="",
                                placeholder="可选，限定某个 chunk",
                            )
                            ingest_asset_id = gr.Textbox(
                                label="Asset ID 过滤",
                                value="",
                                placeholder="可选，限定某个 asset",
                            )
                        with gr.Row():
                            ingest_relation_type = gr.Dropdown(
                                ["all", "direct", "context"],
                                label="Link Relation",
                                value="all",
                            )
                            ingest_structured_limit = gr.Number(
                                value=20,
                                minimum=1,
                                precision=0,
                                label="结构化查询条数",
                            )
                        with gr.Row():
                            ingest_asset_search_btn = gr.Button("查看 Assets", variant="secondary")
                            ingest_chunk_search_btn = gr.Button("查看 Chunks", variant="secondary")
                            ingest_chunk_link_search_btn = gr.Button("查看 Chunk Asset Links", variant="secondary")
                        ingest_asset_summary_md = gr.Markdown(value="尚未查询 ingest assets。")
                        ingest_asset_raw_box = gr.TextArea(
                            label="Assets 原始 JSON",
                            value="",
                            lines=10,
                            max_lines=16,
                            interactive=False,
                            autoscroll=True,
                        )
                        ingest_chunk_summary_md = gr.Markdown(value="尚未查询 ingest chunks。")
                        ingest_chunk_raw_box = gr.TextArea(
                            label="Chunks 原始 JSON",
                            value="",
                            lines=10,
                            max_lines=16,
                            interactive=False,
                            autoscroll=True,
                        )
                        ingest_chunk_link_summary_md = gr.Markdown(value="尚未查询 chunk asset links。")
                        ingest_chunk_link_raw_box = gr.TextArea(
                            label="Chunk Asset Links 原始 JSON",
                            value="",
                            lines=10,
                            max_lines=16,
                            interactive=False,
                            autoscroll=True,
                        )
                with gr.Accordion("审计中心", open=False):
                    with gr.Row():
                        audit_center_refresh_btn = gr.Button("刷新审计", variant="secondary")
                        audit_center_limit = gr.Slider(
                            minimum=5,
                            maximum=50,
                            step=5,
                            value=20,
                            label="最近审计数",
                        )
                        audit_center_status_filter = gr.Dropdown(
                            AUDIT_EVENT_STATUS_FILTER_CHOICES,
                            label="审计状态",
                            value="all",
                        )
                    audit_center_action_filter = gr.Textbox(
                        label="Action 过滤",
                        value="",
                        placeholder="如 parse.async.submit / artifact.publish",
                    )
                    audit_list_md = gr.Markdown(value="暂无审计事件。")
                    audit_selector = gr.Dropdown(
                        label="选择审计事件",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    audit_center_summary = gr.Markdown(value="未选择审计事件。")
                    audit_center_raw_box = gr.TextArea(
                        label="选中审计原始 JSON",
                        value="",
                        lines=10,
                        max_lines=16,
                        interactive=False,
                        autoscroll=True,
                    )
                    audit_center_inspect_btn = gr.Button("刷新审计详情", variant="secondary")
                    with gr.Accordion("审计清理", open=False):
                        with gr.Row():
                            audit_cleanup_status = gr.Dropdown(
                                AUDIT_EVENT_STATUS_FILTER_CHOICES,
                                label="清理状态",
                                value="all",
                            )
                            audit_cleanup_action = gr.Textbox(
                                label="清理 Action",
                                value="",
                                placeholder="留空表示不过滤",
                            )
                        with gr.Row():
                            audit_cleanup_keep_latest = gr.Number(
                                value=50,
                                minimum=0,
                                precision=0,
                                label="保留最近审计数",
                            )
                            audit_cleanup_older_than_days = gr.Number(
                                value=14,
                                minimum=1,
                                precision=0,
                                label="清理多少天前",
                            )
                        with gr.Row():
                            audit_cleanup_dry_run = gr.Checkbox(
                                label="仅预演",
                                value=True,
                            )
                            audit_cleanup_limit = gr.Number(
                                value=500,
                                minimum=1,
                                precision=0,
                                label="最多扫描",
                            )
                        audit_cleanup_btn = gr.Button("执行审计清理", variant="secondary")
                        audit_cleanup_result = gr.TextArea(
                            label="审计清理结果",
                            value="",
                            lines=6,
                            max_lines=10,
                            interactive=False,
                            autoscroll=True,
                        )
                output_file = gr.File(label="转换结果", interactive=False)
                with gr.Blocks():
                    with gr.Tab("Markdown 渲染"):
                        md_kwargs = (
                            {"buttons": ["copy"]}
                            if IS_GRADIO_6
                            else {"show_copy_button": True}
                        )
                        md = gr.Markdown(
                            label="Markdown 渲染",
                            height=1200,
                            line_breaks=True,
                            **md_kwargs,
                        )
                    with gr.Tab("Markdown 文本"):
                        textarea_kwargs = (
                            {"buttons": ["copy"]}
                            if IS_GRADIO_6
                            else {"show_copy_button": True}
                        )
                        md_text = gr.TextArea(
                            lines=45,
                            label="Markdown 文本",
                            **textarea_kwargs,
                        )

        private_api_kwargs = (
            {"api_visibility": "private", "queue": False}
            if IS_GRADIO_6
            else {"queue": False}
        )

        deepdoc_tab.select(
            fn=lambda: "deepdoc",
            inputs=[],
            outputs=[parser_mode_state],
            **private_api_kwargs,
        )
        paddle_tab.select(
            fn=lambda: PADDLEOCR_VL_ENGINE,
            inputs=[],
            outputs=[parser_mode_state],
            **private_api_kwargs,
        )
        markitdown_tab.select(
            fn=lambda: MARKITDOWN_ENGINE,
            inputs=[],
            outputs=[parser_mode_state],
            **private_api_kwargs,
        )
        mineru_tab.select(
            fn=lambda: "mineru",
            inputs=[],
            outputs=[parser_mode_state],
            **private_api_kwargs,
        )

        status_box.change(
            fn=None,
            inputs=[status_box],
            outputs=[],
            js=STATUS_BOX_AUTOSCROLL_JS,
            **private_api_kwargs,
        )

        file_input.change(
            fn=_prepare_preview_update,
            inputs=[file_input],
            outputs=[doc_preview],
            **private_api_kwargs,
            show_progress="hidden",
        ).then(
            fn=_update_pdf_preview,
            inputs=[file_input],
            outputs=[doc_preview],
            **private_api_kwargs,
            show_progress="hidden",
        )

        ops_refresh_btn.click(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )

        self_check_refresh_btn.click(
            fn=_refresh_self_check_center,
            inputs=[self_check_limit, self_check_status_filter, self_check_center_selected_state],
            outputs=[
                self_check_list_md,
                self_check_selector,
                self_check_summary,
                self_check_steps_box,
                self_check_raw_box,
                self_check_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        self_check_status_filter.change(
            fn=_refresh_self_check_center,
            inputs=[self_check_limit, self_check_status_filter, self_check_center_selected_state],
            outputs=[
                self_check_list_md,
                self_check_selector,
                self_check_summary,
                self_check_steps_box,
                self_check_raw_box,
                self_check_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        self_check_limit.release(
            fn=_refresh_self_check_center,
            inputs=[self_check_limit, self_check_status_filter, self_check_center_selected_state],
            outputs=[
                self_check_list_md,
                self_check_selector,
                self_check_summary,
                self_check_steps_box,
                self_check_raw_box,
                self_check_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        self_check_selector.change(
            fn=_inspect_self_check_center,
            inputs=[self_check_selector],
            outputs=[
                self_check_summary,
                self_check_steps_box,
                self_check_raw_box,
                self_check_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        self_check_run_btn.click(
            fn=_run_self_check_console,
            inputs=[
                self_check_force_reparse,
                self_check_force_republish,
                self_check_limit,
                self_check_status_filter,
                self_check_center_selected_state,
            ],
            outputs=[
                self_check_action_result,
                self_check_list_md,
                self_check_selector,
                self_check_summary,
                self_check_steps_box,
                self_check_raw_box,
                self_check_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )
        self_check_cleanup_btn.click(
            fn=_cleanup_self_check_center,
            inputs=[
                self_check_cleanup_status,
                self_check_cleanup_keep_latest,
                self_check_cleanup_older_than_days,
                self_check_cleanup_dry_run,
                self_check_cleanup_limit,
                self_check_limit,
                self_check_status_filter,
                self_check_center_selected_state,
            ],
            outputs=[
                self_check_cleanup_result,
                self_check_list_md,
                self_check_selector,
                self_check_summary,
                self_check_steps_box,
                self_check_raw_box,
                self_check_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )

        run_btn.click(
            fn=prepare_run_ui,
            inputs=[],
            outputs=[
                status_box,
                output_file,
                md,
                md_text,
                artifact_info,
                task_events_box,
                run_btn,
                clear_btn,
                cancel_btn,
                task_id_state,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=run_parse,
            inputs=[
                file_input,
                parser_mode_state,
                deepdoc_layout_model,
                deepdoc_max_pages,
                paddle_use_gpu,
                paddle_prettify_markdown,
                paddle_show_formula_number,
                paddle_use_formula_recognition,
                paddle_table_enable,
                paddle_seal_enable,
                mineru_max_pages,
                mineru_use_gpu,
                mineru_language,
                mineru_is_ocr,
                mineru_formula_enable,
                mineru_table_enable,
            ],
            outputs=[
                status_box,
                output_file,
                md,
                md_text,
                artifact_info,
                task_events_box,
                run_btn,
                clear_btn,
                cancel_btn,
                task_id_state,
            ],
            queue=True,
            show_progress="hidden",
        ).then(
            fn=_refresh_task_center,
            inputs=[task_center_limit, task_center_status_filter, task_center_selected_state],
            outputs=[
                recent_tasks_md,
                task_center_selector,
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_artifact_center,
            inputs=[artifact_center_limit, artifact_center_publish_status_filter, artifact_center_selected_state],
            outputs=[
                artifact_list_md,
                artifact_selector,
                artifact_center_summary,
                artifact_publish_events_box,
                artifact_center_republish_btn,
                artifact_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )

        cancel_btn.click(
            fn=cancel_active_task,
            inputs=[task_id_state],
            outputs=[status_box, task_events_box, cancel_btn],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_task_center,
            inputs=[task_center_limit, task_center_status_filter, task_center_selected_state],
            outputs=[
                recent_tasks_md,
                task_center_selector,
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_artifact_center,
            inputs=[artifact_center_limit, artifact_center_publish_status_filter, artifact_center_selected_state],
            outputs=[
                artifact_list_md,
                artifact_selector,
                artifact_center_summary,
                artifact_publish_events_box,
                artifact_center_republish_btn,
                artifact_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )

        clear_btn.click(
            fn=_reset_ui_outputs,
            inputs=[],
            outputs=[file_input, status_box, output_file, md, md_text, artifact_info, task_events_box, doc_preview, cancel_btn, task_id_state],
            **private_api_kwargs,
            show_progress="hidden",
        )

        task_center_refresh_btn.click(
            fn=_refresh_task_center,
            inputs=[task_center_limit, task_center_status_filter, task_center_selected_state],
            outputs=[
                recent_tasks_md,
                task_center_selector,
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        task_center_status_filter.change(
            fn=_refresh_task_center,
            inputs=[task_center_limit, task_center_status_filter, task_center_selected_state],
            outputs=[
                recent_tasks_md,
                task_center_selector,
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        task_center_limit.release(
            fn=_refresh_task_center,
            inputs=[task_center_limit, task_center_status_filter, task_center_selected_state],
            outputs=[
                recent_tasks_md,
                task_center_selector,
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        task_center_selector.change(
            fn=_inspect_task_center_task,
            inputs=[task_center_selector],
            outputs=[
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        task_center_inspect_btn.click(
            fn=_inspect_task_center_task,
            inputs=[task_center_selector],
            outputs=[
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        task_center_cancel_btn.click(
            fn=cancel_active_task,
            inputs=[task_center_selected_state],
            outputs=[status_box, task_events_box, task_center_cancel_btn],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_task_center,
            inputs=[task_center_limit, task_center_status_filter, task_center_selected_state],
            outputs=[
                recent_tasks_md,
                task_center_selector,
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        task_center_retry_btn.click(
            fn=_retry_task_center_task,
            inputs=[task_center_selected_state],
            outputs=[
                status_box,
                task_center_events,
                task_center_retry_btn,
                task_center_summary,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_task_center,
            inputs=[task_center_limit, task_center_status_filter, task_center_selected_state],
            outputs=[
                recent_tasks_md,
                task_center_selector,
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )
        task_center_callback_retry_btn.click(
            fn=_retry_task_center_callback,
            inputs=[task_center_selected_state],
            outputs=[
                status_box,
                task_center_callback_events,
                task_center_callback_retry_btn,
                task_center_summary,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_task_center,
            inputs=[task_center_limit, task_center_status_filter, task_center_selected_state],
            outputs=[
                recent_tasks_md,
                task_center_selector,
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )
        task_cleanup_btn.click(
            fn=_cleanup_task_center_tasks,
            inputs=[
                task_cleanup_statuses,
                task_cleanup_keep_latest,
                task_cleanup_older_than_days,
                task_cleanup_include_active,
                task_cleanup_dry_run,
                task_cleanup_limit,
                task_center_limit,
                task_center_status_filter,
                task_center_selected_state,
            ],
            outputs=[
                task_cleanup_result,
                recent_tasks_md,
                task_center_selector,
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        task_cleanup_btn.click(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )

        artifact_center_refresh_btn.click(
            fn=_refresh_artifact_center,
            inputs=[artifact_center_limit, artifact_center_publish_status_filter, artifact_center_selected_state],
            outputs=[
                artifact_list_md,
                artifact_selector,
                artifact_center_summary,
                artifact_publish_events_box,
                artifact_center_republish_btn,
                artifact_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        artifact_center_publish_status_filter.change(
            fn=_refresh_artifact_center,
            inputs=[artifact_center_limit, artifact_center_publish_status_filter, artifact_center_selected_state],
            outputs=[
                artifact_list_md,
                artifact_selector,
                artifact_center_summary,
                artifact_publish_events_box,
                artifact_center_republish_btn,
                artifact_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        artifact_center_limit.release(
            fn=_refresh_artifact_center,
            inputs=[artifact_center_limit, artifact_center_publish_status_filter, artifact_center_selected_state],
            outputs=[
                artifact_list_md,
                artifact_selector,
                artifact_center_summary,
                artifact_publish_events_box,
                artifact_center_republish_btn,
                artifact_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        artifact_selector.change(
            fn=_inspect_artifact_center,
            inputs=[artifact_selector],
            outputs=[
                artifact_center_summary,
                artifact_publish_events_box,
                artifact_center_republish_btn,
                artifact_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        artifact_center_inspect_btn.click(
            fn=_inspect_artifact_center,
            inputs=[artifact_selector],
            outputs=[
                artifact_center_summary,
                artifact_publish_events_box,
                artifact_center_republish_btn,
                artifact_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        artifact_center_republish_btn.click(
            fn=_republish_artifact_center,
            inputs=[
                artifact_center_selected_state,
                artifact_center_limit,
                artifact_center_publish_status_filter,
                artifact_center_selected_state,
            ],
            outputs=[
                artifact_action_result,
                artifact_list_md,
                artifact_selector,
                artifact_center_summary,
                artifact_publish_events_box,
                artifact_center_republish_btn,
                artifact_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        artifact_center_republish_btn.click(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )
        artifact_cleanup_btn.click(
            fn=_cleanup_artifact_center,
            inputs=[
                artifact_cleanup_keep_latest,
                artifact_cleanup_older_than_days,
                artifact_cleanup_dry_run,
                artifact_cleanup_limit,
                artifact_center_limit,
                artifact_center_publish_status_filter,
                artifact_center_selected_state,
            ],
            outputs=[
                artifact_action_result,
                artifact_list_md,
                artifact_selector,
                artifact_center_summary,
                artifact_publish_events_box,
                artifact_center_republish_btn,
                artifact_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        artifact_cleanup_btn.click(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )

        ingest_refresh_btn.click(
            fn=_refresh_ingest_console,
            inputs=[
                ingest_parse_id,
                ingest_document_id,
                ingest_parser_engine,
                ingest_file_type,
                ingest_tenant_id,
                ingest_document_limit,
            ],
            outputs=[ingest_stats_md, ingest_documents_md],
            queue=False,
            show_progress="hidden",
        )
        ingest_document_limit.release(
            fn=_refresh_ingest_console,
            inputs=[
                ingest_parse_id,
                ingest_document_id,
                ingest_parser_engine,
                ingest_file_type,
                ingest_tenant_id,
                ingest_document_limit,
            ],
            outputs=[ingest_stats_md, ingest_documents_md],
            queue=False,
            show_progress="hidden",
        )
        ingest_fill_from_task_btn.click(
            fn=_prefill_ingest_target_from_task,
            inputs=[task_center_selected_state],
            outputs=[ingest_parse_id, ingest_document_id],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_ingest_console,
            inputs=[
                ingest_parse_id,
                ingest_document_id,
                ingest_parser_engine,
                ingest_file_type,
                ingest_tenant_id,
                ingest_document_limit,
            ],
            outputs=[ingest_stats_md, ingest_documents_md],
            queue=False,
            show_progress="hidden",
        )
        ingest_fill_from_artifact_btn.click(
            fn=_prefill_ingest_target_from_artifact,
            inputs=[artifact_center_selected_state],
            outputs=[ingest_parse_id, ingest_document_id],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=_refresh_ingest_console,
            inputs=[
                ingest_parse_id,
                ingest_document_id,
                ingest_parser_engine,
                ingest_file_type,
                ingest_tenant_id,
                ingest_document_limit,
            ],
            outputs=[ingest_stats_md, ingest_documents_md],
            queue=False,
            show_progress="hidden",
        )
        ingest_record_search_btn.click(
            fn=_query_ingest_console_records,
            inputs=[
                ingest_record_query,
                ingest_parse_id,
                ingest_document_id,
                ingest_tenant_id,
                ingest_record_limit,
            ],
            outputs=[ingest_record_summary_md, ingest_record_raw_box],
            queue=False,
            show_progress="hidden",
        )
        ingest_asset_search_btn.click(
            fn=_search_ingest_console_assets,
            inputs=[
                ingest_parse_id,
                ingest_document_id,
                ingest_tenant_id,
                ingest_asset_type,
                ingest_structured_limit,
            ],
            outputs=[ingest_asset_summary_md, ingest_asset_raw_box],
            queue=False,
            show_progress="hidden",
        )
        ingest_chunk_search_btn.click(
            fn=_search_ingest_console_chunks,
            inputs=[
                ingest_parse_id,
                ingest_document_id,
                ingest_tenant_id,
                ingest_chunk_id,
                ingest_structured_limit,
            ],
            outputs=[ingest_chunk_summary_md, ingest_chunk_raw_box],
            queue=False,
            show_progress="hidden",
        )
        ingest_chunk_link_search_btn.click(
            fn=_search_ingest_console_chunk_asset_links,
            inputs=[
                ingest_parse_id,
                ingest_document_id,
                ingest_tenant_id,
                ingest_chunk_id,
                ingest_asset_id,
                ingest_relation_type,
                ingest_structured_limit,
            ],
            outputs=[ingest_chunk_link_summary_md, ingest_chunk_link_raw_box],
            queue=False,
            show_progress="hidden",
        )

        audit_center_refresh_btn.click(
            fn=_refresh_ops_audit_center,
            inputs=[audit_center_limit, audit_center_status_filter, audit_center_action_filter, audit_center_selected_state],
            outputs=[
                audit_list_md,
                audit_selector,
                audit_center_summary,
                audit_center_raw_box,
                audit_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        audit_center_status_filter.change(
            fn=_refresh_ops_audit_center,
            inputs=[audit_center_limit, audit_center_status_filter, audit_center_action_filter, audit_center_selected_state],
            outputs=[
                audit_list_md,
                audit_selector,
                audit_center_summary,
                audit_center_raw_box,
                audit_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        audit_center_limit.release(
            fn=_refresh_ops_audit_center,
            inputs=[audit_center_limit, audit_center_status_filter, audit_center_action_filter, audit_center_selected_state],
            outputs=[
                audit_list_md,
                audit_selector,
                audit_center_summary,
                audit_center_raw_box,
                audit_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        audit_center_action_filter.change(
            fn=_refresh_ops_audit_center,
            inputs=[audit_center_limit, audit_center_status_filter, audit_center_action_filter, audit_center_selected_state],
            outputs=[
                audit_list_md,
                audit_selector,
                audit_center_summary,
                audit_center_raw_box,
                audit_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        audit_selector.change(
            fn=_inspect_ops_audit_center,
            inputs=[audit_selector],
            outputs=[audit_center_summary, audit_center_raw_box, audit_center_selected_state],
            queue=False,
            show_progress="hidden",
        )
        audit_center_inspect_btn.click(
            fn=_inspect_ops_audit_center,
            inputs=[audit_selector],
            outputs=[audit_center_summary, audit_center_raw_box, audit_center_selected_state],
            queue=False,
            show_progress="hidden",
        )
        audit_cleanup_btn.click(
            fn=_cleanup_ops_audit_center,
            inputs=[
                audit_cleanup_status,
                audit_cleanup_action,
                audit_cleanup_keep_latest,
                audit_cleanup_older_than_days,
                audit_cleanup_dry_run,
                audit_cleanup_limit,
                audit_center_limit,
                audit_center_status_filter,
                audit_center_action_filter,
                audit_center_selected_state,
            ],
            outputs=[
                audit_cleanup_result,
                audit_list_md,
                audit_selector,
                audit_center_summary,
                audit_center_raw_box,
                audit_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        audit_cleanup_btn.click(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )

        demo.load(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )
        ops_auto_refresh_timer.tick(
            fn=_refresh_ops_overview,
            inputs=[],
            outputs=[ops_health_md, ops_ingest_md, ops_alerts_md, ops_failures_md, ops_activity_md, ops_metrics_md],
            queue=False,
            show_progress="hidden",
        )
        demo.load(
            fn=_refresh_task_center,
            inputs=[task_center_limit, task_center_status_filter, task_center_selected_state],
            outputs=[
                recent_tasks_md,
                task_center_selector,
                task_center_summary,
                task_center_artifact,
                task_center_events,
                task_center_callback_events,
                task_center_cancel_btn,
                task_center_callback_retry_btn,
                task_center_retry_btn,
                task_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        demo.load(
            fn=_restore_latest_task_events,
            inputs=[],
            outputs=[task_events_box],
            queue=False,
            show_progress="hidden",
        )
        demo.load(
            fn=_refresh_artifact_center,
            inputs=[artifact_center_limit, artifact_center_publish_status_filter, artifact_center_selected_state],
            outputs=[
                artifact_list_md,
                artifact_selector,
                artifact_center_summary,
                artifact_publish_events_box,
                artifact_center_republish_btn,
                artifact_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        demo.load(
            fn=_refresh_ingest_console,
            inputs=[
                ingest_parse_id,
                ingest_document_id,
                ingest_parser_engine,
                ingest_file_type,
                ingest_tenant_id,
                ingest_document_limit,
            ],
            outputs=[ingest_stats_md, ingest_documents_md],
            queue=False,
            show_progress="hidden",
        )
        demo.load(
            fn=_refresh_self_check_center,
            inputs=[self_check_limit, self_check_status_filter, self_check_center_selected_state],
            outputs=[
                self_check_list_md,
                self_check_selector,
                self_check_summary,
                self_check_steps_box,
                self_check_raw_box,
                self_check_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )
        demo.load(
            fn=_refresh_ops_audit_center,
            inputs=[audit_center_limit, audit_center_status_filter, audit_center_action_filter, audit_center_selected_state],
            outputs=[
                audit_list_md,
                audit_selector,
                audit_center_summary,
                audit_center_raw_box,
                audit_center_selected_state,
            ],
            queue=False,
            show_progress="hidden",
        )

    demo.queue(default_concurrency_limit=None)
    return demo


if __name__ == "__main__":
    app = build_app()
    nonblocking_launch = os.environ.get("GRADIO_PREVENT_THREAD_LOCK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": int(os.environ.get("GRADIO_PORT", "7860")),
        "root_path": os.environ.get("GRADIO_ROOT_PATH", ""),
        "show_error": True,
    }
    if nonblocking_launch:
        launch_kwargs["prevent_thread_lock"] = True

    launch_params = inspect.signature(app.launch).parameters
    if "show_api" in launch_params:
        launch_kwargs["show_api"] = False
    elif "footer_links" in launch_params:
        launch_kwargs["footer_links"] = ["gradio", "settings"]

    logger.info(
        "Starting Gradio server on %s:%s root_path=%s",
        launch_kwargs.get("server_name"),
        launch_kwargs.get("server_port"),
        launch_kwargs.get("root_path") or "/",
    )

    app.launch(**launch_kwargs)
    if nonblocking_launch:
        logger.info("Gradio server launched in non-blocking mode; entering keepalive loop.")
        while True:
            time.sleep(3600)

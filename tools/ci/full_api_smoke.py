#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class SmokeResult:
    name: str
    method: str
    path: str
    status: int
    expected: list[int]
    passed: bool
    content_type: str = ""
    body_len: int = 0
    notes: str = ""


@dataclass
class SmokeContext:
    base_url: str
    output_dir: Path
    timeout: float
    results: list[SmokeResult] = field(default_factory=list)
    parse_id: str | None = None
    document_id: str | None = None
    asset_parse_id: str | None = None
    asset_id: str | None = None
    asset_filename: str | None = None
    check_id: str | None = None
    task_id: str | None = None
    retry_task_id: str | None = None
    event_id: str | None = None


class CallbackCapture:
    def __init__(self) -> None:
        self.payloads: list[Json] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "CallbackCapture":
        capture = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                    if isinstance(payload, dict):
                        capture.payloads.append(payload)
                except Exception:
                    capture.payloads.append({"raw": raw.decode("utf-8", errors="replace")})
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("callback server is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}/callback"


def _disable_proxy_for_local_base_url(base_url: str) -> None:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        for name in ("NO_PROXY", "no_proxy"):
            existing = os.environ.get(name, "")
            entries = [item.strip() for item in existing.split(",") if item.strip()]
            for value in ("127.0.0.1", "localhost", "::1"):
                if value not in entries:
                    entries.append(value)
            os.environ[name] = ",".join(entries)
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))


def _resolve_url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _read_response(response) -> tuple[int, str, bytes]:
    status = int(response.status)
    content_type = str(response.headers.get("Content-Type") or "")
    body = response.read()
    return status, content_type, body


def _request(
    ctx: SmokeContext,
    *,
    name: str,
    method: str,
    path: str,
    expected: list[int] | tuple[int, ...] = (200,),
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> tuple[int, str, bytes]:
    request = urllib.request.Request(
        _resolve_url(ctx.base_url, path),
        data=data,
        method=method,
        headers=headers or {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout or ctx.timeout) as response:
            status, content_type, body = _read_response(response)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        content_type = str(exc.headers.get("Content-Type") or "")
        body = exc.read()
    expected_values = list(expected)
    passed = status in expected_values
    ctx.results.append(
        SmokeResult(
            name=name,
            method=method,
            path=path,
            status=status,
            expected=expected_values,
            passed=passed,
            content_type=content_type,
            body_len=len(body),
        )
    )
    raw_path = ctx.output_dir / "raw" / f"{len(ctx.results):03d}-{_safe_name(name)}.body"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(body)
    return status, content_type, body


def _request_json(
    ctx: SmokeContext,
    *,
    name: str,
    method: str,
    path: str,
    expected: list[int] | tuple[int, ...] = (200,),
    payload: Json | None = None,
    timeout: float | None = None,
) -> Json:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    status, _, raw = _request(
        ctx,
        name=name,
        method=method,
        path=path,
        expected=expected,
        data=body,
        headers=headers,
        timeout=timeout,
    )
    if status not in expected:
        return {}
    if not raw:
        return {}
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{name} returned non-object JSON")
    return parsed


def _multipart_request(
    ctx: SmokeContext,
    *,
    name: str,
    path: str,
    file_path: Path,
    fields: dict[str, str],
    expected: list[int] | tuple[int, ...] = (200,),
    timeout: float | None = None,
) -> Json:
    boundary = f"deepdoc-full-smoke-{int(time.time() * 1000)}"
    body = bytearray()
    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    status, _, raw = _request(
        ctx,
        name=name,
        method="POST",
        path=path,
        expected=expected,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=timeout,
    )
    if status not in expected:
        return {}
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{name} returned non-object JSON")
    return parsed


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value).strip("-")[:96] or "response"


def _jsonl_rows(raw: bytes) -> list[Json]:
    rows: list[Json] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _first_result(payload: Json) -> Json:
    results = payload.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise RuntimeError("response missing results[0]")
    return results[0]


def _write_worker_heartbeat(state: str, task_id: str | None = None) -> None:
    path = os.environ.get("DEEPDOC_ASYNC_WORKER_HEARTBEAT_FILE")
    if not path:
        return
    payload: Json = {
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if task_id:
        payload["task_id"] = task_id
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_async_worker_until(task_id: str, timeout_seconds: int) -> None:
    import main

    _write_worker_heartbeat("starting", task_id)
    deadline = time.time() + timeout_seconds
    seen = False
    while time.time() < deadline:
        reserved = main.ASYNC_TASK_BROKER.reserve(timeout_seconds=1, task_store=main.ASYNC_TASK_STORE)
        if not reserved:
            if seen:
                return
            continue
        seen = True
        try:
            _write_worker_heartbeat("running", reserved)
            main.run_async_parse_task(reserved)
        finally:
            main.ASYNC_TASK_BROKER.ack(reserved)
            _write_worker_heartbeat("idle", reserved)
        if reserved == task_id:
            return
    raise RuntimeError(f"async worker did not process task before timeout: {task_id}")


def _wait_for_ready(ctx: SmokeContext, timeout_seconds: int) -> Json:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            payload = _request_json(
                ctx,
                name="GET /ready wait",
                method="GET",
                path="/ready",
                expected=(200, 503),
                timeout=5,
            )
            if payload.get("status") == "ready":
                return payload
            last_error = json.dumps(payload, ensure_ascii=False)[:500]
        except Exception as exc:
            last_error = str(exc)
        if ctx.results:
            ctx.results.pop()
        time.sleep(1)
    raise RuntimeError(f"service did not become ready before timeout: {last_error}")


def _make_fixtures(root: Path) -> dict[str, Path]:
    from PIL import Image, ImageDraw
    import fitz

    fixtures = root / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    txt = fixtures / "sample.txt"
    txt.write_text(
        "DocPilot full API smoke validates parse, artifacts, chunks, async tasks, audit, self-check, and ingest query APIs.\n"
        "This document is intentionally small so CPU and GPU runs stay comparable.\n",
        encoding="utf-8",
    )
    png = fixtures / "ocr.png"
    image = Image.new("RGB", (720, 260), "white")
    canvas = ImageDraw.Draw(image)
    canvas.text((36, 36), "DocPilot OCR smoke", fill="black")
    canvas.text((36, 92), "Document ID: FULL-API-2026", fill="black")
    canvas.text((36, 148), "The OCR endpoint should return JSON.", fill="black")
    image.save(png)

    pdf = fixtures / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "DocPilot Full API PDF Smoke", fontsize=18)
    page.insert_text((72, 120), "This PDF is used for a small parser smoke check.", fontsize=11)
    doc.save(pdf, garbage=4, deflate=True)
    doc.close()
    return {"txt": txt, "png": png, "pdf": pdf}


def _assert_nonempty(value: Any, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def _update_last_note(ctx: SmokeContext, note: str) -> None:
    if ctx.results:
        ctx.results[-1].notes = note


def run_smoke(ctx: SmokeContext, *, async_worker_timeout: int) -> Json:
    fixtures = _make_fixtures(ctx.output_dir)
    _write_worker_heartbeat("idle")
    ready = _wait_for_ready(ctx, timeout_seconds=180)

    _request_json(ctx, name="GET /health", method="GET", path="/health")
    _request_json(ctx, name="GET /ready", method="GET", path="/ready")
    _request(ctx, name="GET /metrics", method="GET", path="/metrics")
    _request_json(ctx, name="GET /api/v1/build-info", method="GET", path="/api/v1/build-info")
    _request_json(ctx, name="GET /openapi.json", method="GET", path="/openapi.json")
    _request_json(ctx, name="GET /api/v1/openapi.json", method="GET", path="/api/v1/openapi.json")
    _request_json(ctx, name="GET /docs/openapi.json", method="GET", path="/docs/openapi.json")
    _request(ctx, name="GET /docs/", method="GET", path="/docs/")

    ocr = _multipart_request(
        ctx,
        name="POST /api/v1/ocr",
        path="/api/v1/ocr",
        file_path=fixtures["png"],
        fields={},
        timeout=120,
    )
    _assert_nonempty(ocr.get("ocr_result") is not None, "ocr response missing ocr_result")

    parse_payload = _multipart_request(
        ctx,
        name="POST /api/v1/parse",
        path="/api/v1/parse",
        file_path=fixtures["txt"],
        fields={
            "return_structured": "true",
            "persist_artifacts": "true",
            "include_chunks": "true",
            "publish_ingest": "true",
            "chunk_strategy": "structure_aware",
        },
        timeout=120,
    )
    first = _first_result(parse_payload)
    ctx.parse_id = str(first.get("parse_id") or "")
    ctx.document_id = str(first.get("document_id") or "")
    _assert_nonempty(ctx.parse_id, "parse response missing parse_id")
    _assert_nonempty(first.get("structured"), "parse response missing structured artifact")
    _update_last_note(ctx, f"parse_id={ctx.parse_id}, chunks={first.get('chunk_count')}")

    download_body, download_content_type = _build_multipart_body(fixtures["txt"], {"parser_engine": "deepdoc"})
    _request(
        ctx,
        name="POST /api/v1/parse?download=true",
        method="POST",
        path="/api/v1/parse?download=true",
        data=download_body,
        headers={"Content-Type": download_content_type},
        timeout=120,
    )

    stream_body = _build_multipart_body(
        fixtures["txt"],
        {
            "return_structured": "true",
            "persist_artifacts": "true",
            "include_chunks": "true",
            "chunk_strategy": "structure_aware",
        },
    )
    status, _, raw_stream = _request(
        ctx,
        name="POST /api/v1/parse/stream",
        method="POST",
        path="/api/v1/parse/stream",
        data=stream_body[0],
        headers={"Content-Type": stream_body[1]},
        timeout=120,
    )
    if status == 200 and b"event: done" not in raw_stream:
        raise RuntimeError("parse stream did not emit done event")
    if status == 200 and b'"status": "ok"' not in raw_stream:
        raise RuntimeError("parse stream did not report status=ok")
    if status == 200 and b'"has_error": true' in raw_stream:
        raise RuntimeError("parse stream reported a file error")

    with CallbackCapture() as callback:
        async_payload = _multipart_request(
            ctx,
            name="POST /api/v1/parse/async",
            path="/api/v1/parse/async",
            file_path=fixtures["txt"],
            fields={
                "return_structured": "true",
                "include_chunks": "true",
                "chunk_strategy": "structure_aware",
                "callback_url": callback.url,
                "callback_events": "terminal",
                "callback_max_attempts": "1",
                "callback_timeout_seconds": "5",
            },
            expected=(202,),
            timeout=120,
        )
        ctx.task_id = str(async_payload.get("task_id") or async_payload.get("id") or "")
        _assert_nonempty(ctx.task_id, "async parse response missing task_id")
        _run_async_worker_until(ctx.task_id, async_worker_timeout)
        task_payload = _request_json(
            ctx,
            name="GET /api/v1/tasks/{task_id}",
            method="GET",
            path=f"/api/v1/tasks/{ctx.task_id}",
        )
        if task_payload.get("status") != "succeeded":
            raise RuntimeError(f"async task did not succeed: {task_payload.get('status')}")
        _assert_nonempty(callback.payloads, "async callback server did not receive a terminal event")

        _request_json(ctx, name="GET /api/v1/tasks", method="GET", path="/api/v1/tasks?limit=10")
        _request_json(ctx, name="GET /api/v1/tasks/{task_id}/events", method="GET", path=f"/api/v1/tasks/{ctx.task_id}/events")
        _request(ctx, name="GET /api/v1/tasks/{task_id}/stream", method="GET", path=f"/api/v1/tasks/{ctx.task_id}/stream?timeout_seconds=10&poll_seconds=1")
        _request_json(ctx, name="GET /api/v1/tasks/{task_id}/callback-events", method="GET", path=f"/api/v1/tasks/{ctx.task_id}/callback-events")
        _request_json(
            ctx,
            name="POST /api/v1/tasks/{task_id}/callback/retry",
            method="POST",
            path=f"/api/v1/tasks/{ctx.task_id}/callback/retry",
            payload={"force": True, "requested_by": "full-api-smoke"},
        )
        retry_payload = _request_json(
            ctx,
            name="POST /api/v1/tasks/{task_id}/retry",
            method="POST",
            path=f"/api/v1/tasks/{ctx.task_id}/retry",
            payload={"force": True, "requested_by": "full-api-smoke"},
            expected=(202,),
        )
        retry_task = retry_payload.get("retry_task") if isinstance(retry_payload.get("retry_task"), dict) else {}
        ctx.retry_task_id = str(retry_task.get("task_id") or "")
        if ctx.retry_task_id:
            _run_async_worker_until(ctx.retry_task_id, async_worker_timeout)
        _request_json(
            ctx,
            name="POST /api/v1/tasks/{task_id}/cancel",
            method="POST",
            path=f"/api/v1/tasks/{ctx.task_id}/cancel",
            payload={},
        )

    _request_json(
        ctx,
        name="POST /api/v1/tasks/retry",
        method="POST",
        path="/api/v1/tasks/retry",
        payload={"dry_run": True, "force": True, "task_statuses": ["succeeded"], "limit": 5},
    )
    _request_json(
        ctx,
        name="POST /api/v1/tasks/callbacks/retry",
        method="POST",
        path="/api/v1/tasks/callbacks/retry",
        payload={"dry_run": True, "limit": 5},
    )
    _request_json(
        ctx,
        name="POST /api/v1/tasks/cleanup",
        method="POST",
        path="/api/v1/tasks/cleanup",
        payload={"dry_run": True, "statuses": ["succeeded"], "keep_latest": 1, "limit": 20},
    )

    self_check = _request_json(
        ctx,
        name="POST /api/v1/self-checks/run",
        method="POST",
        path="/api/v1/self-checks/run",
        payload={"suite": "core", "force_reparse": True, "force_republish": True},
        timeout=240,
    )
    ctx.check_id = str(self_check.get("check_id") or "")
    if self_check.get("status") != "passed":
        raise RuntimeError(f"self-check did not pass: {self_check.get('summary')}")
    _request_json(ctx, name="GET /api/v1/self-checks", method="GET", path="/api/v1/self-checks?limit=10")
    _request_json(ctx, name="GET /api/v1/self-checks/{check_id}", method="GET", path=f"/api/v1/self-checks/{ctx.check_id}")
    _request_json(
        ctx,
        name="POST /api/v1/self-checks/cleanup",
        method="POST",
        path="/api/v1/self-checks/cleanup",
        payload={"dry_run": True, "keep_latest": 1, "limit": 20},
    )
    _extract_asset_context_from_self_check(ctx, self_check)

    _request_json(ctx, name="GET /api/v1/artifacts", method="GET", path="/api/v1/artifacts?limit=20")
    _request_json(ctx, name="GET /api/v1/artifacts/{parse_id}/manifest", method="GET", path=f"/api/v1/artifacts/{ctx.parse_id}/manifest")
    _request_json(ctx, name="GET /api/v1/artifacts/{parse_id}/structured", method="GET", path=f"/api/v1/artifacts/{ctx.parse_id}/structured")
    _request(ctx, name="GET /api/v1/artifacts/{parse_id}/markdown", method="GET", path=f"/api/v1/artifacts/{ctx.parse_id}/markdown")
    _, _, chunk_body = _request(ctx, name="GET /api/v1/artifacts/{parse_id}/chunks", method="GET", path=f"/api/v1/artifacts/{ctx.parse_id}/chunks")
    _assert_nonempty(_jsonl_rows(chunk_body), "chunks artifact returned no rows")
    _, _, ingest_body = _request(ctx, name="GET /api/v1/artifacts/{parse_id}/ingest", method="GET", path=f"/api/v1/artifacts/{ctx.parse_id}/ingest")
    _assert_nonempty(_jsonl_rows(ingest_body), "ingest artifact returned no rows")
    _request(ctx, name="GET /api/v1/artifacts/{parse_id}/publish-events", method="GET", path=f"/api/v1/artifacts/{ctx.parse_id}/publish-events")
    _request_json(
        ctx,
        name="POST /api/v1/artifacts/{parse_id}/publish",
        method="POST",
        path=f"/api/v1/artifacts/{ctx.parse_id}/publish",
        payload={"force": True, "requested_by": "full-api-smoke"},
    )
    _request_json(
        ctx,
        name="POST /api/v1/artifacts/publish-retry",
        method="POST",
        path="/api/v1/artifacts/publish-retry",
        payload={"force": True, "limit": 5},
    )
    _request_json(
        ctx,
        name="POST /api/v1/artifacts/cleanup",
        method="POST",
        path="/api/v1/artifacts/cleanup",
        payload={"dry_run": True, "keep_latest": 1, "limit": 20},
    )

    if ctx.asset_parse_id and ctx.asset_filename:
        _request(
            ctx,
            name="GET /api/v1/artifacts/{parse_id}/assets/{filename}",
            method="GET",
            path=f"/api/v1/artifacts/{ctx.asset_parse_id}/assets/{ctx.asset_filename}",
        )

    _request_json(ctx, name="GET /api/v1/ingest/documents", method="GET", path="/api/v1/ingest/documents?limit=20")
    _request_json(ctx, name="GET /api/v1/ingest/stats", method="GET", path=f"/api/v1/ingest/stats?parse_id={ctx.parse_id}")
    _request_json(ctx, name="GET /api/v1/ingest/documents/{parse_id}", method="GET", path=f"/api/v1/ingest/documents/{ctx.parse_id}")
    _request_json(ctx, name="GET /api/v1/ingest/records", method="GET", path=f"/api/v1/ingest/records?parse_id={ctx.parse_id}&limit=20")
    _request_json(ctx, name="GET /api/v1/ingest/chunks", method="GET", path=f"/api/v1/ingest/chunks?parse_id={ctx.parse_id}&limit=20")
    if ctx.asset_parse_id:
        assets_payload = _request_json(ctx, name="GET /api/v1/ingest/assets", method="GET", path=f"/api/v1/ingest/assets?parse_id={ctx.asset_parse_id}&limit=20")
        asset_results = assets_payload.get("results") if isinstance(assets_payload.get("results"), list) else []
        if asset_results and isinstance(asset_results[0], dict):
            ctx.asset_id = str(asset_results[0].get("asset_id") or ctx.asset_id or "")
        if ctx.asset_id:
            _request_json(
                ctx,
                name="GET /api/v1/ingest/assets/{parse_id}/{asset_id}",
                method="GET",
                path=f"/api/v1/ingest/assets/{ctx.asset_parse_id}/{ctx.asset_id}",
            )
        _request_json(
            ctx,
            name="GET /api/v1/ingest/chunk-asset-links",
            method="GET",
            path=f"/api/v1/ingest/chunk-asset-links?parse_id={ctx.asset_parse_id}&limit=20",
        )

    audit_list = _request_json(ctx, name="GET /api/v1/audit/events", method="GET", path="/api/v1/audit/events?limit=20")
    audit_results = audit_list.get("results") if isinstance(audit_list.get("results"), list) else []
    if audit_results and isinstance(audit_results[0], dict):
        ctx.event_id = str(audit_results[0].get("event_id") or "")
    if ctx.event_id:
        _request_json(ctx, name="GET /api/v1/audit/events/{event_id}", method="GET", path=f"/api/v1/audit/events/{ctx.event_id}")
    _request_json(
        ctx,
        name="POST /api/v1/audit/events/cleanup",
        method="POST",
        path="/api/v1/audit/events/cleanup",
        payload={"dry_run": True, "keep_latest": 1},
    )

    summary = {
        "base_url": ctx.base_url,
        "ready": ready,
        "parse_id": ctx.parse_id,
        "asset_parse_id": ctx.asset_parse_id,
        "asset_id": ctx.asset_id,
        "task_id": ctx.task_id,
        "retry_task_id": ctx.retry_task_id,
        "check_id": ctx.check_id,
        "event_id": ctx.event_id,
        "result_count": len(ctx.results),
        "passed_count": sum(1 for result in ctx.results if result.passed),
        "failed_count": sum(1 for result in ctx.results if not result.passed),
        "results": [result.__dict__ for result in ctx.results],
    }
    return summary


def _build_multipart_body(file_path: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"deepdoc-full-smoke-{int(time.time() * 1000)}"
    body = bytearray()
    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _extract_asset_context_from_self_check(ctx: SmokeContext, self_check: Json) -> None:
    steps = self_check.get("steps")
    if not isinstance(steps, list):
        return
    for step in steps:
        if not isinstance(step, dict):
            continue
        details = step.get("details")
        if not isinstance(details, dict):
            continue
        parse_id = str(details.get("parse_id") or "")
        if not parse_id:
            continue
        if int(details.get("asset_count") or details.get("asset_linked_chunk_count") or 0) <= 0:
            continue
        ctx.asset_parse_id = parse_id
        structured = _request_json(
            ctx,
            name="GET /api/v1/artifacts/{asset_parse_id}/structured",
            method="GET",
            path=f"/api/v1/artifacts/{parse_id}/structured",
        )
        assets = structured.get("assets") if isinstance(structured.get("assets"), list) else []
        if assets and isinstance(assets[0], dict):
            ctx.asset_id = str(assets[0].get("asset_id") or "")
            storage = assets[0].get("storage") if isinstance(assets[0].get("storage"), dict) else {}
            relative_path = str(storage.get("relative_path") or storage.get("download_path") or "")
            if relative_path:
                ctx.asset_filename = Path(relative_path).name
        return


def _write_summary(ctx: SmokeContext, summary: Json) -> Path:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ctx.output_dir / "full_api_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run broad DocPilot API smoke coverage against a running service.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--async-worker-timeout", type=int, default=120)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    _disable_proxy_for_local_base_url(base_url)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = SmokeContext(base_url=base_url, output_dir=output_dir, timeout=float(args.timeout))

    try:
        summary = run_smoke(ctx, async_worker_timeout=int(args.async_worker_timeout))
    finally:
        summary = {
            "base_url": ctx.base_url,
            "parse_id": ctx.parse_id,
            "asset_parse_id": ctx.asset_parse_id,
            "asset_id": ctx.asset_id,
            "task_id": ctx.task_id,
            "retry_task_id": ctx.retry_task_id,
            "check_id": ctx.check_id,
            "event_id": ctx.event_id,
            "result_count": len(ctx.results),
            "passed_count": sum(1 for result in ctx.results if result.passed),
            "failed_count": sum(1 for result in ctx.results if not result.passed),
            "results": [result.__dict__ for result in ctx.results],
        } if "summary" not in locals() else summary
        summary_path = _write_summary(ctx, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"summary_path={summary_path}")

    failed = [result for result in ctx.results if not result.passed]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import tempfile
import time
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _fetch_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected object response from {url}")
    return payload


def _fetch_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _parse_jsonl(payload: str) -> list[dict]:
    rows: list[dict] = []
    for line in payload.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        row = json.loads(normalized)
        if not isinstance(row, dict):
            raise RuntimeError("JSONL row is not an object")
        rows.append(row)
    return rows


def _wait_for_ready(base_url: str, timeout_seconds: int) -> dict:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    ready_url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "ready")
    while time.time() < deadline:
        try:
            payload = _fetch_json(ready_url, timeout=5.0)
            if payload.get("status") == "ready":
                return payload
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionResetError, socket.timeout, TimeoutError, OSError, ValueError) as exc:  # pragma: no cover - exercised in CI
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"service did not become ready before timeout: {last_error}")


def _multipart_request(url: str, file_path: Path, fields: dict[str, str]) -> dict:
    boundary = f"deepdoc-ci-{int(time.time() * 1000)}"
    body = bytearray()
    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("parse response is not an object")
    return payload


def _resolve_url(base_url: str, candidate: str) -> str:
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", candidate)


def _disable_proxy_for_local_base_url(base_url: str) -> None:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        urllib.request.install_opener(
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
        )


def _validate_build_info(build_info: dict, *, allow_runtime_build_info: bool) -> None:
    if build_info.get("status") == "ok":
        return
    if allow_runtime_build_info and build_info.get("build_source") == "runtime-fallback":
        return
    raise RuntimeError("build-info endpoint did not return status=ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight container smoke test against a DocPilot instance.")
    parser.add_argument("--base-url", required=True, help="Base URL of the running DocPilot service, for example http://127.0.0.1:18000")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--allow-runtime-build-info",
        action="store_true",
        help="Allow source-tree local runs where /api/v1/build-info reports runtime-fallback instead of embedded metadata.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    _disable_proxy_for_local_base_url(base_url)
    ready = _wait_for_ready(base_url, timeout_seconds=args.timeout_seconds)
    print(f"ready status: {ready.get('status')}")

    docs_payload = urllib.request.urlopen(_resolve_url(base_url, "/docs/"), timeout=10.0).read().decode("utf-8")
    if "SwaggerUIBundle" not in docs_payload and 'id="swagger-ui"' not in docs_payload:
        raise RuntimeError("Swagger UI page missing expected marker")

    openapi = _fetch_json(_resolve_url(base_url, "/openapi.json"))
    if not isinstance(openapi.get("paths"), dict) or "/api/v1/parse" not in openapi["paths"]:
        raise RuntimeError("OpenAPI payload missing /api/v1/parse")

    build_info = _fetch_json(_resolve_url(base_url, "/api/v1/build-info"))
    _validate_build_info(build_info, allow_runtime_build_info=args.allow_runtime_build_info)

    with tempfile.TemporaryDirectory(prefix="deepdoc-ci-") as temp_dir:
        sample_path = Path(temp_dir) / "smoke.txt"
        sample_path.write_text(
            "DocPilot CI smoke validates structured artifacts and markdown export without OCR model downloads.\n",
            encoding="utf-8",
        )
        parse_payload = _multipart_request(
            _resolve_url(base_url, "/api/v1/parse"),
            sample_path,
            {
                "return_structured": "true",
                "persist_artifacts": "true",
                "include_chunks": "true",
                "chunk_strategy": "asset_aware",
            },
        )

    results = parse_payload.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("parse response returned no results")
    first = results[0]
    if not isinstance(first, dict):
        raise RuntimeError("parse result is not an object")
    parse_id = str(first.get("parse_id") or "")
    if not parse_id:
        raise RuntimeError("parse result missing parse_id")

    structured_url = str(((first.get("artifact_urls") or {}).get("structured_url")) or "")
    if not structured_url:
        raise RuntimeError("parse result missing structured_url")
    structured = _fetch_json(_resolve_url(base_url, structured_url))

    document = structured.get("document")
    chunks = structured.get("chunks")
    if not isinstance(document, dict) or str(document.get("file_type") or "").lower() != "txt":
        raise RuntimeError("structured document missing txt file_type")
    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError("structured chunks are missing")
    if not all((chunk.get("metadata") or {}).get("chunk_strategy") == "asset_aware_v1" for chunk in chunks):
        raise RuntimeError("structured chunks did not use asset_aware_v1 strategy")

    artifact_urls = first.get("artifact_urls") if isinstance(first.get("artifact_urls"), dict) else {}
    chunks_url = str(artifact_urls.get("chunks_url") or "")
    ingest_url = str(artifact_urls.get("ingest_url") or "")
    if not chunks_url or not ingest_url:
        raise RuntimeError("parse result missing chunks_url or ingest_url")
    chunk_records = _parse_jsonl(_fetch_text(_resolve_url(base_url, chunks_url)))
    ingest_records = _parse_jsonl(_fetch_text(_resolve_url(base_url, ingest_url)))
    if not chunk_records:
        raise RuntimeError("chunks artifact returned no records")
    if not ingest_records:
        raise RuntimeError("ingest artifact returned no records")
    first_chunk_metadata = chunk_records[0].get("metadata") if isinstance(chunk_records[0].get("metadata"), dict) else {}
    first_ingest_metadata = ingest_records[0].get("metadata") if isinstance(ingest_records[0].get("metadata"), dict) else {}
    if first_chunk_metadata.get("schema_version") != "2026-06-08.chunk.v1":
        raise RuntimeError("chunk export record missing expected schema_version")
    if first_chunk_metadata.get("chunk_strategy") != "asset_aware_v1":
        raise RuntimeError("chunk export record missing asset_aware_v1 strategy")
    if first_ingest_metadata.get("schema_version") != "2026-06-08.ingest.v1":
        raise RuntimeError("ingest export record missing expected schema_version")
    if first_ingest_metadata.get("chunk_schema_version") != "2026-06-08.chunk.v1":
        raise RuntimeError("ingest export record missing expected chunk_schema_version")

    print(
        json.dumps(
            {
                "parse_id": parse_id,
                "chunk_count": len(chunks),
                "chunk_record_count": len(chunk_records),
                "ingest_record_count": len(ingest_records),
                "structured_url": structured_url,
                "build_source": build_info.get("build_source"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

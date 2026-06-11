from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
from pydantic import BaseModel, ConfigDict, Field

from common import logger
from common.parse_artifacts import ParseArtifact, ParseManifest, ChunkExportRecord, IngestExportRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestPublishResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    sink_type: str
    status: str
    record_count: int
    destination: str | None = None
    published_at: str | None = Field(default_factory=_now_iso)
    response_code: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestPublishError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        sink_type: str,
        destination: str | None = None,
        response_code: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.sink_type = sink_type
        self.destination = destination
        self.response_code = response_code
        self.metadata = metadata or {}


class IngestPublishAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(default_factory=lambda: uuid4().hex)
    requested_by: str = "parse"
    attempted_at: str = Field(default_factory=_now_iso)
    sink_type: str
    status: str
    record_count: int
    destination: str | None = None
    response_code: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestPublishState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    sink_type: str
    status: str
    destination: str | None = None
    response_code: int | None = None
    attempt_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_record_count: int = 0
    last_attempt_at: str | None = None
    last_success_at: str | None = None
    last_error: str | None = None
    next_retry_at: str | None = None
    retryable: bool = False
    dead_lettered: bool = False
    last_attempt: IngestPublishAttempt | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestPublisher(ABC):
    sink_type = "none"

    @abstractmethod
    def publish(
        self,
        manifest: ParseManifest,
        records: list[IngestExportRecord],
        *,
        artifact: ParseArtifact | None = None,
        chunk_records: list[ChunkExportRecord] | None = None,
    ) -> IngestPublishResult:
        raise NotImplementedError


class NoopIngestPublisher(IngestPublisher):
    sink_type = "none"

    def publish(
        self,
        manifest: ParseManifest,
        records: list[IngestExportRecord],
        *,
        artifact: ParseArtifact | None = None,
        chunk_records: list[ChunkExportRecord] | None = None,
    ) -> IngestPublishResult:
        return IngestPublishResult(
            enabled=False,
            sink_type=self.sink_type,
            status="disabled",
            record_count=len(records),
        )


class FileIngestPublisher(IngestPublisher):
    sink_type = "file"

    def __init__(self, target_path: str | Path):
        self.target_path = Path(target_path)
        self.target_path.parent.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        manifest: ParseManifest,
        records: list[IngestExportRecord],
        *,
        artifact: ParseArtifact | None = None,
        chunk_records: list[ChunkExportRecord] | None = None,
    ) -> IngestPublishResult:
        with self.target_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {
                            "manifest": manifest.model_dump(mode="json"),
                            "record": record.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return IngestPublishResult(
            enabled=True,
            sink_type=self.sink_type,
            status="published",
            record_count=len(records),
            destination=str(self.target_path),
        )


class HttpIngestPublisher(IngestPublisher):
    sink_type = "http"

    def __init__(
        self,
        url: str,
        *,
        request_timeout: int = 30,
        auth_header: str | None = None,
        auth_token: str | None = None,
        extra_headers: dict[str, str] | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 8.0,
        retry_status_codes: set[int] | None = None,
    ):
        if not url:
            raise ValueError("DEEPDOC_INGEST_HTTP_URL is required when DEEPDOC_INGEST_PUBLISHER=http")
        self.url = url
        self.request_timeout = max(1, int(request_timeout))
        self.auth_header = auth_header
        self.auth_token = auth_token
        self.extra_headers = extra_headers or {}
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.max_backoff_seconds = max(self.backoff_seconds, float(max_backoff_seconds))
        self.retry_status_codes = set(retry_status_codes or {429, 500, 502, 503, 504})

    def _is_retryable(self, response_code: int | None, exc: Exception) -> bool:
        if response_code in self.retry_status_codes:
            return True
        return isinstance(exc, (requests.Timeout, requests.ConnectionError))

    def publish(
        self,
        manifest: ParseManifest,
        records: list[IngestExportRecord],
        *,
        artifact: ParseArtifact | None = None,
        chunk_records: list[ChunkExportRecord] | None = None,
    ) -> IngestPublishResult:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.auth_header and self.auth_token:
            headers[self.auth_header] = self.auth_token
        delay_seconds = self.backoff_seconds
        last_error: Exception | None = None
        last_response_code: int | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = requests.post(
                    self.url,
                    json={
                        "manifest": manifest.model_dump(mode="json"),
                        "records": [record.model_dump(mode="json") for record in records],
                    },
                    headers=headers,
                    timeout=self.request_timeout,
                )
                last_response_code = response.status_code
                retryable_status = response.status_code in self.retry_status_codes
                if retryable_status and attempt < self.max_attempts:
                    time.sleep(delay_seconds)
                    delay_seconds = min(max(delay_seconds * 2, self.backoff_seconds), self.max_backoff_seconds)
                    continue
                response.raise_for_status()
                return IngestPublishResult(
                    enabled=True,
                    sink_type=self.sink_type,
                    status="published",
                    record_count=len(records),
                    destination=self.url,
                    response_code=response.status_code,
                    metadata={"http_attempts": attempt},
                )
            except requests.RequestException as exc:
                response_code = getattr(getattr(exc, "response", None), "status_code", None) or last_response_code
                retryable = self._is_retryable(response_code, exc)
                last_error = exc
                if retryable and attempt < self.max_attempts:
                    time.sleep(delay_seconds)
                    delay_seconds = min(max(delay_seconds * 2, self.backoff_seconds), self.max_backoff_seconds)
                    continue
                raise IngestPublishError(
                    str(exc),
                    sink_type=self.sink_type,
                    destination=self.url,
                    response_code=response_code,
                    metadata={
                        "error": str(exc),
                        "http_attempts": attempt,
                        "retryable": retryable,
                    },
                ) from exc
        raise IngestPublishError(
            str(last_error or "HTTP ingest publish failed"),
            sink_type=self.sink_type,
            destination=self.url,
            response_code=last_response_code,
            metadata={
                "error": str(last_error or "HTTP ingest publish failed"),
                "http_attempts": self.max_attempts,
                "retryable": False,
            },
        )


def _normalize_publish_result(result: IngestPublishResult | dict[str, Any]) -> IngestPublishResult:
    if isinstance(result, IngestPublishResult):
        return result
    return IngestPublishResult.model_validate(result)


def parse_ingest_publish_state(payload: Any) -> IngestPublishState | None:
    if not isinstance(payload, dict):
        return None
    try:
        return IngestPublishState.model_validate(payload)
    except Exception:
        try:
            legacy_result = IngestPublishResult.model_validate(payload)
        except Exception:
            return None
        state, _ = build_ingest_publish_state(previous_state=None, result=legacy_result, requested_by="legacy")
        return state


def build_ingest_publish_state(
    *,
    previous_state: dict[str, Any] | IngestPublishState | None,
    result: IngestPublishResult | dict[str, Any],
    requested_by: str,
    retry_base_delay_seconds: float = 60.0,
    retry_max_delay_seconds: float = 3600.0,
    max_failure_count: int = 5,
) -> tuple[IngestPublishState, IngestPublishAttempt]:
    normalized_result = _normalize_publish_result(result)
    state = previous_state if isinstance(previous_state, IngestPublishState) else parse_ingest_publish_state(previous_state)
    attempt = IngestPublishAttempt(
        requested_by=requested_by,
        attempted_at=normalized_result.published_at or _now_iso(),
        sink_type=normalized_result.sink_type,
        status=normalized_result.status,
        record_count=normalized_result.record_count,
        destination=normalized_result.destination,
        response_code=normalized_result.response_code,
        metadata=dict(normalized_result.metadata or {}),
    )
    success_increment = 1 if normalized_result.status == "published" else 0
    failure_increment = 1 if normalized_result.status == "failed" else 0
    failure_count = (state.failure_count if state else 0) + failure_increment
    retry_delay_seconds = 0.0
    dead_lettered = False
    retryable = False
    next_retry_at = None
    if normalized_result.status == "failed":
        dead_lettered = max_failure_count > 0 and failure_count >= max_failure_count
        retryable = bool(normalized_result.enabled) and not dead_lettered
        if retryable:
            base_delay = max(1.0, float(retry_base_delay_seconds))
            max_delay = max(base_delay, float(retry_max_delay_seconds))
            retry_delay_seconds = min(base_delay * (2 ** max(failure_count - 1, 0)), max_delay)
            next_retry_at = (datetime.fromisoformat(attempt.attempted_at) + timedelta(seconds=retry_delay_seconds)).isoformat()
    state_metadata = dict(getattr(state, "metadata", {}) or {})
    state_metadata.update(normalized_result.metadata or {})
    if normalized_result.status == "published":
        state_metadata.pop("error", None)
        state_metadata.pop("retry_delay_seconds", None)
    if retry_delay_seconds > 0:
        state_metadata["retry_delay_seconds"] = retry_delay_seconds
    next_state = IngestPublishState(
        enabled=normalized_result.enabled,
        sink_type=normalized_result.sink_type,
        status=normalized_result.status,
        destination=normalized_result.destination,
        response_code=normalized_result.response_code,
        attempt_count=(state.attempt_count if state else 0) + 1,
        success_count=(state.success_count if state else 0) + success_increment,
        failure_count=failure_count,
        last_record_count=normalized_result.record_count,
        last_attempt_at=attempt.attempted_at,
        last_success_at=(
            normalized_result.published_at
            if normalized_result.status == "published"
            else (state.last_success_at if state else None)
        ),
        last_error=(
            str((normalized_result.metadata or {}).get("error") or "")
            if normalized_result.status == "failed"
            else None
        )
        or (state.last_error if state and normalized_result.status != "published" else None),
        next_retry_at=next_retry_at,
        retryable=retryable,
        dead_lettered=dead_lettered,
        last_attempt=attempt,
        metadata=state_metadata,
    )
    return next_state, attempt


def create_ingest_publisher() -> IngestPublisher:
    sink_type = (os.environ.get("DEEPDOC_INGEST_PUBLISHER", "none") or "none").strip().lower()
    if sink_type in {"", "none", "disabled"}:
        return NoopIngestPublisher()
    if sink_type == "file":
        target_path = os.environ.get("DEEPDOC_INGEST_FILE_PATH") or os.path.join(
            os.getcwd(), "resources", "artifacts", "published-ingest.jsonl"
        )
        return FileIngestPublisher(target_path)
    if sink_type == "http":
        auth_header = (os.environ.get("DEEPDOC_INGEST_HTTP_AUTH_HEADER") or "").strip() or None
        auth_token = (os.environ.get("DEEPDOC_INGEST_HTTP_AUTH_TOKEN") or "").strip() or None
        extra_headers: dict[str, str] = {}
        raw_headers = (os.environ.get("DEEPDOC_INGEST_HTTP_EXTRA_HEADERS") or "").strip()
        if raw_headers:
            try:
                extra_headers = {
                    str(k): str(v)
                    for k, v in json.loads(raw_headers).items()
                }
            except Exception:
                logger.exception("Invalid DEEPDOC_INGEST_HTTP_EXTRA_HEADERS, expected JSON object")
        return HttpIngestPublisher(
            os.environ.get("DEEPDOC_INGEST_HTTP_URL", "").strip(),
            request_timeout=int(os.environ.get("DEEPDOC_INGEST_HTTP_TIMEOUT", "30")),
            auth_header=auth_header,
            auth_token=auth_token,
            extra_headers=extra_headers,
            max_attempts=int(os.environ.get("DEEPDOC_INGEST_HTTP_MAX_ATTEMPTS", "3")),
            backoff_seconds=float(os.environ.get("DEEPDOC_INGEST_HTTP_BACKOFF_SECONDS", "1")),
            max_backoff_seconds=float(os.environ.get("DEEPDOC_INGEST_HTTP_MAX_BACKOFF_SECONDS", "8")),
            retry_status_codes={
                int(code.strip())
                for code in (os.environ.get("DEEPDOC_INGEST_HTTP_RETRY_STATUS_CODES", "429,500,502,503,504") or "").split(",")
                if code.strip()
            },
        )
    if sink_type == "postgres":
        from common.ingest_postgres import PostgresIngestPublisher, load_postgres_ingest_config_from_env

        return PostgresIngestPublisher(load_postgres_ingest_config_from_env())
    raise RuntimeError(f"Unsupported DEEPDOC_INGEST_PUBLISHER: {sink_type}")

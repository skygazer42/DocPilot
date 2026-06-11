from __future__ import annotations

import json
import os
import re
import shutil
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from common import logger, setting


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


_REDACTABLE_URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"()]+")
_REDACTABLE_HOST_RE = re.compile(r"(host=')([^']+)(')")
_REDACTABLE_FS_PATH_RE = re.compile(r"(?:(?<=\s)|^)(/(?:app|data|home|root|run|srv|tmp|usr|var|work)[^\s'\",)]*)")


def _sanitize_task_sensitive_text(text: str) -> str:
    sanitized = _REDACTABLE_URL_RE.sub("<redacted>", str(text))
    sanitized = _REDACTABLE_HOST_RE.sub(r"\1<redacted>\3", sanitized)
    sanitized = _REDACTABLE_FS_PATH_RE.sub("<redacted>", sanitized)
    return sanitized


def _sanitize_task_payload_value(value: Any, *, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            sanitized[str(key)] = _sanitize_task_payload_value(item, path=path + (str(key),))
        return sanitized
    if isinstance(value, list):
        return [_sanitize_task_payload_value(item, path=path + (str(index),)) for index, item in enumerate(value)]

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
        return _sanitize_task_sensitive_text(value)
    return value


def _safe_slug(value: str, default: str = "document") -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value or "").strip())
    text = text.strip("-.")
    return text or default


class AsyncTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    file_type: str
    size_bytes: int
    sha256: str
    source_path: str


class AsyncTaskEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    created_at: str = Field(default_factory=_now_iso)
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AsyncTaskCallbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    event_types: list[str] = Field(default_factory=lambda: ["terminal"])
    include_result: bool = True
    timeout_seconds: int = 10
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    max_backoff_seconds: float = 10.0
    secret: str | None = None


class AsyncTaskCallbackState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    status: Literal["pending", "delivering", "delivered", "failed", "dead_lettered"] = "pending"
    delivery_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_delivery_id: str | None = None
    last_attempt_at: str | None = None
    last_success_at: str | None = None
    last_error: str | None = None
    last_response_status: int | None = None
    next_retry_at: str | None = None


class AsyncTaskCallbackEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    callback_event_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    delivery_id: str
    created_at: str = Field(default_factory=_now_iso)
    event_type: str
    attempt_no: int
    status: Literal["pending", "succeeded", "failed"]
    request_url: str
    response_status: int | None = None
    duration_ms: int | None = None
    error: str | None = None
    response_body_snippet: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AsyncTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    queue_name: str
    status: Literal["queued", "running", "succeeded", "failed", "cancel_requested", "cancelled"]
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    heartbeat_at: str | None = None
    tenant_id: str | None = None
    requested_by: str = "api"
    auth_subject: str | None = None
    parser_engine: str
    parse_options: dict[str, Any] = Field(default_factory=dict)
    input_files: list[AsyncTaskInput] = Field(default_factory=list)
    result_available: bool = False
    result_summary: dict[str, Any] = Field(default_factory=dict)
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    callback: AsyncTaskCallbackConfig | None = None
    callback_state: AsyncTaskCallbackState | None = None


class AsyncTaskPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_dir: str
    source_dir: str
    task_path: str
    result_path: str
    events_path: str
    callback_events_path: str


class LocalUploadedFile:
    def __init__(self, source_path: str | Path, filename: str):
        self.source_path = Path(source_path)
        self.filename = filename
        self._handle = None

    def _open(self):
        if self._handle is None or self._handle.closed:
            self._handle = self.source_path.open("rb")
        return self._handle

    def seek(self, offset: int, whence: int = 0):
        return self._open().seek(offset, whence)

    def tell(self) -> int:
        return self._open().tell()

    def read(self, size: int = -1) -> bytes:
        return self._open().read(size)

    def save(self, dst: str | os.PathLike[str]) -> None:
        shutil.copyfile(self.source_path, dst)

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()


class AsyncTaskStore:
    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = Path(root_dir or setting.TASKS_DIR)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def get_paths(self, task_id: str) -> AsyncTaskPaths:
        task_dir = self.root_dir / task_id
        source_dir = task_dir / "source"
        task_dir.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir(parents=True, exist_ok=True)
        return AsyncTaskPaths(
            task_id=task_id,
            task_dir=str(task_dir),
            source_dir=str(source_dir),
            task_path=str(task_dir / "task.json"),
            result_path=str(task_dir / "result.json"),
            events_path=str(task_dir / "events.jsonl"),
            callback_events_path=str(task_dir / "callback-events.jsonl"),
        )

    def create_task(self, task: AsyncTask) -> AsyncTaskPaths:
        paths = self.get_paths(task.task_id)
        self.write_task(task)
        self.append_event(
            task.task_id,
            "queued",
            {
                "queue_name": task.queue_name,
                "file_count": len(task.input_files),
                "parser_engine": task.parser_engine,
            },
        )
        return paths

    def write_task(self, task: AsyncTask) -> None:
        paths = self.get_paths(task.task_id)
        Path(paths.task_path).write_text(
            json.dumps(task.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_task(self, task_id: str) -> AsyncTask:
        paths = self.get_paths(task_id)
        return AsyncTask.model_validate(json.loads(Path(paths.task_path).read_text(encoding="utf-8")))

    def save_result(self, task_id: str, payload: dict[str, Any]) -> None:
        paths = self.get_paths(task_id)
        Path(paths.result_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_result(self, task_id: str) -> dict[str, Any] | None:
        paths = self.get_paths(task_id)
        result_path = Path(paths.result_path)
        if not result_path.exists():
            return None
        return json.loads(result_path.read_text(encoding="utf-8"))

    def append_event(self, task_id: str, event_type: str, payload: dict[str, Any] | None = None) -> AsyncTaskEvent:
        paths = self.get_paths(task_id)
        event = AsyncTaskEvent(task_id=task_id, event_type=event_type, payload=payload or {})
        with Path(paths.events_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return event

    def read_events(self, task_id: str) -> list[AsyncTaskEvent]:
        paths = self.get_paths(task_id)
        event_path = Path(paths.events_path)
        if not event_path.exists():
            return []
        events: list[AsyncTaskEvent] = []
        with event_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(AsyncTaskEvent.model_validate(json.loads(line)))
                except Exception:
                    logger.exception("Failed to parse async task event task_id=%s", task_id)
        return events

    def append_callback_event(
        self,
        task_id: str,
        *,
        delivery_id: str,
        event_type: str,
        attempt_no: int,
        status: Literal["pending", "succeeded", "failed"],
        request_url: str,
        response_status: int | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
        response_body_snippet: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncTaskCallbackEvent:
        paths = self.get_paths(task_id)
        event = AsyncTaskCallbackEvent(
            task_id=task_id,
            delivery_id=delivery_id,
            event_type=event_type,
            attempt_no=max(1, int(attempt_no)),
            status=status,
            request_url=request_url,
            response_status=response_status,
            duration_ms=duration_ms,
            error=error,
            response_body_snippet=response_body_snippet,
            metadata=metadata or {},
        )
        with Path(paths.callback_events_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return event

    def read_callback_events(self, task_id: str) -> list[AsyncTaskCallbackEvent]:
        paths = self.get_paths(task_id)
        event_path = Path(paths.callback_events_path)
        if not event_path.exists():
            return []
        events: list[AsyncTaskCallbackEvent] = []
        with event_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(AsyncTaskCallbackEvent.model_validate(json.loads(line)))
                except Exception:
                    logger.exception("Failed to parse async task callback event task_id=%s", task_id)
        return events

    def list_tasks(self, *, limit: int = 20, tenant_id: str | None = None) -> list[AsyncTask]:
        normalized_tenant = str(tenant_id or "").strip() or None
        tasks: list[AsyncTask] = []
        for task_path in sorted(self.root_dir.glob("*/task.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                task = AsyncTask.model_validate(json.loads(task_path.read_text(encoding="utf-8")))
            except Exception:
                logger.exception("Failed to parse async task record from %s", task_path)
                continue
            if normalized_tenant and (str(task.tenant_id or "").strip() or None) != normalized_tenant:
                continue
            tasks.append(task)
            if len(tasks) >= max(1, int(limit)):
                break
        return tasks

    def delete_task(self, task_id: str) -> bool:
        task_dir = self.root_dir / str(task_id or "").strip()
        if not task_dir.exists():
            return False
        shutil.rmtree(task_dir, ignore_errors=False)
        return True

    def cleanup_tasks(
        self,
        *,
        limit: int = 1000,
        tenant_id: str | None = None,
        older_than_days: int | None = None,
        keep_latest: int | None = None,
        statuses: list[str] | None = None,
        include_active: bool = False,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 5000))
        keep_latest = max(0, int(keep_latest or 0))
        statuses_normalized = {
            str(item).strip().lower()
            for item in (statuses or [])
            if str(item).strip()
        }
        active_statuses = {"queued", "running", "cancel_requested"}
        cutoff = None
        if older_than_days is not None:
            cutoff = _now() - timedelta(days=max(0, int(older_than_days)))

        tasks = self.list_tasks(limit=limit, tenant_id=tenant_id)
        keep_ids = {task.task_id for task in tasks[:keep_latest]} if keep_latest > 0 else set()
        candidates: list[dict[str, Any]] = []

        for task in tasks:
            if task.task_id in keep_ids:
                continue
            if not include_active and task.status in active_statuses:
                continue
            if statuses_normalized and task.status not in statuses_normalized:
                continue
            reference_time = (
                _parse_iso_datetime(task.finished_at)
                or _parse_iso_datetime(task.updated_at)
                or _parse_iso_datetime(task.created_at)
            )
            if cutoff is not None and (reference_time is None or reference_time > cutoff):
                continue
            candidates.append(
                {
                    "task_id": task.task_id,
                    "tenant_id": task.tenant_id,
                    "status": task.status,
                    "created_at": task.created_at,
                    "updated_at": task.updated_at,
                    "finished_at": task.finished_at,
                    "result_available": task.result_available,
                }
            )

        deleted = 0
        if not dry_run:
            for item in candidates:
                if self.delete_task(str(item["task_id"])):
                    deleted += 1

        return {
            "scanned": len(tasks),
            "matched": len(candidates),
            "deleted": deleted,
            "dry_run": dry_run,
            "limit": limit,
            "tenant_id": str(tenant_id or "").strip() or None,
            "older_than_days": older_than_days,
            "keep_latest": keep_latest,
            "statuses": sorted(statuses_normalized),
            "include_active": include_active,
            "tasks": candidates,
        }

    def clone_inputs_for_retry(self, source_task: AsyncTask, *, target_task_id: str) -> list[AsyncTaskInput]:
        paths = self.get_paths(target_task_id)
        target_source_dir = Path(paths.source_dir)
        cloned_inputs: list[AsyncTaskInput] = []
        for index, input_file in enumerate(source_task.input_files):
            source_path = Path(input_file.source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"task source file not found: {source_path}")
            safe_name = f"{index:02d}-{uuid4().hex[:8]}-{Path(input_file.filename).name}"
            target_path = target_source_dir / safe_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            cloned_inputs.append(
                input_file.model_copy(
                    update={
                        "source_path": str(target_path),
                    }
                )
            )
        return cloned_inputs

    def check_health(self) -> dict[str, Any]:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "ok", "backend": "local", "root_dir": str(self.root_dir)}


class AsyncTaskBroker(ABC):
    backend_name = "unknown"

    @abstractmethod
    def enqueue(self, task_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def reserve(self, *, timeout_seconds: int, task_store: AsyncTaskStore | None = None) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def ack(self, task_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def check_health(self) -> dict[str, Any]:
        raise NotImplementedError


class RedisAsyncTaskBroker(AsyncTaskBroker):
    backend_name = "redis"

    def __init__(
        self,
        redis_url: str,
        *,
        queue_name: str,
        processing_queue_name: str,
        visibility_timeout_seconds: int,
    ):
        if not redis_url:
            raise ValueError("DEEPDOC_ASYNC_REDIS_URL is required when async broker is enabled")
        try:
            import redis
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("redis package is required for async task broker") from exc
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.processing_queue_name = processing_queue_name
        self.visibility_timeout_seconds = max(30, int(visibility_timeout_seconds))
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._requeue_lock = threading.Lock()

    def enqueue(self, task_id: str) -> None:
        self._client.lpush(self.queue_name, task_id)

    def reserve(self, *, timeout_seconds: int, task_store: AsyncTaskStore | None = None) -> str | None:
        self._requeue_stale(task_store)
        task_id = self._client.brpoplpush(self.queue_name, self.processing_queue_name, timeout=max(1, int(timeout_seconds)))
        if not task_id:
            return None
        return str(task_id)

    def ack(self, task_id: str) -> None:
        self._client.lrem(self.processing_queue_name, 0, task_id)

    def _requeue_stale(self, task_store: AsyncTaskStore | None) -> None:
        if task_store is None or not self._requeue_lock.acquire(blocking=False):
            return
        try:
            now = _now()
            for task_id in self._client.lrange(self.processing_queue_name, 0, -1):
                try:
                    task = task_store.load_task(task_id)
                except Exception:
                    logger.exception("Failed to load async task while checking stale processing task_id=%s", task_id)
                    continue
                if task.status not in {"running", "queued", "cancel_requested"}:
                    self.ack(task_id)
                    continue
                heartbeat_raw = task.heartbeat_at or task.updated_at or task.created_at
                try:
                    heartbeat_at = datetime.fromisoformat(heartbeat_raw)
                except Exception:
                    heartbeat_at = now
                if now - heartbeat_at <= timedelta(seconds=self.visibility_timeout_seconds):
                    continue
                self._client.lrem(self.processing_queue_name, 0, task_id)
                self._client.lpush(self.queue_name, task_id)
                requeued = task.model_copy(
                    update={
                        "status": "queued",
                        "updated_at": _now_iso(),
                        "heartbeat_at": None,
                        "last_error": "task requeued after stale worker visibility timeout",
                    }
                )
                task_store.write_task(requeued)
                task_store.append_event(
                    task_id,
                    "requeued",
                    {"reason": "visibility_timeout", "timeout_seconds": self.visibility_timeout_seconds},
                )
        finally:
            self._requeue_lock.release()

    def queue_lengths(self) -> dict[str, int]:
        return {
            "queued": int(self._client.llen(self.queue_name)),
            "processing": int(self._client.llen(self.processing_queue_name)),
        }

    def check_health(self) -> dict[str, Any]:
        self._client.ping()
        return {
            "status": "ok",
            "backend": self.backend_name,
            "redis_url": self.redis_url,
            "queue_name": self.queue_name,
            "processing_queue_name": self.processing_queue_name,
            "visibility_timeout_seconds": self.visibility_timeout_seconds,
            "queues": self.queue_lengths(),
        }


class NoopAsyncTaskBroker(AsyncTaskBroker):
    backend_name = "none"

    def enqueue(self, task_id: str) -> None:
        raise RuntimeError("Async task broker is disabled")

    def reserve(self, *, timeout_seconds: int, task_store: AsyncTaskStore | None = None) -> str | None:
        return None

    def ack(self, task_id: str) -> None:
        return None

    def check_health(self) -> dict[str, Any]:
        return {"status": "disabled", "backend": self.backend_name}


def create_async_task_store() -> AsyncTaskStore:
    return AsyncTaskStore()


def create_async_task_broker() -> AsyncTaskBroker:
    enabled = _parse_bool(os.environ.get("DEEPDOC_ASYNC_ENABLED"), default=False)
    if not enabled:
        return NoopAsyncTaskBroker()
    backend = (os.environ.get("DEEPDOC_ASYNC_BROKER") or "redis").strip().lower()
    if backend != "redis":
        raise RuntimeError(f"Unsupported DEEPDOC_ASYNC_BROKER: {backend}")
    return RedisAsyncTaskBroker(
        os.environ.get("DEEPDOC_ASYNC_REDIS_URL", ""),
        queue_name=os.environ.get("DEEPDOC_ASYNC_QUEUE_NAME", "deepdoc:async:parse"),
        processing_queue_name=os.environ.get("DEEPDOC_ASYNC_PROCESSING_QUEUE_NAME", "deepdoc:async:parse:processing"),
        visibility_timeout_seconds=int(os.environ.get("DEEPDOC_ASYNC_VISIBILITY_TIMEOUT_SECONDS", "600")),
    )


def build_async_task(
    *,
    queue_name: str,
    parser_engine: str,
    parse_options: dict[str, Any],
    input_files: list[AsyncTaskInput],
    tenant_id: str | None,
    requested_by: str,
    auth_subject: str | None,
    callback: AsyncTaskCallbackConfig | None = None,
) -> AsyncTask:
    now = _now_iso()
    task_id = uuid4().hex
    return AsyncTask(
        task_id=task_id,
        queue_name=queue_name,
        status="queued",
        created_at=now,
        updated_at=now,
        tenant_id=str(tenant_id or "").strip() or None,
        requested_by=requested_by,
        auth_subject=str(auth_subject or "").strip() or None,
        parser_engine=parser_engine,
        parse_options=parse_options,
        input_files=input_files,
        metadata={"file_count": len(input_files)},
        callback=callback,
        callback_state=(
            AsyncTaskCallbackState(
                enabled=True,
                status="pending",
            )
            if callback is not None
            else None
        ),
    )


def build_async_retry_task(
    source_task: AsyncTask,
    *,
    queue_name: str,
    requested_by: str,
    auth_subject: str | None,
    callback: AsyncTaskCallbackConfig | None,
) -> AsyncTask:
    retry_group_id = str((source_task.metadata or {}).get("retry_group_id") or (source_task.metadata or {}).get("original_task_id") or source_task.task_id).strip() or source_task.task_id
    original_task_id = str((source_task.metadata or {}).get("original_task_id") or source_task.task_id).strip() or source_task.task_id
    previous_retry_attempt = int((source_task.metadata or {}).get("retry_attempt") or 0)
    task = build_async_task(
        queue_name=queue_name,
        parser_engine=source_task.parser_engine,
        parse_options=dict(source_task.parse_options or {}),
        input_files=[],
        tenant_id=source_task.tenant_id,
        requested_by=requested_by,
        auth_subject=str(auth_subject or "").strip() or None,
        callback=callback,
    )
    retry_metadata = {
        "retry_group_id": retry_group_id,
        "original_task_id": original_task_id,
        "retried_from_task_id": source_task.task_id,
        "retry_attempt": previous_retry_attempt + 1,
        "retry_requested_by": requested_by,
        "retry_requested_at": _now_iso(),
        "source_task_status": source_task.status,
    }
    return task.model_copy(update={"metadata": {**task.metadata, **retry_metadata}})


def task_access_payload(task: AsyncTask, *, result: dict[str, Any] | None = None, include_result: bool = True) -> dict[str, Any]:
    payload = task.model_dump(mode="json")
    payload["status_url"] = f"/api/v1/tasks/{task.task_id}"
    payload["events_url"] = f"/api/v1/tasks/{task.task_id}/events"
    payload["stream_url"] = f"/api/v1/tasks/{task.task_id}/stream"
    payload["cancel_url"] = f"/api/v1/tasks/{task.task_id}/cancel"
    payload["retry_url"] = f"/api/v1/tasks/{task.task_id}/retry"
    if task.callback is not None:
        callback_payload = task.callback.model_dump(mode="json", exclude={"secret"})
        callback_payload["secret_configured"] = bool(task.callback.secret)
        payload["callback"] = callback_payload
        payload["callback_events_url"] = f"/api/v1/tasks/{task.task_id}/callback-events"
        payload["callback_retry_url"] = f"/api/v1/tasks/{task.task_id}/callback/retry"
    if include_result and result is not None:
        payload["result"] = result
    return _sanitize_task_payload_value(payload)

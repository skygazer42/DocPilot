from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

import requests

from common.async_tasks import AsyncTask, AsyncTaskCallbackState, AsyncTaskStore, task_access_payload
from common.metrics import observe_async_task_callback_delivery

TERMINAL_TASK_STATUSES = {"succeeded", "failed", "cancelled"}
CALLBACK_STATUS_TO_EVENT = {
    "succeeded": "task.succeeded",
    "failed": "task.failed",
    "cancelled": "task.cancelled",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _callback_event_enabled(task: AsyncTask) -> str | None:
    if task.callback is None:
        return None
    status = str(task.status or "").strip().lower()
    if status not in TERMINAL_TASK_STATUSES:
        return None
    event_type = CALLBACK_STATUS_TO_EVENT.get(status)
    configured = {str(item).strip().lower() for item in (task.callback.event_types or []) if str(item).strip()}
    if not configured or "terminal" in configured or str(event_type or "").lower() in configured:
        return event_type
    return None


def _response_snippet(response: requests.Response | None, limit: int = 512) -> str | None:
    if response is None:
        return None
    try:
        content = response.text
    except Exception:
        return None
    if not content:
        return None
    return content[: max(32, int(limit))]


def _build_delivery_headers(
    *,
    body: bytes,
    event_type: str,
    delivery_id: str,
    secret: str | None,
) -> dict[str, str]:
    timestamp = str(int(time.time()))
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "deepdoc-task-callback/1.0",
        "X-DeepDoc-Event-Type": event_type,
        "X-DeepDoc-Delivery-ID": delivery_id,
        "X-DeepDoc-Timestamp": timestamp,
    }
    if secret:
        signing_input = timestamp.encode("utf-8") + b"." + body
        signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).hexdigest()
        headers["X-DeepDoc-Signature-256"] = f"sha256={signature}"
    return headers


def _build_callback_payload(
    *,
    task: AsyncTask,
    result: dict[str, Any] | None,
    event_type: str,
    delivery_id: str,
) -> dict[str, Any]:
    task_payload = task_access_payload(task, result=result, include_result=bool(task.callback and task.callback.include_result))
    return {
        "delivery_id": delivery_id,
        "event_type": event_type,
        "sent_at": _now_iso(),
        "task_id": task.task_id,
        "tenant_id": task.tenant_id,
        "status": task.status,
        "task": task_payload,
        "result_summary": task.result_summary,
    }


def _next_backoff_seconds(base_seconds: float, attempt_no: int, max_backoff_seconds: float) -> float:
    exponent = max(0, int(attempt_no) - 1)
    backoff = float(base_seconds) * float(2**exponent)
    return max(0.0, min(float(max_backoff_seconds), backoff))


def deliver_async_task_callback(
    *,
    task_store: AsyncTaskStore,
    task: AsyncTask,
    result: dict[str, Any] | None,
    force: bool = False,
    requested_by: str = "worker",
    audit_hook: Callable[..., None] | None = None,
) -> dict[str, Any] | None:
    callback = task.callback
    event_type = _callback_event_enabled(task)
    if callback is None or event_type is None:
        return None

    current_state = task.callback_state or AsyncTaskCallbackState(enabled=True, status="pending")
    if current_state.status == "delivered" and not force:
        return {
            "configured": True,
            "skipped": True,
            "reason": "already_delivered",
            "status": "delivered",
            "delivery_count": current_state.delivery_count,
            "success_count": current_state.success_count,
            "failure_count": current_state.failure_count,
        }

    delivery_id = uuid4().hex
    initial_state = current_state.model_copy(
        update={
            "status": "delivering",
            "last_delivery_id": delivery_id,
            "last_error": None,
            "next_retry_at": None,
        }
    )
    task_for_delivery = task.model_copy(update={"callback_state": initial_state})
    task_store.write_task(task_for_delivery)

    max_attempts = max(1, int(callback.max_attempts))
    timeout_seconds = max(1, int(callback.timeout_seconds))
    backoff_seconds = max(0.0, float(callback.backoff_seconds))
    max_backoff_seconds = max(0.0, float(callback.max_backoff_seconds))

    latest_state = initial_state
    for attempt_no in range(1, max_attempts + 1):
        payload = _build_callback_payload(
            task=task_for_delivery,
            result=result,
            event_type=event_type,
            delivery_id=delivery_id,
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = _build_delivery_headers(
            body=body,
            event_type=event_type,
            delivery_id=delivery_id,
            secret=callback.secret,
        )
        response: requests.Response | None = None
        started_at = time.monotonic()
        error_message: str | None = None
        response_status: int | None = None
        response_snippet: str | None = None
        status = "failed"
        try:
            response = requests.post(
                callback.url,
                data=body,
                headers=headers,
                timeout=timeout_seconds,
            )
            response_status = int(response.status_code)
            response_snippet = _response_snippet(response)
            if 200 <= response.status_code < 300:
                status = "succeeded"
            else:
                error_message = f"callback returned HTTP {response.status_code}"
        except Exception as exc:
            error_message = str(exc)
        duration_ms = max(0, int((time.monotonic() - started_at) * 1000))
        task_store.append_callback_event(
            task.task_id,
            delivery_id=delivery_id,
            event_type=event_type,
            attempt_no=attempt_no,
            status=status,
            request_url=callback.url,
            response_status=response_status,
            duration_ms=duration_ms,
            error=error_message,
            response_body_snippet=response_snippet,
            metadata={"requested_by": requested_by},
        )

        delivery_count = int(latest_state.delivery_count) + 1
        if status == "succeeded":
            observe_async_task_callback_delivery(
                event_type=event_type,
                status="delivered",
                duration_ms=duration_ms,
            )
            latest_state = latest_state.model_copy(
                update={
                    "status": "delivered",
                    "delivery_count": delivery_count,
                    "success_count": int(latest_state.success_count) + 1,
                    "last_attempt_at": _now_iso(),
                    "last_success_at": _now_iso(),
                    "last_response_status": response_status,
                    "last_error": None,
                    "next_retry_at": None,
                }
            )
            task_store.write_task(task.model_copy(update={"callback_state": latest_state}))
            task_store.append_event(
                task.task_id,
                "callback_delivered",
                {
                    "delivery_id": delivery_id,
                    "event_type": event_type,
                    "attempt_no": attempt_no,
                    "response_status": response_status,
                },
            )
            if audit_hook is not None:
                audit_hook(
                    "task.callback.delivery",
                    resource_type="async_task",
                    resource_id=task.task_id,
                    status="delivered",
                    payload={
                        "delivery_id": delivery_id,
                        "event_type": event_type,
                        "attempt_no": attempt_no,
                        "response_status": response_status,
                        "duration_ms": duration_ms,
                    },
                    metadata={"requested_by": requested_by, "callback_url": callback.url},
                    tenant_id=task.tenant_id,
                )
            return {
                "configured": True,
                "delivery_id": delivery_id,
                "event_type": event_type,
                "status": "delivered",
                "attempt_no": attempt_no,
                "response_status": response_status,
                "duration_ms": duration_ms,
                "callback_url": callback.url,
            }

        next_retry_at = None
        observe_async_task_callback_delivery(
            event_type=event_type,
            status="failed" if attempt_no < max_attempts else "dead_lettered",
            duration_ms=duration_ms,
        )
        if attempt_no < max_attempts and max_backoff_seconds > 0:
            sleep_seconds = _next_backoff_seconds(backoff_seconds, attempt_no, max_backoff_seconds)
            if sleep_seconds > 0:
                next_retry_at = datetime.fromtimestamp(time.time() + sleep_seconds, tz=timezone.utc).isoformat()
        latest_state = latest_state.model_copy(
            update={
                "status": "failed" if attempt_no < max_attempts else "dead_lettered",
                "delivery_count": delivery_count,
                "failure_count": int(latest_state.failure_count) + 1,
                "last_attempt_at": _now_iso(),
                "last_response_status": response_status,
                "last_error": error_message,
                "next_retry_at": next_retry_at,
            }
        )
        task_store.write_task(task.model_copy(update={"callback_state": latest_state}))
        if attempt_no < max_attempts:
            sleep_seconds = _next_backoff_seconds(backoff_seconds, attempt_no, max_backoff_seconds)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    task_store.append_event(
        task.task_id,
        "callback_failed",
        {
            "delivery_id": delivery_id,
            "event_type": event_type,
            "delivery_count": latest_state.delivery_count,
            "failure_count": latest_state.failure_count,
            "last_error": latest_state.last_error,
        },
    )
    if audit_hook is not None:
        audit_hook(
            "task.callback.delivery",
            resource_type="async_task",
            resource_id=task.task_id,
            status="dead_lettered",
            payload={
                "delivery_id": delivery_id,
                "event_type": event_type,
                "delivery_count": latest_state.delivery_count,
                "failure_count": latest_state.failure_count,
                "last_error": latest_state.last_error,
            },
            metadata={"requested_by": requested_by, "callback_url": callback.url},
            tenant_id=task.tenant_id,
        )
    return {
        "configured": True,
        "delivery_id": delivery_id,
        "event_type": event_type,
        "status": "dead_lettered",
        "delivery_count": latest_state.delivery_count,
        "failure_count": latest_state.failure_count,
        "last_error": latest_state.last_error,
        "callback_url": callback.url,
    }

from __future__ import annotations

from collections import Counter
from typing import Any


def normalize_gpu_devices(devices: list[int] | tuple[int, ...] | None) -> list[int]:
    normalized: list[int] = []
    for value in devices or []:
        try:
            device_id = int(value)
        except Exception:
            continue
        if device_id < 0 or device_id in normalized:
            continue
        normalized.append(device_id)
    return normalized


def dispatch_gpu_page_jobs(
    *,
    task_id: str,
    page_jobs: list[dict[str, Any]],
    devices: list[int] | tuple[int, ...] | None,
) -> dict[str, Any]:
    worker_device_ids = normalize_gpu_devices(devices) or [0]
    assigned_jobs: list[dict[str, Any]] = []
    device_job_counts: Counter[int] = Counter()
    route_counts: Counter[str] = Counter()
    ocr_scope_counts: Counter[str] = Counter()

    for index, job in enumerate(page_jobs):
        worker_device_id = worker_device_ids[index % len(worker_device_ids)]
        route = str(job.get("route") or "unknown").strip() or "unknown"
        ocr_scope = str(job.get("ocr_scope") or "unknown").strip() or "unknown"
        device_job_counts[worker_device_id] += 1
        route_counts[route] += 1
        ocr_scope_counts[ocr_scope] += 1
        assigned_jobs.append(
            {
                **job,
                "task_id": task_id,
                "worker_device_id": worker_device_id,
            }
        )

    return {
        "submitted_job_count": len(assigned_jobs),
        "worker_device_ids": worker_device_ids,
        "device_job_counts": dict(device_job_counts),
        "route_counts": dict(route_counts),
        "ocr_scope_counts": dict(ocr_scope_counts),
        "page_jobs": assigned_jobs,
    }

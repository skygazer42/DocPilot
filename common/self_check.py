from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from common import setting


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SelfCheckStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    name: str
    status: Literal["running", "passed", "failed", "skipped"]
    started_at: str
    finished_at: str | None = None
    duration_ms: int | None = None
    summary: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class SelfCheckRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    suite: str
    status: Literal["running", "passed", "failed"]
    created_at: str
    finished_at: str | None = None
    duration_ms: int | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    steps: list[SelfCheckStep] = Field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def summarize_self_check_run(run: SelfCheckRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "check_id": run.check_id,
        "suite": run.suite,
        "status": run.status,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
        "duration_ms": run.duration_ms,
    }


class SelfCheckPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    root_dir: str
    result_path: str


class SelfCheckStore:
    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = Path(root_dir or setting.SELF_CHECKS_DIR)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def get_paths(self, check_id: str) -> SelfCheckPaths:
        root_dir = self.root_dir / check_id
        root_dir.mkdir(parents=True, exist_ok=True)
        return SelfCheckPaths(
            check_id=check_id,
            root_dir=str(root_dir),
            result_path=str(root_dir / "result.json"),
        )

    def write_run(self, run: SelfCheckRun) -> SelfCheckPaths:
        paths = self.get_paths(run.check_id)
        Path(paths.result_path).write_text(
            json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return paths

    def load_run(self, check_id: str) -> SelfCheckRun:
        paths = self.get_paths(check_id)
        payload = json.loads(Path(paths.result_path).read_text(encoding="utf-8"))
        return SelfCheckRun.model_validate(payload)

    def latest_run(self, *, status: str | None = None) -> SelfCheckRun | None:
        runs = self.list_runs(limit=1, status=status)
        return runs[0] if runs else None

    def check_health(self) -> dict[str, Any]:
        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            latest = self.latest_run()
            payload: dict[str, Any] = {
                "status": "ok",
                "backend": "local",
                "root_dir": str(self.root_dir),
            }
            if latest is not None:
                payload["latest_run"] = summarize_self_check_run(latest)
            return payload
        except Exception as exc:
            return {
                "status": "error",
                "backend": "local",
                "root_dir": str(self.root_dir),
                "error": str(exc),
            }

    def list_runs(self, *, limit: int = 20, status: str | None = None) -> list[SelfCheckRun]:
        runs: list[SelfCheckRun] = []
        for result_path in self.root_dir.glob("*/result.json"):
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                run = SelfCheckRun.model_validate(payload)
            except Exception:
                continue
            if status and run.status != status:
                continue
            runs.append(run)
        runs.sort(
            key=lambda item: (
                item.finished_at or item.created_at or "",
                item.check_id,
            ),
            reverse=True,
        )
        return runs[: max(1, int(limit))]

    def cleanup_runs(
        self,
        *,
        older_than_days: int | None = None,
        keep_latest: int | None = None,
        limit: int = 1000,
        dry_run: bool = True,
        status: str | None = None,
    ) -> dict[str, Any]:
        runs = self.list_runs(limit=max(limit * 2, 2000), status=status)
        candidates = runs
        if older_than_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, int(older_than_days)))
            filtered: list[SelfCheckRun] = []
            for run in candidates:
                created_at = run.finished_at or run.created_at
                try:
                    if datetime.fromisoformat(created_at) <= cutoff:
                        filtered.append(run)
                except Exception:
                    filtered.append(run)
            candidates = filtered
        if keep_latest and keep_latest > 0:
            keep_ids = {run.check_id for run in runs[: int(keep_latest)]}
            candidates = [run for run in candidates if run.check_id not in keep_ids]
        candidates = candidates[: max(1, int(limit))]
        deleted: list[str] = []
        if not dry_run:
            for run in candidates:
                run_dir = self.root_dir / run.check_id
                if run_dir.exists():
                    shutil.rmtree(run_dir, ignore_errors=True)
                deleted.append(run.check_id)
        return {
            "dry_run": bool(dry_run),
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "check_id": run.check_id,
                    "suite": run.suite,
                    "status": run.status,
                    "created_at": run.created_at,
                    "finished_at": run.finished_at,
                }
                for run in candidates
            ],
            "deleted_count": len(deleted),
            "deleted": deleted,
        }


def new_self_check_run(*, suite: str, environment: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> SelfCheckRun:
    check_id = uuid4().hex
    return SelfCheckRun(
        check_id=check_id,
        suite=str(suite or "core").strip() or "core",
        status="running",
        created_at=_now_iso(),
        environment=environment or {},
        metadata=metadata or {},
    )

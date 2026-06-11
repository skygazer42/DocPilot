from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from common import setting


def _parse_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_csv_list(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _parse_optional_int(
    value: object,
    *,
    minimum: int = 0,
    default: int | None = None,
) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    parsed = int(value)
    if parsed < minimum:
        return minimum
    return parsed


class RetentionJanitorRule(BaseModel):
    name: Literal["tasks", "artifacts", "audit_events", "self_checks"]
    enabled: bool = True
    older_than_days: int | None = None
    keep_latest: int | None = None
    limit: int | None = None
    statuses: list[str] = Field(default_factory=list)
    include_active: bool = False

    def summary(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "older_than_days": self.older_than_days,
            "keep_latest": self.keep_latest,
            "limit": self.limit,
            "statuses": list(self.statuses),
            "include_active": self.include_active,
        }


class RetentionJanitorConfig(BaseModel):
    enabled: bool
    required_for_ready: bool = False
    poll_seconds: int = 3600
    heartbeat_seconds: int = 15
    run_on_start: bool = True
    heartbeat_file: str
    health_max_age_seconds: int = 120
    last_run_max_age_seconds: int = 86400
    rules: dict[str, RetentionJanitorRule]

    def summarized_rules(self) -> dict[str, dict[str, object]]:
        return {name: rule.summary() for name, rule in self.rules.items()}


def _build_rule(
    *,
    name: Literal["tasks", "artifacts", "audit_events", "self_checks"],
    enabled_default: bool,
    older_than_days_default: int | None,
    keep_latest_default: int | None,
    limit_default: int | None,
    statuses_default: list[str] | None = None,
    include_active_default: bool = False,
) -> RetentionJanitorRule:
    prefix = f"DEEPDOC_RETENTION_JANITOR_{name.upper()}"
    enabled = _parse_bool(os.environ.get(f"{prefix}_ENABLED"), default=enabled_default)
    older_than_days = _parse_optional_int(
        os.environ.get(f"{prefix}_OLDER_THAN_DAYS"),
        minimum=0 if name in {"tasks", "artifacts", "self_checks"} else 1,
        default=older_than_days_default,
    )
    keep_latest = _parse_optional_int(
        os.environ.get(f"{prefix}_KEEP_LATEST"),
        minimum=0,
        default=keep_latest_default,
    )
    limit = _parse_optional_int(
        os.environ.get(f"{prefix}_LIMIT"),
        minimum=1,
        default=limit_default,
    )
    statuses = _parse_csv_list(os.environ.get(f"{prefix}_STATUSES"))
    if not statuses and statuses_default:
        statuses = list(statuses_default)
    include_active = _parse_bool(
        os.environ.get(f"{prefix}_INCLUDE_ACTIVE"),
        default=include_active_default,
    )
    return RetentionJanitorRule(
        name=name,
        enabled=enabled,
        older_than_days=older_than_days,
        keep_latest=keep_latest,
        limit=limit,
        statuses=statuses,
        include_active=include_active,
    )


def load_retention_janitor_config() -> RetentionJanitorConfig:
    enabled = _parse_bool(os.environ.get("DEEPDOC_RETENTION_JANITOR_ENABLED"), default=False)
    rules = {
        "tasks": _build_rule(
            name="tasks",
            enabled_default=True,
            older_than_days_default=7,
            keep_latest_default=1000,
            limit_default=2000,
            statuses_default=["succeeded", "failed", "cancelled"],
            include_active_default=False,
        ),
        "artifacts": _build_rule(
            name="artifacts",
            enabled_default=True,
            older_than_days_default=30,
            keep_latest_default=1000,
            limit_default=2000,
        ),
        "audit_events": _build_rule(
            name="audit_events",
            enabled_default=True,
            older_than_days_default=14,
            keep_latest_default=5000,
            limit_default=None,
        ),
        "self_checks": _build_rule(
            name="self_checks",
            enabled_default=True,
            older_than_days_default=30,
            keep_latest_default=200,
            limit_default=1000,
        ),
    }
    return RetentionJanitorConfig(
        enabled=enabled,
        required_for_ready=_parse_bool(os.environ.get("DEEPDOC_RETENTION_JANITOR_REQUIRED_FOR_READY"), default=False),
        poll_seconds=max(60, int(os.environ.get("DEEPDOC_RETENTION_JANITOR_POLL_SECONDS", "3600"))),
        heartbeat_seconds=max(5, int(os.environ.get("DEEPDOC_RETENTION_JANITOR_HEARTBEAT_SECONDS", "15"))),
        run_on_start=_parse_bool(os.environ.get("DEEPDOC_RETENTION_JANITOR_RUN_ON_START"), default=True),
        heartbeat_file=str(
            os.environ.get(
                "DEEPDOC_RETENTION_JANITOR_HEARTBEAT_FILE",
                os.path.join(setting.TASKS_DIR, "retention-janitor-heartbeat.json"),
            )
        ).strip(),
        health_max_age_seconds=max(
            30,
            int(os.environ.get("DEEPDOC_RETENTION_JANITOR_HEALTH_MAX_AGE_SECONDS", "120")),
        ),
        last_run_max_age_seconds=max(
            300,
            int(os.environ.get("DEEPDOC_RETENTION_JANITOR_LAST_RUN_MAX_AGE_SECONDS", "86400")),
        ),
        rules=rules,
    )

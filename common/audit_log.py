from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from common import logger, setting
from common.ingest_postgres import _load_psycopg, _redact_dsn, _validate_identifier


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


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: str = Field(default_factory=_now_iso)
    tenant_id: str | None = None
    actor_subject: str | None = None
    actor_mode: str | None = None
    actor_is_admin: bool = False
    actor_scopes: list[str] = Field(default_factory=list)
    request_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    action: str
    resource_type: str
    resource_id: str | None = None
    status: str = "ok"
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditLogStore(ABC):
    backend_name = "unknown"

    @abstractmethod
    def check_health(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def append_event(self, event: AuditEvent) -> AuditEvent:
        raise NotImplementedError

    @abstractmethod
    def get_event(self, event_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def list_events(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
        action: str | None = None,
        resource_type: str | None = None,
        status: str | None = None,
        request_id: str | None = None,
        actor_subject: str | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def cleanup_events(
        self,
        *,
        tenant_id: str | None = None,
        older_than_days: int | None = None,
        keep_latest: int = 0,
        action: str | None = None,
        resource_type: str | None = None,
        status: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        raise NotImplementedError


class LocalAuditLogStore(AuditLogStore):
    backend_name = "local"

    def __init__(self, root_dir: str | Path | None = None):
        self.root_dir = Path(root_dir or setting.AUDIT_DIR)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root_dir / "events.jsonl"

    @staticmethod
    def _normalize_tenant_id(value: str | None) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    def check_health(self) -> dict[str, Any]:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "ok",
            "backend": self.backend_name,
            "root_dir": str(self.root_dir),
            "events_path": str(self.events_path),
        }

    def append_event(self, event: AuditEvent) -> AuditEvent:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return event

    def _load_events(self) -> list[AuditEvent]:
        if not self.events_path.exists():
            return []
        results: list[AuditEvent] = []
        with self.events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(AuditEvent.model_validate(json.loads(line)))
                except Exception:
                    logger.exception("Failed to parse audit log line from %s", self.events_path)
        return results

    def get_event(self, event_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        normalized_event_id = str(event_id or "").strip()
        normalized_tenant = self._normalize_tenant_id(tenant_id)
        for event in self._load_events():
            if event.event_id != normalized_event_id:
                continue
            event_tenant = self._normalize_tenant_id(event.tenant_id)
            if normalized_tenant and event_tenant != normalized_tenant:
                continue
            return event.model_dump(mode="json")
        return None

    def list_events(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
        action: str | None = None,
        resource_type: str | None = None,
        status: str | None = None,
        request_id: str | None = None,
        actor_subject: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_tenant = self._normalize_tenant_id(tenant_id)
        normalized_action = str(action or "").strip() or None
        normalized_resource_type = str(resource_type or "").strip() or None
        normalized_status = str(status or "").strip() or None
        normalized_request_id = str(request_id or "").strip() or None
        normalized_actor_subject = str(actor_subject or "").strip() or None
        events = sorted(
            self._load_events(),
            key=lambda item: (_parse_iso_datetime(item.created_at) or datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )
        results: list[dict[str, Any]] = []
        start = max(0, int(offset))
        end = start + max(1, int(limit))
        for event in events:
            event_tenant = self._normalize_tenant_id(event.tenant_id)
            if normalized_tenant and event_tenant != normalized_tenant:
                continue
            if normalized_action and event.action != normalized_action:
                continue
            if normalized_resource_type and event.resource_type != normalized_resource_type:
                continue
            if normalized_status and event.status != normalized_status:
                continue
            if normalized_request_id and str(event.request_id or "").strip() != normalized_request_id:
                continue
            if normalized_actor_subject and str(event.actor_subject or "").strip() != normalized_actor_subject:
                continue
            results.append(event.model_dump(mode="json"))
        return results[start:end]

    def cleanup_events(
        self,
        *,
        tenant_id: str | None = None,
        older_than_days: int | None = None,
        keep_latest: int = 0,
        action: str | None = None,
        resource_type: str | None = None,
        status: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        normalized_tenant = self._normalize_tenant_id(tenant_id)
        normalized_action = str(action or "").strip() or None
        normalized_resource_type = str(resource_type or "").strip() or None
        normalized_status = str(status or "").strip() or None
        cutoff = None
        if older_than_days is not None:
            cutoff = _now() - timedelta(days=max(1, int(older_than_days)))
        events = sorted(
            self._load_events(),
            key=lambda item: (_parse_iso_datetime(item.created_at) or datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )
        filtered: list[AuditEvent] = []
        kept: list[AuditEvent] = []
        for event in events:
            event_tenant = self._normalize_tenant_id(event.tenant_id)
            if normalized_tenant and event_tenant != normalized_tenant:
                kept.append(event)
                continue
            if normalized_action and event.action != normalized_action:
                kept.append(event)
                continue
            if normalized_resource_type and event.resource_type != normalized_resource_type:
                kept.append(event)
                continue
            if normalized_status and event.status != normalized_status:
                kept.append(event)
                continue
            event_time = _parse_iso_datetime(event.created_at)
            if cutoff is not None and (event_time is None or event_time >= cutoff):
                kept.append(event)
                continue
            filtered.append(event)
        keep_latest = max(0, int(keep_latest or 0))
        victims = filtered[keep_latest:] if keep_latest else filtered
        victim_ids = [event.event_id for event in victims]
        deleted_count = 0
        if not dry_run and victim_ids:
            victim_set = set(victim_ids)
            remaining = [event for event in self._load_events() if event.event_id not in victim_set]
            with self.events_path.open("w", encoding="utf-8") as handle:
                for event in remaining:
                    handle.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
            deleted_count = len(victim_ids)
        return {
            "backend": self.backend_name,
            "dry_run": bool(dry_run),
            "tenant_id": normalized_tenant,
            "older_than_days": max(1, int(older_than_days)) if older_than_days is not None else None,
            "keep_latest": keep_latest,
            "action": normalized_action,
            "resource_type": normalized_resource_type,
            "status": normalized_status,
            "scanned": len(events),
            "candidate_count": len(victims),
            "deleted_count": deleted_count,
            "event_ids": victim_ids[:100],
        }


class PostgresAuditLogStore(AuditLogStore):
    backend_name = "postgres"

    def __init__(self, *, dsn: str, schema: str, connect_timeout: int = 10):
        self.dsn = dsn
        self.schema = _validate_identifier(schema, "schema")
        self.connect_timeout = max(1, int(connect_timeout))
        self.events_table = "audit_events"
        self._schema_ready = False

    def _qualified(self, table_name: str) -> str:
        return f'"{self.schema}"."{table_name}"'

    @staticmethod
    def _normalize_tenant_id(value: str | None) -> str:
        return str(value or "").strip()

    def _connect(self):
        psycopg, dict_row = _load_psycopg()
        return psycopg.connect(
            self.dsn,
            autocommit=False,
            row_factory=dict_row,
            connect_timeout=self.connect_timeout,
        )

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._qualified(self.events_table)} (
                        event_id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        tenant_id TEXT NOT NULL DEFAULT '',
                        actor_subject TEXT,
                        actor_mode TEXT,
                        actor_is_admin BOOLEAN NOT NULL DEFAULT FALSE,
                        actor_scopes JSONB NOT NULL DEFAULT '[]'::jsonb,
                        request_id TEXT,
                        trace_id TEXT,
                        span_id TEXT,
                        action TEXT NOT NULL,
                        resource_type TEXT NOT NULL,
                        resource_id TEXT,
                        status TEXT NOT NULL,
                        payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.events_table}_created_at
                    ON {self._qualified(self.events_table)} (created_at DESC)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.events_table}_tenant_id
                    ON {self._qualified(self.events_table)} (tenant_id, created_at DESC)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.events_table}_action
                    ON {self._qualified(self.events_table)} (action, created_at DESC)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.events_table}_resource_type
                    ON {self._qualified(self.events_table)} (resource_type, created_at DESC)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.events_table}_status
                    ON {self._qualified(self.events_table)} (status, created_at DESC)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.events_table}_request_id
                    ON {self._qualified(self.events_table)} (request_id)
                    """
                )
            conn.commit()
            self._schema_ready = True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def check_health(self) -> dict[str, Any]:
        self.ensure_schema()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database() AS database_name")
                row = cur.fetchone() or {}
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "status": "ok",
            "backend": self.backend_name,
            "schema": self.schema,
            "database": dict(row).get("database_name"),
            "dsn": _redact_dsn(self.dsn),
        }

    def append_event(self, event: AuditEvent) -> AuditEvent:
        self.ensure_schema()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._qualified(self.events_table)} (
                        event_id, created_at, tenant_id, actor_subject, actor_mode, actor_is_admin, actor_scopes,
                        request_id, trace_id, span_id, action, resource_type, resource_id, status, payload, metadata
                    )
                    VALUES (
                        %s, %s::timestamptz, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb
                    )
                    ON CONFLICT (event_id) DO UPDATE SET
                        tenant_id = EXCLUDED.tenant_id,
                        actor_subject = EXCLUDED.actor_subject,
                        actor_mode = EXCLUDED.actor_mode,
                        actor_is_admin = EXCLUDED.actor_is_admin,
                        actor_scopes = EXCLUDED.actor_scopes,
                        request_id = EXCLUDED.request_id,
                        trace_id = EXCLUDED.trace_id,
                        span_id = EXCLUDED.span_id,
                        action = EXCLUDED.action,
                        resource_type = EXCLUDED.resource_type,
                        resource_id = EXCLUDED.resource_id,
                        status = EXCLUDED.status,
                        payload = EXCLUDED.payload,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        event.event_id,
                        event.created_at,
                        self._normalize_tenant_id(event.tenant_id),
                        event.actor_subject,
                        event.actor_mode,
                        bool(event.actor_is_admin),
                        json.dumps(event.actor_scopes, ensure_ascii=False),
                        event.request_id,
                        event.trace_id,
                        event.span_id,
                        event.action,
                        event.resource_type,
                        event.resource_id,
                        event.status,
                        json.dumps(event.payload or {}, ensure_ascii=False),
                        json.dumps(event.metadata or {}, ensure_ascii=False),
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return event

    def get_event(self, event_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        self.ensure_schema()
        normalized_tenant = self._normalize_tenant_id(tenant_id)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = f"SELECT * FROM {self._qualified(self.events_table)} WHERE event_id = %s"
                params: list[Any] = [str(event_id or "").strip()]
                if normalized_tenant:
                    sql += " AND tenant_id = %s"
                    params.append(normalized_tenant)
                cur.execute(sql, params)
                row = cur.fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return dict(row) if row else None

    def list_events(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
        action: str | None = None,
        resource_type: str | None = None,
        status: str | None = None,
        request_id: str | None = None,
        actor_subject: str | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        where_clauses: list[str] = []
        params: list[Any] = []
        normalized_tenant = self._normalize_tenant_id(tenant_id)
        if normalized_tenant:
            where_clauses.append("tenant_id = %s")
            params.append(normalized_tenant)
        if action:
            where_clauses.append("action = %s")
            params.append(str(action).strip())
        if resource_type:
            where_clauses.append("resource_type = %s")
            params.append(str(resource_type).strip())
        if status:
            where_clauses.append("status = %s")
            params.append(str(status).strip())
        if request_id:
            where_clauses.append("request_id = %s")
            params.append(str(request_id).strip())
        if actor_subject:
            where_clauses.append("actor_subject = %s")
            params.append(str(actor_subject).strip())
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = f"""
            SELECT *
            FROM {self._qualified(self.events_table)}
            {where_sql}
            ORDER BY created_at DESC, event_id DESC
            LIMIT %s OFFSET %s
        """
        params.extend([max(1, int(limit)), max(0, int(offset))])
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() or []
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def cleanup_events(
        self,
        *,
        tenant_id: str | None = None,
        older_than_days: int | None = None,
        keep_latest: int = 0,
        action: str | None = None,
        resource_type: str | None = None,
        status: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        self.ensure_schema()
        normalized_tenant = self._normalize_tenant_id(tenant_id)
        where_clauses: list[str] = []
        params: list[Any] = []
        if normalized_tenant:
            where_clauses.append("tenant_id = %s")
            params.append(normalized_tenant)
        if action:
            where_clauses.append("action = %s")
            params.append(str(action).strip())
        if resource_type:
            where_clauses.append("resource_type = %s")
            params.append(str(resource_type).strip())
        if status:
            where_clauses.append("status = %s")
            params.append(str(status).strip())
        if older_than_days is not None:
            where_clauses.append("created_at < NOW() - (%s * INTERVAL '1 day')")
            params.append(max(1, int(older_than_days)))
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        select_sql = f"""
            SELECT event_id
            FROM {self._qualified(self.events_table)}
            {where_sql}
            ORDER BY created_at DESC, event_id DESC
        """
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(select_sql, params)
                rows = [dict(row) for row in (cur.fetchall() or [])]
                keep_latest = max(0, int(keep_latest or 0))
                victims = rows[keep_latest:] if keep_latest else rows
                event_ids = [str(row.get("event_id") or "").strip() for row in victims if str(row.get("event_id") or "").strip()]
                deleted_count = 0
                if event_ids and not dry_run:
                    delete_sql = f"DELETE FROM {self._qualified(self.events_table)} WHERE event_id = ANY(%s)"
                    cur.execute(delete_sql, [event_ids])
                    deleted_count = max(0, int(cur.rowcount or 0))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "backend": self.backend_name,
            "dry_run": bool(dry_run),
            "tenant_id": normalized_tenant or None,
            "older_than_days": max(1, int(older_than_days)) if older_than_days is not None else None,
            "keep_latest": max(0, int(keep_latest or 0)),
            "action": str(action or "").strip() or None,
            "resource_type": str(resource_type or "").strip() or None,
            "status": str(status or "").strip() or None,
            "scanned": len(rows),
            "candidate_count": len(victims),
            "deleted_count": deleted_count,
            "event_ids": event_ids[:100],
        }


def create_audit_log_store() -> AuditLogStore | None:
    backend = (os.environ.get("DEEPDOC_AUDIT_BACKEND") or "auto").strip().lower()
    if backend in {"none", "disabled"}:
        return None
    if backend == "auto":
        if (os.environ.get("DEEPDOC_INGEST_PG_DSN") or "").strip():
            backend = "postgres"
        else:
            backend = "local"
    if backend in {"local", "file"}:
        return LocalAuditLogStore()
    if backend == "postgres":
        dsn = (os.environ.get("DEEPDOC_AUDIT_PG_DSN") or os.environ.get("DEEPDOC_INGEST_PG_DSN") or "").strip()
        if not dsn:
            logger.warning("DEEPDOC_AUDIT_BACKEND=postgres but no PostgreSQL DSN configured, fallback to local")
            return LocalAuditLogStore()
        schema = (
            os.environ.get("DEEPDOC_AUDIT_PG_SCHEMA")
            or os.environ.get("DEEPDOC_INGEST_PG_SCHEMA")
            or "deepdoc_ingest"
        ).strip()
        connect_timeout = int(os.environ.get("DEEPDOC_AUDIT_PG_CONNECT_TIMEOUT") or os.environ.get("DEEPDOC_INGEST_PG_CONNECT_TIMEOUT") or "10")
        return PostgresAuditLogStore(dsn=dsn, schema=schema, connect_timeout=connect_timeout)
    raise RuntimeError(f"Unsupported DEEPDOC_AUDIT_BACKEND: {backend}")

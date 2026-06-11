from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from common.ingest_publisher import IngestPublishResult, IngestPublisher
from common.parse_artifacts import ParseArtifact, ParseManifest, ChunkExportRecord, IngestExportRecord


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str, name: str) -> str:
    normalized = (value or "").strip()
    if not normalized or not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"Invalid PostgreSQL identifier for {name}: {value!r}")
    return normalized


def _redact_dsn(dsn: str) -> str:
    if "@" not in dsn or "://" not in dsn:
        return dsn
    prefix, suffix = dsn.split("://", 1)
    if "@" not in suffix:
        return dsn
    auth, rest = suffix.split("@", 1)
    if ":" not in auth:
        return f"{prefix}://***@{rest}"
    user, _ = auth.split(":", 1)
    return f"{prefix}://{user}:***@{rest}"


@dataclass(frozen=True)
class PostgresIngestConfig:
    dsn: str
    schema: str
    connect_timeout: int


def _load_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except Exception as exc:
        raise RuntimeError(
            "PostgreSQL ingest backend requires psycopg. Install it first, e.g. pip install 'psycopg[binary]'"
        ) from exc
    return psycopg, dict_row


class _PostgresBackendBase:
    def __init__(
        self,
        config: PostgresIngestConfig,
        *,
        connect_fn: Callable[[], Any] | None = None,
    ):
        self.config = config
        self.schema = _validate_identifier(config.schema, "schema")
        self.documents_table = "documents"
        self.records_table = "records"
        self.chunks_table = "chunks"
        self.assets_table = "assets"
        self.chunk_asset_links_table = "chunk_asset_links"
        self.parse_aliases_table = "parse_aliases"
        self._schema_ready = False
        self._connect_fn = connect_fn

    def _qualified(self, table_name: str) -> str:
        return f'"{self.schema}"."{table_name}"'

    @staticmethod
    def _normalized_tenant_id(value: str | None) -> str:
        return str(value or "").strip()

    @staticmethod
    def _normalize_row(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        if hasattr(row, "_mapping"):
            return dict(row._mapping)
        if isinstance(row, (list, tuple)):
            if len(row) == 10:
                return {
                    "parse_id": row[0],
                    "document_id": row[1],
                    "tenant_id": row[2],
                    "artifact_key": row[3],
                    "filename": row[4],
                    "file_type": row[5],
                    "parser_engine": row[6],
                    "created_at": row[7],
                    "updated_at": row[8],
                    "manifest": row[9],
                }
            if len(row) == 8:
                return {
                    "parse_id": row[0],
                    "document_id": row[1],
                    "artifact_key": row[2],
                    "filename": row[3],
                    "file_type": row[4],
                    "parser_engine": row[5],
                    "created_at": row[6],
                    "updated_at": row[7],
                }
            if len(row) == 6:
                return {
                    "request_id": row[0],
                    "tenant_id": row[1],
                    "sequence_no": row[2],
                    "event_type": row[3],
                    "payload": row[4],
                    "created_at": row[5],
                }
            if len(row) == 3:
                return {"database_name": row[0], "extversion": row[1], "status": row[2]}
            if len(row) == 2:
                if isinstance(row[0], str) and not isinstance(row[1], (int, float)):
                    return {"database_name": row[0], "extversion": row[1]}
                return {"payload": row[0], "score": row[1], "distance": row[1]}
            if len(row) == 1:
                value = row[0]
                normalized = {"value": value}
                if isinstance(value, str):
                    normalized.update(
                        {
                            "database_name": value,
                            "extversion": value,
                            "canonical_parse_id": value,
                        }
                    )
                else:
                    normalized["manifest"] = value
                return normalized
        raise TypeError(f"Unsupported PostgreSQL row type: {type(row)!r}")

    @staticmethod
    def _scalar_from_row(row: Any, *preferred_keys: str) -> Any | None:
        if row is None:
            return None
        normalized = _PostgresBackendBase._normalize_row(row)
        for key in preferred_keys:
            if key and key in normalized:
                return normalized[key]
        if "value" in normalized:
            return normalized["value"]
        values = list(normalized.values())
        return values[0] if values else None

    @staticmethod
    def _maybe_json(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _tenant_id_from_manifest(self, manifest: ParseManifest) -> str:
        return self._normalized_tenant_id((manifest.metadata or {}).get("tenant_id"))

    @staticmethod
    def _asset_pk(parse_id: str, asset_id: str) -> str:
        return f"{parse_id}:{asset_id}"

    @staticmethod
    def _chunk_asset_link_id(record_id: str, asset_id: str, relation_type: str) -> str:
        return f"{record_id}:{asset_id}:{relation_type}"

    @staticmethod
    def _annotate_alias_payload(payload: dict[str, Any], *, requested_parse_id: str, canonical_parse_id: str) -> dict[str, Any]:
        if not requested_parse_id or requested_parse_id == canonical_parse_id:
            return payload
        result = dict(payload or {})
        metadata = dict(result.get("metadata") or {})
        metadata.update(
            {
                "alias_parse_id": requested_parse_id,
                "canonical_parse_id": canonical_parse_id,
            }
        )
        result["metadata"] = metadata
        result.setdefault("parse_id", canonical_parse_id)
        return result

    def _build_asset_payload(
        self,
        *,
        manifest: ParseManifest,
        asset_by_id: dict[str, Any],
        resolved_asset_by_id: dict[str, dict[str, Any]],
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        asset_rows: list[dict[str, Any]] = []
        for asset_id, asset in asset_by_id.items():
            resolved = resolved_asset_by_id.get(asset_id) or {}
            storage = getattr(asset, "storage", None)
            payload = asset.model_dump(mode="json")
            if resolved.get("resolved_url"):
                payload["resolved_url"] = resolved["resolved_url"]
            if resolved.get("download_path"):
                payload["download_path"] = resolved["download_path"]
            asset_rows.append(
                {
                    "asset_pk": self._asset_pk(manifest.parse_id, asset_id),
                    "parse_id": manifest.parse_id,
                    "document_id": manifest.document_id,
                    "tenant_id": tenant_id,
                    "asset_id": asset_id,
                    "asset_type": str(getattr(asset, "asset_type", "") or ""),
                    "title": getattr(asset, "title", None),
                    "text": str(getattr(asset, "text", "") or ""),
                    "page_numbers": list(getattr(asset, "page_numbers", []) or []),
                    "positions": payload.get("positions") or [],
                    "width": getattr(asset, "width", None),
                    "height": getattr(asset, "height", None),
                    "sha256": getattr(asset, "sha256", None),
                    "storage_backend": getattr(storage, "backend", None),
                    "storage_relative_path": getattr(storage, "relative_path", None),
                    "storage_absolute_path": getattr(storage, "absolute_path", None),
                    "storage_download_path": getattr(storage, "download_path", None),
                    "storage_resolved_url": resolved.get("resolved_url"),
                    "storage_source_url": getattr(storage, "source_url", None),
                    "media_type": getattr(storage, "media_type", None),
                    "metadata": payload.get("metadata") or {},
                    "payload": payload,
                }
            )
        return asset_rows

    def _build_chunk_payload(
        self,
        *,
        manifest: ParseManifest,
        chunk_records: list[ChunkExportRecord],
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        chunk_rows: list[dict[str, Any]] = []
        for chunk_record in chunk_records:
            record_id = f"{chunk_record.document_id}:{chunk_record.chunk_id}"
            payload = chunk_record.model_dump(mode="json")
            chunk_rows.append(
                {
                    "record_id": record_id,
                    "parse_id": manifest.parse_id,
                    "document_id": manifest.document_id,
                    "tenant_id": tenant_id,
                    "chunk_id": chunk_record.chunk_id,
                    "text": chunk_record.text,
                    "token_count": int(chunk_record.token_count),
                    "page_numbers": list(chunk_record.page_numbers or []),
                    "block_refs": list(chunk_record.block_refs or []),
                    "asset_refs": list(chunk_record.asset_refs or []),
                    "metadata": payload.get("metadata") or {},
                    "payload": payload,
                }
            )
        return chunk_rows

    def _build_chunk_asset_link_payload(
        self,
        *,
        manifest: ParseManifest,
        chunk_records: list[ChunkExportRecord],
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        link_rows: list[dict[str, Any]] = []
        for chunk_record in chunk_records:
            record_id = f"{chunk_record.document_id}:{chunk_record.chunk_id}"
            metadata = dict(chunk_record.metadata or {})
            asset_refs = list(chunk_record.asset_refs or [])
            direct_refs = list(metadata.get("direct_asset_refs") or [])
            context_refs = list(metadata.get("context_asset_refs") or [])
            if not direct_refs and asset_refs:
                direct_refs = list(asset_refs)
            if not context_refs and asset_refs:
                context_refs = [asset_id for asset_id in asset_refs if asset_id not in direct_refs]
            direct_order = {asset_id: idx for idx, asset_id in enumerate(direct_refs)}
            context_order = {asset_id: idx for idx, asset_id in enumerate(context_refs)}
            for asset_id in direct_refs:
                link_rows.append(
                    {
                        "link_id": self._chunk_asset_link_id(record_id, asset_id, "direct"),
                        "parse_id": manifest.parse_id,
                        "document_id": manifest.document_id,
                        "tenant_id": tenant_id,
                        "record_id": record_id,
                        "chunk_id": chunk_record.chunk_id,
                        "asset_pk": self._asset_pk(manifest.parse_id, asset_id),
                        "asset_id": asset_id,
                        "relation_type": "direct",
                        "ordinal": int(direct_order.get(asset_id, 0)),
                        "metadata": {
                            "page_numbers": list(chunk_record.page_numbers or []),
                            "title_path": list(metadata.get("title_path") or []),
                        },
                    }
                )
            for asset_id in context_refs:
                link_rows.append(
                    {
                        "link_id": self._chunk_asset_link_id(record_id, asset_id, "context"),
                        "parse_id": manifest.parse_id,
                        "document_id": manifest.document_id,
                        "tenant_id": tenant_id,
                        "record_id": record_id,
                        "chunk_id": chunk_record.chunk_id,
                        "asset_pk": self._asset_pk(manifest.parse_id, asset_id),
                        "asset_id": asset_id,
                        "relation_type": "context",
                        "ordinal": int(context_order.get(asset_id, 0)),
                        "metadata": {
                            "page_numbers": list(chunk_record.page_numbers or []),
                            "title_path": list(metadata.get("title_path") or []),
                        },
                    }
                )
        return link_rows

    def _find_document_by_artifact_key(
        self,
        artifact_key: str,
        *,
        tenant_id: str | None = None,
        conn=None,
    ) -> dict[str, Any] | None:
        normalized_tenant = self._normalized_tenant_id(tenant_id)
        own_conn = conn is None
        db = conn or self._connect()
        try:
            with db.cursor() as cur:
                sql = f"""
                    SELECT parse_id, document_id, tenant_id, artifact_key, filename, file_type, parser_engine, created_at, updated_at, manifest
                    FROM {self._qualified(self.documents_table)}
                    WHERE artifact_key = %s
                """
                params: list[Any] = [artifact_key]
                if normalized_tenant:
                    sql += " AND tenant_id = %s"
                    params.append(normalized_tenant)
                cur.execute(sql, params)
                row = cur.fetchone()
            if own_conn:
                db.commit()
        except Exception:
            if own_conn:
                db.rollback()
            raise
        finally:
            if own_conn:
                db.close()
        return self._normalize_row(row) if row else None

    def _upsert_parse_alias(
        self,
        *,
        alias_parse_id: str,
        canonical_parse_id: str,
        tenant_id: str | None,
        document_id: str,
        artifact_key: str,
        metadata: dict[str, Any] | None = None,
        conn=None,
    ) -> None:
        normalized_tenant = self._normalized_tenant_id(tenant_id)
        own_conn = conn is None
        db = conn or self._connect()
        try:
            with db.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._qualified(self.parse_aliases_table)} (
                        alias_parse_id, canonical_parse_id, tenant_id, document_id, artifact_key, metadata, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, NOW(), NOW())
                    ON CONFLICT (alias_parse_id) DO UPDATE SET
                        canonical_parse_id = EXCLUDED.canonical_parse_id,
                        tenant_id = EXCLUDED.tenant_id,
                        document_id = EXCLUDED.document_id,
                        artifact_key = EXCLUDED.artifact_key,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    (
                        alias_parse_id,
                        canonical_parse_id,
                        normalized_tenant,
                        document_id,
                        artifact_key,
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )
            if own_conn:
                db.commit()
        except Exception:
            if own_conn:
                db.rollback()
            raise
        finally:
            if own_conn:
                db.close()

    def _resolve_canonical_parse_id(self, parse_id: str | None, *, tenant_id: str | None = None, conn=None) -> str | None:
        normalized_parse_id = str(parse_id or "").strip()
        if not normalized_parse_id:
            return None
        normalized_tenant = self._normalized_tenant_id(tenant_id)
        own_conn = conn is None
        db = conn or self._connect()
        try:
            with db.cursor() as cur:
                sql = f"""
                    SELECT canonical_parse_id
                    FROM {self._qualified(self.parse_aliases_table)}
                    WHERE alias_parse_id = %s
                """
                params: list[Any] = [normalized_parse_id]
                if normalized_tenant:
                    sql += " AND tenant_id = %s"
                    params.append(normalized_tenant)
                cur.execute(sql, params)
                row = cur.fetchone()
            if own_conn:
                db.commit()
        except Exception:
            if own_conn:
                db.rollback()
            raise
        finally:
            if own_conn:
                db.close()
        canonical_parse_id = str(self._scalar_from_row(row, "canonical_parse_id") or "").strip()
        return canonical_parse_id or normalized_parse_id

    def _connect(self):
        if self._connect_fn is not None:
            return self._connect_fn()
        psycopg, dict_row = _load_psycopg()
        return psycopg.connect(
            self.config.dsn,
            autocommit=False,
            row_factory=dict_row,
            connect_timeout=max(1, int(self.config.connect_timeout)),
        )

    def _upsert_structured_entities(
        self,
        cur,
        *,
        manifest: ParseManifest,
        artifact: ParseArtifact | None,
        chunk_records: list[ChunkExportRecord] | None,
        tenant_id: str,
    ) -> None:
        if artifact is None or not chunk_records:
            return
        chunk_rows = self._build_chunk_payload(
            manifest=manifest,
            chunk_records=chunk_records,
            tenant_id=tenant_id,
        )
        resolved_asset_by_id: dict[str, dict[str, Any]] = {}
        for chunk_record in chunk_records:
            for asset in chunk_record.assets:
                resolved_asset_by_id.setdefault(
                    asset.asset_id,
                    {
                        "download_path": asset.download_path,
                        "resolved_url": asset.resolved_url,
                    },
                )
        asset_by_id = {asset.asset_id: asset for asset in (artifact.assets or [])}
        asset_rows = self._build_asset_payload(
            manifest=manifest,
            asset_by_id=asset_by_id,
            resolved_asset_by_id=resolved_asset_by_id,
            tenant_id=tenant_id,
        )
        link_rows = self._build_chunk_asset_link_payload(
            manifest=manifest,
            chunk_records=chunk_records,
            tenant_id=tenant_id,
        )
        if chunk_rows:
            cur.executemany(
                f"""
                INSERT INTO {self._qualified(self.chunks_table)} (
                    record_id, parse_id, document_id, tenant_id, chunk_id, text, token_count, page_numbers,
                    block_refs, asset_refs, metadata, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, NOW())
                ON CONFLICT (record_id) DO UPDATE SET
                    parse_id = EXCLUDED.parse_id,
                    document_id = EXCLUDED.document_id,
                    tenant_id = EXCLUDED.tenant_id,
                    chunk_id = EXCLUDED.chunk_id,
                    text = EXCLUDED.text,
                    token_count = EXCLUDED.token_count,
                    page_numbers = EXCLUDED.page_numbers,
                    block_refs = EXCLUDED.block_refs,
                    asset_refs = EXCLUDED.asset_refs,
                    metadata = EXCLUDED.metadata,
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
                """,
                [
                    (
                        row["record_id"],
                        row["parse_id"],
                        row["document_id"],
                        row["tenant_id"],
                        row["chunk_id"],
                        row["text"],
                        int(row["token_count"]),
                        json.dumps(row["page_numbers"], ensure_ascii=False),
                        json.dumps(row["block_refs"], ensure_ascii=False),
                        json.dumps(row["asset_refs"], ensure_ascii=False),
                        json.dumps(row["metadata"], ensure_ascii=False),
                        json.dumps(row["payload"], ensure_ascii=False),
                    )
                    for row in chunk_rows
                ],
            )
        if asset_rows:
            cur.executemany(
                f"""
                INSERT INTO {self._qualified(self.assets_table)} (
                    asset_pk, parse_id, document_id, tenant_id, asset_id, asset_type, title, text,
                    page_numbers, positions, width, height, sha256, storage_backend, storage_relative_path,
                    storage_absolute_path, storage_download_path, storage_resolved_url, storage_source_url,
                    media_type, metadata, payload, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s::jsonb, %s::jsonb, NOW()
                )
                ON CONFLICT (asset_pk) DO UPDATE SET
                    parse_id = EXCLUDED.parse_id,
                    document_id = EXCLUDED.document_id,
                    tenant_id = EXCLUDED.tenant_id,
                    asset_id = EXCLUDED.asset_id,
                    asset_type = EXCLUDED.asset_type,
                    title = EXCLUDED.title,
                    text = EXCLUDED.text,
                    page_numbers = EXCLUDED.page_numbers,
                    positions = EXCLUDED.positions,
                    width = EXCLUDED.width,
                    height = EXCLUDED.height,
                    sha256 = EXCLUDED.sha256,
                    storage_backend = EXCLUDED.storage_backend,
                    storage_relative_path = EXCLUDED.storage_relative_path,
                    storage_absolute_path = EXCLUDED.storage_absolute_path,
                    storage_download_path = EXCLUDED.storage_download_path,
                    storage_resolved_url = EXCLUDED.storage_resolved_url,
                    storage_source_url = EXCLUDED.storage_source_url,
                    media_type = EXCLUDED.media_type,
                    metadata = EXCLUDED.metadata,
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
                """,
                [
                    (
                        row["asset_pk"],
                        row["parse_id"],
                        row["document_id"],
                        row["tenant_id"],
                        row["asset_id"],
                        row["asset_type"],
                        row["title"],
                        row["text"],
                        json.dumps(row["page_numbers"], ensure_ascii=False),
                        json.dumps(row["positions"], ensure_ascii=False),
                        row["width"],
                        row["height"],
                        row["sha256"],
                        row["storage_backend"],
                        row["storage_relative_path"],
                        row["storage_absolute_path"],
                        row["storage_download_path"],
                        row["storage_resolved_url"],
                        row["storage_source_url"],
                        row["media_type"],
                        json.dumps(row["metadata"], ensure_ascii=False),
                        json.dumps(row["payload"], ensure_ascii=False),
                    )
                    for row in asset_rows
                ],
            )
        if link_rows:
            cur.executemany(
                f"""
                INSERT INTO {self._qualified(self.chunk_asset_links_table)} (
                    link_id, parse_id, document_id, tenant_id, record_id, chunk_id, asset_pk, asset_id,
                    relation_type, ordinal, metadata, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
                ON CONFLICT (link_id) DO UPDATE SET
                    parse_id = EXCLUDED.parse_id,
                    document_id = EXCLUDED.document_id,
                    tenant_id = EXCLUDED.tenant_id,
                    record_id = EXCLUDED.record_id,
                    chunk_id = EXCLUDED.chunk_id,
                    asset_pk = EXCLUDED.asset_pk,
                    asset_id = EXCLUDED.asset_id,
                    relation_type = EXCLUDED.relation_type,
                    ordinal = EXCLUDED.ordinal,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                [
                    (
                        row["link_id"],
                        row["parse_id"],
                        row["document_id"],
                        row["tenant_id"],
                        row["record_id"],
                        row["chunk_id"],
                        row["asset_pk"],
                        row["asset_id"],
                        row["relation_type"],
                        int(row["ordinal"]),
                        json.dumps(row["metadata"], ensure_ascii=False),
                    )
                    for row in link_rows
                ],
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
                    CREATE TABLE IF NOT EXISTS {self._qualified(self.documents_table)} (
                        parse_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL DEFAULT '',
                        artifact_key TEXT NOT NULL UNIQUE,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        parser_engine TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        manifest JSONB NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._qualified(self.records_table)} (
                        record_id TEXT PRIMARY KEY,
                        parse_id TEXT NOT NULL REFERENCES {self._qualified(self.documents_table)}(parse_id) ON DELETE CASCADE,
                        document_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL DEFAULT '',
                        chunk_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        page_numbers JSONB NOT NULL DEFAULT '[]'::jsonb,
                        asset_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                        asset_types JSONB NOT NULL DEFAULT '[]'::jsonb,
                        asset_urls JSONB NOT NULL DEFAULT '[]'::jsonb,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        payload JSONB NOT NULL,
                        search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._qualified(self.chunks_table)} (
                        record_id TEXT PRIMARY KEY REFERENCES {self._qualified(self.records_table)}(record_id) ON DELETE CASCADE,
                        parse_id TEXT NOT NULL REFERENCES {self._qualified(self.documents_table)}(parse_id) ON DELETE CASCADE,
                        document_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL DEFAULT '',
                        chunk_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        page_numbers JSONB NOT NULL DEFAULT '[]'::jsonb,
                        block_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                        asset_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        payload JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (parse_id, chunk_id)
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._qualified(self.assets_table)} (
                        asset_pk TEXT PRIMARY KEY,
                        parse_id TEXT NOT NULL REFERENCES {self._qualified(self.documents_table)}(parse_id) ON DELETE CASCADE,
                        document_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL DEFAULT '',
                        asset_id TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        title TEXT,
                        text TEXT NOT NULL DEFAULT '',
                        page_numbers JSONB NOT NULL DEFAULT '[]'::jsonb,
                        positions JSONB NOT NULL DEFAULT '[]'::jsonb,
                        width INTEGER,
                        height INTEGER,
                        sha256 TEXT,
                        storage_backend TEXT,
                        storage_relative_path TEXT,
                        storage_absolute_path TEXT,
                        storage_download_path TEXT,
                        storage_resolved_url TEXT,
                        storage_source_url TEXT,
                        media_type TEXT,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        payload JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (parse_id, asset_id)
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._qualified(self.chunk_asset_links_table)} (
                        link_id TEXT PRIMARY KEY,
                        parse_id TEXT NOT NULL REFERENCES {self._qualified(self.documents_table)}(parse_id) ON DELETE CASCADE,
                        document_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL DEFAULT '',
                        record_id TEXT NOT NULL REFERENCES {self._qualified(self.chunks_table)}(record_id) ON DELETE CASCADE,
                        chunk_id TEXT NOT NULL,
                        asset_pk TEXT NOT NULL REFERENCES {self._qualified(self.assets_table)}(asset_pk) ON DELETE CASCADE,
                        asset_id TEXT NOT NULL,
                        relation_type TEXT NOT NULL,
                        ordinal INTEGER NOT NULL DEFAULT 0,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (record_id, asset_id, relation_type)
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._qualified(self.parse_aliases_table)} (
                        alias_parse_id TEXT PRIMARY KEY,
                        canonical_parse_id TEXT NOT NULL REFERENCES {self._qualified(self.documents_table)}(parse_id) ON DELETE CASCADE,
                        tenant_id TEXT NOT NULL DEFAULT '',
                        document_id TEXT NOT NULL,
                        artifact_key TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    f"""
                    ALTER TABLE {self._qualified(self.documents_table)}
                    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    f"""
                    ALTER TABLE {self._qualified(self.records_table)}
                    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    f"""
                    ALTER TABLE {self._qualified(self.chunks_table)}
                    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    f"""
                    ALTER TABLE {self._qualified(self.assets_table)}
                    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    f"""
                    ALTER TABLE {self._qualified(self.chunk_asset_links_table)}
                    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.documents_table}_document_id
                    ON {self._qualified(self.documents_table)} (document_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.documents_table}_tenant_id
                    ON {self._qualified(self.documents_table)} (tenant_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.records_table}_parse_id
                    ON {self._qualified(self.records_table)} (parse_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.parse_aliases_table}_canonical_parse_id
                    ON {self._qualified(self.parse_aliases_table)} (canonical_parse_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.parse_aliases_table}_artifact_key
                    ON {self._qualified(self.parse_aliases_table)} (artifact_key)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.parse_aliases_table}_tenant_id
                    ON {self._qualified(self.parse_aliases_table)} (tenant_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.records_table}_tenant_id
                    ON {self._qualified(self.records_table)} (tenant_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.records_table}_document_id
                    ON {self._qualified(self.records_table)} (document_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.chunks_table}_parse_id
                    ON {self._qualified(self.chunks_table)} (parse_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.chunks_table}_document_id
                    ON {self._qualified(self.chunks_table)} (document_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.chunks_table}_tenant_id
                    ON {self._qualified(self.chunks_table)} (tenant_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.assets_table}_parse_id
                    ON {self._qualified(self.assets_table)} (parse_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.assets_table}_document_id
                    ON {self._qualified(self.assets_table)} (document_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.assets_table}_tenant_id
                    ON {self._qualified(self.assets_table)} (tenant_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.assets_table}_asset_type
                    ON {self._qualified(self.assets_table)} (asset_type)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.chunk_asset_links_table}_parse_id
                    ON {self._qualified(self.chunk_asset_links_table)} (parse_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.chunk_asset_links_table}_document_id
                    ON {self._qualified(self.chunk_asset_links_table)} (document_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.chunk_asset_links_table}_tenant_id
                    ON {self._qualified(self.chunk_asset_links_table)} (tenant_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.chunk_asset_links_table}_record_id
                    ON {self._qualified(self.chunk_asset_links_table)} (record_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.chunk_asset_links_table}_asset_pk
                    ON {self._qualified(self.chunk_asset_links_table)} (asset_pk)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.chunk_asset_links_table}_relation_type
                    ON {self._qualified(self.chunk_asset_links_table)} (relation_type)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.schema}_{self.records_table}_search_vector
                    ON {self._qualified(self.records_table)} USING GIN (search_vector)
                    """
                )
            conn.commit()
            self._schema_ready = True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class PostgresIngestPublisher(_PostgresBackendBase, IngestPublisher):
    sink_type = "postgres"

    def publish(
        self,
        manifest: ParseManifest,
        records: list[IngestExportRecord],
        *,
        artifact: ParseArtifact | None = None,
        chunk_records: list[ChunkExportRecord] | None = None,
    ) -> IngestPublishResult:
        self.ensure_schema()
        tenant_id = self._tenant_id_from_manifest(manifest)
        existing_document = self._find_document_by_artifact_key(manifest.artifact_key, tenant_id=tenant_id)
        if existing_document is not None:
            canonical_parse_id = str(existing_document.get("parse_id") or "").strip()
            if canonical_parse_id and canonical_parse_id != manifest.parse_id:
                self._upsert_parse_alias(
                    alias_parse_id=manifest.parse_id,
                    canonical_parse_id=canonical_parse_id,
                    tenant_id=tenant_id,
                    document_id=manifest.document_id,
                    artifact_key=manifest.artifact_key,
                    metadata={
                        "deduplicated": True,
                        "canonical_parse_id": canonical_parse_id,
                        "artifact_key": manifest.artifact_key,
                    },
                )
                return IngestPublishResult(
                    enabled=True,
                    sink_type=self.sink_type,
                    status="published",
                    record_count=len(records),
                    destination=_redact_dsn(self.config.dsn),
                    metadata={
                        "deduplicated": True,
                        "canonical_parse_id": canonical_parse_id,
                        "alias_parse_id": manifest.parse_id,
                        "artifact_key": manifest.artifact_key,
                    },
                )
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                manifest_payload = manifest.model_dump(mode="json")
                cur.execute(
                    f"""
                    INSERT INTO {self._qualified(self.documents_table)} (
                        parse_id, document_id, tenant_id, artifact_key, filename, file_type, parser_engine, created_at, updated_at, manifest, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::timestamptz, NOW(), %s::jsonb, %s::jsonb)
                    ON CONFLICT (parse_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        tenant_id = EXCLUDED.tenant_id,
                        artifact_key = EXCLUDED.artifact_key,
                        filename = EXCLUDED.filename,
                        file_type = EXCLUDED.file_type,
                        parser_engine = EXCLUDED.parser_engine,
                        updated_at = NOW(),
                        manifest = EXCLUDED.manifest,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        manifest.parse_id,
                        manifest.document_id,
                        tenant_id,
                        manifest.artifact_key,
                        manifest.filename,
                        manifest.file_type,
                        manifest.parser_engine,
                        manifest.created_at,
                        json.dumps(manifest_payload, ensure_ascii=False),
                        json.dumps(manifest.metadata or {}, ensure_ascii=False),
                    ),
                )
                cur.executemany(
                    f"""
                    INSERT INTO {self._qualified(self.records_table)} (
                        record_id, parse_id, document_id, tenant_id, chunk_id, text, token_count, page_numbers,
                        asset_refs, asset_types, asset_urls, metadata, payload, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, NOW())
                    ON CONFLICT (record_id) DO UPDATE SET
                        parse_id = EXCLUDED.parse_id,
                        document_id = EXCLUDED.document_id,
                        tenant_id = EXCLUDED.tenant_id,
                        chunk_id = EXCLUDED.chunk_id,
                        text = EXCLUDED.text,
                        token_count = EXCLUDED.token_count,
                        page_numbers = EXCLUDED.page_numbers,
                        asset_refs = EXCLUDED.asset_refs,
                        asset_types = EXCLUDED.asset_types,
                        asset_urls = EXCLUDED.asset_urls,
                        metadata = EXCLUDED.metadata,
                        payload = EXCLUDED.payload,
                        updated_at = NOW()
                    """,
                    [
                        (
                            record.record_id,
                            record.parse_id,
                            record.document_id,
                            tenant_id,
                            record.chunk_id,
                            record.text,
                            int(record.token_count),
                            json.dumps(record.page_numbers, ensure_ascii=False),
                            json.dumps(record.asset_refs, ensure_ascii=False),
                            json.dumps(record.asset_types, ensure_ascii=False),
                            json.dumps(record.asset_urls, ensure_ascii=False),
                            json.dumps(record.metadata or {}, ensure_ascii=False),
                            json.dumps(record.model_dump(mode="json"), ensure_ascii=False),
                        )
                        for record in records
                    ],
                )
                self._upsert_structured_entities(
                    cur,
                    manifest=manifest,
                    artifact=artifact,
                    chunk_records=chunk_records,
                    tenant_id=tenant_id,
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return IngestPublishResult(
            enabled=True,
            sink_type=self.sink_type,
            status="published",
            record_count=len(records),
            destination=_redact_dsn(self.config.dsn),
            metadata={
                "schema": self.schema,
                "documents_table": f"{self.schema}.{self.documents_table}",
                "records_table": f"{self.schema}.{self.records_table}",
                "chunks_table": f"{self.schema}.{self.chunks_table}",
                "assets_table": f"{self.schema}.{self.assets_table}",
                "chunk_asset_links_table": f"{self.schema}.{self.chunk_asset_links_table}",
                "structured_published": artifact is not None and chunk_records is not None,
            },
        )


class PostgresIngestStore(_PostgresBackendBase):
    def check_health(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database() AS database_name")
                row = cur.fetchone()
                database_name = self._normalize_row(row).get("database_name") if row else None
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "status": "ok",
            "schema": self.schema,
            "database": database_name,
        }

    @staticmethod
    def _coerce_count(value: Any) -> int:
        if value is None:
            return 0
        return int(value)

    def _build_document_filter_parts(
        self,
        *,
        tenant_id: str | None = None,
        parse_id: str | None = None,
        document_id: str | None = None,
        parser_engine: str | None = None,
        file_type: str | None = None,
        alias: str = "d",
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        normalized_tenant = self._normalized_tenant_id(tenant_id)
        canonical_parse_id = self._resolve_canonical_parse_id(parse_id, tenant_id=tenant_id) if parse_id else None
        if normalized_tenant:
            clauses.append(f"{alias}.tenant_id = %s")
            params.append(normalized_tenant)
        if canonical_parse_id:
            clauses.append(f"{alias}.parse_id = %s")
            params.append(canonical_parse_id)
        if document_id:
            clauses.append(f"{alias}.document_id = %s")
            params.append(document_id)
        if parser_engine:
            clauses.append(f"{alias}.parser_engine = %s")
            params.append(parser_engine)
        if file_type:
            clauses.append(f"{alias}.file_type = %s")
            params.append(file_type)
        return clauses, params

    def get_stats(
        self,
        *,
        tenant_id: str | None = None,
        parse_id: str | None = None,
        document_id: str | None = None,
        parser_engine: str | None = None,
        file_type: str | None = None,
        include_breakdown: bool = True,
    ) -> dict[str, Any]:
        self.ensure_schema()
        document_filters, document_params = self._build_document_filter_parts(
            tenant_id=tenant_id,
            parse_id=parse_id,
            document_id=document_id,
            parser_engine=parser_engine,
            file_type=file_type,
        )
        document_where_sql = f"WHERE {' AND '.join(document_filters)}" if document_filters else ""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) AS total_documents,
                        COUNT(*) FILTER (
                            WHERE COALESCE(d.manifest->'metadata'->'ingest_publish'->>'status', '') = 'published'
                        ) AS published_documents,
                        COUNT(*) FILTER (
                            WHERE COALESCE(d.manifest->'metadata'->'ingest_publish'->>'status', '') = 'failed'
                        ) AS failed_documents,
                        COUNT(*) FILTER (
                            WHERE COALESCE(d.manifest->'metadata'->'ingest_publish'->>'status', '') = 'disabled'
                        ) AS disabled_documents
                    FROM {self._qualified(self.documents_table)} d
                    {document_where_sql}
                    """,
                    list(document_params),
                )
                document_stats_row = self._normalize_row(cur.fetchone() or {})
                cur.execute(
                    f"""
                    SELECT
                        COUNT(r.record_id) AS total_records,
                        COUNT(*) FILTER (
                            WHERE r.record_id IS NOT NULL
                            AND jsonb_array_length(COALESCE(r.asset_refs, '[]'::jsonb)) > 0
                        ) AS records_with_assets,
                        COUNT(*) FILTER (
                            WHERE r.record_id IS NOT NULL
                            AND jsonb_array_length(COALESCE(r.asset_urls, '[]'::jsonb)) > 0
                        ) AS records_with_asset_urls
                    FROM {self._qualified(self.documents_table)} d
                    LEFT JOIN {self._qualified(self.records_table)} r ON r.parse_id = d.parse_id
                    {document_where_sql}
                    """,
                    list(document_params),
                )
                record_stats_row = self._normalize_row(cur.fetchone() or {})
                cur.execute(
                    f"""
                    SELECT
                        COUNT(a.asset_pk) AS total_assets,
                        COUNT(*) FILTER (
                            WHERE a.asset_pk IS NOT NULL
                            AND (
                                COALESCE(a.storage_download_path, '') <> ''
                                OR COALESCE(a.storage_source_url, '') <> ''
                                OR COALESCE(a.storage_resolved_url, '') <> ''
                            )
                        ) AS assets_with_urls,
                        COUNT(*) FILTER (
                            WHERE a.asset_pk IS NOT NULL
                            AND COALESCE(a.storage_relative_path, '') <> ''
                        ) AS assets_materialized
                    FROM {self._qualified(self.documents_table)} d
                    LEFT JOIN {self._qualified(self.assets_table)} a ON a.parse_id = d.parse_id
                    {document_where_sql}
                    """,
                    list(document_params),
                )
                asset_stats_row = self._normalize_row(cur.fetchone() or {})
                cur.execute(
                    f"""
                    SELECT
                        COUNT(l.link_id) AS total_links,
                        COUNT(*) FILTER (
                            WHERE l.link_id IS NOT NULL AND COALESCE(l.relation_type, '') = 'direct'
                        ) AS direct_links,
                        COUNT(*) FILTER (
                            WHERE l.link_id IS NOT NULL AND COALESCE(l.relation_type, '') = 'context'
                        ) AS context_links
                    FROM {self._qualified(self.documents_table)} d
                    LEFT JOIN {self._qualified(self.chunk_asset_links_table)} l ON l.parse_id = d.parse_id
                    {document_where_sql}
                    """,
                    list(document_params),
                )
                link_stats_row = self._normalize_row(cur.fetchone() or {})
                parser_engine_breakdown: list[dict[str, Any]] = []
                file_type_breakdown: list[dict[str, Any]] = []
                if include_breakdown:
                    cur.execute(
                        f"""
                        SELECT
                            d.parser_engine AS parser_engine,
                            COUNT(DISTINCT d.parse_id) AS document_count,
                            COUNT(r.record_id) AS record_count,
                            COUNT(*) FILTER (
                                WHERE r.record_id IS NOT NULL
                                AND jsonb_array_length(COALESCE(r.asset_refs, '[]'::jsonb)) > 0
                            ) AS records_with_assets
                        FROM {self._qualified(self.documents_table)} d
                        LEFT JOIN {self._qualified(self.records_table)} r ON r.parse_id = d.parse_id
                        {document_where_sql}
                        GROUP BY d.parser_engine
                        ORDER BY document_count DESC, record_count DESC, d.parser_engine ASC
                        """,
                        list(document_params),
                    )
                    parser_engine_breakdown = [self._normalize_row(row) for row in cur.fetchall()]
                    cur.execute(
                        f"""
                        SELECT
                            d.file_type AS file_type,
                            COUNT(DISTINCT d.parse_id) AS document_count,
                            COUNT(r.record_id) AS record_count,
                            COUNT(*) FILTER (
                                WHERE r.record_id IS NOT NULL
                                AND jsonb_array_length(COALESCE(r.asset_refs, '[]'::jsonb)) > 0
                            ) AS records_with_assets
                        FROM {self._qualified(self.documents_table)} d
                        LEFT JOIN {self._qualified(self.records_table)} r ON r.parse_id = d.parse_id
                        {document_where_sql}
                        GROUP BY d.file_type
                        ORDER BY document_count DESC, record_count DESC, d.file_type ASC
                        """,
                        list(document_params),
                    )
                    file_type_breakdown = [self._normalize_row(row) for row in cur.fetchall()]
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        document_stats = {
            "total": self._coerce_count(document_stats_row.get("total_documents")),
            "published": self._coerce_count(document_stats_row.get("published_documents")),
            "failed": self._coerce_count(document_stats_row.get("failed_documents")),
            "disabled": self._coerce_count(document_stats_row.get("disabled_documents")),
        }
        record_stats = {
            "total": self._coerce_count(record_stats_row.get("total_records")),
            "with_assets": self._coerce_count(record_stats_row.get("records_with_assets")),
            "with_asset_urls": self._coerce_count(record_stats_row.get("records_with_asset_urls")),
        }
        asset_stats = {
            "total": self._coerce_count(asset_stats_row.get("total_assets")),
            "with_urls": self._coerce_count(asset_stats_row.get("assets_with_urls")),
            "materialized": self._coerce_count(asset_stats_row.get("assets_materialized")),
        }
        link_stats = {
            "total": self._coerce_count(link_stats_row.get("total_links")),
            "direct": self._coerce_count(link_stats_row.get("direct_links")),
            "context": self._coerce_count(link_stats_row.get("context_links")),
        }
        return {
            "status": "ok",
            "schema": self.schema,
            "filters": {
                "tenant_id": tenant_id,
                "parse_id": parse_id,
                "document_id": document_id,
                "parser_engine": parser_engine,
                "file_type": file_type,
            },
            "documents": document_stats,
            "records": record_stats,
            "assets": asset_stats,
            "chunk_asset_links": link_stats,
            "breakdown": {
                "parser_engines": parser_engine_breakdown if include_breakdown else [],
                "file_types": file_type_breakdown if include_breakdown else [],
            },
        }

    def list_documents(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        tenant_id: str | None = None,
        parser_engine: str | None = None,
        file_type: str | None = None,
        publish_status: str | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        where_clauses: list[str] = []
        params: list[Any] = []
        normalized_tenant = self._normalized_tenant_id(tenant_id)
        if normalized_tenant:
            where_clauses.append("tenant_id = %s")
            params.append(normalized_tenant)
        if parser_engine:
            where_clauses.append("parser_engine = %s")
            params.append(parser_engine)
        if file_type:
            where_clauses.append("file_type = %s")
            params.append(file_type)
        if publish_status:
            where_clauses.append("COALESCE(manifest->'metadata'->'ingest_publish'->>'status', '') = %s")
            params.append(publish_status)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = f"""
            SELECT parse_id, document_id, tenant_id, artifact_key, filename, file_type, parser_engine, created_at, updated_at, manifest
            FROM {self._qualified(self.documents_table)}
            {where_sql}
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
        """
        params.extend([max(1, int(limit)), max(0, int(offset))])
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return [self._normalize_row(row) for row in rows]

    def get_document(self, parse_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
        self.ensure_schema()
        normalized_tenant = self._normalized_tenant_id(tenant_id)
        canonical_parse_id = self._resolve_canonical_parse_id(parse_id, tenant_id=tenant_id)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = f"SELECT manifest FROM {self._qualified(self.documents_table)} WHERE parse_id = %s"
                params: list[Any] = [canonical_parse_id]
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
        if not row:
            return None
        normalized = self._normalize_row(row)
        payload = self._maybe_json(normalized.get("manifest"))
        if isinstance(payload, dict) and canonical_parse_id and canonical_parse_id != parse_id:
            metadata = dict(payload.get("metadata") or {})
            metadata.update(
                {
                    "alias_parse_id": parse_id,
                    "canonical_parse_id": canonical_parse_id,
                }
            )
            payload["metadata"] = metadata
        return payload

    def list_assets(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        tenant_id: str | None = None,
        parse_id: str | None = None,
        document_id: str | None = None,
        asset_type: str | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        where_clauses, params = self._build_document_filter_parts(
            tenant_id=tenant_id,
            parse_id=parse_id,
            document_id=document_id,
            alias="a",
        )
        if asset_type:
            where_clauses.append("a.asset_type = %s")
            params.append(asset_type)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = f"""
            SELECT a.payload
            FROM {self._qualified(self.assets_table)} a
            {where_sql}
            ORDER BY a.updated_at DESC, a.asset_id ASC
            LIMIT %s OFFSET %s
        """
        params.extend([max(1, int(limit)), max(0, int(offset))])
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        results: list[dict[str, Any]] = []
        canonical_parse_id = self._resolve_canonical_parse_id(parse_id, tenant_id=tenant_id) if parse_id else None
        for row in rows:
            payload = self._maybe_json(self._normalize_row(row).get("payload")) or {}
            if isinstance(payload, dict):
                results.append(
                    self._annotate_alias_payload(
                        payload,
                        requested_parse_id=str(parse_id or "").strip(),
                        canonical_parse_id=str(canonical_parse_id or payload.get("parse_id") or "").strip(),
                    )
                )
        return results

    def get_asset(self, parse_id: str, asset_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
        self.ensure_schema()
        normalized_tenant = self._normalized_tenant_id(tenant_id)
        canonical_parse_id = self._resolve_canonical_parse_id(parse_id, tenant_id=tenant_id)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                sql = f"""
                    SELECT payload
                    FROM {self._qualified(self.assets_table)}
                    WHERE parse_id = %s AND asset_id = %s
                """
                params: list[Any] = [canonical_parse_id, asset_id]
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
        if not row:
            return None
        payload = self._maybe_json(self._normalize_row(row).get("payload")) or {}
        if isinstance(payload, dict):
            return self._annotate_alias_payload(
                payload,
                requested_parse_id=parse_id,
                canonical_parse_id=str(canonical_parse_id or parse_id),
            )
        return None

    def list_chunks(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        tenant_id: str | None = None,
        parse_id: str | None = None,
        document_id: str | None = None,
        chunk_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        where_clauses, params = self._build_document_filter_parts(
            tenant_id=tenant_id,
            parse_id=parse_id,
            document_id=document_id,
            alias="c",
        )
        if chunk_id:
            where_clauses.append("c.chunk_id = %s")
            params.append(chunk_id)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = f"""
            SELECT c.payload
            FROM {self._qualified(self.chunks_table)} c
            {where_sql}
            ORDER BY c.updated_at DESC, c.chunk_id ASC
            LIMIT %s OFFSET %s
        """
        params.extend([max(1, int(limit)), max(0, int(offset))])
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        requested_parse_id = str(parse_id or "").strip()
        canonical_parse_id = self._resolve_canonical_parse_id(parse_id, tenant_id=tenant_id) if parse_id else None
        results: list[dict[str, Any]] = []
        for row in rows:
            payload = self._maybe_json(self._normalize_row(row).get("payload")) or {}
            if isinstance(payload, dict):
                results.append(
                    self._annotate_alias_payload(
                        payload,
                        requested_parse_id=requested_parse_id,
                        canonical_parse_id=str(canonical_parse_id or payload.get("parse_id") or "").strip(),
                    )
                )
        return results

    def list_chunk_asset_links(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        tenant_id: str | None = None,
        parse_id: str | None = None,
        document_id: str | None = None,
        chunk_id: str | None = None,
        asset_id: str | None = None,
        relation_type: str | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        where_clauses, params = self._build_document_filter_parts(
            tenant_id=tenant_id,
            parse_id=parse_id,
            document_id=document_id,
            alias="l",
        )
        if chunk_id:
            where_clauses.append("l.chunk_id = %s")
            params.append(chunk_id)
        if asset_id:
            where_clauses.append("l.asset_id = %s")
            params.append(asset_id)
        if relation_type:
            where_clauses.append("l.relation_type = %s")
            params.append(relation_type)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = f"""
            SELECT
                l.link_id,
                l.parse_id,
                l.document_id,
                l.tenant_id,
                l.record_id,
                l.chunk_id,
                l.asset_id,
                l.relation_type,
                l.ordinal,
                l.metadata,
                a.asset_type,
                a.title,
                a.storage_download_path,
                a.storage_resolved_url
            FROM {self._qualified(self.chunk_asset_links_table)} l
            LEFT JOIN {self._qualified(self.assets_table)} a
                ON a.asset_pk = l.asset_pk
            {where_sql}
            ORDER BY l.parse_id DESC, l.chunk_id ASC, l.relation_type ASC, l.ordinal ASC
            LIMIT %s OFFSET %s
        """
        params.extend([max(1, int(limit)), max(0, int(offset))])
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        requested_parse_id = str(parse_id or "").strip()
        canonical_parse_id = self._resolve_canonical_parse_id(parse_id, tenant_id=tenant_id) if parse_id else None
        results: list[dict[str, Any]] = []
        for row in rows:
            normalized = self._normalize_row(row)
            payload = {
                "link_id": normalized.get("link_id"),
                "parse_id": normalized.get("parse_id"),
                "document_id": normalized.get("document_id"),
                "record_id": normalized.get("record_id"),
                "chunk_id": normalized.get("chunk_id"),
                "asset_id": normalized.get("asset_id"),
                "asset_type": normalized.get("asset_type"),
                "asset_title": normalized.get("title"),
                "relation_type": normalized.get("relation_type"),
                "ordinal": int(normalized.get("ordinal") or 0),
                "download_path": normalized.get("storage_download_path"),
                "resolved_url": normalized.get("storage_resolved_url"),
                "metadata": self._maybe_json(normalized.get("metadata")) or {},
            }
            results.append(
                self._annotate_alias_payload(
                    payload,
                    requested_parse_id=requested_parse_id,
                    canonical_parse_id=str(canonical_parse_id or payload.get("parse_id") or "").strip(),
                )
            )
        return results

    def search_records(
        self,
        *,
        query: str | None = None,
        tenant_id: str | None = None,
        parse_id: str | None = None,
        document_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
        mode: str = "text",
    ) -> list[dict[str, Any]]:
        normalized_mode = (mode or "text").strip().lower()
        if normalized_mode != "text":
            raise ValueError("mode must be text")
        return self._search_text_records(
            query=query,
            tenant_id=tenant_id,
            parse_id=parse_id,
            document_id=document_id,
            limit=limit,
            offset=offset,
        )

    def _search_text_records(
        self,
        *,
        query: str | None,
        tenant_id: str | None,
        parse_id: str | None,
        document_id: str | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        where_clauses: list[str] = []
        params: list[Any] = []
        order_sql = "ORDER BY updated_at DESC"
        select_sql = "SELECT payload, NULL::float AS score"
        normalized_tenant = self._normalized_tenant_id(tenant_id)
        canonical_parse_id = self._resolve_canonical_parse_id(parse_id, tenant_id=tenant_id) if parse_id else None
        if normalized_tenant:
            where_clauses.append("tenant_id = %s")
            params.append(normalized_tenant)
        if canonical_parse_id:
            where_clauses.append("parse_id = %s")
            params.append(canonical_parse_id)
        if document_id:
            where_clauses.append("document_id = %s")
            params.append(document_id)
        if query:
            where_clauses.append("search_vector @@ plainto_tsquery('simple', %s)")
            params.append(query)
            select_sql = (
                "SELECT payload, ts_rank_cd(search_vector, plainto_tsquery('simple', %s)) AS score"
            )
            params.insert(0, query)
            order_sql = "ORDER BY score DESC NULLS LAST, updated_at DESC"
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = f"""
            {select_sql}
            FROM {self._qualified(self.records_table)}
            {where_sql}
            {order_sql}
            LIMIT %s OFFSET %s
        """
        params.extend([max(1, int(limit)), max(0, int(offset))])
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        results: list[dict[str, Any]] = []
        for row in rows:
            normalized = self._normalize_row(row)
            payload = self._maybe_json(normalized.get("payload")) or {}
            if normalized.get("score") is not None:
                payload["score"] = float(normalized["score"])
            results.append(payload)
        return results

def load_postgres_ingest_config_from_env() -> PostgresIngestConfig:
    dsn = (os.environ.get("DEEPDOC_INGEST_PG_DSN") or "").strip()
    if not dsn:
        raise RuntimeError("DEEPDOC_INGEST_PG_DSN is required when using PostgreSQL ingest backend")
    schema = (os.environ.get("DEEPDOC_INGEST_PG_SCHEMA") or "deepdoc_ingest").strip()
    connect_timeout = int(os.environ.get("DEEPDOC_INGEST_PG_CONNECT_TIMEOUT", "10"))
    return PostgresIngestConfig(
        dsn=dsn,
        schema=schema,
        connect_timeout=connect_timeout,
    )

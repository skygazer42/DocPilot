import hashlib
import json
import mimetypes
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, Literal
from uuid import uuid4

import requests
import tiktoken
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from common import logger
from common import setting

SCHEMA_VERSION = "1.0.0"
ARTIFACT_PROFILE_VERSION = "2026-06-06-structure-aware-v2"
CHUNK_EXPORT_SCHEMA_VERSION = "2026-06-08.chunk.v1"
INGEST_EXPORT_SCHEMA_VERSION = "2026-06-08.ingest.v1"
ASSET_CONTEXT_SCHEMA_VERSION = "2026-06-08.asset-context.v1"
ASSET_SUMMARY_SCHEMA_VERSION = "2026-06-08.asset-summary.v1"
DEFAULT_CHUNK_MAX_TOKENS = 800
DEFAULT_CHUNK_OVERLAP_TOKENS = 120
DEFAULT_CHUNK_STRATEGY = "structure_aware"
CHUNK_STRATEGY_LABELS = {
    "structure_aware": "structure_aware_v2",
    "page_aware": "page_aware_v1",
    "asset_aware": "asset_aware_v1",
}
CHUNK_STRATEGIES = set(CHUNK_STRATEGY_LABELS)
DEFAULT_TOKEN_ENCODING = "cl100k_base"
DEFAULT_CHUNK_CONTEXT_WINDOW = 2
DEFAULT_CHUNK_TITLE_PATH_DEPTH = 3
DEFAULT_CHUNK_ASSET_SUMMARY_LENGTH = 160


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str, default: str = "document") -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "-", (value or "").strip())
    text = text.strip("-.")
    return text or default


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_dumps_stable(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def normalize_chunk_strategy(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in CHUNK_STRATEGIES:
        return normalized
    return DEFAULT_CHUNK_STRATEGY


def _truncate_text(value: str, max_chars: int) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= max_chars:
        return normalized
    truncated = normalized[:max_chars].rsplit(" ", 1)[0].rstrip(" ,.;:，。；：")
    return (truncated or normalized[:max_chars]).rstrip() + "..."


def _normalized_tenant_id(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _manifest_tenant_id(manifest: "ParseManifest") -> str | None:
    return _normalized_tenant_id((manifest.metadata or {}).get("tenant_id"))


def _load_token_encoder():
    encoding_name = os.environ.get("DEEPDOC_CHUNK_ENCODING", DEFAULT_TOKEN_ENCODING)
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        logger.warning("Unknown tiktoken encoding %s, fallback to %s", encoding_name, DEFAULT_TOKEN_ENCODING)
        return tiktoken.get_encoding(DEFAULT_TOKEN_ENCODING)


_TOKEN_ENCODER = _load_token_encoder()


def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_TOKEN_ENCODER.encode(text))
    except Exception:
        return len(text)


class ParsePosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int
    left: float
    right: float
    top: float
    bottom: float


class AssetStorage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["local", "remote"]
    relative_path: str
    absolute_path: str
    download_path: str
    media_type: str
    source_url: str | None = None


class ParseAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    asset_type: Literal["figure", "table", "seal", "equation", "image", "barcode"]
    title: str | None = None
    text: str = ""
    page_numbers: list[int] = Field(default_factory=list)
    positions: list[ParsePosition] = Field(default_factory=list)
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    storage: AssetStorage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_id: str
    block_type: Literal[
        "text",
        "title",
        "table",
        "figure",
        "equation",
        "seal",
        "barcode",
        "list",
        "reference",
        "unknown",
    ]
    text: str
    page_numbers: list[int] = Field(default_factory=list)
    positions: list[ParsePosition] = Field(default_factory=list)
    token_count: int = 0
    asset_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    token_count: int
    page_numbers: list[int] = Field(default_factory=list)
    block_refs: list[str] = Field(default_factory=list)
    asset_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkAssetView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    asset_type: Literal["figure", "table", "seal", "equation", "image", "barcode"]
    title: str | None = None
    text: str = ""
    page_numbers: list[int] = Field(default_factory=list)
    positions: list[ParsePosition] = Field(default_factory=list)
    download_path: str | None = None
    resolved_url: str | None = None
    media_type: str | None = None
    storage_backend: Literal["local", "remote"] | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkExportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    parse_id: str
    text: str
    token_count: int
    page_numbers: list[int] = Field(default_factory=list)
    block_refs: list[str] = Field(default_factory=list)
    asset_refs: list[str] = Field(default_factory=list)
    assets: list[ChunkAssetView] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestExportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    document_id: str
    parse_id: str
    chunk_id: str
    text: str
    token_count: int
    page_numbers: list[int] = Field(default_factory=list)
    asset_refs: list[str] = Field(default_factory=list)
    asset_types: list[str] = Field(default_factory=list)
    asset_urls: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    artifact_profile_version: str = ARTIFACT_PROFILE_VERSION
    artifact_key: str = ""
    parse_id: str
    document_id: str
    filename: str
    file_type: str
    parser_engine: str
    created_at: str
    storage_backend: str
    source_sha256: str
    source_size_bytes: int
    page_count: int | None = None
    total_page_count: int | None = None
    asset_count: int
    block_count: int
    chunk_count: int
    markdown_url: str
    manifest_url: str = ""
    publish_events_url: str = ""
    structured_url: str
    chunks_url: str
    ingest_url: str
    assets_url_prefix: str
    root_dir: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactKeyIndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_key: str
    parse_id: str
    document_id: str
    filename: str
    created_at: str
    storage_backend: str
    manifest_url: str
    root_dir: str


class ParseDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    parse_id: str
    filename: str
    file_type: str
    parser_engine: str
    created_at: str
    source_sha256: str
    source_size_bytes: int
    page_count: int | None = None
    total_page_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    document: ParseDocument
    markdown: str
    assets: list[ParseAsset] = Field(default_factory=list)
    blocks: list[ParseBlock] = Field(default_factory=list)
    chunks: list[ParseChunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parse_id: str
    root_dir: str
    markdown_path: str
    manifest_path: str
    publish_events_path: str
    structured_path: str
    chunks_path: str
    ingest_path: str
    source_path: str
    assets_dir: str
    markdown_url: str
    manifest_url: str
    publish_events_url: str
    structured_url: str
    chunks_url: str
    ingest_url: str
    assets_url_prefix: str


class ArtifactStore(ABC):
    def __init__(self, public_base: str = "/api/v1/artifacts"):
        self.public_base = public_base.rstrip("/")

    @abstractmethod
    def get_paths(self, parse_id: str, filename: str) -> ArtifactPaths:
        raise NotImplementedError

    @abstractmethod
    def write_markdown(self, paths: ArtifactPaths, markdown: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_source(self, paths: ArtifactPaths, source_bytes: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_structured(self, paths: ArtifactPaths, artifact: ParseArtifact) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_manifest(self, paths: ArtifactPaths, manifest: ParseManifest) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_publish_event(self, paths: ArtifactPaths, event: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_chunks(self, paths: ArtifactPaths, artifact: ParseArtifact) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_ingest(self, paths: ArtifactPaths, artifact: ParseArtifact) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_bytes_asset(
        self,
        *,
        paths: ArtifactPaths,
        payload: bytes,
        asset_id: str,
        extension: str = ".png",
        media_type: str = "image/png",
    ) -> AssetStorage:
        raise NotImplementedError

    @abstractmethod
    def save_image_asset(
        self,
        *,
        paths: ArtifactPaths,
        image: Image.Image,
        asset_id: str,
    ) -> AssetStorage:
        raise NotImplementedError

    @abstractmethod
    def copy_file_asset(
        self,
        *,
        paths: ArtifactPaths,
        source_path: str | Path,
        asset_id: str,
        media_type: str = "image/png",
    ) -> AssetStorage:
        raise NotImplementedError

    @abstractmethod
    def download_asset(
        self,
        *,
        paths: ArtifactPaths,
        source_url: str,
        asset_id: str,
        media_type: str = "image/png",
        request_timeout: int = 60,
    ) -> AssetStorage:
        raise NotImplementedError

    @abstractmethod
    def read_file(self, parse_id: str, relative_path: str, media_type: str | None = None) -> tuple[bytes, str]:
        raise NotImplementedError

    @abstractmethod
    def list_manifests(self, limit: int = 20, tenant_id: str | None = None) -> list[ParseManifest]:
        raise NotImplementedError

    @abstractmethod
    def find_manifest_by_artifact_key(self, artifact_key: str, tenant_id: str | None = None) -> ParseManifest | None:
        raise NotImplementedError

    @abstractmethod
    def delete_artifact(self, manifest: ParseManifest) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def resolve_asset_url(
        self,
        *,
        parse_id: str,
        relative_path: str,
        download_path: str,
        mode: Literal["proxy", "direct", "signed"] = "proxy",
        expires_in: int = 3600,
        media_type: str | None = None,
    ) -> str:
        raise NotImplementedError


def build_document_id(file_bytes: bytes) -> str:
    return _bytes_sha256(file_bytes)


def build_parse_id(document_id: str) -> str:
    return f"{document_id[:12]}-{uuid4().hex[:12]}"


def build_document(
    *,
    filename: str,
    file_type: str,
    parser_engine: str,
    file_bytes: bytes,
    page_count: int | None = None,
    total_page_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParseDocument:
    document_id = build_document_id(file_bytes)
    return ParseDocument(
        document_id=document_id,
        parse_id=build_parse_id(document_id),
        filename=filename,
        file_type=file_type,
        parser_engine=parser_engine,
        created_at=_now_iso(),
        source_sha256=document_id,
        source_size_bytes=len(file_bytes),
        page_count=page_count,
        total_page_count=total_page_count,
        metadata=metadata or {},
    )


def build_artifact_key(document_id: str, artifact_profile: dict[str, Any]) -> str:
    payload = f"{document_id}:{_json_dumps_stable(artifact_profile)}".encode("utf-8")
    return _bytes_sha256(payload)


def build_parse_manifest(
    artifact: ParseArtifact,
    paths: ArtifactPaths,
    *,
    storage_backend: str,
    artifact_key: str,
    extra_metadata: dict[str, Any] | None = None,
) -> ParseManifest:
    return ParseManifest(
        artifact_key=artifact_key,
        parse_id=artifact.document.parse_id,
        document_id=artifact.document.document_id,
        filename=artifact.document.filename,
        file_type=artifact.document.file_type,
        parser_engine=artifact.document.parser_engine,
        created_at=artifact.document.created_at,
        storage_backend=storage_backend,
        source_sha256=artifact.document.source_sha256,
        source_size_bytes=artifact.document.source_size_bytes,
        page_count=artifact.document.page_count,
        total_page_count=artifact.document.total_page_count,
        asset_count=len(artifact.assets),
        block_count=len(artifact.blocks),
        chunk_count=len(artifact.chunks),
        markdown_url=paths.markdown_url,
        manifest_url=paths.manifest_url,
        publish_events_url=paths.publish_events_url,
        structured_url=paths.structured_url,
        chunks_url=paths.chunks_url,
        ingest_url=paths.ingest_url,
        assets_url_prefix=paths.assets_url_prefix,
        root_dir=paths.root_dir,
        metadata={
            **artifact.metadata,
            **artifact.document.metadata,
            **(extra_metadata or {}),
        },
    )


def parse_manifest_payload(payload: dict[str, Any]) -> ParseManifest:
    if not isinstance(payload, dict):
        raise TypeError("manifest payload must be a dict")
    parse_id = str(payload.get("parse_id") or "")
    patched_payload = dict(payload)
    patched_payload.setdefault("artifact_key", str((payload.get("metadata") or {}).get("artifact_key") or ""))
    patched_payload.setdefault("manifest_url", f"/api/v1/artifacts/{parse_id}/manifest" if parse_id else "")
    patched_payload.setdefault("publish_events_url", f"/api/v1/artifacts/{parse_id}/publish-events" if parse_id else "")
    return ParseManifest.model_validate(patched_payload)


def build_artifact_key_index_entry(manifest: ParseManifest) -> ArtifactKeyIndexEntry:
    return ArtifactKeyIndexEntry(
        artifact_key=manifest.artifact_key,
        parse_id=manifest.parse_id,
        document_id=manifest.document_id,
        filename=manifest.filename,
        created_at=manifest.created_at,
        storage_backend=manifest.storage_backend,
        manifest_url=manifest.manifest_url,
        root_dir=manifest.root_dir,
    )


def block_text_for_chunk(block: ParseBlock) -> str:
    text = (block.text or "").strip()
    if block.block_type == "title":
        if not text:
            return ""
        return f"# {text}"
    if block.block_type == "figure":
        if not text:
            return "[Figure]"
        return f"[Figure]\n{text}"
    if block.block_type == "table":
        if not text:
            return "[Table]"
        return f"[Table]\n{text}"
    if block.block_type == "equation":
        if not text:
            return "[Equation]"
        return f"[Equation]\n{text}"
    if block.block_type == "seal":
        if not text:
            return "[Seal]"
        return f"[Seal]\n{text}"
    if block.block_type == "barcode":
        if not text:
            return "[Barcode]"
        return f"[Barcode]\n{text}"
    if not text:
        return ""
    return text


def _is_visual_block(block: ParseBlock) -> bool:
    return block.block_type in {"figure", "table", "equation", "seal", "barcode"}


def _shared_pages(left: ParseBlock, right: ParseBlock) -> bool:
    if not left.page_numbers or not right.page_numbers:
        return False
    return bool(set(left.page_numbers).intersection(right.page_numbers))


def _infer_title_level(text: str) -> int:
    normalized = str(text or "").strip()
    if not normalized:
        return 1
    heading_match = re.match(r"^(#+)\s+", normalized)
    if heading_match:
        return min(6, len(heading_match.group(1)))
    numbering_match = re.match(r"^(\d+(?:\.\d+){0,5})[\s.)、:-]+", normalized)
    if numbering_match:
        return min(6, numbering_match.group(1).count(".") + 1)
    return 1


def _build_title_paths(blocks: list[ParseBlock], *, max_depth: int) -> list[list[str]]:
    current_path: list[str] = []
    title_paths: list[list[str]] = []
    max_depth = max(1, int(max_depth or DEFAULT_CHUNK_TITLE_PATH_DEPTH))
    for block in blocks:
        if block.block_type == "title":
            title_text = " ".join((block.text or "").split()).strip()
            if title_text:
                level = _infer_title_level(title_text)
                if level <= 1:
                    current_path = [title_text]
                else:
                    keep = min(len(current_path), level - 1)
                    current_path = current_path[:keep]
                    current_path.append(title_text)
                if len(current_path) > max_depth:
                    current_path = current_path[-max_depth:]
        title_paths.append(list(current_path))
    return title_paths


def _build_visual_context_maps(
    blocks: list[ParseBlock],
    *,
    window: int,
) -> tuple[list[list[str]], dict[str, str]]:
    context_refs: list[list[str]] = [[] for _ in blocks]
    asset_summary_by_id: dict[str, str] = {}
    window = max(0, int(window or 0))

    for block in blocks:
        if not _is_visual_block(block):
            continue
        summary_text = _truncate_text(block.text or block.block_type.title(), DEFAULT_CHUNK_ASSET_SUMMARY_LENGTH)
        label = f"[{block.block_type.title()}] {summary_text}".strip()
        for asset_id in block.asset_refs:
            if asset_id and asset_id not in asset_summary_by_id:
                asset_summary_by_id[asset_id] = label

    if window <= 0:
        return context_refs, asset_summary_by_id

    for idx, block in enumerate(blocks):
        if _is_visual_block(block):
            continue
        related_refs: list[str] = []
        for neighbor_idx in range(max(0, idx - window), min(len(blocks), idx + window + 1)):
            if neighbor_idx == idx:
                continue
            neighbor = blocks[neighbor_idx]
            if not _is_visual_block(neighbor):
                continue
            if not _shared_pages(block, neighbor):
                continue
            for asset_id in neighbor.asset_refs:
                if asset_id and asset_id not in related_refs:
                    related_refs.append(asset_id)
        context_refs[idx] = related_refs

    return context_refs, asset_summary_by_id


def _format_page_summary(page_numbers: list[int]) -> str:
    pages = [int(page) for page in page_numbers if int(page or 0) > 0]
    if not pages:
        return "unknown page"
    unique_pages = sorted(set(pages))
    if len(unique_pages) == 1:
        return f"page {unique_pages[0]}"
    if len(unique_pages) <= 3:
        return "pages " + ", ".join(str(page) for page in unique_pages)
    return f"pages {unique_pages[0]}-{unique_pages[-1]}"


def _is_markdown_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(bool(re.fullmatch(r":?-{2,}:?", cell.strip())) for cell in cells if cell.strip())


def _infer_markdown_table_shape(text: str) -> tuple[int | None, int | None]:
    rows: list[list[str]] = []
    for line in str(text or "").splitlines():
        normalized = line.strip()
        if "|" not in normalized:
            continue
        cells = [cell.strip() for cell in normalized.strip("|").split("|")]
        if _is_markdown_separator_row(cells):
            continue
        if cells:
            rows.append(cells)
    if not rows:
        return None, None
    return len(rows), max(len(row) for row in rows)


def _int_metadata_value(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _build_asset_summary(asset: ParseAsset) -> tuple[str, dict[str, Any]]:
    asset_type = asset.asset_type
    page_summary = _format_page_summary(asset.page_numbers)
    text = " ".join((asset.text or "").split()).strip()
    text_excerpt = _truncate_text(text, DEFAULT_CHUNK_ASSET_SUMMARY_LENGTH) if text else ""
    facts: dict[str, Any] = {
        "asset_type": asset_type,
        "page_numbers": sorted(set(asset.page_numbers)),
        "text_length": len(asset.text or ""),
    }

    if asset_type == "table":
        inferred_rows, inferred_columns = _infer_markdown_table_shape(asset.text)
        row_count = _int_metadata_value(asset.metadata, "row_count")
        column_count = _int_metadata_value(asset.metadata, "column_count")
        if row_count is None:
            row_count = inferred_rows
        if column_count is None:
            column_count = inferred_columns
        if row_count is not None:
            facts["row_count"] = row_count
        if column_count is not None:
            facts["column_count"] = column_count
        dimensions: list[str] = []
        if row_count is not None:
            dimensions.append(f"{row_count} rows")
        if column_count is not None:
            dimensions.append(f"{column_count} columns")
        summary = f"Table on {page_summary}"
        if dimensions:
            summary += " with " + " and ".join(dimensions)
        if text_excerpt:
            summary += f". Text: {text_excerpt}"
        return summary + ".", facts

    if asset.width is not None and asset.height is not None:
        facts["width"] = int(asset.width)
        facts["height"] = int(asset.height)
        size_text = f", {asset.width}x{asset.height}"
    else:
        size_text = ""

    labels = {
        "figure": "Figure",
        "image": "Image",
        "equation": "Equation",
        "seal": "Seal",
        "barcode": "Barcode",
    }
    label = labels.get(asset_type, asset_type.title())
    summary = f"{label} on {page_summary}{size_text}"
    if text_excerpt:
        text_label = "OCR text" if asset_type == "image" else "Text"
        summary += f". {text_label}: {text_excerpt}"
    if asset_type == "barcode" and asset.metadata.get("barcode_type"):
        facts["barcode_type"] = str(asset.metadata.get("barcode_type"))
    return summary + ".", facts


def _blocks_share_known_page(left: ParseBlock, right: ParseBlock) -> bool:
    if not left.page_numbers or not right.page_numbers:
        return True
    return bool(set(left.page_numbers).intersection(right.page_numbers))


def _chunk_asset_refs(chunk: ParseChunk, metadata_key: str) -> list[str]:
    value = chunk.metadata.get(metadata_key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def enrich_asset_context(
    assets: list[ParseAsset],
    blocks: list[ParseBlock],
    chunks: list[ParseChunk],
    *,
    window: int | None = None,
) -> list[ParseAsset]:
    if not assets:
        return []

    window_size = max(
        0,
        int(
            DEFAULT_CHUNK_CONTEXT_WINDOW
            if window is None
            else window
        ),
    )
    enriched_assets: list[ParseAsset] = []

    for asset in assets:
        direct_block_indexes = [
            idx
            for idx, block in enumerate(blocks)
            if asset.asset_id in block.asset_refs
        ]
        direct_block_refs = [blocks[idx].block_id for idx in direct_block_indexes]
        context_block_refs: list[str] = []
        context_texts: list[str] = []

        for direct_idx in direct_block_indexes:
            direct_block = blocks[direct_idx]
            start = max(0, direct_idx - window_size)
            end = min(len(blocks), direct_idx + window_size + 1)
            for neighbor in blocks[start:end]:
                if neighbor.block_id in direct_block_refs or neighbor.block_id in context_block_refs:
                    continue
                if _is_visual_block(neighbor):
                    continue
                if not _blocks_share_known_page(direct_block, neighbor):
                    continue
                normalized_text = " ".join((neighbor.text or "").split()).strip()
                if not normalized_text:
                    continue
                context_block_refs.append(neighbor.block_id)
                context_texts.append(_truncate_text(normalized_text, DEFAULT_CHUNK_ASSET_SUMMARY_LENGTH))

        direct_chunk_refs = [
            chunk.chunk_id
            for chunk in chunks
            if asset.asset_id in _chunk_asset_refs(chunk, "direct_asset_refs")
        ]
        context_chunk_refs = [
            chunk.chunk_id
            for chunk in chunks
            if asset.asset_id in _chunk_asset_refs(chunk, "context_asset_refs")
        ]
        fallback_chunk_refs = [
            chunk.chunk_id
            for chunk in chunks
            if asset.asset_id in chunk.asset_refs
            and chunk.chunk_id not in direct_chunk_refs
            and chunk.chunk_id not in context_chunk_refs
        ]
        chunk_refs = _dedupe_preserve_order(direct_chunk_refs + context_chunk_refs + fallback_chunk_refs)
        asset_summary, asset_summary_facts = _build_asset_summary(asset)

        enriched_assets.append(
            asset.model_copy(
                update={
                    "metadata": {
                        **asset.metadata,
                        "asset_context_schema_version": ASSET_CONTEXT_SCHEMA_VERSION,
                        "asset_summary_schema_version": ASSET_SUMMARY_SCHEMA_VERSION,
                        "asset_summary_source": "local_rules",
                        "asset_summary": asset_summary,
                        "asset_summary_facts": asset_summary_facts,
                        "direct_block_refs": direct_block_refs,
                        "context_block_refs": context_block_refs,
                        "context_texts": context_texts,
                        "direct_chunk_refs": direct_chunk_refs,
                        "context_chunk_refs": context_chunk_refs,
                        "chunk_refs": chunk_refs,
                    }
                }
            )
        )

    return enriched_assets


def build_chunks(
    blocks: list[ParseBlock],
    *,
    max_tokens: int,
    overlap_tokens: int,
    strategy: str | None = None,
) -> list[ParseChunk]:
    cleaned_blocks = [block for block in blocks if (block.text or "").strip() or block.asset_refs]
    if not cleaned_blocks:
        return []

    max_tokens = max(64, int(max_tokens or DEFAULT_CHUNK_MAX_TOKENS))
    overlap_tokens = max(0, min(int(overlap_tokens or 0), max_tokens // 2))
    chunk_strategy = normalize_chunk_strategy(strategy or os.environ.get("DEEPDOC_CHUNK_STRATEGY"))
    chunk_strategy_label = CHUNK_STRATEGY_LABELS[chunk_strategy]
    title_paths = _build_title_paths(
        cleaned_blocks,
        max_depth=int(os.environ.get("DEEPDOC_CHUNK_TITLE_PATH_DEPTH", str(DEFAULT_CHUNK_TITLE_PATH_DEPTH))),
    )
    context_asset_refs_by_block, asset_summary_by_id = _build_visual_context_maps(
        cleaned_blocks,
        window=int(os.environ.get("DEEPDOC_CHUNK_CONTEXT_WINDOW", str(DEFAULT_CHUNK_CONTEXT_WINDOW))),
    )
    chunks: list[ParseChunk] = []
    cursor = 0

    while cursor < len(cleaned_blocks):
        current_blocks: list[ParseBlock] = []
        current_texts: list[str] = []
        current_tokens = 0
        idx = cursor
        while idx < len(cleaned_blocks):
            block = cleaned_blocks[idx]
            rendered = block_text_for_chunk(block)
            if not rendered:
                idx += 1
                continue
            rendered_tokens = block.token_count or count_tokens(rendered)
            if current_blocks and chunk_strategy == "asset_aware" and _is_visual_block(block):
                break
            if current_blocks and chunk_strategy == "page_aware" and _shared_pages(current_blocks[-1], block) is False:
                current_pages = set(page for current_block in current_blocks for page in current_block.page_numbers)
                next_pages = set(block.page_numbers)
                if current_pages and next_pages:
                    break
            if current_blocks and current_tokens + rendered_tokens > max_tokens:
                break
            current_blocks.append(block)
            current_texts.append(rendered)
            current_tokens += rendered_tokens
            idx += 1
            if chunk_strategy == "asset_aware" and _is_visual_block(block):
                break
            if current_tokens >= max_tokens:
                break

        if not current_blocks:
            block = cleaned_blocks[cursor]
            rendered = block_text_for_chunk(block)
            current_blocks = [block]
            current_texts = [rendered]
            current_tokens = block.token_count or count_tokens(rendered)
            idx = cursor + 1

        title_path = next((path for path in reversed(title_paths[cursor:idx]) if path), [])
        title_prefix = ""
        if title_path:
            last_title = title_path[-1].strip()
            first_block = current_blocks[0]
            if not (first_block.block_type == "title" and first_block.text.strip() == last_title):
                title_prefix = "Section: " + " > ".join(title_path)

        block_refs = [block.block_id for block in current_blocks]
        direct_asset_refs = _dedupe_preserve_order(
            [asset_id for block in current_blocks for asset_id in block.asset_refs]
        )
        raw_context_asset_refs = _dedupe_preserve_order(
            [
                asset_id
                for block_idx in range(cursor, idx)
                for asset_id in context_asset_refs_by_block[block_idx]
            ]
        )
        context_asset_refs = [asset_id for asset_id in raw_context_asset_refs if asset_id not in direct_asset_refs]
        asset_refs = _dedupe_preserve_order(direct_asset_refs + context_asset_refs)
        related_asset_lines = [
            asset_summary_by_id[asset_id]
            for asset_id in context_asset_refs
            if asset_id in asset_summary_by_id
        ]
        chunk_parts: list[str] = []
        if title_prefix:
            chunk_parts.append(title_prefix)
        chunk_parts.extend(part for part in current_texts if part.strip())
        if related_asset_lines:
            chunk_parts.append("Related assets: " + " | ".join(_dedupe_preserve_order(related_asset_lines)))
        chunk_text = "\n\n".join(chunk_parts)
        page_numbers = sorted(
            set(page for block in current_blocks for page in block.page_numbers)
        )
        block_types = [block.block_type for block in current_blocks]
        chunk_index = len(chunks)
        chunks.append(
            ParseChunk(
                chunk_id=f"chunk-{chunk_index:04d}-{_text_sha256(chunk_text)[:12]}",
                text=chunk_text,
                token_count=count_tokens(chunk_text),
                page_numbers=page_numbers,
                block_refs=block_refs,
                asset_refs=asset_refs,
                metadata={
                    "block_count": len(current_blocks),
                    "block_types": block_types,
                    "title_path": title_path,
                    "direct_asset_refs": direct_asset_refs,
                    "context_asset_refs": context_asset_refs,
                    "chunk_strategy": chunk_strategy_label,
                },
            )
        )

        if idx >= len(cleaned_blocks):
            break

        if overlap_tokens <= 0:
            cursor = idx
            continue

        carry_tokens = 0
        carry_count = 0
        for block in reversed(current_blocks):
            rendered = block_text_for_chunk(block)
            carry_tokens += block.token_count or count_tokens(rendered)
            carry_count += 1
            if carry_tokens >= overlap_tokens:
                break
        cursor = max(cursor + 1, idx - carry_count)

    return chunks


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root_dir: str | Path | None = None, public_base: str = "/api/v1/artifacts"):
        super().__init__(public_base=public_base)
        self.root_dir = Path(root_dir or setting.ARTIFACTS_DIR)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_key_index_path(self, artifact_key: str) -> Path:
        shard = artifact_key[:2] or "00"
        return self.root_dir / "_indexes" / "artifact-keys" / shard / f"{artifact_key}.json"

    def get_paths(self, parse_id: str, filename: str) -> ArtifactPaths:
        safe_name = _safe_slug(Path(filename).name, default="document")
        parse_dir = self.root_dir / parse_id
        assets_dir = parse_dir / "assets"
        source_dir = parse_dir / "source"
        parse_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir(parents=True, exist_ok=True)
        return ArtifactPaths(
            parse_id=parse_id,
            root_dir=str(parse_dir),
            markdown_path=str(parse_dir / "markdown.md"),
            manifest_path=str(parse_dir / "manifest.json"),
            publish_events_path=str(parse_dir / "publish-events.jsonl"),
            structured_path=str(parse_dir / "structured.json"),
            chunks_path=str(parse_dir / "chunks.jsonl"),
            ingest_path=str(parse_dir / "ingest.jsonl"),
            source_path=str(source_dir / safe_name),
            assets_dir=str(assets_dir),
            markdown_url=f"{self.public_base}/{parse_id}/markdown",
            manifest_url=f"{self.public_base}/{parse_id}/manifest",
            publish_events_url=f"{self.public_base}/{parse_id}/publish-events",
            structured_url=f"{self.public_base}/{parse_id}/structured",
            chunks_url=f"{self.public_base}/{parse_id}/chunks",
            ingest_url=f"{self.public_base}/{parse_id}/ingest",
            assets_url_prefix=f"{self.public_base}/{parse_id}/assets",
        )

    def write_markdown(self, paths: ArtifactPaths, markdown: str) -> None:
        Path(paths.markdown_path).write_text(markdown or "", encoding="utf-8")

    def write_source(self, paths: ArtifactPaths, source_bytes: bytes) -> None:
        Path(paths.source_path).write_bytes(source_bytes)

    def write_structured(self, paths: ArtifactPaths, artifact: ParseArtifact) -> None:
        Path(paths.structured_path).write_text(
            json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_manifest(self, paths: ArtifactPaths, manifest: ParseManifest) -> None:
        Path(paths.manifest_path).write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        index_path = self._artifact_key_index_path(manifest.artifact_key)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps(build_artifact_key_index_entry(manifest).model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def append_publish_event(self, paths: ArtifactPaths, event: dict[str, Any]) -> None:
        with Path(paths.publish_events_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_chunks(self, paths: ArtifactPaths, artifact: ParseArtifact) -> None:
        chunk_records = build_chunk_export_records(artifact, store=self)
        with Path(paths.chunks_path).open("w", encoding="utf-8") as handle:
            for record in chunk_records:
                handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def write_ingest(self, paths: ArtifactPaths, artifact: ParseArtifact) -> None:
        chunk_records = build_chunk_export_records(artifact, store=self)
        ingest_records = build_ingest_export_records(chunk_records)
        with Path(paths.ingest_path).open("w", encoding="utf-8") as handle:
            for record in ingest_records:
                handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def save_bytes_asset(
        self,
        *,
        paths: ArtifactPaths,
        payload: bytes,
        asset_id: str,
        extension: str = ".png",
        media_type: str = "image/png",
    ) -> AssetStorage:
        ext = extension if extension.startswith(".") else f".{extension}"
        relative_path = f"assets/{asset_id}{ext}"
        absolute_path = Path(paths.root_dir) / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_bytes(payload)
        return AssetStorage(
            backend="local",
            relative_path=relative_path,
            absolute_path=str(absolute_path),
            download_path=f"{paths.assets_url_prefix}/{asset_id}{ext}",
            media_type=media_type,
        )

    def save_image_asset(
        self,
        *,
        paths: ArtifactPaths,
        image: Image.Image,
        asset_id: str,
    ) -> AssetStorage:
        img = image.convert("RGB")
        relative_path = f"assets/{asset_id}.png"
        absolute_path = Path(paths.root_dir) / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(absolute_path, format="PNG")
        return AssetStorage(
            backend="local",
            relative_path=relative_path,
            absolute_path=str(absolute_path),
            download_path=f"{paths.assets_url_prefix}/{asset_id}.png",
            media_type="image/png",
        )

    def copy_file_asset(
        self,
        *,
        paths: ArtifactPaths,
        source_path: str | Path,
        asset_id: str,
        media_type: str = "image/png",
    ) -> AssetStorage:
        source = Path(source_path)
        ext = source.suffix or ".bin"
        payload = source.read_bytes()
        return self.save_bytes_asset(
            paths=paths,
            payload=payload,
            asset_id=asset_id,
            extension=ext,
            media_type=media_type,
        )

    def download_asset(
        self,
        *,
        paths: ArtifactPaths,
        source_url: str,
        asset_id: str,
        media_type: str = "image/png",
        request_timeout: int = 60,
    ) -> AssetStorage:
        response = requests.get(source_url, timeout=request_timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", media_type).split(";")[0].strip() or media_type
        ext = Path(source_url).suffix or ".bin"
        if content_type == "image/png":
            ext = ".png"
        elif content_type == "image/jpeg":
            ext = ".jpg"
        return self.save_bytes_asset(
            paths=paths,
            payload=response.content,
            asset_id=asset_id,
            extension=ext,
            media_type=content_type,
        )

    def read_file(self, parse_id: str, relative_path: str, media_type: str | None = None) -> tuple[bytes, str]:
        target = self.root_dir / parse_id / relative_path
        if not target.exists():
            raise FileNotFoundError(relative_path)
        guessed_type = media_type or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        return target.read_bytes(), guessed_type

    def list_manifests(self, limit: int = 20, tenant_id: str | None = None) -> list[ParseManifest]:
        manifests: list[ParseManifest] = []
        normalized_tenant = _normalized_tenant_id(tenant_id)
        for path in sorted(self.root_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                manifest = parse_manifest_payload(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                logger.exception("Failed to load artifact manifest from %s", path)
                continue
            if normalized_tenant and _manifest_tenant_id(manifest) != normalized_tenant:
                continue
            manifests.append(manifest)
            if len(manifests) >= max(1, int(limit)):
                break
        return manifests

    def find_manifest_by_artifact_key(self, artifact_key: str, tenant_id: str | None = None) -> ParseManifest | None:
        normalized_tenant = _normalized_tenant_id(tenant_id)
        index_path = self._artifact_key_index_path(artifact_key)
        if index_path.exists():
            try:
                entry = ArtifactKeyIndexEntry.model_validate(json.loads(index_path.read_text(encoding="utf-8")))
                payload, _ = self.read_file(entry.parse_id, "manifest.json", "application/json")
                manifest = parse_manifest_payload(json.loads(payload.decode("utf-8")))
                if manifest.artifact_key == artifact_key and (
                    not normalized_tenant or _manifest_tenant_id(manifest) == normalized_tenant
                ):
                    return manifest
            except Exception:
                logger.exception("Failed to use artifact key index %s", index_path)
        for path in sorted(self.root_dir.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                manifest = parse_manifest_payload(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                logger.exception("Failed to load artifact manifest from %s", path)
                continue
            if manifest.artifact_key == artifact_key and (
                not normalized_tenant or _manifest_tenant_id(manifest) == normalized_tenant
            ):
                return manifest
        return None

    def delete_artifact(self, manifest: ParseManifest) -> dict[str, Any]:
        parse_dir = self.root_dir / manifest.parse_id
        deleted_files = 0
        if parse_dir.exists():
            for path in parse_dir.rglob("*"):
                if path.is_file():
                    deleted_files += 1
            shutil.rmtree(parse_dir, ignore_errors=True)
        if manifest.artifact_key:
            index_path = self._artifact_key_index_path(manifest.artifact_key)
            if index_path.exists():
                index_path.unlink(missing_ok=True)
        return {
            "parse_id": manifest.parse_id,
            "artifact_key": manifest.artifact_key,
            "deleted_files": deleted_files,
            "storage_backend": "local",
        }

    def resolve_asset_url(
        self,
        *,
        parse_id: str,
        relative_path: str,
        download_path: str,
        mode: Literal["proxy", "direct", "signed"] = "proxy",
        expires_in: int = 3600,
        media_type: str | None = None,
    ) -> str:
        return download_path


class S3ArtifactStore(ArtifactStore):
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        public_base: str = "/api/v1/artifacts",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        addressing_style: str = "path",
        s3_client=None,
    ):
        super().__init__(public_base=public_base)
        if not bucket:
            raise ValueError("DEEPDOC_ARTIFACT_BUCKET is required when DEEPDOC_ARTIFACT_BACKEND=s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self.addressing_style = addressing_style if addressing_style in {"path", "virtual"} else "path"
        self.public_base_url = (os.environ.get("DEEPDOC_ARTIFACT_PUBLIC_BASE_URL") or "").strip().rstrip("/")
        self.public_url_template = (os.environ.get("DEEPDOC_ARTIFACT_PUBLIC_URL_TEMPLATE") or "").strip()
        self._s3_client = s3_client or self._build_s3_client(
            endpoint_url=endpoint_url,
            region_name=region_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            addressing_style=self.addressing_style,
        )

    @staticmethod
    def _build_s3_client(
        *,
        endpoint_url: str | None,
        region_name: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
        session_token: str | None,
        addressing_style: str,
    ):
        try:
            import boto3
            from botocore.config import Config
        except Exception as exc:
            raise RuntimeError(
                "S3 artifact backend requires boto3. Install it first, e.g. pip install boto3"
            ) from exc

        session = boto3.session.Session()
        return session.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token,
            config=Config(s3={"addressing_style": addressing_style}),
        )

    def _parse_root(self, parse_id: str) -> str:
        return f"s3://{self.bucket}/{self._object_key(parse_id, '')}".rstrip("/")

    def _object_key(self, parse_id: str, relative_path: str) -> str:
        clean_relative = relative_path.lstrip("/")
        parts = [part for part in [self.prefix, parse_id, clean_relative] if part]
        return "/".join(parts)

    def _artifact_key_index_key(self, artifact_key: str) -> str:
        shard = artifact_key[:2] or "00"
        parts = [part for part in [self.prefix, "_indexes", "artifact-keys", shard, f"{artifact_key}.json"] if part]
        return "/".join(parts)

    def _direct_url(self, parse_id: str, relative_path: str) -> str | None:
        key = self._object_key(parse_id, relative_path)
        if self.public_url_template:
            return self.public_url_template.format(
                bucket=self.bucket,
                key=key,
                parse_id=parse_id,
                relative_path=relative_path.lstrip("/"),
            )
        if self.public_base_url:
            return f"{self.public_base_url}/{key}"
        return None

    def _upload_bytes(self, *, parse_id: str, relative_path: str, payload: bytes, media_type: str) -> None:
        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=self._object_key(parse_id, relative_path),
            Body=payload,
            ContentType=media_type,
        )

    def get_paths(self, parse_id: str, filename: str) -> ArtifactPaths:
        safe_name = _safe_slug(Path(filename).name, default="document")
        return ArtifactPaths(
            parse_id=parse_id,
            root_dir=self._parse_root(parse_id),
            markdown_path=f"{self._parse_root(parse_id)}/markdown.md",
            manifest_path=f"{self._parse_root(parse_id)}/manifest.json",
            publish_events_path=f"{self._parse_root(parse_id)}/publish-events.jsonl",
            structured_path=f"{self._parse_root(parse_id)}/structured.json",
            chunks_path=f"{self._parse_root(parse_id)}/chunks.jsonl",
            ingest_path=f"{self._parse_root(parse_id)}/ingest.jsonl",
            source_path=f"{self._parse_root(parse_id)}/source/{safe_name}",
            assets_dir=f"{self._parse_root(parse_id)}/assets",
            markdown_url=f"{self.public_base}/{parse_id}/markdown",
            manifest_url=f"{self.public_base}/{parse_id}/manifest",
            publish_events_url=f"{self.public_base}/{parse_id}/publish-events",
            structured_url=f"{self.public_base}/{parse_id}/structured",
            chunks_url=f"{self.public_base}/{parse_id}/chunks",
            ingest_url=f"{self.public_base}/{parse_id}/ingest",
            assets_url_prefix=f"{self.public_base}/{parse_id}/assets",
        )

    def write_markdown(self, paths: ArtifactPaths, markdown: str) -> None:
        self._upload_bytes(
            parse_id=paths.parse_id,
            relative_path="markdown.md",
            payload=(markdown or "").encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
        )

    def write_source(self, paths: ArtifactPaths, source_bytes: bytes) -> None:
        source_name = Path(paths.source_path).name
        media_type = mimetypes.guess_type(source_name)[0] or "application/octet-stream"
        self._upload_bytes(
            parse_id=paths.parse_id,
            relative_path=f"source/{source_name}",
            payload=source_bytes,
            media_type=media_type,
        )

    def write_structured(self, paths: ArtifactPaths, artifact: ParseArtifact) -> None:
        payload = json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        self._upload_bytes(
            parse_id=paths.parse_id,
            relative_path="structured.json",
            payload=payload,
            media_type="application/json; charset=utf-8",
        )

    def write_manifest(self, paths: ArtifactPaths, manifest: ParseManifest) -> None:
        payload = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        self._upload_bytes(
            parse_id=paths.parse_id,
            relative_path="manifest.json",
            payload=payload,
            media_type="application/json; charset=utf-8",
        )
        index_payload = (
            json.dumps(build_artifact_key_index_entry(manifest).model_dump(mode="json"), ensure_ascii=False, indent=2).encode("utf-8")
            + b"\n"
        )
        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=self._artifact_key_index_key(manifest.artifact_key),
            Body=index_payload,
            ContentType="application/json; charset=utf-8",
        )

    def append_publish_event(self, paths: ArtifactPaths, event: dict[str, Any]) -> None:
        relative_path = "publish-events.jsonl"
        try:
            current_payload, _ = self.read_file(paths.parse_id, relative_path, "application/x-ndjson")
        except FileNotFoundError:
            current_payload = b""
        event_payload = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
        self._upload_bytes(
            parse_id=paths.parse_id,
            relative_path=relative_path,
            payload=current_payload + event_payload,
            media_type="application/x-ndjson; charset=utf-8",
        )

    def write_chunks(self, paths: ArtifactPaths, artifact: ParseArtifact) -> None:
        chunk_records = build_chunk_export_records(artifact, store=self)
        lines = [
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
            for record in chunk_records
        ]
        payload = b"\n".join(lines) + (b"\n" if lines else b"")
        self._upload_bytes(
            parse_id=paths.parse_id,
            relative_path="chunks.jsonl",
            payload=payload,
            media_type="application/x-ndjson; charset=utf-8",
        )

    def write_ingest(self, paths: ArtifactPaths, artifact: ParseArtifact) -> None:
        chunk_records = build_chunk_export_records(artifact, store=self)
        ingest_records = build_ingest_export_records(chunk_records)
        lines = [
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
            for record in ingest_records
        ]
        payload = b"\n".join(lines) + (b"\n" if lines else b"")
        self._upload_bytes(
            parse_id=paths.parse_id,
            relative_path="ingest.jsonl",
            payload=payload,
            media_type="application/x-ndjson; charset=utf-8",
        )

    def save_bytes_asset(
        self,
        *,
        paths: ArtifactPaths,
        payload: bytes,
        asset_id: str,
        extension: str = ".png",
        media_type: str = "image/png",
    ) -> AssetStorage:
        ext = extension if extension.startswith(".") else f".{extension}"
        relative_path = f"assets/{asset_id}{ext}"
        self._upload_bytes(
            parse_id=paths.parse_id,
            relative_path=relative_path,
            payload=payload,
            media_type=media_type,
        )
        return AssetStorage(
            backend="remote",
            relative_path=relative_path,
            absolute_path=f"s3://{self.bucket}/{self._object_key(paths.parse_id, relative_path)}",
            download_path=f"{paths.assets_url_prefix}/{asset_id}{ext}",
            media_type=media_type,
        )

    def save_image_asset(
        self,
        *,
        paths: ArtifactPaths,
        image: Image.Image,
        asset_id: str,
    ) -> AssetStorage:
        img = image.convert("RGB")
        payload = pil_image_to_bytes(img)
        return self.save_bytes_asset(
            paths=paths,
            payload=payload,
            asset_id=asset_id,
            extension=".png",
            media_type="image/png",
        )

    def copy_file_asset(
        self,
        *,
        paths: ArtifactPaths,
        source_path: str | Path,
        asset_id: str,
        media_type: str = "image/png",
    ) -> AssetStorage:
        source = Path(source_path)
        ext = source.suffix or ".bin"
        payload = source.read_bytes()
        resolved_media = media_type or mimetypes.guess_type(str(source))[0] or "application/octet-stream"
        return self.save_bytes_asset(
            paths=paths,
            payload=payload,
            asset_id=asset_id,
            extension=ext,
            media_type=resolved_media,
        )

    def download_asset(
        self,
        *,
        paths: ArtifactPaths,
        source_url: str,
        asset_id: str,
        media_type: str = "image/png",
        request_timeout: int = 60,
    ) -> AssetStorage:
        response = requests.get(source_url, timeout=request_timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", media_type).split(";")[0].strip() or media_type
        ext = Path(source_url).suffix or ".bin"
        if content_type == "image/png":
            ext = ".png"
        elif content_type == "image/jpeg":
            ext = ".jpg"
        return self.save_bytes_asset(
            paths=paths,
            payload=response.content,
            asset_id=asset_id,
            extension=ext,
            media_type=content_type,
        )

    def read_file(self, parse_id: str, relative_path: str, media_type: str | None = None) -> tuple[bytes, str]:
        try:
            result = self._s3_client.get_object(Bucket=self.bucket, Key=self._object_key(parse_id, relative_path))
        except Exception as exc:
            error_code = None
            if hasattr(exc, "response") and isinstance(exc.response, dict):
                error_code = (exc.response.get("Error") or {}).get("Code")
            if error_code in {"NoSuchKey", "404"}:
                raise FileNotFoundError(relative_path) from exc
            raise
        payload = result["Body"].read()
        content_type = result.get("ContentType") or media_type or mimetypes.guess_type(relative_path)[0] or "application/octet-stream"
        return payload, content_type

    def resolve_asset_url(
        self,
        *,
        parse_id: str,
        relative_path: str,
        download_path: str,
        mode: Literal["proxy", "direct", "signed"] = "proxy",
        expires_in: int = 3600,
        media_type: str | None = None,
    ) -> str:
        if mode == "proxy":
            return download_path
        if mode == "direct":
            direct_url = self._direct_url(parse_id, relative_path)
            if direct_url:
                return direct_url
            logger.warning("Artifact direct URL requested but no public URL template/base configured; fallback to proxy")
            return download_path
        try:
            return self._s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": self._object_key(parse_id, relative_path)},
                ExpiresIn=max(60, min(int(expires_in or 3600), 604800)),
            )
        except Exception:
            logger.exception("Failed to generate presigned artifact URL for %s", relative_path)
            return download_path

    def list_manifests(self, limit: int = 20, tenant_id: str | None = None) -> list[ParseManifest]:
        prefix = f"{self.prefix}/" if self.prefix else ""
        manifests: list[ParseManifest] = []
        normalized_tenant = _normalized_tenant_id(tenant_id)
        continuation_token = None
        while True:
            kwargs = {
                "Bucket": self.bucket,
                "Prefix": prefix,
                "MaxKeys": 1000,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._s3_client.list_objects_v2(**kwargs)
            for item in sorted(response.get("Contents", []), key=lambda obj: obj.get("LastModified"), reverse=True):
                key = str(item.get("Key") or "")
                if not key.endswith("/manifest.json") and not key.endswith("manifest.json"):
                    continue
                parse_id = Path(key).parent.name
                try:
                    payload, _ = self.read_file(parse_id, "manifest.json", "application/json")
                    manifest = parse_manifest_payload(json.loads(payload.decode("utf-8")))
                except Exception:
                    logger.exception("Failed to load remote artifact manifest from %s", key)
                    continue
                if normalized_tenant and _manifest_tenant_id(manifest) != normalized_tenant:
                    continue
                manifests.append(manifest)
                if len(manifests) >= max(1, int(limit)):
                    return manifests
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
        return manifests

    def find_manifest_by_artifact_key(self, artifact_key: str, tenant_id: str | None = None) -> ParseManifest | None:
        normalized_tenant = _normalized_tenant_id(tenant_id)
        if artifact_key:
            try:
                result = self._s3_client.get_object(Bucket=self.bucket, Key=self._artifact_key_index_key(artifact_key))
                entry = ArtifactKeyIndexEntry.model_validate(json.loads(result["Body"].read().decode("utf-8")))
                payload, _ = self.read_file(entry.parse_id, "manifest.json", "application/json")
                manifest = parse_manifest_payload(json.loads(payload.decode("utf-8")))
                if manifest.artifact_key == artifact_key and (
                    not normalized_tenant or _manifest_tenant_id(manifest) == normalized_tenant
                ):
                    return manifest
            except Exception as exc:
                error_code = None
                if hasattr(exc, "response") and isinstance(exc.response, dict):
                    error_code = (exc.response.get("Error") or {}).get("Code")
                if error_code not in {"NoSuchKey", "404", None}:
                    logger.exception("Failed to use remote artifact key index %s", artifact_key)
        prefix = f"{self.prefix}/" if self.prefix else ""
        continuation_token = None
        while True:
            kwargs = {
                "Bucket": self.bucket,
                "Prefix": prefix,
                "MaxKeys": 1000,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._s3_client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                key = str(item.get("Key") or "")
                if not key.endswith("/manifest.json") and not key.endswith("manifest.json"):
                    continue
                parse_id = Path(key).parent.name
                try:
                    payload, _ = self.read_file(parse_id, "manifest.json", "application/json")
                    manifest = parse_manifest_payload(json.loads(payload.decode("utf-8")))
                except Exception:
                    logger.exception("Failed to load remote artifact manifest from %s", key)
                    continue
                if manifest.artifact_key == artifact_key and (
                    not normalized_tenant or _manifest_tenant_id(manifest) == normalized_tenant
                ):
                    return manifest
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
        return None

    def delete_artifact(self, manifest: ParseManifest) -> dict[str, Any]:
        parse_prefix = self._object_key(manifest.parse_id, "")
        continuation_token = None
        deleted_objects = 0
        while True:
            kwargs = {
                "Bucket": self.bucket,
                "Prefix": parse_prefix,
                "MaxKeys": 1000,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._s3_client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                key = str(item.get("Key") or "")
                if not key:
                    continue
                if hasattr(self._s3_client, "delete_object"):
                    self._s3_client.delete_object(Bucket=self.bucket, Key=key)
                deleted_objects += 1
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
        if manifest.artifact_key and hasattr(self._s3_client, "delete_object"):
            self._s3_client.delete_object(Bucket=self.bucket, Key=self._artifact_key_index_key(manifest.artifact_key))
        return {
            "parse_id": manifest.parse_id,
            "artifact_key": manifest.artifact_key,
            "deleted_objects": deleted_objects,
            "storage_backend": "s3",
        }


def create_artifact_store(public_base: str = "/api/v1/artifacts") -> ArtifactStore:
    backend = (os.environ.get("DEEPDOC_ARTIFACT_BACKEND", "local") or "local").strip().lower()
    if backend == "local":
        return LocalArtifactStore(public_base=public_base)
    if backend == "s3":
        return S3ArtifactStore(
            bucket=os.environ.get("DEEPDOC_ARTIFACT_BUCKET", "").strip(),
            prefix=os.environ.get("DEEPDOC_ARTIFACT_PREFIX", "deepdoc-artifacts").strip(),
            public_base=public_base,
            endpoint_url=(os.environ.get("DEEPDOC_ARTIFACT_ENDPOINT_URL") or "").strip() or None,
            region_name=(os.environ.get("DEEPDOC_ARTIFACT_REGION") or "").strip() or None,
            access_key_id=(os.environ.get("DEEPDOC_ARTIFACT_ACCESS_KEY_ID") or "").strip() or None,
            secret_access_key=(os.environ.get("DEEPDOC_ARTIFACT_SECRET_ACCESS_KEY") or "").strip() or None,
            session_token=(os.environ.get("DEEPDOC_ARTIFACT_SESSION_TOKEN") or "").strip() or None,
            addressing_style=(os.environ.get("DEEPDOC_ARTIFACT_ADDRESSING_STYLE") or "path").strip().lower(),
        )
    raise RuntimeError(f"Unsupported DEEPDOC_ARTIFACT_BACKEND: {backend}")


def build_remote_storage(source_url: str, media_type: str = "image/png") -> AssetStorage:
    return AssetStorage(
        backend="remote",
        relative_path="",
        absolute_path="",
        download_path=source_url,
        media_type=media_type,
        source_url=source_url,
    )


def build_asset_id(asset_type: str, image_bytes: bytes, page_numbers: list[int]) -> str:
    digest = _bytes_sha256(image_bytes)
    page_part = "-".join(str(page) for page in page_numbers[:4]) or "0"
    return f"{asset_type}-{page_part}-{digest[:16]}"


def pil_image_to_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    bio = BytesIO()
    image.convert("RGB").save(bio, format="PNG")
    return bio.getvalue()


def build_chunk_export_records(
    artifact: ParseArtifact,
    *,
    store: ArtifactStore | None = None,
    asset_url_mode: Literal["proxy", "direct", "signed"] = "proxy",
    signed_url_ttl: int = 3600,
) -> list[ChunkExportRecord]:
    asset_map = {asset.asset_id: asset for asset in artifact.assets}
    records: list[ChunkExportRecord] = []
    for chunk in artifact.chunks:
        chunk_assets: list[ChunkAssetView] = []
        for asset_id in chunk.asset_refs:
            asset = asset_map.get(asset_id)
            if asset is None:
                continue
            download_path = asset.storage.download_path if asset.storage else None
            resolved_url = download_path
            if (
                store is not None
                and asset.storage is not None
                and asset.storage.relative_path
                and download_path
            ):
                resolved_url = store.resolve_asset_url(
                    parse_id=artifact.document.parse_id,
                    relative_path=asset.storage.relative_path,
                    download_path=download_path,
                    mode=asset_url_mode,
                    expires_in=signed_url_ttl,
                    media_type=asset.storage.media_type,
                )
            chunk_assets.append(
                ChunkAssetView(
                    asset_id=asset.asset_id,
                    asset_type=asset.asset_type,
                    title=asset.title,
                    text=asset.text,
                    page_numbers=asset.page_numbers,
                    positions=asset.positions,
                    download_path=download_path,
                    resolved_url=resolved_url,
                    media_type=asset.storage.media_type if asset.storage else None,
                    storage_backend=asset.storage.backend if asset.storage else None,
                    source_url=asset.storage.source_url if asset.storage else None,
                    metadata=asset.metadata,
                )
            )
        records.append(
            ChunkExportRecord(
                chunk_id=chunk.chunk_id,
                document_id=artifact.document.document_id,
                parse_id=artifact.document.parse_id,
                text=chunk.text,
                token_count=chunk.token_count,
                page_numbers=chunk.page_numbers,
                block_refs=chunk.block_refs,
                asset_refs=chunk.asset_refs,
                assets=chunk_assets,
                metadata={
                    **chunk.metadata,
                    "schema_version": CHUNK_EXPORT_SCHEMA_VERSION,
                    "asset_url_mode": asset_url_mode,
                    "parser_engine": artifact.document.parser_engine,
                    "file_type": artifact.document.file_type,
                    "filename": artifact.document.filename,
                    "tenant_id": artifact.document.metadata.get("tenant_id"),
                },
            )
        )
    return records


def build_ingest_export_records(chunk_records: list[ChunkExportRecord]) -> list[IngestExportRecord]:
    ingest_records: list[IngestExportRecord] = []
    for record in chunk_records:
        asset_urls = _dedupe_preserve_order(
            [asset.resolved_url or asset.download_path or "" for asset in record.assets if asset.resolved_url or asset.download_path]
        )
        asset_types = _dedupe_preserve_order([asset.asset_type for asset in record.assets])
        ingest_records.append(
            IngestExportRecord(
                record_id=f"{record.document_id}:{record.chunk_id}",
                document_id=record.document_id,
                parse_id=record.parse_id,
                chunk_id=record.chunk_id,
                text=record.text,
                token_count=record.token_count,
                page_numbers=record.page_numbers,
                asset_refs=record.asset_refs,
                asset_types=asset_types,
                asset_urls=asset_urls,
                metadata={
                    **record.metadata,
                    "schema_version": INGEST_EXPORT_SCHEMA_VERSION,
                    "chunk_schema_version": record.metadata.get("schema_version", CHUNK_EXPORT_SCHEMA_VERSION),
                },
            )
        )
    return ingest_records

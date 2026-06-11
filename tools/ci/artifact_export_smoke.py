#!/usr/bin/env python3
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.parse_artifacts import (
    ASSET_CONTEXT_SCHEMA_VERSION,
    ASSET_SUMMARY_SCHEMA_VERSION,
    CHUNK_EXPORT_SCHEMA_VERSION,
    INGEST_EXPORT_SCHEMA_VERSION,
    LocalArtifactStore,
    ParseArtifact,
    ParseAsset,
    ParseBlock,
    build_artifact_key,
    build_chunks,
    build_document,
    build_parse_manifest,
    enrich_asset_context,
)


def _jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        row = json.loads(normalized)
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL row is not an object in {path}")
        rows.append(row)
    return rows


def _run_smoke(root_dir: Path) -> dict[str, object]:
    file_bytes = b"DocPilot artifact export smoke\n"
    document = build_document(
        filename="artifact-smoke.pdf",
        file_type="pdf",
        parser_engine="deepdoc",
        file_bytes=file_bytes,
        page_count=1,
        total_page_count=1,
        metadata={"tenant_id": "smoke", "chunk_strategy": "asset_aware"},
    )
    table_asset = ParseAsset(
        asset_id="table-smoke-1",
        asset_type="table",
        title="Smoke table",
        text="| A | B |\n|---|---|\n| 1 | 2 |",
        page_numbers=[1],
    )
    blocks = [
        ParseBlock(
            block_id="block-0000",
            block_type="title",
            text="Smoke Document",
            page_numbers=[1],
            token_count=3,
        ),
        ParseBlock(
            block_id="block-0001",
            block_type="text",
            text="Intro paragraph before the table.",
            page_numbers=[1],
            token_count=6,
        ),
        ParseBlock(
            block_id="block-0002",
            block_type="table",
            text=table_asset.text,
            page_numbers=[1],
            token_count=12,
            asset_refs=[table_asset.asset_id],
        ),
        ParseBlock(
            block_id="block-0003",
            block_type="text",
            text="Closing paragraph after the table.",
            page_numbers=[1],
            token_count=6,
        ),
    ]
    chunks = build_chunks(blocks, max_tokens=256, overlap_tokens=0, strategy="asset_aware")
    assets = enrich_asset_context([table_asset], blocks, chunks, window=1)
    artifact = ParseArtifact(
        document=document,
        markdown="# Smoke Document\n\nIntro paragraph before the table.\n\n| A | B |\n\nClosing paragraph after the table.\n",
        assets=assets,
        blocks=blocks,
        chunks=chunks,
        metadata={"parser_engine": "deepdoc", "chunk_strategy": "asset_aware"},
    )
    store = LocalArtifactStore(root_dir=root_dir)
    paths = store.get_paths(document.parse_id, document.filename)
    artifact_profile = {
        "artifact_profile_version": "smoke",
        "file_type": document.file_type,
        "parser_engine": document.parser_engine,
        "include_chunks": True,
        "chunk_strategy": "asset_aware",
    }
    manifest = build_parse_manifest(
        artifact,
        paths,
        storage_backend="local",
        artifact_key=build_artifact_key(document.document_id, artifact_profile),
        extra_metadata={"artifact_profile": artifact_profile},
    )

    store.write_markdown(paths, artifact.markdown)
    store.write_structured(paths, artifact)
    store.write_chunks(paths, artifact)
    store.write_ingest(paths, artifact)
    store.write_manifest(paths, manifest)

    structured = json.loads(Path(paths.structured_path).read_text(encoding="utf-8"))
    chunk_records = _jsonl_rows(Path(paths.chunks_path))
    ingest_records = _jsonl_rows(Path(paths.ingest_path))

    if structured["document"]["parse_id"] != document.parse_id:
        raise RuntimeError("structured artifact parse_id mismatch")
    if len(structured.get("chunks") or []) != len(chunks):
        raise RuntimeError("structured artifact chunk count mismatch")
    table_chunks = [chunk for chunk in structured["chunks"] if chunk.get("block_refs") == ["block-0002"]]
    if len(table_chunks) != 1:
        raise RuntimeError("asset_aware strategy did not keep the table block as its own chunk")
    if table_chunks[0]["metadata"].get("chunk_strategy") != "asset_aware_v1":
        raise RuntimeError("structured table chunk missing asset_aware_v1 strategy")
    structured_assets = structured.get("assets") or []
    if not structured_assets:
        raise RuntimeError("structured artifact has no assets")
    table_asset_metadata = structured_assets[0].get("metadata") or {}
    if table_asset_metadata.get("asset_context_schema_version") != ASSET_CONTEXT_SCHEMA_VERSION:
        raise RuntimeError("structured table asset missing asset context schema version")
    if table_asset_metadata.get("asset_summary_schema_version") != ASSET_SUMMARY_SCHEMA_VERSION:
        raise RuntimeError("structured table asset missing asset summary schema version")
    if table_asset_metadata.get("asset_summary_source") != "local_rules":
        raise RuntimeError("structured table asset missing local rule summary source")
    if not str(table_asset_metadata.get("asset_summary") or "").strip():
        raise RuntimeError("structured table asset missing asset summary")
    if not isinstance(table_asset_metadata.get("asset_summary_facts"), dict):
        raise RuntimeError("structured table asset missing asset summary facts")
    if table_asset_metadata.get("direct_block_refs") != ["block-0002"]:
        raise RuntimeError("structured table asset missing direct block refs")
    if table_asset_metadata.get("context_block_refs") != ["block-0001", "block-0003"]:
        raise RuntimeError("structured table asset missing neighbor context block refs")
    if not chunk_records:
        raise RuntimeError("chunks.jsonl has no records")
    if not ingest_records:
        raise RuntimeError("ingest.jsonl has no records")
    first_chunk_metadata = chunk_records[0].get("metadata") or {}
    first_ingest_metadata = ingest_records[0].get("metadata") or {}
    if first_chunk_metadata.get("schema_version") != CHUNK_EXPORT_SCHEMA_VERSION:
        raise RuntimeError("chunks.jsonl record missing chunk schema version")
    if first_ingest_metadata.get("schema_version") != INGEST_EXPORT_SCHEMA_VERSION:
        raise RuntimeError("ingest.jsonl record missing ingest schema version")
    if first_ingest_metadata.get("chunk_schema_version") != CHUNK_EXPORT_SCHEMA_VERSION:
        raise RuntimeError("ingest.jsonl record missing chunk schema version")
    table_chunk_records = [record for record in chunk_records if record.get("block_refs") == ["block-0002"]]
    if not table_chunk_records or not table_chunk_records[0].get("assets"):
        raise RuntimeError("chunks.jsonl table record missing asset view")
    table_asset_view_metadata = (table_chunk_records[0]["assets"][0].get("metadata") or {})
    if table_asset_view_metadata.get("asset_context_schema_version") != ASSET_CONTEXT_SCHEMA_VERSION:
        raise RuntimeError("chunks.jsonl table asset view missing asset context schema version")
    if table_asset_view_metadata.get("asset_summary_schema_version") != ASSET_SUMMARY_SCHEMA_VERSION:
        raise RuntimeError("chunks.jsonl table asset view missing asset summary schema version")

    return {
        "parse_id": document.parse_id,
        "root_dir": str(root_dir),
        "chunk_count": len(chunks),
        "chunk_record_count": len(chunk_records),
        "asset_context_schema_version": table_asset_metadata.get("asset_context_schema_version"),
        "asset_summary_schema_version": table_asset_metadata.get("asset_summary_schema_version"),
        "ingest_record_count": len(ingest_records),
        "chunk_schema_version": first_chunk_metadata.get("schema_version"),
        "ingest_schema_version": first_ingest_metadata.get("schema_version"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DocPilot artifact/chunk/ingest export without starting the API.")
    parser.add_argument("--output-dir", help="Optional artifact root to keep after the smoke run.")
    args = parser.parse_args()

    if args.output_dir:
        root_dir = Path(args.output_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        summary = _run_smoke(root_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="deepdoc-artifact-smoke-") as temp_dir:
            summary = _run_smoke(Path(temp_dir))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

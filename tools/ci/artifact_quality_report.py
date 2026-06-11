#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CHUNK_EXPORT_SCHEMA_VERSION = "2026-06-08.chunk.v1"
INGEST_EXPORT_SCHEMA_VERSION = "2026-06-08.ingest.v1"
ASSET_CONTEXT_SCHEMA_VERSION = "2026-06-08.asset-context.v1"
ASSET_SUMMARY_SCHEMA_VERSION = "2026-06-08.asset-summary.v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        normalized = line.strip()
        if not normalized:
            continue
        row = json.loads(normalized)
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def _find_parse_dirs(root: Path) -> list[Path]:
    if (root / "structured.json").is_file():
        return [root]
    return sorted(path.parent for path in root.glob("*/structured.json") if path.is_file())


def _failure(failures: list[str], parse_label: str, message: str) -> None:
    failures.append(f"{parse_label}: {message}")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _metadata(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _validate_asset_context(
    *,
    failures: list[str],
    parse_label: str,
    asset_label: str,
    metadata: dict[str, Any],
    known_block_ids: set[str],
    known_chunk_ids: set[str],
) -> None:
    if metadata.get("asset_context_schema_version") != ASSET_CONTEXT_SCHEMA_VERSION:
        _failure(failures, parse_label, f"{asset_label}: asset context schema_version mismatch")

    for key in ("direct_block_refs", "context_block_refs", "direct_chunk_refs", "context_chunk_refs", "chunk_refs"):
        if not isinstance(metadata.get(key), list):
            _failure(failures, parse_label, f"{asset_label}: asset context {key} is missing or not a list")

    for key in ("direct_block_refs", "context_block_refs"):
        unknown_refs = [
            ref for ref in _as_list(metadata.get(key))
            if str(ref or "") and str(ref) not in known_block_ids
        ]
        if unknown_refs:
            _failure(failures, parse_label, f"{asset_label}: unknown asset context {key} {unknown_refs}")

    for key in ("direct_chunk_refs", "context_chunk_refs", "chunk_refs"):
        unknown_refs = [
            ref for ref in _as_list(metadata.get(key))
            if str(ref or "") and str(ref) not in known_chunk_ids
        ]
        if unknown_refs:
            _failure(failures, parse_label, f"{asset_label}: unknown asset context {key} {unknown_refs}")

    if not isinstance(metadata.get("context_texts"), list):
        _failure(failures, parse_label, f"{asset_label}: asset context context_texts is missing or not a list")


def _validate_asset_summary(
    *,
    failures: list[str],
    parse_label: str,
    asset_label: str,
    metadata: dict[str, Any],
) -> None:
    if metadata.get("asset_summary_schema_version") != ASSET_SUMMARY_SCHEMA_VERSION:
        _failure(failures, parse_label, f"{asset_label}: asset summary schema_version mismatch")
    if metadata.get("asset_summary_source") != "local_rules":
        _failure(failures, parse_label, f"{asset_label}: asset summary source mismatch")
    if not str(metadata.get("asset_summary") or "").strip():
        _failure(failures, parse_label, f"{asset_label}: asset summary is missing or empty")
    if not isinstance(metadata.get("asset_summary_facts"), dict):
        _failure(failures, parse_label, f"{asset_label}: asset summary facts is missing or not an object")


def _evaluate_parse_dir(parse_dir: Path) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    structured_path = parse_dir / "structured.json"
    manifest_path = parse_dir / "manifest.json"
    chunks_path = parse_dir / "chunks.jsonl"
    ingest_path = parse_dir / "ingest.jsonl"
    parse_label = parse_dir.name

    try:
        structured = _load_json(structured_path)
    except Exception as exc:
        return (
            {
                "parse_dir": str(parse_dir),
                "parse_id": parse_label,
                "block_count": 0,
                "asset_count": 0,
                "chunk_count": 0,
                "chunk_record_count": 0,
                "ingest_record_count": 0,
            },
            [f"{parse_label}: failed to load structured.json: {exc}"],
        )

    document = _metadata(structured.get("document"))
    parse_id = str(document.get("parse_id") or parse_label)
    parse_label = parse_id
    blocks = _as_list(structured.get("blocks"))
    assets = _as_list(structured.get("assets"))
    chunks = _as_list(structured.get("chunks"))
    known_block_ids = {
        str(block.get("block_id") or "")
        for block in blocks
        if isinstance(block, dict) and str(block.get("block_id") or "")
    }
    known_asset_ids = {
        str(asset.get("asset_id") or "")
        for asset in assets
        if isinstance(asset, dict) and str(asset.get("asset_id") or "")
    }
    known_chunk_ids = {
        str(chunk.get("chunk_id") or "")
        for chunk in chunks
        if isinstance(chunk, dict) and str(chunk.get("chunk_id") or "")
    }

    if not manifest_path.is_file():
        _failure(failures, parse_label, "missing manifest.json")
        manifest: dict[str, Any] = {}
    else:
        try:
            manifest = _load_json(manifest_path)
        except Exception as exc:
            _failure(failures, parse_label, f"failed to load manifest.json: {exc}")
            manifest = {}

    if manifest:
        manifest_parse_id = str(manifest.get("parse_id") or "")
        if manifest_parse_id and manifest_parse_id != parse_id:
            _failure(failures, parse_label, f"manifest parse_id mismatch: {manifest_parse_id}")
        manifest_chunk_count = manifest.get("chunk_count")
        if isinstance(manifest_chunk_count, int) and manifest_chunk_count != len(chunks):
            _failure(failures, parse_label, f"manifest chunk_count mismatch: {manifest_chunk_count} != {len(chunks)}")

    for asset in assets:
        if not isinstance(asset, dict):
            _failure(failures, parse_label, "structured asset is not an object")
            continue
        asset_id = str(asset.get("asset_id") or "<missing>")
        _validate_asset_context(
            failures=failures,
            parse_label=parse_label,
            asset_label=asset_id,
            metadata=_metadata(asset.get("metadata")),
            known_block_ids=known_block_ids,
            known_chunk_ids=known_chunk_ids,
        )
        _validate_asset_summary(
            failures=failures,
            parse_label=parse_label,
            asset_label=asset_id,
            metadata=_metadata(asset.get("metadata")),
        )

    for chunk in chunks:
        if not isinstance(chunk, dict):
            _failure(failures, parse_label, "structured chunk is not an object")
            continue
        chunk_id = str(chunk.get("chunk_id") or "<missing>")
        if not str(chunk.get("text") or "").strip():
            _failure(failures, parse_label, f"{chunk_id}: empty chunk text")
        unknown_blocks = [
            ref for ref in _as_list(chunk.get("block_refs"))
            if str(ref or "") and str(ref) not in known_block_ids
        ]
        if unknown_blocks:
            _failure(failures, parse_label, f"{chunk_id}: unknown block_refs {unknown_blocks}")
        unknown_assets = [
            ref for ref in _as_list(chunk.get("asset_refs"))
            if str(ref or "") and str(ref) not in known_asset_ids
        ]
        if unknown_assets:
            _failure(failures, parse_label, f"{chunk_id}: unknown asset_refs {unknown_assets}")
        if not str(_metadata(chunk.get("metadata")).get("chunk_strategy") or "").strip():
            _failure(failures, parse_label, f"{chunk_id}: missing chunk_strategy")

    if not chunks_path.is_file():
        _failure(failures, parse_label, "missing chunks.jsonl")
        chunk_records: list[dict[str, Any]] = []
    else:
        try:
            chunk_records = _load_jsonl(chunks_path)
        except Exception as exc:
            _failure(failures, parse_label, f"failed to load chunks.jsonl: {exc}")
            chunk_records = []

    if chunk_records and len(chunk_records) != len(chunks):
        _failure(failures, parse_label, f"chunks.jsonl count mismatch: {len(chunk_records)} != {len(chunks)}")
    for record in chunk_records:
        chunk_id = str(record.get("chunk_id") or "<missing>")
        metadata = _metadata(record.get("metadata"))
        if not str(record.get("text") or "").strip():
            _failure(failures, parse_label, f"{chunk_id}: empty chunk export text")
        if metadata.get("schema_version") != CHUNK_EXPORT_SCHEMA_VERSION:
            _failure(failures, parse_label, f"{chunk_id}: chunk schema_version mismatch")
        if not str(metadata.get("chunk_strategy") or "").strip():
            _failure(failures, parse_label, f"{chunk_id}: chunk export missing chunk_strategy")
        for asset in _as_list(record.get("assets")):
            if not isinstance(asset, dict):
                _failure(failures, parse_label, f"{chunk_id}: chunk export asset is not an object")
                continue
            asset_id = str(asset.get("asset_id") or "<missing>")
            _validate_asset_context(
                failures=failures,
                parse_label=parse_label,
                asset_label=f"{chunk_id}:{asset_id}",
                metadata=_metadata(asset.get("metadata")),
                known_block_ids=known_block_ids,
                known_chunk_ids=known_chunk_ids,
            )
            _validate_asset_summary(
                failures=failures,
                parse_label=parse_label,
                asset_label=f"{chunk_id}:{asset_id}",
                metadata=_metadata(asset.get("metadata")),
            )

    if not ingest_path.is_file():
        _failure(failures, parse_label, "missing ingest.jsonl")
        ingest_records: list[dict[str, Any]] = []
    else:
        try:
            ingest_records = _load_jsonl(ingest_path)
        except Exception as exc:
            _failure(failures, parse_label, f"failed to load ingest.jsonl: {exc}")
            ingest_records = []

    if ingest_records and len(ingest_records) != len(chunks):
        _failure(failures, parse_label, f"ingest.jsonl count mismatch: {len(ingest_records)} != {len(chunks)}")
    for record in ingest_records:
        record_id = str(record.get("record_id") or record.get("chunk_id") or "<missing>")
        metadata = _metadata(record.get("metadata"))
        if not str(record.get("text") or "").strip():
            _failure(failures, parse_label, f"{record_id}: empty ingest text")
        if metadata.get("schema_version") != INGEST_EXPORT_SCHEMA_VERSION:
            _failure(failures, parse_label, f"{record_id}: ingest schema_version mismatch")
        if metadata.get("chunk_schema_version") != CHUNK_EXPORT_SCHEMA_VERSION:
            _failure(failures, parse_label, f"{record_id}: ingest chunk_schema_version mismatch")

    asset_linked_chunks = [
        chunk for chunk in chunks
        if isinstance(chunk, dict) and _as_list(chunk.get("asset_refs"))
    ]
    summary = {
        "parse_dir": str(parse_dir),
        "parse_id": parse_id,
        "filename": str(document.get("filename") or ""),
        "file_type": str(document.get("file_type") or ""),
        "parser_engine": str(document.get("parser_engine") or ""),
        "block_count": len(blocks),
        "asset_count": len(assets),
        "chunk_count": len(chunks),
        "chunk_record_count": len(chunk_records),
        "ingest_record_count": len(ingest_records),
        "asset_linked_chunk_count": len(asset_linked_chunks),
    }
    return summary, failures


def evaluate_artifact_root(root: str | Path) -> dict[str, Any]:
    artifact_root = Path(root)
    failures: list[str] = []
    if not artifact_root.exists():
        return {
            "status": "failed",
            "artifact_root": str(artifact_root),
            "parse_count": 0,
            "totals": {
                "block_count": 0,
                "asset_count": 0,
                "chunk_count": 0,
                "chunk_record_count": 0,
                "ingest_record_count": 0,
                "asset_linked_chunk_count": 0,
            },
            "artifacts": [],
            "failures": [f"{artifact_root}: artifact root does not exist"],
        }

    parse_dirs = _find_parse_dirs(artifact_root)
    if not parse_dirs:
        failures.append(f"{artifact_root}: no structured.json files found")

    artifacts: list[dict[str, Any]] = []
    for parse_dir in parse_dirs:
        summary, parse_failures = _evaluate_parse_dir(parse_dir)
        artifacts.append(summary)
        failures.extend(parse_failures)

    totals = {
        "block_count": sum(int(item.get("block_count") or 0) for item in artifacts),
        "asset_count": sum(int(item.get("asset_count") or 0) for item in artifacts),
        "chunk_count": sum(int(item.get("chunk_count") or 0) for item in artifacts),
        "chunk_record_count": sum(int(item.get("chunk_record_count") or 0) for item in artifacts),
        "ingest_record_count": sum(int(item.get("ingest_record_count") or 0) for item in artifacts),
        "asset_linked_chunk_count": sum(int(item.get("asset_linked_chunk_count") or 0) for item in artifacts),
    }
    return {
        "status": "failed" if failures else "passed",
        "artifact_root": str(artifact_root),
        "parse_count": len(artifacts),
        "totals": totals,
        "artifacts": artifacts,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DocPilot structured/chunks/ingest artifacts for business usability.")
    parser.add_argument("artifact_root", help="Artifact root or a single parse artifact directory containing structured.json.")
    parser.add_argument("--output", help="Optional path for the JSON report.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit 1 when quality failures are found.")
    args = parser.parse_args()

    report = evaluate_artifact_root(args.artifact_root)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.fail_on_error and report["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

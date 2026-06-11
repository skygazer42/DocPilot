#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.model_store import (
    MODEL_GROUP_FILES,
    get_group_files,
    get_model_group_provenance,
    get_model_repo,
    get_model_root,
    list_missing_files,
    validate_ocr_dictionaries,
    validate_ocr_recognition_model_alignments,
)


def _model_groups_help() -> str:
    groups = ", ".join(sorted(MODEL_GROUP_FILES))
    return f"Model groups to verify. Supported groups: {groups}, comma-list, or all."


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_local_entries(model_root: Path, groups: str) -> dict[str, dict[str, Any]]:
    missing = list_missing_files(groups, model_root=str(model_root))
    if missing:
        raise RuntimeError(f"local model root is missing required files: {', '.join(missing)}")
    entries: dict[str, dict[str, Any]] = {}
    for relative_path in get_group_files(groups):
        path = model_root / relative_path
        entries[relative_path] = {
            "path": relative_path,
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return entries


def _load_remote_manifest(repo_id: str, token: str | None) -> dict[str, Any]:
    manifest_path = hf_hub_download(repo_id=repo_id, repo_type="model", filename="manifest.json", token=token)
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise RuntimeError("remote manifest.json is not a JSON object")
    return manifest


def _remote_sizes(api: HfApi, repo_id: str, paths: list[str], token: str | None) -> dict[str, int | None]:
    sizes: dict[str, int | None] = {}
    for info in api.get_paths_info(repo_id, paths=paths, repo_type="model", token=token):
        path = getattr(info, "path", None)
        if not path:
            continue
        sizes[path] = getattr(info, "size", None)
    return sizes


def _model_group_provenance_problems(remote_manifest: dict[str, Any], groups: str) -> list[str]:
    expected = get_model_group_provenance(groups)
    actual = remote_manifest.get("model_group_provenance")
    problems: list[str] = []
    if not isinstance(actual, dict):
        return [
            f"missing model_group_provenance for group: {group}"
            for group in sorted(expected)
        ]

    for group, expected_entry in sorted(expected.items()):
        actual_entry = actual.get(group)
        if not isinstance(actual_entry, dict):
            problems.append(f"missing model_group_provenance for group: {group}")
            continue
        for field, expected_value in sorted(expected_entry.items()):
            actual_value = actual_entry.get(field)
            if actual_value != expected_value:
                problems.append(
                    "model_group_provenance mismatch for "
                    f"{group}.{field}: expected {expected_value!r}, got {actual_value!r}"
                )
    return problems


def verify(repo_id: str, model_root: Path, groups: str, remote_only: bool) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    api = HfApi(token=token or None)
    required_paths = get_group_files(groups)
    local_entries = {} if remote_only else _build_local_entries(model_root, groups)
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", token=token or None))
    missing_remote_files = [path for path in required_paths if path not in remote_files]
    remote_manifest = _load_remote_manifest(repo_id, token or None)
    model_group_provenance_problems = _model_group_provenance_problems(remote_manifest, groups)
    remote_entries = {
        item.get("path"): item
        for item in remote_manifest.get("files", [])
        if isinstance(item, dict) and item.get("path")
    }
    missing_manifest_entries = [path for path in required_paths if path not in remote_entries]
    remote_sizes = _remote_sizes(api, repo_id, required_paths, token or None)
    missing_remote_metadata = [path for path in required_paths if path not in remote_sizes]

    size_mismatches = []
    for path in required_paths:
        manifest_size = remote_entries.get(path, {}).get("size_bytes")
        metadata_size = remote_sizes.get(path)
        if manifest_size is not None and metadata_size is not None and manifest_size != metadata_size:
            size_mismatches.append(
                {
                    "path": path,
                    "manifest_size": manifest_size,
                    "remote_metadata_size": metadata_size,
                }
            )

    local_mismatches = []
    if not remote_only:
        for path, local_entry in local_entries.items():
            remote_entry = remote_entries.get(path) or {}
            if (
                local_entry.get("size_bytes") != remote_entry.get("size_bytes")
                or local_entry.get("sha256") != remote_entry.get("sha256")
            ):
                local_mismatches.append(
                    {
                        "path": path,
                        "local_size": local_entry.get("size_bytes"),
                        "remote_size": remote_entry.get("size_bytes"),
                        "local_sha256": local_entry.get("sha256"),
                        "remote_sha256": remote_entry.get("sha256"),
                    }
                )
    ocr_dictionaries = {}
    ocr_dictionary = None
    ocr_dictionary_problems = []
    ocr_recognition_alignments = {}
    ocr_recognition_alignment_problems = []
    ocr_dictionary_paths = [
        path
        for path in required_paths
        if Path(path).name.startswith("ocr") and Path(path).name.endswith(".res")
    ]
    if not remote_only and ocr_dictionary_paths:
        ocr_dictionaries = validate_ocr_dictionaries(
            model_root=model_root,
            relative_paths=ocr_dictionary_paths,
            expected_sha256_by_path={
                path: (remote_entries.get(path) or {}).get("sha256")
                for path in ocr_dictionary_paths
            },
        )
        ocr_dictionary = ocr_dictionaries.get("ocr.res")
        for path, report in ocr_dictionaries.items():
            if report.get("status") != "ok":
                ocr_dictionary_problems.extend(
                    f"{path}: {problem}"
                    for problem in (report.get("problems") or [])
                )
    rec_dictionary_pairs = {
        rec_model_path: dictionary_path
        for rec_model_path, dictionary_path in {
            "rec.onnx": "ocr.res",
            "rec_v5.onnx": "ocr_v5.res",
        }.items()
        if rec_model_path in required_paths and dictionary_path in required_paths
    }
    if not remote_only and rec_dictionary_pairs:
        ocr_recognition_alignments = validate_ocr_recognition_model_alignments(
            model_root=model_root,
            pairs=rec_dictionary_pairs,
        )
        for path, report in ocr_recognition_alignments.items():
            if report.get("status") != "ok":
                ocr_recognition_alignment_problems.extend(
                    f"{path}: {problem}"
                    for problem in (report.get("problems") or [])
                )

    problems = {
        "missing_remote_files": missing_remote_files,
        "missing_manifest_entries": missing_manifest_entries,
        "missing_remote_metadata": missing_remote_metadata,
        "size_mismatches": size_mismatches,
        "local_mismatches": local_mismatches,
        "model_group_provenance_problems": model_group_provenance_problems,
        "ocr_dictionary_problems": ocr_dictionary_problems,
        "ocr_recognition_alignment_problems": ocr_recognition_alignment_problems,
    }
    status = "ok" if all(not value for value in problems.values()) else "failed"
    return {
        "status": status,
        "repo_id": repo_id,
        "groups": groups,
        "remote_only": remote_only,
        "required_file_count": len(required_paths),
        "remote_file_count": len(remote_files),
        "remote_manifest_groups": remote_manifest.get("groups"),
        "remote_manifest_file_count": len(remote_entries),
        "remote_manifest_model_group_provenance": remote_manifest.get("model_group_provenance"),
        "ocr_dictionary": ocr_dictionary,
        "ocr_dictionaries": ocr_dictionaries,
        "ocr_recognition_alignments": ocr_recognition_alignments,
        **problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DeepDoc model files in a Hugging Face model repository.")
    parser.add_argument("--groups", default="all", help=_model_groups_help())
    parser.add_argument("--repo-id", default=get_model_repo(), help="Hugging Face model repository.")
    parser.add_argument("--model-root", default=get_model_root(), help="Local model root used for sha256 comparison.")
    parser.add_argument(
        "--remote-only",
        action="store_true",
        help="Only verify remote files/manifest metadata; skip local sha256 comparison.",
    )
    args = parser.parse_args()

    try:
        result = verify(args.repo_id, Path(args.model_root).resolve(), args.groups, args.remote_only)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

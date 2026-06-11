#!/usr/bin/env python

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import LocalTokenNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.model_store import (
    MODEL_GROUP_FILES,
    get_group_files,
    get_model_group_provenance,
    get_model_repo,
    get_model_root,
    list_missing_files,
)


def _model_groups_help() -> str:
    groups = ", ".join(sorted(MODEL_GROUP_FILES))
    return f"Comma-separated model groups to publish. Supported groups: {groups}, or all."


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_manifest(model_root: str, groups: str, repo_id: str | None = None) -> dict:
    root = Path(model_root)
    files = []
    for rel_path in get_group_files(groups):
        file_path = root / rel_path
        if not file_path.exists():
            continue
        files.append(
            {
                "path": rel_path,
                "size_bytes": file_path.stat().st_size,
                "sha256": _file_sha256(file_path),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_root": str(root),
        "repo_id": repo_id or get_model_repo(),
        "groups": groups,
        "model_group_provenance": get_model_group_provenance(groups),
        "files": files,
    }


def write_manifest(model_root: str, manifest: dict) -> Path:
    manifest_path = Path(model_root) / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def publish(groups: str, repo_id: str, write_manifest_only: bool) -> Path:
    model_root = get_model_root()
    missing = list_missing_files(groups, model_root=model_root)
    if missing:
        raise SystemExit(
            "Cannot publish because local model files are missing: "
            + ", ".join(missing)
        )

    manifest = build_manifest(model_root, groups, repo_id=repo_id)
    manifest_path = write_manifest(model_root, manifest)
    print(f"Manifest written to {manifest_path}")

    if write_manifest_only:
        return manifest_path

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    api = HfApi(token=token or None)
    try:
        api.whoami()
    except LocalTokenNotFoundError:
        raise SystemExit(
            "HF auth missing. Set HF_TOKEN or run `hf auth login` before publishing."
        )
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    allow_patterns = sorted(
        set(get_group_files(groups)).union({"manifest.json", "README.md", ".gitattributes"})
    )
    api.upload_folder(
        folder_path=model_root,
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=allow_patterns,
        commit_message=f"Upload DocPilot model groups: {groups}",
    )
    print(f"Uploaded model groups {groups} to {repo_id}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Publish local DocPilot models to Hugging Face.")
    parser.add_argument(
        "--groups",
        default="core",
        help=_model_groups_help(),
    )
    parser.add_argument(
        "--repo-id",
        default=get_model_repo(),
        help="Target Hugging Face model repository.",
    )
    parser.add_argument(
        "--write-manifest-only",
        action="store_true",
        help="Only write manifest.json locally, do not upload.",
    )
    args = parser.parse_args()
    publish(args.groups, args.repo_id, args.write_manifest_only)


if __name__ == "__main__":
    main()

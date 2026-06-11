from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from common import setting

BUILD_HASH_PATHS: tuple[str, ...] = (
    "common",
    "deepdoc",
    "docker",
    "main.py",
    "gradio_app.py",
    "download_models.py",
    "openapi.json",
    "pyproject.toml",
)


def _read_json_file(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(base_dir: Path, included_paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    seen: set[str] = set()
    for relative in sorted(included_paths):
        target = base_dir / relative
        if not target.exists():
            continue
        candidates: list[Path]
        if target.is_dir():
            candidates = sorted(path for path in target.rglob("*") if path.is_file())
        else:
            candidates = [target]
        for candidate in candidates:
            rel_path = candidate.relative_to(base_dir).as_posix()
            if rel_path in seen:
                continue
            seen.add(rel_path)
            digest.update(rel_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_file(candidate).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _parse_iso_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _short_hash(value: object, *, length: int = 12) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:length]


def _runtime_fallback_info() -> dict[str, object]:
    base_dir = Path(setting.BASE_DIR)
    pyproject_path = base_dir / "pyproject.toml"
    openapi_path = base_dir / "openapi.json"
    build_timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "status": "degraded",
        "embedded": False,
        "error": "embedded build metadata file is missing",
        "package_name": "deepdoc",
        "package_version": os.environ.get("DEEPDOC_BUILD_VERSION", "0.1.0"),
        "build_timestamp": build_timestamp,
        "build_source": "runtime-fallback",
        "image_tag": os.environ.get("DEEPDOC_BUILD_IMAGE_TAG") or None,
        "vcs_ref": os.environ.get("DEEPDOC_BUILD_VCS_REF") or None,
        "pip_extras": os.environ.get("DEEPDOC_PIP_EXTRAS") or None,
        "pyproject_sha256": _sha256_file(pyproject_path) if pyproject_path.exists() else None,
        "openapi_sha256": _sha256_file(openapi_path) if openapi_path.exists() else None,
        "source_tree_sha256": _sha256_tree(base_dir, BUILD_HASH_PATHS),
        "requirements_sha256": None,
        "base_dir": str(base_dir),
    }


@lru_cache(maxsize=1)
def _load_embedded_build_info() -> dict[str, object]:
    build_info_path = Path(setting.BUILD_INFO_PATH)
    payload = _read_json_file(build_info_path)
    if payload is None:
        fallback = _runtime_fallback_info()
        fallback["build_info_path"] = str(build_info_path)
        return fallback
    payload = deepcopy(payload)
    payload.setdefault("status", "ok")
    payload.setdefault("embedded", True)
    payload.setdefault("build_info_path", str(build_info_path))
    return payload


def get_build_info() -> dict[str, object]:
    payload = deepcopy(_load_embedded_build_info())
    build_timestamp = _parse_iso_datetime(payload.get("build_timestamp"))
    if build_timestamp is not None:
        payload["build_timestamp"] = build_timestamp.isoformat()
        payload["build_age_seconds"] = max(0.0, (datetime.now(timezone.utc) - build_timestamp).total_seconds())
        payload["build_timestamp_epoch_seconds"] = int(build_timestamp.timestamp())
    else:
        payload["build_age_seconds"] = None
        payload["build_timestamp_epoch_seconds"] = None
    payload["runtime"] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "service_name": os.environ.get("DEEPDOC_TRACING_SERVICE_NAME", "deepdoc-standalone"),
        "deployment_environment": os.environ.get("DEEPDOC_TRACING_DEPLOYMENT_ENVIRONMENT") or None,
    }
    payload["summary"] = summarize_build_info(payload)
    return payload


def summarize_build_info(payload: dict[str, object] | None) -> dict[str, object]:
    info = payload if isinstance(payload, dict) else {}
    build_timestamp = str(info.get("build_timestamp") or "").strip() or None
    return {
        "status": str(info.get("status") or "unknown").strip() or "unknown",
        "embedded": bool(info.get("embedded")),
        "package_name": str(info.get("package_name") or "deepdoc").strip() or "deepdoc",
        "package_version": str(info.get("package_version") or "0.1.0").strip() or "0.1.0",
        "build_timestamp": build_timestamp,
        "build_age_seconds": info.get("build_age_seconds"),
        "build_source": str(info.get("build_source") or "").strip() or None,
        "image_tag": str(info.get("image_tag") or "").strip() or None,
        "vcs_ref": str(info.get("vcs_ref") or "").strip() or None,
        "vcs_ref_short": _short_hash(info.get("vcs_ref")),
        "source_tree_sha": str(info.get("source_tree_sha256") or "").strip() or None,
        "source_tree_sha12": _short_hash(info.get("source_tree_sha256")),
        "requirements_sha12": _short_hash(info.get("requirements_sha256")),
        "pyproject_sha12": _short_hash(info.get("pyproject_sha256")),
        "openapi_sha12": _short_hash(info.get("openapi_sha256")),
        "pip_extras": str(info.get("pip_extras") or "").strip() or None,
        "build_info_path": str(info.get("build_info_path") or "").strip() or None,
        "error": str(info.get("error") or "").strip() or None,
    }

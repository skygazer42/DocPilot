import os
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
from typing import Iterable, Any

from huggingface_hub import snapshot_download

from common import logger
from common import setting

DEFAULT_MODEL_REPO = "qwqqwq/deepdoc-standalone"
OPTIONAL_METADATA_FILES = {"README.md", ".gitattributes", "manifest.json"}
MODEL_MANIFEST_SCHEMA_VERSION = "2026-06-08.cpu-pipeline-model-manifest.v1"
DEFAULT_OCR_REQUIRED_CHARACTERS = tuple(
    "中国政务人民法院公司一年"
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)
MODEL_GROUP_FILES: dict[str, tuple[str, ...]] = {
    "core": (
        "det.onnx",
        "rec.onnx",
        "ocr.res",
        "layout.onnx",
        "layout.manual.onnx",
        "layout.paper.onnx",
        "layout.laws.onnx",
        "tsr.onnx",
        "updown_concat_xgb.model",
    ),
    "core_v5": (
        "det_v5.onnx",
        "rec_v5.onnx",
        "ocr_v5.res",
    ),
    "layout_v2": (
        "layout/pp_doclayout_plus.onnx",
    ),
    "formula": (
        "formula/image_resizer.onnx",
        "formula/encoder.onnx",
        "formula/decoder.onnx",
        "formula/tokenizer.json",
    ),
    "formula_v2": (
        "formula/config.json",
        "formula/inference.json",
        "formula/inference.pdiparams",
        "formula/inference.yml",
        "formula/pp_formula_net_s.onnx",
    ),
    "seal": ("seal/seal_det.onnx",),
    "handwriting": ("rec_handwriting.onnx",),
    # RapidTable(SLANet-plus, 纯 ONNX/CPU) 表格识别模型组。
    # 实际文件名以 plan Task 0 探针核验的发布产物为准。
    "table_v2": (
        "table/slanet_plus.onnx",
        "table/table_cls.onnx",
    ),
}
PUBLISHED_MODEL_GROUPS: tuple[str, ...] = (
    "core",
    "core_v5",
    "layout_v2",
    "formula",
    "formula_v2",
    "seal",
    "table_v2",
)
MODEL_GROUP_PROVENANCE: dict[str, dict[str, Any]] = {
    "core": {
        "component": "DeepDoc legacy local parser models",
        "license": "Apache-2.0",
        "license_status": "allowed",
        "scope": "document_parser",
        "default_enabled": True,
        "switch": "DEEPDOC_OCR_VERSION=v4",
        "readiness_gate": "core",
    },
    "core_v5": {
        "component": "PP-OCRv5",
        "license": "Apache-2.0",
        "license_status": "allowed",
        "scope": "document_parser",
        "default_enabled": False,
        "switch": "DEEPDOC_OCR_VERSION=v5",
        "readiness_gate": "ocr_v5",
    },
    "layout_v2": {
        "component": "PP-DocLayout",
        "license": "Apache-2.0",
        "license_status": "allowed",
        "scope": "document_parser",
        "default_enabled": False,
        "switch": "DEEPDOC_LAYOUT_ENGINE=ppdoclayout",
        "readiness_gate": "layout_v2",
    },
    "formula": {
        "component": "RapidLaTeXOCR",
        "license": "Apache-2.0",
        "license_status": "allowed",
        "scope": "document_parser",
        "default_enabled": False,
        "switch": "enable_formula=true",
        "readiness_gate": "formula",
    },
    "formula_v2": {
        "component": "PP-FormulaNet-S",
        "license": "Apache-2.0",
        "license_status": "allowed",
        "scope": "document_parser",
        "default_enabled": False,
        "switch": "DEEPDOC_FORMULA_MODE=pp_formula_net_s",
        "readiness_gate": "formula_v2",
    },
    "seal": {
        "component": "DeepDoc seal detector",
        "license": "Apache-2.0",
        "license_status": "allowed",
        "scope": "document_parser",
        "default_enabled": False,
        "switch": "enable_seal=true",
        "readiness_gate": "seal",
    },
    "handwriting": {
        "component": "DeepDoc handwriting fallback recognizer",
        "license": "Apache-2.0",
        "license_status": "allowed",
        "scope": "document_parser",
        "default_enabled": False,
        "switch": "DEEPDOC_HANDWRITING_FALLBACK=1",
        "readiness_gate": "handwriting",
    },
    "table_v2": {
        "component": "SLANet-plus",
        "license": "Apache-2.0",
        "license_status": "allowed",
        "scope": "document_parser",
        "default_enabled": False,
        "switch": "DEEPDOC_TABLE_ENGINE=rapidtable",
        "readiness_gate": "table_v2",
    },
}
OCR_RECOGNITION_DICTIONARY_PAIRS: dict[str, str] = {
    "rec.onnx": "ocr.res",
    "rec_v5.onnx": "ocr_v5.res",
}


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest_expected_sha256(model_root: Path, relative_path: str) -> str | None:
    manifest_path = model_root / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load model manifest from %s", manifest_path)
        return None
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, list):
        return None
    for item in files:
        if not isinstance(item, dict):
            continue
        if item.get("path") == relative_path and item.get("sha256"):
            return str(item.get("sha256"))
    return None


def _read_ocr_dictionary_entries(path: Path) -> list[str]:
    entries: list[str] = []
    with path.open("rb") as handle:
        for raw_line in handle.readlines():
            entries.append(raw_line.decode("utf-8").strip("\n").strip("\r\n"))
    return entries


def _get_onnx_output_class_count(path: str | Path) -> int | None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.get_outputs()
    if not outputs:
        return None
    shape = list(outputs[0].shape or [])
    if not shape:
        return None
    class_count = shape[-1]
    if isinstance(class_count, int) and class_count > 0:
        return int(class_count)
    return None


def validate_ocr_dictionary(
    *,
    model_root: str | Path | None = None,
    relative_path: str = "ocr.res",
    expected_sha256: str | None = None,
    required_characters: str | Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(model_root or get_model_root())
    path = root / relative_path
    required = [
        str(character)
        for character in (
            DEFAULT_OCR_REQUIRED_CHARACTERS
            if required_characters is None
            else required_characters
        )
        if str(character)
    ]
    problems: list[str] = []

    if not path.exists():
        return {
            "status": "failed",
            "path": relative_path,
            "absolute_path": str(path),
            "exists": False,
            "size_bytes": 0,
            "sha256": None,
            "expected_sha256": expected_sha256,
            "sha256_matches": False,
            "line_count": 0,
            "unique_character_count": 0,
            "required_characters": required,
            "missing_required_characters": required,
            "duplicate_characters": [],
            "empty_line_count": 0,
            "problems": [f"missing OCR dictionary file: {relative_path}"],
        }

    entries = _read_ocr_dictionary_entries(path)
    non_empty_entries = [entry for entry in entries if entry]
    entry_set = set(non_empty_entries)
    duplicate_entries = sorted({entry for entry in non_empty_entries if non_empty_entries.count(entry) > 1})
    missing_required_characters = [character for character in required if character not in entry_set]
    actual_sha256 = _file_sha256(path)
    resolved_expected_sha256 = expected_sha256 or _load_manifest_expected_sha256(root, relative_path)
    sha256_matches = bool(resolved_expected_sha256 and resolved_expected_sha256 == actual_sha256)

    if resolved_expected_sha256 and not sha256_matches:
        problems.append("sha256 mismatch")
    if missing_required_characters:
        problems.append(
            "missing required OCR characters: "
            + ", ".join(missing_required_characters)
        )
    if duplicate_entries:
        problems.append(
            "duplicate OCR dictionary entries: "
            + ", ".join(duplicate_entries[:20])
        )
    empty_line_count = len(entries) - len(non_empty_entries)
    if empty_line_count:
        problems.append(f"empty OCR dictionary lines: {empty_line_count}")

    return {
        "status": "failed" if problems else "ok",
        "path": relative_path,
        "absolute_path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha256,
        "expected_sha256": resolved_expected_sha256,
        "sha256_matches": sha256_matches if resolved_expected_sha256 else None,
        "line_count": len(non_empty_entries),
        "unique_character_count": len(entry_set),
        "required_characters": required,
        "missing_required_characters": missing_required_characters,
        "duplicate_characters": duplicate_entries,
        "empty_line_count": empty_line_count,
        "problems": problems,
    }


def _is_ocr_dictionary_path(relative_path: str) -> bool:
    name = Path(relative_path).name
    return name.startswith("ocr") and name.endswith(".res")


def get_ocr_dictionary_paths(groups: str | Iterable[str] | None = "all") -> list[str]:
    return [
        relative_path
        for relative_path in get_group_files(groups)
        if _is_ocr_dictionary_path(relative_path)
    ]


def validate_ocr_dictionaries(
    *,
    model_root: str | Path | None = None,
    relative_paths: Iterable[str] | None = None,
    expected_sha256_by_path: dict[str, str | None] | None = None,
    required_characters: str | Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    paths = sorted(
        {
            path
            for path in (relative_paths if relative_paths is not None else get_ocr_dictionary_paths("all"))
            if _is_ocr_dictionary_path(str(path))
        }
    )
    expected = expected_sha256_by_path or {}
    return {
        relative_path: validate_ocr_dictionary(
            model_root=model_root,
            relative_path=relative_path,
            expected_sha256=expected.get(relative_path),
            required_characters=required_characters,
        )
        for relative_path in paths
    }


def get_ocr_recognition_model_pairs(groups: str | Iterable[str] | None = "all") -> dict[str, str]:
    required_paths = set(get_group_files(groups))
    return {
        rec_model_path: dictionary_path
        for rec_model_path, dictionary_path in OCR_RECOGNITION_DICTIONARY_PAIRS.items()
        if rec_model_path in required_paths and dictionary_path in required_paths
    }


def validate_ocr_recognition_model_alignment(
    *,
    model_root: str | Path | None = None,
    rec_model_path: str = "rec.onnx",
    dictionary_path: str = "ocr.res",
    use_space_char: bool = True,
    ctc_blank: bool = True,
) -> dict[str, Any]:
    root = Path(model_root or get_model_root())
    model_path = root / rec_model_path
    dict_path = root / dictionary_path
    problems: list[str] = []

    dictionary_line_count = 0
    if not dict_path.exists():
        problems.append(f"missing OCR dictionary file: {dictionary_path}")
    else:
        dictionary_line_count = len([entry for entry in _read_ocr_dictionary_entries(dict_path) if entry])

    model_output_class_count: int | None = None
    if not model_path.exists():
        problems.append(f"missing OCR recognition model file: {rec_model_path}")
    else:
        try:
            model_output_class_count = _get_onnx_output_class_count(model_path)
        except Exception as exc:
            problems.append(f"failed to inspect OCR recognition model output: {exc}")

    expected_class_count = (
        dictionary_line_count
        + (1 if use_space_char else 0)
        + (1 if ctc_blank else 0)
    )
    class_count_matches = (
        model_output_class_count is not None
        and model_output_class_count == expected_class_count
    )

    if model_path.exists() and dict_path.exists() and not class_count_matches:
        problems.append(
            "OCR recognition class count mismatch: "
            f"{rec_model_path} outputs {model_output_class_count}, "
            f"expected {expected_class_count} from {dictionary_path} "
            f"({dictionary_line_count} dictionary entries"
            f"{' + space' if use_space_char else ''}"
            f"{' + CTC blank' if ctc_blank else ''})"
        )

    return {
        "status": "failed" if problems else "ok",
        "rec_model_path": rec_model_path,
        "dictionary_path": dictionary_path,
        "absolute_model_path": str(model_path),
        "absolute_dictionary_path": str(dict_path),
        "model_exists": model_path.exists(),
        "dictionary_exists": dict_path.exists(),
        "model_output_class_count": model_output_class_count,
        "dictionary_line_count": dictionary_line_count,
        "use_space_char": use_space_char,
        "ctc_blank": ctc_blank,
        "expected_class_count": expected_class_count,
        "class_count_matches": class_count_matches,
        "problems": problems,
    }


def validate_ocr_recognition_model_alignments(
    *,
    model_root: str | Path | None = None,
    pairs: dict[str, str] | None = None,
    use_space_char: bool = True,
    ctc_blank: bool = True,
) -> dict[str, dict[str, Any]]:
    resolved_pairs = pairs if pairs is not None else get_ocr_recognition_model_pairs("all")
    return {
        rec_model_path: validate_ocr_recognition_model_alignment(
            model_root=model_root,
            rec_model_path=rec_model_path,
            dictionary_path=dictionary_path,
            use_space_char=use_space_char,
            ctc_blank=ctc_blank,
        )
        for rec_model_path, dictionary_path in sorted(resolved_pairs.items())
    }


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_groups(groups: str | Iterable[str] | None) -> list[str]:
    if groups is None:
        return ["core"]
    if isinstance(groups, str):
        raw_items = [item.strip().lower() for item in groups.split(",")]
    else:
        raw_items = [str(item).strip().lower() for item in groups]

    normalized = [item for item in raw_items if item]
    if not normalized:
        return ["core"]
    if any(item in {"all", "*"} for item in normalized):
        return list(MODEL_GROUP_FILES.keys())

    expanded: list[str] = []
    for item in normalized:
        if item == "published":
            expanded.extend(PUBLISHED_MODEL_GROUPS)
        else:
            expanded.append(item)
    normalized = expanded

    unknown = [item for item in normalized if item not in MODEL_GROUP_FILES]
    if unknown:
        raise ValueError(
            f"Unknown model group(s): {', '.join(sorted(unknown))}. "
            f"Supported groups: {', '.join(sorted(MODEL_GROUP_FILES))}"
        )

    deduped: list[str] = []
    for item in normalized:
        if item not in deduped:
            deduped.append(item)
    return deduped


def get_model_root() -> str:
    model_root = os.environ.get("DEEPDOC_MODEL_PATH", setting.MODELS_DIR)
    return os.path.abspath(model_root)


def get_model_repo() -> str:
    repo = (os.environ.get("DEEPDOC_MODEL_REPO") or DEFAULT_MODEL_REPO).strip()
    return repo or DEFAULT_MODEL_REPO


def get_group_files(groups: str | Iterable[str] | None) -> list[str]:
    resolved_groups = _normalize_groups(groups)
    files: list[str] = []
    for group in resolved_groups:
        files.extend(MODEL_GROUP_FILES[group])
    return sorted(set(files))


def get_model_group_provenance(groups: str | Iterable[str] | None = "all") -> dict[str, dict[str, Any]]:
    resolved_groups = _normalize_groups(groups)
    return {
        group: dict(MODEL_GROUP_PROVENANCE[group])
        for group in resolved_groups
        if group in MODEL_GROUP_PROVENANCE
    }


def build_model_manifest(
    *,
    model_root: str | Path | None = None,
    groups: str | Iterable[str] | None = "all",
) -> dict[str, Any]:
    root = Path(model_root or get_model_root())
    resolved_groups = _normalize_groups(groups)
    path_to_groups: dict[str, list[str]] = {}
    for group in resolved_groups:
        for relative_path in MODEL_GROUP_FILES[group]:
            path_to_groups.setdefault(relative_path, []).append(group)

    files: list[dict[str, Any]] = []
    for relative_path in sorted(path_to_groups):
        path = root / relative_path
        entry: dict[str, Any] = {
            "path": relative_path,
            "groups": sorted(path_to_groups[relative_path]),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": _file_sha256(path) if path.exists() else None,
        }
        files.append(entry)

    return {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_root": str(root),
        "repo_id": get_model_repo(),
        "groups": resolved_groups,
        "model_group_provenance": get_model_group_provenance(resolved_groups),
        "files": files,
    }


def list_missing_files(
    groups: str | Iterable[str] | None,
    *,
    model_root: str | None = None,
) -> list[str]:
    root = Path(model_root or get_model_root())
    missing: list[str] = []
    for rel_path in get_group_files(groups):
        if not (root / rel_path).exists():
            missing.append(rel_path)
    return missing


def has_groups(groups: str | Iterable[str] | None, *, model_root: str | None = None) -> bool:
    return not list_missing_files(groups, model_root=model_root)


def download_groups(
    groups: str | Iterable[str] | None,
    *,
    repo_id: str | None = None,
    model_root: str | None = None,
) -> str:
    resolved_groups = _normalize_groups(groups)
    root = os.path.abspath(model_root or get_model_root())
    repo = repo_id or get_model_repo()
    allow_patterns = sorted(
        set(get_group_files(resolved_groups)).union(OPTIONAL_METADATA_FILES)
    )
    os.makedirs(root, exist_ok=True)
    logger.info(
        "Downloading model groups %s from %s to %s",
        ",".join(resolved_groups),
        repo,
        root,
    )
    snapshot_download(
        repo_id=repo,
        local_dir=root,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )
    return root


def ensure_groups(
    groups: str | Iterable[str] | None,
    *,
    auto_download: bool | None = None,
    repo_id: str | None = None,
    model_root: str | None = None,
) -> str:
    resolved_groups = _normalize_groups(groups)
    root = os.path.abspath(model_root or get_model_root())
    missing = list_missing_files(resolved_groups, model_root=root)
    if not missing:
        return root

    if auto_download is None:
        auto_download = _parse_bool(os.environ.get("DEEPDOC_AUTO_DOWNLOAD"), default=True)

    if auto_download:
        download_groups(resolved_groups, repo_id=repo_id, model_root=root)
        missing = list_missing_files(resolved_groups, model_root=root)
        if not missing:
            return root

    raise FileNotFoundError(
        "Missing required model files under "
        f"{root}: {', '.join(missing)}. "
        "Run `python download_models.py published` or set `DEEPDOC_AUTO_DOWNLOAD=1`."
    )


def get_download_groups_from_env(default: str = "published") -> list[str]:
    raw_value = os.environ.get("DEEPDOC_DOWNLOAD_GROUPS", default)
    return _normalize_groups(raw_value)

#!/usr/bin/env python3
"""Check whether the local CPU pipeline upgrade gates are ready.

This is a readiness gate for document parsing pipeline upgrades only. It does
not perform retrieval, vector indexing, question answering, or answer generation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.model_store import MODEL_MANIFEST_SCHEMA_VERSION, build_model_manifest, get_model_root, list_missing_files
from tools.eval_omnidocbench import cpu_pipeline_license_candidates, validate_dataset


FLOAT_TOLERANCE = 1e-6
EVAL_REPORT_SCHEMA_VERSION = "2026-06-08.cpu-pipeline-eval.v1"
PROFILE_REPORT_SCHEMA_VERSION = "2026-06-08.cpu-pipeline-profile.v1"
LICENSE_GATE_SCHEMA_VERSION = "2026-06-08.cpu-pipeline-license-gate.v1"
DATASET_CONTRACT_SCHEMA_VERSION = "2026-06-08.cpu-pipeline-dataset-contract.v1"
REQUIRED_MODEL_GROUPS = ("core_v5", "layout_v2", "table_v2", "formula_v2")
DEFAULT_AB_REPORTS = {
    "ocr_v5": {
        "paired": True,
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "required_summary_fields": ("sample_count", "mean_character_error_rate", "mean_word_error_rate"),
        "baseline_pipeline_config": {"ocr_version": "v4"},
        "candidate_pipeline_config": {"ocr_version": "v5"},
        "lower_or_equal_metrics": ("mean_character_error_rate", "mean_word_error_rate"),
        "higher_or_equal_metrics": (),
    },
    "layout_v2": {
        "paired": True,
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "required_summary_fields": (
            "sample_count",
            "mean_block_type_f1",
            "mean_reading_order_normalized_edit_distance",
            "mean_cross_page_merge_accuracy",
            "mean_chunk_text_coverage",
            "mean_business_field_location_hit_rate",
        ),
        "baseline_pipeline_config": {"layout_engine": "legacy", "reading_order_strategy": "legacy"},
        "candidate_pipeline_config": {"layout_engine": "ppdoclayout", "reading_order_strategy": "rules"},
        "lower_or_equal_metrics": ("mean_reading_order_normalized_edit_distance",),
        "higher_or_equal_metrics": (
            "mean_block_type_f1",
            "mean_cross_page_merge_accuracy",
            "mean_chunk_text_coverage",
            "mean_business_field_location_hit_rate",
        ),
    },
    "table_v2": {
        "paired": True,
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "required_summary_fields": ("sample_count", "mean_table_teds", "mean_table_cell_f1"),
        "baseline_pipeline_config": {"table_engine": "tatr"},
        "candidate_pipeline_config": {"table_engine": "rapidtable"},
        "lower_or_equal_metrics": (),
        "higher_or_equal_metrics": ("mean_table_teds", "mean_table_cell_f1"),
    },
    "formula_v2": {
        "paired": True,
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "required_summary_fields": (
            "sample_count",
            "mean_formula_normalized_edit_distance",
            "mean_formula_exact_match_rate",
            "mean_elapsed_seconds",
        ),
        "baseline_pipeline_config": {"formula_mode": "rapidlatex"},
        "candidate_pipeline_config": {"formula_mode": "pp_formula_net_s"},
        "lower_or_equal_metrics": ("mean_formula_normalized_edit_distance", "mean_elapsed_seconds"),
        "higher_or_equal_metrics": ("mean_formula_exact_match_rate",),
    },
    "profile": {
        "paired": False,
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "required_summary_fields": (),
        "required_report_fields": (
            "total_elapsed_seconds",
            "reading_order_strategy",
            "pipeline_config",
            "stage_summary",
        ),
        "required_stage_names": (
            "rasterize_ocr",
            "layout",
            "table",
            "text_merge",
            "cross_page_text",
            "reading_order",
            "extract_assets",
        ),
        "required_pipeline_config_keys": (
            "ocr_version",
            "layout_engine",
            "table_engine",
            "formula_mode",
            "reading_order_strategy",
        ),
        "pipeline_config": {},
        "lower_or_equal_metrics": (),
        "higher_or_equal_metrics": (),
    },
}
SUMMARY_SAMPLE_METRIC_FIELDS = {
    "mean_character_error_rate": "character_error_rate",
    "mean_word_error_rate": "word_error_rate",
    "mean_block_type_f1": "block_type_f1",
    "mean_reading_order_normalized_edit_distance": "reading_order_normalized_edit_distance",
    "mean_cross_page_merge_accuracy": "cross_page_merge_accuracy",
    "mean_chunk_text_coverage": "chunk_text_coverage",
    "mean_business_field_location_hit_rate": "business_field_location_hit_rate",
    "mean_table_teds": "mean_table_teds",
    "mean_table_cell_f1": "mean_table_cell_f1",
    "mean_formula_normalized_edit_distance": "formula_normalized_edit_distance",
    "mean_formula_exact_match_rate": "formula_exact_match_rate",
    "mean_elapsed_seconds": "elapsed_seconds",
}


def _pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        try:
            import fitz

            with fitz.open(str(path)) as document:
                return int(document.page_count)
        except Exception:
            return 0


def _dataset_report(dataset: str | Path | None, min_pages: int) -> dict[str, Any]:
    if not dataset:
        return {
            "path": None,
            "exists": False,
            "pdf_count": 0,
            "page_count": 0,
            "min_pages": min_pages,
            "meets_min_pages": False,
            "unreadable_pdfs": [],
        }

    root = Path(dataset)
    pdf_paths = sorted(root.rglob("*.pdf")) if root.exists() else []
    unreadable: list[str] = []
    page_count = 0
    for pdf_path in pdf_paths:
        pages = _pdf_page_count(pdf_path)
        if pages <= 0:
            unreadable.append(str(pdf_path))
        page_count += max(0, pages)

    return {
        "path": str(root),
        "exists": root.exists(),
        "pdf_count": len(pdf_paths),
        "page_count": page_count,
        "min_pages": min_pages,
        "meets_min_pages": page_count >= min_pages,
        "unreadable_pdfs": unreadable,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _finite_float(value: Any, *, problem_label: str, problems: list[str]) -> float | None:
    if isinstance(value, bool):
        problems.append(f"{problem_label} must be numeric")
        return None
    try:
        number = float(value)
    except Exception:
        problems.append(f"{problem_label} must be numeric")
        return None
    if not math.isfinite(number):
        problems.append(f"{problem_label} must be finite")
        return None
    return number


def _positive_int(value: Any, *, problem_label: str, problems: list[str]) -> int | None:
    if isinstance(value, bool):
        problems.append(f"{problem_label} must be an integer")
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            problems.append(f"{problem_label} must be finite")
            return None
        if not value.is_integer():
            problems.append(f"{problem_label} must be an integer")
            return None
        number = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            number = int(stripped)
        else:
            parsed = _finite_float(value, problem_label=problem_label, problems=problems)
            if parsed is None:
                return None
            problems.append(f"{problem_label} must be an integer")
            return None
    else:
        parsed = _finite_float(value, problem_label=problem_label, problems=problems)
        if parsed is None:
            return None
        if not parsed.is_integer():
            problems.append(f"{problem_label} must be an integer")
            return None
        number = int(parsed)

    if number <= 0:
        problems.append(f"{problem_label} must be positive")
        return None
    return number


def _validate_summary_sample_metric_means(
    *,
    summary: dict[str, Any],
    samples: list[Any],
    summary_fields: tuple[str, ...],
) -> list[str]:
    problems: list[str] = []
    for summary_field in summary_fields:
        sample_field = SUMMARY_SAMPLE_METRIC_FIELDS.get(summary_field)
        if not sample_field or summary_field not in summary:
            continue
        summary_value = _finite_float(
            summary[summary_field],
            problem_label=f"summary {summary_field}",
            problems=problems,
        )
        values: list[float] = []
        saw_sample_metric = False
        saw_invalid_sample_metric = False
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            if sample.get(sample_field) is None:
                saw_invalid_sample_metric = True
                problems.append(f"samples[{index}].{sample_field} is required for summary {summary_field}")
                continue
            saw_sample_metric = True
            sample_value = _finite_float(
                sample[sample_field],
                problem_label=f"samples[{index}].{sample_field}",
                problems=problems,
            )
            if sample_value is None:
                saw_invalid_sample_metric = True
                continue
            values.append(sample_value)
        if not saw_sample_metric:
            problems.append(
                f"samples missing metric values for summary {summary_field}: sample_field={sample_field}"
            )
            continue
        if summary_value is None or saw_invalid_sample_metric:
            continue
        sample_mean = sum(values) / len(values)
        if abs(summary_value - sample_mean) > FLOAT_TOLERANCE:
            problems.append(
                f"summary {summary_field} must match samples mean: "
                f"summary={_format_float(summary_value)}, samples={_format_float(sample_mean)}"
            )
    return problems


def _license_items_by_name(items: list[Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name and name not in indexed:
            indexed[name] = item
    return indexed


def _required_license_items_by_status(status: str) -> dict[str, dict[str, str]]:
    return {
        item["name"]: item
        for item in cpu_pipeline_license_candidates()
        if item.get("status") == status and str(item.get("name") or "").strip()
    }


def _validate_license_items(
    actual_by_name: dict[str, dict[str, Any]],
    *,
    expected_by_name: dict[str, dict[str, str]],
    problem_prefix: str,
) -> list[str]:
    problems: list[str] = []
    for name, expected in sorted(expected_by_name.items()):
        actual = actual_by_name.get(name)
        if actual is None:
            continue
        for field in ("license", "status"):
            expected_value = expected.get(field)
            actual_value = actual.get(field)
            if actual_value != expected_value:
                problems.append(
                    f"{problem_prefix} {name}.{field} expected {expected_value!r}, got {actual_value!r}"
                )
    return problems


def _validate_license_gate_payload(payload: dict[str, Any], *, problem_prefix: str) -> tuple[str, str | None, list[Any], list[Any], Any, list[str]]:
    actual_schema_version = payload.get("schema_version")
    actual_status = payload.get("status")
    problems: list[str] = []
    if actual_schema_version != LICENSE_GATE_SCHEMA_VERSION:
        problems.append(
            f"{problem_prefix} schema_version expected {LICENSE_GATE_SCHEMA_VERSION}, got {actual_schema_version}"
        )
    if actual_status != "passed":
        problems.append(f"{problem_prefix} status expected passed, got {actual_status}")
    allowed = payload.get("allowed")
    blocked = payload.get("blocked")
    if not isinstance(allowed, list):
        problems.append(f"{problem_prefix} allowed must be a JSON array")
    else:
        required_allowed_by_name = _required_license_items_by_status("allowed")
        allowed_by_name = _license_items_by_name(allowed)
        allowed_names = set(allowed_by_name)
        missing_allowed = sorted(set(required_allowed_by_name) - allowed_names)
        if missing_allowed:
            problems.append(
                f"{problem_prefix} allowed missing required candidates: " + ", ".join(missing_allowed)
            )
        problems.extend(
            _validate_license_items(
                allowed_by_name,
                expected_by_name=required_allowed_by_name,
                problem_prefix=f"{problem_prefix} allowed",
            )
        )
    if not isinstance(blocked, list):
        problems.append(f"{problem_prefix} blocked must be a JSON array")
    else:
        required_blocked_by_name = _required_license_items_by_status("blocked")
        blocked_by_name = _license_items_by_name(blocked)
        blocked_names = set(blocked_by_name)
        missing_blocked = sorted(set(required_blocked_by_name) - blocked_names)
        if missing_blocked:
            problems.append(
                f"{problem_prefix} blocked missing required candidates: " + ", ".join(missing_blocked)
            )
        problems.extend(
            _validate_license_items(
                blocked_by_name,
                expected_by_name=required_blocked_by_name,
                problem_prefix=f"{problem_prefix} blocked",
            )
        )
    return actual_status, actual_schema_version, allowed, blocked, payload.get("rule"), problems


def _license_gate_report(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {
            "status": "missing",
            "path": None,
            "problems": ["missing license gate report path"],
        }

    report_path = Path(path)
    if not report_path.exists():
        return {
            "status": "missing",
            "path": str(report_path),
            "problems": [f"missing license gate report file: {report_path}"],
        }

    payload = _load_json(report_path)
    if payload is None:
        return {
            "status": "failed",
            "path": str(report_path),
            "problems": ["license gate report is not a JSON object"],
        }

    actual_status, actual_schema_version, allowed, blocked, rule, raw_problems = _validate_license_gate_payload(
        payload,
        problem_prefix="license gate",
    )
    problems = [
        problem.removeprefix("license gate ")
        if problem.startswith("license gate schema_version")
        else problem
        for problem in raw_problems
    ]

    return {
        "status": "failed" if problems else "ok",
        "path": str(report_path),
        "schema_version": actual_schema_version,
        "gate_status": actual_status,
        "allowed": allowed,
        "blocked": blocked,
        "rule": rule,
        "problems": problems,
    }


def _validate_eval_dataset_contract_payload(
    payload: dict[str, Any],
    *,
    report_dataset: Any,
    summary_sample_count: Any,
    expected_dataset: str | Path | None = None,
    expected_sample_names: tuple[str, ...] | None = None,
    expected_sample_pdf_paths: dict[str, str] | None = None,
) -> list[str]:
    problems: list[str] = []
    actual_schema_version = payload.get("schema_version")
    actual_status = payload.get("status")
    if actual_schema_version != DATASET_CONTRACT_SCHEMA_VERSION:
        problems.append(
            f"dataset_contract schema_version expected {DATASET_CONTRACT_SCHEMA_VERSION}, got {actual_schema_version}"
        )
    if actual_status != "ok":
        problems.append(f"dataset_contract status expected ok, got {actual_status}")

    contract_dataset = payload.get("dataset")
    if report_dataset is not None and not _same_dataset_path(contract_dataset, report_dataset):
        problems.append(
            "dataset_contract dataset must match report dataset: "
            f"expected={report_dataset}, got={contract_dataset}"
        )
    if expected_dataset is not None and not _same_dataset_path(contract_dataset, expected_dataset):
        problems.append(
            "dataset_contract dataset must match readiness dataset: "
            f"expected={expected_dataset}, got={contract_dataset}"
        )

    contract_sample_count_int = _positive_int(
        payload.get("sample_count"),
        problem_label="dataset_contract sample_count",
        problems=problems,
    )

    summary_sample_count_int = None
    if summary_sample_count is not None:
        summary_count_problems: list[str] = []
        summary_sample_count_int = _positive_int(
            summary_sample_count,
            problem_label="summary sample_count",
            problems=summary_count_problems,
        )
    if contract_sample_count_int is not None and summary_sample_count_int is not None:
        if contract_sample_count_int != summary_sample_count_int:
            problems.append(
                "dataset_contract sample_count must match summary.sample_count: "
                f"dataset_contract={contract_sample_count_int}, summary={summary_sample_count_int}"
            )

    samples = payload.get("samples")
    if not isinstance(samples, list):
        problems.append("dataset_contract samples must be a JSON array")
        return problems
    if contract_sample_count_int is not None and contract_sample_count_int != len(samples):
        problems.append(
            "dataset_contract sample_count must match samples length: "
            f"sample_count={contract_sample_count_int}, samples={len(samples)}"
        )
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            problems.append(f"dataset_contract samples[{index}] must be a JSON object")
    if expected_sample_names is not None:
        actual_names = {
            str(sample.get("name")).strip()
            for sample in samples
            if isinstance(sample, dict) and str(sample.get("name") or "").strip()
        }
        expected_names = set(expected_sample_names)
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        if missing or unexpected:
            problems.append(
                "dataset_contract samples names must match readiness dataset contract: "
                f"missing={','.join(missing) or '-'}, unexpected={','.join(unexpected) or '-'}"
            )
    if expected_sample_pdf_paths is not None:
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            sample_name = str(sample.get("name") or "").strip()
            if not sample_name or sample_name not in expected_sample_pdf_paths:
                continue
            expected_pdf_path = expected_sample_pdf_paths[sample_name]
            if "pdf_path" not in sample or not str(sample.get("pdf_path") or "").strip():
                problems.append(
                    f"dataset_contract samples[{index}].pdf_path is required for readiness dataset contract "
                    f"sample {sample_name}: expected={expected_pdf_path}"
                )
                continue
            actual_pdf_path = sample.get("pdf_path")
            if not _same_dataset_path(actual_pdf_path, expected_pdf_path):
                problems.append(
                    f"dataset_contract samples[{index}].pdf_path must match readiness dataset contract "
                    f"sample {sample_name}: expected={expected_pdf_path}, got={actual_pdf_path}"
                )

    return problems


def _model_group_for_ocr_version(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text == "v5":
        return "core_v5"
    if text == "v4":
        return "core"
    return None


def _model_group_for_layout_engine(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text == "ppdoclayout":
        return "layout_v2"
    if text == "legacy":
        return "core"
    return None


def _model_group_for_table_engine(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text == "rapidtable":
        return "table_v2"
    if text == "tatr":
        return "core"
    return None


def _model_group_for_formula_mode(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text == "pp_formula_net_s":
        return "formula_v2"
    if text == "rapidlatex":
        return "formula"
    return None


def _required_model_groups_for_eval_report(
    *,
    pipeline_config: dict[str, Any],
    expected_pipeline_config: dict[str, str] | None = None,
) -> tuple[str, ...]:
    groups: list[str] = []

    def add(group: str | None) -> None:
        if group and group not in groups:
            groups.append(group)

    effective_expected = expected_pipeline_config or {}
    for config in (effective_expected, pipeline_config):
        if "ocr_version" in config:
            add(_model_group_for_ocr_version(config.get("ocr_version")))
        if "layout_engine" in config:
            add(_model_group_for_layout_engine(config.get("layout_engine")))
        if "table_engine" in config:
            add(_model_group_for_table_engine(config.get("table_engine")))
        if "formula_mode" in config:
            add(_model_group_for_formula_mode(config.get("formula_mode")))

    return tuple(sorted(groups))


def _manifest_files_by_path(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = payload.get("files")
    if not isinstance(files, list):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        relative_path = str(item.get("path") or "").strip()
        if relative_path:
            rows[relative_path] = item
    return rows


def _validate_model_manifest_payload(
    payload: dict[str, Any],
    *,
    required_groups: tuple[str, ...],
    model_root: str | Path,
) -> list[str]:
    problems: list[str] = []
    actual_schema_version = payload.get("schema_version")
    if actual_schema_version != MODEL_MANIFEST_SCHEMA_VERSION:
        problems.append(
            f"model_manifest schema_version expected {MODEL_MANIFEST_SCHEMA_VERSION}, got {actual_schema_version}"
        )

    groups = payload.get("groups")
    if not isinstance(groups, list):
        problems.append("model_manifest groups must be a JSON array")
    else:
        actual_groups = {str(group).strip() for group in groups if str(group or "").strip()}
        missing_groups = sorted(set(required_groups) - actual_groups)
        if missing_groups:
            problems.append(
                "model_manifest groups missing required groups: " + ", ".join(missing_groups)
            )

    current_manifest = build_model_manifest(model_root=model_root, groups=required_groups)
    expected_provenance = current_manifest.get("model_group_provenance")
    actual_provenance = payload.get("model_group_provenance")
    if not isinstance(expected_provenance, dict):
        expected_provenance = {}
    if not isinstance(actual_provenance, dict):
        for group in sorted(expected_provenance):
            problems.append(f"missing model_manifest model_group_provenance for group: {group}")
    else:
        for group, expected_entry in sorted(expected_provenance.items()):
            actual_entry = actual_provenance.get(group)
            if not isinstance(actual_entry, dict):
                problems.append(f"missing model_manifest model_group_provenance for group: {group}")
                continue
            for field, expected_value in sorted(expected_entry.items()):
                actual_value = actual_entry.get(field)
                if actual_value != expected_value:
                    problems.append(
                        "model_manifest model_group_provenance mismatch for "
                        f"{group}.{field}: expected {expected_value!r}, got {actual_value!r}"
                    )

    files = payload.get("files")
    if not isinstance(files, list):
        problems.append("model_manifest files must be a JSON array")
        return problems

    report_files = _manifest_files_by_path(payload)
    for current_file in current_manifest["files"]:
        relative_path = str(current_file["path"])
        report_file = report_files.get(relative_path)
        if report_file is None:
            problems.append(f"model_manifest missing file entry: {relative_path}")
            continue
        if bool(report_file.get("exists")) != bool(current_file.get("exists")):
            problems.append(f"model_manifest file {relative_path} exists mismatch")
        if report_file.get("size_bytes") != current_file.get("size_bytes"):
            problems.append(f"model_manifest file {relative_path} size_bytes mismatch")
        if report_file.get("sha256") != current_file.get("sha256"):
            problems.append(f"model_manifest file {relative_path} sha256 mismatch")

    return problems


def _ab_report(
    name: str,
    path: str | Path | None,
    *,
    expected_pipeline_config: dict[str, str] | None = None,
    model_root: str | Path | None = None,
    expected_dataset: str | Path | None = None,
    expected_sample_names: tuple[str, ...] | None = None,
    expected_sample_pdf_paths: dict[str, str] | None = None,
    expected_profile_dataset: str | Path | None = None,
    expected_profile_sample_names: tuple[str, ...] | None = None,
    expected_profile_pdf_paths: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    spec = DEFAULT_AB_REPORTS[name]
    if not path:
        return {
            "status": "missing",
            "path": None,
            "problems": ["missing A/B report path"],
        }

    report_path = Path(path)
    if not report_path.exists():
        return {
            "status": "missing",
            "path": str(report_path),
            "problems": [f"missing A/B report file: {report_path}"],
        }

    payload = _load_json(report_path)
    if payload is None:
        return {
            "status": "failed",
            "path": str(report_path),
            "problems": ["A/B report is not a JSON object"],
        }

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    pipeline_config = payload.get("pipeline_config") if isinstance(payload.get("pipeline_config"), dict) else {}
    expected_schema_version = spec.get("schema_version")
    actual_schema_version = payload.get("schema_version")
    problems = [
        f"missing summary field: {field}"
        for field in spec["required_summary_fields"]
        if field not in summary
    ]
    if expected_schema_version and actual_schema_version != expected_schema_version:
        problems.append(
            f"schema_version expected {expected_schema_version}, got {actual_schema_version}"
        )
    license_gate = payload.get("license_gate")
    required_model_groups = _required_model_groups_for_eval_report(
        pipeline_config=pipeline_config,
        expected_pipeline_config=expected_pipeline_config,
    )
    if name != "profile":
        if "engine" not in payload:
            problems.append("missing engine report field")
        elif payload.get("engine") != "deepdoc":
            problems.append(f"engine expected deepdoc, got {payload.get('engine')}")
        model_manifest = payload.get("model_manifest")
        if "model_manifest" not in payload:
            problems.append(
                "missing model_manifest report field for required groups: "
                + (", ".join(required_model_groups) if required_model_groups else "-")
            )
        elif not isinstance(model_manifest, dict):
            problems.append("model_manifest must be a JSON object")
        else:
            problems.extend(
                _validate_model_manifest_payload(
                    model_manifest,
                    required_groups=required_model_groups,
                    model_root=model_root or get_model_root(),
                )
            )
        if "license_gate" not in payload:
            problems.append("missing license_gate report field")
        elif not isinstance(license_gate, dict):
            problems.append("license_gate must be a JSON object")
        else:
            _, _, _, _, _, license_gate_problems = _validate_license_gate_payload(
                license_gate,
                problem_prefix="license_gate",
            )
            problems.extend(license_gate_problems)
        dataset_contract = payload.get("dataset_contract")
        if "dataset_contract" not in payload:
            problems.append("missing dataset_contract report field")
        elif not isinstance(dataset_contract, dict):
            problems.append("dataset_contract must be a JSON object")
        else:
            problems.extend(
                _validate_eval_dataset_contract_payload(
                    dataset_contract,
                    report_dataset=payload.get("dataset"),
                    summary_sample_count=summary.get("sample_count"),
                    expected_dataset=expected_dataset,
                    expected_sample_names=expected_sample_names,
                    expected_sample_pdf_paths=expected_sample_pdf_paths,
                )
            )
    else:
        dataset_contract = payload.get("dataset_contract")
        model_manifest = payload.get("model_manifest")
        if "dataset_contract" not in payload:
            problems.append("missing profile field: dataset_contract")
        elif not isinstance(dataset_contract, dict):
            problems.append("profile dataset_contract must be a JSON object")
        else:
            problems.extend(
                _validate_eval_dataset_contract_payload(
                    dataset_contract,
                    report_dataset=payload.get("dataset"),
                    summary_sample_count=None,
                    expected_dataset=expected_profile_dataset,
                    expected_sample_names=expected_profile_sample_names,
                    expected_sample_pdf_paths=expected_sample_pdf_paths,
                )
            )
        if "model_manifest" not in payload:
            problems.append(
                "missing profile field: model_manifest for required groups: "
                + (", ".join(required_model_groups) if required_model_groups else "-")
            )
        elif not isinstance(model_manifest, dict):
            problems.append("profile model_manifest must be a JSON object")
        else:
            problems.extend(
                _validate_model_manifest_payload(
                    model_manifest,
                    required_groups=required_model_groups,
                    model_root=model_root or get_model_root(),
                )
            )
        if "license_gate" not in payload:
            problems.append("missing profile field: license_gate")
        elif not isinstance(license_gate, dict):
            problems.append("profile license_gate must be a JSON object")
        else:
            _, _, _, _, _, license_gate_problems = _validate_license_gate_payload(
                license_gate,
                problem_prefix="profile license_gate",
            )
            problems.extend(license_gate_problems)
    for field in spec.get("required_report_fields", ()):
        if field not in payload:
            problems.append(f"missing profile field: {field}")
    required_stage_names = tuple(spec.get("required_stage_names", ()))
    if required_stage_names:
        stages = payload.get("stages") if isinstance(payload.get("stages"), list) else []
        stage_names = {
            str(stage.get("stage"))
            for stage in stages
            if isinstance(stage, dict) and stage.get("stage") is not None
        }
        missing_stages = [stage for stage in required_stage_names if stage not in stage_names]
        if missing_stages:
            problems.append("missing profile stages: " + ", ".join(missing_stages))
        unexpected_stages = sorted(stage_names - set(required_stage_names))
        if unexpected_stages:
            problems.append("unexpected profile stages: " + ", ".join(unexpected_stages))
        stage_elapsed_values: list[float] = []
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            stage_name = str(stage.get("stage"))
            elapsed_float = _finite_float(
                stage.get("elapsed_seconds"),
                problem_label=f"profile stage {stage_name} elapsed_seconds",
                problems=problems,
            )
            if elapsed_float is None:
                continue
            if elapsed_float < 0:
                problems.append(f"profile stage {stage_name} elapsed_seconds must be non-negative")
                continue
            stage_elapsed_values.append(elapsed_float)
        total_elapsed = None
        if "total_elapsed_seconds" in payload:
            total_elapsed = _finite_float(
                payload.get("total_elapsed_seconds"),
                problem_label="profile total_elapsed_seconds",
                problems=problems,
            )
        if total_elapsed is not None:
            if total_elapsed <= 0:
                problems.append("profile total_elapsed_seconds must be positive")
        if total_elapsed is not None and stage_elapsed_values:
            summed_elapsed = sum(stage_elapsed_values)
            if abs(total_elapsed - summed_elapsed) > FLOAT_TOLERANCE:
                problems.append(
                    "profile total_elapsed_seconds does not match summed stages: "
                    f"total={total_elapsed}, stages={summed_elapsed}"
                )
        if "stage_summary" in payload:
            stage_summary = payload.get("stage_summary")
            if not isinstance(stage_summary, dict):
                problems.append("profile stage_summary must be a JSON object")
            else:
                expected_stage_count = len(stages)
                actual_stage_count = stage_summary.get("stage_count")
                if actual_stage_count != expected_stage_count:
                    problems.append(
                        f"profile stage_summary.stage_count expected {expected_stage_count}, got {actual_stage_count}"
                    )
                share = stage_summary.get("slowest_stage_share")
                share_float = _finite_float(
                    share,
                    problem_label="profile stage_summary.slowest_stage_share",
                    problems=problems,
                )
                if share_float is not None:
                    if share_float < 0 or share_float > 1:
                        problems.append("profile stage_summary.slowest_stage_share must be between 0 and 1")
                if stage_elapsed_values:
                    valid_stages: list[tuple[str, float]] = []
                    for stage in stages:
                        if not isinstance(stage, dict) or stage.get("stage") is None:
                            continue
                        try:
                            elapsed = float(stage.get("elapsed_seconds"))
                        except Exception:
                            continue
                        if math.isfinite(elapsed) and elapsed >= 0:
                            valid_stages.append((str(stage.get("stage")), elapsed))
                    if valid_stages:
                        expected_slowest_stage, expected_slowest_elapsed = sorted(
                            valid_stages,
                            key=lambda row: (-row[1], row[0]),
                        )[0]
                        actual_slowest_stage = stage_summary.get("slowest_stage")
                        if actual_slowest_stage != expected_slowest_stage:
                            problems.append(
                                "profile stage_summary.slowest_stage expected "
                                f"{expected_slowest_stage}, got {actual_slowest_stage}"
                            )
                        actual_slowest_elapsed = _finite_float(
                            stage_summary.get("slowest_stage_elapsed_seconds"),
                            problem_label="profile stage_summary.slowest_stage_elapsed_seconds",
                            problems=problems,
                        )
                        if actual_slowest_elapsed is not None:
                            if abs(actual_slowest_elapsed - expected_slowest_elapsed) > FLOAT_TOLERANCE:
                                problems.append(
                                    "profile stage_summary.slowest_stage_elapsed_seconds expected "
                                    f"{expected_slowest_elapsed}, got {actual_slowest_elapsed}"
                                )
                        summed_elapsed = sum(elapsed for _, elapsed in valid_stages)
                        expected_share = expected_slowest_elapsed / summed_elapsed if summed_elapsed > 0 else 0.0
                        actual_share = share_float
                        if actual_share is not None:
                            if abs(actual_share - expected_share) > FLOAT_TOLERANCE:
                                problems.append(
                                    "profile stage_summary.slowest_stage_share expected "
                                    f"{expected_share}, got {actual_share}"
                                )
                        expected_ranked_stages = sorted(
                            valid_stages,
                            key=lambda row: (-row[1], row[0]),
                        )
                        ranked_payload = stage_summary.get("stages_by_elapsed_seconds")
                        ranked_elapsed_values: list[float | None] = []
                        ranked_share_values: list[float | None] = []
                        if isinstance(ranked_payload, list):
                            for index, actual in enumerate(ranked_payload):
                                if not isinstance(actual, dict):
                                    ranked_elapsed_values.append(None)
                                    ranked_share_values.append(None)
                                    continue
                                ranked_elapsed_values.append(
                                    _finite_float(
                                        actual.get("elapsed_seconds"),
                                        problem_label=(
                                            "profile stage_summary.stages_by_elapsed_seconds"
                                            f"[{index}].elapsed_seconds"
                                        ),
                                        problems=problems,
                                    )
                                )
                                ranked_share = _finite_float(
                                    actual.get("share"),
                                    problem_label=(
                                        "profile stage_summary.stages_by_elapsed_seconds"
                                        f"[{index}].share"
                                    ),
                                    problems=problems,
                                )
                                if ranked_share is not None and (ranked_share < 0 or ranked_share > 1):
                                    problems.append(
                                        "profile stage_summary.stages_by_elapsed_seconds"
                                        f"[{index}].share must be between 0 and 1"
                                    )
                                ranked_share_values.append(ranked_share)
                        ranked_matches = isinstance(ranked_payload, list) and len(ranked_payload) == len(
                            expected_ranked_stages
                        )
                        if ranked_matches:
                            summed_elapsed = sum(elapsed for _, elapsed in expected_ranked_stages)
                            for index, (actual, expected) in enumerate(zip(ranked_payload, expected_ranked_stages)):
                                if not isinstance(actual, dict):
                                    ranked_matches = False
                                    break
                                actual_stage = actual.get("stage")
                                actual_elapsed = ranked_elapsed_values[index]
                                actual_share = ranked_share_values[index]
                                if actual_elapsed is None:
                                    ranked_matches = False
                                    break
                                if actual_stage != expected[0] or abs(actual_elapsed - expected[1]) > FLOAT_TOLERANCE:
                                    ranked_matches = False
                                    break
                                expected_share = expected[1] / summed_elapsed if summed_elapsed > 0 else 0.0
                                if actual_share is not None and abs(actual_share - expected_share) > FLOAT_TOLERANCE:
                                    problems.append(
                                        "profile stage_summary.stages_by_elapsed_seconds"
                                        f"[{index}].share expected {expected_share}, got {actual_share}"
                                    )
                        if not ranked_matches:
                            problems.append(
                                "profile stage_summary.stages_by_elapsed_seconds must include "
                                "the same ordered stages as profile stages"
                            )
    required_pipeline_config_keys = tuple(spec.get("required_pipeline_config_keys", ()))
    if required_pipeline_config_keys and "pipeline_config" in payload:
        if not isinstance(payload.get("pipeline_config"), dict):
            problems.append("profile pipeline_config must be a JSON object")
        else:
            missing_keys = [
                key for key in required_pipeline_config_keys if key not in pipeline_config
            ]
            if missing_keys:
                problems.append(
                    "profile pipeline_config missing keys: " + ", ".join(missing_keys)
                )
            for key in required_pipeline_config_keys:
                if key in payload and key in pipeline_config and payload.get(key) != pipeline_config.get(key):
                    problems.append(
                        f"profile {key} does not match pipeline_config.{key}: "
                        f"top-level={payload.get(key)}, pipeline_config={pipeline_config.get(key)}"
                    )
    if name == "profile" and expected_profile_sample_names is not None:
        sample_name = str(payload.get("sample_name") or "").strip()
        expected_names = set(expected_profile_sample_names)
        if not sample_name:
            problems.append("missing profile field: sample_name")
        elif sample_name not in expected_names:
            problems.append(
                "profile sample_name must be one of dataset contract samples: "
                f"expected={','.join(sorted(expected_names))}, got={sample_name}"
            )
    if name == "profile" and expected_profile_pdf_paths is not None:
        pdf_path = str(payload.get("pdf_path") or "").strip()
        if not pdf_path:
            problems.append("missing profile field: pdf_path")
        else:
            try:
                actual_pdf_path = Path(pdf_path).expanduser().resolve(strict=False)
                expected_pdf_paths = {
                    Path(expected).expanduser().resolve(strict=False)
                    for expected in expected_profile_pdf_paths
                }
            except Exception:
                actual_pdf_path = Path(pdf_path)
                expected_pdf_paths = {Path(expected) for expected in expected_profile_pdf_paths}
            if actual_pdf_path not in expected_pdf_paths:
                problems.append(
                    "profile pdf_path must point to a PDF in readiness dataset: "
                    f"expected one of {expected_profile_dataset}, got {pdf_path}"
                )
    if name == "profile" and expected_sample_pdf_paths is not None:
        sample_name = str(payload.get("sample_name") or "").strip()
        pdf_path = str(payload.get("pdf_path") or "").strip()
        if sample_name and sample_name in expected_sample_pdf_paths and pdf_path:
            expected_pdf_path = expected_sample_pdf_paths[sample_name]
            if not _same_dataset_path(pdf_path, expected_pdf_path):
                problems.append(
                    "profile sample_name/pdf_path must reference the same dataset contract sample: "
                    f"sample_name={sample_name}, expected_pdf_path={expected_pdf_path}, got={pdf_path}"
                )
    if name == "profile" and expected_profile_dataset is not None:
        profile_dataset = payload.get("dataset")
        if "dataset" not in payload or profile_dataset is None or not str(profile_dataset).strip():
            problems.append("missing profile field: dataset")
        elif not _same_dataset_path(profile_dataset, expected_profile_dataset):
            problems.append(
                "profile dataset must match readiness dataset: "
                f"expected={expected_profile_dataset}, got={profile_dataset}"
            )
    for key, expected_value in (expected_pipeline_config or {}).items():
        actual_value = pipeline_config.get(key)
        if actual_value != expected_value:
            problems.append(
                f"pipeline_config.{key} expected {expected_value}, got {actual_value}"
            )
    sample_count = summary.get("sample_count")
    if sample_count is not None:
        sample_count_int = _positive_int(
            sample_count,
            problem_label="summary sample_count",
            problems=problems,
        )
        samples = payload.get("samples")
        if name != "profile" and "samples" not in payload:
            problems.append("missing samples report field")
        elif sample_count_int is not None and "samples" in payload:
            if not isinstance(samples, list):
                problems.append("samples must be a JSON array")
            elif sample_count_int != len(samples):
                problems.append(
                    "summary sample_count must match samples length: "
                    f"sample_count={sample_count_int}, samples={len(samples)}"
                )
            else:
                for index, sample in enumerate(samples):
                    if not isinstance(sample, dict):
                        problems.append(f"samples[{index}] must be a JSON object")
                        continue
                    if "engine" in sample and sample.get("engine") != "deepdoc":
                        problems.append(
                            f"samples[{index}].engine expected deepdoc, got {sample.get('engine')}"
                        )
                    if expected_sample_pdf_paths is None:
                        continue
                    sample_name = str(sample.get("name") or "").strip()
                    if not sample_name or sample_name not in expected_sample_pdf_paths:
                        continue
                    expected_pdf_path = expected_sample_pdf_paths[sample_name]
                    if "pdf_path" not in sample or not str(sample.get("pdf_path") or "").strip():
                        problems.append(
                            f"samples[{index}].pdf_path is required for dataset contract sample {sample_name}: "
                            f"expected={expected_pdf_path}"
                        )
                        continue
                    actual_pdf_path = sample.get("pdf_path")
                    if not _same_dataset_path(actual_pdf_path, expected_pdf_path):
                        problems.append(
                            f"samples[{index}].pdf_path must match dataset contract sample {sample_name}: "
                            f"expected={expected_pdf_path}, got={actual_pdf_path}"
                        )
                problems.extend(
                    _validate_summary_sample_metric_means(
                        summary=summary,
                        samples=samples,
                        summary_fields=tuple(spec["required_summary_fields"]),
                    )
                )
            if isinstance(samples, list) and sample_count_int == len(samples) and expected_sample_names is not None:
                actual_names = {
                    str(sample.get("name")).strip()
                    for sample in samples
                    if isinstance(sample, dict) and str(sample.get("name") or "").strip()
                }
                expected_names = set(expected_sample_names)
                missing = sorted(expected_names - actual_names)
                unexpected = sorted(actual_names - expected_names)
                if missing or unexpected:
                    problems.append(
                        "samples names must match dataset contract: "
                        f"missing={','.join(missing) or '-'}, unexpected={','.join(unexpected) or '-'}"
                    )

    return {
        "status": "failed" if problems else "ok",
        "path": str(report_path),
        "schema_version": payload.get("schema_version"),
        "dataset": payload.get("dataset"),
        "pipeline_config": pipeline_config,
        "summary": summary,
        "model_manifest": payload.get("model_manifest"),
        "license_gate": license_gate,
        "dataset_contract": dataset_contract,
        "problems": problems,
    }


def _same_dataset_path(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left == right
    try:
        return Path(str(left)).expanduser().resolve(strict=False) == Path(str(right)).expanduser().resolve(strict=False)
    except Exception:
        return str(left) == str(right)


def _paired_ab_report(
    name: str,
    report_paths: dict[str, str | Path | None],
    *,
    expected_dataset: str | Path | None = None,
    expected_sample_count: int | None = None,
    expected_sample_names: tuple[str, ...] | None = None,
    expected_sample_pdf_paths: dict[str, str] | None = None,
    model_root: str | Path | None = None,
) -> dict[str, Any]:
    spec = DEFAULT_AB_REPORTS[name]
    baseline = _ab_report(
        name,
        report_paths.get(f"{name}_baseline"),
        expected_pipeline_config=spec.get("baseline_pipeline_config") or {},
        model_root=model_root,
        expected_dataset=expected_dataset,
        expected_sample_names=expected_sample_names,
        expected_sample_pdf_paths=expected_sample_pdf_paths,
    )
    candidate = _ab_report(
        name,
        report_paths.get(f"{name}_candidate"),
        expected_pipeline_config=spec.get("candidate_pipeline_config") or {},
        model_root=model_root,
        expected_dataset=expected_dataset,
        expected_sample_names=expected_sample_names,
        expected_sample_pdf_paths=expected_sample_pdf_paths,
    )
    problems: list[str] = []
    if baseline["status"] != "ok":
        problems.extend(f"baseline: {problem}" for problem in baseline.get("problems", []))
    if candidate["status"] != "ok":
        problems.extend(f"candidate: {problem}" for problem in candidate.get("problems", []))
    if baseline["status"] != "missing" and candidate["status"] != "missing":
        baseline_dataset = baseline.get("dataset")
        candidate_dataset = candidate.get("dataset")
        if baseline_dataset != candidate_dataset:
            problems.append(
                "A/B reports must use the same dataset: "
                f"baseline={baseline_dataset}, candidate={candidate_dataset}"
            )
        elif expected_dataset is not None and not _same_dataset_path(baseline_dataset, expected_dataset):
            problems.append(
                "A/B report dataset must match readiness dataset: "
                f"expected={expected_dataset}, got={baseline_dataset}"
            )
        baseline_sample_count = baseline.get("summary", {}).get("sample_count")
        candidate_sample_count = candidate.get("summary", {}).get("sample_count")
        if baseline_sample_count != candidate_sample_count:
            problems.append(
                "A/B reports must use the same sample_count: "
                f"baseline={baseline_sample_count}, candidate={candidate_sample_count}"
            )
        elif expected_sample_count is not None:
            sample_count_problems: list[str] = []
            actual_sample_count = _positive_int(
                baseline_sample_count,
                problem_label="A/B report sample_count",
                problems=sample_count_problems,
            )
            if actual_sample_count != expected_sample_count:
                problems.append(
                    "A/B report sample_count must match dataset contract: "
                    f"expected={expected_sample_count}, got={baseline_sample_count}"
                )
        baseline_summary = baseline.get("summary", {})
        candidate_summary = candidate.get("summary", {})
        for metric in spec.get("lower_or_equal_metrics", ()):
            baseline_value = baseline_summary.get(metric)
            candidate_value = candidate_summary.get(metric)
            if baseline_value is None or candidate_value is None:
                continue
            try:
                if float(candidate_value) > float(baseline_value):
                    problems.append(
                        f"candidate {metric} regressed: baseline={baseline_value}, candidate={candidate_value}"
                    )
            except Exception:
                problems.append(
                    f"metric {metric} must be numeric for A/B comparison"
                )
        for metric in spec.get("higher_or_equal_metrics", ()):
            baseline_value = baseline_summary.get(metric)
            candidate_value = candidate_summary.get(metric)
            if baseline_value is None or candidate_value is None:
                continue
            try:
                if float(candidate_value) < float(baseline_value):
                    problems.append(
                        f"candidate {metric} regressed: baseline={baseline_value}, candidate={candidate_value}"
                    )
            except Exception:
                problems.append(
                    f"metric {metric} must be numeric for A/B comparison"
                )
    if baseline["status"] == "missing" or candidate["status"] == "missing":
        status = "missing"
    elif baseline["status"] != "ok" or candidate["status"] != "ok" or problems:
        status = "failed"
    else:
        status = "ok"
    return {
        "status": status,
        "baseline": baseline,
        "candidate": candidate,
        "problems": problems,
    }


def check_readiness(
    *,
    model_root: str | Path | None = None,
    dataset: str | Path | None = None,
    min_pages: int = 100,
    report_paths: dict[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    root = Path(model_root or get_model_root())
    reports = report_paths or {}
    failed_gates: list[str] = []

    model_groups: dict[str, dict[str, Any]] = {}
    for group in REQUIRED_MODEL_GROUPS:
        missing = list_missing_files(group, model_root=str(root))
        model_groups[group] = {
            "status": "ok" if not missing else "missing",
            "missing_files": missing,
        }
        if missing:
            failed_gates.append(f"missing_model_group:{group}")

    dataset_status = _dataset_report(dataset, min_pages)
    if not dataset_status["meets_min_pages"]:
        failed_gates.append("dataset_pages_below_minimum")
    if dataset_status["unreadable_pdfs"]:
        failed_gates.append("dataset_has_unreadable_pdfs")
    dataset_contract = validate_dataset(dataset) if dataset else {
        "schema_version": "2026-06-08.cpu-pipeline-dataset-contract.v1",
        "status": "failed",
        "dataset": None,
        "sample_count": 0,
        "problems": ["dataset path is required"],
        "samples": [],
    }
    if dataset_contract["status"] != "ok":
        failed_gates.append("dataset_contract_failed")

    license_gate = _license_gate_report(reports.get("license_gate"))
    if license_gate["status"] != "ok":
        failed_gates.append(
            "missing_license_gate_report"
            if license_gate["status"] == "missing"
            else "failed_license_gate_report"
        )

    ab_reports: dict[str, dict[str, Any]] = {}
    expected_ab_sample_count = None
    if dataset_contract["status"] == "ok":
        expected_count_problems: list[str] = []
        expected_ab_sample_count = _positive_int(
            dataset_contract.get("sample_count"),
            problem_label="dataset_contract sample_count",
            problems=expected_count_problems,
        )
    expected_ab_sample_names = (
        tuple(
            sorted(
                str(sample.get("name")).strip()
                for sample in dataset_contract.get("samples", [])
                if isinstance(sample, dict) and str(sample.get("name") or "").strip()
            )
        )
        if dataset_contract["status"] == "ok"
        else None
    )
    expected_ab_sample_pdf_paths = (
        {
            str(sample.get("name")).strip(): str(sample.get("pdf_path")).strip()
            for sample in dataset_contract.get("samples", [])
            if isinstance(sample, dict)
            and str(sample.get("name") or "").strip()
            and str(sample.get("pdf_path") or "").strip()
        }
        if dataset_contract["status"] == "ok"
        else None
    )
    expected_profile_pdf_paths = (
        tuple(
            sorted(
                str(sample.get("pdf_path")).strip()
                for sample in dataset_contract.get("samples", [])
                if isinstance(sample, dict) and str(sample.get("pdf_path") or "").strip()
            )
        )
        if dataset_contract["status"] == "ok"
        else None
    )
    for name, spec in DEFAULT_AB_REPORTS.items():
        report = (
            _paired_ab_report(
                name,
                reports,
                expected_dataset=dataset,
                expected_sample_count=expected_ab_sample_count,
                expected_sample_names=expected_ab_sample_names,
                expected_sample_pdf_paths=expected_ab_sample_pdf_paths,
                model_root=root,
            )
            if spec["paired"]
            else _ab_report(
                name,
                reports.get(name),
                expected_profile_dataset=dataset,
                expected_profile_sample_names=expected_ab_sample_names,
                expected_profile_pdf_paths=expected_profile_pdf_paths,
                expected_sample_pdf_paths=expected_ab_sample_pdf_paths,
                model_root=root,
            )
        )
        ab_reports[name] = report
        if report["status"] != "ok":
            failed_gates.append(f"missing_ab_report:{name}" if report["status"] == "missing" else f"failed_ab_report:{name}")

    return {
        "schema_version": "2026-06-08.cpu-pipeline-readiness.v1",
        "status": "ok" if not failed_gates else "failed",
        "model_root": str(root),
        "model_groups": model_groups,
        "dataset": dataset_status,
        "dataset_contract": dataset_contract,
        "license_gate": license_gate,
        "ab_reports": ab_reports,
        "failed_gates": failed_gates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DocPilot CPU pipeline upgrade readiness gates.")
    parser.add_argument("--model-root", default=get_model_root(), help="Model root to inspect.")
    parser.add_argument("--dataset", help="Evaluation dataset directory containing real PDFs.")
    parser.add_argument("--min-pages", type=int, default=100, help="Minimum real PDF page count required.")
    parser.add_argument("--ocr-baseline-report", help="OCR baseline JSON report, usually v4.")
    parser.add_argument("--ocr-candidate-report", help="OCR candidate JSON report, usually v5.")
    parser.add_argument("--layout-baseline-report", help="Layout baseline JSON report, usually legacy.")
    parser.add_argument("--layout-candidate-report", help="Layout candidate JSON report, usually ppdoclayout.")
    parser.add_argument("--table-baseline-report", help="Table baseline JSON report, usually tatr.")
    parser.add_argument("--table-candidate-report", help="Table candidate JSON report, usually rapidtable.")
    parser.add_argument("--formula-baseline-report", help="Formula baseline JSON report, usually rapidlatex.")
    parser.add_argument("--formula-candidate-report", help="Formula candidate JSON report, usually pp_formula_net_s.")
    parser.add_argument("--profile-report", help="Pipeline profiling JSON report.")
    parser.add_argument("--license-gate-report", help="CPU pipeline license gate JSON report.")
    parser.add_argument("--out", help="Optional JSON readiness report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = check_readiness(
        model_root=args.model_root,
        dataset=args.dataset,
        min_pages=args.min_pages,
        report_paths={
            "ocr_v5_baseline": args.ocr_baseline_report,
            "ocr_v5_candidate": args.ocr_candidate_report,
            "layout_v2_baseline": args.layout_baseline_report,
            "layout_v2_candidate": args.layout_candidate_report,
            "table_v2_baseline": args.table_baseline_report,
            "table_v2_candidate": args.table_candidate_report,
            "formula_v2_baseline": args.formula_baseline_report,
            "formula_v2_candidate": args.formula_candidate_report,
            "profile": args.profile_report,
            "license_gate": args.license_gate_report,
        },
    )
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

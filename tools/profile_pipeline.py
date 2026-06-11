#!/usr/bin/env python3
"""Profile the local DocPilot CPU parser pipeline by parser stage."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.model_store import build_model_manifest
from tools.eval_omnidocbench import license_gate_report, validate_dataset


def _stage(rows: list[dict[str, Any]], name: str, fn):
    started_at = time.perf_counter()
    result = fn()
    rows.append({"stage": name, "elapsed_seconds": time.perf_counter() - started_at})
    return result


def _stage_summary(rows: list[dict[str, Any]], total_elapsed_seconds: float) -> dict[str, Any]:
    total = float(total_elapsed_seconds)
    ranked = sorted(
        (
            {
                "stage": str(row["stage"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "share": float(row["elapsed_seconds"]) / total if total > 0 else 0.0,
            }
            for row in rows
        ),
        key=lambda row: (-row["elapsed_seconds"], row["stage"]),
    )
    slowest = ranked[0] if ranked else {"stage": None, "elapsed_seconds": 0.0, "share": 0.0}
    return {
        "stage_count": len(rows),
        "slowest_stage": slowest["stage"],
        "slowest_stage_elapsed_seconds": slowest["elapsed_seconds"],
        "slowest_stage_share": slowest["share"],
        "stages_by_elapsed_seconds": ranked,
    }


def _dataset_sample_name(pdf_path: Path, dataset: str | Path | None) -> str:
    if dataset is None:
        return pdf_path.stem
    dataset_path = Path(dataset)
    try:
        relative_path = pdf_path.expanduser().resolve(strict=False).relative_to(
            dataset_path.expanduser().resolve(strict=False)
        )
    except ValueError:
        return pdf_path.stem
    return relative_path.with_suffix("").as_posix()


def _dataset_contract_sample_name(pdf_path: Path, dataset_contract: dict[str, Any]) -> str | None:
    samples = dataset_contract.get("samples")
    if not isinstance(samples, list):
        return None
    try:
        actual_pdf_path = pdf_path.expanduser().resolve(strict=False)
    except Exception:
        actual_pdf_path = pdf_path
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_pdf_path = str(sample.get("pdf_path") or "").strip()
        if not sample_pdf_path:
            continue
        try:
            expected_pdf_path = Path(sample_pdf_path).expanduser().resolve(strict=False)
        except Exception:
            expected_pdf_path = Path(sample_pdf_path)
        if actual_pdf_path == expected_pdf_path:
            sample_name = str(sample.get("name") or "").strip()
            return sample_name or Path(sample_pdf_path).stem
    return None


def _required_model_groups_for_profile(pipeline_config: dict[str, str]) -> tuple[str, ...]:
    groups: list[str] = []

    def add(group: str | None) -> None:
        if group and group not in groups:
            groups.append(group)

    ocr_version = str(pipeline_config.get("ocr_version") or "").strip().lower()
    add("core_v5" if ocr_version == "v5" else "core")

    layout_engine = str(pipeline_config.get("layout_engine") or "").strip().lower()
    if layout_engine == "ppdoclayout":
        add("layout_v2")
    elif layout_engine == "legacy":
        add("core")

    table_engine = str(pipeline_config.get("table_engine") or "").strip().lower()
    if table_engine == "rapidtable":
        add("table_v2")
    elif table_engine == "tatr":
        add("core")

    formula_mode = str(pipeline_config.get("formula_mode") or "").strip().lower()
    if formula_mode == "pp_formula_net_s":
        add("formula_v2")
    elif formula_mode == "rapidlatex":
        add("formula")

    return tuple(sorted(groups))


def _unbound_dataset_contract() -> dict[str, Any]:
    return {
        "schema_version": "2026-06-08.cpu-pipeline-dataset-contract.v1",
        "status": "failed",
        "dataset": None,
        "sample_count": 0,
        "problems": ["dataset path is required"],
        "samples": [],
    }


@contextmanager
def _temporary_env(values: dict[str, str | None]):
    original = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                continue
            os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def profile_pipeline(
    pdf_path: str | Path,
    *,
    dataset: str | Path | None = None,
    zoomin: int = 3,
    table_engine: str | None = None,
    layout_engine: str | None = None,
    ocr_version: str | None = None,
    formula_mode: str | None = None,
    reading_order_strategy: str | None = None,
) -> dict[str, Any]:
    source_path = Path(pdf_path)
    sample_name = _dataset_sample_name(source_path, dataset)
    dataset_contract = _unbound_dataset_contract()
    if dataset is not None:
        dataset_contract = validate_dataset(dataset)
        if dataset_contract["status"] != "ok":
            problems = "; ".join(str(problem) for problem in dataset_contract.get("problems", []))
            raise SystemExit(f"Dataset contract validation failed: {problems}")
        contract_sample_name = _dataset_contract_sample_name(source_path, dataset_contract)
        if contract_sample_name is None:
            raise SystemExit(
                "Profile PDF must be one of dataset contract samples: "
                f"expected one of {dataset}, got {source_path}"
            )
        sample_name = contract_sample_name

    env_overrides = {
        "DEEPDOC_TABLE_ENGINE": table_engine,
        "DEEPDOC_LAYOUT_ENGINE": layout_engine,
        "DEEPDOC_OCR_VERSION": ocr_version,
        "DEEPDOC_FORMULA_MODE": formula_mode,
        "DEEPDOC_READING_ORDER_STRATEGY": reading_order_strategy,
    }
    with _temporary_env(env_overrides):
        from deepdoc.parser.pdf_parser import DeepDocPdfParser

        parser = DeepDocPdfParser()
        rows: list[dict[str, Any]] = []
        source = str(source_path)
        _stage(rows, "rasterize_ocr", lambda: parser.__images__(source, zoomin))
        _stage(rows, "layout", lambda: parser._layouts_rec(zoomin))
        _stage(rows, "table", lambda: parser._table_transformer_job(zoomin))
        _stage(rows, "text_merge", parser._text_merge)
        _stage(rows, "cross_page_text", parser._merge_cross_page_text)
        _stage(rows, "reading_order", lambda: parser._apply_reading_order_strategy(zoomin))
        _stage(rows, "extract_assets", lambda: parser._extract_table_figure(False, zoomin, True, False))
        total = sum(float(row["elapsed_seconds"]) for row in rows)
        pipeline_config = {
            "table_engine": os.environ.get("DEEPDOC_TABLE_ENGINE", "tatr"),
            "layout_engine": os.environ.get("DEEPDOC_LAYOUT_ENGINE", "legacy"),
            "ocr_version": os.environ.get("DEEPDOC_OCR_VERSION", "v4"),
            "formula_mode": os.environ.get("DEEPDOC_FORMULA_MODE", "rapidlatex"),
            "reading_order_strategy": os.environ.get("DEEPDOC_READING_ORDER_STRATEGY", "legacy"),
        }
        payload = {
            "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
            "pdf_path": source,
            "sample_name": sample_name,
            "dataset": str(dataset) if dataset is not None else None,
            "zoomin": zoomin,
            "table_engine": pipeline_config["table_engine"],
            "layout_engine": pipeline_config["layout_engine"],
            "ocr_version": pipeline_config["ocr_version"],
            "formula_mode": pipeline_config["formula_mode"],
            "reading_order_strategy": pipeline_config["reading_order_strategy"],
            "pipeline_config": pipeline_config,
            "license_gate": license_gate_report(),
            "model_manifest": build_model_manifest(groups=_required_model_groups_for_profile(pipeline_config)),
            "dataset_contract": dataset_contract,
            "total_elapsed_seconds": total,
            "stage_summary": _stage_summary(rows, total),
            "stages": rows,
        }
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile DocPilot local PDF parser stages.")
    parser.add_argument("pdf", help="PDF file to profile.")
    parser.add_argument("--dataset", help="Evaluation dataset root, used to record a dataset-relative sample_name.")
    parser.add_argument("--zoomin", type=int, default=3)
    parser.add_argument("--table-engine", choices=["tatr", "rapidtable"])
    parser.add_argument("--layout-engine", choices=["legacy", "ppdoclayout"])
    parser.add_argument("--ocr-version", choices=["v4", "v5"])
    parser.add_argument("--formula-mode", choices=["rapidlatex", "pp_formula_net_s"])
    parser.add_argument("--reading-order-strategy", choices=["legacy", "rules"])
    parser.add_argument("--out", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = profile_pipeline(
        args.pdf,
        dataset=args.dataset,
        zoomin=args.zoomin,
        table_engine=args.table_engine,
        layout_engine=args.layout_engine,
        ocr_version=args.ocr_version,
        formula_mode=args.formula_mode,
        reading_order_strategy=args.reading_order_strategy,
    )
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

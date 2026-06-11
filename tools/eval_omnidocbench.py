#!/usr/bin/env python3
"""DocPilot CPU pipeline evaluation harness.

The tool intentionally evaluates parser outputs and local pipeline metrics. It
does not provide retrieval, vectorization, answer generation, or remote model
chat features.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.model_store import build_model_manifest, get_model_group_provenance


CPU_PIPELINE_PROVENANCE_LICENSE_GROUPS = ("core_v5", "layout_v2", "table_v2", "formula", "formula_v2")
CPU_PIPELINE_EXTRA_LICENSE_CANDIDATES: tuple[dict[str, str], ...] = (
    {"name": "RapidTable", "license": "Apache-2.0", "status": "allowed"},
    {"name": "Docling", "license": "MIT", "status": "allowed"},
    {"name": "DocLayout-YOLO", "license": "AGPL-3.0", "status": "blocked"},
    {"name": "Marker", "license": "GPL-3.0", "status": "blocked"},
    {"name": "Surya", "license": "GPL-3.0", "status": "blocked"},
    {"name": "texify", "license": "GPL-3.0", "status": "blocked"},
)


def _provenance_license_candidates() -> tuple[dict[str, str], ...]:
    candidates: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for provenance in get_model_group_provenance(CPU_PIPELINE_PROVENANCE_LICENSE_GROUPS).values():
        name = str(provenance.get("component") or "").strip()
        if not name or name in seen_names:
            continue
        candidates.append(
            {
                "name": name,
                "license": str(provenance.get("license") or "unknown"),
                "status": str(provenance.get("license_status") or "review"),
            }
        )
        seen_names.add(name)
    return tuple(candidates)


def cpu_pipeline_license_candidates() -> tuple[dict[str, str], ...]:
    candidates: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for item in _provenance_license_candidates():
        candidates.append(dict(item))
        seen_names.add(item["name"])
    for item in CPU_PIPELINE_EXTRA_LICENSE_CANDIDATES:
        name = item["name"]
        if name in seen_names:
            continue
        candidates.append(dict(item))
        seen_names.add(name)
    return tuple(candidates)


CPU_PIPELINE_LICENSE_CANDIDATES: tuple[dict[str, str], ...] = cpu_pipeline_license_candidates()

BLOCK_TYPE_ALIASES = {
    "": "text",
    "unknown": "text",
    "paragraph": "text",
    "content": "text",
    "pdf_native_text": "text",
    "figure caption": "figure",
    "figure_caption": "figure",
    "table caption": "table",
    "table_caption": "table",
    "formula": "equation",
}


@dataclass(frozen=True)
class EvalSample:
    name: str
    pdf_path: Path
    text_path: Path | None
    blocks_path: Path | None
    tables_path: Path | None = None
    formulas_path: Path | None = None
    chunks_path: Path | None = None
    fields_path: Path | None = None


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


def compute_text_edit_distance(predicted: str, expected: str) -> int:
    if predicted == expected:
        return 0
    if not predicted:
        return len(expected)
    if not expected:
        return len(predicted)
    previous = list(range(len(expected) + 1))
    for row_index, predicted_char in enumerate(predicted, 1):
        current = [row_index]
        for col_index, expected_char in enumerate(expected, 1):
            current.append(
                min(
                    previous[col_index] + 1,
                    current[col_index - 1] + 1,
                    previous[col_index - 1] + int(predicted_char != expected_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_distance(predicted: str, expected: str) -> float:
    denominator = max(len(predicted), len(expected), 1)
    return float(compute_text_edit_distance(predicted, expected)) / denominator


def character_error_rate(predicted: str, expected: str) -> float:
    denominator = max(len(expected), 1)
    return float(compute_text_edit_distance(predicted, expected)) / denominator


def _sequence_edit_distance(predicted: list[str], expected: list[str]) -> int:
    if predicted == expected:
        return 0
    if not predicted:
        return len(expected)
    if not expected:
        return len(predicted)
    previous = list(range(len(expected) + 1))
    for row_index, predicted_item in enumerate(predicted, 1):
        current = [row_index]
        for col_index, expected_item in enumerate(expected, 1):
            current.append(
                min(
                    previous[col_index] + 1,
                    current[col_index - 1] + 1,
                    previous[col_index - 1] + int(predicted_item != expected_item),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(predicted: str, expected: str) -> float:
    predicted_words = predicted.split()
    expected_words = expected.split()
    denominator = max(len(expected_words), 1)
    return float(_sequence_edit_distance(predicted_words, expected_words)) / denominator


def _normalize_block_type(block: dict[str, Any]) -> str:
    raw_type = str(
        block.get("block_type")
        or block.get("layout_type")
        or block.get("type")
        or ""
    ).strip().lower()
    return BLOCK_TYPE_ALIASES.get(raw_type, raw_type)


def _normalize_block_text(block: dict[str, Any]) -> str:
    return " ".join(str(block.get("text") or "").split()).strip().lower()


def _normalize_match_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def _block_pages(block: dict[str, Any]) -> list[int]:
    raw_pages = block.get("merged_page_numbers") or block.get("page_numbers")
    if raw_pages is None and block.get("metadata"):
        metadata = block.get("metadata") or {}
        if isinstance(metadata, dict):
            raw_pages = metadata.get("merged_page_numbers") or metadata.get("page_numbers")
    if raw_pages is None and block.get("page_number") is not None:
        raw_pages = [block.get("page_number")]
    if not isinstance(raw_pages, list):
        raw_pages = [raw_pages] if raw_pages is not None else []
    pages: list[int] = []
    for page in raw_pages:
        try:
            pages.append(int(page))
        except (TypeError, ValueError):
            continue
    return sorted(set(pages))


def _is_cross_page_block(block: dict[str, Any]) -> bool:
    metadata = block.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("cross_page"):
        return True
    return len(_block_pages(block)) > 1


def _blocks_match(predicted: dict[str, Any], expected: dict[str, Any]) -> bool:
    if _normalize_block_type(predicted) != _normalize_block_type(expected):
        return False
    predicted_text = _normalize_block_text(predicted)
    expected_text = _normalize_block_text(expected)
    if expected_text and predicted_text:
        return predicted_text == expected_text
    predicted_pages = set(_block_pages(predicted))
    expected_pages = set(_block_pages(expected))
    return not expected_pages or bool(predicted_pages & expected_pages)


def evaluate_blocks(
    *,
    predicted_blocks: list[dict[str, Any]],
    expected_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    predicted_types = [_normalize_block_type(block) for block in predicted_blocks]
    expected_types = [_normalize_block_type(block) for block in expected_blocks]
    predicted_counter = Counter(predicted_types)
    expected_counter = Counter(expected_types)
    true_positive = sum((predicted_counter & expected_counter).values())
    precision = true_positive / len(predicted_types) if predicted_types else None
    recall = true_positive / len(expected_types) if expected_types else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )

    reading_order_edit_distance = _sequence_edit_distance(predicted_types, expected_types)
    reading_order_denominator = max(len(predicted_types), len(expected_types), 1)

    expected_cross_page = [block for block in expected_blocks if _is_cross_page_block(block)]
    predicted_cross_page = [block for block in predicted_blocks if _is_cross_page_block(block)]
    matched_predicted_indexes: set[int] = set()
    matched_cross_page = 0
    for expected in expected_cross_page:
        for index, predicted in enumerate(predicted_cross_page):
            if index in matched_predicted_indexes:
                continue
            if _blocks_match(predicted, expected):
                matched_predicted_indexes.add(index)
                matched_cross_page += 1
                break

    return {
        "predicted_block_count": len(predicted_blocks),
        "expected_block_count": len(expected_blocks),
        "block_type_precision": precision,
        "block_type_recall": recall,
        "block_type_f1": f1,
        "reading_order_edit_distance": reading_order_edit_distance,
        "reading_order_normalized_edit_distance": reading_order_edit_distance / reading_order_denominator,
        "expected_cross_page_block_count": len(expected_cross_page),
        "predicted_cross_page_block_count": len(predicted_cross_page),
        "matched_cross_page_block_count": matched_cross_page,
        "cross_page_merge_accuracy": matched_cross_page / len(expected_cross_page) if expected_cross_page else None,
    }


def _extract_html_cells(table_html: str) -> list[str]:
    try:
        from lxml import html as lxml_html

        root = lxml_html.fromstring(table_html)
        return [" ".join(cell.itertext()).strip() for cell in root.xpath("//td|//th")]
    except Exception:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", table_html, flags=re.IGNORECASE | re.DOTALL)
        return [" ".join(re.sub(r"<[^>]+>", " ", cell).split()).strip() for cell in cells]


def compute_table_cell_f1(predicted_html: str, expected_html: str) -> float | None:
    predicted_cells = [cell for cell in _extract_html_cells(predicted_html) if cell]
    expected_cells = [cell for cell in _extract_html_cells(expected_html) if cell]
    if not predicted_cells and not expected_cells:
        return None
    predicted_counter = Counter(predicted_cells)
    expected_counter = Counter(expected_cells)
    matched = sum((predicted_counter & expected_counter).values())
    precision = matched / len(predicted_cells) if predicted_cells else 0.0
    recall = matched / len(expected_cells) if expected_cells else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def compute_table_teds(predicted_html: str, expected_html: str) -> float | None:
    try:
        from tools.eval_table import compute_teds

        return float(compute_teds(predicted_html, expected_html))
    except Exception:
        return None


def evaluate_tables(*, predicted_tables: list[str], expected_tables: list[str]) -> dict[str, Any]:
    pair_count = min(len(predicted_tables), len(expected_tables))
    teds_values: list[float] = []
    cell_f1_values: list[float] = []
    for predicted_html, expected_html in zip(predicted_tables, expected_tables):
        teds = compute_table_teds(predicted_html, expected_html)
        if teds is not None:
            teds_values.append(teds)
        cell_f1 = compute_table_cell_f1(predicted_html, expected_html)
        if cell_f1 is not None:
            cell_f1_values.append(cell_f1)
    return {
        "predicted_table_count": len(predicted_tables),
        "expected_table_count": len(expected_tables),
        "compared_table_count": pair_count,
        "mean_table_teds": sum(teds_values) / len(teds_values) if teds_values else None,
        "mean_table_cell_f1": sum(cell_f1_values) / len(cell_f1_values) if cell_f1_values else None,
    }


def _load_json_list(path: Path, *, field_names: tuple[str, ...]) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for field_name in field_names:
            value = payload.get(field_name)
            if isinstance(value, list):
                return value
    raise SystemExit(f"Unsupported ground-truth JSON format: {path}")


def _expected_texts(items: list[Any]) -> list[str]:
    texts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            texts.append(text)
    return texts


def _normalize_formula_text(value: Any) -> str:
    text = str(value or "").strip()
    for prefix, suffix in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if text.startswith(prefix) and text.endswith(suffix):
            text = text[len(prefix) : len(text) - len(suffix)]
            break
    return re.sub(r"\s+", "", text.strip())


def _expected_formulas(items: list[Any]) -> list[str]:
    formulas: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("latex") or item.get("formula") or item.get("text") or item.get("content") or ""
        else:
            text = item
        normalized = _normalize_formula_text(text)
        if normalized:
            formulas.append(normalized)
    return formulas


def _predicted_formulas(blocks: list[dict[str, Any]]) -> list[str]:
    formulas: list[str] = []
    for block in blocks:
        if _normalize_block_type(block) != "equation":
            continue
        normalized = _normalize_formula_text(block.get("text"))
        if normalized:
            formulas.append(normalized)
    return formulas


def evaluate_formulas(
    *,
    predicted_blocks: list[dict[str, Any]],
    expected_formulas: list[Any],
) -> dict[str, Any]:
    predicted_texts = _predicted_formulas(predicted_blocks)
    expected_texts = _expected_formulas(expected_formulas)
    compared_count = max(len(predicted_texts), len(expected_texts))
    distances: list[float] = []
    exact_matches = 0
    for index in range(compared_count):
        predicted = predicted_texts[index] if index < len(predicted_texts) else ""
        expected = expected_texts[index] if index < len(expected_texts) else ""
        if predicted == expected and expected:
            exact_matches += 1
        distances.append(normalized_edit_distance(predicted, expected))
    return {
        "predicted_formula_count": len(predicted_texts),
        "expected_formula_count": len(expected_texts),
        "compared_formula_count": compared_count,
        "formula_normalized_edit_distance": sum(distances) / len(distances) if distances else None,
        "formula_exact_match_rate": exact_matches / len(expected_texts) if expected_texts else None,
    }


def _predicted_parse_blocks(blocks: list[dict[str, Any]]):
    from common.parse_artifacts import ParseBlock, count_tokens

    parse_blocks = []
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        block_type = _normalize_block_type(block)
        if block_type not in {"text", "title", "table", "figure", "equation", "seal", "barcode", "list", "reference"}:
            block_type = "text"
        pages = _block_pages(block)
        parse_blocks.append(
            ParseBlock(
                block_id=f"block-{len(parse_blocks):04d}",
                block_type=block_type,
                text=text,
                page_numbers=pages,
                token_count=count_tokens(text),
            )
        )
    return parse_blocks


def evaluate_chunks(
    *,
    predicted_blocks: list[dict[str, Any]],
    expected_chunks: list[Any],
) -> dict[str, Any]:
    from common.parse_artifacts import DEFAULT_CHUNK_MAX_TOKENS, DEFAULT_CHUNK_OVERLAP_TOKENS, build_chunks

    predicted_chunks = build_chunks(
        _predicted_parse_blocks(predicted_blocks),
        max_tokens=DEFAULT_CHUNK_MAX_TOKENS,
        overlap_tokens=DEFAULT_CHUNK_OVERLAP_TOKENS,
        strategy="structure_aware",
    )
    predicted_texts = [_normalize_match_text(chunk.text) for chunk in predicted_chunks]
    expected_texts = _expected_texts(expected_chunks)
    matched = 0
    for expected_text in expected_texts:
        normalized = _normalize_match_text(expected_text)
        if normalized and any(normalized in predicted_text for predicted_text in predicted_texts):
            matched += 1
    return {
        "predicted_chunk_count": len(predicted_chunks),
        "expected_chunk_count": len(expected_texts),
        "matched_chunk_count": matched,
        "chunk_text_coverage": matched / len(expected_texts) if expected_texts else None,
    }


def _field_pages(field: dict[str, Any]) -> list[int]:
    raw_pages = field.get("page_numbers")
    if raw_pages is None:
        raw_pages = field.get("pages")
    if raw_pages is None and field.get("page_number") is not None:
        raw_pages = [field.get("page_number")]
    if raw_pages is None and field.get("page") is not None:
        raw_pages = [field.get("page")]
    if not isinstance(raw_pages, list):
        raw_pages = [raw_pages] if raw_pages is not None else []
    pages: list[int] = []
    for page in raw_pages:
        try:
            pages.append(int(page))
        except (TypeError, ValueError):
            continue
    return sorted(set(pages))


def evaluate_business_fields(
    *,
    predicted_blocks: list[dict[str, Any]],
    expected_fields: list[Any],
) -> dict[str, Any]:
    normalized_blocks = [
        {
            "text": _normalize_match_text(block.get("text")),
            "pages": set(_block_pages(block)),
        }
        for block in predicted_blocks
        if str(block.get("text") or "").strip()
    ]
    normalized_fields = [field for field in expected_fields if isinstance(field, dict) and str(field.get("value") or "").strip()]
    matched = 0
    for field in normalized_fields:
        expected_value = _normalize_match_text(field.get("value"))
        expected_pages = set(_field_pages(field))
        for block in normalized_blocks:
            if expected_value not in block["text"]:
                continue
            if expected_pages and block["pages"] and not expected_pages.intersection(block["pages"]):
                continue
            matched += 1
            break
    return {
        "expected_business_field_count": len(normalized_fields),
        "matched_business_field_count": matched,
        "business_field_location_hit_rate": matched / len(normalized_fields) if normalized_fields else None,
    }


def license_gate_report() -> dict[str, Any]:
    provenance_candidates = _provenance_license_candidates()
    candidates = cpu_pipeline_license_candidates()
    blocked = [item for item in candidates if item["status"] == "blocked"]
    allowed = [item for item in candidates if item["status"] == "allowed"]
    provenance_ready = all(item["status"] == "allowed" for item in provenance_candidates)
    return {
        "schema_version": "2026-06-08.cpu-pipeline-license-gate.v1",
        "status": "passed" if provenance_ready else "review",
        "allowed": allowed,
        "blocked": blocked,
        "rule": "Do not integrate AGPL/GPL components into this Apache-2.0 local parser pipeline.",
    }


def _write_json_payload(payload: dict[str, Any], out: str | Path | None) -> None:
    if not out:
        return
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def discover_samples(dataset: str | Path) -> list[EvalSample]:
    root = Path(dataset)
    samples: list[EvalSample] = []
    for pdf_path in sorted(root.rglob("*.pdf")):
        sample_name = pdf_path.relative_to(root).with_suffix("").as_posix()
        table_path = None
        for candidate in (pdf_path.with_suffix(".gt.tables.html"), pdf_path.with_suffix(".gt.html")):
            if candidate.exists():
                table_path = candidate
                break
        samples.append(
            EvalSample(
                name=sample_name,
                pdf_path=pdf_path,
                text_path=pdf_path.with_suffix(".gt.txt") if pdf_path.with_suffix(".gt.txt").exists() else None,
                blocks_path=pdf_path.with_suffix(".gt.blocks.json") if pdf_path.with_suffix(".gt.blocks.json").exists() else None,
                tables_path=table_path,
                formulas_path=pdf_path.with_suffix(".gt.formulas.json") if pdf_path.with_suffix(".gt.formulas.json").exists() else None,
                chunks_path=pdf_path.with_suffix(".gt.chunks.json") if pdf_path.with_suffix(".gt.chunks.json").exists() else None,
                fields_path=pdf_path.with_suffix(".gt.fields.json") if pdf_path.with_suffix(".gt.fields.json").exists() else None,
            )
        )
    return samples


def _append_json_list_validation(
    *,
    path: Path,
    field_names: tuple[str, ...],
    item_label: str,
    text_fields: tuple[str, ...],
    problems: list[str],
) -> int:
    try:
        items = _load_json_list(path, field_names=field_names)
    except Exception as exc:
        problems.append(f"{path.name}: unsupported JSON format ({exc})")
        return 0
    for index, item in enumerate(items):
        if isinstance(item, str):
            if not item.strip():
                problems.append(f"{path.name}: {item_label}[{index}] is empty")
            continue
        if not isinstance(item, dict):
            problems.append(f"{path.name}: {item_label}[{index}] must be a string or object")
            continue
        if not any(str(item.get(field) or "").strip() for field in text_fields):
            problems.append(f"{path.name}: {item_label}[{index}] missing {'/'.join(text_fields)}")
    return len(items)


def _append_blocks_validation(path: Path, problems: list[str]) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        problems.append(f"{path.name}: invalid JSON ({exc})")
        return 0
    if isinstance(payload, list):
        blocks = payload
    elif isinstance(payload, dict) and isinstance(payload.get("blocks"), list):
        blocks = payload["blocks"]
    else:
        problems.append(f"{path.name}: expected block array or object with blocks array")
        return 0
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            problems.append(f"{path.name}: blocks[{index}] must be an object")
            continue
        if not str(block.get("text") or "").strip():
            problems.append(f"{path.name}: blocks[{index}] missing text")
    return len(blocks)


def _append_tables_validation(path: Path, problems: list[str]) -> int:
    try:
        tables = _load_expected_tables(path)
    except Exception as exc:
        problems.append(f"{path.name}: unsupported table ground-truth format ({exc})")
        return 0
    for index, table in enumerate(tables):
        if "<table" not in table.lower():
            problems.append(f"{path.name}: tables[{index}] missing <table> markup")
    return len(tables)


def validate_dataset(dataset: str | Path) -> dict[str, Any]:
    root = Path(dataset)
    samples = discover_samples(root)
    problems: list[str] = []
    if not root.exists():
        problems.append(f"dataset path does not exist: {root}")
    if not samples:
        problems.append(f"No evaluation samples found under {root}. Expected *.pdf files.")

    known_suffixes = {
        ".gt.txt",
        ".gt.blocks.json",
        ".gt.tables.html",
        ".gt.html",
        ".gt.formulas.json",
        ".gt.chunks.json",
        ".gt.fields.json",
    }
    sample_stems = {sample.name for sample in samples}
    for path in sorted(root.rglob("*.gt.*")):
        matched_suffix = next((suffix for suffix in known_suffixes if path.name.endswith(suffix)), None)
        if matched_suffix is None:
            problems.append(f"{path.name}: unsupported ground-truth file suffix")
            continue
        relative_name = path.relative_to(root).as_posix()
        stem = relative_name[: -len(matched_suffix)]
        if stem not in sample_stems:
            problems.append(f"{path.name}: ground-truth file has no matching PDF")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_counts: dict[str, int] = {}
        if sample.pdf_path.stat().st_size <= 0:
            problems.append(f"{sample.pdf_path.name}: empty PDF file")
        if sample.text_path is not None:
            text = sample.text_path.read_text(encoding="utf-8").strip()
            sample_counts["text"] = 1 if text else 0
            if not text:
                problems.append(f"{sample.text_path.name}: empty text ground truth")
        if sample.blocks_path is not None:
            sample_counts["blocks"] = _append_blocks_validation(sample.blocks_path, problems)
        if sample.tables_path is not None:
            sample_counts["tables"] = _append_tables_validation(sample.tables_path, problems)
        if sample.formulas_path is not None:
            sample_counts["formulas"] = _append_json_list_validation(
                path=sample.formulas_path,
                field_names=("formulas", "equations", "expected_formulas"),
                item_label="formulas",
                text_fields=("latex", "formula", "text", "content"),
                problems=problems,
            )
        if sample.chunks_path is not None:
            sample_counts["chunks"] = _append_json_list_validation(
                path=sample.chunks_path,
                field_names=("chunks", "expected_chunks"),
                item_label="chunks",
                text_fields=("text", "content"),
                problems=problems,
            )
        if sample.fields_path is not None:
            sample_counts["fields"] = _append_json_list_validation(
                path=sample.fields_path,
                field_names=("fields", "business_fields", "expected_fields"),
                item_label="fields",
                text_fields=("value",),
                problems=problems,
            )
        rows.append(
            {
                "name": sample.name,
                "pdf_path": str(sample.pdf_path),
                "has_text_gt": sample.text_path is not None,
                "has_blocks_gt": sample.blocks_path is not None,
                "has_tables_gt": sample.tables_path is not None,
                "has_formulas_gt": sample.formulas_path is not None,
                "has_chunks_gt": sample.chunks_path is not None,
                "has_business_fields_gt": sample.fields_path is not None,
                "ground_truth_counts": sample_counts,
            }
        )

    return {
        "schema_version": "2026-06-08.cpu-pipeline-dataset-contract.v1",
        "status": "ok" if not problems else "failed",
        "dataset": str(root),
        "sample_count": len(samples),
        "problems": problems,
        "samples": rows,
    }


def _parse_deepdoc_blocks(pdf_path: Path, *, engine: str) -> list[dict[str, Any]]:
    if engine != "deepdoc":
        raise SystemExit(f"Unsupported local evaluation engine: {engine}")
    from deepdoc.parser.pdf_parser import DeepDocPdfParser

    parser = DeepDocPdfParser()
    return parser.parse_into_bboxes(str(pdf_path), zoomin=3)


def _blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(str(block.get("text") or "") for block in blocks if str(block.get("text") or "").strip())


def _parse_deepdoc_markdown(pdf_path: Path, *, engine: str) -> str:
    return _blocks_to_text(_parse_deepdoc_blocks(pdf_path, engine=engine))


def _load_expected_blocks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("blocks"), list):
        return [item for item in payload["blocks"] if isinstance(item, dict)]
    raise SystemExit(f"Unsupported block ground-truth format: {path}")


def _load_expected_chunks(path: Path) -> list[Any]:
    return _load_json_list(path, field_names=("chunks", "expected_chunks"))


def _load_expected_business_fields(path: Path) -> list[Any]:
    return _load_json_list(path, field_names=("fields", "business_fields", "expected_fields"))


def _load_expected_formulas(path: Path) -> list[Any]:
    return _load_json_list(path, field_names=("formulas", "equations", "expected_formulas"))


def _load_expected_tables(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(raw)
        if isinstance(payload, list):
            return [str(item) for item in payload if str(item).strip()]
        if isinstance(payload, dict) and isinstance(payload.get("tables"), list):
            return [str(item) for item in payload["tables"] if str(item).strip()]
    try:
        from lxml import html as lxml_html
        from lxml import etree

        root = lxml_html.fromstring(raw)
        tables = root.xpath("//table")
        return [etree.tostring(table, encoding="unicode") for table in tables] or [raw]
    except Exception:
        tables = re.findall(r"<table\b.*?</table>", raw, flags=re.IGNORECASE | re.DOTALL)
        return tables or [raw]


def _extract_predicted_tables(blocks: list[dict[str, Any]]) -> list[str]:
    tables = []
    for block in blocks:
        if _normalize_block_type(block) != "table":
            continue
        text = str(block.get("text") or "").strip()
        if text:
            tables.append(text)
    return tables


def _evaluate_sample(sample: EvalSample, *, engine: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    blocks = _parse_deepdoc_blocks(sample.pdf_path, engine=engine)
    markdown = _blocks_to_text(blocks)
    elapsed = time.perf_counter() - started_at
    row: dict[str, Any] = {
        "name": sample.name,
        "pdf_path": str(sample.pdf_path),
        "engine": engine,
        "elapsed_seconds": elapsed,
        "text_length": len(markdown),
        "has_text_gt": sample.text_path is not None,
        "has_blocks_gt": sample.blocks_path is not None,
        "has_tables_gt": sample.tables_path is not None,
        "has_formulas_gt": sample.formulas_path is not None,
        "has_chunks_gt": sample.chunks_path is not None,
        "has_business_fields_gt": sample.fields_path is not None,
    }
    if sample.text_path is not None:
        expected_text = sample.text_path.read_text(encoding="utf-8")
        row["text_edit_distance"] = compute_text_edit_distance(markdown, expected_text)
        row["text_normalized_edit_distance"] = normalized_edit_distance(markdown, expected_text)
        row["character_error_rate"] = character_error_rate(markdown, expected_text)
        row["word_error_rate"] = word_error_rate(markdown, expected_text)
    if sample.blocks_path is not None:
        row.update(
            evaluate_blocks(
                predicted_blocks=blocks,
                expected_blocks=_load_expected_blocks(sample.blocks_path),
            )
        )
    if sample.tables_path is not None:
        row.update(
            evaluate_tables(
                predicted_tables=_extract_predicted_tables(blocks),
                expected_tables=_load_expected_tables(sample.tables_path),
            )
        )
    if sample.formulas_path is not None:
        row.update(
            evaluate_formulas(
                predicted_blocks=blocks,
                expected_formulas=_load_expected_formulas(sample.formulas_path),
            )
        )
    if sample.chunks_path is not None:
        row.update(
            evaluate_chunks(
                predicted_blocks=blocks,
                expected_chunks=_load_expected_chunks(sample.chunks_path),
            )
        )
    if sample.fields_path is not None:
        row.update(
            evaluate_business_fields(
                predicted_blocks=blocks,
                expected_fields=_load_expected_business_fields(sample.fields_path),
            )
        )
    return row


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed_values = [float(row["elapsed_seconds"]) for row in rows]
    ned_values = [
        float(row["text_normalized_edit_distance"])
        for row in rows
        if "text_normalized_edit_distance" in row
    ]
    return {
        "sample_count": len(rows),
        "mean_elapsed_seconds": sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0,
        "mean_text_normalized_edit_distance": sum(ned_values) / len(ned_values) if ned_values else None,
        "mean_character_error_rate": _mean(rows, "character_error_rate"),
        "mean_word_error_rate": _mean(rows, "word_error_rate"),
        "samples_with_text_gt": len(ned_values),
        "mean_block_type_f1": _mean(rows, "block_type_f1"),
        "mean_reading_order_normalized_edit_distance": _mean(rows, "reading_order_normalized_edit_distance"),
        "mean_cross_page_merge_accuracy": _mean(rows, "cross_page_merge_accuracy"),
        "samples_with_blocks_gt": len([row for row in rows if row.get("has_blocks_gt")]),
        "mean_table_teds": _mean(rows, "mean_table_teds"),
        "mean_table_cell_f1": _mean(rows, "mean_table_cell_f1"),
        "samples_with_tables_gt": len([row for row in rows if row.get("has_tables_gt")]),
        "mean_formula_normalized_edit_distance": _mean(rows, "formula_normalized_edit_distance"),
        "mean_formula_exact_match_rate": _mean(rows, "formula_exact_match_rate"),
        "samples_with_formulas_gt": len([row for row in rows if row.get("has_formulas_gt")]),
        "mean_chunk_text_coverage": _mean(rows, "chunk_text_coverage"),
        "samples_with_chunks_gt": len([row for row in rows if row.get("has_chunks_gt")]),
        "mean_business_field_location_hit_rate": _mean(rows, "business_field_location_hit_rate"),
        "samples_with_business_fields_gt": len([row for row in rows if row.get("has_business_fields_gt")]),
    }


def _required_model_groups_for_eval(pipeline_config: dict[str, str]) -> tuple[str, ...]:
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


def evaluate_dataset(
    *,
    engine: str,
    dataset: str | Path,
    out: str | Path | None = None,
    ocr_version: str | None = None,
    layout_engine: str | None = None,
    table_engine: str | None = None,
    formula_mode: str | None = None,
    reading_order_strategy: str | None = None,
) -> dict[str, Any]:
    dataset_contract = validate_dataset(dataset)
    if dataset_contract["status"] != "ok":
        problems = "; ".join(str(problem) for problem in dataset_contract.get("problems", []))
        raise SystemExit(f"Dataset contract validation failed: {problems}")
    samples = discover_samples(dataset)
    if not samples:
        raise SystemExit(f"No evaluation samples found under {dataset}. Expected *.pdf files.")
    env_overrides = {
        "DEEPDOC_OCR_VERSION": ocr_version,
        "DEEPDOC_LAYOUT_ENGINE": layout_engine,
        "DEEPDOC_TABLE_ENGINE": table_engine,
        "DEEPDOC_FORMULA_MODE": formula_mode,
        "DEEPDOC_READING_ORDER_STRATEGY": reading_order_strategy,
    }
    with _temporary_env(env_overrides):
        rows = [_evaluate_sample(sample, engine=engine) for sample in samples]
        pipeline_config = {
            "ocr_version": os.environ.get("DEEPDOC_OCR_VERSION", "v4"),
            "layout_engine": os.environ.get("DEEPDOC_LAYOUT_ENGINE", "legacy"),
            "table_engine": os.environ.get("DEEPDOC_TABLE_ENGINE", "tatr"),
            "formula_mode": os.environ.get("DEEPDOC_FORMULA_MODE", "rapidlatex"),
            "reading_order_strategy": os.environ.get("DEEPDOC_READING_ORDER_STRATEGY", "legacy"),
        }
    payload = {
        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
        "engine": engine,
        "dataset": str(dataset),
        "pipeline_config": pipeline_config,
        "model_manifest": build_model_manifest(groups=_required_model_groups_for_eval(pipeline_config)),
        "dataset_contract": dataset_contract,
        "summary": _summarize(rows),
        "license_gate": license_gate_report(),
        "samples": rows,
    }
    if out:
        _write_json_payload(payload, out)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DocPilot CPU pipeline on a local PDF dataset.")
    parser.add_argument("--engine", default="deepdoc", choices=["deepdoc"])
    parser.add_argument(
        "--dataset",
        help=(
            "Directory containing *.pdf and optional *.gt.txt, "
            "*.gt.blocks.json, *.gt.tables.html, *.gt.formulas.json, "
            "*.gt.chunks.json, *.gt.fields.json or *.gt.html files."
        ),
    )
    parser.add_argument("--out", help="Optional JSON report path.")
    parser.add_argument("--ocr-version", choices=["v4", "v5"], help="Temporarily set DEEPDOC_OCR_VERSION for this run.")
    parser.add_argument(
        "--layout-engine",
        choices=["legacy", "ppdoclayout"],
        help="Temporarily set DEEPDOC_LAYOUT_ENGINE for this run.",
    )
    parser.add_argument(
        "--table-engine",
        choices=["tatr", "rapidtable"],
        help="Temporarily set DEEPDOC_TABLE_ENGINE for this run.",
    )
    parser.add_argument(
        "--formula-mode",
        choices=["rapidlatex", "pp_formula_net_s"],
        help="Temporarily set DEEPDOC_FORMULA_MODE for this run.",
    )
    parser.add_argument(
        "--reading-order-strategy",
        choices=["legacy", "rules"],
        help="Temporarily set DEEPDOC_READING_ORDER_STRATEGY for this run.",
    )
    parser.add_argument("--license-gate", action="store_true", help="Print only the CPU pipeline license gate report.")
    parser.add_argument(
        "--validate-dataset",
        action="store_true",
        help="Validate the dataset ground-truth contract without parsing PDFs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.license_gate:
        payload = license_gate_report()
        _write_json_payload(payload, args.out)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not args.dataset:
        raise SystemExit("--dataset is required unless --license-gate is used.")
    if args.validate_dataset:
        payload = validate_dataset(args.dataset)
        _write_json_payload(payload, args.out)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["status"] == "ok" else 1
    payload = evaluate_dataset(
        engine=args.engine,
        dataset=args.dataset,
        out=args.out,
        ocr_version=args.ocr_version,
        layout_engine=args.layout_engine,
        table_engine=args.table_engine,
        formula_mode=args.formula_mode,
        reading_order_strategy=args.reading_order_strategy,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

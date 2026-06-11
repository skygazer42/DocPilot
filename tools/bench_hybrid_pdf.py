#!/usr/bin/env python3
"""Summarize hybrid PDF routing decisions for born-digital vs OCR pages."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def analyze_hybrid_pdf(
    pdf_path: str | Path,
    *,
    page_to: int = 299,
    mode: str = "hybrid",
    profile: str = "gpu",
    parser_cls=None,
) -> dict[str, Any]:
    from deepdoc.parser.pdf_hybrid_router import build_pdf_hybrid_plan
    from deepdoc.parser.pdf_parser import detect_pdf_text_layer
    from deepdoc.parser.pdf_parser import DeepDocPdfParser
    import main as deepdoc_main

    source_path = Path(pdf_path).expanduser().resolve(strict=False)
    text_layer = detect_pdf_text_layer(str(source_path), page_from=0, page_to=page_to)
    plan = build_pdf_hybrid_plan(str(source_path), page_from=0, page_to=page_to)
    parse_started_at = time.perf_counter()
    _rows, _tables, parse_meta = deepdoc_main._parse_pdf_from_tmp(
        parser_cls or DeepDocPdfParser,
        str(source_path),
        {
            "parser_engine": "deepdoc",
            "deepdoc_pdf_mode": mode,
            "execution_profile": profile,
            "deepdoc_layout_model": "general",
            "deepdoc_max_pages": page_to,
            "return_structured": False,
            "persist_artifacts": False,
            "include_chunks": False,
            "return_images": False,
            "enable_formula": False,
            "enable_seal": False,
        },
    )
    parse_elapsed_seconds = round(time.perf_counter() - parse_started_at, 6)
    pages = plan.get("pages") or []
    route_examples = [
        {
            "page_number": int(page.get("page_number") or 0),
            "route": str(page.get("route") or ""),
            "ocr_scope": str(page.get("ocr_scope") or ""),
            "complex_block_types": list(page.get("complex_block_types") or []),
            "reasons": list(page.get("reasons") or []),
        }
        for page in pages[: min(10, len(pages))]
    ]
    return {
        "schema_version": "2026-06-11.hybrid-bench.v1",
        "pdf_path": str(source_path),
        "page_to": int(page_to),
        "pdf_text_layer": text_layer,
        "route_summary": plan.get("route_summary") or {},
        "hybrid_route_summary": parse_meta.get("hybrid_route_summary") or plan.get("route_summary") or {},
        "ocr_page_numbers": list(plan.get("ocr_page_numbers") or []),
        "complex_block_page_numbers": list(plan.get("complex_block_page_numbers") or []),
        "native_box_count": len(plan.get("native_boxes") or []),
        "stage_timings": parse_meta.get("stage_timings") or {},
        "ocr_block_count": int(parse_meta.get("ocr_block_count") or 0),
        "complex_block_counts": parse_meta.get("complex_block_counts") or {},
        "parse_elapsed_seconds": parse_elapsed_seconds,
        "route_examples": route_examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize DocPilot hybrid PDF routing decisions.")
    parser.add_argument("--input", required=True, help="Path to the PDF file")
    parser.add_argument("--page-to", type=int, default=299, help="Exclusive upper page bound for analysis")
    parser.add_argument("--mode", default="hybrid", choices=["auto", "native", "ocr", "hybrid"], help="DocPilot PDF mode")
    parser.add_argument("--profile", default="gpu", choices=["auto", "cpu", "gpu"], help="Execution profile")
    args = parser.parse_args()

    payload = analyze_hybrid_pdf(args.input, page_to=args.page_to, mode=args.mode, profile=args.profile)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

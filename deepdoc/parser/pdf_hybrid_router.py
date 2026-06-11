from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from deepdoc.parser.pdf_parser import collect_pdf_page_features_and_native_boxes


HYBRID_PLAN_SCHEMA_VERSION = "2026-06-11.pdf-hybrid-plan.v1"


def _join_native_word_text(left: str, right: str) -> str:
    left = str(left or "").rstrip()
    right = str(right or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    if re.match(r"[A-Za-z0-9]$", left) and re.match(r"[A-Za-z0-9]", right):
        return f"{left} {right}"
    if re.match(r"[\u4e00-\u9fff]$", left) and re.match(r"[\u4e00-\u9fff]", right):
        return f"{left}{right}"
    if left[-1] in "([【《「‘“" or right[:1] in ".,;:!?，。；？！、)]】》」’”":
        return f"{left}{right}"
    return f"{left} {right}"


def _collapse_native_page_boxes(boxes: list[dict[str, Any]], *, page_width: float = 0.0) -> list[dict[str, Any]]:
    if len(boxes) <= 1:
        return [dict(box) for box in boxes]

    sorted_boxes = sorted(
        [dict(box) for box in boxes if str(box.get("text") or "").strip()],
        key=lambda box: (
            float(box.get("top") or 0.0),
            float(box.get("x0") or 0.0),
        ),
    )
    if len(sorted_boxes) <= 1:
        return sorted_boxes

    collapsed: list[dict[str, Any]] = []
    current_run: dict[str, Any] | None = None
    current_mid = 0.0
    current_height = 0.0

    def flush_run() -> None:
        nonlocal current_run, current_mid, current_height
        if current_run is not None:
            collapsed.append(current_run)
        current_run = None
        current_mid = 0.0
        current_height = 0.0

    for box in sorted_boxes:
        top = float(box.get("top") or 0.0)
        bottom = float(box.get("bottom") or 0.0)
        height = max(1.0, bottom - top)
        mid = (top + bottom) / 2.0
        if current_run is None:
            current_run = dict(box)
            current_mid = mid
            current_height = height
            continue

        same_line = abs(mid - current_mid) <= max(2.0, current_height * 0.35)
        horizontal_gap = float(box.get("x0") or 0.0) - float(current_run.get("x1") or 0.0)
        max_join_gap = max(current_height * 2.5, float(page_width or 0.0) * 0.08, 12.0)
        if not same_line or horizontal_gap > max_join_gap:
            flush_run()
            current_run = dict(box)
            current_mid = mid
            current_height = height
            continue

        current_run["text"] = _join_native_word_text(
            str(current_run.get("text") or ""),
            str(box.get("text") or ""),
        )
        current_run["x0"] = min(float(current_run.get("x0") or 0.0), float(box.get("x0") or 0.0))
        current_run["x1"] = max(float(current_run.get("x1") or 0.0), float(box.get("x1") or 0.0))
        current_run["top"] = min(float(current_run.get("top") or 0.0), top)
        current_run["bottom"] = max(float(current_run.get("bottom") or 0.0), bottom)
        if "chars" in current_run or "chars" in box:
            current_run["chars"] = list(current_run.get("chars") or []) + list(box.get("chars") or [])
        current_mid = (float(current_run.get("top") or 0.0) + float(current_run.get("bottom") or 0.0)) / 2.0
        current_height = max(current_height, height)

    flush_run()
    return collapsed


def _classify_page(page: dict[str, Any]) -> tuple[str, list[str], str, list[str]]:
    native_text_char_count = int(page.get("native_text_char_count") or 0)
    native_text_box_count = int(page.get("native_text_box_count") or 0)
    image_count = int(page.get("image_count") or 0)
    font_coverage_ratio = float(page.get("font_coverage_ratio") or 0.0)
    image_area_ratio = float(page.get("image_area_ratio") or 0.0)
    max_image_area_ratio = float(page.get("max_image_area_ratio") or 0.0)
    has_large_image = bool(page.get("has_large_image"))
    reasons: list[str] = []

    if native_text_char_count > 0 and image_count > 0:
        if (
            native_text_char_count >= 8
            and native_text_box_count >= 1
            and font_coverage_ratio >= 0.8
            and image_area_ratio <= 0.03
            and max_image_area_ratio <= 0.03
            and not has_large_image
        ):
            reasons.append("native_text_confident")
            reasons.append("inline_image_tolerated")
            return "digital_clean", reasons, "skip_all_ocr", []
        reasons.append("native_text_with_images")
        if image_area_ratio > 0.15:
            reasons.append("image_area_ratio_high")
        if has_large_image:
            reasons.append("large_image_page")
        reasons.append("complex_visual_page")
        return "digital_mixed", reasons, "complex_blocks_only", ["table"]
    if (
        native_text_char_count >= 8
        and native_text_box_count >= 1
        and font_coverage_ratio >= 0.8
        and image_area_ratio <= 0.15
        and not has_large_image
    ):
        reasons.append("native_text_confident")
        return "digital_clean", reasons, "skip_all_ocr", []
    if native_text_char_count > 0 and font_coverage_ratio >= 0.5:
        reasons.append("native_text_sparse")
        if image_area_ratio > 0.15:
            reasons.append("image_area_ratio_high")
        if has_large_image:
            reasons.append("large_image_page")
        return "digital_mixed", reasons, "full_page", []
    reasons.append("native_text_sparse")
    if image_area_ratio > 0.15:
        reasons.append("image_area_ratio_high")
    if has_large_image:
        reasons.append("large_image_page")
    return "scanned", reasons, "full_page", []


def _build_seed_layouts_for_mixed_page(page: dict[str, Any]) -> list[dict[str, Any]]:
    page_number = int(page.get("page_number") or 0)
    seed_layouts: list[dict[str, Any]] = []
    for index, image_box in enumerate(page.get("image_boxes") or []):
        x0 = float(image_box.get("x0") or 0.0)
        x1 = float(image_box.get("x1") or 0.0)
        top = float(image_box.get("top") or 0.0)
        bottom = float(image_box.get("bottom") or 0.0)
        if x1 <= x0 or bottom <= top:
            continue
        seed_layouts.append(
            {
                "type": "table",
                "score": 1.0,
                "x0": x0,
                "x1": x1,
                "top": top,
                "bottom": bottom,
                "page_number": page_number - 1,
                "source": "native_image_box",
                "layoutno": f"table-seed-{index}",
            }
        )
    return seed_layouts


def build_pdf_hybrid_plan(source, *, page_from: int = 0, page_to: int = 299) -> dict[str, Any]:
    page_features, native_boxes, native_meta = collect_pdf_page_features_and_native_boxes(
        source,
        page_from=page_from,
        page_to=page_to,
        dedupe_chars=False,
    )
    native_boxes_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for box in native_boxes:
        page_number = int(box.get("page_number") or 0)
        if page_number <= 0:
            continue
        native_boxes_by_page[page_number].append(dict(box))

    pages: list[dict[str, Any]] = []
    seed_layouts_by_page: dict[int, list[dict[str, Any]]] = {}
    for feature in page_features:
        page_number = int(feature.get("page_number") or 0)
        route, reasons, ocr_scope, complex_block_types = _classify_page(feature)
        if route == "digital_clean":
            native_boxes_by_page[page_number] = _collapse_native_page_boxes(
                native_boxes_by_page.get(page_number, []),
                page_width=float(feature.get("page_width") or 0.0),
            )
        elif ocr_scope == "complex_blocks_only":
            seed_layouts = _build_seed_layouts_for_mixed_page(feature)
            if seed_layouts:
                seed_layouts_by_page[page_number] = seed_layouts
        pages.append(
            {
                **feature,
                "route": route,
                "reasons": reasons,
                "ocr_scope": ocr_scope,
                "complex_block_types": complex_block_types,
                "native_box_count": len(native_boxes_by_page.get(page_number, [])),
                "seed_layout_count": len(seed_layouts_by_page.get(page_number, [])),
            }
        )

    route_summary = {
        "page_count": len(pages),
        "native_only_page_count": sum(1 for page in pages if page.get("ocr_scope") != "full_page"),
        "ocr_page_count": sum(1 for page in pages if page.get("route") != "digital_clean"),
        "complex_block_only_page_count": sum(1 for page in pages if page.get("ocr_scope") == "complex_blocks_only"),
        "full_page_ocr_page_count": sum(1 for page in pages if page.get("ocr_scope") == "full_page"),
        "digital_clean_pages": sum(1 for page in pages if page.get("route") == "digital_clean"),
        "digital_mixed_pages": sum(1 for page in pages if page.get("route") == "digital_mixed"),
        "scanned_pages": sum(1 for page in pages if page.get("route") == "scanned"),
    }
    ocr_page_numbers = [
        int(page.get("page_number") or 0)
        for page in pages
        if page.get("ocr_scope") == "full_page" and int(page.get("page_number") or 0) > 0
    ]
    complex_block_page_numbers = [
        int(page.get("page_number") or 0)
        for page in pages
        if page.get("ocr_scope") == "complex_blocks_only" and int(page.get("page_number") or 0) > 0
    ]
    collapsed_native_boxes = [
        dict(box)
        for page_number in sorted(native_boxes_by_page.keys())
        for box in native_boxes_by_page.get(page_number, [])
    ]

    return {
        "schema_version": HYBRID_PLAN_SCHEMA_VERSION,
        "page_from": max(0, int(page_from)),
        "page_to": max(0, int(page_to)),
        "page_count": int(native_meta.get("page_count") or len(pages)),
        "total_page_count": native_meta.get("total_page_count"),
        "pages": pages,
        "native_boxes": collapsed_native_boxes,
        "native_boxes_by_page": dict(native_boxes_by_page),
        "seed_layouts_by_page": seed_layouts_by_page,
        "ocr_page_numbers": ocr_page_numbers,
        "complex_block_page_numbers": complex_block_page_numbers,
        "route_summary": route_summary,
        "all_pages_digital_clean": bool(pages) and all(page.get("route") == "digital_clean" for page in pages),
    }


def build_pdf_text_layer_report_from_hybrid_plan(plan: dict[str, Any]) -> dict[str, Any]:
    pages = list(plan.get("pages") or [])
    page_count = len(pages)
    non_whitespace_char_count = sum(int(page.get("native_text_char_count") or 0) for page in pages)
    font_backed_char_count = sum(int(page.get("font_backed_char_count") or 0) for page in pages)
    text_page_count = sum(1 for page in pages if int(page.get("native_text_char_count") or 0) > 0)
    font_coverage_ratio = round(font_backed_char_count / max(non_whitespace_char_count, 1), 6)
    text_page_ratio = round(text_page_count / max(page_count, 1), 6)
    route_summary = plan.get("route_summary") or {}
    if int(route_summary.get("full_page_ocr_page_count") or 0) > 0:
        recommended_mode = "ocr"
    elif int(route_summary.get("complex_block_only_page_count") or 0) > 0:
        recommended_mode = "hybrid"
    else:
        recommended_mode = "native_text"
    return {
        "schema_version": "2026-06-11.pdf-text-layer.v1",
        "status": "derived_from_hybrid_plan",
        "recommended_mode": recommended_mode,
        "has_text_layer": text_page_count > 0,
        "page_from": max(0, int(plan.get("page_from") or 0)),
        "page_to": max(0, int(plan.get("page_to") or 0)),
        "pages_scanned": page_count,
        "total_page_count": plan.get("total_page_count"),
        "text_page_count": text_page_count,
        "non_whitespace_char_count": non_whitespace_char_count,
        "font_backed_char_count": font_backed_char_count,
        "font_coverage_ratio": font_coverage_ratio,
        "text_page_ratio": text_page_ratio,
        "thresholds": {
            "source": "hybrid_plan",
        },
        "problems": [],
    }

#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import asyncio
from common import logger
import math
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from copy import deepcopy
from io import BytesIO
from timeit import default_timer as timer
from typing import Any

import numpy as np
import pdfplumber
import xgboost as xgb
import cv2
import fitz
from PIL import Image
from pypdf import PdfReader as pdf2_read
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common.model_store import ensure_groups
from common.misc_utils import pip_install_torch
from deepdoc.vision import (
    OCR,
    AscendLayoutRecognizer,
    LayoutRecognizer,
    PPDocLayoutRecognizer,
    Recognizer,
    TableStructureRecognizer,
)
from deepdoc.vision.ocr import ensure_parallel_devices_configured
from common.nlp import tokenizer
from common import settings


from common.misc_utils import thread_pool_exec

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


PDF_TEXT_LAYER_SCHEMA_VERSION = "2026-06-08.pdf-text-layer.v1"
_SHARED_COMPONENTS_LOCK = threading.RLock()
_SHARED_PDF_PARSER_COMPONENTS: dict[tuple[str, str, str, str, str, int, str], dict[str, Any]] = {}
_PARALLEL_LIMITER_UNSET = object()


class _PageImagePlaceholder:
    def __init__(self, width: int, height: int):
        self.size = (max(1, int(width)), max(1, int(height)))

    def crop(self, _bbox):
        raise RuntimeError("Page image was intentionally skipped for this page")


def _pdf_parser_recognizer_domain(model_speciess: str | None = None) -> str:
    model_speciess = (model_speciess or "").strip()
    return f"layout.{model_speciess}" if model_speciess else "layout"


def _pdf_parser_layout_recognizer_type() -> str:
    layout_recognizer_type = os.getenv("LAYOUT_RECOGNIZER_TYPE", "onnx").lower()
    if layout_recognizer_type not in ["onnx", "ascend"]:
        raise RuntimeError("Unsupported layout recognizer type.")
    return layout_recognizer_type


def _pdf_parser_layout_engine() -> str:
    layout_engine = (os.getenv("DEEPDOC_LAYOUT_ENGINE") or "legacy").strip().lower()
    if layout_engine in {"", "legacy", "yolo", "yolov10"}:
        return "legacy"
    if layout_engine in {"ppdoclayout", "pp_doclayout", "pp-doclayout"}:
        return "ppdoclayout"
    raise RuntimeError(f"Unsupported DEEPDOC_LAYOUT_ENGINE: {layout_engine}")


def _pdf_parser_ocr_version() -> str:
    version = (os.getenv("DEEPDOC_OCR_VERSION") or "v4").strip().lower()
    if version in {"", "legacy", "v4", "ppocrv4", "pp-ocrv4"}:
        return "v4"
    if version in {"v5", "ppocrv5", "pp-ocrv5"}:
        return "v5"
    raise RuntimeError(f"Unsupported DEEPDOC_OCR_VERSION: {version}")


def _pdf_parser_reading_order_strategy() -> str:
    strategy = (os.getenv("DEEPDOC_READING_ORDER_STRATEGY") or "legacy").strip().lower()
    if strategy in {"", "legacy", "default", "y_first"}:
        return "legacy"
    if strategy in {"rules", "rule", "enhanced", "structured"}:
        return "rules"
    raise RuntimeError(f"Unsupported DEEPDOC_READING_ORDER_STRATEGY: {strategy}")


def _pdf_parser_component_cache_key(model_dir: str, recognizer_domain: str, layout_recognizer_type: str, layout_engine: str, ocr_version: str) -> tuple[str, str, str, str, str, int, str]:
    table_recognizer_type = os.getenv("TABLE_STRUCTURE_RECOGNIZER_TYPE", "onnx").lower()
    return (
        os.path.abspath(model_dir),
        recognizer_domain,
        layout_recognizer_type,
        layout_engine,
        ocr_version,
        int(settings.PARALLEL_DEVICES),
        table_recognizer_type,
    )


def _load_updown_concat_model(model_dir: str):
    model = xgb.Booster()
    try:
        pip_install_torch()
        import torch

        if torch.cuda.is_available():
            model.set_param({"device": "cuda"})
    except Exception:
        logger.info("No torch found.")
    model.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))
    return model


def _parallel_worker_count() -> int:
    try:
        return max(1, int(getattr(settings, "PARALLEL_DEVICES", 0) or 0))
    except Exception:
        return 1


def _build_layout_recognizer(
    *,
    recognizer_domain: str,
    layout_recognizer_type: str,
    layout_engine: str,
    device_id: int | None = None,
):
    if layout_engine == "ppdoclayout":
        if layout_recognizer_type != "onnx":
            raise RuntimeError("DEEPDOC_LAYOUT_ENGINE=ppdoclayout requires LAYOUT_RECOGNIZER_TYPE=onnx")
        logger.debug("Using PPDocLayoutRecognizer")
        return PPDocLayoutRecognizer(recognizer_domain, device_id=device_id)
    if layout_recognizer_type == "ascend":
        logger.debug("Using Ascend LayoutRecognizer")
        return AscendLayoutRecognizer(recognizer_domain)
    logger.debug("Using Onnx LayoutRecognizer")
    return LayoutRecognizer(recognizer_domain, device_id=device_id)


def _build_layout_recognizer_pool(
    *,
    recognizer_domain: str,
    layout_recognizer_type: str,
    layout_engine: str,
) -> list[Any]:
    worker_count = _parallel_worker_count()
    if worker_count <= 1 or layout_recognizer_type != "onnx":
        return [_build_layout_recognizer(
            recognizer_domain=recognizer_domain,
            layout_recognizer_type=layout_recognizer_type,
            layout_engine=layout_engine,
            device_id=None,
        )]
    return [
        _build_layout_recognizer(
            recognizer_domain=recognizer_domain,
            layout_recognizer_type=layout_recognizer_type,
            layout_engine=layout_engine,
            device_id=device_id,
        )
        for device_id in range(worker_count)
    ]


def _build_table_recognizer_pool() -> list[Any]:
    worker_count = _parallel_worker_count()
    table_recognizer_type = os.getenv("TABLE_STRUCTURE_RECOGNIZER_TYPE", "onnx").lower()
    if worker_count <= 1 or table_recognizer_type != "onnx":
        return [TableStructureRecognizer()]
    return [TableStructureRecognizer(device_id=device_id) for device_id in range(worker_count)]


def _build_pdf_parser_components(
    *,
    model_dir: str,
    recognizer_domain: str,
    layout_recognizer_type: str,
    layout_engine: str,
    ocr_version: str,
) -> dict[str, Any]:
    layouters = _build_layout_recognizer_pool(
        recognizer_domain=recognizer_domain,
        layout_recognizer_type=layout_recognizer_type,
        layout_engine=layout_engine,
    )
    tbl_dets = _build_table_recognizer_pool()
    return {
        "ocr": OCR(),
        "layouter": layouters[0],
        "layouters": layouters,
        "tbl_det": tbl_dets[0],
        "tbl_dets": tbl_dets,
        "updown_cnt_mdl": _load_updown_concat_model(model_dir),
        "layout_recognizer_type": layout_recognizer_type,
        "layout_engine": layout_engine,
        "ocr_version": ocr_version,
        "recognizer_domain": recognizer_domain,
        "model_dir": os.path.abspath(model_dir),
    }


def get_shared_pdf_parser_components(
    *,
    model_speciess: str | None = None,
) -> dict[str, Any]:
    model_dir = ensure_groups("core")
    layout_recognizer_type = _pdf_parser_layout_recognizer_type()
    layout_engine = _pdf_parser_layout_engine()
    ocr_version = _pdf_parser_ocr_version()
    recognizer_domain = _pdf_parser_recognizer_domain(model_speciess)
    cache_key = _pdf_parser_component_cache_key(model_dir, recognizer_domain, layout_recognizer_type, layout_engine, ocr_version)
    with _SHARED_COMPONENTS_LOCK:
        cached = _SHARED_PDF_PARSER_COMPONENTS.get(cache_key)
        if cached is None:
            cached = _build_pdf_parser_components(
                model_dir=model_dir,
                recognizer_domain=recognizer_domain,
                layout_recognizer_type=layout_recognizer_type,
                layout_engine=layout_engine,
                ocr_version=ocr_version,
            )
            cached["ref_count"] = 0
            _SHARED_PDF_PARSER_COMPONENTS[cache_key] = cached
        cached["ref_count"] = int(cached.get("ref_count") or 0) + 1
        return cached


def shared_pdf_parser_component_state() -> dict[str, Any]:
    with _SHARED_COMPONENTS_LOCK:
        components = [
            {
                "cache_key": list(cache_key),
                "model_dir": str(component.get("model_dir") or ""),
                "recognizer_domain": str(component.get("recognizer_domain") or ""),
                "layout_recognizer_type": str(component.get("layout_recognizer_type") or ""),
                "layout_engine": str(component.get("layout_engine") or "legacy"),
                "ocr_version": str(component.get("ocr_version") or "v4"),
                "ref_count": int(component.get("ref_count") or 0),
            }
            for cache_key, component in _SHARED_PDF_PARSER_COMPONENTS.items()
        ]
    return {
        "schema_version": "2026-06-08.pdf-parser-components.v1",
        "cached_component_count": len(components),
        "components": components,
    }


def clear_shared_pdf_parser_components() -> None:
    with _SHARED_COMPONENTS_LOCK:
        _SHARED_PDF_PARSER_COMPONENTS.clear()


def _open_pdfplumber_source(source):
    if isinstance(source, (bytes, bytearray)):
        return pdfplumber.open(BytesIO(source))
    return pdfplumber.open(source)


def _open_fitz_source(source):
    if isinstance(source, (bytes, bytearray)):
        return fitz.open(stream=bytes(source), filetype="pdf")
    return fitz.open(source)


def _dedupe_pdf_page(page):
    try:
        return page.dedupe_chars()
    except Exception:
        return page


def _extract_pdf_page_chars(page) -> list[dict[str, Any]]:
    return getattr(page, "chars", []) or []


def _compact_pdf_char(char: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(char.get("text") or ""),
        "page_number": int(char.get("page_number") or 0),
        "x0": float(char.get("x0") or 0.0),
        "x1": float(char.get("x1") or 0.0),
        "top": float(char.get("top") or 0.0),
        "bottom": float(char.get("bottom") or 0.0),
    }


def _chars_within_pdf_box(
    chars: list[dict[str, Any]],
    *,
    x0: float,
    x1: float,
    top: float,
    bottom: float,
    tolerance: float = 0.5,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for char in chars:
        cx = (float(char.get("x0") or 0.0) + float(char.get("x1") or 0.0)) / 2.0
        cy = (float(char.get("top") or 0.0) + float(char.get("bottom") or 0.0)) / 2.0
        if x0 - tolerance <= cx <= x1 + tolerance and top - tolerance <= cy <= bottom + tolerance:
            matched.append(char)
    return matched


def _extract_pdf_page_words(page, *, return_chars: bool = False) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    try:
        words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3,
            keep_blank_chars=False,
            return_chars=return_chars,
        )
    except TypeError:
        try:
            words = page.extract_words(
                x_tolerance=1,
                y_tolerance=3,
                return_chars=return_chars,
            )
        except TypeError:
            try:
                words = page.extract_words(x_tolerance=1, y_tolerance=3, keep_blank_chars=False)
            except TypeError:
                words = page.extract_words(x_tolerance=1, y_tolerance=3)
        except Exception:
            return []
    except Exception:
        return []

    if not return_chars:
        return words

    page_chars = _extract_pdf_page_chars(page)
    compacted_words: list[dict[str, Any]] = []
    for word in words:
        word_chars = word.get("chars") or _chars_within_pdf_box(
            page_chars,
            x0=float(word.get("x0") or 0.0),
            x1=float(word.get("x1") or 0.0),
            top=float(word.get("top") or 0.0),
            bottom=float(word.get("bottom") or 0.0),
        )
        compacted_word = dict(word)
        compacted_word["chars"] = [_compact_pdf_char(char) for char in word_chars if str(char.get("text") or "")]
        compacted_words.append(compacted_word)
    return compacted_words


def _extract_pdf_page_text_lines(page) -> list[dict[str, Any]]:
    try:
        lines = page.extract_text_lines(strip=True, return_chars=False)
    except TypeError:
        lines = page.extract_text_lines(strip=True)
    except Exception:
        lines = []
    if lines:
        return [line for line in lines if str(line.get("text") or "").strip()]

    words = _extract_pdf_page_words(page, return_chars=False)
    return _group_pdf_words_into_text_lines(words)


def _group_pdf_words_into_text_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    line_groups: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda item: (float(item.get("top") or 0.0), float(item.get("x0") or 0.0))):
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        mid = (float(word.get("top") or 0.0) + float(word.get("bottom") or 0.0)) / 2.0
        if line_groups and abs(mid - line_groups[-1]["mid"]) <= max(2.0, line_groups[-1]["height"] * 0.35):
            line_groups[-1]["texts"].append(text)
            line_groups[-1]["height"] = max(line_groups[-1]["height"], float(word.get("bottom") or 0.0) - float(word.get("top") or 0.0))
            continue
        line_groups.append(
            {
                "mid": mid,
                "height": float(word.get("bottom") or 0.0) - float(word.get("top") or 0.0),
                "texts": [text],
            }
        )
    return [{"text": " ".join(group["texts"]).strip()} for group in line_groups if group["texts"]]


def _dedupe_extracted_pdf_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float, float, float]] = set()
    for word in words:
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        key = (
            text,
            round(float(word.get("x0") or 0.0), 2),
            round(float(word.get("x1") or 0.0), 2),
            round(float(word.get("top") or 0.0), 2),
            round(float(word.get("bottom") or 0.0), 2),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(word)
    return deduped


def _looks_like_overlaid_word_text(text: str) -> bool:
    text = str(text or "").strip()
    if len(text) < 4 or len(text) % 2 != 0:
        return False
    duplicated_pairs = sum(1 for index in range(0, len(text), 2) if text[index] == text[index + 1])
    return duplicated_pairs == (len(text) // 2)


def _clip_pdf_box_area(box: dict[str, Any], *, page_width: float, page_height: float) -> float:
    x0 = max(0.0, min(float(box.get("x0") or 0.0), page_width))
    x1 = max(0.0, min(float(box.get("x1") or 0.0), page_width))
    top = max(0.0, min(float(box.get("top") or 0.0), page_height))
    bottom = max(0.0, min(float(box.get("bottom") or 0.0), page_height))
    return max(0.0, x1 - x0) * max(0.0, bottom - top)


def _normalize_pdf_image_boxes(
    images: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    page_width: float,
    page_height: float,
) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for image in images or []:
        box = {
            "x0": max(0.0, min(float(image.get("x0") or 0.0), page_width)),
            "x1": max(0.0, min(float(image.get("x1") or 0.0), page_width)),
            "top": max(0.0, min(float(image.get("top") or 0.0), page_height)),
            "bottom": max(0.0, min(float(image.get("bottom") or 0.0), page_height)),
        }
        if box["x1"] <= box["x0"] or box["bottom"] <= box["top"]:
            continue
        normalized.append(box)
    return normalized


def detect_pdf_text_layer(
    source,
    *,
    page_from: int = 0,
    page_to: int = 299,
    min_non_whitespace_chars: int = 80,
    min_font_coverage_ratio: float = 0.8,
    min_text_page_ratio: float = 0.5,
) -> dict[str, Any]:
    """Inspect whether a PDF has a reliable native text layer.

    The detector is intentionally conservative: scanned/image-only PDFs should
    stay on the OCR/Layout path, while born-digital PDFs with sufficient chars
    and font metadata can use the faster native text path.
    """
    report: dict[str, Any] = {
        "schema_version": PDF_TEXT_LAYER_SCHEMA_VERSION,
        "status": "ok",
        "recommended_mode": "ocr",
        "has_text_layer": False,
        "page_from": max(0, int(page_from)),
        "page_to": max(0, int(page_to)),
        "pages_scanned": 0,
        "total_page_count": None,
        "text_page_count": 0,
        "non_whitespace_char_count": 0,
        "font_backed_char_count": 0,
        "font_coverage_ratio": 0.0,
        "text_page_ratio": 0.0,
        "thresholds": {
            "min_non_whitespace_chars": int(min_non_whitespace_chars),
            "min_font_coverage_ratio": float(min_font_coverage_ratio),
            "min_text_page_ratio": float(min_text_page_ratio),
        },
        "problems": [],
    }
    try:
        with sys.modules[LOCK_KEY_pdfplumber]:
            with _open_pdfplumber_source(source) as pdf:
                total_pages = len(pdf.pages)
                start_page = max(0, int(page_from))
                end_page = min(max(start_page, int(page_to)), total_pages)
                pages = pdf.pages[start_page:end_page]
                report["total_page_count"] = total_pages
                report["pages_scanned"] = len(pages)
                for page in pages:
                    deduped_page = _dedupe_pdf_page(page)
                    chars = _extract_pdf_page_chars(deduped_page)
                    text_chars = [c for c in chars if str(c.get("text") or "").strip()]
                    font_chars = [
                        c
                        for c in text_chars
                        if str(c.get("fontname") or c.get("font") or "").strip()
                        and float(c.get("size") or 0) > 0
                    ]
                    if text_chars:
                        report["text_page_count"] += 1
                    report["non_whitespace_char_count"] += len(text_chars)
                    report["font_backed_char_count"] += len(font_chars)
    except Exception as exc:
        logger.exception("PDF text layer detection failed")
        report["status"] = "error"
        report["problems"] = [str(exc)]
        return report

    char_count = int(report["non_whitespace_char_count"] or 0)
    pages_scanned = int(report["pages_scanned"] or 0)
    font_backed = int(report["font_backed_char_count"] or 0)
    font_coverage = font_backed / max(char_count, 1)
    text_page_ratio = int(report["text_page_count"] or 0) / max(pages_scanned, 1)
    has_text_layer = (
        char_count >= int(min_non_whitespace_chars)
        and font_coverage >= float(min_font_coverage_ratio)
        and text_page_ratio >= float(min_text_page_ratio)
    )
    report["font_coverage_ratio"] = round(font_coverage, 6)
    report["text_page_ratio"] = round(text_page_ratio, 6)
    report["has_text_layer"] = has_text_layer
    report["recommended_mode"] = "native_text" if has_text_layer else "ocr"
    return report


def inspect_pdf_pages(source, *, page_from: int = 0, page_to: int = 299) -> list[dict[str, Any]]:
    pages_report: list[dict[str, Any]] = []
    with sys.modules[LOCK_KEY_pdfplumber]:
        with _open_pdfplumber_source(source) as pdf:
            total_page_count = len(pdf.pages)
            start_page = max(0, int(page_from))
            end_page = min(max(start_page, int(page_to)), total_page_count)
            for local_index, page in enumerate(pdf.pages[start_page:end_page]):
                page_number = start_page + local_index + 1
                page_width = float(getattr(page, "width", 0) or 0.0)
                page_height = float(getattr(page, "height", 0) or 0.0)
                page_area = max(page_width * page_height, 1.0)
                deduped_page = _dedupe_pdf_page(page)
                chars = _extract_pdf_page_chars(deduped_page)
                words = _extract_pdf_page_words(deduped_page, return_chars=False)
                text_lines = _extract_pdf_page_text_lines(deduped_page)
                text_chars = [c for c in chars if str(c.get("text") or "").strip()]
                font_chars = [
                    c
                    for c in text_chars
                    if str(c.get("fontname") or c.get("font") or "").strip()
                    and float(c.get("size") or 0) > 0
                ]
                images = _normalize_pdf_image_boxes(
                    list(getattr(page, "images", []) or []),
                    page_width=page_width,
                    page_height=page_height,
                )
                image_areas = [
                    _clip_pdf_box_area(image, page_width=page_width, page_height=page_height)
                    for image in images
                ]
                total_image_area = sum(image_areas)
                max_image_area = max(image_areas, default=0.0)
                pages_report.append(
                    {
                        "page_number": page_number,
                        "page_width": page_width,
                        "page_height": page_height,
                        "native_text_char_count": len(text_chars),
                        "font_backed_char_count": len(font_chars),
                        "font_coverage_ratio": round(len(font_chars) / max(len(text_chars), 1), 6),
                        "native_text_box_count": len(words),
                        "text_block_count": len(text_lines),
                        "image_count": len(images),
                        "image_area_ratio": round(min(total_image_area / page_area, 1.0), 6),
                        "max_image_area_ratio": round(min(max_image_area / page_area, 1.0), 6),
                        "has_large_image": (max_image_area / page_area) >= 0.85,
                        "image_boxes": images,
                    }
                )
    return pages_report


def collect_pdf_page_features_and_native_boxes(
    source,
    *,
    page_from: int = 0,
    page_to: int = 299,
    include_word_chars: bool = False,
    dedupe_chars: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pages_report: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []
    page_count = 0
    total_page_count = None
    if not dedupe_chars and not include_word_chars:
        doc = _open_fitz_source(source)
        try:
            total_page_count = len(doc)
            start_page = max(0, int(page_from))
            end_page = min(max(start_page, int(page_to)), total_page_count)
            page_count = max(0, end_page - start_page)
            for page_index in range(start_page, end_page):
                page = doc.load_page(page_index)
                page_number = page_index + 1
                page_width = float(page.rect.width or 0.0)
                page_height = float(page.rect.height or 0.0)
                page_area = max(page_width * page_height, 1.0)
                word_boxes = _dedupe_extracted_pdf_words(
                    [
                        {
                            "text": str(word[4] or "").strip(),
                            "x0": float(word[0] or 0.0),
                            "x1": float(word[2] or 0.0),
                            "top": float(word[1] or 0.0),
                            "bottom": float(word[3] or 0.0),
                        }
                        for word in (page.get_text("words") or [])
                        if len(word) >= 5 and str(word[4] or "").strip()
                    ]
                )
                raw_dict = page.get_text("dict") or {}
                text_line_count = 0
                image_count = 0
                image_boxes: list[dict[str, float]] = []
                total_image_area = 0.0
                max_image_area = 0.0
                native_text_char_count = 0
                font_backed_char_count = 0

                for block in raw_dict.get("blocks", []) or []:
                    if int(block.get("type") or 0) == 1:
                        image_count += 1
                        bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
                        image_box = {
                            "x0": float(bbox[0] or 0.0),
                            "top": float(bbox[1] or 0.0),
                            "x1": float(bbox[2] or 0.0),
                            "bottom": float(bbox[3] or 0.0),
                        }
                        image_area = _clip_pdf_box_area(image_box, page_width=page_width, page_height=page_height)
                        total_image_area += image_area
                        max_image_area = max(max_image_area, image_area)
                        image_boxes.append(image_box)
                        continue
                    for line in block.get("lines", []) or []:
                        text_line_count += 1
                        for span in line.get("spans", []) or []:
                            span_text = str(span.get("text") or "")
                            span_char_count = len(span_text.strip()) if span_text.strip() else 0
                            native_text_char_count += span_char_count
                            if str(span.get("font") or "").strip() and float(span.get("size") or 0) > 0:
                                font_backed_char_count += span_char_count

                pages_report.append(
                    {
                        "page_number": page_number,
                        "page_width": page_width,
                        "page_height": page_height,
                        "native_text_char_count": native_text_char_count,
                        "font_backed_char_count": font_backed_char_count,
                        "font_coverage_ratio": round(font_backed_char_count / max(native_text_char_count, 1), 6),
                        "native_text_box_count": len(word_boxes),
                        "text_block_count": text_line_count,
                        "image_count": image_count,
                        "image_area_ratio": round(min(total_image_area / page_area, 1.0), 6),
                        "max_image_area_ratio": round(min(max_image_area / page_area, 1.0), 6),
                        "has_large_image": (max_image_area / page_area) >= 0.85,
                        "image_boxes": image_boxes,
                    }
                )
                for word in word_boxes:
                    boxes.append(
                        {
                            "text": str(word.get("text") or "").strip(),
                            "page_number": page_number,
                            "layout_type": "text",
                            "x0": float(word.get("x0") or 0.0),
                            "x1": float(word.get("x1") or 0.0),
                            "top": float(word.get("top") or 0.0),
                            "bottom": float(word.get("bottom") or 0.0),
                        }
                    )
        finally:
            doc.close()
        return pages_report, boxes, {"page_count": page_count, "total_page_count": total_page_count}

    with sys.modules[LOCK_KEY_pdfplumber]:
        with _open_pdfplumber_source(source) as pdf:
            total_page_count = len(pdf.pages)
            start_page = max(0, int(page_from))
            end_page = min(max(start_page, int(page_to)), total_page_count)
            pages = pdf.pages[start_page:end_page]
            page_count = len(pages)
            for local_index, page in enumerate(pages):
                page_number = start_page + local_index + 1
                page_width = float(getattr(page, "width", 0) or 0.0)
                page_height = float(getattr(page, "height", 0) or 0.0)
                page_area = max(page_width * page_height, 1.0)
                page_for_words = _dedupe_pdf_page(page) if dedupe_chars else page
                prefetched_words = None
                if not dedupe_chars:
                    raw_words = _extract_pdf_page_words(page, return_chars=include_word_chars)
                    if any(_looks_like_overlaid_word_text(word.get("text") or "") for word in raw_words):
                        page_for_words = _dedupe_pdf_page(page)
                    else:
                        page_for_words = page
                        prefetched_words = raw_words
                chars = _extract_pdf_page_chars(page_for_words)
                words = _dedupe_extracted_pdf_words(
                    prefetched_words
                    if prefetched_words is not None
                    else _extract_pdf_page_words(page_for_words, return_chars=include_word_chars)
                )
                text_lines = _group_pdf_words_into_text_lines(words)
                text_chars = [c for c in chars if str(c.get("text") or "").strip()]
                font_chars = [
                    c
                    for c in text_chars
                    if str(c.get("fontname") or c.get("font") or "").strip()
                    and float(c.get("size") or 0) > 0
                ]
                images = list(getattr(page, "images", []) or [])
                image_areas = [
                    _clip_pdf_box_area(image, page_width=page_width, page_height=page_height)
                    for image in images
                ]
                total_image_area = sum(image_areas)
                max_image_area = max(image_areas, default=0.0)
                pages_report.append(
                    {
                        "page_number": page_number,
                        "page_width": page_width,
                        "page_height": page_height,
                        "native_text_char_count": len(text_chars),
                        "font_backed_char_count": len(font_chars),
                        "font_coverage_ratio": round(len(font_chars) / max(len(text_chars), 1), 6),
                        "native_text_box_count": len(words),
                        "text_block_count": len(text_lines),
                        "image_count": len(images),
                        "image_area_ratio": round(min(total_image_area / page_area, 1.0), 6),
                        "max_image_area_ratio": round(min(max_image_area / page_area, 1.0), 6),
                        "has_large_image": (max_image_area / page_area) >= 0.85,
                    }
                )
                for word in words:
                    text = str(word.get("text") or "").strip()
                    if not text:
                        continue
                    box = {
                        "text": text,
                        "page_number": page_number,
                        "layout_type": "text",
                        "x0": float(word.get("x0") or 0.0),
                        "x1": float(word.get("x1") or 0.0),
                        "top": float(word.get("top") or 0.0),
                        "bottom": float(word.get("bottom") or 0.0),
                    }
                    if include_word_chars:
                        box["chars"] = list(word.get("chars") or [])
                    boxes.append(box)
    return pages_report, boxes, {"page_count": page_count, "total_page_count": total_page_count}


def extract_native_pdf_text(
    source,
    *,
    page_from: int = 0,
    page_to: int = 299,
    preserve_geometry: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    page_count = 0
    total_page_count = None
    with sys.modules[LOCK_KEY_pdfplumber]:
        with _open_pdfplumber_source(source) as pdf:
            total_page_count = len(pdf.pages)
            start_page = max(0, int(page_from))
            end_page = min(max(start_page, int(page_to)), total_page_count)
            pages = pdf.pages[start_page:end_page]
            page_count = len(pages)
            for local_index, page in enumerate(pages):
                page_number = start_page + local_index + 1
                if preserve_geometry:
                    deduped_page = _dedupe_pdf_page(page)
                    words = _extract_pdf_page_words(deduped_page, return_chars=True)
                    for word in words:
                        text = str(word.get("text") or "").strip()
                        if not text:
                            continue
                        boxes.append(
                            {
                                "text": text,
                                "page_number": page_number,
                                "layout_type": "text",
                                "x0": float(word.get("x0") or 0.0),
                                "x1": float(word.get("x1") or 0.0),
                                "top": float(word.get("top") or 0.0),
                                "bottom": float(word.get("bottom") or 0.0),
                                "chars": list(word.get("chars") or []),
                            }
                        )
                    continue
                try:
                    text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                except Exception:
                    logger.exception("Native PDF text extraction failed on page=%s", page_number)
                    text = ""
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if not lines:
                    continue
                top_step = max(float(getattr(page, "height", 0) or 0) / max(len(lines), 1), 1.0)
                for line_index, line in enumerate(lines):
                    top = line_index * top_step
                    boxes.append(
                        {
                            "text": line,
                            "page_number": page_number,
                            "layout_type": "text",
                            "x0": 0.0,
                            "x1": float(getattr(page, "width", 0) or 0),
                            "top": top,
                            "bottom": min(top + top_step, float(getattr(page, "height", 0) or top + top_step)),
                        }
                    )
    return boxes, {"page_count": page_count, "total_page_count": total_page_count}


class DeepDocPdfParser:
    def __init__(self, **kwargs):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """

        ensure_parallel_devices_configured()
        self.parallel_limiter = None
        if settings.PARALLEL_DEVICES > 1:
            self.parallel_limiter = int(settings.PARALLEL_DEVICES)

        shared_components = get_shared_pdf_parser_components(
            model_speciess=getattr(self, "model_speciess", None),
        )
        self.ocr = shared_components["ocr"]
        self.layouter = shared_components["layouter"]
        self.layouters = list(shared_components.get("layouters") or [self.layouter])
        self.tbl_det = shared_components["tbl_det"]
        self.tbl_dets = list(shared_components.get("tbl_dets") or [self.tbl_det])
        self.updown_cnt_mdl = shared_components["updown_cnt_mdl"]
        # 表格识别引擎开关：tatr(默认，几何拼装) / rapidtable(SLANet-plus ONNX)。
        # 默认 tatr，行为与改造前完全一致。
        self.table_engine = os.getenv("DEEPDOC_TABLE_ENGINE", "tatr").strip().lower()
        self._rapid_table = None  # 惰性初始化

        self.page_from = 0
        self.column_num = 1
        self._last_cross_page_table_merge_count = 0
        self._last_cross_page_table_merge_groups = []
        self._last_selective_ocr_block_count = 0
        self._last_complex_block_counts: dict[str, int] = {}
        self.complex_block_only_pages: set[int] = set()
        self.hybrid_clean_pages: set[int] = set()

    def _layout_worker_pool(self) -> list[Any]:
        workers = list(getattr(self, "layouters", None) or [])
        if not workers and getattr(self, "layouter", None) is not None:
            workers = [self.layouter]
        return workers or []

    def _table_worker_pool(self) -> list[Any]:
        workers = list(getattr(self, "tbl_dets", None) or [])
        if not workers and getattr(self, "tbl_det", None) is not None:
            workers = [self.tbl_det]
        return workers or []

    def _dispatch_layout_recognition(self, images, page_boxes, zoomin, *, drop=True):
        workers = self._layout_worker_pool()
        if len(workers) <= 1 or len(images) <= 1:
            return self.layouter(images, page_boxes, zoomin, drop=drop)

        jobs: list[dict[str, Any]] = []
        for index, (image, boxes) in enumerate(zip(images, page_boxes)):
            worker = workers[index % len(workers)]
            if len(jobs) <= index % len(workers):
                jobs.append({"worker": worker, "positions": [], "images": [], "boxes": []})
            job = jobs[index % len(workers)]
            job["positions"].append(index)
            job["images"].append(image)
            job["boxes"].append(boxes)

        def run_layout_job(job):
            laid_out_boxes, page_layout = job["worker"](job["images"], job["boxes"], zoomin, drop=drop)
            return job["positions"], laid_out_boxes, page_layout

        ordered_layout = [[] for _ in images]
        laid_out_boxes = []
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            for positions, chunk_boxes, chunk_layout in executor.map(run_layout_job, jobs):
                laid_out_boxes.extend(chunk_boxes)
                for position, layouts in zip(positions, chunk_layout):
                    ordered_layout[position] = layouts
        laid_out_boxes.sort(
            key=lambda box: (
                int(box.get("page_number") or 0),
                float(box.get("top") or 0.0),
                float(box.get("x0") or 0.0),
            )
        )
        return laid_out_boxes, ordered_layout

    def _dispatch_table_structure_recognition(self, images):
        workers = self._table_worker_pool()
        if len(workers) <= 1 or len(images) <= 1:
            return self.tbl_det(images)

        jobs: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            worker_index = index % len(workers)
            if len(jobs) <= worker_index:
                jobs.append({"worker": workers[worker_index], "positions": [], "images": []})
            job = jobs[worker_index]
            job["positions"].append(index)
            job["images"].append(image)

        def run_table_job(job):
            return job["positions"], job["worker"](job["images"])

        ordered_results: list[Any] = [None] * len(images)
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            for positions, chunk_results in executor.map(run_table_job, jobs):
                for position, result in zip(positions, chunk_results):
                    ordered_results[position] = result
        return [result if result is not None else [] for result in ordered_results]

    def __char_width(self, c):
        return (c["x1"] - c["x0"]) // max(len(c["text"]), 1)

    def __height(self, c):
        return c["bottom"] - c["top"]

    def _x_dis(self, a, b):
        return min(
            abs(a["x1"] - b["x0"]),
            abs(a["x0"] - b["x1"]),
            abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2,
        )

    def _y_dis(self, a, b):
        return (b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    def _match_proj(self, b):
        proj_patt = [
            r"第[零一二三四五六七八九十百]+章",
            r"第[零一二三四五六七八九十百]+[条节]",
            r"[零一二三四五六七八九十百]+[、是 　]",
            r"[\(（][零一二三四五六七八九十百]+[）\)]",
            r"[\(（][0-9]+[）\)]",
            r"[0-9]+(、|\.[　 ]|）|\.[^0-9./a-zA-Z_%><-]{4,})",
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[⚫•➢①② ]",
        ]
        return any([re.match(p, b["text"]) for p in proj_patt])

    def _updown_concat_features(self, up, down):
        w = max(self.__char_width(up), self.__char_width(down))
        h = max(self.__height(up), self.__height(down))
        y_dis = self._y_dis(up, down)
        LEN = 6
        tks_down = tokenizer.tokenize(down["text"][:LEN]).split()
        tks_up = tokenizer.tokenize(up["text"][-LEN:]).split()
        tks_all = (
            up["text"][-LEN:].strip()
            + (
                " "
                if re.match(r"[a-zA-Z0-9]+", up["text"][-1] + down["text"][0])
                else ""
            )
            + down["text"][:LEN].strip()
        )
        tks_all = tokenizer.tokenize(tks_all).split()
        fea = [
            up.get("R", -1) == down.get("R", -1),
            y_dis / h,
            down["page_number"] - up["page_number"],
            up["layout_type"] == down["layout_type"],
            up["layout_type"] == "text",
            down["layout_type"] == "text",
            up["layout_type"] == "table",
            down["layout_type"] == "table",
            True if re.search(r"([。？！；!?;+)）]|[a-z]\.)$", up["text"]) else False,
            True if re.search(r"[，：‘“、0-9（+-]$", up["text"]) else False,
            (
                True
                if re.search(r"(^.?[/,?;:\]，。；：’”？！》】）-])", down["text"])
                else False
            ),
            True if re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            (
                True
                if re.search(r"[\(（][^\)）]+$", up["text"])
                and re.search(r"[\)）]", down["text"])
                else False
            ),
            self._match_proj(down),
            True if re.match(r"[A-Z]", down["text"]) else False,
            True if re.match(r"[A-Z]", up["text"][-1]) else False,
            True if re.match(r"[a-z0-9]", up["text"][-1]) else False,
            True if re.match(r"[0-9.%,-]+$", down["text"]) else False,
            (
                up["text"].strip()[-2:] == down["text"].strip()[-2:]
                if len(up["text"].strip()) > 1 and len(down["text"].strip()) > 1
                else False
            ),
            up["x0"] > down["x1"],
            abs(self.__height(up) - self.__height(down))
            / min(self.__height(up), self.__height(down)),
            self._x_dis(up, down) / max(w, 0.000001),
            (len(up["text"]) - len(down["text"]))
            / max(len(up["text"]), len(down["text"])),
            len(tks_all) - len(tks_up) - len(tks_down),
            len(tks_down) - len(tks_up),
            tks_down[-1] == tks_up[-1] if tks_down and tks_up else False,
            max(down["in_row"], up["in_row"]),
            abs(down["in_row"] - up["in_row"]),
            len(tks_down) == 1 and tokenizer.tag(tks_down[0]).find("n") >= 0,
            len(tks_up) == 1 and tokenizer.tag(tks_up[0]).find("n") >= 0,
        ]
        return fea

    @staticmethod
    def sort_X_by_page(arr, threshold):
        # sort using y1 first and then x1
        arr = sorted(arr, key=lambda r: (r["page_number"], r["x0"], r["top"]))
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if (
                    abs(arr[j + 1]["x0"] - arr[j]["x0"]) < threshold
                    and arr[j + 1]["top"] < arr[j]["top"]
                    and arr[j + 1]["page_number"] == arr[j]["page_number"]
                ):
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

    def _has_color(self, o):
        if o.get("ncs", "") == "DeviceGray":
            if (
                o["stroking_color"]
                and o["stroking_color"][0] == 1
                and o["non_stroking_color"]
                and o["non_stroking_color"][0] == 1
            ):
                if re.match(r"[a-zT_\[\]\(\)-]+", o.get("text", "")):
                    return False
        return True

    def _evaluate_table_orientation(self, table_img, sample_ratio=0.3):
        """
        Evaluate the best rotation orientation for a table image.

        Tests 4 rotation angles (0°, 90°, 180°, 270°) and uses OCR
        confidence scores to determine the best orientation.

        Args:
            table_img: PIL Image object of the table region
            sample_ratio: Sampling ratio for quick evaluation

        Returns:
            tuple: (best_angle, best_img, confidence_scores)
                - best_angle: Best rotation angle (0, 90, 180, 270)
                - best_img: Image rotated to best orientation
                - confidence_scores: Dict of scores for each angle
        """

        rotations = [
            (0, "original"),
            (90, "rotate_90"),  # clockwise 90°
            (180, "rotate_180"),  # 180°
            (270, "rotate_270"),  # clockwise 270° (counter-clockwise 90°)
        ]

        results = {}
        best_score = -1
        best_angle = 0
        best_img = table_img
        score_0 = None

        for angle, name in rotations:
            # Rotate image
            if angle == 0:
                rotated_img = table_img
            else:
                # PIL's rotate is counter-clockwise, use negative angle for clockwise
                rotated_img = table_img.rotate(-angle, expand=True)

            # Convert to numpy array for OCR
            img_array = np.array(rotated_img)

            # Perform OCR detection and recognition
            try:
                ocr_results = self.ocr(img_array)

                if ocr_results:
                    # Calculate average confidence
                    scores = [conf for _, (_, conf) in ocr_results]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    total_regions = len(scores)

                    # Combined score: considers both average confidence and number of regions
                    # More regions + higher confidence = better orientation
                    combined_score = avg_score * (1 + 0.1 * min(total_regions, 50) / 50)
                else:
                    avg_score = 0
                    total_regions = 0
                    combined_score = 0

            except Exception as e:
                logger.warning(f"OCR failed for angle {angle}: {e}")
                avg_score = 0
                total_regions = 0
                combined_score = 0

            results[angle] = {
                "avg_confidence": avg_score,
                "total_regions": total_regions,
                "combined_score": combined_score,
            }
            if angle == 0:
                score_0 = combined_score

            logger.debug(
                f"Table orientation {angle}°: avg_conf={avg_score:.4f}, regions={total_regions}, combined={combined_score:.4f}"
            )

            if combined_score > best_score:
                best_score = combined_score
                best_angle = angle
                best_img = rotated_img

        # Absolute threshold rule:
        # Only choose non-0° if it exceeds 0° by more than 0.2 and 0° score is below 0.8.
        if best_angle != 0 and score_0 is not None:
            if not (best_score - score_0 > 0.2 and score_0 < 0.8):
                best_angle = 0
                best_img = table_img
                best_score = score_0

        results[best_angle] = results.get(
            best_angle, {"avg_confidence": 0, "total_regions": 0, "combined_score": 0}
        )

        logger.info(f"Best table orientation: {best_angle}° (score={best_score:.4f})")

        return best_angle, best_img, results

    def _table_transformer_job(self, ZM, auto_rotate=True, need_table_structure=True):
        """
        Process table structure recognition.

        When auto_rotate=True, the complete workflow:
        1. Evaluate table orientation and select the best rotation angle
        2. Use rotated image for table structure recognition (TSR)
        3. Re-OCR the rotated image
        4. Match new OCR results with TSR cell coordinates

        Args:
            ZM: Zoom factor
            auto_rotate: Whether to enable auto orientation correction
        """
        logger.debug("Table processing...")
        imgs, pos = [], []
        tbcnt = [0]
        MARGIN = 10
        self.tb_cpns = []
        self.table_rotations = {}  # Store rotation info for each table
        self.rotated_table_imgs = {}  # Store rotated table images
        self._last_selective_ocr_block_count = 0
        self._last_complex_block_counts = {}

        assert len(self.page_layout) == len(self.page_images)

        # Collect layout info for all tables
        table_layouts = []  # [(page, table_layout, left, top, right, bott), ...]

        table_index = 0
        for p, tbls in enumerate(self.page_layout):  # for page
            tbls = [f for f in tbls if f["type"] == "table"]
            tbcnt.append(len(tbls))
            if not tbls:
                continue
            for tb in tbls:  # for table
                left, top, right, bott = (
                    tb["x0"] - MARGIN,
                    tb["top"] - MARGIN,
                    tb["x1"] + MARGIN,
                    tb["bottom"] + MARGIN,
                )
                left *= ZM
                top *= ZM
                right *= ZM
                bott *= ZM
                pos.append((left, top, p, table_index))  # Add page and table_index

                # Record table layout info
                table_layouts.append(
                    {
                        "page": p,
                        "table_index": table_index,
                        "layout": tb,
                        "coords": (left, top, right, bott),
                    }
                )

                # Crop table image
                table_img = self.page_images[p].crop((left, top, right, bott))

                if auto_rotate:
                    # Evaluate table orientation
                    logger.debug(
                        f"Evaluating orientation for table {table_index} on page {p}"
                    )
                    best_angle, rotated_img, rotation_scores = (
                        self._evaluate_table_orientation(table_img)
                    )

                    # Store rotation info
                    self.table_rotations[table_index] = {
                        "page": p,
                        "original_pos": (left, top, right, bott),
                        "best_angle": best_angle,
                        "scores": rotation_scores,
                        "rotated_size": rotated_img.size,  # (width, height)
                    }

                    # Store the rotated image
                    self.rotated_table_imgs[table_index] = rotated_img
                    if need_table_structure:
                        imgs.append(rotated_img)

                else:
                    self.table_rotations[table_index] = {
                        "page": p,
                        "original_pos": (left, top, right, bott),
                        "best_angle": 0,
                        "scores": {},
                        "rotated_size": table_img.size,
                    }
                    self.rotated_table_imgs[table_index] = table_img
                    if need_table_structure:
                        imgs.append(table_img)

                table_index += 1

        assert len(self.page_images) == len(tbcnt) - 1
        if not table_layouts:
            return

        selective_table_count = self._ocr_selective_table_regions(
            ZM,
            table_layouts,
            getattr(self, "complex_block_only_pages", set()),
        )
        self._last_selective_ocr_block_count = int(selective_table_count or 0)
        self._last_complex_block_counts = {"table": self._last_selective_ocr_block_count}
        if not need_table_structure:
            return

        # Perform table structure recognition (TSR)
        recos = self._dispatch_table_structure_recognition(imgs)

        # If tables were rotated, re-OCR the rotated images and replace table boxes
        if auto_rotate:
            self._ocr_rotated_tables(ZM, table_layouts, recos, tbcnt)

        # Process TSR results (keep original logic but handle rotated coordinates)
        tbcnt = np.cumsum(tbcnt)
        for i in range(len(tbcnt) - 1):  # for page
            pg = []
            for j, tb_items in enumerate(recos[tbcnt[i] : tbcnt[i + 1]]):  # for table
                poss = pos[tbcnt[i] : tbcnt[i + 1]]
                for it in tb_items:  # for table components
                    # TSR coordinates are relative to rotated image, need to record
                    it["x0_rotated"] = it["x0"]
                    it["x1_rotated"] = it["x1"]
                    it["top_rotated"] = it["top"]
                    it["bottom_rotated"] = it["bottom"]

                    # For rotated tables, coordinate transformation to page space requires rotation
                    # Since we already re-OCR'd on rotated image, keep simple processing here
                    it["pn"] = poss[j][2]  # page number
                    it["layoutno"] = j
                    it["table_index"] = poss[j][3]  # table index
                    pg.append(it)
            self.tb_cpns.extend(pg)

        def gather(kwd, fzy=10, ption=0.6):
            eles = Recognizer.sort_Y_firstly(
                [r for r in self.tb_cpns if re.match(kwd, r["label"])], fzy
            )
            eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)
            return Recognizer.sort_Y_firstly(eles, 0)

        # add R,H,C,SP tag to boxes within table layout
        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        clmns = sorted(
            [r for r in self.tb_cpns if re.match(r"table column$", r["label"])],
            key=lambda x: (
                x["pn"],
                x["layoutno"],
                x["x0_rotated"] if "x0_rotated" in x else x["x0"],
            ),
        )
        clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)

        for b in self.boxes:
            if b.get("layout_type", "") != "table":
                continue
            ii = Recognizer.find_overlapped_with_threshold(b, rows, thr=0.3)
            if ii is not None:
                b["R"] = ii
                b["R_top"] = rows[ii]["top"]
                b["R_bott"] = rows[ii]["bottom"]

            ii = Recognizer.find_overlapped_with_threshold(b, headers, thr=0.3)
            if ii is not None:
                b["H_top"] = headers[ii]["top"]
                b["H_bott"] = headers[ii]["bottom"]
                b["H_left"] = headers[ii]["x0"]
                b["H_right"] = headers[ii]["x1"]
                b["H"] = ii

            ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
            if ii is not None:
                b["C"] = ii
                b["C_left"] = clmns[ii]["x0"]
                b["C_right"] = clmns[ii]["x1"]

            ii = Recognizer.find_overlapped_with_threshold(b, spans, thr=0.3)
            if ii is not None:
                b["H_top"] = spans[ii]["top"]
                b["H_bott"] = spans[ii]["bottom"]
                b["H_left"] = spans[ii]["x0"]
                b["H_right"] = spans[ii]["x1"]
                b["SP"] = ii

    def _ocr_rotated_tables(self, ZM, table_layouts, tsr_results, tbcnt):
        """
        Re-OCR rotated table images and update self.boxes.

        Args:
            ZM: Zoom factor
            table_layouts: List of table layout info
            tsr_results: TSR recognition results
            tbcnt: Cumulative table count per page
        """
        tbcnt = np.cumsum(tbcnt)

        def _table_region(layout, page_index):
            table_x0 = layout["x0"]
            table_top = layout["top"]
            table_x1 = layout["x1"]
            table_bottom = layout["bottom"]
            table_top_cum = table_top + self.page_cum_height[page_index]
            table_bottom_cum = table_bottom + self.page_cum_height[page_index]
            return (
                table_x0,
                table_top,
                table_x1,
                table_bottom,
                table_top_cum,
                table_bottom_cum,
            )

        def _collect_table_boxes(
            page_index, table_x0, table_x1, table_top_cum, table_bottom_cum
        ):
            indices = [
                i
                for i, b in enumerate(self.boxes)
                if (
                    int(b.get("page_number") or 0) == self._absolute_page_number(page_index)
                    and b.get("layout_type") == "table"
                    and b["x0"] >= table_x0 - 5
                    and b["x1"] <= table_x1 + 5
                    and b["top"] >= table_top_cum - 5
                    and b["bottom"] <= table_bottom_cum + 5
                )
            ]
            original_boxes = [self.boxes[i] for i in indices]
            insert_at = indices[0] if indices else len(self.boxes)
            for i in reversed(indices):
                self.boxes.pop(i)
            return original_boxes, insert_at

        def _restore_boxes(original_boxes, insert_at):
            for b in original_boxes:
                self.boxes.insert(insert_at, b)
                insert_at += 1
            return insert_at

        def _map_rotated_point(x, y, angle, width, height):
            # Map a point from rotated image coords back to original image coords.
            if angle == 0:
                return x, y
            if angle == 90:
                # clockwise 90: original->rotated (x', y') = (y, width - x)
                # inverse:
                return width - y, x
            if angle == 180:
                return width - x, height - y
            if angle == 270:
                # clockwise 270: original->rotated (x', y') = (height - y, x)
                # inverse:
                return y, height - x
            return x, y

        def _insert_ocr_boxes(
            ocr_results,
            page_index,
            table_x0,
            table_top,
            insert_at,
            table_index,
            best_angle,
            table_w_px,
            table_h_px,
        ):
            added = 0
            for bbox, (text, conf) in ocr_results:
                if conf < 0.5:
                    continue
                mapped = [
                    _map_rotated_point(p[0], p[1], best_angle, table_w_px, table_h_px)
                    for p in bbox
                ]
                x_coords = [p[0] for p in mapped]
                y_coords = [p[1] for p in mapped]
                box_x0 = min(x_coords) / ZM
                box_x1 = max(x_coords) / ZM
                box_top = min(y_coords) / ZM
                box_bottom = max(y_coords) / ZM
                new_box = {
                    "text": text,
                    "x0": box_x0 + table_x0,
                    "x1": box_x1 + table_x0,
                    "top": box_top + table_top + self.page_cum_height[page_index],
                    "bottom": box_bottom + table_top + self.page_cum_height[page_index],
                    "page_number": self._absolute_page_number(page_index),
                    "layout_type": "table",
                    "layoutno": f"table-{table_index}",
                    "_rotated": True,
                    "_rotation_angle": best_angle,
                    "_table_index": table_index,
                    "_rotated_x0": box_x0,
                    "_rotated_x1": box_x1,
                    "_rotated_top": box_top,
                    "_rotated_bottom": box_bottom,
                }
                self.boxes.insert(insert_at, new_box)
                insert_at += 1
                added += 1
            return added

        for tbl_info in table_layouts:
            table_index = tbl_info["table_index"]
            page = tbl_info["page"]
            layout = tbl_info["layout"]
            left, top, right, bott = tbl_info["coords"]

            rotation_info = self.table_rotations.get(table_index, {})
            best_angle = rotation_info.get("best_angle", 0)

            # Get the rotated table image
            rotated_img = self.rotated_table_imgs.get(table_index)
            if rotated_img is None:
                continue

            # If no rotation, keep original OCR boxes untouched.
            if best_angle == 0:
                continue

            # Table region is defined by layout's x0, top, x1, bottom (page-local coords)
            (
                table_x0,
                table_top,
                table_x1,
                table_bottom,
                table_top_cum,
                table_bottom_cum,
            ) = _table_region(layout, page)
            original_boxes, insert_at = _collect_table_boxes(
                page, table_x0, table_x1, table_top_cum, table_bottom_cum
            )

            logger.info(
                f"Re-OCR table {table_index} on page {page} with rotation {best_angle}°"
            )

            # Perform OCR on rotated image
            img_array = np.array(rotated_img)
            ocr_results = self.ocr(img_array)

            if not ocr_results:
                logger.warning(
                    f"No OCR results for rotated table {table_index}, restoring originals"
                )
                _restore_boxes(original_boxes, insert_at)
                continue

            # Add new OCR results to self.boxes
            # OCR coordinates are relative to rotated image, map back to original table coords
            table_w_px = right - left
            table_h_px = bott - top
            added = _insert_ocr_boxes(
                ocr_results,
                page,
                table_x0,
                table_top,
                insert_at,
                table_index,
                best_angle,
                table_w_px,
                table_h_px,
            )

            logger.info(f"Added {added} OCR results from rotated table {table_index}")

    def _ocr_page_boxes(self, pagenum, img, chars, ZM=3, device_id: int | None = None):
        start = timer()
        bxs = self.ocr.detect(np.array(img), device_id)
        logger.info(f"__ocr detecting boxes of a image cost ({timer() - start}s)")

        start = timer()
        if not bxs:
            return []
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = Recognizer.sort_Y_firstly(
            [
                {
                    "x0": b[0][0] / ZM,
                    "x1": b[1][0] / ZM,
                    "top": b[0][1] / ZM,
                    "text": "",
                    "txt": t,
                    "bottom": b[-1][1] / ZM,
                    "chars": [],
                    "page_number": pagenum,
                }
                for b, t in bxs
                if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
            ],
            self.mean_height[pagenum - 1] / 3,
        )

        # merge chars in the same rect
        for c in chars:
            ii = Recognizer.find_overlapped(c, bxs)
            if ii is None:
                self.lefted_chars.append(c)
                continue
            ch = c["bottom"] - c["top"]
            bh = bxs[ii]["bottom"] - bxs[ii]["top"]
            if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != " ":
                self.lefted_chars.append(c)
                continue
            bxs[ii]["chars"].append(c)

        for b in bxs:
            if not b["chars"]:
                del b["chars"]
                continue
            m_ht = np.mean([c["height"] for c in b["chars"]])
            for c in Recognizer.sort_Y_firstly(b["chars"], m_ht):
                if c["text"] == " " and b["text"]:
                    if re.match(r"[0-9a-zA-Zа-яА-Я,.?;:!%%]", b["text"][-1]):
                        b["text"] += " "
                else:
                    b["text"] += c["text"]
            del b["chars"]

        logger.info(f"__ocr sorting {len(chars)} chars cost {timer() - start}s")
        start = timer()
        boxes_to_reg = []
        img_np = np.array(img)
        for b in bxs:
            if not b["text"]:
                left, right, top, bott = (
                    b["x0"] * ZM,
                    b["x1"] * ZM,
                    b["top"] * ZM,
                    b["bottom"] * ZM,
                )
                b["box_image"] = self.ocr.get_rotate_crop_image(
                    img_np,
                    np.array(
                        [[left, top], [right, top], [right, bott], [left, bott]],
                        dtype=np.float32,
                    ),
                )
                boxes_to_reg.append(b)
            del b["txt"]
        texts = self.ocr.recognize_batch(
            [b["box_image"] for b in boxes_to_reg], device_id
        )
        for i in range(len(boxes_to_reg)):
            boxes_to_reg[i]["text"] = texts[i]
            del boxes_to_reg[i]["box_image"]
        logger.info(f"__ocr recognize {len(bxs)} boxes cost {timer() - start}s")
        bxs = [b for b in bxs if b["text"]]
        if bxs and self.mean_height[pagenum - 1] == 0:
            self.mean_height[pagenum - 1] = np.median(
                [b["bottom"] - b["top"] for b in bxs]
            )
        return bxs

    def __ocr(self, pagenum, img, chars, ZM=3, device_id: int | None = None):
        bxs = self._ocr_page_boxes(pagenum, img, chars, ZM=ZM, device_id=device_id)
        page_index = pagenum - int(getattr(self, "page_from", 0) or 0) - 1
        if page_index < 0:
            return
        while len(self.boxes) <= page_index:
            self.boxes.append([])
        self.boxes[page_index] = bxs

    def _layouts_rec(self, ZM, drop=True, page_numbers: set[int] | None = None):
        assert len(self.page_images) == len(self.boxes)

        selected_page_numbers = None
        if page_numbers is not None:
            selected_page_numbers = {int(page_number) for page_number in page_numbers if int(page_number) > 0}

        if selected_page_numbers is None:
            self.boxes, self.page_layout = self._dispatch_layout_recognition(
                self.page_images, self.boxes, ZM, drop=drop
            )
            for i in range(len(self.boxes)):
                self.boxes[i]["top"] += self.page_cum_height[
                    self.boxes[i]["page_number"] - 1
                ]
                self.boxes[i]["bottom"] += self.page_cum_height[
                    self.boxes[i]["page_number"] - 1
                ]
            return

        existing_layout = list(getattr(self, "page_layout", None) or [])
        page_layout = [
            [dict(layout) for layout in existing_layout[index]]
            if index < len(existing_layout) and existing_layout[index]
            else []
            for index in range(len(self.page_images))
        ]
        retained_boxes: list[dict[str, Any]] = []
        selected_images = []
        selected_boxes = []
        selected_indexes: list[int] = []

        for local_index, page_boxes in enumerate(self.boxes):
            page_number = self._absolute_page_number(local_index)
            if page_number in selected_page_numbers:
                selected_indexes.append(local_index)
                selected_images.append(self.page_images[local_index])
                selected_boxes.append(page_boxes)
                continue

            page_offset = self.page_cum_height[page_number - 1]
            for box in page_boxes:
                retained_box = dict(box)
                retained_box["top"] = float(retained_box.get("top") or 0.0) + page_offset
                retained_box["bottom"] = float(retained_box.get("bottom") or 0.0) + page_offset
                retained_boxes.append(retained_box)

        if not selected_images:
            self.page_layout = page_layout
            self.boxes = retained_boxes
            return

        laid_out_boxes, selected_layout = self._dispatch_layout_recognition(
            selected_images, selected_boxes, ZM, drop=drop
        )
        for local_index, layouts in zip(selected_indexes, selected_layout):
            page_layout[local_index] = [dict(layout) for layout in layouts]
        for box in laid_out_boxes:
            box["top"] += self.page_cum_height[
                box["page_number"] - 1
            ]
            box["bottom"] += self.page_cum_height[
                box["page_number"] - 1
            ]
        self.page_layout = page_layout
        self.boxes = sorted(
            retained_boxes + laid_out_boxes,
            key=lambda box: (
                int(box.get("page_number") or 0),
                float(box.get("top") or 0.0),
                float(box.get("x0") or 0.0),
            ),
        )

    def _assign_column(self, boxes, zoomin=3):
        if not boxes:
            return boxes
        if all("col_id" in b for b in boxes):
            return boxes

        by_page = defaultdict(list)
        for b in boxes:
            by_page[b["page_number"]].append(b)

        page_cols = {}

        for pg, bxs in by_page.items():
            if not bxs:
                page_cols[pg] = 1
                continue

            if self._uses_native_fast_column_path(int(pg)):
                self._assign_column_for_hybrid_clean_page(pg, bxs, zoomin=zoomin)
                page_cols[pg] = len({int(box.get("col_id") or 0) for box in bxs}) or 1
                continue

            x0s_raw = np.array([b["x0"] for b in bxs], dtype=float)

            min_x0 = np.min(x0s_raw)
            max_x1 = np.max([b["x1"] for b in bxs])
            width = max_x1 - min_x0

            INDENT_TOL = width * 0.12
            x0s = []
            for x in x0s_raw:
                if abs(x - min_x0) < INDENT_TOL:
                    x0s.append([min_x0])
                else:
                    x0s.append([x])
            x0s = np.array(x0s, dtype=float)

            max_try = min(4, len(bxs))
            if max_try < 2:
                max_try = 1
            best_k = 1
            best_score = -1

            for k in range(1, max_try + 1):
                km = KMeans(n_clusters=k, n_init="auto")
                labels = km.fit_predict(x0s)

                centers = np.sort(km.cluster_centers_.flatten())
                if len(centers) > 1:
                    try:
                        score = silhouette_score(x0s, labels)
                    except ValueError:
                        continue
                else:
                    score = 0
                if score > best_score:
                    best_score = score
                    best_k = k

            page_cols[pg] = best_k
            logger.info(f"[Page {pg}] best_score={best_score:.2f}, best_k={best_k}")

        global_cols = Counter(page_cols.values()).most_common(1)[0][0]
        logger.info(f"Global column_num decided by majority: {global_cols}")

        for pg, bxs in by_page.items():
            if not bxs:
                continue
            if self._uses_native_fast_column_path(int(pg)):
                continue
            k = page_cols[pg]
            if len(bxs) < k:
                k = 1
            x0s = np.array([[b["x0"]] for b in bxs], dtype=float)
            km = KMeans(n_clusters=k, n_init="auto")
            labels = km.fit_predict(x0s)

            centers = km.cluster_centers_.flatten()
            order = np.argsort(centers)

            remap = {orig: new for new, orig in enumerate(order)}

            for b, lb in zip(bxs, labels):
                b["col_id"] = remap[lb]

            grouped = defaultdict(list)
            for b in bxs:
                grouped[b["col_id"]].append(b)

        return boxes

    def _uses_native_fast_column_path(self, page_number: int) -> bool:
        page_number = int(page_number or 0)
        if page_number <= 0:
            return False
        return (
            page_number in getattr(self, "hybrid_clean_pages", set())
            or page_number in getattr(self, "complex_block_only_pages", set())
        )

    def _assign_column_for_hybrid_clean_page(self, page_number: int, boxes, *, zoomin=3):
        if not boxes:
            return

        local_index = self._page_local_index(page_number)
        page_images = getattr(self, "page_images", []) or []
        page_width = 0.0
        if 0 <= local_index < len(page_images):
            page_width = float(page_images[local_index].size[0]) / max(float(zoomin or 1), 1.0)

        mean_width = 8.0
        if getattr(self, "mean_width", None) and 0 <= page_number - 1 < len(self.mean_width):
            mean_width = float(self.mean_width[page_number - 1] or 8.0)

        same_column_tolerance = max(page_width * 0.04, mean_width * 6.0, 18.0)
        ordered_boxes = sorted(
            boxes,
            key=lambda box: (
                float(box.get("x0") or 0.0),
                float(box.get("top") or 0.0),
            ),
        )

        column_centers: list[float] = []
        column_counts: list[int] = []
        assignments: dict[int, int] = {}

        for box in ordered_boxes:
            x0 = float(box.get("x0") or 0.0)
            matched_column = None
            matched_distance = None
            for column_index, center in enumerate(column_centers):
                distance = abs(x0 - center)
                if distance <= same_column_tolerance and (
                    matched_distance is None or distance < matched_distance
                ):
                    matched_column = column_index
                    matched_distance = distance

            if matched_column is None:
                matched_column = len(column_centers)
                column_centers.append(x0)
                column_counts.append(1)
            else:
                count = column_counts[matched_column]
                column_centers[matched_column] = ((column_centers[matched_column] * count) + x0) / (count + 1)
                column_counts[matched_column] = count + 1

            assignments[id(box)] = matched_column

        column_order = {
            original_index: remapped_index
            for remapped_index, original_index in enumerate(
                sorted(range(len(column_centers)), key=lambda index: column_centers[index])
            )
        }
        for box in boxes:
            box["col_id"] = column_order.get(assignments.get(id(box), 0), 0)

    def _text_merge(self, zoomin=3):
        # merge adjusted boxes
        bxs = self._assign_column(self.boxes, zoomin)

        def end_with(b, txt):
            txt = txt.strip()
            tt = b.get("text", "").strip()
            return tt and tt.find(txt) == len(tt) - len(txt)

        def start_with(b, txts):
            tt = b.get("text", "").strip()
            return tt and any([tt.find(t.strip()) == 0 for t in txts])

        # horizontally merge adjacent box with the same layout
        i = 0
        while i < len(bxs) - 1:
            b = bxs[i]
            b_ = bxs[i + 1]

            if b["page_number"] != b_["page_number"] or b.get("col_id") != b_.get(
                "col_id"
            ):
                i += 1
                continue

            if b.get("layoutno", "0") != b_.get("layoutno", "1") or b.get(
                "layout_type", ""
            ) in ["table", "figure", "equation"]:
                i += 1
                continue

            if (
                abs(self._y_dis(b, b_))
                < self.mean_height[bxs[i]["page_number"] - 1] / 3
            ):
                # merge
                bxs[i]["x1"] = b_["x1"]
                bxs[i]["top"] = (b["top"] + b_["top"]) / 2
                bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                bxs[i]["text"] += b_["text"]
                bxs.pop(i + 1)
                continue
            i += 1
        self.boxes = bxs

    def _page_local_index(self, page_number: int) -> int:
        idx = int(page_number) - 1
        page_count = len(getattr(self, "page_images", []) or [])
        if idx >= page_count and getattr(self, "page_from", 0):
            idx = int(page_number) - 1 - int(self.page_from)
        return idx

    def _page_start_height(self, page_number: int) -> tuple[float, float]:
        idx = self._page_local_index(page_number)
        raw_cum = getattr(self, "page_cum_height", None)
        cum = list(raw_cum) if raw_cum is not None else []
        if 0 <= idx and idx + 1 < len(cum):
            start = float(cum[idx])
            return start, max(1.0, float(cum[idx + 1]) - start)

        page_images = getattr(self, "page_images", []) or []
        if 0 <= idx < len(page_images):
            return 0.0, float(page_images[idx].size[1])
        return 0.0, 1.0

    @staticmethod
    def _horizontal_overlap_ratio(a, b) -> float:
        overlap = max(0.0, min(float(a["x1"]), float(b["x1"])) - max(float(a["x0"]), float(b["x0"])))
        width = max(1.0, min(float(a["x1"]) - float(a["x0"]), float(b["x1"]) - float(b["x0"])))
        return overlap / width

    @staticmethod
    def _normalized_margin_text(text: str) -> str:
        text = re.sub(r"\s+", "", (text or "").strip().lower())
        text = re.sub(r"[\-–—_·•|]+", "", text)
        return text

    @staticmethod
    def _box_type(box) -> str:
        return str(box.get("semantic_type") or box.get("layout_type") or "").strip().lower()

    def _is_margin_artifact_candidate(self, box) -> bool:
        text = str(box.get("text") or "").strip()
        if len(text) < 3:
            return False

        box_type = self._box_type(box)
        if "header" in box_type or "footer" in box_type:
            return True
        if "title" in box_type:
            return False

        page_number = int(box.get("page_number", 0) or 0)
        page_start, page_height = self._page_start_height(page_number)
        top_local = float(box.get("top", 0)) - page_start
        bottom_local = float(box.get("bottom", 0)) - page_start
        mh = max(
            1.0,
            float(self.mean_height[min(page_number - 1, len(self.mean_height) - 1)])
            if getattr(self, "mean_height", None) and page_number > 0
            else 10.0,
        )
        margin_band = max(mh * 2.0, page_height * 0.06)
        return top_local <= margin_band or page_height - bottom_local <= margin_band

    def _remove_repeating_margin_artifacts(self, boxes):
        occurrences = defaultdict(set)
        for box in boxes:
            if not self._is_margin_artifact_candidate(box):
                continue
            normalized = self._normalized_margin_text(str(box.get("text") or ""))
            if not normalized:
                continue
            occurrences[normalized].add(int(box.get("page_number", 0) or 0))

        repeating = {text for text, pages in occurrences.items() if len({page for page in pages if page > 0}) >= 2}
        if not repeating:
            return boxes

        kept = []
        removed = 0
        for box in boxes:
            normalized = self._normalized_margin_text(str(box.get("text") or ""))
            if normalized in repeating and self._is_margin_artifact_candidate(box):
                removed += 1
                continue
            kept.append(box)
        if removed:
            logger.info("_final_reading_order_merge: removed %d repeated header/footer boxes", removed)
        return kept

    def _is_new_section_boundary(self, box) -> bool:
        text = str(box.get("text") or "").strip()
        box_type = self._box_type(box)
        if any(token in box_type for token in ("title", "caption", "header", "footer")):
            return True
        if box_type in {"table", "figure", "equation", "seal"}:
            return True
        if self.proj_match(text):
            return True
        if len(text) <= 80 and re.match(r"^[A-Z][A-Za-z0-9 ,/&()：:.-]{2,}$", text) and text[-1:] not in ".。!?！？":
            return True
        return False

    def _rules_reject_cross_page_text_candidate(self, up, down, page_width: float, mw: float) -> bool:
        if self._is_new_section_boundary(down):
            return True
        if self._is_new_section_boundary(up) and self._box_type(up) != "text":
            return True
        indent_delta = float(down.get("x0", 0)) - float(up.get("x0", 0))
        if indent_delta > max(mw * 3.0, page_width * 0.05):
            return True
        return False

    def _bind_nearby_captions(self, boxes) -> None:
        assets_by_page = defaultdict(list)
        captions = []
        for box in boxes:
            box_type = self._box_type(box)
            layout_type = str(box.get("layout_type") or "").strip().lower()
            if layout_type in {"figure", "table", "equation"}:
                assets_by_page[int(box.get("page_number", 0) or 0)].append(box)
            elif "caption" in box_type:
                captions.append(box)

        for caption in captions:
            page_number = int(caption.get("page_number", 0) or 0)
            mh = max(
                1.0,
                float(self.mean_height[min(page_number - 1, len(self.mean_height) - 1)])
                if getattr(self, "mean_height", None) and page_number > 0
                else 10.0,
            )
            best_asset = None
            best_distance = None
            for asset in assets_by_page.get(page_number, []):
                if self._horizontal_overlap_ratio(caption, asset) < 0.15:
                    continue
                vertical_gap = min(
                    abs(float(caption.get("bottom", 0)) - float(asset.get("top", 0))),
                    abs(float(asset.get("bottom", 0)) - float(caption.get("top", 0))),
                )
                if vertical_gap > mh * 5.0:
                    continue
                if best_distance is None or vertical_gap < best_distance:
                    best_asset = asset
                    best_distance = vertical_gap
            if best_asset is None:
                continue

            asset_type = str(best_asset.get("layout_type") or "").strip().lower()
            asset_layoutno = str(best_asset.get("layoutno") or "").strip()
            caption["bound_asset_type"] = asset_type
            if asset_layoutno:
                caption["bound_asset_layoutno"] = asset_layoutno
                caption_layoutnos = list(best_asset.get("caption_layoutnos") or [])
                caption_layoutno = str(caption.get("layoutno") or "").strip()
                if caption_layoutno and caption_layoutno not in caption_layoutnos:
                    caption_layoutnos.append(caption_layoutno)
                    best_asset["caption_layoutnos"] = caption_layoutnos

    @staticmethod
    def _join_cross_page_text(left: str, right: str) -> str:
        left = (left or "").rstrip()
        right = (right or "").lstrip()
        if not left:
            return right
        if not right:
            return left
        if re.match(r"[A-Za-z0-9]$", left) and re.match(r"[A-Za-z0-9]", right):
            return f"{left} {right}"
        if left[-1] in ",;:'\"，、‘“；：-" or right[0] in ".,;:!?，。；？！、":
            return f"{left}{right}"
        if tokenizer.is_chinese(left[-1]) and tokenizer.is_chinese(right[0]):
            return f"{left}{right}"
        return f"{left} {right}"

    def _is_cross_page_text_candidate(self, up, down) -> bool:
        up_text = str(up.get("text") or "").strip()
        down_text = str(down.get("text") or "").strip()
        if not up_text or not down_text:
            return False

        text_layout_types = {"", "text", "paragraph", "content", "unknown"}
        if str(up.get("layout_type") or "").strip().lower() not in text_layout_types:
            return False
        if str(down.get("layout_type") or "").strip().lower() not in text_layout_types:
            return False
        if str(up.get("semantic_type") or "").strip().lower() == "seal":
            return False
        if str(down.get("semantic_type") or "").strip().lower() == "seal":
            return False

        hard_endings = "。？！!?；;."
        if up_text[-1] in hard_endings:
            return False
        if self.proj_match(down_text):
            return False

        up_pages = up.get("merged_page_numbers") or [int(up.get("page_number", 0))]
        last_up_page = max(int(page) for page in up_pages if int(page) > 0)
        down_page = int(down.get("page_number", 0))
        if down_page != last_up_page + 1:
            return False

        if up.get("col_id") is not None and down.get("col_id") is not None and up.get("col_id") != down.get("col_id"):
            return False

        up_start, up_page_height = self._page_start_height(last_up_page)
        down_start, down_page_height = self._page_start_height(down_page)
        up_bottom_local = float(up.get("bottom", 0)) - up_start
        down_top_local = float(down.get("top", 0)) - down_start
        mh = max(1.0, float(self.mean_height[min(down_page - 1, len(self.mean_height) - 1)]) if getattr(self, "mean_height", None) else 10.0)
        mw = max(1.0, float(self.mean_width[min(down_page - 1, len(self.mean_width) - 1)]) if getattr(self, "mean_width", None) else 8.0)

        bottom_band = max(mh * 3.0, up_page_height * 0.08)
        top_band = max(mh * 3.0, down_page_height * 0.08)
        if up_page_height - up_bottom_local > bottom_band:
            return False
        if down_top_local > top_band:
            return False

        if self._horizontal_overlap_ratio(up, down) < 0.55:
            return False

        page_images = getattr(self, "page_images", []) or []
        page_idx = self._page_local_index(down_page)
        page_width = float(page_images[page_idx].size[0]) if 0 <= page_idx < len(page_images) else max(float(up["x1"]), float(down["x1"]))
        if abs(float(up["x0"]) - float(down["x0"])) > max(mw * 4.0, page_width * 0.08):
            return False
        if _pdf_parser_reading_order_strategy() == "rules" and self._rules_reject_cross_page_text_candidate(
            up,
            down,
            page_width,
            mw,
        ):
            return False
        return True

    def _merge_cross_page_text(self):
        if not getattr(self, "boxes", None):
            return

        self.boxes = sorted(
            self.boxes,
            key=lambda b: (
                max(b.get("merged_page_numbers") or [int(b.get("page_number", 0))]),
                b.get("col_id", 0),
                b.get("top", 0),
                b.get("x0", 0),
            ),
        )

        i = 0
        merged_count = 0
        while i + 1 < len(self.boxes):
            up = self.boxes[i]
            down = self.boxes[i + 1]
            if not self._is_cross_page_text_candidate(up, down):
                i += 1
                continue

            down_page = int(down["page_number"])
            down_start, _down_page_height = self._page_start_height(down_page)
            first_page = int(up.get("page_number", 0))
            first_start, _first_page_height = self._page_start_height(first_page)
            down_bottom_local = float(down["bottom"]) - down_start
            merged_pages = sorted(set([int(p) for p in up.get("merged_page_numbers", [first_page])] + [down_page]))
            merged_bottom = first_start + sum(self._page_start_height(page)[1] for page in range(first_page, down_page)) + down_bottom_local

            up["text"] = self._join_cross_page_text(str(up.get("text") or ""), str(down.get("text") or ""))
            up["x0"] = min(float(up["x0"]), float(down["x0"]))
            up["x1"] = max(float(up["x1"]), float(down["x1"]))
            up["bottom"] = max(float(up["bottom"]), merged_bottom)
            up["page_number"] = first_page
            up["merged_page_numbers"] = merged_pages
            up["merge_reason"] = "text_cross_page_continuation"
            up["source_box_count"] = int(up.get("source_box_count") or 1) + int(down.get("source_box_count") or 1)
            source_layoutnos = []
            for box in (up, down):
                if isinstance(box.get("source_layoutnos"), list):
                    source_layoutnos.extend(str(item) for item in box["source_layoutnos"] if str(item).strip())
                elif str(box.get("layoutno") or "").strip():
                    source_layoutnos.append(str(box["layoutno"]))
            if source_layoutnos:
                up["source_layoutnos"] = list(dict.fromkeys(source_layoutnos))
            self.boxes.pop(i + 1)
            merged_count += 1

        if merged_count:
            self.boxes = Recognizer.sort_Y_firstly(self.boxes, 0)
            logger.info("_merge_cross_page_text: merged %d paragraph continuations", merged_count)

    def _naive_vertical_merge(self, zoomin=3):
        # bxs = self._assign_column(self.boxes, zoomin)
        bxs = self.boxes

        grouped = defaultdict(list)
        for b in bxs:
            # grouped[(b["page_number"], b.get("col_id", 0))].append(b)
            grouped[(b["page_number"], "x")].append(b)

        merged_boxes = []
        for (pg, col), bxs in grouped.items():
            bxs = sorted(bxs, key=lambda x: (x["top"], x["x0"]))
            if not bxs:
                continue

            mh = (
                self.mean_height[pg - 1]
                if self.mean_height
                else np.median([b["bottom"] - b["top"] for b in bxs]) or 10
            )

            i = 0
            while i + 1 < len(bxs):
                b = bxs[i]
                b_ = bxs[i + 1]

                if b["page_number"] < b_["page_number"] and re.match(
                    r"[0-9  •一—-]+$", b["text"]
                ):
                    bxs.pop(i)
                    continue

                if not b["text"].strip():
                    bxs.pop(i)
                    continue

                if not b["text"].strip() or b.get("layoutno") != b_.get("layoutno"):
                    i += 1
                    continue

                if b_["top"] - b["bottom"] > mh * 1.5:
                    i += 1
                    continue

                overlap = max(0, min(b["x1"], b_["x1"]) - max(b["x0"], b_["x0"]))
                if overlap / max(1, min(b["x1"] - b["x0"], b_["x1"] - b_["x0"])) < 0.3:
                    i += 1
                    continue

                concatting_feats = [
                    b["text"].strip()[-1] in ",;:'\"，、‘“；：-",
                    len(b["text"].strip()) > 1
                    and b["text"].strip()[-2] in ",;:'\"，‘“、；：",
                    b_["text"].strip()
                    and b_["text"].strip()[0] in "。；？！?”）),，、：",
                ]
                # features for not concating
                feats = [
                    b.get("layoutno", 0) != b_.get("layoutno", 0),
                    b["text"].strip()[-1] in "。？！?",
                    self.is_english and b["text"].strip()[-1] in ".!?",
                    b["page_number"] == b_["page_number"]
                    and b_["top"] - b["bottom"]
                    > self.mean_height[b["page_number"] - 1] * 1.5,
                    b["page_number"] < b_["page_number"]
                    and abs(b["x0"] - b_["x0"])
                    > self.mean_width[b["page_number"] - 1] * 4,
                ]
                # split features
                detach_feats = [b["x1"] < b_["x0"], b["x0"] > b_["x1"]]
                if (any(feats) and not any(concatting_feats)) or any(detach_feats):
                    logger.debug(
                        "{} {} {} {}".format(
                            b["text"],
                            b_["text"],
                            any(feats),
                            any(concatting_feats),
                        )
                    )
                    i += 1
                    continue

                b["text"] = (b["text"].rstrip() + " " + b_["text"].lstrip()).strip()
                b["bottom"] = b_["bottom"]
                b["x0"] = min(b["x0"], b_["x0"])
                b["x1"] = max(b["x1"], b_["x1"])
                bxs.pop(i + 1)

            merged_boxes.extend(bxs)

        # self.boxes = sorted(merged_boxes, key=lambda x: (x["page_number"], x.get("col_id", 0), x["top"]))
        self.boxes = merged_boxes

    def _final_reading_order_merge(self, zoomin=3):
        if not self.boxes:
            return

        if _pdf_parser_reading_order_strategy() == "rules":
            self.boxes = self._assign_column(self.boxes, zoomin=zoomin)
            self.boxes = self._remove_repeating_margin_artifacts(self.boxes)
            self._bind_nearby_captions(self.boxes)

            pages = defaultdict(lambda: defaultdict(list))
            for b in self.boxes:
                pg = b["page_number"]
                col = b.get("col_id", 0)
                pages[pg][col].append(b)

            for pg in pages:
                for col in pages[pg]:
                    pages[pg][col].sort(key=lambda x: (x["top"], x["x0"]))

            new_boxes = []
            for pg in sorted(pages.keys()):
                for col in sorted(pages[pg].keys()):
                    new_boxes.extend(pages[pg][col])

            self.boxes = new_boxes
            return

        self.boxes = self._assign_column(self.boxes, zoomin=zoomin)

        pages = defaultdict(lambda: defaultdict(list))
        for b in self.boxes:
            pg = b["page_number"]
            col = b.get("col_id", 0)
            pages[pg][col].append(b)

        for pg in pages:
            for col in pages[pg]:
                pages[pg][col].sort(key=lambda x: (x["top"], x["x0"]))

        new_boxes = []
        for pg in sorted(pages.keys()):
            for col in sorted(pages[pg].keys()):
                new_boxes.extend(pages[pg][col])

        self.boxes = new_boxes

    def _apply_reading_order_strategy(self, zoomin=3):
        if _pdf_parser_reading_order_strategy() == "rules":
            self._final_reading_order_merge(zoomin=zoomin)

    def _concat_downward(self, concat_between_pages=True):
        self.boxes = Recognizer.sort_Y_firstly(self.boxes, 0)
        return

        # count boxes in the same row as a feature
        for i in range(len(self.boxes)):
            mh = self.mean_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["in_row"] = 0
            j = max(0, i - 12)
            while j < min(i + 12, len(self.boxes)):
                if j == i:
                    j += 1
                    continue
                ydis = self._y_dis(self.boxes[i], self.boxes[j]) / mh
                if abs(ydis) < 1:
                    self.boxes[i]["in_row"] += 1
                elif ydis > 0:
                    break
                j += 1

        # concat between rows
        boxes = deepcopy(self.boxes)
        blocks = []
        while boxes:
            chunks = []

            def dfs(up, dp):
                chunks.append(up)
                i = dp
                while i < min(dp + 12, len(boxes)):
                    ydis = self._y_dis(up, boxes[i])
                    smpg = up["page_number"] == boxes[i]["page_number"]
                    mh = self.mean_height[up["page_number"] - 1]
                    mw = self.mean_width[up["page_number"] - 1]
                    if smpg and ydis > mh * 4:
                        break
                    if not smpg and ydis > mh * 16:
                        break
                    down = boxes[i]
                    if (
                        not concat_between_pages
                        and down["page_number"] > up["page_number"]
                    ):
                        break

                    if up.get("R", "") != down.get("R", "") and up["text"][-1] != "，":
                        i += 1
                        continue

                    if (
                        re.match(r"[0-9]{2,3}/[0-9]{3}$", up["text"])
                        or re.match(r"[0-9]{2,3}/[0-9]{3}$", down["text"])
                        or not down["text"].strip()
                    ):
                        i += 1
                        continue

                    if not down["text"].strip() or not up["text"].strip():
                        i += 1
                        continue

                    if (
                        up["x1"] < down["x0"] - 10 * mw
                        or up["x0"] > down["x1"] + 10 * mw
                    ):
                        i += 1
                        continue

                    if i - dp < 5 and up.get("layout_type") == "text":
                        if up.get("layoutno", "1") == down.get("layoutno", "2"):
                            dfs(down, i + 1)
                            boxes.pop(i)
                            return
                        i += 1
                        continue

                    fea = self._updown_concat_features(up, down)
                    if self.updown_cnt_mdl.predict(xgb.DMatrix([fea]))[0] <= 0.5:
                        i += 1
                        continue
                    dfs(down, i + 1)
                    boxes.pop(i)
                    return

            dfs(boxes[0], 1)
            boxes.pop(0)
            if chunks:
                blocks.append(chunks)

        # concat within each block
        boxes = []
        for b in blocks:
            if len(b) == 1:
                boxes.append(b[0])
                continue
            t = b[0]
            for c in b[1:]:
                t["text"] = t["text"].strip()
                c["text"] = c["text"].strip()
                if not c["text"]:
                    continue
                if t["text"] and re.match(
                    r"[0-9\.a-zA-Z]+$", t["text"][-1] + c["text"][-1]
                ):
                    t["text"] += " "
                t["text"] += c["text"]
                t["x0"] = min(t["x0"], c["x0"])
                t["x1"] = max(t["x1"], c["x1"])
                t["page_number"] = min(t["page_number"], c["page_number"])
                t["bottom"] = c["bottom"]
                if not t["layout_type"] and c["layout_type"]:
                    t["layout_type"] = c["layout_type"]
            boxes.append(t)

        self.boxes = Recognizer.sort_Y_firstly(boxes, 0)

    def _filter_forpages(self):
        if not self.boxes:
            return
        findit = False
        i = 0
        while i < len(self.boxes):
            if not re.match(
                r"(contents|目录|目次|table of contents|致谢|acknowledge)$",
                re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower()),
            ):
                i += 1
                continue
            findit = True
            eng = re.match(r"[0-9a-zA-Z :'.-]{5,}", self.boxes[i]["text"].strip())
            self.boxes.pop(i)
            if i >= len(self.boxes):
                break
            prefix = (
                self.boxes[i]["text"].strip()[:3]
                if not eng
                else " ".join(self.boxes[i]["text"].strip().split()[:2])
            )
            while not prefix:
                self.boxes.pop(i)
                if i >= len(self.boxes):
                    break
                prefix = (
                    self.boxes[i]["text"].strip()[:3]
                    if not eng
                    else " ".join(self.boxes[i]["text"].strip().split()[:2])
                )
            self.boxes.pop(i)
            if i >= len(self.boxes) or not prefix:
                break
            for j in range(i, min(i + 128, len(self.boxes))):
                if not re.match(prefix, self.boxes[j]["text"]):
                    continue
                for k in range(i, j):
                    self.boxes.pop(i)
                break
        if findit:
            return

        page_dirty = [0] * len(self.page_images)
        for b in self.boxes:
            if re.search(r"(··|··|··)", b["text"]):
                page_dirty[b["page_number"] - 1] += 1
        page_dirty = set([i + 1 for i, t in enumerate(page_dirty) if t > 3])
        if not page_dirty:
            return
        i = 0
        while i < len(self.boxes):
            if self.boxes[i]["page_number"] in page_dirty:
                self.boxes.pop(i)
                continue
            i += 1

    def _merge_with_same_bullet(self):
        i = 0
        while i + 1 < len(self.boxes):
            b = self.boxes[i]
            b_ = self.boxes[i + 1]
            if not b["text"].strip():
                self.boxes.pop(i)
                continue
            if not b_["text"].strip():
                self.boxes.pop(i + 1)
                continue

            if (
                b["text"].strip()[0] != b_["text"].strip()[0]
                or b["text"].strip()[0].lower() in set("qwertyuopasdfghjklzxcvbnm")
                or tokenizer.is_chinese(b["text"].strip()[0])
                or b["top"] > b_["bottom"]
            ):
                i += 1
                continue
            b_["text"] = b["text"] + "\n" + b_["text"]
            b_["x0"] = min(b["x0"], b_["x0"])
            b_["x1"] = max(b["x1"], b_["x1"])
            b_["top"] = b["top"]
            self.boxes.pop(i)

    def _recognize_seals(self, zoomin=3):
        """检测印章并以 ``[印章: xxx]`` 行内 Markdown 形式插入 self.boxes。

        - 全页扫描 ``self.page_images``,调用 PP-OCRv4 seal_det。
        - 检测到的多边形做极坐标展开 → 水平条 → 复用 ``self.ocr.recognize_batch``。
        - 每个识别成功的印章在该页插入一个新 box(layout_type="text",位于页末)。
        - 模型缺失或推理失败时静默降级,不影响主解析流程。
        """
        if not getattr(self, "page_images", None):
            return

        try:
            from deepdoc.vision.seal_recognizer import get_seal_recognizer
        except Exception:
            logger.exception("Failed to import seal_recognizer")
            return

        recognizer = get_seal_recognizer()
        if recognizer is None:
            logger.warning("Seal recognition skipped: model not available")
            return

        page_from = getattr(self, "page_from", 0)
        added_total = 0

        for pn, page_img in enumerate(self.page_images):
            try:
                rgb = np.array(page_img)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                logger.exception("_recognize_seals: convert page %s failed", pn)
                continue

            regions = recognizer.detect(bgr)
            if not regions:
                continue

            crops = [region[0] for region in regions]
            try:
                texts = self.ocr.recognize_batch(crops)
            except Exception:
                logger.exception("_recognize_seals: rec batch failed on page %s", pn)
                continue

            cum_offset = self.page_cum_height[pn] if pn < len(self.page_cum_height) else 0
            # 找到该页 boxes 的最大 bottom,确保印章 box 插入到页末尾
            page_boxes = [b for b in self.boxes if b.get("page_number", 0) - 1 - page_from == pn]
            base_bottom = max((b["bottom"] for b in page_boxes), default=cum_offset)

            for idx, ((_, polygon, score), text) in enumerate(zip(regions, texts)):
                text = (text or "").strip()
                if not text:
                    continue
                xs = polygon[:, 0]
                ys = polygon[:, 1]
                x0 = float(np.min(xs)) / zoomin
                x1 = float(np.max(xs)) / zoomin
                top = float(np.min(ys)) / zoomin + cum_offset
                bottom = float(np.max(ys)) / zoomin + cum_offset

                new_box = {
                    "text": f"[印章: {text}]",
                    "score": float(score),
                    "x0": x0,
                    "x1": x1,
                    "top": max(base_bottom + 1 + idx, top),
                    "bottom": max(base_bottom + 1 + idx, bottom),
                    "page_number": pn + 1 + page_from,
                    "layout_type": "text",
                    "semantic_type": "seal",
                    "layoutno": f"seal-{pn}-{idx}",
                    "source_page_index": pn,
                    "source_bbox_local": [x0, float(np.min(ys)) / zoomin, x1, float(np.max(ys)) / zoomin],
                    "source_polygon": polygon.tolist(),
                }
                self.boxes.append(new_box)
                added_total += 1
                logger.info(
                    "_recognize_seals: page=%s text=%r polygon_score=%.3f",
                    pn, text, score,
                )

        if added_total:
            self.boxes.sort(
                key=lambda x: (x.get("page_number", 0), x.get("col_id", 0), x.get("top", 0))
            )
            logger.info("_recognize_seals: added %d seal boxes", added_total)

    def _recognize_formulas(self, zoomin=3):
        """识别 layout 命中的公式区域,替换为 ``$$...$$`` Markdown。

        - 仅处理 ``self.page_layout`` 中 type=="equation" 的区域。
        - 裁剪 ``self.page_images[pn]`` 对应区域,调用 RapidLatexOCR。
        - 删除与公式区域重叠 (IoU>=0.5) 的现有 boxes,插入一个新的公式 box。
        - 失败时静默降级,不影响主解析流程。
        """
        if not getattr(self, "page_layout", None) or not getattr(self, "page_images", None):
            return

        try:
            from deepdoc.vision.formula_recognizer import get_formula_recognizer
        except Exception:
            logger.exception("Failed to import formula_recognizer")
            return

        recognizer = get_formula_recognizer()
        if recognizer is None:
            logger.warning("Formula recognition skipped: model not available")
            return

        page_from = getattr(self, "page_from", 0)
        formulas_per_page: dict[int, list[dict]] = {}
        for pn, layouts in enumerate(self.page_layout):
            formulas = [lt for lt in layouts if lt.get("type") == "equation"]
            if formulas:
                formulas_per_page[pn] = formulas

        if not formulas_per_page:
            logger.info("_recognize_formulas: no equation region in layout")
            return

        to_remove: set[int] = set()
        new_formula_boxes: list[dict] = []
        formula_seq = 0
        for pn, formulas in formulas_per_page.items():
            if pn >= len(self.page_images):
                continue
            page_img = self.page_images[pn]
            cum_offset = self.page_cum_height[pn] if pn < len(self.page_cum_height) else 0

            for fl in formulas:
                x0_img = max(0, int(fl["x0"] * zoomin))
                top_img = max(0, int(fl["top"] * zoomin))
                x1_img = int(fl["x1"] * zoomin)
                bot_img = int(fl["bottom"] * zoomin)
                if x1_img <= x0_img or bot_img <= top_img:
                    continue

                try:
                    crop = page_img.crop((x0_img, top_img, x1_img, bot_img))
                except Exception:
                    logger.exception("_recognize_formulas: crop failed pn=%s", pn)
                    continue

                arr = np.array(crop)
                latex, elapsed = recognizer.predict(arr)
                logger.info(
                    "_recognize_formulas: page=%s region=(%s,%s,%s,%s) latex_len=%s cost=%.3fs",
                    pn, x0_img, top_img, x1_img, bot_img, len(latex), elapsed,
                )
                if not latex:
                    continue

                global_top = float(fl["top"]) + cum_offset
                global_bot = float(fl["bottom"]) + cum_offset
                fl_x0 = float(fl["x0"])
                fl_x1 = float(fl["x1"])

                for i, b in enumerate(self.boxes):
                    if i in to_remove:
                        continue
                    if b.get("page_number", 0) - 1 - page_from != pn:
                        continue
                    ox = max(0.0, min(b["x1"], fl_x1) - max(b["x0"], fl_x0))
                    oy = max(0.0, min(b["bottom"], global_bot) - max(b["top"], global_top))
                    box_area = max(1e-6, (b["x1"] - b["x0"]) * (b["bottom"] - b["top"]))
                    if ox * oy / box_area >= 0.5:
                        to_remove.add(i)

                new_formula_boxes.append(
                    {
                        "text": f"$$ {latex} $$",
                        "score": 1.0,
                        "x0": fl_x0,
                        "x1": fl_x1,
                        "top": global_top,
                        "bottom": global_bot,
                        "page_number": pn + 1 + page_from,
                        "layout_type": "equation",
                        "layoutno": f"equation-{pn}-{formula_seq}",
                        "source_page_index": pn,
                        "source_bbox_local": [
                            float(fl["x0"]),
                            float(fl["top"]),
                            float(fl["x1"]),
                            float(fl["bottom"]),
                        ],
                    }
                )
                formula_seq += 1

        if not new_formula_boxes:
            return

        self.boxes = [b for i, b in enumerate(self.boxes) if i not in to_remove]
        self.boxes.extend(new_formula_boxes)
        self.boxes.sort(
            key=lambda x: (x.get("page_number", 0), x.get("col_id", 0), x.get("top", 0))
        )
        logger.info(
            "_recognize_formulas: replaced %d overlapping boxes with %d formula boxes",
            len(to_remove), len(new_formula_boxes),
        )

    def _get_rapid_table(self):
        if self._rapid_table is None:
            from deepdoc.vision.rapid_table_recognizer import RapidTableRecognizer

            self._rapid_table = RapidTableRecognizer()
        return self._rapid_table

    def _table_html_rapid(self, img, bxs, poss, ZM):
        """rapidtable 引擎的表格 HTML；任何失败都回退 TATR（最坏 = 现状）。"""
        if getattr(self, "_rapid_unavailable", False):
            return self.tbl_det.construct_table(
                bxs, html=True, is_english=self.is_english
            )
        try:
            pages = {b.get("page_number") for b in bxs}
            if len(pages) <= 1 and len(poss) <= 1:
                # 单页表：box 用页面局部坐标，crop_origin 取裁剪框左上角。
                left = poss[-1][1] if poss else float(np.min([b["x0"] for b in bxs]))
                top_local = poss[-1][3] if poss else 0.0
                ocr_boxes, origin, zm = bxs, (left, top_local), ZM
            else:
                # 跨页表：cropout 已把各页裁图按页序垂直拼接为 img；
                # 把 bxs 映射成拼接图的局部像素坐标后直接喂入。
                ocr_boxes = self._rapid_cross_page_local_boxes(bxs, poss, ZM)
                origin, zm = (0, 0), 1
            html = self._get_rapid_table()(
                img, ocr_boxes, crop_origin=origin, zoomin=zm
            )
            if html and "<table" in html:
                return html
            logger.warning("rapidtable empty/invalid output, fallback to TATR")
        except (ImportError, FileNotFoundError) as e:
            # 一次性失败（未装依赖/模型缺失）：标记不可用，避免逐表重试。
            logger.warning(
                "rapidtable unavailable (%s), fallback to TATR for all tables", e
            )
            self._rapid_unavailable = True
        except Exception as e:  # noqa: BLE001
            logger.warning("rapidtable failed (%s), fallback to TATR", e)
        return self.tbl_det.construct_table(
            bxs, html=True, is_english=self.is_english
        )

    def _rapid_cross_page_local_boxes(self, bxs, poss, ZM):
        """把跨页表的 box 映射到 cropout 拼接图的局部像素坐标。

        ⚠️ 坐标系假设（未验证，需按 plan Task 5 Step 1 可视化校验后修正）：
          - poss 各项 = (pn_abs, left, right, top_local, bott_local)，按拼接页序排列；
            top_local/bott_local 已是该页内局部坐标（cropout 已减 page_cum_height）。
          - box.top/bottom 为页面累积坐标；该页 local 索引 = page_number - page_from，
            页面局部 = 累积 - page_cum_height[local_idx]（与 _ocr_rotated_tables 一致）。
          - 拼接图各页 y 起点 = 之前各页裁图像素高度累加 Σ (bott_local-top_local)*ZM。
        """
        y_off, left_of, top_of = {}, {}, {}
        acc = 0.0
        for item in poss:
            pn_abs, left, _right, top_l, bott_l = item
            y_off[pn_abs] = acc
            left_of[pn_abs] = left
            top_of[pn_abs] = top_l
            acc += max(0.0, (bott_l - top_l)) * ZM

        raw_cum = getattr(self, "page_cum_height", None)
        cum = list(raw_cum) if raw_cum is not None else []
        local = []
        for b in bxs:
            pn_abs = b.get("page_number")
            if pn_abs not in y_off:
                continue
            li = pn_abs - self.page_from
            base = cum[li] if 0 <= li < len(cum) else 0.0
            b_top_l = b["top"] - base
            b_bot_l = b["bottom"] - base
            local.append(
                {
                    "text": b.get("text", ""),
                    "x0": (b["x0"] - left_of[pn_abs]) * ZM,
                    "x1": (b["x1"] - left_of[pn_abs]) * ZM,
                    "top": y_off[pn_abs] + (b_top_l - top_of[pn_abs]) * ZM,
                    "bottom": y_off[pn_abs] + (b_bot_l - top_of[pn_abs]) * ZM,
                    "score": b.get("score", 1.0),
                }
            )
        return local

    @staticmethod
    def _table_bounds(bxs):
        return {
            "x0": min(float(b["x0"]) for b in bxs),
            "x1": max(float(b["x1"]) for b in bxs),
            "top": min(float(b["top"]) for b in bxs),
            "bottom": max(float(b["bottom"]) for b in bxs),
        }

    @staticmethod
    def _table_first_line_tokens(bxs) -> list[str]:
        if not bxs:
            return []
        rows = Recognizer.sort_Y_firstly(
            bxs,
            np.mean([max(1.0, float(b["bottom"]) - float(b["top"])) for b in bxs]) / 2,
        )
        first = rows[0]
        first_top = float(first["top"])
        first_height = max(1.0, float(first["bottom"]) - float(first["top"]))
        line = [
            b
            for b in rows
            if abs(float(b["top"]) - first_top) <= first_height * 0.8
        ]
        line = sorted(line, key=lambda b: (float(b["x0"]), float(b["top"])))
        text = " ".join(str(b.get("text") or "") for b in line)
        text = re.sub(r"[\t\r\n|]+", " ", text)
        return [token for token in re.split(r"\s+", text.strip()) if token]

    @classmethod
    def _cross_page_table_structure_compatible(cls, previous_bxs, next_bxs) -> bool:
        previous_bounds = cls._table_bounds(previous_bxs)
        next_bounds = cls._table_bounds(next_bxs)
        overlap = cls._horizontal_overlap_ratio(previous_bounds, next_bounds)
        if overlap < 0.75:
            return False

        previous_width = max(1.0, previous_bounds["x1"] - previous_bounds["x0"])
        next_width = max(1.0, next_bounds["x1"] - next_bounds["x0"])
        width_delta = abs(previous_width - next_width) / max(previous_width, next_width)
        if width_delta > 0.25:
            return False

        previous_tokens = cls._table_first_line_tokens(previous_bxs)
        next_tokens = cls._table_first_line_tokens(next_bxs)
        if len(previous_tokens) > 1 and len(next_tokens) > 1 and len(previous_tokens) != len(next_tokens):
            return False
        return True

    def _extract_table_figure(
        self, need_image, ZM, return_html, need_position, separate_tables_figures=False
    ):
        self._last_cross_page_table_merge_count = 0
        self._last_cross_page_table_merge_groups = []
        tables = {}
        figures = {}
        # extract figure and table boxes
        i = 0
        lst_lout_no = ""
        nomerge_lout_no = []
        while i < len(self.boxes):
            if "layoutno" not in self.boxes[i]:
                i += 1
                continue
            lout_no = (
                str(self.boxes[i]["page_number"]) + "-" + str(self.boxes[i]["layoutno"])
            )
            if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i][
                "layout_type"
            ] in ["table caption", "title", "figure caption", "reference"]:
                nomerge_lout_no.append(lst_lout_no)
            if self.boxes[i]["layout_type"] == "table":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in tables:
                    tables[lout_no] = []
                tables[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            if need_image and self.boxes[i]["layout_type"] == "figure":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in figures:
                    figures[lout_no] = []
                figures[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            i += 1

        # merge table on different pages
        nomerge_lout_no = set(nomerge_lout_no)
        tbls = sorted(
            [(k, bxs) for k, bxs in tables.items()],
            key=lambda x: (x[1][0]["top"], x[1][0]["x0"]),
        )

        i = len(tbls) - 1
        while i - 1 >= 0:
            k0, bxs0 = tbls[i - 1]
            k, bxs = tbls[i]
            i -= 1
            if k0 in nomerge_lout_no:
                continue
            if bxs[0]["page_number"] == bxs0[0]["page_number"]:
                continue
            if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
                continue
            mh = self.mean_height[bxs[0]["page_number"] - 1]
            if self._y_dis(bxs0[-1], bxs[0]) > mh * 23:
                continue
            if not self._cross_page_table_structure_compatible(bxs0, bxs):
                continue
            merged_pages = sorted(
                set(
                    [int(b["page_number"]) for b in tables[k0]]
                    + [int(b["page_number"]) for b in tables[k]]
                )
            )
            group_id = f"table-cross-page-{len(self._last_cross_page_table_merge_groups) + 1}"
            tables[k0].extend(tables[k])
            for box in tables[k0]:
                box["merged_page_numbers"] = merged_pages
                box["merge_reason"] = "table_cross_page_continuation"
                box["cross_page_table_group"] = group_id
            del tables[k]
            self._last_cross_page_table_merge_groups.append(
                {
                    "group_id": group_id,
                    "layout_key": k0,
                    "merged_page_numbers": merged_pages,
                    "source_table_keys": [k0, k],
                }
            )
            self._last_cross_page_table_merge_count += len(merged_pages)

        def x_overlapped(a, b):
            return not any([a["x1"] < b["x0"], a["x0"] > b["x1"]])

        # find captions and pop out
        i = 0
        while i < len(self.boxes):
            c = self.boxes[i]
            # mh = self.mean_height[c["page_number"]-1]
            if not TableStructureRecognizer.is_caption(c):
                i += 1
                continue

            # find the nearest layouts
            def nearest(tbls):
                nonlocal c
                mink = ""
                minv = 1000000000
                for k, bxs in tbls.items():
                    for b in bxs:
                        if b.get("layout_type", "").find("caption") >= 0:
                            continue
                        y_dis = self._y_dis(c, b)
                        x_dis = self._x_dis(c, b) if not x_overlapped(c, b) else 0
                        dis = y_dis * y_dis + x_dis * x_dis
                        if dis < minv:
                            mink = k
                            minv = dis
                return mink, minv

            tk, tv = nearest(tables)
            fk, fv = nearest(figures)
            # if min(tv, fv) > 2000:
            #    i += 1
            #    continue
            if tv < fv and tk:
                tables[tk].insert(0, c)
                logger.debug("TABLE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            elif fk:
                figures[fk].insert(0, c)
                logger.debug("FIGURE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            self.boxes.pop(i)

        def cropout(bxs, ltype, poss):
            nonlocal ZM
            max_page_index = len(self.page_images) - 1

            def local_page_index(page_number):
                idx = page_number - 1 if page_number > 0 else 0
                if idx > max_page_index and self.page_from:
                    idx = page_number - 1 - self.page_from
                return idx

            pn = set()
            for b in bxs:
                idx = local_page_index(b["page_number"])
                if 0 <= idx <= max_page_index:
                    pn.add(idx)
                else:
                    logger.warning(
                        "Skip out-of-range page_number %s (page_from=%s, pages=%s)",
                        b.get("page_number"),
                        self.page_from,
                        len(self.page_images),
                    )

            if not pn:
                return None

            if len(pn) < 2:
                pn = list(pn)[0]
                ht = self.page_cum_height[pn]
                b = {
                    "x0": np.min([b["x0"] for b in bxs]),
                    "top": np.min([b["top"] for b in bxs]) - ht,
                    "x1": np.max([b["x1"] for b in bxs]),
                    "bottom": np.max([b["bottom"] for b in bxs]) - ht,
                }
                louts = [
                    layout for layout in self.page_layout[pn] if layout["type"] == ltype
                ]
                ii = Recognizer.find_overlapped(b, louts, naive=True)
                if ii is not None:
                    b = louts[ii]
                else:
                    logger.warning(
                        f"Missing layout match: {pn + 1},%s"
                        % (bxs[0].get("layoutno", ""))
                    )

                left, top, right, bott = b["x0"], b["top"], b["x1"], b["bottom"]
                if right < left:
                    right = left + 1
                poss.append((pn + self.page_from, left, right, top, bott))
                return self.page_images[pn].crop(
                    (left * ZM, top * ZM, right * ZM, bott * ZM)
                )
            pn = {}
            for b in bxs:
                p = local_page_index(b["page_number"])
                if 0 <= p <= max_page_index:
                    if p not in pn:
                        pn[p] = []
                    pn[p].append(b)
            pn = sorted(pn.items(), key=lambda x: x[0])
            imgs = [cropout(arr, ltype, poss) for p, arr in pn]
            imgs = [img for img in imgs if img is not None]
            if not imgs:
                return None
            pic = Image.new(
                "RGB",
                (
                    int(np.max([i.size[0] for i in imgs])),
                    int(np.sum([m.size[1] for m in imgs])),
                ),
                (245, 245, 245),
            )
            height = 0
            for img in imgs:
                pic.paste(img, (0, int(height)))
                height += img.size[1]
            return pic

        res = []
        positions = []
        figure_results = []
        figure_positions = []
        # crop figure out and add caption
        for k, bxs in figures.items():
            txt = "\n".join([b["text"] for b in bxs])
            if not txt:
                continue

            poss = []

            if separate_tables_figures:
                img = cropout(bxs, "figure", poss)
                if img is None:
                    continue
                figure_results.append((img, [txt]))
                figure_positions.append(poss)
            else:
                img = cropout(bxs, "figure", poss)
                if img is None:
                    continue
                res.append((img, [txt]))
                positions.append(poss)

        for k, bxs in tables.items():
            if not bxs:
                continue
            bxs = Recognizer.sort_Y_firstly(
                bxs, np.mean([(b["bottom"] - b["top"]) / 2 for b in bxs])
            )

            poss = []

            img = cropout(bxs, "table", poss)
            if img is None:
                continue
            if self.table_engine == "rapidtable" and return_html:
                table_html = self._table_html_rapid(img, bxs, poss, ZM)
            else:
                table_html = self.tbl_det.construct_table(
                    bxs, html=return_html, is_english=self.is_english
                )
            res.append((img, table_html))
            positions.append(poss)

        if separate_tables_figures:
            assert len(positions) + len(figure_positions) == len(res) + len(
                figure_results
            )
            if need_position:
                return list(zip(res, positions)), list(
                    zip(figure_results, figure_positions)
                )
            else:
                return res, figure_results
        else:
            assert len(positions) == len(res)
            if need_position:
                return list(zip(res, positions))
            else:
                return res

    def proj_match(self, line):
        if len(line) <= 2:
            return
        if re.match(r"[0-9 ().,%%+/-]+$", line):
            return False
        for p, j in [
            (r"第[零一二三四五六七八九十百]+章", 1),
            (r"第[零一二三四五六七八九十百]+[条节]", 2),
            (r"[零一二三四五六七八九十百]+[、 　]", 3),
            (r"[\(（][零一二三四五六七八九十百]+[）\)]", 4),
            (r"[0-9]+(、|\.[　 ]|\.[^0-9])", 5),
            (r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])", 6),
            (r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 7),
            (r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 8),
            (r".{,48}[：:?？]$", 9),
            (r"[0-9]+）", 10),
            (r"[\(（][0-9]+[）\)]", 11),
            (r"[零一二三四五六七八九十百]+是", 12),
            (r"[⚫•➢✓]", 12),
        ]:
            if re.match(p, line):
                return j
        return

    def _line_tag(self, bx, ZM):
        pn = [bx["page_number"]]
        top = bx["top"] - self.page_cum_height[pn[0] - 1]
        bott = bx["bottom"] - self.page_cum_height[pn[0] - 1]
        page_images_cnt = len(self.page_images)
        if pn[-1] - 1 >= page_images_cnt:
            return ""
        while bott * ZM > self.page_images[pn[-1] - 1].size[1]:
            bott -= self.page_images[pn[-1] - 1].size[1] / ZM
            pn.append(pn[-1] + 1)
            if pn[-1] - 1 >= page_images_cnt:
                return ""

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format(
            "-".join([str(p) for p in pn]), bx["x0"], bx["x1"], top, bott
        )

    def __filterout_scraps(self, boxes, ZM):
        def width(b):
            return b["x1"] - b["x0"]

        def height(b):
            return b["bottom"] - b["top"]

        def usefull(b):
            if b.get("layout_type"):
                return True
            if width(b) > self.page_images[b["page_number"] - 1].size[0] / ZM / 3:
                return True
            if b["bottom"] - b["top"] > self.mean_height[b["page_number"] - 1]:
                return True
            return False

        res = []
        while boxes:
            lines = []
            widths = []
            pw = self.page_images[boxes[0]["page_number"] - 1].size[0] / ZM
            mh = self.mean_height[boxes[0]["page_number"] - 1]
            mj = (
                self.proj_match(boxes[0]["text"])
                or boxes[0].get("layout_type", "") == "title"
            )

            def dfs(line, st):
                nonlocal mh, pw, lines, widths
                lines.append(line)
                widths.append(width(line))
                mmj = (
                    self.proj_match(line["text"])
                    or line.get("layout_type", "") == "title"
                )
                for i in range(st + 1, min(st + 20, len(boxes))):
                    if (boxes[i]["page_number"] - line["page_number"]) > 0:
                        break
                    if (
                        not mmj
                        and self._y_dis(line, boxes[i]) >= 3 * mh
                        and height(line) < 1.5 * mh
                    ):
                        break

                    if not usefull(boxes[i]):
                        continue
                    if mmj or (self._x_dis(boxes[i], line) < pw / 10):
                        # and abs(width(boxes[i])-width_mean)/max(width(boxes[i]),width_mean)<0.5):
                        # concat following
                        dfs(boxes[i], i)
                        boxes.pop(i)
                        break

            try:
                if usefull(boxes[0]):
                    dfs(boxes[0], 0)
                else:
                    logger.debug("WASTE: " + boxes[0]["text"])
            except Exception:
                pass
            boxes.pop(0)
            mw = np.mean(widths)
            if mj or mw / pw >= 0.35 or mw > 200:
                res.append(
                    "\n".join([c["text"] + self._line_tag(c, ZM) for c in lines])
                )
            else:
                logger.debug("REMOVED: " + "<<".join([c["text"] for c in lines]))

        return "\n\n".join(res)

    @staticmethod
    def total_page_number(fnm, binary=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                pdf = (
                    pdfplumber.open(fnm)
                    if not binary
                    else pdfplumber.open(BytesIO(binary))
                )
            total_page = len(pdf.pages)
            pdf.close()
            return total_page
        except Exception:
            logger.exception("total_page_number")

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        self.prepare_pages(fnm, zoomin=zoomin, page_from=page_from, page_to=page_to)
        self.run_page_ocr(zoomin=zoomin, callback=callback)
        self.finalize_page_boxes()
        if not any(self.boxes) and zoomin < 9:
            self.__images__(fnm, zoomin * 3, page_from, page_to, callback)

    def _reset_page_state(self, *, page_from: int = 0):
        self.lefted_chars = []
        self.mean_height = []
        self.mean_width = []
        self.boxes = []
        self.garbages = {}
        self.page_cum_height = [0.0]
        self.page_layout = []
        self.page_from = page_from
        self.page_images = []
        self.page_chars = []
        self.outlines = []
        self.total_page = 0
        self.is_english = False
        self.complex_block_only_pages = set()

    def _absolute_page_number(self, page_index: int) -> int:
        return int(page_index) + int(getattr(self, "page_from", 0) or 0) + 1

    def _table_boxes_in_region(self, page_index: int, table_x0, table_x1, table_top_cum, table_bottom_cum):
        page_number = self._absolute_page_number(page_index)
        return [
            b
            for b in self.boxes
            if (
                int(b.get("page_number") or 0) == page_number
                and b.get("layout_type") == "table"
                and float(b.get("x0") or 0.0) >= table_x0 - 5
                and float(b.get("x1") or 0.0) <= table_x1 + 5
                and float(b.get("top") or 0.0) >= table_top_cum - 5
                and float(b.get("bottom") or 0.0) <= table_bottom_cum + 5
            )
        ]

    def _ocr_selective_table_regions(self, ZM, table_layouts, page_numbers: set[int] | None = None):
        selected_pages = {int(page_number) for page_number in (page_numbers or set()) if int(page_number) > 0}
        if not selected_pages:
            return 0

        added_total = 0
        for tbl_info in table_layouts:
            table_index = int(tbl_info["table_index"])
            page_index = int(tbl_info["page"])
            page_number = self._absolute_page_number(page_index)
            if page_number not in selected_pages:
                continue

            rotation_info = self.table_rotations.get(table_index, {})
            if int(rotation_info.get("best_angle") or 0) != 0:
                continue

            layout = tbl_info["layout"]
            left, top, right, bott = tbl_info["coords"]
            table_x0 = float(layout["x0"])
            table_top = float(layout["top"])
            table_x1 = float(layout["x1"])
            table_top_cum = table_top + float(self.page_cum_height[page_index])
            table_bottom_cum = float(layout["bottom"]) + float(self.page_cum_height[page_index])
            if self._table_boxes_in_region(page_index, table_x0, table_x1, table_top_cum, table_bottom_cum):
                continue

            rotated_img = self.rotated_table_imgs.get(table_index)
            if rotated_img is None:
                continue

            ocr_results = self.ocr(np.array(rotated_img))
            if not ocr_results:
                continue

            insert_at = len(self.boxes)
            for index, box in enumerate(self.boxes):
                if int(box.get("page_number") or 0) == page_number:
                    insert_at = index + 1

            table_w_px = right - left
            table_h_px = bott - top
            for bbox, (text, conf) in ocr_results:
                if conf < 0.5:
                    continue
                x_coords = [float(point[0]) for point in bbox]
                y_coords = [float(point[1]) for point in bbox]
                self.boxes.insert(
                    insert_at,
                    {
                        "text": text,
                        "x0": min(x_coords) / ZM + table_x0,
                        "x1": max(x_coords) / ZM + table_x0,
                        "top": min(y_coords) / ZM + table_top + float(self.page_cum_height[page_index]),
                        "bottom": max(y_coords) / ZM + table_top + float(self.page_cum_height[page_index]),
                        "page_number": page_number,
                        "layout_type": "table",
                        "layoutno": f"table-{table_index}",
                        "_selective_complex_ocr": True,
                        "_table_index": table_index,
                        "_rotated": False,
                        "_rotation_angle": 0,
                        "_table_width_px": table_w_px,
                        "_table_height_px": table_h_px,
                    },
                )
                insert_at += 1
                added_total += 1

        return added_total

    def _load_page_artifacts(
        self,
        fnm,
        *,
        zoomin=3,
        page_from=0,
        page_to=299,
        char_page_numbers: set[int] | None = None,
        image_page_numbers: set[int] | None = None,
    ):
        start = timer()
        selected_char_pages = None
        if char_page_numbers is not None:
            selected_char_pages = {
                int(page_number) for page_number in char_page_numbers if int(page_number) > 0
            }
        selected_image_pages = None
        if image_page_numbers is not None:
            selected_image_pages = {
                int(page_number) for page_number in image_page_numbers if int(page_number) > 0
            }

        if selected_char_pages is not None and not selected_char_pages:
            try:
                with _open_fitz_source(fnm) as doc:
                    start_page = max(0, int(page_from))
                    end_page = min(max(start_page, int(page_to)), len(doc))
                    matrix = fitz.Matrix(float(zoomin), float(zoomin))
                    self.page_images = []
                    self.page_chars = []
                    for page_index in range(start_page, end_page):
                        page = doc.load_page(page_index)
                        absolute_page_number = page_index + 1
                        if selected_image_pages is None or absolute_page_number in selected_image_pages:
                            pix = page.get_pixmap(matrix=matrix, alpha=False)
                            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            self.page_images.append(image)
                        else:
                            self.page_images.append(
                                _PageImagePlaceholder(
                                    round(float(page.rect.width or 0.0) * zoomin),
                                    round(float(page.rect.height or 0.0) * zoomin),
                                )
                            )
                        self.page_chars.append([])
                    self.total_page = len(doc)
                    logger.info(f"__images__ dedupe_chars cost {timer() - start}s")
                    return
            except Exception:
                logger.exception("DeepDocPdfParser fitz fast render fallback failed; retrying with pdfplumber")
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                with (
                    pdfplumber.open(fnm)
                    if isinstance(fnm, (str, os.PathLike))
                    else pdfplumber.open(BytesIO(fnm))
                ) as pdf:
                    self.pdf = pdf
                    pages = list(self.pdf.pages[page_from:page_to])
                    self.page_images = []
                    for local_index, page in enumerate(pages):
                        absolute_page_number = int(page_from) + local_index + 1
                        if selected_image_pages is None or absolute_page_number in selected_image_pages:
                            self.page_images.append(
                                page.to_image(resolution=72 * zoomin, antialias=True).annotated
                            )
                        else:
                            self.page_images.append(
                                _PageImagePlaceholder(
                                    round(float(getattr(page, "width", 0.0) or 0.0) * zoomin),
                                    round(float(getattr(page, "height", 0.0) or 0.0) * zoomin),
                                )
                            )
                    try:
                        self.page_chars = []
                        for local_index, page in enumerate(pages):
                            absolute_page_number = int(page_from) + local_index + 1
                            if selected_char_pages is not None and absolute_page_number not in selected_char_pages:
                                self.page_chars.append([])
                                continue
                            self.page_chars.append([c for c in _dedupe_pdf_page(page).chars if self._has_color(c)])
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract characters for pages {page_from}-{page_to}: {str(e)}"
                        )
                        self.page_chars = [[] for _ in pages]
                    self.total_page = len(self.pdf.pages)
        except Exception as e:
            logger.exception(f"DeepDocPdfParser __images__, exception: {e}")
        logger.info(f"__images__ dedupe_chars cost {timer() - start}s")

    def _load_pdf_outlines(self, fnm):
        self.outlines = []
        try:
            with pdf2_read(fnm if isinstance(fnm, (str, os.PathLike)) else BytesIO(fnm)) as pdf:
                self.pdf = pdf
                outlines = self.pdf.outline

                def dfs(arr, depth):
                    for a in arr:
                        if isinstance(a, dict):
                            self.outlines.append((a["/Title"], depth))
                            continue
                        dfs(a, depth + 1)

                dfs(outlines, 0)
        except Exception as e:
            logger.warning(f"Outlines exception: {e}")

        if not self.outlines:
            logger.warning("Miss outlines")

    def _derive_page_language_hint(self):
        logger.debug("Images converted.")
        english_flags = [
            re.search(
                r"[ a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}",
                "".join(
                    random.choices(
                        [c["text"] for c in self.page_chars[i]],
                        k=min(100, len(self.page_chars[i])),
                    )
                ),
            )
            for i in range(len(self.page_chars))
        ]
        self.is_english = sum([1 if e else 0 for e in english_flags]) > len(self.page_images) / 2

    def prepare_pages(
        self,
        fnm,
        zoomin=3,
        page_from=0,
        page_to=299,
        char_page_numbers: set[int] | None = None,
        image_page_numbers: set[int] | None = None,
        load_outlines: bool = True,
    ):
        self._reset_page_state(page_from=page_from)
        self._load_page_artifacts(
            fnm,
            zoomin=zoomin,
            page_from=page_from,
            page_to=page_to,
            char_page_numbers=char_page_numbers,
            image_page_numbers=image_page_numbers,
        )
        if load_outlines:
            self._load_pdf_outlines(fnm)
        if any(self.page_chars):
            self._derive_page_language_hint()
        else:
            self.is_english = False
        self.boxes = [[] for _ in self.page_images]
        for i, img in enumerate(self.page_images):
            chars = self.page_chars[i] if not self.is_english else []
            self.mean_height.append(
                np.median(sorted([c["height"] for c in chars])) if chars else 0
            )
            self.mean_width.append(
                np.median(sorted([c["width"] for c in chars])) if chars else 8
            )
            self.page_cum_height.append(img.size[1] / zoomin)
        return self

    def seed_page_boxes(self, boxes_by_page: dict[int, list[dict[str, Any]]]):
        if not getattr(self, "boxes", None):
            self.boxes = [[] for _ in getattr(self, "page_images", [])]
        for page_number, page_boxes in (boxes_by_page or {}).items():
            page_index = int(page_number) - int(getattr(self, "page_from", 0) or 0) - 1
            if page_index < 0 or page_index >= len(self.boxes):
                continue
            normalized_boxes = [dict(box) for box in page_boxes if isinstance(box, dict)]
            self.boxes[page_index] = normalized_boxes
            if not normalized_boxes:
                continue
            heights = [
                float(box.get("bottom") or 0.0) - float(box.get("top") or 0.0)
                for box in normalized_boxes
                if float(box.get("bottom") or 0.0) > float(box.get("top") or 0.0)
            ]
            widths = [
                float(box.get("x1") or 0.0) - float(box.get("x0") or 0.0)
                for box in normalized_boxes
                if float(box.get("x1") or 0.0) > float(box.get("x0") or 0.0)
            ]
            if page_index < len(self.mean_height) and self.mean_height[page_index] == 0 and heights:
                self.mean_height[page_index] = float(np.median(heights))
            if page_index < len(self.mean_width) and self.mean_width[page_index] == 8 and widths:
                self.mean_width[page_index] = float(np.median(widths))
        return self

    def seed_page_layouts(self, layouts_by_page: dict[int, list[dict[str, Any]]]):
        if not getattr(self, "page_layout", None):
            self.page_layout = [[] for _ in getattr(self, "page_images", [])]
        for page_number, page_layout in (layouts_by_page or {}).items():
            page_index = int(page_number) - int(getattr(self, "page_from", 0) or 0) - 1
            if page_index < 0 or page_index >= len(self.page_layout):
                continue
            self.page_layout[page_index] = [dict(layout) for layout in page_layout if isinstance(layout, dict)]
        return self

    def run_page_ocr(self, page_numbers: set[int] | None = None, *, zoomin=3, callback=None):
        if page_numbers is None:
            selected_pages = sorted(
                {
                    int(getattr(self, "page_from", 0) or 0) + i + 1
                    for i in range(len(getattr(self, "page_images", []) or []))
                }
            )
        else:
            selected_pages = sorted(page_numbers)

        parallel_limiter_config = getattr(self, "parallel_limiter", _PARALLEL_LIMITER_UNSET)
        if parallel_limiter_config is _PARALLEL_LIMITER_UNSET:
            parallel_device_count = max(1, int(settings.PARALLEL_DEVICES))
        elif isinstance(parallel_limiter_config, int):
            parallel_device_count = max(1, int(parallel_limiter_config))
        elif parallel_limiter_config:
            parallel_device_count = max(1, len(parallel_limiter_config))
        else:
            parallel_device_count = 1

        async def __page_ocr_launcher():
            parallel_limiter = None
            if parallel_device_count > 1:
                parallel_limiter = [asyncio.Semaphore(1) for _ in range(parallel_device_count)]
            if parallel_limiter:
                tasks = []
                for page_number in selected_pages:
                    local_index = page_number - int(getattr(self, "page_from", 0) or 0) - 1
                    if local_index < 0 or local_index >= len(self.page_images):
                        continue
                    img = self.page_images[local_index]
                    chars = self.page_chars[local_index] if not self.is_english else []
                    semaphore = parallel_limiter[local_index % parallel_device_count]

                    async def wrapper(
                        page_number=page_number,
                        local_index=local_index,
                        img=img,
                        chars=chars,
                        semaphore=semaphore,
                    ):
                        async with semaphore:
                            self.boxes[local_index] = await thread_pool_exec(
                                self._ocr_page_boxes,
                                page_number,
                                img,
                                chars,
                                zoomin,
                                local_index % parallel_device_count,
                            )
                        if callback and local_index % 6 == 5:
                            callback((local_index + 1) * 0.6 / max(len(self.page_images), 1))

                    tasks.append(asyncio.create_task(wrapper()))
                    await asyncio.sleep(0)

                try:
                    await asyncio.gather(*tasks, return_exceptions=False)
                except Exception as e:
                    logger.error(f"Error in OCR: {e}")
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                return

            for page_number in selected_pages:
                local_index = page_number - int(getattr(self, "page_from", 0) or 0) - 1
                if local_index < 0 or local_index >= len(self.page_images):
                    continue
                img = self.page_images[local_index]
                chars = self.page_chars[local_index] if not self.is_english else []
                self.boxes[local_index] = self._ocr_page_boxes(page_number, img, chars, zoomin, 0)
                if callback and local_index % 6 == 5:
                    callback((local_index + 1) * 0.6 / max(len(self.page_images), 1))

        start = timer()
        asyncio.run(__page_ocr_launcher())
        logger.info(f"__images__ {len(self.page_images)} pages cost {timer() - start}s")
        return self

    def finalize_page_boxes(self):
        if not self.is_english and not any([c for c in self.page_chars]) and any(self.boxes):
            bxes = [b for bxs in self.boxes for b in bxs]
            self.is_english = re.search(
                r"[ \na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}",
                "".join(
                    [b["text"] for b in random.choices(bxes, k=min(30, len(bxes)))]
                ),
            )

        logger.debug(f"Is it English: {self.is_english}")
        self.page_cum_height = np.cumsum(self.page_cum_height)
        assert len(self.page_cum_height) == len(self.page_images) + 1
        return self

    def __call__(
        self, fnm, need_image=True, zoomin=3, return_html=False, auto_rotate_tables=None
    ):
        """
        Parse a PDF file.

        Args:
            fnm: PDF file path or binary content
            need_image: Whether to extract images
            zoomin: Zoom factor
            return_html: Whether to return tables in HTML format
            auto_rotate_tables: Whether to enable auto orientation correction for tables.
                               None: Use TABLE_AUTO_ROTATE env var setting (default: True)
                               True: Enable auto orientation correction
                               False: Disable auto orientation correction
        """
        if auto_rotate_tables is None:
            auto_rotate_tables = os.getenv("TABLE_AUTO_ROTATE", "true").lower() in (
                "true",
                "1",
                "yes",
            )

        self.__images__(fnm, zoomin)
        self._layouts_rec(zoomin)
        self._table_transformer_job(zoomin, auto_rotate=auto_rotate_tables)
        self._text_merge()
        self._merge_cross_page_text()
        self._concat_downward()
        self._apply_reading_order_strategy(zoomin)
        self._filter_forpages()
        tbls = self._extract_table_figure(need_image, zoomin, return_html, False)
        return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls

    def parse_into_bboxes(self, fnm, callback=None, zoomin=3):
        start = timer()
        self.__images__(fnm, zoomin, callback=callback)
        if callback:
            callback(0.40, "OCR finished ({:.2f}s)".format(timer() - start))

        start = timer()
        self._layouts_rec(zoomin)
        if callback:
            callback(0.63, "Layout analysis ({:.2f}s)".format(timer() - start))

        # Read table auto-rotation setting from environment variable
        auto_rotate_tables = os.getenv("TABLE_AUTO_ROTATE", "true").lower() in (
            "true",
            "1",
            "yes",
        )

        start = timer()
        self._table_transformer_job(zoomin, auto_rotate=auto_rotate_tables)
        if callback:
            callback(0.83, "Table analysis ({:.2f}s)".format(timer() - start))

        start = timer()
        self._text_merge()
        self._merge_cross_page_text()
        self._concat_downward()
        self._naive_vertical_merge(zoomin)
        self._apply_reading_order_strategy(zoomin)
        if callback:
            callback(0.92, "Text merged ({:.2f}s)".format(timer() - start))

        start = timer()
        tbls, figs = self._extract_table_figure(True, zoomin, True, True, True)

        def insert_table_figures(tbls_or_figs, layout_type):
            def min_rectangle_distance(rect1, rect2):
                pn1, left1, right1, top1, bottom1 = rect1
                pn2, left2, right2, top2, bottom2 = rect2
                if (
                    right1 >= left2
                    and right2 >= left1
                    and bottom1 >= top2
                    and bottom2 >= top1
                ):
                    return 0
                if right1 < left2:
                    dx = left2 - right1
                elif right2 < left1:
                    dx = left1 - right2
                else:
                    dx = 0
                if bottom1 < top2:
                    dy = top2 - bottom1
                elif bottom2 < top1:
                    dy = top1 - bottom2
                else:
                    dy = 0
                return math.sqrt(dx * dx + dy * dy)  # + (pn2-pn1)*10000

            for (img, txt), poss in tbls_or_figs:
                bboxes = [
                    (i, (b["page_number"], b["x0"], b["x1"], b["top"], b["bottom"]))
                    for i, b in enumerate(self.boxes)
                ]
                dists = [
                    (
                        min_rectangle_distance(
                            (
                                pn,
                                left,
                                right,
                                top + self.page_cum_height[pn],
                                bott + self.page_cum_height[pn],
                            ),
                            rect,
                        ),
                        i,
                    )
                    for i, rect in bboxes
                    for pn, left, right, top, bott in poss
                ]
                min_i = np.argmin(dists, axis=0)[0]
                min_i, rect = bboxes[dists[min_i][-1]]
                if isinstance(txt, list):
                    txt = "\n".join(txt)
                pn, left, right, top, bott = poss[0]
                if self.boxes[min_i]["bottom"] < top + self.page_cum_height[pn]:
                    min_i += 1
                self.boxes.insert(
                    min_i,
                    {
                        "page_number": pn + 1,
                        "x0": left,
                        "x1": right,
                        "top": top + self.page_cum_height[pn],
                        "bottom": bott + self.page_cum_height[pn],
                        "layout_type": layout_type,
                        "text": txt,
                        "image": img,
                        "positions": [
                            [pn + 1, int(left), int(right), int(top), int(bott)]
                        ],
                    },
                )

        for b in self.boxes:
            b["position_tag"] = self._line_tag(b, zoomin)
            b["image"] = self.crop(b["position_tag"], zoomin)
            b["positions"] = [
                [pos[0][-1] + 1, *pos[1:]]
                for pos in DeepDocPdfParser.extract_positions(b["position_tag"])
            ]

        insert_table_figures(tbls, "table")
        insert_table_figures(figs, "figure")
        if callback:
            callback(1, "Structured ({:.2f}s)".format(timer() - start))
        return deepcopy(self.boxes)

    @staticmethod
    def remove_tag(txt):
        return re.sub(r"@@[\t0-9.-]+?##", "", txt)

    @staticmethod
    def extract_positions(txt):
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = (
                float(left),
                float(right),
                float(top),
                float(bottom),
            )
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def crop(self, text, ZM=3, need_position=False):
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            if need_position:
                return None, None
            return

        if not getattr(self, "page_images", None):
            logger.warning(
                "crop called without page images; skipping image generation."
            )
            if need_position:
                return None, None
            return

        page_count = len(self.page_images)

        filtered_poss = []
        for pns, left, right, top, bottom in poss:
            if not pns:
                logger.warning("Empty page index list in crop; skipping this position.")
                continue
            valid_pns = [p for p in pns if 0 <= p < page_count]
            if not valid_pns:
                logger.warning(
                    f"All page indices {pns} out of range for {page_count} pages; skipping."
                )
                continue
            filtered_poss.append((valid_pns, left, right, top, bottom))

        poss = filtered_poss
        if not poss:
            logger.warning("No valid positions after filtering; skip cropping.")
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        first_page_idx = pos[0][0]
        poss.insert(
            0,
            (
                [first_page_idx],
                pos[1],
                pos[2],
                max(0, pos[3] - 120),
                max(pos[3] - GAP, 0),
            ),
        )
        pos = poss[-1]
        last_page_idx = pos[0][-1]
        if not (0 <= last_page_idx < page_count):
            logger.warning(
                f"Last page index {last_page_idx} out of range for {page_count} pages; skipping crop."
            )
            if need_position:
                return None, None
            return
        last_page_height = self.page_images[last_page_idx].size[1] / ZM
        poss.append(
            (
                [last_page_idx],
                pos[1],
                pos[2],
                min(last_page_height, pos[4] + GAP),
                min(last_page_height, pos[4] + 120),
            )
        )

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            if 0 < ii < len(poss) - 1:
                right = max(left + 10, right)
            else:
                right = left + max_width
            bottom *= ZM
            for pn in pns[1:]:
                if 0 <= pn - 1 < page_count:
                    bottom += self.page_images[pn - 1].size[1]
                else:
                    logger.warning(
                        f"Page index {pn}-1 out of range for {page_count} pages during crop; skipping height accumulation."
                    )

            if not (0 <= pns[0] < page_count):
                logger.warning(
                    f"Base page index {pns[0]} out of range for {page_count} pages during crop; skipping this segment."
                )
                continue

            imgs.append(
                self.page_images[pns[0]].crop(
                    (
                        left * ZM,
                        top * ZM,
                        right * ZM,
                        min(bottom, self.page_images[pns[0]].size[1]),
                    )
                )
            )
            if 0 < ii < len(poss) - 1:
                positions.append(
                    (
                        pns[0] + self.page_from,
                        left,
                        right,
                        top,
                        min(bottom, self.page_images[pns[0]].size[1]) / ZM,
                    )
                )
            bottom -= self.page_images[pns[0]].size[1]
            for pn in pns[1:]:
                if not (0 <= pn < page_count):
                    logger.warning(
                        f"Page index {pn} out of range for {page_count} pages during crop; skipping this page."
                    )
                    continue
                imgs.append(
                    self.page_images[pn].crop(
                        (
                            left * ZM,
                            0,
                            right * ZM,
                            min(bottom, self.page_images[pn].size[1]),
                        )
                    )
                )
                if 0 < ii < len(poss) - 1:
                    positions.append(
                        (
                            pn + self.page_from,
                            left,
                            right,
                            0,
                            min(bottom, self.page_images[pn].size[1]) / ZM,
                        )
                    )
                bottom -= self.page_images[pn].size[1]

        if not imgs:
            if need_position:
                return None, None
            return
        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB", (width, height), (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    def get_position(self, bx, ZM):
        poss = []
        pn = bx["page_number"]
        top = bx["top"] - self.page_cum_height[pn - 1]
        bott = bx["bottom"] - self.page_cum_height[pn - 1]
        poss.append(
            (
                pn,
                bx["x0"],
                bx["x1"],
                top,
                min(bott, self.page_images[pn - 1].size[1] / ZM),
            )
        )
        while bott * ZM > self.page_images[pn - 1].size[1]:
            bott -= self.page_images[pn - 1].size[1] / ZM
            top = 0
            pn += 1
            poss.append(
                (
                    pn,
                    bx["x0"],
                    bx["x1"],
                    top,
                    min(bott, self.page_images[pn - 1].size[1] / ZM),
                )
            )
        return poss


class PlainParser:
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []
        try:
            self.pdf = pdf2_read(
                filename if isinstance(filename, str) else BytesIO(filename)
            )
            total_pages = len(self.pdf.pages)
            start_page = max(0, int(from_page))
            end_page = min(int(to_page), total_pages)
            if start_page < end_page:
                for page in self.pdf.pages[start_page:end_page]:
                    try:
                        text = page.extract_text() or ""
                        lines.extend([t for t in text.split("\n") if t])
                    except Exception:
                        logger.exception("PlainParser extract_text exception")

            try:
                outlines = getattr(self.pdf, "outline", None) or []

                def dfs(arr, depth):
                    if not isinstance(arr, (list, tuple)):
                        return
                    for a in arr:
                        if isinstance(a, dict):
                            title = a.get("/Title")
                            if title:
                                self.outlines.append((title, depth))
                            continue
                        dfs(a, depth + 1)

                dfs(outlines, 0)
            except Exception:
                logger.exception("PlainParser outlines exception")
        except Exception:
            logger.exception("PlainParser exception")
        if not self.outlines:
            logger.warning("Miss outlines")

        return [(line, "") for line in lines], []

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


class VisionParser(DeepDocPdfParser):
    def __init__(self, vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_model = vision_model
        self.outlines = []

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                self.pdf = (
                    pdfplumber.open(fnm)
                    if isinstance(fnm, str)
                    else pdfplumber.open(BytesIO(fnm))
                )
                self.page_images = [
                    p.to_image(resolution=72 * zoomin).annotated
                    for i, p in enumerate(self.pdf.pages[page_from:page_to])
                ]
                self.total_page = len(self.pdf.pages)
        except Exception:
            self.page_images = None
            self.total_page = 0
            logger.exception("VisionParser __images__")

    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        callback = kwargs.get("callback", lambda prog, msg: None)
        zoomin = kwargs.get("zoomin", 3)
        self.__images__(
            fnm=filename,
            zoomin=zoomin,
            page_from=from_page,
            page_to=to_page,
            callback=callback,
        )

        logger.warning("Vision-only parser path is disabled in standalone mode.")
        return [], []


if __name__ == "__main__":
    pass

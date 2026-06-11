import asyncio
import inspect
import tempfile
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import fitz
import main
import numpy as np
from PIL import Image

from deepdoc.parser import pdf_parser


def _positioned_native_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=320, height=220)
    page.insert_text((48, 72), "Alpha", fontsize=12)
    page.insert_text((188, 144), "Beta", fontsize=12)
    payload = doc.tobytes()
    doc.close()
    return payload


def _overlaid_text_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=320, height=220)
    page.insert_text((48, 72), "Alpha", fontsize=12)
    page.insert_text((48, 72), "Alpha", fontsize=12)
    payload = doc.tobytes()
    doc.close()
    return payload


def _text_and_image_pdf_bytes() -> bytes:
    image = Image.new("RGB", (640, 480), "white")
    image_payload = BytesIO()
    image.save(image_payload, format="PNG")

    doc = fitz.open()
    text_page = doc.new_page(width=320, height=220)
    text_page.insert_text((48, 72), "Alpha Beta", fontsize=12)
    text_page.insert_text((48, 102), "Gamma", fontsize=12)

    image_page = doc.new_page(width=320, height=220)
    image_page.insert_image(image_page.rect, stream=image_payload.getvalue())
    payload = doc.tobytes()
    doc.close()
    return payload


def _hybrid_source_pdf_bytes() -> bytes:
    image = Image.new("RGB", (820, 520), "white")
    image_payload = BytesIO()
    image.save(image_payload, format="PNG")

    doc = fitz.open()
    text_page = doc.new_page(width=320, height=220)
    text_page.insert_text((48, 72), "Alpha", fontsize=12)
    text_page.insert_text((188, 144), "Beta", fontsize=12)

    image_page = doc.new_page(width=410, height=260)
    image_page.insert_image(image_page.rect, stream=image_payload.getvalue())
    payload = doc.tobytes()
    doc.close()
    return payload


def _mixed_and_scanned_pdf_bytes() -> bytes:
    inset = Image.new("RGB", (96, 64), "lightgray")
    inset_payload = BytesIO()
    inset.save(inset_payload, format="PNG")

    full_page = Image.new("RGB", (640, 480), "white")
    full_payload = BytesIO()
    full_page.save(full_payload, format="PNG")

    doc = fitz.open()
    mixed_page = doc.new_page(width=320, height=220)
    mixed_page.insert_text((48, 72), "Alpha Beta", fontsize=12)
    mixed_page.insert_text((48, 102), "Gamma", fontsize=12)
    mixed_page.insert_image(fitz.Rect(210, 24, 306, 88), stream=inset_payload.getvalue())

    scanned_page = doc.new_page(width=320, height=220)
    scanned_page.insert_image(scanned_page.rect, stream=full_payload.getvalue())
    payload = doc.tobytes()
    doc.close()
    return payload


def _text_with_small_inline_image_pdf_bytes() -> bytes:
    inset = Image.new("RGB", (24, 24), "lightgray")
    inset_payload = BytesIO()
    inset.save(inset_payload, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=320, height=220)
    page.insert_text((48, 72), "Alpha Beta", fontsize=12)
    page.insert_text((48, 102), "Gamma Delta", fontsize=12)
    page.insert_image(fitz.Rect(270, 20, 294, 44), stream=inset_payload.getvalue())
    payload = doc.tobytes()
    doc.close()
    return payload


def _scanned_only_pdf_bytes(page_count: int = 2) -> bytes:
    full_page = Image.new("RGB", (640, 480), "white")
    full_payload = BytesIO()
    full_page.save(full_payload, format="PNG")

    doc = fitz.open()
    for _ in range(page_count):
        page = doc.new_page(width=320, height=220)
        page.insert_image(page.rect, stream=full_payload.getvalue())
    payload = doc.tobytes()
    doc.close()
    return payload


class PdfHybridRoutingPrimitivesTest(unittest.TestCase):
    def test_collect_page_features_and_native_boxes_supports_fast_word_dedupe(self):
        signature = inspect.signature(pdf_parser.collect_pdf_page_features_and_native_boxes)
        self.assertIn("dedupe_chars", signature.parameters)

        with tempfile.TemporaryDirectory(prefix="deepdoc-fast-word-dedupe-") as temp_dir:
            path = Path(temp_dir) / "overlay.pdf"
            path.write_bytes(_overlaid_text_pdf_bytes())

            pages, boxes, meta = pdf_parser.collect_pdf_page_features_and_native_boxes(
                path,
                page_to=5,
                dedupe_chars=False,
            )

        self.assertEqual(1, meta["page_count"])
        self.assertEqual(1, len(pages))
        self.assertEqual(1, len(boxes))
        self.assertEqual("Alpha", boxes[0]["text"])

    def test_collect_page_features_fast_path_reuses_extracted_words(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-fast-word-reuse-") as temp_dir:
            path = Path(temp_dir) / "native.pdf"
            path.write_bytes(_positioned_native_pdf_bytes())

            original_extract_words = pdf_parser._extract_pdf_page_words
            calls = []

            def counting_extract_words(page, *, return_chars=False):
                calls.append(bool(return_chars))
                return original_extract_words(page, return_chars=return_chars)

            with patch.object(pdf_parser, "_extract_pdf_page_words", side_effect=counting_extract_words):
                pages, boxes, meta = pdf_parser.collect_pdf_page_features_and_native_boxes(
                    path,
                    page_to=5,
                    dedupe_chars=False,
                )

        self.assertEqual(1, meta["page_count"])
        self.assertEqual(1, len(pages))
        self.assertEqual(2, len(boxes))
        self.assertEqual([], calls)

    def test_collect_page_features_fast_path_does_not_require_pdfplumber(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-fast-fitz-") as temp_dir:
            path = Path(temp_dir) / "native.pdf"
            path.write_bytes(_positioned_native_pdf_bytes())

            with patch.object(pdf_parser, "_open_pdfplumber_source", side_effect=AssertionError("pdfplumber disabled")):
                pages, boxes, meta = pdf_parser.collect_pdf_page_features_and_native_boxes(
                    path,
                    page_to=5,
                    dedupe_chars=False,
                )

        self.assertEqual(1, meta["page_count"])
        self.assertEqual(1, len(pages))
        self.assertEqual(["Alpha", "Beta"], [box["text"] for box in boxes[:2]])

    def test_extract_native_pdf_text_supports_positioned_boxes(self):
        signature = inspect.signature(pdf_parser.extract_native_pdf_text)
        self.assertIn("preserve_geometry", signature.parameters)

        with tempfile.TemporaryDirectory(prefix="deepdoc-native-geometry-") as temp_dir:
            path = Path(temp_dir) / "positioned.pdf"
            path.write_bytes(_positioned_native_pdf_bytes())

            boxes, meta = pdf_parser.extract_native_pdf_text(path, page_to=5, preserve_geometry=True)

        self.assertEqual(1, meta["page_count"])
        self.assertEqual(1, meta["total_page_count"])
        self.assertGreaterEqual(len(boxes), 2)
        self.assertEqual(["Alpha", "Beta"], [box["text"] for box in boxes[:2]])
        self.assertEqual("text", boxes[0]["layout_type"])
        self.assertGreater(boxes[0]["x1"], boxes[0]["x0"])
        self.assertGreater(boxes[0]["bottom"], boxes[0]["top"])
        self.assertGreater(boxes[1]["x0"], boxes[0]["x1"])
        self.assertGreater(boxes[1]["top"], boxes[0]["top"])
        self.assertIn("chars", boxes[0])
        self.assertEqual("Alpha", "".join(char["text"] for char in boxes[0]["chars"]))
        self.assertEqual(5, len(boxes[0]["chars"]))
        self.assertTrue(
            {"text", "page_number", "x0", "x1", "top", "bottom"}.issubset(boxes[0]["chars"][0].keys())
        )

    def test_extract_native_pdf_text_geometry_dedupes_overlaid_text_layer(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-native-overlay-") as temp_dir:
            path = Path(temp_dir) / "overlay.pdf"
            path.write_bytes(_overlaid_text_pdf_bytes())

            boxes, meta = pdf_parser.extract_native_pdf_text(path, page_to=5, preserve_geometry=True)

        self.assertEqual(1, meta["page_count"])
        self.assertEqual(1, len(boxes))
        self.assertEqual("Alpha", boxes[0]["text"])
        self.assertEqual("Alpha", "".join(char["text"] for char in boxes[0]["chars"]))
        self.assertEqual(5, len(boxes[0]["chars"]))

    def test_inspect_pdf_pages_reports_text_and_image_features(self):
        self.assertTrue(hasattr(pdf_parser, "inspect_pdf_pages"))

        with tempfile.TemporaryDirectory(prefix="deepdoc-page-inspect-") as temp_dir:
            path = Path(temp_dir) / "mixed.pdf"
            path.write_bytes(_text_and_image_pdf_bytes())

            pages = pdf_parser.inspect_pdf_pages(path, page_to=5)

        self.assertEqual(2, len(pages))
        self.assertEqual(1, pages[0]["page_number"])
        self.assertGreater(pages[0]["native_text_char_count"], 0)
        self.assertGreaterEqual(pages[0]["font_coverage_ratio"], 0.8)
        self.assertEqual(0, pages[0]["image_count"])
        self.assertEqual(3, pages[0]["native_text_box_count"])
        self.assertEqual(2, pages[0]["text_block_count"])
        self.assertEqual(2, pages[1]["page_number"])
        self.assertEqual(0, pages[1]["native_text_char_count"])
        self.assertGreater(pages[1]["image_count"], 0)
        self.assertGreater(pages[1]["image_area_ratio"], 0.9)
        self.assertTrue(pages[1]["has_large_image"])


class FakeSelectiveOcr:
    def __init__(self):
        self.called_pages = []

    def detect(self, img, _device_id=None):
        page_number = 2 if img.shape[1] > 350 else 1
        self.called_pages.append(page_number)
        return [
            (
                np.array([[24, 20], [180, 20], [180, 48], [24, 48]], dtype=np.float32),
                ("scan", 0.99),
            )
        ]

    def get_rotate_crop_image(self, img, _points):
        return img

    def recognize_batch(self, images, _device_id=None):
        return ["OCR text" for _ in images]


class FakeLayouter:
    def __init__(self):
        self.page_box_counts = []

    def __call__(self, page_images, page_boxes, _zoomin, drop=True):
        self.page_box_counts = [len(page) for page in page_boxes]
        flattened = []
        for page in page_boxes:
            for box in page:
                box = dict(box)
                box.setdefault("layout_type", "text")
                flattened.append(box)
        return flattened, [[] for _ in page_images]


class FakeTableRegionOcr:
    def __init__(self):
        self.calls = 0

    def __call__(self, _img):
        self.calls += 1
        return [
            (
                np.array([[10, 8], [74, 8], [74, 26], [10, 26]], dtype=np.float32),
                ("Cell 1", 0.99),
            )
        ]


class RecordingLayoutWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.calls = []

    def __call__(self, page_images, page_boxes, _zoomin, drop=True):
        del page_images, drop
        self.calls.append([int(page[0]["page_number"]) for page in page_boxes if page])
        flattened = []
        page_layout = []
        for page in page_boxes:
            page_layout.append([{"type": "text", "worker_id": self.worker_id}] if page else [])
            for box in page:
                cloned = dict(box)
                cloned["layout_worker_id"] = self.worker_id
                flattened.append(cloned)
        return flattened, page_layout


class RecordingTableWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.calls = []

    def __call__(self, images, thr=0.2):
        del thr
        self.calls.append(len(images))
        return [[{"label": "table", "worker_id": self.worker_id}] for _ in images]


class TrackingHybridParser:
    constructed = 0
    last_instance = None

    def __init__(self):
        type(self).constructed += 1
        type(self).last_instance = self
        self.page_from = 0
        self.page_images = []
        self.page_layout = []
        self.page_cum_height = [0, 220, 440]
        self.total_page = 2
        self.boxes = []
        self.called_pages = []
        self.seeded_pages = []
        self.seeded_layout_pages = []
        self.table_auto_rotate = None
        self.table_need_structure = None
        self.char_page_numbers = None
        self.layout_page_numbers = None
        self.image_page_numbers = None
        self.load_outlines = None

    def prepare_pages(
        self,
        _source,
        zoomin=3,
        page_from=0,
        page_to=299,
        char_page_numbers=None,
        image_page_numbers=None,
        load_outlines=True,
    ):
        del zoomin, page_to
        self.page_from = page_from
        self.char_page_numbers = sorted(int(page_number) for page_number in (char_page_numbers or []))
        self.image_page_numbers = sorted(int(page_number) for page_number in (image_page_numbers or []))
        self.load_outlines = bool(load_outlines)
        self.page_images = [Image.new("RGB", (320, 220), "white"), Image.new("RGB", (320, 220), "white")]
        self.boxes = [[], []]
        self.page_layout = [[], []]
        return self

    def seed_page_boxes(self, boxes_by_page):
        self.seeded_pages = sorted(int(page_number) for page_number in boxes_by_page)
        for page_number, page_boxes in boxes_by_page.items():
            self.boxes[int(page_number) - 1] = [dict(box) for box in page_boxes]
        return self

    def seed_page_layouts(self, layouts_by_page):
        self.seeded_layout_pages = sorted(int(page_number) for page_number in layouts_by_page)
        for page_number, page_layout in layouts_by_page.items():
            self.page_layout[int(page_number) - 1] = [dict(layout) for layout in page_layout]
        return self

    def run_page_ocr(self, page_numbers=None, *, zoomin=3, callback=None):
        del zoomin, callback
        self.called_pages = sorted(page_numbers or [])
        for page_number in self.called_pages:
            self.boxes[page_number - 1] = [
                {
                    "text": f"OCR page {page_number}",
                    "page_number": page_number,
                    "layout_type": "text",
                    "x0": 10,
                    "x1": 100,
                    "top": 10,
                    "bottom": 24,
                }
            ]
        return self

    def finalize_page_boxes(self):
        return self

    def _layouts_rec(self, _zoomin, page_numbers=None):
        self.layout_page_numbers = sorted(int(page_number) for page_number in (page_numbers or []))
        flattened = []
        for page in self.boxes:
            flattened.extend(page)
        self.boxes = flattened
        self.page_layout = [[], []]

    def _table_transformer_job(self, _zoomin, auto_rotate=True, need_table_structure=True):
        self.table_auto_rotate = auto_rotate
        self.table_need_structure = bool(need_table_structure)
        return None

    def _text_merge(self):
        return None

    def _concat_downward(self):
        return None

    def _filter_forpages(self):
        return None

    def _extract_table_figure(self, *args):
        if len(args) >= 5:
            return [], []
        return []


class DeepDocHybridParserRefactorTest(unittest.TestCase):
    def test_assign_column_uses_fast_path_for_hybrid_clean_pages_without_kmeans(self):
        parser = object.__new__(pdf_parser.DeepDocPdfParser)
        parser.page_from = 0
        parser.page_images = [Image.new("RGB", (600, 800), "white")]
        parser.mean_width = [8]
        parser.hybrid_clean_pages = {1}
        parser.complex_block_only_pages = set()

        boxes = [
            {"text": "L1", "page_number": 1, "layout_type": "text", "x0": 48, "x1": 128, "top": 60, "bottom": 78},
            {"text": "L2", "page_number": 1, "layout_type": "text", "x0": 48, "x1": 128, "top": 92, "bottom": 110},
            {"text": "R1", "page_number": 1, "layout_type": "text", "x0": 332, "x1": 412, "top": 60, "bottom": 78},
            {"text": "R2", "page_number": 1, "layout_type": "text", "x0": 332, "x1": 412, "top": 92, "bottom": 110},
        ]

        with patch.object(pdf_parser, "KMeans", side_effect=AssertionError("kmeans should not run for clean pages")):
            assigned = parser._assign_column(boxes, zoomin=1)

        self.assertEqual([0, 0, 1, 1], [box["col_id"] for box in assigned])

    def test_assign_column_uses_fast_path_for_complex_block_only_pages_without_kmeans(self):
        parser = object.__new__(pdf_parser.DeepDocPdfParser)
        parser.page_from = 0
        parser.page_images = [Image.new("RGB", (600, 800), "white")]
        parser.mean_width = [8]
        parser.hybrid_clean_pages = set()
        parser.complex_block_only_pages = {1}

        boxes = [
            {"text": "L1", "page_number": 1, "layout_type": "text", "x0": 48, "x1": 128, "top": 60, "bottom": 78},
            {"text": "L2", "page_number": 1, "layout_type": "text", "x0": 48, "x1": 128, "top": 92, "bottom": 110},
            {"text": "R1", "page_number": 1, "layout_type": "text", "x0": 332, "x1": 412, "top": 60, "bottom": 78},
            {"text": "R2", "page_number": 1, "layout_type": "text", "x0": 332, "x1": 412, "top": 92, "bottom": 110},
        ]

        with patch.object(pdf_parser, "KMeans", side_effect=AssertionError("kmeans should not run for complex-block-only pages")):
            assigned = parser._assign_column(boxes, zoomin=1)

        self.assertEqual([0, 0, 1, 1], [box["col_id"] for box in assigned])

    def test_page_start_height_accepts_numpy_cumulative_heights(self):
        parser = object.__new__(pdf_parser.DeepDocPdfParser)
        parser.page_from = 0
        parser.page_images = [Image.new("RGB", (300, 200), "white"), Image.new("RGB", (300, 240), "white")]
        parser.page_cum_height = np.array([0.0, 200.0, 440.0])

        start, height = parser._page_start_height(2)

        self.assertEqual(200.0, start)
        self.assertEqual(240.0, height)

    def test_load_page_artifacts_skips_char_extraction_for_explicit_empty_selection(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-no-char-pages-") as temp_dir:
            path = Path(temp_dir) / "native.pdf"
            path.write_bytes(_positioned_native_pdf_bytes())

            parser = object.__new__(pdf_parser.DeepDocPdfParser)
            parser._load_page_artifacts(path, zoomin=1, page_from=0, page_to=5, char_page_numbers=set())

        self.assertEqual(1, len(parser.page_chars))
        self.assertEqual([], parser.page_chars[0])

    def test_load_page_artifacts_skips_image_render_for_explicit_empty_selection(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-no-image-pages-") as temp_dir:
            path = Path(temp_dir) / "native.pdf"
            path.write_bytes(_positioned_native_pdf_bytes())

            render_calls = []
            original_to_image = pdf_parser.pdfplumber.page.Page.to_image

            def counting_to_image(page, *args, **kwargs):
                render_calls.append(int(page.page_number))
                return original_to_image(page, *args, **kwargs)

            with patch.object(pdf_parser.pdfplumber.page.Page, "to_image", side_effect=counting_to_image):
                parser = object.__new__(pdf_parser.DeepDocPdfParser)
                parser._load_page_artifacts(
                    path,
                    zoomin=1,
                    page_from=0,
                    page_to=5,
                    char_page_numbers=set(),
                    image_page_numbers=set(),
                )

        self.assertEqual([], render_calls)
        self.assertEqual(1, len(parser.page_images))
        self.assertEqual((320, 220), parser.page_images[0].size)

    def test_load_page_artifacts_uses_fitz_rendering_when_char_extraction_is_skipped(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-fitz-render-") as temp_dir:
            path = Path(temp_dir) / "native.pdf"
            path.write_bytes(_positioned_native_pdf_bytes())

            with patch.object(pdf_parser.pdfplumber, "open", side_effect=AssertionError("pdfplumber disabled")):
                parser = object.__new__(pdf_parser.DeepDocPdfParser)
                parser._load_page_artifacts(
                    path,
                    zoomin=1,
                    page_from=0,
                    page_to=5,
                    char_page_numbers=set(),
                    image_page_numbers={1},
                )

        self.assertEqual(1, len(parser.page_images))
        self.assertEqual((320, 220), parser.page_images[0].size)
        self.assertEqual([[]], parser.page_chars)

    def test_run_page_ocr_allows_explicit_empty_selection(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-empty-ocr-") as temp_dir:
            path = Path(temp_dir) / "hybrid.pdf"
            path.write_bytes(_hybrid_source_pdf_bytes())

            parser = object.__new__(pdf_parser.DeepDocPdfParser)
            parser.ocr = FakeSelectiveOcr()
            parser.layouter = FakeLayouter()
            parser.parallel_limiter = None

            parser.prepare_pages(path, zoomin=1, page_from=0, page_to=5)
            parser.run_page_ocr(page_numbers=set(), zoomin=1)

        self.assertEqual([], parser.ocr.called_pages)
        self.assertEqual([[], []], parser.boxes)

    def test_prepare_seed_selective_ocr_and_layout_pipeline(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-parser-") as temp_dir:
            path = Path(temp_dir) / "hybrid.pdf"
            path.write_bytes(_hybrid_source_pdf_bytes())

            native_boxes, _meta = pdf_parser.extract_native_pdf_text(path, page_to=5, preserve_geometry=True)

            parser = object.__new__(pdf_parser.DeepDocPdfParser)
            parser.ocr = FakeSelectiveOcr()
            parser.layouter = FakeLayouter()
            parser.parallel_limiter = None

            parser.prepare_pages(path, zoomin=1, page_from=0, page_to=5)
            parser.seed_page_boxes(
                {
                    1: [box for box in native_boxes if box["page_number"] == 1],
                }
            )
            parser.run_page_ocr(page_numbers={2})
            parser.finalize_page_boxes()
            parser._layouts_rec(1)

        self.assertEqual([2], parser.ocr.called_pages)
        self.assertEqual([2, 1], parser.layouter.page_box_counts)
        self.assertEqual(2, len(parser.page_layout))
        self.assertEqual(["Alpha", "Beta", "OCR text"], [box["text"] for box in parser.boxes])
        self.assertEqual([0.0, 220.0, 480.0], [float(value) for value in parser.page_cum_height])

    def test_run_page_ocr_reuses_parser_across_event_loops_without_loop_binding_failure(self):
        parser = object.__new__(pdf_parser.DeepDocPdfParser)
        parser.parallel_limiter = [asyncio.Semaphore(1), asyncio.Semaphore(1)]
        parser.page_from = 0
        parser.page_images = [Image.new("RGB", (32, 32), "white") for _ in range(4)]
        parser.page_chars = [[], [], [], []]
        parser.boxes = [[], [], [], []]
        parser.is_english = False

        def fake_ocr_page_boxes(page_number, _img, _chars, _zoomin, _device_id=None):
            time.sleep(0.02)
            return [
                {
                    "text": f"OCR page {page_number}",
                    "page_number": page_number,
                    "layout_type": "text",
                    "x0": 10,
                    "x1": 80,
                    "top": 10,
                    "bottom": 24,
                }
            ]

        parser._ocr_page_boxes = fake_ocr_page_boxes

        with patch.object(pdf_parser.settings, "PARALLEL_DEVICES", 2):
            parser.run_page_ocr(page_numbers={1, 2, 3, 4}, zoomin=1)
            parser.run_page_ocr(page_numbers={1, 2, 3, 4}, zoomin=1)

        self.assertEqual(["OCR page 1"], [box["text"] for box in parser.boxes[0]])
        self.assertEqual(["OCR page 4"], [box["text"] for box in parser.boxes[3]])

    def test_selective_table_region_ocr_inserts_table_boxes_without_full_page_ocr(self):
        parser = object.__new__(pdf_parser.DeepDocPdfParser)
        parser.ocr = FakeTableRegionOcr()
        parser.page_from = 0
        parser.boxes = [
            {
                "text": "Alpha",
                "page_number": 1,
                "layout_type": "text",
                "x0": 24,
                "x1": 80,
                "top": 24,
                "bottom": 42,
            }
        ]
        parser.page_cum_height = np.array([0.0, 220.0])
        parser.table_rotations = {
            0: {
                "page": 0,
                "original_pos": (100, 50, 180, 100),
                "best_angle": 0,
                "scores": {},
                "rotated_size": (80, 50),
            }
        }
        parser.rotated_table_imgs = {0: Image.new("RGB", (80, 50), "white")}

        table_layouts = [
            {
                "page": 0,
                "table_index": 0,
                "layout": {"x0": 100, "top": 50, "x1": 180, "bottom": 100},
                "coords": (100, 50, 180, 100),
            }
        ]

        added = parser._ocr_selective_table_regions(1, table_layouts, {1})

        self.assertEqual(1, added)
        self.assertEqual(1, parser.ocr.calls)
        self.assertTrue(any(box["layout_type"] == "table" for box in parser.boxes))
        self.assertTrue(any(box["text"] == "Cell 1" for box in parser.boxes))
        self.assertTrue(all(int(box["page_number"]) == 1 for box in parser.boxes))

    def test_table_transformer_job_skips_tsr_when_table_structure_not_needed(self):
        parser = object.__new__(pdf_parser.DeepDocPdfParser)
        parser.page_images = [Image.new("RGB", (320, 220), "white")]
        parser.page_layout = [[{"type": "table", "x0": 60, "top": 40, "x1": 180, "bottom": 120}]]
        parser.page_cum_height = np.array([0.0, 220.0])
        parser.boxes = []
        parser.complex_block_only_pages = {1}
        parser._last_selective_ocr_block_count = 0
        parser._last_complex_block_counts = {}

        def fail_dispatch(_imgs):
            raise AssertionError("TSR should be skipped when table structure is not needed")

        parser._dispatch_table_structure_recognition = fail_dispatch
        parser._ocr_selective_table_regions = lambda zm, table_layouts, page_numbers=None: 3

        parser._table_transformer_job(1, auto_rotate=False, need_table_structure=False)

        self.assertEqual(3, parser._last_selective_ocr_block_count)
        self.assertEqual({"table": 3}, parser._last_complex_block_counts)

    def test_dispatch_layout_recognition_spreads_pages_across_worker_pool(self):
        parser = object.__new__(pdf_parser.DeepDocPdfParser)
        worker0 = RecordingLayoutWorker(0)
        worker1 = RecordingLayoutWorker(1)
        parser.layouter = worker0
        parser.layouters = [worker0, worker1]

        images = [Image.new("RGB", (32, 32), "white") for _ in range(4)]
        page_boxes = [
            [{"text": f"P{page_number}", "page_number": page_number, "layout_type": "text", "x0": 0, "x1": 10, "top": 0, "bottom": 10}]
            for page_number in range(1, 5)
        ]

        laid_out_boxes, page_layout = parser._dispatch_layout_recognition(images, page_boxes, 1, drop=True)

        self.assertEqual([[1, 3]], worker0.calls)
        self.assertEqual([[2, 4]], worker1.calls)
        self.assertEqual([0, 1, 0, 1], [int(box["layout_worker_id"]) for box in laid_out_boxes])
        self.assertEqual([0, 1, 0, 1], [int(layout[0]["worker_id"]) for layout in page_layout])

    def test_dispatch_table_structure_recognition_spreads_tables_across_worker_pool(self):
        parser = object.__new__(pdf_parser.DeepDocPdfParser)
        worker0 = RecordingTableWorker(0)
        worker1 = RecordingTableWorker(1)
        parser.tbl_det = worker0
        parser.tbl_dets = [worker0, worker1]

        images = [Image.new("RGB", (24, 24), "white") for _ in range(5)]

        recos = parser._dispatch_table_structure_recognition(images)

        self.assertEqual([3], worker0.calls)
        self.assertEqual([2], worker1.calls)
        self.assertEqual([0, 1, 0, 1, 0], [int(items[0]["worker_id"]) for items in recos])

class HybridRoutingIntegrationTest(unittest.TestCase):
    def test_build_pdf_hybrid_plan_collapses_clean_native_words_into_line_boxes(self):
        from deepdoc.parser.pdf_hybrid_router import build_pdf_hybrid_plan

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-plan-lines-") as temp_dir:
            path = Path(temp_dir) / "clean-lines.pdf"
            path.write_bytes(_text_and_image_pdf_bytes())

            plan = build_pdf_hybrid_plan(path, page_to=5)

        self.assertEqual("digital_clean", plan["pages"][0]["route"])
        self.assertEqual(
            ["Alpha Beta", "Gamma"],
            [box["text"] for box in plan["native_boxes_by_page"][1]],
        )
        self.assertEqual(2, plan["pages"][0]["native_box_count"])

    def test_build_pdf_hybrid_plan_treats_small_inline_image_page_as_clean(self):
        from deepdoc.parser.pdf_hybrid_router import build_pdf_hybrid_plan

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-plan-inline-image-") as temp_dir:
            path = Path(temp_dir) / "inline-image.pdf"
            path.write_bytes(_text_with_small_inline_image_pdf_bytes())

            plan = build_pdf_hybrid_plan(path, page_to=5)

        self.assertEqual("digital_clean", plan["pages"][0]["route"])
        self.assertEqual("skip_all_ocr", plan["pages"][0]["ocr_scope"])
        self.assertIn("native_text_confident", plan["pages"][0]["reasons"])
        self.assertLess(plan["pages"][0]["image_area_ratio"], 0.03)

    def test_build_pdf_hybrid_plan_routes_clean_and_scanned_pages(self):
        from deepdoc.parser.pdf_hybrid_router import build_pdf_hybrid_plan

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-plan-") as temp_dir:
            path = Path(temp_dir) / "mixed.pdf"
            path.write_bytes(_text_and_image_pdf_bytes())

            plan = build_pdf_hybrid_plan(path, page_to=5)

        self.assertEqual(2, len(plan["pages"]))
        self.assertEqual("digital_clean", plan["pages"][0]["route"])
        self.assertEqual("scanned", plan["pages"][1]["route"])
        self.assertEqual("skip_all_ocr", plan["pages"][0]["ocr_scope"])
        self.assertIn("native_text_confident", plan["pages"][0]["reasons"])
        self.assertEqual("full_page", plan["pages"][1]["ocr_scope"])
        self.assertIn("native_text_sparse", plan["pages"][1]["reasons"])
        self.assertEqual(
            {
                "page_count": 2,
                "native_only_page_count": 1,
                "ocr_page_count": 1,
                "complex_block_only_page_count": 0,
                "full_page_ocr_page_count": 1,
                "digital_clean_pages": 1,
                "digital_mixed_pages": 0,
                "scanned_pages": 1,
            },
            plan["route_summary"],
        )

    def test_build_pdf_hybrid_plan_routes_mixed_and_scanned_pages(self):
        from deepdoc.parser.pdf_hybrid_router import build_pdf_hybrid_plan

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-plan-mixed-") as temp_dir:
            path = Path(temp_dir) / "mixed-scanned.pdf"
            path.write_bytes(_mixed_and_scanned_pdf_bytes())

            plan = build_pdf_hybrid_plan(path, page_to=5)

        self.assertEqual("digital_mixed", plan["pages"][0]["route"])
        self.assertEqual("scanned", plan["pages"][1]["route"])
        self.assertEqual("complex_blocks_only", plan["pages"][0]["ocr_scope"])
        self.assertEqual(["table"], plan["pages"][0]["complex_block_types"])
        self.assertIn("native_text_with_images", plan["pages"][0]["reasons"])
        self.assertEqual([1], sorted(int(page_number) for page_number in plan["seed_layouts_by_page"]))
        self.assertEqual("table", plan["seed_layouts_by_page"][1][0]["type"])
        self.assertEqual([2], plan["ocr_page_numbers"])
        self.assertEqual([1], plan["complex_block_page_numbers"])
        self.assertEqual(
            {
                "page_count": 2,
                "native_only_page_count": 1,
                "ocr_page_count": 2,
                "complex_block_only_page_count": 1,
                "full_page_ocr_page_count": 1,
                "digital_clean_pages": 0,
                "digital_mixed_pages": 1,
                "scanned_pages": 1,
            },
            plan["route_summary"],
        )

    def test_parse_pdf_from_tmp_hybrid_gpu_skips_pdf_text_layer_detection(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-skip-text-layer-") as temp_dir:
            path = Path(temp_dir) / "mixed-scan.pdf"
            path.write_bytes(_mixed_and_scanned_pdf_bytes())

            with patch("deepdoc.parser.pdf_parser.detect_pdf_text_layer", side_effect=AssertionError("should not run")):
                boxes, tables, meta = main._parse_pdf_from_tmp(
                    TrackingHybridParser,
                    str(path),
                    {
                        "parser_engine": "deepdoc",
                        "deepdoc_pdf_mode": "hybrid",
                        "execution_profile": "gpu",
                        "deepdoc_layout_model": "general",
                        "return_structured": False,
                        "persist_artifacts": False,
                        "include_chunks": False,
                        "return_images": False,
                        "enable_formula": False,
                        "enable_seal": False,
                    },
                )

        self.assertEqual([], tables)
        self.assertEqual("derived_from_hybrid_plan", meta["pdf_text_layer"]["status"])
        self.assertEqual("ocr", meta["pdf_text_layer"]["recommended_mode"])
        self.assertTrue(any(box["text"] == "Alpha" for box in boxes))

    def test_parse_pdf_from_tmp_hybrid_gpu_all_clean_structured_keeps_chars(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-clean-structured-") as temp_dir:
            path = Path(temp_dir) / "clean.pdf"
            path.write_bytes(_positioned_native_pdf_bytes())

            rows, tables, meta = main._parse_pdf_from_tmp(
                TrackingHybridParser,
                str(path),
                {
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "hybrid",
                    "execution_profile": "gpu",
                    "deepdoc_layout_model": "general",
                    "return_structured": True,
                    "persist_artifacts": False,
                    "include_chunks": False,
                    "return_images": False,
                    "enable_formula": False,
                    "enable_seal": False,
                },
            )

        self.assertEqual([], tables)
        self.assertTrue(any(row[0] == "Alpha" for row in rows))
        structured_boxes = meta["structured_source"]["boxes"]
        self.assertGreaterEqual(len(structured_boxes), 2)
        self.assertIn("chars", structured_boxes[0])
        self.assertEqual("Alpha", "".join(char["text"] for char in structured_boxes[0]["chars"]))

    def test_parse_pdf_from_tmp_uses_hybrid_native_route_on_gpu_for_clean_pdf(self):
        class ForbiddenHybridParser:
            def __init__(self):
                raise AssertionError("digital_clean hybrid pages should not construct the OCR parser")

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-native-gpu-") as temp_dir:
            path = Path(temp_dir) / "native.pdf"
            path.write_bytes(_positioned_native_pdf_bytes())

            rows, tables, meta = main._parse_pdf_from_tmp(
                ForbiddenHybridParser,
                str(path),
                {
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "hybrid",
                    "execution_profile": "gpu",
                    "return_structured": True,
                    "persist_artifacts": False,
                    "include_chunks": True,
                    "return_images": False,
                    "enable_formula": False,
                    "enable_seal": False,
                },
            )

        self.assertEqual([], tables)
        self.assertEqual("hybrid", meta["pdf_parse_mode"])
        self.assertEqual("native_pdf", meta["structured_source"]["engine"])
        self.assertEqual("digital_clean", meta["page_routes"][0]["route"])
        self.assertTrue(any(row[0] == "Alpha" for row in rows))
        self.assertTrue(any(row[0] == "Beta" for row in rows))

    def test_parse_pdf_from_tmp_uses_hybrid_native_route_on_gpu_for_clean_pdf_with_small_inline_image(self):
        class ForbiddenHybridParser:
            def __init__(self):
                raise AssertionError("inline-image clean hybrid pages should not construct the OCR parser")

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-inline-native-gpu-") as temp_dir:
            path = Path(temp_dir) / "inline-native.pdf"
            path.write_bytes(_text_with_small_inline_image_pdf_bytes())

            rows, tables, meta = main._parse_pdf_from_tmp(
                ForbiddenHybridParser,
                str(path),
                {
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "hybrid",
                    "execution_profile": "gpu",
                    "return_structured": False,
                    "persist_artifacts": False,
                    "include_chunks": False,
                    "return_images": False,
                    "enable_formula": False,
                    "enable_seal": False,
                },
            )

        self.assertEqual([], tables)
        self.assertEqual("hybrid", meta["pdf_parse_mode"])
        self.assertEqual("digital_clean", meta["page_routes"][0]["route"])
        self.assertTrue(any(row[0] == "Alpha Beta" for row in rows))
        self.assertTrue(any(row[0] == "Gamma Delta" for row in rows))

    def test_parse_pdf_from_tmp_hybrid_ocrs_only_non_clean_pages(self):
        TrackingHybridParser.constructed = 0
        TrackingHybridParser.last_instance = None

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-selective-ocr-") as temp_dir:
            path = Path(temp_dir) / "clean-scan.pdf"
            path.write_bytes(_text_and_image_pdf_bytes())

            boxes, tables, meta = main._parse_pdf_from_tmp(
                TrackingHybridParser,
                str(path),
                {
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "hybrid",
                    "execution_profile": "gpu",
                    "deepdoc_layout_model": "general",
                    "return_structured": False,
                    "persist_artifacts": False,
                    "include_chunks": False,
                    "return_images": False,
                    "enable_formula": False,
                    "enable_seal": False,
                },
            )

        parser = TrackingHybridParser.last_instance
        self.assertEqual(1, TrackingHybridParser.constructed)
        self.assertIsNotNone(parser)
        self.assertEqual([1], parser.seeded_pages)
        self.assertEqual([2], parser.called_pages)
        self.assertEqual([], parser.char_page_numbers)
        self.assertEqual([2], parser.image_page_numbers)
        self.assertEqual([2], parser.layout_page_numbers)
        self.assertEqual([], tables)
        self.assertEqual("hybrid", meta["pdf_parse_mode"])
        self.assertEqual(["digital_clean", "scanned"], [page["route"] for page in meta["page_routes"][:2]])
        self.assertEqual(
            {
                "page_count": 2,
                "native_only_page_count": 1,
                "ocr_page_count": 1,
                "complex_block_only_page_count": 0,
                "full_page_ocr_page_count": 1,
                "digital_clean_pages": 1,
                "digital_mixed_pages": 0,
                "scanned_pages": 1,
            },
            meta["hybrid_route_summary"],
        )
        self.assertTrue(any(box["text"] == "Alpha Beta" for box in boxes))
        self.assertTrue(any(box["text"] == "Gamma" for box in boxes))
        self.assertTrue(any(box["text"] == "OCR page 2" for box in boxes))

    def test_parse_pdf_from_tmp_hybrid_ocrs_mixed_and_scanned_pages(self):
        TrackingHybridParser.constructed = 0
        TrackingHybridParser.last_instance = None

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-mixed-ocr-") as temp_dir:
            path = Path(temp_dir) / "mixed-scan.pdf"
            path.write_bytes(_mixed_and_scanned_pdf_bytes())

            boxes, tables, meta = main._parse_pdf_from_tmp(
                TrackingHybridParser,
                str(path),
                {
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "hybrid",
                    "execution_profile": "gpu",
                    "deepdoc_layout_model": "general",
                    "return_structured": False,
                    "persist_artifacts": False,
                    "include_chunks": False,
                    "return_images": False,
                    "enable_formula": False,
                    "enable_seal": False,
                },
            )

        parser = TrackingHybridParser.last_instance
        self.assertEqual(1, TrackingHybridParser.constructed)
        self.assertIsNotNone(parser)
        self.assertEqual([1], parser.seeded_pages)
        self.assertEqual([1], parser.seeded_layout_pages)
        self.assertEqual([2], parser.called_pages)
        self.assertEqual([], parser.char_page_numbers)
        self.assertEqual([1, 2], parser.image_page_numbers)
        self.assertEqual([2], parser.layout_page_numbers)
        self.assertEqual([], tables)
        self.assertEqual("hybrid", meta["pdf_parse_mode"])
        self.assertEqual(["digital_mixed", "scanned"], [page["route"] for page in meta["page_routes"][:2]])
        self.assertEqual(
            {
                "page_count": 2,
                "native_only_page_count": 1,
                "ocr_page_count": 2,
                "complex_block_only_page_count": 1,
                "full_page_ocr_page_count": 1,
                "digital_clean_pages": 0,
                "digital_mixed_pages": 1,
                "scanned_pages": 1,
            },
            meta["hybrid_route_summary"],
        )
        self.assertTrue(any(box["text"] == "Alpha" for box in boxes))
        self.assertTrue(any(box["text"] == "OCR page 2" for box in boxes))
        self.assertFalse(parser.table_auto_rotate)
        self.assertFalse(parser.table_need_structure)

    def test_parse_pdf_from_tmp_hybrid_structured_keeps_table_structure_pipeline(self):
        TrackingHybridParser.constructed = 0
        TrackingHybridParser.last_instance = None

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-mixed-structured-") as temp_dir:
            path = Path(temp_dir) / "mixed-scan.pdf"
            path.write_bytes(_mixed_and_scanned_pdf_bytes())

            boxes, tables, meta = main._parse_pdf_from_tmp(
                TrackingHybridParser,
                str(path),
                {
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "hybrid",
                    "execution_profile": "gpu",
                    "deepdoc_layout_model": "general",
                    "return_structured": True,
                    "persist_artifacts": False,
                    "include_chunks": False,
                    "return_images": False,
                    "enable_formula": False,
                    "enable_seal": False,
                },
            )

        parser = TrackingHybridParser.last_instance
        self.assertEqual(1, TrackingHybridParser.constructed)
        self.assertIsNotNone(parser)
        self.assertTrue(parser.table_need_structure)
        self.assertEqual("hybrid", meta["pdf_parse_mode"])
        self.assertEqual([], tables)
        self.assertTrue(any(box["text"] == "Alpha" for box in boxes))

    def test_layouts_rec_preserves_seeded_page_layout_for_unselected_pages(self):
        parser = object.__new__(pdf_parser.DeepDocPdfParser)
        parser.page_from = 0
        parser.page_images = [Image.new("RGB", (320, 220), "white"), Image.new("RGB", (320, 220), "white")]
        parser.page_cum_height = np.array([0.0, 220.0, 440.0])
        parser.boxes = [
            [{"text": "Alpha", "page_number": 1, "layout_type": "text", "x0": 20, "x1": 80, "top": 20, "bottom": 40}],
            [{"text": "Scan", "page_number": 2, "layout_type": "text", "x0": 24, "x1": 84, "top": 24, "bottom": 44}],
        ]
        parser.page_layout = [
            [{"type": "table", "x0": 100, "top": 50, "x1": 180, "bottom": 100, "page_number": 0}],
            [],
        ]

        def fake_dispatch(images, page_boxes, zoomin, *, drop=True):
            del images, zoomin, drop
            flattened = []
            for page in page_boxes:
                flattened.extend(dict(box) for box in page)
            return flattened, [[{"type": "text", "page_number": 1}] for _ in page_boxes]

        parser._dispatch_layout_recognition = fake_dispatch

        parser._layouts_rec(1, page_numbers={2})

        self.assertEqual("table", parser.page_layout[0][0]["type"])
        self.assertEqual("text", parser.page_layout[1][0]["type"])

    def test_parse_pdf_from_tmp_hybrid_scanned_only_skips_char_extraction(self):
        TrackingHybridParser.constructed = 0
        TrackingHybridParser.last_instance = None

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-scanned-only-") as temp_dir:
            path = Path(temp_dir) / "scanned-only.pdf"
            path.write_bytes(_scanned_only_pdf_bytes(2))

            boxes, tables, meta = main._parse_pdf_from_tmp(
                TrackingHybridParser,
                str(path),
                {
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "hybrid",
                    "execution_profile": "gpu",
                    "deepdoc_layout_model": "general",
                    "return_structured": False,
                    "persist_artifacts": False,
                    "include_chunks": False,
                    "return_images": False,
                    "enable_formula": False,
                    "enable_seal": False,
                },
            )

        parser = TrackingHybridParser.last_instance
        self.assertEqual(1, TrackingHybridParser.constructed)
        self.assertIsNotNone(parser)
        self.assertEqual([], parser.char_page_numbers)
        self.assertEqual([1, 2], parser.image_page_numbers)
        self.assertEqual([1, 2], parser.called_pages)
        self.assertEqual([1, 2], parser.layout_page_numbers)
        self.assertFalse(parser.load_outlines)
        self.assertEqual([], tables)
        self.assertEqual(["scanned", "scanned"], [page["route"] for page in meta["page_routes"][:2]])
        self.assertTrue(any(box["text"] == "OCR page 1" for box in boxes))
        self.assertTrue(any(box["text"] == "OCR page 2" for box in boxes))

    def test_bench_hybrid_pdf_reports_stage_timings_and_ocr_counts(self):
        from tools.bench_hybrid_pdf import analyze_hybrid_pdf

        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-bench-") as temp_dir:
            path = Path(temp_dir) / "mixed-scan.pdf"
            path.write_bytes(_mixed_and_scanned_pdf_bytes())

            payload = analyze_hybrid_pdf(
                path,
                page_to=5,
                mode="hybrid",
                profile="gpu",
                parser_cls=TrackingHybridParser,
            )

        self.assertEqual(["table"], payload["route_examples"][0]["complex_block_types"])
        self.assertIn("stage_timings", payload)
        self.assertIn("prepare_pages", payload["stage_timings"])
        self.assertIn("run_page_ocr", payload["stage_timings"])
        self.assertIn("ocr_block_count", payload)
        self.assertIsInstance(payload["ocr_block_count"], int)
        self.assertIn("hybrid_route_summary", payload)
        self.assertEqual(1, payload["hybrid_route_summary"]["complex_block_only_page_count"])


if __name__ == "__main__":
    unittest.main()

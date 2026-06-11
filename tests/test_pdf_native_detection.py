import tempfile
import unittest
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import fitz
from PIL import Image, ImageDraw

import main
from common.branding import PRODUCT_NAME
from deepdoc.parser.pdf_parser import detect_pdf_text_layer, extract_native_pdf_text


def _native_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    text = (
        f"{PRODUCT_NAME} native text layer detection. "
        "This paragraph is embedded as real PDF text, not an image. "
        "The parser should keep this text layer available for fast structured parsing. "
    ) * 4
    page.insert_textbox(fitz.Rect(72, 72, 520, 360), text, fontsize=11)
    payload = doc.tobytes()
    doc.close()
    return payload


def _image_only_pdf_bytes() -> bytes:
    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.text((32, 96), "This text is pixels inside an image-only PDF.", fill="black")
    image_payload = BytesIO()
    image.save(image_payload, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=450, height=130)
    page.insert_image(page.rect, stream=image_payload.getvalue())
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


class ForbiddenOcrParser:
    def __init__(self):
        raise AssertionError("native text PDF should not construct the OCR/layout parser")


class TrackingOcrParser:
    constructed = 0
    image_calls = 0

    def __init__(self):
        type(self).constructed += 1
        self.total_page = 1
        self.page_images = []
        self.boxes = []

    def __images__(self, _tmp_path, zoomin, page_from=0, page_to=299):
        type(self).image_calls += 1
        self.page_images = [Image.new("RGB", (120 * zoomin, 80 * zoomin), "white")]
        self.page_cum_height = [0, 80]
        self.boxes = [
            {
                "text": "OCR fallback text",
                "page_number": 1,
                "x0": 10,
                "x1": 100,
                "top": 10,
                "bottom": 24,
                "layout_type": "text",
            }
        ]

    def _layouts_rec(self, _zoomin, page_numbers=None):
        del page_numbers
        return None

    def _table_transformer_job(self, _zoomin, auto_rotate=True):
        del auto_rotate
        return None

    def _text_merge(self):
        return None

    def _extract_table_figure(self, *args):
        if len(args) >= 5:
            return [], []
        return []

    def _concat_downward(self):
        return None

    def _filter_forpages(self):
        return None


class PdfNativeDetectionTest(unittest.TestCase):
    def test_extract_native_pdf_text_keeps_legacy_line_boxes_by_default(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-native-default-boxes-") as temp_dir:
            path = Path(temp_dir) / "native.pdf"
            path.write_bytes(_native_pdf_bytes())

            boxes, meta = extract_native_pdf_text(path, page_to=5)

        self.assertEqual(1, meta["page_count"])
        self.assertEqual(1, meta["total_page_count"])
        self.assertGreaterEqual(len(boxes), 1)
        self.assertEqual(0.0, boxes[0]["x0"])
        self.assertAlmostEqual(595.0, boxes[0]["x1"])
        self.assertGreater(boxes[0]["bottom"], boxes[0]["top"])
        self.assertEqual("text", boxes[0]["layout_type"])

    def test_deepdoc_pdf_mode_contract_is_documented(self):
        repo_root = Path(__file__).resolve().parents[1]
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        strategy_doc = (repo_root / "docs/PARSER_ENGINE_STRATEGY.md").read_text(encoding="utf-8")
        openapi = json.loads((repo_root / "openapi.json").read_text(encoding="utf-8"))
        parse_schema = (
            openapi["paths"]["/api/v1/parse"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )
        stream_schema = (
            openapi["paths"]["/api/v1/parse/stream"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )
        async_schema = (
            openapi["paths"]["/api/v1/parse/async"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )

        self.assertIn("deepdoc_pdf_mode", api_doc)
        self.assertIn("auto", api_doc)
        self.assertIn("native", api_doc)
        self.assertIn("ocr", api_doc)
        self.assertIn("hybrid", api_doc)
        self.assertIn("execution_profile", api_doc)
        self.assertIn("hybrid", strategy_doc)
        self.assertIn("execution_profile", strategy_doc)
        for schema in (parse_schema, stream_schema, async_schema):
            pdf_mode = schema["properties"].get("deepdoc_pdf_mode")
            self.assertIsInstance(pdf_mode, dict)
            self.assertEqual(["auto", "native", "ocr", "hybrid"], pdf_mode.get("enum"))
            self.assertEqual("auto", pdf_mode.get("default"))
            execution_profile = schema["properties"].get("execution_profile")
            self.assertIsInstance(execution_profile, dict)
            self.assertEqual(["auto", "cpu", "gpu"], execution_profile.get("enum"))
            self.assertEqual("auto", execution_profile.get("default"))

    def test_docpilot_is_documented_as_preferred_parser_engine_alias(self):
        repo_root = Path(__file__).resolve().parents[1]
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        strategy_doc = (repo_root / "docs/PARSER_ENGINE_STRATEGY.md").read_text(encoding="utf-8")

        self.assertIn("`docpilot`", readme)
        self.assertIn("`docpilot`", api_doc)
        self.assertIn("`docpilot`", strategy_doc)

    def test_parse_options_normalize_hybrid_mode_and_execution_profile(self):
        with main.app.test_request_context(
            "/api/v1/parse",
            method="POST",
            data={
                "parser_engine": "deepdoc",
                "deepdoc_pdf_mode": "HYBRID",
                "execution_profile": "GPU",
            },
        ):
            options = main._build_parse_options()

        self.assertEqual("hybrid", options["deepdoc_pdf_mode"])
        self.assertEqual("gpu", options["execution_profile"])

        with main.app.test_request_context(
            "/api/v1/parse",
            method="POST",
            data={"deepdoc_pdf_mode": "bogus", "execution_profile": "bogus"},
        ):
            defaulted = main._build_parse_options()

        self.assertEqual("auto", defaulted["deepdoc_pdf_mode"])
        self.assertEqual("auto", defaulted["execution_profile"])

    def test_parse_options_accept_docpilot_parser_engine_alias(self):
        with main.app.test_request_context(
            "/api/v1/parse",
            method="POST",
            data={"parser_engine": "docpilot"},
        ):
            options = main._build_parse_options()

        self.assertEqual("deepdoc", options["parser_engine"])

    def test_parse_options_honor_env_fallbacks(self):
        with patch.dict(
            "os.environ",
            {
                "DEEPDOC_EXECUTION_PROFILE": "GPU",
                "DEEPDOC_PDF_MODE": "HYBRID",
            },
            clear=False,
        ):
            with main.app.test_request_context("/api/v1/parse", method="POST", data={}):
                options = main._build_parse_options()

        self.assertEqual("gpu", options["execution_profile"])
        self.assertEqual("hybrid", options["deepdoc_pdf_mode"])

    def test_pdf_text_layer_detection_recommends_native_for_real_text_pdf(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-native-pdf-") as temp_dir:
            path = Path(temp_dir) / "native.pdf"
            path.write_bytes(_native_pdf_bytes())

            report = detect_pdf_text_layer(path, page_to=5)

        self.assertEqual("2026-06-08.pdf-text-layer.v1", report["schema_version"])
        self.assertEqual("native_text", report["recommended_mode"])
        self.assertTrue(report["has_text_layer"])
        self.assertGreaterEqual(report["non_whitespace_char_count"], 80)
        self.assertGreaterEqual(report["font_coverage_ratio"], 0.8)

    def test_pdf_text_layer_detection_recommends_ocr_for_image_only_pdf(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-scan-pdf-") as temp_dir:
            path = Path(temp_dir) / "scan.pdf"
            path.write_bytes(_image_only_pdf_bytes())

            report = detect_pdf_text_layer(path, page_to=5)

        self.assertEqual("ocr", report["recommended_mode"])
        self.assertFalse(report["has_text_layer"])
        self.assertEqual(0, report["non_whitespace_char_count"])

    def test_pdf_text_layer_detection_dedupes_overlaid_chars(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-overlay-pdf-") as temp_dir:
            path = Path(temp_dir) / "overlay.pdf"
            path.write_bytes(_overlaid_text_pdf_bytes())

            report = detect_pdf_text_layer(path, page_to=5, min_non_whitespace_chars=1)

        self.assertEqual("native_text", report["recommended_mode"])
        self.assertTrue(report["has_text_layer"])
        self.assertEqual(5, report["non_whitespace_char_count"])
        self.assertEqual(5, report["font_backed_char_count"])

    def test_deepdoc_auto_mode_uses_native_text_without_constructing_ocr_parser(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-native-route-") as temp_dir:
            path = Path(temp_dir) / "native.pdf"
            path.write_bytes(_native_pdf_bytes())

            text_rows, tables, meta = main._parse_pdf_from_tmp(
                ForbiddenOcrParser,
                str(path),
                {
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "auto",
                    "return_structured": True,
                    "persist_artifacts": False,
                    "include_chunks": True,
                    "return_images": False,
                    "enable_formula": False,
                    "enable_seal": False,
                },
            )

        self.assertEqual([], tables)
        self.assertEqual("native_text", meta["pdf_parse_mode"])
        self.assertEqual("native_pdf", meta["structured_source"]["engine"])
        self.assertTrue(any(f"{PRODUCT_NAME} native text layer detection" in row[0] for row in text_rows))

    def test_deepdoc_hybrid_mode_currently_matches_auto_behavior_for_native_text_pdf(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-hybrid-native-route-") as temp_dir:
            path = Path(temp_dir) / "native.pdf"
            path.write_bytes(_native_pdf_bytes())

            text_rows, tables, meta = main._parse_pdf_from_tmp(
                ForbiddenOcrParser,
                str(path),
                {
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "hybrid",
                    "return_structured": True,
                    "persist_artifacts": False,
                    "include_chunks": True,
                    "return_images": False,
                    "enable_formula": False,
                    "enable_seal": False,
                },
            )

        self.assertEqual([], tables)
        self.assertEqual("hybrid", meta["deepdoc_pdf_mode"])
        self.assertEqual("native_text", meta["pdf_parse_mode"])
        self.assertEqual("native_pdf", meta["structured_source"]["engine"])
        self.assertTrue(any(f"{PRODUCT_NAME} native text layer detection" in row[0] for row in text_rows))

    def test_deepdoc_auto_mode_keeps_ocr_layout_for_image_only_pdf(self):
        TrackingOcrParser.constructed = 0
        TrackingOcrParser.image_calls = 0
        with tempfile.TemporaryDirectory(prefix="deepdoc-scan-route-") as temp_dir:
            path = Path(temp_dir) / "scan.pdf"
            path.write_bytes(_image_only_pdf_bytes())

            boxes, tables, meta = main._parse_pdf_from_tmp(
                TrackingOcrParser,
                str(path),
                {
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "auto",
                    "return_structured": False,
                    "persist_artifacts": False,
                    "include_chunks": False,
                    "return_images": False,
                    "enable_formula": False,
                    "enable_seal": False,
                },
            )

        self.assertEqual(1, TrackingOcrParser.constructed)
        self.assertEqual(1, TrackingOcrParser.image_calls)
        self.assertEqual([], tables)
        self.assertEqual("ocr", meta["pdf_parse_mode"])
        self.assertEqual("OCR fallback text", boxes[0]["text"])

    def test_parse_endpoint_returns_structured_native_pdf_chunks(self):
        with main.app.test_client() as client:
            response = client.post(
                "/api/v1/parse",
                data={
                    "file": (BytesIO(_native_pdf_bytes()), "native.pdf"),
                    "parser_engine": "deepdoc",
                    "deepdoc_pdf_mode": "auto",
                    "return_structured": "true",
                    "persist_artifacts": "false",
                    "include_chunks": "true",
                    "chunk_strategy": "page_aware",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(200, response.status_code, response.get_data(as_text=True))
        result = response.get_json()["results"][0]
        self.assertNotIn("error", result)
        self.assertEqual("pdf", result["type"])
        structured = result["structured"]
        self.assertEqual("pdf_native_text", structured["metadata"]["source"])
        self.assertEqual("native_text", structured["document"]["metadata"]["pdf_parse_mode"])
        self.assertGreaterEqual(len(structured["blocks"]), 1)
        self.assertGreaterEqual(len(structured["chunks"]), 1)
        self.assertIn(f"{PRODUCT_NAME} native text layer detection", result["markdown"])


if __name__ == "__main__":
    unittest.main()

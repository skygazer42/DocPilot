import json
import unittest
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import main
from common.parse_artifacts import ParseDocument, build_chunk_export_records
from common.parse_builders import build_image_artifact
from deepdoc.vision.barcode import detect_barcodes


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (120, 80), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _qr_png_bytes(payload: str = "deepdoc://barcode-test") -> bytes:
    encoded = cv2.QRCodeEncoder_create().encode(payload)
    encoded = cv2.resize(encoded, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    canvas = np.full((220, 220), 255, dtype=np.uint8)
    top = (canvas.shape[0] - encoded.shape[0]) // 2
    left = (canvas.shape[1] - encoded.shape[1]) // 2
    canvas[top : top + encoded.shape[0], left : left + encoded.shape[1]] = encoded
    ok, buffer = cv2.imencode(".png", canvas)
    assert ok
    return buffer.tobytes()


def _document() -> ParseDocument:
    return ParseDocument(
        document_id="doc-image",
        parse_id="parse-image",
        filename="sample.png",
        file_type="png",
        parser_engine="deepdoc",
        created_at="2026-06-08T00:00:00+00:00",
        source_sha256="image123",
        source_size_bytes=64,
    )


class _StubOcrEngine:
    def __call__(self, _img):
        return [
            (
                [[10, 12], [100, 12], [100, 36], [10, 36]],
                ("Hello image parse", 0.99),
            )
        ]


class _StubLayoutEngine:
    def __call__(self, _images, ocr_boxes_by_page, scale_factor=1):
        boxes = [dict(box, layout_type="text") for box in ocr_boxes_by_page[0]]
        return boxes, [[{"type": "text", "bbox": [10, 12, 100, 36], "page_number": 0}]]


class ImageParseTest(unittest.TestCase):
    def test_image_parse_contract_is_documented(self):
        repo_root = Path(__file__).resolve().parents[1]
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")
        openapi = json.loads((repo_root / "openapi.json").read_text(encoding="utf-8"))
        parse_schema = (
            openapi["paths"]["/api/v1/parse"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )
        async_schema = (
            openapi["paths"]["/api/v1/parse/async"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )

        self.assertIn("png/jpg/jpeg/bmp/tiff/webp", api_doc)
        self.assertIn("图片文件会走 OCR + Layout 解析链", api_doc)
        self.assertIn("| D6 | **图片直传 `/parse`** | 已落地", roadmap)
        for schema in (parse_schema, async_schema):
            self.assertIn("png/jpg/jpeg/bmp/tiff/webp", schema["properties"]["file"]["description"])

    def test_barcode_detection_contract_is_documented(self):
        repo_root = Path(__file__).resolve().parents[1]
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")

        self.assertIn("| A3 | **二维码/条形码** | 已落地", roadmap)
        self.assertIn("二维码/条形码", api_doc)

    def test_main_infers_image_file_types_for_parse(self):
        for ext in ("png", "jpg", "jpeg", "bmp", "tiff", "webp"):
            with self.subTest(ext=ext):
                self.assertIn(ext, main.IMAGE_FILE_TYPES)
                self.assertEqual(ext, main._infer_file_type(f"sample.{ext}"))

    def test_build_image_artifact_uses_ocr_boxes_chunks_and_image_asset_context(self):
        image = Image.new("RGB", (120, 80), "white")
        boxes = [
            {
                "text": "Hello image parse",
                "score": 0.99,
                "x0": 10,
                "x1": 100,
                "top": 12,
                "bottom": 36,
                "page_number": 0,
                "layout_type": "text",
            }
        ]

        artifact = build_image_artifact(
            document=_document(),
            markdown="Hello image parse",
            image=image,
            boxes=boxes,
            chunk_max_tokens=256,
            chunk_overlap_tokens=0,
            chunk_strategy="asset_aware",
            metadata={"parser_engine": "deepdoc", "chunk_strategy": "asset_aware"},
        )

        self.assertEqual(1, len(artifact.assets))
        self.assertEqual("image", artifact.assets[0].asset_type)
        self.assertEqual("source_image", artifact.assets[0].metadata["source"])
        self.assertEqual("Hello image parse", artifact.blocks[0].text)
        self.assertEqual("text", artifact.blocks[0].block_type)
        self.assertEqual(1, len(artifact.chunks))
        self.assertIn("Hello image parse", artifact.chunks[0].text)
        self.assertEqual("2026-06-08.asset-context.v1", artifact.assets[0].metadata["asset_context_schema_version"])

        records = build_chunk_export_records(artifact)
        self.assertEqual(1, len(records))
        self.assertEqual("image", records[0].assets[0].asset_type)
        self.assertEqual(
            "2026-06-08.asset-context.v1",
            records[0].assets[0].metadata["asset_context_schema_version"],
        )

    def test_detect_barcodes_decodes_qr_code_with_positions(self):
        image = Image.open(BytesIO(_qr_png_bytes())).convert("RGB")

        detections = detect_barcodes(image)

        self.assertEqual(1, len(detections))
        self.assertEqual("qr_code", detections[0]["barcode_type"])
        self.assertEqual("deepdoc://barcode-test", detections[0]["text"])
        self.assertTrue(detections[0]["positions"])

    def test_build_image_artifact_adds_barcode_asset_block_and_chunk(self):
        image = Image.open(BytesIO(_qr_png_bytes())).convert("RGB")
        barcodes = detect_barcodes(image)

        artifact = build_image_artifact(
            document=_document(),
            markdown="",
            image=image,
            boxes=[],
            barcodes=barcodes,
            chunk_max_tokens=256,
            chunk_overlap_tokens=0,
            chunk_strategy="asset_aware",
            metadata={"parser_engine": "deepdoc", "chunk_strategy": "asset_aware"},
        )

        barcode_assets = [asset for asset in artifact.assets if asset.asset_type == "barcode"]
        self.assertEqual(1, len(barcode_assets))
        self.assertEqual("deepdoc://barcode-test", barcode_assets[0].text)
        self.assertEqual("qr_code", barcode_assets[0].metadata["barcode_type"])
        barcode_blocks = [block for block in artifact.blocks if block.block_type == "barcode"]
        self.assertEqual(1, len(barcode_blocks))
        self.assertEqual([barcode_assets[0].asset_id], barcode_blocks[0].asset_refs)
        self.assertTrue(any(barcode_assets[0].asset_id in chunk.asset_refs for chunk in artifact.chunks))

    def test_parse_endpoint_returns_structured_image_artifact(self):
        original_ocr_engine = main.ocr_engine
        original_layout_engine = main.layout_engine
        main.ocr_engine = _StubOcrEngine()
        main.layout_engine = _StubLayoutEngine()
        try:
            with main.app.test_client() as client:
                response = client.post(
                    "/api/v1/parse",
                    data={
                        "file": (BytesIO(_png_bytes()), "sample.png"),
                        "return_structured": "true",
                        "persist_artifacts": "false",
                        "include_chunks": "true",
                        "chunk_strategy": "asset_aware",
                    },
                    content_type="multipart/form-data",
                )
        finally:
            main.ocr_engine = original_ocr_engine
            main.layout_engine = original_layout_engine

        self.assertEqual(200, response.status_code, response.get_data(as_text=True))
        payload = response.get_json()
        result = payload["results"][0]
        self.assertNotIn("error", result)
        self.assertEqual("png", result["type"])
        self.assertIn("Hello image parse", result["markdown"])
        structured = result["structured"]
        self.assertEqual("image", structured["metadata"]["source"])
        self.assertEqual(1, len(structured["assets"]))
        self.assertEqual("image", structured["assets"][0]["asset_type"])
        self.assertEqual("text", structured["blocks"][0]["block_type"])
        self.assertEqual("Hello image parse", structured["blocks"][0]["text"])
        self.assertEqual("asset_aware_v1", structured["chunks"][0]["metadata"]["chunk_strategy"])
        self.assertEqual(
            "2026-06-08.asset-context.v1",
            structured["assets"][0]["metadata"]["asset_context_schema_version"],
        )

    def test_parse_endpoint_returns_structured_barcode_asset_for_image(self):
        original_ocr_engine = main.ocr_engine
        original_layout_engine = main.layout_engine
        main.ocr_engine = _StubOcrEngine()
        main.layout_engine = _StubLayoutEngine()
        try:
            with main.app.test_client() as client:
                response = client.post(
                    "/api/v1/parse",
                    data={
                        "file": (BytesIO(_qr_png_bytes()), "qr.png"),
                        "return_structured": "true",
                        "persist_artifacts": "false",
                        "include_chunks": "true",
                        "chunk_strategy": "asset_aware",
                    },
                    content_type="multipart/form-data",
                )
        finally:
            main.ocr_engine = original_ocr_engine
            main.layout_engine = original_layout_engine

        self.assertEqual(200, response.status_code, response.get_data(as_text=True))
        structured = response.get_json()["results"][0]["structured"]
        barcode_assets = [asset for asset in structured["assets"] if asset["asset_type"] == "barcode"]
        self.assertEqual(1, len(barcode_assets))
        self.assertEqual("deepdoc://barcode-test", barcode_assets[0]["text"])
        self.assertTrue(any(block["block_type"] == "barcode" for block in structured["blocks"]))


if __name__ == "__main__":
    unittest.main()

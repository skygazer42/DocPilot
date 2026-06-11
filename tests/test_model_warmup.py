import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class _WarmupOcrEngine:
    def __init__(self):
        self.images: list[np.ndarray] = []

    def __call__(self, image):
        self.images.append(image)
        return [
            (
                [[2, 3], [20, 3], [20, 14], [2, 14]],
                ("warmup", 0.99),
            )
        ]


class _WarmupLayoutEngine:
    def __init__(self):
        self.calls: list[tuple[list[np.ndarray], list[list[dict[str, object]]], int]] = []

    def __call__(self, images, ocr_boxes_by_page, scale_factor=1):
        self.calls.append((images, ocr_boxes_by_page, scale_factor))
        return ocr_boxes_by_page[0], [[{"type": "text", "bbox": [2, 3, 20, 14], "page_number": 0}]]


class ModelWarmupTest(unittest.TestCase):
    def test_warmup_models_runs_ocr_and_layout_on_white_image(self):
        import main

        original_ocr_engine = main.ocr_engine
        original_layout_engine = main.layout_engine
        ocr_engine = _WarmupOcrEngine()
        layout_engine = _WarmupLayoutEngine()
        main.ocr_engine = ocr_engine
        main.layout_engine = layout_engine
        try:
            result = main.warmup_models(image_size=32, enabled=True, load_if_needed=False)
        finally:
            main.ocr_engine = original_ocr_engine
            main.layout_engine = original_layout_engine

        self.assertEqual("ok", result["status"])
        self.assertEqual("model_warmup", result["source"])
        self.assertEqual(1, len(ocr_engine.images))
        self.assertEqual((32, 32, 3), ocr_engine.images[0].shape)
        self.assertTrue(np.all(ocr_engine.images[0] == 255))
        self.assertEqual(1, len(layout_engine.calls))
        self.assertEqual(1, len(layout_engine.calls[0][1][0]))
        self.assertEqual("warmup", layout_engine.calls[0][1][0][0]["text"])

    def test_warmup_models_skips_when_disabled_by_env(self):
        import main

        with patch.dict(os.environ, {"DEEPDOC_MODEL_WARMUP": "0"}, clear=False):
            result = main.warmup_models(enabled=None, load_if_needed=False)

        self.assertEqual("skipped", result["status"])
        self.assertEqual("disabled", result["reason"])

    def test_startup_path_and_docs_expose_model_warmup(self):
        repo_root = Path(__file__).resolve().parents[1]
        main_source = (repo_root / "main.py").read_text(encoding="utf-8")
        docs = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")

        self.assertIn("warmup_models()", main_source)
        self.assertIn("DEEPDOC_MODEL_WARMUP", docs)
        self.assertIn("| B5 | **Worker 池预热** | 已落地", roadmap)


if __name__ == "__main__":
    unittest.main()

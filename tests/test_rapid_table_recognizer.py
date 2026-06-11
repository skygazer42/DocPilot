import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


class RapidTableRecognizerAdapterTest(unittest.TestCase):
    def test_rapid_table_adapter_supports_keyword_ocr_result_and_dict_html(self):
        from deepdoc.vision import rapid_table_recognizer

        captured = {}

        class FakeRapidTableInput:
            def __init__(self, **kwargs):
                captured["input_kwargs"] = kwargs

        class FakeRapidTable:
            def __init__(self, config):
                captured["config"] = config

            def __call__(self, image, *, ocr_result):
                captured["image_shape"] = image.shape
                captured["ocr_result"] = ocr_result
                return {"pred_html": "<table><tr><td>A</td></tr></table>"}

        fake_module = types.ModuleType("rapid_table")
        fake_module.RapidTable = FakeRapidTable
        fake_module.RapidTableInput = FakeRapidTableInput

        with patch.dict(sys.modules, {"rapid_table": fake_module}):
            with patch.object(rapid_table_recognizer, "ensure_groups", return_value="/models"):
                recognizer = rapid_table_recognizer.RapidTableRecognizer()

        table_image = Image.new("RGB", (120, 80))
        html = recognizer(
            table_image,
            [
                {
                    "text": "A",
                    "x0": 15,
                    "x1": 35,
                    "top": 25,
                    "bottom": 45,
                    "score": 0.9,
                },
                {
                    "text": " ",
                    "x0": 40,
                    "x1": 50,
                    "top": 25,
                    "bottom": 45,
                    "score": 0.8,
                },
            ],
            crop_origin=(10, 20),
            zoomin=2,
        )

        self.assertEqual("<table><tr><td>A</td></tr></table>", html)
        self.assertEqual(
            {"model_type": "slanet_plus", "model_path": "/models/table/slanet_plus.onnx"},
            captured["input_kwargs"],
        )
        self.assertEqual((80, 120, 3), captured["image_shape"])
        self.assertEqual(
            [
                [
                    [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]],
                    "A",
                    0.9,
                ]
            ],
            captured["ocr_result"],
        )
        self.assertIsInstance(np.array(table_image), np.ndarray)


if __name__ == "__main__":
    unittest.main()

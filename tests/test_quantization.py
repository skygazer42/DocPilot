import importlib
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


class FakeSession:
    created_paths: list[str] = []
    created_providers: list[list[str]] = []

    def __init__(self, model_file_path, **kwargs):
        FakeSession.created_paths.append(str(model_file_path))
        FakeSession.created_providers.append(list(kwargs.get("providers") or []))

    def close(self):
        pass


def _touch_model(root: Path, name: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"model\n")
    return path


class QuantizedModelLoadingTest(unittest.TestCase):
    def setUp(self):
        from deepdoc.vision import ocr

        self.ocr = ocr
        self.ocr.clear_loaded_models()
        FakeSession.created_paths = []
        FakeSession.created_providers = []

    def tearDown(self):
        self.ocr.clear_loaded_models()

    def test_load_model_uses_fp32_model_by_default(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-quant-load-") as temp_dir:
            root = Path(temp_dir)
            fp32_path = _touch_model(root, "det.onnx")
            _touch_model(root, "det.int8.onnx")

            with patch.dict(os.environ, {}, clear=True):
                with patch.object(self.ocr.ort, "InferenceSession", FakeSession):
                    with patch.object(self.ocr, "pip_install_torch", lambda: None):
                        self.ocr.load_model(str(root), "det")

        self.assertEqual([str(fp32_path)], FakeSession.created_paths)

    def test_load_model_uses_int8_model_when_enabled(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-quant-load-") as temp_dir:
            root = Path(temp_dir)
            _touch_model(root, "det.onnx")
            int8_path = _touch_model(root, "det.int8.onnx")

            with patch.dict(os.environ, {"DEEPDOC_QUANT": "int8"}, clear=True):
                with patch.object(self.ocr.ort, "InferenceSession", FakeSession):
                    with patch.object(self.ocr, "pip_install_torch", lambda: None):
                        self.ocr.load_model(str(root), "det")

        self.assertEqual([str(int8_path)], FakeSession.created_paths)
        self.assertEqual([["CPUExecutionProvider"]], FakeSession.created_providers)

    def test_load_model_requires_existing_int8_model_when_enabled(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-quant-load-") as temp_dir:
            root = Path(temp_dir)
            _touch_model(root, "det.onnx")

            with patch.dict(os.environ, {"DEEPDOC_QUANT": "int8"}, clear=True):
                with self.assertRaisesRegex(ValueError, "quantized INT8 model"):
                    self.ocr.load_model(str(root), "det")


class QuantizeModelsToolTest(unittest.TestCase):
    def test_discover_model_files_recurses_model_group_subdirectories(self):
        quantize_models = importlib.import_module("tools.quantize_models")

        with tempfile.TemporaryDirectory(prefix="deepdoc-quant-discover-") as temp_dir:
            root = Path(temp_dir)
            det = _touch_model(root, "det.onnx")
            rec = _touch_model(root, "rec.onnx")
            rec_v5 = _touch_model(root, "rec_v5.onnx")
            rec_int8 = _touch_model(root, "rec.int8.onnx")
            layout = _touch_model(root / "layout", "pp_doclayout_plus.onnx")
            formula = _touch_model(root / "formula", "pp_formula_net_s.onnx")
            formula_decoder = _touch_model(root / "formula", "decoder.onnx")
            _touch_model(root / "table", "slanet_plus.int8.onnx")
            (root / "notes.txt").write_text("not a model\n", encoding="utf-8")

            discovered = quantize_models.discover_model_files(root)
            discovered_with_high_risk = quantize_models.discover_model_files(root, include_high_risk=True)

        self.assertEqual(
            sorted([det, layout, formula]),
            discovered,
        )
        self.assertNotIn(rec, discovered)
        self.assertNotIn(rec_v5, discovered)
        self.assertNotIn(rec_int8, discovered)
        self.assertNotIn(formula_decoder, discovered)
        self.assertIn(rec, discovered_with_high_risk)
        self.assertIn(rec_v5, discovered_with_high_risk)
        self.assertIn(formula_decoder, discovered_with_high_risk)

    def test_quantize_cli_help_mentions_recursive_model_dir_discovery(self):
        quantize_models = importlib.import_module("tools.quantize_models")
        buffer = io.StringIO()

        with patch.object(sys, "argv", ["quantize_models.py", "--help"]):
            with redirect_stdout(buffer):
                with self.assertRaises(SystemExit) as raised:
                    quantize_models.parse_args()

        self.assertEqual(0, raised.exception.code)
        help_text = buffer.getvalue().lower()
        self.assertIn("recursively", help_text)
        self.assertIn("subdirectories", help_text)
        self.assertIn("include-risky-sequence-models", help_text)

    def test_high_risk_int8_models_are_named_explicitly(self):
        quantize_models = importlib.import_module("tools.quantize_models")

        risky = [
            Path("rec.onnx"),
            Path("rec_v5.onnx"),
            Path("formula/decoder.onnx"),
        ]
        safe = [
            Path("det.onnx"),
            Path("layout/pp_doclayout_plus.onnx"),
            Path("formula/encoder.onnx"),
            Path("formula/pp_formula_net_s.onnx"),
            Path("table/slanet_plus.onnx"),
        ]

        for model_path in risky:
            self.assertTrue(quantize_models.is_high_risk_int8_model(model_path), model_path)
        for model_path in safe:
            self.assertFalse(quantize_models.is_high_risk_int8_model(model_path), model_path)

    def test_quantize_model_file_invokes_dynamic_quantization(self):
        quantize_models = importlib.import_module("tools.quantize_models")

        with tempfile.TemporaryDirectory(prefix="deepdoc-quant-tool-") as temp_dir:
            root = Path(temp_dir)
            source = _touch_model(root, "rec.onnx")
            target = root / "rec.int8.onnx"
            calls = []

            def fake_quantize_dynamic(model_input, model_output, **kwargs):
                calls.append((str(model_input), str(model_output), kwargs))
                Path(model_output).write_bytes(b"int8\n")

            with patch.object(quantize_models, "quantize_dynamic", fake_quantize_dynamic):
                result = quantize_models.quantize_model_file(source, target)
                target_size = target.stat().st_size

        self.assertEqual(target, result.output_path)
        self.assertEqual(source, result.input_path)
        self.assertEqual(target_size, result.output_size_bytes)
        self.assertEqual(1, len(calls))
        self.assertEqual(str(source), calls[0][0])
        self.assertEqual(str(target), calls[0][1])


if __name__ == "__main__":
    unittest.main()

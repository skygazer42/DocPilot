import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeSession:
    created: list["FakeSession"] = []
    closed: list[str] = []

    def __init__(self, model_file_path, **_kwargs):
        self.model_file_path = str(model_file_path)
        self.closed = False
        FakeSession.created.append(self)

    def close(self):
        self.closed = True
        FakeSession.closed.append(self.model_file_path)


def _touch_model(root: Path, name: str) -> None:
    (root / f"{name}.onnx").write_bytes(f"{name}\n".encode("utf-8"))


class ModelCacheLifecycleTest(unittest.TestCase):
    def setUp(self):
        from deepdoc.vision import ocr

        self.ocr = ocr
        self.original_loaded_models = ocr.loaded_models
        ocr.clear_loaded_models()
        FakeSession.created = []
        FakeSession.closed = []

    def tearDown(self):
        self.ocr.clear_loaded_models()
        self.ocr.loaded_models = self.original_loaded_models

    def _patch_onnx_session(self):
        return patch.object(self.ocr.ort, "InferenceSession", FakeSession)

    def test_load_model_reuses_cached_model_and_refreshes_last_access(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-model-cache-") as temp_dir:
            root = Path(temp_dir)
            _touch_model(root, "det")
            with patch.dict(os.environ, {"DEEPDOC_MODEL_CACHE_MAX_SIZE": "4"}, clear=False):
                with self._patch_onnx_session(), patch.object(self.ocr, "pip_install_torch", lambda: None):
                    first = self.ocr.load_model(str(root), "det")
                    time.sleep(0.01)
                    second = self.ocr.load_model(str(root), "det")
                    state = self.ocr.model_cache_state()

        self.assertIs(first, second)
        self.assertEqual(1, len(FakeSession.created))
        self.assertEqual(1, state["size"])
        self.assertEqual(4, state["max_size"])
        self.assertEqual(0, len(FakeSession.closed))

    def test_load_model_evicts_lru_entry_when_cache_exceeds_limit(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-model-cache-") as temp_dir:
            root = Path(temp_dir)
            _touch_model(root, "det")
            _touch_model(root, "rec")
            with patch.dict(os.environ, {"DEEPDOC_MODEL_CACHE_MAX_SIZE": "1"}, clear=False):
                with self._patch_onnx_session(), patch.object(self.ocr, "pip_install_torch", lambda: None):
                    det = self.ocr.load_model(str(root), "det")
                    rec = self.ocr.load_model(str(root), "rec")

        self.assertIsNot(det, rec)
        self.assertEqual(2, len(FakeSession.created))
        self.assertIn(str(root / "det.onnx"), FakeSession.closed)
        self.assertEqual(1, self.ocr.model_cache_state()["size"])

    def test_prune_loaded_models_releases_idle_entries(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-model-cache-") as temp_dir:
            root = Path(temp_dir)
            _touch_model(root, "det")
            with patch.dict(os.environ, {"DEEPDOC_MODEL_CACHE_IDLE_TTL_SECONDS": "0"}, clear=False):
                with self._patch_onnx_session(), patch.object(self.ocr, "pip_install_torch", lambda: None):
                    self.ocr.load_model(str(root), "det")
                    removed = self.ocr.prune_loaded_models(force=False)

        self.assertEqual(1, removed)
        self.assertIn(str(root / "det.onnx"), FakeSession.closed)
        self.assertEqual(0, self.ocr.model_cache_state()["size"])


if __name__ == "__main__":
    unittest.main()

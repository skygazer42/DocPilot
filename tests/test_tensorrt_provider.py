import os
import sys
import tempfile
import unittest
from builtins import __import__ as builtin_import
from pathlib import Path
from unittest.mock import patch


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1


class FakeTorch:
    cuda = FakeCuda()


class FakeSession:
    created_paths: list[str] = []
    created_providers: list[list[str]] = []
    created_provider_options: list[list[dict]] = []

    def __init__(self, model_file_path, **kwargs):
        FakeSession.created_paths.append(str(model_file_path))
        FakeSession.created_providers.append(list(kwargs.get("providers") or []))
        FakeSession.created_provider_options.append(list(kwargs.get("provider_options") or []))

    def close(self):
        pass


def _touch_model(root: Path, name: str) -> None:
    (root / f"{name}.onnx").write_bytes(f"{name}\n".encode("utf-8"))


class TensorRtProviderTest(unittest.TestCase):
    def setUp(self):
        from deepdoc.vision import ocr

        self.ocr = ocr
        self.ocr.clear_loaded_models()
        FakeSession.created_paths = []
        FakeSession.created_providers = []
        FakeSession.created_provider_options = []

    def tearDown(self):
        self.ocr.clear_loaded_models()

    def test_load_model_uses_tensorrt_provider_and_engine_cache_when_enabled(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-trt-provider-") as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "trt-cache"
            _touch_model(root, "det")

            with patch.dict(
                os.environ,
                {
                    "DEEPDOC_ONNX_PROVIDER": "tensorrt",
                    "DEEPDOC_TENSORRT_CACHE_DIR": str(cache_dir),
                    "DEEPDOC_TENSORRT_FP16": "1",
                    "DEEPDOC_TENSORRT_MAX_WORKSPACE_SIZE": "1073741824",
                },
                clear=True,
            ):
                with patch.dict(sys.modules, {"torch": FakeTorch}):
                    with patch.object(self.ocr, "pip_install_torch", lambda: None):
                        with patch.object(
                            self.ocr.ort,
                            "get_available_providers",
                            lambda: [
                                "TensorrtExecutionProvider",
                                "CUDAExecutionProvider",
                                "CPUExecutionProvider",
                            ],
                        ):
                            with patch.object(self.ocr.ort, "InferenceSession", FakeSession):
                                self.ocr.load_model(str(root), "det")

        self.assertEqual(
            [["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]],
            FakeSession.created_providers,
        )
        trt_options, cuda_options, cpu_options = FakeSession.created_provider_options[0]
        self.assertEqual(str(cache_dir), trt_options["trt_engine_cache_path"])
        self.assertTrue(trt_options["trt_engine_cache_enable"])
        self.assertTrue(trt_options["trt_fp16_enable"])
        self.assertEqual(1073741824, trt_options["trt_max_workspace_size"])
        self.assertEqual(0, cuda_options["device_id"])
        self.assertEqual({}, cpu_options)

    def test_load_model_rejects_tensorrt_when_provider_is_unavailable(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-trt-provider-") as temp_dir:
            root = Path(temp_dir)
            _touch_model(root, "det")

            with patch.dict(os.environ, {"DEEPDOC_ONNX_PROVIDER": "tensorrt"}, clear=True):
                with patch.dict(sys.modules, {"torch": FakeTorch}):
                    with patch.object(self.ocr, "pip_install_torch", lambda: None):
                        with patch.object(self.ocr.ort, "get_available_providers", lambda: ["CUDAExecutionProvider"]):
                            with self.assertRaisesRegex(RuntimeError, "TensorrtExecutionProvider"):
                                self.ocr.load_model(str(root), "det")

    def test_load_model_uses_cuda_provider_without_torch_when_onnxruntime_has_cuda(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-cuda-provider-") as temp_dir:
            root = Path(temp_dir)
            _touch_model(root, "det")

            def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "torch":
                    raise ImportError("torch unavailable")
                return builtin_import(name, globals, locals, fromlist, level)

            with patch.dict(os.environ, {"DEEPDOC_ONNX_PROVIDER": "auto"}, clear=True):
                with patch.object(self.ocr, "pip_install_torch", lambda: None):
                    with patch.object(
                        self.ocr.ort,
                        "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    ):
                        with patch("builtins.__import__", side_effect=import_without_torch):
                            with patch.object(self.ocr.ort, "InferenceSession", FakeSession):
                                self.ocr.load_model(str(root), "det")

        self.assertEqual([["CUDAExecutionProvider"]], FakeSession.created_providers)

    def test_auto_configure_parallel_devices_uses_visible_cuda_device_count_without_torch(self):
        def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                raise ImportError("torch unavailable")
            return builtin_import(name, globals, locals, fromlist, level)

        original_parallel_devices = self.ocr.settings.PARALLEL_DEVICES
        detected_parallel_devices = None
        try:
            self.ocr.settings.PARALLEL_DEVICES = 0
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5"}, clear=True):
                with patch.object(self.ocr, "pip_install_torch", lambda: None):
                    with patch.object(self.ocr.ort, "get_available_providers", lambda: ["CUDAExecutionProvider"]):
                        with patch("builtins.__import__", side_effect=import_without_torch):
                            configured = self.ocr.ensure_parallel_devices_configured()
                            detected_parallel_devices = self.ocr.settings.PARALLEL_DEVICES
        finally:
            self.ocr.settings.PARALLEL_DEVICES = original_parallel_devices

        self.assertEqual(2, configured)
        self.assertEqual(2, detected_parallel_devices)

    def test_auto_configure_parallel_devices_respects_explicit_setting(self):
        original_parallel_devices = self.ocr.settings.PARALLEL_DEVICES
        try:
            self.ocr.settings.PARALLEL_DEVICES = 3
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5"}, clear=True):
                configured = self.ocr.ensure_parallel_devices_configured()
        finally:
            self.ocr.settings.PARALLEL_DEVICES = original_parallel_devices

        self.assertEqual(3, configured)

    def test_pdf_parser_init_uses_auto_configured_parallel_device_count(self):
        from deepdoc.parser import pdf_parser

        original_parallel_devices = self.ocr.settings.PARALLEL_DEVICES
        try:
            self.ocr.settings.PARALLEL_DEVICES = 0
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5"}, clear=True):
                with patch.object(self.ocr, "pip_install_torch", lambda: None):
                    with patch.object(self.ocr.ort, "get_available_providers", lambda: ["CUDAExecutionProvider"]):
                        with patch.object(
                            pdf_parser,
                            "get_shared_pdf_parser_components",
                            return_value={
                                "ocr": object(),
                                "layouter": object(),
                                "tbl_det": object(),
                                "updown_cnt_mdl": object(),
                            },
                        ):
                            parser = pdf_parser.DeepDocPdfParser()
        finally:
            self.ocr.settings.PARALLEL_DEVICES = original_parallel_devices

        self.assertEqual(2, parser.parallel_limiter)


if __name__ == "__main__":
    unittest.main()

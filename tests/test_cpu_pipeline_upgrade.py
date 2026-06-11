import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np


class FakeInput:
    name = "image"
    shape = [1, 3, 48, 320]


class FakePredictor:
    def get_inputs(self):
        return [FakeInput()]


class CpuPipelineUpgradePlanTest(unittest.TestCase):
    @staticmethod
    def _read_text(path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    @staticmethod
    def _dict_path_from_postprocess_calls(mock) -> str:
        for call in mock.call_args_list:
            params = call.args[0]
            if params.get("name") == "CTCLabelDecode":
                return os.path.basename(params["character_dict_path"])
        raise AssertionError("CTCLabelDecode postprocess call not found")

    @staticmethod
    def _license_gate_payload() -> dict:
        eval_tool = importlib.import_module("tools.eval_omnidocbench")
        return eval_tool.license_gate_report()

    @staticmethod
    def _dataset_contract_payload(dataset: Path, sample_names: list[str] | None = None) -> dict:
        names = sample_names or ["sample"]
        return {
            "schema_version": "2026-06-08.cpu-pipeline-dataset-contract.v1",
            "status": "ok",
            "dataset": str(dataset),
            "sample_count": len(names),
            "problems": [],
            "samples": [
                {"name": name, "pdf_path": str(dataset / f"{name}.pdf")}
                for name in names
            ],
        }

    def test_model_store_exposes_cpu_pipeline_model_groups_and_download_commands(self):
        import download_models
        from common.model_store import MODEL_GROUP_FILES, get_group_files

        self.assertIn("core_v5", MODEL_GROUP_FILES)
        self.assertIn("layout_v2", MODEL_GROUP_FILES)
        self.assertIn("table_v2", MODEL_GROUP_FILES)
        self.assertIn("formula_v2", MODEL_GROUP_FILES)
        self.assertIn("det_v5.onnx", MODEL_GROUP_FILES["core_v5"])
        self.assertIn("rec_v5.onnx", MODEL_GROUP_FILES["core_v5"])
        self.assertIn("ocr_v5.res", MODEL_GROUP_FILES["core_v5"])
        self.assertIn("layout/pp_doclayout_plus.onnx", MODEL_GROUP_FILES["layout_v2"])
        self.assertIn("table/slanet_plus.onnx", MODEL_GROUP_FILES["table_v2"])
        self.assertIn("table/table_cls.onnx", MODEL_GROUP_FILES["table_v2"])
        self.assertIn("formula/config.json", MODEL_GROUP_FILES["formula_v2"])
        self.assertIn("formula/inference.json", MODEL_GROUP_FILES["formula_v2"])
        self.assertIn("formula/inference.pdiparams", MODEL_GROUP_FILES["formula_v2"])
        self.assertIn("formula/inference.yml", MODEL_GROUP_FILES["formula_v2"])
        self.assertIn("formula/pp_formula_net_s.onnx", MODEL_GROUP_FILES["formula_v2"])

        with tempfile.TemporaryDirectory(prefix="deepdoc-cpu-models-") as temp_dir:
            buffer = io.StringIO()
            with patch.dict(os.environ, {"DEEPDOC_MODEL_PATH": temp_dir}, clear=False):
                with redirect_stdout(buffer):
                    download_models.print_manifest()

        payload = json.loads(buffer.getvalue())
        self.assertIn("core_v5", payload["groups"])
        self.assertIn("layout_v2", payload["groups"])
        self.assertIn("table_v2", payload["groups"])
        self.assertIn("formula_v2", payload["groups"])
        provenance = payload["model_group_provenance"]
        expected_pipeline_groups = {
            "core_v5": ("PP-OCRv5", "DEEPDOC_OCR_VERSION=v5", "ocr_v5"),
            "layout_v2": ("PP-DocLayout", "DEEPDOC_LAYOUT_ENGINE=ppdoclayout", "layout_v2"),
            "table_v2": ("SLANet-plus", "DEEPDOC_TABLE_ENGINE=rapidtable", "table_v2"),
            "formula_v2": ("PP-FormulaNet-S", "DEEPDOC_FORMULA_MODE=pp_formula_net_s", "formula_v2"),
        }
        for group, (component, switch, readiness_gate) in expected_pipeline_groups.items():
            with self.subTest(group=group):
                self.assertEqual(component, provenance[group]["component"])
                self.assertEqual("Apache-2.0", provenance[group]["license"])
                self.assertEqual("allowed", provenance[group]["license_status"])
                self.assertEqual("document_parser", provenance[group]["scope"])
                self.assertFalse(provenance[group]["default_enabled"])
                self.assertEqual(switch, provenance[group]["switch"])
                self.assertEqual(readiness_gate, provenance[group]["readiness_gate"])
                self.assertNotIn("rag", json.dumps(provenance[group], ensure_ascii=False).lower())
        self.assertIn("core_v5", download_models.SUPPORTED_DOWNLOAD_COMMANDS)
        self.assertIn("layout_v2", download_models.SUPPORTED_DOWNLOAD_COMMANDS)
        self.assertIn("table_v2", download_models.SUPPORTED_DOWNLOAD_COMMANDS)
        self.assertIn("formula_v2", download_models.SUPPORTED_DOWNLOAD_COMMANDS)
        self.assertIn("published", download_models.SUPPORTED_DOWNLOAD_COMMANDS)
        self.assertIn("det_v5.onnx", get_group_files("published"))
        self.assertIn("formula/pp_formula_net_s.onnx", get_group_files("published"))
        self.assertNotIn("rec_handwriting.onnx", get_group_files("published"))

    def test_model_publish_and_verify_help_mentions_cpu_pipeline_model_groups(self):
        repo_root = Path(__file__).resolve().parents[1]
        commands = [
            [sys.executable, "tools/publish_models_to_hf.py", "--help"],
            [sys.executable, "tools/ci/verify_hf_models.py", "--help"],
        ]
        for command in commands:
            with self.subTest(command=command[1]):
                result = subprocess.run(
                    command,
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertEqual(0, result.returncode, result.stderr)
                for expected in ["core_v5", "layout_v2", "table_v2", "formula_v2"]:
                    self.assertIn(expected, result.stdout)

    def test_formula_v2_extra_is_declared_and_locked_for_supported_python_range(self):
        pyproject = self._read_text("pyproject.toml")
        uv_lock = self._read_text("uv.lock")

        self.assertIn('requires-python = ">=3.10,<3.13"', pyproject)
        self.assertIn('requires-python = ">=3.10, <3.13"', uv_lock)
        self.assertIn("formula-v2 = [", pyproject)
        self.assertIn('"paddlex>=3.0.0,<4.0.0"', pyproject)
        self.assertIn("formula-v2 = [", uv_lock)
        self.assertIn('{ name = "paddlex" }', uv_lock)
        self.assertIn('{ name = "paddlex", marker = "extra == \'formula-v2\'", specifier = ">=3.0.0,<4.0.0" }', uv_lock)

    def test_ocr_version_v5_uses_v5_model_names_and_dictionary_with_v4_default(self):
        from deepdoc.vision import ocr

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(ocr, "load_model", return_value=(FakePredictor(), object())) as load_model:
                with patch.object(ocr, "build_post_process", return_value=object()) as build_post_process:
                    recognizer = ocr.TextRecognizer("/models")
                    detector = ocr.TextDetector("/models")

        self.assertEqual("rec", recognizer.model_name)
        self.assertEqual("det", detector.model_name)
        self.assertEqual([3, 48, 320], recognizer.rec_image_shape)
        self.assertEqual("ocr.res", self._dict_path_from_postprocess_calls(build_post_process))
        self.assertEqual(["rec", "det"], [call.args[1] for call in load_model.call_args_list])

        with patch.dict(os.environ, {"DEEPDOC_OCR_VERSION": "v5", "DEEPDOC_OCR_V5_REC_IMAGE_SHAPE": "3,64,512"}, clear=True):
            with patch.object(ocr, "load_model", return_value=(FakePredictor(), object())) as load_model:
                with patch.object(ocr, "build_post_process", return_value=object()) as build_post_process:
                    recognizer = ocr.TextRecognizer("/models")
                    detector = ocr.TextDetector("/models")

        self.assertEqual("rec_v5", recognizer.model_name)
        self.assertEqual("det_v5", detector.model_name)
        self.assertEqual([3, 64, 512], recognizer.rec_image_shape)
        self.assertEqual("ocr_v5.res", self._dict_path_from_postprocess_calls(build_post_process))
        self.assertEqual(["rec_v5", "det_v5"], [call.args[1] for call in load_model.call_args_list])

    def test_ocr_rec_image_shape_can_be_overridden_generically_and_validates_shape(self):
        from deepdoc.vision import ocr

        with patch.dict(os.environ, {"DEEPDOC_REC_IMAGE_SHAPE": "3,32,256"}, clear=True):
            with patch.object(ocr, "load_model", return_value=(FakePredictor(), object())):
                with patch.object(ocr, "build_post_process", return_value=object()):
                    recognizer = ocr.TextRecognizer("/models")

        self.assertEqual([3, 32, 256], recognizer.rec_image_shape)

        with patch.dict(os.environ, {"DEEPDOC_OCR_V5_REC_IMAGE_SHAPE": "3,0,512"}, clear=True):
            with self.assertRaisesRegex(ValueError, "Invalid DEEPDOC_OCR_V5_REC_IMAGE_SHAPE"):
                ocr._ocr_rec_image_shape("rec_v5")

    def test_ppdoclayout_mapping_and_postprocess_are_available(self):
        from deepdoc.vision.layout_recognizer import PPDocLayoutRecognizer

        recognizer = PPDocLayoutRecognizer.__new__(PPDocLayoutRecognizer)
        recognizer.label_list = PPDocLayoutRecognizer.labels
        recognizer.input_names = ["image"]
        recognizer.input_shape = [640, 640]

        outputs = np.array(
            [
                [10, 20, 110, 120, 0.92, 0],
                [30, 40, 130, 140, 0.91, 8],
                [50, 60, 150, 160, 0.15, 4],
            ],
            dtype=np.float32,
        )
        inputs = {"scale_factor": [2.0, 3.0], "pad": [0.0, 0.0]}
        boxes = recognizer.postprocess(outputs, inputs, thr=0.2)

        self.assertEqual(
            [
                {"type": "text", "bbox": [20.0, 60.0, 220.0, 360.0], "score": 0.9200000166893005},
                {"type": "table", "bbox": [60.0, 120.0, 260.0, 420.0], "score": 0.9100000262260437},
            ],
            boxes,
        )

    def test_pdf_parser_layout_engine_uses_ppdoclayout_and_separates_cache_key(self):
        from deepdoc.parser import pdf_parser

        class FakeOcr:
            pass

        class FakeLegacyLayout:
            def __init__(self, domain):
                self.domain = domain

        class FakePPDocLayout(FakeLegacyLayout):
            pass

        class FakeTable:
            pass

        class FakeBooster:
            def load_model(self, _path):
                pass

        pdf_parser.clear_shared_pdf_parser_components()
        try:
            with patch.object(pdf_parser, "OCR", FakeOcr), patch.object(
                pdf_parser, "LayoutRecognizer", FakeLegacyLayout
            ), patch.object(pdf_parser, "PPDocLayoutRecognizer", FakePPDocLayout), patch.object(
                pdf_parser, "TableStructureRecognizer", FakeTable
            ), patch.object(pdf_parser.xgb, "Booster", FakeBooster), patch.object(
                pdf_parser, "ensure_groups", return_value="/models"
            ), patch.object(
                pdf_parser, "pip_install_torch", return_value=None
            ):
                with patch.dict(pdf_parser.os.environ, {"DEEPDOC_LAYOUT_ENGINE": "legacy"}, clear=False):
                    legacy = pdf_parser.DeepDocPdfParser()
                with patch.dict(pdf_parser.os.environ, {"DEEPDOC_LAYOUT_ENGINE": "ppdoclayout"}, clear=False):
                    ppdoc = pdf_parser.DeepDocPdfParser()

            self.assertIsInstance(legacy.layouter, FakeLegacyLayout)
            self.assertIsInstance(ppdoc.layouter, FakePPDocLayout)
            self.assertEqual(2, pdf_parser.shared_pdf_parser_component_state()["cached_component_count"])
            engines = {
                component["layout_engine"]
                for component in pdf_parser.shared_pdf_parser_component_state()["components"]
            }
            self.assertEqual({"legacy", "ppdoclayout"}, engines)
        finally:
            pdf_parser.clear_shared_pdf_parser_components()

    def test_formula_mode_pp_formula_net_uses_formula_v2_group_and_is_default_off(self):
        from deepdoc.vision import formula_recognizer

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual("rapidlatex", formula_recognizer._formula_mode())

        with patch.dict(os.environ, {"DEEPDOC_FORMULA_MODE": "pp_formula_net_s"}, clear=True):
            with patch.object(formula_recognizer, "ensure_groups", return_value="/models") as ensure_groups:
                paths = formula_recognizer._resolve_model_paths()

        self.assertEqual("formula_v2", ensure_groups.call_args.args[0])
        self.assertEqual("/models/formula/pp_formula_net_s.onnx", paths["model_path"])

    def test_pp_formula_net_recognizer_uses_paddlex_and_extracts_rec_formula(self):
        from deepdoc.vision import formula_recognizer

        calls = []

        class FakePaddleXFormulaModel:
            def predict(self, input, batch_size=1):
                calls.append({"input": input, "batch_size": batch_size})
                return [{"res": {"rec_formula": r"\frac{a}{b}"}}]

        def fake_create_model(**kwargs):
            calls.append({"create_model": kwargs})
            return FakePaddleXFormulaModel()

        fake_paddlex = types.SimpleNamespace(create_model=fake_create_model)

        with tempfile.TemporaryDirectory(prefix="deepdoc-pp-formula-") as temp_dir:
            model_root = Path(temp_dir)
            (model_root / "formula").mkdir()
            (model_root / "formula" / "pp_formula_net_s.onnx").write_bytes(b"fake-model")

            with patch.dict(
                os.environ,
                {
                    "DEEPDOC_FORMULA_MODE": "pp_formula_net_s",
                    "DEEPDOC_PP_FORMULA_NET_DEVICE": "cpu",
                    "DEEPDOC_PP_FORMULA_NET_MODEL_NAME": "PP-FormulaNet-S",
                },
                clear=True,
            ):
                with patch.object(formula_recognizer, "ensure_groups", return_value=str(model_root)):
                    with patch.dict(sys.modules, {"paddlex": fake_paddlex}):
                        recognizer = formula_recognizer.FormulaRecognizer()
                        latex, elapsed = recognizer.predict(np.zeros((12, 16, 3), dtype=np.uint8))

        self.assertEqual(r"\frac{a}{b}", latex)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual("PP-FormulaNet-S", calls[0]["create_model"]["model_name"])
        self.assertEqual(str(model_root / "formula"), calls[0]["create_model"]["model_dir"])
        self.assertEqual("cpu", calls[0]["create_model"]["device"])
        self.assertEqual(1, calls[1]["batch_size"])
        self.assertTrue(str(calls[1]["input"]).endswith(".png"))

    def test_eval_and_profile_tools_expose_required_cli_entrypoints(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")
        profile_tool = importlib.import_module("tools.profile_pipeline")

        self.assertTrue(callable(eval_tool.compute_text_edit_distance))
        self.assertTrue(callable(eval_tool.evaluate_dataset))
        self.assertTrue(callable(eval_tool.license_gate_report))
        self.assertTrue(callable(eval_tool.validate_dataset))
        self.assertTrue(callable(profile_tool.profile_pipeline))

        with tempfile.TemporaryDirectory(prefix="deepdoc-empty-eval-") as temp_dir:
            with self.assertRaisesRegex(SystemExit, "No evaluation samples"):
                eval_tool.evaluate_dataset(engine="deepdoc", dataset=temp_dir, out=None)

    def test_eval_tool_validates_dataset_ground_truth_contract(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")

        with tempfile.TemporaryDirectory(prefix="deepdoc-eval-contract-") as temp_dir:
            dataset = Path(temp_dir)
            (dataset / "broken.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "broken.gt.blocks.json").write_text(json.dumps({"blocks": [{"block_type": "text"}]}), encoding="utf-8")
            (dataset / "broken.gt.chunks.json").write_text(json.dumps({"chunks": [{"id": "missing-text"}]}), encoding="utf-8")
            (dataset / "broken.gt.fields.json").write_text(json.dumps({"fields": [{"name": "amount"}]}), encoding="utf-8")
            (dataset / "broken.gt.formulas.json").write_text(json.dumps({"formulas": [{"page": 1}]}), encoding="utf-8")

            report = eval_tool.validate_dataset(dataset)

        self.assertEqual("failed", report["status"])
        self.assertEqual(1, report["sample_count"])
        problems = "\n".join(report["problems"])
        self.assertIn("broken.gt.blocks.json: blocks[0] missing text", problems)
        self.assertIn("broken.gt.chunks.json: chunks[0] missing text/content", problems)
        self.assertIn("broken.gt.fields.json: fields[0] missing value", problems)
        self.assertIn("broken.gt.formulas.json: formulas[0] missing latex/formula/text/content", problems)

    def test_eval_tool_discovers_recursive_dataset_samples_with_relative_names(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")

        with tempfile.TemporaryDirectory(prefix="deepdoc-eval-recursive-contract-") as temp_dir:
            dataset = Path(temp_dir)
            nested = dataset / "contracts"
            nested.mkdir()
            (dataset / "root-sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (nested / "nested-sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (nested / "nested-sample.gt.txt").write_text("nested sample text", encoding="utf-8")

            report = eval_tool.validate_dataset(dataset)

        self.assertEqual("ok", report["status"])
        self.assertEqual(2, report["sample_count"])
        self.assertEqual(["contracts/nested-sample", "root-sample"], [sample["name"] for sample in report["samples"]])
        self.assertTrue(report["samples"][0]["pdf_path"].endswith("contracts/nested-sample.pdf"))

    def test_eval_tool_validate_dataset_cli_writes_out_file_and_returns_failed_status(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="deepdoc-eval-contract-cli-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            output_path = root / "out" / "dataset-contract.json"
            dataset.mkdir()
            (dataset / "broken.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "broken.gt.blocks.json").write_text(
                json.dumps({"blocks": [{"block_type": "text"}]}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/eval_omnidocbench.py",
                    "--validate-dataset",
                    "--dataset",
                    str(dataset),
                    "--out",
                    str(output_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            file_payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(1, result.returncode, result.stderr)
        stdout_payload = json.loads(result.stdout)
        self.assertEqual(stdout_payload, file_payload)
        self.assertEqual("2026-06-08.cpu-pipeline-dataset-contract.v1", file_payload["schema_version"])
        self.assertEqual("failed", file_payload["status"])
        self.assertIn("broken.gt.blocks.json: blocks[0] missing text", file_payload["problems"])

    def test_eval_tool_preflights_dataset_contract_before_parsing(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")

        with tempfile.TemporaryDirectory(prefix="deepdoc-eval-contract-preflight-") as temp_dir:
            dataset = Path(temp_dir)
            (dataset / "broken.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "broken.gt.blocks.json").write_text(
                json.dumps({"blocks": [{"block_type": "text"}]}),
                encoding="utf-8",
            )

            with patch.object(eval_tool, "_evaluate_sample", side_effect=AssertionError("parser should not run")):
                with self.assertRaisesRegex(
                    SystemExit,
                    "Dataset contract validation failed: broken.gt.blocks.json: blocks\\[0\\] missing text",
                ):
                    eval_tool.evaluate_dataset(engine="deepdoc", dataset=dataset, out=None)

    def test_cpu_pipeline_readiness_reports_missing_models_dataset_and_ab_reports(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            model_root.mkdir()
            dataset.mkdir()
            (dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")

            with patch.object(readiness_tool, "_pdf_page_count", return_value=1):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={},
                )

        self.assertEqual("failed", payload["status"])
        self.assertIn("core_v5", payload["model_groups"])
        self.assertIn("det_v5.onnx", payload["model_groups"]["core_v5"]["missing_files"])
        self.assertIn("table_v2", payload["model_groups"])
        self.assertIn("table/slanet_plus.onnx", payload["model_groups"]["table_v2"]["missing_files"])
        self.assertEqual(1, payload["dataset"]["pdf_count"])
        self.assertEqual(1, payload["dataset"]["page_count"])
        self.assertFalse(payload["dataset"]["meets_min_pages"])
        self.assertEqual("missing", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("missing", payload["ab_reports"]["ocr_v5"]["baseline"]["status"])
        self.assertEqual("missing", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertEqual("missing", payload["ab_reports"]["layout_v2"]["status"])
        self.assertIn("dataset_pages_below_minimum", payload["failed_gates"])
        self.assertIn("missing_model_group:core_v5", payload["failed_gates"])
        self.assertIn("missing_model_group:table_v2", payload["failed_gates"])
        self.assertIn("missing_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_fails_bad_dataset_ground_truth_contract(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-contract-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "broken.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "broken.gt.chunks.json").write_text(
                json.dumps({"chunks": [{"id": "missing-text"}]}),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=root / "models",
                    dataset=dataset,
                    min_pages=100,
                    report_paths={},
                )

        self.assertEqual("failed", payload["status"])
        self.assertEqual("failed", payload["dataset_contract"]["status"])
        self.assertIn("broken.gt.chunks.json: chunks[0] missing text/content", payload["dataset_contract"]["problems"])
        self.assertNotIn("dataset_pages_below_minimum", payload["failed_gates"])
        self.assertIn("dataset_contract_failed", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_paired_ab_reports(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-paired-") as temp_dir:
            from common.model_store import build_model_manifest

            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            candidate_report = root / "ocr-v5.json"
            model_root.mkdir()
            dataset.mkdir()
            (dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = {
                "schema_version": "2026-06-08.cpu-pipeline-dataset-contract.v1",
                "status": "ok",
                "dataset": str(dataset),
                "sample_count": 1,
                "problems": [],
                "samples": [{"name": "sample", "pdf_path": str(dataset / "sample.pdf")}],
            }
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "engine": "deepdoc",
                        "dataset": str(dataset),
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {
                            "sample_count": 1,
                            "mean_character_error_rate": 0.1,
                            "mean_word_error_rate": 0.2,
                        },
                        "model_manifest": build_model_manifest(model_root=model_root, groups="core_v5"),
                        "dataset_contract": dataset_contract,
                        "license_gate": self._license_gate_payload(),
                        "samples": [
                            {
                                "name": "sample",
                                "pdf_path": str(dataset / "sample.pdf"),
                                "character_error_rate": 0.1,
                                "word_error_rate": 0.2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=model_root,
                dataset=dataset,
                min_pages=0,
                report_paths={"ocr_v5_candidate": candidate_report},
            )

        self.assertEqual("failed", payload["status"])
        self.assertEqual("missing", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("missing", payload["ab_reports"]["ocr_v5"]["baseline"]["status"])
        self.assertEqual("ok", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn("missing_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_ab_report_pipeline_config(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-config-") as temp_dir:
            root = Path(temp_dir)
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            report_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
            }
            baseline_report.write_text(
                json.dumps({**report_payload, "pipeline_config": {"ocr_version": "v4"}}),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps({**report_payload, "pipeline_config": {"ocr_version": "v4"}}),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={
                    "ocr_v5_baseline": baseline_report,
                    "ocr_v5_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn(
            "pipeline_config.ocr_version expected v5, got v4",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_report_schema_version(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-schema-") as temp_dir:
            root = Path(temp_dir)
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            common_summary = {
                "sample_count": 1,
                "mean_character_error_rate": 0.1,
                "mean_word_error_rate": 0.2,
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema_version": "legacy-eval-report",
                        "dataset": "tools/eval_datasets/biz",
                        "pipeline_config": {"ocr_version": "v4"},
                        "summary": common_summary,
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "dataset": "tools/eval_datasets/biz",
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": common_summary,
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={
                    "ocr_v5_baseline": baseline_report,
                    "ocr_v5_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["baseline"]["status"])
        self.assertIn(
            "schema_version expected 2026-06-08.cpu-pipeline-eval.v1, got legacy-eval-report",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-schema-") as temp_dir:
            root = Path(temp_dir)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "legacy-profile-report",
                        "total_elapsed_seconds": 4.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "layout",
                            "slowest_stage_elapsed_seconds": 2.0,
                            "slowest_stage_share": 0.5,
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 1.0},
                            {"stage": "layout", "elapsed_seconds": 2.0},
                            {"stage": "table", "elapsed_seconds": 0.25},
                            {"stage": "text_merge", "elapsed_seconds": 0.25},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.25},
                            {"stage": "reading_order", "elapsed_seconds": 0.25},
                            {"stage": "extract_assets", "elapsed_seconds": 0.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={"profile": profile},
            )

        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "schema_version expected 2026-06-08.cpu-pipeline-profile.v1, got legacy-profile-report",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_license_gate_report(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-license-") as temp_dir:
            root = Path(temp_dir)
            license_report = root / "license-gate.json"
            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={},
            )

        self.assertEqual("missing", payload["license_gate"]["status"])
        self.assertIn("missing_license_gate_report", payload["failed_gates"])

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-license-") as temp_dir:
            root = Path(temp_dir)
            license_report = root / "license-gate.json"
            license_report.write_text(
                json.dumps(
                    {
                        "schema_version": "legacy-license-gate",
                        "status": "passed",
                        "blocked": [],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={"license_gate": license_report},
            )

        self.assertEqual("failed", payload["license_gate"]["status"])
        self.assertIn(
            "schema_version expected 2026-06-08.cpu-pipeline-license-gate.v1, got legacy-license-gate",
            payload["license_gate"]["problems"],
        )
        self.assertIn("failed_license_gate_report", payload["failed_gates"])

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-license-") as temp_dir:
            root = Path(temp_dir)
            license_report = root / "license-gate.json"
            license_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-license-gate.v1",
                        "status": "review",
                        "blocked": [{"name": "DocLayout-YOLO", "license": "AGPL-3.0"}],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={"license_gate": license_report},
            )

        self.assertEqual("failed", payload["license_gate"]["status"])
        self.assertIn("license gate status expected passed, got review", payload["license_gate"]["problems"])

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-license-") as temp_dir:
            root = Path(temp_dir)
            license_report = root / "license-gate.json"
            license_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-license-gate.v1",
                        "status": "passed",
                        "allowed": [{"name": "PP-OCRv5", "license": "Apache-2.0"}],
                        "blocked": [],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={"license_gate": license_report},
            )

        self.assertEqual("failed", payload["license_gate"]["status"])
        self.assertIn(
            "license gate allowed missing required candidates: Docling, PP-DocLayout, PP-FormulaNet-S, RapidLaTeXOCR, RapidTable, SLANet-plus",
            payload["license_gate"]["problems"],
        )
        self.assertIn(
            "license gate blocked missing required candidates: DocLayout-YOLO, Marker, Surya, texify",
            payload["license_gate"]["problems"],
        )
        self.assertIn("failed_license_gate_report", payload["failed_gates"])

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-license-") as temp_dir:
            root = Path(temp_dir)
            license_report = root / "license-gate.json"
            license_payload = self._license_gate_payload()
            for item in license_payload["allowed"]:
                if item["name"] == "PP-OCRv5":
                    item["license"] = "MIT"
                    break
            license_report.write_text(json.dumps(license_payload), encoding="utf-8")

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={"license_gate": license_report},
            )

        self.assertEqual("failed", payload["license_gate"]["status"])
        self.assertIn(
            "license gate allowed PP-OCRv5.license expected 'Apache-2.0', got 'MIT'",
            payload["license_gate"]["problems"],
        )
        self.assertIn("failed_license_gate_report", payload["failed_gates"])

    def test_cpu_pipeline_readiness_derives_license_gate_expectations_from_current_provenance(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        patched_provenance = dict(model_store.MODEL_GROUP_PROVENANCE)
        patched_provenance["core_v5"] = {
            **patched_provenance["core_v5"],
            "license": "BSD-3-Clause",
            "license_status": "allowed",
        }

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-license-provenance-") as temp_dir:
            root = Path(temp_dir)
            license_report = root / "license-gate.json"
            with patch.object(model_store, "MODEL_GROUP_PROVENANCE", patched_provenance):
                license_report.write_text(json.dumps(eval_tool.license_gate_report()), encoding="utf-8")
                payload = readiness_tool.check_readiness(
                    model_root=root / "models",
                    dataset=root / "dataset",
                    min_pages=0,
                    report_paths={"license_gate": license_report},
                )

        allowed_by_name = {item["name"]: item for item in payload["license_gate"]["allowed"]}
        self.assertEqual("BSD-3-Clause", allowed_by_name["PP-OCRv5"]["license"])
        self.assertEqual("ok", payload["license_gate"]["status"])
        self.assertNotIn("failed_license_gate_report", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_eval_reports_to_embed_license_gate(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-license-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            dataset.mkdir()
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
            }
            baseline_report.write_text(
                json.dumps({**common_payload, "pipeline_config": {"ocr_version": "v4"}}),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "license_gate": {
                            "schema_version": "2026-06-08.cpu-pipeline-license-gate.v1",
                            "status": "passed",
                            "allowed": [{"name": "PP-OCRv5", "license": "Apache-2.0"}],
                            "blocked": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=dataset,
                min_pages=0,
                report_paths={
                    "ocr_v5_baseline": baseline_report,
                    "ocr_v5_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertIn(
            "missing license_gate report field",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertIn(
            "license_gate allowed missing required candidates: Docling, PP-DocLayout, PP-FormulaNet-S, RapidLaTeXOCR, RapidTable, SLANet-plus",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn(
            "license_gate blocked missing required candidates: DocLayout-YOLO, Marker, Surya, texify",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_eval_reports_to_embed_current_dataset_contract(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-contract-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            dataset.mkdir()
            (dataset / "sample-a.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "sample-b.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 2,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "samples": [{"name": "sample-a"}, {"name": "sample-b"}],
                "license_gate": self._license_gate_payload(),
            }
            stale_contract = {
                "schema_version": "2026-06-08.cpu-pipeline-dataset-contract.v1",
                "status": "ok",
                "dataset": str(dataset),
                "sample_count": 1,
                "problems": [],
                "samples": [{"name": "sample-a", "pdf_path": str(dataset / "sample-a.pdf")}],
            }
            baseline_report.write_text(
                json.dumps({**common_payload, "pipeline_config": {"ocr_version": "v4"}}),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "dataset_contract": stale_contract,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=root / "models",
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertIn(
            "missing dataset_contract report field",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertIn(
            "dataset_contract sample_count must match summary.sample_count: dataset_contract=1, summary=2",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn(
            "dataset_contract samples names must match readiness dataset contract: missing=sample-b, unexpected=-",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_rejects_eval_dataset_contract_pdf_path_mismatch(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-contract-pdf-path-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            previous_dataset = root / "previous-dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            previous_dataset.mkdir()
            current_pdf = dataset / "sample.pdf"
            stale_pdf = previous_dataset / "sample.pdf"
            current_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            stale_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            stale_contract = {
                **dataset_contract,
                "samples": [{"name": "sample", "pdf_path": str(stale_pdf)}],
            }
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "engine": "deepdoc",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "license_gate": self._license_gate_payload(),
                "samples": [
                    {
                        "name": "sample",
                        "engine": "deepdoc",
                        "pdf_path": str(current_pdf),
                        "character_error_rate": 0.1,
                        "word_error_rate": 0.2,
                    }
                ],
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core",)),
                        "dataset_contract": dataset_contract,
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                        "dataset_contract": stale_contract,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("ok", payload["ab_reports"]["ocr_v5"]["baseline"]["status"])
        self.assertIn(
            f"dataset_contract samples[0].pdf_path must match readiness dataset contract sample sample: "
            f"expected={current_pdf}, got={stale_pdf}",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_eval_dataset_contract_pdf_paths(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-contract-pdf-path-required-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            current_pdf = dataset / "sample.pdf"
            current_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            contract_without_pdf_path = {
                **dataset_contract,
                "samples": [{"name": "sample"}],
            }
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "engine": "deepdoc",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "license_gate": self._license_gate_payload(),
                "samples": [
                    {
                        "name": "sample",
                        "engine": "deepdoc",
                        "pdf_path": str(current_pdf),
                        "character_error_rate": 0.1,
                        "word_error_rate": 0.2,
                    }
                ],
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core",)),
                        "dataset_contract": dataset_contract,
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                        "dataset_contract": contract_without_pdf_path,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("ok", payload["ab_reports"]["ocr_v5"]["baseline"]["status"])
        self.assertIn(
            f"dataset_contract samples[0].pdf_path is required for readiness dataset contract sample sample: "
            f"expected={current_pdf}",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_eval_dataset_contract_samples_to_be_objects(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-contract-sample-object-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            contract_with_non_object_sample = {
                **dataset_contract,
                "samples": ["sample"],
            }
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "engine": "deepdoc",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "license_gate": self._license_gate_payload(),
                "samples": [
                    {
                        "name": "sample",
                        "engine": "deepdoc",
                        "pdf_path": str(pdf_path),
                        "character_error_rate": 0.1,
                        "word_error_rate": 0.2,
                    }
                ],
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core",)),
                        "dataset_contract": dataset_contract,
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                        "dataset_contract": contract_with_non_object_sample,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn(
            "dataset_contract samples[0] must be a JSON object",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_ab_report_dataset_and_sample_count_match(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-ab-match-") as temp_dir:
            root = Path(temp_dir)
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            base_summary = {
                "mean_character_error_rate": 0.1,
                "mean_word_error_rate": 0.2,
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "dataset": "tools/eval_datasets/biz_a",
                        "pipeline_config": {"ocr_version": "v4"},
                        "summary": {"sample_count": 2, **base_summary},
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "dataset": "tools/eval_datasets/biz_b",
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {"sample_count": 1, **base_summary},
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={
                    "ocr_v5_baseline": baseline_report,
                    "ocr_v5_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertIn(
            "A/B reports must use the same dataset: baseline=tools/eval_datasets/biz_a, candidate=tools/eval_datasets/biz_b",
            payload["ab_reports"]["ocr_v5"]["problems"],
        )
        self.assertIn(
            "A/B reports must use the same sample_count: baseline=2, candidate=1",
            payload["ab_reports"]["ocr_v5"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_eval_report_sample_count_matches_samples(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-samples-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            dataset.mkdir()
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 2,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "samples": [{"name": "only-one-sample"}],
            }
            baseline_report.write_text(
                json.dumps({**common_payload, "pipeline_config": {"ocr_version": "v4"}}),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps({**common_payload, "pipeline_config": {"ocr_version": "v5"}}),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=dataset,
                min_pages=0,
                report_paths={
                    "ocr_v5_baseline": baseline_report,
                    "ocr_v5_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertIn(
            "summary sample_count must match samples length: sample_count=2, samples=1",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertIn(
            "summary sample_count must match samples length: sample_count=2, samples=1",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_integer_sample_counts(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-integer-samples-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "engine": "deepdoc",
                        "dataset": str(dataset),
                        "dataset_contract": {**dataset_contract, "sample_count": 1.5},
                        "license_gate": self._license_gate_payload(),
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {
                            "sample_count": 1.5,
                            "mean_character_error_rate": 0.1,
                            "mean_word_error_rate": 0.2,
                        },
                        "samples": [
                            {
                                "name": "sample",
                                "engine": "deepdoc",
                                "pdf_path": str(pdf_path),
                                "character_error_rate": 0.1,
                                "word_error_rate": 0.2,
                            }
                        ],
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"ocr_v5_candidate": candidate_report},
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn(
            "summary sample_count must be an integer",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn(
            "dataset_contract sample_count must be an integer",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("missing_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_eval_summary_metrics_match_samples(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-summary-metrics-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            (dataset / "sample-a.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "sample-b.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset, ["sample-a", "sample-b"])
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "engine": "deepdoc",
                "dataset": str(dataset),
                "dataset_contract": dataset_contract,
                "license_gate": self._license_gate_payload(),
                "samples": [
                    {
                        "name": "sample-a",
                        "engine": "deepdoc",
                        "pdf_path": str(dataset / "sample-a.pdf"),
                        "character_error_rate": 0.1,
                        "word_error_rate": 0.4,
                    },
                    {
                        "name": "sample-b",
                        "engine": "deepdoc",
                        "pdf_path": str(dataset / "sample-b.pdf"),
                        "character_error_rate": 0.3,
                        "word_error_rate": 0.6,
                    },
                ],
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "summary": {
                            "sample_count": 2,
                            "mean_character_error_rate": 0.9,
                            "mean_word_error_rate": 0.5,
                        },
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core",)),
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {
                            "sample_count": 2,
                            "mean_character_error_rate": 0.2,
                            "mean_word_error_rate": 0.5,
                        },
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=50):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertIn(
            "summary mean_character_error_rate must match samples mean: summary=0.9, samples=0.2",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertEqual("ok", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_sample_metrics_for_summary_metrics(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-missing-sample-metrics-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "engine": "deepdoc",
                        "dataset": str(dataset),
                        "dataset_contract": dataset_contract,
                        "license_gate": self._license_gate_payload(),
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {
                            "sample_count": 1,
                            "mean_character_error_rate": 0.1,
                            "mean_word_error_rate": 0.2,
                        },
                        "samples": [
                            {
                                "name": "sample",
                                "engine": "deepdoc",
                                "pdf_path": str(pdf_path),
                            }
                        ],
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"ocr_v5_candidate": candidate_report},
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn(
            "samples missing metric values for summary mean_character_error_rate: sample_field=character_error_rate",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn(
            "samples missing metric values for summary mean_word_error_rate: sample_field=word_error_rate",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("missing_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_every_sample_metric_for_summary_metrics(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-partial-sample-metrics-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            (dataset / "sample-a.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "sample-b.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset, ["sample-a", "sample-b"])
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "engine": "deepdoc",
                        "dataset": str(dataset),
                        "dataset_contract": dataset_contract,
                        "license_gate": self._license_gate_payload(),
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {
                            "sample_count": 2,
                            "mean_character_error_rate": 0.1,
                            "mean_word_error_rate": 0.2,
                        },
                        "samples": [
                            {
                                "name": "sample-a",
                                "engine": "deepdoc",
                                "pdf_path": str(dataset / "sample-a.pdf"),
                                "character_error_rate": 0.1,
                                "word_error_rate": 0.2,
                            },
                            {
                                "name": "sample-b",
                                "engine": "deepdoc",
                                "pdf_path": str(dataset / "sample-b.pdf"),
                            },
                        ],
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"ocr_v5_candidate": candidate_report},
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn(
            "samples[1].character_error_rate is required for summary mean_character_error_rate",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn(
            "samples[1].word_error_rate is required for summary mean_word_error_rate",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("missing_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_sample_entries_to_be_objects(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-sample-object-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            (dataset / "sample-a.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "sample-b.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset, ["sample-a", "sample-b"])
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "engine": "deepdoc",
                        "dataset": str(dataset),
                        "dataset_contract": dataset_contract,
                        "license_gate": self._license_gate_payload(),
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {
                            "sample_count": 2,
                            "mean_character_error_rate": 0.1,
                            "mean_word_error_rate": 0.2,
                        },
                        "samples": [
                            {
                                "name": "sample-a",
                                "engine": "deepdoc",
                                "pdf_path": str(dataset / "sample-a.pdf"),
                                "character_error_rate": 0.1,
                                "word_error_rate": 0.2,
                            },
                            "sample-b",
                        ],
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"ocr_v5_candidate": candidate_report},
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn(
            "samples[1] must be a JSON object",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("missing_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_rejects_non_finite_summary_and_sample_metrics(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-non-finite-metrics-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "engine": "deepdoc",
                        "dataset": str(dataset),
                        "dataset_contract": dataset_contract,
                        "license_gate": self._license_gate_payload(),
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {
                            "sample_count": 1,
                            "mean_character_error_rate": float("nan"),
                            "mean_word_error_rate": 0.2,
                        },
                        "samples": [
                            {
                                "name": "sample",
                                "engine": "deepdoc",
                                "pdf_path": str(pdf_path),
                                "character_error_rate": float("nan"),
                                "word_error_rate": float("inf"),
                            }
                        ],
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"ocr_v5_candidate": candidate_report},
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn(
            "samples[0].character_error_rate must be finite",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn(
            "samples[0].word_error_rate must be finite",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn(
            "summary mean_character_error_rate must be finite",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("missing_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_rejects_boolean_summary_and_sample_metrics(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-bool-metrics-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "engine": "deepdoc",
                        "dataset": str(dataset),
                        "dataset_contract": dataset_contract,
                        "license_gate": self._license_gate_payload(),
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {
                            "sample_count": 1,
                            "mean_character_error_rate": True,
                            "mean_word_error_rate": 0.2,
                        },
                        "samples": [
                            {
                                "name": "sample",
                                "engine": "deepdoc",
                                "pdf_path": str(pdf_path),
                                "character_error_rate": True,
                                "word_error_rate": False,
                            }
                        ],
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"ocr_v5_candidate": candidate_report},
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn(
            "samples[0].character_error_rate must be numeric",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn(
            "samples[0].word_error_rate must be numeric",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn(
            "summary mean_character_error_rate must be numeric",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("missing_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_eval_report_summary_sample_count(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-sample-count-required-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            current_pdf = dataset / "sample.pdf"
            current_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "engine": "deepdoc",
                "dataset": str(dataset),
                "dataset_contract": dataset_contract,
                "license_gate": self._license_gate_payload(),
                "samples": [
                    {
                        "name": "sample",
                        "engine": "deepdoc",
                        "pdf_path": str(current_pdf),
                        "character_error_rate": 0.1,
                        "word_error_rate": 0.2,
                    }
                ],
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "summary": {
                            "mean_character_error_rate": 0.1,
                            "mean_word_error_rate": 0.2,
                        },
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core",)),
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {
                            "sample_count": 1,
                            "mean_character_error_rate": 0.1,
                            "mean_word_error_rate": 0.2,
                        },
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertIn(
            "missing summary field: sample_count",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertEqual("ok", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_eval_reports_to_include_samples_detail(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-samples-required-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            (dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "dataset_contract": dataset_contract,
                "license_gate": self._license_gate_payload(),
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core",)),
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertIn(
            "missing samples report field",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertIn(
            "missing samples report field",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_eval_reports_to_use_deepdoc_engine(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-eval-engine-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            (dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "dataset_contract": dataset_contract,
                "license_gate": self._license_gate_payload(),
                "samples": [{"name": "sample"}],
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core",)),
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "engine": "plain",
                        "pipeline_config": {"ocr_version": "v5"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertIn(
            "missing engine report field",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertIn(
            "engine expected deepdoc, got plain",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_rejects_eval_sample_engine_mismatch(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-sample-engine-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            (dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "engine": "deepdoc",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "dataset_contract": dataset_contract,
                "license_gate": self._license_gate_payload(),
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core",)),
                        "samples": [
                            {
                                "name": "sample",
                                "engine": "deepdoc",
                                "pdf_path": str(dataset / "sample.pdf"),
                                "character_error_rate": 0.1,
                                "word_error_rate": 0.2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                        "samples": [{"name": "sample", "engine": "plain", "pdf_path": str(dataset / "sample.pdf")}],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("ok", payload["ab_reports"]["ocr_v5"]["baseline"]["status"])
        self.assertIn(
            "samples[0].engine expected deepdoc, got plain",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_rejects_eval_sample_pdf_path_mismatch(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-sample-pdf-path-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            previous_dataset = root / "previous-dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            previous_dataset.mkdir()
            current_pdf = dataset / "sample.pdf"
            stale_pdf = previous_dataset / "sample.pdf"
            current_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            stale_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "engine": "deepdoc",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "dataset_contract": dataset_contract,
                "license_gate": self._license_gate_payload(),
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core",)),
                        "samples": [
                            {
                                "name": "sample",
                                "engine": "deepdoc",
                                "pdf_path": str(current_pdf),
                                "character_error_rate": 0.1,
                                "word_error_rate": 0.2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                        "samples": [{"name": "sample", "engine": "deepdoc", "pdf_path": str(stale_pdf)}],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("ok", payload["ab_reports"]["ocr_v5"]["baseline"]["status"])
        self.assertIn(
            f"samples[0].pdf_path must match dataset contract sample sample: expected={current_pdf}, got={stale_pdf}",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_eval_sample_pdf_paths(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-sample-pdf-path-required-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            current_pdf = dataset / "sample.pdf"
            current_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            dataset_contract = self._dataset_contract_payload(dataset)
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "engine": "deepdoc",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "dataset_contract": dataset_contract,
                "license_gate": self._license_gate_payload(),
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core",)),
                        "samples": [
                            {
                                "name": "sample",
                                "engine": "deepdoc",
                                "pdf_path": str(current_pdf),
                                "character_error_rate": 0.1,
                                "word_error_rate": 0.2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core_v5",)),
                        "samples": [{"name": "sample", "engine": "deepdoc"}],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("ok", payload["ab_reports"]["ocr_v5"]["baseline"]["status"])
        self.assertIn(
            f"samples[0].pdf_path is required for dataset contract sample sample: expected={current_pdf}",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_ab_reports_to_match_current_dataset(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-ab-dataset-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset-current"
            previous_dataset = root / "dataset-previous"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            dataset.mkdir()
            previous_dataset.mkdir()
            (dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (previous_dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            common_summary = {
                "sample_count": 1,
                "mean_character_error_rate": 0.1,
                "mean_word_error_rate": 0.2,
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "dataset": str(previous_dataset),
                        "pipeline_config": {"ocr_version": "v4"},
                        "summary": common_summary,
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "dataset": str(previous_dataset),
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": common_summary,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=root / "models",
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertIn(
            f"A/B report dataset must match readiness dataset: expected={dataset}, got={previous_dataset}",
            payload["ab_reports"]["ocr_v5"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_ab_sample_count_to_match_dataset_contract(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-ab-samples-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            dataset.mkdir()
            (dataset / "sample-a.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "sample-b.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            common_summary = {
                "sample_count": 1,
                "mean_character_error_rate": 0.1,
                "mean_word_error_rate": 0.2,
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "dataset": str(dataset),
                        "pipeline_config": {"ocr_version": "v4"},
                        "summary": common_summary,
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                        "dataset": str(dataset),
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": common_summary,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=root / "models",
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertEqual(2, payload["dataset_contract"]["sample_count"])
        self.assertIn(
            "A/B report sample_count must match dataset contract: expected=2, got=1",
            payload["ab_reports"]["ocr_v5"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_ab_sample_names_to_match_dataset_contract(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-ab-sample-names-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            dataset.mkdir()
            (dataset / "sample-a.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "sample-b.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 2,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "samples": [{"name": "sample-a"}, {"name": "stale-sample"}],
            }
            baseline_report.write_text(
                json.dumps({**common_payload, "pipeline_config": {"ocr_version": "v4"}}),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps({**common_payload, "pipeline_config": {"ocr_version": "v5"}}),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=root / "models",
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertIn(
            "samples names must match dataset contract: missing=sample-b, unexpected=stale-sample",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertIn(
            "samples names must match dataset contract: missing=sample-b, unexpected=stale-sample",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_uses_recursive_dataset_contract_sample_names(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-recursive-samples-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            nested = dataset / "contracts"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            nested.mkdir(parents=True)
            (dataset / "root-sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (nested / "nested-sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 2,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "samples": [{"name": "root-sample"}, {"name": "nested-sample"}],
            }
            baseline_report.write_text(
                json.dumps({**common_payload, "pipeline_config": {"ocr_version": "v4"}}),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps({**common_payload, "pipeline_config": {"ocr_version": "v5"}}),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=root / "models",
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertEqual(["contracts/nested-sample", "root-sample"], [sample["name"] for sample in payload["dataset_contract"]["samples"]])
        self.assertIn(
            "samples names must match dataset contract: missing=contracts/nested-sample, unexpected=nested-sample",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_fails_when_candidate_metrics_regress(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-regression-") as temp_dir:
            root = Path(temp_dir)
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": "tools/eval_datasets/biz",
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "summary": {
                            "sample_count": 1,
                            "mean_character_error_rate": 0.1,
                            "mean_word_error_rate": 0.2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "summary": {
                            "sample_count": 1,
                            "mean_character_error_rate": 0.12,
                            "mean_word_error_rate": 0.25,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={
                    "ocr_v5_baseline": baseline_report,
                    "ocr_v5_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertIn(
            "candidate mean_character_error_rate regressed: baseline=0.1, candidate=0.12",
            payload["ab_reports"]["ocr_v5"]["problems"],
        )
        self.assertIn(
            "candidate mean_word_error_rate regressed: baseline=0.2, candidate=0.25",
            payload["ab_reports"]["ocr_v5"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_layout_cross_page_chunk_and_business_field_metrics(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-layout-metrics-") as temp_dir:
            root = Path(temp_dir)
            baseline_report = root / "layout-baseline.json"
            candidate_report = root / "layout-candidate.json"
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": "tools/eval_datasets/biz",
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"layout_engine": "legacy", "reading_order_strategy": "legacy"},
                        "summary": {
                            "sample_count": 1,
                            "mean_block_type_f1": 0.8,
                            "mean_reading_order_normalized_edit_distance": 0.2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"layout_engine": "ppdoclayout", "reading_order_strategy": "rules"},
                        "summary": {
                            "sample_count": 1,
                            "mean_block_type_f1": 0.9,
                            "mean_reading_order_normalized_edit_distance": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={
                    "layout_v2_baseline": baseline_report,
                    "layout_v2_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["layout_v2"]["baseline"]["status"])
        self.assertIn(
            "missing summary field: mean_chunk_text_coverage",
            payload["ab_reports"]["layout_v2"]["baseline"]["problems"],
        )
        self.assertIn(
            "missing summary field: mean_cross_page_merge_accuracy",
            payload["ab_reports"]["layout_v2"]["baseline"]["problems"],
        )
        self.assertIn(
            "missing summary field: mean_business_field_location_hit_rate",
            payload["ab_reports"]["layout_v2"]["candidate"]["problems"],
        )
        self.assertIn(
            "missing summary field: mean_cross_page_merge_accuracy",
            payload["ab_reports"]["layout_v2"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:layout_v2", payload["failed_gates"])

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-layout-regress-") as temp_dir:
            root = Path(temp_dir)
            baseline_report = root / "layout-baseline.json"
            candidate_report = root / "layout-candidate.json"
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"layout_engine": "legacy", "reading_order_strategy": "legacy"},
                        "summary": {
                            "sample_count": 1,
                            "mean_block_type_f1": 0.8,
                            "mean_reading_order_normalized_edit_distance": 0.2,
                            "mean_cross_page_merge_accuracy": 0.9,
                            "mean_chunk_text_coverage": 0.9,
                            "mean_business_field_location_hit_rate": 0.8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"layout_engine": "ppdoclayout", "reading_order_strategy": "legacy"},
                        "summary": {
                            "sample_count": 1,
                            "mean_block_type_f1": 0.9,
                            "mean_reading_order_normalized_edit_distance": 0.1,
                            "mean_cross_page_merge_accuracy": 0.7,
                            "mean_chunk_text_coverage": 0.7,
                            "mean_business_field_location_hit_rate": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={
                    "layout_v2_baseline": baseline_report,
                    "layout_v2_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["layout_v2"]["status"])
        self.assertIn(
            "candidate mean_cross_page_merge_accuracy regressed: baseline=0.9, candidate=0.7",
            payload["ab_reports"]["layout_v2"]["problems"],
        )
        self.assertIn(
            "candidate mean_chunk_text_coverage regressed: baseline=0.9, candidate=0.7",
            payload["ab_reports"]["layout_v2"]["problems"],
        )
        self.assertIn(
            "candidate mean_business_field_location_hit_rate regressed: baseline=0.8, candidate=0.6",
            payload["ab_reports"]["layout_v2"]["problems"],
        )
        self.assertIn(
            "candidate: pipeline_config.reading_order_strategy expected rules, got legacy",
            payload["ab_reports"]["layout_v2"]["problems"],
        )
        self.assertIn("failed_ab_report:layout_v2", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_table_v2_ab_quality_metrics(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-table-metrics-") as temp_dir:
            root = Path(temp_dir)
            baseline_report = root / "table-baseline.json"
            candidate_report = root / "table-candidate.json"
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": "tools/eval_datasets/table",
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"table_engine": "tatr"},
                        "summary": {"sample_count": 1},
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"table_engine": "rapidtable"},
                        "summary": {"sample_count": 1},
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={
                    "table_v2_baseline": baseline_report,
                    "table_v2_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["table_v2"]["baseline"]["status"])
        self.assertIn(
            "missing summary field: mean_table_teds",
            payload["ab_reports"]["table_v2"]["baseline"]["problems"],
        )
        self.assertIn(
            "missing summary field: mean_table_cell_f1",
            payload["ab_reports"]["table_v2"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:table_v2", payload["failed_gates"])

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-table-regress-") as temp_dir:
            root = Path(temp_dir)
            baseline_report = root / "table-baseline.json"
            candidate_report = root / "table-candidate.json"
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"table_engine": "tatr"},
                        "summary": {
                            "sample_count": 1,
                            "mean_table_teds": 0.9,
                            "mean_table_cell_f1": 0.8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"table_engine": "rapidtable"},
                        "summary": {
                            "sample_count": 1,
                            "mean_table_teds": 0.85,
                            "mean_table_cell_f1": 0.7,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={
                    "table_v2_baseline": baseline_report,
                    "table_v2_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["table_v2"]["status"])
        self.assertIn(
            "candidate mean_table_teds regressed: baseline=0.9, candidate=0.85",
            payload["ab_reports"]["table_v2"]["problems"],
        )
        self.assertIn(
            "candidate mean_table_cell_f1 regressed: baseline=0.8, candidate=0.7",
            payload["ab_reports"]["table_v2"]["problems"],
        )
        self.assertIn("failed_ab_report:table_v2", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_formula_quality_and_timing_metrics(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-formula-metrics-") as temp_dir:
            root = Path(temp_dir)
            baseline_report = root / "formula-baseline.json"
            candidate_report = root / "formula-candidate.json"
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": "tools/eval_datasets/formula",
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"formula_mode": "rapidlatex"},
                        "summary": {"sample_count": 1},
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"formula_mode": "pp_formula_net_s"},
                        "summary": {"sample_count": 1},
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={
                    "formula_v2_baseline": baseline_report,
                    "formula_v2_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["formula_v2"]["baseline"]["status"])
        self.assertIn(
            "missing summary field: mean_formula_normalized_edit_distance",
            payload["ab_reports"]["formula_v2"]["baseline"]["problems"],
        )
        self.assertIn(
            "missing summary field: mean_formula_exact_match_rate",
            payload["ab_reports"]["formula_v2"]["candidate"]["problems"],
        )
        self.assertIn(
            "missing summary field: mean_elapsed_seconds",
            payload["ab_reports"]["formula_v2"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:formula_v2", payload["failed_gates"])

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-formula-regress-") as temp_dir:
            root = Path(temp_dir)
            baseline_report = root / "formula-baseline.json"
            candidate_report = root / "formula-candidate.json"
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"formula_mode": "rapidlatex"},
                        "summary": {
                            "sample_count": 1,
                            "mean_formula_normalized_edit_distance": 0.1,
                            "mean_formula_exact_match_rate": 0.9,
                            "mean_elapsed_seconds": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"formula_mode": "pp_formula_net_s"},
                        "summary": {
                            "sample_count": 1,
                            "mean_formula_normalized_edit_distance": 0.2,
                            "mean_formula_exact_match_rate": 0.8,
                            "mean_elapsed_seconds": 1.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={
                    "formula_v2_baseline": baseline_report,
                    "formula_v2_candidate": candidate_report,
                },
            )

        self.assertEqual("failed", payload["ab_reports"]["formula_v2"]["status"])
        self.assertIn(
            "candidate mean_formula_normalized_edit_distance regressed: baseline=0.1, candidate=0.2",
            payload["ab_reports"]["formula_v2"]["problems"],
        )
        self.assertIn(
            "candidate mean_formula_exact_match_rate regressed: baseline=0.9, candidate=0.8",
            payload["ab_reports"]["formula_v2"]["problems"],
        )
        self.assertIn(
            "candidate mean_elapsed_seconds regressed: baseline=1.0, candidate=1.5",
            payload["ab_reports"]["formula_v2"]["problems"],
        )
        self.assertIn("failed_ab_report:formula_v2", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_profile_report_shape(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-") as temp_dir:
            root = Path(temp_dir)
            bad_profile = root / "profile.json"
            bad_profile.write_text(
                json.dumps({"schema_version": "2026-06-08.cpu-pipeline-profile.v1"}),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={"profile": bad_profile},
            )

        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "missing profile field: total_elapsed_seconds",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "missing profile field: pipeline_config",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "missing profile field: stage_summary",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "missing profile field: license_gate",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "missing profile stages: rasterize_ocr, layout, table, text_merge, cross_page_text, "
            "reading_order, extract_assets",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_profile_pipeline_config_shape(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-config-") as temp_dir:
            root = Path(temp_dir)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "total_elapsed_seconds": 1.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {"reading_order_strategy": "rules"},
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "layout",
                            "slowest_stage_elapsed_seconds": 0.1,
                            "slowest_stage_share": 0.1,
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 0.1},
                            {"stage": "layout", "elapsed_seconds": 0.1},
                            {"stage": "table", "elapsed_seconds": 0.1},
                            {"stage": "text_merge", "elapsed_seconds": 0.1},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.1},
                            {"stage": "reading_order", "elapsed_seconds": 0.1},
                            {"stage": "extract_assets", "elapsed_seconds": 0.1},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={"profile": profile},
            )

        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "profile pipeline_config missing keys: ocr_version, layout_engine, table_engine, formula_mode",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile reading_order_strategy does not match pipeline_config.reading_order_strategy: "
            "top-level=legacy, pipeline_config=rules",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_profile_timing_summary(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-timing-") as temp_dir:
            root = Path(temp_dir)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "total_elapsed_seconds": 0.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 6,
                            "slowest_stage": "layout",
                            "slowest_stage_elapsed_seconds": 3.0,
                            "slowest_stage_share": 1.5,
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 1.0},
                            {"stage": "layout", "elapsed_seconds": "bad"},
                            {"stage": "table", "elapsed_seconds": -0.1},
                            {"stage": "text_merge", "elapsed_seconds": 0.1},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.1},
                            {"stage": "reading_order", "elapsed_seconds": 0.1},
                            {"stage": "extract_assets", "elapsed_seconds": 0.1},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={"profile": profile},
            )

        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "profile total_elapsed_seconds must be positive",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage layout elapsed_seconds must be numeric",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage table elapsed_seconds must be non-negative",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage_summary.stage_count expected 7, got 6",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage_summary.slowest_stage_share must be between 0 and 1",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_rejects_non_finite_profile_timing_values(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        eval_tool = importlib.import_module("tools.eval_omnidocbench")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-finite-timing-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            profile = root / "profile.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "dataset": str(dataset),
                        "dataset_contract": eval_tool.validate_dataset(dataset),
                        "license_gate": self._license_gate_payload(),
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core", "formula")),
                        "pdf_path": str(pdf_path),
                        "sample_name": "sample",
                        "total_elapsed_seconds": float("nan"),
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "rasterize_ocr",
                            "slowest_stage_elapsed_seconds": float("inf"),
                            "slowest_stage_share": float("nan"),
                            "stages_by_elapsed_seconds": [
                                {"stage": "rasterize_ocr", "elapsed_seconds": float("inf")},
                                {"stage": "layout", "elapsed_seconds": 1.5},
                                {"stage": "table", "elapsed_seconds": 1.0},
                                {"stage": "text_merge", "elapsed_seconds": 1.0},
                                {"stage": "cross_page_text", "elapsed_seconds": 0.75},
                                {"stage": "reading_order", "elapsed_seconds": 0.5},
                                {"stage": "extract_assets", "elapsed_seconds": 0.25},
                            ],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": float("inf")},
                            {"stage": "layout", "elapsed_seconds": 1.5},
                            {"stage": "table", "elapsed_seconds": 1.0},
                            {"stage": "text_merge", "elapsed_seconds": 1.0},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.75},
                            {"stage": "reading_order", "elapsed_seconds": 0.5},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=model_root,
                dataset=dataset,
                min_pages=0,
                report_paths={"profile": profile},
            )

        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "profile total_elapsed_seconds must be finite",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage rasterize_ocr elapsed_seconds must be finite",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage_summary.slowest_stage_elapsed_seconds must be finite",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage_summary.slowest_stage_share must be finite",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage_summary.stages_by_elapsed_seconds[0].elapsed_seconds must be finite",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_rejects_boolean_profile_timing_values(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        eval_tool = importlib.import_module("tools.eval_omnidocbench")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-bool-timing-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            profile = root / "profile.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "dataset": str(dataset),
                        "dataset_contract": eval_tool.validate_dataset(dataset),
                        "license_gate": self._license_gate_payload(),
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core", "formula")),
                        "pdf_path": str(pdf_path),
                        "sample_name": "sample",
                        "total_elapsed_seconds": True,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "rasterize_ocr",
                            "slowest_stage_elapsed_seconds": True,
                            "slowest_stage_share": True,
                            "stages_by_elapsed_seconds": [
                                {"stage": "rasterize_ocr", "elapsed_seconds": True},
                                {"stage": "cross_page_text", "elapsed_seconds": 0.0},
                                {"stage": "extract_assets", "elapsed_seconds": 0.0},
                                {"stage": "layout", "elapsed_seconds": 0.0},
                                {"stage": "reading_order", "elapsed_seconds": 0.0},
                                {"stage": "table", "elapsed_seconds": 0.0},
                                {"stage": "text_merge", "elapsed_seconds": 0.0},
                            ],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": True},
                            {"stage": "layout", "elapsed_seconds": 0.0},
                            {"stage": "table", "elapsed_seconds": 0.0},
                            {"stage": "text_merge", "elapsed_seconds": 0.0},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.0},
                            {"stage": "reading_order", "elapsed_seconds": 0.0},
                            {"stage": "extract_assets", "elapsed_seconds": 0.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=model_root,
                dataset=dataset,
                min_pages=0,
                report_paths={"profile": profile},
            )

        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "profile total_elapsed_seconds must be numeric",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage rasterize_ocr elapsed_seconds must be numeric",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage_summary.slowest_stage_elapsed_seconds must be numeric",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage_summary.slowest_stage_share must be numeric",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage_summary.stages_by_elapsed_seconds[0].elapsed_seconds must be numeric",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_profile_summary_consistency(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-consistency-") as temp_dir:
            root = Path(temp_dir)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "total_elapsed_seconds": 10.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "layout",
                            "slowest_stage_elapsed_seconds": 1.0,
                            "slowest_stage_share": 0.1,
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 2.0},
                            {"stage": "layout", "elapsed_seconds": 1.0},
                            {"stage": "table", "elapsed_seconds": 0.5},
                            {"stage": "text_merge", "elapsed_seconds": 0.25},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.25},
                            {"stage": "reading_order", "elapsed_seconds": 0.25},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={"profile": profile},
            )

        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "profile total_elapsed_seconds does not match summed stages: total=10.0, stages=4.5",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage_summary.slowest_stage expected rasterize_ocr, got layout",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "profile stage_summary.slowest_stage_share expected 0.4444444444444444, got 0.1",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_profile_stage_summary_ranked_stages(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-ranked-") as temp_dir:
            root = Path(temp_dir)
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "total_elapsed_seconds": 4.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "layout",
                            "slowest_stage_elapsed_seconds": 1.25,
                            "slowest_stage_share": 1.25 / 4.0,
                            "stages_by_elapsed_seconds": [
                                {"stage": "rasterize_ocr", "elapsed_seconds": 1.0},
                                {"stage": "layout", "elapsed_seconds": 1.25},
                                {"stage": "table", "elapsed_seconds": 0.75},
                            ],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 1.0},
                            {"stage": "layout", "elapsed_seconds": 1.25},
                            {"stage": "table", "elapsed_seconds": 0.75},
                            {"stage": "text_merge", "elapsed_seconds": 0.25},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.25},
                            {"stage": "reading_order", "elapsed_seconds": 0.25},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=root / "models",
                dataset=root / "dataset",
                min_pages=0,
                report_paths={"profile": profile},
            )

        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "profile stage_summary.stages_by_elapsed_seconds must include the same ordered stages as profile stages",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_validates_profile_ranked_stage_shares(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        eval_tool = importlib.import_module("tools.eval_omnidocbench")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-ranked-share-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            profile = root / "profile.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "dataset": str(dataset),
                        "dataset_contract": eval_tool.validate_dataset(dataset),
                        "license_gate": self._license_gate_payload(),
                        "model_manifest": model_store.build_model_manifest(model_root=model_root, groups=("core", "formula")),
                        "pdf_path": str(pdf_path),
                        "sample_name": "sample",
                        "total_elapsed_seconds": 4.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "layout",
                            "slowest_stage_elapsed_seconds": 1.25,
                            "slowest_stage_share": 1.25 / 4.0,
                            "stages_by_elapsed_seconds": [
                                {"stage": "layout", "elapsed_seconds": 1.25, "share": 0.99},
                                {"stage": "rasterize_ocr", "elapsed_seconds": 1.0, "share": 1.0 / 4.0},
                                {"stage": "table", "elapsed_seconds": 0.75, "share": 0.75 / 4.0},
                                {"stage": "cross_page_text", "elapsed_seconds": 0.25, "share": 0.25 / 4.0},
                                {"stage": "extract_assets", "elapsed_seconds": 0.25, "share": 0.25 / 4.0},
                                {"stage": "reading_order", "elapsed_seconds": 0.25, "share": 0.25 / 4.0},
                                {"stage": "text_merge", "elapsed_seconds": 0.25, "share": 0.25 / 4.0},
                            ],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 1.0},
                            {"stage": "layout", "elapsed_seconds": 1.25},
                            {"stage": "table", "elapsed_seconds": 0.75},
                            {"stage": "text_merge", "elapsed_seconds": 0.25},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.25},
                            {"stage": "reading_order", "elapsed_seconds": 0.25},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"profile": profile},
                )

        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "profile stage_summary.stages_by_elapsed_seconds[0].share expected 0.3125, got 0.99",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_rejects_unexpected_profile_stages(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-extra-stage-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            profile = root / "profile.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            current_manifest = importlib.import_module("common.model_store").build_model_manifest(
                model_root=model_root,
                groups=("core", "formula"),
            )
            stages = [
                {"stage": "rasterize_ocr", "elapsed_seconds": 1.0},
                {"stage": "layout", "elapsed_seconds": 1.25},
                {"stage": "table", "elapsed_seconds": 0.75},
                {"stage": "text_merge", "elapsed_seconds": 0.25},
                {"stage": "cross_page_text", "elapsed_seconds": 0.25},
                {"stage": "reading_order", "elapsed_seconds": 0.25},
                {"stage": "extract_assets", "elapsed_seconds": 0.25},
                {"stage": "debug_extra", "elapsed_seconds": 0.25},
            ]
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "dataset": str(dataset),
                        "dataset_contract": self._dataset_contract_payload(dataset),
                        "license_gate": self._license_gate_payload(),
                        "model_manifest": current_manifest,
                        "pdf_path": str(pdf_path),
                        "sample_name": "sample",
                        "total_elapsed_seconds": 4.25,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 8,
                            "slowest_stage": "layout",
                            "slowest_stage_elapsed_seconds": 1.25,
                            "slowest_stage_share": 1.25 / 4.25,
                            "stages_by_elapsed_seconds": [
                                {"stage": "layout", "elapsed_seconds": 1.25},
                                {"stage": "rasterize_ocr", "elapsed_seconds": 1.0},
                                {"stage": "table", "elapsed_seconds": 0.75},
                                {"stage": "cross_page_text", "elapsed_seconds": 0.25},
                                {"stage": "debug_extra", "elapsed_seconds": 0.25},
                                {"stage": "extract_assets", "elapsed_seconds": 0.25},
                                {"stage": "reading_order", "elapsed_seconds": 0.25},
                                {"stage": "text_merge", "elapsed_seconds": 0.25},
                            ],
                        },
                        "stages": stages,
                    }
                ),
                encoding="utf-8",
            )

            payload = readiness_tool.check_readiness(
                model_root=model_root,
                dataset=dataset,
                min_pages=0,
                report_paths={"profile": profile},
            )

        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "unexpected profile stages: debug_extra",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_profile_report_to_match_current_dataset(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-dataset-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "sample-a.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (dataset / "sample-b.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "pdf_path": str(root / "previous-dataset" / "stale-sample.pdf"),
                        "sample_name": "stale-sample",
                        "total_elapsed_seconds": 7.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "rasterize_ocr",
                            "slowest_stage_elapsed_seconds": 2.0,
                            "slowest_stage_share": 2.0 / 7.0,
                            "stages_by_elapsed_seconds": [],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 2.0},
                            {"stage": "layout", "elapsed_seconds": 1.5},
                            {"stage": "table", "elapsed_seconds": 1.0},
                            {"stage": "text_merge", "elapsed_seconds": 1.0},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.75},
                            {"stage": "reading_order", "elapsed_seconds": 0.5},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=root / "models",
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"profile": profile},
                )

        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "profile sample_name must be one of dataset contract samples: expected=sample-a,sample-b, got=stale-sample",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            f"profile pdf_path must point to a PDF in readiness dataset: expected one of {dataset}, got {root / 'previous-dataset' / 'stale-sample.pdf'}",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_profile_dataset_field_to_match_current_dataset(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-dataset-field-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            previous_dataset = root / "previous-dataset"
            profile = root / "profile.json"
            dataset.mkdir()
            previous_dataset.mkdir()
            pdf_path = dataset / "sample-a.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "dataset": str(previous_dataset),
                        "pdf_path": str(pdf_path),
                        "sample_name": "sample-a",
                        "total_elapsed_seconds": 7.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "rasterize_ocr",
                            "slowest_stage_elapsed_seconds": 2.0,
                            "slowest_stage_share": 2.0 / 7.0,
                            "stages_by_elapsed_seconds": [
                                {"stage": "rasterize_ocr", "elapsed_seconds": 2.0, "share": 2.0 / 7.0},
                                {"stage": "layout", "elapsed_seconds": 1.5, "share": 1.5 / 7.0},
                                {"stage": "table", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "text_merge", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "cross_page_text", "elapsed_seconds": 0.75, "share": 0.75 / 7.0},
                                {"stage": "reading_order", "elapsed_seconds": 0.5, "share": 0.5 / 7.0},
                                {"stage": "extract_assets", "elapsed_seconds": 0.25, "share": 0.25 / 7.0},
                            ],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 2.0},
                            {"stage": "layout", "elapsed_seconds": 1.5},
                            {"stage": "table", "elapsed_seconds": 1.0},
                            {"stage": "text_merge", "elapsed_seconds": 1.0},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.75},
                            {"stage": "reading_order", "elapsed_seconds": 0.5},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=root / "models",
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"profile": profile},
                )

        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            f"profile dataset must match readiness dataset: expected={dataset}, got={previous_dataset}",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_profile_dataset_field(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-missing-dataset-field-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            profile = root / "profile.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            current_manifest = importlib.import_module("common.model_store").build_model_manifest(
                model_root=model_root,
                groups=("core", "formula"),
            )
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "dataset_contract": self._dataset_contract_payload(dataset),
                        "license_gate": self._license_gate_payload(),
                        "model_manifest": current_manifest,
                        "pdf_path": str(pdf_path),
                        "sample_name": "sample",
                        "total_elapsed_seconds": 7.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "rasterize_ocr",
                            "slowest_stage_elapsed_seconds": 2.0,
                            "slowest_stage_share": 2.0 / 7.0,
                            "stages_by_elapsed_seconds": [
                                {"stage": "rasterize_ocr", "elapsed_seconds": 2.0, "share": 2.0 / 7.0},
                                {"stage": "layout", "elapsed_seconds": 1.5, "share": 1.5 / 7.0},
                                {"stage": "table", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "text_merge", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "cross_page_text", "elapsed_seconds": 0.75, "share": 0.75 / 7.0},
                                {"stage": "reading_order", "elapsed_seconds": 0.5, "share": 0.5 / 7.0},
                                {"stage": "extract_assets", "elapsed_seconds": 0.25, "share": 0.25 / 7.0},
                            ],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 2.0},
                            {"stage": "layout", "elapsed_seconds": 1.5},
                            {"stage": "table", "elapsed_seconds": 1.0},
                            {"stage": "text_merge", "elapsed_seconds": 1.0},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.75},
                            {"stage": "reading_order", "elapsed_seconds": 0.5},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"profile": profile},
                )

        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "missing profile field: dataset",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_profile_sample_name_to_match_pdf_path(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-sample-path-pair-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            profile = root / "profile.json"
            model_root.mkdir()
            dataset.mkdir()
            sample_a_pdf = dataset / "sample-a.pdf"
            sample_b_pdf = dataset / "sample-b.pdf"
            sample_a_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            sample_b_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            current_manifest = importlib.import_module("common.model_store").build_model_manifest(
                model_root=model_root,
                groups=("core", "formula"),
            )
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "dataset": str(dataset),
                        "dataset_contract": self._dataset_contract_payload(dataset, sample_names=["sample-a", "sample-b"]),
                        "license_gate": self._license_gate_payload(),
                        "model_manifest": current_manifest,
                        "pdf_path": str(sample_b_pdf),
                        "sample_name": "sample-a",
                        "total_elapsed_seconds": 7.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "rasterize_ocr",
                            "slowest_stage_elapsed_seconds": 2.0,
                            "slowest_stage_share": 2.0 / 7.0,
                            "stages_by_elapsed_seconds": [
                                {"stage": "rasterize_ocr", "elapsed_seconds": 2.0, "share": 2.0 / 7.0},
                                {"stage": "layout", "elapsed_seconds": 1.5, "share": 1.5 / 7.0},
                                {"stage": "table", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "text_merge", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "cross_page_text", "elapsed_seconds": 0.75, "share": 0.75 / 7.0},
                                {"stage": "reading_order", "elapsed_seconds": 0.5, "share": 0.5 / 7.0},
                                {"stage": "extract_assets", "elapsed_seconds": 0.25, "share": 0.25 / 7.0},
                            ],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 2.0},
                            {"stage": "layout", "elapsed_seconds": 1.5},
                            {"stage": "table", "elapsed_seconds": 1.0},
                            {"stage": "text_merge", "elapsed_seconds": 1.0},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.75},
                            {"stage": "reading_order", "elapsed_seconds": 0.5},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"profile": profile},
                )

        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            f"profile sample_name/pdf_path must reference the same dataset contract sample: "
            f"sample_name=sample-a, expected_pdf_path={sample_a_pdf}, got={sample_b_pdf}",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_cli_runs_from_repo_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-cli-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            output_path = root / "out" / "readiness.json"
            model_root.mkdir()
            dataset.mkdir()

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/check_cpu_pipeline_readiness.py",
                    "--model-root",
                    str(model_root),
                    "--dataset",
                    str(dataset),
                    "--min-pages",
                    "100",
                    "--out",
                    str(output_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            file_payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(1, result.returncode, result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual("failed", payload["status"])
        self.assertIn("dataset_pages_below_minimum", payload["failed_gates"])
        self.assertEqual(payload, file_payload)

    def test_eval_tool_license_gate_writes_out_file(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="deepdoc-license-gate-cli-") as temp_dir:
            output_path = Path(temp_dir) / "out" / "license-gate.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/eval_omnidocbench.py",
                    "--license-gate",
                    "--out",
                    str(output_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            file_payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(0, result.returncode, result.stderr)
        stdout_payload = json.loads(result.stdout)
        self.assertEqual(stdout_payload, file_payload)
        self.assertEqual("2026-06-08.cpu-pipeline-license-gate.v1", file_payload["schema_version"])
        self.assertEqual("passed", file_payload["status"])
        allowed_names = {item["name"] for item in file_payload["allowed"]}
        blocked_names = {item["name"] for item in file_payload["blocked"]}
        model_store = importlib.import_module("common.model_store")
        provenance_components = {
            item["component"]
            for item in model_store.get_model_group_provenance(("core_v5", "layout_v2", "table_v2", "formula_v2")).values()
        }
        self.assertLessEqual(provenance_components, allowed_names)
        for expected in ["PP-OCRv5", "PP-DocLayout", "SLANet-plus", "PP-FormulaNet-S", "RapidTable", "RapidLaTeXOCR", "Docling"]:
            self.assertIn(expected, allowed_names)
        for expected in ["DocLayout-YOLO", "Marker", "Surya", "texify"]:
            self.assertIn(expected, blocked_names)

    def test_eval_tool_license_gate_derives_upgrade_model_components_from_provenance(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")
        model_store = importlib.import_module("common.model_store")

        patched_provenance = dict(model_store.MODEL_GROUP_PROVENANCE)
        patched_provenance["formula_v2"] = {
            **patched_provenance["formula_v2"],
            "license": "GPL-3.0",
            "license_status": "blocked",
        }

        with patch.object(model_store, "MODEL_GROUP_PROVENANCE", patched_provenance):
            payload = eval_tool.license_gate_report()

        allowed_names = {item["name"] for item in payload["allowed"]}
        blocked_names = {item["name"] for item in payload["blocked"]}

        self.assertNotIn("PP-FormulaNet-S", allowed_names)
        self.assertIn("PP-FormulaNet-S", blocked_names)
        blocked_formula = next(item for item in payload["blocked"] if item["name"] == "PP-FormulaNet-S")
        self.assertEqual("GPL-3.0", blocked_formula["license"])
        self.assertEqual("review", payload["status"])

    def test_eval_tool_applies_and_restores_cpu_pipeline_ab_switches(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")

        with tempfile.TemporaryDirectory(prefix="deepdoc-eval-switches-") as temp_dir:
            dataset = Path(temp_dir)
            sample_path = dataset / "sample.pdf"
            sample_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            seen_env: dict[str, str | None] = {}

            def fake_evaluate_sample(_sample, *, engine):
                self.assertEqual("deepdoc", engine)
                seen_env.update(
                    {
                        "DEEPDOC_OCR_VERSION": os.environ.get("DEEPDOC_OCR_VERSION"),
                        "DEEPDOC_LAYOUT_ENGINE": os.environ.get("DEEPDOC_LAYOUT_ENGINE"),
                        "DEEPDOC_TABLE_ENGINE": os.environ.get("DEEPDOC_TABLE_ENGINE"),
                        "DEEPDOC_FORMULA_MODE": os.environ.get("DEEPDOC_FORMULA_MODE"),
                        "DEEPDOC_READING_ORDER_STRATEGY": os.environ.get("DEEPDOC_READING_ORDER_STRATEGY"),
                    }
                )
                return {
                    "name": _sample.name,
                    "pdf_path": str(_sample.pdf_path),
                    "engine": engine,
                    "elapsed_seconds": 0.0,
                    "text_length": 0,
                    "has_text_gt": False,
                    "has_blocks_gt": False,
                }

            with patch.dict(
                os.environ,
                {
                    "DEEPDOC_OCR_VERSION": "v4",
                    "DEEPDOC_LAYOUT_ENGINE": "legacy",
                    "DEEPDOC_TABLE_ENGINE": "tatr",
                    "DEEPDOC_FORMULA_MODE": "rapidlatex",
                    "DEEPDOC_READING_ORDER_STRATEGY": "legacy",
                },
                clear=False,
            ):
                with patch.object(eval_tool, "_evaluate_sample", side_effect=fake_evaluate_sample):
                    payload = eval_tool.evaluate_dataset(
                        engine="deepdoc",
                        dataset=dataset,
                        out=None,
                        ocr_version="v5",
                        layout_engine="ppdoclayout",
                        table_engine="rapidtable",
                        formula_mode="pp_formula_net_s",
                        reading_order_strategy="rules",
                    )

                self.assertEqual("v4", os.environ.get("DEEPDOC_OCR_VERSION"))
                self.assertEqual("legacy", os.environ.get("DEEPDOC_LAYOUT_ENGINE"))
                self.assertEqual("tatr", os.environ.get("DEEPDOC_TABLE_ENGINE"))
                self.assertEqual("rapidlatex", os.environ.get("DEEPDOC_FORMULA_MODE"))
                self.assertEqual("legacy", os.environ.get("DEEPDOC_READING_ORDER_STRATEGY"))

        self.assertEqual(
            {
                "DEEPDOC_OCR_VERSION": "v5",
                "DEEPDOC_LAYOUT_ENGINE": "ppdoclayout",
                "DEEPDOC_TABLE_ENGINE": "rapidtable",
                "DEEPDOC_FORMULA_MODE": "pp_formula_net_s",
                "DEEPDOC_READING_ORDER_STRATEGY": "rules",
            },
            seen_env,
        )
        self.assertEqual(
            {
                "ocr_version": "v5",
                "layout_engine": "ppdoclayout",
                "table_engine": "rapidtable",
                "formula_mode": "pp_formula_net_s",
                "reading_order_strategy": "rules",
            },
            payload["pipeline_config"],
        )
        self.assertEqual("2026-06-08.cpu-pipeline-model-manifest.v1", payload["model_manifest"]["schema_version"])
        self.assertEqual(
            {"core_v5", "layout_v2", "table_v2", "formula_v2"},
            set(payload["model_manifest"]["groups"]),
        )
        self.assertNotIn("core", payload["model_manifest"]["groups"])

    def test_cpu_pipeline_readiness_requires_eval_report_model_manifest_to_match_current_models(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-model-manifest-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            (dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (model_root / "rec_v5.onnx").write_bytes(b"current-rec-v5")
            (model_root / "ocr_v5.res").write_text("中\n国\n", encoding="utf-8")
            dataset_contract = self._dataset_contract_payload(dataset)
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "dataset_contract": dataset_contract,
                "license_gate": self._license_gate_payload(),
                "samples": [{"name": "sample"}],
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "model_manifest": {
                            "schema_version": "2026-06-08.cpu-pipeline-model-manifest.v1",
                            "groups": ["core_v5"],
                            "files": [
                                {
                                    "path": "rec_v5.onnx",
                                    "exists": True,
                                    "size_bytes": len(b"current-rec-v5"),
                                    "sha256": "stale-sha256",
                                },
                                {
                                    "path": "ocr_v5.res",
                                    "exists": True,
                                    "size_bytes": len("中\n国\n".encode("utf-8")),
                                    "sha256": "stale-sha256",
                                },
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["status"])
        self.assertIn(
            "missing model_manifest report field for required groups: core",
            payload["ab_reports"]["ocr_v5"]["baseline"]["problems"],
        )
        self.assertIn(
            "model_manifest file rec_v5.onnx sha256 mismatch",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn(
            "model_manifest file ocr_v5.res sha256 mismatch",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_cpu_pipeline_readiness_requires_eval_report_model_manifest_provenance(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")
        model_store = importlib.import_module("common.model_store")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-model-provenance-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            baseline_report = root / "ocr-baseline.json"
            candidate_report = root / "ocr-candidate.json"
            model_root.mkdir()
            dataset.mkdir()
            (dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            for relative_path in model_store.MODEL_GROUP_FILES["core"]:
                path = model_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"current-{relative_path}".encode("utf-8"))
            for relative_path in model_store.MODEL_GROUP_FILES["core_v5"]:
                path = model_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"current-{relative_path}".encode("utf-8"))
            dataset_contract = self._dataset_contract_payload(dataset)
            baseline_manifest = model_store.build_model_manifest(model_root=model_root, groups=("core",))
            candidate_manifest = model_store.build_model_manifest(model_root=model_root, groups=("core_v5",))
            candidate_manifest.pop("model_group_provenance")
            common_payload = {
                "schema_version": "2026-06-08.cpu-pipeline-eval.v1",
                "engine": "deepdoc",
                "dataset": str(dataset),
                "summary": {
                    "sample_count": 1,
                    "mean_character_error_rate": 0.1,
                    "mean_word_error_rate": 0.2,
                },
                "dataset_contract": dataset_contract,
                "license_gate": self._license_gate_payload(),
                "samples": [{"name": "sample", "pdf_path": str(dataset / "sample.pdf")}],
            }
            baseline_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v4"},
                        "model_manifest": baseline_manifest,
                    }
                ),
                encoding="utf-8",
            )
            candidate_report.write_text(
                json.dumps(
                    {
                        **common_payload,
                        "pipeline_config": {"ocr_version": "v5"},
                        "model_manifest": candidate_manifest,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={
                        "ocr_v5_baseline": baseline_report,
                        "ocr_v5_candidate": candidate_report,
                    },
                )

        self.assertEqual("failed", payload["ab_reports"]["ocr_v5"]["candidate"]["status"])
        self.assertIn(
            "missing model_manifest model_group_provenance for group: core_v5",
            payload["ab_reports"]["ocr_v5"]["candidate"]["problems"],
        )
        self.assertIn("failed_ab_report:ocr_v5", payload["failed_gates"])

    def test_profile_pipeline_applies_and_restores_cpu_pipeline_switches(self):
        profile_tool = importlib.import_module("tools.profile_pipeline")
        from deepdoc.parser import pdf_parser

        class FakeParser:
            def __images__(self, _source, _zoomin):
                return None

            def _layouts_rec(self, _zoomin):
                return None

            def _table_transformer_job(self, _zoomin):
                return None

            def _text_merge(self):
                return None

            def _merge_cross_page_text(self):
                return None

            def _apply_reading_order_strategy(self, _zoomin):
                return None

            def _extract_table_figure(self, *_args):
                return None

        with patch.dict(
            os.environ,
            {
                "DEEPDOC_OCR_VERSION": "v4",
                "DEEPDOC_LAYOUT_ENGINE": "legacy",
                "DEEPDOC_TABLE_ENGINE": "tatr",
                "DEEPDOC_FORMULA_MODE": "rapidlatex",
                "DEEPDOC_READING_ORDER_STRATEGY": "legacy",
            },
            clear=False,
        ):
            with patch.object(pdf_parser, "DeepDocPdfParser", FakeParser):
                with patch.object(
                    profile_tool.time,
                    "perf_counter",
                    side_effect=[
                        0.0,
                        1.0,
                        1.0,
                        3.0,
                        3.0,
                        4.0,
                        4.0,
                        4.25,
                        4.25,
                        4.75,
                        4.75,
                        5.0,
                        5.0,
                        5.5,
                    ],
                ):
                    payload = profile_tool.profile_pipeline(
                        "/tmp/sample.pdf",
                        table_engine="rapidtable",
                        layout_engine="ppdoclayout",
                        ocr_version="v5",
                        formula_mode="pp_formula_net_s",
                        reading_order_strategy="rules",
                    )

            self.assertEqual("v4", os.environ.get("DEEPDOC_OCR_VERSION"))
            self.assertEqual("legacy", os.environ.get("DEEPDOC_LAYOUT_ENGINE"))
            self.assertEqual("tatr", os.environ.get("DEEPDOC_TABLE_ENGINE"))
            self.assertEqual("rapidlatex", os.environ.get("DEEPDOC_FORMULA_MODE"))
            self.assertEqual("legacy", os.environ.get("DEEPDOC_READING_ORDER_STRATEGY"))

        self.assertEqual("v5", payload["ocr_version"])
        self.assertEqual("sample", payload["sample_name"])
        self.assertEqual("ppdoclayout", payload["layout_engine"])
        self.assertEqual("rapidtable", payload["table_engine"])
        self.assertEqual("pp_formula_net_s", payload["formula_mode"])
        self.assertEqual("rules", payload["reading_order_strategy"])
        self.assertEqual(
            {
                "ocr_version": "v5",
                "layout_engine": "ppdoclayout",
                "table_engine": "rapidtable",
                "formula_mode": "pp_formula_net_s",
                "reading_order_strategy": "rules",
            },
            payload["pipeline_config"],
        )
        self.assertEqual(7, payload["stage_summary"]["stage_count"])
        self.assertEqual("layout", payload["stage_summary"]["slowest_stage"])
        self.assertAlmostEqual(2.0, payload["stage_summary"]["slowest_stage_elapsed_seconds"])
        self.assertAlmostEqual(2.0 / 5.5, payload["stage_summary"]["slowest_stage_share"])
        self.assertEqual("layout", payload["stage_summary"]["stages_by_elapsed_seconds"][0]["stage"])
        self.assertEqual("rasterize_ocr", payload["stage_summary"]["stages_by_elapsed_seconds"][1]["stage"])
        self.assertEqual("2026-06-08.cpu-pipeline-model-manifest.v1", payload["model_manifest"]["schema_version"])
        self.assertIn("core_v5", payload["model_manifest"]["groups"])
        self.assertIn("layout_v2", payload["model_manifest"]["groups"])
        self.assertIn("table_v2", payload["model_manifest"]["groups"])
        self.assertIn("formula_v2", payload["model_manifest"]["groups"])
        self.assertEqual("2026-06-08.cpu-pipeline-license-gate.v1", payload["license_gate"]["schema_version"])
        self.assertEqual("passed", payload["license_gate"]["status"])

    def test_cpu_pipeline_readiness_requires_profile_model_manifest_to_match_current_models(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-model-manifest-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            profile = root / "profile.json"
            model_root.mkdir()
            dataset.mkdir()
            (model_root / "layout").mkdir()
            (model_root / "table").mkdir()
            (model_root / "formula").mkdir()
            (dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            (model_root / "det_v5.onnx").write_bytes(b"current-det-v5")
            (model_root / "rec_v5.onnx").write_bytes(b"current-rec-v5")
            (model_root / "ocr_v5.res").write_text("中\n国\n", encoding="utf-8")
            (model_root / "layout" / "pp_doclayout_plus.onnx").write_bytes(b"current-layout-v2")
            (model_root / "table" / "slanet_plus.onnx").write_bytes(b"current-table-v2")
            (model_root / "table" / "table_cls.onnx").write_bytes(b"current-table-cls")
            (model_root / "formula" / "pp_formula_net_s.onnx").write_bytes(b"current-formula-v2")
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "dataset": str(dataset),
                        "pdf_path": str(dataset / "sample.pdf"),
                        "sample_name": "sample",
                        "total_elapsed_seconds": 7.0,
                        "ocr_version": "v5",
                        "layout_engine": "ppdoclayout",
                        "table_engine": "rapidtable",
                        "formula_mode": "pp_formula_net_s",
                        "reading_order_strategy": "rules",
                        "pipeline_config": {
                            "ocr_version": "v5",
                            "layout_engine": "ppdoclayout",
                            "table_engine": "rapidtable",
                            "formula_mode": "pp_formula_net_s",
                            "reading_order_strategy": "rules",
                        },
                        "model_manifest": {
                            "schema_version": "2026-06-08.cpu-pipeline-model-manifest.v1",
                            "groups": ["core_v5", "layout_v2", "table_v2", "formula_v2"],
                            "files": [
                                {
                                    "path": "rec_v5.onnx",
                                    "exists": True,
                                    "size_bytes": len(b"current-rec-v5"),
                                    "sha256": "stale-sha256",
                                }
                            ],
                        },
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "rasterize_ocr",
                            "slowest_stage_elapsed_seconds": 2.0,
                            "slowest_stage_share": 2.0 / 7.0,
                            "stages_by_elapsed_seconds": [
                                {"stage": "rasterize_ocr", "elapsed_seconds": 2.0, "share": 2.0 / 7.0},
                                {"stage": "layout", "elapsed_seconds": 1.5, "share": 1.5 / 7.0},
                                {"stage": "table", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "text_merge", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "cross_page_text", "elapsed_seconds": 0.75, "share": 0.75 / 7.0},
                                {"stage": "reading_order", "elapsed_seconds": 0.5, "share": 0.5 / 7.0},
                                {"stage": "extract_assets", "elapsed_seconds": 0.25, "share": 0.25 / 7.0},
                            ],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 2.0},
                            {"stage": "layout", "elapsed_seconds": 1.5},
                            {"stage": "table", "elapsed_seconds": 1.0},
                            {"stage": "text_merge", "elapsed_seconds": 1.0},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.75},
                            {"stage": "reading_order", "elapsed_seconds": 0.5},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"profile": profile},
                )

        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "model_manifest file rec_v5.onnx sha256 mismatch",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn(
            "model_manifest missing file entry: det_v5.onnx",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_profile_pipeline_can_record_dataset_relative_sample_name(self):
        profile_tool = importlib.import_module("tools.profile_pipeline")
        from deepdoc.parser import pdf_parser

        class FakeParser:
            def __images__(self, _source, _zoomin):
                return None

            def _layouts_rec(self, _zoomin):
                return None

            def _table_transformer_job(self, _zoomin):
                return None

            def _text_merge(self):
                return None

            def _merge_cross_page_text(self):
                return None

            def _apply_reading_order_strategy(self, _zoomin):
                return None

            def _extract_table_figure(self, *_args):
                return None

        with tempfile.TemporaryDirectory(prefix="deepdoc-profile-recursive-") as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            pdf_path = dataset / "contracts" / "nested-sample.pdf"
            pdf_path.parent.mkdir(parents=True)
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            with patch.object(pdf_parser, "DeepDocPdfParser", FakeParser):
                with patch.object(
                    profile_tool.time,
                    "perf_counter",
                    side_effect=[
                        0.0,
                        1.0,
                        1.0,
                        2.0,
                        2.0,
                        3.0,
                        3.0,
                        4.0,
                        4.0,
                        5.0,
                        5.0,
                        6.0,
                        6.0,
                        7.0,
                    ],
                ):
                    payload = profile_tool.profile_pipeline(pdf_path, dataset=dataset)

        self.assertEqual("contracts/nested-sample", payload["sample_name"])
        self.assertEqual(str(dataset), payload["dataset"])
        self.assertEqual(
            "2026-06-08.cpu-pipeline-dataset-contract.v1",
            payload["dataset_contract"]["schema_version"],
        )
        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertEqual(str(dataset), payload["dataset_contract"]["dataset"])
        self.assertEqual(1, payload["dataset_contract"]["sample_count"])
        self.assertEqual("contracts/nested-sample", payload["dataset_contract"]["samples"][0]["name"])

    def test_profile_pipeline_records_failed_dataset_contract_when_dataset_is_not_bound(self):
        profile_tool = importlib.import_module("tools.profile_pipeline")
        from deepdoc.parser import pdf_parser

        class FakeParser:
            def __images__(self, _source, _zoomin):
                return None

            def _layouts_rec(self, _zoomin):
                return None

            def _table_transformer_job(self, _zoomin):
                return None

            def _text_merge(self):
                return None

            def _merge_cross_page_text(self):
                return None

            def _apply_reading_order_strategy(self, _zoomin):
                return None

            def _extract_table_figure(self, *_args):
                return None

        with patch.object(pdf_parser, "DeepDocPdfParser", FakeParser):
            with patch.object(
                profile_tool.time,
                "perf_counter",
                side_effect=[
                    0.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    6.0,
                    6.0,
                    7.0,
                ],
            ):
                payload = profile_tool.profile_pipeline("/tmp/unbound-profile.pdf")

        self.assertIn("dataset_contract", payload)
        self.assertEqual("2026-06-08.cpu-pipeline-dataset-contract.v1", payload["dataset_contract"]["schema_version"])
        self.assertEqual("failed", payload["dataset_contract"]["status"])
        self.assertIsNone(payload["dataset_contract"]["dataset"])
        self.assertEqual(0, payload["dataset_contract"]["sample_count"])
        self.assertEqual([], payload["dataset_contract"]["samples"])
        self.assertIn("dataset path is required", payload["dataset_contract"]["problems"])

    def test_profile_pipeline_rejects_pdf_outside_dataset_contract(self):
        profile_tool = importlib.import_module("tools.profile_pipeline")
        from deepdoc.parser import pdf_parser

        class ExplodingParser:
            def __init__(self):
                raise AssertionError("parser should not be initialized for a dataset mismatch")

        with tempfile.TemporaryDirectory(prefix="deepdoc-profile-outside-dataset-") as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "sample.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
            outside_pdf = root / "outside.pdf"
            outside_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

            with patch.object(pdf_parser, "DeepDocPdfParser", ExplodingParser):
                with self.assertRaisesRegex(SystemExit, "Profile PDF must be one of dataset contract samples"):
                    profile_tool.profile_pipeline(outside_pdf, dataset=dataset)

    def test_cpu_pipeline_readiness_requires_profile_dataset_contract_to_match_current_dataset(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-contract-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            profile = root / "profile.json"
            model_root.mkdir()
            dataset.mkdir()
            pdf_path = dataset / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
            current_manifest = importlib.import_module("common.model_store").build_model_manifest(
                model_root=model_root,
                groups=("core", "formula"),
            )
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "dataset": str(dataset),
                        "dataset_contract": {
                            "schema_version": "2026-06-08.cpu-pipeline-dataset-contract.v1",
                            "status": "ok",
                            "dataset": str(dataset),
                            "sample_count": 1,
                            "problems": [],
                            "samples": [
                                {
                                    "name": "stale-sample",
                                    "pdf_path": str(dataset / "stale-sample.pdf"),
                                }
                            ],
                        },
                        "pdf_path": str(pdf_path),
                        "sample_name": "sample",
                        "total_elapsed_seconds": 7.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "model_manifest": current_manifest,
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "rasterize_ocr",
                            "slowest_stage_elapsed_seconds": 2.0,
                            "slowest_stage_share": 2.0 / 7.0,
                            "stages_by_elapsed_seconds": [
                                {"stage": "rasterize_ocr", "elapsed_seconds": 2.0, "share": 2.0 / 7.0},
                                {"stage": "layout", "elapsed_seconds": 1.5, "share": 1.5 / 7.0},
                                {"stage": "table", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "text_merge", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "cross_page_text", "elapsed_seconds": 0.75, "share": 0.75 / 7.0},
                                {"stage": "reading_order", "elapsed_seconds": 0.5, "share": 0.5 / 7.0},
                                {"stage": "extract_assets", "elapsed_seconds": 0.25, "share": 0.25 / 7.0},
                            ],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 2.0},
                            {"stage": "layout", "elapsed_seconds": 1.5},
                            {"stage": "table", "elapsed_seconds": 1.0},
                            {"stage": "text_merge", "elapsed_seconds": 1.0},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.75},
                            {"stage": "reading_order", "elapsed_seconds": 0.5},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"profile": profile},
                )

        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            "dataset_contract samples names must match readiness dataset contract: "
            "missing=sample, unexpected=stale-sample",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_cpu_pipeline_readiness_rejects_profile_dataset_contract_pdf_path_mismatch(self):
        readiness_tool = importlib.import_module("tools.check_cpu_pipeline_readiness")

        with tempfile.TemporaryDirectory(prefix="deepdoc-readiness-profile-contract-path-") as temp_dir:
            root = Path(temp_dir)
            model_root = root / "models"
            dataset = root / "dataset"
            previous_dataset = root / "previous-dataset"
            profile = root / "profile.json"
            model_root.mkdir()
            dataset.mkdir()
            previous_dataset.mkdir()
            current_pdf = dataset / "sample.pdf"
            stale_pdf = previous_dataset / "sample.pdf"
            current_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            stale_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
            current_manifest = importlib.import_module("common.model_store").build_model_manifest(
                model_root=model_root,
                groups=("core", "formula"),
            )
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": "2026-06-08.cpu-pipeline-profile.v1",
                        "dataset": str(dataset),
                        "dataset_contract": {
                            "schema_version": "2026-06-08.cpu-pipeline-dataset-contract.v1",
                            "status": "ok",
                            "dataset": str(dataset),
                            "sample_count": 1,
                            "problems": [],
                            "samples": [
                                {
                                    "name": "sample",
                                    "pdf_path": str(stale_pdf),
                                }
                            ],
                        },
                        "pdf_path": str(current_pdf),
                        "sample_name": "sample",
                        "total_elapsed_seconds": 7.0,
                        "ocr_version": "v4",
                        "layout_engine": "legacy",
                        "table_engine": "tatr",
                        "formula_mode": "rapidlatex",
                        "reading_order_strategy": "legacy",
                        "pipeline_config": {
                            "ocr_version": "v4",
                            "layout_engine": "legacy",
                            "table_engine": "tatr",
                            "formula_mode": "rapidlatex",
                            "reading_order_strategy": "legacy",
                        },
                        "model_manifest": current_manifest,
                        "stage_summary": {
                            "stage_count": 7,
                            "slowest_stage": "rasterize_ocr",
                            "slowest_stage_elapsed_seconds": 2.0,
                            "slowest_stage_share": 2.0 / 7.0,
                            "stages_by_elapsed_seconds": [
                                {"stage": "rasterize_ocr", "elapsed_seconds": 2.0, "share": 2.0 / 7.0},
                                {"stage": "layout", "elapsed_seconds": 1.5, "share": 1.5 / 7.0},
                                {"stage": "table", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "text_merge", "elapsed_seconds": 1.0, "share": 1.0 / 7.0},
                                {"stage": "cross_page_text", "elapsed_seconds": 0.75, "share": 0.75 / 7.0},
                                {"stage": "reading_order", "elapsed_seconds": 0.5, "share": 0.5 / 7.0},
                                {"stage": "extract_assets", "elapsed_seconds": 0.25, "share": 0.25 / 7.0},
                            ],
                        },
                        "stages": [
                            {"stage": "rasterize_ocr", "elapsed_seconds": 2.0},
                            {"stage": "layout", "elapsed_seconds": 1.5},
                            {"stage": "table", "elapsed_seconds": 1.0},
                            {"stage": "text_merge", "elapsed_seconds": 1.0},
                            {"stage": "cross_page_text", "elapsed_seconds": 0.75},
                            {"stage": "reading_order", "elapsed_seconds": 0.5},
                            {"stage": "extract_assets", "elapsed_seconds": 0.25},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(readiness_tool, "_pdf_page_count", return_value=100):
                payload = readiness_tool.check_readiness(
                    model_root=model_root,
                    dataset=dataset,
                    min_pages=100,
                    report_paths={"profile": profile},
                )

        self.assertEqual("ok", payload["dataset_contract"]["status"])
        self.assertEqual("failed", payload["ab_reports"]["profile"]["status"])
        self.assertIn(
            f"dataset_contract samples[0].pdf_path must match readiness dataset contract sample sample: "
            f"expected={current_pdf}, got={stale_pdf}",
            payload["ab_reports"]["profile"]["problems"],
        )
        self.assertIn("failed_ab_report:profile", payload["failed_gates"])

    def test_eval_tool_computes_block_order_and_cross_page_metrics(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")

        predicted_blocks = [
            {"layout_type": "title", "text": "Report", "page_number": 1},
            {
                "layout_type": "text",
                "text": "Cross page paragraph",
                "page_number": 1,
                "merged_page_numbers": [1, 2],
            },
            {"layout_type": "table", "text": "A B", "page_number": 2},
        ]
        expected_blocks = [
            {"block_type": "title", "text": "Report", "page_numbers": [1]},
            {
                "block_type": "text",
                "text": "Cross page paragraph",
                "page_numbers": [1, 2],
                "metadata": {"cross_page": True},
            },
            {"block_type": "figure", "text": "Chart", "page_numbers": [2]},
            {"block_type": "table", "text": "A B", "page_numbers": [2]},
        ]

        row = eval_tool.evaluate_blocks(
            predicted_blocks=predicted_blocks,
            expected_blocks=expected_blocks,
        )

        self.assertEqual(3, row["predicted_block_count"])
        self.assertEqual(4, row["expected_block_count"])
        self.assertAlmostEqual(1.0, row["block_type_precision"])
        self.assertAlmostEqual(0.75, row["block_type_recall"])
        self.assertAlmostEqual(6 / 7, row["block_type_f1"])
        self.assertEqual(1, row["reading_order_edit_distance"])
        self.assertAlmostEqual(0.25, row["reading_order_normalized_edit_distance"])
        self.assertAlmostEqual(1.0, row["cross_page_merge_accuracy"])
        self.assertEqual(1, row["expected_cross_page_block_count"])

        summary = eval_tool._summarize([{"elapsed_seconds": 1.0, **row}])
        self.assertAlmostEqual(6 / 7, summary["mean_block_type_f1"])
        self.assertAlmostEqual(0.25, summary["mean_reading_order_normalized_edit_distance"])
        self.assertAlmostEqual(1.0, summary["mean_cross_page_merge_accuracy"])

    def test_eval_tool_uses_structured_parser_path_for_blocks(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")
        from deepdoc.parser import pdf_parser

        class FakeParser:
            def __call__(self, *_args, **_kwargs):
                raise AssertionError("__call__ should not be used for block evaluation")

            def parse_into_bboxes(self, filename, zoomin=3):
                self.filename = filename
                self.zoomin = zoomin
                return [{"layout_type": "text", "text": "structured text", "page_number": 1}]

        with patch.object(pdf_parser, "DeepDocPdfParser", FakeParser):
            blocks = eval_tool._parse_deepdoc_blocks(Path("/tmp/sample.pdf"), engine="deepdoc")

        self.assertEqual([{"layout_type": "text", "text": "structured text", "page_number": 1}], blocks)

    def test_eval_tool_computes_cer_wer_and_table_metrics(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")

        predicted_text = "hello brave world"
        expected_text = "hello world"
        self.assertEqual(6, eval_tool.compute_text_edit_distance(predicted_text, expected_text))
        self.assertAlmostEqual(6 / len(expected_text), eval_tool.character_error_rate(predicted_text, expected_text))
        self.assertAlmostEqual(0.5, eval_tool.word_error_rate(predicted_text, expected_text))

        predicted_tables = ["<table><tr><td>A</td><td>B</td></tr></table>"]
        expected_tables = ["<table><tr><td>A</td><td>C</td></tr></table>"]
        with patch.object(eval_tool, "compute_table_teds", return_value=0.5):
            row = eval_tool.evaluate_tables(predicted_tables=predicted_tables, expected_tables=expected_tables)

        self.assertEqual(1, row["predicted_table_count"])
        self.assertEqual(1, row["expected_table_count"])
        self.assertAlmostEqual(0.5, row["mean_table_teds"])
        self.assertAlmostEqual(0.5, row["mean_table_cell_f1"])

        summary = eval_tool._summarize(
            [
                {
                    "elapsed_seconds": 1.0,
                    "character_error_rate": 6 / len(expected_text),
                    "word_error_rate": 0.5,
                    "has_tables_gt": True,
                    **row,
                }
            ]
        )
        self.assertAlmostEqual(6 / len(expected_text), summary["mean_character_error_rate"])
        self.assertAlmostEqual(0.5, summary["mean_word_error_rate"])
        self.assertAlmostEqual(0.5, summary["mean_table_teds"])
        self.assertAlmostEqual(0.5, summary["mean_table_cell_f1"])
        self.assertEqual(1, summary["samples_with_tables_gt"])

    def test_eval_tool_computes_formula_quality_metrics(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")

        predicted_blocks = [
            {"layout_type": "equation", "text": "$$E=mc^2$$", "page_number": 1},
            {"layout_type": "formula", "text": "\\frac{a}{b}", "page_number": 1},
            {"layout_type": "text", "text": "not a formula", "page_number": 1},
        ]
        expected_formulas = [
            {"latex": "E=mc^2"},
            {"text": "\\frac{a}{c}"},
        ]

        row = eval_tool.evaluate_formulas(
            predicted_blocks=predicted_blocks,
            expected_formulas=expected_formulas,
        )

        self.assertEqual(2, row["predicted_formula_count"])
        self.assertEqual(2, row["expected_formula_count"])
        self.assertAlmostEqual(0.5, row["formula_exact_match_rate"])
        self.assertGreater(row["formula_normalized_edit_distance"], 0)
        self.assertLess(row["formula_normalized_edit_distance"], 1)

        summary = eval_tool._summarize(
            [
                {
                    "elapsed_seconds": 1.0,
                    "has_formulas_gt": True,
                    **row,
                }
            ]
        )
        self.assertAlmostEqual(row["formula_normalized_edit_distance"], summary["mean_formula_normalized_edit_distance"])
        self.assertAlmostEqual(0.5, summary["mean_formula_exact_match_rate"])
        self.assertEqual(1, summary["samples_with_formulas_gt"])

    def test_eval_tool_computes_chunk_coverage_and_business_field_hits(self):
        eval_tool = importlib.import_module("tools.eval_omnidocbench")

        predicted_blocks = [
            {"layout_type": "title", "text": "合同", "page_number": 1},
            {"layout_type": "text", "text": "甲方：常州测试有限公司", "page_number": 1},
            {"layout_type": "text", "text": "合同金额：12345元", "page_number": 2},
        ]
        expected_chunks = [
            {"text": "甲方：常州测试有限公司"},
            {"text": "缺失条款"},
        ]
        expected_fields = [
            {"name": "甲方", "value": "常州测试有限公司", "page_numbers": [1]},
            {"name": "金额", "value": "12345元", "page_numbers": [2]},
            {"name": "乙方", "value": "未出现公司", "page_numbers": [1]},
        ]

        chunk_row = eval_tool.evaluate_chunks(
            predicted_blocks=predicted_blocks,
            expected_chunks=expected_chunks,
        )
        field_row = eval_tool.evaluate_business_fields(
            predicted_blocks=predicted_blocks,
            expected_fields=expected_fields,
        )

        self.assertEqual(2, chunk_row["expected_chunk_count"])
        self.assertEqual(1, chunk_row["matched_chunk_count"])
        self.assertAlmostEqual(0.5, chunk_row["chunk_text_coverage"])
        self.assertEqual(3, field_row["expected_business_field_count"])
        self.assertEqual(2, field_row["matched_business_field_count"])
        self.assertAlmostEqual(2 / 3, field_row["business_field_location_hit_rate"])

        summary = eval_tool._summarize(
            [
                {
                    "elapsed_seconds": 1.0,
                    "has_chunks_gt": True,
                    "has_business_fields_gt": True,
                    **chunk_row,
                    **field_row,
                }
            ]
        )
        self.assertAlmostEqual(0.5, summary["mean_chunk_text_coverage"])
        self.assertAlmostEqual(2 / 3, summary["mean_business_field_location_hit_rate"])
        self.assertEqual(1, summary["samples_with_chunks_gt"])
        self.assertEqual(1, summary["samples_with_business_fields_gt"])

    def test_runtime_docs_and_deploy_configs_expose_cpu_pipeline_switches(self):
        required_switches = [
            "DEEPDOC_OCR_VERSION",
            "DEEPDOC_REC_IMAGE_SHAPE",
            "DEEPDOC_OCR_V4_REC_IMAGE_SHAPE",
            "DEEPDOC_OCR_V5_REC_IMAGE_SHAPE",
            "DEEPDOC_LAYOUT_ENGINE",
            "DEEPDOC_TABLE_ENGINE",
            "DEEPDOC_FORMULA_MODE",
            "DEEPDOC_READING_ORDER_STRATEGY",
        ]

        env_example = self._read_text(".env")
        for expected in [
            "DEEPDOC_OCR_VERSION=v4",
            "DEEPDOC_REC_IMAGE_SHAPE=",
            "DEEPDOC_OCR_V4_REC_IMAGE_SHAPE=",
            "DEEPDOC_OCR_V5_REC_IMAGE_SHAPE=",
            "DEEPDOC_LAYOUT_ENGINE=legacy",
            "DEEPDOC_TABLE_ENGINE=tatr",
            "DEEPDOC_FORMULA_MODE=rapidlatex",
            "DEEPDOC_READING_ORDER_STRATEGY=legacy",
        ]:
            self.assertIn(expected, env_example)

        for compose_path in ["docker-compose.yml"]:
            compose = self._read_text(compose_path)
            for switch in required_switches:
                self.assertIn(switch, compose, compose_path)

        for doc_path in ["README.md", "docs/API.md", "docs/PARSER_ENGINE_STRATEGY.md"]:
            doc = self._read_text(doc_path)
            for switch in required_switches:
                self.assertIn(switch, doc, doc_path)

        dataset_readme = self._read_text("tools/eval_datasets/README.md")
        self.assertIn("<name>.pdf", dataset_readme)
        self.assertIn("<name>.gt.txt", dataset_readme)
        self.assertIn("<name>.gt.blocks.json", dataset_readme)
        self.assertIn("<name>.gt.tables.html", dataset_readme)
        self.assertIn("<name>.gt.formulas.json", dataset_readme)
        self.assertIn("<name>.gt.chunks.json", dataset_readme)
        self.assertIn("<name>.gt.fields.json", dataset_readme)
        self.assertIn("Do not commit", dataset_readme)

        readme = self._read_text("README.md")
        api_doc = self._read_text("docs/API.md")
        strategy_doc = self._read_text("docs/PARSER_ENGINE_STRATEGY.md")
        self.assertIn("python tools/eval_omnidocbench.py --license-gate", readme)
        self.assertIn("python tools/eval_omnidocbench.py --validate-dataset", readme)
        self.assertIn("python tools/check_cpu_pipeline_readiness.py", readme)
        self.assertIn("python tools/eval_omnidocbench.py --engine deepdoc --dataset", api_doc)
        self.assertIn("python tools/eval_omnidocbench.py --validate-dataset", api_doc)
        self.assertIn("python tools/check_cpu_pipeline_readiness.py", api_doc)
        self.assertIn("python tools/profile_pipeline.py", api_doc)
        for expected in [
            "--ocr-baseline-report",
            "--ocr-candidate-report",
            "--layout-baseline-report",
            "--layout-candidate-report",
            "--table-baseline-report",
            "--table-candidate-report",
            "--formula-baseline-report",
            "--formula-candidate-report",
            "--profile-report",
            "--license-gate-report",
        ]:
            self.assertIn(expected, readme)
            self.assertIn(expected, api_doc)
        for expected in [
            "cpu-pipeline-dataset-contract",
            "dataset_contract_failed",
            "A/B 报告必须与本次 readiness 的 `--dataset` 一致",
            "`sample_count` 必须是正整数并等于 dataset contract 识别到的样本数",
            "A/B 报告必须包含 `samples` 明细",
            "`summary.sample_count` 还必须是正整数并等于 `len(samples)`",
            "`samples` 必须是数组，且每一项都必须是 JSON object",
            "summary 中由 samples 明细产生的均值指标必须在每个 sample 中都有对应指标值",
            "summary 和 samples 指标都必须是数值且有限，JSON boolean 不能作为数值字段",
            "summary 均值必须按全量 samples 明细计算并一致",
            "samples 明细里的样本名和声明的 `pdf_path` 也必须匹配 dataset contract",
            "A/B 报告必须声明 `engine=deepdoc`",
            "若 `samples` 明细中声明了 `engine` 也必须是 `deepdoc`",
            "A/B 报告自身也必须包含 `license_gate`",
            "readiness 会校验 A/B 报告内嵌 `license_gate` 的 allowed/blocked 候选覆盖",
            "A/B 报告自身也必须包含 `dataset_contract`",
            "readiness 会校验 A/B 报告内嵌 `dataset_contract` 的 schema/status/sample_count/samples",
            "内嵌 `dataset_contract.samples` 必须是数组，且每一项都必须是 JSON object",
            "递归发现 `--dataset` 下的 PDF",
            "子目录样本名使用 dataset-relative 路径",
            "正式 `--engine deepdoc --dataset ...` 评测也会先执行 dataset contract 预检",
        ]:
            self.assertIn(expected, readme)
            self.assertIn(expected, api_doc)
        for expected in [
            "cpu-pipeline-dataset-contract",
            "dataset_contract_failed",
            "recursive",
            "dataset-relative",
        ]:
            self.assertIn(expected, dataset_readme)
        for expected in [
            "ocr_dictionaries",
            "ocr_recognition_alignments",
            "model_group_provenance",
            "rec_v5.onnx",
            "ocr_v5.res",
            "DEEPDOC_REC_IMAGE_SHAPE",
            "DEEPDOC_OCR_V4_REC_IMAGE_SHAPE",
            "DEEPDOC_OCR_V5_REC_IMAGE_SHAPE",
            "python download_models.py table_v2",
            "全部声明模型组",
            "远端 manifest",
            "模型组来源/许可证",
            "python tools/profile_pipeline.py tools/eval_datasets/biz_mini/contracts/sample.pdf",
            "--dataset tools/eval_datasets/biz_mini",
        ]:
            self.assertIn(expected, readme)
            self.assertIn(expected, api_doc)
        self.assertIn("ocr_recognition_alignments", strategy_doc)
        self.assertIn("DEEPDOC_OCR_V5_REC_IMAGE_SHAPE", strategy_doc)
        for expected in [
            ".gt.txt",
            ".gt.blocks.json",
            ".gt.tables.html",
            "character_error_rate",
            "word_error_rate",
            "block_type_f1",
            "mean_table_teds",
            "mean_table_cell_f1",
            "mean_formula_normalized_edit_distance",
            "mean_formula_exact_match_rate",
            "chunk_text_coverage",
            "business_field_location_hit_rate",
            "DEEPDOC_READING_ORDER_STRATEGY",
            "reading_order_strategy",
            "pipeline_config",
            "model_manifest 会按报告的 `pipeline_config` 只记录相关模型组",
            "profile 报告必须记录 `pipeline_config`（含 `formula_mode`）、`model_manifest`",
            "profile 阶段列表只能包含这 7 个阶段，缺失或额外阶段都会失败",
            "profile 报告自身也必须包含 `dataset_contract`",
            "readiness 会校验 profile 内嵌 `dataset_contract` 的 schema/status/sample_count/samples",
            "每个 sample 都必须提供对应指标值",
            "profile 顶层 `dataset` 必须匹配本次 readiness 的 `--dataset`",
            "profile 顶层 `sample_name` 和 `pdf_path` 也必须指向 dataset contract 的同一样本",
            "profile 内嵌 `dataset_contract.samples[*].pdf_path` 若有声明，也必须与当前 readiness 数据集中的同名 PDF 一致",
            "profile 报告也必须对应本次 readiness 的 `--dataset`",
            "profile 报告中的 `total_elapsed_seconds`、`stages[*].elapsed_seconds`",
            "`stage_summary.stages_by_elapsed_seconds[*].share` 都必须是数值且有限",
            "`stage_summary.stages_by_elapsed_seconds[*]` 的 `share` 必须匹配对应阶段耗时占比",
            "JSON boolean 不能作为耗时或占比字段",
            "模型快照一致",
            "stage_summary",
            "slowest_stage",
            "cross_page_text",
            "extract_assets",
            "license gate",
            "cpu-pipeline-license-gate",
            "include-risky-sequence-models",
            "rec.onnx",
            "formula/decoder.onnx",
            "递归扫描",
            "模型组子目录",
        ]:
            self.assertIn(expected, readme)
            self.assertIn(expected, api_doc)
        self.assertIn("DEEPDOC_READING_ORDER_STRATEGY", strategy_doc)
        self.assertIn("PaddleX PP-FormulaNet-S 适配器", strategy_doc)

    def test_cpu_pipeline_plan_records_landed_scaffolding_and_remaining_gates(self):
        plan = self._read_text("plans/2026-06-08-deepdoc-cpu-pipeline-upgrade.md")
        for expected in [
            "当前实现状态",
            "脚手架已落地",
            "真实业务 PDF 基线仍需数据集后运行",
            "v5 权重/A-B 指标待真实模型与数据集验证",
            "OCR v4/v5 字典 manifest/CI 多字典校验已落地",
            "OCR rec 输出类别数与字典行数对齐校验已落地",
            "OCR rec_image_shape 版本化配置已落地",
            "model_group_provenance",
            "模型组来源/许可证",
            "CPU pipeline readiness 门禁已落地",
            "正式评测报告的 `model_manifest` 会按 `pipeline_config` 记录相关模型组",
            "DEEPDOC_READING_ORDER_STRATEGY=legacy|rules",
            "阅读顺序 rules 策略已落地",
            "`download_models.py all` 覆盖全部声明模型组",
            "profile 报告已统一记录含 `formula_mode` 的 pipeline_config、`dataset`、`sample_name`、`model_manifest`",
            "额外阶段",
            "profile 报告自身也会内嵌 `dataset_contract`",
            "profile 顶层 `dataset` 必须匹配 readiness 当前 `--dataset`",
            "profile `sample_name`/`pdf_path` 必须对应 readiness 当前 dataset contract 中的同一条 PDF 样本",
            "内嵌 `dataset_contract.samples` 必须是数组，且每一项都必须是 JSON object",
            "profile 内嵌 `dataset_contract.samples[*].pdf_path` 若有声明，也必须与当前 readiness 数据集中的同名 PDF 一致",
            "profile `model_manifest` 与当前 `--model-root` 一致",
            "profile 报告需对应 readiness 当前 dataset 中的 PDF",
            "递归发现评测 PDF",
            "dataset-relative 样本名",
            "A/B 报告声明 `engine=deepdoc`",
            "samples 明细中声明的 `engine` 也必须是 `deepdoc`",
            "samples 明细里的样本名和声明的 `pdf_path` 必须匹配 dataset contract",
            "A/B 报告内嵌 `license_gate`",
            "A/B 报告内嵌 `dataset_contract`",
            "A/B 报告必须包含 `samples` 明细",
            "`samples` 必须是数组，且每一项都必须是 JSON object",
            "stage_summary",
            "profile 耗时字段和 `stage_summary` 耗时/占比字段都必须是数值且有限，JSON boolean 不能作为耗时或占比字段",
            "按耗时排序的阶段列表必须包含每个阶段的耗时和占比",
            "最慢阶段",
            "递归扫描模型组子目录",
            "PP-FormulaNet-S 已通过 PaddleX `create_model(...).predict(...)` 适配器接入",
            "真实 PP-FormulaNet-S 权重/PaddleX 环境联调",
            "不引入 RAG、向量化、问答或回答生成",
        ]:
            self.assertIn(expected, plan)


if __name__ == "__main__":
    unittest.main()

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import download_models
from common.model_store import (
    DEFAULT_OCR_REQUIRED_CHARACTERS,
    DEFAULT_MODEL_REPO,
    download_groups,
    get_group_files,
    get_model_group_provenance,
    validate_ocr_dictionary,
    validate_ocr_recognition_model_alignment,
)
from tools import publish_models_to_hf
from tools.ci import verify_hf_models


def _write_ocr_dictionary(root: Path, characters: str, relative_path: str = "ocr.res") -> str:
    payload = "\n".join(characters) + "\n"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_recognition_model(root: Path, relative_path: str = "rec.onnx") -> str:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{relative_path}\n".encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(root: Path, sha256: str | dict[str, str]) -> None:
    sha256_by_path = sha256 if isinstance(sha256, dict) else {"ocr.res": sha256}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": relative_path,
                        "size_bytes": (root / relative_path).stat().st_size,
                        "sha256": file_sha256,
                    }
                    for relative_path, file_sha256 in sha256_by_path.items()
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _add_model_group_provenance(manifest: dict, groups: str) -> dict:
    manifest["model_group_provenance"] = get_model_group_provenance(groups)
    return manifest


class OcrDictionaryValidationTest(unittest.TestCase):
    def test_validate_ocr_dictionary_reports_sha_and_required_character_coverage(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-ocr-dict-") as temp_dir:
            root = Path(temp_dir)
            sha256 = _write_ocr_dictionary(root, "中政A9")
            _write_manifest(root, sha256)

            report = validate_ocr_dictionary(
                model_root=root,
                required_characters="中政A9",
            )

        self.assertEqual("ok", report["status"])
        self.assertEqual("ocr.res", report["path"])
        self.assertEqual(sha256, report["sha256"])
        self.assertEqual(4, report["line_count"])
        self.assertEqual(4, report["unique_character_count"])
        self.assertEqual(sha256, report["expected_sha256"])
        self.assertTrue(report["sha256_matches"])
        self.assertEqual([], report["missing_required_characters"])
        self.assertEqual([], report["problems"])

    def test_validate_ocr_dictionary_fails_on_missing_characters_and_sha_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-ocr-dict-bad-") as temp_dir:
            root = Path(temp_dir)
            _write_ocr_dictionary(root, "中A9")

            report = validate_ocr_dictionary(
                model_root=root,
                expected_sha256="0" * 64,
                required_characters="中国政务A9",
            )

        self.assertEqual("failed", report["status"])
        self.assertFalse(report["sha256_matches"])
        self.assertEqual(["国", "政", "务"], report["missing_required_characters"])
        self.assertIn("sha256 mismatch", report["problems"])
        self.assertIn("missing required OCR characters: 国, 政, 务", report["problems"])

    def test_download_manifest_includes_ocr_dictionary_report(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-ocr-manifest-") as temp_dir:
            root = Path(temp_dir)
            sha256 = _write_ocr_dictionary(root, "".join(DEFAULT_OCR_REQUIRED_CHARACTERS))
            _write_manifest(root, sha256)

            buffer = io.StringIO()
            with patch.dict(os.environ, {"DEEPDOC_MODEL_PATH": str(root)}, clear=False):
                with redirect_stdout(buffer):
                    download_models.print_manifest()

        payload = json.loads(buffer.getvalue())
        ocr_report = payload["ocr_dictionary"]
        self.assertEqual("ok", ocr_report["status"])
        self.assertEqual(sha256, ocr_report["sha256"])
        self.assertEqual(sha256, ocr_report["expected_sha256"])
        self.assertEqual([], ocr_report["missing_required_characters"])

    def test_download_manifest_reports_all_declared_ocr_dictionaries(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-ocr-manifest-v5-") as temp_dir:
            root = Path(temp_dir)
            sha256_by_path = {
                "ocr.res": _write_ocr_dictionary(root, "".join(DEFAULT_OCR_REQUIRED_CHARACTERS), "ocr.res"),
                "ocr_v5.res": _write_ocr_dictionary(root, "".join(DEFAULT_OCR_REQUIRED_CHARACTERS), "ocr_v5.res"),
            }
            _write_manifest(root, sha256_by_path)

            buffer = io.StringIO()
            with patch.dict(os.environ, {"DEEPDOC_MODEL_PATH": str(root)}, clear=False):
                with redirect_stdout(buffer):
                    download_models.print_manifest()

        payload = json.loads(buffer.getvalue())
        self.assertEqual({"ocr.res", "ocr_v5.res"}, set(payload["ocr_dictionaries"]))
        self.assertEqual("ok", payload["ocr_dictionaries"]["ocr.res"]["status"])
        self.assertEqual("ok", payload["ocr_dictionaries"]["ocr_v5.res"]["status"])
        self.assertEqual(sha256_by_path["ocr.res"], payload["ocr_dictionaries"]["ocr.res"]["expected_sha256"])
        self.assertEqual(sha256_by_path["ocr_v5.res"], payload["ocr_dictionaries"]["ocr_v5.res"]["expected_sha256"])

    def test_validate_ocr_recognition_model_alignment_accounts_for_ctc_blank_and_space(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-ocr-align-") as temp_dir:
            root = Path(temp_dir)
            _write_recognition_model(root, "rec.onnx")
            _write_ocr_dictionary(root, "ABC", "ocr.res")

            with patch("common.model_store._get_onnx_output_class_count", return_value=5):
                ok_report = validate_ocr_recognition_model_alignment(
                    model_root=root,
                    rec_model_path="rec.onnx",
                    dictionary_path="ocr.res",
                )
            with patch("common.model_store._get_onnx_output_class_count", return_value=4):
                failed_report = validate_ocr_recognition_model_alignment(
                    model_root=root,
                    rec_model_path="rec.onnx",
                    dictionary_path="ocr.res",
                )

        self.assertEqual("ok", ok_report["status"])
        self.assertEqual(3, ok_report["dictionary_line_count"])
        self.assertEqual(5, ok_report["expected_class_count"])
        self.assertEqual(5, ok_report["model_output_class_count"])
        self.assertTrue(ok_report["class_count_matches"])

        self.assertEqual("failed", failed_report["status"])
        self.assertEqual(5, failed_report["expected_class_count"])
        self.assertEqual(4, failed_report["model_output_class_count"])
        self.assertFalse(failed_report["class_count_matches"])
        self.assertIn("OCR recognition class count mismatch", failed_report["problems"][0])

    def test_download_manifest_reports_ocr_recognition_dictionary_alignments(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-ocr-align-manifest-") as temp_dir:
            root = Path(temp_dir)
            characters = "".join(DEFAULT_OCR_REQUIRED_CHARACTERS)
            sha256_by_path = {
                "ocr.res": _write_ocr_dictionary(root, characters, "ocr.res"),
                "ocr_v5.res": _write_ocr_dictionary(root, characters, "ocr_v5.res"),
                "rec.onnx": _write_recognition_model(root, "rec.onnx"),
                "rec_v5.onnx": _write_recognition_model(root, "rec_v5.onnx"),
            }
            _write_manifest(root, sha256_by_path)
            expected_class_count = len(DEFAULT_OCR_REQUIRED_CHARACTERS) + 2

            buffer = io.StringIO()
            with patch.dict(os.environ, {"DEEPDOC_MODEL_PATH": str(root)}, clear=False):
                with patch("common.model_store._get_onnx_output_class_count", return_value=expected_class_count):
                    with redirect_stdout(buffer):
                        download_models.print_manifest()

        payload = json.loads(buffer.getvalue())
        self.assertEqual({"rec.onnx", "rec_v5.onnx"}, set(payload["ocr_recognition_alignments"]))
        self.assertEqual("ok", payload["ocr_recognition_alignments"]["rec.onnx"]["status"])
        self.assertEqual("ocr.res", payload["ocr_recognition_alignments"]["rec.onnx"]["dictionary_path"])
        self.assertEqual("ok", payload["ocr_recognition_alignments"]["rec_v5.onnx"]["status"])
        self.assertEqual("ocr_v5.res", payload["ocr_recognition_alignments"]["rec_v5.onnx"]["dictionary_path"])

    def test_publish_manifest_includes_model_group_provenance(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-publish-provenance-") as temp_dir:
            manifest = publish_models_to_hf.build_manifest(
                temp_dir,
                "core_v5,layout_v2,table_v2,formula_v2",
                repo_id="repo/deepdoc",
            )

        provenance = manifest["model_group_provenance"]
        self.assertEqual("PP-OCRv5", provenance["core_v5"]["component"])
        self.assertEqual("PP-DocLayout", provenance["layout_v2"]["component"])
        self.assertEqual("SLANet-plus", provenance["table_v2"]["component"])
        self.assertEqual("PP-FormulaNet-S", provenance["formula_v2"]["component"])
        for group in ("core_v5", "layout_v2", "table_v2", "formula_v2"):
            with self.subTest(group=group):
                self.assertEqual("Apache-2.0", provenance[group]["license"])
                self.assertEqual("document_parser", provenance[group]["scope"])
                self.assertFalse(provenance[group]["default_enabled"])

    def test_published_download_group_targets_remote_published_files_only(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-published-download-") as temp_dir:
            with patch("common.model_store.snapshot_download", return_value=temp_dir) as snapshot_download:
                resolved_root = download_groups("published", model_root=temp_dir)

        allow_patterns = snapshot_download.call_args.kwargs["allow_patterns"]
        self.assertEqual(str(Path(temp_dir).resolve()), resolved_root)
        self.assertEqual(DEFAULT_MODEL_REPO, snapshot_download.call_args.kwargs["repo_id"])
        self.assertIn("manifest.json", allow_patterns)
        self.assertIn("det_v5.onnx", allow_patterns)
        self.assertIn("layout/pp_doclayout_plus.onnx", allow_patterns)
        self.assertIn("formula/pp_formula_net_s.onnx", allow_patterns)
        self.assertIn("table/slanet_plus.onnx", allow_patterns)
        self.assertNotIn("rec_handwriting.onnx", allow_patterns)
        self.assertEqual(sorted(set(get_group_files("published")).union({"README.md", ".gitattributes", "manifest.json"})), allow_patterns)

    def test_verify_hf_models_includes_local_ocr_dictionary_report(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-verify-models-") as temp_dir:
            root = Path(temp_dir)
            sha256_by_path: dict[str, str] = {}
            for relative_path in verify_hf_models.get_group_files("core"):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative_path == "ocr.res":
                    sha256_by_path[relative_path] = _write_ocr_dictionary(
                        root,
                        "".join(DEFAULT_OCR_REQUIRED_CHARACTERS),
                    )
                elif relative_path == "rec.onnx":
                    sha256_by_path[relative_path] = _write_recognition_model(root, relative_path)
                else:
                    path.write_bytes(f"{relative_path}\n".encode("utf-8"))
                    sha256_by_path[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
            remote_manifest = {
                "groups": "core",
                "files": [
                    {
                        "path": relative_path,
                        "size_bytes": (root / relative_path).stat().st_size,
                        "sha256": sha256,
                    }
                    for relative_path, sha256 in sha256_by_path.items()
                ],
            }
            _add_model_group_provenance(remote_manifest, "core")
            remote_sizes = {
                relative_path: (root / relative_path).stat().st_size
                for relative_path in sha256_by_path
            }

            class FakeHfApi:
                def __init__(self, token=None):
                    self.token = token

                def list_repo_files(self, repo_id, repo_type, token=None):
                    return list(sha256_by_path)

            with patch.object(verify_hf_models, "HfApi", FakeHfApi):
                with patch.object(verify_hf_models, "_load_remote_manifest", return_value=remote_manifest):
                    with patch.object(verify_hf_models, "_remote_sizes", return_value=remote_sizes):
                        with patch(
                            "common.model_store._get_onnx_output_class_count",
                            return_value=len(DEFAULT_OCR_REQUIRED_CHARACTERS) + 2,
                        ):
                            result = verify_hf_models.verify("repo/deepdoc", root, "core", remote_only=False)

        self.assertEqual("ok", result["status"])
        ocr_report = result["ocr_dictionary"]
        self.assertEqual("ok", ocr_report["status"])
        self.assertEqual(sha256_by_path["ocr.res"], ocr_report["sha256"])
        self.assertEqual(sha256_by_path["ocr.res"], ocr_report["expected_sha256"])
        self.assertEqual([], ocr_report["missing_required_characters"])
        self.assertEqual("ok", result["ocr_recognition_alignments"]["rec.onnx"]["status"])
        self.assertEqual([], result["ocr_recognition_alignment_problems"])

    def test_verify_hf_models_requires_remote_manifest_model_group_provenance(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-verify-provenance-") as temp_dir:
            root = Path(temp_dir)
            sha256_by_path: dict[str, str] = {}
            for relative_path in verify_hf_models.get_group_files("core_v5"):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative_path == "ocr_v5.res":
                    sha256_by_path[relative_path] = _write_ocr_dictionary(
                        root,
                        "".join(DEFAULT_OCR_REQUIRED_CHARACTERS),
                        relative_path,
                    )
                elif relative_path == "rec_v5.onnx":
                    sha256_by_path[relative_path] = _write_recognition_model(root, relative_path)
                else:
                    path.write_bytes(f"{relative_path}\n".encode("utf-8"))
                    sha256_by_path[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
            remote_manifest = {
                "groups": "core_v5",
                "files": [
                    {
                        "path": relative_path,
                        "size_bytes": (root / relative_path).stat().st_size,
                        "sha256": sha256,
                    }
                    for relative_path, sha256 in sha256_by_path.items()
                ],
            }
            remote_sizes = {
                relative_path: (root / relative_path).stat().st_size
                for relative_path in sha256_by_path
            }

            class FakeHfApi:
                def __init__(self, token=None):
                    self.token = token

                def list_repo_files(self, repo_id, repo_type, token=None):
                    return list(sha256_by_path)

            with patch.object(verify_hf_models, "HfApi", FakeHfApi):
                with patch.object(verify_hf_models, "_load_remote_manifest", return_value=remote_manifest):
                    with patch.object(verify_hf_models, "_remote_sizes", return_value=remote_sizes):
                        with patch(
                            "common.model_store._get_onnx_output_class_count",
                            return_value=len(DEFAULT_OCR_REQUIRED_CHARACTERS) + 2,
                        ):
                            result = verify_hf_models.verify("repo/deepdoc", root, "core_v5", remote_only=False)

        self.assertEqual("failed", result["status"])
        self.assertIn(
            "missing model_group_provenance for group: core_v5",
            result["model_group_provenance_problems"],
        )
        self.assertIn("model_group_provenance_problems", result)

    def test_verify_hf_models_reports_every_required_ocr_dictionary(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-verify-models-v5-") as temp_dir:
            root = Path(temp_dir)
            sha256_by_path: dict[str, str] = {}
            for relative_path in verify_hf_models.get_group_files("core,core_v5"):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative_path in {"ocr.res", "ocr_v5.res"}:
                    sha256_by_path[relative_path] = _write_ocr_dictionary(
                        root,
                        "".join(DEFAULT_OCR_REQUIRED_CHARACTERS),
                        relative_path,
                    )
                elif relative_path in {"rec.onnx", "rec_v5.onnx"}:
                    sha256_by_path[relative_path] = _write_recognition_model(root, relative_path)
                else:
                    path.write_bytes(f"{relative_path}\n".encode("utf-8"))
                    sha256_by_path[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
            remote_manifest = {
                "groups": "core,core_v5",
                "files": [
                    {
                        "path": relative_path,
                        "size_bytes": (root / relative_path).stat().st_size,
                        "sha256": sha256,
                    }
                    for relative_path, sha256 in sha256_by_path.items()
                ],
            }
            _add_model_group_provenance(remote_manifest, "core,core_v5")
            remote_sizes = {
                relative_path: (root / relative_path).stat().st_size
                for relative_path in sha256_by_path
            }

            class FakeHfApi:
                def __init__(self, token=None):
                    self.token = token

                def list_repo_files(self, repo_id, repo_type, token=None):
                    return list(sha256_by_path)

            with patch.object(verify_hf_models, "HfApi", FakeHfApi):
                with patch.object(verify_hf_models, "_load_remote_manifest", return_value=remote_manifest):
                    with patch.object(verify_hf_models, "_remote_sizes", return_value=remote_sizes):
                        with patch(
                            "common.model_store._get_onnx_output_class_count",
                            return_value=len(DEFAULT_OCR_REQUIRED_CHARACTERS) + 2,
                        ):
                            result = verify_hf_models.verify("repo/deepdoc", root, "core,core_v5", remote_only=False)

        self.assertEqual("ok", result["status"])
        self.assertEqual({"ocr.res", "ocr_v5.res"}, set(result["ocr_dictionaries"]))
        self.assertEqual("ok", result["ocr_dictionaries"]["ocr.res"]["status"])
        self.assertEqual("ok", result["ocr_dictionaries"]["ocr_v5.res"]["status"])
        self.assertEqual([], result["ocr_dictionary_problems"])
        self.assertEqual("ok", result["ocr_recognition_alignments"]["rec.onnx"]["status"])
        self.assertEqual("ok", result["ocr_recognition_alignments"]["rec_v5.onnx"]["status"])

    def test_verify_hf_models_fails_when_recognition_model_and_dictionary_class_counts_diverge(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-verify-align-bad-") as temp_dir:
            root = Path(temp_dir)
            sha256_by_path: dict[str, str] = {}
            for relative_path in verify_hf_models.get_group_files("core,core_v5"):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                if relative_path in {"ocr.res", "ocr_v5.res"}:
                    sha256_by_path[relative_path] = _write_ocr_dictionary(
                        root,
                        "".join(DEFAULT_OCR_REQUIRED_CHARACTERS),
                        relative_path,
                    )
                elif relative_path in {"rec.onnx", "rec_v5.onnx"}:
                    sha256_by_path[relative_path] = _write_recognition_model(root, relative_path)
                else:
                    path.write_bytes(f"{relative_path}\n".encode("utf-8"))
                    sha256_by_path[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
            remote_manifest = {
                "groups": "core,core_v5",
                "files": [
                    {
                        "path": relative_path,
                        "size_bytes": (root / relative_path).stat().st_size,
                        "sha256": sha256,
                    }
                    for relative_path, sha256 in sha256_by_path.items()
                ],
            }
            _add_model_group_provenance(remote_manifest, "core,core_v5")
            remote_sizes = {
                relative_path: (root / relative_path).stat().st_size
                for relative_path in sha256_by_path
            }

            class FakeHfApi:
                def __init__(self, token=None):
                    self.token = token

                def list_repo_files(self, repo_id, repo_type, token=None):
                    return list(sha256_by_path)

            def fake_output_class_count(path: str | Path) -> int:
                return len(DEFAULT_OCR_REQUIRED_CHARACTERS) + (1 if Path(path).name == "rec_v5.onnx" else 2)

            with patch.object(verify_hf_models, "HfApi", FakeHfApi):
                with patch.object(verify_hf_models, "_load_remote_manifest", return_value=remote_manifest):
                    with patch.object(verify_hf_models, "_remote_sizes", return_value=remote_sizes):
                        with patch("common.model_store._get_onnx_output_class_count", side_effect=fake_output_class_count):
                            result = verify_hf_models.verify("repo/deepdoc", root, "core,core_v5", remote_only=False)

        self.assertEqual("failed", result["status"])
        self.assertEqual("ok", result["ocr_recognition_alignments"]["rec.onnx"]["status"])
        self.assertEqual("failed", result["ocr_recognition_alignments"]["rec_v5.onnx"]["status"])
        self.assertTrue(
            any("rec_v5.onnx" in problem for problem in result["ocr_recognition_alignment_problems"]),
            result["ocr_recognition_alignment_problems"],
        )


if __name__ == "__main__":
    unittest.main()

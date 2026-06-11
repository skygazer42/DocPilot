import unittest
import sys
from unittest.mock import patch

from deepdoc.parser import pdf_parser


class FakeOcr:
    constructed = 0

    def __init__(self):
        type(self).constructed += 1


class FakeLayout:
    constructed = 0

    def __init__(self, domain):
        type(self).constructed += 1
        self.domain = domain


class FakeAscendLayout(FakeLayout):
    pass


class FakeTableStructure:
    constructed = 0

    def __init__(self):
        type(self).constructed += 1


class FakeBooster:
    constructed = 0
    loaded_paths: list[str] = []

    def __init__(self):
        type(self).constructed += 1
        self.params = []

    def set_param(self, params):
        self.params.append(params)

    def load_model(self, path):
        type(self).loaded_paths.append(path)


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return True


class PdfParserSingletonTest(unittest.TestCase):
    def setUp(self):
        pdf_parser.clear_shared_pdf_parser_components()
        FakeOcr.constructed = 0
        FakeLayout.constructed = 0
        FakeAscendLayout.constructed = 0
        FakeTableStructure.constructed = 0
        FakeBooster.constructed = 0
        FakeBooster.loaded_paths = []

    def tearDown(self):
        pdf_parser.clear_shared_pdf_parser_components()

    def test_deepdoc_pdf_parser_reuses_heavy_model_components_between_instances(self):
        with patch.object(pdf_parser, "OCR", FakeOcr), patch.object(
            pdf_parser, "LayoutRecognizer", FakeLayout
        ), patch.object(pdf_parser, "AscendLayoutRecognizer", FakeAscendLayout), patch.object(
            pdf_parser, "TableStructureRecognizer", FakeTableStructure
        ), patch.object(pdf_parser.xgb, "Booster", FakeBooster), patch.object(
            pdf_parser, "ensure_groups", return_value="/models"
        ), patch.object(
            pdf_parser, "pip_install_torch", return_value=None
        ), patch.dict(
            pdf_parser.os.environ,
            {
                "LAYOUT_RECOGNIZER_TYPE": "onnx",
                "DEEPDOC_LAYOUT_MODEL": "manual",
            },
            clear=False,
        ):
            first = pdf_parser.DeepDocPdfParser()
            second = pdf_parser.DeepDocPdfParser()

        self.assertIs(first.ocr, second.ocr)
        self.assertIs(first.layouter, second.layouter)
        self.assertIs(first.tbl_det, second.tbl_det)
        self.assertIs(first.updown_cnt_mdl, second.updown_cnt_mdl)
        self.assertEqual(1, FakeOcr.constructed)
        self.assertEqual(1, FakeLayout.constructed)
        self.assertEqual(0, FakeAscendLayout.constructed)
        self.assertEqual(1, FakeTableStructure.constructed)
        self.assertEqual(1, FakeBooster.constructed)
        self.assertEqual(["/models/updown_concat_xgb.model"], FakeBooster.loaded_paths)

        state = pdf_parser.shared_pdf_parser_component_state()
        self.assertEqual(1, state["cached_component_count"])
        self.assertEqual(2, state["components"][0]["ref_count"])
        self.assertEqual("onnx", state["components"][0]["layout_recognizer_type"])
        self.assertEqual("layout", state["components"][0]["recognizer_domain"])

    def test_parser_instances_keep_parse_state_separate_while_reusing_components(self):
        with patch.object(pdf_parser, "OCR", FakeOcr), patch.object(
            pdf_parser, "LayoutRecognizer", FakeLayout
        ), patch.object(pdf_parser, "AscendLayoutRecognizer", FakeAscendLayout), patch.object(
            pdf_parser, "TableStructureRecognizer", FakeTableStructure
        ), patch.object(pdf_parser.xgb, "Booster", FakeBooster), patch.object(
            pdf_parser, "ensure_groups", return_value="/models"
        ), patch.object(
            pdf_parser, "pip_install_torch", return_value=None
        ), patch.dict(
            pdf_parser.os.environ,
            {"LAYOUT_RECOGNIZER_TYPE": "onnx"},
            clear=False,
        ):
            first = pdf_parser.DeepDocPdfParser()
            second = pdf_parser.DeepDocPdfParser()

        first.boxes = [{"text": "first"}]
        second.boxes = [{"text": "second"}]
        first.page_images = ["first-page"]
        second.page_images = ["second-page"]

        self.assertIs(first.ocr, second.ocr)
        self.assertEqual([{"text": "first"}], first.boxes)
        self.assertEqual([{"text": "second"}], second.boxes)
        self.assertEqual(["first-page"], first.page_images)
        self.assertEqual(["second-page"], second.page_images)

    def test_component_cache_key_separates_layout_backend_and_domain(self):
        with patch.object(pdf_parser, "OCR", FakeOcr), patch.object(
            pdf_parser, "LayoutRecognizer", FakeLayout
        ), patch.object(pdf_parser, "AscendLayoutRecognizer", FakeAscendLayout), patch.object(
            pdf_parser, "TableStructureRecognizer", FakeTableStructure
        ), patch.object(pdf_parser.xgb, "Booster", FakeBooster), patch.object(
            pdf_parser, "ensure_groups", return_value="/models"
        ), patch.object(
            pdf_parser, "pip_install_torch", return_value=None
        ):
            with patch.dict(pdf_parser.os.environ, {"LAYOUT_RECOGNIZER_TYPE": "onnx"}, clear=False):
                onnx_parser = pdf_parser.DeepDocPdfParser()
            with patch.dict(pdf_parser.os.environ, {"LAYOUT_RECOGNIZER_TYPE": "ascend"}, clear=False):
                ascend_parser = pdf_parser.DeepDocPdfParser()

        self.assertIsNot(onnx_parser.layouter, ascend_parser.layouter)
        self.assertEqual(1, FakeLayout.constructed)
        self.assertEqual(1, FakeAscendLayout.constructed)
        self.assertEqual(2, pdf_parser.shared_pdf_parser_component_state()["cached_component_count"])

    def test_load_updown_concat_model_uses_cuda_when_torch_reports_available(self):
        with patch.object(pdf_parser.xgb, "Booster", FakeBooster), patch.dict(sys.modules, {"torch": FakeTorch}):
            model = pdf_parser._load_updown_concat_model("/models")

        self.assertIsInstance(model, FakeBooster)
        self.assertEqual([{"device": "cuda"}], model.params)
        self.assertEqual(["/models/updown_concat_xgb.model"], FakeBooster.loaded_paths)


if __name__ == "__main__":
    unittest.main()

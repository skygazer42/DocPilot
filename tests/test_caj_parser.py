import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import main
from deepdoc.parser.caj_parser import DeepDocCajParser


def _native_pdf_meta():
    return {
        "page_count": 1,
        "total_page_count": 1,
        "pdf_parse_mode": "native_text",
        "deepdoc_pdf_mode": "auto",
        "structured_source": {
            "engine": "native_pdf",
            "boxes": [
                {
                    "text": "Converted CAJ text.",
                    "page_number": 0,
                    "layout_type": "text",
                    "x0": 10,
                    "x1": 100,
                    "top": 20,
                    "bottom": 40,
                }
            ],
        },
    }


class CajParserTest(unittest.TestCase):
    def test_caj_contract_is_pdf_conversion_not_rag_surface(self):
        repo_root = Path(__file__).resolve().parents[1]
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        openapi = (repo_root / "openapi.json").read_text(encoding="utf-8")

        self.assertEqual(("deepdoc.parser.caj_parser", "DeepDocCajParser"), main.PARSER_IMPORTS["caj"])
        self.assertEqual(("deepdoc.parser.caj_parser", "DeepDocCajParser"), main._parser_import_spec("caj"))
        self.assertEqual("caj", main._infer_file_type("sample.caj"))
        self.assertIn("| D4 | **CAJ** | 已落地", roadmap)
        self.assertIn("caj2pdf", api_doc)
        self.assertIn("caj", openapi)
        forbidden_path = "/api/v1/" + "rag"
        self.assertNotIn(forbidden_path, api_doc)

    def test_caj_parser_runs_configured_converter_and_validates_pdf_output(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-caj-parser-") as temp_dir:
            temp_path = Path(temp_dir)
            converter = temp_path / "fake_caj2pdf.py"
            converter.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "Path(sys.argv[2]).write_bytes(b'%PDF-1.4\\n% converted\\n')\n",
                encoding="utf-8",
            )
            input_path = temp_path / "sample.caj"
            input_path.write_bytes(b"CAJ payload")
            output_dir = temp_path / "out"

            with patch.dict(
                os.environ,
                {"DEEPDOC_CAJ2PDF_COMMAND_TEMPLATE": f"python {converter} {{input}} {{output}}"},
                clear=False,
            ):
                result = DeepDocCajParser().convert_to_pdf(input_path, output_dir=output_dir)

            self.assertTrue(Path(result.pdf_path).exists())
            self.assertEqual(b"%PDF", Path(result.pdf_path).read_bytes()[:4])
            self.assertEqual("caj2pdf", result.converter)
            self.assertGreater(result.output_size_bytes, 0)

    def test_parse_endpoint_converts_caj_then_reuses_pdf_structured_artifact(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-caj-endpoint-") as temp_dir:
            converted_pdf = Path(temp_dir) / "converted.pdf"
            converted_pdf.write_bytes(b"%PDF-1.4\n")

            def fake_convert(self, input_path, *, output_dir=None, request_timeout=None):
                return type(
                    "Result",
                    (),
                    {
                        "pdf_path": str(converted_pdf),
                        "output_dir": str(output_dir or temp_dir),
                        "converter": "caj2pdf",
                        "command": ["fake-caj2pdf"],
                        "output_size_bytes": converted_pdf.stat().st_size,
                        "model_dump": lambda self: {
                            "pdf_path": str(converted_pdf),
                            "converter": "caj2pdf",
                            "output_size_bytes": converted_pdf.stat().st_size,
                        },
                    },
                )()

            with patch.object(DeepDocCajParser, "convert_to_pdf", fake_convert), patch.object(
                main,
                "_parse_pdf_from_tmp",
                return_value=([("Converted CAJ text.", "")], [], _native_pdf_meta()),
            ) as parse_pdf:
                response = main.app.test_client().post(
                    "/api/v1/parse",
                    data={
                        "file": (BytesIO(b"CAJ payload"), "sample.caj"),
                        "return_structured": "true",
                        "persist_artifacts": "false",
                        "include_chunks": "true",
                    },
                    content_type="multipart/form-data",
                )

        self.assertEqual(200, response.status_code, response.get_data(as_text=True))
        parse_pdf.assert_called_once()
        result = response.get_json()["results"][0]
        self.assertNotIn("error", result)
        self.assertEqual("caj", result["type"])
        self.assertEqual("deepdoc", result["parser_engine"])
        self.assertIn("Converted CAJ text.", result["markdown"])
        structured = result["structured"]
        self.assertEqual("caj", structured["document"]["file_type"])
        self.assertEqual("caj", structured["document"]["metadata"]["source_file_type"])
        self.assertEqual("pdf", structured["document"]["metadata"]["converted_file_type"])
        self.assertEqual("pdf_native_text", structured["metadata"]["source"])
        self.assertTrue(structured["chunks"])


if __name__ == "__main__":
    unittest.main()

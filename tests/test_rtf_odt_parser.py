import json
import importlib
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import main
from common.parse_artifacts import ParseDocument


def _load_symbol(module_name: str, symbol_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise AssertionError(f"Expected module {module_name} to exist") from exc
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise AssertionError(f"Expected {module_name}.{symbol_name} to exist") from exc


def _rtf_bytes() -> bytes:
    return (
        r"{\rtf1\ansi{\fonttbl{\f0 Arial;}}"
        r"{\b Contract Title}\par "
        r"First paragraph for RTF parsing.\par "
        r"Second paragraph keeps structure.}"
    ).encode("utf-8")


def _odt_bytes() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "content.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
  xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
  xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
  xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
  <office:body>
    <office:text>
      <text:h text:outline-level="1">ODT Heading</text:h>
      <text:p>First ODT paragraph.</text:p>
      <text:list>
        <text:list-item><text:p>First list item</text:p></text:list-item>
        <text:list-item><text:p>Second list item</text:p></text:list-item>
      </text:list>
      <table:table table:name="Amounts">
        <table:table-row>
          <table:table-cell><text:p>Name</text:p></table:table-cell>
          <table:table-cell><text:p>Amount</text:p></table:table-cell>
        </table:table-row>
        <table:table-row>
          <table:table-cell><text:p>Alice</text:p></table:table-cell>
          <table:table-cell><text:p>10</text:p></table:table-cell>
        </table:table-row>
      </table:table>
    </office:text>
  </office:body>
</office:document-content>
""",
        )
    return buffer.getvalue()


def _document(file_type: str) -> ParseDocument:
    return ParseDocument(
        document_id=f"doc-{file_type}",
        parse_id=f"parse-{file_type}",
        filename=f"sample.{file_type}",
        file_type=file_type,
        parser_engine="deepdoc",
        created_at="2026-06-08T00:00:00+00:00",
        source_sha256=f"{file_type}123",
        source_size_bytes=512,
    )


class RtfOdtParserTest(unittest.TestCase):
    def test_rtf_odt_contract_is_native_parser_not_rag_surface(self):
        repo_root = Path(__file__).resolve().parents[1]
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        openapi = json.loads((repo_root / "openapi.json").read_text(encoding="utf-8"))
        parse_schema = (
            openapi["paths"]["/api/v1/parse"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )

        self.assertEqual(("deepdoc.parser.rtf_parser", "DeepDocRtfParser"), main.PARSER_IMPORTS["rtf"])
        self.assertEqual(("deepdoc.parser.odt_parser", "DeepDocOdtParser"), main.PARSER_IMPORTS["odt"])
        self.assertEqual(("deepdoc.parser.rtf_parser", "DeepDocRtfParser"), main._parser_import_spec("rtf"))
        self.assertEqual(("deepdoc.parser.odt_parser", "DeepDocOdtParser"), main._parser_import_spec("odt"))
        self.assertIn("| D3 | **RTF/ODT** | 已落地", roadmap)
        self.assertIn("RTF/ODT", api_doc)
        self.assertIn("RTF", readme)
        self.assertIn("rtf/odt", parse_schema["properties"]["file"]["description"])

    def test_rtf_parser_returns_markdown_and_structured_blocks(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-rtf-") as temp_dir:
            path = Path(temp_dir) / "sample.rtf"
            path.write_bytes(_rtf_bytes())

            DeepDocRtfParser = _load_symbol("deepdoc.parser.rtf_parser", "DeepDocRtfParser")
            markdown, tables, meta = DeepDocRtfParser()(path)

        self.assertEqual([], tables)
        self.assertIn("# Contract Title", markdown)
        self.assertIn("First paragraph for RTF parsing.", markdown)
        structured_source = meta["structured_source"]
        self.assertEqual("rtf", structured_source["engine"])
        self.assertEqual(3, structured_source["block_count"])
        self.assertEqual(["title", "text", "text"], [block["block_type"] for block in structured_source["blocks"]])

    def test_odt_parser_returns_markdown_tables_and_structured_blocks(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-odt-") as temp_dir:
            path = Path(temp_dir) / "sample.odt"
            path.write_bytes(_odt_bytes())

            DeepDocOdtParser = _load_symbol("deepdoc.parser.odt_parser", "DeepDocOdtParser")
            markdown, tables, meta = DeepDocOdtParser()(path)

        self.assertEqual([], tables)
        self.assertIn("# ODT Heading", markdown)
        self.assertIn("- First list item", markdown)
        self.assertIn("| Name | Amount |", markdown)
        structured_source = meta["structured_source"]
        self.assertEqual("odt", structured_source["engine"])
        self.assertEqual(4, structured_source["block_count"])
        self.assertEqual(["title", "text", "list", "table"], [block["block_type"] for block in structured_source["blocks"]])
        self.assertEqual(2, structured_source["blocks"][3]["row_count"])
        self.assertEqual(2, structured_source["blocks"][3]["column_count"])

    def test_rich_text_artifact_builds_blocks_chunks_and_table_asset(self):
        build_rich_text_artifact = _load_symbol("common.parse_builders", "build_rich_text_artifact")

        artifact = build_rich_text_artifact(
            document=_document("odt"),
            markdown="# ODT Heading\n\nFirst ODT paragraph.\n\n| Name | Amount |\n| --- | --- |\n| Alice | 10 |",
            blocks=[
                {"block_type": "title", "text": "ODT Heading", "heading_level": 1},
                {"block_type": "text", "text": "First ODT paragraph."},
                {
                    "block_type": "table",
                    "text": "| Name | Amount |\n| --- | --- |\n| Alice | 10 |",
                    "row_count": 2,
                    "column_count": 2,
                },
            ],
            source="odt",
            source_metadata={"source_name": "sample.odt"},
            chunk_max_tokens=256,
            chunk_overlap_tokens=0,
            chunk_strategy="asset_aware",
            metadata={"parser_engine": "deepdoc", "chunk_strategy": "asset_aware"},
        )

        self.assertEqual("odt", artifact.metadata["source"])
        self.assertEqual(3, len(artifact.blocks))
        self.assertEqual(1, len(artifact.assets))
        self.assertEqual("table", artifact.assets[0].asset_type)
        self.assertEqual([artifact.assets[0].asset_id], artifact.blocks[2].asset_refs)
        self.assertTrue(any(chunk.asset_refs for chunk in artifact.chunks))

    def test_parse_endpoint_returns_structured_rtf_and_odt_artifacts(self):
        cases = [
            ("sample.rtf", _rtf_bytes(), "rtf", "Contract Title"),
            ("sample.odt", _odt_bytes(), "odt", "ODT Heading"),
        ]
        with main.app.test_client() as client:
            for filename, payload, file_type, expected_title in cases:
                with self.subTest(file_type=file_type):
                    response = client.post(
                        "/api/v1/parse",
                        data={
                            "file": (BytesIO(payload), filename),
                            "return_structured": "true",
                            "persist_artifacts": "false",
                            "include_chunks": "true",
                            "chunk_strategy": "asset_aware",
                        },
                        content_type="multipart/form-data",
                    )

                    self.assertEqual(200, response.status_code, response.get_data(as_text=True))
                    result = response.get_json()["results"][0]
                    self.assertNotIn("error", result)
                    self.assertEqual(file_type, result["type"])
                    self.assertEqual("deepdoc", result["parser_engine"])
                    structured = result["structured"]
                    self.assertEqual(file_type, structured["metadata"]["source"])
                    self.assertEqual(expected_title, structured["blocks"][0]["text"])
                    self.assertTrue(structured["chunks"])


if __name__ == "__main__":
    unittest.main()

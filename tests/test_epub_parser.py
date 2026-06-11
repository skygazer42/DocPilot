import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import main
from common.parse_artifacts import ParseDocument
from common.parse_builders import build_epub_artifact
from deepdoc.parser.epub_parser import DeepDocEpubParser


def _minimal_epub_bytes() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
""",
        )
        zf.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="UTF-8"?>
<package version="3.0" unique-identifier="bookid" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Sample EPUB</dc:title>
    <dc:creator>DeepDoc Tests</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="chap1" href="chapters/chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chap2" href="chapters/chapter2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chap1"/>
    <itemref idref="chap2"/>
  </spine>
</package>
""",
        )
        zf.writestr(
            "OEBPS/chapters/chapter1.xhtml",
            """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>Chapter One</title></head>
  <body>
    <h1>Chapter One</h1>
    <p>First paragraph in chapter one.</p>
    <ul><li>First bullet</li><li>Second bullet</li></ul>
  </body>
</html>
""",
        )
        zf.writestr(
            "OEBPS/chapters/chapter2.xhtml",
            """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>Chapter Two</title></head>
  <body>
    <h2>Chapter Two</h2>
    <p>Second chapter paragraph.</p>
    <table>
      <tr><th>Name</th><th>Amount</th></tr>
      <tr><td>Alice</td><td>10</td></tr>
    </table>
  </body>
</html>
""",
        )
    return buffer.getvalue()


def _document() -> ParseDocument:
    return ParseDocument(
        document_id="doc-epub",
        parse_id="parse-epub",
        filename="sample.epub",
        file_type="epub",
        parser_engine="deepdoc",
        created_at="2026-06-08T00:00:00+00:00",
        source_sha256="epub123",
        source_size_bytes=512,
    )


class EpubParserTest(unittest.TestCase):
    def test_epub_contract_is_native_parser_not_markitdown_fallback(self):
        repo_root = Path(__file__).resolve().parents[1]
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")
        openapi = json.loads((repo_root / "openapi.json").read_text(encoding="utf-8"))
        parse_schema = (
            openapi["paths"]["/api/v1/parse"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )

        self.assertEqual(("deepdoc.parser.epub_parser", "DeepDocEpubParser"), main.PARSER_IMPORTS["epub"])
        self.assertEqual(("deepdoc.parser.epub_parser", "DeepDocEpubParser"), main._parser_import_spec("epub"))
        self.assertEqual(("deepdoc.parser.markitdown_parser", "MarkItDownParser"), main._parser_import_spec("epub", "markitdown"))
        self.assertIn("epub", parse_schema["properties"]["file"]["description"])
        self.assertIn("| D1 | **EPUB** | 已落地", roadmap)

    def test_epub_parser_reads_opf_spine_order_and_returns_structured_source(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-epub-") as temp_dir:
            path = Path(temp_dir) / "sample.epub"
            path.write_bytes(_minimal_epub_bytes())

            markdown, tables, meta = DeepDocEpubParser()(path)

        self.assertIn("# Chapter One", markdown)
        self.assertIn("First paragraph in chapter one.", markdown)
        self.assertIn("- First bullet", markdown)
        self.assertIn("## Chapter Two", markdown)
        self.assertIn("| Name | Amount |", markdown)
        self.assertLess(markdown.index("# Chapter One"), markdown.index("## Chapter Two"))
        self.assertEqual([], tables)
        structured_source = meta["structured_source"]
        self.assertEqual("epub", structured_source["engine"])
        self.assertEqual("Sample EPUB", structured_source["metadata"]["title"])
        self.assertEqual(2, structured_source["chapter_count"])
        self.assertEqual(6, len(structured_source["blocks"]))
        self.assertEqual(["title", "text", "list", "title", "text", "table"], [block["block_type"] for block in structured_source["blocks"]])

    def test_epub_artifact_builds_blocks_chunks_and_table_asset(self):
        blocks = [
            {"block_type": "title", "text": "Chapter One", "chapter_index": 0, "href": "chapter1.xhtml"},
            {"block_type": "text", "text": "First paragraph in chapter one.", "chapter_index": 0, "href": "chapter1.xhtml"},
            {"block_type": "table", "text": "| Name | Amount |\n| --- | --- |\n| Alice | 10 |", "chapter_index": 1, "href": "chapter2.xhtml", "row_count": 2, "column_count": 2},
        ]

        artifact = build_epub_artifact(
            document=_document(),
            markdown="# Chapter One\n\nFirst paragraph in chapter one.\n\n| Name | Amount |\n| --- | --- |\n| Alice | 10 |",
            blocks=blocks,
            epub_metadata={"title": "Sample EPUB"},
            chunk_max_tokens=256,
            chunk_overlap_tokens=0,
            chunk_strategy="asset_aware",
            metadata={"parser_engine": "deepdoc", "chunk_strategy": "asset_aware"},
        )

        self.assertEqual("epub", artifact.metadata["source"])
        self.assertEqual(3, len(artifact.blocks))
        self.assertEqual("title", artifact.blocks[0].block_type)
        self.assertEqual("table", artifact.blocks[2].block_type)
        self.assertEqual(1, len(artifact.assets))
        self.assertEqual("table", artifact.assets[0].asset_type)
        self.assertEqual(2, artifact.assets[0].metadata["row_count"])
        self.assertEqual([artifact.assets[0].asset_id], artifact.blocks[2].asset_refs)
        self.assertTrue(any(chunk.asset_refs for chunk in artifact.chunks))

    def test_parse_endpoint_returns_structured_epub_artifact(self):
        with main.app.test_client() as client:
            response = client.post(
                "/api/v1/parse",
                data={
                    "file": (BytesIO(_minimal_epub_bytes()), "sample.epub"),
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
        self.assertEqual("epub", result["type"])
        self.assertEqual("deepdoc", result["parser_engine"])
        structured = result["structured"]
        self.assertEqual("epub", structured["metadata"]["source"])
        self.assertEqual("Sample EPUB", structured["metadata"]["epub_metadata"]["title"])
        self.assertEqual(["title", "text", "list", "title", "text", "table"], [block["block_type"] for block in structured["blocks"]])
        self.assertEqual("table", structured["assets"][0]["asset_type"])
        self.assertTrue(structured["chunks"])


if __name__ == "__main__":
    unittest.main()

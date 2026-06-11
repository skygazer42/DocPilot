import json
import importlib
import unittest
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


def _eml_bytes() -> bytes:
    return (
        "From: Alice <alice@example.com>\r\n"
        "To: Bob <bob@example.com>\r\n"
        "Subject: Contract Notice\r\n"
        "Date: Mon, 8 Jun 2026 09:30:00 +0800\r\n"
        "MIME-Version: 1.0\r\n"
        "Content-Type: multipart/mixed; boundary=deepdoc-mail\r\n"
        "\r\n"
        "--deepdoc-mail\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        "Please review the attached contract.\r\n"
        "The due date is Friday.\r\n"
        "\r\n"
        "--deepdoc-mail\r\n"
        "Content-Type: application/pdf\r\n"
        "Content-Disposition: attachment; filename=\"contract.pdf\"\r\n"
        "\r\n"
        "fake-pdf-bytes\r\n"
        "--deepdoc-mail\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "Content-Disposition: attachment; filename=\"notes.txt\"\r\n"
        "\r\n"
        "Attachment note should be recursively parsed.\r\n"
        "--deepdoc-mail--\r\n"
    ).encode("utf-8")


def _msg_fallback_bytes() -> bytes:
    return (
        "From: Carol <carol@example.com>\r\n"
        "To: DeepDoc <deepdoc@example.com>\r\n"
        "Subject: MSG Fallback Notice\r\n"
        "Date: Mon, 8 Jun 2026 10:00:00 +0800\r\n"
        "\r\n"
        "Fallback MSG body parsed as message text.\r\n"
    ).encode("utf-8")


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


class EmailParserTest(unittest.TestCase):
    def test_email_contract_is_document_parser_not_rag_surface(self):
        repo_root = Path(__file__).resolve().parents[1]
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        openapi = json.loads((repo_root / "openapi.json").read_text(encoding="utf-8"))
        parse_schema = (
            openapi["paths"]["/api/v1/parse"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )

        self.assertEqual(("deepdoc.parser.email_parser", "DeepDocEmailParser"), main.PARSER_IMPORTS["eml"])
        self.assertEqual(("deepdoc.parser.email_parser", "DeepDocEmailParser"), main.PARSER_IMPORTS["msg"])
        self.assertEqual(("deepdoc.parser.email_parser", "DeepDocEmailParser"), main._parser_import_spec("eml"))
        self.assertEqual(("deepdoc.parser.email_parser", "DeepDocEmailParser"), main._parser_import_spec("msg"))
        self.assertIn("| D5 | **邮件 eml/msg** | 已落地", roadmap)
        self.assertIn("eml/msg", api_doc)
        self.assertIn("邮件", readme)
        self.assertIn("eml/msg", parse_schema["properties"]["file"]["description"])

    def test_eml_parser_extracts_headers_body_attachments_and_structured_blocks(self):
        DeepDocEmailParser = _load_symbol("deepdoc.parser.email_parser", "DeepDocEmailParser")

        markdown, tables, meta = DeepDocEmailParser().parser_bytes(
            _eml_bytes(),
            source_name="notice.eml",
            source_type="eml",
        )

        self.assertEqual([], tables)
        self.assertIn("# Contract Notice", markdown)
        self.assertIn("Please review the attached contract.", markdown)
        self.assertIn("contract.pdf", markdown)
        structured_source = meta["structured_source"]
        self.assertEqual("email", structured_source["engine"])
        self.assertEqual("eml", structured_source["metadata"]["source_type"])
        self.assertEqual("Contract Notice", structured_source["metadata"]["subject"])
        self.assertEqual(2, structured_source["metadata"]["attachment_count"])
        self.assertEqual(1, structured_source["metadata"]["parseable_attachment_count"])
        self.assertEqual("contract.pdf", structured_source["attachments"][0]["filename"])
        self.assertEqual(
            ["title", "text", "list", "text"],
            [block["block_type"] for block in structured_source["blocks"]],
        )
        self.assertEqual("attachment_text", structured_source["blocks"][3]["metadata"]["email_block_role"])
        self.assertIn("Attachment note should be recursively parsed.", structured_source["blocks"][3]["text"])

    def test_msg_parser_falls_back_to_message_text_when_extract_msg_is_unavailable(self):
        DeepDocEmailParser = _load_symbol("deepdoc.parser.email_parser", "DeepDocEmailParser")

        markdown, tables, meta = DeepDocEmailParser().parser_bytes(
            _msg_fallback_bytes(),
            source_name="notice.msg",
            source_type="msg",
        )

        self.assertEqual([], tables)
        self.assertIn("# MSG Fallback Notice", markdown)
        self.assertIn("Fallback MSG body parsed as message text.", markdown)
        structured_source = meta["structured_source"]
        self.assertEqual("email", structured_source["engine"])
        self.assertEqual("msg", structured_source["metadata"]["source_type"])
        self.assertEqual("MSG Fallback Notice", structured_source["metadata"]["subject"])

    def test_email_artifact_uses_rich_text_blocks_chunks_and_attachment_list(self):
        build_rich_text_artifact = _load_symbol("common.parse_builders", "build_rich_text_artifact")

        artifact = build_rich_text_artifact(
            document=_document("eml"),
            markdown="# Contract Notice\n\nPlease review the attached contract.\n\n- contract.pdf",
            blocks=[
                {"block_type": "title", "text": "Contract Notice", "heading_level": 1},
                {"block_type": "text", "text": "Please review the attached contract."},
                {"block_type": "list", "text": "- contract.pdf"},
            ],
            source="email",
            source_metadata={"source_name": "notice.eml", "source_type": "eml", "attachment_count": 1},
            chunk_max_tokens=256,
            chunk_overlap_tokens=0,
            chunk_strategy="asset_aware",
            metadata={"parser_engine": "deepdoc", "chunk_strategy": "asset_aware"},
        )

        self.assertEqual("email", artifact.metadata["source"])
        self.assertEqual("eml", artifact.metadata["source_metadata"]["source_type"])
        self.assertEqual(3, len(artifact.blocks))
        self.assertTrue(artifact.chunks)

    def test_parse_endpoint_returns_structured_email_artifacts(self):
        cases = [
            ("notice.eml", _eml_bytes(), "eml", "Contract Notice"),
            ("notice.msg", _msg_fallback_bytes(), "msg", "MSG Fallback Notice"),
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
                    self.assertEqual("email", structured["metadata"]["source"])
                    self.assertEqual(expected_title, structured["blocks"][0]["text"])
                    self.assertTrue(structured["chunks"])


if __name__ == "__main__":
    unittest.main()

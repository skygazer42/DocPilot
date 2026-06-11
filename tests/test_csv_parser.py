import tempfile
import unittest
import json
from io import BytesIO
from pathlib import Path

import main
from common.parse_artifacts import LocalArtifactStore, ParseDocument, build_chunk_export_records
from common.parse_builders import build_csv_artifact
from deepdoc.parser.csv_parser import DeepDocCsvParser


def _document(file_type: str = "csv") -> ParseDocument:
    return ParseDocument(
        document_id="doc-csv",
        parse_id="parse-csv",
        filename=f"sample.{file_type}",
        file_type=file_type,
        parser_engine="deepdoc",
        created_at="2026-06-08T00:00:00+00:00",
        source_sha256="csv123",
        source_size_bytes=64,
    )


class CsvParserTest(unittest.TestCase):
    def test_csv_tsv_contract_is_documented(self):
        repo_root = Path(__file__).resolve().parents[1]
        api_doc = (repo_root / "docs/API.md").read_text(encoding="utf-8")
        roadmap = (repo_root / "plans/optimization-roadmap.md").read_text(encoding="utf-8")
        openapi = json.loads((repo_root / "openapi.json").read_text(encoding="utf-8"))
        parse_schema = (
            openapi["paths"]["/api/v1/parse"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )
        async_schema = (
            openapi["paths"]["/api/v1/parse/async"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        )

        self.assertIn("csv/tsv", api_doc)
        self.assertIn("CSV/TSV", api_doc)
        self.assertIn("| D2 | **CSV/TSV** | 已落地", roadmap)
        for schema in (parse_schema, async_schema):
            self.assertIn("csv/tsv", schema["properties"]["file"]["description"])

    def test_main_routes_csv_and_tsv_to_structured_csv_parser(self):
        self.assertEqual(
            ("deepdoc.parser.csv_parser", "DeepDocCsvParser"),
            main.PARSER_IMPORTS["csv"],
        )
        self.assertEqual(
            ("deepdoc.parser.csv_parser", "DeepDocCsvParser"),
            main.PARSER_IMPORTS["tsv"],
        )
        self.assertEqual(
            ("deepdoc.parser.csv_parser", "DeepDocCsvParser"),
            main._parser_import_spec("csv"),
        )
        self.assertEqual(
            ("deepdoc.parser.csv_parser", "DeepDocCsvParser"),
            main._parser_import_spec("tsv"),
        )

    def test_csv_parser_returns_markdown_and_structured_rows(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-csv-") as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            path.write_text("name,amount\nAlice,10\nBob,20\n", encoding="utf-8")

            markdown, tables, meta = DeepDocCsvParser()(path)

        self.assertEqual([], tables)
        self.assertIn("| name | amount |", markdown)
        self.assertIn("| Alice | 10 |", markdown)
        structured_source = meta["structured_source"]
        self.assertEqual("csv", structured_source["engine"])
        self.assertEqual(",", structured_source["delimiter"])
        self.assertEqual(3, structured_source["row_count"])
        self.assertEqual(2, structured_source["column_count"])
        self.assertEqual([["name", "amount"], ["Alice", "10"], ["Bob", "20"]], structured_source["rows"])

    def test_tsv_parser_uses_tab_delimiter(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-tsv-") as temp_dir:
            path = Path(temp_dir) / "sample.tsv"
            path.write_text("name\tamount\nAlice\t10\n", encoding="utf-8")

            markdown, _tables, meta = DeepDocCsvParser()(path)

        self.assertIn("| name | amount |", markdown)
        self.assertEqual("\t", meta["structured_source"]["delimiter"])
        self.assertEqual([["name", "amount"], ["Alice", "10"]], meta["structured_source"]["rows"])

    def test_csv_artifact_builds_table_asset_block_chunk_and_export_context(self):
        rows = [["name", "amount"], ["Alice", "10"], ["Bob", "20"]]

        artifact = build_csv_artifact(
            document=_document(),
            markdown="| name | amount |\n|---|---|\n| Alice | 10 |\n| Bob | 20 |",
            rows=rows,
            delimiter=",",
            chunk_max_tokens=256,
            chunk_overlap_tokens=0,
            chunk_strategy="asset_aware",
            metadata={"parser_engine": "deepdoc", "chunk_strategy": "asset_aware"},
        )

        self.assertEqual(1, len(artifact.assets))
        self.assertEqual("table", artifact.assets[0].asset_type)
        self.assertEqual(1, len(artifact.blocks))
        self.assertEqual("table", artifact.blocks[0].block_type)
        self.assertEqual([artifact.assets[0].asset_id], artifact.blocks[0].asset_refs)
        self.assertEqual(1, len(artifact.chunks))
        self.assertEqual([artifact.assets[0].asset_id], artifact.chunks[0].asset_refs)
        self.assertEqual("2026-06-08.asset-context.v1", artifact.assets[0].metadata["asset_context_schema_version"])
        self.assertEqual([artifact.blocks[0].block_id], artifact.assets[0].metadata["direct_block_refs"])
        self.assertEqual([artifact.chunks[0].chunk_id], artifact.assets[0].metadata["direct_chunk_refs"])

        records = build_chunk_export_records(artifact)
        self.assertEqual(1, len(records))
        self.assertEqual("table", records[0].assets[0].asset_type)
        self.assertEqual(
            "2026-06-08.asset-context.v1",
            records[0].assets[0].metadata["asset_context_schema_version"],
        )

    def test_structured_artifact_uses_csv_source_metadata(self):
        file_bytes = b"name,amount\nAlice,10\n"
        artifact, artifact_paths, manifest = main._build_structured_artifact(
            filename="sample.csv",
            file_type="csv",
            file_bytes=file_bytes,
            markdown_content="| name | amount |\n|---|---|\n| Alice | 10 |",
            parse_options={
                "parser_engine": "deepdoc",
                "return_structured": True,
                "persist_artifacts": False,
                "include_chunks": True,
                "chunk_max_tokens": 256,
                "chunk_overlap_tokens": 0,
                "chunk_strategy": "asset_aware",
            },
            parse_meta={
                "structured_source": {
                    "engine": "csv",
                    "rows": [["name", "amount"], ["Alice", "10"]],
                    "delimiter": ",",
                }
            },
            artifact_profile={"test": "csv"},
            artifact_key="artifact-key-csv",
        )

        self.assertIsNone(artifact_paths)
        self.assertIsNone(manifest)
        self.assertEqual(1, len(artifact.assets))
        self.assertEqual("table", artifact.assets[0].asset_type)
        self.assertEqual("table", artifact.blocks[0].block_type)
        self.assertEqual("csv", artifact.metadata["source"])
        self.assertEqual("asset_aware_v1", artifact.chunks[0].metadata["chunk_strategy"])

    def test_parse_endpoint_returns_structured_csv_table_artifact(self):
        with main.app.test_client() as client:
            response = client.post(
                "/api/v1/parse",
                data={
                    "file": (BytesIO(b"name,amount\nAlice,10\nBob,20\n"), "sample.csv"),
                    "return_structured": "true",
                    "persist_artifacts": "false",
                    "include_chunks": "true",
                    "chunk_strategy": "asset_aware",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(200, response.status_code, response.get_data(as_text=True))
        payload = response.get_json()
        result = payload["results"][0]
        self.assertNotIn("error", result)
        self.assertEqual("csv", result["type"])
        structured = result["structured"]
        self.assertEqual(1, len(structured["assets"]))
        self.assertEqual("table", structured["assets"][0]["asset_type"])
        self.assertEqual("table", structured["blocks"][0]["block_type"])
        self.assertEqual(structured["assets"][0]["asset_id"], structured["blocks"][0]["asset_refs"][0])
        self.assertEqual("csv", structured["metadata"]["source"])
        self.assertEqual("asset_aware_v1", structured["chunks"][0]["metadata"]["chunk_strategy"])
        self.assertEqual(
            "2026-06-08.asset-context.v1",
            structured["assets"][0]["metadata"]["asset_context_schema_version"],
        )

    def test_parse_endpoint_reuses_persisted_csv_artifact_by_artifact_key(self):
        original_store = main.ARTIFACT_STORE
        with tempfile.TemporaryDirectory(prefix="deepdoc-csv-cache-") as temp_dir:
            main.ARTIFACT_STORE = LocalArtifactStore(root_dir=temp_dir)
            try:
                with main.app.test_client() as client:
                    first_response = client.post(
                        "/api/v1/parse",
                        data={
                            "file": (BytesIO(b"name,amount\nAlice,10\nBob,20\n"), "sample.csv"),
                            "return_structured": "true",
                            "persist_artifacts": "true",
                            "include_chunks": "true",
                            "chunk_strategy": "asset_aware",
                        },
                        content_type="multipart/form-data",
                    )
                    second_response = client.post(
                        "/api/v1/parse",
                        data={
                            "file": (BytesIO(b"name,amount\nAlice,10\nBob,20\n"), "sample.csv"),
                            "return_structured": "true",
                            "persist_artifacts": "true",
                            "include_chunks": "true",
                            "chunk_strategy": "asset_aware",
                            "reuse_artifacts": "true",
                        },
                        content_type="multipart/form-data",
                    )
            finally:
                main.ARTIFACT_STORE = original_store

        self.assertEqual(200, first_response.status_code, first_response.get_data(as_text=True))
        self.assertEqual(200, second_response.status_code, second_response.get_data(as_text=True))
        first_result = first_response.get_json()["results"][0]
        second_result = second_response.get_json()["results"][0]
        self.assertFalse(first_result["cache_hit"])
        self.assertTrue(second_result["cache_hit"])
        self.assertEqual(first_result["parse_id"], second_result["parse_id"])
        self.assertEqual(first_result["document_id"], second_result["document_id"])
        self.assertEqual(first_result["markdown"], second_result["markdown"])
        self.assertEqual(first_result["structured"]["document"]["parse_id"], second_result["structured"]["document"]["parse_id"])
        self.assertEqual(1, second_result["asset_count"])
        self.assertEqual(1, second_result["chunk_count"])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from tools.ci.artifact_quality_report import evaluate_artifact_root


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _valid_asset_metadata() -> dict:
    return {
        "asset_context_schema_version": "2026-06-08.asset-context.v1",
        "asset_summary_schema_version": "2026-06-08.asset-summary.v1",
        "asset_summary_source": "local_rules",
        "asset_summary": "Table on page 1 with 1 rows and 2 columns. Text: | A | B |.",
        "asset_summary_facts": {
            "asset_type": "table",
            "page_numbers": [1],
            "row_count": 1,
            "column_count": 2,
            "text_length": 9,
        },
        "direct_block_refs": ["b2"],
        "context_block_refs": ["b1"],
        "direct_chunk_refs": ["chunk-2"],
        "context_chunk_refs": [],
        "chunk_refs": ["chunk-2"],
        "context_texts": ["Intro"],
    }


class ArtifactQualityReportTest(unittest.TestCase):
    def test_valid_artifact_root_passes_quality_report(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-quality-") as temp_dir:
            root = Path(temp_dir)
            parse_dir = root / "parse-1"
            structured = {
                "schema_version": "1.0.0",
                "document": {
                    "parse_id": "parse-1",
                    "document_id": "doc-1",
                    "filename": "sample.pdf",
                    "file_type": "pdf",
                    "parser_engine": "deepdoc",
                },
                "assets": [
                    {
                        "asset_id": "table-1",
                        "asset_type": "table",
                        "metadata": _valid_asset_metadata(),
                    }
                ],
                "blocks": [
                    {"block_id": "b1", "block_type": "text", "text": "Intro"},
                    {
                        "block_id": "b2",
                        "block_type": "table",
                        "text": "| A | B |",
                        "asset_refs": ["table-1"],
                    },
                ],
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "text": "Intro",
                        "block_refs": ["b1"],
                        "metadata": {"chunk_strategy": "asset_aware_v1"},
                    },
                    {
                        "chunk_id": "chunk-2",
                        "text": "[Table]\n| A | B |",
                        "block_refs": ["b2"],
                        "asset_refs": ["table-1"],
                        "metadata": {"chunk_strategy": "asset_aware_v1"},
                    },
                ],
            }
            _write_json(parse_dir / "structured.json", structured)
            _write_json(parse_dir / "manifest.json", {"parse_id": "parse-1", "chunk_count": 2})
            _write_jsonl(
                parse_dir / "chunks.jsonl",
                [
                    {
                        "chunk_id": "chunk-1",
                        "text": "Intro",
                        "metadata": {
                            "schema_version": "2026-06-08.chunk.v1",
                            "chunk_strategy": "asset_aware_v1",
                        },
                    },
                    {
                        "chunk_id": "chunk-2",
                        "text": "[Table]\n| A | B |",
                        "asset_refs": ["table-1"],
                        "assets": [
                            {
                                "asset_id": "table-1",
                                "asset_type": "table",
                                "metadata": _valid_asset_metadata(),
                            }
                        ],
                        "metadata": {
                            "schema_version": "2026-06-08.chunk.v1",
                            "chunk_strategy": "asset_aware_v1",
                        },
                    },
                ],
            )
            _write_jsonl(
                parse_dir / "ingest.jsonl",
                [
                    {
                        "record_id": "doc-1:chunk-1",
                        "chunk_id": "chunk-1",
                        "text": "Intro",
                        "metadata": {
                            "schema_version": "2026-06-08.ingest.v1",
                            "chunk_schema_version": "2026-06-08.chunk.v1",
                        },
                    },
                    {
                        "record_id": "doc-1:chunk-2",
                        "chunk_id": "chunk-2",
                        "text": "[Table]\n| A | B |",
                        "metadata": {
                            "schema_version": "2026-06-08.ingest.v1",
                            "chunk_schema_version": "2026-06-08.chunk.v1",
                        },
                    },
                ],
            )

            report = evaluate_artifact_root(root)

        self.assertEqual("passed", report["status"])
        self.assertEqual(1, report["parse_count"])
        self.assertEqual(2, report["totals"]["chunk_count"])
        self.assertEqual([], report["failures"])

    def test_invalid_artifact_root_reports_quality_failures(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-quality-") as temp_dir:
            root = Path(temp_dir)
            parse_dir = root / "parse-bad"
            _write_json(
                parse_dir / "structured.json",
                {
                    "document": {"parse_id": "parse-bad", "document_id": "doc-bad"},
                    "assets": [{"asset_id": "asset-missing"}],
                    "blocks": [{"block_id": "b1", "block_type": "text", "text": "Text"}],
                    "chunks": [
                        {
                            "chunk_id": "chunk-bad",
                            "text": "",
                            "block_refs": ["b-missing"],
                            "asset_refs": ["asset-unknown"],
                            "metadata": {},
                        }
                    ],
                },
            )
            _write_json(parse_dir / "manifest.json", {"parse_id": "parse-bad", "chunk_count": 2})
            _write_jsonl(parse_dir / "chunks.jsonl", [{"chunk_id": "chunk-bad", "text": "", "metadata": {}}])

            report = evaluate_artifact_root(root)

        self.assertEqual("failed", report["status"])
        self.assertTrue(any("missing ingest.jsonl" in failure for failure in report["failures"]))
        self.assertTrue(any("empty chunk text" in failure for failure in report["failures"]))
        self.assertTrue(any("unknown block_refs" in failure for failure in report["failures"]))
        self.assertTrue(any("unknown asset_refs" in failure for failure in report["failures"]))
        self.assertTrue(any("chunk schema_version" in failure for failure in report["failures"]))

    def test_asset_without_context_metadata_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-quality-") as temp_dir:
            root = Path(temp_dir)
            parse_dir = root / "parse-missing-asset-context"
            _write_json(
                parse_dir / "structured.json",
                {
                    "document": {
                        "parse_id": "parse-missing-asset-context",
                        "document_id": "doc-asset-context",
                    },
                    "assets": [{"asset_id": "table-1", "asset_type": "table", "metadata": {}}],
                    "blocks": [
                        {"block_id": "b1", "block_type": "text", "text": "Intro"},
                        {
                            "block_id": "b2",
                            "block_type": "table",
                            "text": "| A | B |",
                            "asset_refs": ["table-1"],
                        },
                    ],
                    "chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "text": "Intro",
                            "block_refs": ["b1"],
                            "metadata": {"chunk_strategy": "asset_aware_v1"},
                        },
                        {
                            "chunk_id": "chunk-2",
                            "text": "[Table]\n| A | B |",
                            "block_refs": ["b2"],
                            "asset_refs": ["table-1"],
                            "metadata": {"chunk_strategy": "asset_aware_v1"},
                        },
                    ],
                },
            )
            _write_json(parse_dir / "manifest.json", {"parse_id": "parse-missing-asset-context", "chunk_count": 2})
            _write_jsonl(
                parse_dir / "chunks.jsonl",
                [
                    {
                        "chunk_id": "chunk-1",
                        "text": "Intro",
                        "metadata": {
                            "schema_version": "2026-06-08.chunk.v1",
                            "chunk_strategy": "asset_aware_v1",
                        },
                    },
                    {
                        "chunk_id": "chunk-2",
                        "text": "[Table]\n| A | B |",
                        "asset_refs": ["table-1"],
                        "metadata": {
                            "schema_version": "2026-06-08.chunk.v1",
                            "chunk_strategy": "asset_aware_v1",
                        },
                    },
                ],
            )
            _write_jsonl(
                parse_dir / "ingest.jsonl",
                [
                    {
                        "record_id": "doc-asset-context:chunk-1",
                        "chunk_id": "chunk-1",
                        "text": "Intro",
                        "metadata": {
                            "schema_version": "2026-06-08.ingest.v1",
                            "chunk_schema_version": "2026-06-08.chunk.v1",
                        },
                    },
                    {
                        "record_id": "doc-asset-context:chunk-2",
                        "chunk_id": "chunk-2",
                        "text": "[Table]\n| A | B |",
                        "metadata": {
                            "schema_version": "2026-06-08.ingest.v1",
                            "chunk_schema_version": "2026-06-08.chunk.v1",
                        },
                    },
                ],
            )

            report = evaluate_artifact_root(root)

        self.assertEqual("failed", report["status"])
        self.assertTrue(any("asset context schema_version" in failure for failure in report["failures"]))

    def test_asset_without_summary_metadata_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="deepdoc-quality-") as temp_dir:
            root = Path(temp_dir)
            parse_dir = root / "parse-missing-asset-summary"
            asset_metadata = {
                "asset_context_schema_version": "2026-06-08.asset-context.v1",
                "direct_block_refs": ["b2"],
                "context_block_refs": ["b1"],
                "direct_chunk_refs": ["chunk-2"],
                "context_chunk_refs": [],
                "chunk_refs": ["chunk-2"],
                "context_texts": ["Intro"],
            }
            _write_json(
                parse_dir / "structured.json",
                {
                    "document": {
                        "parse_id": "parse-missing-asset-summary",
                        "document_id": "doc-asset-summary",
                    },
                    "assets": [{"asset_id": "table-1", "asset_type": "table", "metadata": asset_metadata}],
                    "blocks": [
                        {"block_id": "b1", "block_type": "text", "text": "Intro"},
                        {
                            "block_id": "b2",
                            "block_type": "table",
                            "text": "| A | B |",
                            "asset_refs": ["table-1"],
                        },
                    ],
                    "chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "text": "Intro",
                            "block_refs": ["b1"],
                            "metadata": {"chunk_strategy": "asset_aware_v1"},
                        },
                        {
                            "chunk_id": "chunk-2",
                            "text": "[Table]\n| A | B |",
                            "block_refs": ["b2"],
                            "asset_refs": ["table-1"],
                            "metadata": {"chunk_strategy": "asset_aware_v1"},
                        },
                    ],
                },
            )
            _write_json(parse_dir / "manifest.json", {"parse_id": "parse-missing-asset-summary", "chunk_count": 2})
            _write_jsonl(
                parse_dir / "chunks.jsonl",
                [
                    {
                        "chunk_id": "chunk-1",
                        "text": "Intro",
                        "metadata": {
                            "schema_version": "2026-06-08.chunk.v1",
                            "chunk_strategy": "asset_aware_v1",
                        },
                    },
                    {
                        "chunk_id": "chunk-2",
                        "text": "[Table]\n| A | B |",
                        "asset_refs": ["table-1"],
                        "assets": [{"asset_id": "table-1", "asset_type": "table", "metadata": asset_metadata}],
                        "metadata": {
                            "schema_version": "2026-06-08.chunk.v1",
                            "chunk_strategy": "asset_aware_v1",
                        },
                    },
                ],
            )
            _write_jsonl(
                parse_dir / "ingest.jsonl",
                [
                    {
                        "record_id": "doc-asset-summary:chunk-1",
                        "chunk_id": "chunk-1",
                        "text": "Intro",
                        "metadata": {
                            "schema_version": "2026-06-08.ingest.v1",
                            "chunk_schema_version": "2026-06-08.chunk.v1",
                        },
                    },
                    {
                        "record_id": "doc-asset-summary:chunk-2",
                        "chunk_id": "chunk-2",
                        "text": "[Table]\n| A | B |",
                        "metadata": {
                            "schema_version": "2026-06-08.ingest.v1",
                            "chunk_schema_version": "2026-06-08.chunk.v1",
                        },
                    },
                ],
            )

            report = evaluate_artifact_root(root)

        self.assertEqual("failed", report["status"])
        self.assertTrue(any("asset summary schema_version" in failure for failure in report["failures"]))


if __name__ == "__main__":
    unittest.main()

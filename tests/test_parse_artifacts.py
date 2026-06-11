import unittest

from common.parse_artifacts import (
    CHUNK_EXPORT_SCHEMA_VERSION,
    INGEST_EXPORT_SCHEMA_VERSION,
    ASSET_SUMMARY_SCHEMA_VERSION,
    ParseArtifact,
    ParseAsset,
    ParseBlock,
    ParseDocument,
    build_chunk_export_records,
    build_chunks,
    build_ingest_export_records,
    enrich_asset_context,
)


def _document() -> ParseDocument:
    return ParseDocument(
        document_id="doc-1",
        parse_id="parse-1",
        filename="sample.pdf",
        file_type="pdf",
        parser_engine="deepdoc",
        created_at="2026-06-08T00:00:00+00:00",
        source_sha256="abc123",
        source_size_bytes=123,
        metadata={"tenant_id": "tenant-a"},
    )


class ParseArtifactChunkStrategyTest(unittest.TestCase):
    def test_asset_aware_strategy_keeps_table_block_as_own_chunk(self):
        blocks = [
            ParseBlock(
                block_id="b1",
                block_type="text",
                text="Intro paragraph before table.",
                page_numbers=[1],
                token_count=5,
            ),
            ParseBlock(
                block_id="b2",
                block_type="table",
                text="| A | B |\n|---|---|\n| 1 | 2 |",
                page_numbers=[1],
                token_count=12,
                asset_refs=["table-1"],
            ),
            ParseBlock(
                block_id="b3",
                block_type="text",
                text="Paragraph after table.",
                page_numbers=[1],
                token_count=4,
            ),
        ]

        chunks = build_chunks(
            blocks,
            max_tokens=256,
            overlap_tokens=0,
            strategy="asset_aware",
        )

        self.assertEqual(3, len(chunks))
        table_chunks = [chunk for chunk in chunks if chunk.block_refs == ["b2"]]
        self.assertEqual(1, len(table_chunks))
        self.assertEqual(["table"], table_chunks[0].metadata["block_types"])
        self.assertEqual(["table-1"], table_chunks[0].metadata["direct_asset_refs"])
        self.assertEqual("asset_aware_v1", table_chunks[0].metadata["chunk_strategy"])

    def test_page_aware_strategy_does_not_mix_pages_in_one_chunk(self):
        blocks = [
            ParseBlock(
                block_id="p1",
                block_type="text",
                text="Page one paragraph.",
                page_numbers=[1],
                token_count=4,
            ),
            ParseBlock(
                block_id="p2",
                block_type="text",
                text="Page two paragraph.",
                page_numbers=[2],
                token_count=4,
            ),
        ]

        chunks = build_chunks(
            blocks,
            max_tokens=256,
            overlap_tokens=0,
            strategy="page_aware",
        )

        self.assertEqual([[1], [2]], [chunk.page_numbers for chunk in chunks])
        self.assertEqual([["p1"], ["p2"]], [chunk.block_refs for chunk in chunks])
        self.assertTrue(all(chunk.metadata["chunk_strategy"] == "page_aware_v1" for chunk in chunks))

    def test_chunk_and_ingest_exports_include_schema_versions(self):
        asset = ParseAsset(
            asset_id="table-1",
            asset_type="table",
            text="table markdown",
            page_numbers=[1],
        )
        chunks = build_chunks(
            [
                ParseBlock(
                    block_id="b1",
                    block_type="table",
                    text="| A | B |",
                    page_numbers=[1],
                    token_count=5,
                    asset_refs=["table-1"],
                )
            ],
            max_tokens=256,
            overlap_tokens=0,
            strategy="asset_aware",
        )
        artifact = ParseArtifact(
            document=_document(),
            markdown="| A | B |",
            assets=[asset],
            blocks=[],
            chunks=chunks,
        )

        chunk_records = build_chunk_export_records(artifact)
        ingest_records = build_ingest_export_records(chunk_records)

        self.assertEqual(CHUNK_EXPORT_SCHEMA_VERSION, chunk_records[0].metadata["schema_version"])
        self.assertEqual(INGEST_EXPORT_SCHEMA_VERSION, ingest_records[0].metadata["schema_version"])
        self.assertEqual("asset_aware_v1", chunk_records[0].metadata["chunk_strategy"])

    def test_asset_context_enrichment_links_assets_back_to_neighbor_blocks_and_chunks(self):
        asset = ParseAsset(
            asset_id="table-1",
            asset_type="table",
            text="| A | B |",
            page_numbers=[1],
        )
        blocks = [
            ParseBlock(
                block_id="b1",
                block_type="text",
                text="Intro paragraph before table.",
                page_numbers=[1],
                token_count=5,
            ),
            ParseBlock(
                block_id="b2",
                block_type="table",
                text=asset.text,
                page_numbers=[1],
                token_count=5,
                asset_refs=[asset.asset_id],
            ),
            ParseBlock(
                block_id="b3",
                block_type="text",
                text="Paragraph after table.",
                page_numbers=[1],
                token_count=4,
            ),
            ParseBlock(
                block_id="b4",
                block_type="text",
                text="Different page should not be linked.",
                page_numbers=[2],
                token_count=6,
            ),
        ]
        chunks = build_chunks(blocks, max_tokens=256, overlap_tokens=0, strategy="asset_aware")

        enriched_assets = enrich_asset_context([asset], blocks, chunks, window=1)
        metadata = enriched_assets[0].metadata

        self.assertEqual(["b2"], metadata["direct_block_refs"])
        self.assertEqual(["b1", "b3"], metadata["context_block_refs"])
        self.assertEqual(
            ["Intro paragraph before table.", "Paragraph after table."],
            metadata["context_texts"],
        )
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks if chunk.metadata.get("direct_asset_refs") == ["table-1"]],
            metadata["direct_chunk_refs"],
        )
        self.assertEqual(
            [chunk.chunk_id for chunk in chunks if chunk.metadata.get("context_asset_refs") == ["table-1"]],
            metadata["context_chunk_refs"],
        )
        self.assertEqual(
            metadata["direct_chunk_refs"] + metadata["context_chunk_refs"],
            metadata["chunk_refs"],
        )

    def test_asset_context_enrichment_adds_local_rule_summary_metadata(self):
        asset = ParseAsset(
            asset_id="table-1",
            asset_type="table",
            text="| A | B |\n|---|---|\n| 1 | 2 |",
            page_numbers=[1],
            metadata={"row_count": 3, "column_count": 2},
        )
        blocks = [
            ParseBlock(
                block_id="b1",
                block_type="table",
                text=asset.text,
                page_numbers=[1],
                token_count=12,
                asset_refs=[asset.asset_id],
            )
        ]
        chunks = build_chunks(blocks, max_tokens=256, overlap_tokens=0, strategy="asset_aware")

        enriched_assets = enrich_asset_context([asset], blocks, chunks, window=1)
        metadata = enriched_assets[0].metadata

        self.assertEqual(ASSET_SUMMARY_SCHEMA_VERSION, metadata["asset_summary_schema_version"])
        self.assertEqual("local_rules", metadata["asset_summary_source"])
        self.assertEqual(
            {
                "asset_type": "table",
                "page_numbers": [1],
                "row_count": 3,
                "column_count": 2,
                "text_length": len(asset.text),
            },
            metadata["asset_summary_facts"],
        )
        self.assertIn("Table", metadata["asset_summary"])
        self.assertIn("3 rows", metadata["asset_summary"])
        self.assertIn("2 columns", metadata["asset_summary"])

    def test_chunk_export_assets_include_enriched_context_metadata(self):
        asset = ParseAsset(
            asset_id="table-1",
            asset_type="table",
            text="| A | B |",
            page_numbers=[1],
        )
        blocks = [
            ParseBlock(
                block_id="b1",
                block_type="text",
                text="Intro paragraph before table.",
                page_numbers=[1],
                token_count=5,
            ),
            ParseBlock(
                block_id="b2",
                block_type="table",
                text=asset.text,
                page_numbers=[1],
                token_count=5,
                asset_refs=[asset.asset_id],
            ),
        ]
        chunks = build_chunks(blocks, max_tokens=256, overlap_tokens=0, strategy="asset_aware")
        enriched_assets = enrich_asset_context([asset], blocks, chunks, window=1)
        artifact = ParseArtifact(
            document=_document(),
            markdown="Intro paragraph before table.\n\n| A | B |",
            assets=enriched_assets,
            blocks=blocks,
            chunks=chunks,
        )

        records = build_chunk_export_records(artifact)
        exported_asset_metadata = next(
            asset_view.metadata
            for record in records
            for asset_view in record.assets
            if asset_view.asset_id == "table-1"
        )

        self.assertEqual(["b2"], exported_asset_metadata["direct_block_refs"])
        self.assertEqual(["b1"], exported_asset_metadata["context_block_refs"])
        self.assertEqual(["Intro paragraph before table."], exported_asset_metadata["context_texts"])


if __name__ == "__main__":
    unittest.main()

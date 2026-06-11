import unittest
import os
from unittest.mock import patch

from PIL import Image


def _document():
    from common.parse_artifacts import ParseDocument

    return ParseDocument(
        document_id="doc-cross-page",
        parse_id="parse-cross-page",
        filename="cross-page.pdf",
        file_type="pdf",
        parser_engine="deepdoc",
        created_at="2026-06-08T00:00:00+00:00",
        source_sha256="abc123",
        source_size_bytes=123,
    )


class PdfCrossPageMergeTest(unittest.TestCase):
    def _new_parser(self):
        from deepdoc.parser.pdf_parser import DeepDocPdfParser

        parser = object.__new__(DeepDocPdfParser)
        parser.mean_height = [10, 10]
        parser.mean_width = [8, 8]
        parser.page_cum_height = [0, 100, 200]
        parser.page_images = [Image.new("RGB", (600, 100)), Image.new("RGB", (600, 100))]
        parser.page_from = 0
        parser.column_num = 1
        return parser

    def test_cross_page_text_continuation_merges_adjacent_paragraph_boxes(self):
        from deepdoc.parser.pdf_parser import DeepDocPdfParser

        parser = object.__new__(DeepDocPdfParser)
        parser.mean_height = [10, 10]
        parser.mean_width = [8, 8]
        parser.page_cum_height = [0, 100, 200]
        parser.page_images = [Image.new("RGB", (600, 100)), Image.new("RGB", (600, 100))]
        parser.page_from = 0
        parser.column_num = 1
        parser.boxes = [
            {
                "text": "This paragraph continues",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "text-1",
                "x0": 40,
                "x1": 420,
                "top": 92,
                "bottom": 98,
                "col_id": 0,
            },
            {
                "text": "on the next page without a hard break.",
                "page_number": 2,
                "layout_type": "text",
                "layoutno": "text-2",
                "x0": 42,
                "x1": 430,
                "top": 103,
                "bottom": 112,
                "col_id": 0,
            },
            {
                "text": "A separate sentence.",
                "page_number": 2,
                "layout_type": "text",
                "layoutno": "text-3",
                "x0": 42,
                "x1": 430,
                "top": 140,
                "bottom": 150,
                "col_id": 0,
            },
        ]

        parser._merge_cross_page_text()

        self.assertEqual(2, len(parser.boxes))
        self.assertEqual("This paragraph continues on the next page without a hard break.", parser.boxes[0]["text"])
        self.assertEqual([1, 2], parser.boxes[0]["merged_page_numbers"])
        self.assertEqual(1, parser.boxes[0]["page_number"])
        self.assertGreater(parser.boxes[0]["bottom"], 100)
        self.assertEqual("text_cross_page_continuation", parser.boxes[0]["merge_reason"])
        self.assertEqual("A separate sentence.", parser.boxes[1]["text"])

    def test_rules_strategy_rejects_cross_page_merge_when_next_page_starts_new_heading(self):
        parser = self._new_parser()
        parser.boxes = [
            {
                "text": "This paragraph is still open",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "text-1",
                "x0": 40,
                "x1": 420,
                "top": 92,
                "bottom": 98,
                "col_id": 0,
            },
            {
                "text": "Section Summary",
                "page_number": 2,
                "layout_type": "text",
                "semantic_type": "title",
                "layoutno": "text-2",
                "x0": 42,
                "x1": 430,
                "top": 103,
                "bottom": 112,
                "col_id": 0,
            },
        ]

        with patch.dict(os.environ, {"DEEPDOC_READING_ORDER_STRATEGY": "rules"}, clear=False):
            parser._merge_cross_page_text()

        self.assertEqual(2, len(parser.boxes))
        self.assertEqual("This paragraph is still open", parser.boxes[0]["text"])
        self.assertEqual("Section Summary", parser.boxes[1]["text"])

    def test_legacy_reading_order_keeps_repeating_page_headers(self):
        parser = self._new_parser()
        parser.boxes = [
            {
                "text": "Confidential Report",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "header-1",
                "x0": 40,
                "x1": 560,
                "top": 4,
                "bottom": 12,
                "col_id": 0,
            },
            {
                "text": "Body page one",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "body-1",
                "x0": 40,
                "x1": 260,
                "top": 40,
                "bottom": 52,
                "col_id": 0,
            },
            {
                "text": "Confidential Report",
                "page_number": 2,
                "layout_type": "text",
                "layoutno": "header-2",
                "x0": 40,
                "x1": 560,
                "top": 104,
                "bottom": 112,
                "col_id": 0,
            },
            {
                "text": "Body page two",
                "page_number": 2,
                "layout_type": "text",
                "layoutno": "body-2",
                "x0": 40,
                "x1": 260,
                "top": 140,
                "bottom": 152,
                "col_id": 0,
            },
        ]

        with patch.dict(os.environ, {"DEEPDOC_READING_ORDER_STRATEGY": "legacy"}, clear=False):
            parser._final_reading_order_merge()

        self.assertEqual(
            ["Confidential Report", "Body page one", "Confidential Report", "Body page two"],
            [box["text"] for box in parser.boxes],
        )

    def test_rules_reading_order_removes_repeating_headers_and_preserves_column_order(self):
        parser = self._new_parser()
        parser.boxes = [
            {
                "text": "Confidential Report",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "header-1",
                "x0": 40,
                "x1": 560,
                "top": 4,
                "bottom": 12,
                "col_id": 0,
            },
            {
                "text": "Report Title",
                "page_number": 1,
                "layout_type": "title",
                "layoutno": "title-1",
                "x0": 40,
                "x1": 560,
                "top": 20,
                "bottom": 32,
                "col_id": 0,
            },
            {
                "text": "Left column first",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "left-1",
                "x0": 40,
                "x1": 260,
                "top": 42,
                "bottom": 52,
                "col_id": 0,
            },
            {
                "text": "Right column first",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "right-1",
                "x0": 320,
                "x1": 560,
                "top": 38,
                "bottom": 50,
                "col_id": 1,
            },
            {
                "text": "Left column second",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "left-2",
                "x0": 40,
                "x1": 260,
                "top": 68,
                "bottom": 78,
                "col_id": 0,
            },
            {
                "text": "Confidential Report",
                "page_number": 2,
                "layout_type": "text",
                "layoutno": "header-2",
                "x0": 40,
                "x1": 560,
                "top": 104,
                "bottom": 112,
                "col_id": 0,
            },
            {
                "text": "Body page two",
                "page_number": 2,
                "layout_type": "text",
                "layoutno": "body-2",
                "x0": 40,
                "x1": 260,
                "top": 140,
                "bottom": 152,
                "col_id": 0,
            },
        ]

        with patch.dict(os.environ, {"DEEPDOC_READING_ORDER_STRATEGY": "rules"}, clear=False):
            parser._final_reading_order_merge()

        self.assertEqual(
            [
                "Report Title",
                "Left column first",
                "Left column second",
                "Right column first",
                "Body page two",
            ],
            [box["text"] for box in parser.boxes],
        )

    def test_rules_reading_order_binds_nearby_caption_to_asset(self):
        parser = self._new_parser()
        parser.boxes = [
            {
                "text": "Figure 1: quarterly workflow",
                "page_number": 1,
                "layout_type": "figure caption",
                "layoutno": "caption-1",
                "x0": 45,
                "x1": 300,
                "top": 36,
                "bottom": 46,
                "col_id": 0,
            },
            {
                "text": "",
                "page_number": 1,
                "layout_type": "figure",
                "layoutno": "figure-1",
                "x0": 40,
                "x1": 310,
                "top": 50,
                "bottom": 92,
                "col_id": 0,
            },
            {
                "text": "Narrative text",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "text-1",
                "x0": 330,
                "x1": 560,
                "top": 42,
                "bottom": 54,
                "col_id": 1,
            },
        ]

        with patch.dict(os.environ, {"DEEPDOC_READING_ORDER_STRATEGY": "rules"}, clear=False):
            parser._final_reading_order_merge()

        caption = next(box for box in parser.boxes if box["layoutno"] == "caption-1")
        self.assertEqual("figure-1", caption["bound_asset_layoutno"])
        self.assertEqual("figure", caption["bound_asset_type"])

    def test_rules_reading_order_runs_after_default_concat_sort_in_parse_call(self):
        from deepdoc.parser.pdf_parser import DeepDocPdfParser

        parser = self._new_parser()
        parser.boxes = [
            {
                "text": "Report Title",
                "page_number": 1,
                "layout_type": "title",
                "layoutno": "title-1",
                "x0": 40,
                "x1": 560,
                "top": 20,
                "bottom": 32,
                "col_id": 0,
            },
            {
                "text": "Left column first",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "left-1",
                "x0": 40,
                "x1": 260,
                "top": 42,
                "bottom": 52,
                "col_id": 0,
            },
            {
                "text": "Right column first",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "right-1",
                "x0": 320,
                "x1": 560,
                "top": 38,
                "bottom": 50,
                "col_id": 1,
            },
            {
                "text": "Left column second",
                "page_number": 1,
                "layout_type": "text",
                "layoutno": "left-2",
                "x0": 40,
                "x1": 260,
                "top": 68,
                "bottom": 78,
                "col_id": 0,
            },
        ]
        parser.__images__ = lambda _fnm, _zoomin: None
        parser._layouts_rec = lambda _zoomin: None
        parser._table_transformer_job = lambda _zoomin, auto_rotate=None: None
        parser._text_merge = lambda: None
        parser._merge_cross_page_text = lambda: None
        parser._filter_forpages = lambda: None
        parser._extract_table_figure = lambda *_args: []
        parser._DeepDocPdfParser__filterout_scraps = lambda boxes, _zoomin: boxes

        with patch.dict(os.environ, {"DEEPDOC_READING_ORDER_STRATEGY": "rules"}, clear=False):
            boxes, _tables = DeepDocPdfParser.__call__(parser, "sample.pdf")

        self.assertEqual(
            ["Report Title", "Left column first", "Left column second", "Right column first"],
            [box["text"] for box in boxes],
        )

    def test_cross_page_table_merge_marks_continuation_metadata(self):
        from deepdoc.parser.pdf_parser import DeepDocPdfParser

        parser = object.__new__(DeepDocPdfParser)
        parser.mean_height = [10, 10]
        parser.page_images = [Image.new("RGB", (600, 100)), Image.new("RGB", (600, 100))]
        parser.page_cum_height = [0, 100, 200]
        parser.page_layout = [
            [{"type": "table", "x0": 30, "x1": 500, "top": 70, "bottom": 98}],
            [{"type": "table", "x0": 32, "x1": 498, "top": 2, "bottom": 30}],
        ]
        parser.page_from = 0
        parser.is_english = False
        parser.table_engine = "tatr"
        parser.tbl_det = type(
            "FakeTableDetector",
            (),
            {"construct_table": lambda self, bxs, html, is_english: [["A", "B"], ["1", "2"]]},
        )()
        parser.boxes = [
            {
                "text": "A B",
                "page_number": 1,
                "layout_type": "table",
                "layoutno": "table-1",
                "x0": 35,
                "x1": 480,
                "top": 78,
                "bottom": 96,
            },
            {
                "text": "1 2",
                "page_number": 2,
                "layout_type": "table",
                "layoutno": "table-2",
                "x0": 36,
                "x1": 478,
                "top": 106,
                "bottom": 125,
            },
        ]

        tables_with_positions = parser._extract_table_figure(
            need_image=True,
            ZM=3,
            return_html=False,
            need_position=True,
            separate_tables_figures=False,
        )

        self.assertEqual(1, len(tables_with_positions))
        (_image, table_payload), positions = tables_with_positions[0]
        self.assertEqual([["A", "B"], ["1", "2"]], table_payload)
        self.assertEqual([1, 2], [position[0] + 1 for position in positions])
        self.assertEqual(2, parser._last_cross_page_table_merge_count)

    def test_cross_page_table_merge_rejects_incompatible_column_text(self):
        from deepdoc.parser.pdf_parser import DeepDocPdfParser

        parser = object.__new__(DeepDocPdfParser)
        parser.mean_height = [10, 10]
        parser.page_images = [Image.new("RGB", (600, 100)), Image.new("RGB", (600, 100))]
        parser.page_cum_height = [0, 100, 200]
        parser.page_layout = [
            [{"type": "table", "x0": 30, "x1": 500, "top": 70, "bottom": 98}],
            [{"type": "table", "x0": 32, "x1": 498, "top": 2, "bottom": 30}],
        ]
        parser.page_from = 0
        parser.is_english = False
        parser.table_engine = "tatr"
        parser.tbl_det = type(
            "FakeTableDetector",
            (),
            {"construct_table": lambda self, bxs, html, is_english: [[b["text"] for b in bxs]]},
        )()
        parser.boxes = [
            {
                "text": "项目 金额",
                "page_number": 1,
                "layout_type": "table",
                "layoutno": "table-1",
                "x0": 35,
                "x1": 480,
                "top": 78,
                "bottom": 96,
            },
            {
                "text": "姓名 电话 地址",
                "page_number": 2,
                "layout_type": "table",
                "layoutno": "table-2",
                "x0": 36,
                "x1": 478,
                "top": 106,
                "bottom": 125,
            },
        ]

        tables_with_positions = parser._extract_table_figure(
            need_image=True,
            ZM=3,
            return_html=False,
            need_position=True,
            separate_tables_figures=False,
        )

        self.assertEqual(2, len(tables_with_positions))
        self.assertEqual(0, parser._last_cross_page_table_merge_count)
        for (_image, _table_payload), positions in tables_with_positions:
            self.assertEqual(1, len(positions))

    def test_deepdoc_artifact_preserves_cross_page_merge_metadata(self):
        from common.parse_builders import build_deepdoc_artifact

        class FakeParser:
            page_images = [Image.new("RGB", (600, 100)), Image.new("RGB", (600, 100))]

            def get_position(self, box, _zoomin):
                return box["positions_for_test"]

        artifact = build_deepdoc_artifact(
            document=_document(),
            markdown="merged paragraph\n\n| A |\n|---|\n| 1 |",
            parser=FakeParser(),
            boxes=[
                {
                    "text": "merged paragraph",
                    "page_number": 1,
                    "layout_type": "text",
                    "positions_for_test": [
                        (1, 40, 420, 92, 100),
                        (2, 40, 420, 0, 12),
                    ],
                    "merged_page_numbers": [1, 2],
                    "merge_reason": "text_cross_page_continuation",
                    "source_box_count": 2,
                    "source_layoutnos": ["text-1", "text-2"],
                }
            ],
            tables_with_positions=[
                (
                    (Image.new("RGB", (120, 80)), [["A"], ["1"]]),
                    [
                        (0, 35, 480, 78, 96),
                        (1, 36, 478, 6, 25),
                    ],
                )
            ],
            figures_with_positions=[],
            zoomin=3,
            chunk_strategy="asset_aware",
        )

        text_block = next(block for block in artifact.blocks if block.text == "merged paragraph")
        self.assertEqual([1, 2], text_block.metadata["merged_page_numbers"])
        self.assertEqual("text_cross_page_continuation", text_block.metadata["merge_reason"])
        self.assertEqual(2, text_block.metadata["source_box_count"])

        table_asset = next(asset for asset in artifact.assets if asset.asset_type == "table")
        self.assertTrue(table_asset.metadata["cross_page"])
        self.assertEqual([1, 2], table_asset.metadata["merged_page_numbers"])
        self.assertEqual("table_cross_page_continuation", table_asset.metadata["merge_reason"])

        table_block = next(block for block in artifact.blocks if block.block_type == "table")
        self.assertTrue(table_block.metadata["cross_page"])
        self.assertEqual([1, 2], table_block.metadata["merged_page_numbers"])


if __name__ == "__main__":
    unittest.main()

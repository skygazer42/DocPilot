from __future__ import annotations

import re
import zipfile
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from common.markdown_utils import clean_text, table_to_md


ODT_NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
}


def _read_bytes(fnm, binary=None) -> bytes:
    if binary is not None:
        if isinstance(binary, bytes):
            return binary
        if isinstance(binary, bytearray):
            return bytes(binary)
        if isinstance(binary, memoryview):
            return binary.tobytes()
    if isinstance(fnm, (bytes, bytearray, memoryview)):
        return bytes(fnm)
    if isinstance(fnm, (str, PathLike, Path)):
        return Path(fnm).read_bytes()
    data = fnm.read()
    return data if isinstance(data, bytes) else str(data or "").encode("utf-8")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text or "")).strip()


def _tag_name(element: ET.Element) -> str:
    tag = str(element.tag or "")
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag


def _text_content(element: ET.Element) -> str:
    return _normalize_ws(" ".join(part for part in element.itertext() if part))


class DeepDocOdtParser:
    def __call__(self, fnm, binary=None, chunk_token_num=512):
        payload = _read_bytes(fnm, binary)
        source_name = Path(str(fnm)).name if fnm is not None and not isinstance(fnm, bytes) else "document.odt"
        return self.parser_bytes(payload, source_name=source_name, chunk_token_num=chunk_token_num)

    @classmethod
    def parser_bytes(cls, payload: bytes, *, source_name: str = "document.odt", chunk_token_num: int = 512):
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            if "content.xml" not in archive.namelist():
                raise ValueError("Invalid ODT: missing content.xml")
            root = ET.fromstring(archive.read("content.xml"))
        body = root.find(".//office:body/office:text", ODT_NS)
        if body is None:
            body = root
        blocks = cls._extract_blocks(body, source_name=source_name)
        markdown = cls._blocks_to_markdown(blocks)
        structured_source = {
            "engine": "odt",
            "metadata": {
                "source_name": source_name,
                "block_count": len(blocks),
            },
            "blocks": blocks,
            "block_count": len(blocks),
        }
        return markdown, [], {"structured_source": structured_source, "page_count": 1, "total_page_count": 1}

    @classmethod
    def _extract_blocks(cls, body: ET.Element, *, source_name: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for element in list(body):
            tag_name = _tag_name(element)
            if tag_name == "h":
                text = _text_content(element)
                if not text:
                    continue
                level = int(element.attrib.get(f"{{{ODT_NS['text']}}}outline-level") or 1)
                blocks.append(
                    {
                        "block_type": "title",
                        "text": text,
                        "heading_level": max(1, min(6, level)),
                        "source_name": source_name,
                    }
                )
            elif tag_name == "p":
                text = _text_content(element)
                if text:
                    blocks.append({"block_type": "text", "text": text, "source_name": source_name})
            elif tag_name == "list":
                items = cls._list_items(element)
                if items:
                    blocks.append(
                        {
                            "block_type": "list",
                            "text": "\n".join(f"- {item}" for item in items),
                            "source_name": source_name,
                        }
                    )
            elif tag_name == "table":
                rows = cls._table_rows(element)
                table_text = table_to_md(rows).strip()
                if table_text:
                    blocks.append(
                        {
                            "block_type": "table",
                            "text": table_text,
                            "source_name": source_name,
                            "row_count": len(rows),
                            "column_count": max((len(row) for row in rows), default=0),
                        }
                    )
        return blocks

    @staticmethod
    def _list_items(list_element: ET.Element) -> list[str]:
        items: list[str] = []
        for item in list_element.findall("text:list-item", ODT_NS):
            text = _text_content(item)
            if text:
                items.append(text)
        return items

    @staticmethod
    def _table_rows(table_element: ET.Element) -> list[list[str]]:
        rows: list[list[str]] = []
        for row_element in table_element.findall("table:table-row", ODT_NS):
            row: list[str] = []
            for cell in row_element.findall("table:table-cell", ODT_NS):
                repeat = int(cell.attrib.get(f"{{{ODT_NS['table']}}}number-columns-repeated") or 1)
                text = _text_content(cell)
                row.extend([text] * max(1, repeat))
            if any(cell.strip() for cell in row):
                rows.append(row)
        if not rows:
            return []
        max_cols = max(len(row) for row in rows)
        return [row + [""] * (max_cols - len(row)) for row in rows]

    @staticmethod
    def _blocks_to_markdown(blocks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for block in blocks:
            text = str(block.get("text") or "").strip()
            if not text:
                continue
            if block.get("block_type") == "title":
                level = max(1, min(6, int(block.get("heading_level") or 1)))
                parts.append(f"{'#' * level} {text}")
            else:
                parts.append(text)
        return "\n\n".join(parts).strip()

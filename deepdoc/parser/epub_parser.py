from __future__ import annotations

import posixpath
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

from common.markdown_utils import clean_text, table_to_md


CONTAINER_PATH = "META-INF/container.xml"
CONTAINER_NS = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
OPF_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def _zip_path_join(base: str, href: str) -> str:
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(base), href))
    if joined.startswith("../") or joined == "..":
        raise ValueError(f"Unsafe EPUB href: {href}")
    return joined


def _xml_text(root: ET.Element, path: str, namespaces: dict[str, str]) -> str | None:
    item = root.find(path, namespaces)
    if item is None or item.text is None:
        return None
    value = item.text.strip()
    return value or None


def _heading_level(tag_name: str) -> int:
    match = re.fullmatch(r"h([1-6])", tag_name.lower())
    return int(match.group(1)) if match else 1


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text or "")).strip()


class DeepDocEpubParser:
    def __call__(self, fnm, binary=None, chunk_token_num=512):
        if binary is None:
            with open(fnm, "rb") as file:
                binary = file.read()
        source_name = Path(str(fnm)).name if fnm is not None else "document.epub"
        return self.parser_bytes(binary, source_name=source_name, chunk_token_num=chunk_token_num)

    @classmethod
    def parser_bytes(cls, payload: bytes, *, source_name: str = "document.epub", chunk_token_num: int = 512):
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            opf_path = cls._find_opf_path(archive)
            opf_root = ET.fromstring(archive.read(opf_path))
            metadata = cls._extract_metadata(opf_root)
            spine_items = cls._extract_spine_items(opf_root, opf_path)
            blocks: list[dict[str, Any]] = []
            markdown_parts: list[str] = []
            chapters: list[dict[str, Any]] = []
            for chapter_index, item in enumerate(spine_items):
                href = item["href"]
                html_payload = archive.read(href).decode("utf-8", errors="ignore")
                chapter_blocks = cls._extract_chapter_blocks(
                    html_payload,
                    chapter_index=chapter_index,
                    href=href,
                )
                if not chapter_blocks:
                    continue
                blocks.extend(chapter_blocks)
                chapters.append(
                    {
                        "index": chapter_index,
                        "href": href,
                        "block_count": len(chapter_blocks),
                        "title": cls._chapter_title(chapter_blocks),
                    }
                )
                markdown_parts.extend(cls._blocks_to_markdown(chapter_blocks))

        metadata["source_name"] = source_name
        metadata["chapter_count"] = len(chapters)
        structured_source = {
            "engine": "epub",
            "metadata": metadata,
            "chapters": chapters,
            "chapter_count": len(chapters),
            "blocks": blocks,
        }
        markdown = "\n\n".join(part for part in markdown_parts if part.strip()).strip()
        return markdown, [], {"structured_source": structured_source, "page_count": len(chapters), "total_page_count": len(chapters)}

    @staticmethod
    def _find_opf_path(archive: zipfile.ZipFile) -> str:
        if CONTAINER_PATH not in archive.namelist():
            raise ValueError("Invalid EPUB: missing META-INF/container.xml")
        root = ET.fromstring(archive.read(CONTAINER_PATH))
        rootfile = root.find(".//container:rootfile", CONTAINER_NS)
        if rootfile is None:
            raise ValueError("Invalid EPUB: missing rootfile entry")
        opf_path = (rootfile.attrib.get("full-path") or "").strip()
        if not opf_path:
            raise ValueError("Invalid EPUB: empty OPF path")
        if opf_path not in archive.namelist():
            raise ValueError(f"Invalid EPUB: OPF path not found: {opf_path}")
        return opf_path

    @staticmethod
    def _extract_metadata(opf_root: ET.Element) -> dict[str, Any]:
        metadata = {
            "title": _xml_text(opf_root, ".//dc:title", OPF_NS),
            "creator": _xml_text(opf_root, ".//dc:creator", OPF_NS),
            "language": _xml_text(opf_root, ".//dc:language", OPF_NS),
        }
        return {key: value for key, value in metadata.items() if value}

    @staticmethod
    def _extract_spine_items(opf_root: ET.Element, opf_path: str) -> list[dict[str, str]]:
        manifest: dict[str, dict[str, str]] = {}
        for item in opf_root.findall(".//opf:manifest/opf:item", OPF_NS):
            item_id = (item.attrib.get("id") or "").strip()
            href = (item.attrib.get("href") or "").strip()
            media_type = (item.attrib.get("media-type") or "").strip()
            if not item_id or not href:
                continue
            manifest[item_id] = {
                "href": _zip_path_join(opf_path, href),
                "media_type": media_type,
            }
        spine_items: list[dict[str, str]] = []
        for itemref in opf_root.findall(".//opf:spine/opf:itemref", OPF_NS):
            idref = (itemref.attrib.get("idref") or "").strip()
            manifest_item = manifest.get(idref)
            if not manifest_item:
                continue
            media_type = manifest_item.get("media_type") or ""
            if media_type and media_type not in {"application/xhtml+xml", "text/html"}:
                continue
            spine_items.append({"idref": idref, "href": manifest_item["href"], "media_type": media_type})
        return spine_items

    @classmethod
    def _extract_chapter_blocks(cls, html_payload: str, *, chapter_index: int, href: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html_payload, "html.parser")
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()
        body = soup.body or soup
        blocks: list[dict[str, Any]] = []
        for element in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table"], recursive=True):
            if cls._has_block_ancestor(element):
                continue
            tag_name = element.name.lower()
            if tag_name.startswith("h"):
                text = _normalize_ws(element.get_text(" ", strip=True))
                if text:
                    blocks.append(
                        {
                            "block_type": "title",
                            "text": text,
                            "chapter_index": chapter_index,
                            "href": href,
                            "heading_level": _heading_level(tag_name),
                        }
                    )
            elif tag_name == "p":
                text = _normalize_ws(element.get_text(" ", strip=True))
                if text:
                    blocks.append({"block_type": "text", "text": text, "chapter_index": chapter_index, "href": href})
            elif tag_name in {"ul", "ol"}:
                items = []
                for item in element.find_all("li", recursive=False):
                    text = _normalize_ws(item.get_text(" ", strip=True))
                    if not text:
                        continue
                    prefix = "-" if tag_name == "ul" else f"{len(items) + 1}."
                    items.append(f"{prefix} {text}")
                if items:
                    blocks.append(
                        {
                            "block_type": "list",
                            "text": "\n".join(items),
                            "chapter_index": chapter_index,
                            "href": href,
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
                            "chapter_index": chapter_index,
                            "href": href,
                            "row_count": len(rows),
                            "column_count": max((len(row) for row in rows), default=0),
                        }
                    )
        return blocks

    @staticmethod
    def _has_block_ancestor(element) -> bool:
        block_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "li", "table"}
        parent = element.parent
        while parent is not None and getattr(parent, "name", None):
            if parent.name.lower() in block_tags:
                return True
            parent = parent.parent
        return False

    @staticmethod
    def _table_rows(table) -> list[list[str]]:
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [_normalize_ws(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        return rows

    @staticmethod
    def _chapter_title(blocks: list[dict[str, Any]]) -> str | None:
        for block in blocks:
            if block.get("block_type") == "title":
                return str(block.get("text") or "").strip() or None
        return None

    @staticmethod
    def _blocks_to_markdown(blocks: list[dict[str, Any]]) -> list[str]:
        parts: list[str] = []
        for block in blocks:
            block_type = block.get("block_type")
            text = str(block.get("text") or "").strip()
            if not text:
                continue
            if block_type == "title":
                level = max(1, min(6, int(block.get("heading_level") or 1)))
                parts.append(f"{'#' * level} {text}")
            else:
                parts.append(text)
        return parts

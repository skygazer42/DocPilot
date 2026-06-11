from __future__ import annotations

import re
from os import PathLike
from pathlib import Path
from typing import Any

from common.markdown_utils import clean_text
from common.nlp import find_codec


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text or "")).strip()


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


def _plain_text_from_rtf(text: str) -> str:
    try:
        from striprtf.striprtf import rtf_to_text

        return rtf_to_text(text)
    except Exception:
        stripped = re.sub(r"\\par[d]?", "\n", text)
        stripped = re.sub(r"\\'[0-9a-fA-F]{2}", " ", stripped)
        stripped = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", stripped)
        stripped = stripped.replace("{", "").replace("}", "")
        return stripped


def _line_blocks(lines: list[str], *, source_name: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        text = _normalize_ws(line)
        if not text:
            continue
        block_type = "text"
        heading_level = None
        if index == 0 and len(lines) > 1 and len(text) <= 120:
            block_type = "title"
            heading_level = 1
        block: dict[str, Any] = {
            "block_type": block_type,
            "text": text,
            "source_name": source_name,
        }
        if heading_level is not None:
            block["heading_level"] = heading_level
        blocks.append(block)
    return blocks


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


class DeepDocRtfParser:
    def __call__(self, fnm, binary=None, chunk_token_num=512):
        payload = _read_bytes(fnm, binary)
        encoding = find_codec(payload)
        rtf_text = payload.decode(encoding, errors="ignore")
        plain_text = _plain_text_from_rtf(rtf_text)
        source_name = Path(str(fnm)).name if fnm is not None and not isinstance(fnm, bytes) else "document.rtf"
        return self.parser_text(plain_text, source_name=source_name, chunk_token_num=chunk_token_num)

    @classmethod
    def parser_text(cls, text: str, *, source_name: str = "document.rtf", chunk_token_num: int = 512):
        lines = [line.strip() for line in re.split(r"[\r\n]+", text or "") if line.strip()]
        blocks = _line_blocks(lines, source_name=source_name)
        markdown = _blocks_to_markdown(blocks)
        structured_source = {
            "engine": "rtf",
            "metadata": {
                "source_name": source_name,
                "block_count": len(blocks),
            },
            "blocks": blocks,
            "block_count": len(blocks),
        }
        return markdown, [], {"structured_source": structured_source, "page_count": 1, "total_page_count": 1}

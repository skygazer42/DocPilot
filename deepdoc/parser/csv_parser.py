from __future__ import annotations

import csv
from io import StringIO
from os import PathLike
from pathlib import Path

from common.markdown_utils import table_to_md
from common.nlp import find_codec


class DeepDocCsvParser:
    def __call__(self, fnm, binary=None):
        payload = self._read_bytes(fnm, binary)
        encoding = find_codec(payload)
        text = payload.decode(encoding, errors="ignore")
        rows, delimiter = self.parse_rows(text, filename=str(fnm or ""))
        markdown = table_to_md(rows)
        return (
            markdown,
            [],
            {
                "structured_source": {
                    "engine": "csv",
                    "rows": rows,
                    "delimiter": delimiter,
                    "row_count": len(rows),
                    "column_count": max((len(row) for row in rows), default=0),
                }
            },
        )

    @staticmethod
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

    @classmethod
    def parse_rows(cls, text: str, *, filename: str = "") -> tuple[list[list[str]], str]:
        normalized = (text or "").lstrip("\ufeff")
        delimiter = cls._detect_delimiter(normalized, filename=filename)
        reader = csv.reader(StringIO(normalized), delimiter=delimiter)
        rows = [
            [str(cell or "").strip() for cell in row]
            for row in reader
            if any(str(cell or "").strip() for cell in row)
        ]
        if not rows:
            return [], delimiter
        max_cols = max(len(row) for row in rows)
        rows = [row + [""] * (max_cols - len(row)) for row in rows]
        return rows, delimiter

    @staticmethod
    def _detect_delimiter(text: str, *, filename: str = "") -> str:
        if str(filename or "").lower().endswith(".tsv"):
            return "\t"
        sample = "\n".join((text or "").splitlines()[:20])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delimiter = dialect.delimiter
            if delimiter in {",", "\t", ";", "|"}:
                return delimiter
        except csv.Error:
            pass
        if "\t" in sample and sample.count("\t") >= sample.count(","):
            return "\t"
        return ","

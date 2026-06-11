from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any

from common.markdown_utils import clean_text, strip_html_tags
from common.nlp import find_codec


OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


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
    return re.sub(r"[ \t\r\f\v]+", " ", clean_text(text or "")).strip()


def _decode_payload(payload: bytes, charset: str | None = None) -> str:
    if not payload:
        return ""
    candidates = [charset] if charset else []
    candidates.extend([find_codec(payload), "utf-8", "gb18030", "latin1"])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return payload.decode(candidate, errors="replace")
        except Exception:
            continue
    return payload.decode("utf-8", errors="replace")


def _decode_msg_string(payload: bytes) -> str:
    if not payload:
        return ""
    if len(payload) >= 2 and payload[1::2].count(0) >= max(1, len(payload) // 4):
        return payload.decode("utf-16le", errors="ignore").rstrip("\x00")
    return _decode_payload(payload).rstrip("\x00")


def _header_value(value: Any) -> str | None:
    text = _normalize_ws(str(value or ""))
    return text or None


def _address_list(values: list[str | None]) -> list[str]:
    addresses = []
    for name, email_address in getaddresses([value for value in values if value]):
        label = _normalize_ws(name)
        addr = _normalize_ws(email_address)
        if label and addr:
            addresses.append(f"{label} <{addr}>")
        elif addr:
            addresses.append(addr)
        elif label:
            addresses.append(label)
    return addresses


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


class DeepDocEmailParser:
    def __call__(self, fnm, binary=None, chunk_token_num=512):
        payload = _read_bytes(fnm, binary)
        source_name = Path(str(fnm)).name if fnm is not None and not isinstance(fnm, bytes) else "message.eml"
        source_type = Path(source_name).suffix.lower().lstrip(".") or "eml"
        if source_type not in {"eml", "msg"}:
            source_type = "eml"
        return self.parser_bytes(payload, source_name=source_name, source_type=source_type, chunk_token_num=chunk_token_num)

    @classmethod
    def parser_bytes(
        cls,
        payload: bytes,
        *,
        source_name: str = "message.eml",
        source_type: str = "eml",
        chunk_token_num: int = 512,
    ):
        normalized_source_type = (source_type or "eml").strip().lower()
        if normalized_source_type == "msg":
            parsed = cls._parse_msg(payload)
            if parsed is None:
                parsed = cls._parse_eml(payload)
        else:
            parsed = cls._parse_eml(payload)

        subject = parsed.get("subject") or Path(source_name).stem or "Email Message"
        body = _normalize_ws(parsed.get("body") or "")
        attachments = parsed.get("attachments") if isinstance(parsed.get("attachments"), list) else []
        parseable_attachment_count = sum(1 for attachment in attachments if str(attachment.get("parsed_text") or "").strip())
        metadata = {
            "source_name": source_name,
            "source_type": normalized_source_type,
            "subject": subject,
            "from": parsed.get("from"),
            "to": parsed.get("to") or [],
            "cc": parsed.get("cc") or [],
            "date": parsed.get("date"),
            "body_type": parsed.get("body_type") or "text/plain",
            "attachment_count": len(attachments),
            "parseable_attachment_count": parseable_attachment_count,
        }
        blocks = cls._build_blocks(subject=subject, body=body, attachments=attachments, source_name=source_name)
        markdown = _blocks_to_markdown(blocks)
        structured_source = {
            "engine": "email",
            "metadata": metadata,
            "attachments": attachments,
            "blocks": blocks,
            "block_count": len(blocks),
        }
        return markdown, [], {"structured_source": structured_source, "page_count": 1, "total_page_count": 1}

    @staticmethod
    def _build_blocks(
        *,
        subject: str,
        body: str,
        attachments: list[dict[str, Any]],
        source_name: str,
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if subject:
            blocks.append(
                {
                    "block_type": "title",
                    "text": subject,
                    "heading_level": 1,
                    "source_name": source_name,
                }
            )
        if body:
            blocks.append({"block_type": "text", "text": body, "source_name": source_name})
        if attachments:
            lines = []
            for attachment in attachments:
                filename = attachment.get("filename") or "attachment"
                content_type = attachment.get("content_type") or "application/octet-stream"
                size = int(attachment.get("size_bytes") or 0)
                lines.append(f"- {filename} ({content_type}, {size} bytes)")
            blocks.append({"block_type": "list", "text": "\n".join(lines), "source_name": source_name})
            for attachment in attachments:
                parsed_text = _normalize_ws(str(attachment.get("parsed_text") or ""))
                if not parsed_text:
                    continue
                blocks.append(
                    {
                        "block_type": "text",
                        "text": parsed_text,
                        "source_name": source_name,
                        "metadata": {
                            "email_block_role": "attachment_text",
                            "attachment_filename": attachment.get("filename") or "attachment",
                            "attachment_content_type": attachment.get("content_type") or "application/octet-stream",
                        },
                    }
                )
        return blocks

    @staticmethod
    def _parse_eml(payload: bytes) -> dict[str, Any]:
        message = BytesParser(policy=policy.default).parsebytes(payload)
        text_parts: list[str] = []
        html_parts: list[str] = []
        attachments: list[dict[str, Any]] = []

        if message.is_multipart():
            parts = message.walk()
        else:
            parts = [message]
        for part in parts:
            if part.is_multipart():
                continue
            disposition = (part.get_content_disposition() or "").lower()
            filename = part.get_filename()
            content_type = (part.get_content_type() or "application/octet-stream").lower()
            raw_payload = part.get_payload(decode=True) or b""
            if disposition == "attachment" or filename:
                attachments.append(
                    {
                        "filename": filename or "attachment",
                        "content_type": content_type,
                        "size_bytes": len(raw_payload),
                        "parsed_text": DeepDocEmailParser._parse_attachment_text(
                            raw_payload,
                            filename=filename or "attachment",
                            content_type=content_type,
                            charset=part.get_content_charset(),
                        ),
                    }
                )
                continue
            if content_type == "text/plain":
                text_parts.append(_decode_payload(raw_payload, part.get_content_charset()))
            elif content_type == "text/html":
                html_parts.append(strip_html_tags(_decode_payload(raw_payload, part.get_content_charset())))

        body_type = "text/plain"
        body = "\n\n".join(part.strip() for part in text_parts if part.strip())
        if not body:
            body_type = "text/html"
            body = "\n\n".join(part.strip() for part in html_parts if part.strip())

        return {
            "subject": _header_value(message.get("Subject")),
            "from": _header_value(message.get("From")),
            "to": _address_list([message.get("To")]),
            "cc": _address_list([message.get("Cc")]),
            "date": _header_value(message.get("Date")),
            "body": body,
            "body_type": body_type,
            "attachments": attachments,
        }

    @classmethod
    def _parse_msg(cls, payload: bytes) -> dict[str, Any] | None:
        parsed = cls._parse_msg_with_extract_msg(payload)
        if parsed is not None:
            return parsed
        if payload.startswith(OLE_MAGIC):
            parsed = cls._parse_msg_with_olefile(payload)
            if parsed is not None:
                return parsed
        return cls._parse_msg_text_fallback(payload)

    @staticmethod
    def _parse_attachment_text(
        payload: bytes,
        *,
        filename: str,
        content_type: str,
        charset: str | None = None,
    ) -> str:
        suffix = Path(filename or "").suffix.lower().lstrip(".")
        normalized_content_type = (content_type or "").lower()
        if suffix in {"txt", "text", "md", "markdown", "csv", "tsv", "json", "xml"} or normalized_content_type.startswith("text/"):
            return _normalize_ws(_decode_payload(payload, charset))
        if suffix in {"html", "htm"} or normalized_content_type == "text/html":
            return _normalize_ws(strip_html_tags(_decode_payload(payload, charset)))
        if suffix == "eml" or normalized_content_type == "message/rfc822":
            parsed = DeepDocEmailParser._parse_eml(payload)
            subject = parsed.get("subject") or Path(filename).stem
            body = parsed.get("body") or ""
            return _normalize_ws("\n\n".join(part for part in (subject, body) if part))
        return ""

    @staticmethod
    def _parse_msg_with_extract_msg(payload: bytes) -> dict[str, Any] | None:
        try:
            import extract_msg
        except Exception:
            return None
        try:
            message = extract_msg.Message(BytesIO(payload))
            attachments = []
            for attachment in getattr(message, "attachments", []) or []:
                data = getattr(attachment, "data", None) or b""
                filename = getattr(attachment, "longFilename", None) or getattr(attachment, "shortFilename", None) or "attachment"
                attachments.append(
                    {
                        "filename": filename,
                        "content_type": getattr(attachment, "mimetype", None) or "application/octet-stream",
                        "size_bytes": len(data),
                    }
                )
            return {
                "subject": _header_value(getattr(message, "subject", None)),
                "from": _header_value(getattr(message, "sender", None)),
                "to": _address_list([getattr(message, "to", None)]),
                "cc": _address_list([getattr(message, "cc", None)]),
                "date": _header_value(getattr(message, "date", None)),
                "body": getattr(message, "body", None) or strip_html_tags(getattr(message, "htmlBody", None) or ""),
                "body_type": "text/plain",
                "attachments": attachments,
            }
        except Exception:
            return None

    @staticmethod
    def _parse_msg_with_olefile(payload: bytes) -> dict[str, Any] | None:
        try:
            import olefile
        except Exception:
            return None
        try:
            ole = olefile.OleFileIO(BytesIO(payload))
        except Exception:
            return None

        def read_stream(*names: str) -> str | None:
            for name in names:
                path = [name]
                if ole.exists(path):
                    try:
                        return _normalize_ws(_decode_msg_string(ole.openstream(path).read()))
                    except Exception:
                        continue
            return None

        try:
            subject = read_stream("__substg1.0_0037001F", "__substg1.0_0037001E")
            sender = read_stream("__substg1.0_0C1A001F", "__substg1.0_0C1A001E")
            recipients = read_stream("__substg1.0_0E04001F", "__substg1.0_0E04001E")
            body = read_stream("__substg1.0_1000001F", "__substg1.0_1000001E")
            html = read_stream("__substg1.0_1013001F", "__substg1.0_1013001E")
            attachments = DeepDocEmailParser._ole_attachments(ole)
            return {
                "subject": subject,
                "from": sender,
                "to": _address_list([recipients]),
                "cc": [],
                "date": None,
                "body": body or strip_html_tags(html or ""),
                "body_type": "text/plain" if body else "text/html",
                "attachments": attachments,
            }
        finally:
            ole.close()

    @staticmethod
    def _ole_attachments(ole) -> list[dict[str, Any]]:
        attachments: list[dict[str, Any]] = []
        storages = sorted({parts[0] for parts in ole.listdir() if parts and parts[0].startswith("__attach")})
        for storage in storages:
            def read_attachment_stream(*names: str) -> bytes:
                for name in names:
                    path = [storage, name]
                    if ole.exists(path):
                        try:
                            return ole.openstream(path).read()
                        except Exception:
                            continue
                return b""

            filename = _normalize_ws(
                _decode_msg_string(
                    read_attachment_stream("__substg1.0_3707001F", "__substg1.0_3704001F", "__substg1.0_3704001E")
                )
            )
            mime_type = _normalize_ws(_decode_msg_string(read_attachment_stream("__substg1.0_370E001F")))
            data = read_attachment_stream("__substg1.0_37010102")
            attachments.append(
                {
                    "filename": filename or "attachment",
                    "content_type": mime_type or "application/octet-stream",
                    "size_bytes": len(data),
                }
            )
        return attachments

    @staticmethod
    def _parse_msg_text_fallback(payload: bytes) -> dict[str, Any]:
        text = _decode_payload(payload)
        return DeepDocEmailParser._parse_eml(text.encode("utf-8", errors="ignore"))

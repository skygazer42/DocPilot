from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from PIL import Image

from common import logger
from common.markdown_utils import clean_text, table_to_md
from common.parse_artifacts import (
    ArtifactStore,
    DEFAULT_CHUNK_MAX_TOKENS,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_STRATEGY,
    ParseArtifact,
    ParseAsset,
    ParseBlock,
    ParseDocument,
    ParsePosition,
    build_remote_storage,
    build_asset_id,
    build_chunks,
    count_tokens,
    enrich_asset_context,
    pil_image_to_bytes,
)


def _positions_from_tuples(positions: list[tuple[int, float, float, float, float]]) -> list[ParsePosition]:
    return [
        ParsePosition(page=int(page) + 1, left=float(left), right=float(right), top=float(top), bottom=float(bottom))
        for page, left, right, top, bottom in positions
    ]


def _positions_from_box(parser, box: dict[str, Any], zoomin: int) -> list[ParsePosition]:
    try:
        positions = parser.get_position(box, zoomin)
    except Exception:
        logger.exception("Failed to build positions from box")
        return []
    return [
        ParsePosition(page=int(page), left=float(left), right=float(right), top=float(top), bottom=float(bottom))
        for page, left, right, top, bottom in positions
    ]


def _crop_positions_image(parser, positions: list[ParsePosition]) -> Image.Image | None:
    if parser is None or not getattr(parser, "page_images", None) or not positions:
        return None
    first = positions[0]
    page_idx = first.page - 1
    if not (0 <= page_idx < len(parser.page_images)):
        return None
    try:
        page_img = parser.page_images[page_idx]
        left = max(0, int(first.left))
        top = max(0, int(first.top))
        right = max(left + 2, int(first.right))
        bottom = max(top + 2, int(first.bottom))
        return page_img.crop((left, top, right, bottom))
    except Exception:
        logger.exception("Failed to crop fallback image from parser page_images")
        return None


def _normalize_block_type(box: dict[str, Any]) -> str:
    semantic_type = str(box.get("semantic_type") or "").strip().lower()
    if semantic_type == "seal":
        return "seal"
    layout_type = str(box.get("layout_type") or "").strip().lower()
    if layout_type in {"figure", "table", "equation", "seal", "title", "reference"}:
        return layout_type
    if layout_type in {"item", "list"}:
        return "list"
    if layout_type:
        return "text"
    return "unknown"


def _join_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def _cross_page_box_metadata(box: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "merged_page_numbers",
        "merge_reason",
        "source_box_count",
        "source_layoutnos",
        "cross_page_table_group",
    ):
        value = box.get(key)
        if value not in (None, "", []):
            metadata[key] = value
    return metadata


def _cross_page_positions_metadata(positions: list[ParsePosition]) -> dict[str, Any]:
    page_numbers = sorted(set(position.page for position in positions))
    if len(page_numbers) <= 1:
        return {}
    return {
        "cross_page": True,
        "merged_page_numbers": page_numbers,
        "merge_reason": "table_cross_page_continuation",
    }


def _build_visual_asset(
    *,
    store: ArtifactStore | None,
    artifact_paths,
    image: Image.Image,
    asset_type: str,
    text: str,
    positions: list[ParsePosition],
    metadata: dict[str, Any] | None = None,
) -> ParseAsset:
    img_bytes = pil_image_to_bytes(image)
    asset_id = build_asset_id(asset_type, img_bytes, [position.page for position in positions])
    storage = None
    if store and artifact_paths:
        storage = store.save_image_asset(paths=artifact_paths, image=image, asset_id=asset_id)
    return ParseAsset(
        asset_id=asset_id,
        asset_type=asset_type,
        title=text.splitlines()[0][:200] if text else None,
        text=text,
        page_numbers=sorted(set(position.page for position in positions)),
        positions=positions,
        width=image.size[0],
        height=image.size[1],
        sha256=hashlib.sha256(img_bytes).hexdigest(),
        storage=storage,
        metadata=metadata or {},
    )


def _build_file_or_url_asset(
    *,
    store: ArtifactStore | None,
    artifact_paths,
    asset_type: str,
    text: str,
    positions: list[ParsePosition],
    source_path: str | None = None,
    source_url: str | None = None,
    metadata: dict[str, Any] | None = None,
    media_type: str = "image/png",
) -> ParseAsset | None:
    source_identity = source_url or source_path or f"{asset_type}:{text}"
    asset_id = build_asset_id(
        asset_type,
        source_identity.encode("utf-8", errors="ignore"),
        [position.page for position in positions],
    )
    storage = None
    width = None
    height = None
    sha256 = None

    try:
        if source_path and Path(source_path).exists():
            if store and artifact_paths:
                storage = store.copy_file_asset(
                    paths=artifact_paths,
                    source_path=source_path,
                    asset_id=asset_id,
                    media_type=media_type,
                )
            with Image.open(source_path) as img:
                width, height = img.size
            sha256 = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        elif source_url:
            if store and artifact_paths:
                storage = store.download_asset(
                    paths=artifact_paths,
                    source_url=source_url,
                    asset_id=asset_id,
                    media_type=media_type,
                )
            else:
                storage = build_remote_storage(source_url=source_url, media_type=media_type)
        else:
            return None
    except Exception:
        logger.exception("Failed to materialize asset type=%s source=%s", asset_type, source_identity)
        if source_url and storage is None:
            storage = build_remote_storage(source_url=source_url, media_type=media_type)

    return ParseAsset(
        asset_id=asset_id,
        asset_type=asset_type,
        title=text.splitlines()[0][:200] if text else None,
        text=text,
        page_numbers=sorted(set(position.page for position in positions)),
        positions=positions,
        width=width,
        height=height,
        sha256=sha256,
        storage=storage,
        metadata=metadata or {},
    )


def build_generic_artifact(
    *,
    document: ParseDocument,
    markdown: str,
    chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    metadata: dict[str, Any] | None = None,
) -> ParseArtifact:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", markdown or "") if part.strip()]
    blocks: list[ParseBlock] = []
    for idx, paragraph in enumerate(paragraphs):
        block_type = "title" if paragraph.startswith("#") else "text"
        text = clean_text(paragraph)
        blocks.append(
            ParseBlock(
                block_id=f"block-{idx:04d}",
                block_type=block_type,
                text=text,
                token_count=count_tokens(text),
            )
        )
    chunks = build_chunks(
        blocks,
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
        strategy=chunk_strategy,
    )
    return ParseArtifact(
        document=document,
        markdown=markdown,
        blocks=blocks,
        chunks=chunks,
        metadata=metadata or {},
    )


def build_native_pdf_artifact(
    *,
    document: ParseDocument,
    markdown: str,
    boxes: list[dict[str, Any]],
    chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    metadata: dict[str, Any] | None = None,
) -> ParseArtifact:
    blocks: list[ParseBlock] = []
    page_routes_by_number: dict[int, str] = {}
    if isinstance(metadata, dict):
        for item in metadata.get("page_routes") or []:
            if not isinstance(item, dict):
                continue
            page_number = int(item.get("page_number") or 0)
            route = str(item.get("route") or "").strip()
            if page_number > 0 and route:
                page_routes_by_number[page_number] = route
    for box in boxes:
        text = clean_text(str(box.get("text") or "")).strip()
        if not text:
            continue
        page_number = max(1, int(box.get("page_number") or 1))
        positions = []
        if all(key in box for key in ("x0", "x1", "top", "bottom")):
            positions.append(
                ParsePosition(
                    page=page_number,
                    left=float(box.get("x0") or 0),
                    right=float(box.get("x1") or 0),
                    top=float(box.get("top") or 0),
                    bottom=float(box.get("bottom") or 0),
                )
            )
        blocks.append(
            ParseBlock(
                block_id=f"block-{len(blocks):04d}",
                block_type="text",
                text=text,
                page_numbers=[page_number],
                positions=positions,
                token_count=count_tokens(text),
                metadata={
                    "source": "pdf_native_text",
                    **(
                        {"page_route": page_routes_by_number[page_number]}
                        if page_number in page_routes_by_number
                        else {}
                    ),
                },
            )
        )
    chunks = build_chunks(
        blocks,
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
        strategy=chunk_strategy,
    )
    return ParseArtifact(
        document=document,
        markdown=markdown,
        blocks=blocks,
        chunks=chunks,
        metadata={"source": "pdf_native_text", **(metadata or {})},
    )


def build_csv_artifact(
    *,
    document: ParseDocument,
    markdown: str,
    rows: list[list[str]],
    delimiter: str,
    chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    metadata: dict[str, Any] | None = None,
) -> ParseArtifact:
    table_text = clean_text(table_to_md(rows)).strip() or clean_text(markdown or "").strip()
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)
    asset_id = build_asset_id(
        "table",
        (table_text or f"csv:{document.filename}").encode("utf-8", errors="ignore"),
        [1],
    )
    asset = ParseAsset(
        asset_id=asset_id,
        asset_type="table",
        title=f"{document.filename} table",
        text=table_text,
        page_numbers=[1],
        metadata={
            "source": "csv",
            "delimiter": delimiter,
            "row_count": row_count,
            "column_count": column_count,
        },
    )
    block = ParseBlock(
        block_id="block-0000",
        block_type="table",
        text=table_text,
        page_numbers=[1],
        token_count=count_tokens(table_text),
        asset_refs=[asset.asset_id],
        metadata={
            "source": "csv",
            "delimiter": delimiter,
            "row_count": row_count,
            "column_count": column_count,
        },
    )
    chunks = build_chunks(
        [block],
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
        strategy=chunk_strategy,
    )
    assets = enrich_asset_context([asset], [block], chunks)
    return ParseArtifact(
        document=document,
        markdown=markdown,
        assets=assets,
        blocks=[block],
        chunks=chunks,
        metadata={
            **(metadata or {}),
            "source": "csv",
            "delimiter": delimiter,
            "row_count": row_count,
            "column_count": column_count,
        },
    )


def build_epub_artifact(
    *,
    document: ParseDocument,
    markdown: str,
    blocks: list[dict[str, Any]],
    epub_metadata: dict[str, Any] | None = None,
    chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    metadata: dict[str, Any] | None = None,
) -> ParseArtifact:
    parse_blocks: list[ParseBlock] = []
    assets: list[ParseAsset] = []

    for raw_block in blocks:
        text = clean_text(str(raw_block.get("text") or "")).strip()
        block_type = str(raw_block.get("block_type") or "text").strip().lower()
        if block_type not in {"text", "title", "table", "list", "reference", "unknown"}:
            block_type = "text"
        if not text:
            continue
        chapter_index = max(0, int(raw_block.get("chapter_index") or 0))
        page_number = chapter_index + 1
        asset_refs: list[str] = []
        block_metadata = {
            "source": "epub",
            "chapter_index": chapter_index,
            "href": str(raw_block.get("href") or "").strip() or None,
        }
        if raw_block.get("title"):
            block_metadata["chapter_title"] = str(raw_block.get("title") or "").strip()

        if block_type == "table":
            row_count = int(raw_block.get("row_count") or 0)
            column_count = int(raw_block.get("column_count") or 0)
            asset_id = build_asset_id(
                "table",
                (text or f"epub:{document.filename}:{len(assets)}").encode("utf-8", errors="ignore"),
                [page_number],
            )
            asset = ParseAsset(
                asset_id=asset_id,
                asset_type="table",
                title=f"{document.filename} table {len(assets) + 1}",
                text=text,
                page_numbers=[page_number],
                metadata={
                    "source": "epub",
                    "chapter_index": chapter_index,
                    "href": block_metadata["href"],
                    "row_count": row_count,
                    "column_count": column_count,
                },
            )
            assets.append(asset)
            asset_refs.append(asset.asset_id)
            block_metadata["row_count"] = row_count
            block_metadata["column_count"] = column_count

        parse_blocks.append(
            ParseBlock(
                block_id=f"block-{len(parse_blocks):04d}",
                block_type=block_type,
                text=text,
                page_numbers=[page_number],
                token_count=count_tokens(text),
                asset_refs=asset_refs,
                metadata=block_metadata,
            )
        )

    chunks = build_chunks(
        parse_blocks,
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
        strategy=chunk_strategy,
    )
    assets = enrich_asset_context(assets, parse_blocks, chunks)
    return ParseArtifact(
        document=document,
        markdown=markdown,
        assets=assets,
        blocks=parse_blocks,
        chunks=chunks,
        metadata={
            **(metadata or {}),
            "source": "epub",
            "epub_metadata": epub_metadata or {},
            "chapter_count": int((epub_metadata or {}).get("chapter_count") or 0),
        },
    )


def build_rich_text_artifact(
    *,
    document: ParseDocument,
    markdown: str,
    blocks: list[dict[str, Any]],
    source: str,
    source_metadata: dict[str, Any] | None = None,
    chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    metadata: dict[str, Any] | None = None,
) -> ParseArtifact:
    normalized_source = str(source or "rich_text").strip().lower() or "rich_text"
    parse_blocks: list[ParseBlock] = []
    assets: list[ParseAsset] = []

    for raw_block in blocks:
        text = clean_text(str(raw_block.get("text") or "")).strip()
        block_type = str(raw_block.get("block_type") or "text").strip().lower()
        if block_type not in {"text", "title", "table", "list", "reference", "unknown"}:
            block_type = "text"
        if not text:
            continue
        page_number = max(1, int(raw_block.get("page_number") or raw_block.get("page") or 1))
        asset_refs: list[str] = []
        block_metadata: dict[str, Any] = {"source": normalized_source}
        if isinstance(raw_block.get("metadata"), dict):
            block_metadata.update(raw_block["metadata"])
            block_metadata["source"] = normalized_source
        if raw_block.get("heading_level") is not None:
            block_metadata["heading_level"] = int(raw_block.get("heading_level") or 1)

        if block_type == "table":
            row_count = int(raw_block.get("row_count") or 0)
            column_count = int(raw_block.get("column_count") or 0)
            asset_id = build_asset_id(
                "table",
                (text or f"{normalized_source}:{document.filename}:{len(assets)}").encode("utf-8", errors="ignore"),
                [page_number],
            )
            asset = ParseAsset(
                asset_id=asset_id,
                asset_type="table",
                title=f"{document.filename} table {len(assets) + 1}",
                text=text,
                page_numbers=[page_number],
                metadata={
                    "source": normalized_source,
                    "row_count": row_count,
                    "column_count": column_count,
                },
            )
            assets.append(asset)
            asset_refs.append(asset.asset_id)
            block_metadata["row_count"] = row_count
            block_metadata["column_count"] = column_count

        parse_blocks.append(
            ParseBlock(
                block_id=f"block-{len(parse_blocks):04d}",
                block_type=block_type,
                text=text,
                page_numbers=[page_number],
                token_count=count_tokens(text),
                asset_refs=asset_refs,
                metadata=block_metadata,
            )
        )

    chunks = build_chunks(
        parse_blocks,
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
        strategy=chunk_strategy,
    )
    assets = enrich_asset_context(assets, parse_blocks, chunks)
    return ParseArtifact(
        document=document,
        markdown=markdown,
        assets=assets,
        blocks=parse_blocks,
        chunks=chunks,
        metadata={
            **(metadata or {}),
            "source": normalized_source,
            "source_metadata": source_metadata or {},
            "block_count": len(parse_blocks),
        },
    )


def _image_box_positions(box: dict[str, Any]) -> list[ParsePosition]:
    return [
        ParsePosition(
            page=int(box.get("page_number", 0)) + 1,
            left=float(box.get("x0", 0.0)),
            right=float(box.get("x1", 0.0)),
            top=float(box.get("top", 0.0)),
            bottom=float(box.get("bottom", 0.0)),
        )
    ]


def _barcode_positions(barcode: dict[str, Any]) -> list[ParsePosition]:
    positions: list[ParsePosition] = []
    raw_positions = barcode.get("positions")
    if isinstance(raw_positions, list):
        for raw_position in raw_positions:
            if not isinstance(raw_position, dict):
                continue
            try:
                positions.append(
                    ParsePosition(
                        page=max(1, int(raw_position.get("page") or raw_position.get("page_number") or 1)),
                        left=float(raw_position.get("left") or 0.0),
                        right=float(raw_position.get("right") or 0.0),
                        top=float(raw_position.get("top") or 0.0),
                        bottom=float(raw_position.get("bottom") or 0.0),
                    )
                )
            except Exception:
                continue
    if positions:
        return positions

    bbox = barcode.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            return [
                ParsePosition(
                    page=max(1, int(barcode.get("page_number") or 1)),
                    left=float(bbox[0]),
                    right=float(bbox[2]),
                    top=float(bbox[1]),
                    bottom=float(bbox[3]),
                )
            ]
        except Exception:
            return []
    return []


def build_image_artifact(
    *,
    document: ParseDocument,
    markdown: str,
    image: Image.Image,
    boxes: list[dict[str, Any]],
    barcodes: list[dict[str, Any]] | None = None,
    chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    store: ArtifactStore | None = None,
    artifact_paths=None,
    metadata: dict[str, Any] | None = None,
) -> ParseArtifact:
    image_asset = _build_visual_asset(
        store=store,
        artifact_paths=artifact_paths,
        image=image,
        asset_type="image",
        text=clean_text(markdown or "").strip(),
        positions=[
            ParsePosition(
                page=1,
                left=0.0,
                right=float(image.size[0]),
                top=0.0,
                bottom=float(image.size[1]),
            )
        ],
        metadata={
            "source": "source_image",
            "filename": document.filename,
        },
    )
    blocks: list[ParseBlock] = []
    barcode_assets: list[ParseAsset] = []
    for box in boxes:
        text = clean_text(str(box.get("text") or "")).strip()
        if not text:
            continue
        block_type = _normalize_block_type(box)
        if block_type in {"figure", "table", "equation", "seal"}:
            block_type = "text"
        blocks.append(
            ParseBlock(
                block_id=f"block-{len(blocks):04d}",
                block_type=block_type if block_type in {"text", "title", "list", "reference", "unknown"} else "text",
                text=text,
                page_numbers=[int(box.get("page_number", 0)) + 1],
                positions=_image_box_positions(box),
                token_count=count_tokens(text),
                asset_refs=[image_asset.asset_id],
                metadata={
                    "source": "image_ocr",
                    "score": float(box.get("score") or 0.0),
                    "layout_type": str(box.get("layout_type") or "").strip().lower() or None,
                },
            )
        )

    for barcode in barcodes or []:
        text = clean_text(str(barcode.get("text") or "")).strip()
        if not text:
            continue
        positions = _barcode_positions(barcode)
        if not positions:
            continue
        barcode_type = str(barcode.get("barcode_type") or "unknown").strip().lower() or "unknown"
        source = str(barcode.get("source") or "barcode_detector").strip() or "barcode_detector"
        asset_id = build_asset_id(
            "barcode",
            f"{barcode_type}:{text}:{barcode.get('bbox') or ''}".encode("utf-8", errors="ignore"),
            sorted(set(position.page for position in positions)),
        )
        metadata = {
            "source": "barcode_detector",
            "barcode_type": barcode_type,
            "detector": source,
        }
        barcode_asset = ParseAsset(
            asset_id=asset_id,
            asset_type="barcode",
            title=f"{barcode_type}: {text[:120]}",
            text=text,
            page_numbers=sorted(set(position.page for position in positions)),
            positions=positions,
            metadata=metadata,
        )
        barcode_assets.append(barcode_asset)
        blocks.append(
            ParseBlock(
                block_id=f"block-{len(blocks):04d}",
                block_type="barcode",
                text=text,
                page_numbers=barcode_asset.page_numbers,
                positions=positions,
                token_count=count_tokens(text),
                asset_refs=[asset_id],
                metadata=metadata,
            )
        )

    if not blocks and image_asset.text:
        blocks.append(
            ParseBlock(
                block_id="block-0000",
                block_type="text",
                text=image_asset.text,
                page_numbers=[1],
                positions=image_asset.positions,
                token_count=count_tokens(image_asset.text),
                asset_refs=[image_asset.asset_id],
                metadata={"source": "image_ocr"},
            )
        )

    chunks = build_chunks(
        blocks,
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
        strategy=chunk_strategy,
    )
    assets = enrich_asset_context([image_asset, *barcode_assets], blocks, chunks)
    return ParseArtifact(
        document=document,
        markdown=markdown,
        assets=assets,
        blocks=blocks,
        chunks=chunks,
        metadata={
            **(metadata or {}),
            "source": "image",
            "width": image.size[0],
            "height": image.size[1],
            "ocr_box_count": len(boxes),
            "barcode_count": len(barcode_assets),
        },
    )


def build_deepdoc_artifact(
    *,
    document: ParseDocument,
    markdown: str,
    parser,
    boxes: list[dict[str, Any]],
    tables_with_positions: list[tuple[tuple[Any, Any], list[tuple[int, float, float, float, float]]]],
    figures_with_positions: list[tuple[tuple[Any, Any], list[tuple[int, float, float, float, float]]]],
    zoomin: int,
    chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    store: ArtifactStore | None = None,
    artifact_paths=None,
    metadata: dict[str, Any] | None = None,
) -> ParseArtifact:
    assets: list[ParseAsset] = []
    blocks: list[ParseBlock] = []

    def append_block(
        *,
        block_type: str,
        text: str,
        page_numbers: list[int],
        positions: list[ParsePosition],
        asset_refs: list[str] | None = None,
        block_metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized_text = clean_text(text or "").strip()
        if not normalized_text and not asset_refs:
            return
        block_id = f"block-{len(blocks):04d}"
        blocks.append(
            ParseBlock(
                block_id=block_id,
                block_type=block_type if block_type in {"text", "title", "table", "figure", "equation", "seal", "list", "reference", "unknown"} else "unknown",
                text=normalized_text,
                page_numbers=page_numbers,
                positions=positions,
                token_count=count_tokens(normalized_text),
                asset_refs=asset_refs or [],
                metadata=block_metadata or {},
            )
        )

    for box in boxes:
        positions = _positions_from_box(parser, box, zoomin)
        page_numbers = sorted(set(position.page for position in positions)) or [int(box.get("page_number", 0))]
        block_type = _normalize_block_type(box)
        block_metadata: dict[str, Any] = _cross_page_box_metadata(box)

        if block_type == "seal":
            local_bbox = box.get("source_bbox_local") or []
            page_index = int(box.get("source_page_index", page_numbers[0] - 1))
            if local_bbox and 0 <= page_index < len(getattr(parser, "page_images", [])):
                left, top, right, bottom = local_bbox
                page_img = parser.page_images[page_index]
                crop = page_img.crop((left * zoomin, top * zoomin, right * zoomin, bottom * zoomin))
                asset = _build_visual_asset(
                    store=store,
                    artifact_paths=artifact_paths,
                    image=crop,
                    asset_type="seal",
                    text=box.get("text", ""),
                    positions=[
                        ParsePosition(
                            page=page_index + 1,
                            left=float(left),
                            right=float(right),
                            top=float(top),
                            bottom=float(bottom),
                        )
                    ],
                    metadata={"polygon": box.get("source_polygon", [])},
                )
                assets.append(asset)
                block_metadata["source_bbox_local"] = local_bbox
                append_block(
                    block_type="seal",
                    text=str(box.get("text", "")),
                    page_numbers=page_numbers,
                    positions=positions,
                    asset_refs=[asset.asset_id],
                    block_metadata=block_metadata,
                )
                continue

        append_block(
            block_type=block_type,
            text=str(box.get("text", "")),
            page_numbers=page_numbers,
            positions=positions,
            block_metadata=block_metadata,
        )

    for (img, figure_text), poss in figures_with_positions:
        figure_caption = clean_text(_join_text(figure_text)).strip()
        positions = _positions_from_tuples(poss)
        asset = _build_visual_asset(
            store=store,
            artifact_paths=artifact_paths,
            image=img,
            asset_type="figure",
            text=figure_caption,
            positions=positions,
        )
        assets.append(asset)
        append_block(
            block_type="figure",
            text=figure_caption or "[Figure]",
            page_numbers=asset.page_numbers,
            positions=positions,
            asset_refs=[asset.asset_id],
        )

    for (img, table_payload), poss in tables_with_positions:
        table_text = clean_text(table_to_md(table_payload)).strip()
        positions = _positions_from_tuples(poss)
        table_metadata = _cross_page_positions_metadata(positions)
        asset = _build_visual_asset(
            store=store,
            artifact_paths=artifact_paths,
            image=img,
            asset_type="table",
            text=table_text,
            positions=positions,
            metadata=table_metadata,
        )
        assets.append(asset)
        append_block(
            block_type="table",
            text=table_text or "[Table]",
            page_numbers=asset.page_numbers,
            positions=positions,
            asset_refs=[asset.asset_id],
            block_metadata=table_metadata,
        )

    blocks.sort(
        key=lambda block: (
            min(block.page_numbers) if block.page_numbers else 0,
            block.positions[0].top if block.positions else 0,
            block.positions[0].left if block.positions else 0,
            block.block_id,
        )
    )
    normalized_blocks = [
        block.model_copy(update={"block_id": f"block-{idx:04d}"})
        for idx, block in enumerate(blocks)
    ]
    chunks = build_chunks(
        normalized_blocks,
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
        strategy=chunk_strategy,
    )
    assets = enrich_asset_context(assets, normalized_blocks, chunks)
    return ParseArtifact(
        document=document,
        markdown=markdown,
        assets=assets,
        blocks=normalized_blocks,
        chunks=chunks,
        metadata=metadata or {},
    )


def build_paddleocr_artifact(
    *,
    document: ParseDocument,
    markdown: str,
    raw_result: dict[str, Any] | None,
    parser,
    store: ArtifactStore | None = None,
    artifact_paths=None,
    chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    metadata: dict[str, Any] | None = None,
) -> ParseArtifact:
    blocks: list[ParseBlock] = []
    assets: list[ParseAsset] = []
    if not isinstance(raw_result, dict):
        return build_generic_artifact(
            document=document,
            markdown=markdown,
            chunk_max_tokens=chunk_max_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            chunk_strategy=chunk_strategy,
            metadata=metadata or {},
        )

    layout_results = raw_result.get("layoutParsingResults") or raw_result.get("layout_parsing_results", [])
    for page_idx, layout_result in enumerate(layout_results):
        markdown_images = (layout_result.get("markdown") or {}).get("images") or {}
        pruned_result = layout_result.get("prunedResult") or layout_result.get("pruned_result", {})
        parsing_res_list = pruned_result.get("parsing_res_list") or pruned_result.get("parsingResList", [])
        for block in parsing_res_list:
            label = str(
                block.get("block_label")
                or block.get("blockLabel")
                or block.get("block_type")
                or block.get("blockType")
                or "text"
            ).strip().lower()
            bbox = block.get("block_bbox") or block.get("blockBbox") or [0, 0, 0, 0]
            if len(bbox) != 4:
                bbox = [0, 0, 0, 0]
            positions = [
                ParsePosition(
                    page=page_idx + 1,
                    left=float(bbox[0]) / parser._ZOOMIN,
                    right=float(bbox[2]) / parser._ZOOMIN,
                    top=float(bbox[1]) / parser._ZOOMIN,
                    bottom=float(bbox[3]) / parser._ZOOMIN,
                )
            ]
            text = clean_text(
                str(block.get("block_content") or block.get("blockContent") or "").strip()
            )
            asset_refs: list[str] = []
            raw_img = (
                block.get("img_path")
                or block.get("image_path")
                or block.get("block_image")
                or block.get("blockImage")
                or ""
            )
            resolved_img = str(raw_img or "").strip()
            if resolved_img in markdown_images:
                mapped = markdown_images.get(resolved_img)
                if isinstance(mapped, str) and mapped.strip():
                    resolved_img = mapped.strip()
            if label in {"figure", "image", "table", "seal", "stamp"} and resolved_img:
                asset_type = "seal" if label in {"seal", "stamp"} else ("table" if label == "table" else "figure")
                asset = _build_file_or_url_asset(
                    store=store,
                    artifact_paths=artifact_paths,
                    asset_type=asset_type,
                    text=text or f"[{label}]",
                    positions=positions,
                    source_url=resolved_img if resolved_img.startswith(("http://", "https://")) else None,
                    source_path=resolved_img if resolved_img and not resolved_img.startswith(("http://", "https://")) else None,
                    metadata={"label": label},
                )
                if asset is not None:
                    assets.append(asset)
                    asset_refs.append(asset.asset_id)
            elif label in {"figure", "image", "table", "seal", "stamp"}:
                fallback_img = _crop_positions_image(parser, positions)
                if fallback_img is not None:
                    asset_type = "seal" if label in {"seal", "stamp"} else ("table" if label == "table" else "figure")
                    asset = _build_visual_asset(
                        store=store,
                        artifact_paths=artifact_paths,
                        image=fallback_img,
                        asset_type=asset_type,
                        text=text or f"[{label}]",
                        positions=positions,
                        metadata={"label": label, "source": "bbox_crop"},
                    )
                    assets.append(asset)
                    asset_refs.append(asset.asset_id)
            block_type = "seal" if label in {"seal", "stamp"} else ("figure" if label in {"figure", "image"} else ("table" if label == "table" else "text"))
            blocks.append(
                ParseBlock(
                    block_id=f"block-{len(blocks):04d}",
                    block_type=block_type,
                    text=text or f"[{label}]",
                    page_numbers=[page_idx + 1],
                    positions=positions,
                    token_count=count_tokens(text or f"[{label}]"),
                    asset_refs=asset_refs,
                    metadata={"label": label},
                )
            )

    chunks = build_chunks(
        blocks,
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
        strategy=chunk_strategy,
    )
    assets = enrich_asset_context(assets, blocks, chunks)
    return ParseArtifact(
        document=document,
        markdown=markdown,
        assets=assets,
        blocks=blocks,
        chunks=chunks,
        metadata=metadata or {},
    )


def build_mineru_artifact(
    *,
    document: ParseDocument,
    markdown: str,
    raw_outputs: list[dict[str, Any]] | None,
    parser=None,
    store: ArtifactStore | None = None,
    artifact_paths=None,
    chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    metadata: dict[str, Any] | None = None,
) -> ParseArtifact:
    blocks: list[ParseBlock] = []
    assets: list[ParseAsset] = []
    if not isinstance(raw_outputs, list):
        return build_generic_artifact(
            document=document,
            markdown=markdown,
            chunk_max_tokens=chunk_max_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            chunk_strategy=chunk_strategy,
            metadata=metadata or {},
        )

    for output in raw_outputs:
        raw_type = str(output.get("type") or "text").strip().lower()
        page_idx = int(output.get("page_idx", 0))
        page = page_idx + 1
        bbox = output.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        left, top, right, bottom = map(float, bbox)
        if (
            parser is not None
            and getattr(parser, "page_images", None)
            and 0 <= page_idx < len(parser.page_images)
        ):
            page_width, page_height = parser.page_images[page_idx].size
            left = (left / 1000.0) * page_width
            right = (right / 1000.0) * page_width
            top = (top / 1000.0) * page_height
            bottom = (bottom / 1000.0) * page_height
        positions = [
            ParsePosition(
                page=page,
                left=left,
                right=right,
                top=top,
                bottom=bottom,
            )
        ]
        if raw_type == "table":
            text = (
                output.get("table_body", "")
                + "\n".join(output.get("table_caption", []))
                + "\n".join(output.get("table_footnote", []))
            ).strip()
        elif raw_type == "image":
            text = (
                "".join(output.get("image_caption", []))
                + "\n"
                + "".join(output.get("image_footnote", []))
            ).strip()
        elif raw_type == "code":
            text = (output.get("code_body", "") + "\n".join(output.get("code_caption", []))).strip()
        elif raw_type == "list":
            text = "\n".join(output.get("list_items", [])).strip()
        else:
            text = str(output.get("text", "")).strip()
        text = clean_text(text)

        source_path = (
            output.get("table_img_path")
            or output.get("img_path")
            or output.get("equation_img_path")
        )
        asset_refs: list[str] = []
        if raw_type in {"table", "image", "equation"} and source_path:
            asset_type = "table" if raw_type == "table" else ("equation" if raw_type == "equation" else "figure")
            asset = _build_file_or_url_asset(
                store=store,
                artifact_paths=artifact_paths,
                asset_type=asset_type,
                text=text or f"[{raw_type}]",
                positions=positions,
                source_path=source_path,
                metadata={"raw_type": raw_type},
            )
            if asset is not None:
                assets.append(asset)
                asset_refs.append(asset.asset_id)
        elif raw_type in {"table", "image", "equation"}:
            fallback_img = _crop_positions_image(parser, positions)
            if fallback_img is not None:
                asset_type = "table" if raw_type == "table" else ("equation" if raw_type == "equation" else "figure")
                asset = _build_visual_asset(
                    store=store,
                    artifact_paths=artifact_paths,
                    image=fallback_img,
                    asset_type=asset_type,
                    text=text or f"[{raw_type}]",
                    positions=positions,
                    metadata={"raw_type": raw_type, "source": "bbox_crop"},
                )
                assets.append(asset)
                asset_refs.append(asset.asset_id)

        block_type = "figure" if raw_type == "image" else ("table" if raw_type == "table" else ("equation" if raw_type == "equation" else "text"))
        blocks.append(
            ParseBlock(
                block_id=f"block-{len(blocks):04d}",
                block_type=block_type,
                text=text or f"[{raw_type}]",
                page_numbers=[page],
                positions=positions,
                token_count=count_tokens(text or f"[{raw_type}]"),
                asset_refs=asset_refs,
                metadata={"raw_type": raw_type},
            )
        )

    chunks = build_chunks(
        blocks,
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
        strategy=chunk_strategy,
    )
    assets = enrich_asset_context(assets, blocks, chunks)
    return ParseArtifact(
        document=document,
        markdown=markdown,
        assets=assets,
        blocks=blocks,
        chunks=chunks,
        metadata=metadata or {},
    )

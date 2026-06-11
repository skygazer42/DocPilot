from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from common import logger


def _image_to_bgr_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        return rgb[:, :, ::-1].copy()
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a PIL Image or numpy array")
    array = image
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3].copy()
    raise ValueError("image array must be grayscale or RGB/BGR")


def _bbox_from_points(points: Any) -> tuple[float, float, float, float] | None:
    try:
        array = np.asarray(points, dtype=float).reshape(-1, 2)
    except Exception:
        return None
    if array.size == 0:
        return None
    left = float(np.min(array[:, 0]))
    right = float(np.max(array[:, 0]))
    top = float(np.min(array[:, 1]))
    bottom = float(np.max(array[:, 1]))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _position_payload(bbox: tuple[float, float, float, float]) -> list[dict[str, float | int]]:
    left, top, right, bottom = bbox
    return [
        {
            "page": 1,
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
        }
    ]


def _normalize_barcode_type(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"qrcode", "qr"}:
        return "qr_code"
    return normalized or "unknown"


def _dedupe_key(detection: dict[str, Any]) -> tuple[str, str, tuple[int, int, int, int]]:
    bbox = detection.get("bbox") if isinstance(detection.get("bbox"), list) else []
    rounded_bbox = tuple(int(round(float(value))) for value in bbox[:4])
    return str(detection.get("barcode_type") or ""), str(detection.get("text") or ""), rounded_bbox


def _bbox_iou(left_bbox: list[Any], right_bbox: list[Any]) -> float:
    if len(left_bbox) < 4 or len(right_bbox) < 4:
        return 0.0
    try:
        left = [float(value) for value in left_bbox[:4]]
        right = [float(value) for value in right_bbox[:4]]
    except Exception:
        return 0.0
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _is_duplicate_detection(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if (
        str(existing.get("barcode_type") or "") != str(candidate.get("barcode_type") or "")
        or str(existing.get("text") or "") != str(candidate.get("text") or "")
    ):
        return False
    if _dedupe_key(existing) == _dedupe_key(candidate):
        return True
    existing_bbox = existing.get("bbox") if isinstance(existing.get("bbox"), list) else []
    candidate_bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), list) else []
    return _bbox_iou(existing_bbox, candidate_bbox) >= 0.8


def _append_detection(
    detections: list[dict[str, Any]],
    *,
    barcode_type: str,
    text: str,
    bbox: tuple[float, float, float, float],
    source: str,
) -> None:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return
    detection = {
        "barcode_type": _normalize_barcode_type(barcode_type),
        "text": normalized_text,
        "page_number": 1,
        "positions": _position_payload(bbox),
        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        "source": source,
    }
    if any(_is_duplicate_detection(existing, detection) for existing in detections):
        return
    detections.append(detection)


def _detect_opencv_qr(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    import cv2

    detections: list[dict[str, Any]] = []
    detector = cv2.QRCodeDetector()
    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(image_bgr)
    except Exception:
        ok = False
        decoded_info = []
        points = None

    if ok and points is not None:
        for text, qr_points in zip(decoded_info or [], points):
            bbox = _bbox_from_points(qr_points)
            if bbox is None:
                continue
            _append_detection(
                detections,
                barcode_type="qr_code",
                text=str(text or ""),
                bbox=bbox,
                source="opencv_qrcode",
            )
        if detections:
            return detections

    text, points, _ = detector.detectAndDecode(image_bgr)
    bbox = _bbox_from_points(points)
    if bbox is not None:
        _append_detection(
            detections,
            barcode_type="qr_code",
            text=text,
            bbox=bbox,
            source="opencv_qrcode",
        )
    return detections


def _detect_pyzbar(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    try:
        from pyzbar import pyzbar
    except Exception:
        return []

    detections: list[dict[str, Any]] = []
    try:
        decoded_items = pyzbar.decode(image_bgr)
    except Exception:
        logger.exception("pyzbar barcode detection failed")
        return []

    for item in decoded_items:
        text = item.data.decode("utf-8", errors="replace") if getattr(item, "data", None) else ""
        rect = getattr(item, "rect", None)
        polygon = getattr(item, "polygon", None)
        bbox = _bbox_from_points([(point.x, point.y) for point in polygon]) if polygon else None
        if bbox is None and rect is not None:
            bbox = (float(rect.left), float(rect.top), float(rect.left + rect.width), float(rect.top + rect.height))
        if bbox is None:
            continue
        _append_detection(
            detections,
            barcode_type=getattr(item, "type", "") or "unknown",
            text=text,
            bbox=bbox,
            source="pyzbar",
        )
    return detections


def detect_barcodes(image: Image.Image | np.ndarray) -> list[dict[str, Any]]:
    """Detect QR/barcode payloads from a single document image."""
    image_bgr = _image_to_bgr_array(image)
    detections: list[dict[str, Any]] = []

    try:
        for detection in _detect_opencv_qr(image_bgr):
            if not any(_is_duplicate_detection(existing, detection) for existing in detections):
                detections.append(detection)
    except Exception:
        logger.exception("OpenCV QRCodeDetector failed")

    for detection in _detect_pyzbar(image_bgr):
        if not any(_is_duplicate_detection(existing, detection) for existing in detections):
            detections.append(detection)

    return detections

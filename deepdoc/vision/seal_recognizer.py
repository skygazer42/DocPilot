#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
"""
印章检测(PP-OCRv4 seal_det ONNX 封装)。

设计要点:
- **单例 + 懒加载**: 进程级缓存,首次 detect() 时才加载模型。
- **异常隔离**: 模型缺失或推理失败统一返回空列表,不阻塞主解析流程。
- **职责**: 只做 det + 极坐标展开,文字识别在 caller 端复用现有 OCR rec。
- **模型路径**: ``resources/models/seal/seal_det.onnx``。
"""

import math
import os
import threading

import cv2
import numpy as np
import onnxruntime as ort

from common import logger
from common.model_store import ensure_groups
from deepdoc.vision.postprocess import DBPostProcess

_SEAL_MODEL_SUBDIR = "seal"
_SEAL_DET_FILENAME = "seal_det.onnx"
_lock = threading.Lock()
_instance = None
_load_failed = False


def _model_path() -> str:
    return os.path.join(ensure_groups("seal"), _SEAL_MODEL_SUBDIR, _SEAL_DET_FILENAME)


def _preprocess(image: np.ndarray, limit_side_len: int = 1280) -> tuple[np.ndarray, tuple[float, float]]:
    """PaddleOCR DetResizeForTest(max) + ImageNet 归一化 + HWC→CHW + batch。"""
    h, w = image.shape[:2]
    ratio = limit_side_len / float(max(h, w))
    if ratio < 1.0:
        new_h = int(round(h * ratio / 32) * 32)
        new_w = int(round(w * ratio / 32) * 32)
    else:
        new_h = int(round(h / 32) * 32)
        new_w = int(round(w / 32) * 32)
    new_h = max(32, new_h)
    new_w = max(32, new_w)
    resized = cv2.resize(image, (new_w, new_h))
    img = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
    ratio_h, ratio_w = float(new_h) / h, float(new_w) / w
    return img, (ratio_h, ratio_w)


def _unfold_seal(image: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
    """对圆形印章做极坐标展开,返回水平文字条。

    Returns:
        展开后的水平条(BGR uint8 ndarray),失败返回 None。
    """
    poly = polygon.astype(np.float32).reshape(-1, 2)
    if len(poly) < 3:
        return None
    (cx, cy), radius = cv2.minEnclosingCircle(poly)
    radius = int(radius)
    if radius < 15:
        return None

    h, w = image.shape[:2]
    pad = 10
    x0 = max(0, int(cx) - radius - pad)
    y0 = max(0, int(cy) - radius - pad)
    x1 = min(w, int(cx) + radius + pad)
    y1 = min(h, int(cy) + radius + pad)
    patch = image[y0:y1, x0:x1].copy()
    if patch.size == 0:
        return None

    new_cx = float(cx) - x0
    new_cy = float(cy) - y0

    out_w = max(64, int(2 * math.pi * radius))
    out_h = max(32, radius)
    try:
        polar = cv2.warpPolar(
            patch,
            (out_w, out_h),
            (new_cx, new_cy),
            float(radius),
            cv2.WARP_FILL_OUTLIERS | cv2.INTER_LINEAR,
        )
    except Exception:
        logger.exception("warpPolar failed")
        return None

    # 印章弧形文字朝外,极坐标展开后位于外圈、上下颠倒,取近边缘条并翻转
    band = polar[int(out_h * 0.55):, :]
    band = cv2.flip(band, 0)
    if band.size == 0:
        return None
    return band


class SealRecognizer:
    """PP-OCRv4 seal_det 推理 + 极坐标展开。"""

    def __init__(self) -> None:
        model_path = _model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"seal_det.onnx not found at {model_path}. "
                "Run `python download_models.py seal` to fetch."
            )
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(os.environ.get("OCR_INTRA_OP_NUM_THREADS", "2"))
        sess_options.inter_op_num_threads = int(os.environ.get("OCR_INTER_OP_NUM_THREADS", "2"))
        try:
            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception:
            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        self._input_name = self._session.get_inputs()[0].name
        self._postprocess = DBPostProcess(
            thresh=0.3,
            box_thresh=0.7,
            max_candidates=1000,
            unclip_ratio=0.5,
            use_dilation=False,
            score_mode="fast",
            box_type="poly",
        )
        logger.info("SealRecognizer loaded from %s", model_path)

    def detect(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """检测印章并展开成水平条。

        Args:
            image: 整页 BGR ndarray(H, W, 3)。

        Returns:
            列表 [(unfolded_strip, polygon, score), ...];polygon 是原图坐标系下的多边形。
        """
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        try:
            inp, (ratio_h, ratio_w) = _preprocess(image)
            preds = self._session.run(None, {self._input_name: inp})[0]
            shape_list = np.array([[h, w, ratio_h, ratio_w]])
            post_result = self._postprocess({"maps": preds}, shape_list)
        except Exception:
            logger.exception("SealRecognizer.detect inference failed")
            return []

        boxes = post_result[0]["points"] if post_result and "points" in post_result[0] else []
        scores = post_result[0].get("scores", [1.0] * len(boxes))
        results: list[tuple[np.ndarray, np.ndarray, float]] = []
        for poly, score in zip(boxes, scores):
            poly_np = np.array(poly, dtype=np.float32)
            strip = _unfold_seal(image, poly_np)
            if strip is None:
                continue
            results.append((strip, poly_np, float(score)))
        return results


def get_seal_recognizer():
    """获取单例;加载失败后续调用直接返回 None。"""
    global _instance, _load_failed
    if _instance is not None:
        return _instance
    if _load_failed:
        return None
    with _lock:
        if _instance is not None:
            return _instance
        if _load_failed:
            return None
        try:
            _instance = SealRecognizer()
        except Exception as exc:
            _load_failed = True
            logger.error("SealRecognizer init failed: %s", exc)
            return None
    return _instance

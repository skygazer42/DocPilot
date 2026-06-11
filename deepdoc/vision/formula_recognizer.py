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
LaTeX 公式识别封装。

设计要点:
- **单例 + 懒加载**: 进程级缓存,首次 predict() 时才装载模型,避免无公式场景的内存浪费。
- **异常隔离**: 加载或推理失败统一返回 ("", 0.0),不抛出阻塞主解析流程。
- **模型路径**: 读取 ``resources/models/formula/`` 下的 image_resizer.onnx / encoder.onnx / decoder.onnx / tokenizer.json。
- **模型模式**: 默认 RapidLatexOCR; ``DEEPDOC_FORMULA_MODE=pp_formula_net_s`` 时走 PaddleX PP-FormulaNet-S 适配器。
- **依赖**: 可选依赖,需 ``pip install -e ".[formula]"`` 或 ``pip install -e ".[formula-v2]"`` 才能启用。
"""

import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

import cv2
import numpy as np

from common import logger
from common import setting
from common.model_store import ensure_groups

_FORMULA_MODEL_SUBDIR = "formula"
_PP_FORMULA_DEFAULT_MODEL_NAME = "PP-FormulaNet-S"
_lock = threading.Lock()
_instance = None
_load_failed = False


def _formula_mode() -> str:
    mode = (os.environ.get("DEEPDOC_FORMULA_MODE") or "rapidlatex").strip().lower()
    if mode in {"", "rapidlatex", "rapid_latex_ocr", "rapid-latex-ocr"}:
        return "rapidlatex"
    if mode in {"pp_formula_net_s", "pp-formula-net-s", "ppformulanet_s", "ppformulanet"}:
        return "pp_formula_net_s"
    raise ValueError(f"Unsupported DEEPDOC_FORMULA_MODE value: {mode}")


def _resolve_model_paths() -> dict[str, str] | None:
    """返回 formula 四个文件的本地路径;若任一缺失返回 None。"""
    mode = _formula_mode()
    if mode == "pp_formula_net_s":
        try:
            base_root = ensure_groups("formula_v2")
        except Exception:
            return None
        path = os.path.join(base_root, _FORMULA_MODEL_SUBDIR, "pp_formula_net_s.onnx")
        return {"model_path": path, "model_dir": os.path.dirname(path)}

    try:
        base_root = ensure_groups("formula")
    except Exception:
        return None
    base = os.path.join(base_root, _FORMULA_MODEL_SUBDIR)
    paths = {
        "image_resizer_path": os.path.join(base, "image_resizer.onnx"),
        "encoder_path": os.path.join(base, "encoder.onnx"),
        "decoder_path": os.path.join(base, "decoder.onnx"),
        "tokenizer_json": os.path.join(base, "tokenizer.json"),
    }
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    return None


def _pp_formula_device() -> str:
    return (os.environ.get("DEEPDOC_PP_FORMULA_NET_DEVICE") or "cpu").strip() or "cpu"


def _pp_formula_model_name() -> str:
    return (
        os.environ.get("DEEPDOC_PP_FORMULA_NET_MODEL_NAME")
        or _PP_FORMULA_DEFAULT_MODEL_NAME
    ).strip() or _PP_FORMULA_DEFAULT_MODEL_NAME


def _extract_pp_formula_text(result: Any) -> str:
    """Extract LaTeX text from common PaddleX prediction result shapes."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("rec_formula", "formula", "latex", "text", "content"):
            if key in result and result[key] is not None:
                return str(result[key]).strip()
        nested = result.get("res") or result.get("result")
        if nested is not None:
            return _extract_pp_formula_text(nested)
    if isinstance(result, (list, tuple)):
        for item in result:
            text = _extract_pp_formula_text(item)
            if text:
                return text
    for attr in ("rec_formula", "formula", "latex", "text"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if value is not None:
                return str(value).strip()
    return ""


class PPFormulaNetSRecognizer:
    """PaddleX PP-FormulaNet-S adapter.

    PaddleX owns preprocessing and LaTeX decoding for PP-FormulaNet-S. DeepDoc
    only provides the locally downloaded model directory and extracts the
    returned LaTeX text.
    """

    def __init__(self, paths: dict[str, str]) -> None:
        try:
            import paddlex as px  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "paddlex is not installed. "
                'Install via `pip install -e ".[formula-v2]"` to enable '
                "DEEPDOC_FORMULA_MODE=pp_formula_net_s."
            ) from exc

        model_dir = paths.get("model_dir") or os.path.dirname(paths["model_path"])
        if not Path(paths["model_path"]).exists():
            raise FileNotFoundError(
                f"PP-FormulaNet-S model is missing: {paths['model_path']}. "
                "Run `python download_models.py formula_v2` to fetch."
            )
        self._model = px.create_model(
            model_name=_pp_formula_model_name(),
            model_dir=model_dir,
            device=_pp_formula_device(),
        )

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        if image is None or image.size == 0:
            return "", 0.0
        started_at = time.time()
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                temp_path = handle.name
            if not cv2.imwrite(temp_path, image):
                logger.error("PPFormulaNetSRecognizer: failed to encode formula crop")
                return "", 0.0
            result = self._model.predict(input=temp_path, batch_size=1)
            return _extract_pp_formula_text(result), time.time() - started_at
        except Exception:
            logger.exception("PPFormulaNetSRecognizer.predict failed")
            return "", 0.0
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


class FormulaRecognizer:
    """公式识别薄封装,仅暴露 predict()。"""

    def __init__(self) -> None:
        self.mode = _formula_mode()
        if self.mode == "pp_formula_net_s":
            paths = _resolve_model_paths()
            if paths is None:
                raise FileNotFoundError(
                    f"PP-FormulaNet-S model is missing under {setting.MODELS_DIR}/{_FORMULA_MODEL_SUBDIR}. "
                    "Run `python download_models.py formula_v2` to fetch."
                )
            logger.info("FormulaRecognizer: loading PP-FormulaNet-S from %s", paths["model_dir"])
            self._model = PPFormulaNetSRecognizer(paths)
            return

        try:
            from rapid_latex_ocr import LaTeXOCR  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "rapid_latex_ocr is not installed. "
                'Install via `pip install -e ".[formula]"`'
            ) from exc

        paths = _resolve_model_paths()
        if paths is not None:
            logger.info(
                "FormulaRecognizer: loading local models from %s",
                os.path.dirname(paths["encoder_path"]),
            )
            self._model = LaTeXOCR(**paths)
        else:
            raise FileNotFoundError(
                f"Formula models are missing under {setting.MODELS_DIR}/{_FORMULA_MODEL_SUBDIR}. "
                "Run `python download_models.py formula` to fetch."
            )

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """对单张公式图像识别为 LaTeX 字符串。

        Args:
            image: BGR 或 RGB 的 numpy 数组(H, W, 3),由调用方裁剪而来。

        Returns:
            (latex, elapsed_seconds)。识别失败时返回 ("", 0.0)。
        """
        if image is None or image.size == 0:
            return "", 0.0
        if self.mode == "pp_formula_net_s":
            return self._model.predict(image)
        start = time.time()
        try:
            result = self._model(image)
        except Exception:
            logger.exception("FormulaRecognizer.predict failed")
            return "", 0.0

        latex = ""
        if isinstance(result, tuple) and result:
            latex = str(result[0] or "")
        elif isinstance(result, str):
            latex = result
        return latex.strip(), time.time() - start


def get_formula_recognizer():
    """获取单例;加载失败后续调用直接返回 None,不再重试。"""
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
            _instance = FormulaRecognizer()
        except Exception as exc:
            _load_failed = True
            logger.error("FormulaRecognizer init failed: %s", exc)
            return None
    return _instance

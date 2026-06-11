#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import gc
from common import logger
import copy
import time
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from common.model_store import ensure_groups
from common.misc_utils import pip_install_torch
from common import settings
from .operators import *  # noqa: F403
from . import operators
import math
import numpy as np
import cv2
import onnxruntime as ort

from .postprocess import build_post_process

DEFAULT_MODEL_CACHE_MAX_SIZE = 8
DEFAULT_MODEL_CACHE_IDLE_TTL_SECONDS = 3600


@dataclass
class LoadedModelEntry:
    model: tuple[Any, Any]
    model_file_path: str
    model_name: str
    quantization: str
    provider_mode: str
    device_id: int | None
    created_at: float
    last_accessed_at: float


loaded_models: OrderedDict[str, LoadedModelEntry] = OrderedDict()


def _model_cache_max_size() -> int:
    try:
        return max(1, int(os.environ.get("DEEPDOC_MODEL_CACHE_MAX_SIZE", str(DEFAULT_MODEL_CACHE_MAX_SIZE))))
    except Exception:
        return DEFAULT_MODEL_CACHE_MAX_SIZE


def _model_cache_idle_ttl_seconds() -> int:
    try:
        return max(0, int(os.environ.get("DEEPDOC_MODEL_CACHE_IDLE_TTL_SECONDS", str(DEFAULT_MODEL_CACHE_IDLE_TTL_SECONDS))))
    except Exception:
        return DEFAULT_MODEL_CACHE_IDLE_TTL_SECONDS


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _rec_dynamic_batching_enabled() -> bool:
    return _env_flag("DEEPDOC_REC_DYNAMIC_BATCHING", default=True)


def _rec_width_bucket_step() -> int:
    try:
        return max(1, int(os.environ.get("DEEPDOC_REC_WIDTH_BUCKET_STEP", "64")))
    except Exception:
        return 64


def _ocr_version() -> str:
    version = (os.environ.get("DEEPDOC_OCR_VERSION") or "v4").strip().lower()
    if version in {"", "legacy", "v4", "ppocrv4", "pp-ocrv4"}:
        return "v4"
    if version in {"v5", "ppocrv5", "pp-ocrv5"}:
        return "v5"
    raise ValueError(f"Unsupported DEEPDOC_OCR_VERSION value: {version}")


def _ocr_model_group() -> str:
    return "core_v5" if _ocr_version() == "v5" else "core"


def _ocr_det_model_name() -> str:
    return "det_v5" if _ocr_version() == "v5" else "det"


def _ocr_rec_model_name() -> str:
    return "rec_v5" if _ocr_version() == "v5" else "rec"


def _ocr_dictionary_name(model_name: str | None = None) -> str:
    if (model_name or _ocr_rec_model_name()) == "rec_v5":
        return "ocr_v5.res"
    return "ocr.res"


def _parse_rec_image_shape(value: str, env_name: str) -> list[int]:
    try:
        shape = [int(item.strip()) for item in value.split(",")]
    except Exception as exc:
        raise ValueError(f"Invalid {env_name}: {value}") from exc
    if len(shape) != 3 or any(item <= 0 for item in shape):
        raise ValueError(f"Invalid {env_name}: {value}")
    return shape


def _ocr_rec_image_shape(model_name: str | None = None) -> list[int]:
    resolved_model_name = (model_name or _ocr_rec_model_name()).strip()
    version_env = "DEEPDOC_OCR_V5_REC_IMAGE_SHAPE" if resolved_model_name == "rec_v5" else "DEEPDOC_OCR_V4_REC_IMAGE_SHAPE"
    raw_value = os.environ.get(version_env)
    if raw_value:
        return _parse_rec_image_shape(raw_value, version_env)
    raw_value = os.environ.get("DEEPDOC_REC_IMAGE_SHAPE")
    if raw_value:
        return _parse_rec_image_shape(raw_value, "DEEPDOC_REC_IMAGE_SHAPE")
    return [int(v) for v in "3, 48, 320".split(",")]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _handwriting_fallback_enabled() -> bool:
    return _env_flag("DEEPDOC_HANDWRITING_FALLBACK", default=False)


def _handwriting_fallback_threshold() -> float:
    return max(0.0, min(1.0, _env_float("DEEPDOC_HANDWRITING_FALLBACK_THRESHOLD", 0.5)))


def _handwriting_min_score_delta() -> float:
    return max(0.0, _env_float("DEEPDOC_HANDWRITING_MIN_SCORE_DELTA", 0.0))


def _handwriting_model_name() -> str:
    return (os.environ.get("DEEPDOC_HANDWRITING_MODEL_NAME") or "rec_handwriting").strip() or "rec_handwriting"


def _model_quantization_mode() -> str:
    mode = os.environ.get("DEEPDOC_QUANT", "").strip().lower()
    if mode in {"", "0", "off", "false", "none", "fp32"}:
        return "fp32"
    if mode == "int8":
        return "int8"
    raise ValueError(f"Unsupported DEEPDOC_QUANT value: {mode}")


def _resolve_model_file_path(model_dir: str, nm: str) -> tuple[str, str]:
    quantization = _model_quantization_mode()
    if quantization == "int8":
        model_file_path = os.path.join(model_dir, nm + ".int8.onnx")
        if not os.path.exists(model_file_path):
            raise ValueError(
                "not find quantized INT8 model file path {}. "
                "Run python tools/quantize_models.py --model-dir {} first.".format(model_file_path, model_dir)
            )
        return model_file_path, quantization

    model_file_path = os.path.join(model_dir, nm + ".onnx")
    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(model_file_path))
    return model_file_path, quantization


def _close_loaded_model(entry: LoadedModelEntry) -> None:
    try:
        session = entry.model[0] if isinstance(entry.model, tuple) and entry.model else None
        close = getattr(session, "close", None)
        if callable(close):
            close()
    except Exception:
        logger.exception("Failed to close cached OCR model %s", entry.model_file_path)
    finally:
        gc.collect()


def clear_loaded_models() -> int:
    global loaded_models
    removed = 0
    for _key, entry in list(loaded_models.items()):
        _close_loaded_model(entry)
        removed += 1
    loaded_models.clear()
    return removed


def _enforce_model_cache_max_size() -> int:
    global loaded_models
    removed = 0
    max_size = _model_cache_max_size()
    while len(loaded_models) > max_size:
        _key, entry = loaded_models.popitem(last=False)
        _close_loaded_model(entry)
        removed += 1

    return removed


def prune_loaded_models(*, force: bool = False) -> int:
    global loaded_models
    removed = 0
    now = time.time()
    idle_ttl_seconds = _model_cache_idle_ttl_seconds()

    for key, entry in list(loaded_models.items()):
        if not force and (now - entry.last_accessed_at) < idle_ttl_seconds:
            continue
        loaded_models.pop(key, None)
        _close_loaded_model(entry)
        removed += 1

    removed += _enforce_model_cache_max_size()
    return removed


def model_cache_state() -> dict[str, Any]:
    now = time.time()
    return {
        "size": len(loaded_models),
        "max_size": _model_cache_max_size(),
        "idle_ttl_seconds": _model_cache_idle_ttl_seconds(),
        "keys": list(loaded_models.keys()),
        "models": [
            {
                "key": key,
                "model_file_path": entry.model_file_path,
                "model_name": entry.model_name,
                "quantization": entry.quantization,
                "provider_mode": entry.provider_mode,
                "device_id": entry.device_id,
                "age_seconds": max(0.0, now - entry.created_at),
                "idle_seconds": max(0.0, now - entry.last_accessed_at),
            }
            for key, entry in loaded_models.items()
        ],
    }


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(
        op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = getattr(operators, op_name)(**param)
        ops.append(op)
    return ops


def _onnx_provider_mode() -> str:
    mode = os.environ.get("DEEPDOC_ONNX_PROVIDER", "").strip().lower()
    if mode in {"", "auto", "gpu", "cuda"}:
        return "auto"
    if mode == "cpu":
        return "cpu"
    if mode in {"trt", "tensorrt", "tensorrt_execution_provider"}:
        return "tensorrt"
    raise ValueError(f"Unsupported DEEPDOC_ONNX_PROVIDER value: {mode}")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _available_onnx_providers() -> set[str]:
    try:
        return set(ort.get_available_providers())
    except Exception:
        return set()


def _visible_cuda_device_count_from_env() -> int:
    raw_value = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if raw_value == "":
        return 0
    if raw_value == "-1":
        return 0
    devices = [item.strip() for item in raw_value.split(",") if item.strip()]
    return len(devices)


def ensure_parallel_devices_configured() -> int:
    configured = max(0, int(getattr(settings, "PARALLEL_DEVICES", 0) or 0))
    if configured > 0:
        return configured
    available_providers = _available_onnx_providers()
    if "CUDAExecutionProvider" not in available_providers:
        return configured

    detected = 0
    try:
        pip_install_torch()
        import torch

        if torch.cuda.is_available():
            detected = max(0, int(torch.cuda.device_count() or 0))
    except Exception:
        detected = 0

    if detected <= 0:
        detected = _visible_cuda_device_count_from_env()
    if detected <= 0:
        detected = 1

    settings.PARALLEL_DEVICES = detected
    logger.info("Auto-configured PARALLEL_DEVICES=%s for CUDA OCR workers", detected)
    return detected


def _cuda_is_available(quantization: str, device_id: int | None) -> bool:
    if quantization == "int8":
        return False
    available_providers = _available_onnx_providers()
    if "CUDAExecutionProvider" not in available_providers:
        return False
    try:
        pip_install_torch()
        import torch

        target_id = 0 if device_id is None else device_id
        if torch.cuda.is_available():
            return torch.cuda.device_count() > target_id
    except Exception:
        pass
    return True


def _cuda_provider_options(device_id: int | None) -> dict[str, Any]:
    gpu_mem_limit_mb = _env_int("OCR_GPU_MEM_LIMIT_MB", 2048)
    arena_strategy = os.environ.get("OCR_ARENA_EXTEND_STRATEGY", "kNextPowerOfTwo")
    provider_device_id = 0 if device_id is None else device_id
    return {
        "device_id": provider_device_id,
        "gpu_mem_limit": max(gpu_mem_limit_mb, 0) * 1024 * 1024,
        "arena_extend_strategy": arena_strategy,
    }


def _tensorrt_provider_options(device_id: int | None) -> dict[str, Any]:
    cache_dir = os.path.abspath(
        os.environ.get(
            "DEEPDOC_TENSORRT_CACHE_DIR",
            os.path.join(settings.WORK_DIR, "tensorrt_engine_cache"),
        )
    )
    os.makedirs(cache_dir, exist_ok=True)
    options: dict[str, Any] = {
        "device_id": 0 if device_id is None else device_id,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_fp16_enable": _env_flag("DEEPDOC_TENSORRT_FP16", default=False),
    }
    max_workspace_size = _env_int("DEEPDOC_TENSORRT_MAX_WORKSPACE_SIZE", 1 << 30)
    if max_workspace_size > 0:
        options["trt_max_workspace_size"] = max_workspace_size
    return options


def _build_onnx_providers(quantization: str, device_id: int | None) -> tuple[str, list[str], list[dict[str, Any]]]:
    provider_mode = _onnx_provider_mode()
    if quantization == "int8":
        return "cpu", ["CPUExecutionProvider"], []
    if provider_mode == "cpu":
        return "cpu", ["CPUExecutionProvider"], []

    cuda_available = _cuda_is_available(quantization, device_id)
    if provider_mode == "tensorrt":
        available_providers = _available_onnx_providers()
        missing = [
            provider
            for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
            if provider not in available_providers
        ]
        if missing:
            raise RuntimeError(
                "DEEPDOC_ONNX_PROVIDER=tensorrt requires ONNX Runtime providers: {}".format(
                    ", ".join(missing)
                )
            )
        if not cuda_available:
            raise RuntimeError("DEEPDOC_ONNX_PROVIDER=tensorrt requires an available CUDA device")
        return (
            "tensorrt",
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            [_tensorrt_provider_options(device_id), _cuda_provider_options(device_id), {}],
        )

    if cuda_available:
        return "cuda", ["CUDAExecutionProvider"], [_cuda_provider_options(device_id)]
    return "cpu", ["CPUExecutionProvider"], []


def load_model(model_dir, nm, device_id: int | None = None):
    model_file_path, quantization = _resolve_model_file_path(model_dir, nm)
    requested_provider_mode = _onnx_provider_mode()
    model_cached_tag = "|".join(
        [
            model_file_path,
            quantization,
            requested_provider_mode,
            str(device_id) if device_id is not None else "",
        ]
    )

    global loaded_models
    prune_loaded_models(force=False)
    loaded_entry = loaded_models.get(model_cached_tag)
    if loaded_entry:
        loaded_entry.last_accessed_at = time.time()
        loaded_models.move_to_end(model_cached_tag)
        logger.info(f"load_model {model_file_path} reuses cached model")
        return loaded_entry.model

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Prevent CPU oversubscription by allowing explicit thread control in multi-worker environments
    options.intra_op_num_threads = int(os.environ.get("OCR_INTRA_OP_NUM_THREADS", "2"))
    options.inter_op_num_threads = int(os.environ.get("OCR_INTER_OP_NUM_THREADS", "2"))

    # https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
    # Shrink GPU memory after execution
    run_options = ort.RunOptions()
    provider_mode, providers, provider_options = _build_onnx_providers(quantization, device_id)
    if provider_options:
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=providers,
            provider_options=provider_options,
        )
    else:
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=providers,
        )

    if provider_mode in {"cuda", "tensorrt"}:
        cuda_provider_options = (
            provider_options[1] if provider_mode == "tensorrt" else provider_options[0]
        )
        provider_device_id = int(cuda_provider_options["device_id"])
        # Explicit arena shrinkage for GPU to release VRAM back to the system after each run
        if os.environ.get("OCR_GPUMEM_ARENA_SHRINKAGE") == "1":
            run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{provider_device_id}")
            logger.info(
                f"load_model {model_file_path} enabled GPU memory arena shrinkage on device {provider_device_id}")
        if provider_mode == "tensorrt":
            logger.info(
                "load_model %s uses TensorRT (device %s, engine_cache=%s) with CUDA fallback",
                model_file_path,
                provider_device_id,
                provider_options[0].get("trt_engine_cache_path"),
            )
        else:
            logger.info(
                f"load_model {model_file_path} uses GPU (device {provider_device_id}, gpu_mem_limit={cuda_provider_options['gpu_mem_limit']}, arena_strategy={cuda_provider_options['arena_extend_strategy']})"
            )
    else:
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
        logger.info(f"load_model {model_file_path} uses CPU")
    loaded_model = (sess, run_options)
    now = time.time()
    loaded_models[model_cached_tag] = LoadedModelEntry(
        model=loaded_model,
        model_file_path=model_file_path,
        model_name=nm,
        quantization=quantization,
        provider_mode=provider_mode,
        device_id=device_id,
        created_at=now,
        last_accessed_at=now,
    )
    _enforce_model_cache_max_size()
    return loaded_model


class TextRecognizer:
    def __init__(
        self,
        model_dir,
        device_id: int | None = None,
        model_name: str | None = None,
        enable_handwriting_fallback: bool = True,
    ):
        self.model_dir = model_dir
        self.device_id = device_id
        self.model_name = (model_name or _ocr_rec_model_name()).strip() or _ocr_rec_model_name()
        self.enable_handwriting_fallback = enable_handwriting_fallback
        self.handwriting_recognizer = None
        self.rec_image_shape = _ocr_rec_image_shape(self.model_name)
        self.rec_batch_num = 16
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": os.path.join(model_dir, _ocr_dictionary_name(self.model_name)),
            "use_space_char": True
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.run_options = load_model(model_dir, self.model_name, device_id)
        self.input_tensor = self.predictor.get_inputs()[0]

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        w = self.input_tensor.shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_vl(self, img, image_shape):

        imgC, imgH, imgW = image_shape
        img = img[:, :, ::-1]  # bgr2rgb
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        return resized_image

    def resize_norm_img_srn(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        img_black = np.zeros((imgH, imgW))
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        img_np = np.asarray(img_new)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_black[:, 0:img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]

        row, col, c = img_black.shape
        c = 1

        return np.reshape(img_black, (c, row, col)).astype(np.float32)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length):

        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64')
        gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
            (max_text_length, 1)).astype('int64')

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias1 = np.tile(
            gsrm_slf_attn_bias1,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias2 = np.tile(
            gsrm_slf_attn_bias2,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2
        ]

    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]

        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            self.srn_other_inputs(image_shape, num_heads, max_text_length)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
        encoder_word_pos = encoder_word_pos.astype(np.int64)
        gsrm_word_pos = gsrm_word_pos.astype(np.int64)

        return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2)

    def resize_norm_img_sar(self, img, image_shape,
                            width_downsample_ratio=0.25):
        imgC, imgH, imgW_min, imgW_max = image_shape
        h = img.shape[0]
        w = img.shape[1]
        valid_ratio = 1.0
        # make sure new_width is an integral multiple of width_divisor.
        width_divisor = int(1 / width_downsample_ratio)
        # resize
        ratio = w / float(h)
        resize_w = math.ceil(imgH * ratio)
        if resize_w % width_divisor != 0:
            resize_w = round(resize_w / width_divisor) * width_divisor
        if imgW_min is not None:
            resize_w = max(imgW_min, resize_w)
        if imgW_max is not None:
            valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
            resize_w = min(imgW_max, resize_w)
        resized_image = cv2.resize(img, (resize_w, imgH))
        resized_image = resized_image.astype('float32')
        # norm
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        resize_shape = resized_image.shape
        padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im[:, :, 0:resize_w] = resized_image
        pad_shape = padding_im.shape

        return padding_im, resize_shape, pad_shape, valid_ratio

    def resize_norm_img_spin(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # return padding_im
        img = cv2.resize(img, tuple([100, 32]), cv2.INTER_CUBIC)
        img = np.array(img, np.float32)
        img = np.expand_dims(img, -1)
        img = img.transpose((2, 0, 1))
        mean = [127.5]
        std = [127.5]
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        mean = np.float32(mean.reshape(1, -1))
        stdinv = 1 / np.float32(std.reshape(1, -1))
        img -= mean
        img *= stdinv
        return img

    def resize_norm_img_svtr(self, img, image_shape):

        imgC, imgH, imgW = image_shape
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

    def resize_norm_img_abinet(self, img, image_shape):

        imgC, imgH, imgW = image_shape

        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        resized_image = (
            resized_image - mean[None, None, ...]) / std[None, None, ...]
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')

        return resized_image

    def norm_img_can(self, img, image_shape):

        img = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY)  # CAN only predict gray scale image

        if self.rec_image_shape[0] == 1:
            h, w = img.shape
            _, imgH, imgW = self.rec_image_shape
            if h < imgH or w < imgW:
                padding_h = max(imgH - h, 0)
                padding_w = max(imgW - w, 0)
                img_padded = np.pad(img, ((0, padding_h), (0, padding_w)),
                                    'constant',
                                    constant_values=(255))
                img = img_padded

        img = np.expand_dims(img, 0) / 255.0  # h,w,c -> c,h,w
        img = img.astype('float32')

        return img

    def _has_dynamic_input_width(self) -> bool:
        try:
            input_width = self.input_tensor.shape[3:][0]
        except Exception:
            return False
        if isinstance(input_width, str):
            return True
        return input_width is None or input_width <= 0

    def _rec_target_width_bucket(self, wh_ratio: float) -> int:
        _imgC, imgH, imgW = self.rec_image_shape[:3]
        target_width = max(imgW, int(math.ceil(imgH * wh_ratio)))
        bucket_step = _rec_width_bucket_step()
        return int(math.ceil(target_width / bucket_step) * bucket_step)

    def _iter_recognition_batches(self, indices, width_list):
        batch_num = self.rec_batch_num
        imgC, imgH, imgW = self.rec_image_shape[:3]
        base_wh_ratio = imgW / imgH

        if not (_rec_dynamic_batching_enabled() and self._has_dynamic_input_width()):
            for beg_img_no in range(0, len(indices), batch_num):
                end_img_no = min(len(indices), beg_img_no + batch_num)
                batch_indices = [int(indices[ino]) for ino in range(beg_img_no, end_img_no)]
                max_wh_ratio = base_wh_ratio
                for img_index in batch_indices:
                    max_wh_ratio = max(max_wh_ratio, width_list[img_index])
                yield batch_indices, max_wh_ratio
            return

        current_batch: list[int] = []
        current_target_width: int | None = None
        for sorted_pos in range(len(indices)):
            img_index = int(indices[sorted_pos])
            target_width = self._rec_target_width_bucket(width_list[img_index])
            if current_batch and (target_width != current_target_width or len(current_batch) >= batch_num):
                yield current_batch, (current_target_width or imgW) / imgH
                current_batch = []
            current_batch.append(img_index)
            current_target_width = target_width
        if current_batch:
            yield current_batch, (current_target_width or imgW) / imgH

    def _get_handwriting_recognizer(self):
        if self.handwriting_recognizer is not None:
            return self.handwriting_recognizer
        model_name = _handwriting_model_name()
        if model_name == self.model_name:
            raise ValueError("DEEPDOC_HANDWRITING_MODEL_NAME must differ from the primary OCR rec model")
        self.handwriting_recognizer = TextRecognizer(
            self.model_dir,
            self.device_id,
            model_name=model_name,
            enable_handwriting_fallback=False,
        )
        return self.handwriting_recognizer

    def _apply_handwriting_fallback(self, img_list, rec_res):
        if not getattr(self, "enable_handwriting_fallback", True) or not _handwriting_fallback_enabled():
            return rec_res
        threshold = _handwriting_fallback_threshold()
        candidate_indices = []
        candidate_images = []
        for idx, result in enumerate(rec_res):
            score = float(result[1]) if len(result) > 1 else 0.0
            if score < threshold:
                candidate_indices.append(idx)
                candidate_images.append(img_list[idx])
        if not candidate_images:
            return rec_res

        handwriting_recognizer = self._get_handwriting_recognizer()
        handwriting_res, _elapsed = handwriting_recognizer(candidate_images)
        min_delta = _handwriting_min_score_delta()
        updated = [list(item) for item in rec_res]
        for result_idx, handwriting_result in zip(candidate_indices, handwriting_res):
            if not handwriting_result:
                continue
            original_score = float(rec_res[result_idx][1]) if len(rec_res[result_idx]) > 1 else 0.0
            handwriting_text = str(handwriting_result[0] or "")
            handwriting_score = float(handwriting_result[1]) if len(handwriting_result) > 1 else 0.0
            if handwriting_text.strip() and handwriting_score >= original_score + min_delta:
                updated[result_idx] = [handwriting_text, handwriting_score]
        return updated

    def close(self):
        # close session and release manually
        logger.info('Close text recognizer.')
        handwriting_recognizer = getattr(self, "handwriting_recognizer", None)
        if handwriting_recognizer is not None:
            close = getattr(handwriting_recognizer, "close", None)
            if callable(close):
                close()
            self.handwriting_recognizer = None
        if hasattr(self, "predictor"):
            del self.predictor
        gc.collect()

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        st = time.time()

        for batch_indices, max_wh_ratio in self._iter_recognition_batches(indices, width_list):
            norm_img_batch = []
            for img_index in batch_indices:
                norm_img = self.resize_norm_img(img_list[img_index], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = {}
            input_dict[self.input_tensor.name] = norm_img_batch
            for i in range(100000):
                try:
                    outputs = self.predictor.run(None, input_dict, self.run_options)
                    break
                except Exception as e:
                    if i >= 3:
                        raise e
                    time.sleep(5)
            preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[batch_indices[rno]] = rec_result[rno]

        rec_res = self._apply_handwriting_fallback(img_list, rec_res)
        return rec_res, time.time() - st

    def __del__(self):
        self.close()


class TextDetector:
    def __init__(self, model_dir, device_id: int | None = None):
        self.model_dir = model_dir
        self.device_id = device_id
        self.model_name = _ocr_det_model_name()
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': "max",
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {"name": "DBPostProcess", "thresh": 0.3, "box_thresh": 0.5, "max_candidates": 1000,
                              "unclip_ratio": 1.5, "use_dilation": False, "score_mode": "fast", "box_type": "quad"}

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.run_options = load_model(model_dir, self.model_name, device_id)
        self.input_tensor = self.predictor.get_inputs()[0]

        img_h, img_w = self.input_tensor.shape[2:]
        if isinstance(img_h, str) or isinstance(img_w, str):
            pass
        elif img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'image_shape': [img_h, img_w]
                }
            }
        self.preprocess_op = create_operators(pre_process_list)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def close(self):
        logger.info("Close text detector.")
        if hasattr(self, "predictor"):
            del self.predictor
        gc.collect()

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        input_dict = {}
        input_dict[self.input_tensor.name] = img
        for i in range(100000):
            try:
                outputs = self.predictor.run(None, input_dict, self.run_options)
                break
            except Exception as e:
                if i >= 3:
                    raise e
                time.sleep(5)

        post_result = self.postprocess_op({"maps": outputs[0]}, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        return dt_boxes, time.time() - st

    def __del__(self):
        self.close()


class OCR:
    def __init__(self, model_dir=None):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """
        if not model_dir:
            model_dir = ensure_groups(_ocr_model_group())

        ensure_parallel_devices_configured()
        if settings.PARALLEL_DEVICES > 0:
            self.text_detector = []
            self.text_recognizer = []
            for device_id in range(settings.PARALLEL_DEVICES):
                self.text_detector.append(TextDetector(model_dir, device_id))
                self.text_recognizer.append(TextRecognizer(model_dir, device_id))
        else:
            self.text_detector = [TextDetector(model_dir)]
            self.text_recognizer = [TextRecognizer(model_dir)]

        self.drop_score = 0.5
        self.crop_image_res_index = 0

    def get_rotate_crop_image(self, img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            # Try original orientation
            rec_result = self.text_recognizer[0]([dst_img])
            text, score = rec_result[0][0]
            best_score = score
            best_img = dst_img

            # Try clockwise 90° rotation
            rotated_cw = np.rot90(dst_img, k=3)
            rec_result = self.text_recognizer[0]([rotated_cw])
            rotated_cw_text, rotated_cw_score = rec_result[0][0]
            if rotated_cw_score > best_score:
                best_score = rotated_cw_score
                best_img = rotated_cw

            # Try counter-clockwise 90° rotation
            rotated_ccw = np.rot90(dst_img, k=1)
            rec_result = self.text_recognizer[0]([rotated_ccw])
            rotated_ccw_text, rotated_ccw_score = rec_result[0][0]
            if rotated_ccw_score > best_score:
                best_img = rotated_ccw

            # Use the best image
            dst_img = best_img
        return dst_img

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def detect(self, img, device_id: int | None = None):
        if device_id is None:
            device_id = 0

        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            return None, None, time_dict

        start = time.time()
        dt_boxes, elapse = self.text_detector[device_id](img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        return zip(self.sorted_boxes(dt_boxes), [
                   ("", 0) for _ in range(len(dt_boxes))])

    def recognize(self, ori_im, box, device_id: int | None = None):
        if device_id is None:
            device_id = 0

        img_crop = self.get_rotate_crop_image(ori_im, box)

        rec_res, elapse = self.text_recognizer[device_id]([img_crop])
        text, score = rec_res[0]
        if score < self.drop_score:
            return ""
        return text

    def recognize_batch(self, img_list, device_id: int | None = None):
        if device_id is None:
            device_id = 0
        rec_res, elapse = self.text_recognizer[device_id](img_list)
        texts = []
        for i in range(len(rec_res)):
            text, score = rec_res[i]
            if score < self.drop_score:
                text = ""
            texts.append(text)
        return texts

    def __call__(self, img, device_id = 0, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        if device_id is None:
            device_id = 0

        if img is None:
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector[device_id](img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, elapse = self.text_recognizer[device_id](img_crop_list)

        time_dict['rec'] = elapse

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start

        # for bno in range(len(img_crop_list)):
        #    print(f"{bno}, {rec_res[bno]}")

        return list(zip([a.tolist() for a in filter_boxes], filter_rec_res))

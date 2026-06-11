#!/usr/bin/env python3
# ruff: noqa: E402

"""Generate INT8 ONNX models for DocPilot ONNX inference."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import logger
from onnxruntime.quantization import QuantType, quantize_dynamic


HIGH_RISK_INT8_MODEL_NAMES = {"rec.onnx", "rec_v5.onnx", "decoder.onnx"}


@dataclass(frozen=True)
class QuantizationResult:
    input_path: Path
    output_path: Path
    input_size_bytes: int
    output_size_bytes: int


def default_int8_output_path(model_path: Path) -> Path:
    if model_path.suffix != ".onnx":
        raise ValueError(f"Expected an .onnx model file, got {model_path}")
    if model_path.name.endswith(".int8.onnx"):
        raise ValueError(f"Refusing to quantize an already INT8-named model: {model_path}")
    return model_path.with_name(model_path.name.removesuffix(".onnx") + ".int8.onnx")


def quantize_model_file(input_path: str | Path, output_path: str | Path | None = None, *, overwrite: bool = False) -> QuantizationResult:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Model file does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Model path is not a file: {source}")

    target = Path(output_path) if output_path is not None else default_int8_output_path(source)
    if target.exists() and not overwrite:
        raise FileExistsError(f"INT8 model already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        str(source),
        str(target),
        weight_type=QuantType.QInt8,
    )
    return QuantizationResult(
        input_path=source,
        output_path=target,
        input_size_bytes=source.stat().st_size,
        output_size_bytes=target.stat().st_size,
    )


def is_high_risk_int8_model(model_path: str | Path) -> bool:
    return Path(model_path).name in HIGH_RISK_INT8_MODEL_NAMES


def discover_model_files(model_dir: str | Path, *, include_high_risk: bool = False) -> list[Path]:
    root = Path(model_dir)
    if not root.exists():
        raise FileNotFoundError(f"Model directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Model path is not a directory: {root}")
    return sorted(
        path
        for path in root.rglob("*.onnx")
        if not path.name.endswith(".int8.onnx")
        and (include_high_risk or not is_high_risk_int8_model(path))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate .int8.onnx models with ONNX Runtime dynamic quantization.")
    parser.add_argument(
        "--model-dir",
        help=(
            "Directory containing .onnx models. Recursively quantizes every non-.int8 .onnx file "
            "in subdirectories, skipping risky sequence/decoder models by default."
        ),
    )
    parser.add_argument("--model", action="append", default=[], help="Specific .onnx model file to quantize. Can be passed multiple times.")
    parser.add_argument("--output", help="Output path. Only valid when exactly one --model is provided.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .int8.onnx files.")
    parser.add_argument(
        "--include-risky-sequence-models",
        action="store_true",
        help=(
            "Include OCR recognition and formula decoder sequence models during --model-dir discovery. "
            "Use only after module-level accuracy calibration."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_paths = [Path(path) for path in args.model]
    if args.model_dir:
        model_paths.extend(
            discover_model_files(
                args.model_dir,
                include_high_risk=args.include_risky_sequence_models,
            )
        )
    model_paths = sorted(set(model_paths))

    if not model_paths:
        raise SystemExit("No model files provided. Use --model-dir or --model.")
    if args.output and len(model_paths) != 1:
        raise SystemExit("--output is only valid when exactly one --model is provided.")

    for model_path in model_paths:
        output_path = Path(args.output) if args.output else None
        result = quantize_model_file(model_path, output_path, overwrite=args.overwrite)
        ratio = result.output_size_bytes / result.input_size_bytes if result.input_size_bytes else 0.0
        logger.info(
            "Quantized %s -> %s (%s -> %s bytes, ratio=%.3f)",
            result.input_path,
            result.output_path,
            result.input_size_bytes,
            result.output_size_bytes,
            ratio,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

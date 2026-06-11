#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


CORE_MODEL_FILES = (
    "det.onnx",
    "rec.onnx",
    "ocr.res",
    "layout.onnx",
    "layout.manual.onnx",
    "layout.paper.onnx",
    "layout.laws.onnx",
    "tsr.onnx",
    "updown_concat_xgb.model",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create lightweight placeholder core model files for CI smoke runs.")
    parser.add_argument("output_dir", help="Directory that will hold placeholder core model files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in CORE_MODEL_FILES:
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"placeholder\n")

    print(f"created placeholder models under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

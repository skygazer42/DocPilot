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
"""冒烟脚本：表格区域截图 + 占位 OCR 框 -> 非空 HTML <table>。

仅验证 RapidTableRecognizer 管线连通，不验证识别质量
（真实质量在 PDF 端到端 + tools/eval_table.py 的 TEDS 基准里评估）。

前置：pip install -e ".[table]" 且已 download_models.py table_v2。
用法：python deepdoc/vision/t_table.py --image /path/to/table_crop.png
"""
import argparse

from PIL import Image

from deepdoc.vision.rapid_table_recognizer import RapidTableRecognizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="表格区域截图路径")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    # 占位框：用整图当一个文本框，仅验证管线连通；真实对齐在 PDF 端到端校验。
    boxes = [{"text": "smoke", "x0": 0, "x1": w, "top": 0, "bottom": h, "score": 1.0}]

    rec = RapidTableRecognizer()
    html = rec(img, boxes, crop_origin=(0, 0), zoomin=1)
    print(html)
    assert "<table" in html, "expected an HTML table in output"
    print("[t_table] OK")


if __name__ == "__main__":
    main()

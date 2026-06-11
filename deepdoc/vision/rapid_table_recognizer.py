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
"""RapidTable(SLANet-plus, 纯 ONNX/CPU) 表格识别封装：表格图 + OCR 框 -> HTML。

⚠️ 未验证代码：以下 RapidTable 调用基于其常见公开用法编写，尚未在本环境联调。
   集成时务必按实际安装的 rapid-table 版本核验（见 plan Task 0 探针）：
   - 入口类名与构造参数（如何选 slanet_plus 的 ONNX 模型、如何指定本地 model_path）
   - __call__/run 的签名（是否接受 ocr_result 及其格式、是位置参数还是关键字）
   - 返回值结构（HTML 字段名，或是否返回 (html, cell_bboxes, elapse) 元组）
   - 输入图像通道（BGR vs RGB）
"""
import os

import numpy as np

from common import logger
from common.model_store import ensure_groups


class RapidTableRecognizer:
    """表格图 + OCR 框 -> HTML <table>。任何失败都应由调用方回退到 TATR。"""

    def __init__(self):
        try:
            # NOTE: 类名/入参以实际 rapid-table 版本为准（plan Task 0）。
            from rapid_table import RapidTable, RapidTableInput
        except ImportError as exc:
            raise ImportError(
                "rapid-table 未安装。安装可选依赖以启用 DEEPDOC_TABLE_ENGINE=rapidtable："
                ' pip install -e ".[table]"'
            ) from exc

        model_root = ensure_groups("table_v2")
        model_path = os.path.join(model_root, "table", "slanet_plus.onnx")
        self._engine = RapidTable(
            RapidTableInput(model_type="slanet_plus", model_path=model_path)
        )
        logger.info("RapidTableRecognizer initialized (slanet_plus, %s)", model_path)

    @staticmethod
    def _to_local_ocr_result(ocr_boxes, crop_origin, zoomin):
        """页面累积坐标的 box -> 相对裁剪表格图的局部像素四点框。

        crop_origin=(left, top) 为裁剪原点（页面局部坐标，未乘 ZM）。
        返回 RapidOCR 风格 [[ [x0,y0],[x1,y0],[x1,y1],[x0,y1] ], text, score]，
        具体字段顺序以 rapid-table 版本为准。
        """
        ox, oy = crop_origin
        result = []
        for b in ocr_boxes:
            text = (b.get("text") or "").strip()
            if not text:
                continue
            x0 = float((b["x0"] - ox) * zoomin)
            x1 = float((b["x1"] - ox) * zoomin)
            y0 = float((b["top"] - oy) * zoomin)
            y1 = float((b["bottom"] - oy) * zoomin)
            poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            result.append([poly, text, float(b.get("score", 1.0))])
        return result

    @staticmethod
    def _extract_html(output):
        """从 RapidTable 返回值取 HTML，兼容对象、dict、元组和字符串形态。"""
        html = getattr(output, "pred_html", None)
        if html:
            return html
        if isinstance(output, dict):
            for key in ("pred_html", "html", "table_html"):
                html = output.get(key)
                if html:
                    return str(html)
        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            if isinstance(first, str):
                return first
        if isinstance(output, str):
            return output
        return ""

    def __call__(self, table_image, ocr_boxes, crop_origin=(0, 0), zoomin=1):
        # NOTE: 若 RapidTable 期望 BGR，可在此 cv2.cvtColor(..., COLOR_RGB2BGR)。
        img = (
            np.array(table_image)
            if not isinstance(table_image, np.ndarray)
            else table_image
        )
        ocr_result = self._to_local_ocr_result(ocr_boxes, crop_origin, zoomin)
        if not ocr_result:
            return ""
        try:
            output = self._engine(img, ocr_result=ocr_result)
        except TypeError:
            try:
                output = self._engine(img, ocr_results=ocr_result)
            except TypeError:
                output = self._engine(img, ocr_result)
        return self._extract_html(output) or ""

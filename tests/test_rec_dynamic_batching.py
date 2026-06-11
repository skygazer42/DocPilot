import os
import unittest
from unittest.mock import patch

import numpy as np


class FakeInputTensor:
    name = "x"
    shape = [None, 3, 48, "width"]


class FakeStaticInputTensor:
    name = "x"
    shape = [None, 3, 48, 320]


class FakePredictor:
    def __init__(self):
        self.batch_shapes: list[tuple[int, ...]] = []

    def run(self, _output_names, input_dict, _run_options):
        batch = input_dict["x"]
        self.batch_shapes.append(tuple(batch.shape))
        return [batch[:, 0, 0, 0].copy()]


class FakePostprocess:
    def __call__(self, preds):
        return [[f"img-{int(value)}", 0.99] for value in preds]


def _make_image(image_id: int, height: int, width: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, :] = image_id
    return image


class TextRecognizerDynamicBatchingTest(unittest.TestCase):
    def test_dynamic_batching_groups_recognition_by_width_bucket_and_preserves_order(self):
        from deepdoc.vision.ocr import TextRecognizer

        recognizer = object.__new__(TextRecognizer)
        recognizer.rec_image_shape = [3, 48, 320]
        recognizer.rec_batch_num = 4
        recognizer.input_tensor = FakeInputTensor()
        recognizer.predictor = FakePredictor()
        recognizer.run_options = None
        recognizer.postprocess_op = FakePostprocess()

        def fake_resize_norm_img(img, max_wh_ratio):
            target_width = int(round(48 * max_wh_ratio))
            norm_img = np.zeros((3, 48, target_width), dtype=np.float32)
            norm_img[0, 0, 0] = float(img[0, 0, 0])
            return norm_img

        recognizer.resize_norm_img = fake_resize_norm_img
        images = [
            _make_image(1, 10, 90),  # ratio 9.0 -> bucket 448
            _make_image(2, 10, 10),  # ratio 1.0 -> bucket 320
            _make_image(3, 10, 80),  # ratio 8.0 -> bucket 384
            _make_image(4, 10, 70),  # ratio 7.0 -> bucket 384
        ]

        with patch.dict(os.environ, {"DEEPDOC_REC_DYNAMIC_BATCHING": "1", "DEEPDOC_REC_WIDTH_BUCKET_STEP": "64"}):
            rec_res, _elapsed = recognizer(images)

        self.assertEqual([["img-1", 0.99], ["img-2", 0.99], ["img-3", 0.99], ["img-4", 0.99]], rec_res)
        self.assertEqual([(1, 3, 48, 320), (2, 3, 48, 384), (1, 3, 48, 448)], recognizer.predictor.batch_shapes)

    def test_static_recognition_width_keeps_fixed_batching(self):
        from deepdoc.vision.ocr import TextRecognizer

        recognizer = object.__new__(TextRecognizer)
        recognizer.rec_image_shape = [3, 48, 320]
        recognizer.rec_batch_num = 4
        recognizer.input_tensor = FakeStaticInputTensor()
        recognizer.predictor = FakePredictor()
        recognizer.run_options = None
        recognizer.postprocess_op = FakePostprocess()

        def fake_resize_norm_img(img, max_wh_ratio):
            norm_img = np.zeros((3, 48, 320), dtype=np.float32)
            norm_img[0, 0, 0] = float(img[0, 0, 0])
            return norm_img

        recognizer.resize_norm_img = fake_resize_norm_img
        images = [
            _make_image(1, 10, 90),
            _make_image(2, 10, 10),
            _make_image(3, 10, 80),
            _make_image(4, 10, 70),
        ]

        with patch.dict(os.environ, {"DEEPDOC_REC_DYNAMIC_BATCHING": "1", "DEEPDOC_REC_WIDTH_BUCKET_STEP": "64"}):
            rec_res, _elapsed = recognizer(images)

        self.assertEqual([["img-1", 0.99], ["img-2", 0.99], ["img-3", 0.99], ["img-4", 0.99]], rec_res)
        self.assertEqual([(4, 3, 48, 320)], recognizer.predictor.batch_shapes)


if __name__ == "__main__":
    unittest.main()

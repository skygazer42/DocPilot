import unittest

import numpy as np


class FakeOrtSession:
    def __init__(self):
        self.calls: list[dict[str, np.ndarray]] = []

    def run(self, _output_names, input_dict, _run_options):
        self.calls.append(input_dict)
        batch = input_dict["image"]
        outputs = np.zeros((batch.shape[0], 1, 1), dtype=np.float32)
        for batch_index in range(batch.shape[0]):
            outputs[batch_index, 0, 0] = batch[batch_index, 0, 0, 0]
        return [outputs]


class FakeStaticBatchOrtSession:
    def __init__(self):
        self.calls: list[dict[str, np.ndarray]] = []

    def run(self, _output_names, input_dict, _run_options):
        self.calls.append(input_dict)
        batch = input_dict["image"]
        if batch.shape[0] != 1:
            raise RuntimeError("static batch-1 model rejects multi-page inputs")
        outputs = np.zeros((1, 1, 1), dtype=np.float32)
        outputs[0, 0, 0] = batch[0, 0, 0, 0]
        return [outputs]


class RecognizerLayoutBatchingTest(unittest.TestCase):
    def test_recognizer_runs_one_onnx_call_per_layout_batch_and_preserves_page_order(self):
        from deepdoc.vision.recognizer import Recognizer

        recognizer = object.__new__(Recognizer)
        recognizer.ort_sess = FakeOrtSession()
        recognizer.run_options = None
        recognizer.input_names = ["image"]
        recognizer.label_list = ["text"]

        def fake_preprocess(image_list):
            inputs = []
            for page_id, _image in enumerate(image_list, start=1):
                image = np.zeros((1, 3, 8, 8), dtype=np.float32)
                image[0, 0, 0, 0] = page_id
                inputs.append({"image": image, "scale_factor": [1.0, 1.0]})
            return inputs

        def fake_postprocess(boxes, inputs, _thr):
            return [{"page_id": int(boxes[0, 0]), "scale_factor": inputs["scale_factor"]}]

        recognizer.preprocess = fake_preprocess
        recognizer.postprocess = fake_postprocess

        results = recognizer([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)], thr=0.2, batch_size=3)

        self.assertEqual([[{"page_id": 1, "scale_factor": [1.0, 1.0]}],
                          [{"page_id": 2, "scale_factor": [1.0, 1.0]}],
                          [{"page_id": 3, "scale_factor": [1.0, 1.0]}]], results)
        self.assertEqual(1, len(recognizer.ort_sess.calls))
        self.assertEqual((3, 3, 8, 8), recognizer.ort_sess.calls[0]["image"].shape)

    def test_recognizer_skips_batched_attempt_for_static_batch_one_models(self):
        from deepdoc.vision.recognizer import Recognizer

        recognizer = object.__new__(Recognizer)
        recognizer.ort_sess = FakeStaticBatchOrtSession()
        recognizer.run_options = None
        recognizer.input_names = ["image"]
        recognizer.label_list = ["text"]
        recognizer._supports_batched_onnx_inference = False

        def fake_preprocess(image_list):
            inputs = []
            for page_id, _image in enumerate(image_list, start=1):
                image = np.zeros((1, 3, 8, 8), dtype=np.float32)
                image[0, 0, 0, 0] = page_id
                inputs.append({"image": image, "scale_factor": [1.0, 1.0]})
            return inputs

        def fake_postprocess(boxes, inputs, _thr):
            return [{"page_id": int(boxes[0, 0]), "scale_factor": inputs["scale_factor"]}]

        recognizer.preprocess = fake_preprocess
        recognizer.postprocess = fake_postprocess

        results = recognizer([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)], thr=0.2, batch_size=3)

        self.assertEqual(3, len(recognizer.ort_sess.calls))
        self.assertTrue(all(call["image"].shape == (1, 3, 8, 8) for call in recognizer.ort_sess.calls))
        self.assertEqual(1, results[0][0]["page_id"])
        self.assertEqual(3, results[2][0]["page_id"])


if __name__ == "__main__":
    unittest.main()

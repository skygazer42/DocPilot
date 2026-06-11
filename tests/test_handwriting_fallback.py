import os
import unittest
from unittest.mock import patch

import numpy as np


def _image(value: int) -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, :] = value
    return image


class FakeHandwritingRecognizer:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def __call__(self, img_list):
        self.calls.append(list(img_list))
        return self.results, 0.01


class HandwritingFallbackTest(unittest.TestCase):
    def test_handwriting_fallback_is_disabled_by_default(self):
        from deepdoc.vision.ocr import TextRecognizer

        recognizer = object.__new__(TextRecognizer)
        recognizer.handwriting_recognizer = FakeHandwritingRecognizer([["hand", 0.95]])
        recognizer._get_handwriting_recognizer = lambda: recognizer.handwriting_recognizer

        original = [["low", 0.2]]
        with patch.dict(os.environ, {}, clear=True):
            updated = recognizer._apply_handwriting_fallback([_image(1)], original)

        self.assertEqual(original, updated)
        self.assertEqual([], recognizer.handwriting_recognizer.calls)

    def test_handwriting_fallback_replaces_only_low_confidence_rows_when_better(self):
        from deepdoc.vision.ocr import TextRecognizer

        recognizer = object.__new__(TextRecognizer)
        recognizer.handwriting_recognizer = FakeHandwritingRecognizer(
            [["handwritten", 0.88], ["worse", 0.31]]
        )
        recognizer._get_handwriting_recognizer = lambda: recognizer.handwriting_recognizer
        images = [_image(1), _image(2), _image(3)]
        original = [["printed", 0.92], ["uncertain", 0.42], ["bad", 0.35]]

        with patch.dict(
            os.environ,
            {
                "DEEPDOC_HANDWRITING_FALLBACK": "1",
                "DEEPDOC_HANDWRITING_FALLBACK_THRESHOLD": "0.6",
                "DEEPDOC_HANDWRITING_MIN_SCORE_DELTA": "0.05",
            },
            clear=False,
        ):
            updated = recognizer._apply_handwriting_fallback(images, original)

        self.assertEqual([["printed", 0.92], ["handwritten", 0.88], ["bad", 0.35]], updated)
        self.assertEqual(1, len(recognizer.handwriting_recognizer.calls))
        self.assertEqual(2, len(recognizer.handwriting_recognizer.calls[0]))
        self.assertIs(images[1], recognizer.handwriting_recognizer.calls[0][0])
        self.assertIs(images[2], recognizer.handwriting_recognizer.calls[0][1])


if __name__ == "__main__":
    unittest.main()

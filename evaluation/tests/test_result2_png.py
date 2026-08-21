import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from evaluation.evaluate_result2_png import decode_segmentation_png


class Result2PngTest(unittest.TestCase):
    def test_two_color_palettes_decode_to_same_classes(self):
        palette_a = np.array(
            [
                [35, 92, 167],
                [125, 180, 213],
                [219, 241, 247],
                [255, 219, 124],
                [252, 120, 59],
                [208, 10, 0],
            ],
            dtype=np.uint8,
        )
        palette_b = np.array(
            [
                [0, 66, 153],
                [104, 168, 206],
                [213, 239, 246],
                [255, 213, 103],
                [252, 99, 27],
                [208, 10, 0],
            ],
            dtype=np.uint8,
        )
        image = np.vstack([palette_a, palette_b]).reshape(2, -1, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "labels.png")
            Image.fromarray(image, mode="RGB").save(path)
            decoded = decode_segmentation_png(path)
        np.testing.assert_array_equal(decoded[0], np.arange(6))
        np.testing.assert_array_equal(decoded[1], np.arange(6))

    def test_unknown_color_is_rejected(self):
        image = np.array([[[1, 2, 3]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "unknown.png")
            Image.fromarray(image, mode="RGB").save(path)
            with self.assertRaises(ValueError):
                decode_segmentation_png(path)


if __name__ == "__main__":
    unittest.main()

"""Stage 2: binarization of the greyscale signing sheet.

Owned by: Image Preprocessing role.

TODO (Image Preprocessing role): implement every method below.
"""

from __future__ import annotations

import cv2
import numpy as np


class Binarizer:
    """Converts a greyscale image into a binary image where ink /
    printed lines are white (255) and the paper background is black (0),
    which is the convention OpenCV's morphology functions expect
    (image_pipeline/layout.py relies on this convention).
    """

    def adaptive(self, gray: np.ndarray, block_size: int = 25, c: int = 10) -> np.ndarray:
        """Adaptive (locally-thresholded) binarization -- handles uneven
        lighting across a phone photo far better than a single global
        threshold, so use this for the *whole sheet* (line detection).

        TODO: cv2.adaptiveThreshold with cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        and cv2.THRESH_BINARY_INV (inverted, so ink ends up white).
        block_size must be odd -- bump it by 1 if it isn't.
        """
        raise NotImplementedError("TODO: implement Binarizer.adaptive")

    def otsu(self, gray: np.ndarray) -> np.ndarray:
        """Global Otsu threshold -- used for small crops (a single
        signature cell) where a local/adaptive window isn't reliable.

        TODO: cv2.threshold with cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU.
        """
        raise NotImplementedError("TODO: implement Binarizer.otsu")

    def clean(self, binary: np.ndarray, kernel_size: int = 2) -> np.ndarray:
        """Remove salt-and-pepper speckle noise left over from
        binarization without eroding genuine pen strokes.

        TODO: a morphological opening (erode then dilate) with a small
        kernel removes isolated single-pixel speckles while leaving a
        real stroke's connected blob intact. See
        cv2.getStructuringElement + cv2.morphologyEx(..., cv2.MORPH_OPEN, ...).
        """
        raise NotImplementedError("TODO: implement Binarizer.clean")
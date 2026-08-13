"""Stage 1: loading and cleaning up the raw phone-camera photo.

Owned by: Image Preprocessing role.

TODO (Image Preprocessing role): implement every method below. This
stage turns a raw phone photo into a clean, upright, denoised greyscale
image ready for binarization (image_pipeline/binarization.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageLoadError(ValueError):
    pass


class ImagePreprocessor:
    """Loads a signing-sheet photo and prepares it for binarization."""

    def __init__(self, max_width: int = 1600):
        self.max_width = max_width

    def load(self, image_path: str | Path) -> np.ndarray:
        """Load the image from disk as a BGR array.

        TODO: use cv2.imread. Raise ImageLoadError if the path doesn't
        exist (check with Path.exists()) or if cv2.imread returns None
        (it fails silently on bad/corrupt files instead of raising).
        """
        raise NotImplementedError("TODO: implement ImagePreprocessor.load")

    def normalize_size(self, image: np.ndarray) -> np.ndarray:
        """Downscale very large phone photos to self.max_width so that
        later kernel sizes / pixel thresholds behave predictably across
        different phones' camera resolutions.

        TODO: if image width <= self.max_width, return it unchanged.
        Otherwise resize (keeping aspect ratio) with cv2.resize, e.g.
        interpolation=cv2.INTER_AREA (good for shrinking).
        """
        raise NotImplementedError("TODO: implement ImagePreprocessor.normalize_size")

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """TODO: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)."""
        raise NotImplementedError("TODO: implement ImagePreprocessor.to_grayscale")

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Remove phone-camera sensor noise while keeping pen strokes and
        printed table lines crisp.

        TODO: a filter that smooths flat regions but preserves edges
        works best here (a plain Gaussian blur will also soften the
        ruled lines you need later). Look at cv2.bilateralFilter.
        """
        raise NotImplementedError("TODO: implement ImagePreprocessor.denoise")

    def estimate_skew_angle(self, gray: np.ndarray, max_angle: float = 15.0) -> float:
        """Estimate how many degrees the photo is rotated off-square.

        A hand-held photo is rarely perfectly top-down, and even a
        couple of degrees of tilt is enough to break row/column line
        detection later in the pipeline (layout.py), since a "horizontal"
        ruled line then isn't at a single, consistent pixel row.

        TODO: one approach -- run an edge detector (cv2.Canny), then
        cv2.HoughLinesP to find long line segments, compute each
        segment's angle (np.arctan2), keep only near-horizontal ones
        (abs(angle) <= max_angle) and return their median angle in
        degrees. Return 0.0 if you can't find enough lines to trust.
        """
        raise NotImplementedError("TODO: implement ImagePreprocessor.estimate_skew_angle")

    def deskew(self, color: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Rotate the photo so the table's ruled lines are horizontal.

        TODO: call estimate_skew_angle, then (if it's non-trivial) build
        a rotation matrix with cv2.getRotationMatrix2D around the image
        centre and apply it to both `color` and `gray` with
        cv2.warpAffine. Return (rotated_color, rotated_gray, angle_used).
        If the angle is ~0, just return the inputs unchanged with angle 0.0.
        """
        raise NotImplementedError("TODO: implement ImagePreprocessor.deskew")

    def prepare(self, image_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs load -> resize -> greyscale -> deskew -> denoise.

        Returns (color_image, grayscale_image, denoised_image).

        TODO: chain the methods above in that order. Note deskew()
        needs both the color and grayscale images and returns updated
        versions of both -- make sure denoise() runs on the *deskewed*
        grayscale image, not the pre-deskew one.
        """
        raise NotImplementedError("TODO: implement ImagePreprocessor.prepare")
"""ORB feature-based similarity scoring between two signature crops.

Owned by: Recognition role.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MatchResult:
    similarity: float  # 0..1, higher = more alike
    good_matches: int
    keypoints_a: int
    keypoints_b: int


class OrbSignatureMatcher:
    """Compares two signature images using ORB keypoints + a Hamming-
    distance ratio test (Lowe's ratio test), which is robust to the
    different pen colours/thickness students use and does not require
    training data (unlike a learned verifier)."""

    def __init__(self, n_features: int = 500, ratio_threshold: float = 0.75, edge_threshold: int = 7):
        # A cropped signature cell is often only ~30-40px tall. ORB's
        # default edgeThreshold/patchSize (31px) then exceeds the image
        # itself, so every candidate keypoint gets filtered out as too
        # close to the border and detectAndCompute silently returns zero
        # keypoints. Shrinking both to fit small crops is what makes ORB
        # usable here at all.
        self.ratio_threshold = ratio_threshold
        self._orb = cv2.ORB_create(
            nfeatures=n_features, edgeThreshold=edge_threshold, patchSize=edge_threshold
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

   @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def compare(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        gray_a = self._to_gray(image_a)
        gray_b = self._to_gray(image_b)

        kp_a, des_a = self._orb.detectAndCompute(gray_a, None)
        kp_b, des_b = self._orb.detectAndCompute(gray_b, None)

        if des_a is None or des_b is None or len(kp_a) == 0 or len(kp_b) == 0:
            return MatchResult(similarity=0.0, good_matches=0, keypoints_a=len(kp_a or []), keypoints_b=len(kp_b or []))

        raw_matches = self._matcher.knnMatch(des_a, des_b, k=2)
        good = [
            m for m, n in (pair for pair in raw_matches if len(pair) == 2)
            if m.distance < self.ratio_threshold * n.distance
        ]

        smaller_keypoint_count = max(min(len(kp_a), len(kp_b)), 1)
        similarity = min(len(good) / smaller_keypoint_count, 1.0)

        return MatchResult(
            similarity=similarity,
            good_matches=len(good),
            keypoints_a=len(kp_a),
            keypoints_b=len(kp_b),
        )

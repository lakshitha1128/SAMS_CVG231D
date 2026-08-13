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

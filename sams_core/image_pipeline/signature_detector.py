"""Stage 5: deciding whether a signature cell contains a signature.

Owned by: Signature Detection role.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from sams_core.image_pipeline.binarization import Binarizer
from sams_core.models import AttendanceStatus


@dataclass
class SignatureClassification:
    status: AttendanceStatus
    ink_ratio: float


class InkDensitySignatureDetector:
    """Classifies a cropped signature cell as PRESENT/ABSENT based on
    the proportion of "ink" pixels it contains.

    A blank cell (student absent) has only faint scan/paper noise, while
    a signed cell -- regardless of pen colour -- has a much higher ratio
    of dark pixels once binarized. Using ink density instead of colour
    keeps the detector robust to the different coloured pens students use
    (see coursework scenario).
    """

    def __init__(self, ink_ratio_threshold: float = 0.012, binarizer: Binarizer | None = None):
        self.ink_ratio_threshold = ink_ratio_threshold
        self._binarizer = binarizer or Binarizer()

    def classify(self, cell_bgr_or_gray: np.ndarray) -> SignatureClassification:
        if cell_bgr_or_gray.size == 0:
            return SignatureClassification(AttendanceStatus.ABSENT, 0.0)

        if cell_bgr_or_gray.ndim == 3:
            gray = cv2.cvtColor(cell_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_bgr_or_gray

        binary = self._binarizer.otsu(gray)
        binary = self._binarizer.clean(binary, kernel_size=2)

        ink_ratio = float(np.count_nonzero(binary)) / float(binary.size)
        status = (
            AttendanceStatus.PRESENT
            if ink_ratio >= self.ink_ratio_threshold
            else AttendanceStatus.ABSENT
        )
        return SignatureClassification(status=status, ink_ratio=ink_ratio)

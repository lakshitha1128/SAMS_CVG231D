import numpy as np

from sams_core.image_pipeline.signature_detector import InkDensitySignatureDetector
from sams_core.models import AttendanceStatus


def make_cell(signed: bool) -> np.ndarray:
    cell = np.full((60, 200, 3), 255, dtype=np.uint8)
    if signed:
        cell[25:35, 20:180] = (20, 20, 160)  # a pen stroke
    return cell


def test_blank_cell_is_absent():
    detector = InkDensitySignatureDetector()
    result = detector.classify(make_cell(signed=False))
    assert result.status is AttendanceStatus.ABSENT
    assert result.ink_ratio < detector.ink_ratio_threshold


def test_signed_cell_is_present():
    detector = InkDensitySignatureDetector()
    result = detector.classify(make_cell(signed=True))
    assert result.status is AttendanceStatus.PRESENT
    assert result.ink_ratio >= detector.ink_ratio_threshold


def test_empty_crop_is_absent():
    detector = InkDensitySignatureDetector()
    result = detector.classify(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result.status is AttendanceStatus.ABSENT
    assert result.ink_ratio == 0.0

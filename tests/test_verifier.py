import random

import cv2
import numpy as np
import pytest

from sams_core.recognition.verifier import (
    InsufficientSamplesError,
    SignatureSample,
    SignatureVerifier,
)
from tests.generate_sample_sheet import _draw_scribble


def save_signature(path, seed=None):
    image = np.full((80, 240, 3), 255, dtype=np.uint8)
    if seed is not None:
        _draw_scribble(image, 10, 10, 230, 70, random.Random(seed))
    cv2.imwrite(str(path), image)
    return str(path)


def test_raises_when_fewer_than_two_samples(tmp_path):
    verifier = SignatureVerifier()
    samples = [SignatureSample(session_date="2019-06-21", image_path=save_signature(tmp_path / "a.png", 1))]
    with pytest.raises(InsufficientSamplesError):
        verifier.verify("001", samples)


def test_identical_signature_is_flagged_as_match(tmp_path):
    verifier = SignatureVerifier()
    same_path_a = save_signature(tmp_path / "a.png", seed=1)
    same_path_b = save_signature(tmp_path / "b.png", seed=1)
    samples = [
        SignatureSample(session_date="2019-06-21", image_path=same_path_a),
        SignatureSample(session_date="2019-06-28", image_path=same_path_b),
    ]

    report = verifier.verify("001", samples)
    assert report.reference_date == "2019-06-21"
    assert len(report.entries) == 1
    assert report.entries[0].is_match is True


def test_blank_signature_is_flagged_as_mismatch_against_signed_reference(tmp_path):
    verifier = SignatureVerifier()
    reference_path = save_signature(tmp_path / "a.png", seed=1)
    blank_path = save_signature(tmp_path / "b.png", seed=None)
    samples = [
        SignatureSample(session_date="2019-06-21", image_path=reference_path),
        SignatureSample(session_date="2019-06-28", image_path=blank_path),
    ]

    report = verifier.verify("001", samples)
    assert report.entries[0].is_match is False
    assert len(report.mismatches) == 1

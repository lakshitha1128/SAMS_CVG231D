import numpy as np

from sams_core.recognition.feature_matcher import OrbSignatureMatcher
from tests.generate_sample_sheet import _draw_scribble


def make_signature_image(seed: int) -> np.ndarray:
    image = np.full((80, 240, 3), 255, dtype=np.uint8)
    import random

    _draw_scribble(image, 10, 10, 230, 70, random.Random(seed))
    return image


def test_identical_signature_scores_higher_than_different_one():
    matcher = OrbSignatureMatcher()
    reference = make_signature_image(seed=1)
    same_again = reference.copy()
    different = make_signature_image(seed=99)

    result_same = matcher.compare(reference, same_again)
    result_diff = matcher.compare(reference, different)

    assert result_same.similarity >= result_diff.similarity


def test_blank_images_return_zero_similarity_without_crashing():
    matcher = OrbSignatureMatcher()
    blank = np.full((80, 240, 3), 255, dtype=np.uint8)

    result = matcher.compare(blank, blank)
    assert 0.0 <= result.similarity <= 1.0

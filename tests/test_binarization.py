import numpy as np

from sams_core.image_pipeline.binarization import Binarizer


def test_otsu_separates_ink_from_background():
    gray = np.full((100, 100), 255, dtype=np.uint8)
    gray[40:60, 40:60] = 10  # a dark "ink" square on a white page

    binary = Binarizer().otsu(gray)

    assert binary[50, 50] == 255  # ink -> foreground (white)
    assert binary[5, 5] == 0      # background stays background


def test_adaptive_handles_uneven_lighting():
    # A smooth lighting gradient across the page, with a thin dark "ruled
    # line" drawn across it -- adaptive thresholding should still pick up
    # the line even though a single global threshold could not, since the
    # line is much darker than its *local* neighbourhood everywhere.
    row_gradient = np.linspace(120, 220, 100, dtype=np.uint8)
    gray = np.tile(row_gradient, (100, 1))
    gray[49:51, :] = np.clip(gray[49:51, :].astype(int) - 60, 0, 255).astype(np.uint8)

    binary = Binarizer().adaptive(gray)

    assert binary[50, 10] == 255
    assert binary[50, 90] == 255
    assert binary.dtype == np.uint8


def test_clean_removes_isolated_speckles():
    binary = np.zeros((50, 50), dtype=np.uint8)
    binary[10, 10] = 255  # single-pixel speckle
    binary[20:30, 20:30] = 255  # a real 10x10 blob

    cleaned = Binarizer().clean(binary, kernel_size=3)

    assert cleaned[10, 10] == 0
    assert cleaned[25, 25] == 255

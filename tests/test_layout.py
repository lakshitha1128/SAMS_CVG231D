from sams_core.image_pipeline.binarization import Binarizer
from sams_core.image_pipeline.cell_extractor import SignatureCellExtractor
from sams_core.image_pipeline.layout import TableLayoutDetector
from sams_core.image_pipeline.preprocessing import ImagePreprocessor
from tests.generate_sample_sheet import generate_signing_sheet

NAMES = ["John Snow", "James Bond", "Andare"]


def _binary_for(image):
    pre = ImagePreprocessor()
    gray = pre.to_grayscale(image)
    return Binarizer().adaptive(pre.denoise(gray))


def test_detects_student_table_with_correct_row_count():
    image = generate_signing_sheet(NAMES, [True, True, False])
    binary = _binary_for(image)

    table = TableLayoutDetector().find_student_table(binary, expected_rows=len(NAMES))

    # header row + N student rows => N + 2 boundary lines
    assert len(table.row_boundaries) == len(NAMES) + 2
    assert table.width > 0 and table.height > 0


def test_student_row_bands_skip_header():
    image = generate_signing_sheet(NAMES, [True, True, False])
    binary = _binary_for(image)
    table = TableLayoutDetector().find_student_table(binary, expected_rows=len(NAMES))

    bands = SignatureCellExtractor().student_row_bands(table)
    assert len(bands) == len(NAMES)
    for top, bottom in bands:
        assert bottom > top


def test_even_split_fallback_used_when_no_rules_detected(monkeypatch):
    image = generate_signing_sheet(NAMES, [True, False, True])
    binary = _binary_for(image)
    detector = TableLayoutDetector()

    # Force the "no real lines detected" branch regardless of how well
    # the synthetic image's ruling lines are picked up.
    monkeypatch.setattr(detector, "_detect_row_boundaries", lambda *a, **k: [])

    table = detector.find_student_table(binary, expected_rows=len(NAMES))
    assert table.lines_detected is False
    assert len(table.row_boundaries) == len(NAMES) + 2

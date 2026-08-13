"""Orchestrates the full image-processing pipeline: load -> greyscale ->
denoise -> binarize -> locate table -> extract signature cells ->
classify presence -> return attendance records.

Owned by: Project Lead / Architecture role (composition), building on the
Image Preprocessing, Table/ROI Detection and Signature Detection roles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import cv2

from sams_core.config import AppConfig
from sams_core.image_pipeline.binarization import Binarizer
from sams_core.image_pipeline.cell_extractor import SignatureCellExtractor
from sams_core.image_pipeline.layout import TableLayoutDetector
from sams_core.image_pipeline.preprocessing import ImagePreprocessor
from sams_core.image_pipeline.signature_detector import InkDensitySignatureDetector
from sams_core.models import AttendanceRecord, SigningSession, Student
from sams_core.utils.image_io import StepArtifactWriter
from sams_core.utils.logging_utils import ProgressReporter

_DATE_PATTERN = re.compile(r"(\d{1,2})[.\-_](\d{1,2})[.\-_](\d{4})")


class SheetProcessingError(ValueError):
    pass


def parse_session_date(image_path: str | Path) -> date:
    """Extracts the session date from the sheet's filename, e.g.
    ``12.07.2019.png`` -> 2019-07-12 (matches the CLI convention shown in
    the coursework: ``python sams.py 10.07.2019.png info.xml``)."""
    stem = Path(image_path).name
    match = _DATE_PATTERN.search(stem)
    if not match:
        raise SheetProcessingError(
            f"Could not find a dd.mm.yyyy date in filename: {stem}"
        )
    day, month, year = (int(g) for g in match.groups())
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise SheetProcessingError(f"Invalid date in filename {stem}: {exc}") from exc


@dataclass
class PipelineResult:
    session: SigningSession
    records: list[AttendanceRecord]
    artifacts_dir: Path
    table_lines_detected: bool


class AttendanceImagePipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor()
        self.binarizer = Binarizer()
        self.layout_detector = TableLayoutDetector(
            min_row_height_px=config.table_layout.min_row_height_px,
            signature_column_start=config.table_layout.signature_column_start,
        )
        self.cell_extractor = SignatureCellExtractor()
        self.signature_detector = InkDensitySignatureDetector(
            ink_ratio_threshold=config.table_layout.ink_ratio_threshold
        )

    def process(self, image_path: str | Path, students: list[Student]) -> PipelineResult:
        image_path = Path(image_path)
        sheet_name = image_path.stem
        writer = StepArtifactWriter(self.config.processed_dir, sheet_name)
        reporter = ProgressReporter(total_steps=10)

        reporter.step(f"Loading image: {image_path.name}")
        color = self.preprocessor.normalize_size(self.preprocessor.load(image_path))
        writer.save("original", color)

        reporter.step("Converting to greyscale")
        gray = self.preprocessor.to_grayscale(color)
        writer.save("greyscale", gray)

        color, gray, skew_angle = self.preprocessor.deskew(color, gray)
        reporter.step(f"Correcting camera tilt (deskew: {skew_angle:+.2f} deg)")
        writer.save("deskewed", gray)

        reporter.step("Denoising")
        denoised = self.preprocessor.denoise(gray)
        writer.save("denoised", denoised)

        reporter.step("Binarizing (adaptive threshold)")
        binary = self.binarizer.adaptive(denoised)
        writer.save("binarized", binary)

        reporter.step("Detecting table grid lines")
        try:
            table = self.layout_detector.find_student_table(
                binary, expected_rows=len(students)
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the CLI user
            raise SheetProcessingError(f"Table detection failed: {exc}") from exc

        overlay = color.copy()
        cv2.rectangle(
            overlay, (table.x, table.y), (table.right, table.bottom), (0, 255, 0), 3
        )
        for y in table.row_boundaries:
            cv2.line(overlay, (table.x, y), (table.right, y), (255, 0, 0), 1)
        writer.save("table_region", overlay)

        reporter.step(
            "Locating student rows "
            f"({'ruled lines' if table.lines_detected else 'even split fallback'})"
        )

        reporter.step("Extracting signature cells")
        crops = self.cell_extractor.extract(color, table)

        montage = _build_montage([c.display for c in crops])
        if montage is not None:
            writer.save("signature_cells", montage)

        reporter.step("Classifying presence via ink density")
        session_date = parse_session_date(image_path)
        session = SigningSession(session_date=session_date, source_image=image_path.name)

        records: list[AttendanceRecord] = []
        for student, crop in zip(students, crops):
            classification = self.signature_detector.classify(crop.classification)
            sig_path = ""
            if crop.display.size > 0:
                sig_dir = self.config.signatures_dir / student.index
                sig_dir.mkdir(parents=True, exist_ok=True)
                sig_file = sig_dir / f"{session_date.isoformat()}.png"
                cv2.imwrite(str(sig_file), crop.display)
                sig_path = str(sig_file)

            records.append(
                AttendanceRecord(
                    student_index=student.index,
                    status=classification.status,
                    ink_ratio=classification.ink_ratio,
                    signature_image_path=sig_path,
                    session=session,
                )
            )

        if len(crops) != len(students):
            reporter.step(
                f"WARNING: detected {len(crops)} row(s) but info.xml has "
                f"{len(students)} student(s); matched the first "
                f"{min(len(crops), len(students))} by row order"
            )
        else:
            reporter.step(f"Matched {len(records)} student rows by position")

        reporter.done(f"Processed {image_path.name}: {len(records)} attendance record(s)")

        return PipelineResult(
            session=session,
            records=records,
            artifacts_dir=writer.directory,
            table_lines_detected=table.lines_detected,
        )


def _build_montage(crops, max_cell_height: int = 60):
    import numpy as np

    valid = [c for c in crops if c.size > 0]
    if not valid:
        return None

    resized = []
    for crop in valid:
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            continue
        scale = max_cell_height / float(h)
        resized.append(cv2.resize(crop, (max(int(w * scale), 1), max_cell_height)))

    if not resized:
        return None

    max_w = max(c.shape[1] for c in resized)
    padded = []
    for c in resized:
        if c.ndim == 2:
            c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        pad_w = max_w - c.shape[1]
        padded.append(cv2.copyMakeBorder(c, 2, 2, 0, pad_w, cv2.BORDER_CONSTANT, value=(200, 200, 200)))

    return cv2.vconcat(padded)

"""Synthesizes a signing-sheet image for tests and local development, so
the pipeline can be exercised without needing the real scanned sheets
from "CGV Signing Sheets.zip" (add those separately under
data/signing_sheets/).

Owned by: Testing/QA role.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

# Matches sams_core.config.TableLayout defaults.
COLUMN_FRACTIONS = {
    "no": (0.0, 0.06),
    "student_no": (0.06, 0.24),
    "title": (0.24, 0.32),
    "name": (0.32, 0.68),
    "signature": (0.68, 1.0),
}

PEN_COLORS = [(180, 30, 20), (20, 20, 160), (10, 10, 10)]  # BGR: red, blue, black


def generate_signing_sheet(
    student_names: list[str],
    signed_flags: list[bool],
    width: int = 1200,
    row_height: int = 60,
    table_x: int = 100,
    table_top: int = 150,
    seed: int = 0,
) -> np.ndarray:
    """Draws a synthetic ruled table matching the coursework's signing
    sheet layout: No | Student No | Title | Student Name | Signature."""
    assert len(student_names) == len(signed_flags)
    rng = random.Random(seed)

    n_rows = len(student_names) + 1  # + header row
    table_width = width - 2 * table_x
    height = table_top + n_rows * row_height + 100

    image = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.putText(image, "Signing Sheet", (table_x, table_top - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Grid lines
    for row in range(n_rows + 1):
        y = table_top + row * row_height
        cv2.line(image, (table_x, y), (table_x + table_width, y), (0, 0, 0), 2)

    col_x = {}
    for name, (start, end) in COLUMN_FRACTIONS.items():
        col_x[name] = table_x + int(table_width * start)
    col_x["end"] = table_x + table_width
    for x in list(col_x.values()):
        cv2.line(image, (x, table_top), (x, table_top + n_rows * row_height), (0, 0, 0), 2)

    # Header row text
    header_y = table_top + int(row_height * 0.6)
    cv2.putText(image, "No", (col_x["no"] + 5, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "Student No", (col_x["student_no"] + 5, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "Name", (col_x["name"] + 5, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "Signature", (col_x["signature"] + 5, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Student rows
    for i, (name, signed) in enumerate(zip(student_names, signed_flags)):
        row_top = table_top + (i + 1) * row_height
        text_y = row_top + int(row_height * 0.6)

        cv2.putText(image, str(i + 1), (col_x["no"] + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, name, (col_x["name"] + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if signed:
            _draw_scribble(image, col_x["signature"], row_top, col_x["end"], row_top + row_height, rng)

    return image


def _draw_scribble(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, rng: random.Random) -> None:
    color = rng.choice(PEN_COLORS)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    n_points = rng.randint(8, 14)
    points = []
    for i in range(n_points):
        px = x0 + int((x1 - x0) * i / (n_points - 1)) + rng.randint(-5, 5)
        py = cy + rng.randint(-int((y1 - y0) * 0.3), int((y1 - y0) * 0.3))
        points.append((px, py))
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

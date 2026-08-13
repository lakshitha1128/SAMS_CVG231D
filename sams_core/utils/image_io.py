"""Helpers for persisting intermediate image-processing artefacts to
disk, so they can be captured as screenshots for the coursework report.

Owned by: Project Lead / Architecture role.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class StepArtifactWriter:
    """Writes each pipeline step's output image into a per-sheet folder,
    e.g. data/processed/12.07.2019/02_binarized.png
    """

    def __init__(self, processed_dir: Path, sheet_name: str):
        self._dir = Path(processed_dir) / sheet_name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index = 0

    def save(self, label: str, image: np.ndarray) -> Path:
        self._index += 1
        safe_label = label.strip().lower().replace(" ", "_")
        out_path = self._dir / f"{self._index:02d}_{safe_label}.png"
        cv2.imwrite(str(out_path), image)
        return out_path

    @property
    def directory(self) -> Path:
        return self._dir

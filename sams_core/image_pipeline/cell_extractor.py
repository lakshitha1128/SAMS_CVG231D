"""Stage 4: cropping the Signature cell for each student row.

Owned by: Table/ROI Detection role.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sams_core.image_pipeline.layout import TableRegion


class SignatureCellExtractor:
    """Crops each student row's Signature cell.

    Two crops matter here for different reasons, so `extract()` returns
    both: a `display` crop (light margin, just enough to drop the ruled
    grid lines themselves) used for storage, the report montage and the
    bonus signature-recognition feature, and a `classification` crop
    (trimmed well into the row) used only to measure ink density. A long
    cursive underline from the row *above* commonly bleeds across the
    ruled line into the top of the next cell -- trimming that much off
    the display crop would mutilate genuine signatures, but the
    presence/absence decision needs that bleed-over excluded.
    """

    def __init__(
        self,
        display_margin_px: int = 4,
        classification_top_margin_ratio: float = 0.22,
        classification_bottom_margin_px: int = 4,
    ):
        self.display_margin_px = display_margin_px
        self.classification_top_margin_ratio = classification_top_margin_ratio
        self.classification_bottom_margin_px = classification_bottom_margin_px

    def student_row_bands(self, table: TableRegion) -> list[tuple[int, int]]:
        """Returns (top, bottom) y-coordinates for each *student* row,
        skipping the header text row (the first band)."""
        boundaries = table.row_boundaries
        bands = list(zip(boundaries[:-1], boundaries[1:]))
        return bands[1:]  # drop header row band

    def extract(self, image: np.ndarray, table: TableRegion) -> list[SignatureCellCrop]:
        """Crops the signature-column region for every student row."""
        sig_x_start = table.signature_column_x
        sig_x_end = table.right

        crops: list[SignatureCellCrop] = []
        for top, bottom in self.student_row_bands(table):
            row_height = bottom - top

            dy0 = min(top + self.display_margin_px, bottom)
            dy1 = max(bottom - self.display_margin_px, dy0 + 1)
            display = image[dy0:dy1, sig_x_start:sig_x_end]

            top_margin = int(row_height * self.classification_top_margin_ratio)
            cy0 = min(top + top_margin, bottom)
            cy1 = max(bottom - self.classification_bottom_margin_px, cy0 + 1)
            classification = image[cy0:cy1, sig_x_start:sig_x_end]

            crops.append(SignatureCellCrop(display=display, classification=classification))
        return crops


@dataclass
class SignatureCellCrop:
    display: np.ndarray
    classification: np.ndarray

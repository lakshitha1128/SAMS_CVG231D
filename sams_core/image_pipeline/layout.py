"""Stage 3: locating the student table and its row boundaries within the
signing sheet.

Owned by: Table/ROI Detection role.

The signing sheet has a *static* layout (per the coursework spec): a
small header table (Date / Lecturer / Signature) followed by a larger
student table (No / Student No / Title / Student Name / Signature). This
module finds the student table by looking for the tallest ruled-line
grid on the page, then reconstructs its row boundaries.

Because a hand-held phone photo can have faint or broken ruling lines
(most commonly a missing/shadowed bottom border), row detection anchors
on whichever lines it *does* find and extrapolates the rest from the
measured row spacing, rather than trusting the ruled-grid contour's own
(possibly truncated) height. It only falls back to a blind even split of
that contour height when too few real lines were found to measure a
spacing from at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TableRegion:
    x: int
    y: int
    width: int
    height: int
    row_boundaries: list[int]  # absolute y-coordinates, sorted, len == rows+1
    lines_detected: bool  # whether boundaries came from real ruling lines
    signature_column_x: int  # absolute x-coordinate where the Signature column starts

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


class TableNotFoundError(ValueError):
    pass


class TableLayoutDetector:
    def __init__(self, min_row_height_px: int = 20, signature_column_start: float = 0.68):
        self.min_row_height_px = min_row_height_px
        self.signature_column_start = signature_column_start

    def _line_masks(self, binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = binary.shape[:2]
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(width // 15, 10), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(height // 40, 10)))

        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        return horizontal, vertical

    def find_student_table(self, binary: np.ndarray, expected_rows: int) -> TableRegion:
        """`binary` must be an inverted binary image (ink/lines = 255).

        A hand-held photo often has a faint or broken bottom border, which
        makes the ruled-grid *contour* shorter than the real table. Relying
        on that contour's height alone (e.g. for an even split) would then
        silently drift every row boundary upward. Instead: pick the table
        candidate by its *line count* in a generously expanded search
        band (robust even when its own contour is truncated), then anchor
        row boundaries on whichever lines were actually found and
        extrapolate the rest from the measured row spacing.
        """
        horizontal, vertical = self._line_masks(binary)
        grid = cv2.bitwise_or(horizontal, vertical)
        grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise TableNotFoundError("No ruled grid found in the signing sheet")

        img_h, img_w = binary.shape[:2]
        # Only filter by width here -- a table's bounding height is exactly
        # the unreliable signal a broken bottom border corrupts, so it must
        # not gate candidacy, only decide it (via line count, below).
        candidates = [
            (x, y, w, h)
            for (x, y, w, h) in (cv2.boundingRect(c) for c in contours)
            if w > img_w * 0.3
        ]
        if not candidates:
            raise TableNotFoundError("No table-sized region found in the signing sheet")

        # For each candidate, search a band a little beyond its own
        # (possibly truncated) contour for ruled lines sharing its
        # x-range, and keep whichever candidate has the most lines -- the
        # student table has (expected_rows + 2) lines and so outnumbers
        # the small header table regardless of which one's contour
        # happened to be taller. The margin is intentionally modest and
        # biased downward: a candidate's detected top edge is normally
        # already accurate (the first ruled line is rarely missed), it's
        # the *bottom* border that goes missing on a hand-held photo, and
        # a wide search in both directions risks pulling in the other
        # table's lines and corrupting the row-height estimate.
        best = None
        best_lines: list[int] = []
        for (x, y, w, h) in candidates:
            margin_above = max(int(h * 0.05), 10)
            margin_below = max(int(h * 0.6), int(img_h * 0.04))
            search_top = max(y - margin_above, 0)
            search_bottom = min(y + h + margin_below, img_h)
            lines = self._detect_row_boundaries(
                horizontal, x, search_top, w, search_bottom - search_top
            )
            if best is None or len(lines) > len(best_lines):
                best = (x, y, w, h)
                best_lines = lines

        x, y, w, h = best
        row_boundaries, lines_detected = self._resolve_row_boundaries(
            best_lines, fallback_top=y, fallback_height=h, expected_rows=expected_rows
        )

        # Keep the overlay/crop region consistent with whatever boundaries
        # were actually used, even if they extend past the original contour.
        top = min(y, row_boundaries[0])
        bottom = max(y + h, row_boundaries[-1])

        signature_column_x = self._resolve_signature_column(vertical, x, top, w, bottom - top)

        return TableRegion(
            x=x, y=top, width=w, height=bottom - top,
            row_boundaries=row_boundaries,
            lines_detected=lines_detected,
            signature_column_x=signature_column_x,
        )

    def _resolve_signature_column(
        self, vertical_mask: np.ndarray, x: int, y: int, w: int, h: int
    ) -> int:
        """Finds the Name|Signature column divider. A phone photo's
        column lines are rarely detected as cleanly/completely as the
        expected 6 (No | Student No | Title | Name | Signature = 5
        columns), so rather than requiring an exact count, this takes
        whichever *detected* vertical line lands closest to where the
        static layout fraction expects the divider -- falling back to
        that fraction outright if nothing is close enough to trust."""
        expected_x = x + int(w * self.signature_column_start)

        column_lines = self._detect_column_boundaries(vertical_mask, x, y, w, h)
        if not column_lines:
            return expected_x

        closest = min(column_lines, key=lambda cx: abs(cx - expected_x))
        if abs(closest - expected_x) <= w * 0.15:
            return closest
        return expected_x

    def _detect_column_boundaries(
        self, vertical_mask: np.ndarray, x: int, y: int, w: int, h: int
    ) -> list[int]:
        region = vertical_mask[y : y + h, x : x + w]
        col_sums = np.sum(region > 0, axis=0)
        threshold = h * 0.5
        line_cols = np.where(col_sums > threshold)[0]
        if line_cols.size == 0:
            return []

        min_gap = 10
        boundaries: list[int] = []
        cluster_start = line_cols[0]
        prev = line_cols[0]
        for value in line_cols[1:]:
            if value - prev > min_gap // 2:
                boundaries.append(x + (cluster_start + prev) // 2)
                cluster_start = value
            prev = value
        boundaries.append(x + (cluster_start + prev) // 2)

        filtered = [boundaries[0]]
        for b in boundaries[1:]:
            if b - filtered[-1] >= min_gap:
                filtered.append(b)
        return filtered

    def _resolve_row_boundaries(
        self, detected: list[int], fallback_top: int, fallback_height: int, expected_rows: int
    ) -> tuple[list[int], bool]:
        needed = expected_rows + 2

        if len(detected) == needed:
            return detected, True

        if len(detected) >= 3:
            # Anchor on the topmost real line (reliably the header/row-1
            # divider) and extrapolate using the *median* gap between
            # detected lines, which tolerates a handful of missing/broken
            # ones without the whole grid drifting.
            gaps = [b - a for a, b in zip(detected, detected[1:])]
            row_height = float(np.median(gaps))
            anchor = detected[0]
            return [int(round(anchor + i * row_height)) for i in range(needed)], False

        anchor = detected[0] if detected else fallback_top
        return self._even_split(anchor, fallback_height, expected_rows), False

    def _detect_row_boundaries(
        self, horizontal_mask: np.ndarray, x: int, y: int, w: int, h: int
    ) -> list[int]:
        region = horizontal_mask[y : y + h, x : x + w]
        row_sums = np.sum(region > 0, axis=1)
        threshold = w * 0.5
        line_rows = np.where(row_sums > threshold)[0]
        if line_rows.size == 0:
            return []

        # Cluster consecutive pixel rows belonging to the same ruled line.
        boundaries: list[int] = []
        cluster_start = line_rows[0]
        prev = line_rows[0]
        for value in line_rows[1:]:
            if value - prev > self.min_row_height_px // 2:
                boundaries.append(y + (cluster_start + prev) // 2)
                cluster_start = value
            prev = value
        boundaries.append(y + (cluster_start + prev) // 2)

        # Drop clusters that are implausibly close together (noise).
        filtered = [boundaries[0]]
        for b in boundaries[1:]:
            if b - filtered[-1] >= self.min_row_height_px:
                filtered.append(b)
        return filtered

    @staticmethod
    def _even_split(y: int, h: int, expected_rows: int) -> list[int]:
        step = h / float(expected_rows + 1)
        return [int(round(y + i * step)) for i in range(expected_rows + 2)]

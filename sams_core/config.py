"""Central configuration and constants for SAMS.

Owned by: Project Lead / Architecture role.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SIGNING_SHEETS_DIR = DATA_DIR / "signing_sheets"
SIGNATURES_DIR = DATA_DIR / "signatures"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "sams.db"


@dataclass(frozen=True)
class TableLayout:
    """Describes the static layout of the signing-sheet table.

    The signing sheet always has a header block (date / lecturer / hall)
    followed by a student table with four columns: No, Student No, Title
    (+ Name), Signature. Only relative fractions are stored so the layout
    keeps working regardless of the photo's resolution.
    """

    # Fraction of the *student table* width (0..1) where the signature
    # column starts. Everything to the left is No/Student No/Title/Name.
    signature_column_start: float = 0.68

    # Minimum row height in pixels (after perspective/scale normalisation)
    # used to filter out noise lines when reconstructing the grid.
    min_row_height_px: int = 20

    # Fraction of a cell's pixels that must be "ink" for a signature to be
    # considered present. Calibrated against real scanned sheets: blank
    # cells (including a little bleed-over ink from a neighbouring row's
    # signature) measured up to ~0.03; genuine signatures measured 0.07+.
    ink_ratio_threshold: float = 0.045


@dataclass(frozen=True)
class AppConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    signing_sheets_dir: Path = SIGNING_SHEETS_DIR
    signatures_dir: Path = SIGNATURES_DIR
    processed_dir: Path = PROCESSED_DIR
    db_path: Path = DB_PATH
    table_layout: TableLayout = TableLayout()

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.signing_sheets_dir,
            self.signatures_dir,
            self.processed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = AppConfig()

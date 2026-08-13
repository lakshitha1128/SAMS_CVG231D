"""Domain models shared across SAMS.

Owned by: Data Layer role.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class AttendanceStatus(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"


@dataclass(frozen=True)
class Student:
    index: str
    name: str
    batch: str = ""

    def __post_init__(self) -> None:
        if not self.index.strip():
            raise ValueError("Student index cannot be empty")
        if not self.name.strip():
            raise ValueError("Student name cannot be empty")


@dataclass(frozen=True)
class SigningSession:
    """One signing sheet (one lecture occurrence)."""

    session_date: date
    source_image: str
    hall: str = ""
    lecturer: str = ""


@dataclass
class AttendanceRecord:
    student_index: str
    status: AttendanceStatus
    ink_ratio: float = 0.0
    signature_image_path: str = ""
    session: SigningSession | None = field(default=None)

    @property
    def is_present(self) -> bool:
        return self.status is AttendanceStatus.PRESENT

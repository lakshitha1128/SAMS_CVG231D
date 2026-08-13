"""Collects a student's captured signatures over time and reports any
that don't match the rest, per the coursework's "Recognition" bonus
requirement.

Owned by: Recognition role.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from sams_core.recognition.feature_matcher import OrbSignatureMatcher


@dataclass
class SignatureSample:
    session_date: str
    image_path: str


@dataclass
class VerificationEntry:
    session_date: str
    image_path: str
    similarity_to_reference: float
    is_match: bool


@dataclass
class VerificationReport:
    student_index: str
    reference_date: str | None
    entries: list[VerificationEntry]

    @property
    def mismatches(self) -> list[VerificationEntry]:
        return [e for e in self.entries if not e.is_match]

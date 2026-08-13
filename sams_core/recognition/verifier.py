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


class InsufficientSamplesError(ValueError):
    pass


class SignatureVerifier:
    """Flags signatures that likely don't belong to the student, by
    comparing every captured sample against a reference sample (the
    student's earliest recorded signature).

    Calibration caveat: signature cells cropped from a signing sheet are
    tiny (roughly 30x250px), which leaves ORB little texture to work
    with. Measured against this project's own sample sheets, genuine
    same-student comparisons scored ~0.06-0.15 similarity while
    different-student comparisons scored ~0.06-0.28 -- the two
    distributions overlap substantially, so `match_threshold` is a
    reasonable heuristic, not a validated cutoff. Treat a flagged
    mismatch as "worth a manual look" rather than a proven forgery; see
    the comparison chart investigate.py saves for a visual sanity check.
    """

    def __init__(self, matcher: OrbSignatureMatcher | None = None, match_threshold: float = 0.10):
        self.matcher = matcher or OrbSignatureMatcher()
        self.match_threshold = match_threshold

    def verify(self, student_index: str, samples: list[SignatureSample]) -> VerificationReport:
        valid_samples = [s for s in samples if Path(s.image_path).exists()]
        if len(valid_samples) < 2:
            raise InsufficientSamplesError(
                f"Need at least 2 captured signatures for student {student_index} "
                f"to verify consistency; found {len(valid_samples)}. "
                "Process more signing sheets with sams.py first."
            )

        valid_samples.sort(key=lambda s: s.session_date)
        reference = valid_samples[0]
        reference_img = cv2.imread(reference.image_path)

        entries: list[VerificationEntry] = []
        for sample in valid_samples[1:]:
            candidate_img = cv2.imread(sample.image_path)
            result = self.matcher.compare(reference_img, candidate_img)
            entries.append(
                VerificationEntry(
                    session_date=sample.session_date,
                    image_path=sample.image_path,
                    similarity_to_reference=result.similarity,
                    is_match=result.similarity >= self.match_threshold,
                )
            )

        return VerificationReport(
            student_index=student_index,
            reference_date=reference.session_date,
            entries=entries,
        )

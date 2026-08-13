"""Computes the attendance summary statistics shown by infovis.py.

Owned by: Visualization role.
"""

from __future__ import annotations

from dataclasses import dataclass

from sams_core.database import AttendanceDatabase
from sams_core.models import Student


class StudentNotFoundError(ValueError):
    pass


@dataclass
class AttendanceSummary:
    student: Student
    sessions: list[dict]  # rows from AttendanceDatabase.attendance_for_student

    @property
    def total_sessions(self) -> int:
        return len(self.sessions)

    @property
    def present_count(self) -> int:
        return sum(1 for s in self.sessions if s["status"] == "PRESENT")

    @property
    def absent_count(self) -> int:
        return self.total_sessions - self.present_count

    @property
    def attendance_percentage(self) -> float:
        if self.total_sessions == 0:
            return 0.0
        return 100.0 * self.present_count / self.total_sessions


class AttendanceReportBuilder:
    def __init__(self, db: AttendanceDatabase):
        self.db = db

    def build(self, student_index: str) -> AttendanceSummary:
        student = self.db.get_student(student_index)
        if student is None:
            raise StudentNotFoundError(
                f"No student with index '{student_index}' in the database. "
                "Run sams.py on a signing sheet first."
            )
        sessions = self.db.attendance_for_student(student_index)
        return AttendanceSummary(student=student, sessions=sessions)

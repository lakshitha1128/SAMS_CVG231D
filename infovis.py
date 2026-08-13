#!/usr/bin/env python
"""SAMS - attendance visualization.

Shows a summary report of a given student's attendance record.

Usage:
    python infovis.py <student_index>

Example:
    python infovis.py 10009301
"""

from __future__ import annotations

import sys

from sams_core.config import DEFAULT_CONFIG
from sams_core.database import AttendanceDatabase
from sams_core.visualization.report import AttendanceReportBuilder, StudentNotFoundError


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1

    student_index = argv[1]

    db = AttendanceDatabase(DEFAULT_CONFIG.db_path)
    report_builder = AttendanceReportBuilder(db)

    try:
        summary = report_builder.build(student_index)
    except StudentNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"Student       : {summary.student.name} ({summary.student.index})")
    print(f"Sessions      : {summary.total_sessions}")
    print(f"Present       : {summary.present_count}")
    print(f"Absent        : {summary.absent_count}")
    print(f"Attendance %  : {summary.attendance_percentage:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

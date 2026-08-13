import matplotlib.pyplot as plt

from sams_core.models import Student
from sams_core.visualization.charts import AttendanceChartBuilder
from sams_core.visualization.report import AttendanceSummary


def make_summary(sessions):
    return AttendanceSummary(student=Student(index="001", name="John Snow"), sessions=sessions)


def test_build_figure_with_mixed_attendance():
    summary = make_summary(
        [
            {"status": "PRESENT", "session_date": "2019-06-21"},
            {"status": "ABSENT", "session_date": "2019-06-28"},
            {"status": "PRESENT", "session_date": "2019-07-05"},
        ]
    )
    fig = AttendanceChartBuilder().build_figure(summary)
    try:
        assert fig is not None
        assert summary.present_count == 2
        assert summary.absent_count == 1
        assert round(summary.attendance_percentage) == 67
    finally:
        plt.close(fig)


def test_build_figure_with_no_sessions_does_not_crash():
    summary = make_summary([])
    fig = AttendanceChartBuilder().build_figure(summary)
    try:
        assert summary.total_sessions == 0
        assert summary.attendance_percentage == 0.0
    finally:
        plt.close(fig)

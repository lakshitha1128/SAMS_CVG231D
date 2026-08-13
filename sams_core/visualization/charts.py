"""Builds the attendance summary chart shown by infovis.py.

Owned by: Visualization role.

Colour usage follows a fixed status palette (never re-themed): green for
PRESENT, red for ABSENT, always paired with a direct text label so status
is never carried by colour alone.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from sams_core.visualization.report import AttendanceSummary

COLOR_PRESENT = "#0ca30c"
COLOR_ABSENT = "#d03b3b"
COLOR_MUTED_TEXT = "#52514e"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"


class AttendanceChartBuilder:
    def build_figure(self, summary: AttendanceSummary) -> Figure:
        fig = plt.figure(figsize=(11, 5.5), dpi=110, facecolor="#fcfcfb")
        fig.suptitle(
            f"Attendance Summary - {summary.student.name} ({summary.student.index})",
            fontsize=13,
            color="#0b0b0b",
            fontweight="bold",
        )

        timeline_ax = fig.add_axes((0.08, 0.15, 0.55, 0.68))
        donut_ax = fig.add_axes((0.70, 0.12, 0.27, 0.74))

        self._draw_timeline(timeline_ax, summary)
        self._draw_donut(donut_ax, summary)
        return fig

    def _draw_timeline(self, ax, summary: AttendanceSummary) -> None:
        if not summary.sessions:
            ax.text(
                0.5, 0.5, "No attendance recorded yet.\nRun sams.py on a signing sheet first.",
                ha="center", va="center", color=COLOR_MUTED_TEXT,
            )
            ax.axis("off")
            return

        dates = [s["session_date"] for s in summary.sessions]
        statuses = [s["status"] for s in summary.sessions]
        colors = [COLOR_PRESENT if st == "PRESENT" else COLOR_ABSENT for st in statuses]

        y_pos = range(len(dates))
        ax.barh(y_pos, [1] * len(dates), color=colors, height=0.6)

        for y, status in zip(y_pos, statuses):
            label = "Present" if status == "PRESENT" else "Absent"
            ax.text(1.02, y, label, va="center", ha="left", fontsize=9, color="#0b0b0b")

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(dates, fontsize=9, color=COLOR_MUTED_TEXT)
        ax.set_xlim(0, 1.35)
        ax.set_xticks([])
        ax.invert_yaxis()
        ax.set_title("Session-by-session record", fontsize=10, color=COLOR_MUTED_TEXT, loc="left")

        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(COLOR_AXIS)

        ax.legend(
            handles=[
                Patch(facecolor=COLOR_PRESENT, label="Present"),
                Patch(facecolor=COLOR_ABSENT, label="Absent"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            frameon=False,
            fontsize=9,
        )

    def _draw_donut(self, ax, summary: AttendanceSummary) -> None:
        present, absent = summary.present_count, summary.absent_count
        if summary.total_sessions == 0:
            ax.axis("off")
            return

        values = [present, absent] if (present + absent) > 0 else [1]
        colors = [COLOR_PRESENT, COLOR_ABSENT] if (present + absent) > 0 else ["#c3c2b7"]

        ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.35, edgecolor="#fcfcfb", linewidth=2),
        )
        ax.text(
            0, 0.08, f"{summary.attendance_percentage:.0f}%",
            ha="center", va="center", fontsize=20, fontweight="bold", color="#0b0b0b",
        )
        ax.text(
            0, -0.18, "attendance", ha="center", va="center", fontsize=9, color=COLOR_MUTED_TEXT,
        )
        ax.set_title(
            f"{present}/{summary.total_sessions} sessions present",
            fontsize=10, color=COLOR_MUTED_TEXT,
        )
        ax.set_aspect("equal")


def save_figure(fig: Figure, out_path) -> None:
    fig.savefig(out_path, bbox_inches="tight")


def show_figure(fig: Figure) -> None:
    plt.show()

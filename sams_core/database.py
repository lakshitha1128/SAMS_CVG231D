"""SQLite persistence layer for SAMS.

Owned by: Data Layer role.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator

from sams_core.models import AttendanceRecord, AttendanceStatus, SigningSession, Student

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_index TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    batch TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date TEXT NOT NULL,
    source_image TEXT NOT NULL,
    hall TEXT,
    lecturer TEXT,
    UNIQUE(session_date, source_image)
);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    student_index TEXT NOT NULL REFERENCES students(student_index),
    status TEXT NOT NULL CHECK (status IN ('PRESENT', 'ABSENT')),
    ink_ratio REAL NOT NULL DEFAULT 0,
    signature_image_path TEXT,
    UNIQUE(session_id, student_index)
);
"""


class AttendanceDatabase:
    """Thin repository wrapper around a SQLite attendance database."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # -- students ---------------------------------------------------
    def upsert_students(self, students: list[Student]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO students (student_index, name, batch)
                VALUES (?, ?, ?)
                ON CONFLICT(student_index) DO UPDATE SET
                    name = excluded.name, batch = excluded.batch
                """,
                [(s.index, s.name, s.batch) for s in students],
            )

    # -- sessions -----------------------------------------------------
    def get_or_create_session(self, session: SigningSession) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT id FROM sessions WHERE session_date = ? AND source_image = ?",
                (session.session_date.isoformat(), session.source_image),
            )
            row = cur.fetchone()
            if row is not None:
                return int(row["id"])

            cur = conn.execute(
                """
                INSERT INTO sessions (session_date, source_image, hall, lecturer)
                VALUES (?, ?, ?, ?)
                """,
                (
                    session.session_date.isoformat(),
                    session.source_image,
                    session.hall,
                    session.lecturer,
                ),
            )
            return int(cur.lastrowid)

    # -- attendance ---------------------------------------------------
    def save_attendance(self, session_id: int, records: list[AttendanceRecord]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO attendance
                    (session_id, student_index, status, ink_ratio, signature_image_path)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id, student_index) DO UPDATE SET
                    status = excluded.status,
                    ink_ratio = excluded.ink_ratio,
                    signature_image_path = excluded.signature_image_path
                """,
                [
                    (
                        session_id,
                        r.student_index,
                        r.status.value,
                        r.ink_ratio,
                        r.signature_image_path,
                    )
                    for r in records
                ],
            )

    def attendance_for_student(self, student_index: str) -> list[dict]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT a.status, a.ink_ratio, a.signature_image_path,
                       s.session_date, s.source_image, s.hall, s.lecturer
                FROM attendance a
                JOIN sessions s ON s.id = a.session_id
                WHERE a.student_index = ?
                ORDER BY s.session_date
                """,
                (student_index,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_student(self, student_index: str) -> Student | None:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT student_index, name, batch FROM students WHERE student_index = ?",
                (student_index,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return Student(index=row["student_index"], name=row["name"], batch=row["batch"] or "")

    def all_sessions(self) -> list[SigningSession]:
        with self._connect() as conn:
            cur = conn.execute("SELECT session_date, source_image, hall, lecturer FROM sessions ORDER BY session_date")
            return [
                SigningSession(
                    session_date=date.fromisoformat(row["session_date"]),
                    source_image=row["source_image"],
                    hall=row["hall"] or "",
                    lecturer=row["lecturer"] or "",
                )
                for row in cur.fetchall()
            ]

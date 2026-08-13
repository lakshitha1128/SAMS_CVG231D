from datetime import date

from sams_core.database import AttendanceDatabase
from sams_core.models import AttendanceRecord, AttendanceStatus, SigningSession, Student


def make_db(tmp_path):
    return AttendanceDatabase(tmp_path / "test.db")


def test_upsert_and_get_student(tmp_path):
    db = make_db(tmp_path)
    db.upsert_students([Student(index="001", name="John Snow", batch="2016.1")])

    student = db.get_student("001")
    assert student is not None
    assert student.name == "John Snow"

    # Upsert again with a changed name should update, not duplicate.
    db.upsert_students([Student(index="001", name="John Snow Jr", batch="2016.1")])
    assert db.get_student("001").name == "John Snow Jr"


def test_session_is_deduplicated_by_date_and_source(tmp_path):
    db = make_db(tmp_path)
    session = SigningSession(session_date=date(2019, 7, 12), source_image="12.07.2019.png")

    id_a = db.get_or_create_session(session)
    id_b = db.get_or_create_session(session)
    assert id_a == id_b


def test_attendance_roundtrip(tmp_path):
    db = make_db(tmp_path)
    db.upsert_students([Student(index="001", name="John Snow")])
    session = SigningSession(session_date=date(2019, 7, 12), source_image="12.07.2019.png")
    session_id = db.get_or_create_session(session)

    db.save_attendance(
        session_id,
        [AttendanceRecord(student_index="001", status=AttendanceStatus.PRESENT, ink_ratio=0.05)],
    )

    rows = db.attendance_for_student("001")
    assert len(rows) == 1
    assert rows[0]["status"] == "PRESENT"
    assert rows[0]["session_date"] == "2019-07-12"


def test_attendance_upsert_overwrites_status(tmp_path):
    db = make_db(tmp_path)
    db.upsert_students([Student(index="001", name="John Snow")])
    session = SigningSession(session_date=date(2019, 7, 12), source_image="12.07.2019.png")
    session_id = db.get_or_create_session(session)

    db.save_attendance(
        session_id, [AttendanceRecord(student_index="001", status=AttendanceStatus.ABSENT)]
    )
    db.save_attendance(
        session_id,
        [AttendanceRecord(student_index="001", status=AttendanceStatus.PRESENT, ink_ratio=0.2)],
    )

    rows = db.attendance_for_student("001")
    assert len(rows) == 1
    assert rows[0]["status"] == "PRESENT"

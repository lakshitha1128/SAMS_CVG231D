import cv2

from sams_core.config import AppConfig, TableLayout
from sams_core.image_pipeline.pipeline import AttendanceImagePipeline, parse_session_date
from sams_core.models import AttendanceStatus, Student
from tests.generate_sample_sheet import generate_signing_sheet

STUDENTS = [
    Student(index="001", name="John Snow"),
    Student(index="007", name="James Bond"),
    Student(index="009", name="Andare"),
]
SIGNED = [True, True, False]  # Andare is absent, matching the coursework example


def test_parse_session_date_from_filename():
    assert parse_session_date("10.07.2019.png").isoformat() == "2019-07-10"
    assert parse_session_date("data/sheets/1.7.2019.png").isoformat() == "2019-07-01"


def test_full_pipeline_matches_expected_attendance(tmp_path):
    image = generate_signing_sheet([s.name for s in STUDENTS], SIGNED)
    image_path = tmp_path / "10.07.2019.png"
    cv2.imwrite(str(image_path), image)

    config = AppConfig(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        signing_sheets_dir=tmp_path / "data" / "signing_sheets",
        signatures_dir=tmp_path / "data" / "signatures",
        processed_dir=tmp_path / "data" / "processed",
        db_path=tmp_path / "data" / "sams.db",
        table_layout=TableLayout(),
    )
    config.ensure_directories()

    pipeline = AttendanceImagePipeline(config)
    result = pipeline.process(image_path, STUDENTS)

    assert [r.student_index for r in result.records] == [s.index for s in STUDENTS]
    statuses = {r.student_index: r.status for r in result.records}
    assert statuses["001"] is AttendanceStatus.PRESENT
    assert statuses["007"] is AttendanceStatus.PRESENT
    assert statuses["009"] is AttendanceStatus.ABSENT
    assert result.session.session_date.isoformat() == "2019-07-10"

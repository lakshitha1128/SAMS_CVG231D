import pytest

from sams_core.xml_loader import InfoXmlError, StudentRepository


def test_loads_students_from_real_info_xml(project_root):
    repo = StudentRepository.from_xml(project_root / "info.xml")
    assert len(repo) == 6
    student = repo.get("10009301")
    assert student is not None
    assert student.name == "C W M A Shehan Abeyrathne"
    assert student.batch == "2016.1"


def test_missing_file_raises(tmp_path):
    with pytest.raises(InfoXmlError):
        StudentRepository.from_xml(tmp_path / "does_not_exist.xml")


def test_malformed_xml_raises(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<not><valid", encoding="utf-8")
    with pytest.raises(InfoXmlError):
        StudentRepository.from_xml(bad)


def test_missing_student_fields_raise(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text(
        """<?xml version="1.0"?>
        <nsbm><students><batches><batch id="1">
            <student><index>001</index></student>
        </batch></batches></students></nsbm>
        """,
        encoding="utf-8",
    )
    with pytest.raises(InfoXmlError):
        StudentRepository.from_xml(bad)

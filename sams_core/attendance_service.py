"""Application service that ties the XML roster, image pipeline and
database together -- the glue used by sams.py.

Owned by: Project Lead / Architecture role.
"""

from __future__ import annotations

from pathlib import Path

from sams_core.config import AppConfig
from sams_core.database import AttendanceDatabase
from sams_core.image_pipeline.pipeline import AttendanceImagePipeline, PipelineResult
from sams_core.xml_loader import StudentRepository


class AttendanceService:
    def __init__(self, config: AppConfig):
        self.config = config
        config.ensure_directories()
        self.db = AttendanceDatabase(config.db_path)
        self.pipeline = AttendanceImagePipeline(config)

    def process_sheet(self, image_path: str | Path, info_xml_path: str | Path) -> PipelineResult:
        roster = StudentRepository.from_xml(info_xml_path)
        students = roster.all()
        self.db.upsert_students(students)

        result = self.pipeline.process(image_path, students)

        session_id = self.db.get_or_create_session(result.session)
        self.db.save_attendance(session_id, result.records)
        return result

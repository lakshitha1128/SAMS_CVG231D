"""Loads student roster information from the info.xml file supplied by
the admin staff (see Figure 1 of the CS402.3 coursework spec).

Owned by: Data Layer role.

Expected structure::

    <?xml version="1.0"?>
    <nsbm>
        <students>
            <batches>
                <batch id="2016.1">
                    <student>
                        <index>10000409</index>
                        <name>M S Dilshanika Perera</name>
                    </student>
                    ...
                </batch>
            </batches>
        </students>
    </nsbm>

Note: the coursework's Figure 1 shows a batch encoded as a bare ``<15>``
tag. XML element names may not start with a digit, so that figure is a
simplification -- here the batch is represented as ``<batch id="...">``
instead, which is valid XML while preserving the same information.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from sams_core.models import Student


class InfoXmlError(ValueError):
    """Raised when info.xml is missing or malformed."""


class StudentRepository:
    """Reads the student roster from an info.xml file."""

    def __init__(self, students: list[Student]):
        self._students = students
        self._by_index = {s.index: s for s in students}

    @classmethod
    def from_xml(cls, xml_path: str | Path) -> "StudentRepository":
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise InfoXmlError(f"info.xml not found at: {xml_path}")

        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            raise InfoXmlError(f"Could not parse {xml_path}: {exc}") from exc

        root = tree.getroot()
        students: list[Student] = []
        for batch in root.findall(".//batch"):
            batch_id = batch.get("id", "")
            for student_el in batch.findall("student"):
                index_el = student_el.find("index")
                name_el = student_el.find("name")
                if index_el is None or name_el is None:
                    raise InfoXmlError(
                        "Each <student> requires <index> and <name>"
                    )
                index = (index_el.text or "").strip()
                name = (name_el.text or "").strip()
                students.append(Student(index=index, name=name, batch=batch_id))

        if not students:
            raise InfoXmlError(f"No <student> entries found in {xml_path}")

        return cls(students)

    def all(self) -> list[Student]:
        return list(self._students)

    def get(self, index: str) -> Student | None:
        return self._by_index.get(index)

    def __len__(self) -> int:
        return len(self._students)

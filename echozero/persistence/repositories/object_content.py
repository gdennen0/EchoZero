"""Object/content repositories for timeline truth records.
Exists because objects and content are app truth while layers are arrangement rows.
Connects SQLite object-content tables to application-level object/content projections.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from echozero.errors import PersistenceError
from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import (
    ObjectCandidateRecord,
    ObjectContentRecord,
    TimelineObjectRecord,
)


class TimelineObjectRepository(BaseRepository[TimelineObjectRecord]):
    """Read and write timeline object records."""

    def _from_row(self, row: sqlite3.Row) -> TimelineObjectRecord:
        return TimelineObjectRecord(
            id=row["id"],
            song_version_id=row["song_version_id"],
            name=row["name"],
            object_kind=row["object_kind"],
            main_content_id=row["main_content_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def create(self, record: TimelineObjectRecord) -> None:
        self._execute(
            "INSERT INTO timeline_objects "
            "(id, song_version_id, name, object_kind, main_content_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.song_version_id,
                record.name,
                record.object_kind,
                record.main_content_id,
                record.created_at.isoformat(),
            ),
        )

    def get(self, object_id: str) -> TimelineObjectRecord | None:
        row = self._fetchone(
            "SELECT id, song_version_id, name, object_kind, main_content_id, created_at "
            "FROM timeline_objects WHERE id = ?",
            (object_id,),
        )
        return None if row is None else self._from_row(row)

    def list_by_version(self, song_version_id: str) -> list[TimelineObjectRecord]:
        rows = self._fetchall(
            "SELECT id, song_version_id, name, object_kind, main_content_id, created_at "
            "FROM timeline_objects WHERE song_version_id = ? ORDER BY created_at",
            (song_version_id,),
        )
        return [self._from_row(row) for row in rows]

    def update(self, record: TimelineObjectRecord) -> None:
        self._execute(
            "UPDATE timeline_objects SET name = ?, object_kind = ?, main_content_id = ? "
            "WHERE id = ?",
            (record.name, record.object_kind, record.main_content_id, record.id),
        )

    def delete(self, object_id: str) -> None:
        """Delete one timeline object and its cascaded content rows."""

        self._execute("DELETE FROM timeline_objects WHERE id = ?", (object_id,))


class ObjectContentRepository(BaseRepository[ObjectContentRecord]):
    """Read and write object content records."""

    def _from_row(self, row: sqlite3.Row) -> ObjectContentRecord:
        return ObjectContentRecord(
            id=row["id"],
            object_id=row["object_id"],
            revision_id=row["revision_id"],
            content_kind=row["content_kind"],
            payload=json.loads(row["payload_json"] or "{}"),
            source_ref=(json.loads(row["source_ref_json"]) if row["source_ref_json"] else None),
            analysis_build=(
                json.loads(row["analysis_build_json"]) if row["analysis_build_json"] else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def create(self, record: ObjectContentRecord) -> None:
        self._validate_source_ref(record.source_ref)
        self._execute(
            "INSERT INTO object_contents "
            "(id, object_id, revision_id, content_kind, payload_json, source_ref_json, "
            "analysis_build_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.object_id,
                record.revision_id,
                record.content_kind,
                json.dumps(record.payload),
                json.dumps(record.source_ref) if record.source_ref else None,
                json.dumps(record.analysis_build) if record.analysis_build else None,
                record.created_at.isoformat(),
            ),
        )

    def update(self, record: ObjectContentRecord) -> None:
        """Overwrite one object content row."""

        self._validate_source_ref(record.source_ref)
        self._execute(
            "UPDATE object_contents SET object_id = ?, revision_id = ?, content_kind = ?, "
            "payload_json = ?, source_ref_json = ?, analysis_build_json = ?, created_at = ? "
            "WHERE id = ?",
            (
                record.object_id,
                record.revision_id,
                record.content_kind,
                json.dumps(record.payload),
                json.dumps(record.source_ref) if record.source_ref else None,
                json.dumps(record.analysis_build) if record.analysis_build else None,
                record.created_at.isoformat(),
                record.id,
            ),
        )

    def get(self, content_id: str) -> ObjectContentRecord | None:
        row = self._fetchone(
            "SELECT id, object_id, revision_id, content_kind, payload_json, "
            "source_ref_json, analysis_build_json, created_at "
            "FROM object_contents WHERE id = ?",
            (content_id,),
        )
        return None if row is None else self._from_row(row)

    def list_by_object(self, object_id: str) -> list[ObjectContentRecord]:
        rows = self._fetchall(
            "SELECT id, object_id, revision_id, content_kind, payload_json, "
            "source_ref_json, analysis_build_json, created_at "
            "FROM object_contents WHERE object_id = ? ORDER BY created_at",
            (object_id,),
        )
        return [self._from_row(row) for row in rows]

    def delete(self, content_id: str) -> None:
        """Delete one object content row."""

        self._execute("DELETE FROM object_contents WHERE id = ?", (content_id,))

    def _validate_source_ref(self, source_ref: dict[str, object] | None) -> None:
        if not source_ref:
            return
        object_id = str(source_ref.get("object_id") or "").strip()
        content_id = str(source_ref.get("content_id") or "").strip()
        revision_id = str(source_ref.get("revision_id") or "").strip()
        if not object_id or not content_id or not revision_id:
            raise PersistenceError(
                "Object content source_ref must include object_id, content_id, and revision_id."
            )
        row = self._fetchone(
            "SELECT object_id, revision_id FROM object_contents WHERE id = ?",
            (content_id,),
        )
        if row is None:
            raise PersistenceError(
                f"Object content source_ref points at missing content {content_id!r}."
            )
        if row["object_id"] != object_id or row["revision_id"] != revision_id:
            raise PersistenceError(
                "Object content source_ref points at inconsistent content " f"{content_id!r}."
            )


class ObjectCandidateRepository(BaseRepository[ObjectCandidateRecord]):
    """Read and write object candidate records."""

    def _from_row(self, row: sqlite3.Row) -> ObjectCandidateRecord:
        return ObjectCandidateRecord(
            id=row["id"],
            object_id=row["object_id"],
            content_id=row["content_id"],
            label=row["label"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def create(self, record: ObjectCandidateRecord) -> None:
        self._execute(
            "INSERT INTO object_candidates "
            "(id, object_id, content_id, label, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                record.id,
                record.object_id,
                record.content_id,
                record.label,
                record.created_at.isoformat(),
            ),
        )

    def list_by_object(self, object_id: str) -> list[ObjectCandidateRecord]:
        rows = self._fetchall(
            "SELECT id, object_id, content_id, label, created_at "
            "FROM object_candidates WHERE object_id = ? ORDER BY created_at",
            (object_id,),
        )
        return [self._from_row(row) for row in rows]

    def delete_by_object(self, object_id: str) -> None:
        """Delete all candidate rows for one timeline object."""

        self._execute("DELETE FROM object_candidates WHERE object_id = ?", (object_id,))

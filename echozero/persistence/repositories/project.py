"""
ProjectRepository: CRUD operations for Project entities in SQLite.
Exists because the domain layer must not know about SQL — repositories translate
between frozen dataclasses and database rows at the persistence boundary.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import Project, ProjectSettings


class ProjectRepository(BaseRepository[Project]):
    """Read and write Project entities to the projects table."""

    def _from_row(self, row: sqlite3.Row) -> Project:
        """Convert a database row to a Project entity."""
        return Project(
            id=row['id'],
            name=row['name'],
            settings=ProjectSettings(
                sample_rate=row['sample_rate'],
                bpm=row['bpm'],
                bpm_confidence=row['bpm_confidence'],
                timecode_fps=row['timecode_fps'],
            ),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
        )

    def create(self, project: Project) -> None:
        """Insert a new project row."""
        self._execute(
            "INSERT INTO projects "
            "(id, name, sample_rate, bpm, bpm_confidence, timecode_fps, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                project.id,
                project.name,
                project.settings.sample_rate,
                project.settings.bpm,
                project.settings.bpm_confidence,
                project.settings.timecode_fps,
                project.created_at.isoformat(),
                project.updated_at.isoformat(),
            ),
        )

    def get(self, project_id: str) -> Project | None:
        """Return a project by ID, or None if not found."""
        row = self._fetchone(
            "SELECT id, name, sample_rate, bpm, bpm_confidence, timecode_fps, "
            "created_at, updated_at FROM projects WHERE id = ?",
            (project_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list(self) -> list[Project]:
        """Return all projects ordered by name."""
        rows = self._fetchall(
            "SELECT id, name, sample_rate, bpm, bpm_confidence, timecode_fps, "
            "created_at, updated_at FROM projects ORDER BY name"
        )
        return [self._from_row(r) for r in rows]

    def update(self, project: Project) -> None:
        """Overwrite a project row with updated values."""
        self._execute(
            "UPDATE projects SET name = ?, sample_rate = ?, bpm = ?, "
            "bpm_confidence = ?, timecode_fps = ?, updated_at = ? WHERE id = ?",
            (
                project.name,
                project.settings.sample_rate,
                project.settings.bpm,
                project.settings.bpm_confidence,
                project.settings.timecode_fps,
                project.updated_at.isoformat(),
                project.id,
            ),
        )

    def delete(self, project_id: str) -> None:
        """Delete a project by ID. Cascades to songs, versions, layers, takes."""
        self._execute("DELETE FROM projects WHERE id = ?", (project_id,))

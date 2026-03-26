"""
PipelineConfigRepository: CRUD for SongPipelineConfig entities in SQLite.
Exists because per-song pipeline configurations (the EZ1 ActionSet replacement) need
durable storage. Bindings are stored as JSON blobs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import SongPipelineConfig


class PipelineConfigRepository(BaseRepository[SongPipelineConfig]):
    """Read and write SongPipelineConfig entities to the song_pipeline_configs table."""

    def _from_row(self, row: sqlite3.Row) -> SongPipelineConfig:
        """Convert a database row to a SongPipelineConfig entity."""
        return SongPipelineConfig(
            id=row['id'],
            song_version_id=row['song_version_id'],
            pipeline_id=row['pipeline_id'],
            bindings=json.loads(row['bindings']),
            created_at=datetime.fromisoformat(row['created_at']),
        )

    def create(self, config: SongPipelineConfig) -> None:
        """Insert a new pipeline config row."""
        self._execute(
            "INSERT INTO song_pipeline_configs "
            "(id, song_version_id, pipeline_id, bindings, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                config.id,
                config.song_version_id,
                config.pipeline_id,
                json.dumps(config.bindings),
                config.created_at.isoformat(),
            ),
        )

    def get(self, config_id: str) -> SongPipelineConfig | None:
        """Return a pipeline config by ID, or None if not found."""
        row = self._fetchone(
            "SELECT id, song_version_id, pipeline_id, bindings, created_at "
            "FROM song_pipeline_configs WHERE id = ?",
            (config_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list_by_version(self, song_version_id: str) -> list[SongPipelineConfig]:
        """Return all pipeline configs for a song version."""
        rows = self._fetchall(
            "SELECT id, song_version_id, pipeline_id, bindings, created_at "
            "FROM song_pipeline_configs WHERE song_version_id = ? ORDER BY created_at",
            (song_version_id,),
        )
        return [self._from_row(r) for r in rows]

    def delete(self, config_id: str) -> None:
        """Delete a pipeline config by ID."""
        self._execute(
            "DELETE FROM song_pipeline_configs WHERE id = ?", (config_id,)
        )

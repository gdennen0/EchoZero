"""Song default pipeline config repository for SQLite persistence.
Exists because songs own default pipeline profiles separate from version-effective copies.
Connects song-level settings persistence to version creation and copy/apply flows.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import SongDefaultPipelineConfigRecord


class SongDefaultPipelineConfigRepository(BaseRepository[SongDefaultPipelineConfigRecord]):
    """Read and write song-owned default pipeline configs."""

    def _from_row(self, row: sqlite3.Row) -> SongDefaultPipelineConfigRecord:
        return SongDefaultPipelineConfigRecord(
            id=row["id"],
            song_id=row["song_id"],
            template_id=row["template_id"],
            name=row["name"],
            graph_json=row["graph_json"],
            outputs_json=row["outputs_json"],
            knob_values=json.loads(row["knob_values_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            block_overrides=json.loads(row["block_overrides_json"]),
        )

    def create(self, config: SongDefaultPipelineConfigRecord) -> None:
        self._execute(
            "INSERT INTO song_default_pipeline_configs "
            "(id, song_id, template_id, name, graph_json, outputs_json, "
            "knob_values_json, block_overrides_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                config.id,
                config.song_id,
                config.template_id,
                config.name,
                config.graph_json,
                config.outputs_json,
                json.dumps(config.knob_values),
                json.dumps(config.block_overrides),
                config.created_at.isoformat(),
                config.updated_at.isoformat(),
            ),
        )

    def get(self, config_id: str) -> SongDefaultPipelineConfigRecord | None:
        row = self._fetchone(
            "SELECT id, song_id, template_id, name, graph_json, outputs_json, "
            "knob_values_json, block_overrides_json, created_at, updated_at "
            "FROM song_default_pipeline_configs WHERE id = ?",
            (config_id,),
        )
        return None if row is None else self._from_row(row)

    def list_by_song(self, song_id: str) -> list[SongDefaultPipelineConfigRecord]:
        rows = self._fetchall(
            "SELECT id, song_id, template_id, name, graph_json, outputs_json, "
            "knob_values_json, block_overrides_json, created_at, updated_at "
            "FROM song_default_pipeline_configs WHERE song_id = ? ORDER BY created_at",
            (song_id,),
        )
        return [self._from_row(row) for row in rows]

    def update(self, config: SongDefaultPipelineConfigRecord) -> None:
        self._execute(
            "UPDATE song_default_pipeline_configs SET "
            "name = ?, graph_json = ?, outputs_json = ?, knob_values_json = ?, "
            "block_overrides_json = ?, updated_at = ? WHERE id = ?",
            (
                config.name,
                config.graph_json,
                config.outputs_json,
                json.dumps(config.knob_values),
                json.dumps(config.block_overrides),
                config.updated_at.isoformat(),
                config.id,
            ),
        )

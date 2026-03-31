"""
PipelineConfigRepository: CRUD for PipelineConfigRecord entities in SQLite.
Exists because pipeline configurations are first-class persistent project state.
The user's pipeline settings live here — not reconstructed from templates on every run.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import PipelineConfigRecord


class PipelineConfigRepository(BaseRepository[PipelineConfigRecord]):
    """Read and write PipelineConfigRecord entities to the pipeline_configs table."""

    def _from_row(self, row: sqlite3.Row) -> PipelineConfigRecord:
        """Convert a database row to a PipelineConfigRecord entity."""
        return PipelineConfigRecord(
            id=row['id'],
            song_version_id=row['song_version_id'],
            template_id=row['template_id'],
            name=row['name'],
            graph_json=row['graph_json'],
            outputs_json=row['outputs_json'],
            knob_values=json.loads(row['knob_values_json']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            block_overrides=json.loads(row['block_overrides_json']),
        )

    def create(self, config: PipelineConfigRecord) -> None:
        """Insert a new pipeline config."""
        self._execute(
            "INSERT INTO pipeline_configs "
            "(id, song_version_id, template_id, name, graph_json, outputs_json, "
            "knob_values_json, block_overrides_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                config.id,
                config.song_version_id,
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

    def get(self, config_id: str) -> PipelineConfigRecord | None:
        """Return a pipeline config by ID, or None if not found."""
        row = self._fetchone(
            "SELECT id, song_version_id, template_id, name, graph_json, outputs_json, "
            "knob_values_json, block_overrides_json, created_at, updated_at "
            "FROM pipeline_configs WHERE id = ?",
            (config_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list_by_version(self, song_version_id: str) -> list[PipelineConfigRecord]:
        """Return all pipeline configs for a song version, ordered by creation."""
        rows = self._fetchall(
            "SELECT id, song_version_id, template_id, name, graph_json, outputs_json, "
            "knob_values_json, block_overrides_json, created_at, updated_at "
            "FROM pipeline_configs WHERE song_version_id = ? ORDER BY created_at",
            (song_version_id,),
        )
        return [self._from_row(r) for r in rows]

    def list_by_template(self, template_id: str) -> list[PipelineConfigRecord]:
        """Return all configs created from a given template. For migration."""
        rows = self._fetchall(
            "SELECT id, song_version_id, template_id, name, graph_json, outputs_json, "
            "knob_values_json, block_overrides_json, created_at, updated_at "
            "FROM pipeline_configs WHERE template_id = ? ORDER BY created_at",
            (template_id,),
        )
        return [self._from_row(r) for r in rows]

    def update(self, config: PipelineConfigRecord) -> None:
        """Update an existing pipeline config (settings changed)."""
        self._execute(
            "UPDATE pipeline_configs SET "
            "name = ?, graph_json = ?, outputs_json = ?, "
            "knob_values_json = ?, block_overrides_json = ?, updated_at = ? "
            "WHERE id = ?",
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

    def delete(self, config_id: str) -> None:
        """Delete a pipeline config by ID."""
        self._execute(
            "DELETE FROM pipeline_configs WHERE id = ?", (config_id,)
        )

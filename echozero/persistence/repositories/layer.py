"""
LayerRepository: CRUD operations for LayerRecord entities in SQLite.
Exists because persistence layers carry UI state (color, order, visibility) that the
pipeline engine ignores. The repository translates between LayerRecord DTOs and SQL rows.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import LayerRecord


class LayerRepository(BaseRepository[LayerRecord]):
    """Read and write LayerRecord entities to the layers table."""

    def _from_row(self, row: sqlite3.Row) -> LayerRecord:
        """Convert a database row to a LayerRecord entity."""
        source_pipeline_raw = row['source_pipeline']
        source_pipeline: dict[str, Any] | None = None
        if source_pipeline_raw is not None:
            source_pipeline = json.loads(source_pipeline_raw)

        return LayerRecord(
            id=row['id'],
            song_version_id=row['song_version_id'],
            name=row['name'],
            layer_type=row['layer_type'],
            color=row['color'],
            order=row['order'],
            visible=bool(row['visible']),
            locked=bool(row['locked']),
            parent_layer_id=row['parent_layer_id'],
            source_pipeline=source_pipeline,
            created_at=datetime.fromisoformat(row['created_at']),
        )

    def create(self, layer: LayerRecord) -> None:
        """Insert a new layer row."""
        self._execute(
            "INSERT INTO layers "
            '(id, song_version_id, name, layer_type, color, "order", '
            "visible, locked, parent_layer_id, source_pipeline, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                layer.id,
                layer.song_version_id,
                layer.name,
                layer.layer_type,
                layer.color,
                layer.order,
                int(layer.visible),
                int(layer.locked),
                layer.parent_layer_id,
                json.dumps(layer.source_pipeline) if layer.source_pipeline else None,
                layer.created_at.isoformat(),
            ),
        )

    def get(self, layer_id: str) -> LayerRecord | None:
        """Return a layer by ID, or None if not found."""
        row = self._fetchone(
            "SELECT id, song_version_id, name, layer_type, color, "
            '"order", visible, locked, parent_layer_id, source_pipeline, created_at '
            "FROM layers WHERE id = ?",
            (layer_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list_by_version(self, song_version_id: str) -> list[LayerRecord]:
        """Return all layers for a song version, ordered by position."""
        rows = self._fetchall(
            "SELECT id, song_version_id, name, layer_type, color, "
            '"order", visible, locked, parent_layer_id, source_pipeline, created_at '
            'FROM layers WHERE song_version_id = ? ORDER BY "order"',
            (song_version_id,),
        )
        return [self._from_row(r) for r in rows]

    def update(self, layer: LayerRecord) -> None:
        """Overwrite a layer row with updated values."""
        self._execute(
            "UPDATE layers SET name = ?, layer_type = ?, color = ?, "
            '"order" = ?, visible = ?, locked = ?, parent_layer_id = ?, '
            "source_pipeline = ? WHERE id = ?",
            (
                layer.name,
                layer.layer_type,
                layer.color,
                layer.order,
                int(layer.visible),
                int(layer.locked),
                layer.parent_layer_id,
                json.dumps(layer.source_pipeline) if layer.source_pipeline else None,
                layer.id,
            ),
        )

    def delete(self, layer_id: str) -> None:
        """Delete a layer by ID. Cascades to takes."""
        self._execute("DELETE FROM layers WHERE id = ?", (layer_id,))

    def reorder(self, song_version_id: str, layer_ids: list[str]) -> None:
        """Set the order of layers in a version by their ID sequence."""
        for i, layer_id in enumerate(layer_ids):
            self._execute(
                'UPDATE layers SET "order" = ? '
                "WHERE id = ? AND song_version_id = ?",
                (i, layer_id, song_version_id),
            )

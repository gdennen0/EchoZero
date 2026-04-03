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
        source_pipeline_raw = row['source_pipeline']
        source_pipeline: dict[str, Any] | None = None
        if source_pipeline_raw is not None:
            source_pipeline = json.loads(source_pipeline_raw)

        state_flags_raw = row['state_flags_json'] if 'state_flags_json' in row.keys() else '{}'
        provenance_raw = row['provenance_json'] if 'provenance_json' in row.keys() else '{}'

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
            state_flags=json.loads(state_flags_raw or '{}'),
            provenance=json.loads(provenance_raw or '{}'),
        )

    def create(self, layer: LayerRecord) -> None:
        self._execute(
            "INSERT INTO layers "
            '(id, song_version_id, name, layer_type, color, "order", '
            "visible, locked, parent_layer_id, source_pipeline, state_flags_json, provenance_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                json.dumps(layer.state_flags),
                json.dumps(layer.provenance),
                layer.created_at.isoformat(),
            ),
        )

    def get(self, layer_id: str) -> LayerRecord | None:
        row = self._fetchone(
            "SELECT id, song_version_id, name, layer_type, color, "
            '"order", visible, locked, parent_layer_id, source_pipeline, state_flags_json, provenance_json, created_at '
            "FROM layers WHERE id = ?",
            (layer_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list_by_version(self, song_version_id: str) -> list[LayerRecord]:
        rows = self._fetchall(
            "SELECT id, song_version_id, name, layer_type, color, "
            '"order", visible, locked, parent_layer_id, source_pipeline, state_flags_json, provenance_json, created_at '
            'FROM layers WHERE song_version_id = ? ORDER BY "order"',
            (song_version_id,),
        )
        return [self._from_row(r) for r in rows]

    def update(self, layer: LayerRecord) -> None:
        self._execute(
            "UPDATE layers SET name = ?, layer_type = ?, color = ?, "
            '"order" = ?, visible = ?, locked = ?, parent_layer_id = ?, '
            "source_pipeline = ?, state_flags_json = ?, provenance_json = ? WHERE id = ?",
            (
                layer.name,
                layer.layer_type,
                layer.color,
                layer.order,
                int(layer.visible),
                int(layer.locked),
                layer.parent_layer_id,
                json.dumps(layer.source_pipeline) if layer.source_pipeline else None,
                json.dumps(layer.state_flags),
                json.dumps(layer.provenance),
                layer.id,
            ),
        )

    def delete(self, layer_id: str) -> None:
        self._execute("DELETE FROM layers WHERE id = ?", (layer_id,))

    def reorder(self, song_version_id: str, layer_ids: list[str]) -> None:
        for i, layer_id in enumerate(layer_ids):
            self._execute(
                'UPDATE layers SET "order" = ? '
                "WHERE id = ? AND song_version_id = ?",
                (i, layer_id, song_version_id),
            )

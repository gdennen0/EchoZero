"""
TakeRepository: CRUD for Take entities stored as SQLite rows with JSON data blobs.
Exists because Takes bridge the domain layer (frozen EventData/AudioData) and persistence
(SQLite rows). The data_json column stores serialized EventData or AudioData using the
existing serialization module; the repository handles the mapping transparently.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime

from echozero.errors import PersistenceError
from echozero.persistence.base import BaseRepository
from echozero.serialization import serialize_take_data, deserialize_take_data
from echozero.takes import Take, TakeSource

logger = logging.getLogger(__name__)


class TakeRepository(BaseRepository[Take]):
    """Read and write domain Take objects to the takes table."""

    def _from_row(self, row: sqlite3.Row) -> Take:
        """Convert a database row to a domain Take."""
        source = self._deserialize_source(row['source_json'])
        if row['data_json'] is None:
            raise PersistenceError("Take has no data")
        data = deserialize_take_data(json.loads(row['data_json']))

        return Take(
            id=row['id'],
            label=row['label'],
            data=data,
            origin=row['origin'],
            source=source,
            created_at=datetime.fromisoformat(row['created_at']),
            is_main=bool(row['is_main']),
            is_archived=bool(row['is_archived']),
            notes=row['notes'] or "",
        )

    def create(self, layer_id: str, take: Take) -> None:
        """Insert a take row. The layer_id FK is a persistence-only concern."""
        self._execute(
            "INSERT INTO takes "
            "(id, layer_id, label, origin, is_main, is_archived, "
            "source_json, data_json, created_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                take.id,
                layer_id,
                take.label,
                take.origin,
                int(take.is_main),
                int(take.is_archived),
                self._serialize_source(take.source),
                json.dumps(serialize_take_data(take.data)),
                take.created_at.isoformat(),
                take.notes,
            ),
        )

    def get(self, take_id: str) -> Take | None:
        """Return a take by ID, or None if not found."""
        row = self._fetchone(
            "SELECT id, layer_id, label, origin, is_main, is_archived, "
            "source_json, data_json, created_at, notes "
            "FROM takes WHERE id = ?",
            (take_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list_by_layer(self, layer_id: str) -> list[Take]:
        """Return all takes for a layer, ordered by creation time.

        Corrupt rows (e.g. NULL data_json) are skipped with a warning rather than
        crashing the entire listing.
        """
        rows = self._fetchall(
            "SELECT id, layer_id, label, origin, is_main, is_archived, "
            "source_json, data_json, created_at, notes "
            "FROM takes WHERE layer_id = ? ORDER BY created_at",
            (layer_id,),
        )
        results = []
        for r in rows:
            try:
                results.append(self._from_row(r))
            except Exception as exc:
                logger.warning("Skipping corrupt take %s: %s", r['id'], exc)
        return results

    def update(self, take: Take) -> None:
        """Overwrite a take's mutable fields (label, origin, is_main, is_archived, notes, source, data)."""
        self._execute(
            "UPDATE takes SET label = ?, origin = ?, is_main = ?, is_archived = ?, "
            "source_json = ?, data_json = ?, notes = ? WHERE id = ?",
            (
                take.label,
                take.origin,
                int(take.is_main),
                int(take.is_archived),
                self._serialize_source(take.source),
                json.dumps(serialize_take_data(take.data)),
                take.notes,
                take.id,
            ),
        )

    def delete(self, take_id: str) -> None:
        """Delete a take by ID."""
        self._execute("DELETE FROM takes WHERE id = ?", (take_id,))

    def get_main(self, layer_id: str) -> Take | None:
        """Return the main take for a layer, or None if no main exists.

        Returns None if the main take row is corrupt rather than raising.
        """
        row = self._fetchone(
            "SELECT id, layer_id, label, origin, is_main, is_archived, "
            "source_json, data_json, created_at, notes "
            "FROM takes WHERE layer_id = ? AND is_main = 1",
            (layer_id,),
        )
        if row is None:
            return None
        try:
            return self._from_row(row)
        except Exception as exc:
            logger.warning("Main take for layer %s is corrupt: %s", layer_id, exc)
            return None

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _serialize_source(source: TakeSource | None) -> str | None:
        """Serialize a TakeSource to a JSON string, or None."""
        if source is None:
            return None
        return json.dumps(source.to_dict())

    @staticmethod
    def _deserialize_source(raw: str | None) -> TakeSource | None:
        """Reconstruct a TakeSource from a JSON string, or None."""
        if raw is None:
            return None
        return TakeSource.from_dict(json.loads(raw))

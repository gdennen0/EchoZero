"""TimelineRegionRepository: CRUD for timeline region spans in SQLite.
Exists to persist ruler-region truth per song version outside layer/take payload blobs.
Connects region manager edits and timeline batch scopes to storage-backed project state.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import TimelineRegionRecord


class TimelineRegionRepository(BaseRepository[TimelineRegionRecord]):
    """Read and write TimelineRegionRecord rows from the timeline_regions table."""

    def _from_row(self, row: sqlite3.Row) -> TimelineRegionRecord:
        return TimelineRegionRecord(
            id=row["id"],
            song_version_id=row["song_version_id"],
            label=row["label"],
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
            color=row["color"],
            order_index=int(row["order_index"]),
            kind=row["kind"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def create(self, region: TimelineRegionRecord) -> None:
        self._execute(
            "INSERT INTO timeline_regions "
            "(id, song_version_id, label, start_seconds, end_seconds, color, order_index, kind, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                region.id,
                region.song_version_id,
                region.label,
                float(region.start_seconds),
                float(region.end_seconds),
                region.color,
                int(region.order_index),
                region.kind,
                region.created_at.isoformat(),
            ),
        )

    def get(self, region_id: str) -> TimelineRegionRecord | None:
        row = self._fetchone(
            "SELECT id, song_version_id, label, start_seconds, end_seconds, color, order_index, kind, created_at "
            "FROM timeline_regions WHERE id = ?",
            (region_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list_by_version(self, song_version_id: str) -> list[TimelineRegionRecord]:
        rows = self._fetchall(
            "SELECT id, song_version_id, label, start_seconds, end_seconds, color, order_index, kind, created_at "
            "FROM timeline_regions WHERE song_version_id = ? "
            "ORDER BY order_index, start_seconds, end_seconds, id",
            (song_version_id,),
        )
        return [self._from_row(row) for row in rows]

    def update(self, region: TimelineRegionRecord) -> None:
        self._execute(
            "UPDATE timeline_regions SET "
            "label = ?, start_seconds = ?, end_seconds = ?, color = ?, order_index = ?, kind = ? "
            "WHERE id = ?",
            (
                region.label,
                float(region.start_seconds),
                float(region.end_seconds),
                region.color,
                int(region.order_index),
                region.kind,
                region.id,
            ),
        )

    def delete(self, region_id: str) -> None:
        self._execute("DELETE FROM timeline_regions WHERE id = ?", (region_id,))

    def reorder(self, song_version_id: str, region_ids: list[str]) -> None:
        for order_index, region_id in enumerate(region_ids):
            self._execute(
                "UPDATE timeline_regions SET order_index = ? "
                "WHERE id = ? AND song_version_id = ?",
                (int(order_index), region_id, song_version_id),
            )

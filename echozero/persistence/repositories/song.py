"""
SongRepository and SongVersionRepository: CRUD for songs and their audio versions.
Exists because songs are the primary organizational unit in a setlist-based project.
SongRecord ordering and active-version tracking are first-class persistence concerns.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import SongRecord, SongVersionRecord


class SongRepository(BaseRepository[SongRecord]):
    """Read and write SongRecord entities to the songs table."""

    def _from_row(self, row: sqlite3.Row) -> SongRecord:
        """Convert a database row to a SongRecord entity."""
        return SongRecord(
            id=row['id'],
            project_id=row['project_id'],
            title=row['title'],
            artist=row['artist'],
            order=row['order'],
            active_version_id=row['active_version_id'],
        )

    def create(self, song: SongRecord) -> None:
        """Insert a new song row."""
        self._execute(
            'INSERT INTO songs (id, project_id, title, artist, "order", active_version_id) '
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                song.id,
                song.project_id,
                song.title,
                song.artist,
                song.order,
                song.active_version_id,
            ),
        )

    def get(self, song_id: str) -> SongRecord | None:
        """Return a song by ID, or None if not found."""
        row = self._fetchone(
            'SELECT id, project_id, title, artist, "order", active_version_id '
            "FROM songs WHERE id = ?",
            (song_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list_by_project(self, project_id: str) -> list[SongRecord]:
        """Return all songs for a project, ordered by their position."""
        rows = self._fetchall(
            'SELECT id, project_id, title, artist, "order", active_version_id '
            'FROM songs WHERE project_id = ? ORDER BY "order"',
            (project_id,),
        )
        return [self._from_row(r) for r in rows]

    def update(self, song: SongRecord) -> None:
        """Overwrite a song row with updated values."""
        self._execute(
            'UPDATE songs SET title = ?, artist = ?, "order" = ?, '
            "active_version_id = ? WHERE id = ?",
            (song.title, song.artist, song.order, song.active_version_id, song.id),
        )

    def delete(self, song_id: str) -> None:
        """Delete a song by ID. Cascades to versions, layers, takes."""
        self._execute("DELETE FROM songs WHERE id = ?", (song_id,))

    def reorder(self, project_id: str, song_ids: list[str]) -> None:
        """Set the order of songs in a project by their ID sequence."""
        for i, song_id in enumerate(song_ids):
            self._execute(
                'UPDATE songs SET "order" = ? WHERE id = ? AND project_id = ?',
                (i, song_id, project_id),
            )


class SongVersionRepository(BaseRepository[SongVersionRecord]):
    """Read and write SongVersionRecord entities to the song_versions table."""

    def _from_row(self, row: sqlite3.Row) -> SongVersionRecord:
        """Convert a database row to a SongVersionRecord entity."""
        rebuild_plan_raw = row['rebuild_plan_json'] if 'rebuild_plan_json' in row.keys() else '{}'
        try:
            rebuild_plan = json.loads(rebuild_plan_raw) if rebuild_plan_raw else {}
        except json.JSONDecodeError:
            rebuild_plan = {}

        return SongVersionRecord(
            id=row['id'],
            song_id=row['song_id'],
            label=row['label'],
            audio_file=row['audio_file'],
            duration_seconds=row['duration_seconds'],
            original_sample_rate=row['original_sample_rate'],
            audio_hash=row['audio_hash'],
            created_at=datetime.fromisoformat(row['created_at']),
            ma3_timecode_pool_no=_optional_positive_int(row['ma3_timecode_pool_no']),
            rebuild_plan=rebuild_plan,
        )

    def create(self, version: SongVersionRecord) -> None:
        """Insert a new song version row."""
        self._execute(
            "INSERT INTO song_versions "
            "(id, song_id, label, audio_file, duration_seconds, "
            "original_sample_rate, audio_hash, ma3_timecode_pool_no, rebuild_plan_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                version.id,
                version.song_id,
                version.label,
                version.audio_file,
                version.duration_seconds,
                version.original_sample_rate,
                version.audio_hash,
                version.ma3_timecode_pool_no,
                json.dumps(version.rebuild_plan or {}),
                version.created_at.isoformat(),
            ),
        )

    def get(self, version_id: str) -> SongVersionRecord | None:
        """Return a song version by ID, or None if not found."""
        row = self._fetchone(
            "SELECT id, song_id, label, audio_file, duration_seconds, "
            "original_sample_rate, audio_hash, ma3_timecode_pool_no, rebuild_plan_json, created_at "
            "FROM song_versions WHERE id = ?",
            (version_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list_by_song(self, song_id: str) -> list[SongVersionRecord]:
        """Return all versions for a song, ordered by creation time."""
        rows = self._fetchall(
            "SELECT id, song_id, label, audio_file, duration_seconds, "
            "original_sample_rate, audio_hash, ma3_timecode_pool_no, rebuild_plan_json, created_at "
            "FROM song_versions WHERE song_id = ? ORDER BY created_at",
            (song_id,),
        )
        return [self._from_row(r) for r in rows]

    def update(self, version: SongVersionRecord) -> None:
        """Overwrite a song version's mutable fields (label, audio_file, duration, sample_rate, hash)."""
        self._execute(
            "UPDATE song_versions SET label = ?, audio_file = ?, duration_seconds = ?, "
            "original_sample_rate = ?, audio_hash = ?, ma3_timecode_pool_no = ?, "
            "rebuild_plan_json = ? WHERE id = ?",
            (
                version.label,
                version.audio_file,
                version.duration_seconds,
                version.original_sample_rate,
                version.audio_hash,
                version.ma3_timecode_pool_no,
                json.dumps(version.rebuild_plan or {}),
                version.id,
            ),
        )

    def delete(self, version_id: str) -> None:
        """Delete a song version by ID. Cascades to layers and takes."""
        self._execute(
            "DELETE FROM song_versions WHERE id = ?", (version_id,)
        )


def _optional_positive_int(value: object) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None

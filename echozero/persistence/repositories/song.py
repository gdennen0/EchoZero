"""
SongRepository and SongVersionRepository: CRUD for songs and their audio versions.
Exists because songs are the primary organizational unit in a setlist-based project.
Song ordering and active-version tracking are first-class persistence concerns.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import Song, SongVersion


class SongRepository(BaseRepository[Song]):
    """Read and write Song entities to the songs table."""

    def _from_row(self, row: sqlite3.Row) -> Song:
        """Convert a database row to a Song entity."""
        return Song(
            id=row['id'],
            project_id=row['project_id'],
            title=row['title'],
            artist=row['artist'],
            order=row['order'],
            active_version_id=row['active_version_id'],
        )

    def create(self, song: Song) -> None:
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

    def get(self, song_id: str) -> Song | None:
        """Return a song by ID, or None if not found."""
        row = self._fetchone(
            'SELECT id, project_id, title, artist, "order", active_version_id '
            "FROM songs WHERE id = ?",
            (song_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list_by_project(self, project_id: str) -> list[Song]:
        """Return all songs for a project, ordered by their position."""
        rows = self._fetchall(
            'SELECT id, project_id, title, artist, "order", active_version_id '
            'FROM songs WHERE project_id = ? ORDER BY "order"',
            (project_id,),
        )
        return [self._from_row(r) for r in rows]

    def update(self, song: Song) -> None:
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


class SongVersionRepository(BaseRepository[SongVersion]):
    """Read and write SongVersion entities to the song_versions table."""

    def _from_row(self, row: sqlite3.Row) -> SongVersion:
        """Convert a database row to a SongVersion entity."""
        return SongVersion(
            id=row['id'],
            song_id=row['song_id'],
            label=row['label'],
            audio_file=row['audio_file'],
            duration_seconds=row['duration_seconds'],
            original_sample_rate=row['original_sample_rate'],
            audio_hash=row['audio_hash'],
            created_at=datetime.fromisoformat(row['created_at']),
        )

    def create(self, version: SongVersion) -> None:
        """Insert a new song version row."""
        self._execute(
            "INSERT INTO song_versions "
            "(id, song_id, label, audio_file, duration_seconds, "
            "original_sample_rate, audio_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                version.id,
                version.song_id,
                version.label,
                version.audio_file,
                version.duration_seconds,
                version.original_sample_rate,
                version.audio_hash,
                version.created_at.isoformat(),
            ),
        )

    def get(self, version_id: str) -> SongVersion | None:
        """Return a song version by ID, or None if not found."""
        row = self._fetchone(
            "SELECT id, song_id, label, audio_file, duration_seconds, "
            "original_sample_rate, audio_hash, created_at "
            "FROM song_versions WHERE id = ?",
            (version_id,),
        )
        if row is None:
            return None
        return self._from_row(row)

    def list_by_song(self, song_id: str) -> list[SongVersion]:
        """Return all versions for a song, ordered by creation time."""
        rows = self._fetchall(
            "SELECT id, song_id, label, audio_file, duration_seconds, "
            "original_sample_rate, audio_hash, created_at "
            "FROM song_versions WHERE song_id = ? ORDER BY created_at",
            (song_id,),
        )
        return [self._from_row(r) for r in rows]

    def delete(self, version_id: str) -> None:
        """Delete a song version by ID. Cascades to layers and takes."""
        self._execute(
            "DELETE FROM song_versions WHERE id = ?", (version_id,)
        )

"""
Video attachment repositories for song timeline reference media.
Exists to persist the single song-level video and per-version timeline placement.
Connects SQLite project storage to the timeline reference lane projection.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import (
    SongVideoAttachmentRecord,
    SongVideoPlacementRecord,
)


class SongVideoAttachmentRepository(BaseRepository[SongVideoAttachmentRecord]):
    """Read and write the one video attachment allowed per song."""

    def _from_row(self, row: sqlite3.Row) -> SongVideoAttachmentRecord:
        """Convert a database row to a video attachment record."""

        return SongVideoAttachmentRecord(
            id=row["id"],
            song_id=row["song_id"],
            video_file=row["video_file"],
            video_hash=row["video_hash"],
            duration_seconds=float(row["duration_seconds"]),
            extracted_audio_file=row["extracted_audio_file"],
            extracted_audio_hash=row["extracted_audio_hash"],
            width=_optional_int(row["width"]),
            height=_optional_int(row["height"]),
            fps=_optional_float(row["fps"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def upsert(self, record: SongVideoAttachmentRecord) -> None:
        """Insert or replace the one video attachment for a song."""

        self._execute(
            "INSERT INTO song_video_attachments "
            "(id, song_id, video_file, video_hash, duration_seconds, extracted_audio_file, "
            "extracted_audio_hash, width, height, fps, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(song_id) DO UPDATE SET "
            "id = excluded.id, video_file = excluded.video_file, video_hash = excluded.video_hash, "
            "duration_seconds = excluded.duration_seconds, "
            "extracted_audio_file = excluded.extracted_audio_file, "
            "extracted_audio_hash = excluded.extracted_audio_hash, width = excluded.width, "
            "height = excluded.height, fps = excluded.fps, updated_at = excluded.updated_at",
            (
                record.id,
                record.song_id,
                record.video_file,
                record.video_hash,
                record.duration_seconds,
                record.extracted_audio_file,
                record.extracted_audio_hash,
                record.width,
                record.height,
                record.fps,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
            ),
        )

    def get_by_song(self, song_id: str) -> SongVideoAttachmentRecord | None:
        """Return the video attachment for a song, or None if absent."""

        row = self._fetchone(
            "SELECT id, song_id, video_file, video_hash, duration_seconds, "
            "extracted_audio_file, extracted_audio_hash, width, height, fps, "
            "created_at, updated_at FROM song_video_attachments WHERE song_id = ?",
            (song_id,),
        )
        return None if row is None else self._from_row(row)

    def delete_by_song(self, song_id: str) -> None:
        """Delete any video attachment for a song."""

        self._execute("DELETE FROM song_video_attachments WHERE song_id = ?", (song_id,))


class SongVideoPlacementRepository(BaseRepository[SongVideoPlacementRecord]):
    """Read and write per-version video timeline placement."""

    def _from_row(self, row: sqlite3.Row) -> SongVideoPlacementRecord:
        """Convert a database row to a video placement record."""

        return SongVideoPlacementRecord(
            song_version_id=row["song_version_id"],
            video_start_seconds=float(row["video_start_seconds"]),
        )

    def upsert(self, record: SongVideoPlacementRecord) -> None:
        """Insert or update video placement for one song version."""

        self._execute(
            "INSERT INTO song_video_placements (song_version_id, video_start_seconds) "
            "VALUES (?, ?) "
            "ON CONFLICT(song_version_id) DO UPDATE SET "
            "video_start_seconds = excluded.video_start_seconds",
            (record.song_version_id, float(record.video_start_seconds)),
        )

    def get(self, song_version_id: str) -> SongVideoPlacementRecord | None:
        """Return video placement for one song version, or None if absent."""

        row = self._fetchone(
            "SELECT song_version_id, video_start_seconds "
            "FROM song_video_placements WHERE song_version_id = ?",
            (song_version_id,),
        )
        return None if row is None else self._from_row(row)

    def delete(self, song_version_id: str) -> None:
        """Delete video placement for one song version."""

        self._execute(
            "DELETE FROM song_video_placements WHERE song_version_id = ?",
            (song_version_id,),
        )


def _optional_int(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None

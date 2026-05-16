"""
Video reference behavior tests for storage, sync mapping, and playback exclusion.
Exists to pin the song-level video attachment contract introduced for timeline references.
Connects persistence and presentation-facing sync rules without requiring real video playback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from echozero.persistence.entities import SongRecord, SongVersionRecord
from echozero.persistence.session import ProjectStorage
from echozero.persistence.video import ImportedVideo, VideoMetadata
from echozero.application.timeline.video import VideoTimelineMapping


@dataclass(frozen=True, slots=True)
class _FakeImportedVideo:
    video_file: str = "video/ref.mov"
    video_hash: str = "hash-video"
    metadata: VideoMetadata = VideoMetadata(
        duration_seconds=12.5,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
    )
    extracted_audio_file: str | None = "audio/video_refs/ref.wav"
    extracted_audio_hash: str | None = "hash-audio"


def test_video_timeline_mapping_clamps_before_and_after_clip() -> None:
    mapping = VideoTimelineMapping(
        video_path="/tmp/ref.mov",
        start_seconds=2.5,
        duration_seconds=10.0,
    )

    assert mapping.media_seconds_for_song_time(1.0) == 0.0
    assert mapping.media_seconds_for_song_time(7.0) == 4.5
    assert mapping.media_seconds_for_song_time(20.0) == 10.0
    assert not mapping.contains_song_time(1.0)
    assert mapping.contains_song_time(2.5)
    assert mapping.contains_song_time(12.5)
    assert not mapping.contains_song_time(12.6)


def test_project_storage_imports_one_video_per_song_with_per_version_offset(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_import_video(_source_path: Path, _working_dir: Path) -> ImportedVideo:
        fake = _FakeImportedVideo()
        return ImportedVideo(
            video_file=fake.video_file,
            video_hash=fake.video_hash,
            metadata=fake.metadata,
            extracted_audio_file=fake.extracted_audio_file,
            extracted_audio_hash=fake.extracted_audio_hash,
        )

    monkeypatch.setattr("echozero.persistence.video.import_video", fake_import_video)
    storage = ProjectStorage.create_new("Video", working_dir_root=tmp_path)
    try:
        now = datetime.now(timezone.utc)
        song = SongRecord(
            id="song-video",
            project_id=storage.project.id,
            title="Song",
            artist="Artist",
            order=0,
            active_version_id="version-video",
        )
        version = SongVersionRecord(
            id="version-video",
            song_id=song.id,
            label="Main",
            audio_file="audio/song.wav",
            duration_seconds=30.0,
            original_sample_rate=44100,
            audio_hash="hash-song",
            created_at=now,
        )
        storage.songs.create(song)
        storage.song_versions.create(version)
        storage.db.commit()

        first = storage.import_or_replace_song_video(song.id, tmp_path / "a.mov")
        storage.set_song_video_start_seconds(version.id, -1.25)
        second = storage.import_or_replace_song_video(song.id, tmp_path / "b.mov")

        attachment = storage.song_video_attachments.get_by_song(song.id)
        placement = storage.song_video_placements.get(version.id)

        assert attachment is not None
        assert attachment.id == second.id
        assert attachment.id != first.id
        assert attachment.video_file == "video/ref.mov"
        assert attachment.extracted_audio_file == "audio/video_refs/ref.wav"
        assert placement is not None
        assert placement.video_start_seconds == -1.25
    finally:
        storage.close()

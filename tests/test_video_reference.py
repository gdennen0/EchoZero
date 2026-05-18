"""
Video reference behavior tests for storage, sync mapping, and playback exclusion.
Exists to pin the song-level video attachment contract introduced for timeline references.
Connects persistence and presentation-facing sync rules without requiring real video playback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import pytest

from echozero.application.presentation.inspector_contract_context_actions import (
    shared_context_sections,
)
from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import LayerId, TimelineId
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.video import (
    VideoClockSync,
    VideoTimelineMapping,
    video_mapping_from_presentation,
)
from echozero.persistence.entities import (
    SongRecord,
    SongVersionRecord,
    SongVideoAttachmentRecord,
)
from echozero.persistence.session import ProjectStorage
from echozero.persistence.video import ImportedVideo, VideoMetadata

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_video_timeline_mapping_wraps_when_loop_enabled() -> None:
    mapping = VideoTimelineMapping(
        video_path="/tmp/ref.mov",
        start_seconds=2.5,
        duration_seconds=10.0,
        loop_enabled=True,
    )

    assert mapping.media_seconds_for_song_time(1.0) == 0.0
    assert mapping.media_seconds_for_song_time(7.0) == 4.5
    assert mapping.media_seconds_for_song_time(15.0) == pytest.approx(2.5)
    assert not mapping.contains_song_time(1.0)
    assert mapping.contains_song_time(120.0)


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
        storage.set_song_video_loop_enabled(version.id, True)
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
        assert placement.video_loop_enabled is True
    finally:
        storage.close()


def test_video_archive_roundtrip_preserves_project_video_and_presentation_path(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_import_video(_source_path: Path, working_dir: Path) -> ImportedVideo:
        video_path = working_dir / "video" / "ref.mov"
        audio_path = working_dir / "audio" / "video_refs" / "ref.wav"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"video")
        audio_path.write_bytes(b"reference audio")
        return ImportedVideo(
            video_file="video/ref.mov",
            video_hash="hash-video",
            metadata=VideoMetadata(duration_seconds=8.0, has_audio=True),
            extracted_audio_file="audio/video_refs/ref.wav",
            extracted_audio_hash="hash-audio",
        )

    monkeypatch.setattr("echozero.persistence.video.import_video", fake_import_video)
    working_root = tmp_path / "working"
    ez_path = tmp_path / "portable.ez"
    storage = ProjectStorage.create_new("Video", working_dir_root=working_root)
    try:
        song, version = _create_song_with_version(storage)
        storage.import_or_replace_song_video(song.id, tmp_path / "source.mov")
        storage.set_song_video_start_seconds(version.id, 1.5)
        storage.save_as(ez_path)
    finally:
        storage.close()

    with zipfile.ZipFile(ez_path) as archive:
        names = set(archive.namelist())
    assert "video/ref.mov" in names
    assert "audio/video_refs/ref.wav" in names

    reopened = ProjectStorage.open(ez_path, working_dir_root=working_root / "reopened")
    try:
        presentation = _video_presentation(reopened.working_dir / "video" / "ref.mov")
        mapping = video_mapping_from_presentation(presentation)
        assert (reopened.working_dir / "video" / "ref.mov").read_bytes() == b"video"
        assert mapping is not None
        assert mapping.video_path == str(reopened.working_dir / "video" / "ref.mov")
    finally:
        reopened.close()


def test_video_clock_holds_until_positive_offset_then_autoplays() -> None:
    sync = VideoClockSync(drift_threshold_seconds=0.05)
    mapping = VideoTimelineMapping("/tmp/ref.mov", start_seconds=5.0, duration_seconds=10.0)

    before = sync.decision(
        mapping,
        song_seconds=4.9,
        audio_is_playing=True,
        media_seconds=0.0,
    )
    crossing = sync.decision(
        mapping,
        song_seconds=5.01,
        audio_is_playing=True,
        media_seconds=0.0,
    )

    assert before.should_play is False
    assert before.media_seconds == 0.0
    assert crossing.should_play is True
    assert crossing.media_seconds == pytest.approx(0.01)


def test_video_clock_corrects_playback_drift() -> None:
    sync = VideoClockSync(drift_threshold_seconds=0.05)
    mapping = VideoTimelineMapping("/tmp/ref.mov", start_seconds=2.0, duration_seconds=10.0)

    decision = sync.decision(
        mapping,
        song_seconds=6.0,
        audio_is_playing=True,
        media_seconds=3.7,
    )

    assert decision.should_play is True
    assert decision.media_seconds == 4.0
    assert decision.should_seek is True


def test_video_clock_loops_after_clip_end() -> None:
    sync = VideoClockSync(drift_threshold_seconds=0.05)
    mapping = VideoTimelineMapping(
        "/tmp/ref.mov",
        start_seconds=2.0,
        duration_seconds=10.0,
        loop_enabled=True,
    )

    decision = sync.decision(
        mapping,
        song_seconds=24.25,
        audio_is_playing=True,
        media_seconds=2.0,
    )

    assert decision.should_play is True
    assert decision.media_seconds == pytest.approx(2.25)
    assert decision.should_seek is True


def test_video_mapping_from_presentation_preserves_loop_state(tmp_path: Path) -> None:
    video_path = tmp_path / "ref.mov"
    video_path.write_bytes(b"video")

    mapping = video_mapping_from_presentation(
        _video_presentation(video_path, loop_enabled=True)
    )

    assert mapping is not None
    assert mapping.loop_enabled is True


def test_video_layer_context_action_toggles_loop_state() -> None:
    presentation = _video_presentation(Path("/tmp/ref.mov"), loop_enabled=False)
    layer = presentation.layers[0]

    sections = shared_context_sections(
        presentation=presentation,
        layer=layer,
        take=None,
        hit_target=None,
        has_selected_events=False,
        include_layer_transfer_controls=False,
    )
    actions = [action for section in sections for action in section.actions]
    loop_action = next(
        action for action in actions if action.action_id == "video.set_loop_enabled"
    )

    assert loop_action.label == "Enable Video Loop"
    assert loop_action.params["enabled"] is True


def test_runtime_video_update_syncs_presentation_and_clock() -> None:
    class _RuntimeVideo:
        def __init__(self) -> None:
            self.synced = False
            self.updates: list[tuple[float, bool]] = []

        def sync_presentation(self, presentation: TimelinePresentation) -> None:
            self.synced = presentation.title == "Video"

        def update(self, song_seconds: float, is_playing: bool) -> None:
            self.updates.append((song_seconds, is_playing))

    runtime_video = _RuntimeVideo()
    app = TimelineApplication(
        timeline=None,  # type: ignore[arg-type]
        session=None,  # type: ignore[arg-type]
        orchestrator=None,  # type: ignore[arg-type]
        queries=None,  # type: ignore[arg-type]
        sync_service=None,  # type: ignore[arg-type]
        runtime_video=runtime_video,
    )
    app.update_runtime_video(
        song_seconds=3.25,
        is_playing=True,
        presentation=_video_presentation(Path("/tmp/ref.mov"), title="Video"),
    )

    assert runtime_video.synced is True
    assert runtime_video.updates == [(3.25, True)]


def test_video_window_seek_path_does_not_shell_out_to_ffmpeg() -> None:
    source = (_REPO_ROOT / "echozero/ui/qt/video_window.py").read_text()

    assert "subprocess.run" not in source
    assert "_show_seek_frame" not in source


def test_video_window_exposes_error_status_surface() -> None:
    source = (_REPO_ROOT / "echozero/ui/qt/video_window.py").read_text()

    assert "errorOccurred" in source
    assert "mediaStatusChanged" in source
    assert "error_text" in source


def test_replacing_and_removing_video_cleans_only_unreferenced_project_media(
    tmp_path,
    monkeypatch,
) -> None:
    imports = [
        ImportedVideo(
            video_file="video/old.mov",
            video_hash="old",
            metadata=VideoMetadata(duration_seconds=1.0, has_audio=True),
            extracted_audio_file="audio/video_refs/old.wav",
            extracted_audio_hash="old-audio",
        ),
        ImportedVideo(
            video_file="video/new.mov",
            video_hash="new",
            metadata=VideoMetadata(duration_seconds=1.0, has_audio=True),
            extracted_audio_file="audio/video_refs/shared.wav",
            extracted_audio_hash="shared-audio",
        ),
    ]

    def fake_import_video(_source_path: Path, working_dir: Path) -> ImportedVideo:
        imported = imports.pop(0)
        for rel_path in (imported.video_file, imported.extracted_audio_file):
            assert rel_path is not None
            path = working_dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(rel_path.encode())
        return imported

    monkeypatch.setattr("echozero.persistence.video.import_video", fake_import_video)
    storage = ProjectStorage.create_new("Video", working_dir_root=tmp_path)
    try:
        song, _version = _create_song_with_version(storage)
        storage.import_or_replace_song_video(song.id, tmp_path / "old.mov")
        shared_song, _shared_version = _create_song_with_version(storage, suffix="-shared")
        now = datetime.now(timezone.utc)
        storage.song_video_attachments.upsert(
            SongVideoAttachmentRecord(
                id="shared-video",
                song_id=shared_song.id,
                video_file="video/old.mov",
                video_hash="old",
                duration_seconds=1.0,
                extracted_audio_file="audio/video_refs/old.wav",
                extracted_audio_hash="old-audio",
                width=None,
                height=None,
                fps=None,
                created_at=now,
                updated_at=now,
            )
        )
        storage.db.commit()
        outside = tmp_path / "outside.mov"
        outside.write_bytes(b"outside")

        storage.import_or_replace_song_video(song.id, tmp_path / "new.mov")

        assert (storage.working_dir / "video" / "old.mov").exists()
        assert (storage.working_dir / "audio" / "video_refs" / "old.wav").exists()
        assert (storage.working_dir / "video" / "new.mov").exists()
        assert outside.exists()

        storage.remove_song_video(song.id)

        assert not (storage.working_dir / "video" / "new.mov").exists()
        assert not (storage.working_dir / "audio" / "video_refs" / "shared.wav").exists()
        assert (storage.working_dir / "video" / "old.mov").exists()

        storage.remove_song_video(shared_song.id)

        assert not (storage.working_dir / "video" / "old.mov").exists()
        assert not (storage.working_dir / "audio" / "video_refs" / "old.wav").exists()
    finally:
        storage.close()


def _create_song_with_version(
    storage: ProjectStorage,
    *,
    suffix: str = "",
) -> tuple[SongRecord, SongVersionRecord]:
    now = datetime.now(timezone.utc)
    song = SongRecord(
        id=f"song-{storage.project.id}{suffix}",
        project_id=storage.project.id,
        title="Song",
        artist="Artist",
        order=0,
        active_version_id=f"version-{storage.project.id}{suffix}",
    )
    version = SongVersionRecord(
        id=f"version-{storage.project.id}{suffix}",
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
    return song, version


def _video_presentation(
    video_path: Path,
    *,
    title: str = "Presentation",
    loop_enabled: bool = False,
) -> TimelinePresentation:
    return TimelinePresentation(
        timeline_id=TimelineId("timeline-video"),
        title=title,
        layers=[
            LayerPresentation(
                layer_id=LayerId("layer-video"),
                title="Video",
                kind=LayerKind.REFERENCE,
                reference_kind="video",
                video_path=str(video_path),
                video_start_seconds=1.5,
                video_duration_seconds=8.0,
                video_loop_enabled=loop_enabled,
            )
        ],
    )

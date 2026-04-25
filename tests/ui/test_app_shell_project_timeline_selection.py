"""Project timeline selection tests for the Qt app shell.
Exists to lock the extracted song/version resolution seam during hygiene cleanup.
Connects the public baseline builder split to focused selector-state assertions.
"""

from __future__ import annotations

from datetime import datetime, timezone

from echozero.application.shared.ids import SongId, SongVersionId
from echozero.persistence.entities import (
    ProjectRecord,
    ProjectSettingsRecord,
    SongRecord,
    SongVersionRecord,
)
from echozero.ui.qt.app_shell_project_timeline_selection import (
    resolve_project_timeline_selection,
)


def test_resolve_project_timeline_selection_prefers_requested_version_and_keeps_song_bookkeeping():
    project = _project()
    song = SongRecord(
        id="song_alpha",
        project_id=project.id,
        title="Alpha",
        artist="Artist",
        order=0,
        active_version_id="version_main",
    )
    version_main = _version(song_id=song.id, version_id="version_main", label="Main")
    version_festival = _version(
        song_id=song.id,
        version_id="version_festival",
        label="Festival Edit",
        ma3_timecode_pool_no=113,
    )
    storage = _StubProjectStorage(
        project=project,
        songs=[song],
        versions=[version_main, version_festival],
    )

    selection = resolve_project_timeline_selection(
        storage,
        active_song_version_id=SongVersionId(version_festival.id),
    )

    assert selection.active_song is not None
    assert selection.active_song.id == song.id
    assert selection.active_version is not None
    assert selection.active_version.id == version_festival.id
    assert [option.song_version_id for option in selection.available_song_versions] == [
        version_main.id,
        version_festival.id,
    ]
    assert [option.is_active for option in selection.available_song_versions] == [False, True]
    assert selection.available_songs[0].active_version_id == version_main.id
    assert selection.available_songs[0].active_version_label == version_main.label


def test_resolve_project_timeline_selection_keeps_requested_song_when_no_active_version_exists():
    project = _project()
    song = SongRecord(
        id="song_alpha",
        project_id=project.id,
        title="Alpha",
        artist="Artist",
        order=0,
        active_version_id=None,
    )
    archived_version = _version(song_id=song.id, version_id="version_archived", label="Archived")
    storage = _StubProjectStorage(project=project, songs=[song], versions=[archived_version])

    selection = resolve_project_timeline_selection(
        storage,
        active_song_id=SongId(song.id),
    )

    assert selection.active_song is not None
    assert selection.active_song.id == song.id
    assert selection.active_version is None
    assert selection.available_songs[0].is_active is True
    assert selection.available_song_versions[0].song_version_id == archived_version.id
    assert selection.available_song_versions[0].is_active is False


class _StubProjectStorage:
    def __init__(
        self,
        *,
        project: ProjectRecord,
        songs: list[SongRecord],
        versions: list[SongVersionRecord],
    ) -> None:
        self.project = project
        self.songs = _StubSongRepository(songs)
        self.song_versions = _StubSongVersionRepository(versions)


class _StubSongRepository:
    def __init__(self, songs: list[SongRecord]) -> None:
        self._songs = list(songs)
        self._songs_by_id = {song.id: song for song in songs}

    def get(self, song_id: str) -> SongRecord | None:
        return self._songs_by_id.get(song_id)

    def list_by_project(self, project_id: str) -> list[SongRecord]:
        return [song for song in self._songs if song.project_id == project_id]


class _StubSongVersionRepository:
    def __init__(self, versions: list[SongVersionRecord]) -> None:
        self._versions_by_id = {version.id: version for version in versions}
        self._versions_by_song_id: dict[str, list[SongVersionRecord]] = {}
        for version in versions:
            self._versions_by_song_id.setdefault(version.song_id, []).append(version)

    def get(self, version_id: str) -> SongVersionRecord | None:
        return self._versions_by_id.get(version_id)

    def list_by_song(self, song_id: str) -> list[SongVersionRecord]:
        return list(self._versions_by_song_id.get(song_id, []))


def _project() -> ProjectRecord:
    now = datetime.now(timezone.utc)
    return ProjectRecord(
        id="project_alpha",
        name="Alpha Project",
        settings=ProjectSettingsRecord(),
        created_at=now,
        updated_at=now,
    )


def _version(
    *,
    song_id: str,
    version_id: str,
    label: str,
    ma3_timecode_pool_no: int | None = None,
) -> SongVersionRecord:
    return SongVersionRecord(
        id=version_id,
        song_id=song_id,
        label=label,
        audio_file=f"{version_id}.wav",
        duration_seconds=10.0,
        original_sample_rate=44100,
        audio_hash=f"hash-{version_id}",
        created_at=datetime.now(timezone.utc),
        ma3_timecode_pool_no=ma3_timecode_pool_no,
    )

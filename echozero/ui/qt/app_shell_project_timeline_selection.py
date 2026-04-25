"""Project timeline selection resolution for the Qt app shell.
Exists to isolate active song/version resolution and selector option bookkeeping.
Connects project storage song state to the baseline timeline builder without changing its public entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from echozero.application.presentation.models import (
    SongOptionPresentation,
    SongVersionOptionPresentation,
)
from echozero.application.shared.ids import SongId, SongVersionId
from echozero.persistence.entities import SongRecord, SongVersionRecord
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_project_timeline_overlay import (
    available_song_options,
    available_song_version_options,
)


@dataclass(slots=True)
class ProjectTimelineSelection:
    """Resolved active song/version state plus selector options for timeline startup."""

    active_song: SongRecord | None
    active_version: SongVersionRecord | None
    available_songs: list[SongOptionPresentation]
    available_song_versions: list[SongVersionOptionPresentation]


def resolve_project_timeline_selection(
    project_storage: ProjectStorage,
    *,
    active_song_id: SongId | None = None,
    active_song_version_id: SongVersionId | None = None,
) -> ProjectTimelineSelection:
    """Resolve the active song/version pair and presentation options from project storage."""

    songs = project_storage.songs.list_by_project(project_storage.project.id)
    versions_by_song_id, active_versions_by_song_id, version_counts_by_song_id = (
        _build_song_version_maps(project_storage, songs)
    )
    active_song, active_version = _resolve_active_song_and_version(
        project_storage,
        songs=songs,
        requested_song_id=str(active_song_id) if active_song_id is not None else None,
        requested_version_id=(
            str(active_song_version_id) if active_song_version_id is not None else None
        ),
    )
    available_songs = available_song_options(
        songs,
        active_song_id=active_song.id if active_song is not None else None,
        active_versions_by_song_id=active_versions_by_song_id,
        versions_by_song_id=versions_by_song_id,
        version_counts_by_song_id=version_counts_by_song_id,
    )
    available_song_versions = available_song_version_options(
        versions_by_song_id.get(active_song.id, []) if active_song is not None else [],
        active_song_version_id=active_version.id if active_version is not None else None,
    )
    return ProjectTimelineSelection(
        active_song=active_song,
        active_version=active_version,
        available_songs=available_songs,
        available_song_versions=available_song_versions,
    )


def _build_song_version_maps(
    project_storage: ProjectStorage,
    songs: list[SongRecord],
) -> tuple[
    dict[str, list[SongVersionRecord]],
    dict[str, SongVersionRecord | None],
    dict[str, int],
]:
    versions_by_song_id = {
        song.id: project_storage.song_versions.list_by_song(song.id) for song in songs
    }
    active_versions_by_song_id = {
        song.id: next(
            (
                version
                for version in versions_by_song_id.get(song.id, [])
                if version.id == song.active_version_id
            ),
            None,
        )
        for song in songs
    }
    version_counts_by_song_id = {
        song.id: len(versions_by_song_id.get(song.id, [])) for song in songs
    }
    return versions_by_song_id, active_versions_by_song_id, version_counts_by_song_id


def _resolve_active_song_and_version(
    project_storage: ProjectStorage,
    *,
    songs: list[SongRecord],
    requested_song_id: str | None,
    requested_version_id: str | None,
) -> tuple[SongRecord | None, SongVersionRecord | None]:
    active_song = None
    active_version = None
    if requested_version_id is not None:
        active_version = project_storage.song_versions.get(requested_version_id)
        if active_version is not None:
            active_song = project_storage.songs.get(active_version.song_id)
    if active_song is None and requested_song_id is not None:
        active_song = next((song for song in songs if song.id == requested_song_id), None)
    if active_song is None:
        active_song = next((song for song in songs if song.active_version_id), None)
    if (
        active_song is not None
        and active_version is None
        and active_song.active_version_id is not None
    ):
        active_version = project_storage.song_versions.get(active_song.active_version_id)
    return active_song, active_version


__all__ = [
    "ProjectTimelineSelection",
    "resolve_project_timeline_selection",
]

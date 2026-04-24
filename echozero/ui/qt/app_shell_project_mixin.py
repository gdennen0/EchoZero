"""Project lifecycle facade mixin for the Qt app shell runtime.
Exists to isolate song and project lifecycle methods from the orchestration root.
Connects AppShellRuntime to project persistence and baseline timeline reload helpers.
"""

from __future__ import annotations

from pathlib import Path

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.ids import SongId, SongVersionId
from echozero.ui.qt.app_shell_project_lifecycle import (
    ProjectLifecycleShell,
    add_song_from_path,
    add_song_version,
    delete_song,
    delete_song_version,
    list_ma3_timecode_pools,
    new_project,
    open_project,
    refresh_from_storage,
    save_project,
    save_project_as,
    select_song,
    set_song_version_ma3_timecode_pool,
    switch_song_version,
)


class AppShellProjectMixin:
    def new_project(self: ProjectLifecycleShell, name: str = "EchoZero Project") -> None:
        new_project(self, name=name)

    def save_project_as(self: ProjectLifecycleShell, path: str | Path) -> Path:
        return save_project_as(self, path)

    def save_project(self: ProjectLifecycleShell) -> Path:
        return save_project(self)

    def open_project(self: ProjectLifecycleShell, path: str | Path) -> None:
        open_project(self, path)

    def add_song_from_path(
        self: ProjectLifecycleShell,
        title: str,
        audio_path: str | Path,
    ) -> TimelinePresentation:
        return add_song_from_path(self, title, audio_path)

    def select_song(
        self: ProjectLifecycleShell,
        song_id: str | SongId,
    ) -> TimelinePresentation:
        return select_song(self, song_id)

    def switch_song_version(
        self: ProjectLifecycleShell,
        song_version_id: str | SongVersionId,
    ) -> TimelinePresentation:
        return switch_song_version(self, song_version_id)

    def add_song_version(
        self: ProjectLifecycleShell,
        song_id: str | SongId,
        audio_path: str | Path,
        *,
        label: str | None = None,
        activate: bool = True,
    ) -> TimelinePresentation:
        return add_song_version(
            self,
            song_id,
            audio_path,
            label=label,
            activate=activate,
        )

    def delete_song(
        self: ProjectLifecycleShell,
        song_id: str | SongId,
    ) -> TimelinePresentation:
        return delete_song(self, song_id)

    def delete_song_version(
        self: ProjectLifecycleShell,
        song_version_id: str | SongVersionId,
    ) -> TimelinePresentation:
        return delete_song_version(self, song_version_id)

    def list_ma3_timecode_pools(
        self: ProjectLifecycleShell,
    ) -> list[tuple[int, str | None]]:
        return list_ma3_timecode_pools(self)

    def set_song_version_ma3_timecode_pool(
        self: ProjectLifecycleShell,
        song_version_id: str | SongVersionId,
        timecode_pool_no: int | None,
    ) -> TimelinePresentation:
        return set_song_version_ma3_timecode_pool(
            self,
            song_version_id,
            timecode_pool_no,
        )

    def _refresh_from_storage(
        self: ProjectLifecycleShell,
        *,
        active_song_id: SongId | None = None,
        active_song_version_id: SongVersionId | None = None,
    ) -> None:
        refresh_from_storage(
            self,
            active_song_id=active_song_id,
            active_song_version_id=active_song_version_id,
        )

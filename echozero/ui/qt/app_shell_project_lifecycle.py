"""Project lifecycle helpers for the Qt app shell runtime.
Exists to isolate project reload and song/version selection flows from orchestration methods.
Connects ProjectStorage lifecycle and baseline timeline reloads to the Stage Zero shell.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Protocol

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.shared.ids import SongId, SongVersionId
from echozero.application.sync.adapters import MA3SyncBridge
from echozero.application.sync.service import SyncService
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.models import Layer
from echozero.application.timeline.pipeline_run_service import PipelineRunService
from echozero.persistence.entities import SongRecord, SongVersionRecord
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_project_timeline import (
    apply_timeline_presentation_overlay,
    build_project_native_baseline_timeline,
)
from echozero.ui.qt.app_shell_runtime_services import build_runtime_timeline_application
from echozero.ui.qt.app_shell_timeline_state import restore_timeline_targets


class ProjectLifecycleShell(Protocol):
    _app: TimelineApplication
    _draft_layers: list[Layer]
    _is_dirty: bool
    _last_pipeline_run_revision: int
    _pipeline_runs: PipelineRunService
    _sync_bridge: MA3SyncBridge | None
    _sync_service_override: SyncService | None
    project_path: Path | None
    project_storage: ProjectStorage

    @property
    def runtime_audio(self) -> object | None: ...

    @property
    def session(self) -> Session: ...

    def presentation(self) -> TimelinePresentation: ...

    def _build_object_action_services(self) -> None: ...

    def _clear_history(self) -> None: ...

    def _materialize_draft_layers(self, *, song_version_id: str) -> None: ...

    def _select_active_source_layer(self) -> None: ...

    def _sync_runtime_audio_from_presentation(self, presentation: TimelinePresentation) -> None: ...


def new_project(shell: ProjectLifecycleShell, name: str = "EchoZero Project") -> None:
    working_dir_root = shell.project_storage.working_dir.parent
    project_storage = ProjectStorage.create_new(
        name=name,
        working_dir_root=working_dir_root,
    )
    _install_project_runtime(
        shell,
        project_storage=project_storage,
        project_path=None,
        runtime_audio=shell.runtime_audio,
    )
    shell._is_dirty = False
    shell._clear_history()


def save_project_as(shell: ProjectLifecycleShell, path: str | Path) -> Path:
    target_path = Path(path)
    shell.project_storage.save_as(target_path)
    shell.project_path = target_path
    shell._is_dirty = False
    shell._clear_history()
    return target_path


def save_project(shell: ProjectLifecycleShell) -> Path:
    if shell.project_path is None:
        raise RuntimeError("save_project requires an existing project_path")
    return save_project_as(shell, shell.project_path)


def open_project(shell: ProjectLifecycleShell, path: str | Path) -> None:
    target_path = Path(path)
    working_dir_root = shell.project_storage.working_dir.parent
    prior_presentation = shell.presentation()
    if _paths_match(shell.project_path, target_path):
        shell._pipeline_runs.shutdown()
        shell.project_storage.close()
        project_storage = ProjectStorage.open(
            target_path,
            working_dir_root=working_dir_root,
        )
    else:
        project_storage = ProjectStorage.open(
            target_path,
            working_dir_root=working_dir_root,
        )
    _install_project_runtime(
        shell,
        project_storage=project_storage,
        project_path=target_path,
        runtime_audio=shell.runtime_audio,
    )
    restore_timeline_targets(
        timeline=shell._app.timeline,
        prior_presentation=prior_presentation,
        current_presentation=shell.presentation(),
    )
    shell._is_dirty = False
    shell._clear_history()


def add_song_from_path(
    shell: ProjectLifecycleShell,
    title: str,
    audio_path: str | Path,
) -> TimelinePresentation:
    carried_draft_layers = bool(shell._draft_layers)
    song, version = shell.project_storage.import_song(
        title=title,
        audio_source=Path(audio_path),
    )
    shell._materialize_draft_layers(song_version_id=str(version.id))
    refresh_from_storage(
        shell,
        active_song_id=SongId(song.id),
        active_song_version_id=SongVersionId(version.id),
    )
    if carried_draft_layers:
        shell._select_active_source_layer()
    shell._is_dirty = True
    shell._clear_history()
    return shell.presentation()


def select_song(
    shell: ProjectLifecycleShell,
    song_id: str | SongId,
) -> TimelinePresentation:
    song_record = shell.project_storage.songs.get(str(song_id))
    if song_record is None:
        raise ValueError(f"SongRecord not found: {song_id}")

    active_version_id = (
        SongVersionId(song_record.active_version_id)
        if song_record.active_version_id is not None
        else None
    )
    refresh_from_storage(
        shell,
        active_song_id=SongId(song_record.id),
        active_song_version_id=active_version_id,
    )
    shell._clear_history()
    return shell.presentation()


def switch_song_version(
    shell: ProjectLifecycleShell,
    song_version_id: str | SongVersionId,
) -> TimelinePresentation:
    version_record = shell.project_storage.song_versions.get(str(song_version_id))
    if version_record is None:
        raise ValueError(f"SongVersionRecord not found: {song_version_id}")

    song_record = shell.project_storage.songs.get(version_record.song_id)
    if song_record is None:
        raise RuntimeError(f"SongRecord not found for SongVersionRecord '{version_record.id}'")

    if song_record.active_version_id != version_record.id:
        shell.project_storage.songs.update(
            replace(song_record, active_version_id=version_record.id)
        )
        shell.project_storage.commit()
        shell.project_storage.dirty_tracker.mark_dirty(song_record.id)
        shell._is_dirty = True

    refresh_from_storage(
        shell,
        active_song_id=SongId(song_record.id),
        active_song_version_id=SongVersionId(version_record.id),
    )
    shell._clear_history()
    return shell.presentation()


def add_song_version(
    shell: ProjectLifecycleShell,
    song_id: str | SongId,
    audio_path: str | Path,
    *,
    label: str | None = None,
    activate: bool = True,
) -> TimelinePresentation:
    song_record = shell.project_storage.songs.get(str(song_id))
    if song_record is None:
        raise ValueError(f"SongRecord not found: {song_id}")

    version = shell.project_storage.add_song_version(
        song_record.id,
        Path(audio_path),
        label=label,
        activate=activate,
    )
    updated_song = shell.project_storage.songs.get(song_record.id)
    active_version_id = (
        SongVersionId(updated_song.active_version_id)
        if updated_song is not None and updated_song.active_version_id is not None
        else SongVersionId(version.id)
    )
    refresh_from_storage(
        shell,
        active_song_id=SongId(song_record.id),
        active_song_version_id=active_version_id,
    )
    shell._is_dirty = True
    shell._clear_history()
    return shell.presentation()


def delete_song(
    shell: ProjectLifecycleShell,
    song_id: str | SongId,
) -> TimelinePresentation:
    song_record = shell.project_storage.songs.get(str(song_id))
    if song_record is None:
        raise ValueError(f"SongRecord not found: {song_id}")

    next_song_id, next_version_id = _selection_after_song_delete(shell, song_record.id)
    shell.project_storage.delete_song(song_record.id)
    refresh_from_storage(
        shell,
        active_song_id=next_song_id,
        active_song_version_id=next_version_id,
    )
    shell._is_dirty = True
    shell._clear_history()
    return shell.presentation()


def delete_song_version(
    shell: ProjectLifecycleShell,
    song_version_id: str | SongVersionId,
) -> TimelinePresentation:
    version_record = shell.project_storage.song_versions.get(str(song_version_id))
    if version_record is None:
        raise ValueError(f"SongVersionRecord not found: {song_version_id}")

    song_record = shell.project_storage.songs.get(version_record.song_id)
    if song_record is None:
        raise RuntimeError(f"SongRecord not found for SongVersionRecord '{version_record.id}'")

    next_song_id, next_version_id = _selection_after_song_version_delete(
        shell,
        song_id=song_record.id,
        deleted_version_id=version_record.id,
    )
    shell.project_storage.delete_song_version(version_record.id)
    refresh_from_storage(
        shell,
        active_song_id=next_song_id,
        active_song_version_id=next_version_id,
    )
    shell._is_dirty = True
    shell._clear_history()
    return shell.presentation()


def list_ma3_timecode_pools(
    shell: ProjectLifecycleShell,
) -> list[tuple[int, str | None]]:
    method = getattr(shell._app.sync_service, "list_timecodes", None)
    if not callable(method):
        return []

    raw_timecodes = method() or []
    normalized: list[tuple[int, str | None]] = []
    for raw_timecode in raw_timecodes:
        if isinstance(raw_timecode, tuple) and len(raw_timecode) >= 1:
            no = _optional_positive_int(raw_timecode[0])
            name = None if len(raw_timecode) < 2 or raw_timecode[1] in {None, ""} else str(raw_timecode[1])
        elif isinstance(raw_timecode, dict):
            no = _optional_positive_int(raw_timecode.get("number", raw_timecode.get("no")))
            raw_name = raw_timecode.get("name")
            name = None if raw_name in {None, ""} else str(raw_name)
        else:
            no = _optional_positive_int(getattr(raw_timecode, "number", getattr(raw_timecode, "no", None)))
            raw_name = getattr(raw_timecode, "name", None)
            name = None if raw_name in {None, ""} else str(raw_name)
        if no is not None:
            normalized.append((no, name))
    return sorted(set(normalized), key=lambda value: value[0])


def set_song_version_ma3_timecode_pool(
    shell: ProjectLifecycleShell,
    song_version_id: str | SongVersionId,
    timecode_pool_no: int | None,
) -> TimelinePresentation:
    version_record = shell.project_storage.song_versions.get(str(song_version_id))
    if version_record is None:
        raise ValueError(f"SongVersionRecord not found: {song_version_id}")

    song_record = shell.project_storage.songs.get(version_record.song_id)
    if song_record is None:
        raise RuntimeError(f"SongRecord not found for SongVersionRecord '{version_record.id}'")

    normalized_pool_no = _optional_positive_int(timecode_pool_no)
    shell.project_storage.song_versions.update(
        replace(version_record, ma3_timecode_pool_no=normalized_pool_no)
    )
    shell.project_storage.commit()
    shell.project_storage.dirty_tracker.mark_dirty(song_record.id)
    refresh_from_storage(
        shell,
        active_song_id=SongId(song_record.id),
        active_song_version_id=SongVersionId(version_record.id),
    )
    shell._is_dirty = True
    shell._clear_history()
    return shell.presentation()


def refresh_from_storage(
    shell: ProjectLifecycleShell,
    *,
    active_song_id: SongId | None = None,
    active_song_version_id: SongVersionId | None = None,
) -> None:
    current_presentation = shell.presentation()
    with shell.project_storage.locked():
        timeline, overlay, resolved_song_id, resolved_song_version_id = (
            build_project_native_baseline_timeline(
                shell.project_storage,
                active_song_id=active_song_id,
                active_song_version_id=active_song_version_id,
            )
        )
    shell._app.presentation_enricher = lambda presentation: apply_timeline_presentation_overlay(
        presentation,
        overlay=overlay,
    )
    shell._app.replace_timeline(timeline)
    restore_timeline_targets(
        timeline=shell._app.timeline,
        prior_presentation=current_presentation,
        current_presentation=shell.presentation(),
    )
    resolved_active_song_id = active_song_id or resolved_song_id
    resolved_active_song_version_id = active_song_version_id or resolved_song_version_id
    active_version_record = (
        shell.project_storage.song_versions.get(str(resolved_active_song_version_id))
        if resolved_active_song_version_id is not None
        else None
    )
    shell.session.active_song_id = resolved_active_song_id
    shell.session.active_song_version_id = resolved_active_song_version_id
    shell.session.active_song_version_ma3_timecode_pool_no = (
        active_version_record.ma3_timecode_pool_no if active_version_record is not None else None
    )
    shell.session.active_timeline_id = shell._app.timeline.id
    shell._sync_runtime_audio_from_presentation(shell.presentation())


def _replace_project_runtime(
    shell: ProjectLifecycleShell,
    *,
    project_storage: ProjectStorage,
    project_path: Path | None,
    runtime_audio: object | None,
) -> None:
    shell.project_storage = project_storage
    shell.project_path = project_path
    shell._app = build_runtime_timeline_application(
        project_storage=project_storage,
        sync_bridge=shell._sync_bridge,
        sync_service=shell._sync_service_override,
        runtime_audio=runtime_audio,
    )
    shell._last_pipeline_run_revision = 0
    shell._build_object_action_services()
    shell._draft_layers = []


def _install_project_runtime(
    shell: ProjectLifecycleShell,
    *,
    project_storage: ProjectStorage,
    project_path: Path | None,
    runtime_audio: object | None,
) -> None:
    previous_storage = shell.project_storage
    shell._pipeline_runs.shutdown()
    previous_storage.close()
    try:
        _replace_project_runtime(
            shell,
            project_storage=project_storage,
            project_path=project_path,
            runtime_audio=runtime_audio,
        )
    except Exception:
        project_storage.close()
        raise


def _paths_match(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _optional_positive_int(value: object) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _selection_after_song_delete(
    shell: ProjectLifecycleShell,
    deleted_song_id: str,
) -> tuple[SongId | None, SongVersionId | None]:
    current_song_id = str(shell.session.active_song_id) if shell.session.active_song_id is not None else None
    current_version_id = (
        SongVersionId(str(shell.session.active_song_version_id))
        if shell.session.active_song_version_id is not None
        else None
    )
    if current_song_id is not None and current_song_id != deleted_song_id:
        return SongId(current_song_id), current_version_id

    songs = shell.project_storage.songs.list_by_project(shell.project_storage.project.id)
    next_song = _adjacent_song_record(songs, deleted_song_id)
    if next_song is None:
        return None, None
    next_version_id = (
        SongVersionId(next_song.active_version_id)
        if next_song.active_version_id is not None
        else None
    )
    return SongId(next_song.id), next_version_id


def _selection_after_song_version_delete(
    shell: ProjectLifecycleShell,
    *,
    song_id: str,
    deleted_version_id: str,
) -> tuple[SongId | None, SongVersionId | None]:
    current_song_id = str(shell.session.active_song_id) if shell.session.active_song_id is not None else None
    current_version_id = (
        str(shell.session.active_song_version_id)
        if shell.session.active_song_version_id is not None
        else None
    )
    if current_version_id != deleted_version_id:
        if current_song_id is None:
            return None, None
        return SongId(current_song_id), shell.session.active_song_version_id

    versions = shell.project_storage.song_versions.list_by_song(song_id)
    next_version = _adjacent_version_record(versions, deleted_version_id)
    if next_version is not None:
        return SongId(song_id), SongVersionId(next_version.id)
    return _selection_after_song_delete(shell, song_id)


def _adjacent_song_record(
    songs: list[SongRecord],
    deleted_song_id: str,
) -> SongRecord | None:
    remaining_songs = [song for song in songs if song.id != deleted_song_id]
    if not remaining_songs:
        return None

    deleted_index = next(
        (index for index, song in enumerate(songs) if song.id == deleted_song_id),
        len(songs) - 1,
    )
    if deleted_index < len(remaining_songs):
        return remaining_songs[deleted_index]
    return remaining_songs[-1]


def _adjacent_version_record(
    versions: list[SongVersionRecord],
    deleted_version_id: str,
) -> SongVersionRecord | None:
    remaining_versions = [
        version for version in versions if version.id != deleted_version_id
    ]
    if not remaining_versions:
        return None

    deleted_index = next(
        (
            index
            for index, version in enumerate(versions)
            if version.id == deleted_version_id
        ),
        len(versions) - 1,
    )
    if deleted_index < len(remaining_versions):
        return remaining_versions[deleted_index]
    return remaining_versions[-1]

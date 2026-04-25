"""Project lifecycle helpers for the Qt app shell runtime.
Exists to isolate project reload and song/version selection flows from orchestration methods.
Connects ProjectStorage lifecycle and baseline timeline reloads to the Stage Zero shell.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.settings import AppSettingsService
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import SongId, SongVersionId
from echozero.application.sync.adapters import MA3SyncBridge
from echozero.application.sync.service import SyncService
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.models import Layer
from echozero.application.timeline.object_actions import descriptor_for_action
from echozero.application.timeline.pipeline_run_service import PipelineRunService
from echozero.foundry.review_server_controller import ReviewServerController
from echozero.persistence.audio import AudioImportOptions
from echozero.persistence.entities import SongRecord, SongVersionRecord
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_project_timeline import (
    apply_timeline_presentation_overlay,
    build_project_native_baseline_timeline,
)
from echozero.ui.qt.app_shell_runtime_services import build_runtime_timeline_application
from echozero.ui.qt.app_shell_timeline_state import restore_timeline_targets

logger = logging.getLogger(__name__)


class ProjectLifecycleShell(Protocol):
    _app: TimelineApplication
    _draft_layers: list[Layer]
    _is_dirty: bool
    _last_pipeline_run_revision: int
    _pipeline_runs: PipelineRunService
    _review_server_controller: ReviewServerController
    _sync_bridge: MA3SyncBridge | None
    _sync_service_override: SyncService | None
    _app_settings_service: AppSettingsService | None
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

    def run_object_action(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
    ) -> TimelinePresentation: ...


def new_project(shell: ProjectLifecycleShell, name: str = "EchoZero Project") -> None:
    working_dir_root = shell.project_storage.working_dir.parent
    project_storage = ProjectStorage.create_new(
        name=name,
        working_dir_root=working_dir_root,
    )
    shell._review_server_controller.stop()
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
    shell._review_server_controller.stop()
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
    *,
    run_import_pipeline: bool | None = None,
    import_pipeline_action_ids: tuple[str, ...] | None = None,
) -> TimelinePresentation:
    carried_draft_layers = bool(shell._draft_layers)
    audio_import_options = _resolve_audio_import_options(shell)
    song, version = shell.project_storage.import_song(
        title=title,
        audio_source=Path(audio_path),
        audio_import_options=audio_import_options,
    )
    shell._materialize_draft_layers(song_version_id=str(version.id))
    refresh_from_storage(
        shell,
        active_song_id=SongId(song.id),
        active_song_version_id=SongVersionId(version.id),
    )
    if carried_draft_layers:
        shell._select_active_source_layer()
    _run_song_import_pipeline_actions(
        shell,
        run_import_pipeline=run_import_pipeline,
        import_pipeline_action_ids=import_pipeline_action_ids,
    )
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
    transfer_layers: bool = False,
    transfer_layer_ids: list[str] | None = None,
    run_import_pipeline: bool | None = None,
    import_pipeline_action_ids: tuple[str, ...] | None = None,
) -> TimelinePresentation:
    song_record = shell.project_storage.songs.get(str(song_id))
    if song_record is None:
        raise ValueError(f"SongRecord not found: {song_id}")

    audio_import_options = _resolve_audio_import_options(shell)
    version = shell.project_storage.add_song_version(
        song_record.id,
        Path(audio_path),
        label=label,
        activate=activate,
        transfer_layers=transfer_layers,
        transfer_layer_ids=transfer_layer_ids,
        audio_import_options=audio_import_options,
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
    _run_song_import_pipeline_actions(
        shell,
        run_import_pipeline=run_import_pipeline,
        import_pipeline_action_ids=import_pipeline_action_ids,
    )
    shell._is_dirty = True
    shell._clear_history()
    return shell.presentation()


def list_song_version_transfer_layers(
    shell: ProjectLifecycleShell,
    song_id: str | SongId,
) -> list[tuple[str, str]]:
    song_record = shell.project_storage.songs.get(str(song_id))
    if song_record is None:
        raise ValueError(f"SongRecord not found: {song_id}")
    if song_record.active_version_id is None:
        return []

    return [
        (layer.id, layer.name)
        for layer in shell.project_storage.layers.list_by_version(song_record.active_version_id)
    ]


def reorder_songs(
    shell: ProjectLifecycleShell,
    song_ids: list[str],
) -> TimelinePresentation:
    shell.project_storage.reorder_songs(song_ids)
    active_song_id, active_song_version_id = _selection_after_song_reorder(shell, song_ids)
    refresh_from_storage(
        shell,
        active_song_id=active_song_id,
        active_song_version_id=active_song_version_id,
    )
    shell._is_dirty = True
    shell._clear_history()
    return shell.presentation()


def move_song(
    shell: ProjectLifecycleShell,
    song_id: str | SongId,
    *,
    steps: int,
) -> TimelinePresentation:
    songs = shell.project_storage.songs.list_by_project(shell.project_storage.project.id)
    if not songs:
        return shell.presentation()
    song_ids = [song.id for song in songs]
    resolved_song_id = str(song_id)
    if resolved_song_id not in song_ids:
        raise ValueError(f"SongRecord not found: {song_id}")
    if steps == 0:
        return shell.presentation()

    current_index = song_ids.index(resolved_song_id)
    target_index = max(0, min(len(song_ids) - 1, current_index + int(steps)))
    if target_index == current_index:
        return shell.presentation()

    song_ids.insert(target_index, song_ids.pop(current_index))
    return reorder_songs(shell, song_ids)


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


def _resolve_audio_import_options(shell: ProjectLifecycleShell) -> AudioImportOptions:
    settings_service = getattr(shell, "_app_settings_service", None)
    if settings_service is None:
        return AudioImportOptions(strip_ltc_timecode=True)
    try:
        preferences = settings_service.preferences()
    except Exception:
        return AudioImportOptions(strip_ltc_timecode=True)
    return AudioImportOptions(
        strip_ltc_timecode=bool(preferences.song_import.strip_ltc_timecode),
    )


def _resolve_song_import_pipeline_action_ids(
    shell: ProjectLifecycleShell,
    *,
    run_import_pipeline: bool | None,
    import_pipeline_action_ids: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if run_import_pipeline is False:
        return ()
    if import_pipeline_action_ids is not None:
        return _canonical_import_pipeline_action_ids(import_pipeline_action_ids)

    settings_service = getattr(shell, "_app_settings_service", None)
    if settings_service is None:
        return ()
    try:
        preferences = settings_service.preferences()
    except Exception:
        return ()
    return _canonical_import_pipeline_action_ids(
        preferences.song_import.pipeline_action_ids
    )


def _canonical_import_pipeline_action_ids(
    action_ids: tuple[str, ...],
) -> tuple[str, ...]:
    resolved: list[str] = []
    seen: set[str] = set()
    for action_id in action_ids:
        text = action_id.strip() if isinstance(action_id, str) else ""
        if not text:
            continue
        descriptor = descriptor_for_action(text)
        if descriptor is None:
            continue
        canonical_id = descriptor.action_id
        if canonical_id in seen:
            continue
        seen.add(canonical_id)
        resolved.append(canonical_id)
    return tuple(resolved)


def _resolve_import_source_audio_layer_id(shell: ProjectLifecycleShell) -> object | None:
    presentation = shell.presentation()
    for layer in presentation.layers:
        if str(layer.layer_id) == "source_audio":
            return layer.layer_id
    for layer in presentation.layers:
        if layer.kind is LayerKind.AUDIO:
            return layer.layer_id
    return "source_audio"


def _run_song_import_pipeline_actions(
    shell: ProjectLifecycleShell,
    *,
    run_import_pipeline: bool | None,
    import_pipeline_action_ids: tuple[str, ...] | None,
) -> None:
    action_ids = _resolve_song_import_pipeline_action_ids(
        shell,
        run_import_pipeline=run_import_pipeline,
        import_pipeline_action_ids=import_pipeline_action_ids,
    )
    if not action_ids:
        return

    source_layer_id = _resolve_import_source_audio_layer_id(shell)
    if source_layer_id is None:
        return

    for action_id in action_ids:
        try:
            shell.run_object_action(
                action_id,
                object_id=source_layer_id,
                object_type="layer",
            )
        except Exception as exc:
            logger.warning(
                "Import pipeline action '%s' failed: %s",
                action_id,
                exc,
            )


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


def _selection_after_song_reorder(
    shell: ProjectLifecycleShell,
    ordered_song_ids: list[str],
) -> tuple[SongId | None, SongVersionId | None]:
    if not ordered_song_ids:
        return None, None

    current_song_id = str(shell.session.active_song_id) if shell.session.active_song_id is not None else None
    if current_song_id in ordered_song_ids:
        selected_song_id = current_song_id
    else:
        selected_song_id = ordered_song_ids[0]

    song_record = shell.project_storage.songs.get(selected_song_id)
    if song_record is None:
        return None, None
    selected_version_id = (
        SongVersionId(song_record.active_version_id)
        if song_record.active_version_id is not None
        else None
    )
    return SongId(song_record.id), selected_version_id


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

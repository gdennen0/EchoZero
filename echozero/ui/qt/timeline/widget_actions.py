"""Timeline widget action routing and dialog orchestration.
Exists to keep TimelineWidget focused on rendering and input state.
Connects inspector and transfer actions to app intents and canonical object actions.
"""

from __future__ import annotations

from collections.abc import Callable
import inspect
from pathlib import Path
import re
from typing import Protocol, cast

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
    QWidget,
)

from echozero.application.presentation.inspector_contract import InspectorAction
from echozero.application.presentation.models import (
    ManualPullFlowPresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId
from echozero.application.timeline.intents import (
    SetGain,
)
from echozero.application.timeline.object_actions import (
    ObjectActionSettingsSession,
    descriptor_for_action,
    object_action_descriptors,
)
from echozero.models.paths import ensure_installed_models_dir
from echozero.ui.qt.pipeline_settings_browser_dialog import PipelineSettingsBrowserDialog
from echozero.ui.qt.timeline.widget_action_contract_mixin import (
    _AddSongRuntimeShell,
    _AddSongVersionRuntimeShell,
    _DeleteSongRuntimeShell,
    _MoveSongRuntimeShell,
    _ReorderSongsRuntimeShell,
    _SwitchSongVersionRuntimeShell,
    _TimelineRuntimeShell,
    _coerce_event_id,
    _coerce_layer_id,
    _coerce_take_id,
    TimelineWidgetContractActionMixin,
)
from echozero.ui.qt.timeline.widget_action_transfer_mixin import (
    TimelineWidgetTransferActionMixin,
)
from echozero.ui.qt.timeline.manual_pull import (
    ManualPullTimelineSelectionResult,
)

_NATURAL_TOKEN_PATTERN = re.compile(r"(\d+)")
_FINAL_PIPELINE_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})


class _ObjectActionSettingsRuntimeShell(_TimelineRuntimeShell, Protocol):
    def open_object_action_session(
        self,
        action_id: str,
        params: dict[str, object],
        *,
        object_id: LayerId | None = None,
        object_type: str | None = None,
    ) -> ObjectActionSettingsSession: ...

    def dispatch_object_action_command(
        self, session_id: str, command: object
    ) -> ObjectActionSettingsSession: ...


class _RequestObjectActionRuntimeShell(_TimelineRuntimeShell, Protocol):
    def request_object_action_run(
        self,
        action_id: str,
        params: dict[str, object],
        *,
        object_id: LayerId,
        object_type: str,
    ) -> str: ...


class _PipelineRunStateLookupRuntimeShell(_TimelineRuntimeShell, Protocol):
    def get_pipeline_run_state(self, run_id: str) -> object | None: ...


class _RunObjectActionRuntimeShell(_TimelineRuntimeShell, Protocol):
    def run_object_action(
        self,
        action_id: str,
        params: dict[str, object],
        *,
        object_id: LayerId,
        object_type: str,
    ) -> TimelinePresentation: ...

    def reorder_songs(self, song_ids: list[str]) -> TimelinePresentation | None: ...


class TimelineWidgetActionRouter(
    TimelineWidgetContractActionMixin,
    TimelineWidgetTransferActionMixin,
):
    """Routes inspector and transfer actions for the timeline widget."""

    def __init__(
        self,
        *,
        widget: QWidget,
        dispatch: Callable[[object], None],
        get_presentation: Callable[[], TimelinePresentation],
        set_presentation: Callable[[TimelinePresentation], None],
        resolve_runtime_shell: Callable[[], _TimelineRuntimeShell | None],
        selected_event_ids_for_selected_layers: Callable[[], list[EventId]],
        open_manual_pull_timeline_popup: (
            Callable[[ManualPullFlowPresentation], ManualPullTimelineSelectionResult | None] | None
        ) = None,
        input_dialog: type[QInputDialog] = QInputDialog,
        file_dialog: type[QFileDialog] = QFileDialog,
        message_box: type[QMessageBox] = QMessageBox,
        resolve_models_dir: Callable[[], Path] = ensure_installed_models_dir,
    ) -> None:
        self._widget = widget
        self._dispatch = dispatch
        self._get_presentation = get_presentation
        self._set_presentation = set_presentation
        self._resolve_runtime_shell = resolve_runtime_shell
        self._selected_event_ids_for_selected_layers = selected_event_ids_for_selected_layers
        self._open_manual_pull_timeline_popup: Callable[
            [ManualPullFlowPresentation], ManualPullTimelineSelectionResult | None
        ] = (
            open_manual_pull_timeline_popup
            if open_manual_pull_timeline_popup is not None
            else self._default_open_manual_pull_timeline_popup
        )
        self._input_dialog = input_dialog
        self._file_dialog = file_dialog
        self._message_box = message_box
        self._resolve_models_dir = resolve_models_dir

    def import_dropped_audio_path(
        self,
        audio_path: str,
        *,
        force_new_song: bool = False,
        target_song_id: str | None = None,
        target_song_title: str | None = None,
    ) -> bool:
        """Import one dropped audio path as a song or a new version for the active song."""

        return self.import_dropped_audio_paths(
            (audio_path,),
            force_new_song=force_new_song,
            target_song_id=target_song_id,
            target_song_title=target_song_title,
        )

    def import_dropped_audio_paths(
        self,
        audio_paths: tuple[str, ...],
        *,
        force_new_song: bool = False,
        target_song_id: str | None = None,
        target_song_title: str | None = None,
    ) -> bool:
        """Import one or more dropped audio paths as songs or versions."""

        runtime = self._resolve_runtime_shell()
        if runtime is None:
            self._message_box.warning(
                self._widget,
                "Add Song",
                "This runtime does not support adding songs from a drop.",
            )
            return False

        resolved_paths = self._normalized_audio_paths(audio_paths)
        if not resolved_paths:
            return False
        pipeline_action_ids = self._configured_import_pipeline_actions(runtime)
        if len(resolved_paths) == 1:
            return self._import_single_audio_path(
                runtime=runtime,
                audio_path=resolved_paths[0],
                force_new_song=force_new_song,
                target_song_id=target_song_id,
                target_song_title=target_song_title,
                pipeline_action_ids=pipeline_action_ids,
            )

        mode = self._prompt_multi_import_mode(
            count=len(resolved_paths),
            target_song_id=target_song_id,
            target_song_title=target_song_title,
            force_new_song=force_new_song,
        )
        if mode == "cancel":
            return True

        run_pipeline_actions = bool(pipeline_action_ids)
        if pipeline_action_ids:
            decision = self._prompt_run_import_pipeline_actions(
                action_ids=pipeline_action_ids,
                import_count=len(resolved_paths),
            )
            if decision is None:
                return True
            run_pipeline_actions = decision

        if mode == "target_song_versions":
            resolved_target_song_id = target_song_id.strip() if isinstance(target_song_id, str) else ""
            if not resolved_target_song_id:
                self._message_box.warning(
                    self._widget,
                    "Import Songs",
                    "A target song is required to import all files as versions.",
                )
                return False
            return self._import_many_as_versions(
                runtime=runtime,
                target_song_id=resolved_target_song_id,
                audio_paths=resolved_paths,
                run_pipeline_actions=run_pipeline_actions,
                pipeline_action_ids=pipeline_action_ids,
            )

        return self._import_many_as_new_songs(
            runtime=runtime,
            audio_paths=resolved_paths,
            insertion_mode=mode,
            target_song_id=target_song_id,
            run_pipeline_actions=run_pipeline_actions,
            pipeline_action_ids=pipeline_action_ids,
        )

    def _import_single_audio_path(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        audio_path: str,
        force_new_song: bool,
        target_song_id: str | None,
        target_song_title: str | None,
        pipeline_action_ids: tuple[str, ...],
    ) -> bool:
        resolved_path = str(Path(audio_path))
        presentation = self._get_presentation()
        active_song_id = presentation.active_song_id.strip()
        resolved_target_song_id = target_song_id.strip() if target_song_id is not None else ""
        if resolved_target_song_id and not force_new_song:
            if self._prompt_to_add_drop_to_song(
                runtime=runtime,
                song_id=resolved_target_song_id,
                song_title=target_song_title,
                audio_path=resolved_path,
                pipeline_action_ids=pipeline_action_ids,
            ):
                return True
            force_new_song = True
        if active_song_id and not force_new_song:
            version_runtime = cast(_AddSongVersionRuntimeShell | None, runtime)
            if not callable(getattr(version_runtime, "add_song_version", None)):
                self._message_box.warning(
                    self._widget,
                    "Add Version",
                    "This runtime does not support creating a song version from a drop.",
                )
                return False
            active_title = presentation.active_song_title.strip() or "the active song"
            reply = self._message_box.question(
                self._widget,
                "Create New Version",
                (
                    "This timeline already has a source song loaded.\n\n"
                    f"Create a new version of \"{active_title}\" from "
                    f"\"{Path(resolved_path).name}\"?"
                ),
                self._message_box.StandardButton.Yes | self._message_box.StandardButton.No,
                self._message_box.StandardButton.No,
            )
            if reply != self._message_box.StandardButton.Yes:
                return True
            if not self._invoke_add_song_version(
                runtime,
                active_song_id,
                resolved_path,
                run_import_pipeline=bool(pipeline_action_ids),
                pipeline_action_ids=pipeline_action_ids or None,
            ):
                return False
            return True

        title = Path(resolved_path).stem.strip() or "Imported Song"
        if not self._invoke_add_song_from_path(
            runtime,
            title,
            resolved_path,
            run_import_pipeline=bool(pipeline_action_ids),
            pipeline_action_ids=pipeline_action_ids or None,
        ):
            return False
        return True

    def _import_many_as_versions(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        target_song_id: str,
        audio_paths: tuple[str, ...],
        run_pipeline_actions: bool,
        pipeline_action_ids: tuple[str, ...],
    ) -> bool:
        queue_pipeline_runs = self._can_enqueue_import_pipeline_runs(
            runtime,
            run_pipeline_actions=run_pipeline_actions,
            pipeline_action_ids=pipeline_action_ids,
        )
        imported_targets: list[tuple[str, str]] = []
        imported_count, canceled = self._run_import_steps(
            audio_paths=audio_paths,
            title="Import Versions",
            action_label="version",
            show_progress_dialog=not queue_pipeline_runs,
            import_one=lambda audio_path: self._import_one_as_version(
                runtime=runtime,
                target_song_id=target_song_id,
                audio_path=audio_path,
                run_pipeline_actions=run_pipeline_actions and not queue_pipeline_runs,
                pipeline_action_ids=(
                    pipeline_action_ids
                    if not queue_pipeline_runs
                    else ()
                ),
                imported_targets=imported_targets,
                collect_pipeline_target=queue_pipeline_runs,
            ),
        )
        if canceled:
            self._show_import_canceled_summary(
                imported_count=imported_count,
                total_count=len(audio_paths),
                noun="version",
            )
        if queue_pipeline_runs and imported_targets:
            self._enqueue_import_pipeline_runs(
                runtime=runtime,
                imported_targets=tuple(imported_targets),
                action_ids=pipeline_action_ids,
            )
        return imported_count > 0

    def _import_many_as_new_songs(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        audio_paths: tuple[str, ...],
        insertion_mode: str,
        target_song_id: str | None,
        run_pipeline_actions: bool,
        pipeline_action_ids: tuple[str, ...],
    ) -> bool:
        initial_song_order = [
            option.song_id for option in self._get_presentation().available_songs
        ]
        imported_song_ids: list[str] = []
        queue_pipeline_runs = self._can_enqueue_import_pipeline_runs(
            runtime,
            run_pipeline_actions=run_pipeline_actions,
            pipeline_action_ids=pipeline_action_ids,
        )
        imported_targets: list[tuple[str, str]] = []
        imported_count, canceled = self._run_import_steps(
            audio_paths=audio_paths,
            title="Import Songs",
            action_label="song",
            show_progress_dialog=not queue_pipeline_runs,
            import_one=lambda audio_path: self._import_one_as_new_song(
                runtime=runtime,
                audio_path=audio_path,
                run_pipeline_actions=run_pipeline_actions and not queue_pipeline_runs,
                pipeline_action_ids=(
                    pipeline_action_ids
                    if not queue_pipeline_runs
                    else ()
                ),
                initial_song_order=initial_song_order,
                imported_song_ids=imported_song_ids,
                imported_targets=imported_targets,
                collect_pipeline_target=queue_pipeline_runs,
            ),
        )
        imported_any = imported_count > 0
        if imported_any and imported_song_ids and insertion_mode != "new_at_end":
            self._reposition_imported_songs(
                runtime=runtime,
                imported_song_ids=imported_song_ids,
                insertion_mode=insertion_mode,
                target_song_id=target_song_id,
            )
        if canceled:
            self._show_import_canceled_summary(
                imported_count=imported_count,
                total_count=len(audio_paths),
                noun="song",
            )
        if queue_pipeline_runs and imported_targets:
            self._enqueue_import_pipeline_runs(
                runtime=runtime,
                imported_targets=tuple(imported_targets),
                action_ids=pipeline_action_ids,
            )
        return imported_any

    def _import_one_as_new_song(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        audio_path: str,
        run_pipeline_actions: bool,
        pipeline_action_ids: tuple[str, ...],
        initial_song_order: list[str],
        imported_song_ids: list[str],
        imported_targets: list[tuple[str, str]],
        collect_pipeline_target: bool,
    ) -> bool:
        title = Path(audio_path).stem.strip() or "Imported Song"
        if not self._invoke_add_song_from_path(
            runtime,
            title,
            audio_path,
            run_import_pipeline=run_pipeline_actions,
            pipeline_action_ids=pipeline_action_ids or None,
        ):
            return False
        active_song_id = self._get_presentation().active_song_id.strip()
        if (
            active_song_id
            and active_song_id not in initial_song_order
            and active_song_id not in imported_song_ids
        ):
            imported_song_ids.append(active_song_id)
        if collect_pipeline_target:
            self._capture_import_pipeline_target(
                source_label=Path(audio_path).name,
                imported_targets=imported_targets,
            )
        return True

    def _import_one_as_version(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        target_song_id: str,
        audio_path: str,
        run_pipeline_actions: bool,
        pipeline_action_ids: tuple[str, ...],
        imported_targets: list[tuple[str, str]],
        collect_pipeline_target: bool,
    ) -> bool:
        if not self._invoke_add_song_version(
            runtime,
            target_song_id,
            audio_path,
            run_import_pipeline=run_pipeline_actions,
            pipeline_action_ids=pipeline_action_ids or None,
        ):
            return False
        if collect_pipeline_target:
            self._capture_import_pipeline_target(
                source_label=Path(audio_path).name,
                imported_targets=imported_targets,
            )
        return True

    def _run_import_steps(
        self,
        *,
        audio_paths: tuple[str, ...],
        title: str,
        action_label: str,
        show_progress_dialog: bool,
        import_one: Callable[[str], bool],
    ) -> tuple[int, bool]:
        """Run import work file-by-file while yielding UI updates each step."""

        total_count = len(audio_paths)
        if total_count <= 0:
            return (0, False)

        progress_dialog: QProgressDialog | None = None
        if show_progress_dialog and total_count > 1:
            progress_dialog = QProgressDialog(
                f"Preparing to import {total_count} {action_label}s...",
                "Cancel",
                0,
                total_count,
                self._widget,
            )
            progress_dialog.setWindowTitle(title)
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setAutoClose(True)
            progress_dialog.setAutoReset(True)
            progress_dialog.setValue(0)
            progress_dialog.show()
            QApplication.processEvents()

        imported_count = 0
        canceled = False
        for index, audio_path in enumerate(audio_paths, start=1):
            if progress_dialog is not None:
                progress_dialog.setLabelText(
                    f"Importing {action_label} {index} of {total_count}:\n{Path(audio_path).name}"
                )
                progress_dialog.setValue(index - 1)
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    canceled = True
                    break
            else:
                QApplication.processEvents()

            if import_one(audio_path):
                imported_count += 1

            if progress_dialog is not None:
                progress_dialog.setValue(index)
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    canceled = True
                    break
            else:
                QApplication.processEvents()

        if progress_dialog is not None:
            progress_dialog.close()

        return (imported_count, canceled)

    def _capture_import_pipeline_target(
        self,
        *,
        source_label: str,
        imported_targets: list[tuple[str, str]],
    ) -> None:
        presentation = self._get_presentation()
        song_version_id = presentation.active_song_version_id.strip()
        if not song_version_id:
            return
        imported_targets.append((song_version_id, source_label))

    def _can_enqueue_import_pipeline_runs(
        self,
        runtime: _TimelineRuntimeShell,
        *,
        run_pipeline_actions: bool,
        pipeline_action_ids: tuple[str, ...],
    ) -> bool:
        if not run_pipeline_actions or not pipeline_action_ids:
            return False
        request_runtime = cast(_RequestObjectActionRuntimeShell | None, runtime)
        state_runtime = cast(_PipelineRunStateLookupRuntimeShell | None, runtime)
        switch_runtime = cast(_SwitchSongVersionRuntimeShell | None, runtime)
        return (
            callable(getattr(request_runtime, "request_object_action_run", None))
            and callable(getattr(state_runtime, "get_pipeline_run_state", None))
            and callable(getattr(switch_runtime, "switch_song_version", None))
        )

    def _enqueue_import_pipeline_runs(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        imported_targets: tuple[tuple[str, str], ...],
        action_ids: tuple[str, ...],
    ) -> None:
        pending_items = [
            (runtime, song_version_id, action_id, source_label)
            for song_version_id, source_label in imported_targets
            for action_id in action_ids
        ]
        if not pending_items:
            return

        existing_queue = getattr(self, "_import_pipeline_queue_items", None)
        if isinstance(existing_queue, list):
            existing_queue.extend(pending_items)
        else:
            setattr(self, "_import_pipeline_queue_items", pending_items)

        timer = cast(QTimer | None, getattr(self, "_import_pipeline_queue_timer", None))
        if timer is None:
            timer = QTimer(self._widget)
            timer.setInterval(120)
            timer.timeout.connect(self._advance_import_pipeline_queue)
            setattr(self, "_import_pipeline_queue_timer", timer)
        if not timer.isActive():
            timer.start()
        self._advance_import_pipeline_queue()

    def _advance_import_pipeline_queue(self) -> None:
        timer = cast(QTimer | None, getattr(self, "_import_pipeline_queue_timer", None))
        queue = cast(list[tuple[_TimelineRuntimeShell, str, str, str]], getattr(self, "_import_pipeline_queue_items", []))
        current = cast(tuple[_TimelineRuntimeShell, str, str, str] | None, getattr(self, "_import_pipeline_current_run", None))

        if current is not None:
            current_runtime, run_id, _action_id, _source_label = current
            state_runtime = cast(_PipelineRunStateLookupRuntimeShell | None, current_runtime)
            lookup = (
                state_runtime.get_pipeline_run_state
                if state_runtime is not None
                and callable(getattr(state_runtime, "get_pipeline_run_state", None))
                else None
            )
            if callable(lookup):
                try:
                    state = lookup(run_id)
                except Exception:
                    state = None
                status = str(getattr(state, "status", "")).strip().lower() if state is not None else ""
                if status and status not in _FINAL_PIPELINE_RUN_STATUSES:
                    return
            setattr(self, "_import_pipeline_current_run", None)

        while queue:
            queued_runtime, song_version_id, action_id, source_label = queue.pop(0)

            switch_runtime = cast(_SwitchSongVersionRuntimeShell | None, queued_runtime)
            switch_song_version = (
                switch_runtime.switch_song_version
                if switch_runtime is not None
                and callable(getattr(switch_runtime, "switch_song_version", None))
                else None
            )
            if not callable(switch_song_version):
                continue
            try:
                updated = switch_song_version(song_version_id)
            except Exception as exc:
                self._message_box.warning(
                    self._widget,
                    "Import Pipeline Actions",
                    f"{source_label}: {exc}",
                )
                continue
            self._set_presentation(updated if updated is not None else queued_runtime.presentation())

            source_layer_id = self._resolve_import_source_audio_layer_id(queued_runtime)
            if source_layer_id is None:
                continue

            request_runtime = cast(_RequestObjectActionRuntimeShell | None, queued_runtime)
            request_run = (
                request_runtime.request_object_action_run
                if request_runtime is not None
                and callable(getattr(request_runtime, "request_object_action_run", None))
                else None
            )
            if not callable(request_run):
                continue
            try:
                run_id = request_run(
                    action_id,
                    {},
                    object_id=source_layer_id,
                    object_type="layer",
                )
            except Exception as exc:
                self._message_box.warning(
                    self._widget,
                    "Import Pipeline Actions",
                    f"{source_label}: {exc}",
                )
                continue
            if not isinstance(run_id, str) or not run_id.strip():
                continue
            setattr(
                self,
                "_import_pipeline_current_run",
                (queued_runtime, run_id, action_id, source_label),
            )
            return

        if timer is not None and timer.isActive():
            timer.stop()

    def _show_import_canceled_summary(
        self,
        *,
        imported_count: int,
        total_count: int,
        noun: str,
    ) -> None:
        remaining = max(0, total_count - imported_count)
        imported_word = noun if imported_count == 1 else f"{noun}s"
        remaining_word = noun if remaining == 1 else f"{noun}s"
        self._message_box.information(
            self._widget,
            "Import Canceled",
            (
                f"Imported {imported_count} {imported_word} before canceling.\n"
                f"{remaining} {remaining_word} were not imported."
            ),
        )

    def _reposition_imported_songs(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        imported_song_ids: list[str],
        insertion_mode: str,
        target_song_id: str | None,
    ) -> bool:
        reorder_runtime = cast(_ReorderSongsRuntimeShell | None, runtime)
        if not callable(getattr(reorder_runtime, "reorder_songs", None)):
            return False
        current_order = [
            song.song_id for song in self._get_presentation().available_songs
        ]
        imported_set = set(imported_song_ids)
        base_order = [song_id for song_id in current_order if song_id not in imported_set]
        insertion_index = len(base_order)
        resolved_target_song_id = target_song_id.strip() if isinstance(target_song_id, str) else ""
        if resolved_target_song_id in base_order and insertion_mode in {
            "new_before_target",
            "new_after_target",
        }:
            target_index = base_order.index(resolved_target_song_id)
            insertion_index = (
                target_index
                if insertion_mode == "new_before_target"
                else target_index + 1
            )
        reordered_song_ids = (
            base_order[:insertion_index]
            + list(imported_song_ids)
            + base_order[insertion_index:]
        )
        if reordered_song_ids == current_order:
            return True
        assert reorder_runtime is not None
        try:
            updated = reorder_runtime.reorder_songs(reordered_song_ids)
        except Exception as exc:
            self._message_box.warning(self._widget, "Import Songs", str(exc))
            return False
        self._set_presentation(
            updated if updated is not None else runtime.presentation()
        )
        return True

    def _invoke_add_song_from_path(
        self,
        runtime: _TimelineRuntimeShell,
        title: str,
        audio_path: str,
        *,
        run_import_pipeline: bool | None = None,
        pipeline_action_ids: tuple[str, ...] | None = None,
    ) -> bool:
        song_runtime = cast(_AddSongRuntimeShell | None, runtime)
        if not callable(getattr(song_runtime, "add_song_from_path", None)):
            self._message_box.warning(
                self._widget,
                "Add Song",
                "This runtime does not support adding songs from a drop.",
            )
            return False
        assert song_runtime is not None
        supports_native_pipeline_control = self._method_supports_any_kwargs(
            song_runtime.add_song_from_path,
            "run_import_pipeline",
            "import_pipeline_action_ids",
        )
        defer_pipeline_runs, deferred_action_ids = self._resolve_deferred_import_pipeline_runs(
            runtime,
            run_import_pipeline=run_import_pipeline,
            pipeline_action_ids=pipeline_action_ids,
        )
        resolved_run_import_pipeline: bool | None = run_import_pipeline
        resolved_pipeline_action_ids: tuple[str, ...] | None = pipeline_action_ids
        if defer_pipeline_runs:
            resolved_run_import_pipeline = False
            resolved_pipeline_action_ids = None
        try:
            updated = self._invoke_with_supported_kwargs(
                song_runtime.add_song_from_path,
                title,
                audio_path,
                run_import_pipeline=resolved_run_import_pipeline,
                import_pipeline_action_ids=resolved_pipeline_action_ids,
            )
        except Exception as exc:
            self._message_box.warning(self._widget, "Add Song", str(exc))
            return False
        self._set_presentation(updated if updated is not None else runtime.presentation())
        if defer_pipeline_runs:
            imported_targets: list[tuple[str, str]] = []
            self._capture_import_pipeline_target(
                source_label=Path(audio_path).name,
                imported_targets=imported_targets,
            )
            if imported_targets:
                self._enqueue_import_pipeline_runs(
                    runtime=runtime,
                    imported_targets=tuple(imported_targets),
                    action_ids=deferred_action_ids,
                )
        self._run_legacy_import_pipeline_actions_if_needed(
            runtime=runtime,
            source_label=Path(audio_path).name,
            run_import_pipeline=resolved_run_import_pipeline,
            pipeline_action_ids=resolved_pipeline_action_ids,
            supports_native_pipeline_control=supports_native_pipeline_control,
        )
        return True

    def _invoke_add_song_version(
        self,
        runtime: _TimelineRuntimeShell,
        song_id: str,
        audio_path: str,
        *,
        label: str | None = None,
        transfer_layers: bool = False,
        transfer_layer_ids: list[str] | None = None,
        run_import_pipeline: bool | None = None,
        pipeline_action_ids: tuple[str, ...] | None = None,
    ) -> bool:
        version_runtime = cast(_AddSongVersionRuntimeShell | None, runtime)
        if not callable(getattr(version_runtime, "add_song_version", None)):
            self._message_box.warning(
                self._widget,
                "Add Version",
                "This runtime does not support creating a song version from a drop.",
            )
            return False
        assert version_runtime is not None
        supports_native_pipeline_control = self._method_supports_any_kwargs(
            version_runtime.add_song_version,
            "run_import_pipeline",
            "import_pipeline_action_ids",
        )
        defer_pipeline_runs, deferred_action_ids = self._resolve_deferred_import_pipeline_runs(
            runtime,
            run_import_pipeline=run_import_pipeline,
            pipeline_action_ids=pipeline_action_ids,
        )
        resolved_run_import_pipeline: bool | None = run_import_pipeline
        resolved_pipeline_action_ids: tuple[str, ...] | None = pipeline_action_ids
        if defer_pipeline_runs:
            resolved_run_import_pipeline = False
            resolved_pipeline_action_ids = None
        try:
            updated = self._invoke_with_supported_kwargs(
                version_runtime.add_song_version,
                song_id,
                audio_path,
                label=label,
                transfer_layers=transfer_layers,
                transfer_layer_ids=transfer_layer_ids,
                run_import_pipeline=resolved_run_import_pipeline,
                import_pipeline_action_ids=resolved_pipeline_action_ids,
            )
        except Exception as exc:
            self._message_box.warning(self._widget, "Add Version", str(exc))
            return False
        self._set_presentation(updated if updated is not None else runtime.presentation())
        if defer_pipeline_runs:
            imported_targets: list[tuple[str, str]] = []
            self._capture_import_pipeline_target(
                source_label=Path(audio_path).name,
                imported_targets=imported_targets,
            )
            if imported_targets:
                self._enqueue_import_pipeline_runs(
                    runtime=runtime,
                    imported_targets=tuple(imported_targets),
                    action_ids=deferred_action_ids,
                )
        self._run_legacy_import_pipeline_actions_if_needed(
            runtime=runtime,
            source_label=Path(audio_path).name,
            run_import_pipeline=resolved_run_import_pipeline,
            pipeline_action_ids=resolved_pipeline_action_ids,
            supports_native_pipeline_control=supports_native_pipeline_control,
        )
        return True

    def _resolve_deferred_import_pipeline_runs(
        self,
        runtime: _TimelineRuntimeShell,
        *,
        run_import_pipeline: bool | None,
        pipeline_action_ids: tuple[str, ...] | None,
    ) -> tuple[bool, tuple[str, ...]]:
        action_ids = tuple(
            action_id.strip()
            for action_id in (pipeline_action_ids or ())
            if isinstance(action_id, str) and action_id.strip()
        )
        should_defer = self._can_enqueue_import_pipeline_runs(
            runtime,
            run_pipeline_actions=run_import_pipeline is not False,
            pipeline_action_ids=action_ids,
        )
        return (should_defer, action_ids)

    def _run_legacy_import_pipeline_actions_if_needed(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        source_label: str,
        run_import_pipeline: bool | None,
        pipeline_action_ids: tuple[str, ...] | None,
        supports_native_pipeline_control: bool,
    ) -> None:
        """Run import pipeline actions when older runtimes lack import kwargs support."""

        if supports_native_pipeline_control:
            return
        action_ids = tuple(
            action_id.strip()
            for action_id in (pipeline_action_ids or ())
            if isinstance(action_id, str) and action_id.strip()
        )
        if not action_ids:
            return
        if run_import_pipeline is False:
            return

        run_runtime = cast(_RunObjectActionRuntimeShell | None, runtime)
        run_object_action = (
            run_runtime.run_object_action
            if run_runtime is not None and callable(getattr(run_runtime, "run_object_action", None))
            else None
        )
        if not callable(run_object_action):
            return

        source_layer_id = self._resolve_import_source_audio_layer_id(runtime)
        if source_layer_id is None:
            return

        for action_id in action_ids:
            try:
                assert run_runtime is not None
                updated = run_runtime.run_object_action(
                    action_id,
                    {},
                    object_id=source_layer_id,
                    object_type="layer",
                )
            except Exception as exc:
                self._message_box.warning(
                    self._widget,
                    "Import Pipeline Actions",
                    f"{source_label}: {exc}",
                )
                continue
            self._set_presentation(updated if updated is not None else runtime.presentation())

    @staticmethod
    def _method_supports_any_kwargs(
        method: Callable[..., object],
        *kwargs: str,
    ) -> bool:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return True

        parameters = signature.parameters
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return True
        return any(keyword in parameters for keyword in kwargs)

    def _resolve_import_source_audio_layer_id(
        self,
        runtime: _TimelineRuntimeShell,
    ) -> LayerId | None:
        presentation = self._get_presentation()
        for layer in presentation.layers:
            if str(layer.layer_id).strip() == "source_audio":
                return cast(LayerId, layer.layer_id)
        for layer in presentation.layers:
            if layer.kind is LayerKind.AUDIO:
                return cast(LayerId, layer.layer_id)
        runtime_presentation = runtime.presentation()
        for layer in runtime_presentation.layers:
            if str(layer.layer_id).strip() == "source_audio":
                return cast(LayerId, layer.layer_id)
        for layer in runtime_presentation.layers:
            if layer.kind is LayerKind.AUDIO:
                return cast(LayerId, layer.layer_id)
        return cast(LayerId, "source_audio")

    def _configured_import_pipeline_actions(
        self,
        runtime: _TimelineRuntimeShell,
    ) -> tuple[str, ...]:
        service = getattr(runtime, "app_settings_service", None)
        if service is None or not callable(getattr(service, "preferences", None)):
            return ()
        try:
            preferences = service.preferences()
        except Exception:
            return ()
        configured = getattr(preferences, "song_import", None)
        action_ids = (
            tuple(getattr(configured, "pipeline_action_ids", ()))
            if configured is not None
            else ()
        )
        resolved: list[str] = []
        seen: set[str] = set()
        for action_id in action_ids:
            text = str(action_id).strip()
            descriptor = descriptor_for_action(text)
            if descriptor is None:
                continue
            canonical_id = descriptor.action_id
            if canonical_id in seen:
                continue
            seen.add(canonical_id)
            resolved.append(canonical_id)
        return tuple(resolved)

    def _prompt_multi_import_mode(
        self,
        *,
        count: int,
        target_song_id: str | None,
        target_song_title: str | None,
        force_new_song: bool,
    ) -> str:
        options: list[tuple[str, str]] = [
            ("Import as new songs (append to end)", "new_at_end"),
        ]
        resolved_target_song_id = target_song_id.strip() if isinstance(target_song_id, str) else ""
        if resolved_target_song_id:
            target_title = self._resolve_drop_target_song_title(
                resolved_target_song_id,
                target_song_title,
            )
            options.insert(
                0,
                (f'Import as new songs (after "{target_title}")', "new_after_target"),
            )
            options.insert(
                1,
                (f'Import as new songs (before "{target_title}")', "new_before_target"),
            )
            if not force_new_song:
                options.append(
                    (f'Add all as versions to "{target_title}"', "target_song_versions")
                )

        labels = [label for label, _mode in options]
        chosen_label, accepted = self._input_dialog.getItem(
            self._widget,
            f"Import {count} Songs",
            "How should these files be imported?",
            labels,
            0,
            False,
        )
        if not accepted:
            return "cancel"
        for label, mode in options:
            if label == chosen_label:
                return mode
        return "cancel"

    def _prompt_run_import_pipeline_actions(
        self,
        *,
        action_ids: tuple[str, ...],
        import_count: int,
    ) -> bool | None:
        labels = ", ".join(self._action_label(action_id) for action_id in action_ids)
        reply = self._message_box.question(
            self._widget,
            "Import Pipeline Actions",
            (
                f"Run configured pipeline actions for each imported file ({import_count})?\n\n"
                f"Configured: {labels}"
            ),
            self._message_box.StandardButton.Yes
            | self._message_box.StandardButton.No
            | self._message_box.StandardButton.Cancel,
            self._message_box.StandardButton.Yes,
        )
        if reply == self._message_box.StandardButton.Cancel:
            return None
        return reply == self._message_box.StandardButton.Yes

    @staticmethod
    def _invoke_with_supported_kwargs(
        method: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(*args, **kwargs)

        parameters = signature.parameters
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_var_kwargs:
            return method(*args, **kwargs)

        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }
        return method(*args, **supported_kwargs)

    def _action_label(self, action_id: str) -> str:
        descriptor = descriptor_for_action(action_id)
        if descriptor is None:
            return action_id
        return descriptor.label

    def _normalized_audio_paths(self, audio_paths: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        resolved: list[str] = []
        for path in audio_paths:
            normalized = str(Path(str(path)))
            if normalized in seen:
                continue
            seen.add(normalized)
            resolved.append(normalized)
        resolved.sort(key=lambda value: _natural_sort_key(Path(value).name))
        return tuple(resolved)

    def _prompt_to_add_drop_to_song(
        self,
        *,
        runtime: _TimelineRuntimeShell,
        song_id: str,
        song_title: str | None,
        audio_path: str,
        pipeline_action_ids: tuple[str, ...] = (),
    ) -> bool:
        version_runtime = cast(_AddSongVersionRuntimeShell | None, runtime)
        if not callable(getattr(version_runtime, "add_song_version", None)):
            self._message_box.warning(
                self._widget,
                "Add Version",
                "This runtime does not support creating a song version from a drop.",
            )
            return True

        target_title = self._resolve_drop_target_song_title(song_id, song_title)
        reply = self._message_box.question(
            self._widget,
            "Add to Existing Song",
            (
                f'Add "{Path(audio_path).name}" to "{target_title}" as a new version?\n\n'
                "Choose No to import it as a new song instead."
            ),
            self._message_box.StandardButton.Yes
            | self._message_box.StandardButton.No
            | self._message_box.StandardButton.Cancel,
            self._message_box.StandardButton.Yes,
        )
        if reply == self._message_box.StandardButton.Cancel:
            return True
        if reply == self._message_box.StandardButton.No:
            return False
        if not self._invoke_add_song_version(
            runtime,
            song_id,
            audio_path,
            run_import_pipeline=bool(pipeline_action_ids),
            pipeline_action_ids=pipeline_action_ids or None,
        ):
            return True
        return True

    def _resolve_drop_target_song_title(self, song_id: str, song_title: str | None) -> str:
        resolved_title = song_title.strip() if song_title is not None else ""
        if resolved_title:
            return resolved_title
        fallback_title = self._resolve_song_title(song_id)
        return fallback_title if fallback_title else "this song"

    def move_song_up(self, song_id: str) -> None:
        self._move_song(song_id, steps=-1)

    def move_song_down(self, song_id: str) -> None:
        self._move_song(song_id, steps=1)

    def _move_song(self, song_id: str, *, steps: int) -> None:
        runtime = cast(_MoveSongRuntimeShell | None, self._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "move_song", None)):
            self._message_box.warning(
                self._widget,
                "Reorder Setlist",
                "This runtime does not support setlist reordering.",
            )
            return
        try:
            updated = runtime.move_song(song_id, steps=steps)
        except Exception as exc:
            self._message_box.warning(self._widget, "Reorder Setlist", str(exc))
            return
        self._set_presentation(updated if updated is not None else runtime.presentation())

    def move_songs_to_top(self, song_ids: object) -> None:
        self._reorder_selected_songs(song_ids, destination="top")

    def move_songs_to_bottom(self, song_ids: object) -> None:
        self._reorder_selected_songs(song_ids, destination="bottom")

    def reorder_songs(self, song_ids: object) -> None:
        runtime = cast(_ReorderSongsRuntimeShell | None, self._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "reorder_songs", None)):
            self._message_box.warning(
                self._widget,
                "Reorder Setlist",
                "This runtime does not support setlist reordering.",
            )
            return
        ordered_song_ids = self._normalize_song_ids(song_ids)
        if len(ordered_song_ids) < 2:
            return
        current_order = [song.song_id for song in self._get_presentation().available_songs]
        if ordered_song_ids == current_order:
            return
        try:
            updated = runtime.reorder_songs(ordered_song_ids)
        except Exception as exc:
            self._message_box.warning(self._widget, "Reorder Setlist", str(exc))
            return
        self._set_presentation(updated if updated is not None else runtime.presentation())

    def _reorder_selected_songs(self, song_ids: object, *, destination: str) -> None:
        runtime = cast(_ReorderSongsRuntimeShell | None, self._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "reorder_songs", None)):
            self._message_box.warning(
                self._widget,
                "Reorder Setlist",
                "This runtime does not support setlist reordering.",
            )
            return

        selected_song_ids = self._normalize_song_ids(song_ids)
        if len(selected_song_ids) < 2:
            return

        current_order = [song.song_id for song in self._get_presentation().available_songs]
        selected_set = set(selected_song_ids)
        ordered_selection = [song_id for song_id in current_order if song_id in selected_set]
        if len(ordered_selection) < 2:
            return
        remainder = [song_id for song_id in current_order if song_id not in selected_set]
        reordered_song_ids = (
            ordered_selection + remainder
            if destination == "top"
            else remainder + ordered_selection
        )
        if reordered_song_ids == current_order:
            return

        try:
            updated = runtime.reorder_songs(reordered_song_ids)
        except Exception as exc:
            self._message_box.warning(self._widget, "Reorder Setlist", str(exc))
            return
        self._set_presentation(updated if updated is not None else runtime.presentation())

    def delete_songs(self, song_ids: object) -> None:
        runtime = cast(_DeleteSongRuntimeShell | None, self._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "delete_song", None)):
            self._message_box.warning(
                self._widget,
                "Delete Songs",
                "This runtime does not support deleting songs.",
            )
            return

        selected_song_ids = self._normalize_song_ids(song_ids)
        if not selected_song_ids:
            return

        selected_titles = [self._resolve_song_title(song_id) for song_id in selected_song_ids]
        quoted_titles = ", ".join(f'"{title}"' for title in selected_titles)
        reply = self._message_box.question(
            self._widget,
            "Delete Songs",
            (
                f"Delete {len(selected_song_ids)} selected songs and all of their versions?\n\n"
                f"{quoted_titles}\n\n"
                "This cannot be undone."
            ),
            self._message_box.StandardButton.Yes | self._message_box.StandardButton.No,
            self._message_box.StandardButton.No,
        )
        if reply != self._message_box.StandardButton.Yes:
            return

        current_order = [song.song_id for song in self._get_presentation().available_songs]
        selected_set = set(selected_song_ids)
        ordered_selection = [song_id for song_id in current_order if song_id in selected_set]
        deleted_count = 0
        for song_id in ordered_selection:
            try:
                updated = runtime.delete_song(song_id)
            except Exception as exc:
                self._message_box.warning(
                    self._widget,
                    "Delete Songs",
                    f"Failed to delete {self._resolve_song_title(song_id)}: {exc}",
                )
                continue
            deleted_count += 1
            self._set_presentation(updated if updated is not None else runtime.presentation())

        if deleted_count <= 0:
            self._message_box.warning(
                self._widget,
                "Delete Songs",
                "No songs were deleted.",
            )

    def _normalize_song_ids(self, song_ids: object) -> list[str]:
        if not isinstance(song_ids, (list, tuple)):
            return []
        seen: set[str] = set()
        normalized: list[str] = []
        for value in song_ids:
            if not isinstance(value, str):
                continue
            song_id = value.strip()
            if not song_id or song_id in seen:
                continue
            seen.add(song_id)
            normalized.append(song_id)
        return normalized

    def open_pipeline_settings_browser(self) -> None:
        """Open the multi-stage pipeline settings browser for pre-layer editing."""
        self._open_pipeline_settings_browser(preferred_action=None)

    def open_object_action_settings(self, action: InspectorAction) -> None:
        """Open reusable settings in the canonical pipeline browser window."""

        self._open_pipeline_settings_browser(preferred_action=action)

    def _open_pipeline_settings_browser(
        self,
        *,
        preferred_action: InspectorAction | None,
    ) -> None:
        runtime = cast(_ObjectActionSettingsRuntimeShell | None, self._resolve_runtime_shell())
        if runtime is None or not callable(getattr(runtime, "open_object_action_session", None)):
            self._message_box.warning(
                self._widget,
                "Pipeline Settings",
                "This runtime does not support reusable pipeline settings.",
            )
            return
        if not callable(getattr(runtime, "dispatch_object_action_command", None)):
            self._message_box.warning(
                self._widget,
                "Pipeline Settings",
                "This runtime does not support reusable pipeline settings.",
            )
            return

        sessions, initial_action_id, first_error = self._build_pipeline_settings_sessions(
            runtime,
            preferred_action=preferred_action,
        )
        if not sessions:
            self._message_box.warning(
                self._widget,
                "Pipeline Settings",
                first_error or "No reusable pipeline settings are currently available.",
            )
            return

        dialog = PipelineSettingsBrowserDialog(
            tuple(sessions),
            dispatch_command=runtime.dispatch_object_action_command,
            initial_action_id=initial_action_id,
            parent=self._widget,
        )
        dialog.exec()
        self._set_presentation(runtime.presentation())

    def _build_pipeline_settings_sessions(
        self,
        runtime: _ObjectActionSettingsRuntimeShell,
        *,
        preferred_action: InspectorAction | None,
    ) -> tuple[list[ObjectActionSettingsSession], str | None, str | None]:
        preferred_action_id: str | None = None
        preferred_layer_id = None
        preferred_params: dict[str, object] = {}
        if preferred_action is not None:
            preferred_descriptor = descriptor_for_action(preferred_action.action_id)
            preferred_action_id = (
                preferred_descriptor.action_id
                if preferred_descriptor is not None
                else str(preferred_action.action_id).strip()
            )
            preferred_params = dict(preferred_action.params)
            preferred_layer_id = _coerce_layer_id(preferred_params.get("layer_id"))

        ordered_descriptors = list(object_action_descriptors())
        if preferred_action_id:
            ordered_descriptors.sort(
                key=lambda descriptor: descriptor.action_id != preferred_action_id
            )

        sessions: list[ObjectActionSettingsSession] = []
        seen_action_ids: set[str] = set()
        first_error: str | None = None

        for descriptor in ordered_descriptors:
            if descriptor.action_id in seen_action_ids:
                continue
            params: dict[str, object] = {}
            object_id = None
            object_type = descriptor.object_types[0] if descriptor.object_types else None

            if descriptor.action_id == preferred_action_id:
                params = dict(preferred_params)
                explicit_layer_id = _coerce_layer_id(params.get("layer_id"))
                if explicit_layer_id is not None:
                    object_id = explicit_layer_id
                    object_type = "layer"
                elif preferred_layer_id is not None and "layer" in descriptor.object_types:
                    params["layer_id"] = preferred_layer_id
                    object_id = preferred_layer_id
                    object_type = "layer"
            elif preferred_layer_id is not None and "layer" in descriptor.object_types:
                params = {"layer_id": preferred_layer_id}
                object_id = preferred_layer_id
                object_type = "layer"

            try:
                session = runtime.open_object_action_session(
                    descriptor.action_id,
                    params,
                    object_id=object_id,
                    object_type=object_type,
                )
            except Exception as exc:
                if first_error is None:
                    first_error = str(exc)
                continue
            seen_action_ids.add(descriptor.action_id)
            sessions.append(session)

        return sessions, preferred_action_id, first_error

    def _handle_runtime_pipeline_action(self, action_id: str, params: dict[str, object]) -> bool:
        runtime = self._resolve_runtime_shell()
        request_runtime = cast(_RequestObjectActionRuntimeShell | None, runtime)
        run_runtime = cast(_RunObjectActionRuntimeShell | None, runtime)
        request_object_action_run = (
            request_runtime.request_object_action_run
            if request_runtime is not None
            and callable(getattr(request_runtime, "request_object_action_run", None))
            else None
        )
        run_object_action = (
            run_runtime.run_object_action
            if run_runtime is not None and callable(getattr(run_runtime, "run_object_action", None))
            else None
        )
        if not callable(request_object_action_run) and not callable(run_object_action):
            self._message_box.warning(
                self._widget,
                "Pipeline Action",
                f"This runtime does not support '{action_id}'.",
            )
            return True
        layer_id = _coerce_layer_id(params.get("layer_id"))
        if layer_id is None:
            self._message_box.warning(
                self._widget,
                "Pipeline Action",
                f"'{action_id}' requires a target layer.",
            )
            return True
        resolved_params = dict(params)
        descriptor = descriptor_for_action(action_id)
        if descriptor is None:
            self._message_box.warning(
                self._widget,
                "Pipeline Action",
                f"This runtime does not recognize '{action_id}'.",
            )
            return True
        if (
            descriptor.params_schema.get("model_path") == "dialog:file:model"
            and "model_path" not in resolved_params
            and "classify_model_path" not in resolved_params
        ):
            models_dir = self._resolve_models_dir()
            model_path, _ = self._file_dialog.getOpenFileName(
                self._widget,
                "Select Drum Classifier Model",
                str(models_dir),
                "Runtime Models (*.pth *.manifest.json);;PyTorch Models (*.pth);;Artifact Manifests (*.manifest.json);;All Files (*)",
            )
            if not model_path:
                return True
            resolved_params["model_path"] = model_path
        try:
            if callable(request_object_action_run):
                assert request_runtime is not None
                request_runtime.request_object_action_run(
                    action_id,
                    resolved_params,
                    object_id=layer_id,
                    object_type="layer",
                )
                updated = request_runtime.presentation()
            else:
                assert run_runtime is not None and callable(run_object_action)
                updated = run_runtime.run_object_action(
                    action_id,
                    resolved_params,
                    object_id=layer_id,
                    object_type="layer",
                )
        except NotImplementedError as exc:
            self._message_box.warning(self._widget, "Pipeline Action", str(exc))
            return True
        except Exception as exc:
            self._message_box.warning(self._widget, "Pipeline Action", str(exc))
            return True
        self._set_presentation(updated)
        return True


def _natural_sort_key(value: str) -> tuple[object, ...]:
    tokens = _NATURAL_TOKEN_PATTERN.split(value.lower())
    key: list[object] = []
    for token in tokens:
        if not token:
            continue
        if token.isdigit():
            key.append(int(token))
        else:
            key.append(token)
    return tuple(key)

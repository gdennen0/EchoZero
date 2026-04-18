"""Timeline widget action routing and dialog orchestration.
Exists to keep TimelineWidget focused on rendering and input state.
Connects inspector/transfer/runtime actions to existing app intents and runtime methods.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox, QWidget

from echozero.application.presentation.inspector_contract import InspectorAction
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.sync.models import LiveSyncState
from echozero.application.timeline.intents import (
    ApplyTransferPreset,
    ApplyTransferPlan,
    CancelTransferPlan,
    ClearLayerLiveSyncPauseReason,
    ClearSelection,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    DeleteTransferPreset,
    DuplicateSelectedEvents,
    ExitPullFromMA3Workspace,
    ExitPushToMA3Mode,
    NudgeSelectedEvents,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    PreviewTransferPlan,
    SaveTransferPreset,
    Seek,
    SetActivePlaybackTarget,
    SetGain,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    SetPullImportMode,
    SetPushTransferMode,
    SelectAllEvents,
    SelectPullSourceEvents,
    SelectPullSourceTrack,
    SelectPullSourceTracks,
    SelectPullTargetLayer,
    SelectPushTargetTrack,
    TriggerTakeAction,
)
from echozero.models.paths import ensure_installed_models_dir
from echozero.ui.qt.timeline.manual_pull import ManualPullTimelineDialog, ManualPullTimelineSelectionResult


class TimelineWidgetActionRouter:
    """Routes inspector and transfer actions for the timeline widget."""

    def __init__(
        self,
        *,
        widget: QWidget,
        dispatch: Callable[[object], None],
        get_presentation: Callable[[], TimelinePresentation],
        set_presentation: Callable[[TimelinePresentation], None],
        resolve_runtime_shell: Callable[[], object | None],
        selected_event_ids_for_selected_layers: Callable[[], list],
        open_manual_pull_timeline_popup: Callable[[object], ManualPullTimelineSelectionResult | None] | None = None,
        input_dialog=QInputDialog,
        file_dialog=QFileDialog,
        message_box=QMessageBox,
        resolve_models_dir: Callable[[], object] = ensure_installed_models_dir,
    ) -> None:
        self._widget = widget
        self._dispatch = dispatch
        self._get_presentation = get_presentation
        self._set_presentation = set_presentation
        self._resolve_runtime_shell = resolve_runtime_shell
        self._selected_event_ids_for_selected_layers = selected_event_ids_for_selected_layers
        self._open_manual_pull_timeline_popup = open_manual_pull_timeline_popup or self._default_open_manual_pull_timeline_popup
        self._input_dialog = input_dialog
        self._file_dialog = file_dialog
        self._message_box = message_box
        self._resolve_models_dir = resolve_models_dir

    def trigger_contract_action(self, action: InspectorAction) -> None:
        """Execute one inspector contract action against the widget/runtime surface."""
        params = action.params
        action_id = action.action_id
        if action_id == "seek_here":
            time_seconds = params.get("time_seconds")
            if isinstance(time_seconds, (int, float)):
                self._dispatch(Seek(float(time_seconds)))
            return
        if action_id == "nudge_left":
            self._dispatch(NudgeSelectedEvents(direction=-1, steps=int(params.get("steps", 1))))
            return
        if action_id == "nudge_right":
            self._dispatch(NudgeSelectedEvents(direction=1, steps=int(params.get("steps", 1))))
            return
        if action_id == "duplicate":
            self._dispatch(DuplicateSelectedEvents(steps=int(params.get("steps", 1))))
            return
        if action_id == "add_song_from_path":
            self._run_add_song_from_path_action()
            return
        if action_id == "set_active_playback_target":
            layer_id = params.get("layer_id")
            if layer_id is not None:
                self._dispatch(SetActivePlaybackTarget(layer_id=layer_id, take_id=None))
            return
        if action_id in {"gain_down", "gain_unity", "gain_up", "set_gain_custom"}:
            layer_id = params.get("layer_id")
            gain_db = params.get("gain_db")
            if layer_id is not None and isinstance(gain_db, (int, float)):
                self._dispatch(SetGain(layer_id=layer_id, gain_db=float(gain_db)))
            return
        if action_id in {"live_sync_set_off", "live_sync_set_observe", "live_sync_set_armed_write"}:
            self._handle_live_sync_action(action_id, params)
            return
        if action_id == "live_sync_set_pause_reason":
            layer_id = params.get("layer_id")
            pause_reason = params.get("pause_reason")
            if layer_id is not None and isinstance(pause_reason, str) and pause_reason.strip():
                self._dispatch(
                    SetLayerLiveSyncPauseReason(
                        layer_id=layer_id,
                        pause_reason=pause_reason,
                    )
                )
            return
        if action_id == "live_sync_clear_pause_reason":
            layer_id = params.get("layer_id")
            if layer_id is not None:
                self._dispatch(ClearLayerLiveSyncPauseReason(layer_id=layer_id))
            return
        if self.handle_transfer_action(action_id, params):
            return
        if action_id:
            layer_id = params.get("layer_id")
            take_id = params.get("take_id")
            if layer_id is not None and take_id is not None:
                self._dispatch(TriggerTakeAction(layer_id, take_id, action_id))

    def handle_transfer_action(self, action_id: str, params: dict[str, object]) -> bool:
        """Route transfer and runtime-pipeline actions and report whether they were handled."""
        if action_id == "push_to_ma3":
            selected_event_ids = self._selected_event_ids_for_selected_layers()
            self._dispatch(OpenPushToMA3Dialog(selection_event_ids=selected_event_ids))
            return True
        if action_id == "push_select_all_events":
            self._dispatch(SelectAllEvents())
            return True
        if action_id == "push_unselect_all_events":
            self._dispatch(ClearSelection())
            return True
        if action_id == "set_push_transfer_mode":
            return self._handle_set_push_transfer_mode()
        if action_id == "save_transfer_preset":
            return self._handle_save_transfer_preset()
        if action_id in {"apply_transfer_preset", "delete_transfer_preset"}:
            return self._handle_transfer_preset_action(action_id)
        if action_id in {"preview_transfer_plan", "apply_transfer_plan", "cancel_transfer_plan"}:
            return self._handle_transfer_plan_action(action_id, params)
        if action_id in {"extract_stems", "extract_drum_events", "classify_drum_events", "extract_classified_drums"}:
            return self._handle_runtime_pipeline_action(action_id, params)
        if action_id in {
            "select_push_target_track",
            "preview_push_diff",
            "exit_push_mode",
            "pull_from_ma3",
            "select_pull_source_tracks",
            "select_pull_source_events",
            "set_pull_target_layer_mapping",
            "preview_pull_diff",
            "exit_pull_workspace",
        }:
            return self._handle_manual_transfer_workspace_action(action_id, params)
        return False

    def _handle_live_sync_action(self, action_id: str, params: dict[str, object]) -> None:
        layer_id = params.get("layer_id")
        if layer_id is None:
            return
        if action_id == "live_sync_set_armed_write":
            reply = QMessageBox.question(
                self._widget,
                "Arm Live Sync Write",
                "Arm live sync write for this layer? MA3 changes may be written immediately.",
                self._message_box.StandardButton.Yes | self._message_box.StandardButton.No,
                self._message_box.StandardButton.No,
            )
            if reply != self._message_box.StandardButton.Yes:
                return
            state = LiveSyncState.ARMED_WRITE
        elif action_id == "live_sync_set_observe":
            state = LiveSyncState.OBSERVE
        else:
            state = LiveSyncState.OFF
        self._dispatch(SetLayerLiveSyncState(layer_id=layer_id, live_sync_state=state))

    def _run_add_song_from_path_action(self) -> None:
        runtime = self._resolve_runtime_shell()
        if runtime is None or not callable(getattr(runtime, "add_song_from_path", None)):
            self._message_box.warning(
                self._widget,
                "Add Song",
                "This runtime does not support adding songs from a path.",
            )
            return
        title, accepted = self._input_dialog.getText(self._widget, "Add Song", "Song title")
        if not accepted or not title.strip():
            return
        audio_path, _ = self._file_dialog.getOpenFileName(
            self._widget,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.aiff *.aif *.ogg);;All Files (*)",
        )
        if not audio_path:
            return
        try:
            updated = runtime.add_song_from_path(title.strip(), audio_path)
        except Exception as exc:
            self._message_box.warning(self._widget, "Add Song", str(exc))
            return
        self._set_presentation(updated if updated is not None else runtime.presentation())

    def _handle_runtime_pipeline_action(self, action_id: str, params: dict[str, object]) -> bool:
        runtime = self._resolve_runtime_shell()
        layer_id = params.get("layer_id")
        if runtime is None or not callable(getattr(runtime, action_id, None)):
            self._message_box.warning(
                self._widget,
                "Pipeline Action",
                f"This runtime does not support '{action_id}'.",
            )
            return True
        if layer_id is None:
            self._message_box.warning(
                self._widget,
                "Pipeline Action",
                f"'{action_id}' requires a target layer.",
            )
            return True
        call_args: list[object] = [layer_id]
        if action_id == "classify_drum_events":
            models_dir = self._resolve_models_dir()
            model_path, _ = self._file_dialog.getOpenFileName(
                self._widget,
                "Select Drum Classifier Model",
                str(models_dir),
                "Runtime Models (*.pth *.manifest.json);;PyTorch Models (*.pth);;Artifact Manifests (*.manifest.json);;All Files (*)",
            )
            if not model_path:
                return True
            call_args.append(model_path)
        try:
            updated = getattr(runtime, action_id)(*call_args)
        except NotImplementedError as exc:
            self._message_box.warning(self._widget, "Pipeline Action", str(exc))
            return True
        except Exception as exc:
            self._message_box.warning(self._widget, "Pipeline Action", str(exc))
            return True
        self._set_presentation(updated if updated is not None else runtime.presentation())
        return True

    def _handle_set_push_transfer_mode(self) -> bool:
        presentation = self._get_presentation()
        current_mode = (presentation.manual_push_flow.transfer_mode or "merge").strip().lower()
        mode_labels = ["Merge", "Overwrite"]
        default_index = 0 if current_mode == "merge" else 1
        chosen_mode, accepted = self._input_dialog.getItem(
            self._widget,
            "Push Transfer Mode",
            "Transfer mode",
            mode_labels,
            default_index,
            False,
        )
        if not accepted:
            return True
        selected_mode = chosen_mode.strip().lower()
        if selected_mode:
            self._dispatch(SetPushTransferMode(mode=selected_mode))
        return True

    def _handle_save_transfer_preset(self) -> bool:
        preset_name, accepted = self._input_dialog.getText(
            self._widget,
            "Save Transfer Preset",
            "Preset name",
        )
        if not accepted or not preset_name.strip():
            return True
        self._dispatch(SaveTransferPreset(name=preset_name))
        return True

    def _handle_transfer_preset_action(self, action_id: str) -> bool:
        presentation = self._get_presentation()
        if not presentation.transfer_presets:
            return True
        labels = [self._transfer_preset_label(preset) for preset in presentation.transfer_presets]
        title = "Apply Transfer Preset" if action_id == "apply_transfer_preset" else "Delete Transfer Preset"
        chosen_label, accepted = self._input_dialog.getItem(
            self._widget,
            title,
            "Preset",
            labels,
            0,
            False,
        )
        if not accepted:
            return True
        selected_preset = next(
            (preset for preset, label in zip(presentation.transfer_presets, labels) if label == chosen_label),
            None,
        )
        if selected_preset is None:
            return True
        if action_id == "apply_transfer_preset":
            self._dispatch(ApplyTransferPreset(preset_id=selected_preset.preset_id))
        else:
            self._dispatch(DeleteTransferPreset(preset_id=selected_preset.preset_id))
        return True

    def _handle_transfer_plan_action(self, action_id: str, params: dict[str, object]) -> bool:
        plan_id = params.get("plan_id")
        if action_id == "cancel_transfer_plan":
            if isinstance(plan_id, str):
                self._dispatch(CancelTransferPlan(plan_id=plan_id))
            return True
        if not isinstance(plan_id, str):
            return True
        if not self._resolve_blocked_push_rows_for_plan_action(plan_id):
            return True
        if action_id == "preview_transfer_plan":
            self._dispatch(PreviewTransferPlan(plan_id=plan_id))
            plan = self._get_presentation().batch_transfer_plan
            if plan is not None and plan.plan_id == plan_id:
                self._message_box.information(
                    self._widget,
                    "Transfer Plan Preview",
                    self._transfer_plan_preview_summary(
                        operation_type=plan.operation_type,
                        total_rows=len(plan.rows),
                        ready_count=plan.ready_count,
                        blocked_count=plan.blocked_count,
                        applied_count=plan.applied_count,
                        failed_count=plan.failed_count,
                    ),
                )
            return True
        if action_id == "apply_transfer_plan":
            self._dispatch(ApplyTransferPlan(plan_id=plan_id))
            plan = self._get_presentation().batch_transfer_plan
            if plan is not None and plan.plan_id == plan_id:
                self._message_box.information(
                    self._widget,
                    "Transfer Plan Results",
                    self._transfer_plan_apply_summary(
                        operation_type=plan.operation_type,
                        total_rows=len(plan.rows),
                        applied_count=plan.applied_count,
                        failed_count=plan.failed_count,
                        blocked_count=plan.blocked_count,
                    ),
                )
            return True
        return False

    def _handle_manual_transfer_workspace_action(self, action_id: str, params: dict[str, object]) -> bool:
        presentation = self._get_presentation()
        if action_id == "select_push_target_track":
            flow = presentation.manual_push_flow
            layer_id = params.get("layer_id")
            if layer_id is None or not flow.available_tracks:
                return True
            labels = [self._manual_push_track_label(track) for track in flow.available_tracks]
            chosen_label, accepted = self._input_dialog.getItem(
                self._widget,
                "Select Push Target Track",
                "Target track",
                labels,
                0,
                False,
            )
            if not accepted:
                return True
            selected_track = next(
                (track for track, label in zip(flow.available_tracks, labels) if label == chosen_label),
                None,
            )
            if selected_track is None:
                return True
            self._dispatch(SelectPushTargetTrack(target_track_coord=selected_track.coord, layer_id=layer_id))
            return True
        if action_id == "preview_push_diff":
            return self._preview_push_diff(params)
        if action_id == "exit_push_mode":
            self._dispatch(ExitPushToMA3Mode())
            return True
        if action_id == "pull_from_ma3":
            self._dispatch(OpenPullFromMA3Dialog())
            return True
        if action_id == "select_pull_source_tracks":
            return self._select_pull_source_tracks()
        if action_id == "select_pull_source_events":
            return self._select_pull_source_events()
        if action_id == "set_pull_target_layer_mapping":
            return self._set_pull_target_layer_mapping()
        if action_id == "preview_pull_diff":
            return self._preview_pull_diff(params)
        if action_id == "exit_pull_workspace":
            self._dispatch(ExitPullFromMA3Workspace())
            return True
        return False

    def _preview_push_diff(self, params: dict[str, object]) -> bool:
        presentation = self._get_presentation()
        layer_id = params.get("layer_id")
        if layer_id is None:
            return True
        row = next(
            (
                candidate
                for candidate in (presentation.batch_transfer_plan.rows if presentation.batch_transfer_plan else [])
                if candidate.direction == "push" and candidate.source_layer_id == layer_id
            ),
            None,
        )
        if row is None or not row.target_track_coord or not row.selected_event_ids:
            return True
        self._dispatch(
            ConfirmPushToMA3(
                target_track_coord=row.target_track_coord,
                selected_event_ids=list(row.selected_event_ids),
            )
        )
        flow = self._get_presentation().manual_push_flow
        preview = flow.diff_preview
        if flow.diff_gate_open and preview is not None:
            self._message_box.information(
                self._widget,
                "Push Diff Preview",
                self._manual_push_diff_preview_summary(
                    preview.selected_count,
                    preview.target_track_name,
                    preview.target_track_coord,
                ),
            )
        return True

    def _select_pull_source_tracks(self) -> bool:
        flow = self._get_presentation().manual_pull_flow
        if not flow.available_tracks:
            return True
        track_labels = [self._manual_pull_track_label(track) for track in flow.available_tracks]
        chosen_track_label, accepted = self._input_dialog.getItem(
            self._widget,
            "Import from MA3",
            "Source track",
            track_labels,
            0,
            False,
        )
        if not accepted:
            return True
        selected_track = next(
            (track for track, label in zip(flow.available_tracks, track_labels) if label == chosen_track_label),
            None,
        )
        if selected_track is None:
            return True
        next_selected = list(flow.selected_source_track_coords)
        if selected_track.coord not in next_selected:
            next_selected.append(selected_track.coord)
        self._dispatch(SelectPullSourceTracks(source_track_coords=next_selected))
        self._dispatch(SelectPullSourceTrack(source_track_coord=selected_track.coord))
        return True

    def _select_pull_source_events(self) -> bool:
        flow = self._get_presentation().manual_pull_flow
        if not flow.active_source_track_coord or not flow.available_events or not flow.available_target_layers:
            return True
        selection = self._open_manual_pull_timeline_popup(flow)
        if selection is None:
            return True
        self._dispatch(SelectPullSourceEvents(selected_ma3_event_ids=selection.selected_event_ids))
        self._dispatch(SelectPullTargetLayer(target_layer_id=selection.target_layer_id))
        self._dispatch(SetPullImportMode(import_mode=selection.import_mode))
        return True

    def _set_pull_target_layer_mapping(self) -> bool:
        flow = self._get_presentation().manual_pull_flow
        if not flow.available_target_layers:
            return True
        target_labels = [self._manual_pull_target_label(target) for target in flow.available_target_layers]
        chosen_target_label, accepted = self._input_dialog.getItem(
            self._widget,
            "Import from MA3",
            "Destination EZ layer",
            target_labels,
            0,
            False,
        )
        if not accepted:
            return True
        selected_target = next(
            (target for target, label in zip(flow.available_target_layers, target_labels) if label == chosen_target_label),
            None,
        )
        if selected_target is None:
            return True
        self._dispatch(SelectPullTargetLayer(target_layer_id=selected_target.layer_id))
        return True

    def _preview_pull_diff(self, params: dict[str, object]) -> bool:
        presentation = self._get_presentation()
        layer_id = params.get("layer_id")
        if layer_id is None:
            return True
        row = next(
            (
                candidate
                for candidate in (presentation.batch_transfer_plan.rows if presentation.batch_transfer_plan else [])
                if candidate.direction == "pull" and candidate.target_layer_id == layer_id
            ),
            None,
        )
        if row is None or not row.source_track_coord or not row.target_layer_id or not row.selected_ma3_event_ids:
            return True
        self._dispatch(SelectPullSourceTrack(source_track_coord=row.source_track_coord))
        self._dispatch(
            ConfirmPullFromMA3(
                source_track_coord=row.source_track_coord,
                selected_ma3_event_ids=list(row.selected_ma3_event_ids),
                target_layer_id=row.target_layer_id,
                import_mode=row.import_mode,
            )
        )
        flow = self._get_presentation().manual_pull_flow
        preview = flow.diff_preview
        if flow.diff_gate_open and preview is not None:
            self._message_box.information(
                self._widget,
                "Pull Diff Preview",
                self._manual_pull_diff_preview_summary(
                    preview.selected_count,
                    preview.source_track_name,
                    preview.source_track_coord,
                    preview.target_layer_name,
                ),
            )
        return True

    def _resolve_blocked_push_rows_for_plan_action(self, plan_id: str) -> bool:
        presentation = self._get_presentation()
        plan = presentation.batch_transfer_plan
        if plan is None or plan.plan_id != plan_id or plan.operation_type not in {"push", "mixed"}:
            return True
        blocked_rows = [
            row
            for row in plan.rows
            if row.direction == "push" and not row.target_track_coord
        ]
        if not blocked_rows:
            return True
        flow = presentation.manual_push_flow
        if not flow.available_tracks:
            return False
        labels = [self._manual_push_track_label(track) for track in flow.available_tracks]
        for row in blocked_rows:
            chosen_label, accepted = self._input_dialog.getItem(
                self._widget,
                "Map Push Layer",
                f"Target MA3 track for {row.source_label}",
                labels,
                0,
                False,
            )
            if not accepted:
                return False
            selected_track = next(
                (track for track, label in zip(flow.available_tracks, labels) if label == chosen_label),
                None,
            )
            if selected_track is None:
                return False
            self._dispatch(
                SelectPushTargetTrack(
                    target_track_coord=selected_track.coord,
                    layer_id=row.source_layer_id,
                )
            )
            flow = self._get_presentation().manual_push_flow
            labels = [self._manual_push_track_label(track) for track in flow.available_tracks]
        return True

    def _default_open_manual_pull_timeline_popup(self, flow) -> ManualPullTimelineSelectionResult | None:
        source_track = next(
            (track for track in flow.available_tracks if track.coord == flow.active_source_track_coord),
            None,
        )
        source_track_label = (
            self._manual_pull_track_label(source_track)
            if source_track is not None
            else str(flow.active_source_track_coord)
        )
        selected_event_ids = list(
            flow.selected_ma3_event_ids_by_track.get(flow.active_source_track_coord, flow.selected_ma3_event_ids)
        )
        selected_target_layer_id = flow.target_layer_id_by_source_track.get(
            flow.active_source_track_coord,
            flow.target_layer_id,
        )
        dialog = ManualPullTimelineDialog(
            source_track_label=source_track_label,
            events=flow.available_events,
            selected_event_ids=selected_event_ids,
            available_targets=flow.available_target_layers,
            selected_target_layer_id=selected_target_layer_id,
            selected_import_mode=flow.import_mode_by_source_track.get(
                flow.active_source_track_coord,
                flow.import_mode,
            ),
            parent=self._widget,
        )
        if dialog.exec() != ManualPullTimelineDialog.DialogCode.Accepted:
            return None
        return ManualPullTimelineSelectionResult(
            selected_event_ids=dialog.selected_event_ids(),
            target_layer_id=dialog.selected_target_layer_id(),
            import_mode=dialog.selected_import_mode(),
        )

    @staticmethod
    def _manual_push_track_label(track) -> str:
        parts = [track.name, f"({track.coord})"]
        if track.note:
            parts.append(f"- {track.note}")
        if track.event_count is not None:
            parts.append(f"[{track.event_count} existing]")
        return " ".join(parts)

    @staticmethod
    def _manual_push_diff_preview_summary(selected_count: int, target_track_name: str, target_track_coord: str) -> str:
        noun = "event" if selected_count == 1 else "events"
        return (
            f"Prepared diff preview for {selected_count} selected {noun}.\n\n"
            f"Target track: {target_track_name} ({target_track_coord})\n"
            f"No MA3 transfer has been started in this step."
        )

    @staticmethod
    def _manual_pull_track_label(track) -> str:
        parts = [track.name, f"({track.coord})"]
        if track.note:
            parts.append(f"- {track.note}")
        if track.event_count is not None:
            parts.append(f"[{track.event_count} events]")
        return " ".join(parts)

    @staticmethod
    def _manual_pull_target_label(target) -> str:
        return target.name

    @staticmethod
    def _manual_pull_diff_preview_summary(
        selected_count: int,
        source_track_name: str,
        source_track_coord: str,
        target_layer_name: str,
    ) -> str:
        noun = "event" if selected_count == 1 else "events"
        return (
            f"Prepared diff preview for {selected_count} selected {noun}.\n\n"
            f"Source track: {source_track_name} ({source_track_coord})\n"
            f"Target layer: {target_layer_name}\n"
            f"No MA3 import has been started in this step."
        )

    @staticmethod
    def _transfer_plan_preview_summary(
        *,
        operation_type: str,
        total_rows: int,
        ready_count: int,
        blocked_count: int,
        applied_count: int,
        failed_count: int,
    ) -> str:
        return (
            f"{_transfer_plan_operation_label(operation_type)} plan preview complete.\n\n"
            f"Rows: {total_rows}\n"
            f"Ready: {ready_count}\n"
            f"Blocked: {blocked_count}\n"
            f"Applied: {applied_count}\n"
            f"Failed: {failed_count}"
        )

    @staticmethod
    def _transfer_plan_apply_summary(
        *,
        operation_type: str,
        total_rows: int,
        applied_count: int,
        failed_count: int,
        blocked_count: int,
    ) -> str:
        return (
            f"{_transfer_plan_operation_label(operation_type)} plan apply complete.\n\n"
            f"Rows: {total_rows}\n"
            f"Applied: {applied_count}\n"
            f"Failed: {failed_count}\n"
            f"Blocked: {blocked_count}"
        )

    @staticmethod
    def _transfer_preset_label(preset) -> str:
        return f"{preset.name} ({preset.preset_id})"


def _transfer_plan_operation_label(operation_type: str) -> str:
    return (operation_type or "mixed").strip().capitalize()

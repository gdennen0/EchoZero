"""Manual transfer workspace helpers for the timeline widget.
Exists to isolate push/pull workspace routing, blocked-row mapping, and diff dialogs from transfer action entry routing.
Connects transfer workspace presentation state to canonical timeline intents.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from PyQt6.QtWidgets import QInputDialog, QMessageBox, QWidget

from echozero.application.presentation.models import (
    ManualPullFlowPresentation,
    ManualPullTargetOptionPresentation,
    ManualPullTrackOptionPresentation,
    ManualPushTrackOptionPresentation,
    TimelinePresentation,
    TransferPresetPresentation,
)
from echozero.application.shared.ids import EventId
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    ExitPullFromMA3Workspace,
    ExitPushToMA3Mode,
    OpenPullFromMA3Dialog,
    SelectLayer,
    SelectPullSourceEvents,
    SelectPullSourceTimecode,
    SelectPullSourceTrack,
    SelectPullSourceTrackGroup,
    SelectPullSourceTracks,
    SelectPullTargetLayer,
    SelectPushTargetTrack,
)
from echozero.ui.qt.timeline.manual_pull import (
    ManualPullWorkspaceDialog,
)
from echozero.ui.qt.timeline.widget_action_contract_mixin import _coerce_layer_id


class _TransferActionHost(Protocol):
    _widget: QWidget
    _dispatch: Callable[[object], None]
    _get_presentation: Callable[[], TimelinePresentation]
    _selected_event_ids_for_selected_layers: Callable[[], list[EventId]]
    _input_dialog: type[QInputDialog]
    _message_box: type[QMessageBox]

    def _handle_runtime_pipeline_action(
        self, action_id: str, params: dict[str, object]
    ) -> bool: ...


class TimelineWidgetTransferWorkspaceMixin:
    def _handle_manual_transfer_workspace_action(
        self, action_id: str, params: dict[str, object]
    ) -> bool:
        host = cast(_TransferActionHost, self)
        presentation = host._get_presentation()
        if action_id == "select_push_target_track":
            flow = presentation.manual_push_flow
            layer_id = _coerce_layer_id(params.get("layer_id"))
            if layer_id is None or not flow.available_tracks:
                return True
            labels = [self._manual_push_track_label(track) for track in flow.available_tracks]
            chosen_label, accepted = host._input_dialog.getItem(
                host._widget,
                "Select Push Target Track",
                "Target track",
                labels,
                0,
                False,
            )
            if not accepted:
                return True
            selected_track = next(
                (
                    track
                    for track, label in zip(flow.available_tracks, labels)
                    if label == chosen_label
                ),
                None,
            )
            if selected_track is None:
                return True
            host._dispatch(
                SelectPushTargetTrack(target_track_coord=selected_track.coord, layer_id=layer_id)
            )
            return True
        if action_id == "preview_push_diff":
            return self._preview_push_diff(params)
        if action_id == "exit_push_mode":
            host._dispatch(ExitPushToMA3Mode())
            return True
        if action_id == "transfer.workspace_open":
            direction = str(params.get("direction", "")).strip().lower()
            if direction != "pull":
                return False
            layer_id = _coerce_layer_id(params.get("layer_id"))
            if layer_id is not None:
                self._focus_layer_for_transfer_workspace(layer_id)
            return self._run_manual_pull_workspace()
        if action_id == "select_pull_source_tracks":
            return self._run_manual_pull_workspace()
        if action_id == "select_pull_source_events":
            return self._run_manual_pull_workspace()
        if action_id == "set_pull_target_layer_mapping":
            return self._run_manual_pull_workspace()
        if action_id == "preview_pull_diff":
            return self._preview_pull_diff(params)
        if action_id == "exit_pull_workspace":
            host._dispatch(ExitPullFromMA3Workspace())
            return True
        return False

    def _focus_layer_for_transfer_workspace(self, layer_id: object) -> None:
        host = cast(_TransferActionHost, self)
        resolved_layer_id = _coerce_layer_id(layer_id)
        if resolved_layer_id is None:
            return
        presentation = host._get_presentation()
        if presentation.selected_layer_id != resolved_layer_id:
            host._dispatch(SelectLayer(resolved_layer_id))

    def _preview_push_diff(self, params: dict[str, object]) -> bool:
        host = cast(_TransferActionHost, self)
        presentation = host._get_presentation()
        layer_id = _coerce_layer_id(params.get("layer_id"))
        if layer_id is None:
            return True
        row = next(
            (
                candidate
                for candidate in (
                    presentation.batch_transfer_plan.rows
                    if presentation.batch_transfer_plan
                    else []
                )
                if candidate.direction == "push" and candidate.source_layer_id == layer_id
            ),
            None,
        )
        if row is None or not row.target_track_coord or not row.selected_event_ids:
            return True
        host._dispatch(
            ConfirmPushToMA3(
                target_track_coord=row.target_track_coord,
                selected_event_ids=list(row.selected_event_ids),
            )
        )
        flow = host._get_presentation().manual_push_flow
        preview = flow.diff_preview
        if flow.diff_gate_open and preview is not None:
            host._message_box.information(
                host._widget,
                "Push Diff Preview",
                self._manual_push_diff_preview_summary(
                    preview.selected_count,
                    preview.target_track_name,
                    preview.target_track_coord,
                ),
            )
        return True

    def _preview_pull_diff(self, params: dict[str, object]) -> bool:
        host = cast(_TransferActionHost, self)
        presentation = host._get_presentation()
        layer_id = params.get("layer_id")
        if layer_id is None:
            return True
        row = next(
            (
                candidate
                for candidate in (
                    presentation.batch_transfer_plan.rows
                    if presentation.batch_transfer_plan
                    else []
                )
                if candidate.direction == "pull" and candidate.target_layer_id == layer_id
            ),
            None,
        )
        if (
            row is None
            or not row.source_track_coord
            or not row.target_layer_id
            or not row.selected_ma3_event_ids
        ):
            return True
        host._dispatch(SelectPullSourceTrack(source_track_coord=row.source_track_coord))
        host._dispatch(
            ConfirmPullFromMA3(
                source_track_coord=row.source_track_coord,
                selected_ma3_event_ids=list(row.selected_ma3_event_ids),
                target_layer_id=row.target_layer_id,
                import_mode=row.import_mode,
            )
        )
        flow = host._get_presentation().manual_pull_flow
        preview = flow.diff_preview
        if flow.diff_gate_open and preview is not None:
            host._message_box.information(
                host._widget,
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
        host = cast(_TransferActionHost, self)
        presentation = host._get_presentation()
        plan = presentation.batch_transfer_plan
        if plan is None or plan.plan_id != plan_id or plan.operation_type not in {"push", "mixed"}:
            return True
        blocked_rows = [
            row for row in plan.rows if row.direction == "push" and not row.target_track_coord
        ]
        if not blocked_rows:
            return True
        flow = presentation.manual_push_flow
        if not flow.available_tracks:
            return False
        labels = [self._manual_push_track_label(track) for track in flow.available_tracks]
        for row in blocked_rows:
            chosen_label, accepted = host._input_dialog.getItem(
                host._widget,
                "Map Push Layer",
                f"Target MA3 track for {row.source_label}",
                labels,
                0,
                False,
            )
            if not accepted:
                return False
            selected_track = next(
                (
                    track
                    for track, label in zip(flow.available_tracks, labels)
                    if label == chosen_label
                ),
                None,
            )
            if selected_track is None:
                return False
            host._dispatch(
                SelectPushTargetTrack(
                    target_track_coord=selected_track.coord,
                    layer_id=row.source_layer_id,
                )
            )
            flow = host._get_presentation().manual_push_flow
            labels = [self._manual_push_track_label(track) for track in flow.available_tracks]
        return True

    def _run_manual_pull_workspace(self) -> bool:
        host = cast(_TransferActionHost, self)
        flow = host._get_presentation().manual_pull_flow
        accepted = self._open_manual_pull_workspace_dialog(flow, exit_on_cancel=True)
        if not accepted:
            return True
        host._dispatch(ApplyPullFromMA3())
        return True

    def _open_manual_pull_workspace_dialog(
        self,
        flow: ManualPullFlowPresentation,
        *,
        exit_on_cancel: bool,
    ) -> bool:
        host = cast(_TransferActionHost, self)
        if not flow.workspace_active:
            host._dispatch(OpenPullFromMA3Dialog())
            flow = host._get_presentation().manual_pull_flow

        dialog = ManualPullWorkspaceDialog(parent=host._widget)

        def _refresh_dialog() -> ManualPullFlowPresentation:
            refreshed_flow = host._get_presentation().manual_pull_flow
            dialog.set_flow(refreshed_flow)
            return refreshed_flow

        def _select_source_track(source_track_coord: str) -> None:
            host._dispatch(SelectPullSourceTracks(source_track_coords=[source_track_coord]))
            host._dispatch(SelectPullSourceTrack(source_track_coord=source_track_coord))
            _refresh_dialog()

        def _select_source_timecode(timecode_no: int) -> None:
            host._dispatch(SelectPullSourceTimecode(timecode_no=int(timecode_no)))
            _refresh_dialog()

        def _select_source_track_group(track_group_no: int) -> None:
            host._dispatch(SelectPullSourceTrackGroup(track_group_no=int(track_group_no)))
            _refresh_dialog()

        def _select_target_layer(target_layer_id: object) -> None:
            resolved_target_layer_id = _coerce_layer_id(target_layer_id)
            if resolved_target_layer_id is None:
                return
            host._dispatch(SelectPullTargetLayer(target_layer_id=resolved_target_layer_id))
            _refresh_dialog()

        def _select_source_events(selected_event_ids: list[str]) -> None:
            if not selected_event_ids:
                return
            host._dispatch(SelectPullSourceEvents(selected_ma3_event_ids=list(selected_event_ids)))

        dialog.timecode_selected.connect(_select_source_timecode)
        dialog.track_group_selected.connect(_select_source_track_group)
        dialog.track_selected.connect(_select_source_track)
        dialog.target_layer_selected.connect(_select_target_layer)
        dialog.event_selection_changed.connect(_select_source_events)
        _refresh_dialog()
        if dialog.exec() != ManualPullWorkspaceDialog.DialogCode.Accepted:
            if exit_on_cancel:
                host._dispatch(ExitPullFromMA3Workspace())
            return False
        return True

    @staticmethod
    def _manual_push_track_label(track: ManualPushTrackOptionPresentation) -> str:
        name = track.name
        if track.timecode_name:
            timecode_label = TimelineWidgetTransferWorkspaceMixin._manual_push_timecode_label(
                track
            )
            name = f"{timecode_label} · {track.name}"
        parts = [name, f"({track.coord})"]
        if track.number is not None:
            parts.insert(0, f"TR{track.number}")
        if track.note:
            parts.append(f"- {track.note}")
        if track.event_count is not None:
            parts.append(f"[{track.event_count} existing]")
        return " ".join(parts)

    @staticmethod
    def _manual_push_timecode_label(track: ManualPushTrackOptionPresentation) -> str:
        coord = str(track.coord or "").strip().lower()
        tc_no = ""
        if coord.startswith("tc"):
            tc_no = coord[2:].split("_", 1)[0]
        if tc_no and track.timecode_name:
            return f"TC{tc_no} {track.timecode_name}"
        if track.timecode_name:
            return track.timecode_name
        return f"TC{tc_no}" if tc_no else "MA3"

    @staticmethod
    def _manual_push_diff_preview_summary(
        selected_count: int, target_track_name: str, target_track_coord: str
    ) -> str:
        noun = "event" if selected_count == 1 else "events"
        return (
            f"Prepared diff preview for {selected_count} selected {noun}.\n\n"
            f"Target track: {target_track_name} ({target_track_coord})\n"
            f"No MA3 transfer has been started in this step."
        )

    @staticmethod
    def _manual_pull_track_label(track: ManualPullTrackOptionPresentation) -> str:
        parts = [track.name, f"({track.coord})"]
        if track.number is not None:
            parts.insert(0, f"TR{track.number}")
        if track.note:
            parts.append(f"- {track.note}")
        if track.event_count is not None:
            parts.append(f"[{track.event_count} events]")
        return " ".join(parts)

    @staticmethod
    def _manual_pull_target_label(target: ManualPullTargetOptionPresentation) -> str:
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
    def _transfer_preset_label(preset: TransferPresetPresentation) -> str:
        return f"{preset.name} ({preset.preset_id})"


def _transfer_plan_operation_label(operation_type: str) -> str:
    return (operation_type or "mixed").strip().capitalize()


__all__ = ["TimelineWidgetTransferWorkspaceMixin", "_TransferActionHost"]

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
    SelectPullSourceEvents,
    SelectPullSourceTrack,
    SelectPullSourceTracks,
    SelectPullTargetLayer,
    SelectPushTargetTrack,
)
from echozero.ui.qt.timeline.manual_pull import (
    ManualPullTimelineDialog,
    ManualPullTimelineSelectionResult,
)
from echozero.ui.qt.timeline.widget_action_contract_mixin import _coerce_layer_id


class _TransferActionHost(Protocol):
    _widget: QWidget
    _dispatch: Callable[[object], None]
    _get_presentation: Callable[[], TimelinePresentation]
    _selected_event_ids_for_selected_layers: Callable[[], list[EventId]]
    _focus_layer_for_header_action: Callable[[object], None]
    _input_dialog: type[QInputDialog]
    _message_box: type[QMessageBox]
    _open_manual_pull_timeline_popup: Callable[
        [ManualPullFlowPresentation], ManualPullTimelineSelectionResult | None
    ]

    def _handle_runtime_pipeline_action(self, action_id: str, params: dict[str, object]) -> bool: ...


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
        if action_id == "pull_from_ma3":
            layer_id = _coerce_layer_id(params.get("layer_id"))
            if layer_id is not None:
                host._focus_layer_for_header_action(layer_id)
            host._dispatch(OpenPullFromMA3Dialog())
            flow = host._get_presentation().manual_pull_flow
            if flow.workspace_active:
                self._open_manual_pull_workspace_popup(flow)
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
            host._dispatch(ExitPullFromMA3Workspace())
            return True
        return False

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

    def _select_pull_source_tracks(self) -> bool:
        host = cast(_TransferActionHost, self)
        flow = host._get_presentation().manual_pull_flow
        if not flow.available_tracks:
            return True
        track_labels = [self._manual_pull_track_label(track) for track in flow.available_tracks]
        chosen_track_label, accepted = host._input_dialog.getItem(
            host._widget,
            "Import from MA3",
            "Source track",
            track_labels,
            0,
            False,
        )
        if not accepted:
            return True
        selected_track = next(
            (
                track
                for track, label in zip(flow.available_tracks, track_labels)
                if label == chosen_track_label
            ),
            None,
        )
        if selected_track is None:
            return True
        next_selected = list(flow.selected_source_track_coords)
        if selected_track.coord not in next_selected:
            next_selected.append(selected_track.coord)
        host._dispatch(SelectPullSourceTracks(source_track_coords=next_selected))
        host._dispatch(SelectPullSourceTrack(source_track_coord=selected_track.coord))
        return True

    def _select_pull_source_events(self) -> bool:
        host = cast(_TransferActionHost, self)
        flow = host._get_presentation().manual_pull_flow
        if (
            not flow.active_source_track_coord
            or not flow.available_events
            or not flow.available_target_layers
        ):
            return True
        selection = host._open_manual_pull_timeline_popup(flow)
        if selection is None:
            return True
        target_layer_id = _coerce_layer_id(selection.target_layer_id)
        if target_layer_id is None:
            return True
        host._dispatch(SelectPullSourceEvents(selected_ma3_event_ids=selection.selected_event_ids))
        host._dispatch(SelectPullTargetLayer(target_layer_id=target_layer_id))
        return True

    def _set_pull_target_layer_mapping(self) -> bool:
        host = cast(_TransferActionHost, self)
        flow = host._get_presentation().manual_pull_flow
        if not flow.available_target_layers:
            return True
        target_labels = [
            self._manual_pull_target_label(target) for target in flow.available_target_layers
        ]
        chosen_target_label, accepted = host._input_dialog.getItem(
            host._widget,
            "Import from MA3",
            "Destination EZ layer",
            target_labels,
            0,
            False,
        )
        if not accepted:
            return True
        selected_target = next(
            (
                target
                for target, label in zip(flow.available_target_layers, target_labels)
                if label == chosen_target_label
            ),
            None,
        )
        if selected_target is None:
            return True
        host._dispatch(SelectPullTargetLayer(target_layer_id=selected_target.layer_id))
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

    def _default_open_manual_pull_timeline_popup(
        self, flow: ManualPullFlowPresentation
    ) -> ManualPullTimelineSelectionResult | None:
        host = cast(_TransferActionHost, self)
        active_source_track_coord = flow.active_source_track_coord
        source_track = next(
            (
                track
                for track in flow.available_tracks
                if track.coord == active_source_track_coord
            ),
            None,
        )
        source_track_label = (
            self._manual_pull_track_label(source_track)
            if source_track is not None
            else str(active_source_track_coord)
        )
        selected_event_ids = (
            list(flow.selected_ma3_event_ids_by_track.get(active_source_track_coord, []))
            if active_source_track_coord is not None
            else list(flow.selected_ma3_event_ids)
        )
        selected_target_layer_id = (
            flow.target_layer_id_by_source_track.get(active_source_track_coord)
            if active_source_track_coord is not None
            else flow.target_layer_id
        )
        dialog = ManualPullTimelineDialog(
            source_track_label=source_track_label,
            events=flow.available_events,
            selected_event_ids=selected_event_ids,
            available_targets=flow.available_target_layers,
            selected_target_layer_id=selected_target_layer_id,
            selected_import_mode=(
                flow.import_mode_by_source_track.get(active_source_track_coord, flow.import_mode)
                if active_source_track_coord is not None
                else flow.import_mode
            ),
            parent=host._widget,
        )
        if dialog.exec() != ManualPullTimelineDialog.DialogCode.Accepted:
            return None
        return ManualPullTimelineSelectionResult(
            selected_event_ids=dialog.selected_event_ids(),
            target_layer_id=dialog.selected_target_layer_id(),
            import_mode=dialog.selected_import_mode(),
        )

    def _open_manual_pull_workspace_popup(self, flow: ManualPullFlowPresentation) -> bool:
        host = cast(_TransferActionHost, self)
        if not flow.available_tracks:
            host._message_box.warning(
                host._widget,
                "Import Event Layer from MA3",
                "No MA3 source tracks are available to import.",
            )
            return True

        track_labels = [self._manual_pull_track_label(track) for track in flow.available_tracks]
        default_index = 0
        if flow.active_source_track_coord is not None:
            for index, track in enumerate(flow.available_tracks):
                if track.coord == flow.active_source_track_coord:
                    default_index = index
                    break
        chosen_track_label, accepted = host._input_dialog.getItem(
            host._widget,
            "Import Event Layer from MA3",
            "Source MA3 track (all events will be imported)",
            track_labels,
            default_index,
            False,
        )
        if not accepted:
            return True

        selected_track = next(
            (
                track
                for track, label in zip(flow.available_tracks, track_labels)
                if label == chosen_track_label
            ),
            None,
        )
        if selected_track is None:
            return True

        host._dispatch(SelectPullSourceTracks(source_track_coords=[selected_track.coord]))
        host._dispatch(SelectPullSourceTrack(source_track_coord=selected_track.coord))
        refreshed_flow = host._get_presentation().manual_pull_flow
        selected_event_ids = [event.event_id for event in refreshed_flow.available_events]
        if not selected_event_ids:
            host._message_box.warning(
                host._widget,
                "Import Event Layer from MA3",
                "The selected MA3 track has no events to import.",
            )
            return True

        host._dispatch(SelectPullSourceEvents(selected_ma3_event_ids=selected_event_ids))
        host._dispatch(ApplyPullFromMA3())
        return True

    @staticmethod
    def _manual_push_track_label(track: ManualPushTrackOptionPresentation) -> str:
        name = track.name
        if track.timecode_name:
            timecode_label = TimelineWidgetTransferWorkspaceMixin._manual_push_timecode_label(track)
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

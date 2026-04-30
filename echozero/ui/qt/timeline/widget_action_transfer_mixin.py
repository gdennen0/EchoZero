"""Transfer workspace helpers for the timeline widget.
Exists to isolate push/pull workspace routing, transfer presets, and transfer plan dialogs from general widget action handling.
Connects inspector transfer actions to canonical timeline intents and transfer preview dialogs.
"""

from __future__ import annotations

from typing import cast

from echozero.application.timeline.intents import (
    ApplyTransferPlan,
    ApplyTransferPreset,
    CancelTransferPlan,
    ClearSelection,
    DeleteTransferPreset,
    OpenPushToMA3Dialog,
    PreviewTransferPlan,
    SaveTransferPreset,
    SelectAllEvents,
    SetPushTransferMode,
)
from echozero.application.timeline.object_actions import is_object_action, resolve_action_id
from echozero.ui.qt.timeline.widget_action_ma3_push_mixin import (
    TimelineWidgetMA3PushActionMixin,
)
from echozero.ui.qt.timeline.widget_action_transfer_workspace_mixin import (
    TimelineWidgetTransferWorkspaceMixin,
    _TransferActionHost,
)


class TimelineWidgetTransferActionMixin(
    TimelineWidgetMA3PushActionMixin,
    TimelineWidgetTransferWorkspaceMixin,
):
    def handle_transfer_action(self, action_id: str, params: dict[str, object]) -> bool:
        """Route transfer and runtime-pipeline actions and report whether they were handled."""
        host = cast(_TransferActionHost, self)
        resolved_action_id = resolve_action_id(action_id, warn_on_alias=True) or action_id
        if resolved_action_id == "transfer.workspace_open":
            direction = str(params.get("direction", "")).strip().lower()
            if direction == "push":
                return self._handle_send_layer_to_ma3(params)
            if direction == "pull":
                return self._handle_manual_transfer_workspace_action(
                    "transfer.workspace_open",
                    params,
                )
            return False
        if resolved_action_id == "transfer.route_layer_track":
            return self._handle_route_layer_to_ma3_track(params)
        if resolved_action_id == "transfer.send_selection":
            return self._handle_send_selected_events_to_ma3(params)
        if resolved_action_id == "transfer.send_to_track_once":
            return self._handle_send_to_different_track_once(params)
        if resolved_action_id == "push_legacy_mode":
            selected_event_ids = host._selected_event_ids_for_selected_layers()
            host._dispatch(OpenPushToMA3Dialog(selection_event_ids=selected_event_ids))
            return True
        if resolved_action_id == "push_select_all_events":
            host._dispatch(SelectAllEvents())
            return True
        if resolved_action_id == "push_unselect_all_events":
            host._dispatch(ClearSelection())
            return True
        if resolved_action_id == "set_push_transfer_mode":
            return self._handle_set_push_transfer_mode()
        if resolved_action_id == "save_transfer_preset":
            return self._handle_save_transfer_preset()
        if resolved_action_id in {"apply_transfer_preset", "delete_transfer_preset"}:
            return self._handle_transfer_preset_action(resolved_action_id)
        if resolved_action_id in {"transfer.plan_preview", "transfer.plan_apply", "transfer.plan_cancel"}:
            return self._handle_transfer_plan_action(resolved_action_id, params)
        if resolved_action_id in {
            "select_push_target_track",
            "preview_push_diff",
            "exit_push_mode",
            "transfer.workspace_open",
            "select_pull_source_tracks",
            "select_pull_source_events",
            "set_pull_target_layer_mapping",
            "preview_pull_diff",
            "exit_pull_workspace",
        }:
            return self._handle_manual_transfer_workspace_action(resolved_action_id, params)
        if is_object_action(resolved_action_id):
            return host._handle_runtime_pipeline_action(resolved_action_id, params)
        return False

    def _handle_set_push_transfer_mode(self) -> bool:
        host = cast(_TransferActionHost, self)
        presentation = host._get_presentation()
        current_mode = (presentation.manual_push_flow.transfer_mode or "merge").strip().lower()
        mode_labels = ["Merge", "Overwrite"]
        default_index = 0 if current_mode == "merge" else 1
        chosen_mode, accepted = host._input_dialog.getItem(
            host._widget,
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
            host._dispatch(SetPushTransferMode(mode=selected_mode))
        return True

    def _handle_save_transfer_preset(self) -> bool:
        host = cast(_TransferActionHost, self)
        preset_name, accepted = host._input_dialog.getText(
            host._widget,
            "Save Transfer Preset",
            "Preset name",
        )
        if not accepted or not preset_name.strip():
            return True
        host._dispatch(SaveTransferPreset(name=preset_name))
        return True

    def _handle_transfer_preset_action(self, action_id: str) -> bool:
        host = cast(_TransferActionHost, self)
        presentation = host._get_presentation()
        if not presentation.transfer_presets:
            return True
        labels = [self._transfer_preset_label(preset) for preset in presentation.transfer_presets]
        title = (
            "Apply Transfer Preset"
            if action_id == "apply_transfer_preset"
            else "Delete Transfer Preset"
        )
        chosen_label, accepted = host._input_dialog.getItem(
            host._widget,
            title,
            "Preset",
            labels,
            0,
            False,
        )
        if not accepted:
            return True
        selected_preset = next(
            (
                preset
                for preset, label in zip(presentation.transfer_presets, labels)
                if label == chosen_label
            ),
            None,
        )
        if selected_preset is None:
            return True
        if action_id == "apply_transfer_preset":
            host._dispatch(ApplyTransferPreset(preset_id=selected_preset.preset_id))
        else:
            host._dispatch(DeleteTransferPreset(preset_id=selected_preset.preset_id))
        return True

    def _handle_transfer_plan_action(self, action_id: str, params: dict[str, object]) -> bool:
        host = cast(_TransferActionHost, self)
        plan_id = params.get("plan_id")
        if action_id == "transfer.plan_cancel":
            if isinstance(plan_id, str):
                host._dispatch(CancelTransferPlan(plan_id=plan_id))
            return True
        if not isinstance(plan_id, str):
            return True
        if not self._resolve_blocked_push_rows_for_plan_action(plan_id):
            return True
        if action_id == "transfer.plan_preview":
            host._dispatch(PreviewTransferPlan(plan_id=plan_id))
            plan = host._get_presentation().batch_transfer_plan
            if plan is not None and plan.plan_id == plan_id:
                host._message_box.information(
                    host._widget,
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
        if action_id == "transfer.plan_apply":
            host._dispatch(ApplyTransferPlan(plan_id=plan_id))
            plan = host._get_presentation().batch_transfer_plan
            if plan is not None and plan.plan_id == plan_id:
                host._message_box.information(
                    host._widget,
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

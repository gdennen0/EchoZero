"""Timeline widget contract and intent helpers.
Exists to keep application-intent dispatch and inspector action routing out of the widget shell.
Connects canvas signals and object-info actions to canonical timeline intents and runtime actions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QMenu, QWidget

from echozero.application.presentation.inspector_contract import (
    InspectorAction,
    InspectorContract,
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.application.presentation.models import (
    BatchTransferPlanPresentation,
    ManualPullFlowPresentation,
    TimelinePresentation,
)
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.shared.ranges import TimeRange
from echozero.application.timeline.intents import (
    ClearSelection,
    CreateEvent,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveSelectedEventsToAdjacentLayer,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    OpenPullFromMA3Dialog,
    OpenPushToMA3Dialog,
    Seek,
    SelectAdjacentEventInSelectedLayer,
    SelectAdjacentLayer,
    SelectAllEvents,
    SelectEvent,
    SelectLayer,
    SelectTake,
    SetActivePlaybackTarget,
    SetSelectedEvents,
    TimelineIntent,
    ToggleLayerExpanded,
    TriggerTakeAction,
)
from echozero.application.timeline.models import EventRef
from echozero.application.timeline.object_actions import ObjectActionSettingsPlan
from echozero.ui.qt.timeline.manual_pull import (
    ManualPullTimelineSelectionResult,
)
from echozero.ui.qt.timeline.object_info_panel import ObjectInfoPanel


class _TimelineRuntimeShell(Protocol):
    def presentation(self) -> TimelinePresentation: ...

    def describe_object_action(
        self,
        action_id: str,
        params: dict[str, object],
        *,
        object_id: str | None = None,
        object_type: str | None = None,
    ) -> ObjectActionSettingsPlan: ...


class _TimelineWidgetActionRouter(Protocol):
    def open_object_action_settings(self, action: InspectorAction) -> None: ...
    def trigger_contract_action(self, action: InspectorAction) -> None: ...
    def _handle_runtime_pipeline_action(self, action_id: str, params: dict[str, object]) -> bool: ...
    def _default_open_manual_pull_timeline_popup(
        self, flow: ManualPullFlowPresentation
    ) -> ManualPullTimelineSelectionResult | None: ...


class _TimelineWidgetContractHost(Protocol):
    presentation: TimelinePresentation
    _on_intent: Callable[[TimelineIntent], TimelinePresentation | None] | None
    _action_router: _TimelineWidgetActionRouter
    _object_info: ObjectInfoPanel

    def _dispatch(self, intent: TimelineIntent) -> None: ...
    def _focus_layer_for_header_action(self, layer_id: LayerId) -> None: ...
    def _selected_event_ids_for_selected_layers(self) -> list[EventId]: ...
    def _handle_contract_action(self, action: InspectorAction) -> None: ...
    def _resolve_runtime_shell(self) -> _TimelineRuntimeShell | None: ...
    def _resolve_object_action_settings_plans(
        self, contract: InspectorContract
    ) -> tuple[ObjectActionSettingsPlan, ...]: ...


class TimelineWidgetContractMixin:
    def _seek(self: _TimelineWidgetContractHost, position: float) -> None:
        self._dispatch(Seek(position))

    def _select_layer(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
        mode: str = "replace",
    ) -> None:
        self._dispatch(SelectLayer(layer_id, mode=mode))

    def _select_adjacent_layer(
        self: _TimelineWidgetContractHost,
        direction: int,
    ) -> None:
        if direction == 0:
            return
        self._dispatch(SelectAdjacentLayer(direction=direction))

    def _toggle_take_selector(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
    ) -> None:
        self._dispatch(ToggleLayerExpanded(layer_id))

    def _select_take(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
        take_id: TakeId | None,
    ) -> None:
        if take_id is None:
            return
        self._dispatch(SelectTake(layer_id, take_id))

    def _select_event(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
        take_id: TakeId | None,
        event_id: EventId,
        mode: str,
    ) -> None:
        self._dispatch(SelectEvent(layer_id, take_id, event_id, mode=mode))

    def _select_adjacent_event_in_selected_layer(
        self: _TimelineWidgetContractHost,
        direction: int,
    ) -> None:
        if direction == 0:
            return
        self._dispatch(SelectAdjacentEventInSelectedLayer(direction=direction))

    def _set_selected_events(
        self: _TimelineWidgetContractHost,
        event_ids: list[EventId],
        event_refs: list[EventRef],
        anchor_layer_id: LayerId | None,
        anchor_take_id: TakeId | None,
        selected_layer_ids: list[LayerId],
    ) -> None:
        self._dispatch(
            SetSelectedEvents(
                event_ids=list(event_ids),
                event_refs=list(event_refs),
                anchor_layer_id=anchor_layer_id,
                anchor_take_id=anchor_take_id,
                selected_layer_ids=list(selected_layer_ids),
            )
        )

    def _create_event(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
        take_id: TakeId | None,
        start_seconds: float,
        end_seconds: float,
    ) -> None:
        self._dispatch(
            CreateEvent(
                layer_id=layer_id,
                take_id=take_id,
                time_range=TimeRange(
                    start=max(0.0, min(float(start_seconds), float(end_seconds))),
                    end=max(float(start_seconds), float(end_seconds)),
                ),
            )
        )

    def _delete_events(
        self: _TimelineWidgetContractHost,
        event_ids: list[EventId] | list[EventRef],
    ) -> None:
        ids = list(event_ids)
        if not ids:
            return
        if isinstance(ids[0], EventRef):
            event_refs = cast(list[EventRef], ids)
            self._dispatch(
                DeleteEvents(
                    event_ids=[event_ref.event_id for event_ref in event_refs],
                    event_refs=event_refs,
                )
            )
            return
        self._dispatch(DeleteEvents(event_ids=cast(list[EventId], ids)))

    def _clear_selection(self: _TimelineWidgetContractHost) -> None:
        self._dispatch(ClearSelection())

    def _select_all_events(self: _TimelineWidgetContractHost) -> None:
        self._dispatch(SelectAllEvents())

    def _nudge_selected_events(
        self: _TimelineWidgetContractHost,
        direction: int,
        steps: int,
    ) -> None:
        self._dispatch(NudgeSelectedEvents(direction=direction, steps=steps))

    def _duplicate_selected_events(
        self: _TimelineWidgetContractHost,
        steps: int,
    ) -> None:
        self._dispatch(DuplicateSelectedEvents(steps=steps))

    def _trigger_take_action(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
        take_id: TakeId | None,
        action_id: str,
    ) -> None:
        if take_id is None or not action_id:
            return
        self._dispatch(TriggerTakeAction(layer_id, take_id, action_id))

    def _move_selected_events(
        self: _TimelineWidgetContractHost,
        delta_seconds: float,
        target_layer_id: LayerId | None,
    ) -> None:
        self._dispatch(
            MoveSelectedEvents(delta_seconds=delta_seconds, target_layer_id=target_layer_id)
        )

    def _move_selected_events_to_adjacent_layer(
        self: _TimelineWidgetContractHost,
        direction: int,
    ) -> None:
        if direction == 0:
            return
        self._dispatch(MoveSelectedEventsToAdjacentLayer(direction=direction))

    def _set_active_playback_target(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
    ) -> None:
        self._dispatch(SetActivePlaybackTarget(layer_id=layer_id, take_id=None))

    def _open_push_from_layer_action(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
    ) -> None:
        self._focus_layer_for_header_action(layer_id)
        self._handle_contract_action(
            InspectorAction(
                action_id="send_layer_to_ma3",
                label="Send Layer to MA3",
                params={"layer_id": layer_id},
            )
        )

    def _open_pull_from_layer_action(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
    ) -> None:
        self._focus_layer_for_header_action(layer_id)
        self._dispatch(OpenPullFromMA3Dialog())

    def _focus_layer_for_header_action(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
    ) -> None:
        if self.presentation.selected_layer_id != layer_id:
            self._dispatch(SelectLayer(layer_id))

    def _selected_event_ids_for_selected_layers(
        self: _TimelineWidgetContractHost,
    ) -> list[EventId]:
        selected_layer_ids = set(self.presentation.selected_layer_ids)
        if not selected_layer_ids and self.presentation.selected_layer_id is not None:
            selected_layer_ids = {self.presentation.selected_layer_id}
        if not selected_layer_ids:
            return list(self.presentation.selected_event_ids)

        allowed_event_ids: set[EventId] = set()
        for layer in self.presentation.layers:
            if layer.layer_id not in selected_layer_ids:
                continue
            for event in layer.events:
                allowed_event_ids.add(event.event_id)
        return [
            event_id
            for event_id in self.presentation.selected_event_ids
            if event_id in allowed_event_ids
        ]

    def _preview_active_transfer_plan(self: _TimelineWidgetContractHost) -> None:
        plan = self.presentation.batch_transfer_plan
        if plan is None:
            return
        self._handle_contract_action(
            InspectorAction(
                action_id="transfer.plan_preview",
                label=_preview_transfer_plan_label(plan),
                params={"plan_id": plan.plan_id},
            )
        )

    def _apply_active_transfer_plan(self: _TimelineWidgetContractHost) -> None:
        plan = self.presentation.batch_transfer_plan
        if plan is None:
            return
        self._handle_contract_action(
            InspectorAction(
                action_id="transfer.plan_apply",
                label=_apply_transfer_plan_label(plan),
                params={"plan_id": plan.plan_id},
            )
        )

    def _cancel_active_transfer_plan(self: _TimelineWidgetContractHost) -> None:
        plan = self.presentation.batch_transfer_plan
        if plan is None:
            return
        self._handle_contract_action(
            InspectorAction(
                action_id="transfer.plan_cancel",
                label="Cancel Transfer Plan",
                params={"plan_id": plan.plan_id},
            )
        )

    def _handle_contract_action(
        self: _TimelineWidgetContractHost,
        action: InspectorAction,
    ) -> None:
        if action.kind == "settings":
            self._action_router.open_object_action_settings(action)
            return
        self._action_router.trigger_contract_action(action)

    def _open_action_settings_dialog(
        self: _TimelineWidgetContractHost,
        action: InspectorAction,
    ) -> None:
        self._action_router.open_object_action_settings(action)

    def _open_layer_pipeline_actions(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
    ) -> None:
        self._focus_layer_for_header_action(layer_id)
        contract = build_timeline_inspector_contract(
            self.presentation,
            hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=layer_id),
        )
        pipeline_actions = [
            action
            for section in contract.context_sections
            for action in section.actions
            if action.action_id.startswith("timeline.")
        ]
        if not pipeline_actions:
            return

        runtime = self._resolve_runtime_shell()
        describe = (
            getattr(runtime, "describe_object_action", None) if runtime is not None else None
        )
        menu = QMenu(cast(QWidget, self))
        for index, action in enumerate(pipeline_actions):
            if index:
                menu.addSeparator()
            plan = None
            if callable(describe):
                try:
                    plan = describe(
                        action.action_id,
                        action.params,
                        object_id=layer_id,
                        object_type="layer",
                    )
                except Exception:
                    plan = None
            settings_entry = menu.addAction(f"Open {action.label} Settings")
            if settings_entry is not None:
                settings_entry.setData(
                    InspectorAction(
                        action_id=action.action_id,
                        label=action.label,
                        kind="settings",
                        params=dict(action.params),
                    )
                )
            run_entry = menu.addAction(
                f"{plan.run_label} {action.label}" if plan is not None else f"Run {action.label}"
            )
            if run_entry is not None:
                run_entry.setData(
                    InspectorAction(
                        action_id=action.action_id,
                        label=action.label,
                        params=dict(action.params),
                    )
                )

        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            return
        payload = chosen.data()
        if isinstance(payload, InspectorAction):
            self._handle_contract_action(payload)

    def _trigger_contract_action(
        self: _TimelineWidgetContractHost,
        action: InspectorAction,
    ) -> None:
        self._handle_contract_action(action)

    def _handle_runtime_pipeline_action(
        self: _TimelineWidgetContractHost,
        action_id: str,
        params: dict[str, object],
    ) -> bool:
        return self._action_router._handle_runtime_pipeline_action(action_id, params)

    def _open_manual_pull_timeline_popup(
        self: _TimelineWidgetContractHost,
        flow: ManualPullFlowPresentation,
    ) -> ManualPullTimelineSelectionResult | None:
        return self._action_router._default_open_manual_pull_timeline_popup(flow)

    def _resolve_runtime_shell(
        self: _TimelineWidgetContractHost,
    ) -> _TimelineRuntimeShell | None:
        owner = getattr(self._on_intent, "__self__", None)
        if owner is not None and all(
            hasattr(owner, method_name) for method_name in ("presentation",)
        ):
            return cast(_TimelineRuntimeShell, owner)
        runtime = getattr(owner, "runtime", None)
        if runtime is not None and hasattr(runtime, "presentation"):
            return cast(_TimelineRuntimeShell, runtime)
        return None

    def _refresh_object_info_panel(self: _TimelineWidgetContractHost) -> None:
        contract = build_timeline_inspector_contract(self.presentation)
        self._object_info.set_contract(self.presentation, contract)
        self._object_info.set_action_settings_plans(
            self._resolve_object_action_settings_plans(contract)
        )

    def _resolve_object_action_settings_plans(
        self: _TimelineWidgetContractHost,
        contract: InspectorContract,
    ) -> tuple[ObjectActionSettingsPlan, ...]:
        runtime = self._resolve_runtime_shell()
        describe = getattr(runtime, "describe_object_action", None) if runtime is not None else None
        if not callable(describe):
            return ()
        plans: list[ObjectActionSettingsPlan] = []
        object_identity = contract.identity
        for section in contract.context_sections:
            for action in section.actions:
                if not action.action_id.startswith("timeline."):
                    continue
                try:
                    plan = describe(
                        action.action_id,
                        action.params,
                        object_id=(
                            object_identity.object_id if object_identity is not None else None
                        ),
                        object_type=(
                            object_identity.object_type if object_identity is not None else None
                        ),
                    )
                except Exception:
                    continue
                plans.append(plan)
        return tuple(plans)


def _ready_count_label(count: int) -> str:
    noun = "ready row" if count == 1 else "ready rows"
    return f"{count} {noun}"


def _preview_transfer_plan_label(plan: BatchTransferPlanPresentation) -> str:
    return f"Preview Transfer Plan ({_ready_count_label(plan.ready_count)})"


def _apply_transfer_plan_label(plan: BatchTransferPlanPresentation) -> str:
    return f"Apply Transfer Plan ({_ready_count_label(plan.ready_count)})"


__all__ = ["TimelineWidgetContractMixin"]

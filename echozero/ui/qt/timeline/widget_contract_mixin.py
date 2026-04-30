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
    LayerPresentation,
    ManualPullFlowPresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.shared.ranges import TimeRange
from echozero.application.timeline.intents import (
    ClearSelection,
    CommitRejectedEventsReview,
    CommitRejectedEventReview,
    CommitVerifiedEventsReview,
    CreateEvent,
    DeleteEvents,
    DuplicateSelectedEvents,
    ReorderLayer,
    MoveSelectedEventsToAdjacentLayer,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    OpenPushToMA3Dialog,
    Seek,
    SelectAdjacentEventInSelectedLayer,
    SelectAdjacentLayer,
    SelectAllEvents,
    SelectEvent,
    SelectLayer,
    SelectTake,
    SetLayerMute,
    SetLayerSolo,
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
from echozero.ui.FEEL import TIMELINE_ADD_MODE_DEFAULT_EVENT_DURATION_SECONDS


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
    _edit_mode: str
    _fix_action: str
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
        include_demoted: bool = False,
    ) -> None:
        if direction == 0:
            return
        self._dispatch(
            SelectAdjacentEventInSelectedLayer(
                direction=direction,
                include_demoted=bool(include_demoted),
            )
        )

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

    def _create_event_at_playhead(self: _TimelineWidgetContractHost) -> None:
        target_lane = self._resolve_event_creation_lane()
        if target_lane is None:
            return
        layer_id, take_id = target_lane
        start_seconds = max(0.0, float(self.presentation.playhead))
        end_seconds = (
            start_seconds + float(TIMELINE_ADD_MODE_DEFAULT_EVENT_DURATION_SECONDS)
        )
        self._create_event(
            layer_id,
            take_id,
            start_seconds,
            end_seconds,
        )

    def _resolve_event_creation_lane(
        self: _TimelineWidgetContractHost,
    ) -> tuple[LayerId, TakeId | None] | None:
        selected_layer_id = self.presentation.selected_layer_id
        selected_layer = self._find_layer_presentation(selected_layer_id)
        if selected_layer is not None and is_event_like_layer_kind(selected_layer.kind):
            return (
                selected_layer.layer_id,
                self._resolve_selected_take_for_layer(selected_layer),
            )

        for layer in self.presentation.layers:
            if is_event_like_layer_kind(layer.kind):
                return (layer.layer_id, layer.main_take_id)
        return None

    def _find_layer_presentation(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId | None,
    ) -> LayerPresentation | None:
        if layer_id is None:
            return None
        return next(
            (
                layer
                for layer in self.presentation.layers
                if layer.layer_id == layer_id
            ),
            None,
        )

    def _resolve_selected_take_for_layer(
        self: _TimelineWidgetContractHost,
        layer: LayerPresentation,
    ) -> TakeId | None:
        selected_take_id = self.presentation.selected_take_id
        if selected_take_id is None:
            return layer.main_take_id
        if selected_take_id == layer.main_take_id:
            return selected_take_id
        if any(take.take_id == selected_take_id for take in layer.takes):
            return selected_take_id
        return layer.main_take_id

    def _delete_events(
        self: _TimelineWidgetContractHost,
        event_ids: list[EventId] | list[EventRef],
    ) -> None:
        ids = list(event_ids)
        if not ids:
            return
        if self._is_fix_mode_remove():
            rejected_event_refs = self._resolve_event_refs_for_fix_review(ids)
            if len(rejected_event_refs) > 1:
                self._dispatch(
                    CommitRejectedEventsReview(event_refs=rejected_event_refs)
                )
            elif len(rejected_event_refs) == 1:
                event_ref = rejected_event_refs[0]
                self._dispatch(
                    CommitRejectedEventReview(
                        layer_id=event_ref.layer_id,
                        event_id=event_ref.event_id,
                        take_id=event_ref.take_id,
                    )
                )
            if rejected_event_refs:
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

    def _demote_fix_selected_events(
        self: _TimelineWidgetContractHost,
        event_ids: list[EventId] | list[EventRef],
    ) -> None:
        event_refs = self._resolve_event_refs_for_fix_review(event_ids)
        if event_refs:
            self._dispatch(CommitRejectedEventsReview(event_refs=event_refs))
            return

    def _promote_fix_selected_events(
        self: _TimelineWidgetContractHost,
        event_ids: list[EventId] | list[EventRef],
    ) -> None:
        event_refs = self._resolve_event_refs_for_fix_review(event_ids)
        if event_refs:
            self._dispatch(CommitVerifiedEventsReview(event_refs=event_refs))
            return

    def _is_fix_mode_remove(self: _TimelineWidgetContractHost) -> bool:
        return (
            str(getattr(self, "_edit_mode", "")).strip().lower() == "fix"
            and str(getattr(self, "_fix_action", "")).strip().lower() == "remove"
        )

    def _resolve_event_refs_for_fix_review(
        self: _TimelineWidgetContractHost,
        event_ids: list[EventId] | list[EventRef],
    ) -> list[EventRef]:
        if not event_ids:
            return []
        if isinstance(event_ids[0], EventRef):
            return list(cast(list[EventRef], event_ids))

        requested_ids = [str(event_id) for event_id in cast(list[EventId], event_ids)]
        if not requested_ids:
            return []
        requested_id_set = set(requested_ids)

        selected_refs = [
            event_ref
            for event_ref in self.presentation.selected_event_refs
            if str(event_ref.event_id) in requested_id_set
        ]
        if selected_refs:
            return selected_refs

        preferred_take_id = self.presentation.selected_take_id
        preferred_layer_ids = [
            layer_id for layer_id in self.presentation.selected_layer_ids if layer_id is not None
        ]
        if not preferred_layer_ids and self.presentation.selected_layer_id is not None:
            preferred_layer_ids = [self.presentation.selected_layer_id]
        preferred_layers = {str(layer_id) for layer_id in preferred_layer_ids}

        event_records: dict[str, list[EventRef]] = {}
        for layer in self.presentation.layers:
            if layer.main_take_id is not None:
                for event in layer.events:
                    event_records.setdefault(str(event.event_id), []).append(
                        EventRef(
                            layer_id=layer.layer_id,
                            take_id=layer.main_take_id,
                            event_id=event.event_id,
                        )
                    )
            for take in layer.takes:
                for event in take.events:
                    event_records.setdefault(str(event.event_id), []).append(
                        EventRef(
                            layer_id=layer.layer_id,
                            take_id=take.take_id,
                            event_id=event.event_id,
                        )
                    )

        resolved: list[EventRef] = []
        seen: set[tuple[str, str, str]] = set()
        for requested_id in requested_ids:
            matches = list(event_records.get(requested_id, []))
            if not matches:
                continue

            preferred_matches = [
                match
                for match in matches
                if (
                    (preferred_take_id is None or match.take_id == preferred_take_id)
                    and (not preferred_layers or str(match.layer_id) in preferred_layers)
                )
            ]
            if preferred_matches:
                chosen = preferred_matches[0]
            elif preferred_take_id is not None:
                take_matches = [match for match in matches if match.take_id == preferred_take_id]
                chosen = take_matches[0] if take_matches else matches[0]
            else:
                chosen = matches[0]

            key = (str(chosen.layer_id), str(chosen.take_id), str(chosen.event_id))
            if key in seen:
                continue
            resolved.append(chosen)
            seen.add(key)
        return resolved

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
        copy_selected: bool = False,
    ) -> None:
        self._dispatch(
            MoveSelectedEvents(
                delta_seconds=delta_seconds,
                target_layer_id=target_layer_id,
                copy_selected=copy_selected,
            )
        )

    def _move_selected_events_to_adjacent_layer(
        self: _TimelineWidgetContractHost,
        direction: int,
    ) -> None:
        if direction == 0:
            return
        self._dispatch(MoveSelectedEventsToAdjacentLayer(direction=direction))

    def _reorder_layer(
        self: _TimelineWidgetContractHost,
        source_layer_id: LayerId,
        target_after_layer_id: LayerId | None,
        insert_at_start: bool = False,
    ) -> None:
        self._dispatch(
            ReorderLayer(
                source_layer_id=source_layer_id,
                target_after_layer_id=target_after_layer_id,
                insert_at_start=insert_at_start,
            )
        )

    def _toggle_layer_mute_from_header(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
    ) -> None:
        layer = next(
            (
                candidate
                for candidate in self.presentation.layers
                if candidate.layer_id == layer_id
            ),
            None,
        )
        if layer is None:
            return
        if layer.kind is LayerKind.EVENT:
            return
        self._dispatch(SetLayerMute(layer_id=layer_id, muted=not bool(layer.muted)))

    def _toggle_layer_solo_from_header(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
    ) -> None:
        layer = next(
            (
                candidate
                for candidate in self.presentation.layers
                if candidate.layer_id == layer_id
            ),
            None,
        )
        if layer is None:
            return
        if layer.kind is LayerKind.EVENT:
            return
        self._dispatch(SetLayerSolo(layer_id=layer_id, soloed=not bool(layer.soloed)))

    def _open_push_from_layer_action(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
    ) -> None:
        self._focus_layer_for_header_action(layer_id)
        self._handle_contract_action(
            InspectorAction(
                action_id="transfer.workspace_open",
                label="Send Layer to MA3",
                params={"layer_id": layer_id, "direction": "push"},
            )
        )

    def _open_pull_from_layer_action(
        self: _TimelineWidgetContractHost,
        layer_id: LayerId,
    ) -> None:
        self._focus_layer_for_header_action(layer_id)
        self._handle_contract_action(
            InspectorAction(
                action_id="transfer.workspace_open",
                label="Import Event Layer from MA3",
                params={"layer_id": layer_id, "direction": "pull"},
            )
        )

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

    def _preview_selected_event_clip(self: _TimelineWidgetContractHost) -> None:
        action = self._selected_event_preview_action()
        if action is None or not action.enabled:
            return
        self._handle_contract_action(action)

    def _selected_event_preview_action(
        self: _TimelineWidgetContractHost,
    ) -> InspectorAction | None:
        hit_target = self._preview_event_hit_target_for_selection(self.presentation)
        contract = (
            build_timeline_inspector_contract(self.presentation, hit_target=hit_target)
            if hit_target is not None
            else build_timeline_inspector_contract(self.presentation)
        )
        for section in contract.context_sections:
            for action in section.actions:
                if action.action_id == "preview_event_clip":
                    return action
        return None

    @staticmethod
    def _preview_event_hit_target_for_selection(
        presentation: TimelinePresentation,
    ) -> TimelineInspectorHitTarget | None:
        if presentation.selected_event_refs:
            selected_ref = presentation.selected_event_refs[-1]
            return TimelineInspectorHitTarget(
                kind="event",
                layer_id=selected_ref.layer_id,
                take_id=selected_ref.take_id,
                event_id=selected_ref.event_id,
            )
        if not presentation.selected_event_ids:
            return None
        return TimelineInspectorHitTarget(
            kind="event",
            layer_id=presentation.selected_layer_id,
            take_id=presentation.selected_take_id,
            event_id=presentation.selected_event_ids[-1],
        )

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
            if (action.group or "").strip().lower() == "pipeline"
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
                if (action.group or "").strip().lower() != "pipeline":
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

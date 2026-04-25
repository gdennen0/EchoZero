"""Editing facade mixin for the Qt app shell runtime.
Exists to isolate manual-layer creation and intent dispatch on the public shell surface.
Connects AppShellRuntime to undo/history helpers and the timeline edit contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.shared.ids import LayerId
from echozero.application.shared.enums import LayerKind
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    CreateEvent,
    CreateRegion,
    DeleteRegion,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveEvent,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    ReorderLayer,
    SetGain,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    TimelineIntent,
    ToggleLayerExpanded,
    TriggerTakeAction,
    TrimEvent,
    UpdateRegion,
)
from echozero.application.timeline.ma3_push_intents import SetLayerMA3Route
from echozero.application.timeline.models import Layer
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_history import (
    history_label_for_intent,
    is_history_barrier_intent,
    is_storage_backed_undoable_intent,
    is_undoable_intent,
)
from echozero.ui.qt.app_shell_layer_storage import build_manual_layer
from echozero.ui.qt.app_shell_timeline_state import clear_selected_events

_DIRTYING_INTENT_TYPES = (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    CreateEvent,
    CreateRegion,
    DeleteRegion,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveEvent,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    ReorderLayer,
    SetGain,
    SetLayerMA3Route,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
    ToggleLayerExpanded,
    TriggerTakeAction,
    TrimEvent,
    UpdateRegion,
)


class AppShellEditingShell(Protocol):
    _app: TimelineApplication
    _is_dirty: bool

    @property
    def session(self) -> Session: ...
    project_storage: ProjectStorage

    def presentation(self) -> TimelinePresentation: ...

    def _clear_history(self) -> None: ...

    def _run_undoable_operation(
        self,
        *,
        label: str,
        storage_backed: bool,
        mark_dirty: bool,
        operation: Callable[[], TimelinePresentation],
    ) -> TimelinePresentation: ...

    def _store_manual_layer(self, layer: Layer) -> None: ...

    def _sync_storage_backed_timeline(self) -> None: ...

    def _sync_runtime_audio_from_presentation(self, presentation: TimelinePresentation) -> None: ...


class AppShellEditingMixin:
    def add_layer(
        self: AppShellEditingShell,
        kind: LayerKind,
        title: str | None = None,
    ) -> TimelinePresentation:
        def _perform_add_layer() -> TimelinePresentation:
            layer_kind = kind
            if not isinstance(layer_kind, LayerKind):
                try:
                    layer_kind = LayerKind(str(layer_kind))
                except ValueError as exc:
                    raise ValueError(f"Unsupported layer kind '{kind}'.") from exc

            layer_title = (title or "").strip()
            if not layer_title:
                layer_title = f"{layer_kind.value.title()} Layer"

            timeline = self._app.timeline
            new_layer = build_manual_layer(
                timeline=timeline,
                layer_kind=layer_kind,
                layer_title=layer_title,
            )
            timeline.layers.append(new_layer)
            self._store_manual_layer(new_layer)
            timeline.selection.selected_layer_id = new_layer.id
            timeline.selection.selected_layer_ids = [new_layer.id]
            timeline.selection.selected_take_id = None
            clear_selected_events(timeline)
            timeline.playback_target.layer_id = new_layer.id
            timeline.playback_target.take_id = None
            self._sync_runtime_audio_from_presentation(self.presentation())
            self._is_dirty = True
            return self.presentation()

        return self._run_undoable_operation(
            label="Add Layer",
            storage_backed=self.session.active_song_version_id is not None,
            mark_dirty=True,
            operation=_perform_add_layer,
        )

    def delete_layer(
        self: AppShellEditingShell,
        layer_id: str,
    ) -> TimelinePresentation:
        target_layer_id = LayerId(layer_id)

        def _perform_delete_layer() -> TimelinePresentation:
            if target_layer_id == LayerId("source_audio"):
                raise ValueError("Cannot delete source audio layer.")

            timeline = self._app.timeline
            target_layer = next(
                (layer for layer in timeline.layers if layer.id == target_layer_id),
                None,
            )
            if target_layer is None:
                raise ValueError(f"Layer not found: {layer_id}")

            timeline.layers = [layer for layer in timeline.layers if layer.id != target_layer_id]
            previous_selected_layer_id = timeline.selection.selected_layer_id
            self._draft_layers = [
                layer for layer in self._draft_layers if layer.id != target_layer_id
            ]

            selected_layer_ids = [
                layer_id for layer_id in dict.fromkeys(timeline.selection.selected_layer_ids)
                if layer_id != target_layer_id
            ]
            selected_layer_id = timeline.selection.selected_layer_id
            if selected_layer_id == target_layer_id:
                selected_layer_id = selected_layer_ids[0] if selected_layer_ids else (
                    timeline.layers[0].id if timeline.layers else None
                )
            timeline.selection.selected_layer_id = selected_layer_id
            if selected_layer_id is not None and selected_layer_id not in selected_layer_ids:
                selected_layer_ids = [selected_layer_id]
            timeline.selection.selected_layer_ids = selected_layer_ids

            if previous_selected_layer_id == target_layer_id:
                timeline.selection.selected_take_id = None

            if timeline.playback_target.layer_id == target_layer_id:
                timeline.playback_target.layer_id = selected_layer_id or (
                    timeline.layers[0].id if timeline.layers else None
                )
                timeline.playback_target.take_id = None
            clear_selected_events(timeline)

            active_version_id = self.session.active_song_version_id
            if active_version_id is not None:
                with self.project_storage.transaction():
                    for take in target_layer.takes:
                        self.project_storage.takes.delete(str(take.id))
                    self.project_storage.layers.delete(str(target_layer_id))
                    self.project_storage.dirty_tracker.mark_dirty(str(active_version_id))

            self._sync_runtime_audio_from_presentation(self.presentation())
            return self.presentation()

        return self._run_undoable_operation(
            label="Delete Layer",
            storage_backed=self.session.active_song_version_id is not None,
            mark_dirty=True,
            operation=_perform_delete_layer,
        )

    def dispatch(
        self: AppShellEditingShell,
        intent: TimelineIntent,
    ) -> TimelinePresentation:
        if is_undoable_intent(intent):
            return self._run_undoable_operation(
                label=history_label_for_intent(intent) or intent.__class__.__name__,
                storage_backed=is_storage_backed_undoable_intent(intent),
                mark_dirty=isinstance(intent, _DIRTYING_INTENT_TYPES),
                operation=lambda: self._app.dispatch(intent),
            )

        presentation = self._app.dispatch(intent)
        if isinstance(intent, ToggleLayerExpanded):
            self._sync_storage_backed_timeline()
        if is_history_barrier_intent(intent):
            self._clear_history()
        if isinstance(intent, _DIRTYING_INTENT_TYPES):
            self._is_dirty = True
        return presentation

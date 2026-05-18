"""Undo/history helpers for the Qt app shell runtime.
Exists to keep undo semantics at the app boundary instead of inside widgets.
Connects intent classification and snapshot restore logic to AppShellRuntime.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, TypeVar

from echozero.application.mixer.models import MixerState
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.shared.ids import LayerId, SongId, SongVersionId, TimelineId
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.history import UndoHistory, UndoHistoryEntry
from echozero.application.timeline.intents import (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    CommitBoundaryCorrectedEventReview,
    CommitMissedEventsReview,
    CommitMissedEventReview,
    CommitRejectedEventsReview,
    CommitRejectedEventReview,
    CommitRelabeledEventReview,
    CommitVerifiedEventsReview,
    CommitVerifiedEventReview,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
    CreateEvent,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveEvent,
    MoveSelectedEvents,
    PasteCopiedEvents,
    ReorderLayer,
    NudgeSelectedEvents,
    ReplaceSectionCues,
    SelectTake,
    SetGain,
    SnapEventsToBeatGrid,
    TriggerTakeAction,
    TrimEvent,
    UpdateEventLabel,
)
from echozero.application.timeline.ma3_push_intents import SetLayerMA3Route
from echozero.application.timeline.models import Layer, Timeline
from echozero.persistence.session import ProjectStorage

DEFAULT_HISTORY_LIMIT = 100

_UNDOABLE_TAKE_ACTION_IDS = {
    "overwrite_main",
    "promote_take",
    "merge_main",
    "add_selection_to_main",
    "delete_take",
}
_HISTORY_BARRIER_INTENT_TYPES = (
    ApplyPullFromMA3,
    ApplyTransferPlan,
    ConfirmPullFromMA3,
    ConfirmPushToMA3,
)

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RuntimeHistorySnapshot:
    """Bounded runtime state captured around one undoable app-shell operation."""

    timeline: Timeline
    active_song_id: SongId | None
    active_song_version_id: SongVersionId | None
    active_timeline_id: TimelineId | None
    mixer_state: MixerState
    draft_layers: list[Layer]
    is_dirty: bool


@dataclass(frozen=True, slots=True)
class RuntimeScopedHistorySnapshot:
    """Scoped runtime state captured around one hot-path layer edit."""

    layers: dict[LayerId, Layer | None]
    selected_layer_id: LayerId | None
    selected_layer_ids: list[LayerId]
    selected_take_id: object | None
    selected_event_refs: list[object]
    selected_event_ids: list[object]
    section_cues: list[object]
    is_dirty: bool


class HistoryShell(Protocol):
    _app: TimelineApplication
    _draft_layers: list[Layer]
    _history: UndoHistory
    _is_dirty: bool
    project_storage: ProjectStorage

    @property
    def session(self) -> Session: ...

    def presentation(self) -> TimelinePresentation: ...

    def _sync_storage_backed_timeline(self) -> None: ...

    def _sync_storage_backed_layers(self, layer_ids: list[LayerId]) -> None: ...

    def _defer_storage_backed_timeline_sync(self) -> None: ...

    def _defer_storage_backed_layers_sync(self, layer_ids: list[LayerId]) -> None: ...

    def _sync_runtime_audio_from_presentation(
        self, presentation: TimelinePresentation
    ) -> None: ...


def is_undoable_intent(intent: object) -> bool:
    if isinstance(intent, SelectTake):
        return True
    if isinstance(intent, SetLayerMA3Route):
        return True
    if isinstance(
        intent,
        (
            CommitMissedEventReview,
            CommitMissedEventsReview,
            CommitVerifiedEventReview,
            CommitVerifiedEventsReview,
            CommitRejectedEventReview,
            CommitRejectedEventsReview,
            CommitRelabeledEventReview,
            CommitBoundaryCorrectedEventReview,
            CreateEvent,
            DeleteEvents,
            DuplicateSelectedEvents,
            MoveEvent,
            MoveSelectedEvents,
            PasteCopiedEvents,
            ReorderLayer,
            NudgeSelectedEvents,
            ReplaceSectionCues,
            SetGain,
            SnapEventsToBeatGrid,
            TrimEvent,
            UpdateEventLabel,
        ),
    ):
        return True
    if isinstance(intent, TriggerTakeAction):
        return _normalized_take_action_id(intent.action_id) in _UNDOABLE_TAKE_ACTION_IDS
    return False


def is_storage_backed_undoable_intent(intent: object) -> bool:
    if isinstance(intent, (SelectTake, SetGain)):
        return False
    if isinstance(intent, SetLayerMA3Route):
        return True
    if isinstance(intent, TriggerTakeAction):
        return _normalized_take_action_id(intent.action_id) in _UNDOABLE_TAKE_ACTION_IDS
    return isinstance(
        intent,
        (
            CommitMissedEventReview,
            CommitMissedEventsReview,
            CommitVerifiedEventReview,
            CommitVerifiedEventsReview,
            CommitRejectedEventReview,
            CommitRejectedEventsReview,
            CommitRelabeledEventReview,
            CommitBoundaryCorrectedEventReview,
            CreateEvent,
            DeleteEvents,
            DuplicateSelectedEvents,
            MoveEvent,
            MoveSelectedEvents,
            PasteCopiedEvents,
            ReorderLayer,
            NudgeSelectedEvents,
            ReplaceSectionCues,
            TrimEvent,
            SnapEventsToBeatGrid,
            UpdateEventLabel,
        ),
    )


def is_history_barrier_intent(intent: object) -> bool:
    return isinstance(intent, _HISTORY_BARRIER_INTENT_TYPES)


def history_label_for_intent(intent: object) -> str | None:
    if isinstance(intent, SelectTake):
        return "Switch Take"
    if isinstance(intent, SetLayerMA3Route):
        return "Route Layer To MA3"
    if isinstance(intent, CommitMissedEventReview):
        return "Add Missed Event"
    if isinstance(intent, CommitMissedEventsReview):
        return "Add Missed Events"
    if isinstance(intent, CommitVerifiedEventReview):
        return "Verify Event"
    if isinstance(intent, CommitVerifiedEventsReview):
        return "Verify Events"
    if isinstance(intent, CommitRejectedEventReview):
        return "Reject Event"
    if isinstance(intent, CommitRejectedEventsReview):
        return "Reject Events"
    if isinstance(intent, CommitRelabeledEventReview):
        return "Relabel Event"
    if isinstance(intent, CommitBoundaryCorrectedEventReview):
        return "Correct Boundary"
    if isinstance(intent, CreateEvent):
        return "Create Event"
    if isinstance(intent, DeleteEvents):
        return "Delete Events"
    if isinstance(intent, DuplicateSelectedEvents):
        return "Duplicate Events"
    if isinstance(intent, MoveEvent):
        return "Move Event"
    if isinstance(intent, MoveSelectedEvents):
        return "Move Events"
    if isinstance(intent, PasteCopiedEvents):
        return "Paste Events"
    if isinstance(intent, ReorderLayer):
        return "Reorder Layer"
    if isinstance(intent, NudgeSelectedEvents):
        return "Nudge Events"
    if isinstance(intent, SnapEventsToBeatGrid):
        return f"Snap Events to 1/{intent.grid_denominator} Beat Grid"
    if isinstance(intent, ReplaceSectionCues):
        return "Edit Sections"
    if isinstance(intent, SetGain):
        return "Adjust Gain"
    if isinstance(intent, TrimEvent):
        return "Trim Event"
    if isinstance(intent, UpdateEventLabel):
        return "Relabel Event"
    if isinstance(intent, TriggerTakeAction):
        return {
            "overwrite_main": "Overwrite Main",
            "promote_take": "Promote Take",
            "merge_main": "Merge Into Main",
            "add_selection_to_main": "Add Selection To Main",
            "delete_take": "Delete Take",
        }.get(_normalized_take_action_id(intent.action_id))
    return None


def capture_history_snapshot(
    shell: HistoryShell,
    *,
    layer_ids: list[LayerId] | tuple[LayerId, ...] | None = None,
) -> RuntimeHistorySnapshot | RuntimeScopedHistorySnapshot:
    if layer_ids:
        target_ids = [LayerId(str(layer_id)) for layer_id in layer_ids]
        layers_by_id = {layer.id: layer for layer in shell._app.timeline.layers}
        return RuntimeScopedHistorySnapshot(
            layers={
                layer_id: deepcopy(layers_by_id.get(layer_id))
                for layer_id in dict.fromkeys(target_ids)
            },
            selected_layer_id=deepcopy(shell._app.timeline.selection.selected_layer_id),
            selected_layer_ids=deepcopy(shell._app.timeline.selection.selected_layer_ids),
            selected_take_id=deepcopy(shell._app.timeline.selection.selected_take_id),
            selected_event_refs=deepcopy(shell._app.timeline.selection.selected_event_refs),
            selected_event_ids=deepcopy(shell._app.timeline.selection.selected_event_ids),
            section_cues=deepcopy(shell._app.timeline.section_cues),
            is_dirty=shell._is_dirty or shell.project_storage.is_dirty(),
        )
    return RuntimeHistorySnapshot(
        timeline=deepcopy(shell._app.timeline),
        active_song_id=deepcopy(shell.session.active_song_id),
        active_song_version_id=deepcopy(shell.session.active_song_version_id),
        active_timeline_id=deepcopy(shell.session.active_timeline_id),
        mixer_state=deepcopy(shell.session.mixer_state),
        draft_layers=deepcopy(shell._draft_layers),
        is_dirty=shell._is_dirty or shell.project_storage.is_dirty(),
    )


def restore_history_snapshot(
    shell: HistoryShell,
    snapshot: RuntimeHistorySnapshot | RuntimeScopedHistorySnapshot,
    *,
    storage_backed: bool,
) -> None:
    if isinstance(snapshot, RuntimeScopedHistorySnapshot):
        _restore_scoped_history_snapshot(shell, snapshot, storage_backed=storage_backed)
        return
    shell._app.replace_timeline(deepcopy(snapshot.timeline))
    shell.session.active_song_id = deepcopy(snapshot.active_song_id)
    shell.session.active_song_version_id = deepcopy(snapshot.active_song_version_id)
    shell.session.active_timeline_id = deepcopy(snapshot.active_timeline_id)
    shell.session.mixer_state = deepcopy(snapshot.mixer_state)
    shell._draft_layers = deepcopy(snapshot.draft_layers)
    if storage_backed:
        shell._sync_storage_backed_timeline()
    if snapshot.is_dirty:
        shell._is_dirty = True
    else:
        shell._is_dirty = False
        shell.project_storage.dirty_tracker.clear()
    shell._sync_runtime_audio_from_presentation(shell.presentation())


def _restore_scoped_history_snapshot(
    shell: HistoryShell,
    snapshot: RuntimeScopedHistorySnapshot,
    *,
    storage_backed: bool,
) -> None:
    timeline = shell._app.timeline
    layers_by_id = {layer.id: layer for layer in timeline.layers}
    for layer_id, layer_snapshot in snapshot.layers.items():
        if layer_snapshot is None:
            timeline.layers = [layer for layer in timeline.layers if layer.id != layer_id]
            continue
        restored_layer = deepcopy(layer_snapshot)
        if layer_id in layers_by_id:
            timeline.layers = [
                restored_layer if layer.id == layer_id else layer
                for layer in timeline.layers
            ]
        else:
            timeline.layers.append(restored_layer)
    timeline.selection.selected_layer_id = deepcopy(snapshot.selected_layer_id)
    timeline.selection.selected_layer_ids = deepcopy(snapshot.selected_layer_ids)
    timeline.selection.selected_take_id = deepcopy(snapshot.selected_take_id)
    timeline.selection.selected_event_refs = deepcopy(snapshot.selected_event_refs)
    timeline.selection.selected_event_ids = deepcopy(snapshot.selected_event_ids)
    timeline.section_cues = deepcopy(snapshot.section_cues)
    if storage_backed:
        shell._sync_storage_backed_layers(list(snapshot.layers.keys()))
    if snapshot.is_dirty:
        shell._is_dirty = True
    else:
        shell._is_dirty = False
        shell.project_storage.dirty_tracker.clear()
    shell._sync_runtime_audio_from_presentation(shell.presentation())


def clear_history(shell: HistoryShell) -> None:
    shell._history.clear()


def undo(shell: HistoryShell) -> TimelinePresentation:
    entry = shell._history.undo()
    if entry is None:
        return shell.presentation()
    restore_history_snapshot(shell, entry.before, storage_backed=entry.storage_backed)
    return shell.presentation()


def redo(shell: HistoryShell) -> TimelinePresentation:
    entry = shell._history.redo()
    if entry is None:
        return shell.presentation()
    restore_history_snapshot(shell, entry.after, storage_backed=entry.storage_backed)
    return shell.presentation()


def run_undoable_operation(
    shell: HistoryShell,
    *,
    label: str,
    storage_backed: bool,
    mark_dirty: bool,
    operation: Callable[[], _T],
    defer_storage_sync: bool = False,
    storage_layer_ids: list[LayerId] | None = None,
    history_layer_ids: list[LayerId] | None = None,
) -> _T:
    before = capture_history_snapshot(shell, layer_ids=history_layer_ids)
    try:
        result = operation()
    except Exception:
        restore_history_snapshot(shell, before, storage_backed=storage_backed)
        raise
    if storage_backed and defer_storage_sync:
        if storage_layer_ids:
            shell._defer_storage_backed_layers_sync(storage_layer_ids)
        else:
            shell._defer_storage_backed_timeline_sync()
    elif storage_backed:
        if storage_layer_ids:
            shell._sync_storage_backed_layers(storage_layer_ids)
        else:
            shell._sync_storage_backed_timeline()
    after = capture_history_snapshot(shell, layer_ids=history_layer_ids)
    if before == after:
        return result
    if mark_dirty:
        shell._is_dirty = True
    shell._history.push(
        UndoHistoryEntry(
            label=label,
            before=before,
            after=after,
            storage_backed=storage_backed,
        )
    )
    return result


def _normalized_take_action_id(action_id: str) -> str:
    return str(action_id or "").strip().lower()

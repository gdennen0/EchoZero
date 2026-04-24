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
from echozero.application.shared.ids import SongId, SongVersionId, TimelineId
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.history import UndoHistory, UndoHistoryEntry
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
    SelectTake,
    SetGain,
    TriggerTakeAction,
    TrimEvent,
    UpdateRegion,
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

    def _sync_runtime_audio_from_presentation(self, presentation: TimelinePresentation) -> None: ...


def is_undoable_intent(intent: object) -> bool:
    if isinstance(intent, SelectTake):
        return True
    if isinstance(intent, SetLayerMA3Route):
        return True
    if isinstance(
        intent,
        (
            CreateEvent,
            CreateRegion,
            DeleteRegion,
            DeleteEvents,
            DuplicateSelectedEvents,
            MoveEvent,
            MoveSelectedEvents,
            NudgeSelectedEvents,
            SetGain,
            TrimEvent,
            UpdateRegion,
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
            CreateEvent,
            CreateRegion,
            DeleteRegion,
            DeleteEvents,
            DuplicateSelectedEvents,
            MoveEvent,
            MoveSelectedEvents,
            NudgeSelectedEvents,
            TrimEvent,
            UpdateRegion,
        ),
    )


def is_history_barrier_intent(intent: object) -> bool:
    return isinstance(intent, _HISTORY_BARRIER_INTENT_TYPES)


def history_label_for_intent(intent: object) -> str | None:
    if isinstance(intent, SelectTake):
        return "Switch Take"
    if isinstance(intent, SetLayerMA3Route):
        return "Route Layer To MA3"
    if isinstance(intent, CreateEvent):
        return "Create Event"
    if isinstance(intent, CreateRegion):
        return "Create Region"
    if isinstance(intent, UpdateRegion):
        return "Update Region"
    if isinstance(intent, DeleteRegion):
        return "Delete Region"
    if isinstance(intent, DeleteEvents):
        return "Delete Events"
    if isinstance(intent, DuplicateSelectedEvents):
        return "Duplicate Events"
    if isinstance(intent, MoveEvent):
        return "Move Event"
    if isinstance(intent, MoveSelectedEvents):
        return "Move Events"
    if isinstance(intent, NudgeSelectedEvents):
        return "Nudge Events"
    if isinstance(intent, SetGain):
        return "Adjust Gain"
    if isinstance(intent, TrimEvent):
        return "Trim Event"
    if isinstance(intent, TriggerTakeAction):
        return {
            "overwrite_main": "Overwrite Main",
            "promote_take": "Promote Take",
            "merge_main": "Merge Into Main",
            "add_selection_to_main": "Add Selection To Main",
            "delete_take": "Delete Take",
        }.get(_normalized_take_action_id(intent.action_id))
    return None


def capture_history_snapshot(shell: HistoryShell) -> RuntimeHistorySnapshot:
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
    snapshot: RuntimeHistorySnapshot,
    *,
    storage_backed: bool,
) -> None:
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
) -> _T:
    before = capture_history_snapshot(shell)
    try:
        result = operation()
    except Exception:
        restore_history_snapshot(shell, before, storage_backed=storage_backed)
        raise
    if storage_backed:
        shell._sync_storage_backed_timeline()
    after = capture_history_snapshot(shell)
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

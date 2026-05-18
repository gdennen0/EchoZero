"""Timeline command runtime metadata for nonblocking app-shell edits.
Exists to keep hot-path command classification out of Qt widgets and storage helpers.
Connects ordered timeline mutations to scoped history, storage, and playback side-effect lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from echozero.application.shared.ids import LayerId
from echozero.application.timeline.intents import (
    CommitBoundaryCorrectedEventReview,
    CommitMissedEventsReview,
    CommitMissedEventReview,
    CommitRejectedEventsReview,
    CommitRejectedEventReview,
    CommitRelabeledEventReview,
    CommitVerifiedEventsReview,
    CommitVerifiedEventReview,
    CreateEvent,
    DeleteEvents,
    DuplicateSelectedEvents,
    MoveEvent,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    PasteCopiedEvents,
    ReorderLayer,
    ReplaceSectionCues,
    SnapEventsToBeatGrid,
    SetGain,
    TimelineIntent,
    ToggleLayerExpanded,
    TriggerTakeAction,
    TrimEvent,
    UpdateEventLabel,
)
from echozero.application.timeline.ma3_push_intents import SetLayerMA3Route
from echozero.application.timeline.models import Timeline


class PlaybackImpact(StrEnum):
    """Playback side-effect lane requested by one committed timeline command."""

    NONE = "none"
    MIX = "mix"
    EVENT_SCHEDULE = "event_schedule"
    GRAPH = "graph"


@dataclass(frozen=True, slots=True)
class TimelineCommandResult:
    """App-shell command metadata after classifying one ordered mutation."""

    changed_layer_ids: tuple[LayerId, ...] = ()
    history_layer_ids: tuple[LayerId, ...] = ()
    storage_layer_ids: tuple[LayerId, ...] = ()
    playback_impact: PlaybackImpact = PlaybackImpact.NONE
    scoped_history: bool = False


class TimelineCommandRuntime:
    """Classify app-shell timeline commands without mutating canonical truth."""

    def prepare(self, timeline: Timeline, intent: TimelineIntent) -> TimelineCommandResult:
        changed_layer_ids = _changed_layer_ids_for_intent(timeline, intent)
        playback_impact = _playback_impact_for_intent(intent)
        scoped_history = bool(changed_layer_ids) and _supports_scoped_history(intent)
        return TimelineCommandResult(
            changed_layer_ids=changed_layer_ids,
            history_layer_ids=changed_layer_ids if scoped_history else (),
            storage_layer_ids=changed_layer_ids if _can_sync_storage_by_layer(intent) else (),
            playback_impact=playback_impact,
            scoped_history=scoped_history,
        )


def _changed_layer_ids_for_intent(
    timeline: Timeline,
    intent: TimelineIntent,
) -> tuple[LayerId, ...]:
    if isinstance(
        intent,
        (
            CommitVerifiedEventsReview,
            CommitRejectedEventsReview,
        ),
    ):
        return _unique_layer_ids(event_ref.layer_id for event_ref in intent.event_refs)
    if isinstance(intent, CommitMissedEventsReview):
        return _unique_layer_ids(entry.layer_id for entry in intent.intents)
    if isinstance(
        intent,
        (
            CommitMissedEventReview,
            CommitVerifiedEventReview,
            CommitRejectedEventReview,
            CommitRelabeledEventReview,
            CommitBoundaryCorrectedEventReview,
            CreateEvent,
        ),
    ):
        return (LayerId(str(intent.layer_id)),)
    if isinstance(intent, (MoveEvent, TrimEvent, UpdateEventLabel)):
        layer_id = getattr(intent, "layer_id", None)
        if layer_id is not None:
            return (LayerId(str(layer_id)),)
        event_id = getattr(intent, "event_id", None)
        return _unique_layer_ids(
            layer.id
            for layer in timeline.layers
            for take in layer.takes
            for event in take.events
            if event.id == event_id
        )
    if isinstance(intent, DeleteEvents):
        if intent.event_refs:
            return _unique_layer_ids(event_ref.layer_id for event_ref in intent.event_refs)
        event_ids = set(intent.event_ids)
        return _unique_layer_ids(
            layer.id
            for layer in timeline.layers
            for take in layer.takes
            for event in take.events
            if event.id in event_ids
        )
    if isinstance(intent, (MoveSelectedEvents, NudgeSelectedEvents, DuplicateSelectedEvents)):
        selected_refs = list(timeline.selection.selected_event_refs)
        layer_ids = [event_ref.layer_id for event_ref in selected_refs]
        target_layer_id = getattr(intent, "target_layer_id", None)
        if target_layer_id is not None:
            layer_ids.append(LayerId(str(target_layer_id)))
        return _unique_layer_ids(layer_ids)
    if isinstance(intent, SnapEventsToBeatGrid):
        return _changed_layer_ids_for_event_batch_scope(timeline, intent)
    if isinstance(intent, PasteCopiedEvents):
        if intent.target_layer_id is not None:
            return (LayerId(str(intent.target_layer_id)),)
        return _selected_layer_id(timeline)
    if isinstance(intent, SetGain):
        return (LayerId(str(intent.layer_id)),)
    if isinstance(intent, SetLayerMA3Route):
        return (LayerId(str(intent.layer_id)),)
    if isinstance(intent, ToggleLayerExpanded):
        return (LayerId(str(intent.layer_id)),)
    if isinstance(intent, TriggerTakeAction):
        return (LayerId(str(intent.layer_id)),)
    if isinstance(intent, (ReorderLayer, ReplaceSectionCues)):
        return ()
    return ()


def _playback_impact_for_intent(intent: TimelineIntent) -> PlaybackImpact:
    if isinstance(intent, SetGain):
        return PlaybackImpact.MIX
    if isinstance(
        intent,
        (
            CommitVerifiedEventReview,
            CommitVerifiedEventsReview,
            CommitRejectedEventReview,
            CommitRejectedEventsReview,
            MoveEvent,
            MoveSelectedEvents,
            NudgeSelectedEvents,
            SnapEventsToBeatGrid,
            TrimEvent,
        ),
    ):
        return PlaybackImpact.EVENT_SCHEDULE
    if isinstance(
        intent,
        (
            CommitMissedEventReview,
            CommitMissedEventsReview,
            CreateEvent,
            DeleteEvents,
            DuplicateSelectedEvents,
            PasteCopiedEvents,
            ReplaceSectionCues,
            ReorderLayer,
            TriggerTakeAction,
        ),
    ):
        return PlaybackImpact.GRAPH
    return PlaybackImpact.NONE


def _supports_scoped_history(intent: TimelineIntent) -> bool:
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
            NudgeSelectedEvents,
            PasteCopiedEvents,
            SetGain,
            SnapEventsToBeatGrid,
            TrimEvent,
            UpdateEventLabel,
        ),
    )


def _can_sync_storage_by_layer(intent: TimelineIntent) -> bool:
    return _supports_scoped_history(intent) and not isinstance(
        intent,
        (ReplaceSectionCues, ReorderLayer),
    )


def _selected_layer_id(timeline: Timeline) -> tuple[LayerId, ...]:
    selected_layer_id = timeline.selection.selected_layer_id
    if selected_layer_id is None:
        return ()
    return (LayerId(str(selected_layer_id)),)


def _changed_layer_ids_for_event_batch_scope(
    timeline: Timeline,
    intent: SnapEventsToBeatGrid,
) -> tuple[LayerId, ...]:
    scope = intent.scope
    if scope.mode in {"take", "layer_main"} and scope.layer_id is not None:
        return (LayerId(str(scope.layer_id)),)
    if scope.mode == "selected_layers_main":
        selected_layer_ids = list(timeline.selection.selected_layer_ids)
        if not selected_layer_ids and timeline.selection.selected_layer_id is not None:
            selected_layer_ids = [timeline.selection.selected_layer_id]
        return _unique_layer_ids(selected_layer_ids)
    selected_refs = list(timeline.selection.selected_event_refs)
    if selected_refs:
        return _unique_layer_ids(event_ref.layer_id for event_ref in selected_refs)
    event_ids = set(timeline.selection.selected_event_ids)
    return _unique_layer_ids(
        layer.id
        for layer in timeline.layers
        for take in layer.takes
        for event in take.events
        if event.id in event_ids
    )


def _unique_layer_ids(values) -> tuple[LayerId, ...]:
    output: list[LayerId] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        layer_id = LayerId(str(value))
        key = str(layer_id)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(layer_id)
    return tuple(output)


__all__ = [
    "PlaybackImpact",
    "TimelineCommandResult",
    "TimelineCommandRuntime",
]

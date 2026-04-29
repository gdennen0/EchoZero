"""Explicit timeline intents for the new EchoZero application layer."""

from dataclasses import dataclass

from echozero.application.session.models import (
    ManualPullEventOption,
    ManualPullTrackOption,
    ManualPushTrackOption,
)
from echozero.application.shared.cue_numbers import CueNumber, coerce_positive_cue_number
from echozero.application.shared.enums import SyncMode
from echozero.application.shared.ids import EventId, LayerId, RegionId, SectionCueId, TakeId
from echozero.application.shared.ranges import TimeRange
from echozero.application.sync.models import LiveSyncState, coerce_live_sync_state
from echozero.application.timeline.event_batch_scope import EventBatchScope
from echozero.application.timeline.models import EventRef


class TimelineIntent:
    """Marker base type for timeline intents."""


def _coerce_pull_import_mode(raw_mode: str, *, action_name: str) -> str:
    import_mode = (raw_mode or "").strip().lower()
    if import_mode not in {"new_take", "main"}:
        raise ValueError(
            f"{action_name} requires import_mode 'new_take' or 'main'"
        )
    return import_mode


def _is_valid_output_bus(value: str) -> bool:
    token = value.strip().lower()
    if not token.startswith("outputs_"):
        return False
    parts = token.split("_")
    if len(parts) != 3:
        return False
    if not parts[1].isdigit() or not parts[2].isdigit():
        return False
    start = int(parts[1])
    end = int(parts[2])
    return start >= 1 and end >= start


@dataclass(slots=True)
class SelectLayer(TimelineIntent):
    layer_id: LayerId | None
    mode: str = "replace"


@dataclass(slots=True)
class SelectAdjacentLayer(TimelineIntent):
    direction: int


@dataclass(slots=True)
class SelectTake(TimelineIntent):
    layer_id: LayerId
    take_id: TakeId | None


@dataclass(slots=True)
class SetActivePlaybackTarget(TimelineIntent):
    layer_id: LayerId | None
    take_id: TakeId | None = None


@dataclass(slots=True)
class SelectEvent(TimelineIntent):
    layer_id: LayerId
    take_id: TakeId | None
    event_id: EventId | None
    mode: str = "replace"


@dataclass(slots=True)
class SelectAdjacentEventInSelectedLayer(TimelineIntent):
    direction: int
    include_demoted: bool = False


@dataclass(slots=True)
class ClearSelection(TimelineIntent):
    pass


@dataclass(slots=True)
class SelectAllEvents(TimelineIntent):
    pass


@dataclass(slots=True)
class SetSelectedEvents(TimelineIntent):
    """Replace the selected event set while preserving batch-selection anchors."""

    event_ids: list[EventId]
    event_refs: list[EventRef] | None = None
    anchor_layer_id: LayerId | None = None
    anchor_take_id: TakeId | None = None
    selected_layer_ids: list[LayerId] | None = None


@dataclass(slots=True)
class SelectRegion(TimelineIntent):
    region_id: RegionId | None


@dataclass(slots=True)
class CreateRegion(TimelineIntent):
    time_range: TimeRange
    label: str = "Region"
    color: str | None = None
    kind: str = "custom"

    def __post_init__(self) -> None:
        self.label = (self.label or "").strip() or "Region"
        if self.color is not None:
            color = self.color.strip()
            self.color = color or None
        kind = (self.kind or "").strip().lower()
        self.kind = kind or "custom"


@dataclass(slots=True)
class UpdateRegion(TimelineIntent):
    region_id: RegionId
    time_range: TimeRange
    label: str
    color: str | None = None
    kind: str = "custom"

    def __post_init__(self) -> None:
        if self.region_id is None or not str(self.region_id).strip():
            raise ValueError("UpdateRegion requires a non-empty region_id")
        self.label = (self.label or "").strip() or "Region"
        if self.color is not None:
            color = self.color.strip()
            self.color = color or None
        kind = (self.kind or "").strip().lower()
        self.kind = kind or "custom"


@dataclass(slots=True)
class DeleteRegion(TimelineIntent):
    region_id: RegionId

    def __post_init__(self) -> None:
        if self.region_id is None or not str(self.region_id).strip():
            raise ValueError("DeleteRegion requires a non-empty region_id")


@dataclass(slots=True)
class SelectEveryOtherEvents(TimelineIntent):
    """Replace the current selection with every other event inside one resolved scope."""

    scope: EventBatchScope


@dataclass(slots=True)
class RenumberEventCueNumbers(TimelineIntent):
    """Renumber cue numbers across one resolved event batch scope."""

    scope: EventBatchScope
    start_at: int = 1
    step: int = 1

    def __post_init__(self) -> None:
        try:
            start_at = int(self.start_at)
            step = int(self.step)
        except (TypeError, ValueError) as exc:
            raise ValueError("RenumberEventCueNumbers requires integer start_at and step") from exc
        if start_at < 1:
            raise ValueError(f"RenumberEventCueNumbers start_at must be >= 1, got {start_at}")
        if step < 1:
            raise ValueError(f"RenumberEventCueNumbers step must be >= 1, got {step}")
        self.start_at = start_at
        self.step = step


@dataclass(slots=True)
class ToggleLayerExpanded(TimelineIntent):
    layer_id: LayerId


@dataclass(slots=True)
class TriggerTakeAction(TimelineIntent):
    layer_id: LayerId
    take_id: TakeId
    action_id: str


@dataclass(slots=True)
class AddSongFromPath(TimelineIntent):
    title: str
    audio_path: str

    def __post_init__(self) -> None:
        title = (self.title or "").strip()
        audio_path = (self.audio_path or "").strip()
        if not title:
            raise ValueError("AddSongFromPath requires a non-empty title")
        if not audio_path:
            raise ValueError("AddSongFromPath requires a non-empty audio_path")
        self.title = title
        self.audio_path = audio_path


@dataclass(slots=True)
class ExtractStems(TimelineIntent):
    layer_id: LayerId

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("ExtractStems requires a non-empty layer_id")


@dataclass(slots=True)
class ExtractDrumEvents(TimelineIntent):
    layer_id: LayerId

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("ExtractDrumEvents requires a non-empty layer_id")


@dataclass(slots=True)
class CreateEvent(TimelineIntent):
    """Create one event in the target lane and select it immediately."""

    layer_id: LayerId
    time_range: TimeRange
    take_id: TakeId | None = None
    label: str = "Event"
    cue_number: CueNumber = 1
    source_event_id: str | None = None
    payload_ref: str | None = None
    color: str | None = None

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("CreateEvent requires a non-empty layer_id")
        label = (self.label or "").strip()
        self.label = label or "Event"
        try:
            cue_number = coerce_positive_cue_number(self.cue_number)
        except ValueError as exc:
            raise ValueError(
                f"CreateEvent requires positive numeric cue_number, got {self.cue_number!r}"
            ) from exc
        self.cue_number = cue_number
        if self.source_event_id is not None:
            source_event_id = str(self.source_event_id).strip()
            self.source_event_id = source_event_id or None
        if self.payload_ref is not None:
            payload_ref = str(self.payload_ref).strip()
            self.payload_ref = payload_ref or None
        if self.color is not None:
            color = str(self.color).strip()
            self.color = color or None


@dataclass(slots=True)
class CommitMissedEventReview(TimelineIntent):
    """Create one missing event and commit the timeline fix as a review signal."""

    layer_id: LayerId
    time_range: TimeRange
    take_id: TakeId | None = None
    label: str = "Event"
    cue_number: CueNumber = 1
    source_event_id: str | None = None
    payload_ref: str | None = None
    color: str | None = None
    review_note: str | None = None

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("CommitMissedEventReview requires a non-empty layer_id")
        label = (self.label or "").strip()
        self.label = label or "Event"
        try:
            cue_number = coerce_positive_cue_number(self.cue_number)
        except ValueError as exc:
            raise ValueError(
                "CommitMissedEventReview requires positive numeric cue_number, "
                f"got {self.cue_number!r}"
            ) from exc
        self.cue_number = cue_number
        if self.source_event_id is not None:
            source_event_id = str(self.source_event_id).strip()
            self.source_event_id = source_event_id or None
        if self.payload_ref is not None:
            payload_ref = str(self.payload_ref).strip()
            self.payload_ref = payload_ref or None
        if self.color is not None:
            color = str(self.color).strip()
            self.color = color or None
        if self.review_note is not None:
            review_note = str(self.review_note).strip()
            self.review_note = review_note or None


@dataclass(slots=True)
class CommitVerifiedEventReview(TimelineIntent):
    """Commit one explicit verified event review signal."""

    layer_id: LayerId
    event_id: EventId
    take_id: TakeId | None = None
    review_note: str | None = None

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("CommitVerifiedEventReview requires a non-empty layer_id")
        if self.event_id is None or not str(self.event_id).strip():
            raise ValueError("CommitVerifiedEventReview requires a non-empty event_id")
        if self.take_id is not None:
            take_id = str(self.take_id).strip()
            self.take_id = TakeId(take_id) if take_id else None
        if self.review_note is not None:
            review_note = str(self.review_note).strip()
            self.review_note = review_note or None


@dataclass(slots=True)
class CommitRejectedEventReview(TimelineIntent):
    """Demote one false-positive event and commit the rejection as review signal."""

    layer_id: LayerId
    event_id: EventId
    take_id: TakeId | None = None
    review_note: str | None = None

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("CommitRejectedEventReview requires a non-empty layer_id")
        if self.event_id is None or not str(self.event_id).strip():
            raise ValueError("CommitRejectedEventReview requires a non-empty event_id")
        if self.take_id is not None:
            take_id = str(self.take_id).strip()
            self.take_id = TakeId(take_id) if take_id else None
        if self.review_note is not None:
            review_note = str(self.review_note).strip()
            self.review_note = review_note or None


@dataclass(slots=True)
class CommitRelabeledEventReview(TimelineIntent):
    """Relabel one existing event and commit the correction as review signal."""

    layer_id: LayerId
    event_id: EventId
    corrected_label: str
    take_id: TakeId | None = None
    review_note: str | None = None

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("CommitRelabeledEventReview requires a non-empty layer_id")
        if self.event_id is None or not str(self.event_id).strip():
            raise ValueError("CommitRelabeledEventReview requires a non-empty event_id")
        corrected_label = (self.corrected_label or "").strip()
        if not corrected_label:
            raise ValueError("CommitRelabeledEventReview requires a non-empty corrected_label")
        self.corrected_label = corrected_label
        if self.take_id is not None:
            take_id = str(self.take_id).strip()
            self.take_id = TakeId(take_id) if take_id else None
        if self.review_note is not None:
            review_note = str(self.review_note).strip()
            self.review_note = review_note or None


@dataclass(slots=True)
class CommitBoundaryCorrectedEventReview(TimelineIntent):
    """Trim one event boundary and commit the correction as review signal."""

    layer_id: LayerId
    event_id: EventId
    corrected_range: TimeRange
    take_id: TakeId | None = None
    review_note: str | None = None

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("CommitBoundaryCorrectedEventReview requires a non-empty layer_id")
        if self.event_id is None or not str(self.event_id).strip():
            raise ValueError("CommitBoundaryCorrectedEventReview requires a non-empty event_id")
        if self.take_id is not None:
            take_id = str(self.take_id).strip()
            self.take_id = TakeId(take_id) if take_id else None
        if self.review_note is not None:
            review_note = str(self.review_note).strip()
            self.review_note = review_note or None


@dataclass(slots=True)
class CommitMissedEventsReview(TimelineIntent):
    """Commit multiple missed-event review fixes in one undoable app operation."""

    intents: list[CommitMissedEventReview]

    def __post_init__(self) -> None:
        normalized: list[CommitMissedEventReview] = []
        for intent in list(self.intents):
            if not isinstance(intent, CommitMissedEventReview):
                raise ValueError(
                    "CommitMissedEventsReview requires CommitMissedEventReview entries"
                )
            normalized.append(intent)
        if not normalized:
            raise ValueError("CommitMissedEventsReview requires at least one event intent")
        self.intents = normalized


@dataclass(slots=True)
class CommitVerifiedEventsReview(TimelineIntent):
    """Commit multiple explicit verified-event review signals in one app operation."""

    event_refs: list[EventRef]
    review_note: str | None = None

    def __post_init__(self) -> None:
        if self.review_note is not None:
            review_note = str(self.review_note).strip()
            self.review_note = review_note or None
        self.event_refs = _normalized_review_event_refs(
            self.event_refs,
            action_name="CommitVerifiedEventsReview",
        )


@dataclass(slots=True)
class CommitRejectedEventsReview(TimelineIntent):
    """Commit multiple rejected-event review signals in one app operation."""

    event_refs: list[EventRef]
    review_note: str | None = None

    def __post_init__(self) -> None:
        if self.review_note is not None:
            review_note = str(self.review_note).strip()
            self.review_note = review_note or None
        self.event_refs = _normalized_review_event_refs(
            self.event_refs,
            action_name="CommitRejectedEventsReview",
        )


def _normalized_review_event_refs(
    event_refs: list[EventRef],
    *,
    action_name: str,
) -> list[EventRef]:
    normalized: list[EventRef] = []
    seen: set[tuple[str, str, str]] = set()
    for event_ref in list(event_refs):
        layer_id = str(event_ref.layer_id or "").strip()
        event_id = str(event_ref.event_id or "").strip()
        if not layer_id:
            raise ValueError(f"{action_name} requires each event_ref to include layer_id")
        if not event_id:
            raise ValueError(f"{action_name} requires each event_ref to include event_id")
        take_id = (
            None
            if event_ref.take_id is None
            else (TakeId(str(event_ref.take_id).strip()) if str(event_ref.take_id).strip() else None)
        )
        key = (layer_id, str(take_id or ""), event_id)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            EventRef(
                layer_id=LayerId(layer_id),
                take_id=take_id,
                event_id=EventId(event_id),
            )
        )
    if not normalized:
        raise ValueError(f"{action_name} requires at least one event_ref")
    return normalized


@dataclass(slots=True)
class DeleteEvents(TimelineIntent):
    """Delete one or more events by id from the timeline."""

    event_ids: list[EventId]
    event_refs: list[EventRef] | None = None


@dataclass(slots=True)
class MoveEvent(TimelineIntent):
    event_id: EventId
    new_start: float


@dataclass(slots=True)
class MoveSelectedEvents(TimelineIntent):
    delta_seconds: float
    target_layer_id: LayerId | None = None
    copy_selected: bool = False


@dataclass(slots=True)
class MoveSelectedEventsToAdjacentLayer(TimelineIntent):
    direction: int


@dataclass(slots=True)
class ReorderLayer(TimelineIntent):
    source_layer_id: LayerId
    target_after_layer_id: LayerId | None
    insert_at_start: bool = False


@dataclass(slots=True)
class TrimEvent(TimelineIntent):
    event_id: EventId
    new_range: TimeRange


@dataclass(slots=True)
class UpdateEventLabel(TimelineIntent):
    event_id: EventId
    label: str
    layer_id: LayerId | None = None
    take_id: TakeId | None = None

    def __post_init__(self) -> None:
        if self.event_id is None or not str(self.event_id).strip():
            raise ValueError("UpdateEventLabel requires a non-empty event_id")
        label = (self.label or "").strip()
        if not label:
            raise ValueError("UpdateEventLabel requires a non-empty label")
        self.label = label
        if self.layer_id is not None and not str(self.layer_id).strip():
            self.layer_id = None
        if self.take_id is not None:
            take_id = str(self.take_id).strip()
            self.take_id = TakeId(take_id) if take_id else None


@dataclass(slots=True)
class SectionCueEdit:
    cue_id: SectionCueId | None
    start: float
    cue_ref: str
    name: str
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None

    def __post_init__(self) -> None:
        if self.cue_id is not None and not str(self.cue_id).strip():
            self.cue_id = None
        self.start = max(0.0, float(self.start))
        cue_ref = str(self.cue_ref or "").strip()
        if not cue_ref:
            raise ValueError("SectionCueEdit requires a non-empty cue_ref")
        self.cue_ref = cue_ref
        name = str(self.name or "").strip()
        self.name = name or cue_ref
        if self.color is not None:
            color = str(self.color).strip()
            self.color = color or None
        if self.notes is not None:
            notes = str(self.notes).strip()
            self.notes = notes or None
        if self.payload_ref is not None:
            payload_ref = str(self.payload_ref).strip()
            self.payload_ref = payload_ref or None


@dataclass(slots=True)
class ReplaceSectionCues(TimelineIntent):
    cues: list[SectionCueEdit]


@dataclass(slots=True)
class NudgeSelectedEvents(TimelineIntent):
    direction: int
    steps: int = 1


@dataclass(slots=True)
class DuplicateSelectedEvents(TimelineIntent):
    steps: int = 1


@dataclass(slots=True)
class Play(TimelineIntent):
    pass


@dataclass(slots=True)
class Pause(TimelineIntent):
    pass


@dataclass(slots=True)
class Stop(TimelineIntent):
    pass


@dataclass(slots=True)
class Seek(TimelineIntent):
    position: float


@dataclass(slots=True)
class SetFollowCursorEnabled(TimelineIntent):
    enabled: bool

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)


@dataclass(slots=True)
class SetGain(TimelineIntent):
    layer_id: LayerId
    gain_db: float


@dataclass(slots=True)
class SetLayerOutputBus(TimelineIntent):
    layer_id: LayerId
    output_bus: str | None

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("SetLayerOutputBus requires a non-empty layer_id")
        if self.output_bus is None:
            return
        output_bus = str(self.output_bus).strip()
        if not output_bus:
            self.output_bus = None
            return
        if not _is_valid_output_bus(output_bus):
            raise ValueError(
                "SetLayerOutputBus requires output_bus format 'outputs_<start>_<end>'"
            )
        self.output_bus = output_bus


@dataclass(slots=True)
class EnableSync(TimelineIntent):
    mode: SyncMode


@dataclass(slots=True)
class DisableSync(TimelineIntent):
    pass


@dataclass(slots=True)
class EnableExperimentalLiveSync(TimelineIntent):
    pass


@dataclass(slots=True)
class DisableExperimentalLiveSync(TimelineIntent):
    pass


@dataclass(slots=True)
class SetLayerLiveSyncState(TimelineIntent):
    layer_id: LayerId
    live_sync_state: LiveSyncState | str

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("SetLayerLiveSyncState requires a non-empty layer_id")
        self.live_sync_state = coerce_live_sync_state(self.live_sync_state)


@dataclass(slots=True)
class SetLayerLiveSyncPauseReason(TimelineIntent):
    layer_id: LayerId
    pause_reason: str

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("SetLayerLiveSyncPauseReason requires a non-empty layer_id")
        reason = self.pause_reason.strip()
        if not reason:
            raise ValueError("SetLayerLiveSyncPauseReason requires a non-empty pause_reason")
        self.pause_reason = reason


@dataclass(slots=True)
class ClearLayerLiveSyncPauseReason(TimelineIntent):
    layer_id: LayerId

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("ClearLayerLiveSyncPauseReason requires a non-empty layer_id")


@dataclass(slots=True)
class OpenPushToMA3Dialog(TimelineIntent):
    selection_event_ids: list[EventId]


@dataclass(slots=True)
class SetPushTrackOptions(TimelineIntent):
    tracks: list[ManualPushTrackOption]


@dataclass(slots=True)
class SelectPushTargetTrack(TimelineIntent):
    target_track_coord: str
    layer_id: LayerId | None = None

    def __post_init__(self) -> None:
        if not self.target_track_coord or not self.target_track_coord.strip():
            raise ValueError("SelectPushTargetTrack requires a non-empty target_track_coord")


@dataclass(slots=True)
class SetPushTransferMode(TimelineIntent):
    mode: str

    def __post_init__(self) -> None:
        mode = (self.mode or "").strip().lower()
        if mode not in {"merge", "overwrite"}:
            raise ValueError("SetPushTransferMode requires mode 'merge' or 'overwrite'")
        self.mode = mode


@dataclass(slots=True)
class ExitPushToMA3Mode(TimelineIntent):
    pass


@dataclass(slots=True)
class ConfirmPushToMA3(TimelineIntent):
    target_track_coord: str
    selected_event_ids: list[EventId]

    def __post_init__(self) -> None:
        if not self.target_track_coord or not self.target_track_coord.strip():
            raise ValueError("ConfirmPushToMA3 requires a non-empty target_track_coord")
        if not self.selected_event_ids:
            raise ValueError("ConfirmPushToMA3 requires at least one selected_event_id")


@dataclass(slots=True)
class OpenPullFromMA3Dialog(TimelineIntent):
    pass


@dataclass(slots=True)
class ExitPullFromMA3Workspace(TimelineIntent):
    pass


@dataclass(slots=True)
class SelectPullSourceTimecode(TimelineIntent):
    timecode_no: int

    def __post_init__(self) -> None:
        try:
            timecode_no = int(self.timecode_no)
        except (TypeError, ValueError) as exc:
            raise ValueError("SelectPullSourceTimecode requires an integer timecode_no") from exc
        if timecode_no < 1:
            raise ValueError("SelectPullSourceTimecode requires timecode_no >= 1")
        self.timecode_no = timecode_no


@dataclass(slots=True)
class SelectPullSourceTrackGroup(TimelineIntent):
    track_group_no: int

    def __post_init__(self) -> None:
        try:
            track_group_no = int(self.track_group_no)
        except (TypeError, ValueError) as exc:
            raise ValueError("SelectPullSourceTrackGroup requires an integer track_group_no") from exc
        if track_group_no < 1:
            raise ValueError("SelectPullSourceTrackGroup requires track_group_no >= 1")
        self.track_group_no = track_group_no


@dataclass(slots=True)
class SelectPullSourceTracks(TimelineIntent):
    source_track_coords: list[str]

    def __post_init__(self) -> None:
        if not self.source_track_coords:
            raise ValueError("SelectPullSourceTracks requires at least one source_track_coord")
        if not all(coord and str(coord).strip() for coord in self.source_track_coords):
            raise ValueError("SelectPullSourceTracks requires non-empty source_track_coords")


@dataclass(slots=True)
class SetPullTrackOptions(TimelineIntent):
    tracks: list[ManualPullTrackOption]


@dataclass(slots=True)
class SelectPullSourceTrack(TimelineIntent):
    source_track_coord: str

    def __post_init__(self) -> None:
        if not self.source_track_coord or not self.source_track_coord.strip():
            raise ValueError("SelectPullSourceTrack requires a non-empty source_track_coord")


@dataclass(slots=True)
class SetPullSourceEvents(TimelineIntent):
    events: list[ManualPullEventOption]


@dataclass(slots=True)
class SelectPullSourceEvents(TimelineIntent):
    selected_ma3_event_ids: list[str]

    def __post_init__(self) -> None:
        if not self.selected_ma3_event_ids:
            raise ValueError("SelectPullSourceEvents requires at least one selected_ma3_event_id")
        if not all(event_id and str(event_id).strip() for event_id in self.selected_ma3_event_ids):
            raise ValueError("SelectPullSourceEvents requires non-empty selected_ma3_event_ids")


@dataclass(slots=True)
class SelectPullTargetLayer(TimelineIntent):
    target_layer_id: LayerId

    def __post_init__(self) -> None:
        if self.target_layer_id is None or not str(self.target_layer_id).strip():
            raise ValueError("SelectPullTargetLayer requires a non-empty target_layer_id")


@dataclass(slots=True)
class SetPullImportMode(TimelineIntent):
    import_mode: str

    def __post_init__(self) -> None:
        self.import_mode = _coerce_pull_import_mode(
            self.import_mode,
            action_name="SetPullImportMode",
        )


@dataclass(slots=True)
class ConfirmPullFromMA3(TimelineIntent):
    source_track_coord: str
    selected_ma3_event_ids: list[str]
    target_layer_id: LayerId
    import_mode: str = "new_take"

    def __post_init__(self) -> None:
        if not self.source_track_coord or not self.source_track_coord.strip():
            raise ValueError("ConfirmPullFromMA3 requires a non-empty source_track_coord")
        if not self.selected_ma3_event_ids:
            raise ValueError("ConfirmPullFromMA3 requires at least one selected_ma3_event_id")
        if not all(event_id and str(event_id).strip() for event_id in self.selected_ma3_event_ids):
            raise ValueError("ConfirmPullFromMA3 requires non-empty selected_ma3_event_ids")
        if self.target_layer_id is None or not str(self.target_layer_id).strip():
            raise ValueError("ConfirmPullFromMA3 requires a non-empty target_layer_id")
        self.import_mode = _coerce_pull_import_mode(
            self.import_mode,
            action_name="ConfirmPullFromMA3",
        )


@dataclass(slots=True)
class ApplyPullFromMA3(TimelineIntent):
    pass


@dataclass(slots=True)
class SaveTransferPreset(TimelineIntent):
    name: str

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("SaveTransferPreset requires a non-empty name")
        self.name = name


@dataclass(slots=True)
class ApplyTransferPreset(TimelineIntent):
    preset_id: str

    def __post_init__(self) -> None:
        preset_id = self.preset_id.strip()
        if not preset_id:
            raise ValueError("ApplyTransferPreset requires a non-empty preset_id")
        self.preset_id = preset_id


@dataclass(slots=True)
class DeleteTransferPreset(TimelineIntent):
    preset_id: str

    def __post_init__(self) -> None:
        preset_id = self.preset_id.strip()
        if not preset_id:
            raise ValueError("DeleteTransferPreset requires a non-empty preset_id")
        self.preset_id = preset_id


@dataclass(slots=True)
class PreviewTransferPlan(TimelineIntent):
    plan_id: str

    def __post_init__(self) -> None:
        plan_id = self.plan_id.strip()
        if not plan_id:
            raise ValueError("PreviewTransferPlan requires a non-empty plan_id")
        self.plan_id = plan_id


@dataclass(slots=True)
class ApplyTransferPlan(TimelineIntent):
    plan_id: str

    def __post_init__(self) -> None:
        plan_id = self.plan_id.strip()
        if not plan_id:
            raise ValueError("ApplyTransferPlan requires a non-empty plan_id")
        self.plan_id = plan_id


@dataclass(slots=True)
class CancelTransferPlan(TimelineIntent):
    plan_id: str

    def __post_init__(self) -> None:
        plan_id = self.plan_id.strip()
        if not plan_id:
            raise ValueError("CancelTransferPlan requires a non-empty plan_id")
        self.plan_id = plan_id

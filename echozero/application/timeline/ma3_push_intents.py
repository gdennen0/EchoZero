"""Typed MA3 push intents for the operator-first timeline workflow.
Exists to keep layer routing and send actions out of the older batch transfer path.
Connects layer-local MA3 UI actions to the canonical timeline application contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from echozero.application.shared.ids import EventId, LayerId
from echozero.application.timeline.intents import TimelineIntent


class MA3PushScope(str, Enum):
    """Defines which main-take events are included in one MA3 push."""

    LAYER_MAIN = "layer_main"
    SELECTED_EVENTS = "selected_events"


class MA3PushTargetMode(str, Enum):
    """Defines whether a push uses the saved layer route or a one-shot target."""

    SAVED_ROUTE = "saved_route"
    DIFFERENT_TRACK_ONCE = "different_track_once"


class MA3PushApplyMode(str, Enum):
    """Defines how the selected MA3 target track should absorb incoming events."""

    MERGE = "merge"
    OVERWRITE = "overwrite"


class MA3SequenceRefreshRangeMode(str, Enum):
    """Defines which MA3 sequence range should be listed for the operator."""

    ALL = "all"
    CURRENT_SONG = "current_song"


class MA3SequenceCreationMode(str, Enum):
    """Defines how a new MA3 sequence should be created."""

    NEXT_AVAILABLE = "next_available"
    CURRENT_SONG_RANGE = "current_song_range"


def _coerce_ma3_push_scope(raw_scope: MA3PushScope | str) -> MA3PushScope:
    if isinstance(raw_scope, MA3PushScope):
        return raw_scope
    try:
        return MA3PushScope(str(raw_scope).strip().lower())
    except ValueError as exc:
        raise ValueError("PushLayerToMA3 requires scope 'layer_main' or 'selected_events'") from exc


def _coerce_ma3_push_target_mode(raw_mode: MA3PushTargetMode | str) -> MA3PushTargetMode:
    if isinstance(raw_mode, MA3PushTargetMode):
        return raw_mode
    try:
        return MA3PushTargetMode(str(raw_mode).strip().lower())
    except ValueError as exc:
        raise ValueError(
            "PushLayerToMA3 requires target_mode 'saved_route' or 'different_track_once'"
        ) from exc


def _coerce_ma3_push_apply_mode(raw_mode: MA3PushApplyMode | str) -> MA3PushApplyMode:
    if isinstance(raw_mode, MA3PushApplyMode):
        return raw_mode
    try:
        return MA3PushApplyMode(str(raw_mode).strip().lower())
    except ValueError as exc:
        raise ValueError("PushLayerToMA3 requires apply_mode 'merge' or 'overwrite'") from exc


def _coerce_ma3_sequence_refresh_range_mode(
    raw_mode: MA3SequenceRefreshRangeMode | str,
) -> MA3SequenceRefreshRangeMode:
    if isinstance(raw_mode, MA3SequenceRefreshRangeMode):
        return raw_mode
    try:
        return MA3SequenceRefreshRangeMode(str(raw_mode).strip().lower())
    except ValueError as exc:
        raise ValueError("RefreshMA3Sequences requires range_mode 'all' or 'current_song'") from exc


def _coerce_ma3_sequence_creation_mode(
    raw_mode: MA3SequenceCreationMode | str,
) -> MA3SequenceCreationMode:
    if isinstance(raw_mode, MA3SequenceCreationMode):
        return raw_mode
    try:
        return MA3SequenceCreationMode(str(raw_mode).strip().lower())
    except ValueError as exc:
        raise ValueError(
            "CreateMA3Sequence requires creation_mode 'next_available' or 'current_song_range'"
        ) from exc


def _coerce_sequence_action(
    raw_action: "MA3TrackSequenceAction | None",
) -> "MA3TrackSequenceAction | None":
    if raw_action is None or isinstance(raw_action, (AssignMA3TrackSequence, CreateMA3Sequence)):
        return raw_action
    raise ValueError(
        "MA3 sequence_action must be AssignMA3TrackSequence or CreateMA3Sequence when provided"
    )


def _coerce_optional_positive_int(
    raw_value: int | str | None,
    *,
    action_name: str,
    field_name: str,
) -> int | None:
    if raw_value in {None, ""}:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{action_name} requires integer {field_name}") from exc
    if value < 1:
        raise ValueError(f"{action_name} requires {field_name} >= 1")
    return value


def _coerce_required_positive_int(
    raw_value: int | str | None,
    *,
    action_name: str,
    field_name: str,
) -> int:
    value = _coerce_optional_positive_int(
        raw_value,
        action_name=action_name,
        field_name=field_name,
    )
    if value is None:
        raise ValueError(f"{action_name} requires integer {field_name}")
    return value


@dataclass(slots=True)
class RefreshMA3PushTracks(TimelineIntent):
    """Refresh available MA3 target tracks from the sync provider."""

    target_track_coord: str | None = None
    timecode_no: int | None = None
    track_group_no: int | None = None

    def __post_init__(self) -> None:
        target_track_coord = (
            None
            if self.target_track_coord is None
            else str(self.target_track_coord).strip()
        )
        self.target_track_coord = target_track_coord or None
        self.timecode_no = _coerce_optional_positive_int(
            self.timecode_no,
            action_name="RefreshMA3PushTracks",
            field_name="timecode_no",
        )
        self.track_group_no = _coerce_optional_positive_int(
            self.track_group_no,
            action_name="RefreshMA3PushTracks",
            field_name="track_group_no",
        )
        if self.track_group_no is not None and self.timecode_no is None:
            raise ValueError(
                "RefreshMA3PushTracks requires timecode_no when track_group_no is provided"
            )


@dataclass(slots=True)
class RefreshMA3Sequences(TimelineIntent):
    """Refresh available MA3 sequences from the sync provider."""

    range_mode: MA3SequenceRefreshRangeMode | str = MA3SequenceRefreshRangeMode.ALL

    def __post_init__(self) -> None:
        self.range_mode = _coerce_ma3_sequence_refresh_range_mode(self.range_mode)


@dataclass(slots=True)
class AssignMA3TrackSequence(TimelineIntent):
    """Assign one existing MA3 sequence to one MA3 target track."""

    target_track_coord: str
    sequence_no: int

    def __post_init__(self) -> None:
        target_track_coord = str(self.target_track_coord or "").strip()
        if not target_track_coord:
            raise ValueError("AssignMA3TrackSequence requires a non-empty target_track_coord")
        try:
            sequence_no = int(self.sequence_no)
        except (TypeError, ValueError) as exc:
            raise ValueError("AssignMA3TrackSequence requires an integer sequence_no") from exc
        if sequence_no < 1:
            raise ValueError("AssignMA3TrackSequence requires sequence_no >= 1")
        self.target_track_coord = target_track_coord
        self.sequence_no = sequence_no


@dataclass(slots=True)
class CreateMA3Sequence(TimelineIntent):
    """Create one MA3 sequence using the requested creation policy."""

    creation_mode: MA3SequenceCreationMode | str = MA3SequenceCreationMode.NEXT_AVAILABLE
    preferred_name: str | None = None

    def __post_init__(self) -> None:
        self.creation_mode = _coerce_ma3_sequence_creation_mode(self.creation_mode)
        preferred_name = None if self.preferred_name is None else str(self.preferred_name).strip()
        self.preferred_name = preferred_name or None


@dataclass(slots=True)
class CreateMA3Timecode(TimelineIntent):
    """Create one MA3 timecode pool using the next available slot."""

    preferred_name: str | None = None

    def __post_init__(self) -> None:
        preferred_name = None if self.preferred_name is None else str(self.preferred_name).strip()
        self.preferred_name = preferred_name or None


@dataclass(slots=True)
class CreateMA3TrackGroup(TimelineIntent):
    """Create one MA3 track group in the selected timecode pool."""

    timecode_no: int
    preferred_name: str | None = None

    def __post_init__(self) -> None:
        self.timecode_no = _coerce_required_positive_int(
            self.timecode_no,
            action_name="CreateMA3TrackGroup",
            field_name="timecode_no",
        )
        preferred_name = None if self.preferred_name is None else str(self.preferred_name).strip()
        self.preferred_name = preferred_name or None


@dataclass(slots=True)
class CreateMA3Track(TimelineIntent):
    """Create one MA3 track in the selected timecode pool and track group."""

    timecode_no: int
    track_group_no: int
    preferred_name: str | None = None

    def __post_init__(self) -> None:
        self.timecode_no = _coerce_required_positive_int(
            self.timecode_no,
            action_name="CreateMA3Track",
            field_name="timecode_no",
        )
        self.track_group_no = _coerce_required_positive_int(
            self.track_group_no,
            action_name="CreateMA3Track",
            field_name="track_group_no",
        )
        preferred_name = None if self.preferred_name is None else str(self.preferred_name).strip()
        self.preferred_name = preferred_name or None


MA3TrackSequenceAction = AssignMA3TrackSequence | CreateMA3Sequence


@dataclass(slots=True)
class SetLayerMA3Route(TimelineIntent):
    """Persist the saved MA3 target track for one timeline layer."""

    layer_id: LayerId
    target_track_coord: str
    sequence_action: MA3TrackSequenceAction | None = None

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("SetLayerMA3Route requires a non-empty layer_id")
        target_track_coord = str(self.target_track_coord or "").strip()
        if not target_track_coord:
            raise ValueError("SetLayerMA3Route requires a non-empty target_track_coord")
        self.target_track_coord = target_track_coord
        self.sequence_action = _coerce_sequence_action(self.sequence_action)


@dataclass(slots=True)
class PrepareMA3TrackForPush(TimelineIntent):
    """Run MA3-side target-track preparation after sequence assignment."""

    target_track_coord: str

    def __post_init__(self) -> None:
        target_track_coord = str(self.target_track_coord or "").strip()
        if not target_track_coord:
            raise ValueError("PrepareMA3TrackForPush requires a non-empty target_track_coord")
        self.target_track_coord = target_track_coord


@dataclass(slots=True)
class PushLayerToMA3(TimelineIntent):
    """Push main-take events from one layer to one MA3 track."""

    layer_id: LayerId
    scope: MA3PushScope | str = MA3PushScope.LAYER_MAIN
    target_mode: MA3PushTargetMode | str = MA3PushTargetMode.SAVED_ROUTE
    apply_mode: MA3PushApplyMode | str = MA3PushApplyMode.MERGE
    target_track_coord: str | None = None
    selected_event_ids: list[EventId] = field(default_factory=list)
    sequence_action: MA3TrackSequenceAction | None = None

    def __post_init__(self) -> None:
        if self.layer_id is None or not str(self.layer_id).strip():
            raise ValueError("PushLayerToMA3 requires a non-empty layer_id")

        self.scope = _coerce_ma3_push_scope(self.scope)
        self.target_mode = _coerce_ma3_push_target_mode(self.target_mode)
        self.apply_mode = _coerce_ma3_push_apply_mode(self.apply_mode)

        target_track_coord = str(self.target_track_coord or "").strip()
        self.target_track_coord = target_track_coord or None
        self.selected_event_ids = list(dict.fromkeys(self.selected_event_ids))
        self.sequence_action = _coerce_sequence_action(self.sequence_action)

        if self.scope is MA3PushScope.SELECTED_EVENTS and not self.selected_event_ids:
            raise ValueError("PushLayerToMA3 requires selected_event_ids for scope 'selected_events'")
        if (
            self.target_mode is MA3PushTargetMode.DIFFERENT_TRACK_ONCE
            and self.target_track_coord is None
        ):
            raise ValueError(
                "PushLayerToMA3 requires target_track_coord for target_mode 'different_track_once'"
            )


@dataclass(slots=True)
class PollMA3PushOperation(TimelineIntent):
    """Poll one in-flight MA3 push operation and refresh push-flow status when complete."""

    operation_id: str

    def __post_init__(self) -> None:
        operation_id = str(self.operation_id or "").strip()
        if not operation_id:
            raise ValueError("PollMA3PushOperation requires a non-empty operation_id")
        self.operation_id = operation_id


__all__ = [
    "AssignMA3TrackSequence",
    "CreateMA3Timecode",
    "CreateMA3Track",
    "CreateMA3TrackGroup",
    "CreateMA3Sequence",
    "MA3PushApplyMode",
    "MA3PushScope",
    "MA3PushTargetMode",
    "MA3SequenceCreationMode",
    "MA3SequenceRefreshRangeMode",
    "MA3TrackSequenceAction",
    "PollMA3PushOperation",
    "PrepareMA3TrackForPush",
    "PushLayerToMA3",
    "RefreshMA3Sequences",
    "RefreshMA3PushTracks",
    "SetLayerMA3Route",
]

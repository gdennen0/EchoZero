"""Explicit timeline intents for the new EchoZero application layer."""

from dataclasses import dataclass

from echozero.application.shared.ids import LayerId, TakeId, EventId
from echozero.application.shared.enums import SyncMode
from echozero.application.shared.ranges import TimeRange


class TimelineIntent:
    """Marker base type for timeline intents."""


@dataclass(slots=True)
class SelectLayer(TimelineIntent):
    layer_id: LayerId | None


@dataclass(slots=True)
class SelectTake(TimelineIntent):
    layer_id: LayerId
    take_id: TakeId | None


@dataclass(slots=True)
class SelectEvent(TimelineIntent):
    layer_id: LayerId
    take_id: TakeId | None
    event_id: EventId | None
    mode: str = "replace"


@dataclass(slots=True)
class ClearSelection(TimelineIntent):
    pass


@dataclass(slots=True)
class SelectAllEvents(TimelineIntent):
    pass


@dataclass(slots=True)
class ToggleLayerExpanded(TimelineIntent):
    layer_id: LayerId


@dataclass(slots=True)
class TriggerTakeAction(TimelineIntent):
    layer_id: LayerId
    take_id: TakeId
    action_id: str


@dataclass(slots=True)
class MoveEvent(TimelineIntent):
    event_id: EventId
    new_start: float


@dataclass(slots=True)
class MoveSelectedEvents(TimelineIntent):
    delta_seconds: float
    target_layer_id: LayerId | None = None


@dataclass(slots=True)
class TrimEvent(TimelineIntent):
    event_id: EventId
    new_range: TimeRange


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
class ToggleMute(TimelineIntent):
    layer_id: LayerId


@dataclass(slots=True)
class ToggleSolo(TimelineIntent):
    layer_id: LayerId


@dataclass(slots=True)
class SetGain(TimelineIntent):
    layer_id: LayerId
    gain_db: float


@dataclass(slots=True)
class EnableSync(TimelineIntent):
    mode: SyncMode


@dataclass(slots=True)
class DisableSync(TimelineIntent):
    pass


@dataclass(slots=True)
class OpenPushToMA3Dialog(TimelineIntent):
    selection_event_ids: list[EventId]


@dataclass(slots=True)
class ConfirmPushToMA3(TimelineIntent):
    target_track_coord: str
    selected_event_ids: list[EventId]


@dataclass(slots=True)
class OpenPullFromMA3Dialog(TimelineIntent):
    pass


@dataclass(slots=True)
class ConfirmPullFromMA3(TimelineIntent):
    source_track_coord: str
    selected_ma3_event_ids: list[str]
    target_layer_id: LayerId
    import_mode: str = "new_take"

    def __post_init__(self) -> None:
        if self.target_layer_id is None or not str(self.target_layer_id).strip():
            raise ValueError("ConfirmPullFromMA3 requires a non-empty target_layer_id")

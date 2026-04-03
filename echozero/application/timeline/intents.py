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
    event_id: EventId | None


@dataclass(slots=True)
class ToggleLayerExpanded(TimelineIntent):
    layer_id: LayerId


@dataclass(slots=True)
class ToggleTakeSelector(TimelineIntent):
    layer_id: LayerId


@dataclass(slots=True)
class MoveEvent(TimelineIntent):
    event_id: EventId
    new_start: float


@dataclass(slots=True)
class TrimEvent(TimelineIntent):
    event_id: EventId
    new_range: TimeRange


@dataclass(slots=True)
class Play(TimelineIntent):
    pass


@dataclass(slots=True)
class Pause(TimelineIntent):
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

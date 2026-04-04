"""Core timeline application models."""

from dataclasses import dataclass, field

from echozero.application.shared.ids import TimelineId, SongVersionId, LayerId, TakeId, EventId
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ranges import TimeRange
from echozero.application.mixer.models import LayerMixerState
from echozero.application.playback.models import LayerPlaybackState


@dataclass(slots=True)
class LayerSyncState:
    mode: str = "none"
    connected: bool = False
    offset_ms: float = 0.0
    target_ref: str | None = None
    show_manager_block_id: str | None = None
    ma3_track_coord: str | None = None
    derived_from_source: bool = False


@dataclass(slots=True)
class LayerPresentationHints:
    visible: bool = True
    locked: bool = False
    collapsed: bool = False
    height: float = 40.0
    color: str | None = None
    group_id: str | None = None
    group_name: str | None = None
    group_index: int | None = None
    show_take_selector: bool = True
    take_selector_expanded: bool = False
    show_take_lane: bool = False


@dataclass(slots=True)
class Event:
    id: EventId
    take_id: TakeId
    start: float
    end: float
    payload_ref: str | None = None
    label: str = "Event"
    color: str | None = None
    muted: bool = False

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Event.start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"Event.end must be >= start, got start={self.start}, end={self.end}"
            )

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class Take:
    id: TakeId
    layer_id: LayerId
    name: str
    version_label: str | None = None
    events: list[Event] = field(default_factory=list)
    source_ref: str | None = None
    available: bool = True
    is_comped: bool = False


@dataclass(slots=True)
class Layer:
    id: LayerId
    timeline_id: TimelineId
    name: str
    kind: LayerKind
    order_index: int
    takes: list[Take] = field(default_factory=list)
    mixer: LayerMixerState = field(default_factory=LayerMixerState)
    playback: LayerPlaybackState = field(default_factory=LayerPlaybackState)
    sync: LayerSyncState = field(default_factory=LayerSyncState)
    presentation_hints: LayerPresentationHints = field(default_factory=LayerPresentationHints)


@dataclass(slots=True)
class TimelineSelection:
    selected_layer_id: LayerId | None = None
    selected_take_id: TakeId | None = None
    selected_event_ids: list[EventId] = field(default_factory=list)


@dataclass(slots=True)
class TimelineViewport:
    pixels_per_second: float = 100.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0


@dataclass(slots=True)
class Timeline:
    id: TimelineId
    song_version_id: SongVersionId
    start: float = 0.0
    end: float = 0.0
    layers: list[Layer] = field(default_factory=list)
    loop_region: TimeRange | None = None
    selection: TimelineSelection = field(default_factory=TimelineSelection)
    viewport: TimelineViewport = field(default_factory=TimelineViewport)

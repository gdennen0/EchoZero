"""Simple UI-facing presentation models for the new EchoZero application layer."""

from dataclasses import dataclass, field

from echozero.application.shared.ids import TimelineId, LayerId, TakeId, EventId
from echozero.application.shared.enums import LayerKind, FollowMode, PlaybackMode, SyncMode
from echozero.application.shared.ranges import TimeRange


@dataclass(slots=True)
class TakeSummaryPresentation:
    total_take_count: int = 0
    active_take_id: TakeId | None = None
    active_take_name: str | None = None
    available_take_names: list[str] = field(default_factory=list)
    compact_label: str = ""
    can_expand: bool = False


@dataclass(slots=True)
class EventPresentation:
    event_id: EventId
    start: float
    end: float
    label: str
    color: str | None = None
    muted: bool = False
    is_selected: bool = False
    badges: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class LayerPresentation:
    layer_id: LayerId
    title: str
    subtitle: str = ""
    kind: LayerKind = LayerKind.EVENT
    is_selected: bool = False
    is_expanded: bool = False
    active_take_id: TakeId | None = None
    take_summary: TakeSummaryPresentation = field(default_factory=TakeSummaryPresentation)
    events: list[EventPresentation] = field(default_factory=list)
    visible: bool = True
    locked: bool = False
    muted: bool = False
    soloed: bool = False
    gain_db: float = 0.0
    pan: float = 0.0
    playback_mode: PlaybackMode = PlaybackMode.NONE
    playback_enabled: bool = False
    sync_mode: SyncMode = SyncMode.NONE
    sync_connected: bool = False
    color: str | None = None
    badges: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TimelinePresentation:
    timeline_id: TimelineId
    title: str
    layers: list[LayerPresentation] = field(default_factory=list)
    playhead: float = 0.0
    is_playing: bool = False
    loop_region: TimeRange | None = None
    follow_mode: FollowMode = FollowMode.PAGE
    selected_layer_id: LayerId | None = None
    selected_take_id: TakeId | None = None
    selected_event_ids: list[EventId] = field(default_factory=list)
    pixels_per_second: float = 100.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0

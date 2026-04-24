"""Core timeline application models."""

from dataclasses import dataclass, field

from echozero.application.shared.ids import (
    EventId,
    LayerId,
    RegionId,
    SongVersionId,
    TakeId,
    TimelineId,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ranges import TimeRange
from echozero.application.mixer.models import LayerMixerState
from echozero.application.playback.models import LayerPlaybackState
from echozero.application.sync.models import LiveSyncState, coerce_live_sync_state


@dataclass(slots=True)
class LayerSyncState:
    mode: str = "none"
    connected: bool = False
    offset_ms: float = 0.0
    target_ref: str | None = None
    show_manager_block_id: str | None = None
    ma3_track_coord: str | None = None
    derived_from_source: bool = False
    live_sync_state: LiveSyncState = LiveSyncState.OFF
    live_sync_pause_reason: str | None = None
    live_sync_divergent: bool = False

    def __post_init__(self) -> None:
        self.live_sync_state = coerce_live_sync_state(self.live_sync_state)
        if self.live_sync_pause_reason is not None:
            reason = self.live_sync_pause_reason.strip()
            self.live_sync_pause_reason = reason or None


@dataclass(slots=True)
class LayerPresentationHints:
    visible: bool = True
    locked: bool = False
    expanded: bool = False
    height: float = 40.0
    color: str | None = None
    group_id: str | None = None
    group_name: str | None = None
    group_index: int | None = None
    show_take_selector: bool = True
    show_take_lane: bool = False


@dataclass(slots=True)
class LayerStatus:
    stale: bool = False
    manually_modified: bool = False
    stale_reason: str | None = None


@dataclass(slots=True)
class LayerProvenance:
    source_layer_id: LayerId | None = None
    source_song_version_id: SongVersionId | None = None
    source_run_id: str | None = None
    pipeline_id: str | None = None
    output_name: str | None = None


@dataclass(frozen=True, slots=True)
class EventRef:
    layer_id: LayerId
    take_id: TakeId
    event_id: EventId


@dataclass(slots=True)
class Event:
    id: EventId
    take_id: TakeId
    start: float
    end: float
    cue_number: int = 1
    source_event_id: str | None = None
    parent_event_id: str | None = None
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
        try:
            cue_number = int(self.cue_number)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Event.cue_number must be an integer, got {self.cue_number!r}") from exc
        if cue_number < 1:
            raise ValueError(f"Event.cue_number must be >= 1, got {cue_number}")
        self.cue_number = cue_number

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class TimelineRegion:
    id: RegionId
    start: float
    end: float
    label: str = "Region"
    color: str | None = None
    order_index: int = 0
    kind: str = "custom"

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"TimelineRegion.start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"TimelineRegion.end must be >= start, got start={self.start}, end={self.end}"
            )
        self.label = (self.label or "").strip() or "Region"
        self.kind = (self.kind or "").strip().lower() or "custom"

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
    status: LayerStatus = field(default_factory=LayerStatus)
    provenance: LayerProvenance = field(default_factory=LayerProvenance)
    presentation_hints: LayerPresentationHints = field(default_factory=LayerPresentationHints)


@dataclass(slots=True)
class TimelineSelection:
    selected_layer_id: LayerId | None = None
    selected_layer_ids: list[LayerId] = field(default_factory=list)
    selected_take_id: TakeId | None = None
    selected_event_refs: list[EventRef] = field(default_factory=list)
    selected_event_ids: list[EventId] = field(default_factory=list)
    selected_region_id: RegionId | None = None


@dataclass(slots=True)
class TimelinePlaybackTarget:
    layer_id: LayerId | None = None
    take_id: TakeId | None = None


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
    regions: list[TimelineRegion] = field(default_factory=list)
    loop_region: TimeRange | None = None
    selection: TimelineSelection = field(default_factory=TimelineSelection)
    playback_target: TimelinePlaybackTarget = field(default_factory=TimelinePlaybackTarget)
    viewport: TimelineViewport = field(default_factory=TimelineViewport)

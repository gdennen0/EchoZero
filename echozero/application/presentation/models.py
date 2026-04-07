"""Simple UI-facing presentation models for the new EchoZero application layer."""

from dataclasses import dataclass, field

from echozero.application.shared.ids import TimelineId, LayerId, TakeId, EventId
from echozero.application.shared.enums import LayerKind, FollowMode, PlaybackMode, SyncMode
from echozero.application.shared.ranges import TimeRange


@dataclass(slots=True)
class TakeActionPresentation:
    action_id: str
    label: str


@dataclass(slots=True)
class TakeLanePresentation:
    take_id: TakeId
    name: str
    is_main: bool = False
    kind: LayerKind = LayerKind.EVENT
    events: list['EventPresentation'] = field(default_factory=list)
    source_ref: str | None = None
    waveform_key: str | None = None
    source_audio_path: str | None = None
    playback_source_ref: str | None = None
    actions: list[TakeActionPresentation] = field(default_factory=list)


@dataclass(slots=True)
class LayerStatusPresentation:
    stale: bool = False
    manually_modified: bool = False
    source_label: str = ""
    sync_label: str = ""
    stale_reason: str = ""
    source_layer_id: str = ""
    source_song_version_id: str = ""
    pipeline_id: str = ""
    output_name: str = ""
    source_run_id: str = ""


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
    main_take_id: TakeId | None = None
    subtitle: str = ""
    kind: LayerKind = LayerKind.EVENT
    is_selected: bool = False
    is_expanded: bool = False
    events: list[EventPresentation] = field(default_factory=list)
    takes: list[TakeLanePresentation] = field(default_factory=list)
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
    waveform_key: str | None = None
    source_audio_path: str | None = None
    playback_source_ref: str | None = None
    status: LayerStatusPresentation = field(default_factory=LayerStatusPresentation)


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
    current_time_label: str = "00:00.00"
    end_time_label: str = "00:00.00"

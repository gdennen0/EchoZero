"""Simple UI-facing presentation models for the new EchoZero application layer."""

from dataclasses import dataclass, field

from echozero.application.shared.ids import TimelineId, LayerId, TakeId, EventId
from echozero.application.shared.enums import LayerKind, FollowMode, PlaybackMode, SyncMode
from echozero.application.shared.ranges import TimeRange
from echozero.application.sync.models import LiveSyncState


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
class ManualPushTrackOptionPresentation:
    coord: str
    name: str
    note: str | None = None
    event_count: int | None = None


@dataclass(slots=True)
class SyncDiffSummaryPresentation:
    added_count: int = 0
    removed_count: int = 0
    modified_count: int = 0
    unchanged_count: int = 0
    row_count: int = 0


@dataclass(slots=True)
class SyncDiffRowPresentation:
    row_id: str
    action: str
    start: float
    end: float
    label: str
    before: str
    after: str


@dataclass(slots=True)
class ManualPushDiffPreviewPresentation:
    selected_count: int
    target_track_coord: str
    target_track_name: str
    target_track_note: str | None = None
    target_track_event_count: int | None = None
    diff_summary: SyncDiffSummaryPresentation | None = None
    diff_rows: list[SyncDiffRowPresentation] = field(default_factory=list)


@dataclass(slots=True)
class ManualPushFlowPresentation:
    dialog_open: bool = False
    available_tracks: list[ManualPushTrackOptionPresentation] = field(default_factory=list)
    target_track_coord: str | None = None
    diff_gate_open: bool = False
    diff_preview: ManualPushDiffPreviewPresentation | None = None


@dataclass(slots=True)
class ManualPullTrackOptionPresentation:
    coord: str
    name: str
    note: str | None = None
    event_count: int | None = None


@dataclass(slots=True)
class ManualPullEventOptionPresentation:
    event_id: str
    label: str
    start: float | None = None
    end: float | None = None


@dataclass(slots=True)
class ManualPullTargetOptionPresentation:
    layer_id: LayerId
    name: str


@dataclass(slots=True)
class ManualPullDiffPreviewPresentation:
    selected_count: int
    source_track_coord: str
    source_track_name: str
    source_track_note: str | None = None
    source_track_event_count: int | None = None
    target_layer_id: LayerId | None = None
    target_layer_name: str = ""
    import_mode: str = "new_take"
    diff_summary: SyncDiffSummaryPresentation | None = None
    diff_rows: list[SyncDiffRowPresentation] = field(default_factory=list)


@dataclass(slots=True)
class ManualPullFlowPresentation:
    dialog_open: bool = False
    available_tracks: list[ManualPullTrackOptionPresentation] = field(default_factory=list)
    source_track_coord: str | None = None
    available_events: list[ManualPullEventOptionPresentation] = field(default_factory=list)
    selected_ma3_event_ids: list[str] = field(default_factory=list)
    available_target_layers: list[ManualPullTargetOptionPresentation] = field(default_factory=list)
    target_layer_id: LayerId | None = None
    diff_gate_open: bool = False
    diff_preview: ManualPullDiffPreviewPresentation | None = None


@dataclass(slots=True)
class BatchTransferPlanRowPresentation:
    row_id: str
    direction: str
    source_label: str
    target_label: str
    selected_count: int = 0
    status: str = "draft"
    issue: str | None = None


@dataclass(slots=True)
class BatchTransferPlanPresentation:
    plan_id: str
    operation_type: str
    rows: list[BatchTransferPlanRowPresentation] = field(default_factory=list)
    draft_count: int = 0
    ready_count: int = 0
    blocked_count: int = 0
    applied_count: int = 0
    failed_count: int = 0


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
    live_sync_state: LiveSyncState = LiveSyncState.OFF
    live_sync_pause_reason: str | None = None
    live_sync_divergent: bool = False
    sync_target_label: str = ""
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
    follow_mode: FollowMode = FollowMode.CENTER
    selected_layer_id: LayerId | None = None
    selected_take_id: TakeId | None = None
    selected_event_ids: list[EventId] = field(default_factory=list)
    pixels_per_second: float = 100.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0
    experimental_live_sync_enabled: bool = False
    current_time_label: str = "00:00.00"
    end_time_label: str = "00:00.00"
    manual_push_flow: ManualPushFlowPresentation = field(default_factory=ManualPushFlowPresentation)
    manual_pull_flow: ManualPullFlowPresentation = field(default_factory=ManualPullFlowPresentation)
    batch_transfer_plan: BatchTransferPlanPresentation | None = None

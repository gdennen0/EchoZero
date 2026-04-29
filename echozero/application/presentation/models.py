"""Simple UI-facing presentation models for the new EchoZero application layer."""

from dataclasses import dataclass, field

from echozero.application.shared.cue_numbers import CueNumber
from echozero.application.shared.enums import FollowMode, LayerKind, PlaybackMode, SyncMode
from echozero.application.shared.ids import (
    EventId,
    LayerId,
    RegionId,
    SectionCueId,
    TakeId,
    TimelineId,
)
from echozero.application.shared.ranges import TimeRange
from echozero.application.sync.models import LiveSyncState
from echozero.application.timeline.models import EventRef


@dataclass(slots=True)
class TakeActionPresentation:
    action_id: str
    label: str
    compact_label: str | None = None


def default_take_actions(*, has_selection: bool = False) -> list["TakeActionPresentation"]:
    """Return the canonical editorial actions exposed for one subordinate take."""
    actions = [
        TakeActionPresentation(
            action_id="overwrite_main",
            label="Overwrite Main",
            compact_label="Overwrite",
        ),
        TakeActionPresentation(
            action_id="merge_main",
            label="Merge Main",
            compact_label="Merge",
        ),
    ]
    if has_selection:
        actions.insert(
            0,
            TakeActionPresentation(
                action_id="add_selection_to_main",
                label="Add Selection to Main",
                compact_label="Selection",
            ),
        )
    actions.append(
        TakeActionPresentation(
            action_id="delete_take",
            label="Delete Take",
            compact_label="Delete",
        )
    )
    return actions


@dataclass(slots=True)
class TakeLanePresentation:
    take_id: TakeId
    name: str
    is_main: bool = False
    kind: LayerKind = LayerKind.EVENT
    is_selected: bool = False
    is_playback_active: bool = False
    events: list["EventPresentation"] = field(default_factory=list)
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
class LayerHeaderControlPresentation:
    control_id: str
    label: str
    kind: str = "action"
    enabled: bool = True
    active: bool = False


def default_layer_header_controls(
    *,
    kind: LayerKind,
    main_take_id: TakeId | None,
    is_playback_active: bool,
    is_selected: bool = False,
) -> list[LayerHeaderControlPresentation]:
    controls: list[LayerHeaderControlPresentation] = []
    if kind is not LayerKind.SECTION:
        controls.append(
            LayerHeaderControlPresentation(
                control_id="set_active_playback_target",
                label="ACTIVE",
                kind="toggle",
                active=is_playback_active,
            )
        )
    if kind is LayerKind.AUDIO and is_selected:
        controls.append(
            LayerHeaderControlPresentation(
                control_id="layer_pipeline_actions",
                label="Pipelines",
            )
        )
    return controls


@dataclass(slots=True)
class ManualPushTrackOptionPresentation:
    coord: str
    name: str
    number: int | None = None
    timecode_name: str | None = None
    note: str | None = None
    event_count: int | None = None
    sequence_no: int | None = None


@dataclass(slots=True)
class ManualPushTimecodeOptionPresentation:
    number: int
    name: str | None = None


@dataclass(slots=True)
class ManualPushTrackGroupOptionPresentation:
    number: int
    name: str
    track_count: int | None = None


@dataclass(slots=True)
class ManualPushSequenceOptionPresentation:
    number: int
    name: str


@dataclass(slots=True)
class ManualPushSequenceRangePresentation:
    start: int
    end: int
    song_label: str | None = None


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
    push_mode_active: bool = False
    selected_layer_ids: list[LayerId] = field(default_factory=list)
    available_timecodes: list[ManualPushTimecodeOptionPresentation] = field(default_factory=list)
    selected_timecode_no: int | None = None
    available_track_groups: list[ManualPushTrackGroupOptionPresentation] = field(default_factory=list)
    selected_track_group_no: int | None = None
    available_tracks: list[ManualPushTrackOptionPresentation] = field(default_factory=list)
    available_sequences: list[ManualPushSequenceOptionPresentation] = field(default_factory=list)
    current_song_sequence_range: ManualPushSequenceRangePresentation | None = None
    target_track_coord: str | None = None
    transfer_mode: str = "merge"
    diff_gate_open: bool = False
    diff_preview: ManualPushDiffPreviewPresentation | None = None


@dataclass(slots=True)
class ManualPullTrackOptionPresentation:
    coord: str
    name: str
    number: int | None = None
    timecode_name: str | None = None
    note: str | None = None
    event_count: int | None = None


@dataclass(slots=True)
class ManualPullTimecodeOptionPresentation:
    number: int
    name: str | None = None


@dataclass(slots=True)
class ManualPullTrackGroupOptionPresentation:
    number: int
    name: str
    track_count: int | None = None


@dataclass(slots=True)
class ManualPullEventOptionPresentation:
    event_id: str
    label: str
    start: float | None = None
    end: float | None = None
    cue_ref: str | None = None
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None


@dataclass(slots=True)
class ManualPullTargetOptionPresentation:
    layer_id: LayerId
    name: str
    kind: LayerKind | None = None


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
    workspace_active: bool = False
    available_timecodes: list[ManualPullTimecodeOptionPresentation] = field(default_factory=list)
    selected_timecode_no: int | None = None
    available_track_groups: list[ManualPullTrackGroupOptionPresentation] = field(default_factory=list)
    selected_track_group_no: int | None = None
    available_tracks: list[ManualPullTrackOptionPresentation] = field(default_factory=list)
    selected_source_track_coords: list[str] = field(default_factory=list)
    active_source_track_coord: str | None = None
    source_track_coord: str | None = None
    available_events: list[ManualPullEventOptionPresentation] = field(default_factory=list)
    selected_ma3_event_ids: list[str] = field(default_factory=list)
    selected_ma3_event_ids_by_track: dict[str, list[str]] = field(default_factory=dict)
    import_mode: str = "new_take"
    import_mode_by_source_track: dict[str, str] = field(default_factory=dict)
    available_target_layers: list[ManualPullTargetOptionPresentation] = field(default_factory=list)
    target_layer_id: LayerId | None = None
    target_layer_id_by_source_track: dict[str, LayerId] = field(default_factory=dict)
    diff_gate_open: bool = False
    diff_preview: ManualPullDiffPreviewPresentation | None = None


@dataclass(slots=True)
class TransferPresetPresentation:
    preset_id: str
    name: str
    push_target_mapping_by_layer_id: dict[LayerId, str] = field(default_factory=dict)
    pull_target_mapping_by_source_track: dict[str, LayerId] = field(default_factory=dict)


@dataclass(slots=True)
class BatchTransferPlanRowPresentation:
    row_id: str
    direction: str
    source_label: str
    target_label: str
    source_layer_id: LayerId | None = None
    source_track_coord: str | None = None
    target_track_coord: str | None = None
    target_layer_id: LayerId | None = None
    import_mode: str = "new_take"
    selected_event_ids: list[EventId] = field(default_factory=list)
    selected_ma3_event_ids: list[str] = field(default_factory=list)
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
class PipelineRunBannerPresentation:
    run_id: str
    title: str
    status: str
    message: str
    percent: float | None = None
    is_error: bool = False


@dataclass(slots=True)
class SongVersionOptionPresentation:
    song_version_id: str
    label: str
    is_active: bool = False
    ma3_timecode_pool_no: int | None = None


@dataclass(slots=True)
class SongOptionPresentation:
    song_id: str
    title: str
    is_active: bool = False
    active_version_id: str = ""
    active_version_label: str = ""
    version_count: int = 0
    versions: list[SongVersionOptionPresentation] = field(default_factory=list)


@dataclass(slots=True)
class EventPresentation:
    event_id: EventId
    start: float
    end: float
    label: str
    cue_number: CueNumber | None = None
    cue_ref: str | None = None
    color: str | None = None
    notes: str | None = None
    muted: bool = False
    is_selected: bool = False
    badges: list[str] = field(default_factory=list)
    source_event_id: str | None = None
    payload_ref: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class RegionPresentation:
    region_id: RegionId
    start: float
    end: float
    label: str
    color: str | None = None
    kind: str = "custom"
    is_selected: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class SectionCuePresentation:
    cue_id: SectionCueId
    start: float
    cue_ref: str
    name: str
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None


@dataclass(slots=True)
class SectionRegionPresentation:
    cue_id: SectionCueId
    start: float
    end: float
    cue_ref: str
    name: str
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None

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
    is_playback_active: bool = False
    is_expanded: bool = False
    events: list[EventPresentation] = field(default_factory=list)
    takes: list[TakeLanePresentation] = field(default_factory=list)
    visible: bool = True
    locked: bool = False
    gain_db: float = 0.0
    pan: float = 0.0
    output_bus: str | None = None
    playback_mode: PlaybackMode = PlaybackMode.NONE
    playback_enabled: bool = False
    sync_mode: SyncMode = SyncMode.NONE
    sync_connected: bool = False
    live_sync_state: LiveSyncState = LiveSyncState.OFF
    live_sync_pause_reason: str | None = None
    live_sync_divergent: bool = False
    sync_target_label: str = ""
    push_target_label: str = ""
    push_selection_count: int = 0
    push_row_status: str = ""
    push_row_issue: str = ""
    pull_target_label: str = ""
    pull_selection_count: int = 0
    pull_row_status: str = ""
    pull_row_issue: str = ""
    color: str | None = None
    badges: list[str] = field(default_factory=list)
    header_controls: list[LayerHeaderControlPresentation] = field(default_factory=list)
    waveform_key: str | None = None
    source_audio_path: str | None = None
    playback_source_ref: str | None = None
    status: LayerStatusPresentation = field(default_factory=LayerStatusPresentation)

    def __post_init__(self) -> None:
        if not self.header_controls:
            self.header_controls = default_layer_header_controls(
                kind=self.kind,
                main_take_id=self.main_take_id,
                is_playback_active=self.is_playback_active,
                is_selected=self.is_selected,
            )


@dataclass(slots=True)
class TimelinePresentation:
    timeline_id: TimelineId
    title: str
    layers: list[LayerPresentation] = field(default_factory=list)
    section_cues: list[SectionCuePresentation] = field(default_factory=list)
    section_regions: list[SectionRegionPresentation] = field(default_factory=list)
    active_song_id: str = ""
    active_song_title: str = ""
    active_song_version_id: str = ""
    active_song_version_label: str = ""
    active_song_version_ma3_timecode_pool_no: int | None = None
    available_songs: list[SongOptionPresentation] = field(default_factory=list)
    available_song_versions: list[SongVersionOptionPresentation] = field(default_factory=list)
    bpm: float | None = None
    playhead: float = 0.0
    is_playing: bool = False
    loop_region: TimeRange | None = None
    follow_mode: FollowMode = FollowMode.CENTER
    selected_layer_id: LayerId | None = None
    selected_layer_ids: list[LayerId] = field(default_factory=list)
    selected_take_id: TakeId | None = None
    selected_event_refs: list[EventRef] = field(default_factory=list)
    active_playback_layer_id: LayerId | None = None
    active_playback_take_id: TakeId | None = None
    playback_output_channels: int = 0
    selected_event_ids: list[EventId] = field(default_factory=list)
    selected_region_id: RegionId | None = None
    regions: list[RegionPresentation] = field(default_factory=list)
    pixels_per_second: float = 100.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0
    experimental_live_sync_enabled: bool = False
    current_time_label: str = "00:00.00"
    end_time_label: str = "00:00.00"
    manual_push_flow: ManualPushFlowPresentation = field(
        default_factory=ManualPushFlowPresentation
    )
    manual_pull_flow: ManualPullFlowPresentation = field(
        default_factory=ManualPullFlowPresentation
    )
    batch_transfer_plan: BatchTransferPlanPresentation | None = None
    transfer_presets: list[TransferPresetPresentation] = field(default_factory=list)
    pipeline_run_banner: PipelineRunBannerPresentation | None = None

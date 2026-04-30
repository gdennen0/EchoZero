"""Session application models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from echozero.application.shared.cue_numbers import CueNumber
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, ProjectId, SessionId, SongId, SongVersionId, TimelineId
from echozero.application.sync.diff_service import SyncDiffRow, SyncDiffSummary
from echozero.application.transport.models import TransportState
from echozero.application.mixer.models import MixerState
from echozero.application.playback.models import PlaybackState
from echozero.application.sync.models import SyncState

if TYPE_CHECKING:
    from echozero.application.timeline.pipeline_run_service import PipelineRunState


@dataclass(slots=True)
class ManualPushTrackOption:
    coord: str
    name: str
    number: int | None = None
    timecode_name: str | None = None
    note: str | None = None
    event_count: int | None = None
    sequence_no: int | None = None


@dataclass(slots=True)
class ManualPushTimecodeOption:
    number: int
    name: str | None = None


@dataclass(slots=True)
class ManualPushTrackGroupOption:
    number: int
    name: str
    track_count: int | None = None


@dataclass(slots=True)
class ManualPushSequenceOption:
    number: int
    name: str


@dataclass(slots=True)
class ManualPushSequenceRange:
    start: int
    end: int
    song_label: str | None = None


@dataclass(slots=True)
class ManualPushDiffPreview:
    selected_count: int
    target_track_coord: str
    target_track_name: str
    target_track_note: str | None = None
    target_track_event_count: int | None = None
    diff_summary: SyncDiffSummary | None = None
    diff_rows: list[SyncDiffRow] = field(default_factory=list)


@dataclass(slots=True)
class ManualPushFlowState:
    dialog_open: bool = False
    push_mode_active: bool = False
    selected_layer_ids: list[LayerId] = field(default_factory=list)
    selected_event_ids: list[EventId] = field(default_factory=list)
    available_timecodes: list[ManualPushTimecodeOption] = field(default_factory=list)
    selected_timecode_no: int | None = None
    available_track_groups: list[ManualPushTrackGroupOption] = field(default_factory=list)
    selected_track_group_no: int | None = None
    available_tracks: list[ManualPushTrackOption] = field(default_factory=list)
    available_sequences: list[ManualPushSequenceOption] = field(default_factory=list)
    current_song_sequence_range: ManualPushSequenceRange | None = None
    target_track_coord: str | None = None
    transfer_mode: str = "merge"
    diff_gate_open: bool = False
    diff_preview: ManualPushDiffPreview | None = None


@dataclass(slots=True)
class ManualPullTrackOption:
    coord: str
    name: str
    number: int | None = None
    timecode_name: str | None = None
    note: str | None = None
    event_count: int | None = None


@dataclass(slots=True)
class ManualPullTimecodeOption:
    number: int
    name: str | None = None


@dataclass(slots=True)
class ManualPullTrackGroupOption:
    number: int
    name: str
    track_count: int | None = None


@dataclass(slots=True)
class ManualPullEventOption:
    event_id: str
    label: str
    start: float | None = None
    end: float | None = None
    cue_number: CueNumber | None = None
    cue_ref: str | None = None
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None


@dataclass(slots=True)
class ManualPullTargetOption:
    layer_id: LayerId
    name: str
    kind: LayerKind | None = None


@dataclass(slots=True)
class ManualPullDiffPreview:
    selected_count: int
    source_track_coord: str
    source_track_name: str
    source_track_note: str | None = None
    source_track_event_count: int | None = None
    target_layer_id: LayerId | None = None
    target_layer_name: str = ""
    import_mode: str = "new_take"
    diff_summary: SyncDiffSummary | None = None
    diff_rows: list[SyncDiffRow] = field(default_factory=list)


@dataclass(slots=True)
class ManualPullFlowState:
    dialog_open: bool = False
    workspace_active: bool = False
    available_timecodes: list[ManualPullTimecodeOption] = field(default_factory=list)
    selected_timecode_no: int | None = None
    available_track_groups: list[ManualPullTrackGroupOption] = field(default_factory=list)
    selected_track_group_no: int | None = None
    available_tracks: list[ManualPullTrackOption] = field(default_factory=list)
    selected_source_track_coords: list[str] = field(default_factory=list)
    active_source_track_coord: str | None = None
    source_track_coord: str | None = None
    available_events: list[ManualPullEventOption] = field(default_factory=list)
    selected_ma3_event_ids: list[str] = field(default_factory=list)
    selected_ma3_event_ids_by_track: dict[str, list[str]] = field(default_factory=dict)
    import_mode: str = "new_take"
    import_mode_by_source_track: dict[str, str] = field(default_factory=dict)
    available_target_layers: list[ManualPullTargetOption] = field(default_factory=list)
    target_layer_id: LayerId | None = None
    target_layer_id_by_source_track: dict[str, LayerId] = field(default_factory=dict)
    diff_gate_open: bool = False
    diff_preview: ManualPullDiffPreview | None = None


@dataclass(slots=True)
class TransferPresetState:
    preset_id: str
    name: str
    push_target_mapping_by_layer_id: dict[LayerId, str] = field(default_factory=dict)
    pull_target_mapping_by_source_track: dict[str, LayerId] = field(default_factory=dict)


@dataclass(slots=True)
class BatchTransferPlanRowState:
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
class BatchTransferPlanState:
    plan_id: str
    operation_type: str
    rows: list[BatchTransferPlanRowState] = field(default_factory=list)
    draft_count: int = 0
    ready_count: int = 0
    blocked_count: int = 0
    applied_count: int = 0
    failed_count: int = 0


@dataclass(slots=True)
class Session:
    id: SessionId
    project_id: ProjectId
    active_song_id: SongId | None = None
    active_song_version_id: SongVersionId | None = None
    active_song_version_ma3_timecode_pool_no: int | None = None
    project_ma3_push_offset_seconds: float = -1.0
    active_timeline_id: TimelineId | None = None
    transport_state: TransportState = field(default_factory=TransportState)
    mixer_state: MixerState = field(default_factory=MixerState)
    playback_state: PlaybackState = field(default_factory=PlaybackState)
    sync_state: SyncState = field(default_factory=SyncState)
    manual_push_flow: ManualPushFlowState = field(default_factory=ManualPushFlowState)
    manual_pull_flow: ManualPullFlowState = field(default_factory=ManualPullFlowState)
    batch_transfer_plan: BatchTransferPlanState | None = None
    transfer_presets: list[TransferPresetState] = field(default_factory=list)
    pipeline_runs: dict[str, PipelineRunState] = field(default_factory=dict)
    ui_prefs_ref: str | None = None

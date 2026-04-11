"""Session application models."""

from dataclasses import dataclass, field

from echozero.application.shared.ids import EventId, LayerId, ProjectId, SessionId, SongId, SongVersionId, TimelineId
from echozero.application.sync.diff_service import SyncDiffRow, SyncDiffSummary
from echozero.application.transport.models import TransportState
from echozero.application.mixer.models import MixerState
from echozero.application.playback.models import PlaybackState
from echozero.application.sync.models import SyncState


@dataclass(slots=True)
class ManualPushTrackOption:
    coord: str
    name: str
    note: str | None = None
    event_count: int | None = None


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
    selected_event_ids: list[EventId] = field(default_factory=list)
    available_tracks: list[ManualPushTrackOption] = field(default_factory=list)
    target_track_coord: str | None = None
    diff_gate_open: bool = False
    diff_preview: ManualPushDiffPreview | None = None


@dataclass(slots=True)
class ManualPullTrackOption:
    coord: str
    name: str
    note: str | None = None
    event_count: int | None = None


@dataclass(slots=True)
class ManualPullEventOption:
    event_id: str
    label: str
    start: float | None = None
    end: float | None = None


@dataclass(slots=True)
class ManualPullTargetOption:
    layer_id: LayerId
    name: str


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
    available_tracks: list[ManualPullTrackOption] = field(default_factory=list)
    source_track_coord: str | None = None
    available_events: list[ManualPullEventOption] = field(default_factory=list)
    selected_ma3_event_ids: list[str] = field(default_factory=list)
    available_target_layers: list[ManualPullTargetOption] = field(default_factory=list)
    target_layer_id: LayerId | None = None
    diff_gate_open: bool = False
    diff_preview: ManualPullDiffPreview | None = None


@dataclass(slots=True)
class Session:
    id: SessionId
    project_id: ProjectId
    active_song_id: SongId | None = None
    active_song_version_id: SongVersionId | None = None
    active_timeline_id: TimelineId | None = None
    transport_state: TransportState = field(default_factory=TransportState)
    mixer_state: MixerState = field(default_factory=MixerState)
    playback_state: PlaybackState = field(default_factory=PlaybackState)
    sync_state: SyncState = field(default_factory=SyncState)
    manual_push_flow: ManualPushFlowState = field(default_factory=ManualPushFlowState)
    manual_pull_flow: ManualPullFlowState = field(default_factory=ManualPullFlowState)
    ui_prefs_ref: str | None = None

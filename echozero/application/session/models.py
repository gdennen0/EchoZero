"""Session application models."""

from dataclasses import dataclass, field

from echozero.application.shared.ids import EventId, ProjectId, SessionId, SongId, SongVersionId, TimelineId
from echozero.application.transport.models import TransportState
from echozero.application.mixer.models import MixerState
from echozero.application.playback.models import PlaybackState
from echozero.application.sync.models import SyncState


@dataclass(slots=True)
class ManualPushFlowState:
    dialog_open: bool = False
    selected_event_ids: list[EventId] = field(default_factory=list)
    target_track_coord: str | None = None
    diff_gate_open: bool = False


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
    ui_prefs_ref: str | None = None

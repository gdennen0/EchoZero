from __future__ import annotations

import pytest

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import ManualPushTrackOption, Session
from echozero.application.session.service import SessionService
from echozero.application.shared.ids import ProjectId, SessionId, SongVersionId, TimelineId
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import SelectPushTargetTrack, SetPushTrackOptions
from echozero.application.timeline.models import Timeline
from echozero.application.timeline.orchestrator import TimelineOrchestrator
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService


class _SessionService(SessionService):
    def __init__(self, session: Session):
        self._session = session

    def get_session(self) -> Session:
        return self._session

    def set_active_song(self, song_id):
        self._session.active_song_id = song_id
        return self._session

    def set_active_song_version(self, song_version_id):
        self._session.active_song_version_id = song_version_id
        return self._session

    def set_active_timeline(self, timeline_id):
        self._session.active_timeline_id = timeline_id
        return self._session


class _TransportService(TransportService):
    def __init__(self):
        self._state = TransportState()

    def get_state(self) -> TransportState:
        return self._state

    def play(self) -> TransportState:
        self._state.is_playing = True
        return self._state

    def pause(self) -> TransportState:
        self._state.is_playing = False
        return self._state

    def stop(self) -> TransportState:
        self._state.is_playing = False
        self._state.playhead = 0.0
        return self._state

    def seek(self, position: float) -> TransportState:
        self._state.playhead = max(0.0, position)
        return self._state

    def set_loop(self, loop_region, enabled: bool = True) -> TransportState:
        self._state.loop_region = loop_region
        self._state.loop_enabled = enabled
        return self._state


class _MixerService(MixerService):
    def __init__(self):
        self._state = MixerState()

    def get_state(self) -> MixerState:
        return self._state

    def set_layer_state(self, layer_id, state: LayerMixerState) -> MixerState:
        self._state.layer_states[layer_id] = state
        return self._state

    def set_mute(self, layer_id, muted: bool) -> MixerState:
        return self._state

    def set_solo(self, layer_id, soloed: bool) -> MixerState:
        return self._state

    def set_gain(self, layer_id, gain_db: float) -> MixerState:
        return self._state

    def set_pan(self, layer_id, pan: float) -> MixerState:
        return self._state

    def resolve_audibility(self, layers: list) -> list[AudibilityState]:
        return []


class _PlaybackService(PlaybackService):
    def __init__(self):
        self._state = PlaybackState()

    def get_state(self) -> PlaybackState:
        return self._state

    def prepare(self, timeline: Timeline) -> PlaybackState:
        return self._state

    def update_runtime(self, timeline, transport, audibility, sync) -> PlaybackState:
        return self._state

    def stop(self) -> PlaybackState:
        return self._state


class _SyncService(SyncService):
    def __init__(self):
        self._state = SyncState()

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode):
        self._state.mode = mode
        return self._state

    def connect(self):
        self._state.connected = True
        return self._state

    def disconnect(self):
        self._state.connected = False
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        return transport


class _Assembler:
    def assemble(self, timeline, session):
        return timeline


def _build_orchestrator():
    session = Session(
        id=SessionId("session_manual_push_track_picker"),
        project_id=ProjectId("project_manual_push_track_picker"),
        active_timeline_id=TimelineId("timeline_manual_push_track_picker"),
    )
    timeline = Timeline(
        id=TimelineId("timeline_manual_push_track_picker"),
        song_version_id=SongVersionId("song_version_manual_push_track_picker"),
        layers=[],
    )
    orchestrator = TimelineOrchestrator(
        session_service=_SessionService(session),
        transport_service=_TransportService(),
        mixer_service=_MixerService(),
        playback_service=_PlaybackService(),
        sync_service=_SyncService(),
        assembler=_Assembler(),
    )
    return orchestrator, timeline, session


def test_set_push_track_options_updates_session_state():
    orchestrator, timeline, session = _build_orchestrator()
    tracks = [
        ManualPushTrackOption(coord="tc1_tg1_tr1", name="Track 1", note="Main", event_count=12),
        ManualPushTrackOption(coord="tc1_tg1_tr2", name="Track 2"),
    ]

    orchestrator.handle(timeline, SetPushTrackOptions(tracks=tracks))

    assert session.manual_push_flow.available_tracks == tracks


def test_select_push_target_track_sets_target_for_known_coord():
    orchestrator, timeline, session = _build_orchestrator()
    tracks = [
        ManualPushTrackOption(coord="tc1_tg1_tr1", name="Track 1"),
        ManualPushTrackOption(coord="tc1_tg1_tr2", name="Track 2"),
    ]
    orchestrator.handle(timeline, SetPushTrackOptions(tracks=tracks))

    orchestrator.handle(
        timeline,
        SelectPushTargetTrack(target_track_coord="tc1_tg1_tr2"),
    )

    assert session.manual_push_flow.target_track_coord == "tc1_tg1_tr2"


def test_select_push_target_track_raises_for_unknown_coord():
    orchestrator, timeline, _session = _build_orchestrator()
    orchestrator.handle(
        timeline,
        SetPushTrackOptions(
            tracks=[ManualPushTrackOption(coord="tc1_tg1_tr1", name="Track 1")],
        ),
    )

    with pytest.raises(
        ValueError,
        match="SelectPushTargetTrack target_track_coord not found in available_tracks: tc9_tg9_tr9",
    ):
        orchestrator.handle(
            timeline,
            SelectPushTargetTrack(target_track_coord="tc9_tg9_tr9"),
        )


@pytest.mark.parametrize("target_track_coord", ["", "   "])
def test_select_push_target_track_requires_non_empty_coord(target_track_coord):
    with pytest.raises(
        ValueError,
        match="SelectPushTargetTrack requires a non-empty target_track_coord",
    ):
        SelectPushTargetTrack(target_track_coord=target_track_coord)

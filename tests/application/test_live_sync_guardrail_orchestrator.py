from __future__ import annotations

import pytest

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind, SyncMode
from echozero.application.shared.ids import LayerId, ProjectId, SessionId, SongVersionId, TakeId, TimelineId
from echozero.application.sync.models import LiveSyncState, SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import (
    ClearLayerLiveSyncPauseReason,
    DisableExperimentalLiveSync,
    EnableExperimentalLiveSync,
    EnableSync,
    SetLayerLiveSyncPauseReason,
    SetLayerLiveSyncState,
)
from echozero.application.timeline.models import Layer, LayerSyncState, Take, Timeline
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

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._state.mode = mode
        return self._state

    def connect(self) -> SyncState:
        self._state.connected = True
        self._state.health = "healthy"
        return self._state

    def disconnect(self) -> SyncState:
        self._state.connected = False
        self._state.mode = SyncMode.NONE
        self._state.health = "offline"
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        return transport


class _Assembler:
    def assemble(self, timeline, session):
        return timeline


def _build_orchestrator() -> tuple[TimelineOrchestrator, Timeline, Session]:
    session = Session(
        id=SessionId("session_live_sync_guardrails"),
        project_id=ProjectId("project_live_sync_guardrails"),
        active_timeline_id=TimelineId("timeline_live_sync_guardrails"),
    )
    timeline = Timeline(
        id=TimelineId("timeline_live_sync_guardrails"),
        song_version_id=SongVersionId("song_version_live_sync_guardrails"),
        layers=[
            Layer(
                id=LayerId("layer_one"),
                timeline_id=TimelineId("timeline_live_sync_guardrails"),
                name="Layer One",
                kind=LayerKind.EVENT,
                order_index=0,
                takes=[Take(id=TakeId("take_one"), layer_id=LayerId("layer_one"), name="Main")],
                sync=LayerSyncState(),
            ),
            Layer(
                id=LayerId("layer_two"),
                timeline_id=TimelineId("timeline_live_sync_guardrails"),
                name="Layer Two",
                kind=LayerKind.EVENT,
                order_index=1,
                takes=[Take(id=TakeId("take_two"), layer_id=LayerId("layer_two"), name="Main")],
                sync=LayerSyncState(),
            ),
        ],
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


def test_toggle_experimental_live_sync_behavior_updates_session_flag():
    orchestrator, timeline, session = _build_orchestrator()

    orchestrator.handle(timeline, EnableExperimentalLiveSync())
    assert session.sync_state.experimental_live_sync_enabled is True

    orchestrator.handle(timeline, DisableExperimentalLiveSync())
    assert session.sync_state.experimental_live_sync_enabled is False


def test_reject_non_off_live_sync_state_when_experimental_disabled():
    orchestrator, timeline, session = _build_orchestrator()

    assert session.sync_state.experimental_live_sync_enabled is False

    with pytest.raises(
        ValueError,
        match="SetLayerLiveSyncState requires experimental live sync to be enabled for non-off states",
    ):
        orchestrator.handle(
            timeline,
            SetLayerLiveSyncState(
                layer_id=LayerId("layer_one"),
                live_sync_state=LiveSyncState.OBSERVE,
            ),
        )


def test_disable_experimental_live_sync_forces_layer_live_state_reset():
    orchestrator, timeline, _session = _build_orchestrator()
    layer_one, layer_two = timeline.layers
    layer_one.sync.live_sync_state = LiveSyncState.ARMED_WRITE
    layer_one.sync.live_sync_pause_reason = "operator hold"
    layer_one.sync.live_sync_divergent = True
    layer_two.sync.live_sync_state = LiveSyncState.PAUSED
    layer_two.sync.live_sync_pause_reason = "drift"
    layer_two.sync.live_sync_divergent = True

    orchestrator.handle(timeline, EnableExperimentalLiveSync())
    orchestrator.handle(timeline, DisableExperimentalLiveSync())

    for layer in timeline.layers:
        assert layer.sync.live_sync_state is LiveSyncState.OFF
        assert layer.sync.live_sync_pause_reason is None
        assert layer.sync.live_sync_divergent is False


def test_reconnect_downgrades_armed_write_layers_to_paused_with_reason():
    orchestrator, timeline, session = _build_orchestrator()
    layer_one, layer_two = timeline.layers
    layer_one.sync.live_sync_state = LiveSyncState.ARMED_WRITE
    layer_two.sync.live_sync_state = LiveSyncState.OBSERVE
    session.sync_state.experimental_live_sync_enabled = True

    orchestrator.handle(timeline, EnableSync(mode=SyncMode.MA3))

    assert layer_one.sync.live_sync_state is LiveSyncState.PAUSED
    assert layer_one.sync.live_sync_pause_reason == "Live sync reconnected; explicit re-arm required"
    assert layer_two.sync.live_sync_state is LiveSyncState.OBSERVE
    assert layer_two.sync.live_sync_pause_reason is None


def test_pause_reason_set_and_clear_behavior():
    orchestrator, timeline, session = _build_orchestrator()
    session.sync_state.experimental_live_sync_enabled = True

    orchestrator.handle(
        timeline,
        SetLayerLiveSyncState(
            layer_id=LayerId("layer_one"),
            live_sync_state=LiveSyncState.OBSERVE,
        ),
    )
    orchestrator.handle(
        timeline,
        SetLayerLiveSyncPauseReason(
            layer_id=LayerId("layer_one"),
            pause_reason="drift detected",
        ),
    )

    layer = timeline.layers[0]
    assert layer.sync.live_sync_state is LiveSyncState.PAUSED
    assert layer.sync.live_sync_pause_reason == "drift detected"

    orchestrator.handle(
        timeline,
        ClearLayerLiveSyncPauseReason(layer_id=LayerId("layer_one")),
    )

    assert layer.sync.live_sync_state is LiveSyncState.PAUSED
    assert layer.sync.live_sync_pause_reason is None

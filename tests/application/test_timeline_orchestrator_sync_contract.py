from __future__ import annotations

import pytest

from echozero.application.mixer.models import AudibilityState, MixerState, LayerMixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import SyncMode
from echozero.application.shared.ids import ProjectId, SessionId, SongVersionId, TimelineId
from echozero.application.sync.adapters import MA3SyncAdapter
from echozero.application.timeline.intents import DisableSync, EnableSync
from echozero.application.timeline.models import Timeline
from echozero.application.timeline.orchestrator import TimelineOrchestrator
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService


class _Bridge:
    def __init__(self):
        self.connected_calls = 0
        self.disconnected_calls = 0

    def on_ma3_connected(self) -> None:
        self.connected_calls += 1

    def on_ma3_disconnected(self) -> None:
        self.disconnected_calls += 1


class _FailingBridge(_Bridge):
    def on_ma3_connected(self) -> None:
        raise RuntimeError("bridge connect failed")


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


class _Assembler:
    def assemble(self, timeline, session):
        return timeline


def _build_orchestrator(sync_service):
    session = Session(
        id=SessionId("session_sync_contract"),
        project_id=ProjectId("project_sync_contract"),
        active_timeline_id=TimelineId("timeline_sync_contract"),
    )
    timeline = Timeline(
        id=TimelineId("timeline_sync_contract"),
        song_version_id=SongVersionId("song_version_sync_contract"),
        layers=[],
    )

    orchestrator = TimelineOrchestrator(
        session_service=_SessionService(session),
        transport_service=_TransportService(),
        mixer_service=_MixerService(),
        playback_service=_PlaybackService(),
        sync_service=sync_service,
        assembler=_Assembler(),
    )
    return orchestrator, timeline, session


def test_enable_sync_intent_uses_ma3_adapter_bridge_and_updates_session_state():
    bridge = _Bridge()
    sync_service = MA3SyncAdapter(bridge)
    orchestrator, timeline, session = _build_orchestrator(sync_service)

    orchestrator.handle(timeline, EnableSync(mode=SyncMode.MA3))

    assert bridge.connected_calls == 1
    assert session.sync_state.connected is True
    assert session.sync_state.mode == SyncMode.MA3
    assert session.sync_state.health == "healthy"


def test_disable_sync_intent_disconnects_bridge_and_updates_session_state():
    bridge = _Bridge()
    sync_service = MA3SyncAdapter(bridge)
    orchestrator, timeline, session = _build_orchestrator(sync_service)

    orchestrator.handle(timeline, EnableSync(mode=SyncMode.MA3))
    orchestrator.handle(timeline, DisableSync())

    assert bridge.connected_calls == 1
    assert bridge.disconnected_calls == 1
    assert session.sync_state.connected is False
    assert session.sync_state.mode == SyncMode.NONE
    assert session.sync_state.health == "offline"


def test_enable_sync_intent_propagates_bridge_connect_error():
    sync_service = MA3SyncAdapter(_FailingBridge())
    orchestrator, timeline, session = _build_orchestrator(sync_service)

    with pytest.raises(RuntimeError, match="bridge connect failed"):
        orchestrator.handle(timeline, EnableSync(mode=SyncMode.MA3))

    assert session.sync_state.connected is False
    assert session.sync_state.health == "error"

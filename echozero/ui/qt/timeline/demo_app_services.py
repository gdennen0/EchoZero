"""Demo-only timeline service shims for fixture app support.
Exists to keep support-only session, transport, mixer, and playback stubs out of the demo app entrypoint.
Never use these services in the canonical desktop runtime or app-shell path.
"""

from __future__ import annotations

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import PlaybackStatus
from echozero.application.sync.models import SyncState
from echozero.application.transport.service import TransportService
from echozero.application.transport.models import TransportState


class DemoSessionService(SessionService):
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


class DemoTransportService(TransportService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> TransportState:
        return self._session.transport_state

    def play(self) -> TransportState:
        self._session.transport_state.is_playing = True
        return self._session.transport_state

    def pause(self) -> TransportState:
        self._session.transport_state.is_playing = False
        return self._session.transport_state

    def stop(self) -> TransportState:
        self._session.transport_state.is_playing = False
        self._session.transport_state.playhead = 0.0
        return self._session.transport_state

    def seek(self, position: float) -> TransportState:
        self._session.transport_state.playhead = max(0.0, position)
        return self._session.transport_state

    def set_loop(self, loop_region, enabled: bool = True) -> TransportState:
        self._session.transport_state.loop_region = loop_region if enabled else None
        return self._session.transport_state


class DemoMixerService(MixerService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> MixerState:
        return self._session.mixer_state

    def set_layer_state(self, layer_id, state: LayerMixerState) -> MixerState:
        self._session.mixer_state.layer_states[layer_id] = state
        return self._session.mixer_state

    def set_mute(self, layer_id, muted: bool) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.mute = muted
        return self._session.mixer_state

    def set_solo(self, layer_id, soloed: bool) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.solo = soloed
        return self._session.mixer_state

    def set_gain(self, layer_id, gain_db: float) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.gain_db = gain_db
        return self._session.mixer_state

    def set_pan(self, layer_id, pan: float) -> MixerState:
        state = self._session.mixer_state.layer_states.setdefault(layer_id, LayerMixerState())
        state.pan = pan
        return self._session.mixer_state

    def resolve_audibility(self, layers) -> list[AudibilityState]:
        resolved: list[AudibilityState] = []
        for layer in layers:
            resolved.append(
                AudibilityState(layer_id=layer.layer_id, is_audible=True, reason="normal")
            )
        return resolved


class DemoPlaybackService(PlaybackService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> PlaybackState:
        return self._session.playback_state

    def prepare(self, timeline) -> PlaybackState:
        return self._session.playback_state

    def update_runtime(
        self, timeline, transport: TransportState, audibility, sync: SyncState
    ) -> PlaybackState:
        self._session.playback_state.status = (
            PlaybackStatus.PLAYING if transport.is_playing else PlaybackStatus.STOPPED
        )
        return self._session.playback_state

    def stop(self) -> PlaybackState:
        self._session.playback_state.status = PlaybackStatus.STOPPED
        return self._session.playback_state


__all__ = [
    "DemoMixerService",
    "DemoPlaybackService",
    "DemoSessionService",
    "DemoTransportService",
]

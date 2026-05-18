from __future__ import annotations

import pytest

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import SyncMode
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import (
    EventId,
    LayerId,
    ProjectId,
    SectionCueId,
    SessionId,
    SongVersionId,
    TakeId,
    TimelineId,
)
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.models import Event, Layer, SectionCue, Take, Timeline
from echozero.application.timeline.orchestrator import TimelineOrchestrator
from echozero.application.timeline.queries import TimelineQueries
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService


class _SessionService(SessionService):
    def __init__(self, session: Session) -> None:
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
    def __init__(self, state: TransportState) -> None:
        self._state = state
        self.calls: list[tuple[str, float | None]] = []

    def get_state(self) -> TransportState:
        return self._state

    def play(self) -> TransportState:
        self.calls.append(("play", None))
        self._state.is_playing = True
        return self._state

    def pause(self) -> TransportState:
        self.calls.append(("pause", None))
        self._state.is_playing = False
        return self._state

    def stop(self) -> TransportState:
        self.calls.append(("stop", None))
        self._state.is_playing = False
        self._state.playhead = 0.0
        return self._state

    def seek(self, position: float) -> TransportState:
        self.calls.append(("seek", float(position)))
        self._state.playhead = max(0.0, float(position))
        return self._state

    def set_loop(self, loop_region, enabled: bool = True) -> TransportState:
        self._state.loop_region = loop_region if enabled else None
        self._state.loop_enabled = enabled
        return self._state


class _MixerService(MixerService):
    def __init__(self) -> None:
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
    def __init__(self) -> None:
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
    def __init__(self, state: SyncState | None = None) -> None:
        self._state = state or SyncState()

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._state.mode = mode
        return self._state

    def connect(self) -> SyncState:
        self._state.connected = True
        return self._state

    def disconnect(self) -> SyncState:
        self._state.connected = False
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        return transport


def _build_app(
    *,
    playhead: float = 0.0,
    is_playing: bool = False,
    timeline_end: float = 32.0,
) -> tuple[TimelineApplication, _TransportService]:
    transport_state = TransportState(is_playing=is_playing, playhead=playhead)
    session = Session(
        id=SessionId("session_transport"),
        project_id=ProjectId("project_transport"),
        active_timeline_id=TimelineId("timeline_transport"),
        active_song_version_id=SongVersionId("song_version_transport"),
        transport_state=transport_state,
    )
    timeline = Timeline(
        id=TimelineId("timeline_transport"),
        song_version_id=SongVersionId("song_version_transport"),
        end=timeline_end,
        layers=[
            Layer(
                id=LayerId("layer_sections"),
                timeline_id=TimelineId("timeline_transport"),
                name="Sections",
                kind=LayerKind.SECTION,
                order_index=0,
                takes=[
                    Take(
                        id=TakeId("take_sections"),
                        layer_id=LayerId("layer_sections"),
                        name="Main",
                        events=[
                            Event(
                                id=EventId("section_intro"),
                                take_id=TakeId("take_sections"),
                                start=0.0,
                                end=0.05,
                                label="Intro",
                            ),
                            Event(
                                id=EventId("section_verse"),
                                take_id=TakeId("take_sections"),
                                start=8.0,
                                end=8.05,
                                label="Verse",
                            ),
                            Event(
                                id=EventId("section_chorus"),
                                take_id=TakeId("take_sections"),
                                start=24.0,
                                end=24.05,
                                label="Chorus",
                            ),
                        ],
                    ),
                ],
            ),
        ],
        section_cues=[
            SectionCue(SectionCueId("intro"), start=0.0, name="Intro"),
            SectionCue(SectionCueId("verse"), start=8.0, name="Verse"),
            SectionCue(SectionCueId("chorus"), start=24.0, name="Chorus"),
        ],
    )
    transport = _TransportService(transport_state)
    assembler = TimelineAssembler()
    orchestrator = TimelineOrchestrator(
        session_service=_SessionService(session),
        transport_service=transport,
        mixer_service=_MixerService(),
        playback_service=_PlaybackService(),
        sync_service=_SyncService(session.sync_state),
        assembler=assembler,
    )
    return (
        TimelineApplication(
            timeline=timeline,
            session=session,
            orchestrator=orchestrator,
            queries=TimelineQueries(assembler),
            sync_service=orchestrator.sync_service,
        ),
        transport,
    )


def test_external_transport_pause_is_idempotent() -> None:
    app, transport = _build_app(is_playing=True)

    app.apply_external_transport_update({"change": "pause", "action": "pause"})
    app.apply_external_transport_update({"change": "pause", "action": "pause"})

    assert transport.calls == [("pause", None)]
    assert app.session.transport_state.is_playing is False


def test_external_transport_toggle_uses_ez_playing_state() -> None:
    app, transport = _build_app(is_playing=False)

    app.apply_external_transport_update({"change": "play_pause", "action": "play_pause"})
    app.apply_external_transport_update({"change": "play_pause", "action": "play_pause"})

    assert transport.calls == [("play", None), ("pause", None)]
    assert app.session.transport_state.is_playing is False


def test_external_transport_stop_ignores_stale_playhead() -> None:
    app, transport = _build_app(playhead=12.0, is_playing=True)

    presentation = app.apply_external_transport_update(
        {"change": "stop", "action": "stop", "to_seconds": 19.0},
    )

    assert transport.calls == [("stop", None)]
    assert presentation.playhead == pytest.approx(0.0)


def test_external_transport_seek_and_move_clamp_to_timeline() -> None:
    app, transport = _build_app(playhead=4.0, timeline_end=10.0)

    app.apply_external_transport_update({"change": "scrubbed", "to_seconds": 99.0})
    app.apply_external_transport_update(
        {"change": "move", "action": "move", "delta_seconds": -12.0},
        current_playhead_seconds=4.0,
    )

    assert transport.calls == [("seek", 10.0), ("seek", 0.0)]


def test_external_transport_section_jumps_use_ez_section_cues() -> None:
    app, transport = _build_app(playhead=12.0, is_playing=True)

    app.apply_external_transport_update(
        {"change": "jump_next_section", "source": "ez_sections"},
        current_playhead_seconds=12.0,
        current_is_playing=True,
    )
    app.apply_external_transport_update(
        {"change": "jump_previous_section", "source": "ez_sections"},
        current_playhead_seconds=24.0,
        current_is_playing=True,
    )

    assert transport.calls == [("seek", 24.0), ("seek", 8.0)]
    assert app.session.transport_state.is_playing is True

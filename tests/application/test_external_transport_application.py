from __future__ import annotations

import pytest

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.coordination import (
    TransportCommand,
    TransportCommandAction,
)
from echozero.application.playback.models import PlaybackState, PlaybackTimingSnapshot
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
from echozero.application.timeline.intents import Pause, Play, SetPlaybackStart
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
    def __init__(self, state: TransportState, *, apply_seek_to_state: bool = True) -> None:
        self._state = state
        self._apply_seek_to_state = bool(apply_seek_to_state)
        self.calls: list[tuple[str, float | None]] = []

    def get_state(self) -> TransportState:
        return self._state

    def play(self) -> TransportState:
        self.calls.append(("play", None))
        if not self._state.is_playing:
            self._state.playhead = self._state.playback_start_seconds
        self._state.is_playing = True
        return self._state

    def pause(self) -> TransportState:
        self.calls.append(("pause", None))
        self._state.is_playing = False
        self._state.playhead = self._state.playback_start_seconds
        return self._state

    def stop(self) -> TransportState:
        self.calls.append(("stop", None))
        self._state.is_playing = False
        self._state.playhead = 0.0
        return self._state

    def seek(self, position: float) -> TransportState:
        self.calls.append(("seek", float(position)))
        if self._apply_seek_to_state:
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


class _RuntimeAudioRecorder:
    def __init__(self) -> None:
        self.synced_presentations = 0
        self.commands: list[TransportCommand] = []

    def sync_structure_state(self, _presentation) -> None:
        self.synced_presentations += 1

    def enqueue_transport_command(self, command: TransportCommand) -> None:
        self.commands.append(command)


class _RuntimeAudioSnapshotRecorder(_RuntimeAudioRecorder):
    def __init__(
        self,
        *,
        is_playing: bool,
        audible_time_seconds: float,
    ) -> None:
        super().__init__()
        self._snapshot = PlaybackTimingSnapshot(
            audible_time_seconds=float(audible_time_seconds),
            clock_time_seconds=float(audible_time_seconds),
            snapshot_monotonic_seconds=1.0,
            is_playing=bool(is_playing),
        )

    def latest_timing_snapshot(self) -> PlaybackTimingSnapshot:
        return self._snapshot


class _LegacyRuntimeAudioRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float | None]] = []

    def play(self) -> None:
        self.calls.append(("play", None))

    def pause(self) -> None:
        self.calls.append(("pause", None))

    def seek(self, position_seconds: float) -> None:
        self.calls.append(("seek", float(position_seconds)))


def _build_app(
    *,
    playhead: float = 0.0,
    is_playing: bool = False,
    timeline_end: float = 32.0,
    apply_seek_to_state: bool = True,
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
    transport = _TransportService(transport_state, apply_seek_to_state=apply_seek_to_state)
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


def test_app_play_uses_playback_anchor_without_rebuilding_graph() -> None:
    app, _transport = _build_app(playhead=4.0, is_playing=False)
    app.session.transport_state.playback_start = 1.25
    runtime_audio = _RuntimeAudioRecorder()
    app.runtime_audio = runtime_audio

    app.dispatch(Play())

    assert runtime_audio.synced_presentations == 0
    assert [(command.action, command.position_seconds) for command in runtime_audio.commands] == [
        (TransportCommandAction.PLAY, 1.25)
    ]


def test_app_play_after_runtime_end_restarts_from_home_when_session_is_stale() -> None:
    app, _transport = _build_app(playhead=12.0, is_playing=True)
    app.session.transport_state.playback_start = 1.25
    runtime_audio = _RuntimeAudioSnapshotRecorder(
        is_playing=False,
        audible_time_seconds=12.0,
    )
    app.runtime_audio = runtime_audio

    presentation = app.dispatch(Play())

    assert [(command.action, command.position_seconds) for command in runtime_audio.commands] == [
        (TransportCommandAction.PLAY, 1.25)
    ]
    assert app.session.transport_state.is_playing is True
    assert presentation.is_playing is True
    assert presentation.playhead == 1.25


def test_app_pause_sends_pure_runtime_pause_while_session_returns_to_anchor() -> None:
    app, _transport = _build_app(playhead=8.0, is_playing=True)
    app.session.transport_state.playback_start = 2.5
    runtime_audio = _RuntimeAudioRecorder()
    app.runtime_audio = runtime_audio

    presentation = app.dispatch(Pause())

    assert runtime_audio.synced_presentations == 0
    assert [(command.action, command.position_seconds) for command in runtime_audio.commands] == [
        (TransportCommandAction.PAUSE, None)
    ]
    assert presentation.playhead == 2.5
    assert app.session.transport_state.playback_start_seconds == 2.5


def test_app_pause_legacy_adapter_can_still_seek_before_pending_pause() -> None:
    runtime_audio = _LegacyRuntimeAudioRecorder()

    TimelineApplication._enqueue_runtime_transport(
        runtime_audio,
        TransportCommandAction.PAUSE,
        position_seconds=2.5,
    )

    assert runtime_audio.calls == [("seek", 2.5), ("pause", None)]


def test_app_set_playback_start_while_playing_seeks_runtime_and_playhead() -> None:
    app, _transport = _build_app(playhead=8.0, is_playing=True)
    app.session.transport_state.playback_start = 2.5
    runtime_audio = _RuntimeAudioRecorder()
    app.runtime_audio = runtime_audio

    presentation = app.dispatch(SetPlaybackStart(4.75))

    assert [(command.action, command.position_seconds) for command in runtime_audio.commands] == [
        (TransportCommandAction.SEEK, 4.75)
    ]
    assert app.session.transport_state.playback_start_seconds == 4.75
    assert presentation.playback_start == 4.75
    assert presentation.playhead == 4.75


def test_app_set_playback_start_after_runtime_end_only_moves_home() -> None:
    app, _transport = _build_app(playhead=8.0, is_playing=True)
    app.session.transport_state.playback_start = 2.5
    runtime_audio = _RuntimeAudioSnapshotRecorder(
        is_playing=False,
        audible_time_seconds=8.0,
    )
    app.runtime_audio = runtime_audio

    presentation = app.dispatch(SetPlaybackStart(4.75))

    assert runtime_audio.commands == []
    assert app.session.transport_state.is_playing is False
    assert app.session.transport_state.playback_start_seconds == 4.75
    assert presentation.is_playing is False
    assert presentation.playback_start == 4.75
    assert presentation.playhead == 8.0


def test_app_set_playback_start_keeps_paused_playhead_until_next_play() -> None:
    app, _transport = _build_app(playhead=8.0, is_playing=False)

    presentation = app.dispatch(SetPlaybackStart(4.75))

    assert app.session.transport_state.playback_start_seconds == 4.75
    assert presentation.playback_start == 4.75
    assert presentation.playhead == 8.0


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


def test_external_transport_section_jump_presentation_uses_seek_target() -> None:
    app, transport = _build_app(
        playhead=12.0,
        is_playing=True,
        apply_seek_to_state=False,
    )

    presentation = app.apply_external_transport_update(
        {"change": "jump_previous_section", "source": "ez_sections"},
        current_playhead_seconds=12.0,
        current_is_playing=True,
    )

    assert transport.calls == [("seek", 8.0)]
    assert presentation.playhead == pytest.approx(8.0)
    assert presentation.current_time_label == "00:00:08.00"


def test_external_transport_previous_section_can_repeat_while_playing() -> None:
    app, transport = _build_app(playhead=12.0, is_playing=True)

    app.apply_external_transport_update(
        {"change": "jump_previous_section", "source": "ez_sections"},
        current_playhead_seconds=12.0,
        current_is_playing=True,
    )
    app.apply_external_transport_update(
        {"change": "jump_previous_section", "source": "ez_sections"},
        current_playhead_seconds=8.4,
        current_is_playing=True,
    )

    assert transport.calls == [("seek", 8.0), ("seek", 0.0)]
    assert app.session.transport_state.is_playing is True


def test_external_transport_move_after_section_jump_uses_ez_playhead() -> None:
    app, transport = _build_app(playhead=12.0, is_playing=True)

    app.apply_external_transport_update({"change": "jump_next_section", "source": "ez_sections"})
    app.apply_external_transport_update(
        {"change": "move", "action": "move", "delta_seconds": -1.0},
    )

    assert transport.calls == [("seek", 24.0), ("seek", 23.0)]
    assert app.session.transport_state.playhead == pytest.approx(23.0)

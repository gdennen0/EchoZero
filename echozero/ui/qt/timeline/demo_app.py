"""Runnable demo/testing loop for the new Stage Zero timeline shell."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from echozero.application.mixer.models import AudibilityState, MixerState, LayerMixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import EventPresentation, LayerPresentation, LayerStatusPresentation, TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import FollowMode, PlaybackStatus, SyncMode
from echozero.application.shared.ids import EventId, ProjectId, SessionId, SongId, SongVersionId
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.intents import (
    ClearSelection,
    Pause,
    Play,
    Seek,
    SelectEvent,
    SelectLayer,
    SelectTake,
    Stop,
    TimelineIntent,
    ToggleLayerExpanded,
    ToggleMute,
    ToggleSolo,
    SetGain,
    TriggerTakeAction,
)
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService
from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController
from echozero.ui.qt.timeline.widget import TimelineWidget


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
            resolved.append(AudibilityState(layer_id=layer.layer_id, is_audible=not layer.muted, reason='normal'))
        return resolved


class DemoPlaybackService(PlaybackService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> PlaybackState:
        return self._session.playback_state

    def prepare(self, timeline) -> PlaybackState:
        return self._session.playback_state

    def update_runtime(self, timeline, transport: TransportState, audibility, sync: SyncState) -> PlaybackState:
        self._session.playback_state.status = PlaybackStatus.PLAYING if transport.is_playing else PlaybackStatus.STOPPED
        return self._session.playback_state

    def stop(self) -> PlaybackState:
        self._session.playback_state.status = PlaybackStatus.STOPPED
        return self._session.playback_state


class DemoSyncService(SyncService):
    def __init__(self, session: Session):
        self._session = session

    def get_state(self) -> SyncState:
        return self._session.sync_state

    def connect(self) -> SyncState:
        self._session.sync_state.connected = True
        return self._session.sync_state

    def disconnect(self) -> SyncState:
        self._session.sync_state.connected = False
        self._session.sync_state.mode = SyncMode.NONE
        return self._session.sync_state

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._session.sync_state.mode = mode
        return self._session.sync_state

    def align_transport(self, transport: TransportState) -> TransportState:
        return transport


@dataclass(slots=True)
class DemoTimelineApp:
    presentation_state: TimelinePresentation
    session: Session
    runtime_audio: TimelineRuntimeAudioController | None = None

    def presentation(self) -> TimelinePresentation:
        return self.presentation_state

    def dispatch(self, intent: TimelineIntent) -> TimelinePresentation:
        if isinstance(intent, Pause):
            self.session.transport_state.is_playing = False
            if self.runtime_audio is not None:
                self.runtime_audio.pause()
        elif isinstance(intent, Play):
            if self.runtime_audio is not None:
                self.runtime_audio.build_for_presentation(self.presentation_state)
                self.runtime_audio.play()
            self.session.transport_state.is_playing = True
        elif isinstance(intent, Stop):
            if self.runtime_audio is not None:
                self.runtime_audio.stop()
            self.session.transport_state.is_playing = False
            self.session.transport_state.playhead = 0.0
        elif isinstance(intent, Seek):
            if self.runtime_audio is not None:
                self.runtime_audio.seek(intent.position)
            self.session.transport_state.playhead = max(0.0, intent.position)
        elif isinstance(intent, ToggleMute):
            layers = []
            for layer in self.presentation_state.layers:
                if layer.layer_id == intent.layer_id:
                    layers.append(replace(layer, muted=not layer.muted))
                else:
                    layers.append(layer)
            self.presentation_state = replace(self.presentation_state, layers=layers)
            if self.runtime_audio is not None:
                self.runtime_audio.apply_mix_state(self.presentation_state)
        elif isinstance(intent, ToggleSolo):
            layers = []
            for layer in self.presentation_state.layers:
                if layer.layer_id == intent.layer_id:
                    layers.append(replace(layer, soloed=not layer.soloed))
                else:
                    layers.append(layer)
            self.presentation_state = replace(self.presentation_state, layers=layers)
            if self.runtime_audio is not None:
                self.runtime_audio.apply_mix_state(self.presentation_state)
        elif isinstance(intent, SetGain):
            layers = []
            for layer in self.presentation_state.layers:
                if layer.layer_id == intent.layer_id:
                    layers.append(replace(layer, gain_db=float(intent.gain_db)))
                else:
                    layers.append(layer)
            self.presentation_state = replace(self.presentation_state, layers=layers)
            if self.runtime_audio is not None:
                self.runtime_audio.apply_mix_state(self.presentation_state)
        elif isinstance(intent, ToggleLayerExpanded):
            layers = []
            for layer in self.presentation_state.layers:
                if layer.layer_id == intent.layer_id:
                    layers.append(replace(layer, is_expanded=not layer.is_expanded))
                else:
                    layers.append(layer)
            self.presentation_state = replace(self.presentation_state, layers=layers)
        elif isinstance(intent, SelectLayer):
            layers = [
                replace(layer, is_selected=(layer.layer_id == intent.layer_id))
                for layer in self.presentation_state.layers
            ]
            self.presentation_state = replace(
                self.presentation_state,
                layers=layers,
                selected_layer_id=intent.layer_id,
                selected_take_id=None,
                selected_event_ids=[],
            )
        elif isinstance(intent, SelectTake):
            layers = [
                replace(layer, is_selected=(layer.layer_id == intent.layer_id))
                for layer in self.presentation_state.layers
            ]
            self.presentation_state = replace(
                self.presentation_state,
                layers=layers,
                selected_layer_id=intent.layer_id,
                selected_take_id=intent.take_id,
                selected_event_ids=[],
            )
        elif isinstance(intent, SelectEvent):
            layers = _select_event(
                self.presentation_state.layers,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
                event_id=intent.event_id,
            )
            selected_ids = [] if intent.event_id is None else [intent.event_id]
            self.presentation_state = replace(
                self.presentation_state,
                layers=layers,
                selected_layer_id=intent.layer_id,
                selected_take_id=intent.take_id,
                selected_event_ids=selected_ids,
            )
        elif isinstance(intent, ClearSelection):
            layers = _clear_selection(self.presentation_state.layers)
            self.presentation_state = replace(
                self.presentation_state,
                layers=layers,
                selected_layer_id=None,
                selected_take_id=None,
                selected_event_ids=[],
            )
        elif isinstance(intent, TriggerTakeAction):
            self.presentation_state = replace(
                self.presentation_state,
                layers=_apply_take_action(self.presentation_state.layers, intent.layer_id, intent.take_id, intent.action_id),
            )
        if self.runtime_audio is not None:
            # Keep UI transport state in lockstep with the runtime audio clock so
            # layer mix actions (mute/solo) do not snap playhead backward.
            self.session.transport_state.playhead = self.runtime_audio.current_time_seconds()
            self.session.transport_state.is_playing = self.runtime_audio.is_playing()

        self.presentation_state = replace(
            self.presentation_state,
            is_playing=self.session.transport_state.is_playing,
            playhead=self.session.transport_state.playhead,
            current_time_label=_fmt_time(self.session.transport_state.playhead),
        )
        return self.presentation_state


def _fmt_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins:02d}:{secs:05.2f}"


def _apply_take_action(
    layers: list[LayerPresentation],
    layer_id,
    take_id,
    action_id: str,
) -> list[LayerPresentation]:
    updated: list[LayerPresentation] = []
    for layer in layers:
        if layer.layer_id != layer_id:
            updated.append(layer)
            continue

        take = next((candidate for candidate in layer.takes if candidate.take_id == take_id), None)
        if take is None:
            updated.append(layer)
            continue

        next_events = list(layer.events)
        if action_id in {'overwrite_main', 'promote_take'}:
            next_events = _clone_events_for_main(take.events, suffix='ow')
        elif action_id == 'merge_main':
            merged = list(layer.events)
            merged.extend(_clone_events_for_main(take.events, suffix='mg'))
            next_events = sorted(merged, key=lambda e: (e.start, e.end))

        next_status = replace(layer.status, stale=False, manually_modified=True)
        updated.append(replace(layer, events=next_events, status=next_status))
    return updated


def _clone_events_for_main(events: list[EventPresentation], *, suffix: str) -> list[EventPresentation]:
    clones: list[EventPresentation] = []
    for idx, event in enumerate(events, start=1):
        clones.append(
            replace(
                event,
                event_id=EventId(f'{event.event_id}_{suffix}_{idx}'),
                is_selected=False,
            )
        )
    return clones


def _clear_selection(layers: list[LayerPresentation]) -> list[LayerPresentation]:
    updated: list[LayerPresentation] = []
    for layer in layers:
        updated.append(
            replace(
                layer,
                is_selected=False,
                events=[replace(event, is_selected=False) for event in layer.events],
                takes=[
                    replace(take, events=[replace(event, is_selected=False) for event in take.events])
                    for take in layer.takes
                ],
            )
        )
    return updated


def _select_event(
    layers: list[LayerPresentation],
    *,
    layer_id,
    take_id,
    event_id,
) -> list[LayerPresentation]:
    updated: list[LayerPresentation] = []
    for layer in layers:
        is_layer_selected = layer.layer_id == layer_id
        events = [
            replace(event, is_selected=is_layer_selected and take_id is None and event.event_id == event_id)
            for event in layer.events
        ]
        takes = []
        for take in layer.takes:
            takes.append(
                replace(
                    take,
                    events=[
                        replace(
                            event,
                            is_selected=is_layer_selected and take.take_id == take_id and event.event_id == event_id,
                        )
                        for event in take.events
                    ],
                )
            )
        updated.append(replace(layer, is_selected=is_layer_selected, events=events, takes=takes))
    return updated


def build_demo_presentation() -> TimelinePresentation:
    return load_realistic_timeline_fixture()


def build_demo_app() -> DemoTimelineApp:
    presentation = build_demo_presentation()
    session = Session(
        id=SessionId('session_demo'),
        project_id=ProjectId('project_demo'),
        active_song_id=SongId('song_demo'),
        active_song_version_id=SongVersionId('song_version_demo'),
        active_timeline_id=presentation.timeline_id,
        transport_state=TransportState(
            is_playing=presentation.is_playing,
            playhead=presentation.playhead,
            follow_mode=presentation.follow_mode,
        ),
        mixer_state=MixerState(),
        playback_state=PlaybackState(
            status=PlaybackStatus.PLAYING if presentation.is_playing else PlaybackStatus.STOPPED,
            backend_name='demo',
        ),
        sync_state=SyncState(mode=SyncMode.MA3, connected=True, target_ref='show_manager'),
    )
    return DemoTimelineApp(presentation_state=presentation, session=session)


def build_real_data_demo_app(
    audio_path: str | Path,
    *,
    working_root: str | Path,
    song_title: str = "Doechii Nissan Altima",
    runtime_audio: TimelineRuntimeAudioController | None = None,
) -> tuple[DemoTimelineApp, object]:
    from echozero.ui.qt.timeline.real_data_fixture import build_real_data_presentation

    presentation, summary = build_real_data_presentation(
        audio_path=audio_path,
        working_root=working_root,
        song_title=song_title,
    )
    app = build_demo_app()
    app.presentation_state = presentation
    app.session.active_timeline_id = presentation.timeline_id
    app.session.transport_state.is_playing = presentation.is_playing
    app.session.transport_state.playhead = presentation.playhead
    app.runtime_audio = runtime_audio or TimelineRuntimeAudioController()
    app.runtime_audio.build_for_presentation(presentation)
    return app, summary


def main() -> int:
    app = QApplication(sys.argv)
    demo = build_demo_app()
    widget = TimelineWidget(
        demo.presentation(),
        on_intent=demo.dispatch,
        runtime_audio=demo.runtime_audio,
    )
    widget.resize(1440, 720)
    widget.show()
    try:
        return app.exec()
    finally:
        if demo.runtime_audio is not None:
            demo.runtime_audio.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())

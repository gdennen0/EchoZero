from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from echozero.application.mixer.models import AudibilityState, LayerMixerState, MixerState
from echozero.application.mixer.service import MixerService
from echozero.application.playback.models import PlaybackState
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import ManualPullEventOption, ManualPullTrackOption, ManualPushTrackOption, Session
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import PlaybackStatus, SyncMode
from echozero.application.shared.ids import EventId, LayerId, ProjectId, SessionId, SongId, SongVersionId, TakeId, TimelineId
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.models import Event, Layer, LayerPresentationHints, LayerStatus, LayerSyncState, Take, Timeline, TimelineSelection, TimelineViewport
from echozero.application.timeline.orchestrator import TimelineOrchestrator
from echozero.application.transport.models import TransportState
from echozero.application.transport.service import TransportService
from echozero.testing.app_flow import AppFlowHarness
from echozero.testing.gui_dsl import GuiScenario, load_scenario
from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture


@dataclass(slots=True)
class StepTrace:
    index: int
    action: str
    label: str
    status: str
    snapshot: dict[str, object]
    error: str | None = None


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
    def __init__(self, state: TransportState):
        self._state = state

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
        self._state.playhead = max(0.0, float(position))
        return self._state

    def set_loop(self, loop_region, enabled: bool = True) -> TransportState:
        self._state.loop_region = loop_region
        self._state.loop_enabled = enabled
        return self._state


class _MixerService(MixerService):
    def __init__(self, state: MixerState):
        self._state = state

    def get_state(self) -> MixerState:
        return self._state

    def set_layer_state(self, layer_id, state: LayerMixerState) -> MixerState:
        self._state.layer_states[layer_id] = state
        return self._state

    def set_mute(self, layer_id, muted: bool) -> MixerState:
        self._state.layer_states.setdefault(layer_id, LayerMixerState()).mute = muted
        return self._state

    def set_solo(self, layer_id, soloed: bool) -> MixerState:
        self._state.layer_states.setdefault(layer_id, LayerMixerState()).solo = soloed
        return self._state

    def set_gain(self, layer_id, gain_db: float) -> MixerState:
        self._state.layer_states.setdefault(layer_id, LayerMixerState()).gain_db = gain_db
        return self._state

    def set_pan(self, layer_id, pan: float) -> MixerState:
        self._state.layer_states.setdefault(layer_id, LayerMixerState()).pan = pan
        return self._state

    def resolve_audibility(self, layers: list[Layer]) -> list[AudibilityState]:
        return [AudibilityState(layer_id=layer.id, is_audible=not layer.mixer.mute, reason="normal") for layer in layers]


class _PlaybackService(PlaybackService):
    def __init__(self, state: PlaybackState):
        self._state = state

    def get_state(self) -> PlaybackState:
        return self._state

    def prepare(self, timeline: Timeline) -> PlaybackState:
        return self._state

    def update_runtime(self, timeline, transport, audibility, sync) -> PlaybackState:
        self._state.status = PlaybackStatus.PLAYING if transport.is_playing else PlaybackStatus.STOPPED
        return self._state

    def stop(self) -> PlaybackState:
        self._state.status = PlaybackStatus.STOPPED
        return self._state


class _LaneBSyncService(SyncService):
    def __init__(self, state: SyncState | None = None):
        self._state = state or SyncState(mode=SyncMode.NONE, connected=False, target_ref="lane_b")
        self._push_tracks = [
            ManualPushTrackOption(coord="tc1_tg2_tr3", name="Track 3", note="Bass", event_count=8),
            ManualPushTrackOption(coord="tc1_tg2_tr4", name="Track 4", note="Lead", event_count=4),
        ]
        self._pull_tracks = [
            ManualPullTrackOption(coord="tc1_tg2_tr3", name="Track 3", note="Bass", event_count=2),
            ManualPullTrackOption(coord="tc1_tg2_tr4", name="Track 4", note="Lead", event_count=1),
        ]
        self._pull_events = {
            "tc1_tg2_tr3": [
                ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5),
                ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5),
            ],
            "tc1_tg2_tr4": [
                ManualPullEventOption(event_id="ma3_evt_9", label="Cue 9", start=9.0, end=9.5),
            ],
        }

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._state.mode = mode
        return self._state

    def connect(self) -> SyncState:
        if self._state.mode == SyncMode.NONE:
            self._state.mode = SyncMode.MA3
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

    def list_push_track_options(self) -> list[ManualPushTrackOption]:
        return list(self._push_tracks)

    def list_pull_track_options(self) -> list[ManualPullTrackOption]:
        return list(self._pull_tracks)

    def list_pull_source_events(self, source_track_coord: str) -> list[ManualPullEventOption]:
        return list(self._pull_events.get(source_track_coord, []))


class _LaneBApp:
    def __init__(self, *, project_name: str):
        self._project_name = project_name
        self.timeline = _build_timeline_from_fixture()
        self.session = Session(
            id=SessionId("session_lane_b"),
            project_id=ProjectId("project_lane_b"),
            active_song_id=SongId("song_lane_b"),
            active_song_version_id=SongVersionId("song_version_lane_b"),
            active_timeline_id=self.timeline.id,
            transport_state=TransportState(is_playing=False, playhead=0.0),
            mixer_state=MixerState(),
            playback_state=PlaybackState(status=PlaybackStatus.STOPPED, backend_name="lane_b"),
            sync_state=SyncState(mode=SyncMode.NONE, connected=False, target_ref="lane_b"),
        )
        self.runtime_audio = None
        self._sync_service = _LaneBSyncService(self.session.sync_state)
        self._assembler = TimelineAssembler()
        self._orchestrator = TimelineOrchestrator(
            session_service=_SessionService(self.session),
            transport_service=_TransportService(self.session.transport_state),
            mixer_service=_MixerService(self.session.mixer_state),
            playback_service=_PlaybackService(self.session.playback_state),
            sync_service=self._sync_service,
            assembler=self._assembler,
        )

    def presentation(self) -> TimelinePresentation:
        return replace(self._assembler.assemble(self.timeline, self.session), title=self._project_name)

    def dispatch(self, intent) -> TimelinePresentation:
        return replace(self._orchestrator.handle(self.timeline, intent), title=self._project_name)

    def enable_sync(self, mode: SyncMode = SyncMode.MA3) -> SyncState:
        self._sync_service.set_mode(mode)
        state = self._sync_service.connect()
        self.session.sync_state = state
        for layer in self.timeline.layers:
            if layer.kind.value == "event":
                layer.sync.connected = True
                layer.sync.mode = mode.value
        return state

    def disable_sync(self) -> SyncState:
        state = self._sync_service.disconnect()
        self.session.sync_state = state
        for layer in self.timeline.layers:
            layer.sync.connected = False
            layer.sync.mode = SyncMode.NONE.value
        return state


class GuiLaneBRunner:
    def __init__(self, *, scenario: GuiScenario, output_dir: Path | None = None):
        self._scenario = scenario
        self._output_dir = output_dir

    def run(self) -> list[dict[str, object]]:
        trace: list[dict[str, object]] = []
        harness_root = (self._output_dir.parent if self._output_dir is not None else Path("C:/Users/griff/AppData/Local/Temp")) / "gui-lane-b-working"
        harness = AppFlowHarness(
            initial_project_name=self._scenario.name,
            working_dir_root=harness_root,
        )
        harness.runtime._app = _LaneBApp(project_name=self._scenario.name)  # type: ignore[attr-defined]
        harness.widget.set_presentation(harness.presentation())
        _render_for_hit_testing(harness)

        try:
            if self._output_dir is not None:
                self._output_dir.mkdir(parents=True, exist_ok=True)

            for index, step in enumerate(self._scenario.steps):
                label = step.label or step.action
                try:
                    self._execute_step(harness, action=step.action, params=step.params)
                    status = "passed"
                    error = None
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
                trace.append(
                    asdict(
                        StepTrace(
                            index=index,
                            action=step.action,
                            label=label,
                            status=status,
                            error=error,
                            snapshot=_snapshot_harness(harness),
                        )
                    )
                )
                if status == "failed":
                    break

            if self._output_dir is not None:
                (self._output_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
            return trace
        finally:
            harness.runtime._is_dirty = False  # type: ignore[attr-defined]
            harness.shutdown()

    def _execute_step(self, harness: AppFlowHarness, *, action: str, params: dict[str, object]) -> None:
        if action == "trigger_action":
            harness.trigger_action(str(params["action_id"]))
        elif action == "select_first_event":
            _click_first_event(harness, layer_id=str(params["layer_id"]) if "layer_id" in params else None)
        elif action == "nudge_selected_events":
            direction = str(params["direction"])
            steps = int(params.get("steps", 1))
            modifiers = Qt.KeyboardModifier.ShiftModifier if steps >= 10 else Qt.KeyboardModifier.NoModifier
            key = Qt.Key.Key_Left if direction == "left" else Qt.Key.Key_Right
            QTest.keyClick(harness.widget._canvas, key, modifiers)
            QApplication.processEvents()
        elif action == "duplicate_selected_events":
            steps = int(params.get("steps", 1))
            modifiers = Qt.KeyboardModifier.ControlModifier
            if steps >= 10:
                modifiers |= Qt.KeyboardModifier.ShiftModifier
            QTest.keyClick(harness.widget._canvas, Qt.Key.Key_D, modifiers)
            QApplication.processEvents()
        elif action == "open_push_surface":
            _click_layer_surface(harness, surface="push", layer_id=str(params["layer_id"]) if "layer_id" in params else None)
        elif action == "open_pull_surface":
            _click_layer_surface(harness, surface="pull", layer_id=str(params["layer_id"]) if "layer_id" in params else None)
        elif action == "enable_sync":
            harness.enable_sync()
        elif action == "disable_sync":
            harness.disable_sync()
        elif action == "screenshot":
            if self._output_dir is None:
                raise ValueError("screenshot action requires an output_dir")
            screenshot_path = self._output_dir / str(params["filename"])
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            if not harness.widget.grab().save(str(screenshot_path)):
                raise RuntimeError(f"Failed to save screenshot: {screenshot_path}")
        else:
            raise ValueError(f"Unsupported action: {action}")


def run_scenario_file(*, scenario_path: str | Path, output_dir: str | Path | None = None) -> list[dict[str, object]]:
    scenario = load_scenario(scenario_path)
    resolved_output = None if output_dir is None else Path(output_dir)
    return GuiLaneBRunner(scenario=scenario, output_dir=resolved_output).run()


def _build_timeline_from_fixture() -> Timeline:
    presentation = load_realistic_timeline_fixture()
    layers: list[Layer] = []
    for order_index, layer_presentation in enumerate(presentation.layers):
        layer_id = LayerId(str(layer_presentation.layer_id))
        main_take_id = layer_presentation.main_take_id or TakeId(f"{layer_id}:main")
        takes = [
            Take(
                id=TakeId(str(main_take_id)),
                layer_id=layer_id,
                name="Main",
                events=[
                    Event(
                        id=EventId(str(event.event_id)),
                        take_id=TakeId(str(main_take_id)),
                        start=float(event.start),
                        end=float(event.end),
                        label=event.label,
                        color=event.color,
                        muted=bool(event.muted),
                    )
                    for event in layer_presentation.events
                ],
            )
        ]
        for take_presentation in layer_presentation.takes:
            take_id = TakeId(str(take_presentation.take_id))
            takes.append(
                Take(
                    id=take_id,
                    layer_id=layer_id,
                    name=take_presentation.name,
                    source_ref=take_presentation.source_ref,
                    events=[
                        Event(
                            id=EventId(str(event.event_id)),
                            take_id=take_id,
                            start=float(event.start),
                            end=float(event.end),
                            label=event.label,
                            color=event.color,
                            muted=bool(event.muted),
                        )
                        for event in take_presentation.events
                    ],
                )
            )
        layers.append(
            Layer(
                id=layer_id,
                timeline_id=TimelineId(str(presentation.timeline_id)),
                name=layer_presentation.title,
                kind=layer_presentation.kind,
                order_index=order_index,
                takes=takes,
                mixer=LayerMixerState(
                    mute=layer_presentation.muted,
                    solo=layer_presentation.soloed,
                    gain_db=layer_presentation.gain_db,
                    pan=layer_presentation.pan,
                ),
                sync=LayerSyncState(
                    mode=layer_presentation.sync_mode.value,
                    connected=layer_presentation.sync_connected,
                    target_ref=layer_presentation.sync_target_label or None,
                    ma3_track_coord=layer_presentation.sync_target_label or None,
                    live_sync_state=layer_presentation.live_sync_state,
                    live_sync_pause_reason=layer_presentation.live_sync_pause_reason or None,
                    live_sync_divergent=layer_presentation.live_sync_divergent,
                ),
                status=LayerStatus(
                    stale=layer_presentation.status.stale,
                    manually_modified=layer_presentation.status.manually_modified,
                    stale_reason=layer_presentation.status.stale_reason or None,
                ),
                presentation_hints=LayerPresentationHints(
                    visible=layer_presentation.visible,
                    locked=layer_presentation.locked,
                    expanded=layer_presentation.is_expanded,
                    color=layer_presentation.color,
                ),
            )
        )
    return Timeline(
        id=TimelineId(str(presentation.timeline_id)),
        song_version_id=SongVersionId("song_version_lane_b"),
        layers=layers,
        selection=TimelineSelection(
            selected_layer_id=LayerId(str(presentation.selected_layer_id)) if presentation.selected_layer_id is not None else None,
            selected_layer_ids=[LayerId(str(layer_id)) for layer_id in presentation.selected_layer_ids],
            selected_take_id=TakeId(str(presentation.selected_take_id)) if presentation.selected_take_id is not None else None,
            selected_event_ids=[EventId(str(event_id)) for event_id in presentation.selected_event_ids],
        ),
        viewport=TimelineViewport(
            pixels_per_second=presentation.pixels_per_second,
            scroll_x=presentation.scroll_x,
            scroll_y=presentation.scroll_y,
        ),
    )


def _render_for_hit_testing(harness: AppFlowHarness) -> None:
    harness.widget.resize(1200, 720)
    harness.widget.show()
    harness.widget.activateWindow()
    harness.widget.setFocus()
    harness.widget.repaint()
    QApplication.processEvents()
    harness.widget._canvas.repaint()
    QApplication.processEvents()


def _click_rect(widget, rect) -> None:
    center = rect.center().toPoint()
    QTest.mouseClick(widget._canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(center.x(), center.y()))
    QApplication.processEvents()


def _click_first_event(harness: AppFlowHarness, *, layer_id: str | None) -> None:
    for rect, candidate_layer_id, _take_id, _event_id in harness.widget._canvas._event_rects:
        if layer_id is None or str(candidate_layer_id) == layer_id:
            _click_rect(harness.widget, rect)
            return
    if layer_id is None:
        raise AssertionError("No event rects are available")
    raise AssertionError(f"No event rect found for layer_id={layer_id}")


def _click_layer_surface(harness: AppFlowHarness, *, surface: str, layer_id: str | None) -> None:
    rects = harness.widget._canvas._push_rects if surface == "push" else harness.widget._canvas._pull_rects
    for rect, candidate_layer_id in rects:
        if layer_id is None or str(candidate_layer_id) == layer_id:
            _click_rect(harness.widget, rect)
            return
    if layer_id is None:
        raise AssertionError(f"No {surface} rects are available")
    raise AssertionError(f"No {surface} rect found for layer_id={layer_id}")


def _snapshot_harness(harness: AppFlowHarness) -> dict[str, object]:
    presentation = harness.presentation()
    event_counts = {str(layer.layer_id): len(layer.events) for layer in presentation.layers}
    return {
        "selected_layer_id": None if presentation.selected_layer_id is None else str(presentation.selected_layer_id),
        "selected_take_id": None if presentation.selected_take_id is None else str(presentation.selected_take_id),
        "selected_event_ids": [str(event_id) for event_id in presentation.selected_event_ids],
        "event_counts": event_counts,
        "push_mode_active": bool(presentation.manual_push_flow.push_mode_active),
        "pull_workspace_active": bool(presentation.manual_pull_flow.workspace_active),
        "batch_transfer_plan_id": None if presentation.batch_transfer_plan is None else presentation.batch_transfer_plan.plan_id,
        "sync_connected": bool(harness.runtime.session.sync_state.connected),
        "sync_mode": harness.runtime.session.sync_state.mode.value,
    }

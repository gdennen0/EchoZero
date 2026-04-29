from __future__ import annotations

import shutil
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from echozero.application.presentation.inspector_contract import (
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ranges import TimeRange
from echozero.application.timeline.intents import CreateEvent, Play, ToggleLayerExpanded
from echozero.application.timeline.ma3_push_intents import (
    CreateMA3Sequence,
    MA3PushApplyMode,
    MA3SequenceCreationMode,
)
from echozero.audio.engine import AudioEngine
from echozero.testing.analysis_mocks import (
    build_mock_analysis_service,
    write_test_tone_wav,
    write_test_wav,
)
from echozero.testing.app_flow import AppFlowHarness
from echozero.ui.qt.timeline.widget_action_ma3_push_mixin import _ManualPushRoutePopupResult
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_app_flow_harness")


def _repo_local_temp_root() -> Path:
    root = _TEST_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


class _FakeStream:
    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def _fake_stream_factory(**kwargs):
    return _FakeStream(**kwargs)


def _install_deterministic_runtime_audio(
    harness: AppFlowHarness,
) -> TimelineRuntimeAudioController:
    runtime_audio = TimelineRuntimeAudioController(
        engine=AudioEngine(stream_factory=_fake_stream_factory)
    )
    harness.install_runtime_audio(runtime_audio)
    return runtime_audio


def _monitor_layer(runtime_audio: TimelineRuntimeAudioController):
    return runtime_audio.engine.mixer.get_layer(TimelineRuntimeAudioController._MONITOR_LAYER_ID)


def _route_monitor_to_layer(harness: AppFlowHarness, layer_id) -> None:
    harness.widget.set_presentation(
        replace(
            harness.runtime.presentation(),
            active_playback_layer_id=layer_id,
            active_playback_take_id=None,
        )
    )
    harness._app.processEvents()


def _contract_action(harness: AppFlowHarness, action_id: str):
    contract = build_timeline_inspector_contract(harness.widget.presentation)
    return next(
        action
        for section in contract.context_sections
        for action in section.actions
        if action.action_id == action_id
    )


def _layer_contract_action(harness: AppFlowHarness, layer_id, action_id: str):
    contract = build_timeline_inspector_contract(
        harness.widget.presentation,
        hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=layer_id),
    )
    return next(
        action
        for section in contract.context_sections
        for action in section.actions
        if action.action_id == action_id
    )


def test_app_flow_harness_dispatch_and_launcher_actions():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working")

    try:
        harness.runtime.add_song_from_path(
            "Harness Song",
            write_test_wav(temp_root / "fixtures" / "harness-song.wav"),
        )
        harness.widget.set_presentation(harness.runtime.presentation())
        harness._app.processEvents()

        layer_id = harness.presentation().layers[0].layer_id

        harness.dispatch(ToggleLayerExpanded(layer_id))
        assert harness.is_dirty is True

        save_path = harness.queue_save_path(temp_root / "saved-project.ez")
        harness.trigger_action("save_as")

        assert harness.project_path == save_path
        assert save_path.exists()
        assert harness.is_dirty is False

        harness.trigger_action("new")
        assert harness.project_path is None
        assert harness.is_dirty is False
        assert harness.presentation().title == "EchoZero Project"
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_exposes_launcher_menus():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working-menus")

    try:
        menu_bar = harness.widget._launcher_menu_bar
        assert menu_bar.isHidden() is False
        assert [action.text() for action in menu_bar.actions()] == ["&File", "&Edit"]

        file_menu = menu_bar.actions()[0].menu()
        edit_menu = menu_bar.actions()[1].menu()

        assert file_menu is not None
        assert [
            action.text()
            for action in file_menu.actions()
            if action.isSeparator() is False
        ] == [
            "&New Project",
            "&Open Project",
            "Open &Recent Project",
            "&Save Project",
            "Save Project &As...",
            "Enable &Phone Review Service",
            "Disable Phone Review Service",
            "Open &Questionable Review",
            "Open &All-Events Review",
            "Open Latest Review Dataset &Folder",
            "Open Latest Review Dataset &Record",
            "Create Project &Specialized Model",
        ]
        assert edit_menu is not None
        assert [
            action.text()
            for action in edit_menu.actions()
            if action.isSeparator() is False
        ] == ["&Undo", "&Redo"]
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_sync_with_simulated_ma3_connects_bridge():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(simulate_ma3=True, working_dir_root=temp_root / "working-sync")

    try:
        state = harness.enable_sync()
        assert harness.ma3_bridge is not None
        assert state.connected is True
        assert state.mode.value == "ma3"
        assert harness.ma3_bridge.connected is True
        assert harness.ma3_bridge.connect_calls == 1

        disabled = harness.disable_sync()
        assert disabled.connected is False
        assert disabled.mode.value == "none"
        assert harness.ma3_bridge.connected is False
        assert harness.ma3_bridge.disconnect_calls == 1
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_sync_push_transfer_updates_simulated_ma3_snapshot(monkeypatch):
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(simulate_ma3=True, working_dir_root=temp_root / "working-sync-push")

    try:
        harness.enable_sync()
        presentation = harness.runtime.add_layer(LayerKind.EVENT, "Push Layer")
        harness.widget.set_presentation(presentation)
        harness._app.processEvents()

        layer_id = harness.presentation().layers[0].layer_id
        harness.dispatch(
            CreateEvent(
                layer_id=layer_id,
                take_id=None,
                time_range=TimeRange(1.0, 1.5),
            )
        )

        def _choose_ma3_push_option(_parent, _title, prompt, items, *_args):
            if "has no assigned MA3 sequence" in prompt:
                return ("Create next available sequence", True)
            raise AssertionError(f"Unexpected MA3 push prompt: {prompt}")

        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget_action_ma3_push_mixin."
            "TimelineWidgetMA3PushActionMixin._open_manual_push_route_popup",
            lambda *_args, **_kwargs: _ManualPushRoutePopupResult(
                target_track_coord="tc1_tg2_tr4",
                sequence_action=CreateMA3Sequence(
                    creation_mode=MA3SequenceCreationMode.NEXT_AVAILABLE,
                    preferred_name="Harness Song - Push Layer",
                ),
                apply_mode=MA3PushApplyMode.MERGE,
            ),
        )
        monkeypatch.setattr(
            "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
            _choose_ma3_push_option,
        )
        harness.widget._trigger_contract_action(
            _layer_contract_action(harness, layer_id, "send_layer_to_ma3")
        )

        assert harness.ma3_bridge is not None
        remote_events = harness.ma3_bridge.list_track_events("tc1_tg2_tr4")
        assert [event.label for event in remote_events] == ["Event", "Cue 9"]
        layer = next(layer for layer in harness.presentation().layers if layer.layer_id == layer_id)
        assert layer.sync_target_label == "tc1_tg2_tr4"
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_layer_contract_hides_pull_actions_from_ui_surface():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(simulate_ma3=True, working_dir_root=temp_root / "working-sync-pull")

    try:
        presentation = harness.runtime.add_layer(LayerKind.EVENT, "Pull Target")
        harness.widget.set_presentation(presentation)
        harness._app.processEvents()

        layer_id = harness.presentation().layers[0].layer_id
        harness.dispatch(
            CreateEvent(
                layer_id=layer_id,
                take_id=None,
                time_range=TimeRange(0.25, 0.5),
            )
        )
        contract = build_timeline_inspector_contract(
            harness.widget.presentation,
            hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=layer_id),
        )
        action_ids = {
            action.action_id for section in contract.context_sections for action in section.actions
        }

        assert "route_layer_to_ma3_track" in action_ids
        assert "send_layer_to_ma3" in action_ids
        assert "send_selected_events_to_ma3" in action_ids
        assert "send_to_different_track_once" in action_ids
        assert "pull_from_ma3" in action_ids
        assert "select_pull_source_tracks" not in action_ids
        assert "transfer.plan_apply" not in action_ids
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_osc_loopback_helpers():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(simulate_ma3_osc=True, working_dir_root=temp_root / "working-osc")

    try:
        harness.send_ma3_osc("/cmd", 7, "flash")
        capture = harness.wait_for_ma3_osc("/cmd", timeout=1.0)

        assert harness.ma3_osc_loopback is not None
        assert harness.ma3_osc_loopback.is_running is True
        assert capture is not None
        assert capture.args == (7, "flash")
        assert [message.path for message in harness.ma3_osc_messages()] == ["/cmd"]

        harness.clear_ma3_osc()
        assert harness.ma3_osc_messages() == []
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_shutdown_stops_osc_loopback():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(
        simulate_ma3_osc=True, working_dir_root=temp_root / "working-osc-stop"
    )

    loopback = harness.ma3_osc_loopback
    assert loopback is not None
    thread = loopback.thread

    harness.shutdown()

    try:
        assert loopback.is_running is False
        assert thread is not None
        assert thread.is_alive() is False
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_playback_advances_playhead_with_runtime_audio():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working-playback")

    try:
        runtime_audio = _install_deterministic_runtime_audio(harness)
        harness.runtime.add_song_from_path(
            "Playback Song",
            write_test_tone_wav(
                temp_root / "fixtures" / "playback-song.wav",
                duration_seconds=1.5,
                frequency_hz=220.0,
            ),
        )
        harness.widget.set_presentation(harness.runtime.presentation())
        harness._app.processEvents()

        harness.dispatch(Play())
        start = harness.widget.presentation.playhead
        harness.advance_playback(iterations=8)
        end = harness.widget.presentation.playhead

        assert runtime_audio.is_playing() is True
        assert end > start
        assert harness.widget.presentation.current_time_label != "00:00.00"
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_route_switch_keeps_transport_running_and_advances_playhead():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(
        working_dir_root=temp_root / "working-switch",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        runtime_audio = _install_deterministic_runtime_audio(harness)
        harness.runtime.add_song_from_path(
            "Route Song",
            write_test_tone_wav(
                temp_root / "fixtures" / "route-song.wav", duration_seconds=1.5, frequency_hz=220.0
            ),
        )
        after_stems = harness.runtime.extract_stems("source_audio")
        harness.widget.set_presentation(after_stems)
        harness._app.processEvents()

        drums_layer = next(
            layer for layer in harness.presentation().layers if layer.title == "Drums"
        )

        harness.dispatch(Play())
        harness.advance_playback(iterations=6)
        before_switch = harness.widget.presentation.playhead
        source_monitor = _monitor_layer(runtime_audio)
        assert source_monitor is not None
        assert source_monitor.name == "Route Song"

        _route_monitor_to_layer(harness, drums_layer.layer_id)
        assert runtime_audio.is_playing() is True
        drums_monitor = _monitor_layer(runtime_audio)
        assert drums_monitor is not None
        assert drums_monitor.name == "Drums"

        harness.advance_playback(iterations=6)
        after_switch = harness.widget.presentation.playhead

        assert after_switch > before_switch
        assert after_switch > 0.0
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_derived_audio_layers_produce_distinct_playback_output():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(
        working_dir_root=temp_root / "working-derived",
        analysis_service=build_mock_analysis_service(),
    )

    try:
        runtime_audio = _install_deterministic_runtime_audio(harness)
        harness.runtime.add_song_from_path(
            "Derived Playback Song",
            write_test_tone_wav(
                temp_root / "fixtures" / "derived-song.wav",
                duration_seconds=1.5,
                frequency_hz=220.0,
            ),
        )
        after_stems = harness.runtime.extract_stems("source_audio")
        harness.widget.set_presentation(after_stems)
        harness._app.processEvents()

        drums_layer = next(
            layer for layer in harness.presentation().layers if layer.title == "Drums"
        )
        bass_layer = next(
            layer for layer in harness.presentation().layers if layer.title == "Bass"
        )

        _route_monitor_to_layer(harness, drums_layer.layer_id)
        drums_mix = runtime_audio.engine.mixer.read_mix(0, 512)
        drums_monitor = _monitor_layer(runtime_audio)

        _route_monitor_to_layer(harness, bass_layer.layer_id)
        bass_mix = runtime_audio.engine.mixer.read_mix(0, 512)
        bass_monitor = _monitor_layer(runtime_audio)

        assert drums_monitor is not None
        assert drums_monitor.name == "Drums"
        assert bass_monitor is not None
        assert bass_monitor.name == "Bass"
        assert np.max(np.abs(drums_mix)) > 0.0
        assert np.max(np.abs(bass_mix)) > 0.0
        assert not np.allclose(drums_mix, bass_mix)
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


@pytest.mark.slow
def test_app_flow_harness_real_pipeline_playback_survives_active_song_switch():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working-real-switch")

    try:
        runtime_audio = _install_deterministic_runtime_audio(harness)

        first_audio = write_test_tone_wav(
            temp_root / "fixtures" / "real-switch-song-1.wav",
            duration_seconds=1.2,
            frequency_hz=220.0,
        )
        second_audio = write_test_tone_wav(
            temp_root / "fixtures" / "real-switch-song-2.wav",
            duration_seconds=1.2,
            frequency_hz=440.0,
        )

        harness.runtime.add_song_from_path("Switch Song A", first_audio)
        song_a_id = str(harness.runtime.session.active_song_id)
        harness.runtime.extract_stems("source_audio")

        harness.runtime.add_song_from_path("Switch Song B", second_audio)
        song_b_id = str(harness.runtime.session.active_song_id)
        harness.runtime.extract_stems("source_audio")

        presentation = harness.runtime.select_song(song_a_id)
        harness.widget.set_presentation(presentation)
        harness._app.processEvents()

        assert len(harness.presentation().layers) >= 5
        assert harness.presentation().layers[0].title == "Switch Song A"

        harness.dispatch(Play())
        harness.advance_playback(iterations=6)
        before_switch = harness.widget.presentation.playhead
        monitor_a = _monitor_layer(runtime_audio)

        switched = harness.runtime.select_song(song_b_id)
        harness.widget.set_presentation(switched)
        harness._app.processEvents()

        assert runtime_audio.is_playing() is True
        monitor_b = _monitor_layer(runtime_audio)
        harness.advance_playback(iterations=6)
        after_switch = harness.widget.presentation.playhead

        assert monitor_a is not None
        assert monitor_a.name == "Switch Song A"
        assert monitor_b is not None
        assert monitor_b.name == "Switch Song B"
        assert str(harness.runtime.session.active_song_id) == song_b_id
        assert harness.presentation().layers[0].title == "Switch Song B"
        assert after_switch > before_switch > 0.0
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


@pytest.mark.slow
def test_app_flow_harness_real_pipeline_playback_routes_main_then_each_stem():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(working_dir_root=temp_root / "working-real-route-sequence")

    try:
        runtime_audio = _install_deterministic_runtime_audio(harness)
        audio_path = write_test_tone_wav(
            temp_root / "fixtures" / "playback-route-demo-real-long-audio.wav",
            frequency_hz=220.0,
            duration_seconds=24.0,
            amplitude=0.45,
        )

        harness.runtime.add_song_from_path("Route Sequence Song", audio_path)
        harness.runtime.extract_stems("source_audio")
        harness.widget.set_presentation(harness.runtime.presentation())
        harness._app.processEvents()

        expected_titles = ["Route Sequence Song", "Drums", "Bass", "Vocals", "Other"]
        routed_titles: list[str] = []
        playheads: list[float] = []

        harness.dispatch(Play())

        for expected_title, layer in zip(expected_titles, harness.presentation().layers):
            _route_monitor_to_layer(harness, layer.layer_id)
            harness.advance_playback(iterations=10)
            monitor = _monitor_layer(runtime_audio)

            assert monitor is not None
            routed_titles.append(monitor.name)
            playheads.append(harness.widget.presentation.playhead)

        assert runtime_audio.is_playing() is True
        assert routed_titles == expected_titles
        assert all(current > previous for previous, current in zip(playheads, playheads[1:]))
        assert playheads[0] > 0.0
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)

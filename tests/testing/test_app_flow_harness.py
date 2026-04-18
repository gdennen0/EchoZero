from __future__ import annotations

from dataclasses import replace
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from echozero.application.timeline.intents import Play, ToggleLayerExpanded
from echozero.audio.engine import AudioEngine
from echozero.testing.analysis_mocks import build_mock_analysis_service, write_test_tone_wav, write_test_wav
from echozero.testing.app_flow import AppFlowHarness
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


def _install_deterministic_runtime_audio(harness: AppFlowHarness) -> TimelineRuntimeAudioController:
    runtime_audio = TimelineRuntimeAudioController(engine=AudioEngine(stream_factory=_fake_stream_factory))
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


def test_app_flow_harness_osc_loopback_helpers():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(simulate_ma3_osc=True, working_dir_root=temp_root / "working-osc")

    try:
        harness.send_ma3_osc("/ma3/exec", 7, "flash")
        capture = harness.wait_for_ma3_osc("/ma3/exec", timeout=1.0)

        assert harness.ma3_osc_loopback is not None
        assert harness.ma3_osc_loopback.is_running is True
        assert capture is not None
        assert capture.args == (7, "flash")
        assert [message.path for message in harness.ma3_osc_messages()] == ["/ma3/exec"]

        harness.clear_ma3_osc()
        assert harness.ma3_osc_messages() == []
    finally:
        harness.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_flow_harness_shutdown_stops_osc_loopback():
    temp_root = _repo_local_temp_root()
    harness = AppFlowHarness(simulate_ma3_osc=True, working_dir_root=temp_root / "working-osc-stop")

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
            write_test_tone_wav(temp_root / "fixtures" / "playback-song.wav", duration_seconds=1.5, frequency_hz=220.0),
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
            write_test_tone_wav(temp_root / "fixtures" / "route-song.wav", duration_seconds=1.5, frequency_hz=220.0),
        )
        after_stems = harness.runtime.extract_stems("source_audio")
        harness.widget.set_presentation(after_stems)
        harness._app.processEvents()

        drums_layer = next(layer for layer in harness.presentation().layers if layer.title == "Drums")

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
            write_test_tone_wav(temp_root / "fixtures" / "derived-song.wav", duration_seconds=1.5, frequency_hz=220.0),
        )
        after_stems = harness.runtime.extract_stems("source_audio")
        harness.widget.set_presentation(after_stems)
        harness._app.processEvents()

        drums_layer = next(layer for layer in harness.presentation().layers if layer.title == "Drums")
        bass_layer = next(layer for layer in harness.presentation().layers if layer.title == "Bass")

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

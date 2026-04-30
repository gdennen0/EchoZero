"""Runtime-audio app-shell flow support cases.
Exists to isolate runtime-audio rebuild and canonical build assertions from project and pipeline support tests.
Connects the compatibility wrapper to the bounded audio support slice.
"""

import shutil

from echozero.application.settings import AudioOutputRuntimeConfig
from echozero.application.shared.enums import LayerKind
from echozero.application.timeline.intents import Play, SelectEvent
from echozero.audio.engine import AudioEngine
from echozero.testing.analysis_mocks import build_mock_analysis_service, write_test_wav
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.app_shell_runtime_services import build_runtime_audio_controller
from echozero.ui.qt.timeline.runtime_audio import TimelineRuntimeAudioController
from tests.ui.app_shell_runtime_flow_shared_support import (
    _CountedRuntimeAudio,
    _repo_local_temp_root,
)
from tests.ui.runtime_audio_shared_support import _fake_stream_factory


def test_app_shell_runtime_add_song_from_path_defers_runtime_audio_build_until_playback():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    counted = _CountedRuntimeAudio()
    runtime.runtime_audio = counted

    try:
        assert counted.build_calls == 0

        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav")
        )

        assert counted.build_calls == 0
        assert runtime.presentation().layers[0].title == "Imported Song"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_apply_audio_output_config_rebuilds_runtime_audio_controller():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        original_runtime_audio = runtime.runtime_audio

        runtime.apply_audio_output_config(
            AudioOutputRuntimeConfig(
                output_device=3,
                sample_rate=48000,
                channels=2,
                stream_latency="low",
                stream_blocksize=512,
                prime_output_buffers_using_stream_callback=False,
            )
        )

        rebuilt_runtime_audio = runtime.runtime_audio

        assert rebuilt_runtime_audio is not None
        assert rebuilt_runtime_audio is not original_runtime_audio
        assert rebuilt_runtime_audio.engine.sample_rate == 48000
        assert rebuilt_runtime_audio.engine._channels == 2
        assert rebuilt_runtime_audio.engine._output_device == 3
        assert rebuilt_runtime_audio.engine._stream_latency == "low"
        assert rebuilt_runtime_audio.engine._stream_blocksize == 512
        assert rebuilt_runtime_audio.engine._prime_output_buffers_using_stream_callback is False
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_build_runtime_audio_controller_defaults_to_engine_continuous_audio():
    controller = build_runtime_audio_controller()

    try:
        assert controller is not None
    finally:
        controller.shutdown()


def test_app_shell_runtime_add_layer_after_song_defers_runtime_audio_build_while_stopped():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    counted = _CountedRuntimeAudio()
    runtime.runtime_audio = counted

    try:
        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav")
        )
        counted.build_calls = 0

        runtime.add_layer(LayerKind.EVENT, "Event Layer")

        assert counted.build_calls == 0
        assert any(layer.title == "Event Layer" for layer in runtime.presentation().layers)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_layer_after_song_rebuilds_runtime_audio_while_playing():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    counted = _CountedRuntimeAudio()
    runtime.runtime_audio = counted

    try:
        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav")
        )
        runtime.dispatch(Play())
        counted.build_calls = 0

        runtime.add_layer(LayerKind.EVENT, "Event Layer")

        assert counted.build_calls == 1
        assert any(layer.title == "Event Layer" for layer in runtime.presentation().layers)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_song_after_draft_layer_keeps_draft_above_source():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_layer(LayerKind.EVENT, "Draft Layer")
        presentation = runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "draft-import.wav")
        )

        assert [layer.title for layer in presentation.layers[:2]] == [
            "Draft Layer",
            "Imported Song",
        ]

        source_layer = next(layer for layer in presentation.layers if layer.title == "Imported Song")
        assert presentation.selected_layer_id == source_layer.layer_id
        assert presentation.selected_layer_id == source_layer.layer_id
        assert runtime.session.active_song_version_id is not None
        assert [
            layer.name
            for layer in runtime.project_storage.layers.list_by_version(
                str(runtime.session.active_song_version_id)
            )
        ] == ["Draft Layer"]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_layer_after_song_survives_storage_refresh():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "refresh-import.wav")
        )
        runtime.add_layer(LayerKind.EVENT, "Event Layer")

        song_id = runtime.session.active_song_id
        song_version_id = runtime.session.active_song_version_id
        assert song_id is not None
        assert song_version_id is not None

        runtime._refresh_from_storage(
            active_song_id=song_id,
            active_song_version_id=song_version_id,
        )
        refreshed = runtime.presentation()

        assert [layer.title for layer in refreshed.layers[:2]] == [
            "Imported Song",
            "Event Layer",
        ]
        assert refreshed.selected_layer_id == refreshed.layers[1].layer_id
        assert [
            layer.name
            for layer in runtime.project_storage.layers.list_by_version(str(song_version_id))
        ] == ["Event Layer"]
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_layer_clears_stale_selected_event_refs():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "add-layer-clear.wav")
        )
        after_stems = runtime.extract_stems("source_audio")
        drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")
        first_pass = runtime.extract_drum_events(drums_layer.layer_id)
        onsets_layer = next(layer for layer in first_pass.layers if layer.title == "Onsets")

        selected = runtime.dispatch(
            SelectEvent(
                onsets_layer.layer_id,
                onsets_layer.main_take_id,
                onsets_layer.events[0].event_id,
            )
        )
        assert selected.selected_event_refs

        added = runtime.add_layer(LayerKind.EVENT, "Notes")
        added_layer = next(layer for layer in added.layers if layer.title == "Notes")
        updated_onsets = next(layer for layer in added.layers if layer.title == "Onsets")

        assert added.selected_layer_id == added_layer.layer_id
        assert added.selected_event_ids == []
        assert added.selected_event_refs == []
        assert all(event.is_selected is False for event in updated_onsets.events)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_play_dispatch_rebuilds_runtime_audio():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    counted = _CountedRuntimeAudio()
    runtime.runtime_audio = counted

    try:
        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav")
        )
        counted.build_calls = 0
        runtime.dispatch(Play())
        assert counted.build_calls == 1
        assert counted.play_calls == 1
        assert runtime.presentation().is_playing is True
        assert runtime.session.transport_state.is_playing is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_add_song_syncs_backend_playback_state_metadata():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.runtime_audio = TimelineRuntimeAudioController(
            engine=AudioEngine(stream_factory=_fake_stream_factory),
        )
        runtime.add_song_from_path(
            "Imported Song", write_test_wav(temp_root / "fixtures" / "import.wav")
        )
        runtime.dispatch(Play())

        assert runtime.session.playback_state.backend_name == "sounddevice"
        assert (
            runtime.session.playback_state.active_layer_id
            == runtime.presentation().selected_layer_id
        )
        assert runtime.session.playback_state.active_sources
        assert runtime.session.playback_state.output_sample_rate > 0
        assert runtime.session.playback_state.output_channels > 0
        assert runtime.session.playback_state.diagnostics.output_device == "default"
        assert runtime.session.playback_state.diagnostics.last_transition == "play"
        assert runtime.session.playback_state.diagnostics.last_track_sync_reason != ""
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_canonical_build_starts_with_native_empty_timeline_state():
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"

    runtime = build_app_shell(
        working_dir_root=working_root,
        initial_project_name="Native Baseline",
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        presentation = runtime.presentation()

        assert presentation.title == "Native Baseline"
        assert presentation.timeline_id == f"timeline_{runtime.project_storage.project.id}"
        assert runtime.session.project_id == runtime.project_storage.project.id
        assert runtime.session.active_song_id is None
        assert runtime.session.active_song_version_id is None
        assert runtime.session.active_timeline_id == presentation.timeline_id
        assert presentation.layers == []
        assert presentation.selected_layer_id is None
        assert presentation.selected_layer_ids == []
        assert presentation.playhead == 0.0
        assert presentation.current_time_label == "00:00.00"
        assert presentation.end_time_label == "00:00.00"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)

__all__ = [name for name in globals() if name.startswith("test_")]

"""Process-isolated runtime-audio IPC support cases.
Exists to verify the app-facing playback client/service contract under the new isolated runtime.
Connects process lifecycle, IPC envelope behavior, and diagnostics metadata to runtime audio flows.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from echozero.application.playback.models import (
    PlaybackDiagnostics,
    PlaybackState,
    PlaybackTimingSnapshot,
)
from echozero.application.playback.process_client import ProcessPlaybackClient
from echozero.application.playback.process_service import PlaybackProcessService
from echozero.application.playback.process_shared import (
    PlaybackIpcError,
    decode_playback_state,
    encode_playback_state,
    encode_timing_snapshot,
)
from echozero.application.settings import AudioOutputRuntimeConfig
from echozero.application.shared.ids import LayerId
from echozero.ui.qt.timeline.demo_app import build_demo_app


@pytest.fixture
def process_runtime_audio() -> ProcessPlaybackClient:
    client = ProcessPlaybackClient()
    try:
        yield client
    finally:
        client.shutdown()


def test_process_runtime_audio_health_and_version(
    process_runtime_audio: ProcessPlaybackClient,
) -> None:
    health = process_runtime_audio.health()

    assert bool(health.get("ok", False)) is True
    assert str(health.get("version", "")) == "1"
    assert int(health.get("pid", 0) or 0) > 0
    assert str(health.get("ws_url", "")).startswith("ws://")


def test_process_runtime_audio_snapshot_includes_process_diagnostics(
    process_runtime_audio: ProcessPlaybackClient,
) -> None:
    presentation = build_demo_app().presentation()
    state = process_runtime_audio.snapshot_state(presentation)

    assert state.diagnostics.audio_process_connected is True
    assert state.diagnostics.audio_process_pid is not None
    assert state.diagnostics.ipc_rtt_ms >= 0.0
    assert state.diagnostics.latency_profile == "low"


def test_process_runtime_audio_accepts_compact_sync_and_signature(
    process_runtime_audio: ProcessPlaybackClient,
) -> None:
    presentation = build_demo_app().presentation()

    process_runtime_audio.sync_structure_state(presentation)
    process_runtime_audio.sync_mix_state(presentation)

    signature = process_runtime_audio.presentation_signature(presentation)

    assert isinstance(signature, tuple)


def test_process_runtime_audio_snapshot_state_uses_request_payload_authoritatively(
    process_runtime_audio: ProcessPlaybackClient,
) -> None:
    presentation = build_demo_app().presentation()
    assert len(presentation.layers) >= 2
    base = replace(
        presentation,
        selected_layer_id=LayerId(str(presentation.layers[0].layer_id)),
    )
    updated = replace(
        presentation,
        selected_layer_id=LayerId(str(presentation.layers[1].layer_id)),
    )

    process_runtime_audio.sync_structure_state(base)
    state_base = process_runtime_audio.snapshot_state(base)
    state_updated = process_runtime_audio.snapshot_state(updated)

    assert state_base.active_layer_id == base.selected_layer_id
    assert state_updated.active_layer_id == updated.selected_layer_id


def test_process_client_transport_ipc_updates_timing_snapshots(monkeypatch) -> None:
    client = ProcessPlaybackClient.__new__(ProcessPlaybackClient)
    client._shutdown = False
    commands: list[tuple[str, dict[str, object]]] = []
    state = {"playing": False, "seconds": 0.0}

    def _fake_command(operation: str, params: dict[str, object]) -> dict[str, object]:
        commands.append((operation, dict(params)))
        if operation == "play":
            state["playing"] = True
            return {}
        if operation == "pause":
            state["playing"] = False
            return {}
        if operation == "seek":
            state["seconds"] = float(params["position_seconds"])
            return {"position_seconds": state["seconds"]}
        if operation == "timing_snapshot":
            snapshot = PlaybackTimingSnapshot(
                audible_time_seconds=float(state["seconds"]),
                clock_time_seconds=float(state["seconds"]),
                snapshot_monotonic_seconds=10.0,
                is_playing=bool(state["playing"]),
                sample_position=24000,
                frame_index=15,
                timecode_label="00:00:00:15",
                display_label="00:00:00:15",
            )
            return {"value": encode_timing_snapshot(snapshot)}
        return {}

    monkeypatch.setattr(client, "_command", _fake_command)

    client.play()
    client.seek(0.5)
    playing_snapshot = client.timing_snapshot()
    client.pause()
    paused_snapshot = client.timing_snapshot()

    assert [operation for operation, _ in commands] == [
        "play",
        "seek",
        "timing_snapshot",
        "pause",
        "timing_snapshot",
    ]
    assert playing_snapshot.is_playing is True
    assert paused_snapshot.is_playing is False
    assert playing_snapshot.audible_time_seconds == 0.5
    assert paused_snapshot.audible_time_seconds == 0.5
    assert playing_snapshot.display_label == "00:00:00:15"


def test_process_client_reconfigure_device_serializes_hardware_settings(monkeypatch) -> None:
    client = ProcessPlaybackClient.__new__(ProcessPlaybackClient)
    client._shutdown = False
    commands: list[tuple[str, dict[str, object]]] = []

    def _fake_command(operation: str, params: dict[str, object]) -> dict[str, object]:
        commands.append((operation, dict(params)))
        return {"latency_profile": "balanced", "device_reinit_count": 2}

    monkeypatch.setattr(client, "_command", _fake_command)

    response = client.reconfigure_device(
        device_spec={
            "output_device": "Scarlett 4i4",
            "sample_rate": 48000,
            "channels": 4,
            "stream_latency": "high",
            "stream_blocksize": 512,
            "prime_output_buffers_using_stream_callback": False,
        },
        profile="balanced",
    )

    assert response == {"latency_profile": "balanced", "device_reinit_count": 2}
    assert commands == [
        (
            "reconfigure_device",
            {
                "device_spec": {
                    "output_device": "Scarlett 4i4",
                    "sample_rate": 48000,
                    "channels": 4,
                    "stream_latency": "high",
                    "stream_blocksize": 512,
                    "prime_output_buffers_using_stream_callback": False,
                },
                "profile": "balanced",
            },
        )
    ]


def test_process_client_serializes_audio_diagnostics_capture_commands(monkeypatch, tmp_path) -> None:
    client = ProcessPlaybackClient.__new__(ProcessPlaybackClient)
    client._shutdown = False
    client._audio_diagnostics_capture_status = {"active": False}
    commands: list[tuple[str, dict[str, object]]] = []

    def _fake_command(operation: str, params: dict[str, object]) -> dict[str, object]:
        commands.append((operation, dict(params)))
        if operation == "start_audio_diagnostics_capture":
            return {"active": True, "capture_id": "cap-1"}
        if operation == "stop_audio_diagnostics_capture":
            return {"active": False, "bundle_path": "artifacts/audio-diagnostics/cap-1"}
        return {}

    monkeypatch.setattr(client, "_command", _fake_command)

    started = client.start_audio_diagnostics_capture(
        output_dir=tmp_path,
        include_audio_buffers=False,
        max_audio_blocks=2,
    )
    stopped = client.stop_audio_diagnostics_capture()

    assert started == {"active": True, "capture_id": "cap-1"}
    assert stopped == {"active": False, "bundle_path": "artifacts/audio-diagnostics/cap-1"}
    assert client.audio_diagnostics_capture_status() == stopped
    assert commands == [
        (
            "start_audio_diagnostics_capture",
            {
                "output_dir": str(tmp_path),
                "include_audio_buffers": False,
                "max_audio_blocks": 2,
            },
        ),
        ("stop_audio_diagnostics_capture", {}),
    ]


def test_playback_state_ipc_round_trips_device_reinit_diagnostics() -> None:
    state = PlaybackState(
        output_sample_rate=48000,
        output_channels=4,
        diagnostics=PlaybackDiagnostics(
            output_device="Scarlett 4i4",
            stream_latency="high",
            stream_blocksize=512,
            device_reinit_count=3,
            recent_audio_runtime_events=(
                {
                    "source": "audio_engine",
                    "kind": "overlay-start",
                    "reason": "play-overlay",
                    "clock_samples": 128,
                },
            ),
        ),
    )

    restored = decode_playback_state(encode_playback_state(state))

    assert restored.output_sample_rate == 48000
    assert restored.output_channels == 4
    assert restored.diagnostics.output_device == "Scarlett 4i4"
    assert restored.diagnostics.stream_latency == "high"
    assert restored.diagnostics.stream_blocksize == 512
    assert restored.diagnostics.device_reinit_count == 3
    assert restored.diagnostics.recent_audio_runtime_events == (
        {
            "source": "audio_engine",
            "kind": "overlay-start",
            "reason": "play-overlay",
            "clock_samples": 128,
        },
    )


def test_process_service_reconfigure_device_restores_projection_time_and_play_state(
    monkeypatch,
) -> None:
    projection = object()
    rebuilt_controllers: list[_FakeDeviceReconfigureController] = []

    service = PlaybackProcessService.__new__(PlaybackProcessService)
    service._base_audio_config = AudioOutputRuntimeConfig(
        output_device="old-device",
        sample_rate=44100,
        channels=2,
        stream_latency="low",
        stream_blocksize=128,
        prime_output_buffers_using_stream_callback=True,
    )
    service._latest_projection = projection
    old_controller = _FakeDeviceReconfigureController(seconds=12.5, playing=True)
    service._controller = old_controller
    service._device_reinit_count = 0

    def _build_controller(self):
        controller = _FakeDeviceReconfigureController(seconds=0.0, playing=False)
        rebuilt_controllers.append(controller)
        return controller

    monkeypatch.setattr(PlaybackProcessService, "_build_controller", _build_controller)

    service._reconfigure_device(
        {
            "output_device": "new-device",
            "sample_rate": 48000,
            "channels": 4,
            "stream_latency": "high",
            "stream_blocksize": 512,
            "prime_output_buffers_using_stream_callback": False,
        }
    )

    rebuilt = rebuilt_controllers[0]
    assert old_controller.shutdown_called is True
    assert service._controller is rebuilt
    assert service._device_reinit_count == 1
    assert service._base_audio_config == AudioOutputRuntimeConfig(
        output_device="new-device",
        sample_rate=48000,
        channels=4,
        stream_latency="high",
        stream_blocksize=512,
        prime_output_buffers_using_stream_callback=False,
    )
    assert service._controller.synced_projection is projection
    assert service._controller.seek_seconds == 12.5
    assert service._controller.play_called is True


class _FakeDeviceReconfigureController:
    def __init__(self, *, seconds: float, playing: bool) -> None:
        self._seconds = float(seconds)
        self._playing = bool(playing)
        self.shutdown_called = False
        self.synced_projection = None
        self.seek_seconds: float | None = None
        self.play_called = False

    def current_time_seconds(self) -> float:
        return self._seconds

    def is_playing(self) -> bool:
        return self._playing

    def shutdown(self) -> None:
        self.shutdown_called = True

    def sync_structure_state(self, projection) -> None:
        self.synced_projection = projection

    def seek(self, position_seconds: float) -> None:
        self.seek_seconds = float(position_seconds)

    def play(self) -> None:
        self.play_called = True
        self._playing = True


def test_process_runtime_audio_shutdown_is_idempotent() -> None:
    client = ProcessPlaybackClient()
    client.shutdown()
    client.shutdown()

    with pytest.raises(PlaybackIpcError):
        _ = client.current_time_seconds()


def test_process_runtime_audio_spawn_passes_token_as_equals_assignment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_popen(command, stdout=None, stderr=None):
        captured["command"] = list(command)
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return SimpleNamespace()

    monkeypatch.setattr(
        "echozero.application.playback.process_client.subprocess.Popen", _fake_popen
    )

    client = ProcessPlaybackClient.__new__(ProcessPlaybackClient)
    client._host = "127.0.0.1"
    client._port = 18080
    client._ws_port = 18081
    client._token = "-leading-dash-token"
    client._audio_output_config = None
    client._audio_config_file = None

    _ = ProcessPlaybackClient._spawn_service_process(client)

    command = captured.get("command")
    assert isinstance(command, list)
    assert command[:3] == [
        sys.executable,
        "-m",
        "echozero.application.playback.process_service_entry",
    ]
    assert "--token" not in command
    assert "--token=-leading-dash-token" in command


def test_process_runtime_audio_spawn_uses_packaged_service_mode_when_frozen(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_popen(command, stdout=None, stderr=None):
        captured["command"] = list(command)
        return SimpleNamespace()

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        "echozero.application.playback.process_client.subprocess.Popen", _fake_popen
    )

    client = ProcessPlaybackClient.__new__(ProcessPlaybackClient)
    client._host = "127.0.0.1"
    client._port = 18080
    client._ws_port = 18081
    client._token = "token"
    client._audio_output_config = None
    client._audio_config_file = None

    _ = ProcessPlaybackClient._spawn_service_process(client)

    command = captured.get("command")
    assert isinstance(command, list)
    assert command[:2] == [sys.executable, "--playback-service"]
    assert "-m" not in command

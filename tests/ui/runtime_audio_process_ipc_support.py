"""Process-isolated runtime-audio IPC support cases.
Exists to verify the app-facing playback client/service contract under the new isolated runtime.
Connects process lifecycle, IPC envelope behavior, and diagnostics metadata to runtime audio flows.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from echozero.application.playback.process_client import ProcessPlaybackClient
from echozero.application.playback.process_shared import PlaybackIpcError
from echozero.ui.qt.timeline.demo_app import build_demo_app


@pytest.fixture
def process_runtime_audio() -> ProcessPlaybackClient:
    client = ProcessPlaybackClient()
    try:
        yield client
    finally:
        client.shutdown()


def test_process_runtime_audio_health_and_version(process_runtime_audio: ProcessPlaybackClient) -> None:
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

    monkeypatch.setattr("echozero.application.playback.process_client.subprocess.Popen", _fake_popen)

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
    assert "--token" not in command
    assert "--token=-leading-dash-token" in command

"""Process-isolated runtime-audio IPC support cases.
Exists to verify the app-facing playback client/service contract under the new isolated runtime.
Connects process lifecycle, IPC envelope behavior, and diagnostics metadata to runtime audio flows.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from echozero.application.playback.models import PlaybackTimingSnapshot
from echozero.application.playback.process_client import ProcessPlaybackClient
from echozero.application.playback.process_shared import PlaybackIpcError, encode_timing_snapshot
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
    assert playing_snapshot.display_label == "00:00:00:15"


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
    assert "--token" not in command
    assert "--token=-leading-dash-token" in command

"""
Playback process client: app-process adapter for isolated runtime-audio playback.
Exists because UI/app actions must not share execution contention with real-time audio callbacks.
Connects existing runtime-audio controller calls to a child playback service over HTTP and WebSocket.
"""

from __future__ import annotations

import http.client
import json
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from echozero.application.playback.models import PlaybackState, PlaybackTimingSnapshot
from echozero.application.playback.coordination import (
    TransportCommand,
    TransportCommandAction,
    TransportSnapshot,
)
from echozero.application.playback.process_shared import (
    PLAYBACK_IPC_COMMAND_PATH,
    PLAYBACK_IPC_HEALTH_PATH,
    PLAYBACK_IPC_HOST,
    PLAYBACK_IPC_TOKEN_HEADER,
    PLAYBACK_IPC_VERSION,
    PlaybackIpcError,
    decode_playback_state,
    decode_timing_snapshot,
)
from echozero.application.playback.sync_projection import PlaybackSyncPayload
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.settings import AudioOutputRuntimeConfig
from echozero.errors import InfrastructureError

try:
    from websockets.sync.client import connect as ws_connect
except ImportError as exc:  # pragma: no cover - environment contract
    raise InfrastructureError("websockets package is required for playback process IPC") from exc


class ProcessPlaybackClient:
    """Runtime-audio client that proxies calls to the playback child process."""

    _PUSHED_TIMING_STALE_SECONDS = 0.75

    _SHUTDOWN_NOOP_OPERATIONS = frozenset(
        {
            "sync_structure_state",
            "sync_mix_state",
            "drain_pending_structure_sync",
            "record_coalesced_structural_edits",
            "play",
            "pause",
            "stop",
            "seek",
            "preview_clip",
            "enqueue_structure",
            "enqueue_mix",
            "enqueue_seek",
            "enqueue_preview",
            "clear_playback_graph",
            "reconfigure_device",
            "start_audio_diagnostics_capture",
            "stop_audio_diagnostics_capture",
            "audio_diagnostics_capture_status",
            "drain_events",
            "shutdown",
        }
    )

    def __init__(
        self,
        *,
        audio_output_config: AudioOutputRuntimeConfig | None = None,
        service_start_timeout_seconds: float = 6.0,
    ) -> None:
        self._audio_output_config = audio_output_config
        self._service_start_timeout_seconds = max(1.0, float(service_start_timeout_seconds))
        self._host = PLAYBACK_IPC_HOST
        self._port = 0
        self._ws_port = 0
        self._token = secrets.token_urlsafe(24)
        self._command_counter = 0
        self._http = None
        self._process = None
        self._ws_url = ""
        self._started = False
        self._shutdown = False
        self._ws_stop_event = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._audio_config_file: Path | None = None
        self._latest_sync_payload: dict[str, object] | None = None
        self._latest_timing_snapshot: PlaybackTimingSnapshot | None = None
        self._latest_transport_snapshot: TransportSnapshot | None = None
        self._latest_timing_snapshot_received_monotonic = 0.0

        self._audio_process_connected = False
        self._audio_process_pid: int | None = None
        self._ipc_rtt_ms = 0.0
        self._last_ipc_error: str | None = None
        self._latency_profile = ""
        self._latency_profile_switch_count = 0
        self._last_latency_profile_reason: str | None = None
        self._startup_stderr: str | None = None
        self._last_local_sync_change_kind = ""
        self._last_local_projection_build_ms = 0.0
        self._last_local_sync_classify_ms = 0.0
        self._last_ipc_command = ""
        self._audio_diagnostics_capture_status: dict[str, object] = {"active": False}

        self.start()

    def start(self) -> None:
        """Start the child playback process and initialize IPC channels."""

        if self._started and not self._shutdown:
            return
        self._shutdown = False
        self._port = _reserve_local_port()
        self._ws_port = _reserve_local_port()
        while self._ws_port == self._port:
            self._ws_port = _reserve_local_port()
        self._http = http.client.HTTPConnection(self._host, self._port, timeout=3.0)
        self._process = self._spawn_service_process()
        self._wait_for_health()
        self._started = True
        self._audio_process_connected = True
        self._start_ws_listener()

    def health(self) -> dict[str, object]:
        """Return playback process health payload."""

        return self._health_request()

    def shutdown(self) -> None:
        """Stop the child playback process and release IPC resources."""

        if self._shutdown:
            return
        try:
            self._command("shutdown", {})
        except Exception:
            pass
        self._shutdown = True
        self._ws_stop_event.set()
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=1.0)
            self._ws_thread = None
        if self._process is not None:
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=1.0)
            self._process = None
        if self._http is not None:
            try:
                self._http.close()
            except Exception:
                pass
            self._http = None
        if self._audio_config_file is not None:
            self._audio_config_file.unlink(missing_ok=True)
            self._audio_config_file = None
        self._audio_process_connected = False

    def sync_structure_state(self, presentation: TimelinePresentation) -> None:
        payload = PlaybackSyncPayload.from_presentation(presentation).to_dict()
        self._latest_sync_payload = payload
        _ = self._command("sync_structure_state", {"payload": payload})

    def sync_mix_state(self, presentation: TimelinePresentation) -> None:
        payload = PlaybackSyncPayload.from_presentation(presentation).to_dict()
        self._latest_sync_payload = payload
        _ = self._command("sync_mix_state", {"payload": payload})

    def enqueue_structure(self, generation: int, presentation: TimelinePresentation) -> int:
        payload = PlaybackSyncPayload.from_presentation(presentation).to_dict()
        self._latest_sync_payload = payload
        response = self._command(
            "enqueue_structure",
            {"generation": int(generation), "payload": payload},
        )
        return int(response.get("generation", generation) or generation)

    def enqueue_mix(self, presentation: TimelinePresentation) -> None:
        payload = PlaybackSyncPayload.from_presentation(presentation).to_dict()
        self._latest_sync_payload = payload
        _ = self._command("enqueue_mix", {"payload": payload})

    def enqueue_seek(self, target_samples: int, *, seek_id: str = "") -> dict[str, object]:
        response = self._command(
            "enqueue_seek",
            {"target_samples": int(target_samples), "seek_id": str(seek_id)},
        )
        return dict(response)

    def enqueue_preview(
        self,
        *,
        source_ref: str,
        start_seconds: float,
        end_seconds: float,
        gain_db: float = 0.0,
    ) -> bool:
        response = self._command(
            "enqueue_preview",
            {
                "clip_spec": {
                    "source_ref": str(source_ref),
                    "start_seconds": float(start_seconds),
                    "end_seconds": float(end_seconds),
                    "gain_db": float(gain_db),
                }
            },
        )
        return bool(response.get("played", False))

    def reconfigure_device(
        self,
        *,
        device_spec: dict[str, object],
        profile: str = "",
    ) -> dict[str, object]:
        self._clear_latest_timing_snapshot()
        response = self._command(
            "reconfigure_device",
            {
                "device_spec": dict(device_spec),
                "profile": str(profile),
            },
        )
        self._clear_latest_timing_snapshot()
        return dict(response)

    def drain_events(self) -> list[dict[str, object]]:
        response = self._command("drain_events", {})
        events = response.get("events")
        if not isinstance(events, list):
            return []
        output: list[dict[str, object]] = []
        for item in events:
            if isinstance(item, dict):
                output.append(item)
        return output

    def start_audio_diagnostics_capture(
        self,
        *,
        output_dir: str | Path | None = None,
        include_audio_buffers: bool = True,
        max_audio_blocks: int = 64,
    ) -> dict[str, object]:
        response = self._command(
            "start_audio_diagnostics_capture",
            {
                "output_dir": str(output_dir) if output_dir is not None else None,
                "include_audio_buffers": bool(include_audio_buffers),
                "max_audio_blocks": int(max_audio_blocks),
            },
        )
        self._audio_diagnostics_capture_status = dict(response)
        return dict(response)

    def stop_audio_diagnostics_capture(self) -> dict[str, object]:
        response = self._command("stop_audio_diagnostics_capture", {})
        self._audio_diagnostics_capture_status = dict(response)
        return dict(response)

    def audio_diagnostics_capture_status(self) -> dict[str, object]:
        return dict(self._audio_diagnostics_capture_status)

    def drain_pending_structure_sync(self) -> None:
        _ = self._command("drain_pending_structure_sync", {})

    def clear_playback_graph(self, *, reason: str = "clear-playback-graph") -> None:
        self._latest_sync_payload = None
        self._clear_latest_timing_snapshot()
        _ = self._command("clear_playback_graph", {"reason": str(reason or "")})

    def record_coalesced_structural_edits(self, count: int = 1) -> None:
        _ = self._command("record_coalesced_structural_edits", {"count": int(count)})

    def play(self) -> None:
        self._clear_latest_timing_snapshot()
        _ = self._command("play", {})

    def pause(self) -> None:
        self._clear_latest_timing_snapshot()
        _ = self._command("pause", {})

    def stop(self) -> None:
        self._clear_latest_timing_snapshot()
        _ = self._command("stop", {})

    def seek(self, position_seconds: float) -> None:
        self._clear_latest_timing_snapshot()
        _ = self._command("seek", {"position_seconds": float(position_seconds)})

    def preview_clip(
        self,
        source_ref: str,
        *,
        start_seconds: float,
        end_seconds: float,
        gain_db: float = 0.0,
    ) -> bool:
        response = self._command(
            "preview_clip",
            {
                "source_ref": str(source_ref),
                "start_seconds": float(start_seconds),
                "end_seconds": float(end_seconds),
                "gain_db": float(gain_db),
            },
        )
        return bool(response.get("played", False))

    def current_time_seconds(self) -> float:
        snapshot = self.latest_timing_snapshot()
        if snapshot is not None:
            return max(0.0, float(snapshot.audible_time_seconds))
        response = self._command("current_time_seconds", {})
        return float(response.get("value", 0.0) or 0.0)

    def is_playing(self) -> bool:
        snapshot = self.latest_timing_snapshot()
        if snapshot is not None:
            return bool(snapshot.is_playing)
        response = self._command("is_playing", {})
        return bool(response.get("value", False))

    def timing_snapshot(self) -> PlaybackTimingSnapshot:
        cached = self.latest_timing_snapshot()
        if cached is not None:
            return cached
        response = self._command("timing_snapshot", {})
        value = response.get("value")
        if not isinstance(value, dict):
            raise PlaybackIpcError("timing_snapshot response missing value payload")
        return decode_timing_snapshot(value)

    def latest_timing_snapshot(self) -> PlaybackTimingSnapshot | None:
        """Return the latest pushed timing snapshot without HTTP IPC."""

        if not bool(getattr(self, "_audio_process_connected", False)):
            return None
        snapshot = getattr(self, "_latest_timing_snapshot", None)
        received_at = float(getattr(self, "_latest_timing_snapshot_received_monotonic", 0.0) or 0.0)
        if snapshot is None or received_at <= 0.0:
            return None
        if time.monotonic() - received_at > self._PUSHED_TIMING_STALE_SECONDS:
            return None
        return snapshot

    def latest_transport_snapshot(self) -> TransportSnapshot | None:
        """Return the latest pushed transport snapshot without HTTP IPC."""

        if self.latest_timing_snapshot() is None:
            return None
        return getattr(self, "_latest_transport_snapshot", None)

    def enqueue_transport_command(self, command: TransportCommand) -> None:
        """Apply one transport command through the coordinator-compatible seam."""

        action = command.action
        if action is TransportCommandAction.PLAY:
            if command.position_seconds is not None:
                self.seek(float(command.position_seconds))
            self.play()
            return
        if action is TransportCommandAction.PAUSE:
            if command.position_seconds is not None:
                self.seek(float(command.position_seconds))
            self.pause()
            return
        if action is TransportCommandAction.STOP:
            self.stop()
            return
        if action in {
            TransportCommandAction.SEEK,
            TransportCommandAction.SCRUB_UPDATE,
            TransportCommandAction.SCRUB_COMMIT,
        }:
            self.seek(float(command.position_seconds or 0.0))

    def snapshot_state(self, presentation: TimelinePresentation) -> PlaybackState:
        payload = PlaybackSyncPayload.from_presentation(presentation).to_dict()
        self._latest_sync_payload = payload
        response = self._command("snapshot_state", {"payload": payload})
        value = response.get("value")
        if not isinstance(value, dict):
            raise PlaybackIpcError("snapshot_state response missing value payload")
        state = decode_playback_state(value)
        diagnostics = state.diagnostics
        diagnostics.audio_process_connected = bool(self._audio_process_connected)
        diagnostics.audio_process_pid = self._audio_process_pid
        diagnostics.ipc_rtt_ms = float(self._ipc_rtt_ms)
        diagnostics.last_ipc_error = self._last_ipc_error
        diagnostics.latency_profile = self._latency_profile or diagnostics.latency_profile
        diagnostics.latency_profile_switch_count = max(
            int(diagnostics.latency_profile_switch_count),
            int(self._latency_profile_switch_count),
        )
        if self._last_latency_profile_reason is not None:
            diagnostics.last_latency_profile_reason = self._last_latency_profile_reason
        diagnostics.local_projection_build_ms = float(self._last_local_projection_build_ms)
        diagnostics.local_sync_classify_ms = float(self._last_local_sync_classify_ms)
        diagnostics.last_local_sync_change_kind = self._last_local_sync_change_kind
        diagnostics.last_ipc_command = self._last_ipc_command
        return state

    def presentation_signature(
        self, presentation: TimelinePresentation
    ) -> tuple[tuple[str, str], ...]:
        payload = PlaybackSyncPayload.from_presentation(presentation).to_dict()
        self._latest_sync_payload = payload
        response = self._command("presentation_signature", {"payload": payload})
        raw = response.get("value", ()) or ()
        signature: list[tuple[str, str]] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                signature.append((str(item[0]), str(item[1])))
        return tuple(signature)

    def record_local_sync_decision(
        self,
        change_kind: str,
        *,
        projection_build_ms: float = 0.0,
        classify_ms: float = 0.0,
    ) -> None:
        """Record one local app-process sync decision for diagnostics."""

        self._last_local_sync_change_kind = str(change_kind or "")
        self._last_local_projection_build_ms = max(0.0, float(projection_build_ms))
        self._last_local_sync_classify_ms = max(0.0, float(classify_ms))

    def _clear_latest_timing_snapshot(self) -> None:
        self._latest_timing_snapshot = None
        self._latest_transport_snapshot = None
        self._latest_timing_snapshot_received_monotonic = 0.0

    def _next_command_id(self) -> str:
        self._command_counter += 1
        return f"cmd-{self._command_counter}"

    def _command(self, operation: str, params: dict[str, object]) -> dict[str, object]:
        if self._shutdown:
            if str(operation) in self._SHUTDOWN_NOOP_OPERATIONS:
                return {}
            raise PlaybackIpcError("playback process client is shut down")
        command_id = self._next_command_id()
        payload = {
            "version": PLAYBACK_IPC_VERSION,
            "command_id": command_id,
            "operation": operation,
            "params": params,
        }
        self._last_ipc_command = str(operation)
        started = time.perf_counter()
        response = self._request_json("POST", PLAYBACK_IPC_COMMAND_PATH, payload)
        self._ipc_rtt_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        meta = response.get("meta")
        if isinstance(meta, dict):
            self._ipc_rtt_ms = float(meta.get("ipc_rtt_ms", self._ipc_rtt_ms) or self._ipc_rtt_ms)
        result = response.get("result")
        if not isinstance(result, dict):
            raise PlaybackIpcError(f"Invalid command result for operation '{operation}'")
        return result

    def _health_request(self) -> dict[str, object]:
        payload = self._request_json("GET", PLAYBACK_IPC_HEALTH_PATH, None)
        if str(payload.get("version", "")) != PLAYBACK_IPC_VERSION:
            raise PlaybackIpcError(
                f"Playback IPC version mismatch: expected {PLAYBACK_IPC_VERSION}, got {payload.get('version')}"
            )
        self._audio_process_pid = int(payload.get("pid", 0) or 0)
        self._ws_url = str(payload.get("ws_url", "") or "")
        return payload

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        if self._http is None:
            raise PlaybackIpcError("HTTP channel is not initialized")

        headers = {
            "Accept": "application/json",
            PLAYBACK_IPC_TOKEN_HEADER: self._token,
        }
        body = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":"))
            headers["Content-Type"] = "application/json"
        try:
            self._http.request(method, path, body=body, headers=headers)
            response = self._http.getresponse()
            raw = response.read()
        except Exception as exc:
            self._last_ipc_error = f"{type(exc).__name__}: {exc}"
            self._reconnect_http_channel()
            raise PlaybackIpcError(f"Playback IPC request failed: {exc}") from exc

        try:
            decoded = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as exc:
            self._last_ipc_error = f"JSONDecodeError: {exc}"
            raise PlaybackIpcError(f"Playback IPC returned invalid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise PlaybackIpcError("Playback IPC response must be a JSON object")
        if response.status >= 400 or not bool(decoded.get("ok", False)):
            error = str(decoded.get("error", f"http_{response.status}"))
            self._last_ipc_error = error
            raise PlaybackIpcError(error)
        return decoded

    def _spawn_service_process(self):
        command = _build_service_process_command(
            host=self._host,
            port=self._port,
            ws_port=self._ws_port,
            token=self._token,
        )
        if self._audio_output_config is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
            ) as handle:
                payload = {
                    "output_device": self._audio_output_config.output_device,
                    "sample_rate": self._audio_output_config.sample_rate,
                    "channels": self._audio_output_config.channels,
                    "master_output_bus": self._audio_output_config.master_output_bus,
                    "stream_latency": self._audio_output_config.stream_latency,
                    "stream_blocksize": self._audio_output_config.stream_blocksize,
                    "prime_output_buffers_using_stream_callback": (
                        self._audio_output_config.prime_output_buffers_using_stream_callback
                    ),
                }
                json.dump(payload, handle)
                self._audio_config_file = Path(handle.name)
            command.extend(["--audio-config-json", str(self._audio_config_file)])
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return process

    def _wait_for_health(self) -> None:
        deadline = time.time() + self._service_start_timeout_seconds
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                startup_error = self._collect_startup_stderr()
                if startup_error:
                    raise PlaybackIpcError(
                        f"Playback service process exited during startup: {startup_error}"
                    )
                raise PlaybackIpcError("Playback service process exited during startup")
            try:
                self._health_request()
                return
            except Exception:
                time.sleep(0.05)
                continue
        startup_error = self._terminate_failed_start()
        detail = f": {startup_error}" if startup_error else ""
        raise PlaybackIpcError(f"Timed out waiting for playback service health{detail}")

    def _terminate_failed_start(self) -> str | None:
        process = self._process
        if process is None:
            return None
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
            except Exception:
                pass
        return self._collect_startup_stderr()

    def _collect_startup_stderr(self) -> str | None:
        if self._startup_stderr is not None:
            return self._startup_stderr or None
        process = self._process
        if process is None or process.stderr is None:
            return None
        try:
            stderr_text = process.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            stderr_text = ""
        self._startup_stderr = stderr_text
        return stderr_text or None

    def _reconnect_http_channel(self) -> None:
        if self._http is not None:
            try:
                self._http.close()
            except Exception:
                pass
        self._http = http.client.HTTPConnection(self._host, self._port, timeout=3.0)

    def _start_ws_listener(self) -> None:
        self._ws_stop_event.clear()
        if not self._ws_url:
            return
        self._ws_thread = threading.Thread(
            target=self._run_ws_listener,
            name="ez-playback-client-events",
            daemon=True,
        )
        self._ws_thread.start()

    def _run_ws_listener(self) -> None:
        while not self._ws_stop_event.is_set() and not self._shutdown:
            try:
                with ws_connect(
                    self._ws_url,
                    additional_headers={PLAYBACK_IPC_TOKEN_HEADER: self._token},
                    open_timeout=2,
                    ping_interval=10,
                    ping_timeout=10,
                    close_timeout=1,
                    proxy=None,
                ) as websocket:
                    self._audio_process_connected = True
                    for message in websocket:
                        if self._ws_stop_event.is_set() or self._shutdown:
                            return
                        self._handle_ws_event(message)
            except Exception:
                self._audio_process_connected = False
                if self._ws_stop_event.is_set() or self._shutdown:
                    return
                time.sleep(0.15)

    def _handle_ws_event(self, raw_message: object) -> None:
        if not isinstance(raw_message, str):
            return
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        event_type = str(payload.get("type", "") or "")
        body = payload.get("payload")
        if not isinstance(body, dict):
            body = {}

        if event_type == "service-started":
            if body.get("pid") is not None:
                self._audio_process_pid = int(body.get("pid", 0) or 0)
            self._audio_process_connected = True
            return

        if event_type == "latency-profile-switched":
            self._latency_profile = str(
                body.get("profile", self._latency_profile) or self._latency_profile
            )
            self._latency_profile_switch_count = int(
                body.get("switch_count", self._latency_profile_switch_count)
                or self._latency_profile_switch_count
            )
            self._last_latency_profile_reason = (
                str(body.get("reason")) if body.get("reason") is not None else None
            )
            return

        if event_type == "command-error":
            error_text = body.get("error")
            if error_text is not None:
                self._last_ipc_error = str(error_text)
            return

        if event_type == "timing-snapshot":
            snapshot_payload = body.get("snapshot")
            if isinstance(snapshot_payload, dict):
                snapshot = decode_timing_snapshot(snapshot_payload)
                self._latest_timing_snapshot = snapshot
                self._latest_timing_snapshot_received_monotonic = time.monotonic()
                self._latest_transport_snapshot = TransportSnapshot.from_timing_snapshot(
                    snapshot,
                    generation_id=(
                        str(body.get("structural_generation"))
                        if body.get("structural_generation") is not None
                        else None
                    ),
                )


def _build_service_process_command(
    *,
    host: str,
    port: int,
    ws_port: int,
    token: str,
) -> list[str]:
    """Build the playback service command for source and frozen runtimes."""
    command = [sys.executable]
    if getattr(sys, "frozen", False):
        command.append("--playback-service")
    else:
        command.extend(["-m", "echozero.application.playback.process_service_entry"])
    command.extend(
        [
            "--host",
            host,
            "--port",
            str(port),
            "--ws-port",
            str(ws_port),
            f"--token={token}",
        ]
    )
    return command


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind((PLAYBACK_IPC_HOST, 0))
        handle.listen(1)
        return int(handle.getsockname()[1])


__all__ = ["ProcessPlaybackClient"]

"""
Playback process service: dedicated runtime-audio host with HTTP + WebSocket IPC.
Exists because audio callback stability requires isolation from UI-thread and app-process contention.
Connects process IPC commands to the canonical PlaybackController and emits runtime telemetry events.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

from echozero.application.playback.models import PlaybackState
from echozero.application.playback.process_shared import (
    PLAYBACK_IPC_COMMAND_PATH,
    PLAYBACK_IPC_HEALTH_PATH,
    PLAYBACK_IPC_TOKEN_HEADER,
    PLAYBACK_IPC_VERSION,
    encode_playback_state,
    encode_timing_snapshot,
)
from echozero.application.playback.process_events_hub import PlaybackEventsHub
from echozero.application.playback.runtime import PlaybackController
from echozero.application.playback.sync_projection import PlaybackSyncPayload, RuntimeSyncProjection
from echozero.application.settings import AudioOutputRuntimeConfig
from echozero.audio.engine import AudioEngine
from echozero.errors import InfrastructureError


@dataclass(slots=True, frozen=True)
class _LatencyProfileSpec:
    name: str
    stream_latency: str | float
    stream_blocksize: int


@dataclass(slots=True)
class _PlaybackServiceTelemetry:
    latency_profile: str = "low"
    latency_profile_switch_count: int = 0
    last_latency_profile_reason: str | None = None
    structural_generation: int = 0
    ipc_rtt_ms: float = 0.0
    last_ipc_error: str | None = None


class PlaybackProcessService:
    """Dedicated process host for playback runtime commands and telemetry."""

    _LATENCY_PROFILE_SPECS: tuple[_LatencyProfileSpec, ...] = (
        _LatencyProfileSpec(name="ultra_low", stream_latency="low", stream_blocksize=0),
        _LatencyProfileSpec(name="low", stream_latency="low", stream_blocksize=128),
        _LatencyProfileSpec(name="balanced", stream_latency="high", stream_blocksize=256),
    )
    _GLITCH_STEP_UP_THRESHOLD = 3
    _GLITCH_STEP_UP_WINDOW_SECONDS = 2.0
    _GLITCH_STEP_DOWN_STABLE_SECONDS = 30.0
    _ADAPTIVE_PROFILE_SUSPEND_AFTER_RUNNING_SEEK_SECONDS = 2.5

    def __init__(
        self,
        *,
        host: str,
        port: int,
        ws_port: int,
        token: str,
        base_audio_config: AudioOutputRuntimeConfig | None,
    ) -> None:
        self._host = host
        self._port = int(port)
        self._token = token
        self._base_audio_config = base_audio_config
        self._profile_index = 1
        self._pending_profile_index: int | None = None
        self._telemetry = _PlaybackServiceTelemetry(
            latency_profile=self._LATENCY_PROFILE_SPECS[self._profile_index].name
        )
        self._service_started_monotonic = time.perf_counter()
        self._latest_projection: RuntimeSyncProjection | None = None
        self._shutdown_requested = False
        self._last_reason_seen = ""
        self._last_glitch_count = 0
        self._glitch_burst_events: list[tuple[float, int]] = []
        self._last_glitch_change_monotonic = time.perf_counter()
        self._adaptive_profile_suspend_until_monotonic = 0.0
        self._device_reinit_count = 0
        self._rt_event_queue: deque[dict[str, object]] = deque(maxlen=2048)

        self._controller = self._build_controller()
        self._events_hub = PlaybackEventsHub(host=host, port=ws_port)
        self._http_server = self._build_http_server()
        self._http_server.timeout = 0.1

    def run(self) -> int:
        self._events_hub.start()
        self._publish_event("service-started", {"pid": self.pid, "ws_url": self._events_hub.ws_url})
        try:
            while not self._shutdown_requested:
                self._http_server.handle_request()
                self._tick()
        finally:
            self._shutdown_requested = True
            self._events_hub.shutdown()
            self._http_server.server_close()
            self._controller.shutdown()
        return 0

    @property
    def pid(self) -> int:
        return int(os.getpid())

    def health_payload(self) -> dict[str, object]:
        return {
            "ok": not self._shutdown_requested,
            "version": PLAYBACK_IPC_VERSION,
            "pid": self.pid,
            "ws_url": self._events_hub.ws_url,
            "uptime_seconds": max(0.0, time.perf_counter() - self._service_started_monotonic),
        }

    def dispatch_command(self, payload: dict[str, object]) -> dict[str, object]:
        command_id = str(payload.get("command_id", "") or "")
        operation = str(payload.get("operation", "") or "").strip().lower()
        params = payload.get("params")
        if not isinstance(params, dict):
            params = {}
        started = time.perf_counter()
        result: dict[str, object]
        try:
            result = self._dispatch(operation, params)
            self._telemetry.last_ipc_error = None
        except Exception as exc:
            self._telemetry.last_ipc_error = f"{type(exc).__name__}: {exc}"
            self._publish_event(
                "command-error",
                {
                    "command_id": command_id,
                    "operation": operation,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        self._telemetry.ipc_rtt_ms = elapsed_ms
        envelope = {
            "ok": True,
            "version": PLAYBACK_IPC_VERSION,
            "command_id": command_id,
            "operation": operation,
            "result": result,
            "meta": {
                "ipc_rtt_ms": elapsed_ms,
                "structural_generation": int(self._telemetry.structural_generation),
                "last_track_sync_reason": str(
                    getattr(self._controller, "_last_track_sync_reason", "") or ""
                ),
            },
        }
        self._publish_event(
            "command-complete",
            {
                "command_id": command_id,
                "operation": operation,
                "ipc_rtt_ms": elapsed_ms,
                "structural_generation": int(self._telemetry.structural_generation),
            },
        )
        return envelope

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def _dispatch(self, operation: str, params: dict[str, object]) -> dict[str, object]:
        if operation == "sync_structure_state":
            projection = self._require_projection(params)
            self._latest_projection = projection
            self._controller.sync_structure_state(projection)
            self._telemetry.structural_generation = int(
                getattr(self._controller, "_latest_requested_generation", 0)
            )
            self._push_rt_event(
                "structure-enqueued",
                {
                    "generation": int(self._telemetry.structural_generation),
                    "reason": "sync_structure_state",
                },
            )
            return {"queued_generation": int(self._telemetry.structural_generation)}

        if operation == "sync_mix_state":
            projection = self._require_projection(params)
            self._latest_projection = projection
            self._controller.sync_mix_state(projection)
            self._push_rt_event("mix-enqueued", {"reason": "sync_mix_state"})
            return {}

        if operation == "drain_pending_structure_sync":
            self._controller.drain_pending_structure_sync()
            return {}

        if operation == "record_coalesced_structural_edits":
            count = int(params.get("count", 1) or 1)
            self._controller.record_coalesced_structural_edits(count)
            return {"count": count}

        if operation == "play":
            self._controller.play()
            return {}

        if operation == "pause":
            self._controller.pause()
            return {}

        if operation == "stop":
            self._controller.stop()
            return {}

        if operation == "seek":
            position = float(params.get("position_seconds", 0.0) or 0.0)
            was_playing = bool(self._controller.is_playing())
            self._controller.seek(position)
            self._push_rt_event(
                "seek-enqueued",
                {
                    "position_seconds": position,
                    "seek_id": str(params.get("seek_id", "") or ""),
                },
            )
            if was_playing:
                self._adaptive_profile_suspend_until_monotonic = (
                    time.perf_counter()
                    + self._ADAPTIVE_PROFILE_SUSPEND_AFTER_RUNNING_SEEK_SECONDS
                )
            return {"position_seconds": position}

        if operation == "preview_clip":
            source_ref = str(params.get("source_ref", "") or "")
            played = self._controller.preview_clip(
                source_ref,
                start_seconds=float(params.get("start_seconds", 0.0) or 0.0),
                end_seconds=float(params.get("end_seconds", 0.0) or 0.0),
                gain_db=float(params.get("gain_db", 0.0) or 0.0),
            )
            self._push_rt_event(
                "preview-enqueued",
                {
                    "source_ref": source_ref,
                    "played": bool(played),
                },
            )
            return {"played": bool(played)}

        if operation == "enqueue_structure":
            projection = self._require_projection(params)
            generation = int(params.get("generation", 0) or 0)
            self._latest_projection = projection
            self._controller.sync_structure_state(projection)
            resolved_generation = int(getattr(self._controller, "_latest_requested_generation", 0) or 0)
            if generation <= 0:
                generation = resolved_generation
            self._telemetry.structural_generation = int(resolved_generation)
            self._push_rt_event(
                "structure-enqueued",
                {"generation": int(generation), "reason": "enqueue_structure"},
            )
            return {"generation": int(generation)}

        if operation == "enqueue_mix":
            projection = self._require_projection(params)
            self._latest_projection = projection
            self._controller.sync_mix_state(projection)
            self._push_rt_event("mix-enqueued", {"reason": "enqueue_mix"})
            return {}

        if operation == "enqueue_seek":
            target_samples = int(params.get("target_samples", 0) or 0)
            seek_id = str(params.get("seek_id", "") or "")
            position_seconds = float(target_samples) / float(max(1, self._controller.engine.sample_rate))
            self._controller.seek(position_seconds)
            self._push_rt_event(
                "seek-enqueued",
                {
                    "target_samples": target_samples,
                    "seek_id": seek_id,
                },
            )
            return {"seek_id": seek_id, "position_seconds": position_seconds}

        if operation == "enqueue_preview":
            clip_spec = params.get("clip_spec")
            if not isinstance(clip_spec, dict):
                raise InfrastructureError("enqueue_preview requires clip_spec mapping")
            source_ref = str(clip_spec.get("source_ref", "") or "")
            played = self._controller.preview_clip(
                source_ref,
                start_seconds=float(clip_spec.get("start_seconds", 0.0) or 0.0),
                end_seconds=float(clip_spec.get("end_seconds", 0.0) or 0.0),
                gain_db=float(clip_spec.get("gain_db", 0.0) or 0.0),
            )
            self._push_rt_event(
                "preview-enqueued",
                {
                    "source_ref": source_ref,
                    "played": bool(played),
                },
            )
            return {"played": bool(played)}

        if operation == "reconfigure_device":
            device_spec = params.get("device_spec")
            if not isinstance(device_spec, dict):
                device_spec = {}
            profile_name = str(params.get("profile", "") or "").strip().lower()
            if profile_name:
                for index, spec in enumerate(self._LATENCY_PROFILE_SPECS):
                    if spec.name == profile_name:
                        self._profile_index = index
                        self._telemetry.latency_profile = spec.name
                        break
            self._reconfigure_device(device_spec)
            return {
                "latency_profile": self._telemetry.latency_profile,
                "device_reinit_count": int(self._device_reinit_count),
            }

        if operation == "drain_events":
            return {"events": self._drain_rt_events()}

        if operation == "current_time_seconds":
            return {"value": float(self._controller.current_time_seconds())}

        if operation == "is_playing":
            return {"value": bool(self._controller.is_playing())}

        if operation == "timing_snapshot":
            snapshot = self._controller.timing_snapshot()
            return {"value": encode_timing_snapshot(snapshot)}

        if operation == "presentation_signature":
            projection = self._require_projection(params)
            return {
                "value": [list(item) for item in self._controller.presentation_signature(projection)]
            }

        if operation == "snapshot_state":
            projection = self._require_projection(params, allow_empty=True)
            self._latest_projection = projection
            state = self._controller.snapshot_state(projection)
            self._apply_service_diagnostics(state)
            return {"value": encode_playback_state(state)}

        if operation == "shutdown":
            self.request_shutdown()
            return {"accepted": True}

        raise InfrastructureError(f"Unsupported playback IPC operation: {operation}")

    def _require_projection(
        self,
        params: dict[str, object],
        *,
        allow_empty: bool = False,
    ) -> RuntimeSyncProjection:
        payload = params.get("payload")
        if isinstance(payload, dict):
            projection = PlaybackSyncPayload.from_dict(payload).to_runtime_projection()
            return projection
        if allow_empty:
            return RuntimeSyncProjection(
                layers=[],
                selected_layer_id=None,
                selected_take_id=None,
                playback_output_channels=0,
            )
        raise InfrastructureError("Missing playback sync payload")

    def _apply_service_diagnostics(self, state: PlaybackState) -> None:
        diagnostics = state.diagnostics
        diagnostics.audio_process_connected = True
        diagnostics.audio_process_pid = self.pid
        diagnostics.ipc_rtt_ms = float(self._telemetry.ipc_rtt_ms)
        diagnostics.last_ipc_error = self._telemetry.last_ipc_error
        diagnostics.latency_profile = self._telemetry.latency_profile
        diagnostics.latency_profile_switch_count = int(self._telemetry.latency_profile_switch_count)
        diagnostics.last_latency_profile_reason = self._telemetry.last_latency_profile_reason
        diagnostics.device_reinit_count = int(self._device_reinit_count)
        diagnostics.rt_command_queue_depth = int(len(self._rt_event_queue))

    def _tick(self) -> None:
        self._apply_pending_profile_if_safe()
        self._controller.drain_pending_structure_sync()
        self._emit_reason_update_if_changed()
        self._sample_glitch_and_adapt_profile()

    def _apply_pending_profile_if_safe(self) -> None:
        if self._pending_profile_index is None:
            return
        if self._controller.is_playing():
            return
        pending = int(self._pending_profile_index)
        self._pending_profile_index = None
        self._switch_latency_profile(pending, reason="deferred-profile-apply")

    def _emit_reason_update_if_changed(self) -> None:
        reason = str(getattr(self._controller, "_last_track_sync_reason", "") or "")
        if reason == self._last_reason_seen:
            return
        self._last_reason_seen = reason
        self._push_rt_event(
            "reason-updated",
            {
                "reason": reason,
                "structural_generation": int(self._telemetry.structural_generation),
            },
        )
        self._publish_event(
            "track-sync-reason",
            {
                "reason": reason,
                "structural_generation": int(self._telemetry.structural_generation),
            },
        )

    def _sample_glitch_and_adapt_profile(self) -> None:
        glitch_count = int(self._controller.engine.glitch_count)
        now = time.perf_counter()
        if now < self._adaptive_profile_suspend_until_monotonic:
            self._last_glitch_count = glitch_count
            return
        if glitch_count > self._last_glitch_count:
            delta = glitch_count - self._last_glitch_count
            self._last_glitch_count = glitch_count
            self._last_glitch_change_monotonic = now
            self._glitch_burst_events.append((now, delta))
        threshold_window = now - self._GLITCH_STEP_UP_WINDOW_SECONDS
        self._glitch_burst_events = [
            (timestamp, delta)
            for timestamp, delta in self._glitch_burst_events
            if timestamp >= threshold_window
        ]
        burst_total = sum(delta for _, delta in self._glitch_burst_events)
        if (
            burst_total >= self._GLITCH_STEP_UP_THRESHOLD
            and self._profile_index < len(self._LATENCY_PROFILE_SPECS) - 1
        ):
            self._switch_latency_profile(self._profile_index + 1, reason="auto-up-glitch-threshold")
            self._glitch_burst_events.clear()
            return

        stable_for_seconds = now - self._last_glitch_change_monotonic
        if (
            stable_for_seconds >= self._GLITCH_STEP_DOWN_STABLE_SECONDS
            and self._profile_index > 0
        ):
            self._switch_latency_profile(self._profile_index - 1, reason="auto-down-stable-window")

    def _switch_latency_profile(self, next_index: int, *, reason: str) -> None:
        if next_index == self._profile_index:
            return
        if self._controller.is_playing():
            self._pending_profile_index = max(
                0,
                min(next_index, len(self._LATENCY_PROFILE_SPECS) - 1),
            )
            self._telemetry.last_latency_profile_reason = "deferred-while-playing"
            self._publish_event(
                "latency-profile-switch-deferred",
                {
                    "current_profile": self._telemetry.latency_profile,
                    "pending_profile": self._LATENCY_PROFILE_SPECS[self._pending_profile_index].name,
                    "reason": reason,
                },
            )
            return
        self._profile_index = max(0, min(next_index, len(self._LATENCY_PROFILE_SPECS) - 1))
        self._telemetry.latency_profile = self._LATENCY_PROFILE_SPECS[self._profile_index].name
        self._telemetry.latency_profile_switch_count += 1
        self._telemetry.last_latency_profile_reason = reason
        projection = self._latest_projection
        current_time = float(self._controller.current_time_seconds())
        was_playing = bool(self._controller.is_playing())
        self._controller.shutdown()
        self._controller = self._build_controller()
        if projection is not None:
            self._controller.sync_structure_state(projection)
            if current_time > 0.0:
                self._controller.seek(current_time)
            if was_playing:
                self._controller.play()
        self._publish_event(
            "latency-profile-switched",
            {
                "profile": self._telemetry.latency_profile,
                "reason": reason,
                "switch_count": int(self._telemetry.latency_profile_switch_count),
            },
        )
        self._push_rt_event(
            "latency-profile-switched",
            {
                "profile": self._telemetry.latency_profile,
                "reason": reason,
                "switch_count": int(self._telemetry.latency_profile_switch_count),
            },
        )

    def _build_http_server(self) -> HTTPServer:
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                if self.path != PLAYBACK_IPC_HEALTH_PATH:
                    self._write_json(404, {"ok": False, "error": "not_found"})
                    return
                if not self._is_authorized():
                    self._write_json(401, {"ok": False, "error": "unauthorized"})
                    return
                self._write_json(200, service.health_payload())

            def do_POST(self) -> None:  # noqa: N802
                if self.path != PLAYBACK_IPC_COMMAND_PATH:
                    self._write_json(404, {"ok": False, "error": "not_found"})
                    return
                if not self._is_authorized():
                    self._write_json(401, {"ok": False, "error": "unauthorized"})
                    return
                payload = self._read_json_body()
                if payload is None:
                    self._write_json(400, {"ok": False, "error": "invalid_json"})
                    return
                try:
                    response = service.dispatch_command(payload)
                    self._write_json(200, response)
                except Exception as exc:
                    self._write_json(
                        500,
                        {
                            "ok": False,
                            "version": PLAYBACK_IPC_VERSION,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )

            def _is_authorized(self) -> bool:
                return self.headers.get(PLAYBACK_IPC_TOKEN_HEADER) == service._token

            def _read_json_body(self) -> dict[str, object] | None:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    return None
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return None
                if not isinstance(payload, dict):
                    return None
                return payload

            def _write_json(self, status: int, payload: dict[str, object]) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return None

        return HTTPServer((self._host, self._port), Handler)

    def _build_controller(self) -> PlaybackController:
        profile = self._LATENCY_PROFILE_SPECS[self._profile_index]
        base = self._base_audio_config

        def _engine_factory() -> AudioEngine:
            return AudioEngine(
                sample_rate=(base.sample_rate if base is not None else None),
                channels=(base.channels if base is not None else None),
                stream_latency=profile.stream_latency,
                stream_blocksize=profile.stream_blocksize,
                prime_output_buffers_using_stream_callback=(
                    bool(base.prime_output_buffers_using_stream_callback)
                    if base is not None
                    else True
                ),
                output_device=(base.output_device if base is not None else None),
            )

        return PlaybackController(
            engine_factory=_engine_factory,
        )

    def _publish_event(self, event_type: str, payload: dict[str, object]) -> None:
        self._events_hub.publish(
            {
                "type": event_type,
                "version": PLAYBACK_IPC_VERSION,
                "ts_monotonic": time.perf_counter(),
                "payload": payload,
            }
        )

    def _push_rt_event(self, event_type: str, payload: dict[str, object]) -> None:
        self._rt_event_queue.append(
            {
                "type": str(event_type),
                "ts_monotonic": time.perf_counter(),
                "payload": dict(payload),
            }
        )

    def _drain_rt_events(self) -> list[dict[str, object]]:
        events = list(self._rt_event_queue)
        self._rt_event_queue.clear()
        return events

    def _reconfigure_device(self, device_spec: dict[str, object]) -> None:
        current = self._base_audio_config or AudioOutputRuntimeConfig()
        self._base_audio_config = AudioOutputRuntimeConfig(
            output_device=device_spec.get("output_device", current.output_device),
            sample_rate=(
                int(device_spec["sample_rate"])
                if device_spec.get("sample_rate") is not None
                else current.sample_rate
            ),
            channels=(
                int(device_spec["channels"])
                if device_spec.get("channels") is not None
                else current.channels
            ),
            stream_latency=device_spec.get("stream_latency", current.stream_latency),
            stream_blocksize=(
                int(device_spec["stream_blocksize"])
                if device_spec.get("stream_blocksize") is not None
                else current.stream_blocksize
            ),
            prime_output_buffers_using_stream_callback=bool(
                device_spec.get(
                    "prime_output_buffers_using_stream_callback",
                    current.prime_output_buffers_using_stream_callback,
                )
            ),
        )
        projection = self._latest_projection
        current_time = float(self._controller.current_time_seconds())
        was_playing = bool(self._controller.is_playing())
        self._controller.shutdown()
        self._controller = self._build_controller()
        self._device_reinit_count += 1
        if projection is not None:
            self._controller.sync_structure_state(projection)
            if current_time > 0.0:
                self._controller.seek(current_time)
            if was_playing:
                self._controller.play()

__all__ = ["PlaybackProcessService"]

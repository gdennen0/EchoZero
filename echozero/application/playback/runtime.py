"""
Playback controller for EZ runtime audio.
Exists because the app should sync DAW-style playback tracks without exposing engine internals.
Connects timeline presentation state to one backend-agnostic runtime playback controller.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from datetime import datetime, timezone
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from echozero.application.playback.models import (
    PlaybackDiagnostics,
    PlaybackSource,
    PlaybackState,
    PlaybackTimingSnapshot,
)
from echozero.application.playback.audio_diagnostics import (
    timestamped_capture_id,
    write_audio_diagnostics_bundle,
)
from echozero.application.playback.timecode import PlaybackTimecodeAuthority, TimebaseSpec
from echozero.application.playback.tracks import (
    PlaybackMixPlan,
    PlaybackTrack,
    PlaybackTrackBuilder,
    PlaybackTrackPlan,
)
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.enums import PlaybackStatus
from echozero.audio.engine import AudioEngine
from echozero.audio.layer import AudioTrack
from echozero.audio.file_cache import load_audio_file
from echozero.audio.output_backend import DEFAULT_BUFFER_SIZE
from echozero.output_routing import parse_output_bus_spans

_SOUNDDEVICE_BACKEND = "sounddevice"
_RUNTIME_EVENT_LIMIT = 32


def _db_to_linear(gain_db: float) -> float:
    return float(10.0 ** (float(gain_db) / 20.0))


def _load_runtime_audio(path: str | Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = load_audio_file(path)
    return samples.astype(np.float32, copy=False), int(sample_rate)


class PlaybackController:
    """Playback facade that keeps EZ on one simple track-based runtime surface."""

    _PRIMARY_TRACK_ID = "__ez_primary_track__"
    _ROUTED_TRACK_PREFIX = "__ez_route__"
    _PREVIEW_TRACK_ID = "__ez_preview_track__"

    def __init__(
        self,
        engine: AudioEngine | None = None,
        *,
        engine_factory: Callable[[], AudioEngine] | None = None,
        preview_engine: AudioEngine | None = None,
        preview_engine_factory: Callable[[], AudioEngine] | None = None,
        audio_loader: Callable[[str | Path], tuple[np.ndarray, int]] = _load_runtime_audio,
        timebase: TimebaseSpec | None = None,
    ) -> None:
        self._engine = engine or (
            engine_factory() if engine_factory is not None else AudioEngine()
        )
        self._engine_factory = engine_factory
        _ = preview_engine
        _ = preview_engine_factory
        self._track_builder = PlaybackTrackBuilder(audio_loader)
        self._async_track_builder = PlaybackTrackBuilder(audio_loader)
        self._loaded_track_signature: tuple[tuple[str, str], ...] = ()
        self._loaded_track_structure_signature: tuple[tuple[str, str], ...] = ()
        self._loaded_uses_track_routing = False
        self._structure_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ez-structure-sync",
        )
        self._pending_structure_futures: dict[Future[PlaybackTrackPlan], int] = {}
        self._generation_terminal_outcomes: dict[int, str] = {}
        self._next_generation = 1
        self._latest_requested_generation = 0
        self._latest_ready_generation = 0
        self._owner_thread_id = threading.get_ident()
        self._shutdown_state = "running"
        self._preview_active = False
        self._last_transition = ""
        self._last_track_sync_reason = ""
        self._structural_rebuild_count = 0
        self._coalesced_edit_count = 0
        self._last_structural_rebuild_ms = 0.0
        self._max_structural_rebuild_ms = 0.0
        self._last_local_projection_build_ms = 0.0
        self._last_local_sync_classify_ms = 0.0
        self._last_local_sync_change_kind = ""
        self._last_ipc_command = ""
        self._rt_last_apply_latency_ms = 0.0
        self._rt_last_seek_apply_latency_ms = 0.0
        self._timebase = timebase or TimebaseSpec.from_legacy_fps(30)
        self._latest_presentation: TimelinePresentation | None = None
        self._device_reinit_count = 0
        self._last_device_reinit_reason = ""
        self._last_route_resolution_summary = ""
        self._runtime_event_sequence = 0
        self._recent_runtime_events = deque(maxlen=_RUNTIME_EVENT_LIMIT)
        self._audio_diagnostics_capture: dict[str, object] | None = None
        self._last_audio_diagnostics_capture_result: dict[str, object] | None = None

    @property
    def engine(self) -> AudioEngine:
        return self._engine

    def sync_presentation(self, presentation: TimelinePresentation) -> None:
        """Compatibility alias for callers that still say `sync_presentation`."""

        self.sync_structure_state(presentation)

    def sync_structure_state(self, presentation: TimelinePresentation) -> None:
        """Sync the currently playable EZ presentation into engine tracks."""

        if self._shutdown_state != "running":
            self._last_track_sync_reason = "structure-async-shutdown-ignored"
            return
        self.drain_pending_structure_sync()
        self._latest_presentation = presentation
        resolved = self._with_resolved_output_channels(presentation)
        if self.is_playing():
            self._queue_async_structure_sync(resolved)
            return
        self._cancel_pending_structure_sync(reason="structure-sync-superseded")
        self._sync_track_plan(self._track_builder.build_track_plan(resolved))

    def sync_mix_state(self, presentation: TimelinePresentation) -> None:
        """Sync track gain and routing changes for the current presentation."""

        if self._shutdown_state != "running":
            return
        self.drain_pending_structure_sync()
        self._latest_presentation = presentation
        resolved = self._with_resolved_output_channels(presentation)
        mix_plan = self._track_builder.build_mix_plan(resolved)
        if self._apply_track_mix_state(mix_plan):
            self._last_track_sync_reason = "mix-state-pending-structure-sync"
            return
        self._last_track_sync_reason = "mix-state-applied"

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        """Compatibility alias for callers that still say `build_for_presentation`."""

        self.sync_structure_state(presentation)

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        """Compatibility alias for callers that still say `apply_mix_state`."""

        self.sync_mix_state(presentation)

    def record_coalesced_structural_edits(self, count: int = 1) -> None:
        self._coalesced_edit_count += max(0, int(count))

    def play(self) -> None:
        self._sync_preview_state()
        if self._preview_active:
            self.stop_preview()
        self._last_transition = "play"
        self._engine.play()

    def pause(self) -> None:
        self._last_transition = "pause"
        self._engine.pause()

    def stop(self) -> None:
        self._last_transition = "stop"
        self._engine.stop()

    def seek(self, position_seconds: float) -> None:
        self._last_transition = "seek"
        started = time.perf_counter()
        self._engine.seek_seconds(position_seconds)
        self._rt_last_seek_apply_latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)

    def current_time_seconds(self) -> float:
        self.drain_pending_structure_sync()
        self._sync_preview_state()
        return float(self._engine.audible_time_seconds)

    def timing_snapshot(self) -> PlaybackTimingSnapshot:
        self.drain_pending_structure_sync()
        self._sync_preview_state()
        snapshot_time = getattr(self._engine, "_last_audible_time_seconds", None)
        snapshot_monotonic = getattr(self._engine, "_last_audible_monotonic_seconds", None)
        clock_time = float(self._engine.clock.position_seconds)
        is_playing = bool(self._engine.transport.is_playing)
        if snapshot_time is None:
            snapshot_time = float(self._engine.audible_time_seconds)
            snapshot_monotonic = None
        sample_position = int(
            round(max(0.0, float(snapshot_time)) * float(self._engine.sample_rate))
        )
        timecode = PlaybackTimecodeAuthority(
            self._timebase,
            sample_rate=max(1, int(self._engine.sample_rate)),
        ).position_from_samples(sample_position)
        lock_state = "locked" if timecode.is_locked else "unlocked"
        return PlaybackTimingSnapshot(
            audible_time_seconds=max(0.0, float(snapshot_time)),
            clock_time_seconds=max(0.0, clock_time),
            snapshot_monotonic_seconds=(
                max(0.0, float(snapshot_monotonic)) if snapshot_monotonic is not None else None
            ),
            is_playing=is_playing,
            sample_position=sample_position,
            frame_index=timecode.frame_index,
            timecode_label=timecode.label,
            display_label=timecode.label,
            timecode_mode=str(timecode.mode.value),
            timecode_lock_state=lock_state,
            drift_ppm=timecode.drift_ppm,
            drift_ms=timecode.drift_ms,
        )

    def is_playing(self) -> bool:
        self.drain_pending_structure_sync()
        self._sync_preview_state()
        return bool(self._engine.transport.is_playing)

    def shutdown(self) -> None:
        if self._shutdown_state == "shutdown":
            return
        self._shutdown_state = "shutting_down"
        self._cancel_pending_structure_sync(reason="shutdown")
        if self._pending_structure_futures:
            wait(list(self._pending_structure_futures.keys()), timeout=0.05)
            self._drain_completed_structure_outcomes_without_apply(outcome="cancelled")
        self._structure_executor.shutdown(wait=False, cancel_futures=True)
        self._shutdown_state = "shutdown"
        self.stop_preview()
        self._engine.clear_tracks()
        self._loaded_track_signature = ()
        self._loaded_track_structure_signature = ()
        self._loaded_uses_track_routing = False
        self._engine.shutdown()

    def reconfigure_device(
        self,
        *,
        device_spec: dict[str, object],
        profile: str = "",
    ) -> dict[str, object]:
        """Rebuild the output engine for one hardware/settings change."""

        _ = profile
        if self._shutdown_state != "running":
            return {
                "device_reinit_count": int(self._device_reinit_count),
                "reason": "shutdown",
            }
        presentation = self._latest_presentation
        current_time_seconds = self._current_engine_time_seconds()
        was_playing = bool(self._engine.transport.is_playing)
        reason = self._classify_device_reconfigure_reason(device_spec)
        self._cancel_pending_structure_sync(reason=reason)
        self._engine.shutdown()
        self._engine = self._build_engine_for_device_spec(device_spec)
        self._loaded_track_signature = ()
        self._loaded_track_structure_signature = ()
        self._loaded_uses_track_routing = False
        self._preview_active = False
        self._device_reinit_count += 1
        self._last_device_reinit_reason = reason
        self._last_track_sync_reason = reason
        if presentation is not None:
            self.sync_structure_state(presentation)
            if current_time_seconds > 0.0:
                self.seek(current_time_seconds)
            if was_playing and self._engine.tracks:
                self.play()
        return {
            "device_reinit_count": int(self._device_reinit_count),
            "reason": reason,
            "output_sample_rate": int(self._engine.sample_rate),
            "output_channels": int(self._engine.output_channels),
        }

    def preview_clip(
        self,
        source_ref: str,
        *,
        start_seconds: float,
        end_seconds: float,
        gain_db: float = 0.0,
    ) -> bool:
        source_path = str(source_ref).strip()
        if not source_path:
            return False
        source_buffer, sample_rate = self._track_builder.load_source_buffer(
            f"preview:{source_path}",
            source_path,
        )
        if source_buffer.size == 0:
            return False
        start_sample = max(0, int(round(float(start_seconds) * sample_rate)))
        end_sample = max(start_sample, int(round(float(end_seconds) * sample_rate)))
        end_sample = min(end_sample, int(source_buffer.shape[0]))
        if end_sample <= start_sample:
            return False
        preview_buffer = np.array(
            source_buffer[start_sample:end_sample], dtype=np.float32, copy=True
        )
        if preview_buffer.size == 0:
            return False
        replacing_preview = bool(self._preview_active or self._engine.overlay_active)
        self.stop_preview()
        self._last_transition = "preview_start"
        played = self._engine.play_overlay(
            preview_buffer,
            sample_rate,
            volume=_db_to_linear(gain_db),
        )
        self._preview_active = bool(played)
        if played:
            self._record_runtime_event(
                "preview-replace" if replacing_preview else "preview-start",
                source_ref=source_path,
                start_seconds=max(0.0, float(start_seconds)),
                end_seconds=max(0.0, float(end_seconds)),
                duration_seconds=max(0.0, float(end_seconds) - float(start_seconds)),
                frames=int(preview_buffer.shape[0]),
                sample_rate=int(sample_rate),
            )
        return bool(played)

    def stop_preview(self) -> None:
        was_preview_active = bool(self._preview_active or self._engine.overlay_active)
        self._last_transition = "preview_stop"
        self._preview_active = False
        self._engine.stop_overlay()
        if was_preview_active:
            self._record_runtime_event("preview-stop")

    def start_audio_diagnostics_capture(
        self,
        *,
        output_dir: str | Path | None = None,
        include_audio_buffers: bool = True,
        max_audio_blocks: int = 64,
    ) -> dict[str, object]:
        """Start one bounded runtime-audio diagnostics capture."""

        capture_id = timestamped_capture_id()
        started = datetime.now(timezone.utc).isoformat()
        capture: dict[str, object] = {
            "capture_id": capture_id,
            "active": True,
            "started_at_utc": started,
            "started_monotonic_seconds": float(time.monotonic()),
            "output_dir": str(output_dir) if output_dir is not None else None,
            "include_audio_buffers": bool(include_audio_buffers),
            "max_audio_blocks": max(0, min(256, int(max_audio_blocks))),
        }
        self._audio_diagnostics_capture = capture
        self._last_audio_diagnostics_capture_result = None
        start_engine_capture = getattr(self._engine, "start_diagnostics_capture", None)
        engine_status: dict[str, object] = {}
        if callable(start_engine_capture):
            engine_status = dict(
                start_engine_capture(
                    include_audio_buffers=bool(include_audio_buffers),
                    max_audio_blocks=int(capture["max_audio_blocks"]),
                )
            )
        self._record_runtime_event(
            "diagnostics-capture-start",
            capture_id=capture_id,
            include_audio_buffers=bool(include_audio_buffers),
        )
        return {
            **capture,
            "engine": engine_status,
        }

    def stop_audio_diagnostics_capture(self) -> dict[str, object]:
        """Stop diagnostics capture and write a timestamped bundle."""

        capture = dict(self._audio_diagnostics_capture or {})
        if not capture:
            result = dict(self._last_audio_diagnostics_capture_result or {})
            result.setdefault("active", False)
            result.setdefault("bundle_path", None)
            return result

        capture["active"] = False
        capture["stopped_at_utc"] = datetime.now(timezone.utc).isoformat()
        capture["stopped_monotonic_seconds"] = float(time.monotonic())
        self._record_runtime_event(
            "diagnostics-capture-stop",
            capture_id=str(capture.get("capture_id", "")),
        )
        stop_engine_capture = getattr(self._engine, "stop_diagnostics_capture", None)
        engine_capture: dict[str, object] = {}
        if callable(stop_engine_capture):
            engine_capture = dict(stop_engine_capture())
        audio_blocks = engine_capture.get("audio_blocks", ())
        playback_state = self._snapshot_state_for_diagnostics_bundle()
        runtime_events = tuple(playback_state.diagnostics.recent_audio_runtime_events)
        result = write_audio_diagnostics_bundle(
            capture_id=str(capture.get("capture_id") or timestamped_capture_id()),
            capture=capture,
            playback_state=playback_state,
            device_config=self._audio_diagnostics_device_config(),
            runtime_sensor_events=runtime_events,
            audio_blocks=audio_blocks if isinstance(audio_blocks, (list, tuple)) else (),
            output_dir=capture.get("output_dir"),
        )
        self._audio_diagnostics_capture = None
        self._last_audio_diagnostics_capture_result = dict(result)
        return dict(result)

    def audio_diagnostics_capture_status(self) -> dict[str, object]:
        """Return current or most recent diagnostics capture metadata."""

        if self._audio_diagnostics_capture is not None:
            engine_status: dict[str, object] = {}
            status = getattr(self._engine, "diagnostics_capture_status", None)
            if callable(status):
                engine_status = dict(status())
            return {
                **self._audio_diagnostics_capture,
                "engine": engine_status,
            }
        result = dict(self._last_audio_diagnostics_capture_result or {})
        result.setdefault("active", False)
        return result

    def _snapshot_state_for_diagnostics_bundle(self) -> PlaybackState:
        presentation = self._latest_presentation
        if presentation is not None:
            return self.snapshot_state(presentation)
        return PlaybackState(
            status=PlaybackStatus.PLAYING if self.is_playing() else PlaybackStatus.STOPPED,
            latency_ms=float(self._engine.reported_output_latency_seconds) * 1000.0,
            backend_name=self._engine.backend_name or _SOUNDDEVICE_BACKEND,
            output_sample_rate=int(self._engine.sample_rate),
            output_channels=int(self._engine.output_channels),
            diagnostics=PlaybackDiagnostics(
                glitch_count=int(self._engine.glitch_count),
                last_audio_status=self._format_audio_status(self._engine.last_audio_status),
                output_device=self._format_output_device(self._engine.output_device),
                resolved_output_device=self._format_output_device(
                    self._engine.resolved_output_device
                ),
                output_device_name=self._engine.resolved_output_device_name,
                stream_latency=self._engine.stream_latency,
                stream_blocksize=int(self._engine.stream_blocksize),
                prime_output_buffers_using_stream_callback=bool(
                    self._engine.prime_output_buffers_using_stream_callback
                ),
                requested_output_sample_rate=self._engine.output_config.requested_sample_rate,
                requested_output_channels=self._engine.output_config.requested_channels,
                device_max_output_channels=int(
                    self._engine.output_config.device_max_output_channels
                ),
                hardware_resolution_reason=self._engine.output_config.hardware_resolution_reason,
                sample_rate_resolution_reason=(
                    self._engine.output_config.sample_rate_resolution_reason
                ),
                channel_resolution_reason=self._engine.output_config.channel_resolution_reason,
                route_resolution_summary=self._last_route_resolution_summary,
                last_transition=self._last_transition,
                transition_state=self._transition_state(),
                last_track_sync_reason=self._last_track_sync_reason,
                ramp_samples_remaining=int(self._engine.ramp_samples_remaining),
                last_discontinuity_reason=self._engine.last_discontinuity_reason,
                last_ramp_reason=self._engine.last_ramp_reason,
                structural_rebuild_count=self._structural_rebuild_count,
                coalesced_edit_count=self._coalesced_edit_count,
                last_structural_rebuild_ms=self._last_structural_rebuild_ms,
                max_structural_rebuild_ms=self._max_structural_rebuild_ms,
                local_projection_build_ms=self._last_local_projection_build_ms,
                local_sync_classify_ms=self._last_local_sync_classify_ms,
                last_local_sync_change_kind=self._last_local_sync_change_kind,
                last_ipc_command=self._last_ipc_command,
                rt_last_apply_latency_ms=self._rt_last_apply_latency_ms,
                rt_last_seek_apply_latency_ms=self._rt_last_seek_apply_latency_ms,
                device_reinit_count=int(self._device_reinit_count),
                last_device_reinit_reason=self._last_device_reinit_reason,
                recent_audio_runtime_events=self._recent_audio_runtime_events(),
            ),
        )

    def _audio_diagnostics_device_config(self) -> dict[str, object]:
        config = self._engine.output_config
        payload = asdict(config)
        payload.update(
            {
                "backend_name": self._engine.backend_name or _SOUNDDEVICE_BACKEND,
                "sample_rate": int(self._engine.sample_rate),
                "channels": int(self._engine.output_channels),
                "stream_latency": self._engine.stream_latency,
                "stream_blocksize": int(self._engine.stream_blocksize),
                "reported_output_latency_seconds": float(
                    self._engine.reported_output_latency_seconds
                ),
            }
        )
        return payload

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

    def presentation_signature(
        self,
        presentation: TimelinePresentation,
    ) -> tuple[tuple[str, str], ...]:
        resolved = self._with_resolved_output_channels(presentation)
        return self._track_builder.build_track_signature(resolved)

    def snapshot_state(self, presentation: TimelinePresentation) -> PlaybackState:
        self.drain_pending_structure_sync()
        self._latest_presentation = presentation
        resolved = self._with_resolved_output_channels(presentation)
        active_tracks = self._track_builder.describe_selected_tracks(resolved)
        active_sources = [
            PlaybackSource(
                layer_id=playback_track.source_layer_id,
                take_id=playback_track.source_take_id,
                source_ref=playback_track.source_ref,
                mode=playback_track.mode,
            )
            for playback_track in active_tracks
        ]
        return PlaybackState(
            status=PlaybackStatus.PLAYING if self.is_playing() else PlaybackStatus.STOPPED,
            active_sources=active_sources,
            latency_ms=float(self._engine.reported_output_latency_seconds) * 1000.0,
            backend_name=self._engine.backend_name or _SOUNDDEVICE_BACKEND,
            active_layer_id=presentation.selected_layer_id,
            active_take_id=presentation.selected_take_id,
            output_sample_rate=int(self._engine.sample_rate),
            output_channels=int(self._engine.output_channels),
            diagnostics=PlaybackDiagnostics(
                glitch_count=int(self._engine.glitch_count),
                last_audio_status=self._format_audio_status(self._engine.last_audio_status),
                output_device=self._format_output_device(self._engine.output_device),
                resolved_output_device=self._format_output_device(
                    self._engine.resolved_output_device
                ),
                output_device_name=self._engine.resolved_output_device_name,
                stream_latency=self._engine.stream_latency,
                stream_blocksize=int(self._engine.stream_blocksize),
                prime_output_buffers_using_stream_callback=bool(
                    self._engine.prime_output_buffers_using_stream_callback
                ),
                requested_output_sample_rate=self._engine.output_config.requested_sample_rate,
                requested_output_channels=self._engine.output_config.requested_channels,
                device_max_output_channels=int(
                    self._engine.output_config.device_max_output_channels
                ),
                hardware_resolution_reason=self._engine.output_config.hardware_resolution_reason,
                sample_rate_resolution_reason=(
                    self._engine.output_config.sample_rate_resolution_reason
                ),
                channel_resolution_reason=self._engine.output_config.channel_resolution_reason,
                route_resolution_summary=self._last_route_resolution_summary,
                last_transition=self._last_transition,
                transition_state=self._transition_state(),
                last_track_sync_reason=self._last_track_sync_reason,
                ramp_samples_remaining=int(self._engine.ramp_samples_remaining),
                last_discontinuity_reason=self._engine.last_discontinuity_reason,
                last_ramp_reason=self._engine.last_ramp_reason,
                timecode_mode="internal_generated",
                timecode_lock_state="locked",
                drift_ppm=None,
                drift_ms=None,
                structural_rebuild_count=self._structural_rebuild_count,
                coalesced_edit_count=self._coalesced_edit_count,
                last_structural_rebuild_ms=self._last_structural_rebuild_ms,
                max_structural_rebuild_ms=self._max_structural_rebuild_ms,
                local_projection_build_ms=self._last_local_projection_build_ms,
                local_sync_classify_ms=self._last_local_sync_classify_ms,
                last_local_sync_change_kind=self._last_local_sync_change_kind,
                last_ipc_command=self._last_ipc_command,
                rt_last_apply_latency_ms=self._rt_last_apply_latency_ms,
                rt_last_seek_apply_latency_ms=self._rt_last_seek_apply_latency_ms,
                device_reinit_count=int(self._device_reinit_count),
                last_device_reinit_reason=self._last_device_reinit_reason,
                recent_audio_runtime_events=self._recent_audio_runtime_events(),
            ),
        )

    def _record_runtime_event(self, kind: str, **metrics: object) -> None:
        """Append one bounded controller-side audio runtime diagnostic event."""

        try:
            self._runtime_event_sequence += 1
            event: dict[str, object] = {
                "seq": int(self._runtime_event_sequence),
                "source": "playback_controller",
                "kind": str(kind or "runtime-event"),
                "monotonic_seconds": float(time.monotonic()),
                "is_playing": bool(self._engine.transport.is_playing),
                "overlay_active": bool(self._engine.overlay_active),
            }
            for key, value in metrics.items():
                if value is None:
                    continue
                if isinstance(value, (str, bool, int, float)):
                    event[str(key)] = value
                    continue
                try:
                    event[str(key)] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    event[str(key)] = str(value)
            self._recent_runtime_events.append(event)
        except Exception:
            return

    def _recent_audio_runtime_events(self) -> tuple[dict[str, object], ...]:
        engine_events = getattr(self._engine, "recent_runtime_events", ())
        try:
            events = [dict(event) for event in self._recent_runtime_events]
            events.extend(dict(event) for event in engine_events)
        except Exception:
            events = [dict(event) for event in self._recent_runtime_events]
        events.sort(key=lambda event: float(event.get("monotonic_seconds", 0.0) or 0.0))
        return tuple(events[-_RUNTIME_EVENT_LIMIT:])

    def _transition_state(self) -> str:
        if self._preview_active:
            return "preview"
        if self._engine.transport.is_playing:
            return "playing"
        if str(self._last_transition) == "pause":
            return "paused"
        return "stopped"

    def _with_resolved_output_channels(
        self,
        presentation: TimelinePresentation,
    ) -> TimelinePresentation:
        configured_channels = int(getattr(presentation, "playback_output_channels", 0) or 0)
        required_channels = self._max_required_output_channels_for_routes(presentation)
        hardware_channels = int(self._engine.output_channels)
        self._last_route_resolution_summary = self._route_resolution_summary(
            presentation,
            hardware_channels=hardware_channels,
        )
        if hardware_channels > 0:
            resolved_channels = hardware_channels
        elif configured_channels > 0 and configured_channels >= required_channels:
            return presentation
        else:
            resolved_channels = max(
                1,
                configured_channels,
                required_channels,
            )
        return replace(
            presentation,
            playback_output_channels=resolved_channels,
        )

    @staticmethod
    def _max_required_output_channels_for_routes(presentation: TimelinePresentation) -> int:
        required = 0
        for layer in getattr(presentation, "layers", ()):
            output_bus = getattr(layer, "output_bus", None)
            if not isinstance(output_bus, str):
                continue
            for parsed in parse_output_bus_spans(output_bus):
                _start_channel, end_channel = parsed
                required = max(required, end_channel)
        return max(0, int(required))

    @staticmethod
    def _route_resolution_summary(
        presentation: TimelinePresentation,
        *,
        hardware_channels: int,
    ) -> str:
        clamped: list[str] = []
        for layer in getattr(presentation, "layers", ()):
            output_bus = getattr(layer, "output_bus", None)
            if not isinstance(output_bus, str):
                continue
            over_tokens: list[str] = []
            for start_channel, end_channel in parse_output_bus_spans(output_bus):
                if end_channel > hardware_channels:
                    over_tokens.append(f"outputs_{start_channel}_{end_channel}")
            if over_tokens:
                layer_id = str(getattr(layer, "layer_id", "unknown"))
                clamped.append(f"{layer_id}:{'/'.join(over_tokens)}->pruned")
        if clamped:
            return "routes-exceed-hardware:" + ",".join(clamped)
        return "routes-fit-hardware"

    @staticmethod
    def _classify_device_reconfigure_reason(device_spec: dict[str, object]) -> str:
        if "output_device" in device_spec:
            return "device-change"
        if any(
            key in device_spec
            for key in (
                "sample_rate",
                "channels",
                "stream_latency",
                "stream_blocksize",
                "prime_output_buffers_using_stream_callback",
                "master_output_bus",
            )
        ):
            return "settings-change"
        return "device-refresh"

    def _build_engine_for_device_spec(self, device_spec: dict[str, object]) -> AudioEngine:
        if self._engine_factory is not None:
            return self._engine_factory()
        sample_rate = self._optional_int(device_spec.get("sample_rate"))
        channels = self._optional_int(device_spec.get("channels"))
        stream_blocksize = self._optional_int(device_spec.get("stream_blocksize"))
        return AudioEngine(
            sample_rate=sample_rate,
            channels=channels,
            buffer_size=self._optional_int(device_spec.get("buffer_size")) or DEFAULT_BUFFER_SIZE,
            stream_latency=device_spec.get("stream_latency"),
            stream_blocksize=stream_blocksize,
            prime_output_buffers_using_stream_callback=bool(
                device_spec.get("prime_output_buffers_using_stream_callback", True)
            ),
            output_device=device_spec.get("output_device"),
            master_output_bus=device_spec.get("master_output_bus"),
        )

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _sync_preview_state(self) -> None:
        self._preview_active = bool(self._engine.overlay_active)

    def _sync_track_plan(self, track_plan: PlaybackTrackPlan) -> None:
        self._track_builder.prune_cache(track_plan.cache_keys)
        resume_seconds = self._current_engine_time_seconds()
        resume_playing = bool(self._engine.transport.is_playing)
        structure_signature = self._track_structure_signature(track_plan.tracks)
        should_reload_tracks = (
            structure_signature != self._loaded_track_structure_signature
            or track_plan.uses_track_routing != self._loaded_uses_track_routing
        )
        if should_reload_tracks:
            if (
                structure_signature != self._loaded_track_structure_signature
                and track_plan.uses_track_routing != self._loaded_uses_track_routing
            ):
                reason = "track-signature-and-routing-changed"
            elif structure_signature != self._loaded_track_structure_signature:
                reason = "track-signature-changed"
            else:
                reason = "routing-mode-changed"
            self._replace_engine_tracks(
                track_plan,
                resume_seconds=resume_seconds,
                resume_playing=resume_playing,
                reason=reason,
            )
            return
        mix_plan = PlaybackMixPlan(
            tracks=track_plan.tracks,
            signature=track_plan.signature,
            uses_track_routing=track_plan.uses_track_routing,
        )
        if self._apply_track_mix_state(mix_plan):
            self._last_track_sync_reason = "structure-sync-track-gap"
            return
        self._last_track_sync_reason = "structure-sync-mix-applied"

    def drain_pending_structure_sync(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            return
        if self._shutdown_state != "running":
            self._drain_completed_structure_outcomes_without_apply(outcome="cancelled")
            return
        if not self._pending_structure_futures:
            return
        ready: list[tuple[int, PlaybackTrackPlan]] = []
        for future, generation in list(self._pending_structure_futures.items()):
            if not future.done():
                continue
            self._pending_structure_futures.pop(future, None)
            if future.cancelled():
                self._mark_generation_terminal_outcome(generation, "cancelled")
                continue
            try:
                plan = future.result()
            except Exception:
                if generation >= self._latest_requested_generation:
                    self._last_track_sync_reason = "structure-async-failed"
                self._mark_generation_terminal_outcome(generation, "failed")
                continue
            ready.append((generation, plan))

        if not ready:
            return

        for generation, plan in sorted(ready, key=lambda entry: entry[0]):
            if generation < self._latest_requested_generation:
                self._last_track_sync_reason = "structure-async-stale-dropped"
                self._mark_generation_terminal_outcome(generation, "stale-dropped")
                continue
            resume_seconds = self._current_engine_time_seconds()
            resume_playing = bool(self._engine.transport.is_playing)
            self._replace_engine_tracks(
                plan,
                resume_seconds=resume_seconds,
                resume_playing=resume_playing,
                reason="structure-async-applied",
                resolve_audio=False,
                preserve_transport_clock=resume_playing,
            )
            self._latest_ready_generation = generation
            self._mark_generation_terminal_outcome(generation, "applied")

    def _queue_async_structure_sync(self, presentation: TimelinePresentation) -> None:
        if self._shutdown_state != "running":
            self._last_track_sync_reason = "structure-async-shutdown-ignored"
            return
        generation = self._next_generation
        self._next_generation += 1
        self._latest_requested_generation = generation

        superseded_generations = [
            (future, pending_generation)
            for future, pending_generation in self._pending_structure_futures.items()
            if pending_generation < generation and not future.done()
        ]
        if superseded_generations:
            self._coalesced_edit_count += len(superseded_generations)
            for future, pending_generation in superseded_generations:
                if future.cancel():
                    self._mark_generation_terminal_outcome(pending_generation, "cancelled")
                    self._pending_structure_futures.pop(future, None)
            self._last_track_sync_reason = "structure-async-cancelled"

        future = self._structure_executor.submit(
            self._prepare_structure_track_plan_async,
            presentation,
        )
        self._pending_structure_futures[future] = generation
        self._last_track_sync_reason = "structure-async-queued"

    def _cancel_pending_structure_sync(self, *, reason: str) -> None:
        for future, generation in list(self._pending_structure_futures.items()):
            if not future.done():
                future.cancel()
                self._mark_generation_terminal_outcome(generation, "cancelled")
        self._pending_structure_futures.clear()
        _ = reason

    def _drain_completed_structure_outcomes_without_apply(self, *, outcome: str) -> None:
        if not self._pending_structure_futures:
            return
        for future, generation in list(self._pending_structure_futures.items()):
            if not future.done():
                continue
            self._pending_structure_futures.pop(future, None)
            self._mark_generation_terminal_outcome(generation, outcome)

    def _mark_generation_terminal_outcome(self, generation: int, outcome: str) -> None:
        if generation <= 0:
            return
        if generation in self._generation_terminal_outcomes:
            return
        self._generation_terminal_outcomes[generation] = outcome

    def _prepare_structure_track_plan_async(
        self,
        presentation: TimelinePresentation,
    ) -> PlaybackTrackPlan:
        track_plan = self._async_track_builder.build_track_plan(presentation)
        return track_plan

    def _current_engine_time_seconds(self) -> float:
        self._sync_preview_state()
        return float(self._engine.audible_time_seconds)

    def _replace_engine_tracks(
        self,
        track_plan: PlaybackTrackPlan,
        *,
        resume_seconds: float,
        resume_playing: bool,
        reason: str,
        resolve_audio: bool = True,
        preserve_transport_clock: bool = False,
    ) -> None:
        started = time.perf_counter()
        engine_tracks = []
        for playback_track in track_plan.tracks:
            if resolve_audio or playback_track.buffer is None or playback_track.sample_rate <= 0:
                self._track_builder.resolve_audio(playback_track)
            engine_track = playback_track.to_audio_track(
                engine_track_id=self._engine_track_id(track_plan, playback_track),
                engine_sample_rate=self._engine.sample_rate,
            )
            engine_track.muted = bool(playback_track.muted)
            engine_tracks.append(engine_track)
        self._engine.replace_tracks(engine_tracks)
        self._loaded_track_signature = track_plan.signature
        self._loaded_track_structure_signature = self._track_structure_signature(track_plan.tracks)
        self._loaded_uses_track_routing = track_plan.uses_track_routing
        if preserve_transport_clock:
            elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            self._rt_last_apply_latency_ms = elapsed_ms
            self._structural_rebuild_count += 1
            self._last_structural_rebuild_ms = elapsed_ms
            if elapsed_ms > self._max_structural_rebuild_ms:
                self._max_structural_rebuild_ms = elapsed_ms
            self._last_track_sync_reason = reason
            return
        if resume_seconds > 0.0:
            self._engine.seek_seconds(resume_seconds)
        if resume_playing and engine_tracks:
            self._engine.play()
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        self._rt_last_apply_latency_ms = elapsed_ms
        self._structural_rebuild_count += 1
        self._last_structural_rebuild_ms = elapsed_ms
        if elapsed_ms > self._max_structural_rebuild_ms:
            self._max_structural_rebuild_ms = elapsed_ms
        self._last_track_sync_reason = reason

    def _apply_track_mix_state(self, mix_plan: PlaybackMixPlan) -> bool:
        if mix_plan.uses_track_routing != self._loaded_uses_track_routing:
            return True
        structure_signature = self._track_structure_signature(mix_plan.tracks)
        if structure_signature != self._loaded_track_structure_signature:
            return True
        desired_mix: dict[str, tuple[bool, float, str | None]] = {}
        for playback_track in mix_plan.tracks:
            engine_track_id = self._engine_track_id(mix_plan, playback_track)
            desired_mix[engine_track_id] = (
                bool(playback_track.muted),
                _db_to_linear(playback_track.gain_db),
                playback_track.output_bus,
            )

        existing_tracks = list(self._engine.tracks)
        existing_ids = [str(track.id) for track in existing_tracks]
        desired_ids = [
            self._engine_track_id(mix_plan, playback_track) for playback_track in mix_plan.tracks
        ]
        if len(existing_ids) != len(desired_ids) or set(existing_ids) != set(desired_ids):
            return True

        apply_mix_updates = getattr(self._engine, "apply_track_mix_updates", None)
        if callable(apply_mix_updates):
            _ = apply_mix_updates(desired_mix)
            return False

        applied_change = False
        next_tracks: list[AudioTrack] = []
        for engine_track in existing_tracks:
            track_id = str(engine_track.id)
            desired = desired_mix.get(track_id)
            if desired is None:
                return True
            muted, volume, output_bus = desired
            if (
                bool(engine_track.muted) == muted
                and abs(float(engine_track.volume) - float(volume)) <= 1e-6
                and engine_track.output_bus == output_bus
            ):
                next_tracks.append(engine_track)
                continue
            applied_change = True
            cloned = AudioTrack(
                layer_id=str(engine_track.id),
                name=str(engine_track.name),
                buffer=engine_track.buffer,
                sample_rate=int(engine_track.sample_rate),
                offset=int(engine_track.offset),
                volume=float(volume),
                output_bus=output_bus,
            )
            cloned.muted = bool(muted)
            cloned.solo = bool(engine_track.solo)
            next_tracks.append(cloned)

        if applied_change:
            self._engine.replace_tracks(next_tracks)
        return False

    @staticmethod
    def _track_structure_signature(
        tracks: tuple[PlaybackTrack, ...] | list[PlaybackTrack],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (str(playback_track.track_id), str(playback_track.source_key))
            for playback_track in tracks
        )

    def _engine_track_id(
        self,
        track_plan: PlaybackTrackPlan | PlaybackMixPlan,
        playback_track: PlaybackTrack,
    ) -> str:
        if track_plan.uses_track_routing:
            return f"{self._ROUTED_TRACK_PREFIX}{playback_track.track_id}"
        return self._PRIMARY_TRACK_ID

    @staticmethod
    def _format_audio_status(status: object) -> str | None:
        if status is None:
            return None
        value = str(status).strip()
        return value or None

    @staticmethod
    def _format_output_device(output_device: object) -> str | None:
        if output_device is None:
            return "default"
        value = str(output_device).strip()
        return value or "default"


PresentationPlaybackRuntime = PlaybackController

__all__ = [
    "PlaybackController",
    "PresentationPlaybackRuntime",
]

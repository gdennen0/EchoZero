"""
AudioEngine: Transport, mixer, and output stream for EZ playback.
Exists because EZ needs one simple DAW-style engine surface independent of any device library.
Connects application playback control to one output backend through a narrow callback-driven contract.
"""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Callable

import numpy as np

from echozero.audio.clock import Clock, ClockSubscriber
from echozero.audio.crossfade import CrossfadeBuffer
from echozero.audio.layer import AudioLayer, AudioTrack, resample_buffer
from echozero.audio.mixer import Mixer
from echozero.audio.output_backend import (
    DEFAULT_BUFFER_SIZE,
    AudioOutputBackend,
    AudioOutputConfig,
)
from echozero.audio.sounddevice_backend import (
    SounddeviceBackend,
    _resolve_output_defaults,
    _resolve_stream_defaults,
)
from echozero.audio.transport import Transport

DEFAULT_SCRATCH_FRAMES = 32768
_DECLICK_DURATION_SECONDS = 0.004
_OVERLAY_EDGE_FADE_SECONDS = 0.008
_TRANSPORT_RELEASE_DURATION_SECONDS = 0.02
_DECLICK_DELTA_THRESHOLD = 0.05
_RUNTIME_EVENT_LIMIT = 32


def _declick_ramp_samples(sample_rate: int) -> int:
    """Return the standard render-boundary declick length for a sample rate."""

    return max(2, int(round(max(1, int(sample_rate)) * _DECLICK_DURATION_SECONDS)))


def _smooth_fade_out_curve(samples: int) -> np.ndarray:
    """Return a transport release fade with zero slope at both ends."""

    sample_count = max(2, int(samples))
    progress = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    smoothstep = progress * progress * (np.float32(3.0) - (np.float32(2.0) * progress))
    return (np.float32(1.0) - smoothstep).astype(np.float32, copy=False)


def _copy_with_edge_declick_fades(
    buffer: np.ndarray,
    fade_in: np.ndarray,
    fade_out: np.ndarray,
) -> np.ndarray:
    """Copy one preview/overlay buffer and apply non-mutating edge fades."""

    faded = np.array(buffer, dtype=np.float32, copy=True)
    if faded.size == 0:
        return faded
    frames = int(faded.shape[0])
    if frames <= 1:
        return faded
    if frames < 4:
        fade_in_frames = frames
        fade_out_frames = frames
    else:
        fade_in_frames = min(frames // 2, int(fade_in.shape[0]))
        fade_out_frames = min(frames // 2, int(fade_out.shape[0]))
    if fade_in_frames > 1:
        fade_in_ramp = (
            fade_in[:fade_in_frames]
            if fade_in_frames == int(fade_in.shape[0])
            else np.linspace(0.0, 1.0, fade_in_frames, dtype=np.float32)
        )
        if faded.ndim == 1:
            faded[:fade_in_frames] *= fade_in_ramp
        else:
            faded[:fade_in_frames] *= fade_in_ramp[:, None]
    if fade_out_frames > 1:
        tail_start = frames - fade_out_frames
        fade_out_ramp = (
            fade_out[:fade_out_frames]
            if fade_out_frames == int(fade_out.shape[0])
            else np.linspace(1.0, 0.0, fade_out_frames, dtype=np.float32)
        )
        if faded.ndim == 1:
            faded[tail_start:frames] *= fade_out_ramp
        else:
            faded[tail_start:frames] *= fade_out_ramp[:, None]
    return faded


def _apply_declick_fade_out_slice(
    buffer: np.ndarray,
    start: int,
    stop: int,
    fade_out: np.ndarray,
) -> None:
    """Apply one in-place fade to an output slice that must end at silence."""

    fade_frames = max(0, int(stop) - int(start))
    if fade_frames <= 1:
        if fade_frames == 1:
            buffer[start:stop] = 0.0
        return
    if fade_frames == int(fade_out.shape[0]):
        if buffer.ndim == 1:
            buffer[start:stop] *= fade_out[:fade_frames]
        else:
            buffer[start:stop] *= fade_out[:fade_frames, None]
        return
    denominator = np.float32(fade_frames - 1)
    for index in range(fade_frames):
        gain = np.float32(1.0) - (np.float32(index) / denominator)
        buffer[start + index] *= gain


def _create_audio_buffer(frames: int, channels: int) -> np.ndarray:
    """Allocate one scratch buffer matching the engine output channel layout."""

    if channels <= 1:
        return np.zeros(frames, dtype=np.float32)
    return np.zeros((frames, channels), dtype=np.float32)


class AudioEngine:
    """Playback engine that owns one transport clock, mixer, and output stream."""

    __slots__ = (
        "_backend",
        "_output_config",
        "_clock",
        "_transport",
        "_mixer",
        "_crossfade",
        "_stream",
        "_buffer_size",
        "_channels",
        "_stream_blocksize",
        "_stream_latency",
        "_prime_output_buffers_using_stream_callback",
        "_output_device",
        "_active",
        "_end_of_content",
        "_reported_output_latency_seconds",
        "_last_audible_time_seconds",
        "_last_audible_monotonic_seconds",
        "_output_scratch",
        "_pre_scratch",
        "_post_scratch",
        "_glitch_count",
        "_last_status",
        "_last_output_tail",
        "_last_callback_was_playing",
        "_pending_pause_reason",
        "_pending_pause_fade_index",
        "_pending_pause_bridge_delta",
        "_pending_pause_bridge_active",
        "_pending_declick",
        "_pending_declick_reason",
        "_declick_ramp_samples",
        "_declick_fade_in",
        "_declick_fade_out",
        "_overlay_edge_fade_in",
        "_overlay_edge_fade_out",
        "_transport_release_fade_out",
        "_declick_correction_delta",
        "_declick_correction_total",
        "_declick_correction_remaining",
        "_last_discontinuity_reason",
        "_last_ramp_reason",
        "_overlay_buffer",
        "_overlay_playback_buffer",
        "_overlay_read_index",
        "_overlay_volume",
        "_overlay_release_buffer",
        "_overlay_release_read_index",
        "_overlay_release_volume",
        "_transport_release_buffer",
        "_transport_release_read_index",
        "_transport_release_active_frames",
        "_runtime_event_sequence",
        "_recent_runtime_events",
        "_diagnostics_capture_active",
        "_diagnostics_capture_include_audio",
        "_diagnostics_capture_blocks",
        "_diagnostics_capture_max_blocks",
        "_diagnostics_capture_sequence",
    )

    def __init__(
        self,
        sample_rate: int | None = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        channels: int | None = None,
        stream_factory: Callable[..., Any] | None = None,
        stream_blocksize: int | None = None,
        stream_latency: str | float | None = None,
        prime_output_buffers_using_stream_callback: bool = True,
        output_device: int | str | None = None,
        master_output_bus: object = None,
        backend: AudioOutputBackend | None = None,
    ) -> None:
        self._backend = backend or SounddeviceBackend(stream_factory=stream_factory)
        self._output_config = self._backend.resolve_output_config(
            sample_rate=sample_rate,
            channels=channels,
            buffer_size=buffer_size,
            output_device=output_device,
            stream_blocksize=stream_blocksize,
            stream_latency=stream_latency,
            prime_output_buffers_using_stream_callback=(
                prime_output_buffers_using_stream_callback
            ),
        )
        self._clock = Clock(sample_rate=self._output_config.sample_rate)
        self._transport = Transport(self._clock)
        self._mixer = Mixer()
        self._mixer.master_output_bus = master_output_bus
        self._mixer.configure_gain_smoothing(sample_rate=self._clock.sample_rate)
        self._crossfade = CrossfadeBuffer(
            crossfade_samples=int(self._output_config.sample_rate * 0.004)
        )
        self._stream: Any = None
        self._buffer_size = int(buffer_size)
        self._channels = int(self._output_config.channels)
        self._stream_blocksize = int(self._output_config.blocksize)
        self._stream_latency = self._output_config.latency
        self._prime_output_buffers_using_stream_callback = (
            self._output_config.prime_output_buffers_using_stream_callback
        )
        self._output_device = self._output_config.output_device
        self._active = False
        self._end_of_content = False
        self._reported_output_latency_seconds = 0.0
        self._last_audible_time_seconds: float | None = None
        self._last_audible_monotonic_seconds: float | None = None
        scratch_size = max(buffer_size * 2, DEFAULT_SCRATCH_FRAMES)
        self._output_scratch = _create_audio_buffer(scratch_size, self._channels)
        self._pre_scratch = _create_audio_buffer(scratch_size, self._channels)
        self._post_scratch = _create_audio_buffer(scratch_size, self._channels)
        self._glitch_count = 0
        self._last_status: Any = None
        if self._channels <= 1:
            self._last_output_tail = np.zeros(1, dtype=np.float32)
        else:
            self._last_output_tail = np.zeros(self._channels, dtype=np.float32)
        self._last_callback_was_playing = False
        self._pending_pause_reason = ""
        self._pending_pause_fade_index = 0
        self._pending_pause_bridge_delta = np.zeros_like(self._last_output_tail)
        self._pending_pause_bridge_active = False
        self._pending_declick = True
        self._pending_declick_reason = "engine-startup"
        self._declick_ramp_samples = _declick_ramp_samples(self._output_config.sample_rate)
        self._declick_fade_in = np.linspace(
            0.0,
            1.0,
            self._declick_ramp_samples,
            dtype=np.float32,
        )
        self._declick_fade_out = np.linspace(
            1.0,
            0.0,
            self._declick_ramp_samples,
            dtype=np.float32,
        )
        overlay_edge_fade_samples = max(
            self._declick_ramp_samples,
            int(round(self._output_config.sample_rate * _OVERLAY_EDGE_FADE_SECONDS)),
        )
        self._overlay_edge_fade_in = np.linspace(
            0.0,
            1.0,
            overlay_edge_fade_samples,
            dtype=np.float32,
        )
        self._overlay_edge_fade_out = np.linspace(
            1.0,
            0.0,
            overlay_edge_fade_samples,
            dtype=np.float32,
        )
        transport_release_samples = max(
            2,
            int(round(self._output_config.sample_rate * _TRANSPORT_RELEASE_DURATION_SECONDS)),
        )
        self._transport_release_fade_out = _smooth_fade_out_curve(transport_release_samples)
        self._declick_correction_delta = np.zeros_like(self._last_output_tail)
        self._declick_correction_total = 0
        self._declick_correction_remaining = 0
        self._last_discontinuity_reason: str | None = "engine-startup"
        self._last_ramp_reason: str | None = None
        self._overlay_buffer: np.ndarray | None = None
        self._overlay_playback_buffer: np.ndarray | None = None
        self._overlay_read_index = 0
        self._overlay_volume = np.float32(1.0)
        self._overlay_release_buffer: np.ndarray | None = None
        self._overlay_release_read_index = 0
        self._overlay_release_volume = np.float32(1.0)
        self._transport_release_buffer = _create_audio_buffer(
            transport_release_samples,
            self._channels,
        )
        self._transport_release_read_index = 0
        self._transport_release_active_frames = 0
        self._runtime_event_sequence = 0
        self._recent_runtime_events = deque(maxlen=_RUNTIME_EVENT_LIMIT)
        self._diagnostics_capture_active = False
        self._diagnostics_capture_include_audio = False
        self._diagnostics_capture_blocks = deque(maxlen=0)
        self._diagnostics_capture_max_blocks = 0
        self._diagnostics_capture_sequence = 0

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def transport(self) -> Transport:
        return self._transport

    @property
    def mixer(self) -> Mixer:
        return self._mixer

    @property
    def tracks(self) -> tuple[AudioTrack, ...]:
        """Snapshot of loaded playback tracks."""

        return self._mixer.tracks

    @property
    def layers(self) -> tuple[AudioTrack, ...]:
        """Compatibility alias for callers that still say `layers`."""

        return self.tracks

    @property
    def sample_rate(self) -> int:
        return int(self._clock.sample_rate)

    @property
    def output_channels(self) -> int:
        return self._channels

    @property
    def master_output_bus(self) -> str | None:
        """Default route for tracks without an explicit output bus."""

        return self._mixer.master_output_bus

    def set_master_output_bus(self, output_bus: object) -> None:
        """Set the default route for un-routed master/song playback."""

        before = self._mixer.master_output_bus
        self._mixer.master_output_bus = output_bus
        if self._mixer.master_output_bus != before:
            self._request_declick("master-output-bus-changed")

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def reported_output_latency_seconds(self) -> float:
        return self._reported_output_latency_seconds

    @property
    def audible_time_seconds(self) -> float:
        clock_time = float(self._clock.position_seconds)
        if not self._transport.is_playing:
            return clock_time
        snapshot = self._last_audible_time_seconds
        snapshot_monotonic = self._last_audible_monotonic_seconds
        if snapshot is None or snapshot_monotonic is None:
            return max(0.0, clock_time - self._reported_output_latency_seconds)
        extrapolated = snapshot + max(0.0, time.monotonic() - snapshot_monotonic)
        return max(0.0, min(extrapolated, clock_time))

    @property
    def reached_end(self) -> bool:
        return self._end_of_content

    @property
    def glitch_count(self) -> int:
        return self._glitch_count

    @property
    def last_audio_status(self) -> Any:
        return self._last_status

    @property
    def backend_name(self) -> str:
        return str(getattr(self._backend, "name", "unknown"))

    @property
    def output_device(self) -> int | str | None:
        return self._output_device

    @property
    def output_config(self) -> AudioOutputConfig:
        """Return the resolved hardware and stream config for diagnostics."""

        return self._output_config

    @property
    def resolved_output_device(self) -> int | str | None:
        """Return the concrete backend output device selected for this engine."""

        return self._output_config.resolved_output_device

    @property
    def resolved_output_device_name(self) -> str | None:
        """Return the backend-reported output device name when available."""

        return self._output_config.resolved_output_device_name

    @property
    def stream_latency(self) -> str | float | None:
        return self._stream_latency

    @property
    def stream_blocksize(self) -> int:
        return self._stream_blocksize

    @property
    def prime_output_buffers_using_stream_callback(self) -> bool:
        return self._prime_output_buffers_using_stream_callback

    @property
    def ramp_samples_remaining(self) -> int:
        """Return current declick/gain ramp samples left for diagnostics."""

        pending_declick = self._declick_ramp_samples if self._pending_declick else 0
        pending_pause = 0
        if self._pending_pause_reason:
            pending_pause = int(self._transport_release_fade_out.shape[0]) - int(
                self._pending_pause_fade_index
            )
        correction_remaining = int(self._declick_correction_remaining)
        mixer_remaining = int(getattr(self._mixer, "ramp_samples_remaining", 0))
        return max(0, pending_declick, pending_pause, correction_remaining, mixer_remaining)

    @property
    def last_discontinuity_reason(self) -> str | None:
        """Return the last output-boundary discontinuity reason."""

        return self._last_discontinuity_reason

    @property
    def last_ramp_reason(self) -> str | None:
        """Return the last ramp/declick reason."""

        return self._last_ramp_reason

    @property
    def recent_runtime_events(self) -> tuple[dict[str, object], ...]:
        """Return recent playback-thread sensor events for diagnostics."""

        return tuple(dict(event) for event in self._recent_runtime_events)

    def start_diagnostics_capture(
        self,
        *,
        include_audio_buffers: bool = True,
        max_audio_blocks: int = 64,
    ) -> dict[str, object]:
        """Arm a bounded dev diagnostics capture on the callback path."""

        max_blocks = max(0, min(256, int(max_audio_blocks)))
        self._diagnostics_capture_active = True
        self._diagnostics_capture_include_audio = bool(include_audio_buffers and max_blocks > 0)
        self._diagnostics_capture_max_blocks = max_blocks
        self._diagnostics_capture_blocks = deque(maxlen=max_blocks)
        self._diagnostics_capture_sequence = 0
        self._record_runtime_event(
            "diagnostics-capture-start",
            reason="manual-capture",
            include_audio_buffers=bool(self._diagnostics_capture_include_audio),
            max_audio_blocks=int(max_blocks),
        )
        return self.diagnostics_capture_status()

    def stop_diagnostics_capture(self) -> dict[str, object]:
        """Disarm diagnostics capture and return buffered callback blocks."""

        was_active = bool(self._diagnostics_capture_active)
        blocks = tuple(dict(block) for block in self._diagnostics_capture_blocks)
        self._diagnostics_capture_active = False
        self._diagnostics_capture_include_audio = False
        self._diagnostics_capture_blocks = deque(maxlen=0)
        self._diagnostics_capture_max_blocks = 0
        if was_active:
            self._record_runtime_event(
                "diagnostics-capture-stop",
                reason="manual-capture",
                captured_audio_blocks=len(blocks),
            )
        return {
            "active": False,
            "audio_blocks": blocks,
            "audio_block_count": len(blocks),
        }

    def diagnostics_capture_status(self) -> dict[str, object]:
        """Return the current bounded diagnostics capture state."""

        return {
            "active": bool(self._diagnostics_capture_active),
            "include_audio_buffers": bool(self._diagnostics_capture_include_audio),
            "audio_block_count": int(len(self._diagnostics_capture_blocks)),
            "max_audio_blocks": int(self._diagnostics_capture_max_blocks),
        }

    def create_track(
        self,
        layer_id: str,
        buffer: np.ndarray,
        sample_rate: int,
        name: str | None = None,
        offset: int = 0,
        volume: float = 1.0,
        output_bus: str | None = None,
    ) -> AudioTrack:
        """Create one engine-ready playback track."""

        return AudioTrack(
            layer_id=layer_id,
            name=name or layer_id,
            buffer=buffer,
            sample_rate=sample_rate,
            offset=offset,
            volume=volume,
            engine_sample_rate=self._clock.sample_rate,
            output_bus=output_bus,
        )

    def set_track(self, track: AudioTrack) -> AudioTrack:
        """Add or replace one playback track by ID."""

        self._mixer.remove_track(track.id)
        self._mixer.add_track(track)
        return track

    def load_track(
        self,
        track_id: str,
        buffer: np.ndarray,
        sample_rate: int,
        *,
        name: str | None = None,
        offset: int = 0,
        volume: float = 1.0,
        output_bus: str | None = None,
    ) -> AudioTrack:
        """Create and register one playback track in a single call."""

        track = self.create_track(
            track_id,
            buffer,
            sample_rate,
            name=name,
            offset=offset,
            volume=volume,
            output_bus=output_bus,
        )
        return self.set_track(track)

    def replace_tracks(self, tracks: list[AudioTrack]) -> None:
        """Atomically replace the current playback track set."""

        self._mixer.replace_tracks(tracks)
        self._request_declick("tracks-replaced")

    def apply_track_mix_updates(
        self,
        updates: dict[str, tuple[bool, float, str | None]],
    ) -> bool:
        """Apply mix-only updates without replacing engine track objects."""

        _will_apply, requires_declick = self._track_mix_update_change_flags(updates)
        if requires_declick:
            self._request_declick("mix-update")
        applied, applied_requires_declick = self._mixer.apply_track_mix_updates(updates)
        if applied and not self._transport.is_playing:
            self._mixer.snap_track_mix_envelopes(updates)
        if applied_requires_declick:
            self._request_declick("mix-update")
        return bool(applied)

    def clear_tracks(self) -> None:
        """Remove every playback track from the engine mixer."""

        self._mixer.clear_tracks()
        self._request_declick("tracks-cleared")

    def add_layer(
        self,
        layer_id: str,
        buffer: np.ndarray,
        sample_rate: int,
        name: str | None = None,
        offset: int = 0,
        volume: float = 1.0,
        output_bus: str | None = None,
    ) -> AudioTrack:
        """Compatibility alias for callers that still say `add_layer`."""

        return self.load_track(
            layer_id,
            buffer,
            sample_rate,
            name=name,
            offset=offset,
            volume=volume,
            output_bus=output_bus,
        )

    def remove_track(self, track_id: str) -> AudioTrack | None:
        """Remove one playback track from the mixer."""

        return self._mixer.remove_track(track_id)

    def remove_layer(self, layer_id: str) -> AudioTrack | None:
        """Compatibility alias for callers that still say `remove_layer`."""

        return self.remove_track(layer_id)

    def get_track(self, track_id: str) -> AudioTrack | None:
        """Look up one playback track by ID."""

        return self._mixer.get_track(track_id)

    def get_layer(self, layer_id: str) -> AudioTrack | None:
        """Compatibility alias for callers that still say `get_layer`."""

        return self.get_track(layer_id)

    def play(self) -> None:
        self._end_of_content = False
        self._clear_pending_pause()
        self._clear_transport_release()
        if not self._active:
            self._open_stream()
        self._request_declick("play")
        self._transport.play()

    def pause(self) -> None:
        self._request_pending_pause("pause")

    def stop(self) -> None:
        self._end_of_content = False
        self._clear_pending_pause()
        self._schedule_transport_release_tail("stop")
        self._transport.stop()
        self._last_audible_time_seconds = 0.0
        self._last_audible_monotonic_seconds = None
        self.stop_overlay()
        self._request_declick("stop")

    def seek(self, position_samples: int) -> None:
        self._end_of_content = False
        self._clear_pending_pause()
        self._clear_transport_release()
        self._transport.seek(position_samples)
        self._last_audible_time_seconds = self._clock.position_seconds
        self._last_audible_monotonic_seconds = None
        self._request_declick("seek")

    def seek_seconds(self, seconds: float) -> None:
        self.seek(int(seconds * self._clock.sample_rate))

    def toggle_play_pause(self) -> None:
        self._end_of_content = False
        if self._transport.is_playing:
            if self._pending_pause_reason:
                self._clear_pending_pause()
                self._last_discontinuity_reason = "toggle-play"
                self._last_ramp_reason = "pending-pause-cancelled"
                self._request_declick("toggle-play")
                return
            self._request_pending_pause("toggle-pause")
            return
        if not self._active:
            self._open_stream()
        self._clear_pending_pause()
        self._clear_transport_release()
        self._request_declick("toggle-play")
        self._transport.toggle_play_pause()

    def shutdown(self) -> None:
        self._clear_pending_pause()
        self._transport.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False
        self._reported_output_latency_seconds = 0.0
        self._last_audible_time_seconds = None
        self._last_audible_monotonic_seconds = None
        self._request_declick("shutdown")
        self.stop_overlay()
        self._clear_pending_pause()
        self._clear_transport_release()

    def request_declick(self) -> None:
        """Force one output-boundary declick on next callback buffer."""

        self._request_declick("manual")

    def _request_declick(self, reason: str) -> None:
        self._pending_declick = True
        self._pending_declick_reason = str(reason or "unspecified")
        self._last_discontinuity_reason = self._pending_declick_reason
        self._record_runtime_event(
            "discontinuity-request",
            reason=self._pending_declick_reason,
        )

    def _record_runtime_event(
        self,
        kind: str,
        *,
        reason: str = "",
        **metrics: object,
    ) -> None:
        """Append one bounded, best-effort runtime sensor event."""

        try:
            self._runtime_event_sequence += 1
            event: dict[str, object] = {
                "seq": int(self._runtime_event_sequence),
                "source": "audio_engine",
                "kind": str(kind or "runtime-event"),
                "reason": str(reason or ""),
                "monotonic_seconds": float(time.monotonic()),
                "clock_samples": int(self._clock.position),
                "clock_seconds": float(self._clock.position_seconds),
                "is_playing": bool(self._transport.is_playing),
                "overlay_active": bool(self.overlay_active),
                "ramp_samples_remaining": int(self.ramp_samples_remaining),
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

    def _track_mix_update_change_flags(
        self,
        updates: dict[str, tuple[bool, float, str | None]],
    ) -> tuple[bool, bool]:
        applied_change = False
        requires_declick = False
        if not updates:
            return (False, False)
        for track in self._mixer.tracks:
            desired = updates.get(str(track.id))
            if desired is None:
                continue
            muted, volume, output_bus = desired
            if bool(track.muted) != bool(muted):
                applied_change = True
                requires_declick = True
            if abs(float(track.volume) - float(volume)) > 1e-6:
                applied_change = True
            if track.output_bus != output_bus:
                applied_change = True
                requires_declick = True
        return (applied_change, requires_declick)

    @property
    def overlay_active(self) -> bool:
        return self._overlay_buffer is not None

    def play_overlay(
        self,
        buffer: np.ndarray,
        sample_rate: int,
        *,
        volume: float = 1.0,
    ) -> bool:
        """Play one-shot overlay audio on the main stream without a second engine."""

        if buffer.size == 0 or sample_rate <= 0:
            self.stop_overlay()
            return False
        replacing_overlay = self._overlay_playback_buffer is not None
        source = np.array(buffer, dtype=np.float32, copy=True)
        prepared = source
        source_channel_count = 1 if source.ndim <= 1 else int(source.shape[1])
        resampled = bool(int(sample_rate) != int(self._clock.sample_rate))
        if int(sample_rate) != int(self._clock.sample_rate):
            prepared = resample_buffer(prepared, int(sample_rate), int(self._clock.sample_rate))
        if prepared.size == 0:
            self.stop_overlay()
            return False
        if self._channels <= 1:
            if prepared.ndim == 2:
                prepared = np.mean(prepared, axis=1, dtype=np.float32)
            prepared = np.asarray(prepared, dtype=np.float32)
        else:
            if prepared.ndim == 1:
                prepared = np.repeat(prepared[:, None], self._channels, axis=1)
            elif prepared.shape[1] < self._channels:
                pad_channels = self._channels - int(prepared.shape[1])
                pad = np.repeat(prepared[:, -1:], pad_channels, axis=1)
                prepared = np.concatenate((prepared, pad), axis=1)
            elif prepared.shape[1] > self._channels:
                prepared = np.asarray(prepared[:, : self._channels], dtype=np.float32)
        playback = _copy_with_edge_declick_fades(
            prepared,
            self._overlay_edge_fade_in,
            self._overlay_edge_fade_out,
        )
        self._schedule_overlay_release_tail()
        self._overlay_buffer = source
        self._overlay_playback_buffer = playback
        self._overlay_read_index = 0
        self._overlay_volume = np.float32(max(0.0, float(volume)))
        self._record_runtime_event(
            "overlay-replace" if replacing_overlay else "overlay-start",
            reason="play-overlay",
            source_frames=int(source.shape[0]),
            playback_frames=int(playback.shape[0]),
            sample_rate=int(sample_rate),
            output_sample_rate=int(self._clock.sample_rate),
            source_channels=int(source_channel_count),
            output_channels=int(self._channels),
            resampled=resampled,
            resample_source_rate=int(sample_rate),
            resample_target_rate=int(self._clock.sample_rate),
            volume=float(self._overlay_volume),
        )
        self._request_declick("overlay-start")
        if not self._active:
            self._open_stream()
        return True

    def stop_overlay(self) -> None:
        """Stop one-shot overlay playback and clear staged overlay samples."""

        stopping_overlay = self._overlay_playback_buffer is not None
        self._schedule_overlay_release_tail()
        self._clear_overlay()
        if stopping_overlay:
            self._record_runtime_event("overlay-stop", reason="stop-overlay")
        self._request_declick("overlay-stop")

    def _clear_overlay(self) -> None:
        self._overlay_buffer = None
        self._overlay_playback_buffer = None
        self._overlay_read_index = 0
        self._overlay_volume = np.float32(1.0)

    def _schedule_overlay_release_tail(self) -> None:
        overlay = self._overlay_playback_buffer
        start = int(self._overlay_read_index)
        if overlay is None or start <= 0 or start >= int(overlay.shape[0]):
            return
        available = int(overlay.shape[0]) - start
        release_frames = min(available, int(self._overlay_edge_fade_out.shape[0]))
        if release_frames <= 1:
            return
        release = np.array(overlay[start : start + release_frames], dtype=np.float32, copy=True)
        if release_frames == int(self._overlay_edge_fade_out.shape[0]):
            release_ramp = self._overlay_edge_fade_out[:release_frames]
        else:
            release_ramp = np.linspace(1.0, 0.0, release_frames, dtype=np.float32)
        if release.ndim == 1:
            release *= release_ramp
        else:
            release *= release_ramp[:, None]
        self._overlay_release_buffer = release
        self._overlay_release_read_index = 0
        self._overlay_release_volume = np.float32(self._overlay_volume)
        self._record_runtime_event(
            "overlay-release",
            reason="overlay-replaced-or-stopped",
            release_frames=int(release_frames),
            overlay_read_index=int(start),
            overlay_frames=int(overlay.shape[0]),
        )

    def _schedule_transport_release_tail(
        self,
        reason: str,
    ) -> None:
        if not self._transport.is_playing or self._mixer.track_count <= 0:
            self._clear_transport_release()
            return
        release_frames = int(self._transport_release_fade_out.shape[0])
        release_frames = min(release_frames, int(self._pre_scratch.shape[0]))
        if release_frames <= 1:
            self._clear_transport_release()
            return
        release = self._transport_release_buffer[:release_frames]
        release_position = int(self._clock.position)
        self._mixer.read_mix_into(release, release_position, release_frames)
        if not np.any(np.abs(release) > 1e-7):
            self._clear_transport_release()
            return
        release_ramp = self._transport_release_fade_out[:release_frames]
        if release.ndim == 1:
            release *= release_ramp
        else:
            release *= release_ramp[:, None]
        self._transport_release_read_index = 0
        self._transport_release_active_frames = int(release_frames)
        self._record_runtime_event(
            "transport-release",
            reason=str(reason or "transport-release"),
            release_frames=int(release_frames),
            clock_samples=int(self._clock.position),
            release_start_samples=int(release_position),
            peak_abs=float(np.max(np.abs(release))) if release.size else 0.0,
        )

    def _request_pending_pause(self, reason: str) -> None:
        if not self._transport.is_playing:
            self._transport.pause()
            self._clear_pending_pause()
            return
        if not self._active:
            self._open_stream()
        if not self._pending_pause_reason:
            self._pending_pause_fade_index = 0
        self._pending_pause_reason = str(reason or "pause")
        self._pending_declick = False
        self._pending_declick_reason = ""
        self._clear_declick_correction()
        self._last_discontinuity_reason = self._pending_pause_reason
        self._last_ramp_reason = "pending-pause"
        self._record_runtime_event(
            "pending-pause-request",
            reason=self._pending_pause_reason,
            clock_samples=int(self._clock.position),
            fade_frames=int(self._transport_release_fade_out.shape[0]),
        )

    def _clear_pending_pause(self) -> None:
        self._pending_pause_reason = ""
        self._pending_pause_fade_index = 0
        self._pending_pause_bridge_delta[:] = 0.0
        self._pending_pause_bridge_active = False

    def _clear_declick_correction(self) -> None:
        self._declick_correction_total = 0
        self._declick_correction_remaining = 0
        self._declick_correction_delta = np.zeros_like(self._last_output_tail)

    def _apply_pending_pause_envelope(
        self,
        mixed: np.ndarray,
        *,
        frames: int,
        position: int,
    ) -> None:
        if not self._pending_pause_reason or frames <= 0:
            return
        start = int(self._pending_pause_fade_index)
        total = int(self._transport_release_fade_out.shape[0])
        if start <= 0:
            self._record_runtime_event(
                "pending-pause-start",
                reason=self._pending_pause_reason,
                clock_samples=int(self._clock.position),
                fade_start_samples=int(position),
                fade_frames=int(total),
            )
            self._capture_pending_pause_bridge(mixed)
        if start >= total:
            self._finish_pending_pause(frames=0)
            return
        fade_frames = min(int(frames), total - start)
        self._apply_pending_pause_bridge(mixed, start=start, frames=fade_frames)
        ramp = self._transport_release_fade_out[start : start + fade_frames]
        if mixed.ndim == 1:
            mixed[:fade_frames] *= ramp
        else:
            mixed[:fade_frames] *= ramp[:, None]
        self._pending_pause_fade_index = start + fade_frames
        self._last_ramp_reason = "pending-pause"
        if fade_frames < int(frames):
            mixed[fade_frames:frames] = 0.0
            self._finish_pending_pause(frames=fade_frames)
        elif self._pending_pause_fade_index >= total:
            self._finish_pending_pause(frames=fade_frames)

    def _finish_pending_pause(self, *, frames: int) -> None:
        reason = self._pending_pause_reason or "pause"
        total = int(self._transport_release_fade_out.shape[0])
        self._transport.pause()
        self._last_audible_monotonic_seconds = None
        self._record_runtime_event(
            "pending-pause-complete",
            reason=reason,
            clock_samples=int(self._clock.position),
            fade_frames=int(total),
            rendered_frames=int(frames),
        )
        self._clear_pending_pause()

    def _capture_pending_pause_bridge(self, mixed: np.ndarray) -> None:
        tail = np.asarray(self._last_output_tail, dtype=np.float32)
        if mixed.ndim == 1:
            delta_sample = np.float32(mixed[0]) - np.float32(tail[0])
            if abs(float(delta_sample)) < _DECLICK_DELTA_THRESHOLD:
                return
            self._pending_pause_bridge_delta[0] = delta_sample
            peak_delta = abs(float(delta_sample))
        else:
            delta = np.asarray(mixed[0], dtype=np.float32) - tail
            if not bool(np.any(np.abs(delta) >= _DECLICK_DELTA_THRESHOLD)):
                return
            self._pending_pause_bridge_delta[:] = delta
            peak_delta = float(np.max(np.abs(delta)))
        self._pending_pause_bridge_active = True
        self._last_ramp_reason = "pending-pause-bridge"
        self._record_runtime_event(
            "pending-pause-bridge",
            reason=self._pending_pause_reason,
            frames=int(self._transport_release_fade_out.shape[0]),
            peak_delta=peak_delta,
        )

    def _apply_pending_pause_bridge(
        self,
        mixed: np.ndarray,
        *,
        start: int,
        frames: int,
    ) -> None:
        if not self._pending_pause_bridge_active or frames <= 0:
            return
        bridge = self._transport_release_fade_out[start : start + frames]
        if mixed.ndim == 1:
            mixed[:frames] -= self._pending_pause_bridge_delta[0] * bridge
            return
        mixed[:frames] -= self._pending_pause_bridge_delta[None, :] * bridge[:, None]

    def _clear_transport_release(self) -> None:
        self._transport_release_read_index = 0
        self._transport_release_active_frames = 0

    def _open_stream(self) -> None:
        if self._active:
            return
        self._stream = self._backend.open_output_stream(
            self._audio_callback,
            self._output_config,
        )
        self._stream.start()
        self._reported_output_latency_seconds = self._coerce_output_latency_seconds(
            getattr(self._stream, "latency", 0.0)
        )
        self._active = True

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """Render one output buffer on the real-time thread."""

        if status:
            self._glitch_count += 1
            self._last_status = status
        if frames > len(self._output_scratch):
            outdata[:] = 0
            self._glitch_count += 1
            self._last_status = (
                f"callback_frames_exceeded_scratch:{frames}>{len(self._output_scratch)}"
            )
            return
        mixed = self._output_scratch[:frames]
        end_fade_position = -1
        end_fade_duration = 0
        render_position = int(self._clock.position)
        callback_is_playing = bool(self._transport.is_playing)
        transport_state_changed = callback_is_playing != bool(self._last_callback_was_playing)
        if not callback_is_playing:
            if self._pending_pause_reason:
                self._clear_pending_pause()
            mixed[:] = 0.0
        else:
            position = self._clock.advance(frames)
            render_position = int(position)
            self._update_callback_timing_snapshot(time_info)
            duration = self._mixer.duration_samples
            if duration > 0 and not self._clock.loop_enabled and position >= duration:
                mixed[:] = 0.0
                self._transport.pause()
                self._end_of_content = True
            else:
                wrap_offset = self._clock.last_wrap_offset
                loop_region = self._clock.loop_region
                if wrap_offset >= 0 and loop_region is not None:
                    pre_frames = wrap_offset
                    if pre_frames > 0:
                        self._mixer.read_mix_into(self._pre_scratch, position, pre_frames)
                    remaining = frames - pre_frames
                    loop_length = loop_region.end - loop_region.start
                    post_filled = 0
                    read_position = loop_region.start
                    while post_filled < remaining:
                        chunk = min(loop_length, remaining - post_filled)
                        self._mixer.read_mix_into(
                            self._post_scratch[post_filled:],
                            read_position,
                            chunk,
                        )
                        read_position = loop_region.start + (
                            (read_position - loop_region.start + chunk) % loop_length
                        )
                        post_filled += chunk
                    if pre_frames > 0:
                        mixed[:pre_frames] = self._pre_scratch[:pre_frames]
                    mixed[pre_frames:frames] = self._post_scratch[:remaining]
                    crossfade_length = min(self._crossfade.length, pre_frames, remaining)
                    if pre_frames > 0 and crossfade_length > 0:
                        tail = self._pre_scratch[pre_frames - crossfade_length : pre_frames]
                        head = self._post_scratch[:crossfade_length]
                        self._crossfade.apply(
                            mixed,
                            tail,
                            head,
                            pre_frames - crossfade_length,
                            crossfade_length,
                        )
                else:
                    self._mixer.read_mix_into(self._output_scratch, position, frames)
                if duration > 0 and not self._clock.loop_enabled:
                    end_fade_position = int(position)
                    end_fade_duration = int(duration)

        self._mix_transport_release_into(mixed, frames)
        self._mix_overlay_release_into(mixed, frames)
        self._mix_overlay_into(mixed, frames)
        self._apply_pending_pause_envelope(
            mixed,
            frames=frames,
            position=render_position,
        )

        self._sanitize_output_samples(mixed, frames)
        boundary_declick_reason = self._pending_declick_reason
        if transport_state_changed and not self._pending_declick:
            boundary_declick_reason = "transport-state-changed"
            self._last_discontinuity_reason = boundary_declick_reason
        self._apply_boundary_declick(
            mixed,
            frames=frames,
            force=bool(self._pending_declick) or transport_state_changed,
            reason=boundary_declick_reason,
        )
        if end_fade_duration > 0:
            self._apply_end_of_content_fade(
                mixed,
                position=end_fade_position,
                frames=frames,
                duration=end_fade_duration,
            )
        self._pending_declick = False
        self._last_callback_was_playing = bool(self._transport.is_playing)
        self._capture_output_callback_block(mixed, frames=frames)
        if self._channels == 1:
            outdata[:, 0] = mixed if mixed.ndim == 1 else mixed[:, 0]
            return
        if mixed.ndim == 1:
            outdata[:, :] = mixed[:, None]
            return
        outdata[:, :] = mixed[:, : self._channels]

    def _capture_output_callback_block(self, mixed: np.ndarray, *, frames: int) -> None:
        if not self._diagnostics_capture_active or not self._diagnostics_capture_include_audio:
            return
        if frames <= 0 or self._diagnostics_capture_max_blocks <= 0:
            return
        try:
            block = np.array(mixed[:frames], dtype=np.float32, copy=True)
            self._diagnostics_capture_sequence += 1
            self._diagnostics_capture_blocks.append(
                {
                    "seq": int(self._diagnostics_capture_sequence),
                    "kind": "output_callback_mixed",
                    "monotonic_seconds": float(time.monotonic()),
                    "clock_samples": int(self._clock.position),
                    "clock_seconds": float(self._clock.position_seconds),
                    "frames": int(frames),
                    "channels": int(self._channels),
                    "sample_rate": int(self._clock.sample_rate),
                    "is_playing": bool(self._transport.is_playing),
                    "overlay_active": bool(self.overlay_active),
                    "peak_abs": float(np.max(np.abs(block))) if block.size else 0.0,
                    "rms": float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0,
                    "samples": block,
                }
            )
        except Exception:
            return

    def _mix_overlay_into(self, mixed: np.ndarray, frames: int) -> None:
        overlay = self._overlay_playback_buffer
        if overlay is None:
            return
        start = int(self._overlay_read_index)
        if start >= int(overlay.shape[0]):
            self._clear_overlay()
            return
        available = int(overlay.shape[0]) - start
        chunk_frames = min(int(frames), available)
        if chunk_frames <= 0:
            self._clear_overlay()
            return
        overlay_view = overlay[start : start + chunk_frames]
        if start == 0:
            self._last_ramp_reason = "overlay-start"
        if start + chunk_frames >= int(overlay.shape[0]):
            self._last_ramp_reason = "overlay-end"
        if mixed.ndim == 1:
            if overlay_view.ndim == 2:
                overlay_view = overlay_view[:, 0]
            mixed[:chunk_frames] += overlay_view * self._overlay_volume
        else:
            if overlay_view.ndim == 1:
                mixed[:chunk_frames, :] += overlay_view[:, None] * self._overlay_volume
            else:
                mixed[:chunk_frames, : overlay_view.shape[1]] += (
                    overlay_view * self._overlay_volume
                )
        self._overlay_read_index = start + chunk_frames
        if self._overlay_read_index >= int(overlay.shape[0]):
            self._clear_overlay()

    def _mix_overlay_release_into(self, mixed: np.ndarray, frames: int) -> None:
        release = self._overlay_release_buffer
        if release is None:
            return
        start = int(self._overlay_release_read_index)
        if start >= int(release.shape[0]):
            self._clear_overlay_release()
            return
        available = int(release.shape[0]) - start
        chunk_frames = min(int(frames), available)
        if chunk_frames <= 0:
            self._clear_overlay_release()
            return
        release_view = release[start : start + chunk_frames]
        if mixed.ndim == 1:
            if release_view.ndim == 2:
                release_view = release_view[:, 0]
            mixed[:chunk_frames] += release_view * self._overlay_release_volume
        else:
            if release_view.ndim == 1:
                mixed[:chunk_frames, :] += release_view[:, None] * self._overlay_release_volume
            else:
                mixed[:chunk_frames, : release_view.shape[1]] += (
                    release_view * self._overlay_release_volume
                )
        self._overlay_release_read_index = start + chunk_frames
        self._last_ramp_reason = "overlay-release"
        if self._overlay_release_read_index >= int(release.shape[0]):
            self._clear_overlay_release()

    def _mix_transport_release_into(self, mixed: np.ndarray, frames: int) -> None:
        release = self._transport_release_buffer
        active_frames = int(self._transport_release_active_frames)
        if active_frames <= 0:
            return
        start = int(self._transport_release_read_index)
        if start >= active_frames:
            self._clear_transport_release()
            return
        available = active_frames - start
        chunk_frames = min(int(frames), available)
        if chunk_frames <= 0:
            self._clear_transport_release()
            return
        release_view = release[start : start + chunk_frames]
        if mixed.ndim == 1:
            if release_view.ndim == 2:
                release_view = release_view[:, 0]
            mixed[:chunk_frames] += release_view
        else:
            if release_view.ndim == 1:
                mixed[:chunk_frames, :] += release_view[:, None]
            else:
                mixed[:chunk_frames, : release_view.shape[1]] += release_view
        self._transport_release_read_index = start + chunk_frames
        self._last_ramp_reason = "transport-release"
        if self._transport_release_read_index >= active_frames:
            self._clear_transport_release()

    def _apply_end_of_content_fade(
        self,
        mixed: np.ndarray,
        *,
        position: int,
        frames: int,
        duration: int,
    ) -> None:
        end_offset = int(duration) - int(position)
        if end_offset <= 0 or end_offset > int(frames):
            return
        fade_frames = min(end_offset, int(self._declick_ramp_samples))
        fade_start = max(0, end_offset - fade_frames)
        _apply_declick_fade_out_slice(mixed, fade_start, end_offset, self._declick_fade_out)
        if end_offset < int(frames):
            mixed[end_offset:frames] = 0.0
        if mixed.ndim == 1:
            self._last_output_tail[0] = np.float32(mixed[frames - 1])
        else:
            self._last_output_tail[:] = np.asarray(mixed[frames - 1], dtype=np.float32)
        self._transport.pause()
        self._end_of_content = True
        self._last_ramp_reason = "end-of-content"

    def _clear_overlay_release(self) -> None:
        self._overlay_release_buffer = None
        self._overlay_release_read_index = 0
        self._overlay_release_volume = np.float32(1.0)

    def add_clock_subscriber(self, sub: ClockSubscriber) -> None:
        self._clock.add_subscriber(sub)

    def remove_clock_subscriber(self, sub: ClockSubscriber) -> None:
        self._clock.remove_subscriber(sub)

    @staticmethod
    def _coerce_output_latency_seconds(latency: Any) -> float:
        if isinstance(latency, (tuple, list)):
            latency = latency[-1] if latency else 0.0
        try:
            return max(0.0, float(latency))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _sanitize_output_samples(buffer: np.ndarray, frames: int) -> None:
        out = buffer[:frames]
        np.nan_to_num(out, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(out, -1.0, 1.0, out=out)

    def _apply_boundary_declick(
        self,
        buffer: np.ndarray,
        *,
        frames: int,
        force: bool,
        reason: str | None = None,
    ) -> None:
        if frames <= 0:
            return

        if force:
            if buffer.ndim == 1:
                start_sample = np.array([float(buffer[0])], dtype=np.float32)
            else:
                start_sample = np.asarray(buffer[0], dtype=np.float32)
            prior_tail = np.asarray(self._last_output_tail, dtype=np.float32)
            delta = start_sample - prior_tail
            if bool(np.any(np.abs(delta) >= _DECLICK_DELTA_THRESHOLD)):
                self._declick_correction_delta = np.array(delta, dtype=np.float32, copy=True)
                self._declick_correction_total = int(self._declick_ramp_samples)
                self._declick_correction_remaining = int(self._declick_ramp_samples)
                discontinuity_reason = str(reason or self._pending_declick_reason)
                self._last_ramp_reason = discontinuity_reason
                self._record_runtime_event(
                    "callback-discontinuity",
                    reason=discontinuity_reason,
                    frames=int(frames),
                    peak_delta=float(np.max(np.abs(delta))),
                )

        self._apply_declick_correction(buffer, frames=frames)
        if buffer.ndim == 1:
            self._last_output_tail[0] = np.float32(buffer[frames - 1])
            return
        self._last_output_tail[:] = np.asarray(buffer[frames - 1], dtype=np.float32)

    def _apply_declick_correction(self, buffer: np.ndarray, *, frames: int) -> None:
        remaining = int(self._declick_correction_remaining)
        total = int(self._declick_correction_total)
        if frames <= 0 or remaining <= 0 or total <= 1:
            return
        declick_frames = min(int(frames), remaining)
        if declick_frames <= 0:
            return
        start = total - remaining
        stop = start + declick_frames
        ramp = self._declick_fade_out[start:stop]
        if ramp.shape[0] != declick_frames:
            return
        delta = np.asarray(self._declick_correction_delta, dtype=np.float32)
        if buffer.ndim == 1:
            buffer[:declick_frames] -= delta[0] * ramp
        else:
            buffer[:declick_frames] -= delta[None, :] * ramp[:, None]
        remaining -= declick_frames
        self._declick_correction_remaining = max(0, remaining)
        if self._declick_correction_remaining == 0:
            self._declick_correction_total = 0
            self._declick_correction_delta = np.zeros_like(self._last_output_tail)

    def _update_callback_timing_snapshot(self, time_info: Any) -> None:
        output_latency_seconds = self._reported_output_latency_seconds
        callback_now = time.monotonic()
        measured_latency = self._extract_output_latency_seconds(time_info)
        if measured_latency is not None:
            output_latency_seconds = measured_latency
            self._reported_output_latency_seconds = measured_latency
        self._last_audible_time_seconds = max(
            0.0,
            float(self._clock.position_seconds) - output_latency_seconds,
        )
        self._last_audible_monotonic_seconds = callback_now

    @staticmethod
    def _extract_output_latency_seconds(time_info: Any) -> float | None:
        current_time = AudioEngine._coerce_callback_time_value(time_info, "currentTime")
        output_dac_time = AudioEngine._coerce_callback_time_value(
            time_info,
            "outputBufferDacTime",
        )
        if current_time is None or output_dac_time is None:
            return None
        return max(0.0, output_dac_time - current_time)

    @staticmethod
    def _coerce_callback_time_value(time_info: Any, field: str) -> float | None:
        if time_info is None:
            return None
        if isinstance(time_info, dict):
            value = time_info.get(field)
        else:
            value = getattr(time_info, field, None)
            if value is None:
                try:
                    value = time_info[field]
                except Exception:
                    value = None
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


__all__ = [
    "AudioEngine",
    "_resolve_output_defaults",
    "_resolve_stream_defaults",
]

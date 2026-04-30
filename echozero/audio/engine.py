"""
AudioEngine: Transport, mixer, and output stream for EZ playback.
Exists because EZ needs one simple DAW-style engine surface independent of any device library.
Connects application playback control to one output backend through a narrow callback-driven contract.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from echozero.audio.clock import Clock, ClockSubscriber
from echozero.audio.crossfade import CrossfadeBuffer
from echozero.audio.layer import AudioLayer, AudioTrack
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
    def stream_latency(self) -> str | float | None:
        return self._stream_latency

    @property
    def stream_blocksize(self) -> int:
        return self._stream_blocksize

    @property
    def prime_output_buffers_using_stream_callback(self) -> bool:
        return self._prime_output_buffers_using_stream_callback

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

    def clear_tracks(self) -> None:
        """Remove every playback track from the engine mixer."""

        self._mixer.clear_tracks()

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
        if not self._active:
            self._open_stream()
        self._transport.play()

    def pause(self) -> None:
        self._transport.pause()
        self._last_audible_monotonic_seconds = None

    def stop(self) -> None:
        self._end_of_content = False
        self._transport.stop()
        self._last_audible_time_seconds = 0.0
        self._last_audible_monotonic_seconds = None

    def seek(self, position_samples: int) -> None:
        self._end_of_content = False
        self._transport.seek(position_samples)
        self._last_audible_time_seconds = self._clock.position_seconds
        self._last_audible_monotonic_seconds = None

    def seek_seconds(self, seconds: float) -> None:
        self.seek(int(seconds * self._clock.sample_rate))

    def toggle_play_pause(self) -> None:
        self._end_of_content = False
        if not self._active:
            self._open_stream()
        self._transport.toggle_play_pause()

    def shutdown(self) -> None:
        self._transport.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False
        self._reported_output_latency_seconds = 0.0
        self._last_audible_time_seconds = None
        self._last_audible_monotonic_seconds = None

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

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Render one output buffer on the real-time thread."""

        if status:
            self._glitch_count += 1
            self._last_status = status
        if frames > len(self._output_scratch):
            outdata[:] = 0
            self._glitch_count += 1
            self._last_status = f"callback_frames_exceeded_scratch:{frames}>{len(self._output_scratch)}"
            return
        if not self._transport.is_playing:
            outdata[:] = 0
            return

        position = self._clock.advance(frames)
        self._update_callback_timing_snapshot(time_info)
        duration = self._mixer.duration_samples
        if duration > 0 and not self._clock.loop_enabled and position >= duration:
            outdata[:] = 0
            self._transport.pause()
            self._end_of_content = True
            return

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
            mixed = self._output_scratch[:frames]
            if pre_frames > 0:
                mixed[:pre_frames] = self._pre_scratch[:pre_frames]
            mixed[pre_frames:frames] = self._post_scratch[:remaining]
            crossfade_length = min(self._crossfade.length, pre_frames, remaining)
            if pre_frames > 0 and crossfade_length > 0:
                tail = self._pre_scratch[pre_frames - crossfade_length:pre_frames]
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
            mixed = self._output_scratch[:frames]

        self._sanitize_output_samples(mixed, frames)
        if self._channels == 1:
            outdata[:, 0] = mixed if mixed.ndim == 1 else mixed[:, 0]
            return
        if mixed.ndim == 1:
            outdata[:, :] = mixed[:, None]
            return
        outdata[:, :] = mixed[:, :self._channels]

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

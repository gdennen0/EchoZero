"""Core audio playback engine for EchoZero runtime.
Exists to own the sounddevice stream, transport clock, and layer mixer for playback.
Connects application playback control to process-local audio I/O without UI semantics.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from echozero.audio.clock import Clock, ClockSubscriber
from echozero.audio.crossfade import CrossfadeBuffer
from echozero.audio.layer import AudioLayer
from echozero.audio.mixer import Mixer
from echozero.audio.transport import Transport


# Default audio settings
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BUFFER_SIZE = 256  # ~5.8ms at 44100 — low latency, safe for modern hardware
DEFAULT_CHANNELS = 1       # injected/test streams keep legacy mono defaults
DEFAULT_SCRATCH_FRAMES = 32768
_AUTO_SAMPLE_RATE_PREFERENCE = (48000, 44100, 96000, 88200, 32000)


def _coerce_sample_rate(value: Any, *, default: int = DEFAULT_SAMPLE_RATE) -> int:
    """Convert a backend-provided sample-rate value to a positive integer."""
    try:
        resolved = int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)
    return max(1, resolved)


def _is_output_sample_rate_supported(
    sounddevice_module: Any,
    *,
    output_device: int | str | None,
    channels: int,
    sample_rate: int,
) -> bool:
    """Return True when the device accepts this output format."""
    check_output_settings = getattr(sounddevice_module, "check_output_settings", None)
    if not callable(check_output_settings):
        return False
    try:
        check_output_settings(
            device=output_device,
            channels=max(1, int(channels)),
            dtype="float32",
            samplerate=int(sample_rate),
        )
    except Exception:
        return False
    return True


def _select_auto_output_sample_rate(
    sounddevice_module: Any,
    *,
    output_device: int | str | None,
    channels: int,
    default_sample_rate: int,
) -> int:
    """Choose a stable auto sample rate for the current output device.

    Prefers the device-reported default when explicitly supported, then falls
    back to a bounded list of common rates.
    """
    resolved_default = _coerce_sample_rate(default_sample_rate)
    candidates: list[int] = []
    for candidate in (resolved_default, *_AUTO_SAMPLE_RATE_PREFERENCE):
        sample_rate = _coerce_sample_rate(candidate)
        if sample_rate not in candidates:
            candidates.append(sample_rate)

    for sample_rate in candidates:
        if _is_output_sample_rate_supported(
            sounddevice_module,
            output_device=output_device,
            channels=channels,
            sample_rate=sample_rate,
        ):
            return sample_rate

    return resolved_default


def _resolve_output_defaults(
    stream_factory: Callable[..., Any] | None,
    *,
    output_device: int | str | None = None,
) -> tuple[int, int]:
    """Prefer the default real output device format when using sounddevice directly.

    Deterministic test paths that inject a stream_factory keep the legacy defaults.
    """
    if stream_factory is not None:
        return DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS

    try:
        import sounddevice as sd

        resolved_output_device = sd.default.device[1] if output_device is None else output_device
        device_info = sd.query_devices(resolved_output_device)
        max_output_channels = int(device_info.get("max_output_channels", DEFAULT_CHANNELS))
        channels = 2 if max_output_channels >= 2 else max(1, max_output_channels)
        sample_rate = _select_auto_output_sample_rate(
            sd,
            output_device=resolved_output_device,
            channels=channels,
            default_sample_rate=device_info.get("default_samplerate", DEFAULT_SAMPLE_RATE),
        )
        return sample_rate, max(1, channels)
    except Exception:
        return DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS


def _resolve_stream_defaults(
    stream_factory: Callable[..., Any] | None,
    *,
    buffer_size: int,
    blocksize: int | None,
    latency: str | float | None,
    prime_output_buffers_using_stream_callback: bool,
) -> tuple[int, str | float, bool]:
    """Choose safer real-device stream settings without perturbing injected test streams.

    The app path should prefer stable playback over minimum latency. Injected
    stream factories keep the historical aggressive defaults so deterministic
    tests do not change behavior.
    """
    if stream_factory is not None:
        resolved_blocksize = 0 if blocksize is None else int(blocksize)
        resolved_latency: str | float = "low" if latency is None else latency
        return (
            resolved_blocksize,
            resolved_latency,
            bool(prime_output_buffers_using_stream_callback),
        )

    # PortAudio/sounddevice recommends blocksize=0 unless the callback truly
    # requires a fixed frame count. Our callback already handles variable-size
    # buffers, and host-chosen block sizes are typically more robust on real
    # devices than forcing a nominal buffer size such as 256 frames.
    resolved_blocksize = 0 if blocksize is None else int(blocksize)
    resolved_latency = "high" if latency is None else latency
    return (
        max(0, resolved_blocksize),
        resolved_latency,
        bool(prime_output_buffers_using_stream_callback),
    )


def _create_audio_buffer(frames: int, channels: int) -> np.ndarray:
    """Allocate a scratch buffer matching the engine output channel layout."""
    if channels <= 1:
        return np.zeros(frames, dtype=np.float32)
    return np.zeros((frames, channels), dtype=np.float32)


class AudioEngine:
    """DAW-grade audio playback engine.

    Owns: Clock, Transport, Mixer, sounddevice stream.

    Usage:
        engine = AudioEngine()
        engine.add_layer("drums", drums_buffer, 44100)
        engine.play()
        # ... later ...
        engine.stop()
        engine.shutdown()

    The audio callback runs on a real-time thread. It:
    1. Checks transport state (playing?)
    2. Reads the clock position via lock-free advance
    3. Asks the mixer for mixed audio (pre-allocated, clipped)
    4. Checks for end-of-content → auto-stop
    5. Writes to the output buffer
    """

    __slots__ = (
        "_clock", "_transport", "_mixer", "_crossfade",
        "_stream", "_buffer_size", "_channels",
        "_stream_factory", "_stream_blocksize",
        "_stream_latency", "_prime_output_buffers_using_stream_callback",
        "_output_device",
        "_active", "_end_of_content", "_reported_output_latency_seconds",
        "_last_audible_time_seconds",
        "_last_audible_monotonic_seconds",
        "_output_scratch",       # A2: pre-allocated output buffer for loop-wrap path
        "_pre_scratch",          # pre-allocated buffer for pre-wrap audio
        "_post_scratch",         # pre-allocated buffer for post-wrap audio
        "_glitch_count",         # A10: sounddevice underrun/overrun counter
        "_last_status",          # A10: last non-None sounddevice status object
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
    ) -> None:
        resolved_sample_rate, resolved_channels = _resolve_output_defaults(
            stream_factory,
            output_device=output_device,
        )
        sample_rate = int(sample_rate or resolved_sample_rate)
        channels = int(channels or resolved_channels)
        self._clock = Clock(sample_rate=sample_rate)
        self._transport = Transport(self._clock)
        self._mixer = Mixer()
        self._crossfade = CrossfadeBuffer(
            crossfade_samples=int(sample_rate * 0.004)  # 4ms
        )
        self._buffer_size = buffer_size
        self._channels = channels
        self._stream: Any = None
        self._stream_factory = stream_factory
        self._stream_blocksize = stream_blocksize
        self._stream_latency = stream_latency
        self._prime_output_buffers_using_stream_callback = (
            prime_output_buffers_using_stream_callback
        )
        self._output_device = output_device
        self._active = False
        self._end_of_content = False  # set by callback, read by main thread
        self._reported_output_latency_seconds = 0.0
        self._last_audible_time_seconds: float | None = None
        self._last_audible_monotonic_seconds: float | None = None

        # A2: pre-allocated output scratch buffers. Sized to 2× buffer_size so that
        # split-read pre/post segments can each be a full buffer's worth.
        scratch_size = max(buffer_size * 2, DEFAULT_SCRATCH_FRAMES)
        self._output_scratch: np.ndarray = _create_audio_buffer(scratch_size, channels)
        self._pre_scratch: np.ndarray = _create_audio_buffer(scratch_size, channels)
        self._post_scratch: np.ndarray = _create_audio_buffer(scratch_size, channels)

        # A10: glitch tracking
        self._glitch_count: int = 0
        self._last_status: Any = None

    # -- Public properties --------------------------------------------------

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
    def sample_rate(self) -> int:
        return self._clock.sample_rate

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def reported_output_latency_seconds(self) -> float:
        """Best-effort output latency reported by the backend stream."""
        return self._reported_output_latency_seconds

    @property
    def audible_time_seconds(self) -> float:
        """Best-effort output-aligned transport time.

        Uses the latest callback-aligned audible snapshot plus monotonic
        extrapolation between callbacks. This keeps the UI smoother without
        making the UI timer the source of truth.
        """
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
        """True if playback stopped because content ended. Reset on play/seek."""
        return self._end_of_content

    @property
    def glitch_count(self) -> int:
        """A10: Number of audio glitches (underruns/overruns) reported by sounddevice."""
        return self._glitch_count

    @property
    def last_audio_status(self) -> Any:
        """A10: Last non-None sounddevice status object. None if no glitches yet."""
        return self._last_status

    # -- Layer management ---------------------------------------------------

    def add_layer(
        self,
        layer_id: str,
        buffer: np.ndarray,
        sample_rate: int,
        name: str | None = None,
        offset: int = 0,
        volume: float = 1.0,
        output_bus: str | None = None,
    ) -> AudioLayer:
        """Create and add a layer to the mixer.

        If the buffer's sample rate differs from the engine's, it is resampled
        automatically. The original sample rate is preserved on the layer for reference.
        """
        layer = AudioLayer(
            layer_id=layer_id,
            name=name or layer_id,
            buffer=buffer,
            sample_rate=sample_rate,
            offset=offset,
            volume=volume,
            engine_sample_rate=self._clock.sample_rate,
            output_bus=output_bus,
        )
        self._mixer.add_layer(layer)
        return layer

    def remove_layer(self, layer_id: str) -> AudioLayer | None:
        return self._mixer.remove_layer(layer_id)

    # -- Transport controls -------------------------------------------------

    def play(self) -> None:
        """Start or resume playback. Opens audio stream if needed."""
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
        self._end_of_content = False
        self._transport.seek_seconds(seconds)
        self._last_audible_time_seconds = self._clock.position_seconds
        self._last_audible_monotonic_seconds = None

    def toggle_play_pause(self) -> None:
        self._end_of_content = False
        if not self._active:
            self._open_stream()
        self._transport.toggle_play_pause()

    # -- Stream management --------------------------------------------------

    def _open_stream(self) -> None:
        if self._active:
            return

        blocksize, latency, prime_output = _resolve_stream_defaults(
            self._stream_factory,
            buffer_size=self._buffer_size,
            blocksize=self._stream_blocksize,
            latency=self._stream_latency,
            prime_output_buffers_using_stream_callback=(
                self._prime_output_buffers_using_stream_callback
            ),
        )

        stream_kwargs = {
            "samplerate": self._clock.sample_rate,
            "blocksize": blocksize,
            "channels": self._channels,
            "dtype": "float32",
            "latency": latency,
            "prime_output_buffers_using_stream_callback": prime_output,
            "callback": self._audio_callback,
        }
        if self._output_device is not None:
            stream_kwargs["device"] = self._output_device

        if self._stream_factory is not None:
            self._stream = self._stream_factory(**stream_kwargs)
        else:
            import sounddevice as sd
            self._stream = sd.OutputStream(**stream_kwargs)

        self._stream.start()
        self._reported_output_latency_seconds = self._coerce_output_latency_seconds(
            getattr(self._stream, "latency", 0.0)
        )
        self._active = True

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

    # -- The callback (HOT PATH) -------------------------------------------

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Called by sounddevice on the real-time audio thread.

        Lock-free. No allocations. No I/O. No exceptions.

        Loop crossfade: when the clock wraps at a loop boundary, we do a split
        read — tail audio from before the wrap, head audio from after — and
        blend them with an equal-power crossfade to eliminate the click.

        A10: if status is truthy, a glitch (underrun/overrun) occurred.
        """
        # A10: track glitches without raising
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

        if not self._transport.is_playing:
            outdata[:] = 0
            return

        # Advance clock (lock-free)
        position = self._clock.advance(frames)
        self._update_callback_timing_snapshot(time_info)

        # End-of-content check (auto-pause, skip if looping)
        duration = self._mixer.duration_samples
        if duration > 0 and not self._clock.loop_enabled:
            if position >= duration:
                outdata[:] = 0
                self._transport.pause()
                self._end_of_content = True
                return

        # Check if a loop wrap happened within this buffer
        wrap_offset = self._clock.last_wrap_offset
        loop_region = self._clock.loop_region

        # A11: wrap_offset == 0 is valid — wrap at the very first sample of this buffer.
        # Previously `wrap_offset > 0` missed this case, leaving it as a hard cut.
        # When wrap_offset == 0 there is no pre-wrap audio (pre segment is empty),
        # so we skip straight to reading from loop_start for all `frames` samples.
        if wrap_offset >= 0 and loop_region is not None:
            # SPLIT READ: the buffer spans (or starts at) a loop boundary.
            loop_len = loop_region.end - loop_region.start

            # Part 1: pre-wrap audio (position → loop_region.end)
            # wrap_offset == 0 means no pre-wrap samples — skip this segment.
            pre_frames = wrap_offset  # 0 when wrap_offset == 0
            if pre_frames > 0:
                # A6: use read_mix_into so the result lands in _pre_scratch directly
                self._mixer.read_mix_into(self._pre_scratch, position, pre_frames)

            # Part 2: post-wrap audio (loop_start → loop_start + remaining).
            # A4: if remaining > loop_len, tile by reading in loop_len chunks.
            remaining = frames - pre_frames
            post_filled = 0
            read_pos = loop_region.start
            while post_filled < remaining:
                chunk = min(loop_len, remaining - post_filled)
                self._mixer.read_mix_into(
                    self._post_scratch[post_filled:], read_pos, chunk
                )
                # Advance within the loop, wrapping if necessary
                read_pos = loop_region.start + ((read_pos - loop_region.start + chunk) % loop_len)
                post_filled += chunk

            # Assemble into _output_scratch (A2: no third read_mix call)
            out = self._output_scratch[:frames]
            if pre_frames > 0:
                out[:pre_frames] = self._pre_scratch[:pre_frames]
            out[pre_frames:frames] = self._post_scratch[:remaining]

            # CROSSFADE at the splice point to eliminate click.
            # Only apply if there IS a pre-wrap segment to blend from.
            if pre_frames > 0:
                xfade = self._crossfade
                xfade_len = min(xfade.length, pre_frames, remaining)
                if xfade_len > 0:
                    # tail = last xfade_len samples of pre-wrap region
                    tail = self._pre_scratch[pre_frames - xfade_len:pre_frames]
                    # head = first xfade_len samples of post-wrap region
                    head = self._post_scratch[:xfade_len]
                    xfade.apply(out, tail, head, pre_frames - xfade_len, xfade_len)

            # A3: clip crossfade output — equal-power peaks at √2 ≈ 1.414
            mixed = out

        else:
            # Normal path: no loop wrap, straight read into _output_scratch
            # A6: use read_mix_into to avoid a copy
            self._mixer.read_mix_into(self._output_scratch, position, frames)
            mixed = self._output_scratch[:frames]

        self._sanitize_output_samples(mixed, frames)

        # Write to output
        if self._channels == 1:
            if mixed.ndim == 1:
                outdata[:, 0] = mixed
            else:
                outdata[:, 0] = mixed[:, 0]
        else:
            if mixed.ndim == 1:
                outdata[:, :] = mixed[:, None]
            else:
                outdata[:, :] = mixed[:, :self._channels]

    # -- Clock subscriber management ----------------------------------------

    def add_clock_subscriber(self, sub: ClockSubscriber) -> None:
        self._clock.add_subscriber(sub)

    def remove_clock_subscriber(self, sub: ClockSubscriber) -> None:
        self._clock.remove_subscriber(sub)

    @staticmethod
    def _coerce_output_latency_seconds(latency: Any) -> float:
        """Normalize backend latency reporting to output-latency seconds."""
        if isinstance(latency, (tuple, list)):
            if not latency:
                return 0.0
            latency = latency[-1]
        try:
            return max(0.0, float(latency))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _sanitize_output_samples(buffer: np.ndarray, frames: int) -> None:
        """Clamp RT output to a finite, device-safe float32 range."""
        out = buffer[:frames]
        np.nan_to_num(out, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(out, -1.0, 1.0, out=out)

    def _update_callback_timing_snapshot(self, time_info: Any) -> None:
        output_latency_seconds = self._reported_output_latency_seconds
        callback_now = time.monotonic()
        if time_info is not None:
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
        output_dac_time = AudioEngine._coerce_callback_time_value(time_info, "outputBufferDacTime")
        if current_time is None or output_dac_time is None:
            return None
        return max(0.0, output_dac_time - current_time)

    @staticmethod
    def _coerce_callback_time_value(time_info: Any, field: str) -> float | None:
        if time_info is None:
            return None
        value = None
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

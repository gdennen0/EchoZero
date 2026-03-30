"""
AudioEngine: Top-level audio playback engine.

Exists because someone needs to own the sounddevice stream, wire the clock to
the mixer, and present a clean API to the rest of the application.

This is the single entry point for audio playback. The application creates one
AudioEngine, adds layers, and calls play/pause/stop. Everything else is internal.

Inspired by: Reaper's audio system, JUCE AudioDeviceManager, Ableton's audio engine.

Process-agnostic: no Qt, no pipeline engine, no persistence. Just numpy + sounddevice.

Ship-ready guarantees:
- Lock-free audio callback (no mutex, no GIL contention beyond atomic reads)
- Zero per-callback allocations (pre-allocated scratch buffers)
- Hard clipping on output (prevents DAC distortion)
- Auto-stop at end of content (unless looping)
- Sample rate conversion on layer add (mismatched rates handled)
- Thread-safe subscriber add/remove while playing
- Crossfade output clipped to [-1, 1] (equal-power peaks at √2 ≈ 1.414 otherwise)
- Glitch counter tracks sounddevice underrun/overrun events
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from echozero.audio.clock import Clock, ClockSubscriber
from echozero.audio.crossfade import CrossfadeBuffer, DEFAULT_CROSSFADE_SAMPLES
from echozero.audio.layer import AudioLayer
from echozero.audio.mixer import Mixer
from echozero.audio.transport import Transport, TransportState


# Default audio settings
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BUFFER_SIZE = 256  # ~5.8ms at 44100 — low latency, safe for modern hardware
DEFAULT_CHANNELS = 1       # mono output for v1


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
        "_stream_factory", "_active", "_end_of_content",
        "_output_scratch",       # A2: pre-allocated output buffer for loop-wrap path
        "_pre_scratch",          # pre-allocated buffer for pre-wrap audio
        "_post_scratch",         # pre-allocated buffer for post-wrap audio
        "_glitch_count",         # A10: sounddevice underrun/overrun counter
        "_last_status",          # A10: last non-None sounddevice status object
    )

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        channels: int = DEFAULT_CHANNELS,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
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
        self._active = False
        self._end_of_content = False  # set by callback, read by main thread

        # A2: pre-allocated output scratch buffers. Sized to 2× buffer_size so that
        # split-read pre/post segments can each be a full buffer's worth.
        scratch_size = max(buffer_size * 2, 8192)
        self._output_scratch: np.ndarray = np.zeros(scratch_size, dtype=np.float32)
        self._pre_scratch: np.ndarray = np.zeros(scratch_size, dtype=np.float32)
        self._post_scratch: np.ndarray = np.zeros(scratch_size, dtype=np.float32)

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

    def stop(self) -> None:
        self._end_of_content = False
        self._transport.stop()

    def seek(self, position_samples: int) -> None:
        self._end_of_content = False
        self._transport.seek(position_samples)

    def seek_seconds(self, seconds: float) -> None:
        self._end_of_content = False
        self._transport.seek_seconds(seconds)

    def toggle_play_pause(self) -> None:
        self._end_of_content = False
        if not self._active:
            self._open_stream()
        self._transport.toggle_play_pause()

    # -- Stream management --------------------------------------------------

    def _open_stream(self) -> None:
        if self._active:
            return

        if self._stream_factory is not None:
            self._stream = self._stream_factory(
                samplerate=self._clock.sample_rate,
                blocksize=self._buffer_size,
                channels=self._channels,
                dtype="float32",
                callback=self._audio_callback,
            )
        else:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=self._clock.sample_rate,
                blocksize=self._buffer_size,
                channels=self._channels,
                dtype="float32",
                callback=self._audio_callback,
            )

        self._stream.start()
        self._active = True

    def shutdown(self) -> None:
        self._transport.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False

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

        if not self._transport.is_playing:
            outdata[:] = 0
            return

        # Advance clock (lock-free)
        position = self._clock.advance(frames)

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
            np.clip(out[:frames], -1.0, 1.0, out=out[:frames])

            mixed = out

        else:
            # Normal path: no loop wrap, straight read into _output_scratch
            # A6: use read_mix_into to avoid a copy
            self._mixer.read_mix_into(self._output_scratch, position, frames)
            mixed = self._output_scratch[:frames]

        # Write to output
        if self._channels == 1:
            outdata[:, 0] = mixed
        else:
            for ch in range(self._channels):
                outdata[:, ch] = mixed

    # -- Clock subscriber management ----------------------------------------

    def add_clock_subscriber(self, sub: ClockSubscriber) -> None:
        self._clock.add_subscriber(sub)

    def remove_clock_subscriber(self, sub: ClockSubscriber) -> None:
        self._clock.remove_subscriber(sub)

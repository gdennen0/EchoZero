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
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from echozero.audio.clock import Clock, ClockSubscriber
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
        "_clock", "_transport", "_mixer",
        "_stream", "_buffer_size", "_channels",
        "_stream_factory", "_active", "_end_of_content",
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
        self._buffer_size = buffer_size
        self._channels = channels
        self._stream: Any = None
        self._stream_factory = stream_factory
        self._active = False
        self._end_of_content = False  # set by callback, read by main thread

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
        """
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

        # Mix all active layers (pre-allocated, clipped)
        mixed = self._mixer.read_mix(position, frames)

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

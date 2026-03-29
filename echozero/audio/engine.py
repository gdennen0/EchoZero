"""
AudioEngine: Top-level audio playback engine.

Exists because someone needs to own the sounddevice stream, wire the clock to
the mixer, and present a clean API to the rest of the application.

This is the single entry point for audio playback. The application creates one
AudioEngine, adds layers, and calls play/pause/stop. Everything else is internal.

Inspired by: Reaper's audio system, JUCE AudioDeviceManager, Ableton's audio engine.

Process-agnostic: no Qt, no pipeline engine, no persistence. Just numpy + sounddevice.
"""

from __future__ import annotations

from pathlib import Path
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
    2. Reads the clock position
    3. Asks the mixer for mixed audio at that position
    4. Advances the clock
    5. Writes to the output buffer

    That's it. Everything else is state management.
    """

    __slots__ = (
        "_clock", "_transport", "_mixer",
        "_stream", "_buffer_size", "_channels",
        "_stream_factory", "_active",
    )

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        channels: int = DEFAULT_CHANNELS,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize the audio engine.

        Args:
            sample_rate: Output sample rate.
            buffer_size: Frames per audio callback.
            channels: Output channels (1=mono, 2=stereo).
            stream_factory: Injectable stream constructor for testing.
                            Defaults to sounddevice.OutputStream.
        """
        self._clock = Clock(sample_rate=sample_rate)
        self._transport = Transport(self._clock)
        self._mixer = Mixer()
        self._buffer_size = buffer_size
        self._channels = channels
        self._stream: Any = None
        self._stream_factory = stream_factory
        self._active = False

    # -- Public properties --------------------------------------------------

    @property
    def clock(self) -> Clock:
        """The master clock. Subscribe to it for position updates."""
        return self._clock

    @property
    def transport(self) -> Transport:
        """Transport controls (play/pause/stop/seek)."""
        return self._transport

    @property
    def mixer(self) -> Mixer:
        """The mixer. Add/remove layers, adjust volume/mute/solo."""
        return self._mixer

    @property
    def sample_rate(self) -> int:
        return self._clock.sample_rate

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def is_active(self) -> bool:
        """Whether the audio stream is open and running."""
        return self._active

    # -- Layer management (convenience wrappers) ----------------------------

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

        Args:
            layer_id: Unique ID for this layer.
            buffer: Audio samples (float32 or will be converted).
            sample_rate: Sample rate of the buffer.
            name: Display name (defaults to layer_id).
            offset: Start position in timeline samples.
            volume: Initial volume [0.0, 1.0].

        Returns:
            The created AudioLayer.
        """
        layer = AudioLayer(
            layer_id=layer_id,
            name=name or layer_id,
            buffer=buffer,
            sample_rate=sample_rate,
            offset=offset,
            volume=volume,
        )
        self._mixer.add_layer(layer)
        return layer

    def remove_layer(self, layer_id: str) -> AudioLayer | None:
        """Remove a layer from the mixer."""
        return self._mixer.remove_layer(layer_id)

    # -- Transport controls (convenience wrappers) --------------------------

    def play(self) -> None:
        """Start or resume playback. Opens audio stream if needed."""
        if not self._active:
            self._open_stream()
        self._transport.play()

    def pause(self) -> None:
        """Pause playback."""
        self._transport.pause()

    def stop(self) -> None:
        """Stop playback, return to start."""
        self._transport.stop()

    def seek(self, position_samples: int) -> None:
        """Seek to position in samples."""
        self._transport.seek(position_samples)

    def seek_seconds(self, seconds: float) -> None:
        """Seek to position in seconds."""
        self._transport.seek_seconds(seconds)

    def toggle_play_pause(self) -> None:
        """Toggle between play and pause."""
        if not self._active:
            self._open_stream()
        self._transport.toggle_play_pause()

    # -- Stream management --------------------------------------------------

    def _open_stream(self) -> None:
        """Open the audio output stream."""
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
        """Close the audio stream and release resources."""
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

        This runs every ~5ms. It MUST be fast:
        - No allocations (beyond numpy buffers which are pre-allocated)
        - No locks (mixer uses snapshot pattern)
        - No I/O
        - No exceptions (sounddevice silently drops the callback on error)
        """
        if not self._transport.is_playing:
            outdata[:] = 0
            return

        # Clock position for this buffer
        position = self._clock.advance(frames)

        # Mix all active layers
        mixed = self._mixer.read_mix(position, frames)

        # Write to output buffer
        if self._channels == 1:
            outdata[:, 0] = mixed
        else:
            # Mono mix → duplicate to all channels
            for ch in range(self._channels):
                outdata[:, ch] = mixed

    # -- Clock subscriber management ----------------------------------------

    def add_clock_subscriber(self, sub: ClockSubscriber) -> None:
        """Add a subscriber to the master clock."""
        self._clock.add_subscriber(sub)

    def remove_clock_subscriber(self, sub: ClockSubscriber) -> None:
        """Remove a clock subscriber."""
        self._clock.remove_subscriber(sub)

"""
AudioLayer: A single audio track in the mixer.

Exists because multi-track playback requires individual control over each stem/source.
Each layer holds a loaded audio buffer and provides read access for the mixer's
summing callback.

Inspired by: Reaper track items, Ableton clip slots, Logic regions.
Keeps it simple — a layer is a buffer with volume/mute/solo and a sample offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class AudioLayer:
    """One audio track. Holds samples, provides chunk reads for the mixer.

    The buffer is a 1D float32 numpy array (mono). Stereo support is a future
    extension — EchoZero v1 is mono analysis, and summing mono stems is the
    primary use case.

    Attributes:
        id: Unique layer identifier (matches domain Layer.id).
        name: Human-readable label.
        buffer: Audio samples, float32, normalized [-1, 1].
        sample_rate: Sample rate of the buffer.
        offset: Start position in the timeline (samples). Allows layers to
                start at different times (e.g., stems from different songs in a setlist).
        volume: Linear gain [0.0, 1.0]. Default 1.0.
        muted: Whether this layer is muted.
        solo: Whether this layer is soloed.
    """

    __slots__ = (
        "id", "name", "buffer", "sample_rate", "offset",
        "volume", "muted", "solo",
    )

    def __init__(
        self,
        layer_id: str,
        name: str,
        buffer: np.ndarray,
        sample_rate: int,
        offset: int = 0,
        volume: float = 1.0,
    ) -> None:
        if buffer.dtype != np.float32:
            buffer = buffer.astype(np.float32)
        self.id: str = layer_id
        self.name: str = name
        self.buffer: np.ndarray = buffer
        self.sample_rate: int = sample_rate
        self.offset: int = offset
        self.volume: float = volume
        self.muted: bool = False
        self.solo: bool = False

    @property
    def duration_samples(self) -> int:
        """Total samples in this layer's buffer."""
        return len(self.buffer)

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration_samples / self.sample_rate

    @property
    def end_sample(self) -> int:
        """Last sample position in timeline coordinates."""
        return self.offset + self.duration_samples

    def read_samples(self, position: int, frames: int) -> np.ndarray:
        """Read a chunk of audio from this layer at the given timeline position.

        Returns a float32 array of length `frames`. Samples outside the buffer
        range are zero (silence). This is the hot path — called every audio callback.

        Args:
            position: Timeline position in samples.
            frames: Number of samples to read.

        Returns:
            float32 array of shape (frames,).
        """
        out = np.zeros(frames, dtype=np.float32)

        # Position relative to this layer's buffer
        local_pos = position - self.offset

        # Completely before or after this layer
        if local_pos >= self.duration_samples or local_pos + frames <= 0:
            return out

        # Calculate overlap
        buf_start = max(0, local_pos)
        buf_end = min(self.duration_samples, local_pos + frames)
        out_start = buf_start - local_pos
        out_end = out_start + (buf_end - buf_start)

        out[out_start:out_end] = self.buffer[buf_start:buf_end]
        return out

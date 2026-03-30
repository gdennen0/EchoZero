"""
AudioLayer: A single audio track in the mixer.

Exists because multi-track playback requires individual control over each stem/source.
Each layer holds a loaded audio buffer and provides read access for the mixer's
summing callback.

Inspired by: Reaper track items, Ableton clip slots, Logic regions.
Keeps it simple — a layer is a buffer with volume/mute/solo and a sample offset.

Audio thread contract: read_into() writes into a pre-allocated scratch buffer.
No allocations on the hot path.
"""

from __future__ import annotations

import numpy as np


def resample_buffer(buffer: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio buffer from source_sr to target_sr using linear interpolation.

    For production quality we'd use libsamplerate or scipy.signal.resample_poly,
    but linear interp is allocation-free-friendly and good enough for preview playback.
    High-quality resampling can be swapped in later.

    Args:
        buffer: Source audio, float32.
        source_sr: Source sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled float32 buffer at target_sr.
    """
    if source_sr == target_sr:
        return buffer
    ratio = target_sr / source_sr
    new_len = int(len(buffer) * ratio)
    indices = np.arange(new_len, dtype=np.float64) / ratio
    idx_floor = np.floor(indices).astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, len(buffer) - 1)
    frac = (indices - idx_floor).astype(np.float32)
    return buffer[idx_floor] * (1.0 - frac) + buffer[idx_ceil] * frac


class AudioLayer:
    """One audio track. Holds samples, provides chunk reads for the mixer.

    The buffer is a 1D float32 numpy array (mono), resampled to the engine's
    sample rate on construction. All read operations are in engine sample space.

    Attributes:
        id: Unique layer identifier (matches domain Layer.id).
        name: Human-readable label.
        buffer: Audio samples, float32, normalized [-1, 1], at engine sample rate.
        sample_rate: Engine sample rate (buffer is already resampled to this).
        original_sample_rate: Source file's sample rate before resampling.
        offset: Start position in the timeline (samples). Allows layers to
                start at different times.
        volume: Linear gain [0.0, ...]. Default 1.0. Not clamped — allow boost.
        muted: Whether this layer is muted.
        solo: Whether this layer is soloed.
    """

    __slots__ = (
        "id", "name", "buffer", "sample_rate", "original_sample_rate",
        "offset", "volume", "muted", "solo",
    )

    def __init__(
        self,
        layer_id: str,
        name: str,
        buffer: np.ndarray,
        sample_rate: int,
        offset: int = 0,
        volume: float = 1.0,
        engine_sample_rate: int | None = None,
    ) -> None:
        if buffer.dtype != np.float32:
            buffer = buffer.astype(np.float32)

        self.original_sample_rate: int = sample_rate

        # Resample to engine rate if needed
        target_sr = engine_sample_rate or sample_rate
        if sample_rate != target_sr:
            buffer = resample_buffer(buffer, sample_rate, target_sr)

        self.id: str = layer_id
        self.name: str = name
        self.buffer: np.ndarray = buffer
        self.sample_rate: int = target_sr
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

    def read_into(self, out: np.ndarray, position: int, frames: int) -> None:
        """Read audio into a pre-allocated buffer. Zero-allocation hot path.

        Writes into `out[0:frames]`. Regions outside the layer's range are zeroed.
        This is the primary method called by the mixer on the audio thread.

        Args:
            out: Pre-allocated float32 buffer, at least `frames` long.
            position: Timeline position in samples.
            frames: Number of samples to read.
        """
        out[:frames] = 0.0

        local_pos = position - self.offset

        # Completely outside this layer
        if local_pos >= self.duration_samples or local_pos + frames <= 0:
            return

        # Calculate overlap
        buf_start = max(0, local_pos)
        buf_end = min(self.duration_samples, local_pos + frames)
        out_start = buf_start - local_pos
        out_end = out_start + (buf_end - buf_start)

        out[out_start:out_end] = self.buffer[buf_start:buf_end]

    def read_samples(self, position: int, frames: int) -> np.ndarray:
        """Read a chunk of audio (allocating version, for non-hot-path use).

        Returns a float32 array of length `frames`. For the audio callback,
        prefer read_into() with a pre-allocated scratch buffer.
        """
        out = np.zeros(frames, dtype=np.float32)
        self.read_into(out, position, frames)
        return out

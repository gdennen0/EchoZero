"""
AudioTrack: One DAW-style playback track for the mixer.
Exists because EZ needs one simple, reusable way to play audio from any layer shape.
Connects loaded mono or stereo buffers to the mixer's callback read contract.
"""

from __future__ import annotations

from math import gcd

import numpy as np

try:
    from scipy.signal import resample_poly as _resample_poly
except ImportError:
    _resample_poly = None


def _resample_linear(buffer: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample with linear interpolation as a bounded fallback path."""
    ratio = target_sr / source_sr
    new_len = int(len(buffer) * ratio)
    indices = np.arange(new_len, dtype=np.float64) / ratio
    idx_floor = np.floor(indices).astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, len(buffer) - 1)
    frac = (indices - idx_floor).astype(np.float32)
    if buffer.ndim == 1:
        return buffer[idx_floor] * (1.0 - frac) + buffer[idx_ceil] * frac
    frac_2d = frac[:, None]
    return buffer[idx_floor] * (1.0 - frac_2d) + buffer[idx_ceil] * frac_2d


def resample_buffer(buffer: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio buffer from source_sr to target_sr.

    Uses `scipy.signal.resample_poly` for playback-quality conversion and falls
    back to linear interpolation if scipy is unavailable or parameters are invalid.

    Args:
        buffer: Source audio, float32.
        source_sr: Source sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled float32 buffer at target_sr.
    """
    # A9: guard against empty buffer (avoids arange/indexing errors on zero-length input)
    if len(buffer) == 0:
        return buffer

    if source_sr == target_sr:
        return buffer

    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")

    ratio_gcd = gcd(int(source_sr), int(target_sr))
    up = int(target_sr) // ratio_gcd
    down = int(source_sr) // ratio_gcd
    axis = 0 if buffer.ndim > 1 else -1

    if _resample_poly is None:
        return _resample_linear(buffer, source_sr, target_sr)

    try:
        resampled = _resample_poly(buffer, up, down, axis=axis)
    except ValueError:
        resampled = _resample_linear(buffer, source_sr, target_sr)
    return np.asarray(resampled, dtype=np.float32)


class AudioTrack:
    """One playback track. Holds samples and serves chunk reads for the mixer.

    The buffer may be mono `(frames,)` or multi-channel `(frames, channels)`,
    resampled to the engine's sample rate on construction. All read operations
    are in engine sample space.

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
        "offset", "volume", "muted", "solo", "output_bus",
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
        output_bus: str | None = None,
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
        self.output_bus: str | None = (
            output_bus.strip() if isinstance(output_bus, str) and output_bus.strip() else None
        )

    @property
    def duration_samples(self) -> int:
        """Total samples in this layer's buffer."""
        return len(self.buffer)

    @property
    def channel_count(self) -> int:
        """Number of audio channels carried by this layer."""
        if self.buffer.ndim == 1:
            return 1
        return int(self.buffer.shape[1])

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

        Raises:
            ValueError: If frames > len(out) — caller passed an undersized buffer.
        """
        # A8: explicit bounds check to catch caller bugs early
        if frames > len(out):
            raise ValueError(
                f"frames ({frames}) > buffer length ({len(out)}): "
                "caller must provide a buffer at least `frames` long"
            )
        if out.ndim not in (1, 2):
            raise ValueError(f"Unsupported output rank {out.ndim}; expected 1D or 2D audio buffer")

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

        source = self.buffer[buf_start:buf_end]
        destination = out[out_start:out_end]

        if destination.ndim == 1:
            if source.ndim == 1:
                destination[:] = source
                return

            destination[:] = source[:, 0]
            for channel_index in range(1, source.shape[1]):
                destination[:] += source[:, channel_index]
            destination[:] /= float(source.shape[1])
            return

        if source.ndim == 1:
            destination[:] = source[:, None]
            return

        source_channels = source.shape[1]
        output_channels = destination.shape[1]
        if source_channels == output_channels:
            destination[:] = source
            return
        if source_channels == 1:
            destination[:] = source[:, :1]
            return
        if output_channels == 1:
            destination[:, 0] = source[:, 0]
            for channel_index in range(1, source_channels):
                destination[:, 0] += source[:, channel_index]
            destination[:, 0] /= float(source_channels)
            return

        copied_channels = min(source_channels, output_channels)
        destination[:, :copied_channels] = source[:, :copied_channels]
        if copied_channels < output_channels:
            destination[:, copied_channels:] = source[:, copied_channels - 1:copied_channels]

    def read_samples(self, position: int, frames: int) -> np.ndarray:
        """Read a chunk of audio (allocating version, for non-hot-path use).

        Returns mono audio as `(frames,)` or multi-channel audio as
        `(frames, channels)`. For the audio callback, prefer read_into() with a
        pre-allocated scratch buffer.
        """
        if self.channel_count == 1:
            out = np.zeros(frames, dtype=np.float32)
        else:
            out = np.zeros((frames, self.channel_count), dtype=np.float32)
        self.read_into(out, position, frames)
        return out


AudioLayer = AudioTrack

__all__ = ["AudioTrack", "AudioLayer", "resample_buffer"]

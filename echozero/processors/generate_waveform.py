"""
GenerateWaveformProcessor: Computes multi-LOD peak arrays from audio for UI rendering.
Exists because waveform visualization requires pre-computed min/max peaks at multiple
zoom levels — this block replaces the old ingest waveform step (D276).
Used by ExecutionEngine when running blocks of type 'GenerateWaveform'.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from echozero.domain.types import AudioData, WaveformData, WaveformPeaks
from echozero.errors import ExecutionError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok

# Default LOD window sizes: finest to coarsest.
# LOD 0 = 64 samples/peak (zoomed in, see transients)
# LOD 1 = 256 samples/peak (medium)
# LOD 2 = 1024 samples/peak (coarse)
# LOD 3 = 4096 samples/peak (overview, full song in view)
DEFAULT_WINDOW_SIZES = (64, 256, 1024, 4096)


def compute_peaks(samples: np.ndarray, window_size: int) -> np.ndarray:
    """Compute min/max peak pairs for a given window size.

    Args:
        samples: 1-D numpy array of audio samples (mono).
        window_size: Number of samples per peak window.

    Returns:
        Array of shape (N, 2) where [:,0] is min and [:,1] is max per window.
    """
    n_samples = len(samples)
    if n_samples == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Truncate to even multiple of window_size, handle remainder separately
    n_complete = (n_samples // window_size) * window_size
    peaks_list = []

    if n_complete > 0:
        reshaped = samples[:n_complete].reshape(-1, window_size)
        mins = reshaped.min(axis=1)
        maxs = reshaped.max(axis=1)
        peaks_list.append(np.column_stack((mins, maxs)))

    # Handle remainder samples (partial final window)
    if n_complete < n_samples:
        remainder = samples[n_complete:]
        peaks_list.append(
            np.array([[remainder.min(), remainder.max()]], dtype=np.float32)
        )

    return np.vstack(peaks_list).astype(np.float32)


def _default_load_samples(file_path: str, sample_rate: int) -> np.ndarray:
    """Load audio samples as mono float32 using librosa. Production default."""
    try:
        import librosa
    except ImportError:
        raise NotImplementedError(
            "Default sample loading requires librosa. "
            "Install with: pip install librosa"
        )
    y, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    return y


class GenerateWaveformProcessor:
    """Computes multi-LOD waveform peaks from upstream AudioData.

    Input port: audio_in (AudioData)
    Output port: waveform_out (WaveformData)
    """

    def __init__(
        self,
        load_samples_fn: Callable[[str, int], np.ndarray] | None = None,
        window_sizes: tuple[int, ...] = DEFAULT_WINDOW_SIZES,
    ) -> None:
        self._load_samples_fn = load_samples_fn or _default_load_samples
        self._window_sizes = window_sizes

    def execute(self, block_id: str, context: ExecutionContext) -> Result[WaveformData]:
        """Load audio samples from upstream, compute peaks at each LOD level."""
        # Read audio input
        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None:
            return err(
                ExecutionError(
                    f"Block '{block_id}' has no audio input — "
                    f"connect an audio source to 'audio_in'"
                )
            )

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="generate_waveform",
                percent=0.0,
                message="Loading audio samples",
            )
        )

        # Load raw samples
        try:
            samples = self._load_samples_fn(audio.file_path, audio.sample_rate)
        except Exception as exc:
            return err(
                ExecutionError(
                    f"Failed to load samples from '{audio.file_path}': {exc}"
                )
            )

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="generate_waveform",
                percent=0.2,
                message="Computing waveform peaks",
            )
        )

        # Compute peaks at each LOD level
        lods: list[WaveformPeaks] = []
        n_lods = len(self._window_sizes)
        for i, window_size in enumerate(self._window_sizes):
            peaks = compute_peaks(samples, window_size)
            lods.append(WaveformPeaks(window_size=window_size, peaks=peaks))

            # Progress: 20% for loading + 80% spread across LODs
            frac = 0.2 + 0.8 * (i + 1) / n_lods
            context.progress_bus.publish(
                ProgressReport(
                    block_id=block_id,
                    phase="generate_waveform",
                    percent=frac,
                    message=f"LOD {i} complete (window={window_size})",
                )
            )

        return ok(
            WaveformData(
                lods=tuple(lods),
                sample_rate=audio.sample_rate,
                duration=audio.duration,
                channel_count=audio.channel_count,
            )
        )

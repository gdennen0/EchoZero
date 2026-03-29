"""
EQBandsProcessor: Multi-band parametric equalizer.
Exists because frequency-selective analysis requires precise EQ —
single-band AudioFilter isn't enough for multi-band surgical work.
Used by ExecutionEngine when running blocks of type 'EQBands'.

Each band is defined by freq_low, freq_high, and gain_db.
Bands are applied in series: isolate range with bandpass, scale by (gain - 1),
add back to running signal. Frequencies outside all bands pass through unchanged.

The injectable function pattern allows testing without scipy.
"""

from __future__ import annotations

from typing import Any, Callable

from echozero.domain.types import AudioData
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


# ---------------------------------------------------------------------------
# Band type + defaults
# ---------------------------------------------------------------------------

DEFAULT_BANDS = [
    {"freq_low": 60.0, "freq_high": 250.0, "gain_db": 0.0},
    {"freq_low": 250.0, "freq_high": 2000.0, "gain_db": 0.0},
    {"freq_low": 2000.0, "freq_high": 8000.0, "gain_db": 0.0},
]


def validate_bands(bands: list[dict[str, Any]], sample_rate: int) -> list[str]:
    """Validate band definitions. Returns list of error strings (empty = valid)."""
    errors: list[str] = []
    nyquist = sample_rate / 2.0

    for i, band in enumerate(bands):
        if not isinstance(band, dict):
            errors.append(f"Band {i}: must be a dict")
            continue
        freq_low = band.get("freq_low", 0)
        freq_high = band.get("freq_high", 0)
        gain_db = band.get("gain_db", 0)

        if not isinstance(freq_low, (int, float)) or freq_low < 20:
            errors.append(f"Band {i}: freq_low must be >= 20 Hz, got {freq_low}")
        if not isinstance(freq_high, (int, float)) or freq_high > nyquist:
            errors.append(f"Band {i}: freq_high must be <= {nyquist} Hz (Nyquist), got {freq_high}")
        if isinstance(freq_low, (int, float)) and isinstance(freq_high, (int, float)):
            if freq_low >= freq_high:
                errors.append(f"Band {i}: freq_low ({freq_low}) must be < freq_high ({freq_high})")
        if not isinstance(gain_db, (int, float)) or abs(gain_db) > 48:
            errors.append(f"Band {i}: gain_db must be between -48 and 48, got {gain_db}")

    return errors


# ---------------------------------------------------------------------------
# EQ function signature for DI
# ---------------------------------------------------------------------------

EQBandsFn = Callable[
    [
        str,   # file_path
        int,   # sample_rate
        list,  # bands (list of dicts)
        int,   # filter_order
    ],
    tuple[str, int, float],  # (output_file_path, sample_rate, duration)
]


def _default_eq_bands(
    file_path: str,
    sample_rate: int,
    bands: list[dict[str, Any]],
    filter_order: int,
) -> tuple[str, int, float]:
    """Apply multi-band EQ. Requires scipy + soundfile."""
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import butter, sosfilt
    except ImportError:
        raise ExecutionError(
            "scipy and soundfile are required. Install with: pip install scipy soundfile"
        )

    audio, sr = sf.read(file_path)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    nyquist = sr / 2.0
    result = audio.astype(np.float64).copy()

    for band in bands:
        gain_db = float(band.get("gain_db", 0.0))
        if abs(gain_db) < 0.01:
            continue

        freq_low = max(20.0, float(band["freq_low"]))
        freq_high = min(float(band["freq_high"]), nyquist - 1.0)
        if freq_low >= freq_high:
            continue

        wn_low = max(0.001, freq_low / nyquist)
        wn_high = min(0.999, freq_high / nyquist)
        if wn_low >= wn_high:
            continue

        try:
            sos = butter(filter_order, [wn_low, wn_high], btype="band", output="sos")
        except Exception:
            continue

        linear_gain = 10.0 ** (gain_db / 20.0)
        mix_factor = linear_gain - 1.0

        for ch in range(result.shape[1]):
            bandpassed = sosfilt(sos, result[:, ch])
            result[:, ch] += mix_factor * bandpassed

    # Clamp
    np.clip(result, -1.0, 1.0, out=result)

    # Write output
    import tempfile
    import os

    fd, output_file = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(output_file, result, sr)

    duration = audio.shape[0] / sr
    return output_file, sr, duration


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class EQBandsProcessor:
    """Multi-band parametric EQ processor."""

    def __init__(self, eq_fn: EQBandsFn | None = None) -> None:
        self._eq_fn = eq_fn or _default_eq_bands

    def execute(self, block_id: str, context: ExecutionContext) -> Result[AudioData]:
        """Read upstream audio, apply multi-band EQ, return filtered AudioData."""
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="eq_bands",
                percent=0.0,
                message="Starting multi-band EQ",
            )
        )

        # Read audio input
        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None:
            return err(
                ExecutionError(
                    f"Block '{block_id}' has no audio input — "
                    f"connect an audio source to 'audio_in'"
                )
            )

        # Read settings
        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        bands = settings.get("bands", DEFAULT_BANDS)
        filter_order = settings.get("filter_order", 4)

        if not isinstance(bands, list):
            return err(ValidationError(
                f"Block '{block_id}': 'bands' must be a list of band dicts"
            ))
        if not isinstance(filter_order, int) or filter_order < 1 or filter_order > 8:
            return err(ValidationError(
                f"Block '{block_id}': filter_order must be 1-8, got {filter_order}"
            ))

        # Validate bands
        band_errors = validate_bands(bands, audio.sample_rate)
        if band_errors:
            return err(ValidationError(
                f"Block '{block_id}' band validation errors: {'; '.join(band_errors)}"
            ))

        active_bands = [b for b in bands if abs(float(b.get("gain_db", 0))) >= 0.01]

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="eq_bands",
                percent=0.1,
                message=f"Applying {len(active_bands)} active EQ band(s)",
            )
        )

        # Apply EQ
        try:
            output_file, sr, duration = self._eq_fn(
                file_path=audio.file_path,
                sample_rate=audio.sample_rate,
                bands=bands,
                filter_order=filter_order,
            )
        except (ValidationError, ExecutionError) as exc:
            return err(exc)
        except Exception as exc:
            return err(ExecutionError(
                f"EQ processing failed for block '{block_id}': {exc}"
            ))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="eq_bands",
                percent=1.0,
                message=f"EQ complete — {len(active_bands)} band(s) applied",
            )
        )

        return ok(AudioData(
            sample_rate=sr,
            duration=duration,
            file_path=output_file,
            channel_count=audio.channel_count,
        ))



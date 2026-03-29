"""
AudioFilterProcessor: Parametric EQ and filtering via scipy.signal.
Exists because per-stem filtering is essential for frequency-selective analysis —
you can't accurately detect kick onsets in the bass_out without removing high frequencies.
Used by ExecutionEngine when running blocks of type 'AudioFilter'.
"""

from __future__ import annotations

from typing import Callable

from echozero.domain.types import AudioData
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok

# Supported filter types
VALID_FILTER_TYPES = {
    "lowpass": "Low-pass: removes frequencies above cutoff",
    "highpass": "High-pass: removes frequencies below cutoff",
    "bandpass": "Band-pass: keeps frequencies between low and high cutoff",
    "bandstop": "Band-stop (notch): removes frequencies between low and high cutoff",
    "lowshelf": "Low shelf: boosts/cuts frequencies below cutoff",
    "highshelf": "High shelf: boosts/cuts frequencies above cutoff",
    "peak": "Peak (peaking EQ): boosts/cuts frequencies around center",
}

# Filter function signature for DI
FilterFn = Callable[
    [
        str,  # file_path
        int,  # sample_rate
        str,  # filter_type
        float,  # freq (or center freq for peak)
        float,  # gain_db (only used for shelf/peak)
        float,  # Q (quality factor)
    ],
    tuple[str, int, float],  # (output_file_path, sample_rate, duration)
]


def _default_filter(
    file_path: str,
    sample_rate: int,
    filter_type: str,
    freq: float,
    gain_db: float,
    Q: float,
) -> tuple[str, int, float]:
    """Apply an IIR filter to audio file. Requires scipy + soundfile installed."""
    try:
        import soundfile as sf
        import numpy as np
        from scipy import signal
    except ImportError:
        raise ExecutionError(
            "scipy and soundfile are required. Install with: pip install scipy soundfile"
        )

    # Read audio
    audio, sr = sf.read(file_path)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    # Normalize gain_db to linear scale for shelf/peak filters
    gain_linear = 10.0 ** (gain_db / 20.0)

    # Design the filter based on type
    nyquist = sample_rate / 2.0
    normalized_freq = freq / nyquist

    if filter_type == "lowpass":
        # Simple Butterworth lowpass
        if normalized_freq >= 1.0:
            raise ValidationError(f"Filter frequency {freq} Hz >= Nyquist {nyquist} Hz")
        sos = signal.butter(4, normalized_freq, btype="low", output="sos")

    elif filter_type == "highpass":
        if normalized_freq >= 1.0:
            raise ValidationError(f"Filter frequency {freq} Hz >= Nyquist {nyquist} Hz")
        sos = signal.butter(4, normalized_freq, btype="high", output="sos")

    elif filter_type == "bandpass":
        # Bandpass requires two frequencies; use freq as center and Q to compute bandwidth
        if Q <= 0:
            raise ValidationError(f"Q must be positive for bandpass, got {Q}")
        bandwidth = freq / Q
        low_freq = freq - bandwidth / 2.0
        high_freq = freq + bandwidth / 2.0
        if low_freq <= 0 or high_freq >= nyquist:
            raise ValidationError(
                f"Bandpass frequencies out of range: low={low_freq}, high={high_freq}, nyquist={nyquist}"
            )
        sos = signal.butter(
            4,
            [low_freq / nyquist, high_freq / nyquist],
            btype="band",
            output="sos",
        )

    elif filter_type == "bandstop":
        # Bandstop (notch): similar to bandpass but removes the band
        if Q <= 0:
            raise ValidationError(f"Q must be positive for bandstop, got {Q}")
        bandwidth = freq / Q
        low_freq = freq - bandwidth / 2.0
        high_freq = freq + bandwidth / 2.0
        if low_freq <= 0 or high_freq >= nyquist:
            raise ValidationError(
                f"Bandstop frequencies out of range: low={low_freq}, high={high_freq}, nyquist={nyquist}"
            )
        sos = signal.butter(
            4,
            [low_freq / nyquist, high_freq / nyquist],
            btype="bandstop",
            output="sos",
        )

    elif filter_type == "lowshelf":
        # Low shelf: boost/cut below cutoff
        if normalized_freq >= 1.0:
            raise ValidationError(f"Filter frequency {freq} Hz >= Nyquist {nyquist} Hz")
        # scipy.signal.iirfilter with btype='low' for shelf, but we need to use butter + manual gain
        # Actually, scipy doesn't have direct shelf filters. We'll use a simple approach:
        # design a lowpass and blend with dry signal based on gain
        sos = signal.butter(2, normalized_freq, btype="low", output="sos")

    elif filter_type == "highshelf":
        # High shelf: boost/cut above cutoff
        if normalized_freq >= 1.0:
            raise ValidationError(f"Filter frequency {freq} Hz >= Nyquist {nyquist} Hz")
        sos = signal.butter(2, normalized_freq, btype="high", output="sos")

    elif filter_type == "peak":
        # Peak: boost/cut around center frequency
        # Approximate with bandpass + gain blend
        if Q <= 0:
            raise ValidationError(f"Q must be positive for peak, got {Q}")
        bandwidth = freq / Q
        low_freq = freq - bandwidth / 2.0
        high_freq = freq + bandwidth / 2.0
        if low_freq <= 0 or high_freq >= nyquist:
            raise ValidationError(
                f"Peak frequencies out of range: low={low_freq}, high={high_freq}, nyquist={nyquist}"
            )
        sos = signal.butter(
            2,
            [low_freq / nyquist, high_freq / nyquist],
            btype="band",
            output="sos",
        )

    else:
        raise ValidationError(f"Unknown filter type '{filter_type}'")

    # Apply filter to each channel
    filtered = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        filtered[:, ch] = signal.sosfilt(sos, audio[:, ch])

    # For shelf/peak filters, blend with dry signal based on gain
    if filter_type in ("lowshelf", "highshelf", "peak"):
        if gain_db != 0:
            # Blend: wet_level = gain_linear (for boost) or blend for cut
            if gain_db > 0:
                # Boost: mix dry + wet
                filtered = audio + (filtered - audio) * (gain_linear - 1.0)
            else:
                # Cut: reduce filtered signal
                filtered = audio + (filtered - audio) * (1.0 - 1.0 / gain_linear)

    # Write filtered audio
    import tempfile
    import os
    fd, output_file = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    sf.write(output_file, filtered, sample_rate)

    # Compute duration
    duration = audio.shape[0] / sample_rate

    return output_file, sample_rate, duration


class AudioFilterProcessor:
    """Applies parametric EQ and filtering to audio."""

    def __init__(self, filter_fn: FilterFn | None = None) -> None:
        self._filter_fn = filter_fn or _default_filter

    def execute(self, block_id: str, context: ExecutionContext) -> Result[AudioData]:
        """Read upstream audio, apply filter, return filtered AudioData."""
        # Report start
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="audio_filter",
                percent=0.0,
                message="Starting audio filtering",
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
        filter_type = settings.get("filter_type")
        freq = settings.get("freq")
        gain_db = settings.get("gain_db", 0.0)
        Q = settings.get("Q", 1.0)

        # Validate required settings
        if filter_type is None:
            return err(
                ValidationError(f"Block '{block_id}' is missing required setting 'filter_type'")
            )
        if freq is None:
            return err(
                ValidationError(f"Block '{block_id}' is missing required setting 'freq'")
            )

        if filter_type not in VALID_FILTER_TYPES:
            return err(
                ValidationError(
                    f"Unknown filter_type '{filter_type}'. Valid: {', '.join(VALID_FILTER_TYPES.keys())}"
                )
            )

        if not isinstance(freq, (int, float)) or freq <= 0:
            return err(ValidationError(f"freq must be a positive number, got {freq}"))

        if not isinstance(gain_db, (int, float)):
            return err(ValidationError(f"gain_db must be a number, got {gain_db}"))

        if not isinstance(Q, (int, float)) or Q <= 0:
            return err(ValidationError(f"Q must be a positive number, got {Q}"))

        # Report progress
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="audio_filter",
                percent=0.1,
                message=f"Applying {filter_type} filter at {freq} Hz",
            )
        )

        # Apply filter
        try:
            output_file, sample_rate, duration = self._filter_fn(
                file_path=audio.file_path,
                sample_rate=audio.sample_rate,
                filter_type=filter_type,
                freq=freq,
                gain_db=gain_db,
                Q=Q,
            )
        except (ValidationError, ExecutionError) as exc:
            return err(exc)
        except Exception as exc:
            return err(
                ExecutionError(f"Filtering failed for block '{block_id}': {exc}")
            )

        # Report complete
        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="audio_filter",
                percent=1.0,
                message="Filtering complete",
            )
        )

        return ok(
            AudioData(
                sample_rate=sample_rate,
                duration=duration,
                file_path=output_file,
                channel_count=audio.channel_count,
            )
        )



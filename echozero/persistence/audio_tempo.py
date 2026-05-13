"""
Tempo detection for imported song audio.
Exists because imported songs should gain basic musical timing truth at ingest time.
Connects persistence import flows to version-scoped BPM and beat-anchor metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioTempoMetadata:
    """Version-scoped tempo metadata derived from imported audio."""

    bpm: float
    bpm_confidence: float | None
    beat_anchor_seconds: float


def detect_audio_tempo(audio_path: Path) -> AudioTempoMetadata | None:
    """Return BPM plus a first-beat anchor for one audio file when detection succeeds."""

    try:
        import librosa
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency failure is environment-specific
        logger.warning("Tempo detection unavailable for '%s': %s", audio_path, exc)
        return None

    try:
        loaded_audio = _load_tempo_analysis_audio(audio_path, sample_rate=22050)
        if loaded_audio is None:
            return None
        samples, sample_rate = loaded_audio
        if samples.size == 0:
            return None
        onset_envelope = librosa.onset.onset_strength(y=samples, sr=sample_rate)
        filename_bpm_hint = _filename_bpm_hint(audio_path)
        tempo_value, beat_frames = _track_beats(
            onset_envelope=onset_envelope,
            sample_rate=sample_rate,
            filename_bpm_hint=filename_bpm_hint,
        )
        bpm = _coerce_bpm(tempo_value)
        if bpm is None or bpm <= 0.0:
            return None
        bpm = _resolve_reported_bpm(bpm, filename_bpm_hint=filename_bpm_hint)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        beat_anchor_seconds = _first_beat_anchor_seconds(beat_times)
        confidence = _estimate_bpm_confidence(beat_times)
    except Exception as exc:  # pragma: no cover - decode/model failure is environment-specific
        logger.warning("Tempo detection failed for '%s': %s", audio_path, exc)
        return None

    return AudioTempoMetadata(
        bpm=bpm,
        bpm_confidence=confidence,
        beat_anchor_seconds=beat_anchor_seconds,
    )


def _load_tempo_analysis_audio(
    audio_path: Path,
    *,
    sample_rate: int,
):
    """Load mono analysis samples, preferring the program channel when LTC is present."""

    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except Exception:
        try:
            samples, resolved_sample_rate = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        except Exception:
            return None
        return np.asarray(samples, dtype=np.float32), int(resolved_sample_rate)

    program_channel_index = _program_channel_index(audio_path)
    if program_channel_index is None:
        try:
            samples, resolved_sample_rate = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        except Exception:
            return None
        return np.asarray(samples, dtype=np.float32), int(resolved_sample_rate)

    try:
        channel_samples, resolved_sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
    except Exception:
        samples, resolved_sample_rate = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        return np.asarray(samples, dtype=np.float32), int(resolved_sample_rate)

    if getattr(channel_samples, "ndim", 0) != 2 or channel_samples.shape[1] <= program_channel_index:
        samples, resolved_sample_rate = librosa.load(str(audio_path), sr=sample_rate, mono=True)
        return np.asarray(samples, dtype=np.float32), int(resolved_sample_rate)

    program_samples = np.asarray(channel_samples[:, program_channel_index], dtype=np.float32)
    if int(resolved_sample_rate) != int(sample_rate):
        program_samples = librosa.resample(
            program_samples,
            orig_sr=int(resolved_sample_rate),
            target_sr=int(sample_rate),
        )
        resolved_sample_rate = int(sample_rate)
    return program_samples, int(resolved_sample_rate)


def _program_channel_index(audio_path: Path) -> int | None:
    """Return the stereo program-channel index when one LTC channel is detectable."""

    try:
        from echozero.persistence.audio import detect_ltc_channel
    except Exception:
        return None
    try:
        ltc_channel = detect_ltc_channel(audio_path, mode="aggressive")
    except TypeError:
        ltc_channel = detect_ltc_channel(audio_path)
    except Exception:
        return None
    if ltc_channel == "left":
        return 1
    if ltc_channel == "right":
        return 0
    return None


def _track_beats(
    *,
    onset_envelope,
    sample_rate: int,
    filename_bpm_hint: float | None,
):
    """Track beats with an optional filename-derived BPM prior."""

    import librosa

    if filename_bpm_hint is not None and filename_bpm_hint > 0.0:
        return librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            start_bpm=float(filename_bpm_hint),
        )
    return librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sample_rate,
    )


def _filename_bpm_hint(audio_path: Path) -> float | None:
    """Extract a BPM hint from filenames like `SongName_85bpm_v01.wav`."""

    match = re.search(
        r"(?<!\d)(\d{2,3}(?:\.\d+)?)\s*bpm(?![a-z0-9])",
        audio_path.stem,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        bpm = float(match.group(1))
    except (TypeError, ValueError):
        return None
    return bpm if 30.0 <= bpm <= 300.0 else None


def _resolve_reported_bpm(measured_bpm: float, *, filename_bpm_hint: float | None) -> float:
    """Prefer the explicit filename BPM when the measured tempo converges nearby."""

    if filename_bpm_hint is None or filename_bpm_hint <= 0.0:
        return measured_bpm
    if abs(float(measured_bpm) - float(filename_bpm_hint)) <= 2.0:
        return float(filename_bpm_hint)
    return measured_bpm


def _coerce_bpm(raw_value: object) -> float | None:
    try:
        if hasattr(raw_value, "item"):
            raw_value = raw_value.item()
        bpm = float(raw_value)
    except (AttributeError, TypeError, ValueError):
        return None
    return bpm if bpm > 0.0 else None


def _first_beat_anchor_seconds(beat_times: object) -> float:
    try:
        first = float(beat_times[0])  # type: ignore[index]
    except (IndexError, TypeError, ValueError):
        return 0.0
    return max(0.0, first)


def _estimate_bpm_confidence(beat_times: object) -> float | None:
    try:
        import numpy as np

        beat_values = np.asarray(beat_times, dtype=np.float64)
    except Exception:
        return None
    if beat_values.size < 3:
        return None
    intervals = np.diff(beat_values)
    mean_interval = float(intervals.mean()) if intervals.size else 0.0
    if mean_interval <= 1e-6:
        return None
    variation = float(intervals.std()) / mean_interval
    confidence = max(0.0, min(1.0, 1.0 - variation))
    return confidence

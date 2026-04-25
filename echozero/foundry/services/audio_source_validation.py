"""Audio source validation helpers for Foundry ingest and training.
Exists to keep invalid-source checks consistent across dataset ingest and runtime sample loading.
Connects dataset/trainer services to shared zero-byte, decode, and finite-audio guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


class InvalidAudioSourceError(ValueError):
    """Raised when a dataset sample source cannot be decoded into usable audio."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class AudioSourceMetadata:
    """Basic metadata discovered while validating an audio source."""

    duration_ms: float
    sample_rate: int
    channels: int


def inspect_audio_source(path: Path) -> AudioSourceMetadata:
    """Validate an audio source without loading the full waveform into memory."""
    size_bytes = _file_size_bytes(path)
    if size_bytes <= 0:
        raise InvalidAudioSourceError("zero_byte", "audio source is zero bytes")

    try:
        info = sf.info(path)
    except Exception as exc:
        raise InvalidAudioSourceError("unreadable", f"audio source could not be decoded: {exc}") from exc

    if int(info.samplerate) <= 0:
        raise InvalidAudioSourceError("invalid_sample_rate", "audio source has an invalid sample rate")
    if int(info.frames) <= 0:
        raise InvalidAudioSourceError("empty_audio", "audio source contains no frames")

    duration_ms = (float(info.frames) / float(info.samplerate)) * 1000.0
    return AudioSourceMetadata(
        duration_ms=duration_ms,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
    )


def load_audio_source(path: Path, *, sample_rate: int, max_length: int) -> np.ndarray:
    """Decode, normalize, and pad audio for training feature extraction."""
    size_bytes = _file_size_bytes(path)
    if size_bytes <= 0:
        raise InvalidAudioSourceError("zero_byte", "audio source is zero bytes")

    try:
        audio, file_sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        raise InvalidAudioSourceError("unreadable", f"audio source could not be decoded: {exc}") from exc

    if int(file_sample_rate) <= 0:
        raise InvalidAudioSourceError("invalid_sample_rate", "audio source has an invalid sample rate")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise InvalidAudioSourceError("empty_audio", "audio source contains no frames")

    if file_sample_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=file_sample_rate, target_sr=sample_rate)
        audio = np.asarray(audio, dtype=np.float32)

    if not np.isfinite(audio).all():
        raise InvalidAudioSourceError("non_finite_audio", "audio source contains non-finite samples")

    if len(audio) > max_length:
        audio = audio[:max_length]
    elif len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)))

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak
    audio = audio.astype(np.float32)

    if not np.isfinite(audio).all():
        raise InvalidAudioSourceError("non_finite_audio", "normalized audio source contains non-finite samples")
    return audio


def _file_size_bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError as exc:
        raise InvalidAudioSourceError("unreadable", f"audio source metadata could not be read: {exc}") from exc


__all__ = ["AudioSourceMetadata", "InvalidAudioSourceError", "inspect_audio_source", "load_audio_source"]

"""Audio tempo metadata detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AudioTempoMetadata:
    bpm: float | None = None
    bpm_confidence: float | None = None
    beat_anchor_seconds: float | None = None


def detect_audio_tempo(path: str | Path) -> AudioTempoMetadata:
    """Return tempo metadata when optional MIR dependencies can detect it."""

    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(path), sr=None, mono=True)
        if y.size == 0:
            return AudioTempoMetadata()
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo.reshape(-1)[0]) if tempo.size else 0.0
        anchor = None
        if len(beats):
            anchor = float(librosa.frames_to_time([beats[0]], sr=sr)[0])
        return AudioTempoMetadata(
            bpm=float(tempo) if tempo else None,
            bpm_confidence=0.5 if tempo else None,
            beat_anchor_seconds=anchor,
        )
    except Exception:
        return AudioTempoMetadata()


__all__ = ["AudioTempoMetadata", "detect_audio_tempo"]

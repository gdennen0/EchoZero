"""Heuristic drum-classifier preview for real-data timeline visualization.

This is a UI/dev-loop preview classifier (not production ML inference).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DrumHit:
    time: float
    label: str
    confidence: float


def classify_drum_hits(
    audio_file: str | Path,
    *,
    onset_threshold: float = 0.02,
    min_gap: float = 0.04,
    sample_rate: int = 44100,
) -> dict[str, list[DrumHit]]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("librosa is required for drum preview classification") from exc

    path = str(audio_file)
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    if y.size == 0:
        return {"kick": [], "snare": [], "hihat": [], "clap": []}

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=False, delta=onset_threshold)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # gap filtering to avoid duplicate detections
    if min_gap > 0 and len(onset_times) > 1:
        filtered = [onset_times[0]]
        for t in onset_times[1:]:
            if t - filtered[-1] >= min_gap:
                filtered.append(t)
        onset_times = filtered

    buckets: dict[str, list[DrumHit]] = {"kick": [], "snare": [], "hihat": [], "clap": []}

    win = int(sr * 0.03)
    half = max(8, win // 2)
    for t in onset_times:
        idx = int(t * sr)
        lo = max(0, idx - half)
        hi = min(len(y), idx + half)
        segment = y[lo:hi]
        if segment.size < 8:
            continue

        # feature extraction
        rms = float(np.sqrt(np.mean(segment * segment)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(segment, frame_length=min(2048, segment.size), hop_length=max(1, segment.size // 4))))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))

        spec = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
        freqs = np.fft.rfftfreq(segment.size, d=1.0 / sr)
        total = float(spec.sum()) + 1e-9
        low_ratio = float(spec[freqs <= 200.0].sum() / total)

        label, conf = label_drum_hit(low_ratio=low_ratio, centroid_hz=centroid, zcr=zcr, rms=rms)
        buckets[label].append(DrumHit(time=float(t), label=label, confidence=conf))

    return buckets


def label_drum_hit(*, low_ratio: float, centroid_hz: float, zcr: float, rms: float) -> tuple[str, float]:
    """Simple feature-threshold classifier for preview lanes."""
    if low_ratio > 0.34 and centroid_hz < 1700.0:
        return "kick", min(0.99, 0.55 + low_ratio)
    if centroid_hz > 5200.0 and zcr > 0.18:
        return "hihat", min(0.99, 0.45 + min(0.4, zcr))
    if 2200.0 <= centroid_hz <= 5200.0 and rms > 0.015:
        return "snare", min(0.99, 0.45 + min(0.4, rms * 8.0))
    return "clap", 0.45

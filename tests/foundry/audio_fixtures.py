from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def write_percussion_dataset(root: Path, *, sample_rate: int = 22050, sample_count: int = 4) -> None:
    (root / "kick").mkdir(parents=True, exist_ok=True)
    (root / "snare").mkdir(parents=True, exist_ok=True)
    for index in range(sample_count):
        sf.write(
            root / "kick" / f"k{index + 1}.wav",
            _kick_wave(index, sample_rate),
            sample_rate,
        )
        sf.write(
            root / "snare" / f"s{index + 1}.wav",
            _snare_wave(index, sample_rate),
            sample_rate,
        )


def _kick_wave(index: int, sample_rate: int) -> np.ndarray:
    duration = 0.18 + (index * 0.005)
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    envelope = np.exp(-28.0 * t)
    base = np.sin(2.0 * np.pi * (62.0 + (index * 4.0)) * t)
    overtone = 0.25 * np.sin(2.0 * np.pi * (124.0 + (index * 3.0)) * t)
    click = np.exp(-400.0 * t) * np.sin(2.0 * np.pi * 1500.0 * t)
    return (0.9 * envelope * (base + overtone) + 0.08 * click).astype(np.float32)


def _snare_wave(index: int, sample_rate: int) -> np.ndarray:
    duration = 0.16 + (index * 0.004)
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(1000 + index)
    noise = rng.standard_normal(len(t), dtype=np.float32)
    tone = np.sin(2.0 * np.pi * (210.0 + (index * 8.0)) * t)
    envelope = np.exp(-24.0 * t)
    body = 0.55 * envelope * noise
    crack = 0.35 * np.exp(-220.0 * t) * tone
    return (body + crack).astype(np.float32)

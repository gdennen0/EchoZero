"""Visual Lab waveform preview data.
Exists to make waveform UI states visible without depending on local audio cache files.
Connects lab current-state previews to the same cached-waveform shape used by Qt inspectors.
"""

from __future__ import annotations

from math import pi, sin

import numpy as np

from echozero.ui.qt.timeline.object_info_panel_preview import EventPreviewState
from echozero.ui.qt.timeline.waveform_cache import CachedWaveform
from echozero.ui.qt.timeline.waveform_cache import _put_cached_waveform

FUN_WAVEFORM_KEY = "visual-lab:fun-sine-waveform"
FUN_WAVEFORM_SOURCE = "visual-lab://synthetic/fun-sine"


def build_fun_waveform_peaks(column_count: int = 420) -> np.ndarray:
    """Build a readable synthetic min/max waveform with sine and goofy pulse accents."""

    count = max(8, int(column_count))
    peaks = np.zeros((count, 2), dtype=np.float32)
    for index in range(count):
        t = index / max(1, count - 1)
        carrier = abs(sin(2.0 * pi * (3.0 * t + 0.32 * sin(2.0 * pi * t))))
        wobble = 0.34 + (0.28 * abs(sin(2.0 * pi * 9.0 * t)))
        duck = 1.0 - (0.45 * abs(sin(2.0 * pi * 1.5 * t)))
        silly_pop = 0.0
        for center in (0.16, 0.31, 0.53, 0.71, 0.84):
            silly_pop = max(silly_pop, max(0.0, 1.0 - abs(t - center) / 0.018))
        amplitude = min(1.0, (carrier * wobble * duck) + (0.42 * silly_pop) + 0.08)
        asymmetry = 0.72 + (0.22 * sin(2.0 * pi * 5.0 * t))
        peaks[index, 0] = -float(amplitude * asymmetry)
        peaks[index, 1] = float(amplitude)
    return peaks


def register_fun_waveform_preview(key: str = FUN_WAVEFORM_KEY) -> CachedWaveform:
    """Register the Visual Lab synthetic waveform in the current waveform cache."""

    cached = CachedWaveform(
        sample_rate=48_000,
        window_size=512,
        peaks=build_fun_waveform_peaks(),
    )
    _put_cached_waveform(key, cached)
    return cached


def build_fun_event_preview_state(key: str = FUN_WAVEFORM_KEY) -> EventPreviewState:
    """Build an inspector-compatible event preview backed by the synthetic waveform."""

    register_fun_waveform_preview(key)
    return EventPreviewState(
        layer_id="current_drum_events",
        take_id="current_drums_main",
        event_id="current_snare_1",
        source_ref=FUN_WAVEFORM_SOURCE,
        source_audio_path=None,
        waveform_key=key,
        start_seconds=0.36,
        end_seconds=3.20,
        duration_seconds=2.84,
    )

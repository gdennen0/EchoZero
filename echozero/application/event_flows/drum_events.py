"""
Drum-event settings helpers.
Exists because pipeline templates and persisted compact settings need one shared
mapping from simple sensitivity presets to low-level onset/classifier thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_PRESET_DELTAS: dict[str, tuple[float, float]] = {
    "more_events": (-0.05, -0.04),
    "balanced": (0.0, 0.0),
    "custom": (0.0, 0.0),
    "fewer_events": (0.05, 0.04),
}


def apply_drum_event_sensitivity_preset(
    values: Mapping[str, Any],
    *,
    sensitivity: object = "balanced",
) -> dict[str, Any]:
    """Apply a named sensitivity preset to drum-event threshold values.

    Presets deliberately adjust only onset and positive-classification thresholds.
    `custom` and unknown values leave the supplied values untouched so advanced/manual
    controls remain authoritative.
    """

    result = dict(values)
    preset = str(sensitivity or "balanced").strip().lower()
    if preset not in _PRESET_DELTAS:
        return result
    onset_delta, positive_delta = _PRESET_DELTAS[preset]
    if onset_delta == 0.0 and positive_delta == 0.0:
        return result
    for key, value in tuple(result.items()):
        if key.endswith("_onset_threshold"):
            result[key] = _clamped_threshold(value, onset_delta)
        elif key.endswith("_positive_threshold") or key == "positive_threshold":
            result[key] = _clamped_threshold(value, positive_delta)
    return result


def _clamped_threshold(value: Any, delta: float) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = 0.0
    threshold = min(1.0, max(0.0, threshold + delta))
    return round(threshold, 6)


__all__ = ["apply_drum_event_sensitivity_preset"]

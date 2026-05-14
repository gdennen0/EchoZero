"""
Drum-event settings helpers.
Exists because pipeline templates and persisted compact settings need one shared
mapping from simple sensitivity presets to low-level onset/classifier thresholds.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_PRESET_DELTAS: dict[str, tuple[float, float]] = {
    "more_events": (-0.05, -0.04),
    "balanced": (0.0, 0.0),
    "custom": (0.0, 0.0),
    "fewer_events": (0.05, 0.04),
}


@dataclass(frozen=True, slots=True)
class DrumModelReadiness:
    label: str
    status: str
    is_ready: bool
    value: str = ""


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


def model_readiness_from_fields(fields: Iterable[object]) -> tuple[DrumModelReadiness, ...]:
    """Summarize per-drum model field readiness from settings fields."""

    readiness: list[DrumModelReadiness] = []
    for field in fields:
        key = str(getattr(field, "key", ""))
        if not key.endswith("_model_path"):
            continue
        label = _model_label_from_key(key)
        value = str(getattr(field, "value", "") or "")
        enabled = bool(getattr(field, "enabled", True))
        options = tuple(getattr(field, "options", ()) or ())
        selected_option = next(
            (option for option in options if str(getattr(option, "value", "")) == value),
            None,
        )
        metadata = getattr(selected_option, "metadata", {}) if selected_option is not None else {}
        status = str(getattr(metadata, "get", lambda _k, _d=None: _d)("status", "") or "")
        if not status:
            status = "ready" if enabled and value else "missing"
        readiness.append(
            DrumModelReadiness(
                label=label,
                status=status,
                is_ready=status.lower() == "ready" and bool(value),
                value=value,
            )
        )
    return tuple(readiness)


def _clamped_threshold(value: Any, delta: float) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = 0.0
    threshold = min(1.0, max(0.0, threshold + delta))
    return round(threshold, 6)


def _model_label_from_key(key: str) -> str:
    label = key[: -len("_model_path")].replace("_", " ").strip()
    return label or "model"


__all__ = [
    "DrumModelReadiness",
    "apply_drum_event_sensitivity_preset",
    "model_readiness_from_fields",
]

"""
Drum event-flow contract for compact extraction controls.
Exists to translate musician-facing event type and sensitivity choices into existing pipeline knobs.
Connects timeline settings, model readiness, and drum extraction templates without adding persistence behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

DRUM_EVENT_LABELS: tuple[str, ...] = ("kick", "snare", "clap", "cymbal")
DRUM_EVENT_SENSITIVITY_OPTIONS: tuple[str, ...] = (
    "more_events",
    "balanced",
    "fewer_events",
    "custom",
)

_ONSET_THRESHOLD_KEYS: tuple[str, ...] = tuple(
    f"{label}_onset_threshold" for label in DRUM_EVENT_LABELS
)
_POSITIVE_THRESHOLD_KEYS: tuple[str, ...] = (
    "positive_threshold",
    *(f"{label}_positive_threshold" for label in DRUM_EVENT_LABELS),
)
_BALANCED_ONSET_THRESHOLDS: Mapping[str, float] = {
    "kick_onset_threshold": 0.15,
    "snare_onset_threshold": 0.15,
    "clap_onset_threshold": 0.35,
    "cymbal_onset_threshold": 0.40,
}
_BALANCED_POSITIVE_THRESHOLDS: Mapping[str, float] = {
    "positive_threshold": 0.60,
    "kick_positive_threshold": 0.50,
    "snare_positive_threshold": 0.65,
    "clap_positive_threshold": 0.60,
    "cymbal_positive_threshold": 0.60,
}


class _FieldLike(Protocol):
    key: str
    value: object
    enabled: bool
    options: Sequence[object]


@dataclass(frozen=True, slots=True)
class CompactDrumEventSettings:
    """Compact user-facing settings for finding drum events in audio."""

    event_labels: tuple[str, ...] = ("kick", "snare")
    sensitivity: str = "balanced"
    model_mode: str = "auto"
    output_mode: str = "event_layers"


@dataclass(frozen=True, slots=True)
class DrumEventModelReadiness:
    """Readiness summary for one drum-event classifier output."""

    label: str
    status: str
    selected_model: str = ""
    ready_count: int = 0

    @property
    def is_ready(self) -> bool:
        """Return true when the selected event type has at least one compatible model."""

        return self.status == "ready"


def drum_event_type_options(
    labels: Sequence[str] = DRUM_EVENT_LABELS,
) -> tuple[tuple[str, str], ...]:
    """Return stable event-type option pairs for compact settings surfaces."""

    return tuple((label, _display_label(label)) for label in normalize_drum_event_labels(labels))


def normalize_drum_event_labels(value: object) -> tuple[str, ...]:
    """Normalize selected drum event labels and common cymbal aliases."""

    if isinstance(value, str):
        raw_values: tuple[object, ...] = tuple(value.split(","))
    elif isinstance(value, (list, tuple, set)):
        raw_values = tuple(value)
    else:
        raw_values = () if value is None else (value,)
    labels: list[str] = []
    for raw_value in raw_values:
        label = _normalize_drum_event_label(raw_value)
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def compile_drum_event_sensitivity_knobs(sensitivity: object) -> dict[str, float]:
    """Compile a compact sensitivity preset to existing detection and classification knobs."""

    preset = str(sensitivity or "balanced").strip().lower()
    if preset == "custom":
        return {}
    if preset not in DRUM_EVENT_SENSITIVITY_OPTIONS:
        preset = "balanced"
    if preset == "balanced":
        return {**_BALANCED_ONSET_THRESHOLDS, **_BALANCED_POSITIVE_THRESHOLDS}
    if preset == "more_events":
        return {
            **{
                key: _clamp_probability(value - 0.05)
                for key, value in _BALANCED_ONSET_THRESHOLDS.items()
            },
            **{
                key: _clamp_probability(value - 0.04)
                for key, value in _BALANCED_POSITIVE_THRESHOLDS.items()
            },
        }
    return {
        **{
            key: _clamp_probability(value + 0.07)
            for key, value in _BALANCED_ONSET_THRESHOLDS.items()
        },
        **{
            key: _clamp_probability(value + 0.06)
            for key, value in _BALANCED_POSITIVE_THRESHOLDS.items()
        },
    }


def model_readiness_from_fields(
    fields: Sequence[_FieldLike],
) -> tuple[DrumEventModelReadiness, ...]:
    """Summarize compatible model availability from action-setting model picker fields."""

    fields_by_key = {field.key: field for field in fields}
    readiness: list[DrumEventModelReadiness] = []
    for label in DRUM_EVENT_LABELS:
        field = fields_by_key.get(f"{label}_model_path")
        if field is None:
            continue
        ready_count = sum(
            1
            for option in field.options
            if getattr(option, "metadata", {}).get("status") == "ready"
        )
        selected_model = str(field.value or "").strip()
        if not field.enabled:
            status = "missing"
        elif selected_model:
            status = "ready"
        elif ready_count:
            status = "select"
        else:
            status = "missing"
        readiness.append(
            DrumEventModelReadiness(
                label=label,
                status=status,
                selected_model=selected_model,
                ready_count=ready_count,
            )
        )
    return tuple(readiness)


def should_preserve_custom_sensitivity_values(values: Mapping[str, object]) -> bool:
    """Return whether raw knobs look intentionally customized from balanced defaults."""

    balanced = compile_drum_event_sensitivity_knobs("balanced")
    for key in (*_ONSET_THRESHOLD_KEYS, *_POSITIVE_THRESHOLD_KEYS):
        if key not in values or key not in balanced:
            continue
        try:
            current = float(values[key])
        except (TypeError, ValueError):
            continue
        if abs(current - balanced[key]) > 1e-9:
            return True
    return False


def apply_drum_event_sensitivity_preset(
    values: Mapping[str, object],
    *,
    sensitivity: object,
) -> dict[str, object]:
    """Return pipeline values with a sensitivity preset applied where safe."""

    preset = str(sensitivity or "balanced").strip().lower()
    if preset == "custom" or (
        preset == "balanced" and should_preserve_custom_sensitivity_values(values)
    ):
        return dict(values)
    compiled = compile_drum_event_sensitivity_knobs(preset)
    if not compiled:
        return dict(values)
    return {**dict(values), **compiled}


def _normalize_drum_event_label(value: object) -> str:
    label = str(value or "").strip().lower()
    if label in {"symbol", "cymbol"}:
        return "cymbal"
    return label


def _display_label(label: str) -> str:
    return str(label).replace("_", " ").title()


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, round(value, 3)))

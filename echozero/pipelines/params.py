"""
Pipeline Knobs: typed parameter metadata for UI auto-generation.

Exists because pipeline builders need typed, validated parameter metadata that the
visual editor can inspect to render appropriate widgets — without the builder knowing
anything about the UI.

A Knob is the single source of truth for "what can the user tweak?" on a pipeline.
The knob() convenience function builds one with validation. extract_knobs() pulls
them from a function signature. validate_bindings() checks user values at runtime.

Provides rich metadata (widgets, constraints, groups) for pipeline parameters.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Widget enum
# ---------------------------------------------------------------------------


class KnobWidget(Enum):
    """Widget types for the visual editor's block inspector.

    AUTO lets the system infer from the parameter's Python type and metadata.
    Specialized widgets (frequency, gain, time_range) carry domain-specific
    defaults so the UI doesn't need to know about audio conventions.
    """

    AUTO = auto()

    # Primitives
    SLIDER = auto()          # float/int with min/max
    DROPDOWN = auto()        # select one from options
    TOGGLE = auto()          # bool on/off
    TEXT = auto()             # free-form string
    NUMBER = auto()          # numeric input without slider

    # File / model
    FILE_PICKER = auto()     # browse for files (audio, config, etc.)
    MODEL_PICKER = auto()    # file_picker + model registry resolution

    # Visual
    COLOR_PICKER = auto()    # hex/rgb color

    # Audio-specific
    FREQUENCY = auto()       # log-scale slider, 20 Hz – 20 kHz
    GAIN = auto()            # linear slider, −48 dB – +48 dB
    TIME_RANGE = auto()      # start/end time pair (seconds)

    # Multi-value
    MULTI_SELECT = auto()    # select multiple from options


# ---------------------------------------------------------------------------
# Knob
# ---------------------------------------------------------------------------

# Widgets that require min_value and max_value
_RANGE_WIDGETS = frozenset({
    KnobWidget.SLIDER,
    KnobWidget.FREQUENCY,
    KnobWidget.GAIN,
})

# Widgets that require options
_OPTION_WIDGETS = frozenset({
    KnobWidget.DROPDOWN,
    KnobWidget.MULTI_SELECT,
})


@dataclass(frozen=True)
class Knob:
    """Immutable parameter definition — one per tweakable knob on a pipeline.

    Created via the ``knob()`` convenience function which handles AUTO inference
    and validates constraints at construction time.
    """

    default: Any
    label: str = ""
    description: str = ""
    widget: KnobWidget = KnobWidget.AUTO
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    options: tuple[str, ...] | None = None
    hidden: bool = False
    advanced: bool = False
    group: str = ""
    units: str = ""
    log_scale: bool = False
    file_types: tuple[str, ...] | None = None
    depends_on: str | None = None
    placeholder: str = ""


# ---------------------------------------------------------------------------
# AUTO inference
# ---------------------------------------------------------------------------


def _infer_widget(default: Any, *, min_value, max_value, options, file_types) -> KnobWidget:
    """Infer the best widget from the default value's type and supplied metadata."""

    if isinstance(default, bool):
        return KnobWidget.TOGGLE

    if isinstance(default, (int, float)) and min_value is not None and max_value is not None:
        return KnobWidget.SLIDER

    if isinstance(default, (int, float)):
        return KnobWidget.NUMBER

    if isinstance(default, str):
        if options is not None:
            return KnobWidget.DROPDOWN
        if file_types is not None:
            return KnobWidget.FILE_PICKER
        return KnobWidget.TEXT

    if isinstance(default, Path):
        return KnobWidget.FILE_PICKER

    if isinstance(default, (list, tuple)) and options is not None:
        return KnobWidget.MULTI_SELECT

    # Fallback — TEXT is the safest universal widget
    return KnobWidget.TEXT


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate(
    widget: KnobWidget,
    default: Any,
    min_value: float | None,
    max_value: float | None,
    step: float | None,
    options: tuple[str, ...] | None,
) -> list[str]:
    """Return a list of validation errors (empty = valid)."""

    errors: list[str] = []

    # Range constraints
    if widget in _RANGE_WIDGETS:
        if min_value is None or max_value is None:
            errors.append(f"{widget.name} requires both min_value and max_value")
        elif min_value >= max_value:
            errors.append(f"min_value ({min_value}) must be less than max_value ({max_value})")

    # Even for non-range widgets, if both min/max are given they should be sane
    if min_value is not None and max_value is not None and min_value >= max_value:
        if widget not in _RANGE_WIDGETS:  # already reported above
            errors.append(f"min_value ({min_value}) must be less than max_value ({max_value})")

    # Options constraints
    if widget in _OPTION_WIDGETS:
        if options is None or len(options) == 0:
            errors.append(f"{widget.name} requires non-empty options")

    # Step
    if step is not None and step <= 0:
        errors.append(f"step must be positive, got {step}")

    # Default within range
    if (
        min_value is not None
        and max_value is not None
        and isinstance(default, (int, float))
        and not isinstance(default, bool)
    ):
        if not (min_value <= default <= max_value):
            errors.append(
                f"default ({default}) outside range [{min_value}, {max_value}]"
            )

    # Default in options
    if options is not None and widget == KnobWidget.DROPDOWN:
        if isinstance(default, str) and default not in options:
            errors.append(f"default '{default}' not in options {options}")

    return errors


# ---------------------------------------------------------------------------
# param() builder
# ---------------------------------------------------------------------------


def knob(
    default: Any,
    *,
    label: str = "",
    description: str = "",
    widget: KnobWidget = KnobWidget.AUTO,
    min_value: float | None = None,
    max_value: float | None = None,
    step: float | None = None,
    options: tuple[str, ...] | None = None,
    hidden: bool = False,
    advanced: bool = False,
    group: str = "",
    units: str = "",
    log_scale: bool = False,
    file_types: tuple[str, ...] | None = None,
    depends_on: str | None = None,
    placeholder: str = "",
) -> Knob:
    """Build a validated Knob.

    If *widget* is ``AUTO``, the concrete widget is inferred from the default
    value's Python type and the presence of *min_value*/*max_value*/*options*.

    Audio-specific widgets (FREQUENCY, GAIN) inject sensible defaults for
    min_value, max_value, units, and log_scale when not explicitly overridden.

    Raises ``ValueError`` if the definition is inconsistent (e.g. SLIDER
    without min/max, dropdown without options, default out of range).
    """

    # Audio-specific widget defaults
    if widget == KnobWidget.FREQUENCY:
        min_value = min_value if min_value is not None else 20.0
        max_value = max_value if max_value is not None else 20_000.0
        units = units or "Hz"
        log_scale = True  # always log for frequency

    if widget == KnobWidget.GAIN:
        min_value = min_value if min_value is not None else -48.0
        max_value = max_value if max_value is not None else 48.0
        units = units or "dB"

    # Infer widget from type if AUTO
    resolved_widget = widget
    if widget == KnobWidget.AUTO:
        resolved_widget = _infer_widget(
            default,
            min_value=min_value,
            max_value=max_value,
            options=options,
            file_types=file_types,
        )

    # Validate
    errors = _validate(resolved_widget, default, min_value, max_value, step, options)
    if errors:
        raise ValueError(
            f"Invalid Knob: {'; '.join(errors)}"
        )

    return Knob(
        default=default,
        label=label,
        description=description,
        widget=resolved_widget,
        min_value=min_value,
        max_value=max_value,
        step=step,
        options=options,
        hidden=hidden,
        advanced=advanced,
        group=group,
        units=units,
        log_scale=log_scale,
        file_types=file_types,
        depends_on=depends_on,
        placeholder=placeholder,
    )


# ---------------------------------------------------------------------------
# extract_knobs / validate_bindings
# ---------------------------------------------------------------------------


def extract_knobs(fn: Callable) -> dict[str, Knob]:
    """Inspect a callable's signature and return all parameters whose defaults are Knob.

    This is how the visual editor discovers what widgets to render for a pipeline
    builder function — zero registration needed.

    Example::

        def my_pipeline(
            threshold: float = param(0.5, min_value=0.0, max_value=1.0, label="Threshold"),
            method: str = param("rms", options=("rms", "spectral_flux"), label="Method"),
        ) -> Graph:
            ...

        params = extract_knobs(my_pipeline)
        # {"threshold": Knob(...), "method": Knob(...)}
    """

    sig = inspect.signature(fn)
    params: dict[str, Knob] = {}

    for name, p in sig.parameters.items():
        if isinstance(p.default, Knob):
            params[name] = p.default

    return params


def validate_bindings(
    params: dict[str, Knob],
    bindings: dict[str, Any],
) -> list[str]:
    """Validate user-supplied bindings against extracted Knobs.

    Returns a list of human-readable error strings (empty = valid).
    Checks:
      - Unknown keys (not in params)
      - Type mismatches (with int→float coercion)
      - Values outside [min_value, max_value]
      - Values not in options (for dropdown)
    """

    errors: list[str] = []

    known = set(params.keys())

    # Unknown keys
    for key in bindings:
        if key not in known:
            errors.append(f"Unknown parameter: '{key}'")

    for key, pdef in params.items():
        if key not in bindings:
            continue  # use default — that's fine

        value = bindings[key]

        # Type check (loose — allow int where float expected)
        expected = type(pdef.default)
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            pass  # int→float coercion OK
        elif not isinstance(value, expected):
            errors.append(
                f"'{key}': expected {expected.__name__}, got {type(value).__name__}"
            )
            continue  # skip further checks if type is wrong

        # Range check
        if (
            pdef.min_value is not None
            and pdef.max_value is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            if not (pdef.min_value <= value <= pdef.max_value):
                errors.append(
                    f"'{key}': value {value} outside [{pdef.min_value}, {pdef.max_value}]"
                )

        # Options check (dropdown)
        if pdef.widget == KnobWidget.DROPDOWN and pdef.options is not None:
            if value not in pdef.options:
                errors.append(f"'{key}': value '{value}' not in {pdef.options}")

        # Multi-select: each element must be in options
        if pdef.widget == KnobWidget.MULTI_SELECT and pdef.options is not None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    if item not in pdef.options:
                        errors.append(
                            f"'{key}': item '{item}' not in {pdef.options}"
                        )

    return errors

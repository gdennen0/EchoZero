"""Settings contracts for timeline object actions.
Exists to keep editable field metadata typed and reusable across surfaces.
Connects application-owned settings plans to neutral UI form rendering.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ObjectActionSettingOption:
    """One selectable option for a dropdown-backed settings field."""

    value: str
    label: str


@dataclass(slots=True, frozen=True)
class ObjectActionSettingField:
    """Typed presentation contract for one editable settings field."""

    key: str
    label: str
    value: object
    default_value: object
    persisted_value: object | None = None
    is_dirty: bool = False
    widget: str = "text"
    description: str = ""
    enabled: bool = True
    advanced: bool = False
    placeholder: str = ""
    units: str = ""
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    options: tuple[ObjectActionSettingOption, ...] = ()


@dataclass(slots=True, frozen=True)
class ObjectActionSettingsPlan:
    """Rendered settings plan for one object action in one scope."""

    action_id: str
    title: str
    object_id: str
    object_type: str
    pipeline_template_id: str
    editable_fields: tuple[ObjectActionSettingField, ...] = ()
    advanced_fields: tuple[ObjectActionSettingField, ...] = ()
    locked_bindings: tuple[tuple[str, str], ...] = ()
    has_prior_outputs: bool = False
    run_label: str = "Run"
    settings_label: str = "Open Settings"
    rerun_hint: str = ""
    summary: str = ""
    warnings: tuple[str, ...] = ()
    operation_id: str | None = None
    is_running: bool = False
    operation_status: str = ""
    operation_message: str = ""
    operation_fraction: float | None = None
    operation_error: str | None = None

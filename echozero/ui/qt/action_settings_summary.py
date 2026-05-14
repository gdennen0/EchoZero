"""
Confirmation-summary helpers for action settings dialogs.
Exists so settings and pipeline-browser dialogs present the same compact
operator confirmation before long-running or destructive-ish pipeline runs.
"""

from __future__ import annotations

from collections.abc import Iterable

from echozero.application.timeline.object_actions import (
    ObjectActionSettingField,
    ObjectActionSettingsSession,
)


def build_action_confirmation_summary(session: ObjectActionSettingsSession) -> str:
    """Build a short confirmation summary for sessions that request one."""

    if not session.plan.requires_settings_confirmation:
        return ""
    fields = {field.key: field for field in _all_fields(session)}
    lines = ["Confirm before run"]

    event_types = _selected_option_labels(fields.get("target_drum_labels"))
    if event_types:
        lines.append(f"Event types: {', '.join(event_types)}")

    sensitivity = _selected_option_label(fields.get("sensitivity_preset"))
    if sensitivity:
        lines.append(f"Sensitivity: {sensitivity}")

    ready_models = _ready_model_count(fields.values())
    if ready_models is not None:
        lines.append(f"Models: {ready_models} ready")

    if session.plan.summary:
        lines.append(session.plan.summary)
    for warning in session.plan.warnings:
        if warning:
            lines.append(str(warning))
    return "\n".join(lines)


def _all_fields(session: ObjectActionSettingsSession) -> tuple[ObjectActionSettingField, ...]:
    return tuple(session.plan.editable_fields) + tuple(session.plan.advanced_fields)


def _selected_option_labels(field: ObjectActionSettingField | None) -> tuple[str, ...]:
    if field is None:
        return ()
    raw_value = field.value
    if isinstance(raw_value, str):
        values = (raw_value,)
    elif isinstance(raw_value, Iterable):
        values = tuple(str(value) for value in raw_value)
    else:
        values = (str(raw_value),)
    labels: list[str] = []
    options = {str(option.value): option.label for option in field.options}
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        labels.append(options.get(text, text.replace("_", " ").title()))
    return tuple(labels)


def _selected_option_label(field: ObjectActionSettingField | None) -> str:
    labels = _selected_option_labels(field)
    return labels[0] if labels else ""


def _ready_model_count(fields: Iterable[ObjectActionSettingField]) -> int | None:
    count = 0
    saw_model_field = False
    for field in fields:
        if not field.key.endswith("_model_path"):
            continue
        saw_model_field = True
        current_value = str(field.value or "")
        if not current_value:
            continue
        matching_option = next(
            (option for option in field.options if str(option.value) == current_value),
            None,
        )
        status = str((matching_option.metadata if matching_option else {}).get("status", ""))
        if status.lower() == "ready":
            count += 1
    return count if saw_model_field else None


__all__ = ["build_action_confirmation_summary"]

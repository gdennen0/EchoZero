"""
Action settings summary text for Qt confirmation surfaces.
Exists so run-confirmation dialogs show the same operator-readable settings recap.
Connects application-owned object-action sessions to compact dialog header copy.
"""

from __future__ import annotations

from collections.abc import Sequence

from echozero.application.timeline.object_actions import (
    ObjectActionSettingField,
    ObjectActionSettingsSession,
)

_DRUM_ACTION_IDS = frozenset(
    {
        "timeline.extract_classified_drums",
        "timeline.extract_song_drum_events",
    }
)


def build_action_confirmation_summary(session: ObjectActionSettingsSession) -> str:
    """Return a compact always-visible recap for actions that require confirmation."""

    if not session.plan.requires_settings_confirmation:
        return ""
    if session.action_id in _DRUM_ACTION_IDS:
        return _drum_confirmation_summary(session)
    if session.action_id == "timeline.extract_song_sections":
        method = _selected_option_label(session, key="detect_method")
        return _join_summary_parts(
            "Confirm before run",
            (f"Method: {method}" if method else "", _target_part(session)),
        )
    return _join_summary_parts("Confirm before run", (_target_part(session),))


def _drum_confirmation_summary(session: ObjectActionSettingsSession) -> str:
    event_types = _multi_option_labels(session, key="target_drum_labels")
    sensitivity = _selected_option_label(session, key="sensitivity_preset")
    model_status = _model_status_part(session)
    return _join_summary_parts(
        "Confirm before run",
        (
            f"Event types: {', '.join(event_types)}" if event_types else "",
            f"Sensitivity: {sensitivity}" if sensitivity else "",
            model_status,
            _target_part(session),
        ),
    )


def _target_part(session: ObjectActionSettingsSession) -> str:
    target = session.plan.summary or session.plan.object_id or session.plan.object_type
    return f"Target: {target}" if str(target or "").strip() else ""


def _model_status_part(session: ObjectActionSettingsSession) -> str:
    model_fields = tuple(
        field
        for field in (*session.plan.editable_fields, *session.plan.advanced_fields)
        if field.key.endswith("_model_path")
    )
    if not model_fields:
        return ""
    values = session.values
    ready = 0
    missing_labels: list[str] = []
    for field in model_fields:
        label = _label_from_model_field(field)
        selected = str(values.get(field.key, field.value) or "").strip()
        has_ready_option = any(
            getattr(option, "metadata", {}).get("status") == "ready"
            for option in field.options
        )
        if field.enabled and (selected or has_ready_option):
            ready += 1
            continue
        missing_labels.append(label)
    parts = [f"Models: {ready} ready"]
    if missing_labels:
        parts.append(f"missing {', '.join(missing_labels)}")
    return " (".join(parts) + (")" if len(parts) > 1 else "")


def _selected_option_label(session: ObjectActionSettingsSession, *, key: str) -> str:
    field = _field_by_key(session, key)
    if field is None:
        return ""
    value = session.values.get(key, field.value)
    for option in field.options:
        if option.value == value:
            return option.label
    return str(value or "").strip()


def _multi_option_labels(session: ObjectActionSettingsSession, *, key: str) -> tuple[str, ...]:
    field = _field_by_key(session, key)
    if field is None:
        return ()
    selected_values = _coerce_selected_values(session.values.get(key, field.value))
    labels_by_value = {str(option.value): option.label for option in field.options}
    return tuple(labels_by_value.get(value, value.replace("_", " ").title()) for value in selected_values)


def _field_by_key(
    session: ObjectActionSettingsSession,
    key: str,
) -> ObjectActionSettingField | None:
    for field in (*session.plan.editable_fields, *session.plan.advanced_fields):
        if field.key == key:
            return field
    return None


def _coerce_selected_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values: Sequence[object] = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_values = tuple(value)
    else:
        raw_values = () if value is None else (value,)
    values: list[str] = []
    for raw_value in raw_values:
        text = str(raw_value or "").strip()
        if text and text not in values:
            values.append(text)
    return tuple(values)


def _label_from_model_field(field: ObjectActionSettingField) -> str:
    raw_label = field.key[: -len("_model_path")] if field.key.endswith("_model_path") else field.label
    return raw_label.replace("_", " ").title()


def _join_summary_parts(prefix: str, parts: Sequence[str]) -> str:
    clean_parts = tuple(part for part in parts if str(part or "").strip())
    if not clean_parts:
        return prefix
    return f"{prefix}: " + " · ".join(clean_parts)

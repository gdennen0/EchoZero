"""Action-settings adapter over the neutral EchoZero settings-page form.
Exists to keep object-action settings on the shared settings renderer instead of a custom form.
Connects action-owned settings plans to the reusable Qt settings-page surface.
"""

from __future__ import annotations

from dataclasses import replace

from echozero.application.event_flows.drum_events import model_readiness_from_fields
from echozero.application.settings import (
    SettingsField,
    SettingsFieldSurface,
    SettingsFieldWidget,
    SettingsOption,
    SettingsPage,
    SettingsSection,
)
from echozero.application.timeline.object_actions import (
    ObjectActionSettingField,
    ObjectActionSettingsPlan,
)
from echozero.ui.qt.settings_page_form import SettingsPageForm


class ActionSettingsForm(SettingsPageForm):
    """Embeddable renderer/editor for one object action settings plan."""

    def set_plan(self, plan: ObjectActionSettingsPlan) -> None:
        """Render one object-action settings plan into the shared settings form."""

        self.set_page(
            _page_from_action_plan(plan),
            empty_message="No editable settings for this action.",
        )


def _page_from_action_plan(plan: ObjectActionSettingsPlan) -> SettingsPage:
    if plan.action_id in _COMPACT_DRUM_EVENT_ACTION_IDS:
        return _drum_events_page_from_action_plan(plan)

    fields = (
        *_fields_from_action_fields(plan.editable_fields, surface=SettingsFieldSurface.PRIMARY),
        *_fields_from_action_fields(plan.advanced_fields, surface=SettingsFieldSurface.ADVANCED),
    )
    return SettingsPage(
        key=plan.action_id,
        title=plan.title,
        summary=plan.summary,
        sections=(
            SettingsSection(
                key=f"{plan.action_id}.stage",
                title="Stage Settings",
                fields=tuple(fields),
            ),
        ),
        warnings=plan.warnings,
    )


_COMPACT_DRUM_EVENT_ACTION_IDS = frozenset(
    {
        "timeline.extract_classified_drums",
        "timeline.extract_song_drum_events",
    }
)


def _drum_events_page_from_action_plan(plan: ObjectActionSettingsPlan) -> SettingsPage:
    primary = _field_lookup(
        _fields_from_action_fields(plan.editable_fields, surface=SettingsFieldSurface.PRIMARY)
    )
    advanced = _field_lookup(
        _fields_from_action_fields(plan.advanced_fields, surface=SettingsFieldSurface.ADVANCED)
    )
    used_keys: set[str] = set()

    event_fields = _rename_fields(
        _take_fields(primary, used_keys, ("target_drum_labels",)),
        labels={"target_drum_labels": "Event Types"},
    )
    sensitivity_fields = _take_fields(primary, used_keys, ("sensitivity_preset",))
    model_fields = _secondary_fields(
        _take_fields(primary, used_keys, _round_robin_label_keys(("model_path",)))
    )
    model_status = _model_status_description(model_fields)

    sections = [
        SettingsSection(
            key=f"{plan.action_id}.events",
            title="Events to find",
            description="Select the drum event types EchoZero should look for in this audio.",
            fields=event_fields,
            preferred_columns=1,
        ),
        SettingsSection(
            key=f"{plan.action_id}.quality",
            title="Quality",
            description=(
                "Use Sensitivity first. Choose Custom only when you need the raw "
                "threshold and onset controls in Advanced."
            ),
            fields=sensitivity_fields,
            preferred_columns=1,
        ),
        SettingsSection(
            key=f"{plan.action_id}.models",
            title="Model status",
            description=model_status,
            fields=model_fields,
            preferred_columns=1,
        ),
    ]
    remaining_fields = _secondary_fields(
        tuple(field for key, field in primary.items() if key not in used_keys)
    )
    advanced_fields = (*remaining_fields, *tuple(advanced.values()))
    if advanced_fields:
        sections.append(
            SettingsSection(
                key=f"{plan.action_id}.advanced",
                title="Advanced / Custom recovery",
                description=(
                    "Raw model, threshold, filter, onset, device, and assignment knobs. "
                    "Leave these closed unless Sensitivity needs manual override."
                ),
                fields=advanced_fields,
                preferred_columns=2,
            )
        )
    return SettingsPage(
        key=plan.action_id,
        title="Extract Drum Events",
        summary=plan.summary,
        sections=tuple(section for section in sections if section.fields),
        warnings=plan.warnings,
    )


_CLASSIFIED_DRUM_LABELS = ("kick", "snare", "clap", "cymbal")

_CLASSIFIED_DRUM_CONFIDENCE_SUFFIXES = (
    "positive_threshold",
    "min_event_peak",
    "min_event_rms",
    "min_separation_ms",
)

_CLASSIFIED_DRUM_DETECTION_SUFFIXES = (
    "filter_enabled",
    "filter_freq",
    "onset_threshold",
)


def _round_robin_label_keys(suffixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"{label}_{suffix}"
        for suffix in suffixes
        for label in _CLASSIFIED_DRUM_LABELS
    )


def _field_lookup(fields: tuple[SettingsField, ...]) -> dict[str, SettingsField]:
    return {field.key: field for field in fields}


def _rename_fields(
    fields: tuple[SettingsField, ...],
    *,
    labels: dict[str, str],
) -> tuple[SettingsField, ...]:
    return tuple(replace(field, label=labels.get(field.key, field.label)) for field in fields)


def _secondary_fields(fields: tuple[SettingsField, ...]) -> tuple[SettingsField, ...]:
    return tuple(replace(field, surface=SettingsFieldSurface.ADVANCED) for field in fields)


def _take_fields(
    fields_by_key: dict[str, SettingsField],
    used_keys: set[str],
    keys: tuple[str, ...],
) -> tuple[SettingsField, ...]:
    selected: list[SettingsField] = []
    for key in keys:
        field = fields_by_key.get(key)
        if field is None:
            continue
        selected.append(field)
        used_keys.add(key)
    return tuple(selected)


def _model_status_description(fields: tuple[SettingsField, ...]) -> str:
    readiness = model_readiness_from_fields(fields)
    if not readiness:
        return "Model choices are available in Advanced when compatible classifiers are installed."
    ready = tuple(item for item in readiness if item.is_ready)
    missing = tuple(item for item in readiness if item.status == "missing")
    parts = [f"{len(ready)} of {len(readiness)} event types ready."]
    if missing:
        labels = ", ".join(item.label.title() for item in missing)
        parts.append(f"Missing compatible models: {labels}.")
    parts.append("Open Advanced to choose or inspect exact model files.")
    return " ".join(parts)


def _model_card_fields(fields: tuple[SettingsField, ...]) -> tuple[SettingsField, ...]:
    return tuple(_model_card_field(field) for field in fields)


def _model_card_field(field: SettingsField) -> SettingsField:
    label = field.label
    status = "Missing" if not field.enabled else "Ready" if str(field.value or "") else "Select"
    if "·" not in label:
        label = f"{label} · {status}"
    description = field.description
    if field.options:
        selected_label = _selected_option_label(field)
        if selected_label:
            description = f"Selected: {selected_label}. {description}".strip()
    return replace(field, label=label, description=description)


def _selected_option_label(field: SettingsField) -> str:
    for option in field.options:
        if option.value == field.value and str(option.value or ""):
            return option.label
    return ""


def _fields_from_action_fields(
    fields: tuple[ObjectActionSettingField, ...],
    *,
    surface: SettingsFieldSurface,
) -> tuple[SettingsField, ...]:
    return tuple(
        SettingsField(
            key=field.key,
            label=field.label,
            value=field.value,
            default_value=field.default_value,
            persisted_value=field.persisted_value,
            is_dirty=field.is_dirty,
            widget=_widget_for_action_field(field),
            description=field.description,
            enabled=field.enabled,
            surface=surface,
            placeholder=field.placeholder,
            units=field.units,
            min_value=field.min_value,
            max_value=field.max_value,
            step=field.step,
            options=tuple(
                SettingsOption(
                    value=option.value,
                    label=option.label,
                    metadata=option.metadata,
                )
                for option in field.options
            ),
        )
        for field in fields
    )


def _widget_for_action_field(field: ObjectActionSettingField) -> SettingsFieldWidget:
    widget_name = str(field.widget or "text").strip().lower()
    if widget_name == "checkbox_group":
        return SettingsFieldWidget.CHECKBOX_GROUP
    if widget_name == "dropdown":
        return SettingsFieldWidget.DROPDOWN
    if widget_name == "toggle":
        return SettingsFieldWidget.TOGGLE
    if widget_name == "number":
        return SettingsFieldWidget.NUMBER
    return SettingsFieldWidget.TEXT

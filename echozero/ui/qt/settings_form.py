"""Action-settings adapter over the neutral EchoZero settings-page form.
Exists to keep object-action settings on the shared settings renderer instead of a custom form.
Connects action-owned settings plans to the reusable Qt settings-page surface.
"""

from __future__ import annotations

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
                SettingsOption(value=option.value, label=option.label) for option in field.options
            ),
        )
        for field in fields
    )


def _widget_for_action_field(field: ObjectActionSettingField) -> SettingsFieldWidget:
    widget_name = str(field.widget or "text").strip().lower()
    if widget_name == "dropdown":
        return SettingsFieldWidget.DROPDOWN
    if widget_name == "toggle":
        return SettingsFieldWidget.TOGGLE
    if widget_name == "number":
        return SettingsFieldWidget.NUMBER
    return SettingsFieldWidget.TEXT

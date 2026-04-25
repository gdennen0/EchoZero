"""Session models and commands for object-action settings interactions.
Exists to keep draft state, scope policy, and copy policy in the application lane.
Connects reusable settings surfaces to typed object-action session commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from echozero.application.timeline.object_actions.settings import ObjectActionSettingsPlan


@dataclass(slots=True, frozen=True)
class ObjectActionCopySource:
    """One available source the operator can copy settings from."""

    source_id: str
    label: str
    scope: str
    song_id: str | None = None
    version_id: str | None = None
    description: str = ""


@dataclass(slots=True, frozen=True)
class ObjectActionSettingsCopyPreview:
    """Preview of a pending copy operation before it mutates settings."""

    source_id: str
    summary: str
    changes: tuple[tuple[str, object, object], ...] = ()


@dataclass(slots=True, frozen=True)
class ObjectActionSessionFieldValue:
    """Persisted and draft value pair for one editable settings field."""

    key: str
    persisted_value: object
    draft_value: object

    @property
    def is_dirty(self) -> bool:
        """Report whether the draft differs from the persisted value."""

        return self.draft_value != self.persisted_value


@dataclass(slots=True, frozen=True)
class ObjectActionSettingsScopeChoice:
    """One selectable settings scope in the current object-action session."""

    scope: str
    label: str
    can_run: bool = False
    has_unsaved_changes: bool = False


@dataclass(slots=True, frozen=True)
class ObjectActionSettingsScopeState:
    """Draft and persisted state for one scope within an object-action session."""

    scope: str
    label: str
    field_values: tuple[ObjectActionSessionFieldValue, ...] = ()
    can_run: bool = False

    @property
    def draft_values(self) -> dict[str, object]:
        """Return current draft values keyed by field."""

        return {field.key: field.draft_value for field in self.field_values}

    @property
    def persisted_values(self) -> dict[str, object]:
        """Return persisted values keyed by field."""

        return {field.key: field.persisted_value for field in self.field_values}

    @property
    def has_unsaved_changes(self) -> bool:
        """Report whether any field in this scope is dirty."""

        return any(field.is_dirty for field in self.field_values)


@dataclass(slots=True, frozen=True)
class ObjectActionSettingsCopyPolicy:
    """Available copy sources and preview state for the current session scope."""

    target_scope: str
    target_label: str
    sources: tuple[ObjectActionCopySource, ...] = ()
    selected_source_id: str | None = None
    preview: ObjectActionSettingsCopyPreview | None = None


@dataclass(slots=True, frozen=True)
class ObjectActionSettingsSession:
    """Application-owned settings session for one object action."""

    session_id: str
    action_id: str
    object_id: str
    object_type: str
    scope: str
    plan: ObjectActionSettingsPlan
    scope_states: tuple[ObjectActionSettingsScopeState, ...] = ()
    copy_policy: ObjectActionSettingsCopyPolicy = field(
        default_factory=lambda: ObjectActionSettingsCopyPolicy(
            target_scope="version",
            target_label="This Version",
        )
    )
    can_save: bool = True
    can_save_and_run: bool = False
    run_disabled_reason: str = ""

    @property
    def current_scope_state(self) -> ObjectActionSettingsScopeState:
        """Return the scope state backing the currently visible plan."""

        match = next((state for state in self.scope_states if state.scope == self.scope), None)
        if match is None:
            raise ValueError(f"Unknown session scope '{self.scope}'.")
        return match

    @property
    def scope_choices(self) -> tuple[ObjectActionSettingsScopeChoice, ...]:
        """Return the current selectable scope choices for the session."""

        return tuple(
            ObjectActionSettingsScopeChoice(
                scope=state.scope,
                label=state.label,
                can_run=state.can_run,
                has_unsaved_changes=state.has_unsaved_changes,
            )
            for state in self.scope_states
        )

    @property
    def available_scopes(self) -> tuple[str, ...]:
        """Return scope ids in presentation order."""

        return tuple(choice.scope for choice in self.scope_choices)

    @property
    def values(self) -> dict[str, object]:
        """Return current draft values for compatibility callers."""

        return self.current_scope_state.draft_values

    @property
    def copy_sources(self) -> tuple[ObjectActionCopySource, ...]:
        """Return copy sources valid for the currently selected scope."""

        return self.copy_policy.sources

    @property
    def selected_copy_source_id(self) -> str | None:
        """Return the currently selected copy source, if any."""

        return self.copy_policy.selected_source_id

    @property
    def copy_preview(self) -> ObjectActionSettingsCopyPreview | None:
        """Return the active copy preview, if any."""

        return self.copy_policy.preview

    @property
    def has_unsaved_changes(self) -> bool:
        """Report whether the current scope has unsaved edits."""

        return self.current_scope_state.has_unsaved_changes


class ObjectActionSessionCommand:
    """Marker base type for object-action session commands."""


@dataclass(slots=True, frozen=True)
class SetSessionFieldValue(ObjectActionSessionCommand):
    """Replace one editable field in the current session scope."""

    key: str
    value: object


@dataclass(slots=True, frozen=True)
class ReplaceSessionValues(ObjectActionSessionCommand):
    """Replace multiple editable fields in the current session scope."""

    values: dict[str, object]


@dataclass(slots=True, frozen=True)
class ChangeSessionScope(ObjectActionSessionCommand):
    """Switch the active session scope."""

    scope: str


@dataclass(slots=True, frozen=True)
class PreviewCopySource(ObjectActionSessionCommand):
    """Preview copying settings from one source into the current scope."""

    source_id: str


@dataclass(slots=True, frozen=True)
class ApplyCopySource(ObjectActionSessionCommand):
    """Apply copied settings from one source into the current scope."""

    source_id: str


@dataclass(slots=True, frozen=True)
class ResetSessionDefaults(ObjectActionSessionCommand):
    """Reset editable settings in the current scope back to their defaults."""

    pass


@dataclass(slots=True, frozen=True)
class SaveSessionToDefaults(ObjectActionSessionCommand):
    """Persist current draft values into this song's default settings scope."""

    pass


@dataclass(slots=True, frozen=True)
class SaveSession(ObjectActionSessionCommand):
    """Persist the current scope draft values."""

    pass


@dataclass(slots=True, frozen=True)
class SaveAndRunSession(ObjectActionSessionCommand):
    """Persist the current draft values, then execute the object action."""

    pass


@dataclass(slots=True, frozen=True)
class RunSession(ObjectActionSessionCommand):
    """Compatibility command for callers still using the old run name."""

    pass

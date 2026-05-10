"""Timeline object-actions lane.
Exists to keep object-action descriptors, settings contracts, and orchestration under one application boundary.
Connects callers to object-action contracts directly while lazily exposing the execution service to avoid import cycles.
"""

from echozero.application.timeline.object_actions.descriptors import (
    ActionAlias,
    ActionDescriptor,
    SONG_ADD_DESCRIPTOR,
    action_descriptors,
    action_aliases,
    canonical_action_id,
    canonical_action_ids,
    descriptor_for_action,
    is_object_action,
    object_action_descriptors,
    pipeline_actions_for_audio_layer,
    resolve_action_id,
    workflow_descriptor_for_action,
)
from echozero.application.timeline.object_actions.session import (
    ApplyCopySource,
    ChangeSessionScope,
    ObjectActionCopySource,
    ObjectActionSessionFieldValue,
    ObjectActionSettingsCopyPreview,
    ObjectActionSettingsCopyPolicy,
    ObjectActionSettingsScopeChoice,
    ObjectActionSettingsScopeState,
    ObjectActionSettingsSession,
    PreviewCopySource,
    ResetSessionDefaults,
    ReplaceSessionValues,
    RunSession,
    SaveAndRunSession,
    SaveSessionToDefaults,
    SaveSession,
    SetSessionFieldValue,
)
from echozero.application.timeline.object_actions.settings import (
    ObjectActionSettingField,
    ObjectActionSettingOption,
    ObjectActionSettingsPlan,
)

__all__ = [
    "ActionDescriptor",
    "ActionAlias",
    "ApplyCopySource",
    "ChangeSessionScope",
    "ObjectActionService",
    "ObjectActionCopySource",
    "ObjectActionSessionFieldValue",
    "ObjectActionSettingField",
    "ObjectActionSettingOption",
    "ObjectActionSettingsCopyPreview",
    "ObjectActionSettingsCopyPolicy",
    "ObjectActionSettingsPlan",
    "ObjectActionSettingsScopeChoice",
    "ObjectActionSettingsScopeState",
    "ObjectActionSettingsSession",
    "PreviewCopySource",
    "ResetSessionDefaults",
    "ReplaceSessionValues",
    "RunSession",
    "SaveAndRunSession",
    "SONG_ADD_DESCRIPTOR",
    "SaveSessionToDefaults",
    "SaveSession",
    "SetSessionFieldValue",
    "action_aliases",
    "action_descriptors",
    "canonical_action_id",
    "canonical_action_ids",
    "descriptor_for_action",
    "is_object_action",
    "object_action_descriptors",
    "pipeline_actions_for_audio_layer",
    "resolve_action_id",
    "workflow_descriptor_for_action",
]


def __getattr__(name: str):
    if name == "ObjectActionService":
        from echozero.application.timeline.object_actions.service import ObjectActionService

        return ObjectActionService
    raise AttributeError(name)

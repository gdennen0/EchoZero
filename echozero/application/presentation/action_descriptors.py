"""Presentation-facing re-exports for timeline action descriptors.
Exists to keep presentation imports pointed at one canonical descriptor vocabulary.
Connects presentation callers to the timeline object-action descriptor surface.
"""

from echozero.application.timeline.object_actions.descriptors import (
    ActionAlias,
    ActionDescriptor,
    SONG_ADD_DESCRIPTOR,
    action_aliases,
    action_descriptors,
    canonical_action_id,
    canonical_action_ids,
    descriptor_for_action,
    is_object_action,
    object_action_descriptors,
    pipeline_actions_for_audio_layer,
    resolve_action_id,
    workflow_descriptor_for_action,
)

__all__ = [
    "ActionAlias",
    "ActionDescriptor",
    "SONG_ADD_DESCRIPTOR",
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

"""Canonical descriptors for timeline object actions and related operator actions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class ActionDescriptor:
    """Stable action identity with optional legacy aliases and workflow metadata."""

    action_id: str
    label: str
    aliases: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    params_schema: dict[str, object] = field(default_factory=dict)
    workflow_id: str | None = None
    pipeline_template_id: str | None = None
    static_params: dict[str, object] = field(default_factory=dict)
    binding_resolver_id: str | None = None
    runtime_param_coercer_id: str | None = None


SONG_ADD_DESCRIPTOR = ActionDescriptor(
    action_id="song.add",
    label="Add Song",
    aliases=("add_song_from_path",),
    object_types=("timeline",),
    groups=("tools",),
    params_schema={"title": "dialog:text", "audio_path": "dialog:file"},
)
EXTRACT_STEMS_DESCRIPTOR = ActionDescriptor(
    action_id="timeline.extract_stems",
    label="Extract Stems",
    object_types=("layer",),
    groups=("object_action", "tools"),
    params_schema={"layer_id": "required"},
    workflow_id="layer.audio.extract_stems",
    pipeline_template_id="stem_separation",
    binding_resolver_id="extract_stems",
)
EXTRACT_SONG_DRUM_EVENTS_DESCRIPTOR = ActionDescriptor(
    action_id="timeline.extract_song_drum_events",
    label="Extract Drum Events",
    object_types=("layer",),
    groups=("object_action", "tools"),
    params_schema={"layer_id": "required"},
    workflow_id="layer.audio.extract_song_drum_events",
    pipeline_template_id="extract_song_drum_events",
    binding_resolver_id="extract_song_drum_events",
)
EXTRACT_DRUM_EVENTS_DESCRIPTOR = ActionDescriptor(
    action_id="timeline.extract_drum_events",
    label="Extract Onsets",
    object_types=("layer",),
    groups=("object_action", "tools"),
    params_schema={"layer_id": "required"},
    workflow_id="layer.audio.extract_drum_events",
    pipeline_template_id="onset_detection",
    binding_resolver_id="extract_drum_events",
)
CLASSIFY_DRUM_EVENTS_DESCRIPTOR = ActionDescriptor(
    action_id="timeline.classify_drum_events",
    label="Classify Drum Events",
    object_types=("layer",),
    groups=("object_action", "tools"),
    params_schema={"layer_id": "required", "model_path": "dialog:file:model"},
    workflow_id="layer.audio.classify_drum_events",
    pipeline_template_id="drum_classification",
    binding_resolver_id="classify_drum_events",
    runtime_param_coercer_id="classify_drum_events",
)
EXTRACT_CLASSIFIED_DRUMS_DESCRIPTOR = ActionDescriptor(
    action_id="timeline.extract_classified_drums",
    label="Extract Classified Drums",
    object_types=("layer",),
    groups=("object_action", "tools"),
    params_schema={"layer_id": "required"},
    workflow_id="layer.audio.extract_classified_drums",
    pipeline_template_id="extract_classified_drums",
    binding_resolver_id="extract_classified_drums",
)


_DESCRIPTORS_BY_ID: dict[str, ActionDescriptor] = {
    descriptor.action_id: descriptor
    for descriptor in (
        SONG_ADD_DESCRIPTOR,
        EXTRACT_STEMS_DESCRIPTOR,
        EXTRACT_SONG_DRUM_EVENTS_DESCRIPTOR,
        EXTRACT_DRUM_EVENTS_DESCRIPTOR,
        CLASSIFY_DRUM_EVENTS_DESCRIPTOR,
        EXTRACT_CLASSIFIED_DRUMS_DESCRIPTOR,
    )
}
_ALIASES_TO_ACTION_ID: dict[str, str] = {
    alias: descriptor.action_id
    for descriptor in _DESCRIPTORS_BY_ID.values()
    for alias in descriptor.aliases
}


def descriptor_for_action(action_id: str) -> ActionDescriptor | None:
    canonical_id = canonical_action_id(action_id)
    if canonical_id is None:
        return None
    return _DESCRIPTORS_BY_ID.get(canonical_id)


def workflow_descriptor_for_action(action_id: str) -> ActionDescriptor | None:
    descriptor = descriptor_for_action(action_id)
    if descriptor is None or descriptor.workflow_id is None or descriptor.pipeline_template_id is None:
        return None
    return descriptor


def is_object_action(action_id: str) -> bool:
    descriptor = descriptor_for_action(action_id)
    return descriptor is not None and "object_action" in descriptor.groups


def action_descriptors() -> tuple[ActionDescriptor, ...]:
    return tuple(_DESCRIPTORS_BY_ID[action_id] for action_id in sorted(_DESCRIPTORS_BY_ID))


def object_action_descriptors() -> tuple[ActionDescriptor, ...]:
    return tuple(descriptor for descriptor in action_descriptors() if "object_action" in descriptor.groups)


def pipeline_actions_for_audio_layer(
    *,
    is_stem_capable: bool,
    is_drum_capable: bool,
    is_song_drum_capable: bool = False,
) -> tuple[ActionDescriptor, ...]:
    descriptors: list[ActionDescriptor] = []
    if is_stem_capable:
        descriptors.append(EXTRACT_STEMS_DESCRIPTOR)
    if is_song_drum_capable:
        descriptors.append(EXTRACT_SONG_DRUM_EVENTS_DESCRIPTOR)
    if is_drum_capable:
        descriptors.extend(
            (
                EXTRACT_CLASSIFIED_DRUMS_DESCRIPTOR,
                EXTRACT_DRUM_EVENTS_DESCRIPTOR,
            )
        )
    return tuple(descriptors)


def canonical_action_id(action_id: str) -> str | None:
    if action_id in _DESCRIPTORS_BY_ID:
        return action_id
    return _ALIASES_TO_ACTION_ID.get(action_id)

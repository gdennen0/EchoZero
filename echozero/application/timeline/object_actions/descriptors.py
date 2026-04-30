"""Canonical descriptors for timeline object actions and related operator actions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

_LOG = logging.getLogger(__name__)
_ALIAS_WARNED_IDS: set[str] = set()


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


@dataclass(slots=True, frozen=True)
class ActionAlias:
    """Compatibility alias metadata for transitional primitive migration."""

    alias_id: str
    canonical_id: str
    remove_after_release: str = "next_release"
    deprecation_note: str = "Legacy alias accepted for one release only."


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
EXTRACT_SONG_SECTIONS_DESCRIPTOR = ActionDescriptor(
    action_id="timeline.extract_song_sections",
    label="Extract Song Sections",
    object_types=("layer",),
    groups=("object_action", "tools"),
    params_schema={"layer_id": "required"},
    workflow_id="layer.audio.extract_song_sections",
    pipeline_template_id="extract_song_sections",
    binding_resolver_id="extract_song_sections",
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
        EXTRACT_SONG_SECTIONS_DESCRIPTOR,
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
_CANONICAL_NON_OBJECT_ACTION_IDS: set[str] = {
    "add_event_layer",
    "add_section_layer",
    "add_selection_to_main",
    "add_smpte_layer",
    "add_smpte_layer_from_import_split",
    "app.new",
    "app.open",
    "app.save",
    "app.save_as",
    "capture.screenshot",
    "delete_layer",
    "delete_take",
    "gain_down",
    "gain_unity",
    "gain_up",
    "import_smpte_audio_to_layer",
    "live_sync_clear_pause_reason",
    "live_sync_set_armed_write",
    "live_sync_set_observe",
    "live_sync_set_off",
    "live_sync_set_pause_reason",
    "merge_main",
    "overwrite_main",
    "preview_event_clip",
    "project.settings.set_ma3_push_offset",
    "seek_here",
    "set_layer_mute_off",
    "set_layer_mute_on",
    "set_layer_solo_off",
    "set_layer_solo_on",
    "selection.event",
    "selection.first_event",
    "selection.find_similar_sounding",
    "selection.layer",
    "selection.renumber_cues_from_one",
    "selection.select_every_other",
    "song.delete",
    "song.select",
    "song.version.add",
    "song.version.delete",
    "song.version.set_ma3_timecode_pool",
    "song.version.switch",
    "sync.disable",
    "sync.enable",
    "timeline.duplicate_selection",
    "timeline.nudge_selection",
    "transfer.plan_apply",
    "transfer.plan_cancel",
    "transfer.plan_preview",
    "transfer.route_layer_track",
    "transfer.send_selection",
    "transfer.send_to_track_once",
    "transfer.workspace_open",
    "transport.pause",
    "transport.play",
    "transport.stop",
}
_ALIASES: tuple[ActionAlias, ...] = (
    ActionAlias(alias_id="add_song_from_path", canonical_id="song.add"),
    ActionAlias(alias_id="extract_stems", canonical_id="timeline.extract_stems"),
    ActionAlias(alias_id="extract_drum_events", canonical_id="timeline.extract_drum_events"),
    ActionAlias(alias_id="classify_drum_events", canonical_id="timeline.classify_drum_events"),
    ActionAlias(
        alias_id="extract_classified_drums",
        canonical_id="timeline.extract_classified_drums",
    ),
    ActionAlias(alias_id="select_first_event", canonical_id="selection.first_event"),
    ActionAlias(alias_id="nudge", canonical_id="timeline.nudge_selection"),
    ActionAlias(alias_id="nudge_left", canonical_id="timeline.nudge_selection"),
    ActionAlias(alias_id="nudge_right", canonical_id="timeline.nudge_selection"),
    ActionAlias(alias_id="nudge_selected_events", canonical_id="timeline.nudge_selection"),
    ActionAlias(alias_id="duplicate", canonical_id="timeline.duplicate_selection"),
    ActionAlias(alias_id="duplicate_selected_events", canonical_id="timeline.duplicate_selection"),
    ActionAlias(alias_id="open_push_surface", canonical_id="transfer.workspace_open"),
    ActionAlias(alias_id="open_pull_surface", canonical_id="transfer.workspace_open"),
    ActionAlias(alias_id="pull_from_ma3", canonical_id="transfer.workspace_open"),
    ActionAlias(alias_id="push_to_ma3", canonical_id="transfer.workspace_open"),
    ActionAlias(alias_id="send_to_ma3", canonical_id="transfer.workspace_open"),
    ActionAlias(alias_id="send_layer_to_ma3", canonical_id="transfer.workspace_open"),
    ActionAlias(alias_id="apply_transfer_plan", canonical_id="transfer.plan_apply"),
    ActionAlias(alias_id="preview_transfer_plan", canonical_id="transfer.plan_preview"),
    ActionAlias(alias_id="cancel_transfer_plan", canonical_id="transfer.plan_cancel"),
    ActionAlias(alias_id="route_layer_to_ma3_track", canonical_id="transfer.route_layer_track"),
    ActionAlias(alias_id="send_selected_events_to_ma3", canonical_id="transfer.send_selection"),
    ActionAlias(alias_id="send_to_different_track_once", canonical_id="transfer.send_to_track_once"),
    ActionAlias(alias_id="enable_sync", canonical_id="sync.enable"),
    ActionAlias(alias_id="disable_sync", canonical_id="sync.disable"),
    ActionAlias(alias_id="screenshot", canonical_id="capture.screenshot"),
)
_ALIASES_BY_ID: dict[str, ActionAlias] = {alias.alias_id: alias for alias in _ALIASES}


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
        descriptors.extend(
            (
                EXTRACT_SONG_DRUM_EVENTS_DESCRIPTOR,
                EXTRACT_SONG_SECTIONS_DESCRIPTOR,
            )
        )
    if is_drum_capable:
        descriptors.extend(
            (
                EXTRACT_CLASSIFIED_DRUMS_DESCRIPTOR,
                EXTRACT_DRUM_EVENTS_DESCRIPTOR,
            )
        )
    return tuple(descriptors)


def canonical_action_id(action_id: str) -> str | None:
    return resolve_action_id(action_id, warn_on_alias=False)


def resolve_action_id(action_id: str, *, warn_on_alias: bool = False) -> str | None:
    if (
        action_id in _DESCRIPTORS_BY_ID
        or action_id in _CANONICAL_NON_OBJECT_ACTION_IDS
        or action_id.startswith("set_layer_output_bus_")
    ):
        return action_id
    if action_id in _ALIASES_TO_ACTION_ID:
        if warn_on_alias:
            _warn_deprecated_alias(action_id, _ALIASES_TO_ACTION_ID[action_id])
        return _ALIASES_TO_ACTION_ID[action_id]
    alias = _ALIASES_BY_ID.get(action_id)
    if alias is None:
        return None
    if warn_on_alias:
        _warn_deprecated_alias(
            alias.alias_id,
            alias.canonical_id,
            remove_after_release=alias.remove_after_release,
            note=alias.deprecation_note,
        )
    return alias.canonical_id


def canonical_action_ids() -> tuple[str, ...]:
    ids = set(_DESCRIPTORS_BY_ID)
    ids.update(_CANONICAL_NON_OBJECT_ACTION_IDS)
    return tuple(sorted(ids))


def action_aliases() -> tuple[ActionAlias, ...]:
    return _ALIASES


def _warn_deprecated_alias(
    alias_id: str,
    canonical_id: str,
    *,
    remove_after_release: str = "next_release",
    note: str = "Legacy alias accepted for one release only.",
) -> None:
    if alias_id in _ALIAS_WARNED_IDS:
        return
    _ALIAS_WARNED_IDS.add(alias_id)
    _LOG.warning(
        "Deprecated action alias '%s' resolved to '%s'. Remove after %s. %s",
        alias_id,
        canonical_id,
        remove_after_release,
        note,
    )

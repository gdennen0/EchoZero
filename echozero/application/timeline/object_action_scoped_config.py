"""Scoped config helpers for object-action settings.
Exists to isolate version-vs-song-default config lookup, persistence, and default hydration.
Connects object-action settings flows to typed ProjectStorage config records.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol, TypeAlias

from echozero.application.session.models import Session
from echozero.persistence.entities import PipelineConfigRecord, SongDefaultPipelineConfigRecord
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry
from echozero.result import Err
from echozero.services.orchestrator import Orchestrator

ObjectActionConfigRecord: TypeAlias = PipelineConfigRecord | SongDefaultPipelineConfigRecord


class ScopedConfigShell(Protocol):
    _analysis_service: Orchestrator

    @property
    def project_storage(self) -> ProjectStorage: ...

    @property
    def session(self) -> Session: ...

    def _require_active_song_id(self, action_name: str) -> str: ...

    def _require_active_song_version_id(self, action_name: str) -> str: ...

    def _resolve_object_action_object_bindings(
        self,
        action_id: str,
        *,
        layer_id: object,
        params: dict[str, object],
    ) -> dict[str, object]: ...

    def _coerce_object_action_runtime_params(
        self,
        action_id: str,
        *,
        params: dict[str, object],
    ) -> dict[str, object]: ...

    def _extract_classified_drums_model_defaults(self) -> dict[str, object]: ...


def load_scoped_action_config(
    shell: ScopedConfigShell,
    template_id: str,
    *,
    scope: str,
    song_id: str | None = None,
    song_version_id: str | None = None,
) -> ObjectActionConfigRecord:
    active_song_id = (
        str(shell.session.active_song_id) if shell.session.active_song_id is not None else None
    )
    active_song_version_id = (
        str(shell.session.active_song_version_id)
        if shell.session.active_song_version_id is not None
        else None
    )
    if scope == "song_default" and (song_id is None or song_id == active_song_id):
        return require_object_action_config(shell, template_id, scope=scope)
    if scope == "version" and (
        song_version_id is None or song_version_id == active_song_version_id
    ):
        return require_object_action_config(shell, template_id, scope=scope)
    return resolve_scoped_action_config(
        shell,
        template_id,
        scope=scope,
        song_id=song_id,
        song_version_id=song_version_id,
    )


def require_object_action_config(
    shell: ScopedConfigShell,
    template_id: str,
    *,
    scope: str = "version",
) -> ObjectActionConfigRecord:
    if scope == "song_default":
        song_id = shell._require_active_song_id(template_id)
        configs = shell.project_storage.song_default_pipeline_configs.list_by_song(song_id)
        match = next((config for config in configs if config.template_id == template_id), None)
        if match is not None:
            return hydrate_object_action_config_defaults(shell, match, scope=scope)
        song_version_id = shell._require_active_song_version_id(template_id)
        created = shell._analysis_service.create_config(
            shell.project_storage, song_version_id, template_id
        )
        if isinstance(created, Err):
            raise RuntimeError(
                f"Failed to create pipeline config for '{template_id}': {created.error}"
            )
        default_config = SongDefaultPipelineConfigRecord.from_version_config(
            created.value,
            song_id=song_id,
        )
        shell.project_storage.song_default_pipeline_configs.create(default_config)
        shell.project_storage.commit()
        return hydrate_object_action_config_defaults(shell, default_config, scope=scope)

    song_version_id = shell._require_active_song_version_id(template_id)
    version_configs = shell.project_storage.pipeline_configs.list_by_version(song_version_id)
    version_match = next(
        (config for config in version_configs if config.template_id == template_id), None
    )
    if version_match is not None:
        return hydrate_object_action_config_defaults(shell, version_match, scope=scope)
    created = shell._analysis_service.create_config(
        shell.project_storage, song_version_id, template_id
    )
    if isinstance(created, Err):
        raise RuntimeError(
            f"Failed to create pipeline config for '{template_id}': {created.error}"
        )
    return hydrate_object_action_config_defaults(shell, created.value, scope=scope)


def persist_object_action_params(
    shell: ScopedConfigShell,
    config: ObjectActionConfigRecord,
    *,
    action_id: str,
    pipeline_template_id: str,
    params: dict[str, object],
    scope: str = "version",
) -> ObjectActionConfigRecord:
    template = get_registry().get(pipeline_template_id)
    if template is None:
        raise ValueError(f"Pipeline template not found: {pipeline_template_id}")
    object_bindings = shell._resolve_object_action_object_bindings(
        action_id,
        layer_id=params.get("layer_id"),
        params=params,
    )
    runtime_params = shell._coerce_object_action_runtime_params(action_id, params=params)
    updates = {
        key: value
        for key, value in runtime_params.items()
        if key in template.knobs and key not in object_bindings
    }
    if not updates:
        return config
    updated = config.with_knob_values(updates, knob_metadata=template.knobs)
    store_scoped_action_config(shell, updated, scope=scope)
    return updated


def store_scoped_action_config(
    shell: ScopedConfigShell,
    config: ObjectActionConfigRecord,
    *,
    scope: str,
) -> None:
    if scope == "song_default":
        if not isinstance(config, SongDefaultPipelineConfigRecord):
            raise TypeError("song_default scope requires a SongDefaultPipelineConfigRecord.")
        shell.project_storage.song_default_pipeline_configs.update(config)
    else:
        if not isinstance(config, PipelineConfigRecord):
            raise TypeError("version scope requires a PipelineConfigRecord.")
        shell.project_storage.pipeline_configs.update(config)
    shell.project_storage.commit()


def resolve_scoped_action_config(
    shell: ScopedConfigShell,
    template_id: str,
    *,
    scope: str,
    song_id: str | None = None,
    song_version_id: str | None = None,
) -> ObjectActionConfigRecord:
    if scope == "song_default":
        resolved_song_id = song_id or shell._require_active_song_id(template_id)
        configs = shell.project_storage.song_default_pipeline_configs.list_by_song(
            resolved_song_id
        )
        match = next((config for config in configs if config.template_id == template_id), None)
        if match is None:
            raise ValueError(
                f"No song default settings found for '{template_id}' on song '{resolved_song_id}'."
            )
        return hydrate_object_action_config_defaults(shell, match, scope=scope)

    resolved_version_id = song_version_id or shell._require_active_song_version_id(template_id)
    version_configs = shell.project_storage.pipeline_configs.list_by_version(resolved_version_id)
    version_match = next(
        (config for config in version_configs if config.template_id == template_id), None
    )
    if version_match is None:
        raise ValueError(
            f"No version settings found for '{template_id}' on version '{resolved_version_id}'."
        )
    return hydrate_object_action_config_defaults(shell, version_match, scope=scope)


def hydrate_object_action_config_defaults(
    shell: ScopedConfigShell,
    config: ObjectActionConfigRecord,
    *,
    scope: str,
) -> ObjectActionConfigRecord:
    if config.template_id not in {"extract_classified_drums", "extract_song_drum_events"}:
        return config
    template = get_registry().get(config.template_id)
    if template is None:
        return config
    defaults = shell._extract_classified_drums_model_defaults()
    updates = {
        key: value
        for key, value in defaults.items()
        if _should_refresh_binary_model_default(
            current_value=config.knob_values.get(key),
            default_value=value,
        )
    }
    target_label_default = defaults.get("target_drum_labels")
    if "target_drum_labels" in template.knobs and _should_refresh_target_label_default(
        current_value=config.knob_values.get("target_drum_labels"),
        default_value=target_label_default,
    ):
        updates["target_drum_labels"] = target_label_default
    assignment_mode = str(config.knob_values.get("assignment_mode", "")).strip().lower()
    if "assignment_mode" in template.knobs and assignment_mode not in {
        "independent",
        "exclusive_max",
    }:
        updates["assignment_mode"] = "independent"

    graph_sync_updates = {
        key: value
        for key, value in config.knob_values.items()
        if key in template.knobs
    }
    graph_sync_updates.update(updates)
    if not graph_sync_updates:
        return config
    updated = config.with_knob_values(graph_sync_updates, knob_metadata=template.knobs)
    updated = _refresh_drum_event_template_graph_if_needed(updated, template=template)
    if updated.graph_json == config.graph_json and updated.knob_values == config.knob_values:
        return config
    store_scoped_action_config(shell, updated, scope=scope)
    return updated


def _refresh_drum_event_template_graph_if_needed(
    config: ObjectActionConfigRecord,
    *,
    template: object,
) -> ObjectActionConfigRecord:
    if config.template_id not in {"extract_classified_drums", "extract_song_drum_events"}:
        return config
    if not _drum_event_graph_needs_template_refresh(config):
        return config

    from echozero.serialization import serialize_graph

    knob_defaults = {
        key: knob.default
        for key, knob in getattr(template, "knobs", {}).items()
    }
    values = {**knob_defaults, **config.knob_values}
    pipeline = template.build_pipeline(values)
    outputs_json = json.dumps(
        [
            {
                "name": output.name,
                "block_id": output.port_ref.block_id,
                "port_name": output.port_ref.port_name,
            }
            for output in pipeline.outputs
        ]
    )
    return replace(
        config,
        graph_json=json.dumps(serialize_graph(pipeline.graph)),
        outputs_json=outputs_json,
        updated_at=datetime.now(timezone.utc),
    )


def _drum_event_graph_needs_template_refresh(config: ObjectActionConfigRecord) -> bool:
    try:
        graph_data = json.loads(config.graph_json)
    except (TypeError, ValueError):
        return True
    blocks = graph_data.get("blocks")
    connections = graph_data.get("connections")
    if not isinstance(blocks, list) or not isinstance(connections, list):
        return True
    block_ids = {str(block.get("id")) for block in blocks if isinstance(block, dict)}
    connection_inputs = {
        str(connection.get("target_input_name"))
        for connection in connections
        if isinstance(connection, dict)
        and str(connection.get("target_block_id")) == "classify_drums"
    }
    target_labels = _normalize_target_label_values(config.knob_values.get("target_drum_labels"))
    if "clap" in target_labels and (
        "clap_filter" not in block_ids
        or "clap_onsets" not in block_ids
        or "clap_events_in" not in connection_inputs
    ):
        return True
    if "cymbal" in target_labels and (
        "cymbal_filter" not in block_ids
        or "cymbal_onsets" not in block_ids
        or "cymbal_events_in" not in connection_inputs
    ):
        return True
    return False


def _should_refresh_binary_model_default(
    *,
    current_value: object,
    default_value: object,
) -> bool:
    current_text = str(current_value or "").strip()
    if current_text:
        return False
    return bool(str(default_value or "").strip())


def _should_refresh_target_label_default(
    *,
    current_value: object,
    default_value: object,
) -> bool:
    default_labels = _normalize_target_label_values(default_value)
    if not default_labels:
        return False
    current_labels = _normalize_target_label_values(current_value)
    if not current_labels:
        return True
    legacy_auto_defaults = {
        ("kick", "snare"),
        ("kick", "snare", "clap"),
    }
    return current_labels in legacy_auto_defaults and set(current_labels) < set(default_labels)


def _normalize_target_label_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values: tuple[object, ...] = tuple(value.split(","))
    elif isinstance(value, (list, tuple, set)):
        raw_values = tuple(value)
    else:
        raw_values = () if value is None else (value,)
    normalized: list[str] = []
    for raw_value in raw_values:
        label = str(raw_value or "").strip().lower()
        if label in {"symbol", "cymbol"}:
            label = "cymbal"
        if label and label not in normalized:
            normalized.append(label)
    return tuple(normalized)

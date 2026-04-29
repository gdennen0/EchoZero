"""Scoped config helpers for object-action settings.
Exists to isolate version-vs-song-default config lookup, persistence, and default hydration.
Connects object-action settings flows to typed ProjectStorage config records.
"""

from __future__ import annotations

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
    active_song_id = str(shell.session.active_song_id) if shell.session.active_song_id is not None else None
    active_song_version_id = (
        str(shell.session.active_song_version_id)
        if shell.session.active_song_version_id is not None
        else None
    )
    if scope == "song_default" and (song_id is None or song_id == active_song_id):
        return require_object_action_config(shell, template_id, scope=scope)
    if scope == "version" and (song_version_id is None or song_version_id == active_song_version_id):
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
        created = shell._analysis_service.create_config(shell.project_storage, song_version_id, template_id)
        if isinstance(created, Err):
            raise RuntimeError(f"Failed to create pipeline config for '{template_id}': {created.error}")
        default_config = SongDefaultPipelineConfigRecord.from_version_config(
            created.value,
            song_id=song_id,
        )
        shell.project_storage.song_default_pipeline_configs.create(default_config)
        shell.project_storage.commit()
        return hydrate_object_action_config_defaults(shell, default_config, scope=scope)

    song_version_id = shell._require_active_song_version_id(template_id)
    version_configs = shell.project_storage.pipeline_configs.list_by_version(song_version_id)
    version_match = next((config for config in version_configs if config.template_id == template_id), None)
    if version_match is not None:
        return hydrate_object_action_config_defaults(shell, version_match, scope=scope)
    created = shell._analysis_service.create_config(shell.project_storage, song_version_id, template_id)
    if isinstance(created, Err):
        raise RuntimeError(f"Failed to create pipeline config for '{template_id}': {created.error}")
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
        configs = shell.project_storage.song_default_pipeline_configs.list_by_song(resolved_song_id)
        match = next((config for config in configs if config.template_id == template_id), None)
        if match is None:
            raise ValueError(
                f"No song default settings found for '{template_id}' on song '{resolved_song_id}'."
            )
        return hydrate_object_action_config_defaults(shell, match, scope=scope)

    resolved_version_id = song_version_id or shell._require_active_song_version_id(template_id)
    version_configs = shell.project_storage.pipeline_configs.list_by_version(resolved_version_id)
    version_match = next((config for config in version_configs if config.template_id == template_id), None)
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
    assignment_mode = str(config.knob_values.get("assignment_mode", "")).strip().lower()
    if "assignment_mode" in template.knobs and assignment_mode not in {
        "independent",
        "exclusive_max",
    }:
        updates["assignment_mode"] = "independent"
    if not updates:
        return config
    updated = config.with_knob_values(updates, knob_metadata=template.knobs)
    store_scoped_action_config(shell, updated, scope=scope)
    return updated


def _should_refresh_binary_model_default(
    *,
    current_value: object,
    default_value: object,
) -> bool:
    current_text = str(current_value or "").strip()
    if current_text:
        return False
    return bool(str(default_value or "").strip())

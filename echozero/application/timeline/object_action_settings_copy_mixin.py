"""Copy-policy helpers for object-action settings sessions.
Exists to isolate scope selection, copy previews, and persisted-value loading from the core settings service.
Connects version-vs-song-default settings sessions to copy flows and session rebuilds.
"""

from __future__ import annotations

from typing import Protocol, cast

from echozero.application.session.models import Session
from echozero.application.timeline.object_action_scoped_config import ObjectActionConfigRecord
from echozero.application.timeline.object_actions.descriptors import ActionDescriptor
from echozero.application.timeline.object_actions.session import (
    ObjectActionCopySource,
    ObjectActionSettingsCopyPolicy,
    ObjectActionSettingsCopyPreview,
    ObjectActionSettingsScopeState,
    ObjectActionSettingsSession,
)
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry

class ObjectActionSettingsCopyShell(Protocol):
    @property
    def session(self) -> Session: ...

    @property
    def project_storage(self) -> ProjectStorage: ...

    def save(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> object: ...

    def _build_session(
        self,
        *,
        session_id: str,
        action_id: str,
        params: dict[str, object] | None,
        object_id: object | None,
        object_type: str | None,
        scope: str,
        drafts_by_scope: dict[str, dict[str, object]] | None = None,
        selected_copy_source_id: str | None = None,
        copy_preview: ObjectActionSettingsCopyPreview | None = None,
    ) -> ObjectActionSettingsSession: ...

    def _load_scoped_action_config(
        self,
        template_id: str,
        *,
        scope: str,
        song_id: str | None = None,
        song_version_id: str | None = None,
    ) -> ObjectActionConfigRecord: ...

    def _store_scoped_action_config(
        self,
        config: ObjectActionConfigRecord,
        *,
        scope: str,
    ) -> None: ...

    def _require_workflow(self, action_id: str) -> tuple[ActionDescriptor, str]: ...

    def _resolve_params(
        self,
        action_id: str,
        params: dict[str, object] | None,
        *,
        object_id: object | None,
        object_type: str | None,
    ) -> dict[str, object]: ...

class ObjectActionSettingsCopyMixin:
    def preview_copy(
        self: ObjectActionSettingsCopyShell,
        action_id: str,
        *,
        source_scope: str,
        target_scope: str,
        source_song_id: str | None = None,
        source_version_id: str | None = None,
        target_song_id: str | None = None,
        target_version_id: str | None = None,
        keys: list[str] | None = None,
    ) -> dict[str, object]:
        return preview_copy_between_scopes(
            self,
            action_id,
            source_scope=source_scope,
            target_scope=target_scope,
            source_song_id=source_song_id,
            source_version_id=source_version_id,
            target_song_id=target_song_id,
            target_version_id=target_version_id,
            keys=keys,
        )

    def apply_copy(
        self: ObjectActionSettingsCopyShell,
        action_id: str,
        *,
        source_scope: str,
        target_scope: str,
        source_song_id: str | None = None,
        source_version_id: str | None = None,
        target_song_id: str | None = None,
        target_version_id: str | None = None,
        keys: list[str] | None = None,
    ) -> dict[str, object]:
        return apply_copy_between_scopes(
            self,
            action_id,
            source_scope=source_scope,
            target_scope=target_scope,
            source_song_id=source_song_id,
            source_version_id=source_version_id,
            target_song_id=target_song_id,
            target_version_id=target_version_id,
            keys=keys,
        )

    def _build_copy_policy(
        self: ObjectActionSettingsCopyShell,
        scope: str,
        scope_states: tuple[ObjectActionSettingsScopeState, ...],
        *,
        selected_copy_source_id: str | None,
        copy_preview: ObjectActionSettingsCopyPreview | None,
    ) -> ObjectActionSettingsCopyPolicy:
        return build_copy_policy(
            self,
            scope,
            scope_states,
            selected_copy_source_id=selected_copy_source_id,
            copy_preview=copy_preview,
        )

    def _available_session_scopes(
        self: ObjectActionSettingsCopyShell,
    ) -> tuple[str, ...]:
        return available_session_scopes(self)

    @staticmethod
    def _scope_label(scope: str) -> str:
        return scope_label(scope)

    def _preview_copy_source(
        self: ObjectActionSettingsCopyShell,
        settings_session: ObjectActionSettingsSession,
        *,
        source_id: str,
    ) -> ObjectActionSettingsSession:
        return preview_copy_source(self, settings_session, source_id=source_id)

    def _apply_copy_source(
        self: ObjectActionSettingsCopyShell,
        settings_session: ObjectActionSettingsSession,
        *,
        source_id: str,
    ) -> ObjectActionSettingsSession:
        return apply_copy_source(self, settings_session, source_id=source_id)

    def _load_scope_persisted_values(
        self: ObjectActionSettingsCopyShell,
        action_id: str,
        *,
        scope: str,
        song_id: str | None = None,
        song_version_id: str | None = None,
    ) -> dict[str, object]:
        return load_scope_persisted_values(
            self,
            action_id,
            scope=scope,
            song_id=song_id,
            song_version_id=song_version_id,
        )

    def _session_object_params(
        self: ObjectActionSettingsCopyShell,
        settings_session: ObjectActionSettingsSession,
    ) -> dict[str, object]:
        return session_object_params(self, settings_session)

    @staticmethod
    def _require_copy_source(
        settings_session: ObjectActionSettingsSession,
        source_id: str,
    ) -> ObjectActionCopySource:
        return require_copy_source(settings_session, source_id)


def build_copy_policy(
    shell: ObjectActionSettingsCopyShell,
    scope: str,
    scope_states: tuple[ObjectActionSettingsScopeState, ...],
    *,
    selected_copy_source_id: str | None,
    copy_preview: ObjectActionSettingsCopyPreview | None,
) -> ObjectActionSettingsCopyPolicy:
    sources = discover_copy_sources(
        shell,
        scope=scope,
        available_scopes=tuple(state.scope for state in scope_states),
    )
    effective_selected_source_id = (
        selected_copy_source_id
        if any(source.source_id == selected_copy_source_id for source in sources)
        else None
    )
    effective_preview = (
        copy_preview
        if copy_preview is not None and copy_preview.source_id == effective_selected_source_id
        else None
    )
    return ObjectActionSettingsCopyPolicy(
        target_scope=scope,
        target_label=scope_label(scope),
        sources=sources,
        selected_source_id=effective_selected_source_id,
        preview=effective_preview,
    )


def available_session_scopes(shell: ObjectActionSettingsCopyShell) -> tuple[str, ...]:
    scopes: list[str] = []
    if shell.session.active_song_version_id is not None:
        scopes.append("version")
    if shell.session.active_song_id is not None:
        scopes.append("song_default")
    if not scopes:
        scopes.append("version")
    return tuple(scopes)


def scope_label(scope: str) -> str:
    return "Song Default" if scope == "song_default" else "This Version"


def discover_copy_sources(
    shell: ObjectActionSettingsCopyShell,
    *,
    scope: str,
    available_scopes: tuple[str, ...],
) -> tuple[ObjectActionCopySource, ...]:
    sources: list[ObjectActionCopySource] = []
    if scope == "version" and "song_default" in available_scopes:
        if shell.session.active_song_id is not None:
            sources.append(
                ObjectActionCopySource(
                    source_id="song_default",
                    label="Song Default",
                    scope="song_default",
                    song_id=str(shell.session.active_song_id),
                    description="Copy the saved song defaults into this version.",
                )
            )
    elif scope == "song_default" and "version" in available_scopes:
        if shell.session.active_song_version_id is not None:
            sources.append(
                ObjectActionCopySource(
                    source_id="this_version",
                    label="This Version",
                    scope="version",
                    version_id=str(shell.session.active_song_version_id),
                    description="Copy the current version's effective settings into the song defaults.",
                )
            )
    return tuple(sources)


def preview_copy_source(
    shell: ObjectActionSettingsCopyShell,
    settings_session: ObjectActionSettingsSession,
    *,
    source_id: str,
) -> ObjectActionSettingsSession:
    source = require_copy_source(settings_session, source_id)
    copy_preview = build_session_copy_preview(shell, settings_session, source)
    return shell._build_session(
        session_id=settings_session.session_id,
        action_id=settings_session.action_id,
        params=session_object_params(shell, settings_session),
        object_id=settings_session.object_id,
        object_type=settings_session.object_type,
        scope=settings_session.scope,
        drafts_by_scope={
            state.scope: state.draft_values
            for state in settings_session.scope_states
        },
        selected_copy_source_id=source_id,
        copy_preview=copy_preview,
    )


def apply_copy_source(
    shell: ObjectActionSettingsCopyShell,
    settings_session: ObjectActionSettingsSession,
    *,
    source_id: str,
) -> ObjectActionSettingsSession:
    source = require_copy_source(settings_session, source_id)
    source_values = load_scope_persisted_values(
        shell,
        settings_session.action_id,
        scope=source.scope,
        song_id=source.song_id,
        song_version_id=source.version_id,
    )
    merged_values = {
        **settings_session.current_scope_state.draft_values,
        **source_values,
    }
    object_params = session_object_params(shell, settings_session)
    shell.save(
        settings_session.action_id,
        {**object_params, **merged_values},
        object_id=settings_session.object_id,
        object_type=settings_session.object_type,
        scope=settings_session.scope,
    )
    drafts_by_scope = {
        state.scope: state.draft_values
        for state in settings_session.scope_states
    }
    drafts_by_scope[settings_session.scope] = merged_values
    return shell._build_session(
        session_id=settings_session.session_id,
        action_id=settings_session.action_id,
        params=object_params,
        object_id=settings_session.object_id,
        object_type=settings_session.object_type,
        scope=settings_session.scope,
        drafts_by_scope=drafts_by_scope,
        selected_copy_source_id=source_id,
    )


def build_session_copy_preview(
    shell: ObjectActionSettingsCopyShell,
    settings_session: ObjectActionSettingsSession,
    source: ObjectActionCopySource,
) -> ObjectActionSettingsCopyPreview:
    source_values = load_scope_persisted_values(
        shell,
        settings_session.action_id,
        scope=source.scope,
        song_id=source.song_id,
        song_version_id=source.version_id,
    )
    target_values = settings_session.current_scope_state.draft_values
    changes = []
    for key in sorted(set(source_values) | set(target_values)):
        current_value = target_values.get(key)
        source_value = source_values.get(key)
        if source_value == current_value:
            continue
        changes.append((key, current_value, source_value))
    return ObjectActionSettingsCopyPreview(
        source_id=source.source_id,
        summary=f"{source.label} -> {scope_label(settings_session.scope)}",
        changes=tuple(changes),
    )

def preview_copy_between_scopes(
    shell: ObjectActionSettingsCopyShell,
    action_id: str,
    *,
    source_scope: str,
    target_scope: str,
    source_song_id: str | None = None,
    source_version_id: str | None = None,
    target_song_id: str | None = None,
    target_version_id: str | None = None,
    keys: list[str] | None = None,
) -> dict[str, object]:
    with shell.project_storage.locked():
        _workflow, pipeline_template_id = shell._require_workflow(action_id)
        source = shell._load_scoped_action_config(
            pipeline_template_id,
            scope=source_scope,
            song_id=source_song_id,
            song_version_id=source_version_id,
        )
        target = shell._load_scoped_action_config(
            pipeline_template_id,
            scope=target_scope,
            song_id=target_song_id,
            song_version_id=target_version_id,
        )
        selected_keys = sorted(set(keys or source.knob_values.keys()))
        changes = []
        for key in selected_keys:
            source_value = source.knob_values.get(key)
            target_value = target.knob_values.get(key)
            if source_value == target_value:
                continue
            changes.append(
                {
                    "key": key,
                    "from": source_value,
                    "to": target_value,
                    "apply": source_value,
                }
            )
        return {
            "action_id": action_id,
            "template_id": pipeline_template_id,
            "source_scope": source_scope,
            "target_scope": target_scope,
            "changes": changes,
        }

def apply_copy_between_scopes(
    shell: ObjectActionSettingsCopyShell,
    action_id: str,
    *,
    source_scope: str,
    target_scope: str,
    source_song_id: str | None = None,
    source_version_id: str | None = None,
    target_song_id: str | None = None,
    target_version_id: str | None = None,
    keys: list[str] | None = None,
) -> dict[str, object]:
    with shell.project_storage.locked():
        preview = preview_copy_between_scopes(
            shell,
            action_id,
            source_scope=source_scope,
            target_scope=target_scope,
            source_song_id=source_song_id,
            source_version_id=source_version_id,
            target_song_id=target_song_id,
            target_version_id=target_version_id,
            keys=keys,
        )
        _workflow, pipeline_template_id = shell._require_workflow(action_id)
        target = shell._load_scoped_action_config(
            pipeline_template_id,
            scope=target_scope,
            song_id=target_song_id,
            song_version_id=target_version_id,
        )
        template = get_registry().get(pipeline_template_id)
        assert template is not None
        changes = cast(list[dict[str, object]], preview["changes"])
        updates = {str(item["key"]): item["apply"] for item in changes}
        if updates:
            updated = target.with_knob_values(updates, knob_metadata=template.knobs)
            shell._store_scoped_action_config(updated, scope=target_scope)
        return preview


def load_scope_persisted_values(
    shell: ObjectActionSettingsCopyShell,
    action_id: str,
    *,
    scope: str,
    song_id: str | None = None,
    song_version_id: str | None = None,
) -> dict[str, object]:
    _workflow, pipeline_template_id = shell._require_workflow(action_id)
    config = shell._load_scoped_action_config(
        pipeline_template_id,
        scope=scope,
        song_id=song_id,
        song_version_id=song_version_id,
    )
    return dict(config.knob_values)


def session_object_params(
    shell: ObjectActionSettingsCopyShell,
    settings_session: ObjectActionSettingsSession,
) -> dict[str, object]:
    return shell._resolve_params(
        settings_session.action_id,
        None,
        object_id=settings_session.object_id,
        object_type=settings_session.object_type,
    )


def require_copy_source(
    settings_session: ObjectActionSettingsSession,
    source_id: str,
) -> ObjectActionCopySource:
    match = next((source for source in settings_session.copy_sources if source.source_id == source_id), None)
    if match is None:
        raise ValueError(f"Unknown copy source '{source_id}'.")
    return match

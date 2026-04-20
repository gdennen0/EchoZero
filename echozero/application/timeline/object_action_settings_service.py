"""Application-level settings lane for timeline object actions.
Exists to keep object-action settings resolution and persistence out of Qt surfaces.
Connects pipeline templates, scoped config storage, copy semantics, and runtime bindings.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from echozero.application.timeline.object_actions.descriptors import workflow_descriptor_for_action
from echozero.application.timeline.object_actions.session import (
    ApplyCopySource,
    ChangeSessionScope,
    ObjectActionCopySource,
    ObjectActionSessionFieldValue,
    ObjectActionSettingsCopyPreview,
    ObjectActionSettingsCopyPolicy,
    ObjectActionSettingsScopeState,
    ObjectActionSettingsSession,
    PreviewCopySource,
    ReplaceSessionValues,
    RunSession,
    SaveAndRunSession,
    SaveSession,
    SetSessionFieldValue,
)
from echozero.application.timeline.object_actions.settings import (
    ObjectActionSettingField,
    ObjectActionSettingOption,
    ObjectActionSettingsPlan,
)
from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.session.models import Session
from echozero.application.shared.enums import LayerKind
from echozero.inference_eval.runtime_preflight import resolve_runtime_model_path
from echozero.persistence.entities import SongDefaultPipelineConfigRecord
from echozero.persistence.session import ProjectStorage
from echozero.models.paths import ensure_installed_models_dir
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles
from echozero.pipelines.params import KnobWidget
from echozero.pipelines.registry import get_registry
from echozero.result import is_err, unwrap
from echozero.runtime_models.bundle_compat import upgrade_installed_runtime_bundles
from echozero.services.orchestrator import AnalysisService


class ObjectActionSettingsService:
    """Own object-action settings routing, persistence, copy, and runtime binding resolution."""

    def __init__(
        self,
        *,
        project_storage_getter: Callable[[], ProjectStorage],
        session_getter: Callable[[], Session],
        presentation_getter: Callable[[], TimelinePresentation],
        require_layer: Callable[[object], LayerPresentation],
        analysis_service: AnalysisService,
    ) -> None:
        self._project_storage_getter = project_storage_getter
        self._session_getter = session_getter
        self._presentation_getter = presentation_getter
        self._require_layer = require_layer
        self._analysis_service = analysis_service
        self._settings_sessions: dict[str, ObjectActionSettingsSession] = {}

    def open_session(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsSession:
        session_id = f"object_action_session_{uuid.uuid4().hex[:12]}"
        settings_session = self._build_session(
            session_id=session_id,
            action_id=action_id,
            params=params,
            object_id=object_id,
            object_type=object_type,
            scope=scope,
        )
        self._settings_sessions[session_id] = settings_session
        return settings_session

    def dispatch_command(self, session_id: str, command: object) -> ObjectActionSettingsSession:
        settings_session = self._settings_sessions.get(session_id)
        if settings_session is None:
            raise ValueError(f"Unknown object-action settings session '{session_id}'.")

        if isinstance(command, SetSessionFieldValue):
            updated = self._update_session_field(settings_session, key=command.key, value=command.value)
        elif isinstance(command, ReplaceSessionValues):
            updated = self._replace_session_values(settings_session, values=command.values)
        elif isinstance(command, ChangeSessionScope):
            updated = self._rebuild_session(settings_session, scope=command.scope)
        elif isinstance(command, PreviewCopySource):
            updated = self._preview_copy_source(settings_session, source_id=command.source_id)
        elif isinstance(command, ApplyCopySource):
            updated = self._apply_copy_source(settings_session, source_id=command.source_id)
        elif isinstance(command, SaveSession):
            self._save_session_scope(settings_session)
            updated = self._rebuild_session(settings_session)
        elif isinstance(command, (RunSession, SaveAndRunSession)):
            self._save_and_run_session(settings_session)
            updated = self._rebuild_session(settings_session)
        else:
            raise ValueError(f"Unsupported object-action settings command '{type(command).__name__}'.")

        self._settings_sessions[session_id] = updated
        return updated

    def run(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        persist_scope: str | None = "version",
    ) -> None:
        workflow, resolved_params, layer_id = self._resolve_execution_context(
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
        )
        config = self._require_object_action_config(
            workflow.pipeline_template_id,
            scope=persist_scope or "version",
        )
        if persist_scope is not None:
            self._persist_object_action_params(
                config,
                action_id=action_id,
                params=resolved_params,
                scope=persist_scope,
            )
        self._execute_object_action(
            action_id,
            config.id,
            layer_id=layer_id,
            params=resolved_params,
        )

    def save(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsPlan:
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None:
            raise ValueError(f"Unsupported object action '{action_id}'.")
        resolved_params = self._resolve_params(action_id, params, object_id=object_id, object_type=object_type)
        config = self._require_object_action_config(workflow.pipeline_template_id, scope=scope)
        self._persist_object_action_params(config, action_id=action_id, params=resolved_params, scope=scope)
        return self.describe(
            action_id,
            resolved_params,
            object_id=object_id,
            object_type=object_type,
            scope=scope,
        )

    def describe(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsPlan:
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None:
            raise ValueError(f"Unsupported object action '{action_id}'.")
        if object_type is not None and object_type not in workflow.object_types:
            raise ValueError(
                f"{action_id} does not support object_type '{object_type}'. Expected one of {workflow.object_types}."
            )

        resolved_params = self._resolve_params(action_id, params, object_id=object_id, object_type=object_type)
        layer_id = resolved_params.get("layer_id")
        if "layer_id" in workflow.params_schema and layer_id is None:
            raise ValueError(f"{action_id} requires a target layer.")

        template = get_registry().get(workflow.pipeline_template_id)
        if template is None:
            raise ValueError(f"Pipeline template not found: {workflow.pipeline_template_id}")

        config = self._require_object_action_config(workflow.pipeline_template_id, scope=scope)
        defaults = {key: knob.default for key, knob in template.knobs.items()}
        persisted_values = dict(config.knob_values)
        object_bindings = self._resolve_object_action_object_bindings(action_id, layer_id=layer_id, params=resolved_params)
        editable_fields = self._build_object_action_setting_fields(
            action_id,
            defaults=defaults,
            persisted_values=persisted_values,
            object_bindings=object_bindings,
            params=resolved_params,
        )
        layer = self._require_layer(layer_id) if layer_id is not None else None
        has_prior_outputs = self._has_prior_outputs_for_action(
            pipeline_template_id=workflow.pipeline_template_id,
            source_layer_id=layer_id,
        )
        rerun_hint = ""
        if has_prior_outputs:
            rerun_hint = "Existing outputs detected. Run again as-is or tweak settings before creating another take."

        locked_bindings = tuple(
            (key, self._format_locked_binding_value(value))
            for key, value in sorted(object_bindings.items())
        )
        summary = layer.title if layer is not None else workflow.label
        return ObjectActionSettingsPlan(
            action_id=action_id,
            title=workflow.label,
            object_id=str(object_id if object_id is not None else layer_id or ""),
            object_type=object_type or (workflow.object_types[0] if workflow.object_types else "object"),
            pipeline_template_id=workflow.pipeline_template_id,
            editable_fields=tuple(field for field in editable_fields if not field.advanced),
            advanced_fields=tuple(field for field in editable_fields if field.advanced),
            locked_bindings=locked_bindings,
            has_prior_outputs=has_prior_outputs,
            run_label="Run Again" if has_prior_outputs else "Run",
            settings_label="Open Settings",
            rerun_hint=rerun_hint,
            summary=f"{summary} · {'Song Default' if scope == 'song_default' else 'This Version'}",
        )

    def preview_copy(
        self,
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
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None:
            raise ValueError(f"Unsupported object action '{action_id}'.")
        source = self._load_scoped_action_config(
            workflow.pipeline_template_id,
            scope=source_scope,
            song_id=source_song_id,
            song_version_id=source_version_id,
        )
        target = self._load_scoped_action_config(
            workflow.pipeline_template_id,
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
            changes.append({"key": key, "from": source_value, "to": target_value, "apply": source_value})
        return {
            "action_id": action_id,
            "template_id": workflow.pipeline_template_id,
            "source_scope": source_scope,
            "target_scope": target_scope,
            "changes": changes,
        }

    def apply_copy(
        self,
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
        preview = self.preview_copy(
            action_id,
            source_scope=source_scope,
            target_scope=target_scope,
            source_song_id=source_song_id,
            source_version_id=source_version_id,
            target_song_id=target_song_id,
            target_version_id=target_version_id,
            keys=keys,
        )
        workflow = workflow_descriptor_for_action(action_id)
        assert workflow is not None
        target = self._load_scoped_action_config(
            workflow.pipeline_template_id,
            scope=target_scope,
            song_id=target_song_id,
            song_version_id=target_version_id,
        )
        template = get_registry().get(workflow.pipeline_template_id)
        assert template is not None
        updates = {item["key"]: item["apply"] for item in preview["changes"]}
        if updates:
            updated = target.with_knob_values(updates, knob_metadata=template.knobs)
            self._store_scoped_action_config(updated, scope=target_scope)
        return preview

    @property
    def project_storage(self) -> ProjectStorage:
        return self._project_storage_getter()

    @property
    def session(self) -> Session:
        return self._session_getter()

    def presentation(self) -> TimelinePresentation:
        return self._presentation_getter()

    def _resolve_execution_context(
        self,
        action_id: str,
        params: dict[str, object] | None,
        *,
        object_id: object | None,
        object_type: str | None,
    ):
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None:
            raise ValueError(f"Unsupported object action '{action_id}'.")
        if object_type is not None and object_type not in workflow.object_types:
            raise ValueError(
                f"{action_id} does not support object_type '{object_type}'. Expected one of {workflow.object_types}."
            )
        resolved_params = self._resolve_params(action_id, params, object_id=object_id, object_type=object_type)
        layer_id = resolved_params.get("layer_id")
        if "layer_id" in workflow.params_schema and layer_id is None:
            raise ValueError(f"{action_id} requires a target layer.")
        return workflow, resolved_params, layer_id

    def _execute_object_action(
        self,
        action_id: str,
        config_id: str,
        *,
        layer_id,
        params: dict[str, object],
    ) -> None:
        runtime_bindings = self._resolve_object_action_runtime_bindings(
            action_id,
            layer_id=layer_id,
            params=params,
        )
        result = self._analysis_service.execute(
            self.project_storage,
            config_id,
            runtime_bindings=runtime_bindings,
        )
        if is_err(result):
            raise RuntimeError(f"{action_id} failed: {result.error}")

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
    ) -> ObjectActionSettingsSession:
        resolved_params = self._resolve_params(
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
        )
        resolved_object_params, initial_draft_values = self._split_session_params(
            action_id,
            resolved_params,
        )
        available_scopes = self._available_session_scopes()
        effective_scope = scope if scope in available_scopes else available_scopes[0]
        scope_states: list[ObjectActionSettingsScopeState] = []
        current_plan: ObjectActionSettingsPlan | None = None
        for candidate_scope in available_scopes:
            scope_state, plan = self._build_scope_state(
                action_id,
                object_id=object_id,
                object_type=object_type,
                scope=candidate_scope,
                object_params=resolved_object_params,
                draft_values={
                    **((drafts_by_scope or {}).get(candidate_scope, {})),
                    **(initial_draft_values if candidate_scope == effective_scope else {}),
                },
            )
            scope_states.append(scope_state)
            if candidate_scope == effective_scope:
                current_plan = plan
        assert current_plan is not None
        copy_policy = self._build_copy_policy(
            effective_scope,
            tuple(scope_states),
            selected_copy_source_id=selected_copy_source_id,
            copy_preview=copy_preview,
        )
        current_scope_state = next(state for state in scope_states if state.scope == effective_scope)
        workflow = workflow_descriptor_for_action(action_id)
        resolved_object_type = (
            object_type
            or (
                workflow.object_types[0]
                if workflow is not None and workflow.object_types
                else "object"
            )
        )
        return ObjectActionSettingsSession(
            session_id=session_id,
            action_id=action_id,
            object_id=str(object_id if object_id is not None else resolved_object_params.get("layer_id", "")),
            object_type=resolved_object_type,
            scope=effective_scope,
            plan=current_plan,
            scope_states=tuple(scope_states),
            copy_policy=copy_policy,
            can_save=True,
            can_save_and_run=current_scope_state.can_run,
            run_disabled_reason=""
            if current_scope_state.can_run
            else "Reruns use this version's effective settings. Switch to This Version to run.",
        )

    def _rebuild_session(
        self,
        settings_session: ObjectActionSettingsSession,
        *,
        scope: str | None = None,
        selected_copy_source_id: str | None = None,
        copy_preview: ObjectActionSettingsCopyPreview | None = None,
    ) -> ObjectActionSettingsSession:
        drafts_by_scope = {
            state.scope: state.draft_values
            for state in settings_session.scope_states
        }
        return self._build_session(
            session_id=settings_session.session_id,
            action_id=settings_session.action_id,
            params=self._session_object_params(settings_session),
            object_id=settings_session.object_id,
            object_type=settings_session.object_type,
            scope=scope or settings_session.scope,
            drafts_by_scope=drafts_by_scope,
            selected_copy_source_id=selected_copy_source_id,
            copy_preview=copy_preview,
        )

    def _build_scope_state(
        self,
        action_id: str,
        *,
        object_id: object | None,
        object_type: str | None,
        scope: str,
        object_params: dict[str, object],
        draft_values: dict[str, object],
    ) -> tuple[ObjectActionSettingsScopeState, ObjectActionSettingsPlan]:
        plan = self.describe(
            action_id,
            {**object_params, **draft_values},
            object_id=object_id,
            object_type=object_type,
            scope=scope,
        )
        field_values = tuple(
            ObjectActionSessionFieldValue(
                key=field.key,
                persisted_value=field.persisted_value,
                draft_value=field.value,
            )
            for field in (*plan.editable_fields, *plan.advanced_fields)
        )
        return (
            ObjectActionSettingsScopeState(
                scope=scope,
                label=self._scope_label(scope),
                field_values=field_values,
                can_run=scope == "version",
            ),
            plan,
        )

    def _build_copy_policy(
        self,
        scope: str,
        scope_states: tuple[ObjectActionSettingsScopeState, ...],
        *,
        selected_copy_source_id: str | None,
        copy_preview: ObjectActionSettingsCopyPreview | None,
    ) -> ObjectActionSettingsCopyPolicy:
        sources = self._discover_copy_sources(
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
            target_label=self._scope_label(scope),
            sources=sources,
            selected_source_id=effective_selected_source_id,
            preview=effective_preview,
        )

    def _available_session_scopes(self) -> tuple[str, ...]:
        scopes: list[str] = []
        if self.session.active_song_version_id is not None:
            scopes.append("version")
        if self.session.active_song_id is not None:
            scopes.append("song_default")
        if not scopes:
            scopes.append("version")
        return tuple(scopes)

    @staticmethod
    def _scope_label(scope: str) -> str:
        return "Song Default" if scope == "song_default" else "This Version"

    def _split_session_params(
        self,
        action_id: str,
        params: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None:
            return dict(params), {}
        template = get_registry().get(workflow.pipeline_template_id)
        if template is None:
            return dict(params), {}
        object_params = dict(params)
        draft_values: dict[str, object] = {}
        for key in tuple(object_params):
            if key in template.knobs:
                draft_values[key] = object_params.pop(key)
        return object_params, draft_values

    def _discover_copy_sources(
        self,
        *,
        scope: str,
        available_scopes: tuple[str, ...],
    ) -> tuple[ObjectActionCopySource, ...]:
        sources: list[ObjectActionCopySource] = []
        if scope == "version" and "song_default" in available_scopes:
            if self.session.active_song_id is not None:
                sources.append(
                    ObjectActionCopySource(
                        source_id="song_default",
                        label="Song Default",
                        scope="song_default",
                        song_id=str(self.session.active_song_id),
                        description="Copy the saved song defaults into this version.",
                    )
                )
        elif scope == "song_default" and "version" in available_scopes:
            if self.session.active_song_version_id is not None:
                sources.append(
                    ObjectActionCopySource(
                        source_id="this_version",
                        label="This Version",
                        scope="version",
                        version_id=str(self.session.active_song_version_id),
                        description="Copy the current version's effective settings into the song defaults.",
                    )
                )
        return tuple(sources)

    def _preview_copy_source(
        self,
        settings_session: ObjectActionSettingsSession,
        *,
        source_id: str,
    ) -> ObjectActionSettingsSession:
        source = self._require_copy_source(settings_session, source_id)
        copy_preview = self._build_session_copy_preview(settings_session, source)
        return self._rebuild_session(
            settings_session,
            selected_copy_source_id=source_id,
            copy_preview=copy_preview,
        )

    def _apply_copy_source(
        self,
        settings_session: ObjectActionSettingsSession,
        *,
        source_id: str,
    ) -> ObjectActionSettingsSession:
        source = self._require_copy_source(settings_session, source_id)
        source_values = self._load_scope_persisted_values(
            settings_session.action_id,
            scope=source.scope,
            song_id=source.song_id,
            song_version_id=source.version_id,
        )
        merged_values = {
            **settings_session.current_scope_state.draft_values,
            **source_values,
        }
        self.save(
            settings_session.action_id,
            {**self._session_object_params(settings_session), **merged_values},
            object_id=settings_session.object_id,
            object_type=settings_session.object_type,
            scope=settings_session.scope,
        )
        drafts_by_scope = {
            state.scope: state.draft_values
            for state in settings_session.scope_states
        }
        drafts_by_scope[settings_session.scope] = merged_values
        return self._build_session(
            session_id=settings_session.session_id,
            action_id=settings_session.action_id,
            params=self._session_object_params(settings_session),
            object_id=settings_session.object_id,
            object_type=settings_session.object_type,
            scope=settings_session.scope,
            drafts_by_scope=drafts_by_scope,
            selected_copy_source_id=source_id,
        )

    def _build_session_copy_preview(
        self,
        settings_session: ObjectActionSettingsSession,
        source: ObjectActionCopySource,
    ) -> ObjectActionSettingsCopyPreview:
        source_values = self._load_scope_persisted_values(
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
            summary=f"{source.label} -> {self._scope_label(settings_session.scope)}",
            changes=tuple(changes),
        )

    def _update_session_field(
        self,
        settings_session: ObjectActionSettingsSession,
        *,
        key: str,
        value: object,
    ) -> ObjectActionSettingsSession:
        current_values = settings_session.current_scope_state.draft_values
        if key not in current_values:
            raise ValueError(f"Unknown settings field '{key}'.")
        return self._replace_session_values(settings_session, values={key: value})

    def _replace_session_values(
        self,
        settings_session: ObjectActionSettingsSession,
        *,
        values: dict[str, object],
    ) -> ObjectActionSettingsSession:
        drafts_by_scope = {
            state.scope: state.draft_values
            for state in settings_session.scope_states
        }
        current_values = dict(drafts_by_scope.get(settings_session.scope, {}))
        unknown_keys = sorted(set(values) - set(current_values))
        if unknown_keys:
            raise ValueError(f"Unknown settings fields: {', '.join(unknown_keys)}")
        current_values.update(values)
        drafts_by_scope[settings_session.scope] = current_values
        return self._build_session(
            session_id=settings_session.session_id,
            action_id=settings_session.action_id,
            params=self._session_object_params(settings_session),
            object_id=settings_session.object_id,
            object_type=settings_session.object_type,
            scope=settings_session.scope,
            drafts_by_scope=drafts_by_scope,
        )

    def _save_session_scope(self, settings_session: ObjectActionSettingsSession) -> None:
        self.save(
            settings_session.action_id,
            {**self._session_object_params(settings_session), **settings_session.values},
            object_id=settings_session.object_id,
            object_type=settings_session.object_type,
            scope=settings_session.scope,
        )

    def _save_and_run_session(self, settings_session: ObjectActionSettingsSession) -> None:
        if settings_session.scope != "version":
            raise ValueError("Reruns use this version's effective settings. Switch to This Version to run.")
        self._save_session_scope(settings_session)
        self.run(
            settings_session.action_id,
            {**self._session_object_params(settings_session), **settings_session.values},
            object_id=settings_session.object_id,
            object_type=settings_session.object_type,
            persist_scope=None,
        )

    def _load_scope_persisted_values(
        self,
        action_id: str,
        *,
        scope: str,
        song_id: str | None = None,
        song_version_id: str | None = None,
    ) -> dict[str, object]:
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None:
            raise ValueError(f"Unsupported object action '{action_id}'.")
        config = self._load_scoped_action_config(
            workflow.pipeline_template_id,
            scope=scope,
            song_id=song_id,
            song_version_id=song_version_id,
        )
        return dict(config.knob_values)

    def _load_scoped_action_config(
        self,
        template_id: str,
        *,
        scope: str,
        song_id: str | None = None,
        song_version_id: str | None = None,
    ):
        active_song_id = str(self.session.active_song_id) if self.session.active_song_id is not None else None
        active_song_version_id = (
            str(self.session.active_song_version_id)
            if self.session.active_song_version_id is not None
            else None
        )
        if scope == "song_default" and (song_id is None or song_id == active_song_id):
            return self._require_object_action_config(template_id, scope=scope)
        if scope == "version" and (song_version_id is None or song_version_id == active_song_version_id):
            return self._require_object_action_config(template_id, scope=scope)
        return self._resolve_scoped_action_config(
            template_id,
            scope=scope,
            song_id=song_id,
            song_version_id=song_version_id,
        )

    def _session_object_params(self, settings_session: ObjectActionSettingsSession) -> dict[str, object]:
        return self._resolve_params(
            settings_session.action_id,
            None,
            object_id=settings_session.object_id,
            object_type=settings_session.object_type,
        )

    @staticmethod
    def _require_copy_source(
        settings_session: ObjectActionSettingsSession,
        source_id: str,
    ) -> ObjectActionCopySource:
        match = next((source for source in settings_session.copy_sources if source.source_id == source_id), None)
        if match is None:
            raise ValueError(f"Unknown copy source '{source_id}'.")
        return match

    @staticmethod
    def _resolve_params(
        action_id: str,
        params: dict[str, object] | None,
        *,
        object_id: object | None,
        object_type: str | None,
    ) -> dict[str, object]:
        resolved_params = dict(params or {})
        if object_type == "layer" and object_id is not None and "layer_id" not in resolved_params:
            resolved_params["layer_id"] = object_id
        return resolved_params

    def _resolve_object_action_runtime_bindings(
        self,
        action_id: str,
        *,
        layer_id,
        params: dict[str, object],
    ) -> dict[str, object]:
        bindings = self._resolve_object_action_object_bindings(action_id, layer_id=layer_id, params=params)
        bindings.pop("layer_id", None)
        return bindings

    def _resolve_object_action_object_bindings(
        self,
        action_id: str,
        *,
        layer_id,
        params: dict[str, object],
    ) -> dict[str, object]:
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None:
            raise ValueError(f"Unsupported object action '{action_id}'.")
        if workflow.binding_resolver_id is None:
            return {}
        resolver = self._object_action_binding_resolvers().get(workflow.binding_resolver_id)
        if resolver is None:
            raise ValueError(
                f"Unsupported object action binding resolver '{workflow.binding_resolver_id}' for '{action_id}'."
            )
        layer = self._require_layer(layer_id) if "layer_id" in workflow.params_schema else None
        return resolver(layer=layer, params=params)

    def _coerce_object_action_runtime_params(
        self,
        action_id: str,
        *,
        params: dict[str, object],
    ) -> dict[str, object]:
        workflow = workflow_descriptor_for_action(action_id)
        resolved = dict(params)
        if workflow is None or workflow.runtime_param_coercer_id is None:
            return resolved
        coercer = self._object_action_runtime_param_coercers().get(workflow.runtime_param_coercer_id)
        if coercer is None:
            raise ValueError(
                f"Unsupported object action runtime param coercer '{workflow.runtime_param_coercer_id}' for '{action_id}'."
            )
        return coercer(resolved)

    def _object_action_binding_resolvers(
        self,
    ) -> dict[str, Callable[..., dict[str, object]]]:
        return {
            "extract_stems": self._resolve_extract_stems_object_bindings,
            "extract_drum_events": self._resolve_extract_drum_events_object_bindings,
            "classify_drum_events": self._resolve_classify_drum_events_object_bindings,
            "extract_classified_drums": self._resolve_extract_classified_drums_object_bindings,
        }

    def _object_action_runtime_param_coercers(
        self,
    ) -> dict[str, Callable[[dict[str, object]], dict[str, object]]]:
        return {
            "classify_drum_events": self._coerce_classify_drum_events_runtime_params,
        }

    def _resolve_extract_stems_object_bindings(
        self,
        *,
        layer: LayerPresentation | None,
        params: dict[str, object],
    ) -> dict[str, object]:
        assert layer is not None
        return self._bindings_for_extract_stems(layer)

    def _resolve_extract_drum_events_object_bindings(
        self,
        *,
        layer: LayerPresentation | None,
        params: dict[str, object],
    ) -> dict[str, object]:
        assert layer is not None
        return self._bindings_for_extract_drum_events(layer)

    def _resolve_classify_drum_events_object_bindings(
        self,
        *,
        layer: LayerPresentation | None,
        params: dict[str, object],
    ) -> dict[str, object]:
        assert layer is not None
        return self._bindings_for_classify_drum_events(layer, params=params, include_runtime_overrides=False)

    def _resolve_extract_classified_drums_object_bindings(
        self,
        *,
        layer: LayerPresentation | None,
        params: dict[str, object],
    ) -> dict[str, object]:
        assert layer is not None
        return self._bindings_for_extract_classified_drums(layer)

    @staticmethod
    def _coerce_classify_drum_events_runtime_params(params: dict[str, object]) -> dict[str, object]:
        resolved = dict(params)
        model_path = resolved.pop("model_path", None)
        if model_path is not None and "classify_model_path" not in resolved:
            resolved["classify_model_path"] = model_path
        classify_model_path = resolved.get("classify_model_path")
        if classify_model_path is not None:
            resolved["classify_model_path"] = str(resolve_runtime_model_path(classify_model_path))
        return resolved

    def _build_object_action_setting_fields(
        self,
        action_id: str,
        *,
        defaults: dict[str, object],
        persisted_values: dict[str, object],
        object_bindings: dict[str, object],
        params: dict[str, object],
    ) -> tuple[ObjectActionSettingField, ...]:
        workflow = workflow_descriptor_for_action(action_id)
        assert workflow is not None
        template = get_registry().get(workflow.pipeline_template_id)
        assert template is not None
        resolved_params = self._coerce_object_action_runtime_params(action_id, params=params)
        fields: list[ObjectActionSettingField] = []
        for key, knob in template.knobs.items():
            if key in object_bindings:
                continue
            persisted_value = persisted_values.get(key, defaults.get(key, knob.default))
            value = resolved_params.get(key, persisted_value)
            fields.append(
                ObjectActionSettingField(
                    key=key,
                    label=knob.label or key.replace("_", " ").title(),
                    value=value,
                    default_value=knob.default,
                    persisted_value=persisted_value,
                    is_dirty=value != persisted_value,
                    widget=self._knob_widget_name(knob.widget),
                    description=knob.description,
                    advanced=knob.advanced,
                    placeholder=knob.placeholder,
                    units=knob.units,
                    min_value=knob.min_value,
                    max_value=knob.max_value,
                    step=knob.step,
                    options=tuple(
                        ObjectActionSettingOption(value=option, label=option.replace("_", " ").title())
                        for option in (knob.options or ())
                    ),
                )
            )
        return tuple(fields)

    def _require_object_action_config(self, template_id: str, *, scope: str = "version"):
        if scope == "song_default":
            song_id = self._require_active_song_id(template_id)
            configs = self.project_storage.song_default_pipeline_configs.list_by_song(song_id)
            match = next((config for config in configs if config.template_id == template_id), None)
            if match is not None:
                return match
            song_version_id = self._require_active_song_version_id(template_id)
            created = self._analysis_service.create_config(self.project_storage, song_version_id, template_id)
            if is_err(created):
                raise RuntimeError(f"Failed to create pipeline config for '{template_id}': {created.error}")
            default_config = SongDefaultPipelineConfigRecord.from_version_config(unwrap(created), song_id=song_id)
            self.project_storage.song_default_pipeline_configs.create(default_config)
            self.project_storage.commit()
            return default_config
        song_version_id = self._require_active_song_version_id(template_id)
        configs = self.project_storage.pipeline_configs.list_by_version(song_version_id)
        match = next((config for config in configs if config.template_id == template_id), None)
        if match is not None:
            return match
        created = self._analysis_service.create_config(self.project_storage, song_version_id, template_id)
        if is_err(created):
            raise RuntimeError(f"Failed to create pipeline config for '{template_id}': {created.error}")
        return unwrap(created)

    def _persist_object_action_params(self, config, *, action_id: str, params: dict[str, object], scope: str = "version"):
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None:
            raise ValueError(f"Unsupported object action '{action_id}'.")
        template = get_registry().get(workflow.pipeline_template_id)
        if template is None:
            raise ValueError(f"Pipeline template not found: {workflow.pipeline_template_id}")
        object_bindings = self._resolve_object_action_object_bindings(
            action_id,
            layer_id=params.get("layer_id"),
            params=params,
        )
        runtime_params = self._coerce_object_action_runtime_params(action_id, params=params)
        updates = {
            key: value
            for key, value in runtime_params.items()
            if key in template.knobs and key not in object_bindings
        }
        if updates:
            config = config.with_knob_values(updates, knob_metadata=template.knobs)
            self._store_scoped_action_config(config, scope=scope)
        return config

    def _store_scoped_action_config(self, config, *, scope: str) -> None:
        if scope == "song_default":
            self.project_storage.song_default_pipeline_configs.update(config)
        else:
            self.project_storage.pipeline_configs.update(config)
        self.project_storage.commit()

    def _resolve_scoped_action_config(
        self,
        template_id: str,
        *,
        scope: str,
        song_id: str | None = None,
        song_version_id: str | None = None,
    ):
        if scope == "song_default":
            resolved_song_id = song_id or self._require_active_song_id(template_id)
            configs = self.project_storage.song_default_pipeline_configs.list_by_song(resolved_song_id)
            match = next((config for config in configs if config.template_id == template_id), None)
            if match is None:
                raise ValueError(f"No song default settings found for '{template_id}' on song '{resolved_song_id}'.")
            return match
        resolved_version_id = song_version_id or self._require_active_song_version_id(template_id)
        configs = self.project_storage.pipeline_configs.list_by_version(resolved_version_id)
        match = next((config for config in configs if config.template_id == template_id), None)
        if match is None:
            raise ValueError(f"No version settings found for '{template_id}' on version '{resolved_version_id}'.")
        return match

    def _require_active_song_id(self, action_name: str) -> str:
        if self.session.active_song_id is None:
            raise RuntimeError(f"{action_name} requires an active song.")
        return str(self.session.active_song_id)

    def _require_active_song_version_id(self, action_name: str) -> str:
        if self.session.active_song_version_id is None:
            raise RuntimeError(f"{action_name} requires an active song version.")
        return str(self.session.active_song_version_id)

    @staticmethod
    def _knob_widget_name(widget: KnobWidget) -> str:
        mapping = {
            KnobWidget.TOGGLE: "toggle",
            KnobWidget.DROPDOWN: "dropdown",
            KnobWidget.FILE_PICKER: "file",
            KnobWidget.MODEL_PICKER: "file",
            KnobWidget.SLIDER: "number",
            KnobWidget.NUMBER: "number",
            KnobWidget.FREQUENCY: "number",
            KnobWidget.GAIN: "number",
        }
        return mapping.get(widget, "text")

    def _has_prior_outputs_for_action(self, *, pipeline_template_id: str, source_layer_id) -> bool:
        if source_layer_id is None:
            return False
        return any(
            layer.status.pipeline_id == pipeline_template_id
            and (
                str(layer.status.source_layer_id) == str(source_layer_id)
                or str(source_layer_id) == "source_audio"
            )
            for layer in self.presentation().layers
        )

    @staticmethod
    def _format_locked_binding_value(value: object) -> str:
        text = str(value)
        return text if len(text) <= 72 else f"{text[:69]}..."

    def _bindings_for_extract_stems(self, layer: LayerPresentation) -> dict[str, object]:
        if layer.kind is not LayerKind.AUDIO:
            raise ValueError(f"timeline.extract_stems requires an audio layer, got {layer.kind.name.lower()}.")
        if str(layer.layer_id) != "source_audio":
            raise NotImplementedError(
                "timeline.extract_stems currently runs only from the imported song layer. "
                "Derived-audio reruns are deferred until arbitrary-layer pipeline input is wired."
            )
        if not layer.source_audio_path:
            raise ValueError("timeline.extract_stems requires a resolved source audio path.")
        return {"audio_file": layer.source_audio_path}

    def _bindings_for_extract_drum_events(self, layer: LayerPresentation) -> dict[str, object]:
        self._validate_drum_derived_audio_layer(layer, action_name="timeline.extract_drum_events")
        return {"audio_file": layer.source_audio_path}

    def _bindings_for_classify_drum_events(
        self,
        layer: LayerPresentation,
        *,
        params: dict[str, object],
        include_runtime_overrides: bool = True,
    ) -> dict[str, object]:
        self._validate_drum_derived_audio_layer(layer, action_name="timeline.classify_drum_events")
        bindings = {"audio_file": layer.source_audio_path}
        if not include_runtime_overrides:
            return bindings
        model_path = params.get("classify_model_path", params.get("model_path"))
        resolved_model_path = resolve_runtime_model_path(model_path)
        if not str(resolved_model_path).strip():
            raise ValueError("timeline.classify_drum_events requires a non-empty model path.")
        bindings["classify_model_path"] = str(resolved_model_path)
        return bindings

    def _bindings_for_extract_classified_drums(self, layer: LayerPresentation) -> dict[str, object]:
        self._validate_drum_derived_audio_layer(layer, action_name="timeline.extract_classified_drums")
        upgrade_installed_runtime_bundles(ensure_installed_models_dir())
        bundles = resolve_installed_binary_drum_bundles()
        return {
            "audio_file": layer.source_audio_path,
            "kick_model_path": str(bundles["kick"].manifest_path),
            "snare_model_path": str(bundles["snare"].manifest_path),
        }

    @staticmethod
    def _validate_drum_derived_audio_layer(layer: LayerPresentation, *, action_name: str) -> None:
        if layer.kind is not LayerKind.AUDIO:
            raise ValueError(
                f"{action_name} requires an audio layer, got {layer.kind.name.lower()}."
            )
        if not layer.source_audio_path:
            raise RuntimeError(f"{action_name} requires a source audio path on the selected layer.")

        title_lower = layer.title.lower()
        source_label = (layer.status.source_label if layer.status is not None else "")
        source_label_lower = source_label.lower()
        badges = {str(badge).strip().lower() for badge in layer.badges}
        if "drum" not in title_lower and "drums" not in badges and "drum" not in source_label_lower:
            raise NotImplementedError(
                f"{action_name} currently runs only from drum-derived audio layers. "
                "Select a drums layer produced by stem separation."
            )

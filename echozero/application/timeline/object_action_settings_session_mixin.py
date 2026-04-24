"""Session lifecycle helpers for object-action settings.
Exists to keep scoped draft mutation, scope switching, and copy-session orchestration out of the core settings service root.
Connects settings plans, copy policy, and run eligibility into one bounded session-editing seam.
"""

from __future__ import annotations

import uuid

from echozero.application.session.models import Session
from echozero.application.timeline.object_action_scoped_config import (
    ObjectActionConfigRecord,
)
from echozero.application.timeline.object_action_settings_copy_mixin import (
    ObjectActionSettingsCopyMixin,
)
from echozero.application.timeline.object_actions.descriptors import (
    ActionDescriptor,
    workflow_descriptor_for_action,
)
from echozero.application.timeline.object_actions.session import (
    ApplyCopySource,
    ChangeSessionScope,
    ObjectActionSessionFieldValue,
    ObjectActionSettingsCopyPreview,
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
    ObjectActionSettingsPlan,
)
from echozero.application.timeline.pipeline_run_service import (
    PipelineRunService,
    PipelineRunState,
)
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry


class ObjectActionSettingsSessionMixin(ObjectActionSettingsCopyMixin):
    """Owns object-action settings session state, draft mutation, and scope flow."""

    _settings_sessions: dict[str, ObjectActionSettingsSession]

    @property
    def project_storage(self) -> ProjectStorage:
        raise NotImplementedError

    @property
    def session(self) -> Session:
        raise NotImplementedError

    def describe(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsPlan:
        raise NotImplementedError

    def run(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        persist_scope: str | None = "version",
    ) -> None:
        raise NotImplementedError

    def save(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsPlan:
        raise NotImplementedError

    def _lookup_active_run(
        self,
        action_id: str,
        *,
        object_id: object | None,
        object_type: str | None,
    ) -> PipelineRunState | None:
        raise NotImplementedError

    @staticmethod
    def _resolve_params(
        action_id: str,
        params: dict[str, object] | None,
        *,
        object_id: object | None,
        object_type: str | None,
    ) -> dict[str, object]:
        raise NotImplementedError

    def _load_scoped_action_config(
        self,
        template_id: str,
        *,
        scope: str,
        song_id: str | None = None,
        song_version_id: str | None = None,
    ) -> ObjectActionConfigRecord:
        raise NotImplementedError

    def _store_scoped_action_config(
        self,
        config: ObjectActionConfigRecord,
        *,
        scope: str,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def _require_workflow(action_id: str) -> tuple[ActionDescriptor, str]:
        raise NotImplementedError

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
        with self.project_storage.locked():
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

    def dispatch_command(
        self,
        session_id: str,
        command: object,
    ) -> ObjectActionSettingsSession:
        settings_session = self._settings_sessions.get(session_id)
        if settings_session is None:
            raise ValueError(f"Unknown object-action settings session '{session_id}'.")

        with self.project_storage.locked():
            if isinstance(command, SetSessionFieldValue):
                updated = self._update_session_field(
                    settings_session,
                    key=command.key,
                    value=command.value,
                )
            elif isinstance(command, ReplaceSessionValues):
                updated = self._replace_session_values(
                    settings_session,
                    values=command.values,
                )
            elif isinstance(command, ChangeSessionScope):
                updated = self._rebuild_session(settings_session, scope=command.scope)
            elif isinstance(command, PreviewCopySource):
                updated = self._preview_copy_source(
                    settings_session,
                    source_id=command.source_id,
                )
            elif isinstance(command, ApplyCopySource):
                updated = self._apply_copy_source(
                    settings_session,
                    source_id=command.source_id,
                )
            elif isinstance(command, SaveSession):
                self._save_session_scope(settings_session)
                updated = self._rebuild_session(settings_session)
            elif isinstance(command, (RunSession, SaveAndRunSession)):
                self._save_and_run_session(settings_session)
                updated = self._rebuild_session(settings_session)
            else:
                raise ValueError(
                    f"Unsupported object-action settings command '{type(command).__name__}'."
                )

        self._settings_sessions[session_id] = updated
        return updated

    def refresh_session(
        self,
        session_id: str,
    ) -> ObjectActionSettingsSession:
        settings_session = self._settings_sessions.get(session_id)
        if settings_session is None:
            raise ValueError(f"Unknown object-action settings session '{session_id}'.")
        with self.project_storage.locked():
            refreshed = self._rebuild_session(settings_session)
        self._settings_sessions[session_id] = refreshed
        return refreshed

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
        current_scope_state = next(
            state for state in scope_states if state.scope == effective_scope
        )
        workflow = workflow_descriptor_for_action(action_id)
        resolved_object_type = (
            object_type
            or (
                workflow.object_types[0]
                if workflow is not None and workflow.object_types
                else "object"
            )
        )
        active_run = self._lookup_active_run(
            action_id,
            object_id=(
                object_id
                if object_id is not None
                else resolved_object_params.get("layer_id", "")
            ),
            object_type=resolved_object_type,
        )
        run_is_active = PipelineRunService.is_active(active_run)
        return ObjectActionSettingsSession(
            session_id=session_id,
            action_id=action_id,
            object_id=str(
                object_id
                if object_id is not None
                else resolved_object_params.get("layer_id", "")
            ),
            object_type=resolved_object_type,
            scope=effective_scope,
            plan=current_plan,
            scope_states=tuple(scope_states),
            copy_policy=copy_policy,
            can_save=True,
            can_save_and_run=current_scope_state.can_run and not run_is_active,
            run_disabled_reason=(
                "This stage is already running."
                if run_is_active
                else (
                    ""
                    if current_scope_state.can_run
                    else "Reruns use this version's effective settings. Switch to This Version to run."
                )
            ),
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
            state.scope: state.draft_values for state in settings_session.scope_states
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

    def _split_session_params(
        self,
        action_id: str,
        params: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None or workflow.pipeline_template_id is None:
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
            state.scope: state.draft_values for state in settings_session.scope_states
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

    def _save_session_scope(
        self,
        settings_session: ObjectActionSettingsSession,
    ) -> None:
        self.save(
            settings_session.action_id,
            {**self._session_object_params(settings_session), **settings_session.values},
            object_id=settings_session.object_id,
            object_type=settings_session.object_type,
            scope=settings_session.scope,
        )

    def _save_and_run_session(
        self,
        settings_session: ObjectActionSettingsSession,
    ) -> None:
        if settings_session.scope != "version":
            raise ValueError(
                "Reruns use this version's effective settings. Switch to This Version to run."
            )
        self._save_session_scope(settings_session)
        self.run(
            settings_session.action_id,
            {**self._session_object_params(settings_session), **settings_session.values},
            object_id=settings_session.object_id,
            object_type=settings_session.object_type,
            persist_scope=None,
        )

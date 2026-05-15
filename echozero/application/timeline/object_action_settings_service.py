"""Application-level execution lane for timeline object actions.
Exists to keep object-action settings resolution, persistence, and execution out of Qt surfaces.
Connects pipeline templates, scoped config storage, and session helpers through one explicit owner.
"""

from __future__ import annotations

from collections.abc import Callable

from echozero.application.settings import AppSettingsService
from echozero.application.timeline.object_action_scoped_config import (
    ProjectPipelineDefaultsSyncResult,
    apply_app_defaults_to_project,
)
from echozero.application.timeline.object_action_settings_session_mixin import (
    ObjectActionSettingsSessionMixin,
)
from echozero.application.timeline.object_action_settings_context_mixin import (
    ObjectActionSettingsContextMixin,
)
from echozero.application.timeline.object_action_settings_persistence_mixin import (
    ObjectActionSettingsPersistenceMixin,
)
from echozero.application.timeline.object_action_settings_plan_mixin import (
    ObjectActionSettingsPlanMixin,
)
from echozero.application.timeline.object_action_settings_runtime_mixin import (
    ObjectActionSettingsRuntimeMixin,
)
from echozero.application.timeline.object_actions.session import ObjectActionSettingsSession
from echozero.application.timeline.object_actions.settings import ObjectActionSettingsPlan
from echozero.application.timeline.operation_progress_service import (
    OperationProgressState,
    PreparedOperation,
)
from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.session.models import Session
from echozero.inference_eval.runtime_preflight import resolve_runtime_model_path  # noqa: F401
from echozero.models.paths import ensure_installed_models_dir  # noqa: F401
from echozero.models.runtime_bundle_selection import (
    resolve_installed_binary_drum_bundles,
)  # noqa: F401
from echozero.persistence.session import ProjectStorage
from echozero.result import Err, unwrap
from echozero.runtime_models.bundle_compat import upgrade_installed_runtime_bundles  # noqa: F401
from echozero.services.orchestrator import Orchestrator

__all__ = [
    "ObjectActionExecutionService",
    "ObjectActionSettingsService",
    "ensure_installed_models_dir",
    "resolve_installed_binary_drum_bundles",
    "resolve_runtime_model_path",
    "upgrade_installed_runtime_bundles",
]


class ObjectActionExecutionService(
    ObjectActionSettingsContextMixin,
    ObjectActionSettingsPersistenceMixin,
    ObjectActionSettingsPlanMixin,
    ObjectActionSettingsRuntimeMixin,
    ObjectActionSettingsSessionMixin,
):
    """Own object-action settings routing, persistence, and execution."""

    def __init__(
        self,
        *,
        project_storage_getter: Callable[[], ProjectStorage],
        session_getter: Callable[[], Session],
        presentation_getter: Callable[[], TimelinePresentation],
        require_layer: Callable[[object], LayerPresentation],
        analysis_service: Orchestrator,
        app_settings_service: AppSettingsService | None = None,
        active_run_lookup: (
            Callable[[str, object | None, str | None], OperationProgressState | None] | None
        ) = None,
    ) -> None:
        self._project_storage_getter = project_storage_getter
        self._session_getter = session_getter
        self._presentation_getter = presentation_getter
        self._require_layer = require_layer
        self._analysis_service = analysis_service
        self._app_settings_service = app_settings_service
        self._active_run_lookup = active_run_lookup
        self._settings_sessions: dict[str, ObjectActionSettingsSession] = {}

    def run(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        persist_scope: str | None = "version",
    ) -> None:
        prepared = self.prepare_run(
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
            persist_scope=persist_scope,
        )
        result = self._analysis_service.execute(
            self.project_storage,
            prepared.config_id,
            runtime_bindings=prepared.runtime_bindings,
        )
        if isinstance(result, Err):
            raise RuntimeError(f"{action_id} failed: {result.error}")
        self.persist_generated_source_layer_id(
            analysis_result=unwrap(result),
            source_layer_id=prepared.source_layer_id,
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
        with self.project_storage.locked():
            scope = self._normalize_edit_scope(scope)
            workflow, pipeline_template_id = self._require_workflow(action_id)
            resolved_params = self._resolve_params(
                action_id, params, object_id=object_id, object_type=object_type
            )
            config = self._require_object_action_config(pipeline_template_id, scope=scope)
            updated_config = self._persist_object_action_params(
                config,
                action_id=action_id,
                params=resolved_params,
                scope=scope,
            )
            if updated_config is not config:
                self._mark_scope_persist_dirty(scope=scope, config=updated_config)
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
        with self.project_storage.locked():
            return self._describe_settings_plan(
                action_id,
                params,
                object_id=object_id,
                object_type=object_type,
                scope=self._normalize_edit_scope(scope),
            )

    @property
    def project_storage(self) -> ProjectStorage:
        return self._project_storage_getter()

    @property
    def session(self) -> Session:
        return self._session_getter()

    def presentation(self) -> TimelinePresentation:
        return self._presentation_getter()

    def prepare_run(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        persist_scope: str | None = "version",
    ) -> PreparedOperation:
        """Resolve and optionally persist one object-action run without executing it."""

        with self.project_storage.locked():
            workflow, pipeline_template_id, resolved_params, layer_id = (
                self._resolve_execution_context(
                    action_id,
                    params,
                    object_id=object_id,
                    object_type=object_type,
                )
            )
            if layer_id is not None:
                layer = self._require_layer(layer_id)
                layer_id = layer.layer_id
                resolved_params = {**resolved_params, "layer_id": layer_id}
            config = self._require_object_action_config(
                pipeline_template_id,
                scope=persist_scope or "version",
            )
            if persist_scope is not None:
                prior = config
                config = self._persist_object_action_params(
                    config,
                    action_id=action_id,
                    params=resolved_params,
                    scope=persist_scope,
                )
                if config is not prior:
                    self._mark_scope_persist_dirty(scope=persist_scope, config=config)
            workflow_id = workflow.workflow_id
            if workflow_id is None:
                raise ValueError(f"Unsupported object action '{action_id}'.")
            return PreparedOperation(
                action_id=action_id,
                workflow_id=workflow_id,
                pipeline_template_id=pipeline_template_id,
                config_id=config.id,
                display_label=workflow.label,
                object_id=str(object_id if object_id is not None else layer_id or ""),
                object_type=object_type
                or (workflow.object_types[0] if workflow.object_types else "object"),
                source_layer_id=str(layer_id) if layer_id is not None else None,
                song_id=(
                    str(self.session.active_song_id)
                    if self.session.active_song_id is not None
                    else None
                ),
                song_version_id=(
                    str(self.session.active_song_version_id)
                    if self.session.active_song_version_id is not None
                    else None
                ),
                runtime_bindings=self._resolve_object_action_runtime_bindings(
                    action_id,
                    layer_id=layer_id,
                    params=resolved_params,
                ),
            )

    def _execute_object_action(
        self,
        action_id: str,
        config_id: str,
        *,
        layer_id: object | None,
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
        if isinstance(result, Err):
            raise RuntimeError(f"{action_id} failed: {result.error}")
        self.persist_generated_source_layer_id(
            analysis_result=unwrap(result),
            source_layer_id=layer_id,
        )

    def _normalize_edit_scope(self, scope: str) -> str:
        if (
            scope == "version"
            and self.session.active_song_version_id is None
            and self.session.active_song_id is not None
        ):
            return "song_default"
        if (
            scope == "version"
            and self.session.active_song_version_id is None
            and self.session.active_song_id is None
            and self._app_settings_service is not None
        ):
            return "app_default"
        return scope

    def _can_edit_app_defaults(self) -> bool:
        return self._app_settings_service is not None

    def _load_app_pipeline_defaults(self, template_id: str) -> dict[str, object]:
        if self._app_settings_service is None:
            return {}
        return self._app_settings_service.pipeline_defaults_for_template(template_id)

    def _store_app_pipeline_defaults(
        self,
        template_id: str,
        values: dict[str, object],
    ) -> None:
        if self._app_settings_service is None:
            raise RuntimeError("Application defaults are unavailable in this runtime.")
        self._app_settings_service.replace_pipeline_defaults(template_id, values)

    def apply_app_defaults_to_project(
        self,
        *,
        template_ids: tuple[str, ...] | None = None,
    ) -> ProjectPipelineDefaultsSyncResult:
        with self.project_storage.locked():
            return apply_app_defaults_to_project(self, template_ids=template_ids)


ObjectActionSettingsService = ObjectActionExecutionService

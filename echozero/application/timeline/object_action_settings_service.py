"""Application-level settings lane for timeline object actions.
Exists to keep object-action settings resolution and persistence out of Qt surfaces.
Connects pipeline templates, scoped config storage, and extracted copy/runtime helper slices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from echozero.application.timeline.object_action_settings_session_mixin import (
    ObjectActionSettingsSessionMixin,
)
from echozero.application.timeline.object_actions.descriptors import ActionDescriptor, workflow_descriptor_for_action
from echozero.application.timeline.object_action_scoped_config import (
    ObjectActionConfigRecord,
    load_scoped_action_config as _load_scoped_action_config,
)
from echozero.application.timeline.object_action_scoped_config import (
    persist_object_action_params as _persist_object_action_params,
)
from echozero.application.timeline.object_action_scoped_config import (
    require_object_action_config as _require_object_action_config,
)
from echozero.application.timeline.object_action_scoped_config import (
    store_scoped_action_config as _store_scoped_action_config,
)
from echozero.application.timeline.object_action_settings_runtime_mixin import ObjectActionSettingsRuntimeMixin
from echozero.application.timeline.object_actions.session import ObjectActionSettingsSession
from echozero.application.timeline.object_actions.settings import ObjectActionSettingsPlan
from echozero.application.timeline.pipeline_run_service import (
    PipelineRunService,
    PipelineRunState,
    PreparedPipelineRun,
)
from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.session.models import Session
from echozero.inference_eval.runtime_preflight import resolve_runtime_model_path  # noqa: F401
from echozero.models.paths import ensure_installed_models_dir  # noqa: F401
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles  # noqa: F401
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import get_registry
from echozero.result import Err, unwrap
from echozero.runtime_models.bundle_compat import upgrade_installed_runtime_bundles  # noqa: F401
from echozero.services.orchestrator import AnalysisResult, AnalysisService

__all__ = [
    "ObjectActionSettingsService",
    "ensure_installed_models_dir",
    "resolve_installed_binary_drum_bundles",
    "resolve_runtime_model_path",
    "upgrade_installed_runtime_bundles",
]


class ObjectActionSettingsService(
    ObjectActionSettingsSessionMixin,
    ObjectActionSettingsRuntimeMixin,
):
    """Own object-action settings routing, persistence, and helper delegation."""

    def __init__(
        self,
        *,
        project_storage_getter: Callable[[], ProjectStorage],
        session_getter: Callable[[], Session],
        presentation_getter: Callable[[], TimelinePresentation],
        require_layer: Callable[[object], LayerPresentation],
        analysis_service: AnalysisService,
        active_run_lookup: Callable[[str, object | None, str | None], PipelineRunState | None] | None = None,
    ) -> None:
        self._project_storage_getter = project_storage_getter
        self._session_getter = session_getter
        self._presentation_getter = presentation_getter
        self._require_layer = require_layer
        self._analysis_service = analysis_service
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
            workflow, pipeline_template_id = self._require_workflow(action_id)
            resolved_params = self._resolve_params(action_id, params, object_id=object_id, object_type=object_type)
            config = self._require_object_action_config(pipeline_template_id, scope=scope)
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
        with self.project_storage.locked():
            workflow, pipeline_template_id = self._require_workflow(action_id)
            if object_type is not None and object_type not in workflow.object_types:
                raise ValueError(
                    f"{action_id} does not support object_type '{object_type}'. Expected one of {workflow.object_types}."
                )

            resolved_params = self._resolve_params(action_id, params, object_id=object_id, object_type=object_type)
            layer_id = resolved_params.get("layer_id")
            if "layer_id" in workflow.params_schema and layer_id is None:
                raise ValueError(f"{action_id} requires a target layer.")

            template = get_registry().get(pipeline_template_id)
            if template is None:
                raise ValueError(f"Pipeline template not found: {pipeline_template_id}")

            config = self._require_object_action_config(pipeline_template_id, scope=scope)
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
                pipeline_template_id=pipeline_template_id,
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
            active_run = self._lookup_active_run(
                action_id,
                object_id=(object_id if object_id is not None else layer_id),
                object_type=(object_type or (workflow.object_types[0] if workflow.object_types else "object")),
            )
            is_running = PipelineRunService.is_active(active_run)
            run_label = "Run Again" if has_prior_outputs else "Run"
            warnings: tuple[str, ...] = ()
            if is_running:
                run_label = "Running..."
            elif active_run is not None and active_run.status == "failed" and active_run.error:
                warnings = (active_run.error,)
            return ObjectActionSettingsPlan(
                action_id=action_id,
                title=workflow.label,
                object_id=str(object_id if object_id is not None else layer_id or ""),
                object_type=object_type or (workflow.object_types[0] if workflow.object_types else "object"),
                pipeline_template_id=pipeline_template_id,
                editable_fields=tuple(field for field in editable_fields if not field.advanced),
                advanced_fields=tuple(field for field in editable_fields if field.advanced),
                locked_bindings=locked_bindings,
                has_prior_outputs=has_prior_outputs,
                run_label=run_label,
                settings_label="Open Settings",
                rerun_hint=rerun_hint,
                summary=f"{summary} · {'Song Default' if scope == 'song_default' else 'This Version'}",
                warnings=warnings,
                run_id=active_run.run_id if active_run is not None else None,
                is_running=is_running,
                run_status=active_run.status if active_run is not None else "",
                run_message=active_run.message if active_run is not None else "",
                run_percent=active_run.percent if active_run is not None else None,
                run_error=active_run.error if active_run is not None else None,
            )

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
    ) -> tuple[ActionDescriptor, str, dict[str, object], object | None]:
        workflow, pipeline_template_id = self._require_workflow(action_id)
        if object_type is not None and object_type not in workflow.object_types:
            raise ValueError(
                f"{action_id} does not support object_type '{object_type}'. Expected one of {workflow.object_types}."
            )
        resolved_params = self._resolve_params(action_id, params, object_id=object_id, object_type=object_type)
        layer_id = resolved_params.get("layer_id")
        if "layer_id" in workflow.params_schema and layer_id is None:
            raise ValueError(f"{action_id} requires a target layer.")
        return workflow, pipeline_template_id, resolved_params, layer_id

    def prepare_run(
        self,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        persist_scope: str | None = "version",
    ) -> PreparedPipelineRun:
        """Resolve and optionally persist one object-action run without executing it."""

        with self.project_storage.locked():
            workflow, pipeline_template_id, resolved_params, layer_id = self._resolve_execution_context(
                action_id,
                params,
                object_id=object_id,
                object_type=object_type,
            )
            config = self._require_object_action_config(
                pipeline_template_id,
                scope=persist_scope or "version",
            )
            if persist_scope is not None:
                config = self._persist_object_action_params(
                    config,
                    action_id=action_id,
                    params=resolved_params,
                    scope=persist_scope,
                )
            workflow_id = workflow.workflow_id
            if workflow_id is None:
                raise ValueError(f"Unsupported object action '{action_id}'.")
            return PreparedPipelineRun(
                action_id=action_id,
                workflow_id=workflow_id,
                pipeline_template_id=pipeline_template_id,
                config_id=config.id,
                display_label=workflow.label,
                object_id=str(object_id if object_id is not None else layer_id or ""),
                object_type=object_type or (workflow.object_types[0] if workflow.object_types else "object"),
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

    def persist_generated_source_layer_id(
        self,
        *,
        analysis_result: AnalysisResult,
        source_layer_id: object | None,
    ) -> None:
        if source_layer_id is None:
            return
        persisted_source_layer_id = str(source_layer_id)
        updated_version_ids: set[str] = set()
        with self.project_storage.transaction():
            for generated_layer_id in analysis_result.layer_ids:
                layer_record = self.project_storage.layers.get(generated_layer_id)
                if layer_record is None:
                    continue
                provenance = dict(layer_record.provenance)
                if provenance.get("source_layer_id") == persisted_source_layer_id:
                    continue
                provenance["source_layer_id"] = persisted_source_layer_id
                self.project_storage.layers.update(
                    replace(layer_record, provenance=provenance)
                )
                updated_version_ids.add(str(layer_record.song_version_id))
        for song_version_id in updated_version_ids:
            self.project_storage.dirty_tracker.mark_dirty(song_version_id)

    def _load_scoped_action_config(
        self,
        template_id: str,
        *,
        scope: str,
        song_id: str | None = None,
        song_version_id: str | None = None,
    ) -> ObjectActionConfigRecord:
        return _load_scoped_action_config(
            self,
            template_id,
            scope=scope,
            song_id=song_id,
            song_version_id=song_version_id,
        )

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

    def _lookup_active_run(
        self,
        action_id: str,
        *,
        object_id: object | None,
        object_type: str | None,
    ) -> PipelineRunState | None:
        if self._active_run_lookup is None:
            return None
        return self._active_run_lookup(action_id, object_id, object_type)

    def _require_object_action_config(
        self,
        template_id: str,
        *,
        scope: str = "version",
    ) -> ObjectActionConfigRecord:
        return _require_object_action_config(self, template_id, scope=scope)

    def _persist_object_action_params(
        self,
        config: ObjectActionConfigRecord,
        *,
        action_id: str,
        params: dict[str, object],
        scope: str = "version",
    ) -> ObjectActionConfigRecord:
        _workflow, pipeline_template_id = self._require_workflow(action_id)
        return _persist_object_action_params(
            self,
            config,
            action_id=action_id,
            pipeline_template_id=pipeline_template_id,
            params=params,
            scope=scope,
        )

    def _store_scoped_action_config(
        self,
        config: ObjectActionConfigRecord,
        *,
        scope: str,
    ) -> None:
        _store_scoped_action_config(self, config, scope=scope)

    def _require_active_song_id(self, action_name: str) -> str:
        if self.session.active_song_id is None:
            raise RuntimeError(f"{action_name} requires an active song.")
        return str(self.session.active_song_id)

    def _require_active_song_version_id(self, action_name: str) -> str:
        if self.session.active_song_version_id is None:
            raise RuntimeError(f"{action_name} requires an active song version.")
        return str(self.session.active_song_version_id)

    @staticmethod
    def _require_workflow(action_id: str) -> tuple[ActionDescriptor, str]:
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None or workflow.pipeline_template_id is None:
            raise ValueError(f"Unsupported object action '{action_id}'.")
        return workflow, workflow.pipeline_template_id

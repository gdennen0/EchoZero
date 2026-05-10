"""Object-action settings plan assembly.
Exists to isolate settings-plan shaping and active-run presentation from the service root.
Connects workflow descriptors, scoped config, and runtime field builders to UI-facing plans.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from echozero.application.presentation.models import LayerPresentation
from echozero.application.timeline.object_actions.descriptors import ActionDescriptor
from echozero.application.timeline.object_actions.settings import ObjectActionSettingsPlan
from echozero.application.timeline.operation_progress_service import (
    OperationProgressService,
    OperationProgressState,
)
from echozero.pipelines.registry import get_registry


class ObjectActionSettingsPlanShell(Protocol):
    _require_layer: Callable[[object], LayerPresentation]

    def _require_workflow(self, action_id: str) -> tuple[ActionDescriptor, str]: ...

    @staticmethod
    def _resolve_params(
        action_id: str,
        params: dict[str, object] | None,
        *,
        object_id: object | None,
        object_type: str | None,
    ) -> dict[str, object]: ...

    def _require_object_action_config(self, template_id: str, *, scope: str = "version"): ...

    def _resolve_object_action_object_bindings(
        self,
        action_id: str,
        *,
        layer_id: object | None,
        params: dict[str, object],
    ) -> dict[str, object]: ...

    def _build_object_action_setting_fields(
        self,
        action_id: str,
        *,
        defaults: dict[str, object],
        persisted_values: dict[str, object],
        object_bindings: dict[str, object],
        params: dict[str, object],
    ) -> tuple[object, ...]: ...

    def _has_prior_outputs_for_action(
        self,
        *,
        pipeline_template_id: str,
        source_layer_id: object | None,
    ) -> bool: ...

    @staticmethod
    def _format_locked_binding_value(value: object) -> str: ...

    def _lookup_active_run(
        self,
        action_id: str,
        *,
        object_id: object | None,
        object_type: str | None,
    ) -> OperationProgressState | None: ...


class ObjectActionSettingsPlanMixin:
    """Builds UI-facing settings plans for object actions."""

    def _describe_settings_plan(
        self: ObjectActionSettingsPlanShell,
        action_id: str,
        params: dict[str, object] | None = None,
        *,
        object_id: object | None = None,
        object_type: str | None = None,
        scope: str = "version",
    ) -> ObjectActionSettingsPlan:
        workflow, pipeline_template_id = self._require_workflow(action_id)
        if object_type is not None and object_type not in workflow.object_types:
            raise ValueError(
                f"{action_id} does not support object_type '{object_type}'. Expected one of {workflow.object_types}."
            )

        resolved_params = self._resolve_params(
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
        )
        layer_id = resolved_params.get("layer_id")
        requires_layer = "layer_id" in workflow.params_schema
        missing_target_layer = requires_layer and layer_id is None

        template = get_registry().get(pipeline_template_id)
        if template is None:
            raise ValueError(f"Pipeline template not found: {pipeline_template_id}")

        config = self._require_object_action_config(pipeline_template_id, scope=scope)
        defaults = {key: knob.default for key, knob in template.knobs.items()}
        persisted_values = dict(config.knob_values)
        object_bindings = self._resolve_object_action_object_bindings(
            action_id,
            layer_id=layer_id,
            params=resolved_params,
        )
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
        summary = layer.title if layer is not None else "No target layer selected"
        active_operation = self._lookup_active_run(
            action_id,
            object_id=(object_id if object_id is not None else layer_id),
            object_type=(
                object_type or (workflow.object_types[0] if workflow.object_types else "object")
            ),
        )
        is_running = OperationProgressService.is_active(active_operation)
        run_label = "Run Again" if has_prior_outputs else "Run"
        warnings: tuple[str, ...] = (
            (
                (
                    "Select a target layer to run this action. "
                    "You can still save reusable pipeline settings now."
                ),
            )
            if missing_target_layer
            else ()
        )
        if is_running:
            run_label = "Running..."
        elif (
            active_operation is not None
            and active_operation.status == "failed"
            and active_operation.error
        ):
            warnings = (*warnings, active_operation.error)
        return ObjectActionSettingsPlan(
            action_id=action_id,
            title=workflow.label,
            object_id=str(object_id if object_id is not None else layer_id or ""),
            object_type=object_type
            or (workflow.object_types[0] if workflow.object_types else "object"),
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
            operation_id=active_operation.operation_id if active_operation is not None else None,
            is_running=is_running,
            operation_status=active_operation.status if active_operation is not None else "",
            operation_message=active_operation.message if active_operation is not None else "",
            operation_fraction=(
                active_operation.fraction_complete if active_operation is not None else None
            ),
            operation_error=active_operation.error if active_operation is not None else None,
        )

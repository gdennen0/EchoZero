"""Object-action settings context helpers.
Exists to isolate workflow lookup, param normalization, and active-run lookup from the service root.
Connects action descriptors and operation progress state to the public execution service.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from echozero.application.timeline.object_actions.descriptors import (
    ActionDescriptor,
    workflow_descriptor_for_action,
)
from echozero.application.timeline.operation_progress_service import (
    OperationProgressState,
)


class ObjectActionSettingsContextShell(Protocol):
    _active_run_lookup: (
        Callable[[str, object | None, str | None], OperationProgressState | None] | None
    )

    @staticmethod
    def _require_workflow(action_id: str) -> tuple[ActionDescriptor, str]: ...


class ObjectActionSettingsContextMixin:
    """Provides workflow resolution and param normalization for object actions."""

    def _resolve_execution_context(
        self: ObjectActionSettingsContextShell,
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
        resolved_params = self._resolve_params(
            action_id,
            params,
            object_id=object_id,
            object_type=object_type,
        )
        layer_id = resolved_params.get("layer_id")
        if "layer_id" in workflow.params_schema and layer_id is None:
            raise ValueError(f"{action_id} requires a target layer.")
        return workflow, pipeline_template_id, resolved_params, layer_id

    @staticmethod
    def _resolve_params(
        action_id: str,
        params: dict[str, object] | None,
        *,
        object_id: object | None,
        object_type: str | None,
    ) -> dict[str, object]:
        del action_id
        resolved_params = dict(params or {})
        normalized_layer_id = ObjectActionSettingsContextMixin._normalize_optional_object_id(
            resolved_params.get("layer_id")
        )
        if normalized_layer_id is None:
            resolved_params.pop("layer_id", None)
        else:
            resolved_params["layer_id"] = normalized_layer_id

        normalized_object_id = ObjectActionSettingsContextMixin._normalize_optional_object_id(
            object_id
        )
        if (
            object_type == "layer"
            and normalized_object_id is not None
            and "layer_id" not in resolved_params
        ):
            resolved_params["layer_id"] = normalized_object_id
        return resolved_params

    @staticmethod
    def _normalize_optional_object_id(value: object | None) -> object | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    def _lookup_active_run(
        self: ObjectActionSettingsContextShell,
        action_id: str,
        *,
        object_id: object | None,
        object_type: str | None,
    ) -> OperationProgressState | None:
        if self._active_run_lookup is None:
            return None
        return self._active_run_lookup(action_id, object_id, object_type)

    @staticmethod
    def _require_workflow(action_id: str) -> tuple[ActionDescriptor, str]:
        workflow = workflow_descriptor_for_action(action_id)
        if workflow is None or workflow.pipeline_template_id is None:
            raise ValueError(f"Unsupported object action '{action_id}'.")
        return workflow, workflow.pipeline_template_id

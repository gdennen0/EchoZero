"""Runtime binding helpers for object-action settings.
Exists to isolate binding resolution, runtime-param coercion, and model-path defaults from the core settings service.
Connects object-action descriptors to typed layer bindings and runtime model selection.
"""

from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Protocol

from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.timeline.object_action_model_picker_options import (
    build_runtime_model_picker_options,
)
from echozero.application.timeline.object_actions.descriptors import ActionDescriptor, workflow_descriptor_for_action
from echozero.application.timeline.object_actions.settings import (
    ObjectActionSettingField,
    ObjectActionSettingOption,
)
from echozero.pipelines.params import KnobWidget
from echozero.pipelines.registry import get_registry


class ObjectActionSettingsRuntimeShell(Protocol):
    _require_layer: Callable[[object], LayerPresentation]

    def presentation(self) -> TimelinePresentation: ...

    def _require_workflow(self, action_id: str) -> tuple[ActionDescriptor, str]: ...


class ObjectActionSettingsRuntimeMixin:
    def _resolve_object_action_runtime_bindings(
        self: ObjectActionSettingsRuntimeShell,
        action_id: str,
        *,
        layer_id: object | None,
        params: dict[str, object],
    ) -> dict[str, object]:
        return resolve_object_action_runtime_bindings(
            self,
            action_id,
            layer_id=layer_id,
            params=params,
        )

    def _resolve_object_action_object_bindings(
        self: ObjectActionSettingsRuntimeShell,
        action_id: str,
        *,
        layer_id: object | None,
        params: dict[str, object],
    ) -> dict[str, object]:
        return resolve_object_action_object_bindings(
            self,
            action_id,
            layer_id=layer_id,
            params=params,
        )

    def _coerce_object_action_runtime_params(
        self: ObjectActionSettingsRuntimeShell,
        action_id: str,
        *,
        params: dict[str, object],
    ) -> dict[str, object]:
        return coerce_object_action_runtime_params(action_id, params=params)

    def _build_object_action_setting_fields(
        self: ObjectActionSettingsRuntimeShell,
        action_id: str,
        *,
        defaults: dict[str, object],
        persisted_values: dict[str, object],
        object_bindings: dict[str, object],
        params: dict[str, object],
    ) -> tuple[ObjectActionSettingField, ...]:
        return build_object_action_setting_fields(
            self,
            action_id,
            defaults=defaults,
            persisted_values=persisted_values,
            object_bindings=object_bindings,
            params=params,
        )

    def _has_prior_outputs_for_action(
        self: ObjectActionSettingsRuntimeShell,
        *,
        pipeline_template_id: str,
        source_layer_id: object | None,
    ) -> bool:
        return has_prior_outputs_for_action(
            self,
            pipeline_template_id=pipeline_template_id,
            source_layer_id=source_layer_id,
        )

    @staticmethod
    def _format_locked_binding_value(value: object) -> str:
        return format_locked_binding_value(value)

    def _extract_classified_drums_model_defaults(
        self: ObjectActionSettingsRuntimeShell,
    ) -> dict[str, object]:
        return extract_classified_drums_model_defaults()


def resolve_object_action_runtime_bindings(
    shell: ObjectActionSettingsRuntimeShell,
    action_id: str,
    *,
    layer_id: object | None,
    params: dict[str, object],
) -> dict[str, object]:
    bindings = resolve_object_action_object_bindings(
        shell,
        action_id,
        layer_id=layer_id,
        params=params,
    )
    bindings.pop("layer_id", None)
    return bindings


def resolve_object_action_object_bindings(
    shell: ObjectActionSettingsRuntimeShell,
    action_id: str,
    *,
    layer_id: object | None,
    params: dict[str, object],
) -> dict[str, object]:
    workflow = workflow_descriptor_for_action(action_id)
    if workflow is None:
        raise ValueError(f"Unsupported object action '{action_id}'.")
    if workflow.binding_resolver_id is None:
        return {}
    resolver = _object_action_binding_resolvers(shell).get(workflow.binding_resolver_id)
    if resolver is None:
        raise ValueError(
            f"Unsupported object action binding resolver '{workflow.binding_resolver_id}' for '{action_id}'."
        )
    if "layer_id" in workflow.params_schema and layer_id is None:
        return {}
    layer = shell._require_layer(layer_id) if "layer_id" in workflow.params_schema else None
    return resolver(layer=layer, params=params)


def coerce_object_action_runtime_params(
    action_id: str,
    *,
    params: dict[str, object],
) -> dict[str, object]:
    workflow = workflow_descriptor_for_action(action_id)
    resolved = dict(params)
    if workflow is None or workflow.runtime_param_coercer_id is None:
        return resolved
    coercer = _object_action_runtime_param_coercers().get(workflow.runtime_param_coercer_id)
    if coercer is None:
        raise ValueError(
            f"Unsupported object action runtime param coercer '{workflow.runtime_param_coercer_id}' for '{action_id}'."
        )
    return coercer(resolved)


def build_object_action_setting_fields(
    shell: ObjectActionSettingsRuntimeShell,
    action_id: str,
    *,
    defaults: dict[str, object],
    persisted_values: dict[str, object],
    object_bindings: dict[str, object],
    params: dict[str, object],
) -> tuple[ObjectActionSettingField, ...]:
    _workflow, pipeline_template_id = shell._require_workflow(action_id)
    template = get_registry().get(pipeline_template_id)
    assert template is not None
    resolved_params = coerce_object_action_runtime_params(action_id, params=params)
    fields: list[ObjectActionSettingField] = []
    for key, knob in template.knobs.items():
        if key in object_bindings:
            continue
        persisted_value = persisted_values.get(key, defaults.get(key, knob.default))
        value = resolved_params.get(key, persisted_value)
        widget_name = _knob_widget_name(knob.widget)
        options = tuple(
            ObjectActionSettingOption(value=option, label=option.replace("_", " ").title())
            for option in (knob.options or ())
        )
        model_options = build_runtime_model_picker_options(knob=knob, value=value)
        if model_options:
            widget_name = "dropdown"
            options = model_options
        fields.append(
            ObjectActionSettingField(
                key=key,
                label=knob.label or key.replace("_", " ").title(),
                value=value,
                default_value=knob.default,
                persisted_value=persisted_value,
                is_dirty=value != persisted_value,
                widget=widget_name,
                description=knob.description,
                advanced=knob.advanced,
                placeholder=knob.placeholder,
                units=knob.units,
                min_value=knob.min_value,
                max_value=knob.max_value,
                step=knob.step,
                options=options,
            )
        )
    return tuple(fields)


def has_prior_outputs_for_action(
    shell: ObjectActionSettingsRuntimeShell,
    *,
    pipeline_template_id: str,
    source_layer_id: object | None,
) -> bool:
    if source_layer_id is None:
        return False
    return any(
        layer.status.pipeline_id == pipeline_template_id
        and (
            str(layer.status.source_layer_id) == str(source_layer_id)
            or str(source_layer_id) == "source_audio"
        )
        for layer in shell.presentation().layers
    )


def format_locked_binding_value(value: object) -> str:
    text = str(value)
    return text if len(text) <= 72 else f"{text[:69]}..."


def extract_classified_drums_model_defaults() -> dict[str, object]:
    from echozero.application.timeline.object_action_settings_service import (
        ensure_installed_models_dir,
        resolve_installed_binary_drum_bundles,
        upgrade_installed_runtime_bundles,
    )

    models_dir = ensure_installed_models_dir()
    upgrade_installed_runtime_bundles(models_dir)
    defaults: dict[str, object] = {}
    for label in ("kick", "snare"):
        try:
            bundles = _resolve_installed_binary_drum_bundles_compat(
                resolve_installed_binary_drum_bundles,
                models_dir=models_dir,
                labels=(label,),
            )
        except FileNotFoundError:
            continue
        defaults[f"{label}_model_path"] = str(bundles[label].manifest_path)
    return defaults


def _resolve_installed_binary_drum_bundles_compat(
    resolver: Callable[..., dict[str, object]],
    *,
    models_dir: object,
    labels: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Call bundle resolvers across the real keyword-only API and zero-arg test doubles."""

    try:
        parameters = inspect.signature(resolver).parameters
    except (TypeError, ValueError):
        kwargs = {"models_dir": models_dir}
        if labels is not None:
            kwargs["labels"] = labels
        return resolver(**kwargs)

    kwargs: dict[str, object] = {}
    if "models_dir" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs["models_dir"] = models_dir
    if labels is not None and (
        "labels" in parameters
        or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    ):
        kwargs["labels"] = labels
    if kwargs:
        return resolver(**kwargs)
    if not parameters:
        return resolver()
    if labels is not None and len(parameters) > 1:
        return resolver(models_dir, labels)
    return resolver(models_dir)


def _object_action_binding_resolvers(
    shell: ObjectActionSettingsRuntimeShell,
) -> dict[str, Callable[..., dict[str, object]]]:
    return {
        "extract_stems": lambda *, layer, params: _resolve_extract_stems_object_bindings(
            shell,
            layer=layer,
            params=params,
        ),
        "extract_song_drum_events": lambda *, layer, params: _resolve_extract_song_drum_events_object_bindings(
            shell,
            layer=layer,
            params=params,
        ),
        "extract_song_sections": lambda *, layer, params: _resolve_extract_song_sections_object_bindings(
            shell,
            layer=layer,
            params=params,
        ),
        "extract_drum_events": lambda *, layer, params: _resolve_extract_drum_events_object_bindings(
            shell,
            layer=layer,
            params=params,
        ),
        "classify_drum_events": lambda *, layer, params: _resolve_classify_drum_events_object_bindings(
            shell,
            layer=layer,
            params=params,
        ),
        "extract_classified_drums": lambda *, layer, params: _resolve_extract_classified_drums_object_bindings(
            shell,
            layer=layer,
            params=params,
        ),
    }


def _object_action_runtime_param_coercers() -> dict[str, Callable[[dict[str, object]], dict[str, object]]]:
    return {
        "classify_drum_events": _coerce_classify_drum_events_runtime_params,
    }


def _resolve_extract_stems_object_bindings(
    shell: ObjectActionSettingsRuntimeShell,
    *,
    layer: LayerPresentation | None,
    params: dict[str, object],
) -> dict[str, object]:
    del shell, params
    assert layer is not None
    return _bindings_for_extract_stems(layer)


def _resolve_extract_song_drum_events_object_bindings(
    shell: ObjectActionSettingsRuntimeShell,
    *,
    layer: LayerPresentation | None,
    params: dict[str, object],
) -> dict[str, object]:
    del shell, params
    assert layer is not None
    return _bindings_for_extract_song_drum_events(layer)


def _resolve_extract_drum_events_object_bindings(
    shell: ObjectActionSettingsRuntimeShell,
    *,
    layer: LayerPresentation | None,
    params: dict[str, object],
) -> dict[str, object]:
    del shell, params
    assert layer is not None
    return _bindings_for_extract_drum_events(layer)


def _resolve_extract_song_sections_object_bindings(
    shell: ObjectActionSettingsRuntimeShell,
    *,
    layer: LayerPresentation | None,
    params: dict[str, object],
) -> dict[str, object]:
    del params
    assert layer is not None
    return _bindings_for_extract_song_sections(shell, layer=layer)


def _resolve_classify_drum_events_object_bindings(
    shell: ObjectActionSettingsRuntimeShell,
    *,
    layer: LayerPresentation | None,
    params: dict[str, object],
) -> dict[str, object]:
    del shell
    assert layer is not None
    return _bindings_for_classify_drum_events(layer, params=params, include_runtime_overrides=False)


def _resolve_extract_classified_drums_object_bindings(
    shell: ObjectActionSettingsRuntimeShell,
    *,
    layer: LayerPresentation | None,
    params: dict[str, object],
) -> dict[str, object]:
    del shell, params
    assert layer is not None
    return _bindings_for_extract_classified_drums(layer)


def _coerce_classify_drum_events_runtime_params(params: dict[str, object]) -> dict[str, object]:
    from echozero.application.timeline.object_action_settings_service import resolve_runtime_model_path

    resolved = dict(params)
    model_path = resolved.pop("model_path", None)
    if model_path is not None and "classify_model_path" not in resolved:
        resolved["classify_model_path"] = model_path
    classify_model_path = resolved.get("classify_model_path")
    if classify_model_path is not None:
        resolved["classify_model_path"] = str(
            resolve_runtime_model_path(str(classify_model_path))
        )
    return resolved


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


def _bindings_for_extract_stems(layer: LayerPresentation) -> dict[str, object]:
    return _bindings_for_song_audio_pipeline_action(
        layer,
        action_name="timeline.extract_stems",
    )


def _bindings_for_extract_song_drum_events(layer: LayerPresentation) -> dict[str, object]:
    return _bindings_for_song_audio_pipeline_action(
        layer,
        action_name="timeline.extract_song_drum_events",
    )


def _bindings_for_extract_song_sections(
    shell: ObjectActionSettingsRuntimeShell,
    *,
    layer: LayerPresentation,
) -> dict[str, object]:
    if layer.kind is LayerKind.AUDIO:
        return _bindings_for_song_audio_pipeline_action(
            layer,
            action_name="timeline.extract_song_sections",
        )
    if layer.kind is not LayerKind.SECTION:
        raise ValueError(
            "timeline.extract_song_sections requires an audio or section layer, "
            f"got {layer.kind.name.lower()}."
        )
    source_layer = _resolve_source_song_audio_layer(shell)
    if source_layer is None or not source_layer.source_audio_path:
        raise RuntimeError(
            "timeline.extract_song_sections requires a source song audio layer with a resolved path."
        )
    return {"audio_file": str(source_layer.source_audio_path)}


def _bindings_for_song_audio_pipeline_action(
    layer: LayerPresentation,
    *,
    action_name: str,
) -> dict[str, object]:
    if layer.kind is not LayerKind.AUDIO:
        raise ValueError(f"{action_name} requires an audio layer, got {layer.kind.name.lower()}.")
    if str(layer.layer_id) != "source_audio":
        raise NotImplementedError(
            f"{action_name} currently runs only from the imported song layer. "
            "Derived-audio reruns are deferred until arbitrary-layer pipeline input is wired."
        )
    if not layer.source_audio_path:
        raise ValueError(f"{action_name} requires a resolved source audio path.")
    return {"audio_file": str(layer.source_audio_path)}


def _resolve_source_song_audio_layer(
    shell: ObjectActionSettingsRuntimeShell,
) -> LayerPresentation | None:
    presentation = shell.presentation()
    audio_layers = [
        candidate
        for candidate in presentation.layers
        if candidate.kind is LayerKind.AUDIO and bool(candidate.source_audio_path)
    ]
    if not audio_layers:
        return None
    direct_source = next(
        (candidate for candidate in audio_layers if str(candidate.layer_id).strip() == "source_audio"),
        None,
    )
    if direct_source is not None:
        return direct_source
    canonical_song_layer = next(
        (
            candidate
            for candidate in audio_layers
            if candidate.status is None
            or (
                not str(candidate.status.source_layer_id or "").strip()
                and not str(candidate.status.pipeline_id or "").strip()
            )
        ),
        None,
    )
    if canonical_song_layer is not None:
        return canonical_song_layer
    return audio_layers[0]


def _bindings_for_extract_drum_events(layer: LayerPresentation) -> dict[str, object]:
    _validate_drum_derived_audio_layer(layer, action_name="timeline.extract_drum_events")
    return {"audio_file": str(layer.source_audio_path)}


def _bindings_for_classify_drum_events(
    layer: LayerPresentation,
    *,
    params: dict[str, object],
    include_runtime_overrides: bool = True,
) -> dict[str, object]:
    from echozero.application.timeline.object_action_settings_service import resolve_runtime_model_path

    _validate_drum_derived_audio_layer(layer, action_name="timeline.classify_drum_events")
    bindings: dict[str, object] = {"audio_file": str(layer.source_audio_path)}
    if not include_runtime_overrides:
        return bindings
    model_path = params.get("classify_model_path", params.get("model_path"))
    resolved_model_path = resolve_runtime_model_path(str(model_path))
    if not str(resolved_model_path).strip():
        raise ValueError("timeline.classify_drum_events requires a non-empty model path.")
    bindings["classify_model_path"] = str(resolved_model_path)
    return bindings


def _bindings_for_extract_classified_drums(layer: LayerPresentation) -> dict[str, object]:
    _validate_drum_derived_audio_layer(layer, action_name="timeline.extract_classified_drums")
    return {"audio_file": str(layer.source_audio_path)}


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

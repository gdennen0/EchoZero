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
from echozero.application.timeline.object_content import is_imported_song_object_id
from echozero.application.timeline.object_actions.descriptors import (
    ActionDescriptor,
    workflow_descriptor_for_action,
)
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
            ObjectActionSettingOption(
                value=option, label=_option_label_for_setting(key=key, option=option)
            )
            for option in (knob.options or ())
        )
        widget_name, options, value, persisted_value = _apply_custom_drum_output_field_behavior(
            key=key,
            widget_name=widget_name,
            options=options,
            value=value,
            persisted_value=persisted_value,
        )
        field_description = knob.description
        field_enabled = True
        model_options = build_runtime_model_picker_options(
            knob=knob,
            value=value,
            key=key,
            action_id=action_id,
        )
        if model_options:
            widget_name = "dropdown"
            options = model_options
            if _is_missing_binary_drum_model_picker(model_options):
                field_enabled = False
            field_description = _model_picker_description(
                key=key,
                options=model_options,
                fallback=field_description,
            )
        fields.append(
            ObjectActionSettingField(
                key=key,
                label=knob.label or key.replace("_", " ").title(),
                value=value,
                default_value=knob.default,
                persisted_value=persisted_value,
                is_dirty=value != persisted_value,
                widget=widget_name,
                description=field_description,
                enabled=field_enabled,
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
    source_layer_ids = _source_layer_id_candidates(shell, source_layer_id)
    return any(
        layer.status.pipeline_id == pipeline_template_id
        and str(layer.status.source_layer_id or "") in source_layer_ids
        for layer in shell.presentation().layers
    )


def _is_missing_binary_drum_model_picker(
    options: tuple[ObjectActionSettingOption, ...],
) -> bool:
    return len(options) == 1 and options[0].metadata.get("status") == "missing"


def _model_picker_description(
    *,
    key: str,
    options: tuple[ObjectActionSettingOption, ...],
    fallback: str,
) -> str:
    label = _model_label_from_key(key)
    if label is None:
        return fallback
    ready_options = tuple(
        option for option in options if option.metadata.get("status") == "ready"
    )
    if not ready_options:
        return (
            f"No compatible {label.title()} classifier is installed. "
            f"Install a Foundry export with classes [{label}, other] to enable this output."
        )
    current_default_count = sum(
        1 for option in ready_options if option.metadata.get("is_current_default") is True
    )
    default_note = " Current default is listed first." if current_default_count else ""
    return (
        f"{len(ready_options)} compatible {label.title()} classifier model"
        f"{'s' if len(ready_options) != 1 else ''} available."
        f"{default_note} Only [{label}, other] bundles are shown."
    )


def _model_label_from_key(key: str) -> str | None:
    text = str(key or "").strip().lower()
    if not text.endswith("_model_path"):
        return None
    label = text[: -len("_model_path")]
    if label in {"symbol", "cymbol"}:
        return "cymbal"
    if label in {"kick", "snare", "clap", "cymbal"}:
        return label
    return None


def format_locked_binding_value(value: object) -> str:
    text = str(value)
    return text if len(text) <= 72 else f"{text[:69]}..."


def _option_label_for_setting(*, key: str, option: str) -> str:
    if key == "sensitivity_preset":
        if option == "more_events":
            return "More Events"
        if option == "balanced":
            return "Balanced"
        if option == "fewer_events":
            return "Fewer Events"
        if option == "custom":
            return "Custom / Advanced"
    if key == "assignment_mode":
        if option == "independent":
            return "Independent (Allow Overlaps)"
        if option == "exclusive_max":
            return "Winner Takes Similar Hits"
    if key == "detect_method":
        if option == "mir_self_similarity":
            return "MIR Part Boundaries (Recommended)"
        if option == "mfcc_sequence_pooling":
            return "MFCC Sequence Pooling (Legacy)"
        if option == "determine_sections_style":
            return "Experimental (determine_sections-style)"
    return option.replace("_", " ").title()


def extract_classified_drums_model_defaults() -> dict[str, object]:
    from echozero.application.timeline.object_action_settings_service import (
        ensure_installed_models_dir,
        resolve_installed_binary_drum_bundles,
        upgrade_installed_runtime_bundles,
    )
    from echozero.models.classifier_model_catalog import build_runtime_classifier_model_catalog

    models_dir = ensure_installed_models_dir()
    upgrade_installed_runtime_bundles(models_dir)
    catalog = build_runtime_classifier_model_catalog(models_dir=models_dir)
    defaults: dict[str, object] = {}
    installed_labels: list[str] = []
    for label in ("kick", "snare", "clap", "cymbal"):
        candidates = catalog.candidates_for_label(label)
        if candidates:
            defaults[f"{label}_model_path"] = str(candidates[0].manifest_path)
            installed_labels.append(label)
            continue
        try:
            bundles = _resolve_installed_binary_drum_bundles_compat(
                resolve_installed_binary_drum_bundles,
                models_dir=models_dir,
                labels=(label,),
            )
        except FileNotFoundError:
            continue
        bundle = bundles.get(label)
        if bundle is None:
            continue
        defaults[f"{label}_model_path"] = str(bundle.manifest_path)
        installed_labels.append(label)
    if installed_labels:
        defaults["target_drum_labels"] = tuple(installed_labels)
    return defaults


def _source_layer_id_candidates(
    shell: ObjectActionSettingsRuntimeShell,
    source_layer_id: object,
) -> set[str]:
    source_id = str(source_layer_id or "").strip()
    if not source_id:
        return set()
    candidates = {source_id}
    if source_id == "source_audio":
        candidates.update(
            str(layer.layer_id)
            for layer in shell.presentation().layers
            if _is_imported_song_audio_layer(layer)
        )
    return candidates


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
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        kwargs["models_dir"] = models_dir
    if labels is not None and (
        "labels" in parameters
        or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
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
        "extract_note_contour": lambda *, layer, params: _resolve_extract_note_contour_object_bindings(
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


def _object_action_runtime_param_coercers() -> (
    dict[str, Callable[[dict[str, object]], dict[str, object]]]
):
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


def _resolve_extract_note_contour_object_bindings(
    shell: ObjectActionSettingsRuntimeShell,
    *,
    layer: LayerPresentation | None,
    params: dict[str, object],
) -> dict[str, object]:
    del shell, params
    assert layer is not None
    return _bindings_for_extract_note_contour(layer)


def _resolve_classify_drum_events_object_bindings(
    shell: ObjectActionSettingsRuntimeShell,
    *,
    layer: LayerPresentation | None,
    params: dict[str, object],
) -> dict[str, object]:
    del shell
    assert layer is not None
    return _bindings_for_classify_drum_events(
        layer, params=params, include_runtime_overrides=False
    )


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
    from echozero.application.timeline.object_action_settings_service import (
        resolve_runtime_model_path,
    )

    resolved = dict(params)
    model_path = resolved.pop("model_path", None)
    if model_path is not None and "classify_model_path" not in resolved:
        resolved["classify_model_path"] = model_path
    classify_model_path = resolved.get("classify_model_path")
    if classify_model_path is not None:
        resolved["classify_model_path"] = str(resolve_runtime_model_path(str(classify_model_path)))
    return resolved


def _knob_widget_name(widget: KnobWidget) -> str:
    mapping = {
        KnobWidget.MULTI_SELECT: "checkbox_group",
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


def _apply_custom_drum_output_field_behavior(
    *,
    key: str,
    widget_name: str,
    options: tuple[ObjectActionSettingOption, ...],
    value: object,
    persisted_value: object,
) -> tuple[str, tuple[ObjectActionSettingOption, ...], object, object]:
    if key != "target_drum_labels":
        return widget_name, options, value, persisted_value

    installed_labels = _installed_drum_output_labels()
    resolved_value = _normalize_drum_label_values(value)
    resolved_persisted = _normalize_drum_label_values(persisted_value)
    fallback_selection = installed_labels or ("kick", "snare")
    option_values = tuple(
        dict.fromkeys(
            tuple(_normalize_drum_label_value(option.value) for option in options if option.value)
            + installed_labels
        )
    )
    label_options = tuple(
        ObjectActionSettingOption(
            value=label,
            label=label.replace("_", " ").title(),
        )
        for label in option_values
        if label
    )
    default_value: object = resolved_value or fallback_selection
    persisted = resolved_persisted or fallback_selection
    return "checkbox_group", label_options or options, default_value, persisted


def _normalize_drum_label_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values: tuple[object, ...] = tuple(value.split(","))
    elif isinstance(value, (list, tuple, set)):
        raw_values = tuple(value)
    else:
        raw_values = () if value is None else (value,)
    return tuple(
        dict.fromkeys(
            label for label in (_normalize_drum_label_value(item) for item in raw_values) if label
        )
    )


def _normalize_drum_label_value(value: object) -> str:
    label = str(value or "").strip().lower()
    if label in {"symbol", "cymbol"}:
        return "cymbal"
    return label


def _installed_drum_output_labels() -> tuple[str, ...]:
    from echozero.models.runtime_bundle_selection import list_installed_binary_drum_bundle_labels

    try:
        return list_installed_binary_drum_bundle_labels()
    except FileNotFoundError:
        return ()


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
    if not _is_imported_song_audio_layer(layer):
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
        (candidate for candidate in audio_layers if _is_imported_song_audio_layer(candidate)), None
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
    _validate_stem_derived_audio_layer(layer, action_name="timeline.extract_drum_events")
    return {"audio_file": str(layer.source_audio_path)}


def _bindings_for_extract_note_contour(layer: LayerPresentation) -> dict[str, object]:
    _validate_audio_layer(layer, action_name="timeline.extract_note_contour")
    return {"audio_file": str(layer.source_audio_path)}


def _bindings_for_classify_drum_events(
    layer: LayerPresentation,
    *,
    params: dict[str, object],
    include_runtime_overrides: bool = True,
) -> dict[str, object]:
    from echozero.application.timeline.object_action_settings_service import (
        resolve_runtime_model_path,
    )

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
        raise ValueError(f"{action_name} requires an audio layer, got {layer.kind.name.lower()}.")
    if not layer.source_audio_path:
        raise RuntimeError(f"{action_name} requires a source audio path on the selected layer.")

    title_lower = layer.title.lower()
    source_label = layer.status.source_label if layer.status is not None else ""
    source_label_lower = source_label.lower()
    badges = {str(badge).strip().lower() for badge in layer.badges}
    if "drum" not in title_lower and "drums" not in badges and "drum" not in source_label_lower:
        raise NotImplementedError(
            f"{action_name} currently runs only from drum-derived audio layers. "
            "Select a drums layer produced by stem separation."
        )


def _validate_audio_layer(layer: LayerPresentation, *, action_name: str) -> None:
    if layer.kind is not LayerKind.AUDIO:
        raise ValueError(f"{action_name} requires an audio layer, got {layer.kind.name.lower()}.")
    if not layer.source_audio_path:
        raise RuntimeError(f"{action_name} requires a source audio path on the selected layer.")


def _validate_stem_derived_audio_layer(layer: LayerPresentation, *, action_name: str) -> None:
    if layer.kind is not LayerKind.AUDIO:
        raise ValueError(f"{action_name} requires an audio layer, got {layer.kind.name.lower()}.")
    if not layer.source_audio_path:
        raise RuntimeError(f"{action_name} requires a source audio path on the selected layer.")
    title_lower = layer.title.lower()
    source_label = layer.status.source_label if layer.status is not None else ""
    source_label_lower = source_label.lower()
    badges = {str(badge).strip().lower() for badge in layer.badges}
    output_name = (layer.status.output_name if layer.status is not None else "").strip().lower()
    if (
        output_name not in {"drums", "bass", "vocals", "other"}
        and "drum" not in title_lower
        and "drums" not in badges
        and "drum" not in source_label_lower
    ):
        raise NotImplementedError(
            f"{action_name} currently runs only from stem-derived audio layers. "
            "Select a drums, bass, vocals, or other stem layer produced by stem separation."
        )


def _is_imported_song_audio_layer(layer: LayerPresentation) -> bool:
    if layer.kind is not LayerKind.AUDIO:
        return False
    if is_imported_song_object_id(layer.object_id):
        return True
    return layer.status is None or (
        not str(layer.status.source_layer_id or "").strip()
        and not str(layer.status.pipeline_id or "").strip()
    )

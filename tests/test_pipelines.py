"""
Pipeline infrastructure tests: registry, decorator, templates, knobs, validation.
Exercises PipelineRegistry, @pipeline_template decorator, and onset_detection template.
"""

from __future__ import annotations

import pytest

from echozero.domain.graph import Graph
from echozero.pipelines.params import Knob, KnobWidget, knob
from echozero.pipelines.registry import (
    PipelineRegistry,
    PipelineTemplate,
    get_registry,
    pipeline_template,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry() -> PipelineRegistry:
    """Fresh registry for each test — no global state leakage."""
    return PipelineRegistry()


def _dummy_builder() -> Graph:
    return Graph()


def _make_template(
    id: str = 'test_pipeline',
    name: str = 'Test Pipeline',
    description: str = 'A test pipeline',
    knobs: dict[str, Knob] | None = None,
) -> PipelineTemplate:
    return PipelineTemplate(
        id=id,
        name=name,
        description=description,
        knobs=knobs or {},
        builder=_dummy_builder,
    )


# ===================================================================
# PipelineRegistry
# ===================================================================

class TestPipelineRegistry:
    """Registry basic CRUD: register, get, list, ids."""

    def test_register_and_get(self, registry: PipelineRegistry) -> None:
        t = _make_template()
        registry.register(t)
        assert registry.get('test_pipeline') is t

    def test_get_nonexistent_returns_none(self, registry: PipelineRegistry) -> None:
        assert registry.get('no_such') is None

    def test_list_returns_all(self, registry: PipelineRegistry) -> None:
        t1 = _make_template(id='a', name='A')
        t2 = _make_template(id='b', name='B')
        registry.register(t1)
        registry.register(t2)
        templates = registry.list()
        assert len(templates) == 2
        assert t1 in templates
        assert t2 in templates

    def test_ids_returns_all_keys(self, registry: PipelineRegistry) -> None:
        registry.register(_make_template(id='x'))
        registry.register(_make_template(id='y'))
        ids = registry.ids()
        assert 'x' in ids
        assert 'y' in ids

    def test_empty_registry(self, registry: PipelineRegistry) -> None:
        assert registry.list() == []
        assert registry.ids() == []

    def test_register_overwrites_duplicate_id(self, registry: PipelineRegistry) -> None:
        t1 = _make_template(id='dup', name='First')
        t2 = _make_template(id='dup', name='Second')
        registry.register(t1)
        registry.register(t2)
        assert registry.get('dup').name == 'Second'
        assert len(registry.list()) == 1


# ===================================================================
# PipelineTemplate
# ===================================================================

class TestPipelineTemplate:
    """PipelineTemplate: build, frozen, equality."""

    def test_build_returns_graph(self) -> None:
        t = _make_template()
        result = t.build()
        assert isinstance(result, Graph)

    def test_frozen(self) -> None:
        t = _make_template()
        with pytest.raises(AttributeError):
            t.id = 'changed'

    def test_template_fields(self) -> None:
        t = _make_template(
            id='my_id',
            name='My Name',
            description='My description',
        )
        assert t.id == 'my_id'
        assert t.name == 'My Name'
        assert t.description == 'My description'
        assert t.knobs == {}


# ===================================================================
# @pipeline_template decorator
# ===================================================================

class TestPipelineTemplateDecorator:
    """Decorator auto-registers builder functions in the global registry."""

    def test_decorator_registers_in_global_registry(self) -> None:
        """The onset_detection template should be auto-registered via import."""
        import echozero.pipelines.templates.onset_detection  # noqa: F401
        reg = get_registry()
        assert reg.get('onset_detection') is not None

    def test_decorator_preserves_function(self) -> None:
        """Decorated function is still callable directly."""
        import echozero.pipelines.templates.onset_detection as mod
        from echozero.pipelines.pipeline import Pipeline
        result = mod.build_onset_detection()
        assert isinstance(result, Pipeline)

    def test_decorator_registers_metadata(self) -> None:
        reg = get_registry()
        t = reg.get('onset_detection')
        assert t is not None
        assert t.name == 'Onset Detection'
        assert t.description == 'Detect note onsets in audio'
        assert len(t.knobs) == 6


# ===================================================================
# Onset Detection template
# ===================================================================

class TestOnsetDetectionTemplate:
    """Onset detection template builds a valid 2-block graph."""

    @pytest.fixture
    def template(self) -> PipelineTemplate:
        import echozero.pipelines.templates.onset_detection  # noqa: F401
        reg = get_registry()
        t = reg.get('onset_detection')
        assert t is not None
        return t

    def test_builds_valid_graph(self, template: PipelineTemplate) -> None:
        g = template.build()
        assert isinstance(g, Graph)

    def test_has_two_blocks(self, template: PipelineTemplate) -> None:
        g = template.build()
        assert len(g.blocks) == 2

    def test_has_load_audio_block(self, template: PipelineTemplate) -> None:
        g = template.build()
        assert 'load_audio' in g.blocks

    def test_has_detect_onsets_block(self, template: PipelineTemplate) -> None:
        g = template.build()
        assert 'detect_onsets' in g.blocks

    def test_has_one_connection(self, template: PipelineTemplate) -> None:
        g = template.build()
        assert len(g.connections) == 1

    def test_connection_wiring(self, template: PipelineTemplate) -> None:
        g = template.build()
        conn = g.connections[0]
        assert conn.source_block_id == 'load_audio'
        assert conn.source_output_name == 'audio_out'
        assert conn.target_block_id == 'detect_onsets'
        assert conn.target_input_name == 'audio_in'

    def test_load_audio_settings(self, template: PipelineTemplate) -> None:
        g = template.build()
        load = g.blocks['load_audio']
        assert 'file_path' in load.settings
        assert 'target_sample_rate' in load.settings

    def test_detect_onsets_settings(self, template: PipelineTemplate) -> None:
        g = template.build()
        detect = g.blocks['detect_onsets']
        assert detect.settings['threshold'] == 0.3
        assert detect.settings['min_gap'] == 0.05
        assert detect.settings['method'] == 'default'
        assert detect.settings['backtrack'] is True
        assert detect.settings['timing_offset_ms'] == 0.0

    def test_knobs_audio_file(self, template: PipelineTemplate) -> None:
        k = template.knobs['audio_file']
        assert isinstance(k, Knob)
        assert k.widget == KnobWidget.FILE_PICKER

    def test_knobs_threshold(self, template: PipelineTemplate) -> None:
        k = template.knobs['threshold']
        assert isinstance(k, Knob)
        assert k.default == 0.3
        assert k.min_value == 0.0
        assert k.max_value == 1.0

    def test_knobs_method(self, template: PipelineTemplate) -> None:
        k = template.knobs['method']
        assert isinstance(k, Knob)
        assert k.widget == KnobWidget.DROPDOWN
        assert 'default' in k.options

    def test_knobs_backtrack(self, template: PipelineTemplate) -> None:
        k = template.knobs['backtrack']
        assert isinstance(k, Knob)
        assert k.default is True


class TestDrumClassificationTemplate:
    """Drum classification template exposes onset and classifier controls."""

    @pytest.fixture
    def template(self) -> PipelineTemplate:
        import echozero.pipelines.templates.drum_classification  # noqa: F401

        reg = get_registry()
        t = reg.get("drum_classification")
        assert t is not None
        return t

    def test_knobs_include_onset_and_classifier_controls(self, template: PipelineTemplate) -> None:
        assert "onset_threshold" in template.knobs
        assert "onset_min_gap" in template.knobs
        assert "onset_method" in template.knobs
        assert "onset_backtrack" in template.knobs
        assert "onset_timing_offset_ms" in template.knobs
        assert "classify_model_path" in template.knobs
        assert "classify_device" in template.knobs
        assert "classify_batch_size" in template.knobs

    def test_detect_onsets_block_receives_onset_knob_values(self, template: PipelineTemplate) -> None:
        pipeline = template.build_pipeline(
            bindings={
                "onset_threshold": 0.2,
                "onset_min_gap": 0.08,
                "onset_method": "hfc",
                "onset_backtrack": False,
                "onset_timing_offset_ms": -12.0,
                "classify_model_path": "model.pth",
            }
        )

        detect = pipeline.graph.blocks["detect_onsets"]
        assert detect.settings["threshold"] == 0.2
        assert detect.settings["min_gap"] == 0.08
        assert detect.settings["method"] == "hfc"
        assert detect.settings["backtrack"] is False
        assert detect.settings["timing_offset_ms"] == -12.0

    def test_classify_block_receives_audio_context(self, template: PipelineTemplate) -> None:
        pipeline = template.build_pipeline(bindings={"classify_model_path": "model.pth"})
        assert any(
            connection.source_block_id == "load_audio"
            and connection.target_block_id == "classify"
            and connection.target_input_name == "audio_in"
            for connection in pipeline.graph.connections
        )


# ===================================================================
# Knob
# ===================================================================

class TestKnob:
    """Knob dataclass: frozen, auto-inference, validation."""

    def test_basic_knob(self) -> None:
        k = knob(0.5, label='Test', min_value=0.0, max_value=1.0)
        assert isinstance(k, Knob)
        assert k.default == 0.5
        assert k.widget == KnobWidget.SLIDER

    def test_frozen(self) -> None:
        k = knob(0.5, label='Test', min_value=0.0, max_value=1.0)
        with pytest.raises(AttributeError):
            k.default = 0.9

    def test_auto_toggle(self) -> None:
        k = knob(True, label='Enabled')
        assert k.widget == KnobWidget.TOGGLE

    def test_auto_dropdown(self) -> None:
        k = knob('a', label='Choice', options=('a', 'b', 'c'))
        assert k.widget == KnobWidget.DROPDOWN

    def test_auto_text(self) -> None:
        k = knob('hello', label='Name')
        assert k.widget == KnobWidget.TEXT

    def test_frequency_knob(self) -> None:
        k = knob(440.0, widget=KnobWidget.FREQUENCY)
        assert k.min_value == 20.0
        assert k.max_value == 20000.0
        assert k.units == 'Hz'
        assert k.log_scale is True

    def test_gain_knob(self) -> None:
        k = knob(0.0, widget=KnobWidget.GAIN)
        assert k.min_value == -48.0
        assert k.max_value == 48.0
        assert k.units == 'dB'


# ===================================================================
# validate_bindings (via template)
# ===================================================================

class TestValidateBindings:
    """PipelineTemplate.validate_bindings using Knobs."""

    @pytest.fixture
    def template_with_knobs(self) -> PipelineTemplate:
        return _make_template(
            knobs={
                'threshold': knob(0.5, label='Threshold', min_value=0.0, max_value=1.0),
                'method': knob('default', label='Method',
                               widget=KnobWidget.DROPDOWN, options=('default', 'hfc')),
                'count': knob(10, label='Count', min_value=1, max_value=100),
            },
        )

    def test_valid_bindings(self, template_with_knobs: PipelineTemplate) -> None:
        errors = template_with_knobs.validate_bindings({
            'threshold': 0.8,
            'method': 'hfc',
            'count': 5,
        })
        assert errors == []

    def test_unknown_key(self, template_with_knobs: PipelineTemplate) -> None:
        errors = template_with_knobs.validate_bindings({'bogus': 42})
        assert len(errors) == 1
        assert 'bogus' in errors[0]

    def test_out_of_range(self, template_with_knobs: PipelineTemplate) -> None:
        errors = template_with_knobs.validate_bindings({'threshold': 5.0})
        assert len(errors) == 1
        assert 'threshold' in errors[0]

    def test_wrong_type(self, template_with_knobs: PipelineTemplate) -> None:
        errors = template_with_knobs.validate_bindings({'threshold': 'not_a_float'})
        assert len(errors) == 1
        assert 'threshold' in errors[0]

    def test_invalid_option(self, template_with_knobs: PipelineTemplate) -> None:
        errors = template_with_knobs.validate_bindings({'method': 'invalid'})
        assert len(errors) == 1
        assert 'method' in errors[0]

    def test_empty_bindings_ok(self, template_with_knobs: PipelineTemplate) -> None:
        errors = template_with_knobs.validate_bindings({})
        assert errors == []

    def test_no_knobs_reports_unknown(self) -> None:
        t = _make_template()
        errors = t.validate_bindings({'anything': 'goes'})
        assert errors == []  # No knobs = nothing to validate

    def test_int_to_float_coercion(self, template_with_knobs: PipelineTemplate) -> None:
        errors = template_with_knobs.validate_bindings({'threshold': 1})
        assert errors == []


# ===================================================================
# Pipeline output declarations
# ===================================================================

class TestPipelineOutputs:
    """Pipeline.output() declares named outputs on built pipelines."""

    def test_onset_detection_has_output(self) -> None:
        import echozero.pipelines.templates.onset_detection as mod
        p = mod.build_onset_detection()
        assert len(p.outputs) == 1
        assert p.outputs[0].name == 'onsets'
        assert p.outputs[0].port_ref.block_id == 'detect_onsets'
        assert p.outputs[0].port_ref.port_name == 'events_out'

    def test_stem_separation_has_four_outputs(self) -> None:
        import echozero.pipelines.templates.stem_separation as mod
        p = mod.build_stem_separation()
        names = [o.name for o in p.outputs]
        assert names == ['drums', 'bass', 'vocals', 'other']

    def test_stem_separation_output_refs(self) -> None:
        import echozero.pipelines.templates.stem_separation as mod
        p = mod.build_stem_separation()
        for out in p.outputs:
            assert out.port_ref.block_id == 'separate'
            assert out.port_ref.port_name == f'{out.name}_out'

    def test_build_pipeline_returns_pipeline_with_outputs(self) -> None:
        reg = get_registry()
        t = reg.get('onset_detection')
        p = t.build_pipeline()
        assert len(p.outputs) == 1
        assert p.outputs[0].name == 'onsets'


# ===================================================================
# Full Analysis template
# ===================================================================

class TestFullAnalysisTemplate:
    """Full analysis pipeline: separate → detect onsets per stem."""

    @pytest.fixture
    def template(self) -> PipelineTemplate:
        import echozero.pipelines.templates.full_analysis  # noqa: F401
        reg = get_registry()
        t = reg.get('full_analysis')
        assert t is not None
        return t

    def test_registered(self, template: PipelineTemplate) -> None:
        assert template.id == 'full_analysis'
        assert template.name == 'Full Analysis'

    def test_builds_valid_graph(self, template: PipelineTemplate) -> None:
        g = template.build()
        assert isinstance(g, Graph)

    def test_has_six_blocks(self, template: PipelineTemplate) -> None:
        """load + sep + 4 onset detectors = 6 blocks (no classify without model)."""
        g = template.build()
        assert len(g.blocks) == 6

    def test_has_five_connections(self, template: PipelineTemplate) -> None:
        """load→sep + sep→4 detectors = 5 connections."""
        g = template.build()
        assert len(g.connections) == 5

    def test_outputs_without_classify(self, template: PipelineTemplate) -> None:
        p = template.build_pipeline()
        names = [o.name for o in p.outputs]
        assert 'drums_onsets' in names
        assert 'bass_onsets' in names
        assert 'vocals_onsets' in names
        assert 'other_onsets' in names
        assert len(names) == 4

    def test_outputs_with_classify(self, template: PipelineTemplate) -> None:
        p = template.build_pipeline(bindings={'classify_model_path': 'model.pth'})
        names = [o.name for o in p.outputs]
        assert 'drums_classified' in names
        assert 'drums_onsets' not in names
        assert len(names) == 4

    def test_has_seven_blocks_with_classify(self, template: PipelineTemplate) -> None:
        g = template.build(bindings={'classify_model_path': 'model.pth'})
        assert len(g.blocks) == 7

    def test_knobs(self, template: PipelineTemplate) -> None:
        assert 'threshold' in template.knobs
        assert 'model' in template.knobs
        assert 'device' in template.knobs
        assert 'classify_model_path' in template.knobs

    def test_topological_sort_valid(self, template: PipelineTemplate) -> None:
        g = template.build()
        order = g.topological_sort()
        # load must come before sep, sep before onsets
        assert order.index('load') < order.index('sep')
        assert order.index('sep') < order.index('drums_onsets')
        assert order.index('sep') < order.index('bass_onsets')

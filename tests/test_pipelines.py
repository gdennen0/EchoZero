"""
Pipeline infrastructure tests: registry, decorator, templates, promoted params, validation.
Exercises PipelineRegistry, @pipeline_template decorator, and onset_detection template.
"""

from __future__ import annotations

import pytest

from echozero.domain.graph import Graph
from echozero.pipelines.registry import (
    PipelineRegistry,
    PipelineTemplate,
    PromotedParam,
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
    promoted_params: tuple[PromotedParam, ...] = (),
) -> PipelineTemplate:
    return PipelineTemplate(
        id=id,
        name=name,
        description=description,
        promoted_params=promoted_params,
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
        assert t.promoted_params == ()


# ===================================================================
# @pipeline_template decorator
# ===================================================================

class TestPipelineTemplateDecorator:
    """Decorator auto-registers builder functions in the global registry."""

    def test_decorator_registers_in_global_registry(self) -> None:
        """The onset_detection template should be auto-registered via import."""
        # Force the import to ensure registration happens
        import echozero.pipelines.templates.onset_detection  # noqa: F401
        reg = get_registry()
        assert reg.get('onset_detection') is not None

    def test_decorator_preserves_function(self) -> None:
        """Decorated function is still callable directly."""
        import echozero.pipelines.templates.onset_detection as mod
        result = mod.build_onset_detection()
        assert isinstance(result, Graph)

    def test_decorator_registers_metadata(self) -> None:
        reg = get_registry()
        t = reg.get('onset_detection')
        assert t is not None
        assert t.name == 'Onset Detection'
        assert t.description == 'Detect note onsets in audio'
        assert len(t.promoted_params) == 3


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
        assert conn.source_output_name == 'audio'
        assert conn.target_block_id == 'detect_onsets'
        assert conn.target_input_name == 'audio'

    def test_load_audio_settings(self, template: PipelineTemplate) -> None:
        g = template.build()
        load = g.blocks['load_audio']
        assert 'file_path' in load.settings.entries
        assert 'target_sample_rate' in load.settings.entries

    def test_detect_onsets_settings(self, template: PipelineTemplate) -> None:
        g = template.build()
        detect = g.blocks['detect_onsets']
        assert detect.settings.entries['threshold'] == 0.3
        assert detect.settings.entries['method'] == 'default'

    def test_promoted_params_audio_file(self, template: PipelineTemplate) -> None:
        params = {p.key: p for p in template.promoted_params}
        p = params['audio_file']
        assert p.required is True
        assert p.type is str
        assert p.name == 'Audio File'

    def test_promoted_params_threshold(self, template: PipelineTemplate) -> None:
        params = {p.key: p for p in template.promoted_params}
        p = params['threshold']
        assert p.required is False
        assert p.type is float
        assert p.default == 0.3

    def test_promoted_params_method(self, template: PipelineTemplate) -> None:
        params = {p.key: p for p in template.promoted_params}
        p = params['method']
        assert p.required is False
        assert p.type is str
        assert p.default == 'default'


# ===================================================================
# PromotedParam
# ===================================================================

class TestPromotedParam:
    """PromotedParam dataclass: defaults, frozen, fields."""

    def test_defaults(self) -> None:
        p = PromotedParam('key', 'Name', str)
        assert p.required is False
        assert p.default is None
        assert p.description == ''

    def test_frozen(self) -> None:
        p = PromotedParam('key', 'Name', str)
        with pytest.raises(AttributeError):
            p.key = 'other'

    def test_all_fields(self) -> None:
        p = PromotedParam(
            key='k',
            name='N',
            type=float,
            required=True,
            default=1.5,
            description='desc',
        )
        assert p.key == 'k'
        assert p.name == 'N'
        assert p.type is float
        assert p.required is True
        assert p.default == 1.5
        assert p.description == 'desc'

    def test_equality(self) -> None:
        p1 = PromotedParam('k', 'N', str)
        p2 = PromotedParam('k', 'N', str)
        assert p1 == p2


# ===================================================================
# validate_bindings
# ===================================================================

class TestValidateBindings:
    """PipelineTemplate.validate_bindings: required, type checking, valid."""

    @pytest.fixture
    def template_with_params(self) -> PipelineTemplate:
        return _make_template(
            promoted_params=(
                PromotedParam('file', 'File', str, required=True),
                PromotedParam('threshold', 'Threshold', float, default=0.5),
                PromotedParam('count', 'Count', int, default=10),
            ),
        )

    def test_valid_bindings_no_errors(self, template_with_params: PipelineTemplate) -> None:
        errors = template_with_params.validate_bindings({
            'file': '/path/to/audio.wav',
            'threshold': 0.8,
            'count': 5,
        })
        assert errors == []

    def test_valid_with_only_required(self, template_with_params: PipelineTemplate) -> None:
        errors = template_with_params.validate_bindings({'file': 'test.wav'})
        assert errors == []

    def test_missing_required_param(self, template_with_params: PipelineTemplate) -> None:
        errors = template_with_params.validate_bindings({})
        assert len(errors) == 1
        assert 'file' in errors[0].lower() or 'File' in errors[0]

    def test_wrong_type_string_for_float(self, template_with_params: PipelineTemplate) -> None:
        errors = template_with_params.validate_bindings({
            'file': 'test.wav',
            'threshold': 'not_a_float',
        })
        assert len(errors) == 1
        assert 'threshold' in errors[0]

    def test_wrong_type_float_for_int(self, template_with_params: PipelineTemplate) -> None:
        errors = template_with_params.validate_bindings({
            'file': 'test.wav',
            'count': 3.14,
        })
        assert len(errors) == 1
        assert 'count' in errors[0]

    def test_multiple_errors(self, template_with_params: PipelineTemplate) -> None:
        errors = template_with_params.validate_bindings({
            'threshold': 'bad',
            'count': 'bad',
        })
        # missing required 'file' + wrong type for threshold + wrong type for count
        assert len(errors) == 3

    def test_empty_bindings_only_required_fails(self, template_with_params: PipelineTemplate) -> None:
        errors = template_with_params.validate_bindings({})
        # Only the required param should fail
        assert len(errors) == 1

    def test_extra_bindings_ignored(self, template_with_params: PipelineTemplate) -> None:
        errors = template_with_params.validate_bindings({
            'file': 'test.wav',
            'unknown_key': 42,
        })
        assert errors == []

    def test_no_promoted_params(self) -> None:
        t = _make_template()
        errors = t.validate_bindings({'anything': 'goes'})
        assert errors == []

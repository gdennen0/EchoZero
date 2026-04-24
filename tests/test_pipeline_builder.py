"""Tests for the new Pipeline construction API (pipelines/pipeline.py + block_specs.py)."""

from __future__ import annotations

import pytest

from echozero.domain.enums import Direction, PortType
from echozero.errors import ValidationError
from echozero.pipelines.block_specs import (
    AudioFilter,
    BlockSpec,
    Classify,
    DetectOnsets,
    LoadAudio,
    PortSpec,
    Separator,
)
from echozero.pipelines.params import Knob, KnobWidget, knob
from echozero.pipelines.pipeline import BlockHandle, Pipeline, PipelineOutput, PortRef
from echozero.pipelines.registry import get_registry, pipeline_template


# ---------------------------------------------------------------------------
# BlockHandle / PortRef basics
# ---------------------------------------------------------------------------


class TestBlockHandle:
    def test_attribute_access_returns_port_ref(self):
        handle = BlockHandle("load_1", {"audio_out": "AUDIO"})
        ref = handle.audio_out
        assert isinstance(ref, PortRef)
        assert ref.block_id == "load_1"
        assert ref.port_name == "audio_out"

    def test_invalid_attribute_raises(self):
        handle = BlockHandle("load_1", {"audio_out": "AUDIO"})
        with pytest.raises(AttributeError, match="no output port 'bogus'"):
            _ = handle.bogus

    def test_multi_output_attribute_access(self):
        handle = BlockHandle("sep_1", {
            "drums_out": "AUDIO",
            "bass_out": "AUDIO",
            "vocals_out": "AUDIO",
            "other_out": "AUDIO",
        })
        drums = handle.drums_out
        assert isinstance(drums, PortRef)
        assert drums.block_id == "sep_1"
        assert drums.port_name == "drums_out"

    def test_repr(self):
        handle = BlockHandle("load_1", {"audio_out": "AUDIO"})
        assert "load_1" in repr(handle)


class TestPortRef:
    def test_fields(self):
        ref = PortRef("block_1", "audio_out")
        assert ref.block_id == "block_1"
        assert ref.port_name == "audio_out"

    def test_repr(self):
        ref = PortRef("block_1", "audio_out")
        assert "block_1" in repr(ref)
        assert "audio_out" in repr(ref)


# ---------------------------------------------------------------------------
# Pipeline.add() — block creation
# ---------------------------------------------------------------------------


class TestPipelineAdd:
    def test_add_creates_block_in_graph(self):
        p = Pipeline("test")
        p.add(LoadAudio())
        assert len(p.graph.blocks) == 1
        block = list(p.graph.blocks.values())[0]
        assert block.block_type == "LoadAudio"

    def test_add_returns_block_handle(self):
        p = Pipeline("test")
        handle = p.add(LoadAudio())
        assert isinstance(handle, BlockHandle)

    def test_add_with_connection(self):
        p = Pipeline("test")
        load = p.add(LoadAudio())
        p.add(DetectOnsets(threshold=0.3), audio_in=load.audio_out)

        assert len(p.graph.blocks) == 2
        assert len(p.graph.connections) == 1
        conn = p.graph.connections[0]
        assert conn.source_output_name == "audio_out"
        assert conn.target_input_name == "audio_in"

    def test_add_settings_passed_through(self):
        p = Pipeline("test")
        p.add(DetectOnsets(threshold=0.5, min_gap=0.05), id="det")
        block = p.graph.blocks["det"]
        assert block.settings["threshold"] == 0.5
        assert block.settings["min_gap"] == 0.05

    def test_add_auto_id_generation(self):
        p = Pipeline("test")
        p.add(DetectOnsets())
        p.add(DetectOnsets())
        block_ids = sorted(p.graph.blocks.keys())
        assert block_ids == ["DetectOnsets_1", "DetectOnsets_2"]

    def test_add_explicit_id(self):
        p = Pipeline("test")
        p.add(LoadAudio(), id="my_audio")
        assert "my_audio" in p.graph.blocks

    def test_add_duplicate_id_raises(self):
        p = Pipeline("test")
        p.add(LoadAudio(), id="dupe")
        with pytest.raises(ValidationError, match="Duplicate block ID"):
            p.add(DetectOnsets(), id="dupe")

    def test_add_mixed_explicit_and_auto_ids(self):
        p = Pipeline("test")
        p.add(LoadAudio(), id="my_audio")
        p.add(DetectOnsets())  # auto
        assert "my_audio" in p.graph.blocks
        auto_ids = [bid for bid in p.graph.blocks if bid.startswith("DetectOnsets_")]
        assert len(auto_ids) == 1

    def test_add_invalid_connection_type_raises(self):
        p = Pipeline("test")
        with pytest.raises(ValidationError, match="expected a PortRef"):
            p.add(DetectOnsets(), audio_in="not_a_port_ref")

    def test_knob_defaults_resolved_in_settings(self):
        """Knob objects passed as settings should be resolved to their .default values."""
        threshold_knob = knob(0.3, label="Sensitivity", min_value=0.0, max_value=1.0)
        p = Pipeline("test")
        p.add(DetectOnsets(threshold=threshold_knob), id="det")
        block = p.graph.blocks["det"]
        assert block.settings["threshold"] == 0.3


# ---------------------------------------------------------------------------
# Pipeline.output() — named outputs
# ---------------------------------------------------------------------------


class TestPipelineOutput:
    def test_output_registers_named_output(self):
        p = Pipeline("test")
        load = p.add(LoadAudio())
        onsets = p.add(DetectOnsets(), audio_in=load.audio_out)
        p.output("onsets", onsets.events_out)

        assert len(p.outputs) == 1
        out = p.outputs[0]
        assert out.name == "onsets"
        assert out.port_ref.port_name == "events_out"

    def test_output_with_block_handle_uses_first_port(self):
        p = Pipeline("test")
        load = p.add(LoadAudio())
        p.output("audio", load)

        out = p.outputs[0]
        assert out.port_ref.port_name == "audio_out"

    def test_multiple_outputs(self):
        p = Pipeline("test")
        load = p.add(LoadAudio())
        stems = p.add(Separator(), audio_in=load.audio_out)
        p.output("drums", stems.drums_out)
        p.output("bass", stems.bass_out)

        assert len(p.outputs) == 2
        names = [o.name for o in p.outputs]
        assert "drums" in names
        assert "bass" in names


# ---------------------------------------------------------------------------
# Separator multi-output
# ---------------------------------------------------------------------------


class TestSeparator:
    def test_separator_multi_output(self):
        p = Pipeline("test")
        load = p.add(LoadAudio())
        stems = p.add(Separator(), audio_in=load.audio_out)
        p.add(DetectOnsets(), audio_in=stems.drums_out)

        assert len(p.graph.blocks) == 3
        assert len(p.graph.connections) == 2

    def test_separator_multiple_stems(self):
        p = Pipeline("test")
        load = p.add(LoadAudio())
        stems = p.add(Separator(), audio_in=load.audio_out)
        p.add(DetectOnsets(), audio_in=stems.drums_out)
        p.add(DetectOnsets(), audio_in=stems.bass_out)

        assert len(p.graph.blocks) == 4
        assert len(p.graph.connections) == 3

    def test_separator_bad_port_raises(self):
        p = Pipeline("test")
        load = p.add(LoadAudio())
        stems = p.add(Separator(), audio_in=load.audio_out)
        with pytest.raises(AttributeError, match="no output port 'piano_out'"):
            _ = stems.piano_out


# ---------------------------------------------------------------------------
# Full pipeline construction
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_analyze_all_structure(self):
        """Full pipeline: load → separate → detect onsets on drums + bass → classify drums."""
        p = Pipeline("analyze_all", name="Full Analysis")
        load = p.add(LoadAudio())
        stems = p.add(Separator(device="auto"), audio_in=load.audio_out)
        drums_onsets = p.add(DetectOnsets(threshold=0.3), audio_in=stems.drums_out)
        drums_classified = p.add(
            Classify(device="auto"),
            events_in=drums_onsets.events_out,
            audio_in=stems.drums_out,
        )
        bass_onsets = p.add(DetectOnsets(), audio_in=stems.bass_out)

        p.output("drums_classified", drums_classified.events_out)
        p.output("bass_onsets", bass_onsets.events_out)

        # 5 blocks: LoadAudio, SeparateAudio, DetectOnsets x2, PyTorchAudioClassify
        assert len(p.graph.blocks) == 5
        block_types = [b.block_type for b in p.graph.blocks.values()]
        assert block_types.count("LoadAudio") == 1
        assert block_types.count("SeparateAudio") == 1
        assert block_types.count("DetectOnsets") == 2
        assert block_types.count("PyTorchAudioClassify") == 1

        # 5 connections
        assert len(p.graph.connections) == 5

        # Valid DAG
        order = p.graph.topological_sort()
        assert len(order) == 5

        # 2 named outputs
        assert len(p.outputs) == 2

    def test_pipeline_with_audio_filter(self):
        """Audio filter in chain."""
        p = Pipeline("filtered")
        load = p.add(LoadAudio())
        filtered = p.add(AudioFilter(filter_type="highpass", freq=100.0), audio_in=load.audio_out)
        p.add(DetectOnsets(), audio_in=filtered.audio_out)

        assert len(p.graph.blocks) == 3
        filter_block = [b for b in p.graph.blocks.values() if b.block_type == "AudioFilter"][0]
        assert filter_block.settings["filter_type"] == "highpass"


# ---------------------------------------------------------------------------
# Pipeline with knobs / rebuild
# ---------------------------------------------------------------------------


class TestPipelineKnobs:
    def test_pipeline_with_knobs_rebuilds_with_different_values(self):
        """Pipeline templates should produce different graphs with different knob values."""

        @pipeline_template(id="test_knob_rebuild", name="Knob Rebuild Test")
        def test_knob_rebuild(
            threshold=knob(0.3, label="Sensitivity", min_value=0.0, max_value=1.0),
        ):
            p = Pipeline("test_knob_rebuild")
            load = p.add(LoadAudio())
            p.add(DetectOnsets(threshold=threshold), id="onsets", audio_in=load.audio_out)
            return p

        template = get_registry().get("test_knob_rebuild")

        # Build with default
        graph_default = template.build()
        assert graph_default.blocks["onsets"].settings["threshold"] == 0.3

        # Build with override
        graph_override = template.build(bindings={"threshold": 0.7})
        assert graph_override.blocks["onsets"].settings["threshold"] == 0.7

    def test_partial_bindings(self):
        """Only bound knobs should change; others keep defaults."""

        @pipeline_template(id="test_partial_bind_new", name="Partial Bind")
        def test_partial_bind_new(
            threshold=knob(0.3, label="Threshold", min_value=0.0, max_value=1.0),
            device=knob("auto", label="Device", widget=KnobWidget.DROPDOWN, options=("auto", "cpu", "cuda")),
        ):
            p = Pipeline("test_partial_bind_new")
            load = p.add(LoadAudio())
            p.add(DetectOnsets(threshold=threshold), id="onsets", audio_in=load.audio_out)
            p.add(Separator(device=device), id="sep", audio_in=load.audio_out)
            return p

        template = get_registry().get("test_partial_bind_new")
        graph = template.build(bindings={"threshold": 0.8})

        assert graph.blocks["onsets"].settings["threshold"] == 0.8
        assert graph.blocks["sep"].settings["device"] == "auto"


# ---------------------------------------------------------------------------
# Graph validity
# ---------------------------------------------------------------------------


class TestGraphValidity:
    def test_built_graph_is_acyclic(self):
        p = Pipeline("test")
        load = p.add(LoadAudio())
        onsets = p.add(DetectOnsets(), audio_in=load.audio_out)
        p.add(Classify(), events_in=onsets.events_out)

        assert not p.graph.has_cycle()
        order = p.graph.topological_sort()
        assert len(order) == 3

    def test_topological_order_respects_dependencies(self):
        p = Pipeline("test")
        load = p.add(LoadAudio(), id="load")
        onsets = p.add(DetectOnsets(), id="onsets", audio_in=load.audio_out)
        p.add(Classify(), id="classify", events_in=onsets.events_out)

        order = p.graph.topological_sort()
        load_idx = order.index("load")
        onset_idx = order.index("onsets")
        classify_idx = order.index("classify")
        assert load_idx < onset_idx < classify_idx


# ---------------------------------------------------------------------------
# BlockSpec validation
# ---------------------------------------------------------------------------


class TestBlockSpecs:
    def test_load_audio_spec(self):
        spec = LoadAudio()
        assert spec.block_type == "LoadAudio"
        assert len(spec.input_ports) == 0
        assert len(spec.output_ports) == 1
        assert spec.output_ports[0].name == "audio_out"

    def test_separator_spec_has_four_outputs(self):
        spec = Separator()
        output_names = {p.name for p in spec.output_ports}
        assert output_names == {"drums_out", "bass_out", "vocals_out", "other_out"}

    def test_detect_onsets_spec(self):
        spec = DetectOnsets(threshold=0.5)
        assert spec.block_type == "DetectOnsets"
        assert spec.settings["threshold"] == 0.5

    def test_classify_spec_accepts_audio_context(self):
        spec = Classify(model_path="/tmp/model.manifest.json")
        input_names = [port.name for port in spec.input_ports]
        assert input_names == ["events_in", "audio_in"]

    def test_settings_override(self):
        spec = LoadAudio(file_path="/test.wav", target_sample_rate=22050)
        assert spec.settings["file_path"] == "/test.wav"
        assert spec.settings["target_sample_rate"] == 22050

    def test_all_specs_port_directions(self):
        """All input ports should have direction INPUT, output ports OUTPUT."""
        from echozero.domain.enums import Direction
        specs = [LoadAudio(), Separator(), DetectOnsets(), AudioFilter(), Classify()]
        for spec in specs:
            for port in spec.input_ports:
                assert port.direction == Direction.INPUT, f"{spec.block_type} input port {port.name}"
            for port in spec.output_ports:
                assert port.direction == Direction.OUTPUT, f"{spec.block_type} output port {port.name}"

    def test_output_port_names_follow_convention(self):
        """Output ports should end with '_out', input ports with '_in'."""
        specs = [LoadAudio(), Separator(), DetectOnsets(), AudioFilter(), Classify()]
        for spec in specs:
            for port in spec.output_ports:
                assert port.name.endswith("_out"), f"{spec.block_type} output port {port.name}"
            for port in spec.input_ports:
                assert port.name.endswith("_in"), f"{spec.block_type} input port {port.name}"

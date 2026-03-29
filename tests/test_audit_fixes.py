"""
Tests for audit fixes (WI-1 through WI-8).
Verifies immutability, boundary enforcement, and serialization round-trips.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import (
    AudioData, Block, BlockSettings, Connection, Event, EventData, Layer, Port,
)
from echozero.errors import ValidationError
from echozero.pipelines.block_specs import (
    AudioFilter, AudioNegate, Classify, DatasetViewer, DetectOnsets,
    EQBands, ExportAudio, ExportAudioDataset, ExportMA2, LoadAudio,
    PortSpec, Separator, TranscribeNotes,
)
from echozero.pipelines.pipeline import Pipeline
from echozero.serialization import (
    deserialize_pipeline, serialize_graph, serialize_pipeline,
)


# ===================================================================
# WI-1: BlockSettings immutability
# ===================================================================

class TestBlockSettingsImmutable:
    def test_item_not_assignable(self):
        bs = BlockSettings({"a": 1})
        with pytest.raises(TypeError):
            bs["a"] = 2

    def test_item_not_deletable(self):
        bs = BlockSettings({"a": 1})
        with pytest.raises(TypeError):
            del bs["a"]

    def test_getitem(self):
        bs = BlockSettings({"a": 1, "b": "hello"})
        assert bs["a"] == 1
        assert bs["b"] == "hello"

    def test_keys(self):
        bs = BlockSettings({"x": 1, "y": 2})
        assert set(bs.keys()) == {"x", "y"}

    def test_get_with_default(self):
        bs = BlockSettings({"a": 1})
        assert bs.get("a") == 1
        assert bs.get("missing", 42) == 42

    def test_attribute_not_settable(self):
        bs = BlockSettings({"a": 1})
        with pytest.raises(AttributeError):
            bs.foo = "bar"

    def test_constructor_copies_input(self):
        d = {"a": 1}
        bs = BlockSettings(d)
        d["a"] = 999
        assert bs["a"] == 1

    def test_default_empty(self):
        bs = BlockSettings()
        assert len(bs) == 0

    def test_equality(self):
        assert BlockSettings({"a": 1}) == BlockSettings({"a": 1})

    def test_equality_with_dict(self):
        assert BlockSettings({"a": 1}) == {"a": 1}

    def test_inequality(self):
        assert BlockSettings({"a": 1}) != BlockSettings({"a": 2})

    def test_json_round_trip(self):
        bs = BlockSettings({"threshold": 0.3, "method": "default"})
        assert json.loads(json.dumps(dict(bs))) == {"threshold": 0.3, "method": "default"}

    def test_contains(self):
        bs = BlockSettings({"a": 1})
        assert "a" in bs
        assert "b" not in bs

    def test_len(self):
        assert len(BlockSettings({"a": 1, "b": 2})) == 2

    def test_iter(self):
        assert set(BlockSettings({"a": 1, "b": 2})) == {"a", "b"}


# ===================================================================
# WI-2: Graph.blocks read-only
# ===================================================================

class TestGraphBlocksReadOnly:
    def test_blocks_not_directly_assignable(self):
        g = Graph()
        g.add_block(Block(
            id="a", name="A", block_type="T",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(),
        ))
        with pytest.raises(TypeError):
            g.blocks["injected"] = Block(
                id="injected", name="X", block_type="T",
                category=BlockCategory.PROCESSOR,
                input_ports=(), output_ports=(),
            )

    def test_blocks_not_deletable(self):
        g = Graph()
        g.add_block(Block(
            id="a", name="A", block_type="T",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(),
        ))
        with pytest.raises(TypeError):
            del g.blocks["a"]

    def test_blocks_readable(self):
        g = Graph()
        g.add_block(Block(
            id="a", name="A", block_type="T",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(),
        ))
        assert "a" in g.blocks
        assert g.blocks["a"].name == "A"

    def test_add_block_still_works(self):
        g = Graph()
        g.add_block(Block(
            id="a", name="A", block_type="T",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(),
        ))
        assert len(g.blocks) == 1

    def test_remove_block_still_works(self):
        g = Graph()
        g.add_block(Block(
            id="a", name="A", block_type="T",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(),
        ))
        g.remove_block("a")
        assert len(g.blocks) == 0

    def test_replace_block_works(self):
        g = Graph()
        g.add_block(Block(
            id="a", name="A", block_type="T",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(),
        ))
        g.replace_block(Block(
            id="a", name="A Updated", block_type="T",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(),
        ))
        assert g.blocks["a"].name == "A Updated"

    def test_replace_block_nonexistent_raises(self):
        g = Graph()
        with pytest.raises(ValidationError):
            g.replace_block(Block(
                id="nope", name="N", block_type="T",
                category=BlockCategory.PROCESSOR,
                input_ports=(), output_ports=(),
            ))

    def test_connections_returns_copy(self):
        g = Graph()
        g.add_block(Block(
            id="a", name="A", block_type="T",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(Port("out", PortType.AUDIO, Direction.OUTPUT),),
        ))
        g.add_block(Block(
            id="b", name="B", block_type="T",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(),
        ))
        g.add_connection(Connection("a", "out", "b", "in"))
        conns = g.connections
        conns.clear()  # mutate the copy
        assert len(g.connections) == 1  # original unchanged


# ===================================================================
# WI-3: Duplicate pipeline output names
# ===================================================================

class TestDuplicateOutputNames:
    def test_duplicate_name_raises(self):
        p = Pipeline("test")
        load = p.add(LoadAudio(file_path="x.wav"), id="load")
        p.output("onsets", load.audio_out)
        with pytest.raises(ValidationError, match="Duplicate"):
            p.output("onsets", load.audio_out)

    def test_different_names_allowed(self):
        p = Pipeline("test")
        load = p.add(LoadAudio(file_path="x.wav"), id="load")
        p.output("output_a", load.audio_out)
        p.output("output_b", load.audio_out)
        assert len(p.outputs) == 2


# ===================================================================
# WI-7: Pipeline serialization round-trip
# ===================================================================

class TestPipelineSerializationRoundTrip:
    def test_serialize_deserialize_onset_pipeline(self):
        from echozero.pipelines.templates.onset_detection import build_onset_detection
        p = build_onset_detection(audio_file="test.wav", threshold=0.5)

        data = serialize_pipeline(p)
        p2 = deserialize_pipeline(data)

        assert p2.id == p.id
        assert p2.name == p.name
        assert len(p2.graph.blocks) == len(p.graph.blocks)
        assert len(p2.graph.connections) == len(p.graph.connections)
        assert len(p2.outputs) == len(p.outputs)
        assert p2.outputs[0].name == "onsets"
        assert p2.outputs[0].port_ref.block_id == "detect_onsets"
        assert p2.outputs[0].port_ref.port_name == "events_out"

    def test_serialize_deserialize_multi_output(self):
        from echozero.pipelines.templates.full_analysis import build_full_analysis
        p = build_full_analysis(audio_file="test.wav")

        data = serialize_pipeline(p)
        p2 = deserialize_pipeline(data)

        assert len(p2.outputs) == 4
        output_names = [o.name for o in p2.outputs]
        assert "drums_onsets" in output_names
        assert "bass_onsets" in output_names

    def test_json_round_trip(self):
        from echozero.pipelines.templates.onset_detection import build_onset_detection
        p = build_onset_detection(audio_file="test.wav")

        data = serialize_pipeline(p)
        json_str = json.dumps(data)
        data2 = json.loads(json_str)
        p2 = deserialize_pipeline(data2)

        assert p2.id == p.id
        assert len(p2.outputs) == 1

    def test_empty_pipeline_round_trip(self):
        p = Pipeline("empty", name="Empty")
        data = serialize_pipeline(p)
        p2 = deserialize_pipeline(data)
        assert p2.id == "empty"
        assert len(p2.outputs) == 0
        assert len(p2.graph.blocks) == 0


# ===================================================================
# WI-8: PortSpec uses enums
# ===================================================================

class TestPortSpecEnums:
    def test_port_spec_uses_port_type_enum(self):
        spec = LoadAudio()
        for port in spec.output_ports:
            assert isinstance(port.port_type, PortType)

    def test_port_spec_uses_direction_enum(self):
        spec = DetectOnsets()
        for port in spec.input_ports:
            assert isinstance(port.direction, Direction)
        for port in spec.output_ports:
            assert isinstance(port.direction, Direction)

    def test_all_specs_use_enums(self):
        """Every block spec should use enum types, never strings."""
        all_specs = [
            LoadAudio(), Separator(), DetectOnsets(), AudioFilter(),
            Classify(), TranscribeNotes(), ExportMA2(), ExportAudio(),
            EQBands(), AudioNegate(), ExportAudioDataset(), DatasetViewer(),
        ]
        for spec in all_specs:
            for port in spec.input_ports + spec.output_ports:
                assert isinstance(port.port_type, PortType), (
                    f"{spec.block_type} port {port.name}: "
                    f"port_type is {type(port.port_type)}, expected PortType"
                )
                assert isinstance(port.direction, Direction), (
                    f"{spec.block_type} port {port.name}: "
                    f"direction is {type(port.direction)}, expected Direction"
                )


# ===================================================================
# WI-5: Take limit (tested via Orchestrator)
# ===================================================================

class TestTakeLimit:
    def test_default_limit_exists(self):
        from echozero.services.orchestrator import Orchestrator
        assert Orchestrator.DEFAULT_MAX_TAKES_PER_LAYER == 20


"""
Serialization tests: Verify round-trip fidelity for Graph and Take persistence.
Exists because data loss during save/load is catastrophic — every field must survive the round-trip.
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import (
    AudioData,
    Block,
    BlockSettings,
    Connection,
    Event,
    EventData,
    Layer,
    Port,
)
from echozero.takes import Take, TakeLayer, TakeSource
from echozero.serialization import (
    deserialize_graph,
    deserialize_take,
    deserialize_take_layer,
    load_project,
    save_project,
    serialize_graph,
    serialize_take,
    serialize_take_layer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_out(name: str = "audio_out") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _audio_in(name: str = "audio_in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


def _event_out(name: str = "event_out") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.OUTPUT)


def _event_in(name: str = "event_in") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.INPUT)


def _make_block(
    block_id: str,
    block_type: str = "LoadAudio",
    input_ports: tuple[Port, ...] = (),
    output_ports: tuple[Port, ...] = (),
    control_ports: tuple[Port, ...] = (),
    settings: dict[str, Any] | None = None,
) -> Block:
    return Block(
        id=block_id,
        name=f"Block {block_id}",
        block_type=block_type,
        category=BlockCategory.PROCESSOR,
        input_ports=input_ports,
        output_ports=output_ports,
        control_ports=control_ports,
        settings=BlockSettings(settings or {}),
    )


def _make_event(time: float, label: str = "test") -> Event:
    return Event(
        id=f"evt-{time}",
        time=time,
        duration=0.1,
        classifications={"type": label},
        metadata={"conf": 0.9},
        origin="pipeline",
    )


def _make_event_data(*times: float) -> EventData:
    events = tuple(_make_event(t) for t in times)
    layer = Layer(id="layer-1", name="Onsets", events=events)
    return EventData(layers=(layer,))


def _make_take(
    label: str = "Test Take",
    times: tuple[float, ...] = (1.0, 2.0),
    is_main: bool = False,
    origin: str = "pipeline",
    with_source: bool = True,
) -> Take:
    source = None
    if with_source:
        source = TakeSource(
            block_id="block-1",
            block_type="DetectOnsets",
            settings_snapshot={"threshold": 0.5, "hop_size": 512},
            run_id="run-001",
        )
    return Take.create(
        data=_make_event_data(*times),
        label=label,
        origin=origin,
        source=source,
        is_main=is_main,
    )


# ---------------------------------------------------------------------------
# Graph round-trip
# ---------------------------------------------------------------------------


class TestGraphSerialization:
    """Verify Graph serialize/deserialize round-trip fidelity."""

    def test_round_trip_single_block(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))

        data = serialize_graph(graph)
        restored = deserialize_graph(data)

        assert "a" in restored.blocks
        assert restored.blocks["a"].name == "Block a"
        assert restored.blocks["a"].block_type == "LoadAudio"

    def test_round_trip_preserves_all_block_fields(self) -> None:
        graph = Graph()
        graph.add_block(
            _make_block(
                "b",
                block_type="DetectOnsets",
                input_ports=(_audio_in(),),
                output_ports=(_event_out(),),
                settings={"threshold": 0.8, "min_gap": 0.1},
            )
        )

        data = serialize_graph(graph)
        restored = deserialize_graph(data)

        block = restored.blocks["b"]
        assert block.block_type == "DetectOnsets"
        assert block.category == BlockCategory.PROCESSOR
        assert len(block.input_ports) == 1
        assert block.input_ports[0].name == "audio_in"
        assert block.input_ports[0].port_type == PortType.AUDIO
        assert block.input_ports[0].direction == Direction.INPUT
        assert len(block.output_ports) == 1
        assert block.output_ports[0].port_type == PortType.EVENT
        assert block.settings == {"threshold": 0.8, "min_gap": 0.1}

    def test_connections_survive_round_trip(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="audio_out",
                target_block_id="b",
                target_input_name="audio_in",
            )
        )

        data = serialize_graph(graph)
        restored = deserialize_graph(data)

        assert len(restored.connections) == 1
        conn = restored.connections[0]
        assert conn.source_block_id == "a"
        assert conn.source_output_name == "audio_out"
        assert conn.target_block_id == "b"
        assert conn.target_input_name == "audio_in"

    def test_enum_values_serialize_as_strings(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))

        data = serialize_graph(graph)

        block_data = data["blocks"][0]
        assert block_data["category"] == "PROCESSOR"
        assert block_data["state"] == "FRESH"
        port_data = block_data["output_ports"][0]
        assert port_data["port_type"] == "AUDIO"
        assert port_data["direction"] == "OUTPUT"

    def test_state_is_round_tripped(self) -> None:
        """S1 audit fix: state is now serialized and restored, not forced STALE."""
        graph = Graph()
        graph.add_block(_make_block("a"))
        graph.add_block(_make_block("b"))
        # Default block state is FRESH; after serialization it should stay FRESH
        data = serialize_graph(graph)
        restored = deserialize_graph(data)

        assert restored.blocks["a"].state == BlockState.FRESH
        assert restored.blocks["b"].state == BlockState.FRESH

    def test_stale_state_is_preserved(self) -> None:
        """STALE blocks should stay STALE after round-trip."""
        graph = Graph()
        graph.add_block(_make_block("a"))
        graph.set_block_state("a", BlockState.STALE)

        data = serialize_graph(graph)
        restored = deserialize_graph(data)

        assert restored.blocks["a"].state == BlockState.STALE

    def test_empty_graph_round_trip(self) -> None:
        graph = Graph()

        data = serialize_graph(graph)
        restored = deserialize_graph(data)

        assert len(restored.blocks) == 0
        assert len(restored.connections) == 0

    def test_diamond_topology_survives_round_trip(self) -> None:
        """Diamond: a -> b, a -> c, b -> d, c -> d."""
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(
            Port(name="out1", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="out2", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        )))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),), output_ports=(_event_out(),)))
        graph.add_block(_make_block("c", input_ports=(
            Port(name="audio_in", port_type=PortType.AUDIO, direction=Direction.INPUT),
        ), output_ports=(_event_out("event_out2"),)))
        graph.add_block(_make_block("d", input_ports=(
            Port(name="event_in", port_type=PortType.EVENT, direction=Direction.INPUT),
        )))

        graph.add_connection(Connection(
            source_block_id="a", source_output_name="out1",
            target_block_id="b", target_input_name="audio_in",
        ))
        graph.add_connection(Connection(
            source_block_id="a", source_output_name="out2",
            target_block_id="c", target_input_name="audio_in",
        ))
        graph.add_connection(Connection(
            source_block_id="b", source_output_name="event_out",
            target_block_id="d", target_input_name="event_in",
        ))

        data = serialize_graph(graph)
        restored = deserialize_graph(data)

        assert set(restored.blocks.keys()) == {"a", "b", "c", "d"}
        assert len(restored.connections) == 3
        topo = restored.topological_sort()
        assert topo.index("a") < topo.index("b")
        assert topo.index("a") < topo.index("c")
        assert topo.index("b") < topo.index("d")

    def test_block_with_settings_survives_round_trip(self) -> None:
        graph = Graph()
        graph.add_block(
            _make_block("a", settings={"file_path": "/audio/test.wav", "volume": 0.75})
        )

        data = serialize_graph(graph)
        restored = deserialize_graph(data)

        assert restored.blocks["a"].settings == {
            "file_path": "/audio/test.wav",
            "volume": 0.75,
        }


# ---------------------------------------------------------------------------
# Take round-trip
# ---------------------------------------------------------------------------


class TestTakeSerialization:
    """Verify Take serialize/deserialize round-trip fidelity."""

    def test_round_trip_take_with_source(self) -> None:
        take = _make_take(label="Onset Run 1", times=(1.0, 2.5, 3.7), is_main=True)

        data = serialize_take(take)
        restored = deserialize_take(data)

        assert restored.id == take.id
        assert restored.label == "Onset Run 1"
        assert restored.origin == "pipeline"
        assert restored.is_main is True
        assert restored.source is not None
        assert restored.source.block_type == "DetectOnsets"
        assert restored.source.settings_snapshot == {"threshold": 0.5, "hop_size": 512}
        assert restored.source.run_id == "run-001"

    def test_round_trip_take_without_source(self) -> None:
        take = _make_take(label="Manual", origin="user", with_source=False)

        data = serialize_take(take)
        restored = deserialize_take(data)

        assert restored.source is None
        assert restored.origin == "user"

    def test_round_trip_preserves_event_data(self) -> None:
        take = _make_take(times=(1.0, 2.0, 3.0))

        data = serialize_take(take)
        restored = deserialize_take(data)

        assert isinstance(restored.data, EventData)
        assert len(restored.data.layers) == 1
        events = restored.data.layers[0].events
        assert len(events) == 3
        assert events[0].time == 1.0
        assert events[1].time == 2.0
        assert events[2].time == 3.0

    def test_round_trip_preserves_event_fields(self) -> None:
        take = _make_take(times=(1.5,))

        data = serialize_take(take)
        restored = deserialize_take(data)

        event = restored.data.layers[0].events[0]
        assert event.id == "evt-1.5"
        assert event.duration == 0.1
        assert event.classifications == {"type": "test"}
        assert event.metadata == {"conf": 0.9}
        assert event.origin == "pipeline"

    def test_round_trip_audio_data_take(self) -> None:
        audio = AudioData(
            sample_rate=44100,
            duration=180.0,
            file_path="stems/vocals.wav",
            channel_count=2,
        )
        take = Take.create(
            data=audio,
            label="Vocals Stem",
            origin="pipeline",
            is_main=True,
        )

        data = serialize_take(take)
        restored = deserialize_take(data)

        assert isinstance(restored.data, AudioData)
        assert restored.data.sample_rate == 44100
        assert restored.data.duration == 180.0
        assert restored.data.file_path == "stems/vocals.wav"
        assert restored.data.channel_count == 2

    def test_round_trip_preserves_notes(self) -> None:
        take = Take.create(
            data=_make_event_data(1.0),
            label="With Notes",
            origin="user",
            notes="Adjusted for venue delay",
            is_main=True,
        )

        data = serialize_take(take)
        restored = deserialize_take(data)

        assert restored.notes == "Adjusted for venue delay"

    def test_round_trip_preserves_timestamp(self) -> None:
        take = _make_take()

        data = serialize_take(take)
        restored = deserialize_take(data)

        # Allow minor rounding from ISO format
        assert abs(
            (restored.created_at - take.created_at).total_seconds()
        ) < 0.001


# ---------------------------------------------------------------------------
# TakeLayer round-trip
# ---------------------------------------------------------------------------


class TestTakeLayerSerialization:
    """Verify TakeLayer serialize/deserialize round-trip."""

    def test_round_trip_layer_with_takes(self) -> None:
        main = _make_take(label="Main", times=(1.0, 2.0), is_main=True)
        run1 = _make_take(label="Run 1", times=(1.1, 2.1))
        run2 = _make_take(label="Run 2", times=(1.2, 2.2))
        layer = TakeLayer(layer_id="onsets", takes=[main, run1, run2])

        data = serialize_take_layer(layer)
        restored = deserialize_take_layer(data)

        assert restored.layer_id == "onsets"
        assert restored.take_count == 3
        assert restored.main_take().label == "Main"
        assert restored.takes[1].label == "Run 1"
        assert restored.takes[2].label == "Run 2"

    def test_round_trip_empty_layer(self) -> None:
        layer = TakeLayer(layer_id="empty", takes=[])

        data = serialize_take_layer(layer)
        restored = deserialize_take_layer(data)

        assert restored.layer_id == "empty"
        assert restored.take_count == 0

    def test_round_trip_preserves_main_flag(self) -> None:
        main = _make_take(label="Main", is_main=True)
        other = _make_take(label="Other")
        layer = TakeLayer(layer_id="test", takes=[main, other])

        data = serialize_take_layer(layer)
        restored = deserialize_take_layer(data)

        assert restored.main_take().id == main.id
        assert not restored.takes[1].is_main


# ---------------------------------------------------------------------------
# Project save/load
# ---------------------------------------------------------------------------


class TestProjectSaveLoad:
    """Verify full project save/load via tmp file."""

    def test_save_and_load_round_trip(self, tmp_path: Any) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(Connection(
            source_block_id="a", source_output_name="audio_out",
            target_block_id="b", target_input_name="audio_in",
        ))

        main = _make_take(label="Main", times=(1.0, 2.0), is_main=True)
        run1 = _make_take(label="Run 1", times=(1.1, 2.1))
        take_layers = [TakeLayer(layer_id="onsets", takes=[main, run1])]

        path = str(tmp_path / "project.json")
        save_project(path, graph, take_layers)

        restored_graph, restored_layers = load_project(path)

        assert set(restored_graph.blocks.keys()) == {"a", "b"}
        assert len(restored_graph.connections) == 1
        # S1 audit fix: state is now round-tripped. Default state is FRESH.
        assert restored_graph.blocks["a"].state == BlockState.FRESH

        assert len(restored_layers) == 1
        assert restored_layers[0].layer_id == "onsets"
        assert restored_layers[0].take_count == 2
        assert restored_layers[0].main_take().label == "Main"

    def test_save_creates_valid_json(self, tmp_path: Any) -> None:
        graph = Graph()
        graph.add_block(_make_block("a"))

        path = str(tmp_path / "project.json")
        save_project(path, graph)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["version"] == "2.1.0"
        assert "graph" in data
        assert "take_layers" in data

    def test_all_blocks_fresh_after_load(self, tmp_path: Any) -> None:
        """S1 audit fix: state is now round-tripped. Blocks default to FRESH."""
        graph = Graph()
        graph.add_block(_make_block("x"))
        graph.add_block(_make_block("y"))

        path = str(tmp_path / "project.json")
        save_project(path, graph)

        restored_graph, _ = load_project(path)

        for block in restored_graph.blocks.values():
            assert block.state == BlockState.FRESH

    def test_load_project_without_takes(self, tmp_path: Any) -> None:
        """Backward compat: project file without take_layers key."""
        graph = Graph()
        graph.add_block(_make_block("a"))

        path = str(tmp_path / "project.json")
        save_project(path, graph)

        restored_graph, take_layers = load_project(path)
        assert len(take_layers) == 0


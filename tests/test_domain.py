"""
Domain entity tests: Verify all frozen dataclasses, enums, and Graph invariants.
Exists because the domain layer is the foundation — every invariant must be proven correct.
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

import uuid

import pytest

from echozero.domain import (
    AudioData,
    Block,
    BlockCategory,
    BlockSettings,
    BlockState,
    Connection,
    Direction,
    Event,
    EventData,
    Graph,
    Layer,
    Port,
    PortType,
)
from echozero.errors import ValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_port(
    name: str,
    port_type: PortType = PortType.AUDIO,
    direction: Direction = Direction.INPUT,
) -> Port:
    """Create a port with sensible defaults."""
    return Port(name=name, port_type=port_type, direction=direction)


def _make_block(
    block_id: str | None = None,
    name: str = "TestBlock",
    block_type: str = "LoadAudio",
    input_ports: tuple[Port, ...] = (),
    output_ports: tuple[Port, ...] = (),
) -> Block:
    """Create a block with sensible defaults."""
    return Block(
        id=block_id or uuid.uuid4().hex,
        name=name,
        block_type=block_type,
        category=BlockCategory.PROCESSOR,
        input_ports=input_ports,
        output_ports=output_ports,
    )


def _audio_in(name: str = "audio_in") -> Port:
    return _make_port(name, PortType.AUDIO, Direction.INPUT)


def _audio_out(name: str = "audio_out") -> Port:
    return _make_port(name, PortType.AUDIO, Direction.OUTPUT)


def _event_in(name: str = "events_in") -> Port:
    return _make_port(name, PortType.EVENT, Direction.INPUT)


def _event_out(name: str = "events_out") -> Port:
    return _make_port(name, PortType.EVENT, Direction.OUTPUT)


# ---------------------------------------------------------------------------
# Block and Port creation
# ---------------------------------------------------------------------------


class TestBlockCreation:
    """Verify frozen block and port construction."""

    def test_block_has_correct_attributes(self) -> None:
        port_in = _audio_in()
        port_out = _audio_out()
        block = _make_block(
            block_id="b1",
            name="LoadAudio",
            input_ports=(port_in,),
            output_ports=(port_out,),
        )

        assert block.id == "b1"
        assert block.name == "LoadAudio"
        assert block.block_type == "LoadAudio"
        assert block.category == BlockCategory.PROCESSOR
        assert block.input_ports == (port_in,)
        assert block.output_ports == (port_out,)
        assert block.control_ports == ()
        assert block.state == BlockState.FRESH
        assert block.settings.entries == {}

    def test_port_has_correct_attributes(self) -> None:
        port = Port(name="main", port_type=PortType.OSC, direction=Direction.OUTPUT)

        assert port.name == "main"
        assert port.port_type == PortType.OSC
        assert port.direction == Direction.OUTPUT

    def test_block_is_frozen(self) -> None:
        block = _make_block(block_id="b1")
        with pytest.raises(AttributeError):
            block.name = "changed"  # type: ignore[misc]

    def test_block_with_custom_settings(self) -> None:
        settings = BlockSettings(entries={"threshold": 0.5, "min_gap": 0.05})
        block = Block(
            id="b1",
            name="DetectOnsets",
            block_type="DetectOnsets",
            category=BlockCategory.PROCESSOR,
            input_ports=(_audio_in(),),
            output_ports=(_event_out(),),
            settings=settings,
        )

        assert block.settings.entries["threshold"] == 0.5
        assert block.settings.entries["min_gap"] == 0.05


# ---------------------------------------------------------------------------
# Graph: adding and storing blocks
# ---------------------------------------------------------------------------


class TestGraphBlocks:
    """Verify block addition, removal, and duplicate rejection."""

    def test_add_blocks_stores_them(self) -> None:
        graph = Graph()
        block_a = _make_block(block_id="a")
        block_b = _make_block(block_id="b")
        graph.add_block(block_a)
        graph.add_block(block_b)

        assert len(graph.blocks) == 2
        assert graph.blocks["a"] is block_a
        assert graph.blocks["b"] is block_b

    def test_duplicate_block_id_raises(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a"))

        with pytest.raises(ValidationError, match="Duplicate block ID"):
            graph.add_block(_make_block(block_id="a"))

    def test_remove_block(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a"))
        graph.remove_block("a")

        assert len(graph.blocks) == 0

    def test_remove_nonexistent_block_raises(self) -> None:
        graph = Graph()

        with pytest.raises(ValidationError, match="Block not found"):
            graph.remove_block("ghost")

    def test_remove_block_removes_its_connections(self) -> None:
        graph = Graph()
        block_a = _make_block(block_id="a", output_ports=(_audio_out(),))
        block_b = _make_block(block_id="b", input_ports=(_audio_in(),))
        graph.add_block(block_a)
        graph.add_block(block_b)
        graph.add_connection(Connection("a", "audio_out", "b", "audio_in"))

        graph.remove_block("a")

        assert len(graph.connections) == 0
        assert "b" in graph.blocks


# ---------------------------------------------------------------------------
# Graph: set_block_state
# ---------------------------------------------------------------------------


class TestGraphSetBlockState:
    """Verify block state mutation via set_block_state."""

    def test_set_block_state_updates_state(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a"))
        assert graph.blocks["a"].state == BlockState.FRESH

        graph.set_block_state("a", BlockState.STALE)

        assert graph.blocks["a"].state == BlockState.STALE

    def test_set_block_state_preserves_other_fields(self) -> None:
        graph = Graph()
        original = _make_block(block_id="a", name="MyBlock", output_ports=(_audio_out(),))
        graph.add_block(original)

        graph.set_block_state("a", BlockState.ERROR)

        updated = graph.blocks["a"]
        assert updated.state == BlockState.ERROR
        assert updated.id == original.id
        assert updated.name == original.name
        assert updated.block_type == original.block_type
        assert updated.category == original.category
        assert updated.output_ports == original.output_ports

    def test_set_block_state_nonexistent_raises(self) -> None:
        graph = Graph()

        with pytest.raises(ValidationError, match="Block not found"):
            graph.set_block_state("ghost", BlockState.STALE)

    def test_set_block_state_to_same_state(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a"))

        graph.set_block_state("a", BlockState.FRESH)
        assert graph.blocks["a"].state == BlockState.FRESH

    def test_set_block_state_all_states(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a"))

        for state in BlockState:
            graph.set_block_state("a", state)
            assert graph.blocks["a"].state == state


# ---------------------------------------------------------------------------
# Graph: valid connections
# ---------------------------------------------------------------------------


class TestGraphConnections:
    """Verify connection creation and storage."""

    def test_add_audio_connection(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block(block_id="b", input_ports=(_audio_in(),)))

        conn = Connection("a", "audio_out", "b", "audio_in")
        graph.add_connection(conn)

        assert len(graph.connections) == 1
        assert graph.connections[0] == conn

    def test_add_event_connection(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a", output_ports=(_event_out(),)))
        graph.add_block(_make_block(block_id="b", input_ports=(_event_in(),)))

        conn = Connection("a", "events_out", "b", "events_in")
        graph.add_connection(conn)

        assert len(graph.connections) == 1

    def test_event_fan_in_allowed(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a", output_ports=(_event_out(),)))
        graph.add_block(_make_block(block_id="b", output_ports=(_event_out(),)))
        graph.add_block(_make_block(block_id="c", input_ports=(_event_in(),)))

        graph.add_connection(Connection("a", "events_out", "c", "events_in"))
        graph.add_connection(Connection("b", "events_out", "c", "events_in"))

        assert len(graph.connections) == 2

    def test_remove_connection(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block(block_id="b", input_ports=(_audio_in(),)))

        conn = Connection("a", "audio_out", "b", "audio_in")
        graph.add_connection(conn)
        graph.remove_connection(conn)

        assert len(graph.connections) == 0


# ---------------------------------------------------------------------------
# Graph: connection invariant violations
# ---------------------------------------------------------------------------


class TestGraphConnectionRejections:
    """Verify all connection invariant violations raise ValidationError."""

    def test_reject_cross_type_connection(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a", output_ports=(_event_out(),)))
        graph.add_block(_make_block(block_id="b", input_ports=(_audio_in(),)))

        with pytest.raises(ValidationError, match="Port type mismatch"):
            graph.add_connection(Connection("a", "events_out", "b", "audio_in"))

    def test_reject_audio_fan_in(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block(block_id="b", output_ports=(_audio_out(),)))
        graph.add_block(_make_block(block_id="c", input_ports=(_audio_in(),)))

        graph.add_connection(Connection("a", "audio_out", "c", "audio_in"))

        with pytest.raises(ValidationError, match="fan-in not allowed"):
            graph.add_connection(Connection("b", "audio_out", "c", "audio_in"))

    def test_reject_self_connection(self) -> None:
        graph = Graph()
        graph.add_block(
            _make_block(
                block_id="a",
                input_ports=(_audio_in(),),
                output_ports=(_audio_out(),),
            )
        )

        with pytest.raises(ValidationError, match="Self-connections"):
            graph.add_connection(Connection("a", "audio_out", "a", "audio_in"))

    def test_reject_nonexistent_source_block(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="b", input_ports=(_audio_in(),)))

        with pytest.raises(ValidationError, match="Source block not found"):
            graph.add_connection(Connection("ghost", "audio_out", "b", "audio_in"))

    def test_reject_nonexistent_target_block(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a", output_ports=(_audio_out(),)))

        with pytest.raises(ValidationError, match="Target block not found"):
            graph.add_connection(Connection("a", "audio_out", "ghost", "audio_in"))

    def test_reject_nonexistent_source_port(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block(block_id="b", input_ports=(_audio_in(),)))

        with pytest.raises(ValidationError, match="Output port.*not found"):
            graph.add_connection(Connection("a", "no_such_port", "b", "audio_in"))

    def test_reject_nonexistent_target_port(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block(block_id="b", input_ports=(_audio_in(),)))

        with pytest.raises(ValidationError, match="Input port.*not found"):
            graph.add_connection(Connection("a", "audio_out", "b", "no_such_port"))

    def test_reject_cycle_two_blocks(self) -> None:
        graph = Graph()
        graph.add_block(
            _make_block(
                block_id="a",
                input_ports=(_audio_in(),),
                output_ports=(_audio_out(),),
            )
        )
        graph.add_block(
            _make_block(
                block_id="b",
                input_ports=(_audio_in(),),
                output_ports=(_audio_out(),),
            )
        )

        graph.add_connection(Connection("a", "audio_out", "b", "audio_in"))

        with pytest.raises(ValidationError, match="cycle"):
            graph.add_connection(Connection("b", "audio_out", "a", "audio_in"))

    def test_reject_cycle_three_blocks(self) -> None:
        graph = Graph()
        for bid in ("a", "b", "c"):
            graph.add_block(
                _make_block(
                    block_id=bid,
                    input_ports=(_event_in(),),
                    output_ports=(_event_out(),),
                )
            )

        graph.add_connection(Connection("a", "events_out", "b", "events_in"))
        graph.add_connection(Connection("b", "events_out", "c", "events_in"))

        with pytest.raises(ValidationError, match="cycle"):
            graph.add_connection(Connection("c", "events_out", "a", "events_in"))

    def test_cycle_rejection_does_not_corrupt_graph(self) -> None:
        graph = Graph()
        graph.add_block(
            _make_block(
                block_id="a",
                input_ports=(_audio_in(),),
                output_ports=(_audio_out(),),
            )
        )
        graph.add_block(
            _make_block(
                block_id="b",
                input_ports=(_audio_in(),),
                output_ports=(_audio_out(),),
            )
        )
        graph.add_connection(Connection("a", "audio_out", "b", "audio_in"))

        with pytest.raises(ValidationError):
            graph.add_connection(Connection("b", "audio_out", "a", "audio_in"))

        # Graph still has exactly the one valid connection
        assert len(graph.connections) == 1
        assert graph.connections[0].source_block_id == "a"


# ---------------------------------------------------------------------------
# Graph: topological sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    """Verify execution order computation."""

    def test_chain_a_b_c(self) -> None:
        graph = Graph()
        for bid in ("a", "b", "c"):
            graph.add_block(
                _make_block(
                    block_id=bid,
                    input_ports=(_event_in(),),
                    output_ports=(_event_out(),),
                )
            )

        graph.add_connection(Connection("a", "events_out", "b", "events_in"))
        graph.add_connection(Connection("b", "events_out", "c", "events_in"))

        order = graph.topological_sort()

        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_diamond_a_bc_d(self) -> None:
        graph = Graph()
        graph.add_block(
            _make_block(block_id="a", output_ports=(_event_out("e1"), _event_out("e2")))
        )
        graph.add_block(
            _make_block(block_id="b", input_ports=(_event_in(),), output_ports=(_event_out(),))
        )
        graph.add_block(
            _make_block(block_id="c", input_ports=(_event_in(),), output_ports=(_event_out(),))
        )
        graph.add_block(_make_block(block_id="d", input_ports=(_event_in(),)))

        graph.add_connection(Connection("a", "e1", "b", "events_in"))
        graph.add_connection(Connection("a", "e2", "c", "events_in"))
        graph.add_connection(Connection("b", "events_out", "d", "events_in"))
        graph.add_connection(Connection("c", "events_out", "d", "events_in"))

        order = graph.topological_sort()

        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_disconnected_blocks_all_present(self) -> None:
        graph = Graph()
        graph.add_block(_make_block(block_id="x"))
        graph.add_block(_make_block(block_id="y"))
        graph.add_block(_make_block(block_id="z"))

        order = graph.topological_sort()

        assert set(order) == {"x", "y", "z"}
        assert len(order) == 3


# ---------------------------------------------------------------------------
# Graph: downstream_of
# ---------------------------------------------------------------------------


class TestDownstreamOf:
    """Verify transitive descendant discovery."""

    def test_downstream_of_chain(self) -> None:
        graph = Graph()
        for bid in ("a", "b", "c"):
            graph.add_block(
                _make_block(
                    block_id=bid,
                    input_ports=(_event_in(),),
                    output_ports=(_event_out(),),
                )
            )

        graph.add_connection(Connection("a", "events_out", "b", "events_in"))
        graph.add_connection(Connection("b", "events_out", "c", "events_in"))

        assert graph.downstream_of("a") == {"b", "c"}
        assert graph.downstream_of("b") == {"c"}
        assert graph.downstream_of("c") == set()

    def test_downstream_of_diamond(self) -> None:
        graph = Graph()
        graph.add_block(
            _make_block(block_id="a", output_ports=(_event_out("e1"), _event_out("e2")))
        )
        graph.add_block(
            _make_block(block_id="b", input_ports=(_event_in(),), output_ports=(_event_out(),))
        )
        graph.add_block(
            _make_block(block_id="c", input_ports=(_event_in(),), output_ports=(_event_out(),))
        )
        graph.add_block(_make_block(block_id="d", input_ports=(_event_in(),)))

        graph.add_connection(Connection("a", "e1", "b", "events_in"))
        graph.add_connection(Connection("a", "e2", "c", "events_in"))
        graph.add_connection(Connection("b", "events_out", "d", "events_in"))
        graph.add_connection(Connection("c", "events_out", "d", "events_in"))

        assert graph.downstream_of("a") == {"b", "c", "d"}

    def test_downstream_of_nonexistent_block_raises(self) -> None:
        graph = Graph()

        with pytest.raises(ValidationError, match="Block not found"):
            graph.downstream_of("ghost")


# ---------------------------------------------------------------------------
# Event, Layer, EventData
# ---------------------------------------------------------------------------


class TestEventEntities:
    """Verify event, layer, and event-data construction and semantics."""

    def test_event_creation_with_classification(self) -> None:
        event = Event(
            id="e1",
            time=1.5,
            duration=0.25,
            classifications={"onset_model/type": "kick"},
            metadata={"confidence": 0.95},
            origin="DetectOnsets",
        )

        assert event.id == "e1"
        assert event.time == 1.5
        assert event.duration == 0.25
        assert event.classifications["onset_model/type"] == "kick"
        assert event.metadata["confidence"] == 0.95
        assert event.origin == "DetectOnsets"

    def test_marker_event_has_zero_duration(self) -> None:
        marker = Event(
            id="m1",
            time=2.0,
            duration=0.0,
            classifications={},
            metadata={},
            origin="Manual",
        )

        assert marker.duration == 0.0

    def test_block_event_has_positive_duration(self) -> None:
        block_event = Event(
            id="be1",
            time=3.0,
            duration=1.5,
            classifications={},
            metadata={},
            origin="SegmentAudio",
        )

        assert block_event.duration > 0

    def test_layer_contains_events(self) -> None:
        events = (
            Event("e1", 0.0, 0.0, {}, {}, "Test"),
            Event("e2", 1.0, 0.5, {}, {}, "Test"),
        )
        layer = Layer(id="l1", name="Onsets", events=events)

        assert layer.id == "l1"
        assert layer.name == "Onsets"
        assert len(layer.events) == 2
        assert layer.events[0].time == 0.0
        assert layer.events[1].time == 1.0

    def test_event_data_immutable_layers(self) -> None:
        layer = Layer(id="l1", name="Onsets", events=())
        event_data = EventData(layers=(layer,))

        assert len(event_data.layers) == 1
        assert event_data.layers[0].name == "Onsets"

        with pytest.raises(AttributeError):
            event_data.layers = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Connection equality
# ---------------------------------------------------------------------------


class TestConnectionEquality:
    """Verify connection identity is based on endpoint tuple."""

    def test_same_endpoints_are_equal(self) -> None:
        conn1 = Connection("a", "out", "b", "in")
        conn2 = Connection("a", "out", "b", "in")

        assert conn1 == conn2

    def test_different_endpoints_are_not_equal(self) -> None:
        conn1 = Connection("a", "out", "b", "in")
        conn2 = Connection("a", "out", "c", "in")

        assert conn1 != conn2

    def test_same_endpoints_same_hash(self) -> None:
        conn1 = Connection("a", "out", "b", "in")
        conn2 = Connection("a", "out", "b", "in")

        assert hash(conn1) == hash(conn2)


# ---------------------------------------------------------------------------
# AudioData
# ---------------------------------------------------------------------------


class TestAudioData:
    """Verify audio data construction."""

    def test_audio_data_attributes(self) -> None:
        audio = AudioData(
            sample_rate=44100,
            duration=3.5,
            file_path="/audio/song.wav",
            channel_count=2,
        )

        assert audio.sample_rate == 44100
        assert audio.duration == 3.5
        assert audio.file_path == "/audio/song.wav"
        assert audio.channel_count == 2

    def test_audio_data_defaults_to_mono(self) -> None:
        audio = AudioData(sample_rate=48000, duration=1.0, file_path="/a.wav")

        assert audio.channel_count == 1

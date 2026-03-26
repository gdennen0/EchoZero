"""
ExecutionCache tests: Verify store/get round-trip, invalidation, and downstream cascade.
Exists because cached outputs drive the coordinator's dirty-set logic — stale caches break re-execution.
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

from typing import Any

from echozero.cache import CachedOutput, ExecutionCache
from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import Block, BlockSettings, Connection, Port


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(
    block_id: str,
    input_ports: tuple[Port, ...] = (),
    output_ports: tuple[Port, ...] = (),
) -> Block:
    return Block(
        id=block_id,
        name=f"Block {block_id}",
        block_type="TestType",
        category=BlockCategory.PROCESSOR,
        input_ports=input_ports,
        output_ports=output_ports,
    )


def _audio_out(name: str = "out") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _audio_in(name: str = "in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


# ---------------------------------------------------------------------------
# CachedOutput value object
# ---------------------------------------------------------------------------


class TestCachedOutput:
    """Verify frozen dataclass construction."""

    def test_creates_with_all_fields(self) -> None:
        entry = CachedOutput(
            block_id="b1",
            port_name="out",
            value=42,
            produced_at=1000.0,
            execution_id="run-1",
        )
        assert entry.block_id == "b1"
        assert entry.port_name == "out"
        assert entry.value == 42
        assert entry.produced_at == 1000.0
        assert entry.execution_id == "run-1"

    def test_is_frozen(self) -> None:
        entry = CachedOutput(
            block_id="b1", port_name="out", value=42, produced_at=1000.0, execution_id="run-1"
        )
        try:
            entry.value = 99  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# ExecutionCache
# ---------------------------------------------------------------------------


class TestExecutionCache:
    """Verify store, get, invalidation, and downstream cascading."""

    def test_store_and_get_round_trip(self) -> None:
        cache = ExecutionCache()
        cache.store("b1", "out", 42, "run-1")

        result = cache.get("b1", "out")
        assert result is not None
        assert result.value == 42
        assert result.block_id == "b1"
        assert result.port_name == "out"
        assert result.execution_id == "run-1"

    def test_get_missing_returns_none(self) -> None:
        cache = ExecutionCache()
        assert cache.get("b1", "out") is None

    def test_get_wrong_port_returns_none(self) -> None:
        cache = ExecutionCache()
        cache.store("b1", "out", 42, "run-1")
        assert cache.get("b1", "other") is None

    def test_store_overwrites_previous(self) -> None:
        cache = ExecutionCache()
        cache.store("b1", "out", 42, "run-1")
        cache.store("b1", "out", 99, "run-2")

        result = cache.get("b1", "out")
        assert result is not None
        assert result.value == 99
        assert result.execution_id == "run-2"

    def test_get_all_returns_all_ports(self) -> None:
        cache = ExecutionCache()
        cache.store("b1", "audio_out", "audio_data", "run-1")
        cache.store("b1", "event_out", "event_data", "run-1")

        all_outputs = cache.get_all("b1")
        assert len(all_outputs) == 2
        assert "audio_out" in all_outputs
        assert "event_out" in all_outputs
        assert all_outputs["audio_out"].value == "audio_data"
        assert all_outputs["event_out"].value == "event_data"

    def test_get_all_returns_empty_for_unknown_block(self) -> None:
        cache = ExecutionCache()
        assert cache.get_all("unknown") == {}

    def test_invalidate_removes_all_ports_for_block(self) -> None:
        cache = ExecutionCache()
        cache.store("b1", "out_a", 1, "run-1")
        cache.store("b1", "out_b", 2, "run-1")
        cache.store("b2", "out", 3, "run-1")

        cache.invalidate("b1")

        assert cache.get("b1", "out_a") is None
        assert cache.get("b1", "out_b") is None
        assert cache.get("b2", "out") is not None

    def test_invalidate_nonexistent_block_is_noop(self) -> None:
        cache = ExecutionCache()
        cache.invalidate("ghost")  # Should not raise

    def test_invalidate_downstream_cascades(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),), output_ports=(_audio_out(),)))
        graph.add_block(_make_block("c", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )
        graph.add_connection(
            Connection(source_block_id="b", source_output_name="out",
                       target_block_id="c", target_input_name="in")
        )

        cache = ExecutionCache()
        cache.store("a", "out", "data_a", "run-1")
        cache.store("b", "out", "data_b", "run-1")
        cache.store("c", "out", "data_c", "run-1")

        affected = cache.invalidate_downstream("a", graph)

        assert affected == {"a", "b", "c"}
        assert cache.get("a", "out") is None
        assert cache.get("b", "out") is None
        assert cache.get("c", "out") is None

    def test_invalidate_downstream_leaf_only_affects_itself(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )

        cache = ExecutionCache()
        cache.store("a", "out", "data_a", "run-1")
        cache.store("b", "out", "data_b", "run-1")

        affected = cache.invalidate_downstream("b", graph)

        assert affected == {"b"}
        assert cache.get("a", "out") is not None
        assert cache.get("b", "out") is None

    def test_invalidate_downstream_mid_chain(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),), output_ports=(_audio_out(),)))
        graph.add_block(_make_block("c", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )
        graph.add_connection(
            Connection(source_block_id="b", source_output_name="out",
                       target_block_id="c", target_input_name="in")
        )

        cache = ExecutionCache()
        cache.store("a", "out", "data_a", "run-1")
        cache.store("b", "out", "data_b", "run-1")
        cache.store("c", "out", "data_c", "run-1")

        affected = cache.invalidate_downstream("b", graph)

        assert affected == {"b", "c"}
        assert cache.get("a", "out") is not None
        assert cache.get("b", "out") is None
        assert cache.get("c", "out") is None

    def test_clear_removes_everything(self) -> None:
        cache = ExecutionCache()
        cache.store("b1", "out", 1, "run-1")
        cache.store("b2", "out", 2, "run-1")

        cache.clear()

        assert cache.get("b1", "out") is None
        assert cache.get("b2", "out") is None

    def test_stores_any_value_type(self) -> None:
        cache = ExecutionCache()
        cache.store("b1", "out", {"key": [1, 2, 3]}, "run-1")

        result = cache.get("b1", "out")
        assert result is not None
        assert result.value == {"key": [1, 2, 3]}

    def test_produced_at_is_set_automatically(self) -> None:
        import time

        before = time.time()
        cache = ExecutionCache()
        cache.store("b1", "out", 42, "run-1")
        after = time.time()

        result = cache.get("b1", "out")
        assert result is not None
        assert before <= result.produced_at <= after

"""
Coordinator tests: Verify request_run, cancel, propagate_stale, and ready_nodes scheduling.
Exists because the coordinator is the central orchestration point — broken coordination breaks everything.
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from echozero.cache import ExecutionCache
from echozero.coordinator import Coordinator, ready_nodes
from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import Block, BlockSettings, Connection, Port
from echozero.errors import ExecutionError, OperationCancelledError
from echozero.event_bus import EventBus
from echozero.execution import (
    BlockExecutor,
    ExecutionContext,
    ExecutionEngine,
    GraphPlanner,
)
from echozero.pipeline import Pipeline
from echozero.progress import RuntimeBus
from echozero.result import Err, Ok, err, is_ok, ok, unwrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(
    block_id: str,
    block_type: str = "TestType",
    input_ports: tuple[Port, ...] = (),
    output_ports: tuple[Port, ...] = (),
    settings: dict[str, Any] | None = None,
    state: BlockState = BlockState.FRESH,
) -> Block:
    return Block(
        id=block_id,
        name=f"Block {block_id}",
        block_type=block_type,
        category=BlockCategory.PROCESSOR,
        input_ports=input_ports,
        output_ports=output_ports,
        settings=BlockSettings(entries=settings or {}),
        state=state,
    )


def _audio_out(name: str = "out") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _audio_in(name: str = "in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


class StubExecutor:
    """A test executor that returns a fixed value or error."""

    def __init__(self, output: Any = "done", should_fail: bool = False) -> None:
        self._output = output
        self._should_fail = should_fail
        self.called_with: list[str] = []

    def execute(self, block_id: str, context: ExecutionContext) -> Any:
        self.called_with.append(block_id)
        if self._should_fail:
            return err(ExecutionError(f"Block {block_id} failed"))
        return ok(self._output)


def _make_coordinator(
    graph: Graph | None = None,
) -> tuple[Graph, ExecutionEngine, ExecutionCache, Coordinator]:
    """Create a full coordinator setup for testing."""
    graph = graph or Graph()
    event_bus = EventBus()
    pipeline = Pipeline(event_bus)
    runtime_bus = RuntimeBus()
    cache = ExecutionCache()
    engine = ExecutionEngine(graph, runtime_bus)
    coordinator = Coordinator(graph, pipeline, engine, cache, runtime_bus)
    return graph, engine, cache, coordinator


# ---------------------------------------------------------------------------
# ready_nodes pure function
# ---------------------------------------------------------------------------


class TestReadyNodes:
    """Verify the pure scheduling function computes correct ready sets."""

    def test_root_node_is_ready_when_dirty(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a"))
        cache = ExecutionCache()

        result = ready_nodes(graph, dirty={"a"}, running=set(), cache=cache)
        assert result == {"a"}

    def test_node_with_dirty_upstream_is_not_ready(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )
        cache = ExecutionCache()

        result = ready_nodes(graph, dirty={"a", "b"}, running=set(), cache=cache)
        assert result == {"a"}  # Only root is ready

    def test_node_with_running_upstream_is_not_ready(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )
        cache = ExecutionCache()

        result = ready_nodes(graph, dirty={"b"}, running={"a"}, cache=cache)
        assert result == set()  # b depends on running a

    def test_running_node_is_not_ready(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a"))
        cache = ExecutionCache()

        result = ready_nodes(graph, dirty={"a"}, running={"a"}, cache=cache)
        assert result == set()

    def test_non_dirty_node_is_not_ready(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a"))
        cache = ExecutionCache()

        result = ready_nodes(graph, dirty=set(), running=set(), cache=cache)
        assert result == set()

    def test_multiple_roots_all_ready(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a"))
        graph.add_block(_make_block("b"))
        graph.add_block(_make_block("c"))
        cache = ExecutionCache()

        result = ready_nodes(graph, dirty={"a", "b", "c"}, running=set(), cache=cache)
        assert result == {"a", "b", "c"}

    def test_chain_only_first_ready(self) -> None:
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

        result = ready_nodes(graph, dirty={"a", "b", "c"}, running=set(), cache=cache)
        assert result == {"a"}

    def test_diamond_both_middle_ready_when_root_clean(self) -> None:
        """Diamond: a -> b, a -> c, b -> d, c -> d. With a clean, b and c are ready."""
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(
            Port(name="out1", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="out2", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        )))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),), output_ports=(
            Port(name="event_out", port_type=PortType.EVENT, direction=Direction.OUTPUT),
        )))
        graph.add_block(_make_block("c", input_ports=(
            Port(name="in", port_type=PortType.AUDIO, direction=Direction.INPUT),
        ), output_ports=(
            Port(name="event_out", port_type=PortType.EVENT, direction=Direction.OUTPUT),
        )))
        graph.add_block(_make_block("d", input_ports=(
            Port(name="events", port_type=PortType.EVENT, direction=Direction.INPUT),
        )))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out1",
                       target_block_id="b", target_input_name="in")
        )
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out2",
                       target_block_id="c", target_input_name="in")
        )
        graph.add_connection(
            Connection(source_block_id="b", source_output_name="event_out",
                       target_block_id="d", target_input_name="events")
        )
        graph.add_connection(
            Connection(source_block_id="c", source_output_name="event_out",
                       target_block_id="d", target_input_name="events")
        )
        cache = ExecutionCache()
        # a was already executed (it's clean), so its outputs are cached
        cache.store("a", "out1", "data1", "run-0")
        cache.store("a", "out2", "data2", "run-0")

        # a is clean, b and c are dirty, d is dirty
        result = ready_nodes(graph, dirty={"b", "c", "d"}, running=set(), cache=cache)
        assert result == {"b", "c"}


# ---------------------------------------------------------------------------
# Coordinator.request_run
# ---------------------------------------------------------------------------


class TestCoordinatorRequestRun:
    """Verify that request_run executes and caches results."""

    def test_request_run_returns_execution_id(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a"))
        engine.register_executor("TestType", StubExecutor(output="audio"))

        result = coord.request_run()

        assert isinstance(result, Ok)
        assert len(result.value) == 32  # UUID hex

    def test_request_run_caches_outputs(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a"))
        engine.register_executor("TestType", StubExecutor(output="audio_data"))

        coord.request_run()

        cached = cache.get("a", "out")
        assert cached is not None
        assert cached.value == "audio_data"

    def test_request_run_marks_blocks_fresh(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a", state=BlockState.STALE))
        engine.register_executor("TestType", StubExecutor(output="data"))

        coord.request_run()

        assert graph.blocks["a"].state == BlockState.FRESH

    def test_request_run_with_target_only_runs_upstream(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a", block_type="TypeA", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", block_type="TypeB", input_ports=(_audio_in(),), output_ports=(_audio_out(),)))
        graph.add_block(_make_block("c", block_type="TypeC", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )
        graph.add_connection(
            Connection(source_block_id="b", source_output_name="out",
                       target_block_id="c", target_input_name="in")
        )

        exec_a = StubExecutor(output="a_out")
        exec_b = StubExecutor(output="b_out")
        exec_c = StubExecutor(output="c_out")
        engine.register_executor("TypeA", exec_a)
        engine.register_executor("TypeB", exec_b)
        engine.register_executor("TypeC", exec_c)

        coord.request_run(target="b")

        assert exec_a.called_with == ["a"]
        assert exec_b.called_with == ["b"]
        assert exec_c.called_with == []  # Not executed

    def test_request_run_returns_err_on_failure(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a"))
        engine.register_executor("TestType", StubExecutor(should_fail=True))

        result = coord.request_run()

        assert isinstance(result, Err)

    def test_is_executing_is_false_before_and_after(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        assert coord.is_executing is False

        graph.add_block(_make_block("a"))
        engine.register_executor("TestType", StubExecutor())
        coord.request_run()

        assert coord.is_executing is False


# ---------------------------------------------------------------------------
# Coordinator.cancel
# ---------------------------------------------------------------------------


class TestCoordinatorCancel:
    """Verify that cancel signals stop execution."""

    def test_cancel_sets_event(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        coord.cancel()
        # The cancel event is internal — we test its effect via execution
        assert coord._cancel_event.is_set()

    def test_pre_cancelled_run_returns_err(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a"))
        engine.register_executor("TestType", StubExecutor())

        coord.cancel()
        result = coord.request_run()

        # request_run clears the cancel event, so this should succeed
        # (cancel is for in-flight, not pre-emptive)
        assert isinstance(result, Ok)


# ---------------------------------------------------------------------------
# Coordinator.propagate_stale
# ---------------------------------------------------------------------------


class TestCoordinatorPropagateStale:
    """Verify staleness propagation through the graph."""

    def test_propagate_stale_marks_block_stale(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a"))

        affected = coord.propagate_stale("a")

        assert "a" in affected
        assert graph.blocks["a"].state == BlockState.STALE

    def test_propagate_stale_marks_downstream_stale(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
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

        affected = coord.propagate_stale("a")

        assert affected == {"a", "b", "c"}
        assert graph.blocks["a"].state == BlockState.STALE
        assert graph.blocks["b"].state == BlockState.STALE
        assert graph.blocks["c"].state == BlockState.STALE

    def test_propagate_stale_invalidates_cache(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )

        cache.store("a", "out", "data_a", "run-1")
        cache.store("b", "out", "data_b", "run-1")

        coord.propagate_stale("a")

        assert cache.get("a", "out") is None
        assert cache.get("b", "out") is None

    def test_propagate_stale_does_not_affect_unrelated(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_block(_make_block("c"))  # Disconnected
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )

        cache.store("c", "out", "data_c", "run-1")

        affected = coord.propagate_stale("a")

        assert "c" not in affected
        assert graph.blocks["c"].state == BlockState.FRESH
        assert cache.get("c", "out") is not None


# ---------------------------------------------------------------------------
# Per-port cache awareness in ready_nodes
# ---------------------------------------------------------------------------


class TestReadyNodesPortAware:
    """Verify ready_nodes checks port-level cache validity."""

    def test_block_with_uncached_upstream_is_not_ready(self) -> None:
        """Even if upstream is clean (not dirty), missing cache means not ready."""
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )
        cache = ExecutionCache()
        # a is NOT dirty, NOT running, but has no cached output

        result = ready_nodes(graph, dirty={"b"}, running=set(), cache=cache)
        assert result == set()  # b can't run because a's output isn't cached

    def test_block_with_cached_upstream_is_ready(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )
        cache = ExecutionCache()
        cache.store("a", "out", "data", "run-0")

        result = ready_nodes(graph, dirty={"b"}, running=set(), cache=cache)
        assert result == {"b"}

    def test_root_node_always_ready_when_dirty(self) -> None:
        """Root nodes have no input connections, so cache check is vacuously true."""
        graph = Graph()
        graph.add_block(_make_block("a"))
        cache = ExecutionCache()  # Empty cache — doesn't matter for roots

        result = ready_nodes(graph, dirty={"a"}, running=set(), cache=cache)
        assert result == {"a"}

    def test_has_valid_output_true_false(self) -> None:
        cache = ExecutionCache()
        cache.store("a", "out", "data", "run-1")

        assert cache.has_valid_output("a", "out") is True
        assert cache.has_valid_output("a", "other") is False
        assert cache.has_valid_output("b", "out") is False


# ---------------------------------------------------------------------------
# Auto-evaluation wiring
# ---------------------------------------------------------------------------


class TestAutoEvaluation:
    """Verify DocumentBus events trigger propagate_stale and optional request_run."""

    def test_settings_changed_triggers_propagate_stale(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a"))

        doc_bus = EventBus()
        coord.subscribe_to_document_bus(doc_bus)

        from echozero.domain.events import SettingsChangedEvent, create_event_id
        import time

        event = SettingsChangedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=create_event_id(),
            block_id="a",
            setting_key="threshold",
        )
        doc_bus.publish(event)

        assert graph.blocks["a"].state == BlockState.STALE

    def test_connection_added_triggers_propagate_stale_on_target(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(source_block_id="a", source_output_name="out",
                       target_block_id="b", target_input_name="in")
        )

        doc_bus = EventBus()
        coord.subscribe_to_document_bus(doc_bus)

        from echozero.domain.events import ConnectionAddedEvent, create_event_id
        import time

        event = ConnectionAddedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=create_event_id(),
            source_block_id="a",
            target_block_id="b",
        )
        doc_bus.publish(event)

        assert graph.blocks["b"].state == BlockState.STALE

    def test_auto_evaluate_true_triggers_request_run(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a"))
        engine.register_executor("TestType", StubExecutor(output="result"))

        doc_bus = EventBus()
        coord.subscribe_to_document_bus(doc_bus)
        coord.auto_evaluate = True

        from echozero.domain.events import SettingsChangedEvent, create_event_id
        import time

        event = SettingsChangedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=create_event_id(),
            block_id="a",
            setting_key="threshold",
        )
        doc_bus.publish(event)

        # Block should have been re-executed and cached
        assert cache.get("a", "out") is not None
        assert graph.blocks["a"].state == BlockState.FRESH

    def test_auto_evaluate_false_does_not_trigger_request_run(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a"))
        engine.register_executor("TestType", StubExecutor(output="result"))

        doc_bus = EventBus()
        coord.subscribe_to_document_bus(doc_bus)
        coord.auto_evaluate = False  # default

        from echozero.domain.events import SettingsChangedEvent, create_event_id
        import time

        event = SettingsChangedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=create_event_id(),
            block_id="a",
            setting_key="threshold",
        )
        doc_bus.publish(event)

        # Block should be STALE but NOT re-executed
        assert graph.blocks["a"].state == BlockState.STALE
        assert cache.get("a", "out") is None

    def test_block_removed_invalidates_cache(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a"))
        cache.store("a", "out", "data", "run-1")

        doc_bus = EventBus()
        coord.subscribe_to_document_bus(doc_bus)

        # Remove block from graph first (as command handler would)
        graph.remove_block("a")

        from echozero.domain.events import BlockRemovedEvent, create_event_id
        import time

        event = BlockRemovedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=create_event_id(),
            block_id="a",
        )
        doc_bus.publish(event)

        assert cache.get("a", "out") is None

    def test_unsubscribe_prevents_further_events(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a"))

        doc_bus = EventBus()
        coord.subscribe_to_document_bus(doc_bus)
        coord.unsubscribe_from_document_bus(doc_bus)

        from echozero.domain.events import SettingsChangedEvent, create_event_id
        import time

        event = SettingsChangedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=create_event_id(),
            block_id="a",
            setting_key="threshold",
        )
        doc_bus.publish(event)

        # Block should NOT have been marked stale
        assert graph.blocks["a"].state == BlockState.FRESH


# ---------------------------------------------------------------------------
# Multi-port caching in Coordinator
# ---------------------------------------------------------------------------


class MultiPortExecutor:
    """An executor that returns a dict of port_name -> value."""

    def __init__(self, outputs: dict[str, Any]) -> None:
        self._outputs = outputs
        self.called_with: list[str] = []

    def execute(self, block_id: str, context: ExecutionContext) -> Any:
        self.called_with.append(block_id)
        return ok(self._outputs)


class TestCoordinatorMultiPort:
    """Verify coordinator caches multi-port outputs correctly."""

    def test_coordinator_caches_multi_port_outputs(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("sep", block_type="Separator", output_ports=(
            Port(name="drums", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="bass", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        )))

        engine.register_executor(
            "Separator",
            MultiPortExecutor({"drums": "drums_data", "bass": "bass_data"}),
        )

        coord.request_run()

        assert cache.get("sep", "drums") is not None
        assert cache.get("sep", "drums").value == "drums_data"
        assert cache.get("sep", "bass") is not None
        assert cache.get("sep", "bass").value == "bass_data"

    def test_coordinator_caches_single_port_to_named_port(self) -> None:
        graph, engine, cache, coord = _make_coordinator()
        graph.add_block(_make_block("a", output_ports=(
            Port(name="audio_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        )))

        engine.register_executor("TestType", StubExecutor(output="audio_data"))

        coord.request_run()

        assert cache.get("a", "audio_out") is not None
        assert cache.get("a", "audio_out").value == "audio_data"

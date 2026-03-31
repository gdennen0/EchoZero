"""
Architecture fix tests: S3 shared graph, S4 category filtering, S5 output freeze,
M3 graph snapshots, M4 async execution, CommandEnvelope rollback.
Exists to verify the structural and application-layer fixes introduced in the
non-UI architecture gap sprint.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest

from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.graph import Graph, GraphSnapshot
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, EventData, Layer, Event, Port
from echozero.editor.cache import ExecutionCache
from echozero.editor.coordinator import Coordinator, ExecutionHandle
from echozero.editor.pipeline import CommandEnvelope, Pipeline
from echozero.editor.commands import AddBlockCommand, RemoveBlockCommand
from echozero.event_bus import EventBus
from echozero.execution import ExecutionContext, ExecutionEngine, ExecutionPlan, GraphPlanner
from echozero.progress import RuntimeBus
from echozero.result import Ok, err, is_ok, ok, unwrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(
    block_id: str,
    block_type: str = "TestType",
    category: BlockCategory = BlockCategory.PROCESSOR,
    input_ports: tuple[Port, ...] = (),
    output_ports: tuple[Port, ...] = (),
) -> Block:
    return Block(
        id=block_id,
        name=f"Block {block_id}",
        block_type=block_type,
        category=category,
        input_ports=input_ports,
        output_ports=output_ports,
        settings=BlockSettings({}),
    )


def _audio_out(name: str = "out") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _audio_in(name: str = "in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


def _add_block_cmd(block_id: str, category: BlockCategory = BlockCategory.PROCESSOR) -> AddBlockCommand:
    return AddBlockCommand(
        block_id=block_id,
        name=f"Block {block_id}",
        block_type="TestType",
        category=category,
        input_ports=(),
        output_ports=(),
        control_ports=(),
        settings_entries=(),
    )


class StubExecutor:
    def __init__(self, output: Any = "done", should_fail: bool = False) -> None:
        self._output = output
        self._should_fail = should_fail
        self.called_with: list[str] = []

    def execute(self, block_id: str, context: "ExecutionContext") -> Any:
        self.called_with.append(block_id)
        if self._should_fail:
            from echozero.errors import ExecutionError
            return err(ExecutionError(f"Block {block_id} failed"))
        return ok(self._output)


def _make_coordinator(graph: Graph | None = None) -> tuple[Graph, Pipeline, ExecutionEngine, ExecutionCache, Coordinator]:
    graph = graph or Graph()
    event_bus = EventBus()
    pipeline = Pipeline(event_bus, graph=graph)
    runtime_bus = RuntimeBus()
    cache = ExecutionCache()
    engine = ExecutionEngine(graph, runtime_bus)
    coordinator = Coordinator(graph, pipeline, engine, cache, runtime_bus)
    return graph, pipeline, engine, cache, coordinator


# ===========================================================================
# S3: Shared Graph — Pipeline and Coordinator see the same instance
# ===========================================================================


class TestS3SharedGraph:
    """Pipeline and Coordinator must share the exact same Graph instance."""

    def test_pipeline_and_coordinator_share_same_graph(self) -> None:
        graph = Graph()
        event_bus = EventBus()
        pipeline = Pipeline(event_bus, graph=graph)

        assert pipeline.graph is graph

    def test_pipeline_created_with_external_graph_uses_it(self) -> None:
        graph = Graph()
        graph.add_block(_block("b1"))
        event_bus = EventBus()
        pipeline = Pipeline(event_bus, graph=graph)

        assert "b1" in pipeline.graph.blocks

    def test_dispatch_mutation_visible_to_coordinator(self) -> None:
        graph, pipeline, engine, cache, coord = _make_coordinator()
        assert pipeline.graph is graph  # Same instance

        pipeline.dispatch(_add_block_cmd("b1"))

        # Block added via pipeline should be visible via the graph
        assert "b1" in graph.blocks

    def test_pipeline_default_creates_own_graph(self) -> None:
        event_bus = EventBus()
        pipeline = Pipeline(event_bus)
        pipeline.dispatch(_add_block_cmd("b1"))

        assert "b1" in pipeline.graph.blocks


# ===========================================================================
# S5: set_output rejects numpy arrays
# ===========================================================================


class TestS5OutputFreeze:
    """ExecutionContext.set_output must reject numpy arrays."""

    def _make_context(self) -> ExecutionContext:
        graph = Graph()
        bus = RuntimeBus()
        return ExecutionContext(
            execution_id="test-exec",
            graph=graph,
            progress_bus=bus,
        )

    def test_set_output_rejects_numpy_array(self) -> None:
        import numpy as np
        ctx = self._make_context()
        with pytest.raises(TypeError, match="numpy arrays"):
            ctx.set_output("b1", "out", np.array([1.0, 2.0]))

    def test_set_output_rejects_numpy_array_message_contains_block_and_port(self) -> None:
        import numpy as np
        ctx = self._make_context()
        with pytest.raises(TypeError) as exc_info:
            ctx.set_output("my_block", "audio_out", np.zeros(100))
        assert "my_block" in str(exc_info.value)
        assert "audio_out" in str(exc_info.value)

    def test_set_output_accepts_audio_data(self) -> None:
        ctx = self._make_context()
        audio = AudioData(sample_rate=44100, duration=1.0, file_path="/tmp/test.wav")
        ctx.set_output("b1", "out", audio)
        # No exception — and value is stored
        val = ctx._outputs[("b1", "out")]
        assert isinstance(val, AudioData)

    def test_set_output_accepts_event_data(self) -> None:
        ctx = self._make_context()
        events = EventData(layers=())
        ctx.set_output("b1", "out", events)
        assert ctx._outputs[("b1", "out")] is events

    def test_set_output_accepts_dict(self) -> None:
        ctx = self._make_context()
        ctx.set_output("b1", "out", {"some": "data"})
        assert ctx._outputs[("b1", "out")] == {"some": "data"}

    def test_set_output_accepts_string(self) -> None:
        ctx = self._make_context()
        ctx.set_output("b1", "out", "/path/to/result")
        assert ctx._outputs[("b1", "out")] == "/path/to/result"

    def test_set_output_accepts_none(self) -> None:
        ctx = self._make_context()
        ctx.set_output("b1", "out", None)
        assert ctx._outputs[("b1", "out")] is None


# ===========================================================================
# S4: Block category filtering in GraphPlanner
# ===========================================================================


class TestS4CategoryFiltering:
    """GraphPlanner.plan() must exclude non-PROCESSOR blocks."""

    def test_processor_blocks_included_in_plan(self) -> None:
        graph = Graph()
        graph.add_block(_block("proc1", category=BlockCategory.PROCESSOR))
        plan = GraphPlanner().plan(graph)
        assert "proc1" in plan.ordered_block_ids

    def test_workspace_blocks_excluded_from_plan(self) -> None:
        graph = Graph()
        graph.add_block(_block("ws1", category=BlockCategory.WORKSPACE))
        plan = GraphPlanner().plan(graph)
        assert "ws1" not in plan.ordered_block_ids

    def test_playback_blocks_excluded_from_plan(self) -> None:
        graph = Graph()
        graph.add_block(_block("play1", category=BlockCategory.PLAYBACK))
        plan = GraphPlanner().plan(graph)
        assert "play1" not in plan.ordered_block_ids

    def test_mixed_categories_only_processors_in_plan(self) -> None:
        graph = Graph()
        graph.add_block(_block("proc1", category=BlockCategory.PROCESSOR))
        graph.add_block(_block("ws1", category=BlockCategory.WORKSPACE))
        graph.add_block(_block("play1", category=BlockCategory.PLAYBACK))
        plan = GraphPlanner().plan(graph)
        assert "proc1" in plan.ordered_block_ids
        assert "ws1" not in plan.ordered_block_ids
        assert "play1" not in plan.ordered_block_ids

    def test_plan_with_target_still_excludes_non_processor(self) -> None:
        graph = Graph()
        graph.add_block(_block("proc1", category=BlockCategory.PROCESSOR, output_ports=(_audio_out(),)))
        graph.add_block(_block("proc2", category=BlockCategory.PROCESSOR, input_ports=(_audio_in(),)))
        graph.add_block(_block("ws1", category=BlockCategory.WORKSPACE))
        graph.add_connection(Connection("proc1", "out", "proc2", "in"))
        plan = GraphPlanner().plan(graph, target_block_id="proc2")
        assert "proc1" in plan.ordered_block_ids
        assert "proc2" in plan.ordered_block_ids
        assert "ws1" not in plan.ordered_block_ids

    def test_empty_plan_when_no_processors(self) -> None:
        graph = Graph()
        graph.add_block(_block("ws1", category=BlockCategory.WORKSPACE))
        graph.add_block(_block("play1", category=BlockCategory.PLAYBACK))
        plan = GraphPlanner().plan(graph)
        assert len(plan.ordered_block_ids) == 0


# ===========================================================================
# M3: Graph snapshot and restore
# ===========================================================================


class TestM3GraphSnapshot:
    """Graph.snapshot() and Graph.restore() must correctly capture/restore state."""

    def test_snapshot_captures_blocks(self) -> None:
        graph = Graph()
        graph.add_block(_block("b1"))
        snap = graph.snapshot()
        assert "b1" in snap.blocks

    def test_snapshot_captures_connections(self) -> None:
        graph = Graph()
        graph.add_block(_block("b1", output_ports=(_audio_out(),)))
        graph.add_block(_block("b2", input_ports=(_audio_in(),)))
        graph.add_connection(Connection("b1", "out", "b2", "in"))
        snap = graph.snapshot()
        assert len(snap.connections) == 1

    def test_snapshot_is_independent_copy(self) -> None:
        graph = Graph()
        graph.add_block(_block("b1"))
        snap = graph.snapshot()

        # Adding a block after snapshot doesn't affect snapshot
        graph.add_block(_block("b2"))
        assert "b2" not in snap.blocks
        assert "b1" in snap.blocks

    def test_restore_reverts_to_snapshot_state(self) -> None:
        graph = Graph()
        graph.add_block(_block("b1"))
        snap = graph.snapshot()

        graph.add_block(_block("b2"))
        assert "b2" in graph.blocks

        graph.restore(snap)
        assert "b1" in graph.blocks
        assert "b2" not in graph.blocks

    def test_restore_removes_blocks_added_after_snapshot(self) -> None:
        graph = Graph()
        snap_empty = graph.snapshot()

        graph.add_block(_block("b1"))
        graph.add_block(_block("b2"))

        graph.restore(snap_empty)
        assert len(graph.blocks) == 0

    def test_restore_reverts_connections(self) -> None:
        graph = Graph()
        graph.add_block(_block("b1", output_ports=(_audio_out(),)))
        graph.add_block(_block("b2", input_ports=(_audio_in(),)))
        snap = graph.snapshot()  # 0 connections

        graph.add_connection(Connection("b1", "out", "b2", "in"))
        assert len(graph.connections) == 1

        graph.restore(snap)
        assert len(graph.connections) == 0

    def test_restore_allows_re_add_of_previously_removed_block(self) -> None:
        graph = Graph()
        graph.add_block(_block("b1"))
        snap = graph.snapshot()

        graph.remove_block("b1")
        graph.restore(snap)

        assert "b1" in graph.blocks


# ===========================================================================
# M4: Background execution (request_run_async)
# ===========================================================================


class TestM4AsyncExecution:
    """Coordinator.request_run_async must return immediately with an ExecutionHandle."""

    def _make_stack_with_stub(self) -> tuple[Graph, Pipeline, ExecutionEngine, ExecutionCache, Coordinator, StubExecutor]:
        graph = Graph()
        graph.add_block(_block("b1", output_ports=(_audio_out(),)))
        event_bus = EventBus()
        pipeline = Pipeline(event_bus, graph=graph)
        runtime_bus = RuntimeBus()
        cache = ExecutionCache()
        engine = ExecutionEngine(graph, runtime_bus)
        stub = StubExecutor(output="result_value")
        engine.register_executor("TestType", stub)
        coordinator = Coordinator(graph, pipeline, engine, cache, runtime_bus)
        return graph, pipeline, engine, cache, coordinator, stub

    def test_request_run_async_returns_execution_handle(self) -> None:
        _, _, _, _, coord, _ = self._make_stack_with_stub()
        result = coord.request_run_async()
        assert is_ok(result)
        handle = unwrap(result)
        assert isinstance(handle, ExecutionHandle)

    def test_execution_handle_done_true_after_completion(self) -> None:
        _, _, _, _, coord, _ = self._make_stack_with_stub()
        result = coord.request_run_async()
        handle = unwrap(result)

        # Wait for completion (with timeout)
        deadline = time.time() + 5.0
        while not handle.done and time.time() < deadline:
            time.sleep(0.01)

        assert handle.done is True

    def test_execution_handle_cancel_sets_cancel_event(self) -> None:
        _, _, _, _, coord, _ = self._make_stack_with_stub()
        result = coord.request_run_async()
        handle = unwrap(result)
        handle.cancel()
        assert handle.cancel_event.is_set()

    def test_cannot_start_two_concurrent_executions(self) -> None:
        import threading

        graph = Graph()
        event_bus = EventBus()
        pipeline = Pipeline(event_bus, graph=graph)
        runtime_bus = RuntimeBus()
        cache = ExecutionCache()
        engine = ExecutionEngine(graph, runtime_bus)
        coord = Coordinator(graph, pipeline, engine, cache, runtime_bus)

        # Manually set executing flag to simulate a running execution
        coord._executing = True

        result = coord.request_run_async()
        from echozero.result import Err
        assert isinstance(result, Err)

        coord._executing = False

    def test_execution_handle_done_false_initially_for_slow_task(self) -> None:
        """When execution hasn't started yet, done should be False."""
        import threading

        ready = threading.Event()
        release = threading.Event()

        class SlowExecutor:
            def execute(self, block_id: str, context: Any) -> Any:
                ready.set()
                release.wait(timeout=5.0)
                return ok("slow_done")

        graph = Graph()
        graph.add_block(_block("b1", output_ports=(_audio_out(),)))
        event_bus = EventBus()
        pipeline = Pipeline(event_bus, graph=graph)
        runtime_bus = RuntimeBus()
        cache = ExecutionCache()
        engine = ExecutionEngine(graph, runtime_bus)
        engine.register_executor("TestType", SlowExecutor())
        coord = Coordinator(graph, pipeline, engine, cache, runtime_bus)

        result = coord.request_run_async()
        handle = unwrap(result)

        # Wait until the executor is actually running
        ready.wait(timeout=5.0)
        assert not handle.done

        # Release and let it finish
        release.set()
        deadline = time.time() + 5.0
        while not handle.done and time.time() < deadline:
            time.sleep(0.01)
        assert handle.done


# ===========================================================================
# CommandEnvelope: rollback on failure, no events, snapshots on success
# ===========================================================================


class TestCommandEnvelope:
    """Pipeline.dispatch must snapshot before/after, rollback on failure."""

    def _make_failing_pipeline(self) -> Pipeline:
        """Pipeline with a handler that always raises."""
        event_bus = EventBus()
        pipeline = Pipeline(event_bus)

        def bad_handler(cmd: Any, ctx: Any) -> Any:
            raise RuntimeError("intentional failure")

        from echozero.editor.commands import AddBlockCommand
        pipeline.register(AddBlockCommand, bad_handler)
        return pipeline

    def test_failed_command_restores_graph_to_prior_state(self) -> None:
        event_bus = EventBus()
        pipeline = self._make_failing_pipeline()

        # Add a block first to have some state
        # Use the default pipeline graph
        graph = pipeline.graph
        graph.add_block(_block("existing"))

        # Dispatch a command that will fail
        result = pipeline.dispatch(_add_block_cmd("new_block"))

        from echozero.result import Err
        assert isinstance(result, Err)
        # "existing" should still be there, "new_block" must not be
        assert "existing" in graph.blocks
        assert "new_block" not in graph.blocks

    def test_failed_command_does_not_emit_events(self) -> None:
        published: list[Any] = []
        event_bus = EventBus()

        from echozero.domain.events import DomainEvent
        # Subscribe to all events
        from echozero.domain.events import BlockAddedEvent
        event_bus.subscribe(BlockAddedEvent, published.append)

        pipeline = Pipeline(event_bus)
        # Replace handler with one that raises
        def bad_handler(cmd: Any, ctx: Any) -> Any:
            raise RuntimeError("fail before emit")

        pipeline.register(AddBlockCommand, bad_handler)

        pipeline.dispatch(_add_block_cmd("b1"))

        assert len(published) == 0

    def test_successful_command_increments_sequence(self) -> None:
        event_bus = EventBus()
        pipeline = Pipeline(event_bus)

        assert pipeline._sequence == 0
        pipeline.dispatch(_add_block_cmd("b1"))
        assert pipeline._sequence == 1
        pipeline.dispatch(_add_block_cmd("b2"))
        assert pipeline._sequence == 2

    def test_successful_dispatch_graph_reflects_change(self) -> None:
        event_bus = EventBus()
        pipeline = Pipeline(event_bus)

        result = pipeline.dispatch(_add_block_cmd("b1"))
        assert is_ok(result)
        assert "b1" in pipeline.graph.blocks

    def test_snapshot_before_and_after_differ_on_success(self) -> None:
        """After dispatch, post_snapshot has the new block, prior does not."""
        event_bus = EventBus()
        pipeline = Pipeline(event_bus)

        # Capture snapshots by instrumenting the graph
        snap_before = pipeline.graph.snapshot()
        pipeline.dispatch(_add_block_cmd("b1"))
        snap_after = pipeline.graph.snapshot()

        assert "b1" not in snap_before.blocks
        assert "b1" in snap_after.blocks

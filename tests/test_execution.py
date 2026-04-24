"""
Execution engine tests: Verify planning, engine dispatch, report publishing, and fail-fast.
Exists because the execution engine orchestrates block runs — incorrect ordering or missed errors break pipelines.
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import Block, BlockSettings, Connection, Port
from echozero.errors import ExecutionError, OperationCancelledError
from echozero.execution import (
    BlockExecutor,
    ExecutionContext,
    ExecutionEngine,
    ExecutionPlan,
    GraphPlanner,
)
from echozero.progress import (
    ExecutionCompletedReport,
    ExecutionStartedReport,
    ProgressReport,
    RuntimeBus,
    RuntimeReport,
)
from echozero.result import Err, Ok, err, ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(
    block_id: str,
    block_type: str = "TestType",
    input_ports: tuple[Port, ...] = (),
    output_ports: tuple[Port, ...] = (),
    settings: dict[str, Any] | None = None,
) -> Block:
    """Create a block with sensible defaults for testing."""
    return Block(
        id=block_id,
        name=f"Block {block_id}",
        block_type=block_type,
        category=BlockCategory.PROCESSOR,
        input_ports=input_ports,
        output_ports=output_ports,
        settings=BlockSettings(settings or {}),
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


# ---------------------------------------------------------------------------
# GraphPlanner
# ---------------------------------------------------------------------------


class TestGraphPlanner:
    """Verify plan generation for full and targeted execution."""

    def test_plan_all_blocks_returns_topo_order(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),), output_ports=(_audio_out(),)))
        graph.add_block(_make_block("c", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )
        graph.add_connection(
            Connection(
                source_block_id="b",
                source_output_name="out",
                target_block_id="c",
                target_input_name="in",
            )
        )

        planner = GraphPlanner()
        plan = planner.plan(graph)

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.execution_id) == 32
        # a must come before b, b before c
        ids = list(plan.ordered_block_ids)
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_plan_with_target_returns_only_upstream_chain(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),), output_ports=(_audio_out(),)))
        graph.add_block(_make_block("c", input_ports=(_audio_in(),)))
        graph.add_block(_make_block("d"))  # Disconnected block
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )
        graph.add_connection(
            Connection(
                source_block_id="b",
                source_output_name="out",
                target_block_id="c",
                target_input_name="in",
            )
        )

        planner = GraphPlanner()
        plan = planner.plan(graph, target_block_id="b")

        # Should include a and b (upstream of b + b itself), but NOT c or d
        assert set(plan.ordered_block_ids) == {"a", "b"}
        ids = list(plan.ordered_block_ids)
        assert ids.index("a") < ids.index("b")

    def test_plan_with_target_leaf_returns_full_chain(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),), output_ports=(_audio_out(),)))
        graph.add_block(_make_block("c", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )
        graph.add_connection(
            Connection(
                source_block_id="b",
                source_output_name="out",
                target_block_id="c",
                target_input_name="in",
            )
        )

        planner = GraphPlanner()
        plan = planner.plan(graph, target_block_id="c")

        assert set(plan.ordered_block_ids) == {"a", "b", "c"}

    def test_plan_with_target_root_returns_only_itself(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )

        planner = GraphPlanner()
        plan = planner.plan(graph, target_block_id="a")

        assert plan.ordered_block_ids == ("a",)

    def test_plan_disconnected_blocks_all_included(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("x"))
        graph.add_block(_make_block("y"))
        graph.add_block(_make_block("z"))

        planner = GraphPlanner()
        plan = planner.plan(graph)

        assert set(plan.ordered_block_ids) == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# ExecutionEngine
# ---------------------------------------------------------------------------


class TestExecutionEngine:
    """Verify block dispatch, output collection, fail-fast, and report publishing."""

    def _make_engine(
        self,
    ) -> tuple[Graph, RuntimeBus, ExecutionEngine]:
        graph = Graph()
        runtime_bus = RuntimeBus()
        engine = ExecutionEngine(graph, runtime_bus)
        return graph, runtime_bus, engine

    def test_runs_blocks_in_order_and_collects_outputs(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )

        exec_a = StubExecutor(output="audio_data")
        exec_b = StubExecutor(output="processed_data")
        engine.register_executor("TestType", exec_a)

        # Both blocks use same type, so same executor
        plan = GraphPlanner().plan(graph)
        result = engine.run(plan)

        assert isinstance(result, Ok)
        assert "a" in result.value
        assert "b" in result.value

    def test_executor_called_in_topo_order(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("first", block_type="TypeA", output_ports=(_audio_out(),)))
        graph.add_block(
            _make_block("second", block_type="TypeB", input_ports=(_audio_in(),))
        )
        graph.add_connection(
            Connection(
                source_block_id="first",
                source_output_name="out",
                target_block_id="second",
                target_input_name="in",
            )
        )

        exec_a = StubExecutor(output="out_a")
        exec_b = StubExecutor(output="out_b")
        engine.register_executor("TypeA", exec_a)
        engine.register_executor("TypeB", exec_b)

        plan = GraphPlanner().plan(graph)
        engine.run(plan)

        assert exec_a.called_with == ["first"]
        assert exec_b.called_with == ["second"]

    def test_fails_fast_on_first_block_error(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("a", block_type="Good", output_ports=(_audio_out(),)))
        graph.add_block(
            _make_block("b", block_type="Bad", input_ports=(_audio_in(),), output_ports=(_audio_out(),))
        )
        graph.add_block(_make_block("c", block_type="Good", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )
        graph.add_connection(
            Connection(
                source_block_id="b",
                source_output_name="out",
                target_block_id="c",
                target_input_name="in",
            )
        )

        good_exec = StubExecutor(output="ok")
        bad_exec = StubExecutor(should_fail=True)
        engine.register_executor("Good", good_exec)
        engine.register_executor("Bad", bad_exec)

        plan = GraphPlanner().plan(graph)
        result = engine.run(plan)

        assert isinstance(result, Err)
        assert "b" in str(result.error)
        # Block "c" should never have been called
        assert "c" not in good_exec.called_with

    def test_publishes_execution_started_and_completed_reports(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("a"))

        reports: list[RuntimeReport] = []
        runtime_bus.subscribe(reports.append)

        engine.register_executor("TestType", StubExecutor())
        plan = GraphPlanner().plan(graph)
        engine.run(plan)

        started = [r for r in reports if isinstance(r, ExecutionStartedReport)]
        completed = [r for r in reports if isinstance(r, ExecutionCompletedReport)]
        assert len(started) == 1
        assert started[0].block_id == "a"
        assert len(completed) == 1
        assert completed[0].block_id == "a"
        assert completed[0].success is True

    def test_publishes_completed_with_failure_on_error(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("a"))

        reports: list[RuntimeReport] = []
        runtime_bus.subscribe(reports.append)

        engine.register_executor("TestType", StubExecutor(should_fail=True))
        plan = GraphPlanner().plan(graph)
        engine.run(plan)

        completed = [r for r in reports if isinstance(r, ExecutionCompletedReport)]
        assert len(completed) == 1
        assert completed[0].success is False
        assert completed[0].error is not None

    def test_publishes_progress_reports(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("a"))

        reports: list[ProgressReport] = []
        runtime_bus.subscribe(lambda r: reports.append(r) if isinstance(r, ProgressReport) else None)

        engine.register_executor("TestType", StubExecutor())
        plan = GraphPlanner().plan(graph)
        engine.run(plan)

        assert len(reports) == 2
        assert reports[0].block_id == "a"
        assert reports[0].percent == 0.0
        assert reports[1].block_id == "a"
        assert reports[1].percent == 1.0

    def test_progress_reports_on_failure(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("a"))

        reports: list[ProgressReport] = []
        runtime_bus.subscribe(lambda r: reports.append(r) if isinstance(r, ProgressReport) else None)

        engine.register_executor("TestType", StubExecutor(should_fail=True))
        plan = GraphPlanner().plan(graph)
        engine.run(plan)

        # Only the start progress should have been published (0%), not completion (1.0)
        assert len(reports) == 1
        assert reports[0].percent == 0.0

    def test_unknown_block_type_returns_err(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("a", block_type="UnknownType"))

        plan = GraphPlanner().plan(graph)
        result = engine.run(plan)

        assert isinstance(result, Err)
        assert "No executor registered" in str(result.error)
        assert "UnknownType" in str(result.error)

    def test_block_not_in_graph_returns_err(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        # Create a plan referencing a block that doesn't exist
        plan = ExecutionPlan(execution_id="test-run", ordered_block_ids=("ghost",))

        result = engine.run(plan)

        assert isinstance(result, Err)
        assert "Block not found" in str(result.error)

    def test_execution_plan_is_frozen(self) -> None:
        plan = ExecutionPlan(execution_id="x", ordered_block_ids=("a", "b"))
        try:
            plan.execution_id = "y"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass

    def test_execution_context_holds_references(self) -> None:
        graph = Graph()
        runtime_bus = RuntimeBus()
        ctx = ExecutionContext(
            execution_id="run-1", graph=graph, progress_bus=runtime_bus
        )
        assert ctx.execution_id == "run-1"
        assert ctx.graph is graph
        assert ctx.progress_bus is runtime_bus

    def test_multiple_blocks_all_outputs_collected(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("x", block_type="Alpha"))
        graph.add_block(_make_block("y", block_type="Beta"))

        engine.register_executor("Alpha", StubExecutor(output=42))
        engine.register_executor("Beta", StubExecutor(output="hello"))

        plan = GraphPlanner().plan(graph)
        result = engine.run(plan)

        assert isinstance(result, Ok)
        assert result.value["x"] == {"out": 42}
        assert result.value["y"] == {"out": "hello"}

    def test_cancellation_stops_execution(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )

        cancel = threading.Event()
        cancel.set()  # Pre-cancel

        engine.register_executor("TestType", StubExecutor())
        plan = GraphPlanner().plan(graph)
        result = engine.run(plan, cancel_event=cancel)

        assert isinstance(result, Err)
        assert isinstance(result.error, OperationCancelledError)


# ---------------------------------------------------------------------------
# ExecutionContext data flow
# ---------------------------------------------------------------------------


class TestExecutionContextDataFlow:
    """Verify get_input/set_output resolve connections and check types."""

    def test_set_output_then_get_input_resolves(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )

        ctx = ExecutionContext(
            execution_id="test", graph=graph, progress_bus=RuntimeBus()
        )
        ctx.set_output("a", "out", "audio_data")
        result = ctx.get_input("b", "in")

        assert result == "audio_data"

    def test_get_input_returns_none_when_no_connection(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a"))

        ctx = ExecutionContext(
            execution_id="test", graph=graph, progress_bus=RuntimeBus()
        )
        result = ctx.get_input("a", "in")

        assert result is None

    def test_get_input_returns_none_when_upstream_not_produced(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )

        ctx = ExecutionContext(
            execution_id="test", graph=graph, progress_bus=RuntimeBus()
        )
        result = ctx.get_input("b", "in")

        assert result is None

    def test_get_input_type_check_raises_on_mismatch(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )

        ctx = ExecutionContext(
            execution_id="test", graph=graph, progress_bus=RuntimeBus()
        )
        ctx.set_output("a", "out", "not_an_int")

        with pytest.raises(ExecutionError, match="Type mismatch"):
            ctx.get_input("b", "in", expected_type=int)

    def test_get_input_type_check_passes_on_match(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )

        ctx = ExecutionContext(
            execution_id="test", graph=graph, progress_bus=RuntimeBus()
        )
        ctx.set_output("a", "out", 42)

        result = ctx.get_input("b", "in", expected_type=int)
        assert result == 42

    def test_set_output_overwrites_previous(self) -> None:
        graph = Graph()
        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(
            Connection(
                source_block_id="a",
                source_output_name="out",
                target_block_id="b",
                target_input_name="in",
            )
        )

        ctx = ExecutionContext(
            execution_id="test", graph=graph, progress_bus=RuntimeBus()
        )
        ctx.set_output("a", "out", "first")
        ctx.set_output("a", "out", "second")

        result = ctx.get_input("b", "in")
        assert result == "second"

    def test_context_has_cancel_event(self) -> None:
        ctx = ExecutionContext(
            execution_id="test", graph=Graph(), progress_bus=RuntimeBus()
        )
        assert isinstance(ctx.cancel_event, threading.Event)
        assert not ctx.cancel_event.is_set()


# ---------------------------------------------------------------------------
# Multi-port output handling
# ---------------------------------------------------------------------------


class MultiPortExecutor:
    """An executor that returns a dict of port_name -> value."""

    def __init__(self, outputs: dict[str, Any]) -> None:
        self._outputs = outputs
        self.called_with: list[str] = []

    def execute(self, block_id: str, context: ExecutionContext) -> Any:
        self.called_with.append(block_id)
        return ok(self._outputs)


class InputReadingExecutor:
    """An executor that reads an input port and returns the value."""

    def __init__(self, input_port: str) -> None:
        self._input_port = input_port
        self.received_input: Any = None

    def execute(self, block_id: str, context: ExecutionContext) -> Any:
        self.received_input = context.get_input(block_id, self._input_port)
        return ok(self.received_input)


class TestMultiPortOutput:
    """Verify multi-port output handling in the execution engine."""

    def _make_engine(self) -> tuple[Graph, RuntimeBus, ExecutionEngine]:
        graph = Graph()
        runtime_bus = RuntimeBus()
        engine = ExecutionEngine(graph, runtime_bus)
        return graph, runtime_bus, engine

    def test_dict_output_stores_per_port(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("sep", block_type="Separator", output_ports=(
            Port(name="drums", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="bass", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        )))

        engine.register_executor(
            "Separator",
            MultiPortExecutor({"drums": "drums_data", "bass": "bass_data"}),
        )

        plan = GraphPlanner().plan(graph)
        result = engine.run(plan)

        assert isinstance(result, Ok)
        assert result.value["sep"] == {"drums": "drums_data", "bass": "bass_data"}

    def test_single_value_uses_first_output_port_name(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("a", block_type="TypeA", output_ports=(
            Port(name="audio_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        )))
        graph.add_block(_make_block("b", block_type="TypeB", input_ports=(
            Port(name="audio_in", port_type=PortType.AUDIO, direction=Direction.INPUT),
        )))
        graph.add_connection(Connection(
            source_block_id="a", source_output_name="audio_out",
            target_block_id="b", target_input_name="audio_in",
        ))

        reader = InputReadingExecutor("audio_in")
        engine.register_executor("TypeA", StubExecutor(output="audio_data"))
        engine.register_executor("TypeB", reader)

        plan = GraphPlanner().plan(graph)
        result = engine.run(plan)

        assert isinstance(result, Ok)
        assert reader.received_input == "audio_data"

    def test_multi_port_feeds_downstream_correctly(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("sep", block_type="Separator", output_ports=(
            Port(name="drums", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="bass", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        )))
        graph.add_block(_make_block("proc", block_type="Processor", input_ports=(
            Port(name="audio_in", port_type=PortType.AUDIO, direction=Direction.INPUT),
        )))
        graph.add_connection(Connection(
            source_block_id="sep", source_output_name="drums",
            target_block_id="proc", target_input_name="audio_in",
        ))

        reader = InputReadingExecutor("audio_in")
        engine.register_executor(
            "Separator",
            MultiPortExecutor({"drums": "drums_data", "bass": "bass_data"}),
        )
        engine.register_executor("Processor", reader)

        plan = GraphPlanner().plan(graph)
        result = engine.run(plan)

        assert isinstance(result, Ok)
        assert reader.received_input == "drums_data"

    def test_multi_port_ignores_undeclared_extra_outputs_and_feeds_declared_port(self) -> None:
        graph, runtime_bus, engine = self._make_engine()
        graph.add_block(_make_block("sep", block_type="Separator", output_ports=(
            Port(name="drums_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="bass_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="vocals_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="other_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        )))
        graph.add_block(_make_block("proc", block_type="Processor", input_ports=(
            Port(name="audio_in", port_type=PortType.AUDIO, direction=Direction.INPUT),
        )))
        graph.add_connection(Connection(
            source_block_id="sep", source_output_name="drums_out",
            target_block_id="proc", target_input_name="audio_in",
        ))

        reader = InputReadingExecutor("audio_in")
        engine.register_executor(
            "Separator",
            MultiPortExecutor({"drums_out": "drums_data", "no_drums_out": "remainder_data"}),
        )
        engine.register_executor("Processor", reader)

        plan = GraphPlanner().plan(graph)
        result = engine.run(plan)

        assert isinstance(result, Ok)
        assert result.value["sep"] == {"drums_out": "drums_data"}
        assert reader.received_input == "drums_data"

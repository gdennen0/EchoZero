"""
Execution engine: Plans and runs blocks in topological order through the graph.
Exists because block execution requires dependency-aware scheduling with progress and event reporting.
Used by the Coordinator to orchestrate block runs; delegates actual work to BlockExecutor implementations.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from echozero.domain.graph import Graph
from echozero.errors import ExecutionError, OperationCancelledError
from echozero.progress import (
    ExecutionCompletedReport,
    ExecutionStartedReport,
    ProgressReport,
    RuntimeBus,
)
from echozero.result import Err, Ok, Result, err, is_err, ok

T = TypeVar("T")


@dataclass(frozen=True)
class ExecutionPlan:
    """An ordered list of block IDs to execute, identified by a unique run ID."""

    execution_id: str
    ordered_block_ids: tuple[str, ...]


@dataclass
class ExecutionContext:
    """Runtime context passed to each BlockExecutor during a plan run."""

    execution_id: str
    graph: Graph
    progress_bus: RuntimeBus
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _outputs: dict[tuple[str, str], Any] = field(default_factory=dict)

    def get_input(
        self,
        block_id: str,
        input_port_name: str,
        expected_type: type[T] = object,  # type: ignore[assignment]
    ) -> T | None:
        """Resolve an input port's value by looking up the upstream connection's output.

        Returns None if no connection exists or the upstream output hasn't been produced yet.
        Raises ExecutionError if the value doesn't match expected_type.
        """
        # Find the connection that feeds this input port
        for conn in self.graph.connections:
            if conn.target_block_id == block_id and conn.target_input_name == input_port_name:
                value = self._outputs.get(
                    (conn.source_block_id, conn.source_output_name)
                )
                if value is None:
                    return None
                if expected_type is not object and not isinstance(value, expected_type):
                    raise ExecutionError(
                        f"Type mismatch on input '{input_port_name}' of block '{block_id}': "
                        f"expected {expected_type.__name__}, got {type(value).__name__}"
                    )
                return value  # type: ignore[return-value]
        return None

    def set_output(self, block_id: str, output_port_name: str, value: Any) -> None:
        """Store a block's output value, keyed by (block_id, port_name)."""
        self._outputs[(block_id, output_port_name)] = value


class BlockExecutor(Protocol):
    """Protocol for block-type-specific execution logic."""

    def execute(self, block_id: str, context: ExecutionContext) -> Result[Any]:
        """Execute the block and return its output or an error."""
        ...


class GraphPlanner:
    """Creates execution plans from the graph's topological order."""

    def plan(self, graph: Graph, target_block_id: str | None = None) -> ExecutionPlan:
        """Build an execution plan — all blocks or only upstream of a target."""
        execution_id = uuid.uuid4().hex
        topo_order = graph.topological_sort()

        if target_block_id is None:
            return ExecutionPlan(
                execution_id=execution_id,
                ordered_block_ids=tuple(topo_order),
            )

        # Collect upstream dependencies: everything that target depends on
        upstream = self._upstream_of(graph, target_block_id)
        upstream.add(target_block_id)

        # Filter topo order to only include upstream blocks, preserving order
        filtered = [bid for bid in topo_order if bid in upstream]
        return ExecutionPlan(
            execution_id=execution_id,
            ordered_block_ids=tuple(filtered),
        )

    @staticmethod
    def _upstream_of(graph: Graph, block_id: str) -> set[str]:
        """Return all transitive ancestors of a block (reverse of downstream_of)."""
        # Build reverse adjacency: for each connection, target depends on source
        reverse_adj: dict[str, list[str]] = {bid: [] for bid in graph.blocks}
        for conn in graph.connections:
            reverse_adj[conn.target_block_id].append(conn.source_block_id)

        visited: set[str] = set()
        stack = list(reverse_adj[block_id])
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(reverse_adj[current])

        return visited


class ExecutionEngine:
    """Runs an execution plan, dispatching each block to its registered executor."""

    def __init__(
        self,
        graph: Graph,
        runtime_bus: RuntimeBus,
    ) -> None:
        self._graph = graph
        self._runtime_bus = runtime_bus
        self._executors: dict[str, BlockExecutor] = {}

    def register_executor(self, block_type: str, executor: BlockExecutor) -> None:
        """Register an executor for a given block type."""
        self._executors[block_type] = executor

    def run(
        self,
        plan: ExecutionPlan,
        cancel_event: threading.Event | None = None,
    ) -> Result[dict[str, Any]]:
        """Execute all blocks in plan order. Fail-fast on first error or cancellation."""
        outputs: dict[str, Any] = {}
        _cancel = cancel_event or threading.Event()
        context = ExecutionContext(
            execution_id=plan.execution_id,
            graph=self._graph,
            progress_bus=self._runtime_bus,
            cancel_event=_cancel,
        )

        for block_id in plan.ordered_block_ids:
            # Check cancellation between blocks
            if _cancel.is_set():
                return err(OperationCancelledError("Execution cancelled"))

            block = self._graph.blocks.get(block_id)
            if block is None:
                return err(ExecutionError(f"Block not found in graph: {block_id}"))

            executor = self._executors.get(block.block_type)
            if executor is None:
                return err(
                    ExecutionError(
                        f"No executor registered for block type: {block.block_type}"
                    )
                )

            # Signal start via RuntimeBus
            self._runtime_bus.publish(
                ExecutionStartedReport(
                    block_id=block_id,
                    execution_id=plan.execution_id,
                )
            )
            self._runtime_bus.publish(
                ProgressReport(
                    block_id=block_id,
                    phase="execute",
                    percent=0.0,
                    message=f"Starting {block.block_type}",
                )
            )

            # Execute
            try:
                result = executor.execute(block_id, context)
            except Exception as exc:
                self._runtime_bus.publish(
                    ExecutionCompletedReport(
                        block_id=block_id,
                        execution_id=plan.execution_id,
                        success=False,
                        error=f"Executor raised: {exc}",
                    )
                )
                chained = ExecutionError(f"Executor for '{block.block_type}' raised: {exc}")
                chained.__cause__ = exc
                return err(chained)

            if is_err(result):
                # Signal failure
                assert isinstance(result, Err)
                self._runtime_bus.publish(
                    ExecutionCompletedReport(
                        block_id=block_id,
                        execution_id=plan.execution_id,
                        success=False,
                        error=str(result.error),
                    )
                )
                return err(result.error)

            # Signal success and store output
            # Output dict is ALWAYS normalized to {port_name: value}.
            # This ensures the Orchestrator can resolve by port name without
            # guessing whether a processor returned a dict or a single value.
            assert isinstance(result, Ok)
            result_value = result.value

            # Detect multi-port: executor returned a dict whose keys match
            # the block's declared output port names.
            output_port_names = {p.name for p in block.output_ports}
            is_multi_port = (
                isinstance(result_value, dict)
                and len(output_port_names) > 1
                and result_value.keys() <= output_port_names
            )

            if is_multi_port:
                # Multi-port: executor returned {port_name: value}
                for pname, pvalue in result_value.items():
                    context.set_output(block_id, pname, pvalue)
                outputs[block_id] = result_value
            else:
                # Single-port (or single output port): wrap in {port_name: value}
                pname = block.output_ports[0].name if block.output_ports else "out"
                context.set_output(block_id, pname, result_value)
                outputs[block_id] = {pname: result_value}
            self._runtime_bus.publish(
                ExecutionCompletedReport(
                    block_id=block_id,
                    execution_id=plan.execution_id,
                    success=True,
                )
            )
            self._runtime_bus.publish(
                ProgressReport(
                    block_id=block_id,
                    phase="execute",
                    percent=1.0,
                    message=f"Completed {block.block_type}",
                )
            )

        return ok(outputs)

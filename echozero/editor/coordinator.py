"""
Coordinator: Thin orchestration layer implementing the core reactive loop.
Exists because the system needs a single place that ties mutation → dirty → cancel → launch.
Used by the application layer to request runs, propagate staleness, and cancel in-flight work.
"""

from __future__ import annotations

import threading
from typing import Any

from echozero.editor.cache import ExecutionCache
from echozero.editor.staleness import (
    StaleReason,
    StaleTracker,
    connection_changed_reason,
    setting_changed_reason,
)
from echozero.domain.enums import BlockState
from echozero.domain.events import (
    BlockRemovedEvent,
    ConnectionAddedEvent,
    ConnectionRemovedEvent,
    SettingsChangedEvent,
)
from echozero.domain.graph import Graph
from echozero.event_bus import EventBus
from echozero.execution import ExecutionEngine, GraphPlanner
from echozero.editor.pipeline import Pipeline
from echozero.progress import RuntimeBus
from echozero.result import Err, Result, err, is_ok, ok, unwrap


# ---------------------------------------------------------------------------
# Pure scheduling function
# ---------------------------------------------------------------------------


def ready_nodes(
    graph: Graph,
    dirty: set[str],
    running: set[str],
    cache: ExecutionCache,
) -> set[str]:
    """Return block IDs whose upstream dependencies are all satisfied.

    A block is ready when:
    - It IS in the dirty set (needs execution)
    - It is NOT currently running
    - All upstream neighbors are NOT dirty and NOT running
    - All connected input ports have cached upstream outputs (or are root nodes)
    """
    # Build reverse adjacency: block → set of blocks it depends on
    deps: dict[str, set[str]] = {bid: set() for bid in graph.blocks}
    for conn in graph.connections:
        deps[conn.target_block_id].add(conn.source_block_id)

    result: set[str] = set()
    for block_id in dirty:
        if block_id in running:
            continue
        if block_id not in graph.blocks:
            continue
        # All upstream must be not-dirty and not-running
        upstream = deps[block_id]
        if not all(u not in dirty and u not in running for u in upstream):
            continue
        # All connected input ports must have cached upstream outputs
        all_inputs_cached = True
        for conn in graph.connections:
            if conn.target_block_id == block_id:
                if not cache.has_valid_output(conn.source_block_id, conn.source_output_name):
                    all_inputs_cached = False
                    break
        if all_inputs_cached:
            result.add(block_id)
    return result


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class Coordinator:
    """Implements the core operations: request_run, cancel, propagate_stale, auto-evaluation."""

    def __init__(
        self,
        graph: Graph,
        pipeline: Pipeline,
        engine: ExecutionEngine,
        cache: ExecutionCache,
        runtime_bus: RuntimeBus,
        stale_tracker: StaleTracker | None = None,
    ) -> None:
        self._graph = graph
        self._pipeline = pipeline
        self._engine = engine
        self._cache = cache
        self._runtime_bus = runtime_bus
        self._stale_tracker = stale_tracker or StaleTracker()
        self._cancel_event: threading.Event = threading.Event()
        self._executing: bool = False
        self._auto_evaluate: bool = False

    @property
    def stale_tracker(self) -> StaleTracker:
        """Access the stale reason tracker for UI queries."""
        return self._stale_tracker

    @property
    def is_executing(self) -> bool:
        """Whether an execution is currently in progress."""
        return self._executing

    @property
    def auto_evaluate(self) -> bool:
        """When True, document changes trigger automatic re-evaluation."""
        return self._auto_evaluate

    @auto_evaluate.setter
    def auto_evaluate(self, value: bool) -> None:
        self._auto_evaluate = value

    def request_run(self, target: str | None = None) -> Result[str]:
        """Plan and execute blocks, caching results per-port.

        Returns the execution_id on success, or an Err on failure.
        """
        if self._executing:
            from echozero.errors import ExecutionError
            return err(ExecutionError("Execution already in progress. Cancel first."))
        self._cancel_event.clear()
        planner = GraphPlanner()
        plan = planner.plan(self._graph, target_block_id=target)

        self._executing = True
        try:
            result = self._engine.run(plan, cancel_event=self._cancel_event)
        finally:
            self._executing = False

        if is_ok(result):
            outputs = unwrap(result)
            for block_id, value in outputs.items():
                if isinstance(value, dict):
                    # Multi-port: executor returned {port_name: value}
                    for port_name, port_value in value.items():
                        self._cache.store(block_id, port_name, port_value, plan.execution_id)
                else:
                    # Single-port: use first output port name (or 'out' as fallback)
                    block = self._graph.blocks[block_id]
                    port_name = block.output_ports[0].name if block.output_ports else "out"
                    self._cache.store(block_id, port_name, value, plan.execution_id)
                # Mark block as FRESH and clear stale reasons
                self._graph.set_block_state(block_id, BlockState.FRESH)
                self._stale_tracker.clear(block_id)
            return ok(plan.execution_id)

        assert isinstance(result, Err)
        return err(result.error)

    def cancel(self) -> None:
        """Signal cancellation to in-flight execution."""
        self._cancel_event.set()

    def propagate_stale(
        self,
        block_id: str,
        reason: StaleReason | None = None,
    ) -> set[str]:
        """Mark a block and all downstream as STALE, invalidating their cache entries.

        If a reason is provided, it's recorded on every affected block so the UI
        can show WHY each block is stale. Reasons accumulate — multiple changes
        before re-run produce multiple reasons per block.

        Returns the set of affected block IDs.
        """
        affected = self._cache.invalidate_downstream(block_id, self._graph)
        for bid in affected:
            if bid in self._graph.blocks:
                self._graph.set_block_state(bid, BlockState.STALE)
        if reason is not None:
            self._stale_tracker.add_reason_to_downstream(affected, reason)
        return affected

    # -- Document bus wiring ------------------------------------------------

    def subscribe_to_document_bus(self, document_bus: EventBus) -> None:
        """Wire DocumentBus events to propagate_stale and optional request_run."""
        document_bus.subscribe(SettingsChangedEvent, self._on_settings_changed)
        document_bus.subscribe(ConnectionAddedEvent, self._on_connection_added)
        document_bus.subscribe(ConnectionRemovedEvent, self._on_connection_removed)
        document_bus.subscribe(BlockRemovedEvent, self._on_block_removed)

    def unsubscribe_from_document_bus(self, document_bus: EventBus) -> None:
        """Remove all coordinator subscriptions from the document bus."""
        document_bus.unsubscribe(SettingsChangedEvent, self._on_settings_changed)
        document_bus.unsubscribe(ConnectionAddedEvent, self._on_connection_added)
        document_bus.unsubscribe(ConnectionRemovedEvent, self._on_connection_removed)
        document_bus.unsubscribe(BlockRemovedEvent, self._on_block_removed)

    def _on_settings_changed(self, event: SettingsChangedEvent) -> None:
        block = self._graph.blocks.get(event.block_id)
        block_name = block.name if block else event.block_id
        reason = setting_changed_reason(
            event.block_id, block_name, event.setting_key,
            old_value=event.old_value,
            new_value=event.new_value,
        )
        self.propagate_stale(event.block_id, reason=reason)
        if self._auto_evaluate:
            self.request_run()

    def _on_connection_added(self, event: ConnectionAddedEvent) -> None:
        target = self._graph.blocks.get(event.target_block_id)
        target_name = target.name if target else event.target_block_id
        reason = connection_changed_reason(
            event.target_block_id, target_name, "added",
        )
        self.propagate_stale(event.target_block_id, reason=reason)
        if self._auto_evaluate:
            self.request_run()

    def _on_connection_removed(self, event: ConnectionRemovedEvent) -> None:
        target = self._graph.blocks.get(event.target_block_id)
        target_name = target.name if target else event.target_block_id
        reason = connection_changed_reason(
            event.target_block_id, target_name, "removed",
        )
        self.propagate_stale(event.target_block_id, reason=reason)
        if self._auto_evaluate:
            self.request_run()

    def _on_block_removed(self, event: BlockRemovedEvent) -> None:
        self._cache.invalidate(event.block_id)
        self._stale_tracker.clear(event.block_id)

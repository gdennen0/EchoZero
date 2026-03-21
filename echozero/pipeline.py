"""
Pipeline: Single entry point for all mutations to the EchoZero domain model.
Exists because direct graph manipulation is banned — every change flows through dispatch().
Routes commands to handlers, collects domain events, and flushes them to EventBus on success.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from echozero.commands import Command
from echozero.domain.events import DomainEvent
from echozero.domain.graph import Graph
from echozero.errors import DomainError
from echozero.event_bus import EventBus
from echozero.result import Err, Ok, Result, err, ok

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Command context — mutable workspace for handlers
# ---------------------------------------------------------------------------


@dataclass
class CommandContext:
    """Mutable workspace passed to command handlers during dispatch."""

    graph: Graph
    collected_events: list[DomainEvent] = field(default_factory=list)

    def collect(self, event: DomainEvent) -> None:
        """Stage a domain event to be flushed after successful commit."""
        self.collected_events.append(event)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Routes commands to registered handlers, manages event collect/flush lifecycle."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._graph = Graph()
        self._handlers: dict[type[Command], Callable[..., Any]] = {}

    @property
    def graph(self) -> Graph:
        """Read-only access to the pipeline's graph for queries."""
        return self._graph

    def register(self, command_type: type[Command], handler_fn: Callable[..., Any]) -> None:
        """Bind a command type to its handler function."""
        self._handlers[command_type] = handler_fn

    def dispatch(self, command: Command) -> Result[Any]:
        """Execute a command through its registered handler.

        On success: flushes collected events to EventBus, returns Ok.
        On failure: discards collected events, returns Err.
        """
        handler = self._handlers.get(type(command))
        if handler is None:
            return err(DomainError(f"No handler registered for {type(command).__name__}"))

        context = CommandContext(graph=self._graph)

        try:
            result_value = handler(command, context)
        except Exception as exc:
            # Discard collected events on failure
            logger.warning("Handler for %s failed: %s", type(command).__name__, exc)
            return err(exc)

        # Success: flush events to bus
        for event in context.collected_events:
            self._event_bus.publish(event)

        return ok(result_value)

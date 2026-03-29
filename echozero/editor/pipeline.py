"""
Pipeline: Single entry point for all mutations to the EchoZero domain model.
Exists because direct graph manipulation is banned — every change flows through dispatch().
Routes commands to handlers, collects domain events, and flushes them to EventBus on success.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from echozero.editor.commands import (
    AddBlockCommand,
    AddConnectionCommand,
    ChangeBlockSettingsCommand,
    Command,
    RemoveBlockCommand,
    RemoveConnectionCommand,
)
from echozero.domain.enums import Direction, PortType
from echozero.domain.events import (
    BlockAddedEvent,
    BlockRemovedEvent,
    ConnectionAddedEvent,
    ConnectionRemovedEvent,
    DomainEvent,
    SettingsChangedEvent,
    create_event_id,
)
from echozero.domain.graph import Graph
from echozero.domain.types import Block, BlockSettings, Connection, Port
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
        _register_default_handlers(self)

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


# ---------------------------------------------------------------------------
# Default command handlers
# ---------------------------------------------------------------------------


def _handle_add_block(command: AddBlockCommand, context: CommandContext) -> str:
    """Create a Block from the command and add it to the graph."""
    block = Block(
        id=command.block_id,
        name=command.name,
        block_type=command.block_type,
        category=command.category,
        input_ports=tuple(
            Port(name=n, port_type=PortType[pt], direction=Direction[d])
            for n, pt, d in command.input_ports
        ),
        output_ports=tuple(
            Port(name=n, port_type=PortType[pt], direction=Direction[d])
            for n, pt, d in command.output_ports
        ),
        control_ports=tuple(
            Port(name=n, port_type=PortType[pt], direction=Direction[d])
            for n, pt, d in command.control_ports
        ),
        settings=BlockSettings(dict(command.settings_entries)),
    )
    context.graph.add_block(block)
    context.collect(
        BlockAddedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=command.correlation_id,
            block_id=command.block_id,
            block_type=command.block_type,
        )
    )
    return command.block_id


def _handle_remove_block(command: RemoveBlockCommand, context: CommandContext) -> None:
    """Remove a block and its connections from the graph."""
    context.graph.remove_block(command.block_id)
    context.collect(
        BlockRemovedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=command.correlation_id,
            block_id=command.block_id,
        )
    )


def _handle_add_connection(command: AddConnectionCommand, context: CommandContext) -> None:
    """Create a Connection and add it to the graph."""
    conn = Connection(
        source_block_id=command.source_block_id,
        source_output_name=command.source_output_name,
        target_block_id=command.target_block_id,
        target_input_name=command.target_input_name,
    )
    context.graph.add_connection(conn)
    context.collect(
        ConnectionAddedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=command.correlation_id,
            source_block_id=command.source_block_id,
            target_block_id=command.target_block_id,
        )
    )


def _handle_remove_connection(command: RemoveConnectionCommand, context: CommandContext) -> None:
    """Remove a connection from the graph."""
    conn = Connection(
        source_block_id=command.source_block_id,
        source_output_name=command.source_output_name,
        target_block_id=command.target_block_id,
        target_input_name=command.target_input_name,
    )
    context.graph.remove_connection(conn)
    context.collect(
        ConnectionRemovedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=command.correlation_id,
            source_block_id=command.source_block_id,
            target_block_id=command.target_block_id,
        )
    )


def _handle_change_settings(command: ChangeBlockSettingsCommand, context: CommandContext) -> None:
    """Update a single setting on a block."""
    block = context.graph.blocks[command.block_id]
    old_value = block.settings.get(command.setting_key)
    new_entries = dict(block.settings)
    new_entries[command.setting_key] = command.new_value
    context.graph.blocks[command.block_id] = replace(
        block, settings=BlockSettings(new_entries)
    )
    context.collect(
        SettingsChangedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=command.correlation_id,
            block_id=command.block_id,
            setting_key=command.setting_key,
            old_value=old_value,
            new_value=command.new_value,
        )
    )


def _register_default_handlers(pipeline: Pipeline) -> None:
    """Register the standard set of command handlers on a Pipeline instance."""
    pipeline.register(AddBlockCommand, _handle_add_block)
    pipeline.register(RemoveBlockCommand, _handle_remove_block)
    pipeline.register(AddConnectionCommand, _handle_add_connection)
    pipeline.register(RemoveConnectionCommand, _handle_remove_connection)
    pipeline.register(ChangeBlockSettingsCommand, _handle_change_settings)


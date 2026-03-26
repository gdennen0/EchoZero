"""
Pipeline tests: Verify command dispatch, event collect/flush, and graph mutations.
Exists because the pipeline is the single entry point for all mutations — correctness is critical.
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

import time
import uuid

import pytest

from echozero.commands import (
    AddBlockCommand,
    AddConnectionCommand,
    ChangeBlockSettingsCommand,
    Command,
    RemoveBlockCommand,
    RemoveConnectionCommand,
)
from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.events import (
    BlockAddedEvent,
    BlockRemovedEvent,
    ConnectionAddedEvent,
    DomainEvent,
    create_event_id,
)
from echozero.domain.types import Block, Connection, Port
from echozero.event_bus import EventBus
from echozero.pipeline import CommandContext, Pipeline
from echozero.result import Err, Ok

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_add_block_command(
    block_id: str | None = None,
    name: str = "TestBlock",
    block_type: str = "LoadAudio",
    category: BlockCategory = BlockCategory.PROCESSOR,
    input_ports: tuple[tuple[str, str, str], ...] = (),
    output_ports: tuple[tuple[str, str, str], ...] = (),
) -> AddBlockCommand:
    """Create an AddBlockCommand with sensible defaults."""
    return AddBlockCommand(
        block_id=block_id or uuid.uuid4().hex,
        name=name,
        block_type=block_type,
        category=category,
        input_ports=input_ports,
        output_ports=output_ports,
    )


def _make_block_added_event(
    correlation_id: str = "cmd-1",
    block_id: str = "b1",
    block_type: str = "LoadAudio",
) -> BlockAddedEvent:
    """Create a BlockAddedEvent with sensible defaults."""
    return BlockAddedEvent(
        event_id=create_event_id(),
        timestamp=time.time(),
        correlation_id=correlation_id,
        block_id=block_id,
        block_type=block_type,
    )


def _add_block_handler(command: AddBlockCommand, context: CommandContext) -> str:
    """Standard handler that adds a block to the graph and emits an event."""
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


def _remove_block_handler(command: RemoveBlockCommand, context: CommandContext) -> str:
    """Standard handler that removes a block from the graph and emits an event."""
    context.graph.remove_block(command.block_id)
    context.collect(
        BlockRemovedEvent(
            event_id=create_event_id(),
            timestamp=time.time(),
            correlation_id=command.correlation_id,
            block_id=command.block_id,
        )
    )
    return command.block_id


def _add_connection_handler(command: AddConnectionCommand, context: CommandContext) -> None:
    """Standard handler that adds a connection to the graph and emits an event."""
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


@pytest.fixture
def bus() -> EventBus:
    """Provide a fresh EventBus for each test."""
    return EventBus()


@pytest.fixture
def pipeline(bus: EventBus) -> Pipeline:
    """Provide a Pipeline pre-registered with standard handlers."""
    p = Pipeline(event_bus=bus)
    p.register(AddBlockCommand, _add_block_handler)
    p.register(RemoveBlockCommand, _remove_block_handler)
    p.register(AddConnectionCommand, _add_connection_handler)
    return p


# ---------------------------------------------------------------------------
# Handler registration and dispatch
# ---------------------------------------------------------------------------


class TestHandlerDispatch:
    """Verify command routing and handler invocation."""

    def test_registered_handler_called_with_correct_command(self, bus: EventBus) -> None:
        received_commands: list[Command] = []

        def spy_handler(command: Command, context: CommandContext) -> None:
            received_commands.append(command)

        p = Pipeline(event_bus=bus)
        p.register(AddBlockCommand, spy_handler)

        cmd = _make_add_block_command()
        p.dispatch(cmd)

        assert len(received_commands) == 1
        assert received_commands[0] is cmd

    def test_dispatch_returns_ok_on_success(self, pipeline: Pipeline) -> None:
        cmd = _make_add_block_command(block_id="b1")
        result = pipeline.dispatch(cmd)

        assert isinstance(result, Ok)
        assert result.value == "b1"

    def test_dispatch_returns_err_on_handler_exception(self, pipeline: Pipeline) -> None:
        # Adding duplicate block triggers ValidationError
        cmd = _make_add_block_command(block_id="b1")
        pipeline.dispatch(cmd)

        result = pipeline.dispatch(cmd)

        assert isinstance(result, Err)
        assert "Duplicate block ID" in str(result.error)

    def test_unknown_command_returns_err(self, bus: EventBus) -> None:
        p = Pipeline(event_bus=bus)
        cmd = ChangeBlockSettingsCommand(block_id="b1", setting_key="vol", new_value=1.0)
        result = p.dispatch(cmd)

        assert isinstance(result, Err)
        assert "No handler registered" in str(result.error)

    def test_multiple_handlers_for_different_types(self, bus: EventBus) -> None:
        add_calls: list[str] = []
        remove_calls: list[str] = []

        def on_add(cmd: AddBlockCommand, ctx: CommandContext) -> None:
            add_calls.append(cmd.block_id)

        def on_remove(cmd: RemoveBlockCommand, ctx: CommandContext) -> None:
            remove_calls.append(cmd.block_id)

        p = Pipeline(event_bus=bus)
        p.register(AddBlockCommand, on_add)
        p.register(RemoveBlockCommand, on_remove)

        p.dispatch(_make_add_block_command(block_id="b1"))
        p.dispatch(RemoveBlockCommand(block_id="b2"))

        assert add_calls == ["b1"]
        assert remove_calls == ["b2"]


# ---------------------------------------------------------------------------
# Event collect/flush lifecycle
# ---------------------------------------------------------------------------


class TestEventLifecycle:
    """Verify events are flushed on success and discarded on failure."""

    def test_events_flushed_to_bus_on_success(self, bus: EventBus, pipeline: Pipeline) -> None:
        received: list[DomainEvent] = []
        bus.subscribe(BlockAddedEvent, received.append)

        cmd = _make_add_block_command(block_id="b1")
        pipeline.dispatch(cmd)

        assert len(received) == 1
        assert isinstance(received[0], BlockAddedEvent)
        assert received[0].block_id == "b1"

    def test_events_discarded_on_failure(self, bus: EventBus) -> None:
        received: list[DomainEvent] = []
        bus.subscribe(BlockAddedEvent, received.append)

        def failing_handler(command: Command, context: CommandContext) -> None:
            context.collect(_make_block_added_event(block_id="ghost"))
            raise RuntimeError("Handler exploded")

        p = Pipeline(event_bus=bus)
        p.register(AddBlockCommand, failing_handler)

        result = p.dispatch(_make_add_block_command())

        assert isinstance(result, Err)
        assert len(received) == 0


# ---------------------------------------------------------------------------
# Graph mutations through pipeline
# ---------------------------------------------------------------------------


class TestGraphMutations:
    """Verify commands correctly mutate the graph via pipeline dispatch."""

    def test_add_block_creates_block_in_graph(self, pipeline: Pipeline) -> None:
        cmd = _make_add_block_command(block_id="b1", name="Load Audio")
        pipeline.dispatch(cmd)

        assert "b1" in pipeline.graph.blocks
        assert pipeline.graph.blocks["b1"].name == "Load Audio"

    def test_remove_block_removes_from_graph(self, pipeline: Pipeline) -> None:
        pipeline.dispatch(_make_add_block_command(block_id="b1"))
        pipeline.dispatch(RemoveBlockCommand(block_id="b1"))

        assert "b1" not in pipeline.graph.blocks

    def test_add_connection_creates_connection_in_graph(self, pipeline: Pipeline) -> None:
        pipeline.dispatch(
            _make_add_block_command(
                block_id="src",
                output_ports=(("out", "AUDIO", "OUTPUT"),),
            )
        )
        pipeline.dispatch(
            _make_add_block_command(
                block_id="tgt",
                input_ports=(("in", "AUDIO", "INPUT"),),
            )
        )

        cmd = AddConnectionCommand(
            source_block_id="src",
            source_output_name="out",
            target_block_id="tgt",
            target_input_name="in",
        )
        result = pipeline.dispatch(cmd)

        assert isinstance(result, Ok)
        assert len(pipeline.graph.connections) == 1
        assert pipeline.graph.connections[0].source_block_id == "src"
        assert pipeline.graph.connections[0].target_block_id == "tgt"

    def test_remove_block_cascades_connection_removal(self, pipeline: Pipeline) -> None:
        pipeline.dispatch(
            _make_add_block_command(
                block_id="src",
                output_ports=(("out", "AUDIO", "OUTPUT"),),
            )
        )
        pipeline.dispatch(
            _make_add_block_command(
                block_id="tgt",
                input_ports=(("in", "AUDIO", "INPUT"),),
            )
        )
        pipeline.dispatch(
            AddConnectionCommand(
                source_block_id="src",
                source_output_name="out",
                target_block_id="tgt",
                target_input_name="in",
            )
        )

        assert len(pipeline.graph.connections) == 1

        pipeline.dispatch(RemoveBlockCommand(block_id="src"))

        assert len(pipeline.graph.connections) == 0
        assert "src" not in pipeline.graph.blocks
        assert "tgt" in pipeline.graph.blocks


# ---------------------------------------------------------------------------
# Command properties
# ---------------------------------------------------------------------------


class TestCommandProperties:
    """Verify command identity and undoability classification."""

    def test_editable_commands_are_undoable(self) -> None:
        assert AddBlockCommand(block_id="b1").is_undoable is True
        assert RemoveBlockCommand(block_id="b1").is_undoable is True
        assert (
            AddConnectionCommand(
                source_block_id="s",
                source_output_name="o",
                target_block_id="t",
                target_input_name="i",
            ).is_undoable
            is True
        )
        assert (
            RemoveConnectionCommand(
                source_block_id="s",
                source_output_name="o",
                target_block_id="t",
                target_input_name="i",
            ).is_undoable
            is True
        )

    def test_change_settings_command_is_undoable(self) -> None:
        assert ChangeBlockSettingsCommand(block_id="b1", setting_key="x", new_value=1).is_undoable is True

    def test_command_ids_are_unique(self) -> None:
        cmd_a = AddBlockCommand(block_id="b1")
        cmd_b = AddBlockCommand(block_id="b2")

        assert len(cmd_a.command_id) == 32
        assert cmd_a.command_id != cmd_b.command_id

    def test_correlation_id_auto_generated(self) -> None:
        cmd = AddBlockCommand(block_id="b1")

        assert len(cmd.correlation_id) == 32
        assert cmd.correlation_id != cmd.command_id

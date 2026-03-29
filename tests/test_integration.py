"""
Integration tests: End-to-end verification of multi-block pipeline execution.
Exists because unit tests verify components in isolation — these prove the full architecture works together.
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import replace
from typing import Any

import pytest

from echozero.editor.cache import ExecutionCache
from echozero.editor.commands import (
    AddBlockCommand,
    AddConnectionCommand,
    ChangeBlockSettingsCommand,
    Command,
)
from echozero.editor.coordinator import Coordinator
from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.events import (
    BlockAddedEvent,
    ConnectionAddedEvent,
    DomainEvent,
    SettingsChangedEvent,
    create_event_id,
)
from echozero.domain.graph import Graph
from echozero.domain.types import (
    AudioData,
    Block,
    BlockSettings,
    Connection,
    Event,
    EventData,
    Layer,
    Port,
)
from echozero.errors import OperationCancelledError
from echozero.event_bus import EventBus
from echozero.execution import ExecutionContext, ExecutionEngine, GraphPlanner
from echozero.editor.pipeline import CommandContext, Pipeline
from echozero.processors.detect_onsets import DetectOnsetsProcessor
from echozero.processors.load_audio import AudioFileInfo, LoadAudioProcessor
from echozero.progress import RuntimeBus, RuntimeReport
from echozero.result import Err, Ok, err, ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_out(name: str = "audio_out") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _audio_in(name: str = "audio_in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


def _event_out(name: str = "event_out") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.OUTPUT)


def _event_in(name: str = "event_in") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.INPUT)


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
        settings=BlockSettings(settings or {}),
        state=state,
    )


# Pipeline command handlers

def _add_block_handler(command: AddBlockCommand, context: CommandContext) -> str:
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


def _add_connection_handler(command: AddConnectionCommand, context: CommandContext) -> None:
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


def _change_settings_handler(command: ChangeBlockSettingsCommand, context: CommandContext) -> None:
    block = context.graph.blocks[command.block_id]
    new_entries = dict(block.settings)
    new_entries[command.setting_key] = command.new_value
    context.graph.replace_block(replace(
        block, settings=BlockSettings(new_entries)
    ))
    old_value = block.settings.get(command.setting_key)
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


def _make_full_stack() -> tuple[Graph, EventBus, Pipeline, ExecutionEngine, ExecutionCache, Coordinator, RuntimeBus]:
    """Build a full coordinator + pipeline + engine stack for integration tests.

    Uses pipeline.graph as the single shared graph — commands mutate it directly.
    """
    event_bus = EventBus()
    pipeline = Pipeline(event_bus)
    pipeline.register(AddBlockCommand, _add_block_handler)
    pipeline.register(AddConnectionCommand, _add_connection_handler)
    pipeline.register(ChangeBlockSettingsCommand, _change_settings_handler)
    graph = pipeline.graph  # Shared graph — same instance everywhere
    runtime_bus = RuntimeBus()
    cache = ExecutionCache()
    engine = ExecutionEngine(graph, runtime_bus)
    coordinator = Coordinator(graph, pipeline, engine, cache, runtime_bus)
    return graph, event_bus, pipeline, engine, cache, coordinator, runtime_bus


class StubExecutor:
    """A test executor that returns a fixed value or error."""

    def __init__(self, output: Any = "done", should_fail: bool = False) -> None:
        self._output = output
        self._should_fail = should_fail
        self.called_with: list[str] = []

    def execute(self, block_id: str, context: ExecutionContext) -> Any:
        self.called_with.append(block_id)
        if self._should_fail:
            return err(OperationCancelledError(f"Block {block_id} failed"))
        return ok(self._output)


# ---------------------------------------------------------------------------
# Pipeline Build + Execute
# ---------------------------------------------------------------------------


class TestPipelineBuildAndExecute:
    """Verify a 2-block chain built via Pipeline commands and executed via Coordinator."""

    def test_load_audio_to_detect_onsets_chain(self, tmp_path: Any) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        graph, event_bus, pipeline, engine, cache, coordinator, _ = _make_full_stack()

        # Build graph via Pipeline commands
        pipeline.dispatch(AddBlockCommand(
            block_id="load1",
            name="Load Audio",
            block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(("audio_out", "AUDIO", "OUTPUT"),),
        ))
        pipeline.dispatch(AddBlockCommand(
            block_id="onset1",
            name="Detect Onsets",
            block_type="DetectOnsets",
            category=BlockCategory.PROCESSOR,
            input_ports=(("audio_in", "AUDIO", "INPUT"),),
            output_ports=(("event_out", "EVENT", "OUTPUT"),),
        ))
        pipeline.dispatch(AddConnectionCommand(
            source_block_id="load1",
            source_output_name="audio_out",
            target_block_id="onset1",
            target_input_name="audio_in",
        ))

        # Inject settings directly (as command handler would)
        block = graph.blocks["load1"]
        graph.replace_block(replace(
            block, settings=BlockSettings({"file_path": str(audio_file)})
        ))

        # Register processors
        engine.register_executor(
            "LoadAudio",
            LoadAudioProcessor(
                audio_info_fn=lambda p: AudioFileInfo(
                    sample_rate=44100, duration=5.0, channels=2
                )
            ),
        )
        engine.register_executor(
            "DetectOnsets",
            DetectOnsetsProcessor(
                onset_detect_fn=lambda fp, sr, th, mg: [0.5, 1.0, 1.5]
            ),
        )

        # Execute
        result = coordinator.request_run()
        assert isinstance(result, Ok)

        # Verify AudioData cached for LoadAudio
        load_cached = cache.get("load1", "audio_out")
        assert load_cached is not None
        assert isinstance(load_cached.value, AudioData)
        assert load_cached.value.sample_rate == 44100

        # Verify EventData cached for DetectOnsets
        onset_cached = cache.get("onset1", "event_out")
        assert onset_cached is not None
        assert isinstance(onset_cached.value, EventData)
        assert len(onset_cached.value.layers[0].events) == 3

    def test_events_have_correct_structure(self, tmp_path: Any) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        graph, _, pipeline, engine, cache, coordinator, _ = _make_full_stack()

        pipeline.dispatch(AddBlockCommand(
            block_id="load1", name="Load", block_type="LoadAudio",
            output_ports=(("audio_out", "AUDIO", "OUTPUT"),),
        ))
        pipeline.dispatch(AddBlockCommand(
            block_id="onset1", name="Onsets", block_type="DetectOnsets",
            input_ports=(("audio_in", "AUDIO", "INPUT"),),
            output_ports=(("event_out", "EVENT", "OUTPUT"),),
        ))
        pipeline.dispatch(AddConnectionCommand(
            source_block_id="load1", source_output_name="audio_out",
            target_block_id="onset1", target_input_name="audio_in",
        ))

        block = graph.blocks["load1"]
        graph.replace_block(replace(
            block, settings=BlockSettings({"file_path": str(audio_file)})
        ))

        engine.register_executor("LoadAudio", LoadAudioProcessor(
            audio_info_fn=lambda p: AudioFileInfo(44100, 5.0, 2)
        ))
        engine.register_executor("DetectOnsets", DetectOnsetsProcessor(
            onset_detect_fn=lambda fp, sr, th, mg: [0.5, 1.0]
        ))

        coordinator.request_run()

        event_data = cache.get("onset1", "event_out").value
        events = event_data.layers[0].events
        assert events[0].id == "onset1_onset_0"
        assert events[0].time == 0.5
        assert events[0].origin == "onset1"
        assert events[1].id == "onset1_onset_1"
        assert events[1].time == 1.0


# ---------------------------------------------------------------------------
# Staleness Propagation
# ---------------------------------------------------------------------------


class TestStalenessPropagation:
    """Verify that changing upstream settings propagates staleness and re-run refreshes."""

    def test_change_settings_marks_stale_and_rerun_refreshes(self, tmp_path: Any) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        graph, event_bus, pipeline, engine, cache, coordinator, _ = _make_full_stack()
        coordinator.subscribe_to_document_bus(event_bus)

        # Build and set up
        pipeline.dispatch(AddBlockCommand(
            block_id="load1", name="Load", block_type="LoadAudio",
            output_ports=(("audio_out", "AUDIO", "OUTPUT"),),
        ))
        pipeline.dispatch(AddBlockCommand(
            block_id="onset1", name="Onsets", block_type="DetectOnsets",
            input_ports=(("audio_in", "AUDIO", "INPUT"),),
            output_ports=(("event_out", "EVENT", "OUTPUT"),),
        ))
        pipeline.dispatch(AddConnectionCommand(
            source_block_id="load1", source_output_name="audio_out",
            target_block_id="onset1", target_input_name="audio_in",
        ))

        block = graph.blocks["load1"]
        graph.replace_block(replace(
            block, settings=BlockSettings({"file_path": str(audio_file)})
        ))

        engine.register_executor("LoadAudio", LoadAudioProcessor(
            audio_info_fn=lambda p: AudioFileInfo(44100, 5.0, 2)
        ))
        engine.register_executor("DetectOnsets", DetectOnsetsProcessor(
            onset_detect_fn=lambda fp, sr, th, mg: [0.5, 1.0, 1.5]
        ))

        # First run — both become FRESH
        coordinator.request_run()
        assert graph.blocks["load1"].state == BlockState.FRESH
        assert graph.blocks["onset1"].state == BlockState.FRESH

        # Change load settings → both marked STALE via DocumentBus
        pipeline.dispatch(ChangeBlockSettingsCommand(
            block_id="load1", setting_key="volume", new_value=0.8,
        ))
        assert graph.blocks["load1"].state == BlockState.STALE
        assert graph.blocks["onset1"].state == BlockState.STALE

        # Re-run → both FRESH again
        coordinator.request_run()
        assert graph.blocks["load1"].state == BlockState.FRESH
        assert graph.blocks["onset1"].state == BlockState.FRESH


# ---------------------------------------------------------------------------
# Targeted Execution
# ---------------------------------------------------------------------------


class TestTargetedExecution:
    """Verify that request_run with target only executes upstream blocks."""

    def test_target_b_skips_c(self) -> None:
        graph, _, pipeline, engine, cache, coordinator, _ = _make_full_stack()

        graph.add_block(_make_block("a", block_type="TypeA", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", block_type="TypeB",
                                     input_ports=(_audio_in(),), output_ports=(_audio_out(),)))
        graph.add_block(_make_block("c", block_type="TypeC", input_ports=(_audio_in(),)))
        graph.add_connection(Connection(
            source_block_id="a", source_output_name="audio_out",
            target_block_id="b", target_input_name="audio_in",
        ))
        graph.add_connection(Connection(
            source_block_id="b", source_output_name="audio_out",
            target_block_id="c", target_input_name="audio_in",
        ))

        exec_a = StubExecutor(output="a_out")
        exec_b = StubExecutor(output="b_out")
        exec_c = StubExecutor(output="c_out")
        engine.register_executor("TypeA", exec_a)
        engine.register_executor("TypeB", exec_b)
        engine.register_executor("TypeC", exec_c)

        result = coordinator.request_run(target="b")

        assert isinstance(result, Ok)
        assert exec_a.called_with == ["a"]
        assert exec_b.called_with == ["b"]
        assert exec_c.called_with == []  # Not executed


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    """Verify pre-cancellation returns Err(OperationCancelledError)."""

    def test_pre_cancelled_execution(self) -> None:
        graph, _, pipeline, engine, cache, coordinator, _ = _make_full_stack()

        graph.add_block(_make_block("a", output_ports=(_audio_out(),)))
        graph.add_block(_make_block("b", input_ports=(_audio_in(),)))
        graph.add_connection(Connection(
            source_block_id="a", source_output_name="audio_out",
            target_block_id="b", target_input_name="audio_in",
        ))

        # Use an executor that sets cancel before the second block runs
        call_count = 0

        class CancellingExecutor:
            def execute(self, block_id: str, context: ExecutionContext) -> Any:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # After executing first block, cancel for second
                    context.cancel_event.set()
                return ok("data")

        engine.register_executor("TestType", CancellingExecutor())

        result = coordinator.request_run()

        assert isinstance(result, Err)
        assert isinstance(result.error, OperationCancelledError)


# ---------------------------------------------------------------------------
# Override Reconciliation — REMOVED
# ---------------------------------------------------------------------------
# OverrideStore has been replaced by the Take System (echozero.takes).
# Override reconciliation tests were removed as part of that migration.
# See: sass/seminars/echozero-take-system/synthesis.md


# ---------------------------------------------------------------------------
# Auto-Evaluation
# ---------------------------------------------------------------------------


class TestAutoEvaluation:
    """Verify auto_evaluate triggers propagate_stale and request_run on DocumentBus events."""

    def test_auto_evaluate_triggers_run_on_settings_change(self, tmp_path: Any) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        graph, event_bus, pipeline, engine, cache, coordinator, _ = _make_full_stack()
        coordinator.subscribe_to_document_bus(event_bus)
        coordinator.auto_evaluate = True

        # Build graph
        pipeline.dispatch(AddBlockCommand(
            block_id="load1", name="Load", block_type="LoadAudio",
            output_ports=(("audio_out", "AUDIO", "OUTPUT"),),
        ))

        block = graph.blocks["load1"]
        graph.replace_block(replace(
            block, settings=BlockSettings({"file_path": str(audio_file)})
        ))

        engine.register_executor("LoadAudio", LoadAudioProcessor(
            audio_info_fn=lambda p: AudioFileInfo(44100, 5.0, 2)
        ))

        # First run to get FRESH
        coordinator.request_run()
        assert graph.blocks["load1"].state == BlockState.FRESH

        # Dispatch a settings change through Pipeline — DocumentBus should trigger auto-run
        pipeline.dispatch(ChangeBlockSettingsCommand(
            block_id="load1", setting_key="volume", new_value=0.5,
        ))

        # auto_evaluate should have triggered propagate_stale AND request_run
        # After request_run, block should be FRESH again
        assert graph.blocks["load1"].state == BlockState.FRESH
        assert cache.get("load1", "audio_out") is not None


# ---------------------------------------------------------------------------
# Multi-Port Output
# ---------------------------------------------------------------------------


class TestMultiPortOutput:
    """Verify multi-port executors store per-port outputs and feed downstream correctly."""

    def test_multi_port_executor_caches_both_ports(self) -> None:
        graph, _, pipeline, engine, cache, coordinator, _ = _make_full_stack()

        graph.add_block(_make_block(
            "split", block_type="SplitProcessor",
            output_ports=(
                _audio_out("audio_out"),
                _event_out("events_out"),
            ),
        ))

        class SplitProcessor:
            def execute(self, block_id: str, context: ExecutionContext) -> Any:
                return ok({
                    "audio_out": AudioData(sample_rate=44100, duration=5.0,
                                           file_path="/test.wav", channel_count=2),
                    "events_out": EventData(layers=(
                        Layer(id="l1", name="Events", events=(
                            Event(id="e1", time=0.5, duration=0.0,
                                  classifications={}, metadata={}, origin="split"),
                        )),
                    )),
                })

        engine.register_executor("SplitProcessor", SplitProcessor())
        coordinator.request_run()

        # Verify both ports cached separately
        audio_cached = cache.get("split", "audio_out")
        assert audio_cached is not None
        assert isinstance(audio_cached.value, AudioData)

        events_cached = cache.get("split", "events_out")
        assert events_cached is not None
        assert isinstance(events_cached.value, EventData)

    def test_downstream_reads_specific_port(self) -> None:
        graph, _, pipeline, engine, cache, coordinator, _ = _make_full_stack()

        graph.add_block(_make_block(
            "split", block_type="SplitProcessor",
            output_ports=(
                _audio_out("audio_out"),
                _event_out("events_out"),
            ),
        ))
        graph.add_block(_make_block(
            "consumer", block_type="Consumer",
            input_ports=(_audio_in(),),
        ))
        graph.add_connection(Connection(
            source_block_id="split", source_output_name="audio_out",
            target_block_id="consumer", target_input_name="audio_in",
        ))

        class SplitProcessor:
            def execute(self, block_id: str, context: ExecutionContext) -> Any:
                return ok({
                    "audio_out": AudioData(sample_rate=48000, duration=3.0,
                                           file_path="/split.wav", channel_count=1),
                    "events_out": EventData(layers=()),
                })

        class Consumer:
            def __init__(self) -> None:
                self.received: Any = None

            def execute(self, block_id: str, context: ExecutionContext) -> Any:
                self.received = context.get_input(block_id, "audio_in", AudioData)
                return ok(self.received)

        consumer = Consumer()
        engine.register_executor("SplitProcessor", SplitProcessor())
        engine.register_executor("Consumer", consumer)

        coordinator.request_run()

        assert consumer.received is not None
        assert isinstance(consumer.received, AudioData)
        assert consumer.received.sample_rate == 48000






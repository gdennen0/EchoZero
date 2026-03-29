"""
PyTorchAudioClassifyProcessor tests: Verify event classification and error handling.
Exists because classification is the final step in the analysis pipeline.
Tests assert on correct EventData output with classification metadata.
"""

from __future__ import annotations

from typing import Any

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
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
from echozero.execution import ExecutionContext
from echozero.processors.pytorch_audio_classify import PyTorchAudioClassifyProcessor
from echozero.progress import ProgressReport, RuntimeBus
from echozero.result import Err, Ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event_in(name: str = "events_in") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.INPUT)


def _event_out(name: str = "events_out") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.OUTPUT)


def _audio_in(name: str = "audio_in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


def _audio_out(name: str = "audio_out") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _make_graph_with_classify(
    classify_settings: dict[str, Any] | None = None,
    include_audio: bool = False,
) -> Graph:
    """Create a DetectOnsets -> PyTorchAudioClassify graph (optionally with audio input)."""
    graph = Graph()
    onset_block = Block(
        id="onset1",
        name="Detect Onsets",
        block_type="DetectOnsets",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(_event_out(),),
    )
    graph.add_block(onset_block)

    input_ports = [_event_in()]
    if include_audio:
        input_ports.append(_audio_in())

    # Default model_path if not explicitly provided
    default_settings = {"model_path": "/test/model.pth"}
    if classify_settings is not None:
        default_settings.update(classify_settings)
    
    classify_block = Block(
        id="classify1",
        name="PyTorch Audio Classify",
        block_type="PyTorchAudioClassify",
        category=BlockCategory.PROCESSOR,
        input_ports=tuple(input_ports),
        output_ports=(_event_out(),),
        settings=BlockSettings(default_settings),
    )
    graph.add_block(classify_block)
    graph.add_connection(
        Connection(
            source_block_id="onset1",
            source_output_name="events_out",
            target_block_id="classify1",
            target_input_name="events_in",
        )
    )
    return graph


def _make_context(graph: Graph) -> ExecutionContext:
    """Create an ExecutionContext for testing."""
    return ExecutionContext(
        execution_id="test-run",
        graph=graph,
        progress_bus=RuntimeBus(),
    )


MOCK_EVENT_1 = Event(
    id="event_1",
    time=0.5,
    duration=0.1,
    classifications={},
    metadata={},
    origin="onset1",
)
MOCK_EVENT_2 = Event(
    id="event_2",
    time=2.0,
    duration=0.1,
    classifications={},
    metadata={},
    origin="onset1",
)
MOCK_EVENT_3 = Event(
    id="event_3",
    time=4.5,
    duration=0.1,
    classifications={},
    metadata={},
    origin="onset1",
)

MOCK_EVENT_DATA = EventData(
    layers=(
        Layer(
            id="onsets",
            name="Detected Onsets",
            events=(MOCK_EVENT_1, MOCK_EVENT_2, MOCK_EVENT_3),
        ),
    ),
)

MOCK_AUDIO = AudioData(
    sample_rate=44100,
    duration=5.0,
    file_path="/test/audio.wav",
    channel_count=2,
)


def _mock_classify_fn(
    events: list[Event],
    audio_file: str | None,
    model_path: str,
    device: str,
    batch_size: int,
) -> list[Event]:
    """Classify events with dummy logic: simple time-based rules."""
    classified = []
    for event in events:
        if event.time < 1.0:
            cls = "kick"
        elif event.time < 3.0:
            cls = "snare"
        else:
            cls = "hihat"

        classified.append(
            Event(
                id=event.id,
                time=event.time,
                duration=event.duration,
                classifications={"class": cls, "confidence": 0.95},
                metadata={**event.metadata, "classified": True},
                origin=event.origin,
            )
        )
    return classified


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestPyTorchAudioClassifySuccess:
    """Verify correct EventData is returned with classifications."""

    def test_returns_event_data_with_classifications(self) -> None:
        graph = _make_graph_with_classify()
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Ok)
        event_data = result.value
        assert isinstance(event_data, EventData)
        assert len(event_data.layers) == 1
        assert len(event_data.layers[0].events) == 3

    def test_events_are_classified_correctly(self) -> None:
        graph = _make_graph_with_classify()
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Ok)
        events = result.value.layers[0].events

        # Check first event (time < 1.0 → kick)
        e0 = events[0]
        assert e0.id == "event_1"
        assert e0.classifications["class"] == "kick"
        assert e0.metadata["classified"] is True

        # Check second event (1.0 <= time < 3.0 → snare)
        e1 = events[1]
        assert e1.id == "event_2"
        assert e1.classifications["class"] == "snare"

        # Check third event (time >= 3.0 → hihat)
        e2 = events[2]
        assert e2.id == "event_3"
        assert e2.classifications["class"] == "hihat"

    def test_preserves_event_metadata(self) -> None:
        event_with_meta = Event(
            id="e1",
            time=0.5,
            duration=0.1,
            classifications={},
            metadata={"foo": "bar", "index": 42},
            origin="onset1",
        )
        event_data = EventData(
            layers=(Layer(id="layer1", name="Layer 1", events=(event_with_meta,)),)
        )

        graph = _make_graph_with_classify()
        context = _make_context(graph)
        context.set_output("onset1", "events_out", event_data)

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Ok)
        classified_event = result.value.layers[0].events[0]
        assert classified_event.metadata["foo"] == "bar"
        assert classified_event.metadata["index"] == 42
        assert classified_event.metadata["classified"] is True

    def test_preserves_layer_structure(self) -> None:
        # Multiple layers
        layer1 = Layer(
            id="onsets",
            name="Detected Onsets",
            events=(MOCK_EVENT_1,),
        )
        layer2 = Layer(
            id="peaks",
            name="Detected Peaks",
            events=(MOCK_EVENT_2, MOCK_EVENT_3),
        )
        event_data = EventData(layers=(layer1, layer2))

        graph = _make_graph_with_classify()
        context = _make_context(graph)
        context.set_output("onset1", "events_out", event_data)

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Ok)
        result_layers = result.value.layers
        assert len(result_layers) == 2
        assert result_layers[0].id == "onsets"
        assert result_layers[0].name == "Detected Onsets"
        assert len(result_layers[0].events) == 1
        assert result_layers[1].id == "peaks"
        assert result_layers[1].name == "Detected Peaks"
        assert len(result_layers[1].events) == 2

    def test_empty_events_returns_empty_event_data(self) -> None:
        empty_event_data = EventData(
            layers=(Layer(id="empty", name="Empty", events=()),)
        )

        graph = _make_graph_with_classify()
        context = _make_context(graph)
        context.set_output("onset1", "events_out", empty_event_data)

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Ok)
        assert len(result.value.layers[0].events) == 0

    def test_accepts_optional_audio_input(self) -> None:
        graph = _make_graph_with_classify(include_audio=True)
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)
        # Note: we don't set audio_out here to test that it's optional
        # In a real pipeline, it would be connected if available

        received_args = []

        def spy_fn(
            events: list[Event],
            audio_file: str | None,
            model_path: str,
            device: str,
            batch_size: int,
        ) -> list[Event]:
            received_args.append(audio_file)
            return _mock_classify_fn(events, audio_file, model_path, device, batch_size)

        processor = PyTorchAudioClassifyProcessor(classify_fn=spy_fn)
        result = processor.execute("classify1", context)

        # Should succeed even without audio input
        assert isinstance(result, Ok)
        assert len(received_args) == 1
        # audio_file should be None when not connected
        assert received_args[0] is None

    def test_settings_passed_to_classify_fn(self) -> None:
        graph = _make_graph_with_classify(
            classify_settings={
                "model_path": "/models/custom.pth",
                "device": "cuda",
                "batch_size": 64,
            }
        )
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        received_args = []

        def spy_fn(
            events: list[Event],
            audio_file: str | None,
            model_path: str,
            device: str,
            batch_size: int,
        ) -> list[Event]:
            received_args.append((model_path, device, batch_size))
            return _mock_classify_fn(events, audio_file, model_path, device, batch_size)

        processor = PyTorchAudioClassifyProcessor(classify_fn=spy_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Ok)
        assert len(received_args) == 1
        assert received_args[0] == ("/models/custom.pth", "cuda", 64)

    def test_default_device_is_cpu(self) -> None:
        graph = _make_graph_with_classify(
            classify_settings={"model_path": "/test/model.pth"}
        )
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        received_args = []

        def spy_fn(
            events: list[Event],
            audio_file: str | None,
            model_path: str,
            device: str,
            batch_size: int,
        ) -> list[Event]:
            received_args.append(device)
            return _mock_classify_fn(events, audio_file, model_path, device, batch_size)

        processor = PyTorchAudioClassifyProcessor(classify_fn=spy_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Ok)
        assert len(received_args) == 1
        assert received_args[0] == "cpu"

    def test_default_batch_size_is_32(self) -> None:
        graph = _make_graph_with_classify(
            classify_settings={"model_path": "/test/model.pth"}
        )
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        received_args = []

        def spy_fn(
            events: list[Event],
            audio_file: str | None,
            model_path: str,
            device: str,
            batch_size: int,
        ) -> list[Event]:
            received_args.append(batch_size)
            return _mock_classify_fn(events, audio_file, model_path, device, batch_size)

        processor = PyTorchAudioClassifyProcessor(classify_fn=spy_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Ok)
        assert len(received_args) == 1
        assert received_args[0] == 32


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestPyTorchAudioClassifyErrors:
    """Verify Err results for missing inputs and invalid settings."""

    def test_missing_event_input_returns_err(self) -> None:
        graph = _make_graph_with_classify()
        context = _make_context(graph)
        # No upstream events set

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Err)
        assert "event input" in str(result.error).lower()

    def test_missing_model_path_returns_err(self) -> None:
        # Create graph without model_path - manually create block
        graph = Graph()
        onset_block = Block(
            id="onset1",
            name="Detect Onsets",
            block_type="DetectOnsets",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(_event_out(),),
        )
        graph.add_block(onset_block)
        
        classify_block = Block(
            id="classify1",
            name="PyTorch Audio Classify",
            block_type="PyTorchAudioClassify",
            category=BlockCategory.PROCESSOR,
            input_ports=(_event_in(),),
            output_ports=(_event_out(),),
            settings=BlockSettings({}),  # No model_path
        )
        graph.add_block(classify_block)
        graph.add_connection(
            Connection(
                source_block_id="onset1",
                source_output_name="events_out",
                target_block_id="classify1",
                target_input_name="events_in",
            )
        )
        
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Err)
        assert "model_path" in str(result.error).lower()

    def test_invalid_batch_size_returns_err(self) -> None:
        graph = _make_graph_with_classify(
            classify_settings={
                "model_path": "/test/model.pth",
                "batch_size": 0,
            }
        )
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Err)
        assert "batch_size" in str(result.error).lower()

    def test_negative_batch_size_returns_err(self) -> None:
        graph = _make_graph_with_classify(
            classify_settings={
                "model_path": "/test/model.pth",
                "batch_size": -1,
            }
        )
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Err)

    def test_non_integer_batch_size_returns_err(self) -> None:
        graph = _make_graph_with_classify(
            classify_settings={
                "model_path": "/test/model.pth",
                "batch_size": "big",
            }
        )
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Err)

    def test_classify_fn_exception_returns_err(self) -> None:
        graph = _make_graph_with_classify()
        context = _make_context(graph)
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        def exploding_fn(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("torch crashed")

        processor = PyTorchAudioClassifyProcessor(classify_fn=exploding_fn)
        result = processor.execute("classify1", context)

        assert isinstance(result, Err)
        assert "torch crashed" in str(result.error)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


class TestPyTorchAudioClassifyProgress:
    """Verify progress reports are published."""

    def test_progress_reports_published(self) -> None:
        graph = _make_graph_with_classify()
        runtime_bus = RuntimeBus()
        context = ExecutionContext(
            execution_id="test-run",
            graph=graph,
            progress_bus=runtime_bus,
        )
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        reports: list[ProgressReport] = []
        runtime_bus.subscribe(
            lambda r: reports.append(r) if isinstance(r, ProgressReport) else None
        )

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        processor.execute("classify1", context)

        assert len(reports) >= 2
        assert reports[0].percent == 0.0
        assert reports[0].block_id == "classify1"
        assert reports[-1].percent == 1.0

    def test_reports_event_count(self) -> None:
        graph = _make_graph_with_classify()
        runtime_bus = RuntimeBus()
        context = ExecutionContext(
            execution_id="test-run",
            graph=graph,
            progress_bus=runtime_bus,
        )
        context.set_output("onset1", "events_out", MOCK_EVENT_DATA)

        reports: list[ProgressReport] = []
        runtime_bus.subscribe(
            lambda r: reports.append(r) if isinstance(r, ProgressReport) else None
        )

        processor = PyTorchAudioClassifyProcessor(classify_fn=_mock_classify_fn)
        processor.execute("classify1", context)

        # Check that at least one report mentions 3 events
        event_count_reports = [r for r in reports if "3 events" in r.message]
        assert len(event_count_reports) > 0


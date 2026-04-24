"""
DetectOnsetsProcessor tests: Verify onset detection, event creation, and error handling.
Exists because onset detection is the first block that consumes upstream data — correctness is critical.
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

from typing import Any

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Event, EventData, Layer, Port
from echozero.execution import ExecutionContext
from echozero.processors.detect_onsets import DetectOnsetsProcessor
from echozero.progress import ProgressReport, RuntimeBus, RuntimeReport
from echozero.result import Err, Ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_out(name: str = "audio_out") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _audio_in(name: str = "audio_in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


def _event_out(name: str = "event_out") -> Port:
    return Port(name=name, port_type=PortType.EVENT, direction=Direction.OUTPUT)


def _make_graph_with_chain(
    onset_settings: dict[str, Any] | None = None,
) -> Graph:
    """Create a LoadAudio -> DetectOnsets graph."""
    graph = Graph()
    load_block = Block(
        id="load1",
        name="Load Audio",
        block_type="LoadAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(_audio_out(),),
        settings=BlockSettings({"file_path": "/test/audio.wav"}),
    )
    onset_block = Block(
        id="onset1",
        name="Detect Onsets",
        block_type="DetectOnsets",
        category=BlockCategory.PROCESSOR,
        input_ports=(_audio_in(),),
        output_ports=(_event_out(),),
        settings=BlockSettings(onset_settings or {}),
    )
    graph.add_block(load_block)
    graph.add_block(onset_block)
    graph.add_connection(
        Connection(
            source_block_id="load1",
            source_output_name="audio_out",
            target_block_id="onset1",
            target_input_name="audio_in",
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


MOCK_AUDIO = AudioData(
    sample_rate=44100,
    duration=5.0,
    file_path="/test/audio.wav",
    channel_count=2,
)


def _mock_onset_fn(
    file_path: str, sample_rate: int, threshold: float, min_gap: float
) -> list[float]:
    """Return fixed onset times for testing."""
    return [0.5, 1.0, 1.5]


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestDetectOnsetsSuccess:
    """Verify correct EventData is returned with mocked onset detection."""

    def test_returns_event_data_with_three_events(self) -> None:
        graph = _make_graph_with_chain()
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = DetectOnsetsProcessor(onset_detect_fn=_mock_onset_fn)
        result = processor.execute("onset1", context)

        assert isinstance(result, Ok)
        event_data = result.value
        assert isinstance(event_data, EventData)
        assert len(event_data.layers) == 1
        assert len(event_data.layers[0].events) == 3

    def test_correct_event_objects_created(self) -> None:
        graph = _make_graph_with_chain()
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = DetectOnsetsProcessor(onset_detect_fn=_mock_onset_fn)
        result = processor.execute("onset1", context)

        assert isinstance(result, Ok)
        events = result.value.layers[0].events

        # Check first event
        e0 = events[0]
        assert e0.id == "onset1_onset_0"
        assert e0.time == 0.5
        assert e0.duration == 0.0
        assert e0.classifications == {}
        assert e0.metadata["threshold"] == 0.5
        assert e0.metadata["min_gap"] == 0.05
        assert e0.metadata["method"] == "default"
        assert e0.metadata["backtrack"] is True
        assert e0.metadata["timing_offset_ms"] == 0.0
        assert e0.metadata["index"] == 0
        assert e0.origin == "onset1"

        # Check second event
        e1 = events[1]
        assert e1.id == "onset1_onset_1"
        assert e1.time == 1.0
        assert e1.metadata["index"] == 1

        # Check third event
        e2 = events[2]
        assert e2.id == "onset1_onset_2"
        assert e2.time == 1.5
        assert e2.metadata["index"] == 2

    def test_layer_has_correct_id_and_name(self) -> None:
        graph = _make_graph_with_chain()
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = DetectOnsetsProcessor(onset_detect_fn=_mock_onset_fn)
        result = processor.execute("onset1", context)

        assert isinstance(result, Ok)
        layer = result.value.layers[0]
        assert layer.id == "onset1_onsets"
        assert layer.name == "Detected Onsets"

    def test_empty_onset_list_produces_empty_layer(self) -> None:
        graph = _make_graph_with_chain()
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = DetectOnsetsProcessor(
            onset_detect_fn=lambda fp, sr, th, mg: []
        )
        result = processor.execute("onset1", context)

        assert isinstance(result, Ok)
        assert len(result.value.layers[0].events) == 0

    def test_settings_passed_to_onset_detect_fn(self) -> None:
        graph = _make_graph_with_chain(
            onset_settings={
                "threshold": 0.8,
                "min_gap": 0.1,
                "method": "hfc",
                "backtrack": False,
                "timing_offset_ms": -15.0,
            }
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        received_args: list[tuple[str, int, float, float, str, bool, float]] = []

        def spy_fn(
            fp: str,
            sr: int,
            th: float,
            mg: float,
            *,
            method: str,
            backtrack: bool,
            timing_offset_ms: float,
        ) -> list[float]:
            received_args.append((fp, sr, th, mg, method, backtrack, timing_offset_ms))
            return [1.0]

        processor = DetectOnsetsProcessor(onset_detect_fn=spy_fn)
        processor.execute("onset1", context)

        assert len(received_args) == 1
        assert received_args[0] == (
            "/test/audio.wav",
            44100,
            0.8,
            0.1,
            "hfc",
            False,
            -15.0,
        )

    def test_default_settings_used_when_not_specified(self) -> None:
        graph = _make_graph_with_chain(onset_settings={})
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        received_args: list[tuple[float, float]] = []

        def spy_fn(fp: str, sr: int, th: float, mg: float) -> list[float]:
            received_args.append((th, mg))
            return []

        processor = DetectOnsetsProcessor(onset_detect_fn=spy_fn)
        processor.execute("onset1", context)

        assert received_args[0] == (0.5, 0.05)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestDetectOnsetsErrors:
    """Verify Err results for missing input, failed detection, and missing block."""

    def test_missing_audio_input_returns_err(self) -> None:
        graph = _make_graph_with_chain()
        context = _make_context(graph)
        # No upstream audio set

        processor = DetectOnsetsProcessor(onset_detect_fn=_mock_onset_fn)
        result = processor.execute("onset1", context)

        assert isinstance(result, Err)
        assert "audio input" in str(result.error).lower()

    def test_onset_detect_fn_raising_returns_err(self) -> None:
        graph = _make_graph_with_chain()
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        def exploding_fn(fp: str, sr: int, th: float, mg: float) -> list[float]:
            raise RuntimeError("librosa crashed")

        processor = DetectOnsetsProcessor(onset_detect_fn=exploding_fn)
        result = processor.execute("onset1", context)

        assert isinstance(result, Err)
        assert "librosa crashed" in str(result.error)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


class TestDetectOnsetsProgress:
    """Verify progress reports are published at 0%, 50%, and 100%."""

    def test_progress_reports_published(self) -> None:
        graph = _make_graph_with_chain()
        runtime_bus = RuntimeBus()
        context = ExecutionContext(
            execution_id="test-run",
            graph=graph,
            progress_bus=runtime_bus,
        )
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        reports: list[ProgressReport] = []
        runtime_bus.subscribe(
            lambda r: reports.append(r) if isinstance(r, ProgressReport) else None
        )

        processor = DetectOnsetsProcessor(onset_detect_fn=_mock_onset_fn)
        processor.execute("onset1", context)

        assert len(reports) == 3
        assert reports[0].percent == 0.0
        assert reports[0].block_id == "onset1"
        assert reports[1].percent == 0.5
        assert reports[2].percent == 1.0

    def test_metadata_includes_settings_in_events(self) -> None:
        graph = _make_graph_with_chain(
            onset_settings={"threshold": 0.3, "min_gap": 0.02}
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = DetectOnsetsProcessor(onset_detect_fn=_mock_onset_fn)
        result = processor.execute("onset1", context)

        assert isinstance(result, Ok)
        event = result.value.layers[0].events[0]
        assert event.metadata["threshold"] == 0.3
        assert event.metadata["min_gap"] == 0.02
        assert event.metadata["method"] == "default"
        assert event.metadata["backtrack"] is True

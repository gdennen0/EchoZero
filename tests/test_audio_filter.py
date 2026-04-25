"""
AudioFilterProcessor tests: Verify filtering, parameter handling, and error cases.
Exists because filtering is the foundation for frequency-selective analysis.
Tests assert on correct AudioData output and proper error handling.
"""

from __future__ import annotations

from typing import Any

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Port
from echozero.execution import ExecutionContext
from echozero.processors.audio_filter import AudioFilterProcessor
from echozero.progress import ProgressReport, RuntimeBus
from echozero.result import Err, Ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_in(name: str = "audio_in") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.INPUT)


def _audio_out(name: str = "audio_out") -> Port:
    return Port(name=name, port_type=PortType.AUDIO, direction=Direction.OUTPUT)


def _make_graph_with_filter(
    filter_settings: dict[str, Any] | None = None,
) -> Graph:
    """Create a LoadAudio -> AudioFilter graph."""
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
    filter_block = Block(
        id="filter1",
        name="Audio Filter",
        block_type="AudioFilter",
        category=BlockCategory.PROCESSOR,
        input_ports=(_audio_in(),),
        output_ports=(_audio_out(),),
        settings=BlockSettings(filter_settings or {}),
    )
    graph.add_block(load_block)
    graph.add_block(filter_block)
    graph.add_connection(
        Connection(
            source_block_id="load1",
            source_output_name="audio_out",
            target_block_id="filter1",
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


def _mock_filter_fn(
    file_path: str,
    sample_rate: int,
    filter_type: str,
    freq: float,
    gain_db: float,
    Q: float,
) -> tuple[str, int, float]:
    """Return a dummy output file for testing."""
    return "/tmp/filtered_audio.wav", sample_rate, 5.0


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestAudioFilterSuccess:
    """Verify correct AudioData is returned with mocked filtering."""

    def test_returns_audio_data_with_correct_metadata(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={"filter_type": "lowpass", "freq": 5000.0}
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)
        audio_data = result.value
        assert isinstance(audio_data, AudioData)
        assert audio_data.sample_rate == 44100
        assert audio_data.duration == 5.0
        assert audio_data.channel_count == 2

    def test_lowpass_filter_creates_output(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={"filter_type": "lowpass", "freq": 3000.0}
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)
        assert result.value.file_path == "/tmp/filtered_audio.wav"

    def test_highpass_filter_creates_output(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={"filter_type": "highpass", "freq": 100.0}
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)

    def test_bandpass_filter_with_Q(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={
                "filter_type": "bandpass",
                "freq": 1000.0,
                "Q": 2.0,
            }
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)

    def test_bandstop_filter_with_Q(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={
                "filter_type": "bandstop",
                "freq": 60.0,
                "Q": 1.0,
            }
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)

    def test_lowshelf_filter_with_gain(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={
                "filter_type": "lowshelf",
                "freq": 500.0,
                "gain_db": 6.0,
            }
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)

    def test_highshelf_filter_with_negative_gain(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={
                "filter_type": "highshelf",
                "freq": 8000.0,
                "gain_db": -3.0,
            }
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)

    def test_peak_filter_with_Q_and_gain(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={
                "filter_type": "peak",
                "freq": 2000.0,
                "Q": 3.0,
                "gain_db": 12.0,
            }
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)

    def test_default_gain_db_zero(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={"filter_type": "lowpass", "freq": 1000.0}
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        received_args = []

        def spy_fn(
            file_path: str,
            sample_rate: int,
            filter_type: str,
            freq: float,
            gain_db: float,
            Q: float,
        ) -> tuple[str, int, float]:
            received_args.append((filter_type, freq, gain_db, Q))
            return "/tmp/out.wav", sample_rate, 5.0

        processor = AudioFilterProcessor(filter_fn=spy_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)
        assert len(received_args) == 1
        assert received_args[0][2] == 0.0  # gain_db should be 0.0

    def test_default_Q_is_one(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={
                "filter_type": "bandpass",
                "freq": 1000.0,
            }
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        received_args = []

        def spy_fn(
            file_path: str,
            sample_rate: int,
            filter_type: str,
            freq: float,
            gain_db: float,
            Q: float,
        ) -> tuple[str, int, float]:
            received_args.append((filter_type, freq, gain_db, Q))
            return "/tmp/out.wav", sample_rate, 5.0

        processor = AudioFilterProcessor(filter_fn=spy_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)
        assert len(received_args) == 1
        assert received_args[0][3] == 1.0  # Q should be 1.0

    def test_disabled_filter_bypasses_processing_and_returns_input_audio(self) -> None:
        graph = _make_graph_with_filter(filter_settings={"enabled": False})
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        calls: list[tuple[Any, ...]] = []

        def spy_fn(
            file_path: str,
            sample_rate: int,
            filter_type: str,
            freq: float,
            gain_db: float,
            Q: float,
        ) -> tuple[str, int, float]:
            calls.append((file_path, sample_rate, filter_type, freq, gain_db, Q))
            return "/tmp/out.wav", sample_rate, 5.0

        processor = AudioFilterProcessor(filter_fn=spy_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)
        assert calls == []
        assert result.value == MOCK_AUDIO

    def test_disabled_filter_accepts_false_like_string(self) -> None:
        graph = _make_graph_with_filter(filter_settings={"enabled": "false"})
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)
        assert result.value == MOCK_AUDIO


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestAudioFilterErrors:
    """Verify Err results for invalid settings and missing inputs."""

    def test_missing_audio_input_returns_err(self) -> None:
        graph = _make_graph_with_filter()
        context = _make_context(graph)
        # No upstream audio set

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Err)
        assert "audio input" in str(result.error).lower()

    def test_missing_filter_type_returns_err(self) -> None:
        graph = _make_graph_with_filter(filter_settings={"freq": 1000.0})
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Err)
        assert "filter_type" in str(result.error).lower()

    def test_missing_freq_returns_err(self) -> None:
        graph = _make_graph_with_filter(filter_settings={"filter_type": "lowpass"})
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Err)
        assert "freq" in str(result.error).lower()

    def test_invalid_filter_type_returns_err(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={"filter_type": "invalid_filter", "freq": 1000.0}
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Err)
        assert "unknown" in str(result.error).lower()

    def test_negative_freq_returns_err(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={"filter_type": "lowpass", "freq": -1000.0}
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Err)
        assert "positive" in str(result.error).lower()

    def test_zero_freq_returns_err(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={"filter_type": "lowpass", "freq": 0}
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Err)

    def test_non_numeric_freq_returns_err(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={"filter_type": "lowpass", "freq": "bad"}
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Err)

    def test_non_numeric_gain_returns_err(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={
                "filter_type": "lowpass",
                "freq": 1000.0,
                "gain_db": "high",
            }
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Err)
        assert "gain_db" in str(result.error).lower()

    def test_zero_Q_returns_err(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={
                "filter_type": "bandpass",
                "freq": 1000.0,
                "Q": 0,
            }
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Err)
        assert "positive" in str(result.error).lower()

    def test_filter_fn_exception_returns_err(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={"filter_type": "lowpass", "freq": 1000.0}
        )
        context = _make_context(graph)
        context.set_output("load1", "audio_out", MOCK_AUDIO)

        def exploding_fn(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("scipy crashed")

        processor = AudioFilterProcessor(filter_fn=exploding_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Err)
        assert "scipy crashed" in str(result.error)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


class TestAudioFilterProgress:
    """Verify progress reports are published."""

    def test_progress_reports_published(self) -> None:
        graph = _make_graph_with_filter(
            filter_settings={"filter_type": "lowpass", "freq": 1000.0}
        )
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

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        processor.execute("filter1", context)

        assert len(reports) >= 2
        assert reports[0].percent == 0.0
        assert reports[0].block_id == "filter1"
        assert reports[-1].percent == 1.0

    def test_bypass_progress_reports_complete(self) -> None:
        graph = _make_graph_with_filter(filter_settings={"enabled": False})
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

        processor = AudioFilterProcessor(filter_fn=_mock_filter_fn)
        result = processor.execute("filter1", context)

        assert isinstance(result, Ok)
        assert len(reports) >= 2
        assert reports[0].percent == 0.0
        assert reports[-1].percent == 1.0
        assert reports[-1].message == "Filtering bypassed"

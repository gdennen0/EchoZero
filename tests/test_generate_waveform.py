"""
GenerateWaveformProcessor tests: Verify multi-LOD peak computation with mocked audio.
Exists because waveform rendering depends on correct peak data at every zoom level.
Tests assert on output shape, values, and edge cases per STYLE.md testing rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Port, WaveformData, WaveformPeaks
from echozero.execution import ExecutionContext
from echozero.processors.generate_waveform import (
    DEFAULT_WINDOW_SIZES,
    GenerateWaveformProcessor,
    compute_peaks,
)
from echozero.progress import ProgressReport, RuntimeBus
from echozero.result import Err, Ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_load_block(block_id: str = "load1") -> Block:
    """Create a LoadAudio block that produces audio_out."""
    return Block(
        id=block_id,
        name="Load Audio",
        block_type="LoadAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(
            Port(name="audio_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        ),
    )


def _make_waveform_block(block_id: str = "waveform1") -> Block:
    """Create a GenerateWaveform block with audio_in and waveform_out."""
    return Block(
        id=block_id,
        name="Generate Waveform",
        block_type="GenerateWaveform",
        category=BlockCategory.PROCESSOR,
        input_ports=(
            Port(name="audio_in", port_type=PortType.AUDIO, direction=Direction.INPUT),
        ),
        output_ports=(
            Port(name="waveform_out", port_type=PortType.WAVEFORM, direction=Direction.OUTPUT),
        ),
    )


def _make_connected_graph(
    load_id: str = "load1",
    waveform_id: str = "waveform1",
) -> Graph:
    """Create a graph with LoadAudio -> GenerateWaveform connected."""
    from echozero.domain.types import Connection

    graph = Graph()
    graph.add_block(_make_load_block(load_id))
    graph.add_block(_make_waveform_block(waveform_id))
    graph.add_connection(
        Connection(
            source_block_id=load_id,
            source_output_name="audio_out",
            target_block_id=waveform_id,
            target_input_name="audio_in",
        )
    )
    return graph


def _make_context(
    graph: Graph,
    upstream_audio: AudioData | None = None,
    load_id: str = "load1",
) -> ExecutionContext:
    """Create an ExecutionContext, optionally with upstream audio pre-set."""
    ctx = ExecutionContext(
        execution_id="test-run",
        graph=graph,
        progress_bus=RuntimeBus(),
    )
    if upstream_audio is not None:
        ctx.set_output(load_id, "audio_out", upstream_audio)
    return ctx


def _fake_samples(n: int, value: float = 0.5) -> np.ndarray:
    """Create a constant-value sample array of length n."""
    return np.full(n, value, dtype=np.float32)


def _sine_samples(n: int, freq: float = 440.0, sr: int = 44100) -> np.ndarray:
    """Create a sine wave sample array."""
    t = np.arange(n, dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Unit tests: compute_peaks
# ---------------------------------------------------------------------------


class TestComputePeaks:
    """Verify the core peak computation function."""

    def test_exact_multiple_of_window(self) -> None:
        samples = np.array([1, -1, 2, -2, 3, -3, 4, -4], dtype=np.float32)
        peaks = compute_peaks(samples, window_size=4)
        assert peaks.shape == (2, 2)
        np.testing.assert_array_equal(peaks[0], [-2.0, 2.0])
        np.testing.assert_array_equal(peaks[1], [-4.0, 4.0])

    def test_with_remainder(self) -> None:
        samples = np.array([1, -1, 2, -2, 5], dtype=np.float32)
        peaks = compute_peaks(samples, window_size=4)
        assert peaks.shape == (2, 2)
        np.testing.assert_array_equal(peaks[0], [-2.0, 2.0])
        np.testing.assert_array_equal(peaks[1], [5.0, 5.0])

    def test_empty_samples(self) -> None:
        samples = np.array([], dtype=np.float32)
        peaks = compute_peaks(samples, window_size=64)
        assert peaks.shape == (0, 2)

    def test_fewer_samples_than_window(self) -> None:
        samples = np.array([0.5, -0.3], dtype=np.float32)
        peaks = compute_peaks(samples, window_size=64)
        assert peaks.shape == (1, 2)
        np.testing.assert_almost_equal(peaks[0, 0], -0.3)
        np.testing.assert_almost_equal(peaks[0, 1], 0.5)

    def test_single_sample(self) -> None:
        samples = np.array([0.7], dtype=np.float32)
        peaks = compute_peaks(samples, window_size=64)
        assert peaks.shape == (1, 2)
        np.testing.assert_almost_equal(peaks[0, 0], 0.7)
        np.testing.assert_almost_equal(peaks[0, 1], 0.7)

    def test_constant_signal(self) -> None:
        samples = np.full(256, 0.5, dtype=np.float32)
        peaks = compute_peaks(samples, window_size=64)
        assert peaks.shape == (4, 2)
        for row in peaks:
            np.testing.assert_almost_equal(row[0], 0.5)
            np.testing.assert_almost_equal(row[1], 0.5)

    def test_output_dtype_is_float32(self) -> None:
        samples = np.arange(128, dtype=np.float64)
        peaks = compute_peaks(samples, window_size=64)
        assert peaks.dtype == np.float32


# ---------------------------------------------------------------------------
# Processor tests: success path
# ---------------------------------------------------------------------------


class TestGenerateWaveformSuccess:
    """Verify correct WaveformData returned with mocked sample loading."""

    def test_returns_waveform_data_with_all_lods(self) -> None:
        n_samples = 8192
        mock_samples = _sine_samples(n_samples)

        graph = _make_connected_graph()
        audio = AudioData(sample_rate=44100, duration=0.186, file_path="/fake/audio.wav")
        ctx = _make_context(graph, upstream_audio=audio)

        processor = GenerateWaveformProcessor(
            load_samples_fn=lambda path, sr: mock_samples,
        )
        result = processor.execute("waveform1", ctx)

        assert isinstance(result, Ok)
        waveform = result.value
        assert isinstance(waveform, WaveformData)
        assert len(waveform.lods) == len(DEFAULT_WINDOW_SIZES)
        assert waveform.sample_rate == 44100
        assert waveform.duration == 0.186
        assert waveform.channel_count == 1

    def test_lod_window_sizes_match_defaults(self) -> None:
        mock_samples = _fake_samples(4096)

        graph = _make_connected_graph()
        audio = AudioData(sample_rate=44100, duration=0.1, file_path="/fake.wav")
        ctx = _make_context(graph, upstream_audio=audio)

        processor = GenerateWaveformProcessor(
            load_samples_fn=lambda path, sr: mock_samples,
        )
        result = processor.execute("waveform1", ctx)

        assert isinstance(result, Ok)
        for lod, expected_ws in zip(result.value.lods, DEFAULT_WINDOW_SIZES):
            assert lod.window_size == expected_ws

    def test_lod_peak_counts_decrease_with_coarser_lods(self) -> None:
        n_samples = 16384
        mock_samples = _fake_samples(n_samples)

        graph = _make_connected_graph()
        audio = AudioData(sample_rate=44100, duration=0.37, file_path="/fake.wav")
        ctx = _make_context(graph, upstream_audio=audio)

        processor = GenerateWaveformProcessor(
            load_samples_fn=lambda path, sr: mock_samples,
        )
        result = processor.execute("waveform1", ctx)

        assert isinstance(result, Ok)
        peak_counts = [lod.peaks.shape[0] for lod in result.value.lods]
        # Each coarser LOD should have fewer or equal peaks
        for i in range(len(peak_counts) - 1):
            assert peak_counts[i] >= peak_counts[i + 1]

    def test_custom_window_sizes(self) -> None:
        mock_samples = _fake_samples(1024)

        graph = _make_connected_graph()
        audio = AudioData(sample_rate=22050, duration=0.046, file_path="/fake.wav")
        ctx = _make_context(graph, upstream_audio=audio)

        custom_windows = (32, 128)
        processor = GenerateWaveformProcessor(
            load_samples_fn=lambda path, sr: mock_samples,
            window_sizes=custom_windows,
        )
        result = processor.execute("waveform1", ctx)

        assert isinstance(result, Ok)
        assert len(result.value.lods) == 2
        assert result.value.lods[0].window_size == 32
        assert result.value.lods[1].window_size == 128

    def test_passes_correct_path_and_sr_to_load_fn(self) -> None:
        calls: list[tuple[str, int]] = []

        def spy_load(path: str, sr: int) -> np.ndarray:
            calls.append((path, sr))
            return _fake_samples(1024)

        graph = _make_connected_graph()
        audio = AudioData(sample_rate=48000, duration=0.02, file_path="/my/song.wav")
        ctx = _make_context(graph, upstream_audio=audio)

        processor = GenerateWaveformProcessor(load_samples_fn=spy_load)
        processor.execute("waveform1", ctx)

        assert calls == [("/my/song.wav", 48000)]

    def test_preserves_channel_count_from_audio(self) -> None:
        graph = _make_connected_graph()
        audio = AudioData(
            sample_rate=44100, duration=1.0, file_path="/fake.wav", channel_count=2
        )
        ctx = _make_context(graph, upstream_audio=audio)

        processor = GenerateWaveformProcessor(
            load_samples_fn=lambda path, sr: _fake_samples(44100),
        )
        result = processor.execute("waveform1", ctx)

        assert isinstance(result, Ok)
        assert result.value.channel_count == 2

    def test_progress_reports_emitted(self) -> None:
        reports: list[ProgressReport] = []

        graph = _make_connected_graph()
        audio = AudioData(sample_rate=44100, duration=0.1, file_path="/fake.wav")
        ctx = _make_context(graph, upstream_audio=audio)
        ctx.progress_bus.subscribe(
            lambda r: reports.append(r) if isinstance(r, ProgressReport) else None
        )

        processor = GenerateWaveformProcessor(
            load_samples_fn=lambda path, sr: _fake_samples(4096),
        )
        processor.execute("waveform1", ctx)

        # Should have: 1 loading + 1 computing + 4 LOD completions = 6 reports
        assert len(reports) >= 5
        assert reports[0].percent == 0.0
        assert reports[-1].percent == 1.0


# ---------------------------------------------------------------------------
# Processor tests: error paths
# ---------------------------------------------------------------------------


class TestGenerateWaveformErrors:
    """Verify Err results for missing input and load failures."""

    def test_no_audio_input_returns_err(self) -> None:
        graph = Graph()
        graph.add_block(_make_waveform_block())
        ctx = ExecutionContext(
            execution_id="test", graph=graph, progress_bus=RuntimeBus()
        )

        processor = GenerateWaveformProcessor(
            load_samples_fn=lambda path, sr: _fake_samples(100),
        )
        result = processor.execute("waveform1", ctx)

        assert isinstance(result, Err)
        assert "no audio input" in str(result.error)

    def test_load_samples_failure_returns_err(self) -> None:
        graph = _make_connected_graph()
        audio = AudioData(sample_rate=44100, duration=1.0, file_path="/bad/file.wav")
        ctx = _make_context(graph, upstream_audio=audio)

        def exploding_load(path: str, sr: int) -> np.ndarray:
            raise RuntimeError("Corrupted audio file")

        processor = GenerateWaveformProcessor(load_samples_fn=exploding_load)
        result = processor.execute("waveform1", ctx)

        assert isinstance(result, Err)
        assert "Corrupted audio file" in str(result.error)

    def test_empty_audio_produces_empty_lods(self) -> None:
        graph = _make_connected_graph()
        audio = AudioData(sample_rate=44100, duration=0.0, file_path="/empty.wav")
        ctx = _make_context(graph, upstream_audio=audio)

        processor = GenerateWaveformProcessor(
            load_samples_fn=lambda path, sr: np.array([], dtype=np.float32),
        )
        result = processor.execute("waveform1", ctx)

        assert isinstance(result, Ok)
        for lod in result.value.lods:
            assert lod.peaks.shape == (0, 2)

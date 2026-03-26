"""
LoadAudioProcessor tests: Verify audio file loading with mocked and error paths.
Exists because the LoadAudio block is the pipeline entry point — broken loading breaks everything.
Tests assert on output values per STYLE.md testing rules; no smoke-only checks.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Port
from echozero.execution import ExecutionContext
from echozero.processors.load_audio import AudioFileInfo, LoadAudioProcessor
from echozero.progress import RuntimeBus
from echozero.result import Err, Ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_with_load_block(
    block_id: str = "load1",
    settings: dict[str, Any] | None = None,
) -> Graph:
    """Create a graph containing a single LoadAudio block."""
    graph = Graph()
    block = Block(
        id=block_id,
        name="Load Audio",
        block_type="LoadAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(Port(name="audio_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),),
        settings=BlockSettings(entries=settings or {}),
    )
    graph.add_block(block)
    return graph


def _make_context(graph: Graph, execution_id: str = "test-run") -> ExecutionContext:
    """Create an ExecutionContext for testing."""
    return ExecutionContext(
        execution_id=execution_id,
        graph=graph,
        progress_bus=RuntimeBus(),
    )


def _mock_audio_info(
    sample_rate: int = 44100,
    duration: float = 2.5,
    channels: int = 2,
) -> AudioFileInfo:
    """Create a mock AudioFileInfo return value."""
    return AudioFileInfo(sample_rate=sample_rate, duration=duration, channels=channels)


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestLoadAudioSuccess:
    """Verify correct AudioData is returned with mocked audio info."""

    def test_returns_audio_data_with_correct_fields(self, tmp_path: Any) -> None:
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)  # Fake file

        graph = _make_graph_with_load_block(settings={"file_path": str(audio_file)})
        context = _make_context(graph)

        processor = LoadAudioProcessor(
            audio_info_fn=lambda path: _mock_audio_info(
                sample_rate=48000, duration=3.0, channels=1
            )
        )
        result = processor.execute("load1", context)

        assert isinstance(result, Ok)
        audio_data = result.value
        assert isinstance(audio_data, AudioData)
        assert audio_data.sample_rate == 48000
        assert audio_data.duration == 3.0
        assert audio_data.file_path == str(audio_file)
        assert audio_data.channel_count == 1

    def test_stereo_file_reports_two_channels(self, tmp_path: Any) -> None:
        audio_file = tmp_path / "stereo.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        graph = _make_graph_with_load_block(settings={"file_path": str(audio_file)})
        context = _make_context(graph)

        processor = LoadAudioProcessor(
            audio_info_fn=lambda path: _mock_audio_info(channels=2)
        )
        result = processor.execute("load1", context)

        assert isinstance(result, Ok)
        assert result.value.channel_count == 2

    def test_passes_correct_path_to_info_fn(self, tmp_path: Any) -> None:
        audio_file = tmp_path / "check_path.wav"
        audio_file.write_bytes(b"data")

        received_paths: list[str] = []

        def spy_info(path: str) -> AudioFileInfo:
            received_paths.append(path)
            return _mock_audio_info()

        graph = _make_graph_with_load_block(settings={"file_path": str(audio_file)})
        context = _make_context(graph)

        processor = LoadAudioProcessor(audio_info_fn=spy_info)
        processor.execute("load1", context)

        assert received_paths == [str(audio_file)]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestLoadAudioErrors:
    """Verify Err results for missing settings, missing files, and info failures."""

    def test_missing_file_path_setting_returns_err(self) -> None:
        graph = _make_graph_with_load_block(settings={})
        context = _make_context(graph)

        processor = LoadAudioProcessor(audio_info_fn=lambda p: _mock_audio_info())
        result = processor.execute("load1", context)

        assert isinstance(result, Err)
        assert "file_path" in str(result.error)

    def test_file_not_found_returns_err(self) -> None:
        graph = _make_graph_with_load_block(
            settings={"file_path": "/nonexistent/audio.wav"}
        )
        context = _make_context(graph)

        processor = LoadAudioProcessor(audio_info_fn=lambda p: _mock_audio_info())
        result = processor.execute("load1", context)

        assert isinstance(result, Err)
        assert "not found" in str(result.error)

    def test_audio_info_fn_raising_returns_err(self, tmp_path: Any) -> None:
        audio_file = tmp_path / "bad.wav"
        audio_file.write_bytes(b"corrupted")

        graph = _make_graph_with_load_block(settings={"file_path": str(audio_file)})
        context = _make_context(graph)

        def exploding_info(path: str) -> AudioFileInfo:
            raise RuntimeError("Cannot read format")

        processor = LoadAudioProcessor(audio_info_fn=exploding_info)
        result = processor.execute("load1", context)

        assert isinstance(result, Err)
        assert "Cannot read format" in str(result.error)

    def test_block_not_in_graph_returns_err(self) -> None:
        graph = Graph()
        context = _make_context(graph)

        processor = LoadAudioProcessor(audio_info_fn=lambda p: _mock_audio_info())
        result = processor.execute("ghost", context)

        assert isinstance(result, Err)
        assert "Block not found" in str(result.error)

    def test_err_type_is_specific_for_missing_setting(self) -> None:
        from echozero.errors import ValidationError

        graph = _make_graph_with_load_block(settings={})
        context = _make_context(graph)

        processor = LoadAudioProcessor(audio_info_fn=lambda p: _mock_audio_info())
        result = processor.execute("load1", context)

        assert isinstance(result, Err)
        assert isinstance(result.error, ValidationError)

    def test_err_type_is_execution_error_for_missing_file(self) -> None:
        from echozero.errors import ExecutionError

        graph = _make_graph_with_load_block(
            settings={"file_path": "/no/such/file.wav"}
        )
        context = _make_context(graph)

        processor = LoadAudioProcessor(audio_info_fn=lambda p: _mock_audio_info())
        result = processor.execute("load1", context)

        assert isinstance(result, Err)
        assert isinstance(result.error, ExecutionError)

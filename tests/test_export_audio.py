"""Tests for ExportAudioProcessor."""

from __future__ import annotations

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Port
from echozero.execution import ExecutionEngine, GraphPlanner
from echozero.processors.export_audio import ExportAudioProcessor, SUPPORTED_FORMATS
from echozero.progress import RuntimeBus
from echozero.result import Ok, is_err, is_ok, unwrap


class MockLoadAudio:
    def execute(self, block_id, context):
        return Ok(AudioData(
            sample_rate=44100, duration=5.0,
            file_path="song.wav", channel_count=2,
        ))


def _build_graph(output_dir="/tmp/export", fmt="wav", filename=None) -> Graph:
    settings = {"output_dir": output_dir, "format": fmt}
    if filename:
        settings["filename"] = filename

    g = Graph()
    g.add_block(Block(
        id="load", name="Load", block_type="LoadAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(), output_ports=(
            Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
        ),
        settings=BlockSettings({"file_path": "song.wav"}),
    ))
    g.add_block(Block(
        id="export", name="Export", block_type="ExportAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
        output_ports=(),
        settings=BlockSettings(settings),
    ))
    g.add_connection(Connection("load", "audio_out", "export", "audio_in"))
    return g


def _run(graph, capture=None):
    exports = capture if capture is not None else []

    def fake_export(source, output, fmt):
        exports.append({"source": source, "output": output, "format": fmt})
        return output

    bus = RuntimeBus()
    engine = ExecutionEngine(graph, bus)
    engine.register_executor("LoadAudio", MockLoadAudio())
    engine.register_executor("ExportAudio", ExportAudioProcessor(fake_export))
    plan = GraphPlanner().plan(graph)
    return engine.run(plan), exports


class TestExportAudioProcessor:
    def test_exports_successfully(self):
        result, _ = _run(_build_graph())
        assert is_ok(result)

    def test_writes_to_output_dir(self):
        exports = []
        _run(_build_graph(output_dir="/tmp/out"), capture=exports)
        assert len(exports) == 1
        assert "/tmp/out" in exports[0]["output"]

    def test_default_filename_from_source(self):
        exports = []
        _run(_build_graph(), capture=exports)
        assert "song.wav" in exports[0]["output"]

    def test_custom_filename(self):
        exports = []
        _run(_build_graph(filename="custom"), capture=exports)
        assert "custom.wav" in exports[0]["output"]

    def test_format_passed_through(self):
        exports = []
        _run(_build_graph(fmt="flac"), capture=exports)
        assert exports[0]["format"] == "flac"

    def test_unsupported_format_returns_error(self):
        result, _ = _run(_build_graph(fmt="xyz"))
        assert is_err(result)

    def test_missing_output_dir_returns_error(self):
        g = _build_graph()
        g.replace_block(Block(
            id="export", name="E", block_type="ExportAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(),
            settings=BlockSettings({"format": "wav"}),
        ))
        result, _ = _run(g)
        assert is_err(result)

    def test_no_audio_input_returns_error(self):
        g = Graph()
        g.add_block(Block(
            id="export", name="E", block_type="ExportAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(),
            settings=BlockSettings({"output_dir": "/tmp", "format": "wav"}),
        ))
        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)
        engine.register_executor("ExportAudio", ExportAudioProcessor(lambda s, o, f: o))
        result = engine.run(GraphPlanner().plan(g))
        assert is_err(result)

    def test_supported_formats_constant(self):
        assert "wav" in SUPPORTED_FORMATS
        assert "mp3" in SUPPORTED_FORMATS
        assert "flac" in SUPPORTED_FORMATS


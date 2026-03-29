"""Tests for EQBandsProcessor."""

from __future__ import annotations

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Port
from echozero.execution import ExecutionEngine, GraphPlanner
from echozero.processors.eq_bands import (
    DEFAULT_BANDS,
    EQBandsProcessor,
    validate_bands,
)
from echozero.progress import RuntimeBus
from echozero.result import Ok, is_err, is_ok, unwrap


class MockLoadAudio:
    def execute(self, block_id, context):
        return Ok(AudioData(
            sample_rate=44100, duration=5.0,
            file_path="test.wav", channel_count=2,
        ))


def _fake_eq(file_path, sample_rate, bands, filter_order):
    return ("/tmp/eq_out.wav", sample_rate, 5.0)


def _build_graph(bands=None, filter_order=4) -> Graph:
    settings = {"filter_order": filter_order}
    if bands is not None:
        settings["bands"] = bands

    g = Graph()
    g.add_block(Block(
        id="load", name="Load", block_type="LoadAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(), output_ports=(
            Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
        ),
        settings=BlockSettings({"file_path": "test.wav"}),
    ))
    g.add_block(Block(
        id="eq", name="EQ", block_type="EQBands",
        category=BlockCategory.PROCESSOR,
        input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
        output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
        settings=BlockSettings(settings),
    ))
    g.add_connection(Connection("load", "audio_out", "eq", "audio_in"))
    return g


def _run(graph, eq_fn=_fake_eq):
    bus = RuntimeBus()
    engine = ExecutionEngine(graph, bus)
    engine.register_executor("LoadAudio", MockLoadAudio())
    engine.register_executor("EQBands", EQBandsProcessor(eq_fn))
    plan = GraphPlanner().plan(graph)
    return engine.run(plan)


# ---------------------------------------------------------------------------
# validate_bands
# ---------------------------------------------------------------------------

class TestValidateBands:
    def test_valid_bands(self):
        bands = [{"freq_low": 100, "freq_high": 1000, "gain_db": 6.0}]
        assert validate_bands(bands, 44100) == []

    def test_freq_low_below_20(self):
        bands = [{"freq_low": 5, "freq_high": 1000, "gain_db": 0}]
        errors = validate_bands(bands, 44100)
        assert len(errors) == 1
        assert "freq_low" in errors[0]

    def test_freq_high_above_nyquist(self):
        bands = [{"freq_low": 100, "freq_high": 25000, "gain_db": 0}]
        errors = validate_bands(bands, 44100)
        assert len(errors) == 1
        assert "freq_high" in errors[0]

    def test_low_equals_high(self):
        bands = [{"freq_low": 500, "freq_high": 500, "gain_db": 0}]
        errors = validate_bands(bands, 44100)
        assert len(errors) == 1

    def test_gain_too_high(self):
        bands = [{"freq_low": 100, "freq_high": 1000, "gain_db": 60}]
        errors = validate_bands(bands, 44100)
        assert len(errors) == 1

    def test_default_bands_valid(self):
        assert validate_bands(DEFAULT_BANDS, 44100) == []


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestEQBandsProcessor:
    def test_produces_audio_data(self):
        result = _run(_build_graph())
        assert is_ok(result)
        outputs = unwrap(result)
        audio = outputs["eq"]["audio_out"]
        assert isinstance(audio, AudioData)

    def test_preserves_sample_rate(self):
        result = _run(_build_graph())
        audio = unwrap(result)["eq"]["audio_out"]
        assert audio.sample_rate == 44100

    def test_custom_bands(self):
        bands = [{"freq_low": 200, "freq_high": 5000, "gain_db": 3.0}]
        result = _run(_build_graph(bands=bands))
        assert is_ok(result)

    def test_invalid_filter_order_returns_error(self):
        result = _run(_build_graph(filter_order=0))
        assert is_err(result)

    def test_invalid_filter_order_too_high(self):
        result = _run(_build_graph(filter_order=9))
        assert is_err(result)

    def test_invalid_bands_type_returns_error(self):
        g = _build_graph()
        g.replace_block(Block(
            id="eq", name="EQ", block_type="EQBands",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
            settings=BlockSettings({"bands": "not_a_list"}),
        ))
        result = _run(g)
        assert is_err(result)

    def test_no_audio_input_returns_error(self):
        g = Graph()
        g.add_block(Block(
            id="eq", name="EQ", block_type="EQBands",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
            settings=BlockSettings({}),
        ))
        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)
        engine.register_executor("EQBands", EQBandsProcessor(_fake_eq))
        result = engine.run(GraphPlanner().plan(g))
        assert is_err(result)

    def test_eq_fn_failure_returns_error(self):
        def failing_eq(*args, **kwargs):
            raise RuntimeError("scipy missing")
        result = _run(_build_graph(), eq_fn=failing_eq)
        assert is_err(result)




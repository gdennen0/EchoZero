"""Tests for AudioNegateProcessor."""

from __future__ import annotations

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import (
    AudioData, Block, BlockSettings, Connection, Event, EventData, Layer, Port,
)
from echozero.execution import ExecutionEngine, GraphPlanner
from echozero.processors.audio_negate import AudioNegateProcessor, VALID_MODES
from echozero.progress import RuntimeBus
from echozero.result import Ok, is_err, is_ok, unwrap


class MockLoadAudio:
    def execute(self, block_id, context):
        return Ok(AudioData(
            sample_rate=44100, duration=5.0,
            file_path="test.wav", channel_count=2,
        ))


def _fake_negate(audio_file, sample_rate, regions, mode, fade_ms, attenuation_db):
    return ("/tmp/negated.wav", sample_rate, 5.0)


def _make_event_data() -> EventData:
    return EventData(layers=(
        Layer(id="onsets", name="Onsets", events=(
            Event(id="e1", time=0.5, duration=0.2, classifications={}, metadata={}, origin="test"),
            Event(id="e2", time=1.5, duration=0.3, classifications={}, metadata={}, origin="test"),
        )),
    ))


def _build_graph(mode="silence", fade_ms=10.0, attenuation_db=-20.0) -> Graph:
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
        id="events_src", name="Events", block_type="EventSource",
        category=BlockCategory.PROCESSOR,
        input_ports=(), output_ports=(
            Port("events_out", PortType.EVENT, Direction.OUTPUT),
        ),
    ))
    g.add_block(Block(
        id="negate", name="Negate", block_type="AudioNegate",
        category=BlockCategory.PROCESSOR,
        input_ports=(
            Port("audio_in", PortType.AUDIO, Direction.INPUT),
            Port("events_in", PortType.EVENT, Direction.INPUT),
        ),
        output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
        settings=BlockSettings({
            "mode": mode,
            "fade_ms": fade_ms,
            "attenuation_db": attenuation_db,
        }),
    ))
    g.add_connection(Connection("load", "audio_out", "negate", "audio_in"))
    g.add_connection(Connection("events_src", "events_out", "negate", "events_in"))
    return g


class FakeEventSource:
    def execute(self, block_id, context):
        from echozero.result import ok
        return ok(_make_event_data())


def _run(graph, negate_fn=_fake_negate):
    bus = RuntimeBus()
    engine = ExecutionEngine(graph, bus)
    engine.register_executor("LoadAudio", MockLoadAudio())
    engine.register_executor("EventSource", FakeEventSource())
    engine.register_executor("AudioNegate", AudioNegateProcessor(negate_fn))
    plan = GraphPlanner().plan(graph)
    return engine.run(plan)


class TestAudioNegateProcessor:
    def test_silence_mode(self):
        result = _run(_build_graph(mode="silence"))
        assert is_ok(result)
        audio = unwrap(result)["negate"]["audio_out"]
        assert isinstance(audio, AudioData)

    def test_attenuate_mode(self):
        result = _run(_build_graph(mode="attenuate"))
        assert is_ok(result)

    def test_invalid_mode_returns_error(self):
        result = _run(_build_graph(mode="invalid"))
        assert is_err(result)

    def test_subtract_mode_not_supported_v1(self):
        result = _run(_build_graph(mode="subtract"))
        assert is_err(result)

    def test_preserves_sample_rate(self):
        result = _run(_build_graph())
        audio = unwrap(result)["negate"]["audio_out"]
        assert audio.sample_rate == 44100

    def test_no_audio_returns_error(self):
        g = Graph()
        g.add_block(Block(
            id="events_src", name="E", block_type="EventSource",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("events_out", PortType.EVENT, Direction.OUTPUT),
            ),
        ))
        g.add_block(Block(
            id="negate", name="N", block_type="AudioNegate",
            category=BlockCategory.PROCESSOR,
            input_ports=(
                Port("audio_in", PortType.AUDIO, Direction.INPUT),
                Port("events_in", PortType.EVENT, Direction.INPUT),
            ),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
            settings=BlockSettings({"mode": "silence"}),
        ))
        g.add_connection(Connection("events_src", "events_out", "negate", "events_in"))
        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)
        engine.register_executor("EventSource", FakeEventSource())
        engine.register_executor("AudioNegate", AudioNegateProcessor(_fake_negate))
        result = engine.run(GraphPlanner().plan(g))
        assert is_err(result)

    def test_no_events_returns_error(self):
        g = Graph()
        g.add_block(Block(
            id="load", name="L", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("audio_out", PortType.AUDIO, Direction.OUTPUT),
            ),
            settings=BlockSettings({"file_path": "test.wav"}),
        ))
        g.add_block(Block(
            id="negate", name="N", block_type="AudioNegate",
            category=BlockCategory.PROCESSOR,
            input_ports=(
                Port("audio_in", PortType.AUDIO, Direction.INPUT),
                Port("events_in", PortType.EVENT, Direction.INPUT),
            ),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
            settings=BlockSettings({"mode": "silence"}),
        ))
        g.add_connection(Connection("load", "audio_out", "negate", "audio_in"))
        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)
        engine.register_executor("LoadAudio", MockLoadAudio())
        engine.register_executor("AudioNegate", AudioNegateProcessor(_fake_negate))
        result = engine.run(GraphPlanner().plan(g))
        assert is_err(result)

    def test_zero_duration_events_returns_audio_unchanged(self):
        """Events with zero duration should result in unchanged audio."""
        class ZeroDurationSource:
            def execute(self, block_id, context):
                from echozero.result import ok
                return ok(EventData(layers=(
                    Layer(id="l", name="L", events=(
                        Event(id="e1", time=0.5, duration=0.0, classifications={},
                              metadata={}, origin="test"),
                    )),
                )))

        g = _build_graph()
        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)
        engine.register_executor("LoadAudio", MockLoadAudio())
        engine.register_executor("EventSource", ZeroDurationSource())
        engine.register_executor("AudioNegate", AudioNegateProcessor(_fake_negate))
        result = engine.run(GraphPlanner().plan(g))
        assert is_ok(result)
        # Returns original audio since no regions have duration > 0
        audio = unwrap(result)["negate"]["audio_out"]
        assert audio.file_path == "test.wav"  # unchanged

    def test_invalid_fade_ms_returns_error(self):
        result = _run(_build_graph(fade_ms=200))
        assert is_err(result)

    def test_positive_attenuation_returns_error(self):
        result = _run(_build_graph(attenuation_db=6.0))
        assert is_err(result)

    def test_valid_modes_constant(self):
        assert "silence" in VALID_MODES
        assert "attenuate" in VALID_MODES
        assert "subtract" in VALID_MODES

    def test_negate_fn_failure_returns_error(self):
        def failing(*args, **kwargs):
            raise RuntimeError("boom")
        result = _run(_build_graph(), negate_fn=failing)
        assert is_err(result)



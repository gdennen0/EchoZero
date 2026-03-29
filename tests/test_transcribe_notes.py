"""Tests for TranscribeNotesProcessor."""

from __future__ import annotations

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import AudioData, Block, BlockSettings, Connection, Port
from echozero.execution import ExecutionContext, ExecutionEngine, GraphPlanner
from echozero.processors.transcribe_notes import (
    NoteInfo,
    TranscribeNotesProcessor,
    midi_to_frequency,
    midi_to_note_name,
)
from echozero.progress import RuntimeBus
from echozero.result import Ok, is_err, is_ok, unwrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockLoadAudio:
    def execute(self, block_id, context):
        return Ok(AudioData(
            sample_rate=44100, duration=10.0,
            file_path="test.wav", channel_count=2,
        ))


def _fake_transcribe(
    file_path, sample_rate, onset_threshold, frame_threshold,
    min_note_length, min_frequency, max_frequency,
) -> list[NoteInfo]:
    return [
        NoteInfo(start_time=0.5, end_time=1.0, midi_note=60, velocity=100),  # C4
        NoteInfo(start_time=1.5, end_time=2.0, midi_note=64, velocity=80),   # E4
        NoteInfo(start_time=2.5, end_time=3.5, midi_note=60, velocity=90),   # C4 again
        NoteInfo(start_time=3.0, end_time=3.8, midi_note=67, velocity=110),  # G4
    ]


def _empty_transcribe(*args, **kwargs) -> list[NoteInfo]:
    return []


def _failing_transcribe(*args, **kwargs) -> list[NoteInfo]:
    raise RuntimeError("Model crashed")


def _build_graph() -> Graph:
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
        id="transcribe", name="Transcribe", block_type="TranscribeNotes",
        category=BlockCategory.PROCESSOR,
        input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
        output_ports=(Port("events_out", PortType.EVENT, Direction.OUTPUT),),
        settings=BlockSettings({
            "onset_threshold": 0.5,
            "frame_threshold": 0.3,
            "min_note_length": 0.058,
            "min_frequency": 27.5,
            "max_frequency": 4186.0,
        }),
    ))
    g.add_connection(Connection("load", "audio_out", "transcribe", "audio_in"))
    return g


def _run(graph, transcribe_fn=_fake_transcribe):
    bus = RuntimeBus()
    engine = ExecutionEngine(graph, bus)
    engine.register_executor("LoadAudio", MockLoadAudio())
    engine.register_executor("TranscribeNotes", TranscribeNotesProcessor(transcribe_fn))
    planner = GraphPlanner()
    plan = planner.plan(graph)
    return engine.run(plan)


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestMidiHelpers:
    def test_midi_to_note_name_c4(self):
        assert midi_to_note_name(60) == "C4"

    def test_midi_to_note_name_a4(self):
        assert midi_to_note_name(69) == "A4"

    def test_midi_to_frequency_a4(self):
        assert abs(midi_to_frequency(69) - 440.0) < 0.01

    def test_midi_to_frequency_a3(self):
        assert abs(midi_to_frequency(57) - 220.0) < 0.01


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestTranscribeNotesProcessor:
    def test_produces_event_data(self):
        result = _run(_build_graph())
        assert is_ok(result)
        outputs = unwrap(result)
        event_data = outputs["transcribe"]["events_out"]
        assert len(event_data.layers) > 0

    def test_groups_notes_by_pitch(self):
        result = _run(_build_graph())
        outputs = unwrap(result)
        event_data = outputs["transcribe"]["events_out"]
        layer_names = [l.name for l in event_data.layers]
        assert "C4" in layer_names
        assert "E4" in layer_names
        assert "G4" in layer_names

    def test_c4_layer_has_two_events(self):
        result = _run(_build_graph())
        outputs = unwrap(result)
        event_data = outputs["transcribe"]["events_out"]
        c4_layer = next(l for l in event_data.layers if l.name == "C4")
        assert len(c4_layer.events) == 2

    def test_event_has_duration(self):
        result = _run(_build_graph())
        outputs = unwrap(result)
        event_data = outputs["transcribe"]["events_out"]
        c4_layer = next(l for l in event_data.layers if l.name == "C4")
        event = c4_layer.events[0]
        assert event.duration == pytest.approx(0.5)

    def test_event_has_midi_metadata(self):
        result = _run(_build_graph())
        outputs = unwrap(result)
        event_data = outputs["transcribe"]["events_out"]
        c4_layer = next(l for l in event_data.layers if l.name == "C4")
        event = c4_layer.events[0]
        assert event.metadata["midi_note"] == 60
        assert event.metadata["velocity"] == 100
        assert "frequency_hz" in event.metadata

    def test_event_has_note_classification(self):
        result = _run(_build_graph())
        outputs = unwrap(result)
        event_data = outputs["transcribe"]["events_out"]
        c4_layer = next(l for l in event_data.layers if l.name == "C4")
        assert c4_layer.events[0].classifications["note"] == "C4"

    def test_empty_transcription_returns_empty_layers(self):
        result = _run(_build_graph(), _empty_transcribe)
        assert is_ok(result)
        outputs = unwrap(result)
        event_data = outputs["transcribe"]["events_out"]
        assert len(event_data.layers) == 0

    def test_transcription_failure_returns_error(self):
        result = _run(_build_graph(), _failing_transcribe)
        assert is_err(result)

    def test_no_audio_input_returns_error(self):
        g = Graph()
        g.add_block(Block(
            id="transcribe", name="T", block_type="TranscribeNotes",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(Port("events_out", PortType.EVENT, Direction.OUTPUT),),
            settings=BlockSettings({}),
        ))
        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)
        engine.register_executor("TranscribeNotes", TranscribeNotesProcessor(_fake_transcribe))
        plan = GraphPlanner().plan(g)
        result = engine.run(plan)
        assert is_err(result)

    def test_invalid_onset_threshold_returns_error(self):
        g = _build_graph()
        # Replace transcribe block with invalid threshold
        g.replace_block(Block(
            id="transcribe", name="T", block_type="TranscribeNotes",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(Port("events_out", PortType.EVENT, Direction.OUTPUT),),
            settings=BlockSettings({"onset_threshold": 5.0}),
        ))
        result = _run(g)
        assert is_err(result)

    def test_total_event_count(self):
        result = _run(_build_graph())
        outputs = unwrap(result)
        event_data = outputs["transcribe"]["events_out"]
        total = sum(len(l.events) for l in event_data.layers)
        assert total == 4




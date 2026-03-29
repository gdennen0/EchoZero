"""Tests for ExportAudioDatasetProcessor."""

from __future__ import annotations

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import (
    AudioData, Block, BlockSettings, Connection, Event, EventData, Layer, Port,
)
from echozero.execution import ExecutionEngine, GraphPlanner
from echozero.processors.export_audio_dataset import ExportAudioDatasetProcessor
from echozero.progress import RuntimeBus
from echozero.result import Ok, is_err, is_ok, unwrap


class MockLoadAudio:
    def execute(self, block_id, context):
        return Ok(AudioData(
            sample_rate=44100, duration=5.0,
            file_path="song.wav", channel_count=2,
        ))


def _make_event_data() -> EventData:
    return EventData(layers=(
        Layer(id="drums", name="Drums", events=(
            Event(id="kick_1", time=0.5, duration=0.2,
                  classifications={"class": "kick"}, metadata={}, origin="test"),
            Event(id="snare_1", time=1.0, duration=0.3,
                  classifications={"class": "snare"}, metadata={}, origin="test"),
            Event(id="kick_2", time=1.5, duration=0.2,
                  classifications={"class": "kick"}, metadata={}, origin="test"),
        )),
    ))


def _make_no_duration_events() -> EventData:
    return EventData(layers=(
        Layer(id="l", name="L", events=(
            Event(id="e1", time=0.5, duration=0.0, classifications={}, metadata={}, origin="t"),
        )),
    ))


class FakeEventSource:
    def __init__(self, event_data=None):
        self._data = event_data or _make_event_data()

    def execute(self, block_id, context):
        from echozero.result import ok
        return ok(self._data)


def _build_graph(output_dir="/tmp/dataset", fmt="wav", organize=True, min_dur=0.01) -> Graph:
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
        id="events_src", name="Events", block_type="EventSource",
        category=BlockCategory.PROCESSOR,
        input_ports=(), output_ports=(
            Port("events_out", PortType.EVENT, Direction.OUTPUT),
        ),
    ))
    g.add_block(Block(
        id="export", name="Export", block_type="ExportAudioDataset",
        category=BlockCategory.PROCESSOR,
        input_ports=(
            Port("audio_in", PortType.AUDIO, Direction.INPUT),
            Port("events_in", PortType.EVENT, Direction.INPUT),
        ),
        output_ports=(),
        settings=BlockSettings({
            "output_dir": output_dir,
            "format": fmt,
            "organize_by_class": organize,
            "min_duration": min_dur,
        }),
    ))
    g.add_connection(Connection("load", "audio_out", "export", "audio_in"))
    g.add_connection(Connection("events_src", "events_out", "export", "events_in"))
    return g


def _run(graph, export_fn=None, event_source=None):
    captured = []

    def fake_export(source_audio, sample_rate, output_dir, fmt, clips):
        captured.append({"clips": clips, "output_dir": output_dir})
        return len(clips)

    fn = export_fn or fake_export
    bus = RuntimeBus()
    engine = ExecutionEngine(graph, bus)
    engine.register_executor("LoadAudio", MockLoadAudio())
    engine.register_executor("EventSource", event_source or FakeEventSource())
    engine.register_executor("ExportAudioDataset", ExportAudioDatasetProcessor(fn))
    plan = GraphPlanner().plan(graph)
    return engine.run(plan), captured


class TestExportAudioDatasetProcessor:
    def test_exports_successfully(self):
        result, _ = _run(_build_graph())
        assert is_ok(result)

    def test_returns_stats(self):
        result, _ = _run(_build_graph())
        stats = unwrap(result)["export"]["out"]
        assert stats["clips_exported"] == 3
        assert stats["output_dir"] == "/tmp/dataset"
        assert "kick" in stats["classes"]
        assert "snare" in stats["classes"]

    def test_organize_by_class(self):
        _, captured = _run(_build_graph(organize=True))
        clips = captured[0]["clips"]
        kick_clips = [c for c in clips if c["label"] == "kick"]
        assert all("kick" in c["output_dir"] for c in kick_clips)

    def test_flat_organization(self):
        _, captured = _run(_build_graph(organize=False))
        clips = captured[0]["clips"]
        assert all(c["output_dir"] == "/tmp/dataset" for c in clips)

    def test_min_duration_filter(self):
        result, captured = _run(_build_graph(min_dur=0.25))
        stats = unwrap(result)["export"]["out"]
        # Only snare_1 (0.3s) passes the 0.25s filter
        assert stats["clips_exported"] == 1

    def test_no_events_with_duration_returns_error(self):
        result, _ = _run(
            _build_graph(),
            event_source=FakeEventSource(_make_no_duration_events()),
        )
        assert is_err(result)

    def test_missing_output_dir_returns_error(self):
        g = _build_graph()
        g.replace_block(Block(
            id="export", name="E", block_type="ExportAudioDataset",
            category=BlockCategory.PROCESSOR,
            input_ports=(
                Port("audio_in", PortType.AUDIO, Direction.INPUT),
                Port("events_in", PortType.EVENT, Direction.INPUT),
            ),
            output_ports=(),
            settings=BlockSettings({}),
        ))
        result, _ = _run(g)
        assert is_err(result)

    def test_export_fn_failure_returns_error(self):
        def failing(*args, **kwargs):
            raise RuntimeError("disk full")
        result, _ = _run(_build_graph(), export_fn=failing)
        assert is_err(result)

    def test_clip_filenames_use_event_ids(self):
        _, captured = _run(_build_graph())
        clips = captured[0]["clips"]
        filenames = [c["filename"] for c in clips]
        assert "kick_1.wav" in filenames
        assert "snare_1.wav" in filenames



"""Tests for ExportMA2Processor."""

from __future__ import annotations

import pytest

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import (
    Block, BlockSettings, Connection, Event, EventData, Layer, Port,
)
from echozero.execution import ExecutionContext, ExecutionEngine, GraphPlanner
from echozero.processors.export_ma2 import (
    ExportMA2Processor,
    build_ma2_xml,
    seconds_to_timecode,
)
from echozero.progress import RuntimeBus
from echozero.result import is_err, is_ok, unwrap


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestSecondsToTimecode:
    def test_zero(self):
        assert seconds_to_timecode(0.0, 30) == "00:00:00.00"

    def test_one_second(self):
        assert seconds_to_timecode(1.0, 30) == "00:00:01.00"

    def test_fractional(self):
        # 0.5s at 30fps = 15 frames
        assert seconds_to_timecode(0.5, 30) == "00:00:00.15"

    def test_one_minute(self):
        assert seconds_to_timecode(60.0, 30) == "00:01:00.00"

    def test_one_hour(self):
        assert seconds_to_timecode(3600.0, 30) == "01:00:00.00"

    def test_25fps(self):
        assert seconds_to_timecode(1.0, 25) == "00:00:01.00"

    def test_25fps_half_second(self):
        # 0.5s at 25fps = 12.5 → rounds to 12 frames
        assert seconds_to_timecode(0.5, 25) == "00:00:00.12"

    def test_29_97_ndf(self):
        assert seconds_to_timecode(1.0, 29.97) == "00:00:01.00"

    def test_29_97_drop_frame_minute_skip(self):
        # 1800 frames at 29.97fps lands on DF boundary 00:01:00.02
        assert seconds_to_timecode(60.06, 29.97, drop_frame=True) == "00:01:00.02"


class TestBuildMA2XML:
    def test_empty_events(self):
        xml = build_ma2_xml([], frame_rate=30)
        assert "<MA2Timecode" in xml
        assert "</MA2Timecode>" in xml

    def test_single_event(self):
        events = [{"time": 1.0, "label": "Kick", "cue": "1"}]
        xml = build_ma2_xml(events, frame_rate=30)
        assert 'timecode="00:00:01.00"' in xml
        assert 'label="Kick"' in xml
        assert 'cue="1"' in xml

    def test_multiple_events_sorted(self):
        events = [
            {"time": 0.5, "label": "A"},
            {"time": 1.5, "label": "B"},
        ]
        xml = build_ma2_xml(events, frame_rate=30)
        assert xml.index("A") < xml.index("B")

    def test_track_name(self):
        xml = build_ma2_xml([], track_name="MySong")
        assert 'trackName="MySong"' in xml

    def test_frame_rate_in_header(self):
        xml = build_ma2_xml([], frame_rate=25)
        assert 'frameRate="25"' in xml

    def test_29_97_frame_rate_in_header(self):
        xml = build_ma2_xml([], frame_rate=29.97)
        assert 'frameRate="29.97"' in xml


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def _make_event_data() -> EventData:
    return EventData(layers=(
        Layer(id="drums", name="Drums", events=(
            Event(id="e1", time=0.5, duration=0.1,
                  classifications={"class": "kick"}, metadata={}, origin="test"),
            Event(id="e2", time=1.0, duration=0.1,
                  classifications={"class": "snare"}, metadata={}, origin="test"),
            Event(id="e3", time=1.5, duration=0.1,
                  classifications={"class": "kick"}, metadata={}, origin="test"),
        )),
    ))


class TestExportMA2Processor:
    def _build_and_run(self, output_path="/tmp/test.xml", frame_rate=30, capture=None):
        """Build graph, inject events, run processor."""
        written_files = capture if capture is not None else []

        def fake_export(xml_content, path):
            written_files.append({"path": path, "content": xml_content})
            return path

        g = Graph()
        # We need a "source" block that produces EventData
        g.add_block(Block(
            id="source", name="Source", block_type="EventSource",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("events_out", PortType.EVENT, Direction.OUTPUT),
            ),
        ))
        g.add_block(Block(
            id="export", name="Export", block_type="ExportMA2",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("events_in", PortType.EVENT, Direction.INPUT),),
            output_ports=(),
            settings=BlockSettings({
                "output_path": output_path,
                "frame_rate": frame_rate,
                "track_name": "TestTrack",
            }),
        ))
        g.add_connection(Connection("source", "events_out", "export", "events_in"))

        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)

        # Fake source executor that returns EventData
        class FakeSource:
            def execute(self, block_id, context):
                from echozero.result import ok
                return ok(_make_event_data())

        engine.register_executor("EventSource", FakeSource())
        engine.register_executor("ExportMA2", ExportMA2Processor(fake_export))

        plan = GraphPlanner().plan(g)
        return engine.run(plan), written_files

    def test_exports_successfully(self):
        result, _ = self._build_and_run()
        assert is_ok(result)

    def test_writes_to_output_path(self):
        files = []
        self._build_and_run(output_path="/tmp/out.xml", capture=files)
        assert len(files) == 1
        assert files[0]["path"] == "/tmp/out.xml"

    def test_xml_contains_events(self):
        files = []
        self._build_and_run(capture=files)
        content = files[0]["content"]
        assert "kick" in content
        assert "snare" in content

    def test_xml_has_three_events(self):
        files = []
        self._build_and_run(capture=files)
        content = files[0]["content"]
        assert content.count("<Event ") == 3

    def test_invalid_frame_rate(self):
        result, _ = self._build_and_run(frame_rate=60)
        assert is_err(result)

    def test_invalid_drop_frame_for_30fps(self):
        g = Graph()
        g.add_block(Block(
            id="source", name="Source", block_type="EventSource",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("events_out", PortType.EVENT, Direction.OUTPUT),
            ),
        ))
        g.add_block(Block(
            id="export", name="Export", block_type="ExportMA2",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("events_in", PortType.EVENT, Direction.INPUT),),
            output_ports=(),
            settings=BlockSettings({
                "output_path": "/tmp/test.xml",
                "frame_rate": 30,
                "drop_frame": True,
            }),
        ))
        g.add_connection(Connection("source", "events_out", "export", "events_in"))

        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)

        class FakeSource:
            def execute(self, block_id, context):
                from echozero.result import ok
                return ok(_make_event_data())

        engine.register_executor("EventSource", FakeSource())
        engine.register_executor("ExportMA2", ExportMA2Processor(lambda c, p: p))

        result = engine.run(GraphPlanner().plan(g))
        assert is_err(result)

    def test_missing_output_path(self):
        g = Graph()
        g.add_block(Block(
            id="source", name="S", block_type="EventSource",
            category=BlockCategory.PROCESSOR,
            input_ports=(), output_ports=(
                Port("events_out", PortType.EVENT, Direction.OUTPUT),
            ),
        ))
        g.add_block(Block(
            id="export", name="E", block_type="ExportMA2",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("events_in", PortType.EVENT, Direction.INPUT),),
            output_ports=(),
            settings=BlockSettings({}),
        ))
        g.add_connection(Connection("source", "events_out", "export", "events_in"))

        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)

        class FakeSource:
            def execute(self, block_id, context):
                from echozero.result import ok
                return ok(_make_event_data())

        engine.register_executor("EventSource", FakeSource())
        engine.register_executor("ExportMA2", ExportMA2Processor(lambda c, p: p))
        result = engine.run(GraphPlanner().plan(g))
        assert is_err(result)

    def test_no_events_input_returns_error(self):
        g = Graph()
        g.add_block(Block(
            id="export", name="E", block_type="ExportMA2",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("events_in", PortType.EVENT, Direction.INPUT),),
            output_ports=(),
            settings=BlockSettings({"output_path": "/tmp/x.xml"}),
        ))
        bus = RuntimeBus()
        engine = ExecutionEngine(g, bus)
        engine.register_executor("ExportMA2", ExportMA2Processor(lambda c, p: p))
        result = engine.run(GraphPlanner().plan(g))
        assert is_err(result)

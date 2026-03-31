"""
Staleness tracking tests: Verify D280 stale cascade with human-readable reasons.
Exists because users need to know WHY blocks are stale, not just that they are.
Tests cover reason creation, accumulation, downstream propagation, and clearing.
"""

from __future__ import annotations

import pytest

from echozero.editor.staleness import (
    StaleReason,
    StaleTracker,
    connection_changed_reason,
    setting_changed_reason,
)


# ---------------------------------------------------------------------------
# StaleReason factory tests
# ---------------------------------------------------------------------------


class TestStaleReasonFactories:
    """Verify human-readable reason creation."""

    def test_setting_changed_with_values(self) -> None:
        reason = setting_changed_reason(
            "sep1", "Separator", "model", "htdemucs", "htdemucs_ft"
        )
        assert "Separator" in str(reason)
        assert "model" in str(reason)
        assert "htdemucs" in str(reason)
        assert "htdemucs_ft" in str(reason)

    def test_setting_changed_without_values(self) -> None:
        reason = setting_changed_reason("det1", "Detect Onsets", "threshold")
        assert "Detect Onsets" in str(reason)
        assert "threshold" in str(reason)

    def test_connection_added(self) -> None:
        reason = connection_changed_reason("det1", "Detect Onsets", "added")
        assert "added" in str(reason)
        assert "Detect Onsets" in str(reason)

    def test_connection_removed(self) -> None:
        reason = connection_changed_reason("det1", "Detect Onsets", "removed")
        assert "removed" in str(reason)


# ---------------------------------------------------------------------------
# StaleTracker tests
# ---------------------------------------------------------------------------


class TestStaleTracker:
    """Verify reason accumulation, querying, and clearing."""

    def test_starts_empty(self) -> None:
        tracker = StaleTracker()
        assert tracker.stale_count() == 0
        assert not tracker.is_stale("any_block")

    def test_add_reason_makes_block_stale(self) -> None:
        tracker = StaleTracker()
        reason = setting_changed_reason("b1", "Block 1", "threshold")
        tracker.add_reason("b1", reason)

        assert tracker.is_stale("b1")
        assert tracker.stale_count() == 1

    def test_get_reasons_returns_added_reasons(self) -> None:
        tracker = StaleTracker()
        r1 = setting_changed_reason("b1", "Block 1", "threshold")
        r2 = setting_changed_reason("b1", "Block 1", "min_gap")
        tracker.add_reason("b1", r1)
        tracker.add_reason("b1", r2)

        reasons = tracker.get_reasons("b1")
        assert len(reasons) == 2
        assert "threshold" in str(reasons[0])
        assert "min_gap" in str(reasons[1])

    def test_reasons_accumulate_across_blocks(self) -> None:
        tracker = StaleTracker()
        tracker.add_reason("b1", setting_changed_reason("b1", "B1", "x"))
        tracker.add_reason("b2", setting_changed_reason("b2", "B2", "y"))

        assert tracker.stale_count() == 2
        assert tracker.is_stale("b1")
        assert tracker.is_stale("b2")

    def test_duplicate_reasons_not_added(self) -> None:
        tracker = StaleTracker()
        reason = setting_changed_reason("b1", "Block 1", "threshold")
        tracker.add_reason("b1", reason)
        tracker.add_reason("b1", reason)  # same description

        assert len(tracker.get_reasons("b1")) == 1

    def test_different_reasons_same_block_accumulate(self) -> None:
        tracker = StaleTracker()
        tracker.add_reason("b1", setting_changed_reason("b1", "B1", "threshold"))
        tracker.add_reason("b1", setting_changed_reason("b1", "B1", "model"))

        assert len(tracker.get_reasons("b1")) == 2

    def test_clear_removes_reasons_for_block(self) -> None:
        tracker = StaleTracker()
        tracker.add_reason("b1", setting_changed_reason("b1", "B1", "x"))
        tracker.add_reason("b2", setting_changed_reason("b2", "B2", "y"))

        tracker.clear("b1")
        assert not tracker.is_stale("b1")
        assert tracker.is_stale("b2")

    def test_clear_all(self) -> None:
        tracker = StaleTracker()
        tracker.add_reason("b1", setting_changed_reason("b1", "B1", "x"))
        tracker.add_reason("b2", setting_changed_reason("b2", "B2", "y"))

        tracker.clear_all()
        assert tracker.stale_count() == 0

    def test_clear_nonexistent_is_safe(self) -> None:
        tracker = StaleTracker()
        tracker.clear("ghost")  # should not raise

    def test_get_reasons_empty_for_fresh_block(self) -> None:
        tracker = StaleTracker()
        assert tracker.get_reasons("fresh_block") == ()

    def test_add_reason_to_downstream(self) -> None:
        tracker = StaleTracker()
        reason = setting_changed_reason("b1", "B1", "threshold")
        tracker.add_reason_to_downstream({"b2", "b3", "b4"}, reason)

        assert tracker.is_stale("b2")
        assert tracker.is_stale("b3")
        assert tracker.is_stale("b4")
        # Source block b1 is NOT in the downstream set
        assert not tracker.is_stale("b1")

    def test_get_all_stale(self) -> None:
        tracker = StaleTracker()
        tracker.add_reason("b1", setting_changed_reason("b1", "B1", "x"))
        tracker.add_reason("b2", setting_changed_reason("b2", "B2", "y"))

        all_stale = tracker.get_all_stale()
        assert set(all_stale.keys()) == {"b1", "b2"}

    def test_summary_single_reason(self) -> None:
        tracker = StaleTracker()
        tracker.add_reason("b1", setting_changed_reason("b1", "B1", "threshold"))

        summary = tracker.summary("b1")
        assert summary is not None
        assert "threshold" in summary

    def test_summary_multiple_reasons(self) -> None:
        tracker = StaleTracker()
        tracker.add_reason("b1", setting_changed_reason("b1", "B1", "threshold"))
        tracker.add_reason("b1", setting_changed_reason("b1", "B1", "model"))

        summary = tracker.summary("b1")
        assert summary is not None
        assert "2 changes" in summary
        assert "threshold" in summary
        assert "model" in summary

    def test_summary_none_for_fresh(self) -> None:
        tracker = StaleTracker()
        assert tracker.summary("fresh") is None


# ---------------------------------------------------------------------------
# Coordinator integration tests
# ---------------------------------------------------------------------------


class TestCoordinatorStaleness:
    """Verify that Coordinator wires stale reasons through propagation and clears on run."""

    def _make_coordinator(self):
        """Build a minimal coordinator with a linear graph: load → detect → classify."""
        from echozero.domain.enums import BlockCategory, Direction, PortType
        from echozero.domain.graph import Graph
        from echozero.domain.types import Block, BlockSettings, Connection, Port
        from echozero.editor.cache import ExecutionCache
        from echozero.editor.coordinator import Coordinator
        from echozero.editor.pipeline import Pipeline
        from echozero.execution import ExecutionEngine
        from echozero.progress import RuntimeBus
        from echozero.result import Ok

        graph = Graph()
        graph.add_block(Block(
            id="load", name="Load Audio", block_type="LoadAudio",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(Port("audio_out", PortType.AUDIO, Direction.OUTPUT),),
            settings=BlockSettings({"file_path": "test.wav"}),
        ))
        graph.add_block(Block(
            id="detect", name="Detect Onsets", block_type="DetectOnsets",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("audio_in", PortType.AUDIO, Direction.INPUT),),
            output_ports=(Port("events_out", PortType.EVENT, Direction.OUTPUT),),
            settings=BlockSettings({"threshold": 0.3}),
        ))
        graph.add_block(Block(
            id="classify", name="Classify", block_type="Classify",
            category=BlockCategory.PROCESSOR,
            input_ports=(Port("events_in", PortType.EVENT, Direction.INPUT),),
            output_ports=(Port("events_out", PortType.EVENT, Direction.OUTPUT),),
        ))
        graph.add_connection(Connection("load", "audio_out", "detect", "audio_in"))
        graph.add_connection(Connection("detect", "events_out", "classify", "events_in"))

        from echozero.event_bus import EventBus

        runtime_bus = RuntimeBus()
        engine = ExecutionEngine(graph, runtime_bus)
        cache = ExecutionCache()
        event_bus = EventBus()
        pipeline = Pipeline(event_bus, graph=graph)
        tracker = StaleTracker()

        coord = Coordinator(graph, pipeline, engine, cache, runtime_bus, stale_tracker=tracker)
        return coord, graph, tracker

    def test_propagate_stale_with_reason(self) -> None:
        coord, graph, tracker = self._make_coordinator()
        reason = setting_changed_reason("load", "Load Audio", "file_path")

        affected = coord.propagate_stale("load", reason=reason)

        # load + detect + classify all affected
        assert "detect" in affected
        assert "classify" in affected
        # Reasons recorded on downstream blocks
        assert tracker.is_stale("detect")
        assert tracker.is_stale("classify")
        assert "file_path" in tracker.summary("detect")

    def test_propagate_stale_without_reason_still_works(self) -> None:
        coord, graph, tracker = self._make_coordinator()

        affected = coord.propagate_stale("load")

        assert "detect" in affected
        # No reasons recorded (backward compat)
        assert not tracker.is_stale("detect")

    def test_multiple_changes_accumulate_reasons(self) -> None:
        coord, graph, tracker = self._make_coordinator()

        r1 = setting_changed_reason("detect", "Detect Onsets", "threshold")
        r2 = setting_changed_reason("detect", "Detect Onsets", "min_gap")
        coord.propagate_stale("detect", reason=r1)
        coord.propagate_stale("detect", reason=r2)

        # classify is downstream of detect, should have both reasons
        reasons = tracker.get_reasons("classify")
        assert len(reasons) == 2

    def test_successful_run_clears_reasons(self) -> None:
        from echozero.result import Ok

        coord, graph, tracker = self._make_coordinator()

        # Make things stale
        reason = setting_changed_reason("load", "Load Audio", "file_path")
        coord.propagate_stale("load", reason=reason)
        assert tracker.is_stale("detect")

        # Mock executors that return Ok
        class MockExecutor:
            def execute(self, block_id, context):
                return Ok(value={"mock": "data"})

        coord._engine.register_executor("LoadAudio", MockExecutor())
        coord._engine.register_executor("DetectOnsets", MockExecutor())
        coord._engine.register_executor("Classify", MockExecutor())

        # Run should clear reasons
        coord.request_run()

        assert not tracker.is_stale("load")
        assert not tracker.is_stale("detect")
        assert not tracker.is_stale("classify")

    def test_stale_tracker_accessible_via_property(self) -> None:
        coord, _, tracker = self._make_coordinator()
        assert coord.stale_tracker is tracker

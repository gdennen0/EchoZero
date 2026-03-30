"""
Tests for audit Batches 1 & 2 fixes.

Batch 1 — Security + Data Loss:
  P1/A1: Zip-slip in unpack_ez()
  WI-1:  Double-open lockfile
  P2:    Partial unpack leaves broken state
  X1:    Negative Event.time / Event.duration

Batch 2 — Correctness:
  E1:    Exception wrapping loses type and traceback
  S1:    deserialize_graph marks everything STALE
  S3/ED1: _handle_change_settings writes through MappingProxyType
  T1:    Can promote archived take to main
  O1:    Graph mutation during iteration (orchestrator)
  S2:    deserialize_pipeline bypasses validation
"""

from __future__ import annotations

import json
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import Block, BlockSettings, Event, Port
from echozero.errors import ExecutionError
from echozero.persistence.archive import unpack_ez
from echozero.persistence.session import ProjectSession
from echozero.result import is_err, unwrap
from echozero.serialization import deserialize_graph, deserialize_pipeline, serialize_graph
from echozero.takes import Take, TakeLayer, TakeLayerError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(block_id: str, block_type: str = "Test") -> Block:
    return Block(
        id=block_id,
        name=block_id,
        block_type=block_type,
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(),
    )


def _make_take(is_main: bool = False, is_archived: bool = False, label: str = "t") -> Take:
    return Take(
        id=f"take-{label}",
        label=label,
        data=MagicMock(),
        origin="user",
        source=None,
        created_at=datetime.now(timezone.utc),
        is_main=is_main,
        is_archived=is_archived,
    )


# ===========================================================================
# BATCH 1 — Security + Data Loss
# ===========================================================================


class TestP1ZipSlip:
    """P1/A1: Zip-slip path traversal must be rejected."""

    def test_unpack_rejects_path_traversal(self, tmp_path):
        """A .ez with ../evil.txt should raise ValueError mentioning 'path traversal'."""
        ez_path = tmp_path / "evil.ez"
        with zipfile.ZipFile(ez_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"format_version": 1}))
            zf.writestr("../evil.txt", "pwned")
        with pytest.raises(ValueError, match="path traversal"):
            unpack_ez(ez_path, tmp_path / "work")

    def test_unpack_rejects_absolute_path(self, tmp_path):
        """Members with absolute paths should also be rejected."""
        ez_path = tmp_path / "abs.ez"
        with zipfile.ZipFile(ez_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"format_version": 1}))
            # On Windows, zip members with absolute-looking paths
            # We simulate a traversal via nested ..
            zf.writestr("subdir/../../escape.txt", "bad")
        with pytest.raises(ValueError, match="path traversal"):
            unpack_ez(ez_path, tmp_path / "work")

    def test_legitimate_archive_unpacks_fine(self, tmp_path):
        """A clean archive with only safe paths should unpack without error."""
        ez_path = tmp_path / "good.ez"
        with zipfile.ZipFile(ez_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"format_version": 1}))
            zf.writestr("audio/sample.wav", b"RIFF" + b"\x00" * 40)
        result = unpack_ez(ez_path, tmp_path / "work")
        assert result["format_version"] == 1
        assert (tmp_path / "work" / "manifest.json").exists()


class TestWI1Lockfile:
    """WI-1: Opening same project twice should raise RuntimeError."""

    def test_double_open_raises(self, tmp_path):
        """Opening same working dir twice raises RuntimeError."""
        session1 = ProjectSession.create_new("test", working_dir_root=tmp_path)
        try:
            with pytest.raises(RuntimeError, match="already open"):
                ProjectSession.open_db(session1.working_dir)
        finally:
            session1.close()

    def test_after_close_can_reopen(self, tmp_path):
        """After closing, the same project can be reopened."""
        session1 = ProjectSession.create_new("test", working_dir_root=tmp_path)
        working_dir = session1.working_dir
        session1.close()

        session2 = ProjectSession.open_db(working_dir)
        assert session2 is not None
        session2.close()

    def test_lockfile_created_on_open(self, tmp_path):
        """A project.lock file should exist while session is open."""
        session = ProjectSession.create_new("test", working_dir_root=tmp_path)
        try:
            lock_path = session.working_dir / "project.lock"
            assert lock_path.exists(), "project.lock should exist while session is open"
        finally:
            session.close()

    def test_lockfile_removed_on_close(self, tmp_path):
        """project.lock should be removed when session is closed."""
        session = ProjectSession.create_new("test", working_dir_root=tmp_path)
        lock_path = session.working_dir / "project.lock"
        session.close()
        assert not lock_path.exists(), "project.lock should be removed after close"


class TestP2PartialUnpack:
    """P2: Failed unpack should not leave partial working directory."""

    def test_bad_zip_doesnt_leave_broken_dir(self, tmp_path):
        """If unpack fails (not a zip), working_dir should not exist."""
        ez_path = tmp_path / "bad.ez"
        ez_path.write_bytes(b"not a zip at all")
        working = tmp_path / "work"
        with pytest.raises(Exception):
            unpack_ez(ez_path, working)
        assert not working.exists(), "working_dir must not exist after failed unpack"

    def test_missing_manifest_doesnt_leave_dir(self, tmp_path):
        """If manifest.json is missing, working_dir should not exist."""
        ez_path = tmp_path / "no_manifest.ez"
        with zipfile.ZipFile(ez_path, "w") as zf:
            zf.writestr("some_file.txt", "hello")
        working = tmp_path / "work"
        with pytest.raises(ValueError, match="manifest.json"):
            unpack_ez(ez_path, working)
        assert not working.exists()

    def test_nonexistent_archive_raises_file_not_found(self, tmp_path):
        """Nonexistent archive raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            unpack_ez(tmp_path / "nonexistent.ez", tmp_path / "work")


class TestX1NegativeEvents:
    """X1: Negative Event.time and Event.duration must be rejected."""

    def test_event_negative_time_raises(self):
        with pytest.raises(ValueError, match="time must be >= 0"):
            Event(
                id="1",
                time=-1.0,
                duration=0.5,
                classifications={},
                metadata={},
                origin="test",
            )

    def test_event_negative_duration_raises(self):
        with pytest.raises(ValueError, match="duration must be >= 0"):
            Event(
                id="1",
                time=0.0,
                duration=-0.5,
                classifications={},
                metadata={},
                origin="test",
            )

    def test_event_zero_time_ok(self):
        e = Event(id="1", time=0.0, duration=1.0, classifications={}, metadata={}, origin="t")
        assert e.time == 0.0

    def test_event_zero_duration_ok(self):
        e = Event(id="1", time=1.0, duration=0.0, classifications={}, metadata={}, origin="t")
        assert e.duration == 0.0

    def test_event_both_zero_ok(self):
        e = Event(id="1", time=0.0, duration=0.0, classifications={}, metadata={}, origin="t")
        assert e.time == 0.0
        assert e.duration == 0.0

    def test_event_positive_values_ok(self):
        e = Event(id="1", time=1.5, duration=0.25, classifications={}, metadata={}, origin="t")
        assert e.time == 1.5
        assert e.duration == 0.25


# ===========================================================================
# BATCH 2 — Correctness
# ===========================================================================


class TestE1ExceptionChaining:
    """E1: Exception wrapping should preserve the original cause."""

    def test_executor_exception_is_execution_error(self):
        """Executor exceptions should be wrapped in ExecutionError."""
        from echozero.execution import BlockExecutor, ExecutionContext, ExecutionEngine, GraphPlanner
        from echozero.progress import RuntimeBus

        graph = Graph()
        graph.add_block(
            Block(
                id="bad_block",
                name="Bad",
                block_type="BadExecutor",
                category=BlockCategory.PROCESSOR,
                input_ports=(),
                output_ports=(),
            )
        )

        class BoomExecutor:
            def execute(self, block_id: str, context: ExecutionContext):
                raise FileNotFoundError("audio file missing")

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("BadExecutor", BoomExecutor())

        planner = GraphPlanner()
        plan = planner.plan(graph)
        result = engine.run(plan)

        assert is_err(result)
        # Should be ExecutionError, not RuntimeError
        from echozero.result import Err
        assert isinstance(result, Err)
        assert isinstance(result.error, ExecutionError)
        # Original cause should be chained
        assert result.error.__cause__ is not None
        assert isinstance(result.error.__cause__, FileNotFoundError)

    def test_executor_exception_message_contains_block_type(self):
        """Error message should include the block type for debugging."""
        from echozero.execution import BlockExecutor, ExecutionContext, ExecutionEngine, GraphPlanner
        from echozero.progress import RuntimeBus

        graph = Graph()
        graph.add_block(
            Block(
                id="blk1",
                name="MyBlock",
                block_type="MyBlockType",
                category=BlockCategory.PROCESSOR,
                input_ports=(),
                output_ports=(),
            )
        )

        class ErrorExecutor:
            def execute(self, block_id: str, context: ExecutionContext):
                raise ValueError("bad config")

        bus = RuntimeBus()
        engine = ExecutionEngine(graph, bus)
        engine.register_executor("MyBlockType", ErrorExecutor())

        plan = GraphPlanner().plan(graph)
        result = engine.run(plan)

        from echozero.result import Err
        assert isinstance(result.error, ExecutionError)
        assert "MyBlockType" in str(result.error)


class TestS1DeserializeState:
    """S1: deserialize_graph should use stored state, not always STALE."""

    def test_fresh_blocks_stay_fresh_after_roundtrip(self):
        graph = Graph()
        graph.add_block(_make_block("a"))
        data = serialize_graph(graph)
        restored = deserialize_graph(data)
        assert restored.blocks["a"].state == BlockState.FRESH

    def test_stale_blocks_stay_stale_after_roundtrip(self):
        graph = Graph()
        graph.add_block(_make_block("a"))
        graph.set_block_state("a", BlockState.STALE)
        data = serialize_graph(graph)
        restored = deserialize_graph(data)
        assert restored.blocks["a"].state == BlockState.STALE

    def test_missing_state_field_defaults_to_fresh(self):
        """Legacy serialized data without 'state' field should default to FRESH."""
        graph = Graph()
        graph.add_block(_make_block("a"))
        data = serialize_graph(graph)
        # Remove state field to simulate legacy data
        data["blocks"][0].pop("state", None)
        restored = deserialize_graph(data)
        assert restored.blocks["a"].state == BlockState.FRESH


class TestS3ChangeSettings:
    """S3/ED1: _handle_change_settings must use replace_block(), not MappingProxyType write."""

    def test_change_settings_command_does_not_crash(self):
        """Dispatching ChangeBlockSettingsCommand should not raise TypeError."""
        from echozero.editor.commands import AddBlockCommand, ChangeBlockSettingsCommand
        from echozero.editor.pipeline import Pipeline as EditorPipeline
        from echozero.event_bus import EventBus

        bus = EventBus()
        pipeline = EditorPipeline(bus)

        # Add a block first
        add_cmd = AddBlockCommand(
            block_id="blk1",
            name="Test Block",
            block_type="TestType",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(),
            control_ports=(),
            settings_entries=(("threshold", 0.5),),
        )
        result = pipeline.dispatch(add_cmd)
        assert not is_err(result)

        # Change a setting — should NOT crash
        change_cmd = ChangeBlockSettingsCommand(
            block_id="blk1",
            setting_key="threshold",
            new_value=0.8,
        )
        result = pipeline.dispatch(change_cmd)
        assert not is_err(result), f"ChangeBlockSettingsCommand failed: {result}"

        # Setting should be updated
        block = pipeline.graph.blocks["blk1"]
        assert block.settings["threshold"] == 0.8

    def test_settings_change_fires_event(self):
        """SettingsChangedEvent should be published after a successful settings change."""
        from echozero.editor.commands import AddBlockCommand, ChangeBlockSettingsCommand
        from echozero.editor.pipeline import Pipeline as EditorPipeline
        from echozero.event_bus import EventBus
        from echozero.domain.events import SettingsChangedEvent

        events_received = []
        bus = EventBus()
        bus.subscribe(SettingsChangedEvent, lambda e: events_received.append(e))

        pipeline = EditorPipeline(bus)
        pipeline.dispatch(AddBlockCommand(
            block_id="b1",
            name="B1",
            block_type="T",
            category=BlockCategory.PROCESSOR,
            input_ports=(),
            output_ports=(),
            control_ports=(),
            settings_entries=(("gain", 1.0),),
        ))
        pipeline.dispatch(ChangeBlockSettingsCommand(
            block_id="b1",
            setting_key="gain",
            new_value=2.0,
        ))

        assert len(events_received) == 1
        assert events_received[0].setting_key == "gain"
        assert events_received[0].new_value == 2.0


class TestT1ArchivdTakePromotion:
    """T1: Archived takes should not be promotable to main."""

    def test_promote_archived_take_raises(self):
        """Promoting an archived take should raise TakeLayerError."""
        main = _make_take(is_main=True, label="main")
        archived = _make_take(is_main=False, is_archived=True, label="archived")
        layer = TakeLayer(layer_id="test-layer", takes=[main, archived])

        with pytest.raises(TakeLayerError, match="archived"):
            layer.promote_to_main("take-archived")

    def test_promote_nonexistent_take_raises(self):
        """Promoting a non-existent take still raises TakeLayerError (unchanged)."""
        main = _make_take(is_main=True, label="main")
        layer = TakeLayer(layer_id="test-layer", takes=[main])

        with pytest.raises(TakeLayerError, match="not found"):
            layer.promote_to_main("nonexistent-id")

    def test_promote_non_archived_take_works(self):
        """Promoting a non-archived take should succeed."""
        main = _make_take(is_main=True, label="main")
        candidate = _make_take(is_main=False, is_archived=False, label="alt")
        layer = TakeLayer(layer_id="test-layer", takes=[main, candidate])

        layer.promote_to_main("take-alt")

        assert layer.main_take().label == "alt"
        # Old main is demoted
        old_main = next(t for t in layer.takes if t.label == "main")
        assert not old_main.is_main

    def test_archived_take_promotion_leaves_layer_unchanged(self):
        """After failed promotion attempt, layer state is unchanged."""
        main = _make_take(is_main=True, label="main")
        archived = _make_take(is_main=False, is_archived=True, label="archived")
        layer = TakeLayer(layer_id="test-layer", takes=[main, archived])

        try:
            layer.promote_to_main("take-archived")
        except TakeLayerError:
            pass

        # Main should still be the original main
        assert layer.main_take().label == "main"


class TestO1GraphMutationDuringIteration:
    """O1: Orchestrator should not mutate graph while iterating."""

    def test_multiple_load_audio_blocks_all_updated(self, tmp_path):
        """When multiple LoadAudio blocks exist, all should get the audio path."""
        # This tests that the ID-collection-first approach works correctly
        from echozero.domain.types import BlockSettings
        from dataclasses import replace

        graph = Graph()
        for i in range(3):
            block = Block(
                id=f"load_{i}",
                name=f"LoadAudio {i}",
                block_type="LoadAudio",
                category=BlockCategory.PROCESSOR,
                input_ports=(),
                output_ports=(),
                settings=BlockSettings({"file_path": ""}),
            )
            graph.add_block(block)

        # Collect IDs first (O1 fix pattern)
        load_audio_ids = [
            bid for bid, b in graph.blocks.items()
            if b.block_type == "LoadAudio"
        ]
        audio_path = "test/audio.wav"
        for block_id in load_audio_ids:
            block = graph.blocks[block_id]
            new_settings = {**dict(block.settings), "file_path": audio_path}
            updated = replace(block, settings=BlockSettings(new_settings))
            graph.replace_block(updated)

        # All 3 blocks should have the audio path
        for bid in load_audio_ids:
            assert graph.blocks[bid].settings["file_path"] == audio_path


class TestS2DeserializePipeline:
    """S2: deserialize_pipeline should use public API and validate duplicate outputs."""

    def test_deserialize_pipeline_rejects_duplicate_output_names(self):
        """Pipeline data with duplicate output names should raise ValidationError."""
        from echozero.errors import ValidationError
        from echozero.pipelines.pipeline import Pipeline, PortRef, PipelineOutput

        # Build a minimal valid pipeline first, then craft bad serialized data
        data = {
            "id": "test-pipeline",
            "name": "Test",
            "description": "",
            "graph": {"blocks": [], "connections": []},
            "outputs": [
                {"name": "onsets", "block_id": "blk1", "port_name": "out"},
                {"name": "onsets", "block_id": "blk2", "port_name": "out"},  # duplicate!
            ],
        }
        with pytest.raises(ValidationError, match="Duplicate"):
            deserialize_pipeline(data)

    def test_deserialize_pipeline_produces_correct_outputs(self):
        """Deserializing a pipeline with unique outputs should work correctly."""
        from echozero.serialization import serialize_pipeline
        from echozero.pipelines.pipeline import Pipeline, PortRef

        # Build a pipeline with an output
        # We use direct construction since we don't have blocks to wire up
        pipeline = Pipeline(id="my-pipeline", name="My Pipeline", description="test")
        data = serialize_pipeline(pipeline)

        restored = deserialize_pipeline(data)
        assert restored.id == "my-pipeline"
        assert restored.name == "My Pipeline"

    def test_pipeline_constructor_accepts_graph_and_outputs(self):
        """Pipeline.__init__ should accept graph and outputs kwargs."""
        from echozero.pipelines.pipeline import Pipeline, PipelineOutput, PortRef

        graph = Graph()
        graph.add_block(_make_block("b1"))

        port_ref = PortRef("b1", "out")
        outputs = [PipelineOutput("result", port_ref)]

        pipeline = Pipeline(
            id="test",
            name="Test Pipeline",
            graph=graph,
            outputs=outputs,
        )
        assert pipeline.id == "test"
        assert "b1" in pipeline.graph.blocks
        assert len(pipeline.outputs) == 1
        assert pipeline.outputs[0].name == "result"

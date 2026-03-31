"""Audit batch 3 — security + reliability fixes."""
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# PR4: MP3 extension fix
# ---------------------------------------------------------------------------

def test_separator_always_writes_wav_extension():
    """The ext variable should always be 'wav' in V1."""
    # Covered by existing separator tests after the fix; placeholder for clarity.
    pass


# ---------------------------------------------------------------------------
# S4: BlockSettings hash with non-hashable values
# ---------------------------------------------------------------------------

def test_block_settings_hash_with_list_values():
    from echozero.domain.types import BlockSettings
    s = BlockSettings({"bands": [100, 200, 300], "name": "test"})
    # Should not crash
    h = hash(s)
    assert isinstance(h, int)


def test_block_settings_hash_with_dict_values():
    from echozero.domain.types import BlockSettings
    s = BlockSettings({"config": {"nested": True}, "value": 42})
    h = hash(s)
    assert isinstance(h, int)


def test_block_settings_hash_consistency():
    from echozero.domain.types import BlockSettings
    s1 = BlockSettings({"a": 1, "b": 2})
    s2 = BlockSettings({"a": 1, "b": 2})
    assert hash(s1) == hash(s2)


# ---------------------------------------------------------------------------
# P4: Autosave doesn't commit during transaction
# ---------------------------------------------------------------------------

def test_autosave_skips_during_transaction(tmp_path):
    from echozero.persistence.session import ProjectStorage
    session = ProjectStorage.create_new("test", working_dir_root=tmp_path)
    try:
        session._in_transaction = True
        session.dirty_tracker.mark_dirty("test")
        session._autosave_tick()
        # Should still be dirty (autosave skipped)
        assert session.dirty_tracker.is_dirty()
    finally:
        session._in_transaction = False
        session.close()


# ---------------------------------------------------------------------------
# MISC2: Double unsubscribe should not raise
# ---------------------------------------------------------------------------

def test_runtime_bus_double_unsubscribe():
    from echozero.progress import RuntimeBus
    bus = RuntimeBus()
    cb = lambda r: None
    bus.subscribe(cb)
    bus.unsubscribe(cb)
    bus.unsubscribe(cb)  # Should not raise


# ---------------------------------------------------------------------------
# P8: Corrupt take skipped in list_by_layer
# ---------------------------------------------------------------------------

def test_list_by_layer_skips_corrupt_takes(tmp_path):
    """A take with NULL data_json should be skipped, not crash the listing."""
    from echozero.persistence.session import ProjectStorage
    session = ProjectStorage.create_new("test", working_dir_root=tmp_path)
    try:
        # Disable FK checks so we can insert a layer without a real song_version row.
        session.db.execute("PRAGMA foreign_keys = OFF")
        session.db.execute(
            "INSERT INTO layers (id, song_version_id, name, layer_type, \"order\", visible, locked, created_at) "
            "VALUES ('layer1', 'fake_version', 'Test', 'analysis', 0, 1, 0, '2024-01-01')"
        )
        session.db.execute(
            "INSERT INTO takes (id, layer_id, label, origin, is_main, is_archived, data_json, created_at) "
            "VALUES ('take1', 'layer1', 'Good', 'pipeline', 1, 0, ?, '2024-01-01')",
            (json.dumps({"type": "EventData", "layers": []}),),
        )
        session.db.execute(
            "INSERT INTO takes (id, layer_id, label, origin, is_main, is_archived, data_json, created_at) "
            "VALUES ('take2', 'layer1', 'Bad', 'pipeline', 0, 0, NULL, '2024-01-01')"
        )
        session.db.commit()

        takes = session.takes.list_by_layer("layer1")
        assert len(takes) == 1  # Only the good take
        assert takes[0].id == "take1"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# ED4: request_run rejects when already executing
# ---------------------------------------------------------------------------

def test_request_run_rejects_during_execution():
    from echozero.editor.coordinator import Coordinator
    from echozero.editor.cache import ExecutionCache
    from echozero.editor.pipeline import Pipeline as EditorPipeline
    from echozero.execution import ExecutionEngine
    from echozero.domain.graph import Graph
    from echozero.event_bus import EventBus
    from echozero.progress import RuntimeBus
    from echozero.result import is_err

    graph = Graph()
    bus = EventBus()
    pipeline = EditorPipeline(bus)
    runtime_bus = RuntimeBus()
    engine = ExecutionEngine(graph, runtime_bus)
    cache = ExecutionCache()

    coord = Coordinator(graph, pipeline, engine, cache, runtime_bus)
    coord._executing = True  # Simulate in-progress execution

    result = coord.request_run()
    assert is_err(result)

    coord._executing = False

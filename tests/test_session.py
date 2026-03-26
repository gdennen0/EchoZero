"""
ProjectSession + DirtyTracker tests: lifecycle, persistence round-trips, autosave, recovery.
Exercises the session layer against real SQLite working directories in temp folders.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from echozero.domain.enums import (
    BlockCategory,
    BlockState,
    Direction,
    PortType,
)
from echozero.domain.events import (
    BlockAddedEvent,
    BlockRemovedEvent,
    BlockStateChangedEvent,
    ConnectionAddedEvent,
    ConnectionRemovedEvent,
    SettingsChangedEvent,
    create_event_id,
)
from echozero.domain.graph import Graph
from echozero.domain.types import (
    Block,
    BlockSettings,
    Connection,
    Event,
    EventData,
    Layer,
    Port,
)
from echozero.event_bus import EventBus
from echozero.persistence.dirty import DirtyTracker
from echozero.persistence.entities import (
    LayerRecord,
    Project,
    ProjectSettings,
    Song,
    SongVersion,
)
from echozero.persistence.repositories import PipelineConfigRepository
from echozero.persistence.session import ProjectSession
from echozero.takes import Take, TakeSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_event(correlation_id: str = "") -> dict:
    """Common fields for constructing a DomainEvent."""
    return {
        "event_id": create_event_id(),
        "timestamp": time.time(),
        "correlation_id": correlation_id or _uid(),
    }


def _make_graph() -> Graph:
    """Build a small two-block graph for round-trip testing."""
    graph = Graph()
    b1 = Block(
        id="b1",
        name="Source",
        block_type="audio_source",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(
            Port(name="audio_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        ),
        settings=BlockSettings(entries={"file": "test.wav"}),
    )
    b2 = Block(
        id="b2",
        name="Detector",
        block_type="onset_detector",
        category=BlockCategory.PROCESSOR,
        input_ports=(
            Port(name="audio_in", port_type=PortType.AUDIO, direction=Direction.INPUT),
        ),
        output_ports=(
            Port(name="events_out", port_type=PortType.EVENT, direction=Direction.OUTPUT),
        ),
        settings=BlockSettings(entries={"threshold": 0.3}),
    )
    graph.add_block(b1)
    graph.add_block(b2)
    graph.add_connection(Connection(
        source_block_id="b1",
        source_output_name="audio_out",
        target_block_id="b2",
        target_input_name="audio_in",
    ))
    return graph


def _make_song(project_id: str, **kw) -> Song:
    defaults = dict(
        id=_uid(), project_id=project_id, title="Song A",
        artist="Artist", order=0, active_version_id=None,
    )
    defaults.update(kw)
    return Song(**defaults)


def _make_version(song_id: str, **kw) -> SongVersion:
    defaults = dict(
        id=_uid(), song_id=song_id, label="Studio Mix",
        audio_file="audio/song.wav", duration_seconds=180.0,
        original_sample_rate=44100, audio_hash="abc123", created_at=_now(),
    )
    defaults.update(kw)
    return SongVersion(**defaults)


def _make_layer(song_version_id: str, **kw) -> LayerRecord:
    defaults = dict(
        id=_uid(), song_version_id=song_version_id, name="Drums",
        layer_type="analysis", color="#FF0000", order=0,
        visible=True, locked=False, parent_layer_id=None,
        source_pipeline=None, created_at=_now(),
    )
    defaults.update(kw)
    return LayerRecord(**defaults)


def _make_event_data() -> EventData:
    return EventData(layers=(
        Layer(id=_uid(), name="onsets", events=(
            Event(
                id=_uid(), time=1.0, duration=0.1,
                classifications={"type": "kick"}, metadata={}, origin="pipeline",
            ),
        )),
    ))


def _make_take(is_main: bool = False, **kw) -> Take:
    defaults = dict(
        id=_uid(), label="Take 1", data=_make_event_data(),
        origin="pipeline", source=None, created_at=_now(),
        is_main=is_main, notes="",
    )
    defaults.update(kw)
    return Take(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_root(tmp_path):
    """Provide a temporary working dir root for each test."""
    return tmp_path / "working"


@pytest.fixture
def event_bus():
    bus = EventBus()
    yield bus
    bus.clear()


# ---------------------------------------------------------------------------
# DirtyTracker unit tests
# ---------------------------------------------------------------------------


class TestDirtyTracker:
    def test_starts_clean(self):
        tracker = DirtyTracker()
        assert tracker.is_dirty() is False
        assert tracker.dirty_ids == set()

    def test_mark_dirty_without_entity(self):
        tracker = DirtyTracker()
        tracker.mark_dirty()
        assert tracker.is_dirty() is True
        assert tracker.dirty_ids == set()

    def test_mark_dirty_with_entity(self):
        tracker = DirtyTracker()
        tracker.mark_dirty("song_123")
        assert tracker.is_dirty() is True
        assert tracker.dirty_ids == {"song_123"}

    def test_clear_resets(self):
        tracker = DirtyTracker()
        tracker.mark_dirty("a")
        tracker.mark_dirty("b")
        tracker.clear()
        assert tracker.is_dirty() is False
        assert tracker.dirty_ids == set()

    def test_dirty_ids_returns_copy(self):
        tracker = DirtyTracker()
        tracker.mark_dirty("x")
        ids = tracker.dirty_ids
        ids.add("y")
        assert "y" not in tracker.dirty_ids

    def test_subscribes_to_block_added(self, event_bus):
        tracker = DirtyTracker(event_bus)
        event_bus.publish(BlockAddedEvent(
            **_make_event(), block_id="blk1", block_type="onset_detector",
        ))
        assert tracker.is_dirty() is True
        assert "blk1" in tracker.dirty_ids

    def test_subscribes_to_block_removed(self, event_bus):
        tracker = DirtyTracker(event_bus)
        event_bus.publish(BlockRemovedEvent(**_make_event(), block_id="blk1"))
        assert tracker.is_dirty() is True
        assert "blk1" in tracker.dirty_ids

    def test_subscribes_to_connection_added(self, event_bus):
        tracker = DirtyTracker(event_bus)
        event_bus.publish(ConnectionAddedEvent(
            **_make_event(), source_block_id="a", target_block_id="b",
        ))
        assert tracker.is_dirty() is True

    def test_subscribes_to_connection_removed(self, event_bus):
        tracker = DirtyTracker(event_bus)
        event_bus.publish(ConnectionRemovedEvent(
            **_make_event(), source_block_id="a", target_block_id="b",
        ))
        assert tracker.is_dirty() is True

    def test_block_state_changed_does_not_dirty(self, event_bus):
        """BlockStateChangedEvent is transient execution state — not a structural mutation."""
        tracker = DirtyTracker(event_bus)
        event_bus.publish(BlockStateChangedEvent(
            **_make_event(), block_id="blk1",
            old_state=BlockState.STALE, new_state=BlockState.FRESH,
        ))
        assert tracker.is_dirty() is False

    def test_subscribes_to_settings_changed(self, event_bus):
        tracker = DirtyTracker(event_bus)
        event_bus.publish(SettingsChangedEvent(
            **_make_event(), block_id="blk1", setting_key="threshold",
        ))
        assert tracker.is_dirty() is True
        assert "blk1" in tracker.dirty_ids

    def test_multiple_events_accumulate_ids(self, event_bus):
        tracker = DirtyTracker(event_bus)
        event_bus.publish(BlockAddedEvent(**_make_event(), block_id="a", block_type="t"))
        event_bus.publish(BlockAddedEvent(**_make_event(), block_id="b", block_type="t"))
        assert tracker.dirty_ids == {"a", "b"}

    def test_no_event_bus_still_works(self):
        tracker = DirtyTracker(None)
        assert tracker.is_dirty() is False
        tracker.mark_dirty("x")
        assert tracker.is_dirty() is True

    def test_unsubscribe(self, event_bus):
        tracker = DirtyTracker(event_bus)
        tracker._unsubscribe()
        event_bus.publish(BlockAddedEvent(**_make_event(), block_id="blk1", block_type="t"))
        assert tracker.is_dirty() is False


# ---------------------------------------------------------------------------
# ProjectSession — creation and basic lifecycle
# ---------------------------------------------------------------------------


class TestSessionCreate:
    def test_create_new_creates_working_dir(self, tmp_root):
        session = ProjectSession.create_new("Test", working_dir_root=tmp_root)
        try:
            assert session.working_dir.exists()
            assert (session.working_dir / "project.db").exists()
        finally:
            session.close()

    def test_create_new_project_entity(self, tmp_root):
        session = ProjectSession.create_new(
            "My Show",
            settings=ProjectSettings(bpm=120.0),
            working_dir_root=tmp_root,
        )
        try:
            assert session.project.name == "My Show"
            assert session.project.settings.bpm == 120.0
            assert session.project.id is not None
        finally:
            session.close()

    def test_create_new_project_persisted(self, tmp_root):
        session = ProjectSession.create_new("Persisted", working_dir_root=tmp_root)
        try:
            got = session.projects.get(session.project.id)
            assert got is not None
            assert got.name == "Persisted"
        finally:
            session.close()

    def test_create_new_default_settings(self, tmp_root):
        session = ProjectSession.create_new("Defaults", working_dir_root=tmp_root)
        try:
            assert session.project.settings.sample_rate == 44100
            assert session.project.settings.bpm is None
        finally:
            session.close()

    def test_create_new_starts_clean(self, tmp_root):
        session = ProjectSession.create_new("Clean", working_dir_root=tmp_root)
        try:
            assert session.is_dirty() is False
        finally:
            session.close()

    def test_wal_mode_enabled(self, tmp_root):
        """Verify WAL mode is set on new sessions."""
        session = ProjectSession.create_new("WAL", working_dir_root=tmp_root)
        try:
            mode = session.db.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"
        finally:
            session.close()

    def test_row_factory_is_sqlite_row(self, tmp_root):
        """Verify row_factory is set to sqlite3.Row."""
        session = ProjectSession.create_new("RowFactory", working_dir_root=tmp_root)
        try:
            assert session.db.row_factory is sqlite3.Row
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Open from working directory
# ---------------------------------------------------------------------------


class TestSessionOpenDb:
    def test_open_db_loads_project(self, tmp_root):
        session1 = ProjectSession.create_new("Reopen", working_dir_root=tmp_root)
        wd = session1.working_dir
        pid = session1.project.id
        session1.close()

        session2 = ProjectSession.open_db(wd)
        try:
            assert session2.project.id == pid
            assert session2.project.name == "Reopen"
        finally:
            session2.close()

    def test_open_db_missing_dir_raises(self, tmp_root):
        with pytest.raises(FileNotFoundError):
            ProjectSession.open_db(tmp_root / "nonexistent")

    def test_open_db_empty_db_raises(self, tmp_root):
        wd = tmp_root / "empty"
        wd.mkdir(parents=True)
        conn = sqlite3.connect(str(wd / "project.db"))
        from echozero.persistence.schema import init_db
        init_db(conn)
        conn.close()

        with pytest.raises(ValueError, match="No project found"):
            ProjectSession.open_db(wd)

    def test_open_db_has_wal_mode(self, tmp_root):
        """Verify WAL mode is set when reopening an existing DB."""
        session1 = ProjectSession.create_new("WALReopen", working_dir_root=tmp_root)
        wd = session1.working_dir
        session1.close()

        session2 = ProjectSession.open_db(wd)
        try:
            mode = session2.db.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"
            assert session2.db.row_factory is sqlite3.Row
        finally:
            session2.close()


# ---------------------------------------------------------------------------
# Transaction control
# ---------------------------------------------------------------------------


class TestSessionTransaction:
    def test_transaction_commits_on_success(self, tmp_root):
        session = ProjectSession.create_new("TxnSuccess", working_dir_root=tmp_root)
        try:
            song = _make_song(session.project.id, title="Atomic Song")
            version = _make_version(song.id)

            with session.transaction():
                session.songs.create(song)
                session.song_versions.create(version)

            # Data should be committed and readable after reopen
            wd = session.working_dir
            session.close()

            session2 = ProjectSession.open_db(wd)
            try:
                songs = session2.songs.list_by_project(session2.project.id)
                assert len(songs) == 1
                assert songs[0].title == "Atomic Song"
                versions = session2.song_versions.list_by_song(song.id)
                assert len(versions) == 1
            finally:
                session2.close()
        except Exception:
            session.close()
            raise

    def test_transaction_rolls_back_on_error(self, tmp_root):
        session = ProjectSession.create_new("TxnRollback", working_dir_root=tmp_root)
        try:
            song = _make_song(session.project.id, title="Rolled Back")

            with pytest.raises(ValueError, match="boom"):
                with session.transaction():
                    session.songs.create(song)
                    raise ValueError("boom")

            # Song should NOT be persisted after rollback
            songs = session.songs.list_by_project(session.project.id)
            assert len(songs) == 0
        finally:
            session.close()

    def test_commit_explicitly(self, tmp_root):
        session = ProjectSession.create_new("ExplicitCommit", working_dir_root=tmp_root)
        try:
            song = _make_song(session.project.id, title="Committed")
            session.songs.create(song)
            session.commit()

            # Verify persisted across reopen
            wd = session.working_dir
            session.close()

            session2 = ProjectSession.open_db(wd)
            try:
                songs = session2.songs.list_by_project(session2.project.id)
                assert len(songs) == 1
                assert songs[0].title == "Committed"
            finally:
                session2.close()
        except Exception:
            session.close()
            raise

    def test_commit_after_close_raises(self, tmp_root):
        session = ProjectSession.create_new("ClosedCommit", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.commit()

    def test_transaction_after_close_raises(self, tmp_root):
        session = ProjectSession.create_new("ClosedTxn", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            with session.transaction():
                pass


# ---------------------------------------------------------------------------
# Save and round-trip
# ---------------------------------------------------------------------------


class TestSessionSave:
    def test_save_clears_dirty(self, tmp_root):
        session = ProjectSession.create_new("Dirty", working_dir_root=tmp_root)
        try:
            session.dirty_tracker.mark_dirty()
            assert session.is_dirty() is True
            session.save()
            assert session.is_dirty() is False
        finally:
            session.close()

    def test_save_as_creates_ez_file(self, tmp_root, tmp_path):
        """save_as() creates a .ez archive file."""
        session = ProjectSession.create_new("SaveAs", working_dir_root=tmp_root)
        try:
            ez_path = tmp_path / "output.ez"
            session.save_as(ez_path)
            assert ez_path.exists()

            import zipfile
            assert zipfile.is_zipfile(ez_path)
            with zipfile.ZipFile(ez_path, "r") as zf:
                assert "manifest.json" in zf.namelist()
                assert "project.db" in zf.namelist()
        finally:
            session.close()

    def test_save_as_clears_dirty(self, tmp_root, tmp_path):
        """save_as() clears the dirty tracker."""
        session = ProjectSession.create_new("DirtySaveAs", working_dir_root=tmp_root)
        try:
            session.dirty_tracker.mark_dirty()
            assert session.is_dirty() is True
            ez_path = tmp_path / "output.ez"
            session.save_as(ez_path)
            assert session.is_dirty() is False
        finally:
            session.close()

    def test_save_as_after_close_raises(self, tmp_root, tmp_path):
        """save_as() raises RuntimeError after close."""
        session = ProjectSession.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.save_as(tmp_path / "output.ez")

    def test_full_hierarchy_round_trip(self, tmp_root):
        """Create -> add songs/versions/layers/takes -> save -> reopen -> verify."""
        session = ProjectSession.create_new(
            "Tour 2026",
            settings=ProjectSettings(sample_rate=48000, bpm=128.0),
            working_dir_root=tmp_root,
        )
        pid = session.project.id
        wd = session.working_dir

        # Build hierarchy
        song = _make_song(pid, title="Opening Act", artist="The Band")
        session.songs.create(song)

        version = _make_version(song.id, label="Final Mix")
        session.song_versions.create(version)

        layer = _make_layer(version.id, name="Onsets", color="#FF5500")
        session.layers.create(layer)

        take = _make_take(is_main=True, label="Initial Analysis")
        session.takes.create(layer.id, take)

        session.save()
        session.close()

        # Reopen and verify
        session2 = ProjectSession.open_db(wd)
        try:
            assert session2.project.name == "Tour 2026"
            assert session2.project.settings.bpm == 128.0

            songs = session2.songs.list_by_project(pid)
            assert len(songs) == 1
            assert songs[0].title == "Opening Act"

            versions = session2.song_versions.list_by_song(song.id)
            assert len(versions) == 1
            assert versions[0].label == "Final Mix"

            layers = session2.layers.list_by_version(version.id)
            assert len(layers) == 1
            assert layers[0].color == "#FF5500"

            got_take = session2.takes.get_main(layer.id)
            assert got_take is not None
            assert got_take.label == "Initial Analysis"
        finally:
            session2.close()


# ---------------------------------------------------------------------------
# DirtyTracker integration with session
# ---------------------------------------------------------------------------


class TestSessionDirtyIntegration:
    def test_event_bus_mutations_dirty_session(self, tmp_root):
        bus = EventBus()
        session = ProjectSession.create_new(
            "Evented", event_bus=bus, working_dir_root=tmp_root,
        )
        try:
            assert session.is_dirty() is False
            bus.publish(BlockAddedEvent(
                **_make_event(), block_id="blk1", block_type="t",
            ))
            assert session.is_dirty() is True
        finally:
            session.close()

    def test_save_clears_event_bus_dirty(self, tmp_root):
        bus = EventBus()
        session = ProjectSession.create_new(
            "Evented2", event_bus=bus, working_dir_root=tmp_root,
        )
        try:
            bus.publish(BlockAddedEvent(
                **_make_event(), block_id="blk1", block_type="t",
            ))
            assert session.is_dirty() is True
            session.save()
            assert session.is_dirty() is False
        finally:
            session.close()

    def test_manual_dirty_via_persistence_op(self, tmp_root):
        session = ProjectSession.create_new("Manual", working_dir_root=tmp_root)
        try:
            session.dirty_tracker.mark_dirty("song_xyz")
            assert session.is_dirty() is True
            assert "song_xyz" in session.dirty_tracker.dirty_ids
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Graph save/load
# ---------------------------------------------------------------------------


class TestSessionGraph:
    def test_save_and_load_graph(self, tmp_root):
        session = ProjectSession.create_new("Graphed", working_dir_root=tmp_root)
        try:
            graph = _make_graph()
            session.save_graph(graph)

            loaded = session.load_graph()
            assert loaded is not None
            assert "b1" in loaded.blocks
            assert "b2" in loaded.blocks
            assert len(loaded.connections) == 1
            assert loaded.connections[0].source_block_id == "b1"
        finally:
            session.close()

    def test_load_graph_returns_none_when_empty(self, tmp_root):
        session = ProjectSession.create_new("NoGraph", working_dir_root=tmp_root)
        try:
            assert session.load_graph() is None
        finally:
            session.close()

    def test_graph_persists_across_sessions(self, tmp_root):
        """save_graph marks dirty but doesn't auto-commit; save() commits."""
        session = ProjectSession.create_new("GraphPersist", working_dir_root=tmp_root)
        wd = session.working_dir
        graph = _make_graph()
        session.save_graph(graph)
        session.save()
        session.close()

        session2 = ProjectSession.open_db(wd)
        try:
            loaded = session2.load_graph()
            assert loaded is not None
            assert len(loaded.blocks) == 2
            assert len(loaded.connections) == 1
        finally:
            session2.close()

    def test_save_graph_marks_dirty(self, tmp_root):
        """save_graph should mark dirty but NOT auto-commit."""
        session = ProjectSession.create_new("GraphDirty", working_dir_root=tmp_root)
        try:
            assert session.is_dirty() is False
            graph = _make_graph()
            session.save_graph(graph)
            assert session.is_dirty() is True
        finally:
            session.close()

    def test_graph_block_settings_round_trip(self, tmp_root):
        session = ProjectSession.create_new("Settings", working_dir_root=tmp_root)
        try:
            graph = _make_graph()
            session.save_graph(graph)
            loaded = session.load_graph()
            assert loaded.blocks["b1"].settings.entries == {"file": "test.wav"}
            assert loaded.blocks["b2"].settings.entries == {"threshold": 0.3}
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Autosave
# ---------------------------------------------------------------------------


class TestSessionAutosave:
    def test_autosave_commits_dirty(self, tmp_root):
        session = ProjectSession.create_new("Autosave", working_dir_root=tmp_root)
        try:
            session.dirty_tracker.mark_dirty()
            assert session.is_dirty() is True

            # Start autosave with very short interval
            session.start_autosave(interval_seconds=0.1)
            time.sleep(0.3)

            assert session.is_dirty() is False
        finally:
            session.close()

    def test_autosave_does_not_fire_when_clean(self, tmp_root):
        session = ProjectSession.create_new("Clean", working_dir_root=tmp_root)
        try:
            assert session.is_dirty() is False
            session.start_autosave(interval_seconds=0.1)
            time.sleep(0.3)
            # Should still be clean
            assert session.is_dirty() is False
        finally:
            session.close()

    def test_stop_autosave(self, tmp_root):
        session = ProjectSession.create_new("Stop", working_dir_root=tmp_root)
        try:
            session.start_autosave(interval_seconds=0.1)
            session.stop_autosave()
            session.dirty_tracker.mark_dirty()
            time.sleep(0.3)
            # Autosave was stopped, so dirty should remain
            assert session.is_dirty() is True
        finally:
            session.close()

    def test_autosave_stops_on_close(self, tmp_root):
        session = ProjectSession.create_new("AutoClose", working_dir_root=tmp_root)
        session.start_autosave(interval_seconds=0.1)
        session.close()
        # Should not raise — timer is cancelled
        assert session._autosave_timer is None


# ---------------------------------------------------------------------------
# Close and context manager
# ---------------------------------------------------------------------------


class TestSessionClose:
    def test_close_sets_closed(self, tmp_root):
        session = ProjectSession.create_new("Close", working_dir_root=tmp_root)
        session.close()
        assert session._closed is True

    def test_operations_after_close_raise(self, tmp_root):
        session = ProjectSession.create_new("Closed", working_dir_root=tmp_root)
        session.close()

        with pytest.raises(RuntimeError, match="closed"):
            session.save()

        with pytest.raises(RuntimeError, match="closed"):
            _ = session.projects

        with pytest.raises(RuntimeError, match="closed"):
            _ = session.songs

        with pytest.raises(RuntimeError, match="closed"):
            session.save_graph(_make_graph())

        with pytest.raises(RuntimeError, match="closed"):
            session.load_graph()

        with pytest.raises(RuntimeError, match="closed"):
            session.start_autosave()

    def test_double_close_is_safe(self, tmp_root):
        session = ProjectSession.create_new("DoubleClose", working_dir_root=tmp_root)
        session.close()
        session.close()  # Should not raise

    def test_context_manager(self, tmp_root):
        with ProjectSession.create_new("Context", working_dir_root=tmp_root) as session:
            assert session.project.name == "Context"
            session.dirty_tracker.mark_dirty()
        assert session._closed is True

    def test_context_manager_closes_on_exception(self, tmp_root):
        with pytest.raises(ValueError):
            with ProjectSession.create_new("Error", working_dir_root=tmp_root) as session:
                raise ValueError("boom")
        assert session._closed is True

    def test_working_dir_survives_close(self, tmp_root):
        session = ProjectSession.create_new("Survive", working_dir_root=tmp_root)
        wd = session.working_dir
        session.close()
        # Working dir is NOT deleted — enables crash recovery
        assert wd.exists()
        assert (wd / "project.db").exists()


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    def test_check_recovery_detects_working_dir(self, tmp_path):
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "test.ez"
        ez_path.touch()

        # No working dir yet
        assert ProjectSession.check_recovery(ez_path, working_dir_root=tmp_root) is False

        # Simulate: create the working dir with a DB
        import hashlib
        digest = hashlib.sha256(str(ez_path.resolve()).encode()).hexdigest()[:16]
        wd = tmp_root / digest
        wd.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(wd / "project.db"))
        conn.row_factory = sqlite3.Row
        from echozero.persistence.schema import init_db
        init_db(conn)
        from echozero.persistence.repositories import ProjectRepository
        ProjectRepository(conn).create(Project(
            id=_uid(), name="Recovered",
            settings=ProjectSettings(), created_at=_now(), updated_at=_now(),
        ))
        conn.commit()
        conn.close()

        assert ProjectSession.check_recovery(ez_path, working_dir_root=tmp_root) is True

    def test_recover_opens_existing_working_dir(self, tmp_path):
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "recover.ez"
        ez_path.touch()

        import hashlib
        digest = hashlib.sha256(str(ez_path.resolve()).encode()).hexdigest()[:16]
        wd = tmp_root / digest
        wd.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(wd / "project.db"))
        conn.row_factory = sqlite3.Row
        from echozero.persistence.schema import init_db
        init_db(conn)
        from echozero.persistence.repositories import ProjectRepository
        ProjectRepository(conn).create(Project(
            id="recover_id", name="Recovered",
            settings=ProjectSettings(), created_at=_now(), updated_at=_now(),
        ))
        conn.commit()
        conn.close()

        session = ProjectSession.recover(ez_path, working_dir_root=tmp_root)
        try:
            assert session.project.name == "Recovered"
            assert session.project.id == "recover_id"
        finally:
            session.close()

    def test_discard_recovery_deletes_working_dir(self, tmp_path):
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "discard.ez"
        ez_path.touch()

        import hashlib
        digest = hashlib.sha256(str(ez_path.resolve()).encode()).hexdigest()[:16]
        wd = tmp_root / digest
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "project.db").touch()

        assert wd.exists()
        ProjectSession.discard_recovery(ez_path, working_dir_root=tmp_root)
        assert not wd.exists()

    def test_discard_recovery_nonexistent_is_safe(self, tmp_path):
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "ghost.ez"
        ez_path.touch()
        # Should not raise
        ProjectSession.discard_recovery(ez_path, working_dir_root=tmp_root)


# ---------------------------------------------------------------------------
# Multiple simultaneous projects
# ---------------------------------------------------------------------------


class TestMultipleProjects:
    def test_two_projects_independent(self, tmp_root):
        s1 = ProjectSession.create_new("Project A", working_dir_root=tmp_root)
        s2 = ProjectSession.create_new("Project B", working_dir_root=tmp_root)
        try:
            assert s1.working_dir != s2.working_dir
            assert s1.project.id != s2.project.id

            # Mutations in one don't affect the other
            song = _make_song(s1.project.id, title="Only in A")
            s1.songs.create(song)

            assert len(s1.songs.list_by_project(s1.project.id)) == 1
            assert len(s2.songs.list_by_project(s2.project.id)) == 0
        finally:
            s1.close()
            s2.close()

    def test_two_projects_both_save(self, tmp_root):
        s1 = ProjectSession.create_new("A", working_dir_root=tmp_root)
        s2 = ProjectSession.create_new("B", working_dir_root=tmp_root)

        s1.dirty_tracker.mark_dirty()
        s2.dirty_tracker.mark_dirty()
        s1.save()
        s2.save()

        assert s1.is_dirty() is False
        assert s2.is_dirty() is False

        s1.close()
        s2.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_close_without_save_does_not_raise(self, tmp_root):
        session = ProjectSession.create_new("NoSave", working_dir_root=tmp_root)
        session.dirty_tracker.mark_dirty()
        session.close()  # Should not raise

    def test_repo_accessors_return_correct_types(self, tmp_root):
        from echozero.persistence.repositories import (
            LayerRepository,
            PipelineConfigRepository,
            ProjectRepository,
            SongRepository,
            SongVersionRepository,
            TakeRepository,
        )
        session = ProjectSession.create_new("Types", working_dir_root=tmp_root)
        try:
            assert isinstance(session.projects, ProjectRepository)
            assert isinstance(session.songs, SongRepository)
            assert isinstance(session.song_versions, SongVersionRepository)
            assert isinstance(session.layers, LayerRepository)
            assert isinstance(session.takes, TakeRepository)
            assert isinstance(session.pipeline_configs, PipelineConfigRepository)
        finally:
            session.close()

    def test_save_graph_then_overwrite(self, tmp_root):
        session = ProjectSession.create_new("Overwrite", working_dir_root=tmp_root)
        try:
            g1 = _make_graph()
            session.save_graph(g1)

            # Overwrite with empty graph
            g2 = Graph()
            session.save_graph(g2)

            loaded = session.load_graph()
            assert loaded is not None
            assert len(loaded.blocks) == 0
            assert len(loaded.connections) == 0
        finally:
            session.close()

    def test_create_new_custom_settings(self, tmp_root):
        settings = ProjectSettings(
            sample_rate=96000, bpm=160.0,
            bpm_confidence=0.99, timecode_fps=29.97,
        )
        session = ProjectSession.create_new(
            "Custom", settings=settings, working_dir_root=tmp_root,
        )
        try:
            assert session.project.settings == settings
            got = session.projects.get(session.project.id)
            assert got.settings == settings
        finally:
            session.close()


# ---------------------------------------------------------------------------
# ProjectSession.open() test (item 19)
# ---------------------------------------------------------------------------


class TestSessionOpen:
    def test_open_via_ez_path(self, tmp_path):
        """Open a project via the ez_path-based factory."""
        import hashlib
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "test.ez"
        ez_path.touch()

        # Create working dir manually matching the hash
        digest = hashlib.sha256(str(ez_path.resolve()).encode()).hexdigest()[:16]
        wd = tmp_root / digest
        wd.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(wd / "project.db"))
        conn.row_factory = sqlite3.Row
        from echozero.persistence.schema import init_db
        from echozero.persistence.repositories import ProjectRepository
        init_db(conn)
        ProjectRepository(conn).create(Project(
            id="open_test_id", name="OpenTest",
            settings=ProjectSettings(), created_at=_now(), updated_at=_now(),
        ))
        conn.commit()
        conn.close()

        session = ProjectSession.open(ez_path, working_dir_root=tmp_root)
        try:
            assert session.project.id == "open_test_id"
            assert session.project.name == "OpenTest"
        finally:
            session.close()

    def test_open_nonexistent_ez_path_raises(self, tmp_path):
        """Open with ez_path that doesn't exist should raise."""
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "nonexistent.ez"
        # File doesn't exist at all
        with pytest.raises(FileNotFoundError):
            ProjectSession.open(ez_path, working_dir_root=tmp_root)

    def test_open_invalid_ez_file_raises(self, tmp_path):
        """Open with an invalid (non-ZIP) .ez file should raise."""
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "bad.ez"
        ez_path.write_bytes(b"not a zip file")
        with pytest.raises(Exception):  # BadZipFile or ValueError
            ProjectSession.open(ez_path, working_dir_root=tmp_root)


# ---------------------------------------------------------------------------
# DirtyTracker.last_saved_at test (item 16)
# ---------------------------------------------------------------------------


class TestDirtyTrackerLastSavedAt:
    def test_last_saved_at_starts_none(self):
        tracker = DirtyTracker()
        assert tracker.last_saved_at is None

    def test_last_saved_at_set_after_clear(self):
        tracker = DirtyTracker()
        tracker.mark_dirty()
        tracker.clear()
        assert tracker.last_saved_at is not None

    def test_last_saved_at_is_utc(self):
        tracker = DirtyTracker()
        tracker.mark_dirty()
        tracker.clear()
        ts = tracker.last_saved_at
        assert ts is not None
        assert ts.tzinfo is not None

    def test_last_saved_at_updates_on_each_clear(self):
        import time as _time
        tracker = DirtyTracker()
        tracker.mark_dirty()
        tracker.clear()
        first = tracker.last_saved_at
        _time.sleep(0.01)
        tracker.mark_dirty()
        tracker.clear()
        second = tracker.last_saved_at
        assert second > first

    def test_last_saved_at_via_session_save(self, tmp_root):
        session = ProjectSession.create_new("SavedAt", working_dir_root=tmp_root)
        try:
            assert session.dirty_tracker.last_saved_at is None
            session.dirty_tracker.mark_dirty()
            session.save()
            assert session.dirty_tracker.last_saved_at is not None
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Concurrent thread test (item 20)
# ---------------------------------------------------------------------------


class TestConcurrentThreadSafety:
    def test_autosave_and_manual_save_race(self, tmp_root):
        """Autosave and manual save running simultaneously must not error."""
        import threading

        session = ProjectSession.create_new("ThreadSafe", working_dir_root=tmp_root)
        errors = []

        def manual_save_loop():
            for _ in range(20):
                try:
                    session.dirty_tracker.mark_dirty("manual")
                    session.save()
                except Exception as exc:
                    errors.append(exc)

        try:
            session.start_autosave(interval_seconds=0.01)

            threads = [threading.Thread(target=manual_save_loop) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)

            assert errors == [], f"Thread errors: {errors}"
        finally:
            session.close()

    def test_concurrent_repo_writes(self, tmp_root):
        """Multiple threads writing via repos simultaneously must not corrupt."""
        import threading

        session = ProjectSession.create_new("ConcWrite", working_dir_root=tmp_root)
        errors = []

        def write_songs(n):
            for i in range(5):
                try:
                    song = _make_song(
                        session.project.id,
                        title=f"Thread{n}_Song{i}",
                    )
                    with session.transaction():
                        session.songs.create(song)
                except Exception as exc:
                    errors.append(exc)

        try:
            threads = [threading.Thread(target=write_songs, args=(n,)) for n in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)

            assert errors == [], f"Thread errors: {errors}"
            songs = session.songs.list_by_project(session.project.id)
            assert len(songs) == 15
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Autosave polling test (item 24)
# ---------------------------------------------------------------------------


class TestAutosavePolling:
    def test_autosave_commits_dirty_with_polling(self, tmp_root):
        """Use polling instead of fixed sleep for autosave test."""
        session = ProjectSession.create_new("PollSave", working_dir_root=tmp_root)
        try:
            session.dirty_tracker.mark_dirty()
            assert session.is_dirty() is True
            session.start_autosave(interval_seconds=0.05)

            # Poll with deadline instead of fixed sleep
            import time
            deadline = time.monotonic() + 2.0
            while session.is_dirty() and time.monotonic() < deadline:
                time.sleep(0.02)

            assert session.is_dirty() is False
        finally:
            session.close()

    def test_autosave_not_fired_when_clean_polling(self, tmp_root):
        """Autosave with clean state - polled variant."""
        session = ProjectSession.create_new("PollClean", working_dir_root=tmp_root)
        try:
            assert session.is_dirty() is False
            session.start_autosave(interval_seconds=0.05)

            import time
            time.sleep(0.2)

            assert session.is_dirty() is False
        finally:
            session.close()

    def test_stop_autosave_preserves_dirty_polling(self, tmp_root):
        """Stopping autosave should keep dirty state."""
        session = ProjectSession.create_new("PollStop", working_dir_root=tmp_root)
        try:
            session.start_autosave(interval_seconds=0.05)
            session.stop_autosave()
            session.dirty_tracker.mark_dirty()

            import time
            time.sleep(0.2)

            assert session.is_dirty() is True
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Operations-after-close for all repo accessors (item 25)
# ---------------------------------------------------------------------------


class TestOperationsAfterClose:
    def test_song_versions_after_close_raises(self, tmp_root):
        session = ProjectSession.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.song_versions

    def test_layers_after_close_raises(self, tmp_root):
        session = ProjectSession.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.layers

    def test_takes_after_close_raises(self, tmp_root):
        session = ProjectSession.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.takes

    def test_pipeline_configs_after_close_raises(self, tmp_root):
        session = ProjectSession.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.pipeline_configs

    def test_commit_after_close_raises(self, tmp_root):
        session = ProjectSession.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.commit()

    def test_transaction_after_close_raises(self, tmp_root):
        session = ProjectSession.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            with session.transaction():
                pass

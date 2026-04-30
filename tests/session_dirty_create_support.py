"""Dirty-tracker and create/open session support cases.
Exists to isolate creation and transaction coverage from save, lifecycle, and edge-case support tests.
Connects the compatibility wrapper to the bounded dirty/create slice.
"""

from tests.session_shared_support import *  # noqa: F401,F403

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
            old_value=0.3, new_value=0.5,
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
# ProjectStorage — creation and basic lifecycle
# ---------------------------------------------------------------------------


class TestSessionCreate:
    def test_create_new_creates_working_dir(self, tmp_root):
        session = ProjectStorage.create_new("Test", working_dir_root=tmp_root)
        try:
            assert session.working_dir.exists()
            assert (session.working_dir / "project.db").exists()
        finally:
            session.close()

    def test_create_new_project_entity(self, tmp_root):
        session = ProjectStorage.create_new(
            "My Show",
            settings=ProjectSettingsRecord(bpm=120.0),
            working_dir_root=tmp_root,
        )
        try:
            assert session.project.name == "My Show"
            assert session.project.settings.bpm == 120.0
            assert session.project.id is not None
        finally:
            session.close()

    def test_create_new_project_persisted(self, tmp_root):
        session = ProjectStorage.create_new("Persisted", working_dir_root=tmp_root)
        try:
            got = session.projects.get(session.project.id)
            assert got is not None
            assert got.name == "Persisted"
        finally:
            session.close()

    def test_create_new_default_settings(self, tmp_root):
        session = ProjectStorage.create_new("Defaults", working_dir_root=tmp_root)
        try:
            assert session.project.settings.sample_rate == 44100
            assert session.project.settings.bpm is None
        finally:
            session.close()

    def test_create_new_starts_clean(self, tmp_root):
        session = ProjectStorage.create_new("Clean", working_dir_root=tmp_root)
        try:
            assert session.is_dirty() is False
        finally:
            session.close()

    def test_wal_mode_enabled(self, tmp_root):
        """Verify WAL mode is set on new sessions."""
        session = ProjectStorage.create_new("WAL", working_dir_root=tmp_root)
        try:
            mode = session.db.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"
        finally:
            session.close()

    def test_row_factory_is_sqlite_row(self, tmp_root):
        """Verify row_factory is set to sqlite3.Row."""
        session = ProjectStorage.create_new("RowFactory", working_dir_root=tmp_root)
        try:
            assert session.db.row_factory is sqlite3.Row
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Open from working directory
# ---------------------------------------------------------------------------


class TestSessionOpenDb:
    def test_open_db_loads_project(self, tmp_root):
        session1 = ProjectStorage.create_new("Reopen", working_dir_root=tmp_root)
        wd = session1.working_dir
        pid = session1.project.id
        session1.close()

        session2 = ProjectStorage.open_db(wd)
        try:
            assert session2.project.id == pid
            assert session2.project.name == "Reopen"
        finally:
            session2.close()

    def test_open_db_missing_dir_raises(self, tmp_root):
        with pytest.raises(FileNotFoundError):
            ProjectStorage.open_db(tmp_root / "nonexistent")

    def test_open_db_empty_db_raises(self, tmp_root):
        wd = tmp_root / "empty"
        wd.mkdir(parents=True)
        conn = sqlite3.connect(str(wd / "project.db"))
        from echozero.persistence.schema import init_db
        init_db(conn)
        conn.close()

        with pytest.raises(ValueError, match="No project found"):
            ProjectStorage.open_db(wd)

    def test_open_db_has_wal_mode(self, tmp_root):
        """Verify WAL mode is set when reopening an existing DB."""
        session1 = ProjectStorage.create_new("WALReopen", working_dir_root=tmp_root)
        wd = session1.working_dir
        session1.close()

        session2 = ProjectStorage.open_db(wd)
        try:
            mode = session2.db.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"
            assert session2.db.row_factory is sqlite3.Row
        finally:
            session2.close()

    def test_open_db_repairs_missing_timeline_regions_table(self, tmp_root):
        wd = tmp_root / "legacy_missing_regions"
        wd.mkdir(parents=True)
        conn = sqlite3.connect(str(wd / "project.db"))
        now = _now().isoformat()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS _meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                sample_rate INTEGER NOT NULL DEFAULT 44100,
                bpm REAL,
                bpm_confidence REAL,
                timecode_fps REAL,
                ma3_push_offset_seconds REAL NOT NULL DEFAULT -1.0,
                graph_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO projects "
            "(id, name, sample_rate, bpm, bpm_confidence, timecode_fps, "
            "ma3_push_offset_seconds, graph_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("project_legacy", "Legacy", 44100, None, None, None, -1.0, None, now, now),
        )
        conn.commit()
        conn.close()

        session = ProjectStorage.open_db(wd)
        try:
            assert session.timeline_regions.list_by_version("version_missing") == []
            tables = {
                row[0]
                for row in session.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "timeline_regions" in tables
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Transaction control
# ---------------------------------------------------------------------------


class TestSessionTransaction:
    def test_transaction_commits_on_success(self, tmp_root):
        session = ProjectStorage.create_new("TxnSuccess", working_dir_root=tmp_root)
        try:
            song = _make_song(session.project.id, title="Atomic SongRecord")
            version = _make_version(song.id)

            with session.transaction():
                session.songs.create(song)
                session.song_versions.create(version)

            # Data should be committed and readable after reopen
            wd = session.working_dir
            session.close()

            session2 = ProjectStorage.open_db(wd)
            try:
                songs = session2.songs.list_by_project(session2.project.id)
                assert len(songs) == 1
                assert songs[0].title == "Atomic SongRecord"
                versions = session2.song_versions.list_by_song(song.id)
                assert len(versions) == 1
            finally:
                session2.close()
        except Exception:
            session.close()
            raise

    def test_transaction_rolls_back_on_error(self, tmp_root):
        session = ProjectStorage.create_new("TxnRollback", working_dir_root=tmp_root)
        try:
            song = _make_song(session.project.id, title="Rolled Back")

            with pytest.raises(ValueError, match="boom"):
                with session.transaction():
                    session.songs.create(song)
                    raise ValueError("boom")

            # SongRecord should NOT be persisted after rollback
            songs = session.songs.list_by_project(session.project.id)
            assert len(songs) == 0
        finally:
            session.close()

    def test_commit_explicitly(self, tmp_root):
        session = ProjectStorage.create_new("ExplicitCommit", working_dir_root=tmp_root)
        try:
            song = _make_song(session.project.id, title="Committed")
            session.songs.create(song)
            session.commit()

            # Verify persisted across reopen
            wd = session.working_dir
            session.close()

            session2 = ProjectStorage.open_db(wd)
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
        session = ProjectStorage.create_new("ClosedCommit", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.commit()

    def test_transaction_after_close_raises(self, tmp_root):
        session = ProjectStorage.create_new("ClosedTxn", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            with session.transaction():
                pass


# ---------------------------------------------------------------------------
# Save and round-trip
# ---------------------------------------------------------------------------



__all__ = [name for name in globals() if name.startswith("Test")]

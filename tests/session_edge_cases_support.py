"""Edge-case session support cases.
Exists to keep reopen, thread-safety, polling, and post-close coverage separate from the primary session support slices.
Connects the compatibility wrapper to the bounded edge-case slice.
"""

from tests.session_shared_support import *  # noqa: F401,F403

class TestEdgeCases:
    def test_close_without_save_does_not_raise(self, tmp_root):
        session = ProjectStorage.create_new("NoSave", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("Types", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("Overwrite", working_dir_root=tmp_root)
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
        settings = ProjectSettingsRecord(
            sample_rate=96000, bpm=160.0,
            bpm_confidence=0.99, timecode_fps=29.97,
        )
        session = ProjectStorage.create_new(
            "Custom", settings=settings, working_dir_root=tmp_root,
        )
        try:
            assert session.project.settings == settings
            got = session.projects.get(session.project.id)
            assert got.settings == settings
        finally:
            session.close()


# ---------------------------------------------------------------------------
# ProjectStorage.open() test (item 19)
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
        ProjectRepository(conn).create(ProjectRecord(
            id="open_test_id", name="OpenTest",
            settings=ProjectSettingsRecord(), created_at=_now(), updated_at=_now(),
        ))
        conn.commit()
        conn.close()

        session = ProjectStorage.open(ez_path, working_dir_root=tmp_root)
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
            ProjectStorage.open(ez_path, working_dir_root=tmp_root)

    def test_open_invalid_ez_file_raises(self, tmp_path):
        """Open with an invalid (non-ZIP) .ez file should raise."""
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "bad.ez"
        ez_path.write_bytes(b"not a zip file")
        with pytest.raises(Exception):  # BadZipFile or ValueError
            ProjectStorage.open(ez_path, working_dir_root=tmp_root)


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
        session = ProjectStorage.create_new("SavedAt", working_dir_root=tmp_root)
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

        session = ProjectStorage.create_new("ThreadSafe", working_dir_root=tmp_root)
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

        session = ProjectStorage.create_new("ConcWrite", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("PollSave", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("PollClean", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("PollStop", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.song_versions

    def test_layers_after_close_raises(self, tmp_root):
        session = ProjectStorage.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.layers

    def test_takes_after_close_raises(self, tmp_root):
        session = ProjectStorage.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.takes

    def test_pipeline_configs_after_close_raises(self, tmp_root):
        session = ProjectStorage.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.pipeline_configs

    def test_commit_after_close_raises(self, tmp_root):
        session = ProjectStorage.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.commit()

    def test_transaction_after_close_raises(self, tmp_root):
        session = ProjectStorage.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            with session.transaction():
                pass


__all__ = [name for name in globals() if name.startswith("Test")]

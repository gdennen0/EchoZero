"""Lifecycle session support cases.
Exists to isolate autosave, close, crash-recovery, and multi-project behavior from other session support tests.
Connects the compatibility wrapper to the bounded lifecycle slice.
"""

from tests.session_shared_support import *  # noqa: F401,F403

class TestSessionAutosave:
    def test_autosave_commits_dirty(self, tmp_root):
        session = ProjectStorage.create_new("Autosave", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("Clean", working_dir_root=tmp_root)
        try:
            assert session.is_dirty() is False
            session.start_autosave(interval_seconds=0.1)
            time.sleep(0.3)
            # Should still be clean
            assert session.is_dirty() is False
        finally:
            session.close()

    def test_stop_autosave(self, tmp_root):
        session = ProjectStorage.create_new("Stop", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("AutoClose", working_dir_root=tmp_root)
        session.start_autosave(interval_seconds=0.1)
        session.close()
        # Should not raise — timer is cancelled
        assert session._autosave_timer is None


# ---------------------------------------------------------------------------
# Close and context manager
# ---------------------------------------------------------------------------


class TestSessionClose:
    def test_close_sets_closed(self, tmp_root):
        session = ProjectStorage.create_new("Close", working_dir_root=tmp_root)
        session.close()
        assert session._closed is True

    def test_operations_after_close_raise(self, tmp_root):
        session = ProjectStorage.create_new("Closed", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("DoubleClose", working_dir_root=tmp_root)
        session.close()
        session.close()  # Should not raise

    def test_context_manager(self, tmp_root):
        with ProjectStorage.create_new("Context", working_dir_root=tmp_root) as session:
            assert session.project.name == "Context"
            session.dirty_tracker.mark_dirty()
        assert session._closed is True

    def test_context_manager_closes_on_exception(self, tmp_root):
        with pytest.raises(ValueError):
            with ProjectStorage.create_new("Error", working_dir_root=tmp_root) as session:
                raise ValueError("boom")
        assert session._closed is True

    def test_working_dir_survives_close(self, tmp_root):
        session = ProjectStorage.create_new("Survive", working_dir_root=tmp_root)
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
        assert ProjectStorage.check_recovery(ez_path, working_dir_root=tmp_root) is False

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
        ProjectRepository(conn).create(ProjectRecord(
            id=_uid(), name="Recovered",
            settings=ProjectSettingsRecord(), created_at=_now(), updated_at=_now(),
        ))
        conn.commit()
        conn.close()

        assert ProjectStorage.check_recovery(ez_path, working_dir_root=tmp_root) is True

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
        ProjectRepository(conn).create(ProjectRecord(
            id="recover_id", name="Recovered",
            settings=ProjectSettingsRecord(), created_at=_now(), updated_at=_now(),
        ))
        conn.commit()
        conn.close()

        session = ProjectStorage.recover(ez_path, working_dir_root=tmp_root)
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
        ProjectStorage.discard_recovery(ez_path, working_dir_root=tmp_root)
        assert not wd.exists()

    def test_discard_recovery_nonexistent_is_safe(self, tmp_path):
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "ghost.ez"
        ez_path.touch()
        # Should not raise
        ProjectStorage.discard_recovery(ez_path, working_dir_root=tmp_root)


# ---------------------------------------------------------------------------
# Multiple simultaneous projects
# ---------------------------------------------------------------------------


class TestMultipleProjects:
    def test_two_projects_independent(self, tmp_root):
        s1 = ProjectStorage.create_new("ProjectRecord A", working_dir_root=tmp_root)
        s2 = ProjectStorage.create_new("ProjectRecord B", working_dir_root=tmp_root)
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
        s1 = ProjectStorage.create_new("A", working_dir_root=tmp_root)
        s2 = ProjectStorage.create_new("B", working_dir_root=tmp_root)

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



__all__ = [name for name in globals() if name.startswith("Test")]

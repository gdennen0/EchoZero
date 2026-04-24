"""Save and graph session support cases.
Exists to keep save-path and graph round-trip coverage separate from lifecycle and edge-case support tests.
Connects the compatibility wrapper to the bounded save/graph slice.
"""

from tests.session_shared_support import *  # noqa: F401,F403

class TestSessionSave:
    def test_save_clears_dirty(self, tmp_root):
        session = ProjectStorage.create_new("Dirty", working_dir_root=tmp_root)
        try:
            session.dirty_tracker.mark_dirty()
            assert session.is_dirty() is True
            session.save()
            assert session.is_dirty() is False
        finally:
            session.close()

    def test_save_as_creates_ez_file(self, tmp_root, tmp_path):
        """save_as() creates a .ez archive file."""
        session = ProjectStorage.create_new("SaveAs", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("DirtySaveAs", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("Closed", working_dir_root=tmp_root)
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.save_as(tmp_path / "output.ez")

    def test_full_hierarchy_round_trip(self, tmp_root):
        """Create -> add songs/versions/layers/takes -> save -> reopen -> verify."""
        session = ProjectStorage.create_new(
            "Tour 2026",
            settings=ProjectSettingsRecord(sample_rate=48000, bpm=128.0),
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
        session2 = ProjectStorage.open_db(wd)
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
        session = ProjectStorage.create_new(
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
        session = ProjectStorage.create_new(
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
        session = ProjectStorage.create_new("Manual", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("Graphed", working_dir_root=tmp_root)
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
        session = ProjectStorage.create_new("NoGraph", working_dir_root=tmp_root)
        try:
            assert session.load_graph() is None
        finally:
            session.close()

    def test_graph_persists_across_sessions(self, tmp_root):
        """save_graph marks dirty but doesn't auto-commit; save() commits."""
        session = ProjectStorage.create_new("GraphPersist", working_dir_root=tmp_root)
        wd = session.working_dir
        graph = _make_graph()
        session.save_graph(graph)
        session.save()
        session.close()

        session2 = ProjectStorage.open_db(wd)
        try:
            loaded = session2.load_graph()
            assert loaded is not None
            assert len(loaded.blocks) == 2
            assert len(loaded.connections) == 1
        finally:
            session2.close()

    def test_save_graph_marks_dirty(self, tmp_root):
        """save_graph should mark dirty but NOT auto-commit."""
        session = ProjectStorage.create_new("GraphDirty", working_dir_root=tmp_root)
        try:
            assert session.is_dirty() is False
            graph = _make_graph()
            session.save_graph(graph)
            assert session.is_dirty() is True
        finally:
            session.close()

    def test_graph_block_settings_round_trip(self, tmp_root):
        session = ProjectStorage.create_new("Settings", working_dir_root=tmp_root)
        try:
            graph = _make_graph()
            session.save_graph(graph)
            loaded = session.load_graph()
            assert loaded.blocks["b1"].settings == {"file": "test.wav"}
            assert loaded.blocks["b2"].settings == {"threshold": 0.3}
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Autosave
# ---------------------------------------------------------------------------



__all__ = [name for name in globals() if name.startswith("Test")]

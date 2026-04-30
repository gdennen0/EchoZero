"""Round-trip persistence support cases.
Exists to isolate reload and update-path coverage from core and integrity persistence support tests.
Connects the compatibility wrapper to the bounded persistence round-trip slice.
"""

from tests.persistence_shared_support import *  # noqa: F401,F403

class TestRoundTrip:
    def test_full_hierarchy_round_trip(self, conn):
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        vr = SongVersionRepository(conn)
        lr = LayerRepository(conn)
        tr = TakeRepository(conn)

        # Create hierarchy
        project = _make_project(
            name="Tour 2026",
            settings=ProjectSettingsRecord(sample_rate=48000, bpm=128.0),
        )
        pr.create(project)

        song = _make_song(project.id, title="Opening Act", artist="The Band")
        sr.create(song)

        version = _make_version(song.id, label="Final Mix", duration_seconds=312.5)
        vr.create(version)

        layer = _make_layer(
            version.id, name="Onsets", layer_type="analysis",
            color="#FF5500", source_pipeline={"name": "onset_detector", "version": "2.0"},
        )
        lr.create(layer)

        source = TakeSource(
            block_id="blk_onset", block_type="onset_detector",
            settings_snapshot={"threshold": 0.25, "min_gap": 0.1},
            run_id="run_001",
        )
        take = _make_take(
            is_main=True, label="Initial Analysis", source=source,
            notes="First pass with default settings",
        )
        tr.create(layer.id, take)
        conn.commit()

        # Read everything back
        got_p = pr.get(project.id)
        assert got_p.name == "Tour 2026"
        assert got_p.settings.bpm == 128.0

        got_s = sr.list_by_project(project.id)
        assert len(got_s) == 1
        assert got_s[0].title == "Opening Act"

        got_v = vr.list_by_song(song.id)
        assert len(got_v) == 1
        assert got_v[0].duration_seconds == 312.5

        got_l = lr.list_by_version(version.id)
        assert len(got_l) == 1
        assert got_l[0].color == "#FF5500"
        assert got_l[0].source_pipeline["name"] == "onset_detector"

        got_t = tr.get_main(layer.id)
        assert got_t.label == "Initial Analysis"
        assert got_t.source.block_type == "onset_detector"
        assert got_t.notes == "First pass with default settings"
        assert isinstance(got_t.data, EventData)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_project_no_songs(self, conn):
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        p = _make_project()
        pr.create(p)
        conn.commit()
        assert sr.list_by_project(p.id) == []

    def test_duplicate_project_id_raises(self, conn):
        pr = ProjectRepository(conn)
        p = _make_project()
        pr.create(p)
        conn.commit()
        with pytest.raises(PersistenceError):
            pr.create(p)

    def test_song_with_missing_project_fk_raises(self, conn):
        sr = SongRepository(conn)
        s = _make_song("nonexistent_project_id")
        with pytest.raises(PersistenceError):
            sr.create(s)

    def test_version_with_missing_song_fk_raises(self, conn):
        vr = SongVersionRepository(conn)
        v = _make_version("nonexistent_song_id")
        with pytest.raises(PersistenceError):
            vr.create(v)

    def test_layer_with_missing_version_fk_raises(self, conn):
        lr = LayerRepository(conn)
        layer = _make_layer("nonexistent_version_id")
        with pytest.raises(PersistenceError):
            lr.create(layer)

    def test_take_with_missing_layer_fk_raises(self, conn):
        tr = TakeRepository(conn)
        t = _make_take()
        with pytest.raises(PersistenceError):
            tr.create("nonexistent_layer_id", t)

    def test_duplicate_song_id_raises(self, conn):
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        p = _make_project()
        pr.create(p)
        conn.commit()
        s = _make_song(p.id)
        sr.create(s)
        conn.commit()
        with pytest.raises(PersistenceError):
            sr.create(s)

    def test_project_with_null_optional_settings(self, conn):
        pr = ProjectRepository(conn)
        p = _make_project(settings=ProjectSettingsRecord())
        pr.create(p)
        conn.commit()
        got = pr.get(p.id)
        assert got.settings.bpm is None
        assert got.settings.bpm_confidence is None
        assert got.settings.timecode_fps is None
        assert got.settings.ma3_push_offset_seconds == pytest.approx(-1.0)

    def test_layer_with_all_optional_fields_null(self, conn):
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        vr = SongVersionRepository(conn)
        lr = LayerRepository(conn)

        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        v = _make_version(s.id)
        vr.create(v)
        conn.commit()
        layer = _make_layer(
            v.id, color=None, parent_layer_id=None, source_pipeline=None,
        )
        lr.create(layer)
        conn.commit()
        got = lr.get(layer.id)
        assert got.color is None
        assert got.parent_layer_id is None
        assert got.source_pipeline is None


# ---------------------------------------------------------------------------
# SongVersionRepository.update (item 14)
# ---------------------------------------------------------------------------


class TestSongVersionUpdate:
    def _setup(self, conn) -> tuple[SongVersionRepository, str]:
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        conn.commit()
        return SongVersionRepository(conn), s.id

    def test_update_label(self, conn):
        vr, song_id = self._setup(conn)
        v = _make_version(song_id, label="Original")
        vr.create(v)
        conn.commit()
        updated = replace(v, label="Remastered")
        vr.update(updated)
        conn.commit()
        got = vr.get(v.id)
        assert got.label == "Remastered"

    def test_update_audio_file(self, conn):
        vr, song_id = self._setup(conn)
        v = _make_version(song_id, audio_file="audio/old.wav")
        vr.create(v)
        conn.commit()
        updated = replace(v, audio_file="audio/new.wav")
        vr.update(updated)
        conn.commit()
        got = vr.get(v.id)
        assert got.audio_file == "audio/new.wav"

    def test_update_duration(self, conn):
        vr, song_id = self._setup(conn)
        v = _make_version(song_id, duration_seconds=100.0)
        vr.create(v)
        conn.commit()
        updated = replace(v, duration_seconds=250.5)
        vr.update(updated)
        conn.commit()
        got = vr.get(v.id)
        assert got.duration_seconds == 250.5

    def test_update_sample_rate(self, conn):
        vr, song_id = self._setup(conn)
        v = _make_version(song_id, original_sample_rate=44100)
        vr.create(v)
        conn.commit()
        updated = replace(v, original_sample_rate=96000)
        vr.update(updated)
        conn.commit()
        got = vr.get(v.id)
        assert got.original_sample_rate == 96000

    def test_update_audio_hash(self, conn):
        vr, song_id = self._setup(conn)
        v = _make_version(song_id, audio_hash="old_hash")
        vr.create(v)
        conn.commit()
        updated = replace(v, audio_hash="new_hash_abc")
        vr.update(updated)
        conn.commit()
        got = vr.get(v.id)
        assert got.audio_hash == "new_hash_abc"

    def test_update_preserves_created_at(self, conn):
        vr, song_id = self._setup(conn)
        v = _make_version(song_id)
        vr.create(v)
        conn.commit()
        updated = replace(v, label="Changed")
        vr.update(updated)
        conn.commit()
        got = vr.get(v.id)
        assert got.created_at == v.created_at

    def test_update_round_trip(self, conn):
        vr, song_id = self._setup(conn)
        v = _make_version(song_id, label="V1", audio_file="a.wav",
                          duration_seconds=200.0, original_sample_rate=48000,
                          audio_hash="hash1")
        vr.create(v)
        conn.commit()
        updated = replace(v, label="V2", audio_file="b.wav",
                          duration_seconds=300.0, original_sample_rate=96000,
                          audio_hash="hash2")
        vr.update(updated)
        conn.commit()
        got = vr.get(v.id)
        assert got.label == "V2"
        assert got.audio_file == "b.wav"
        assert got.duration_seconds == 300.0
        assert got.original_sample_rate == 96000
        assert got.audio_hash == "hash2"


# ---------------------------------------------------------------------------
# Take.is_archived round-trip (item 15)
# ---------------------------------------------------------------------------


class TestTakeIsArchived:
    def _setup(self, conn) -> tuple[TakeRepository, str]:
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        vr = SongVersionRepository(conn)
        lr = LayerRepository(conn)
        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        v = _make_version(s.id)
        vr.create(v)
        layer = _make_layer(v.id)
        lr.create(layer)
        conn.commit()
        return TakeRepository(conn), layer.id

    def test_default_not_archived(self, conn):
        tr, lid = self._setup(conn)
        t = _make_take(is_main=True)
        tr.create(lid, t)
        conn.commit()
        got = tr.get(t.id)
        assert got.is_archived is False

    def test_create_archived(self, conn):
        tr, lid = self._setup(conn)
        t = _make_take(is_main=True, is_archived=True)
        tr.create(lid, t)
        conn.commit()
        got = tr.get(t.id)
        assert got.is_archived is True

    def test_update_to_archived(self, conn):
        tr, lid = self._setup(conn)
        t = _make_take(is_main=True)
        tr.create(lid, t)
        conn.commit()
        updated = replace(t, is_archived=True)
        tr.update(updated)
        conn.commit()
        got = tr.get(t.id)
        assert got.is_archived is True

    def test_update_to_unarchived(self, conn):
        tr, lid = self._setup(conn)
        t = _make_take(is_main=True, is_archived=True)
        tr.create(lid, t)
        conn.commit()
        updated = replace(t, is_archived=False)
        tr.update(updated)
        conn.commit()
        got = tr.get(t.id)
        assert got.is_archived is False

    def test_archived_round_trip(self, conn):
        tr, lid = self._setup(conn)
        t = _make_take(is_main=True, is_archived=True, label="Archived Take")
        tr.create(lid, t)
        conn.commit()
        got = tr.get(t.id)
        assert got.is_archived is True
        assert got.label == "Archived Take"


# ---------------------------------------------------------------------------
# Duplicate ID tests (item 18)
# ---------------------------------------------------------------------------



__all__ = [name for name in globals() if name.startswith("Test")]

"""Core persistence repository support cases.
Exists to isolate schema and base CRUD coverage from layering, round-trip, and integrity support tests.
Connects the compatibility wrapper to the bounded core persistence slice.
"""

from tests.persistence_shared_support import *  # noqa: F401,F403

class TestSchema:
    def test_init_db_creates_tables(self, conn):
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "_meta" in tables
        assert "projects" in tables
        assert "songs" in tables
        assert "song_versions" in tables
        assert "layers" in tables
        assert "takes" in tables
        assert "timeline_regions" in tables
        assert "pipeline_configs" in tables

    def test_schema_version_is_set(self, conn):
        assert get_schema_version(conn) == SCHEMA_VERSION

    def test_set_and_get_schema_version(self, conn):
        set_schema_version(conn, 42)
        conn.commit()
        assert get_schema_version(conn) == 42

    def test_init_db_is_idempotent(self, conn):
        init_db(conn)
        assert get_schema_version(conn) == SCHEMA_VERSION

    def test_foreign_keys_enabled(self, conn):
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_projects_table_has_graph_json_column(self, conn):
        """Verify graph_json is in the DDL, not added via ALTER TABLE."""
        columns = {
            row['name']
            for row in conn.execute("PRAGMA table_info(projects)").fetchall()
        }
        assert 'graph_json' in columns


# ---------------------------------------------------------------------------
# BaseRepository contract
# ---------------------------------------------------------------------------


class TestBaseRepository:
    def test_base_repository_is_abstract(self):
        """Cannot instantiate BaseRepository directly."""
        with pytest.raises(TypeError):
            BaseRepository(None)

    def test_subclass_must_implement_from_row(self, conn):
        """Subclass without _from_row raises TypeError."""
        with pytest.raises(TypeError):
            class BadRepo(BaseRepository[str]):
                pass
            BadRepo(conn)

    def test_subclass_with_from_row_works(self, conn):
        """Concrete subclass can use _execute, _fetchone, _fetchall."""
        class StringRepo(BaseRepository[str]):
            def _from_row(self, row: sqlite3.Row) -> str:
                return str(row[0])

        repo = StringRepo(conn)
        # _execute works
        repo._execute(
            "INSERT INTO projects "
            "(id, name, sample_rate, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("test_id", "Test", 44100, _now().isoformat(), _now().isoformat()),
        )
        conn.commit()

        # _fetchone works
        row = repo._fetchone("SELECT id FROM projects WHERE id = ?", ("test_id",))
        assert row is not None
        assert repo._from_row(row) == "test_id"

        # _fetchall works
        rows = repo._fetchall("SELECT id FROM projects")
        assert len(rows) == 1

    def test_all_repos_extend_base_repository(self):
        """Every repository class extends BaseRepository."""
        assert issubclass(ProjectRepository, BaseRepository)
        assert issubclass(SongRepository, BaseRepository)
        assert issubclass(SongVersionRepository, BaseRepository)
        assert issubclass(LayerRepository, BaseRepository)
        assert issubclass(TakeRepository, BaseRepository)
        assert issubclass(PipelineConfigRepository, BaseRepository)
        assert issubclass(TimelineRegionRepository, BaseRepository)


# ---------------------------------------------------------------------------
# ProjectRecord CRUD
# ---------------------------------------------------------------------------


class TestProjectRepository:
    def test_create_and_get(self, conn):
        repo = ProjectRepository(conn)
        p = _make_project(name="My Show", settings=ProjectSettingsRecord(bpm=120.0))
        repo.create(p)
        conn.commit()
        got = repo.get(p.id)
        assert got is not None
        assert got.id == p.id
        assert got.name == "My Show"
        assert got.settings.sample_rate == 44100
        assert got.settings.bpm == 120.0

    def test_list_projects(self, conn):
        repo = ProjectRepository(conn)
        repo.create(_make_project(name="Beta"))
        repo.create(_make_project(name="Alpha"))
        conn.commit()
        projects = repo.list()
        assert len(projects) == 2
        assert projects[0].name == "Alpha"
        assert projects[1].name == "Beta"

    def test_update_project(self, conn):
        repo = ProjectRepository(conn)
        p = _make_project()
        repo.create(p)
        conn.commit()
        updated = replace(p, name="Updated", updated_at=_now())
        repo.update(updated)
        conn.commit()
        got = repo.get(p.id)
        assert got.name == "Updated"

    def test_delete_project(self, conn):
        repo = ProjectRepository(conn)
        p = _make_project()
        repo.create(p)
        conn.commit()
        repo.delete(p.id)
        conn.commit()
        assert repo.get(p.id) is None

    def test_get_nonexistent_returns_none(self, conn):
        repo = ProjectRepository(conn)
        assert repo.get("nonexistent") is None

    def test_datetime_round_trip(self, conn):
        repo = ProjectRepository(conn)
        now = _now()
        p = _make_project(created_at=now, updated_at=now)
        repo.create(p)
        conn.commit()
        got = repo.get(p.id)
        assert got.created_at == now
        assert got.updated_at == now

    def test_settings_round_trip(self, conn):
        repo = ProjectRepository(conn)
        s = ProjectSettingsRecord(sample_rate=48000, bpm=140.5, bpm_confidence=0.95, timecode_fps=29.97)
        p = _make_project(settings=s)
        repo.create(p)
        conn.commit()
        got = repo.get(p.id)
        assert got.settings == s


# ---------------------------------------------------------------------------
# SongRecord CRUD
# ---------------------------------------------------------------------------


class TestSongRepository:
    def _setup(self, conn) -> tuple[ProjectRepository, SongRepository, ProjectRecord]:
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        p = _make_project()
        pr.create(p)
        conn.commit()
        return pr, sr, p

    def test_create_and_get(self, conn):
        _, sr, p = self._setup(conn)
        s = _make_song(p.id, title="Don't Stop")
        sr.create(s)
        conn.commit()
        got = sr.get(s.id)
        assert got is not None
        assert got.title == "Don't Stop"
        assert got.project_id == p.id

    def test_list_by_project_ordered(self, conn):
        _, sr, p = self._setup(conn)
        sr.create(_make_song(p.id, title="C", order=2))
        sr.create(_make_song(p.id, title="A", order=0))
        sr.create(_make_song(p.id, title="B", order=1))
        conn.commit()
        songs = sr.list_by_project(p.id)
        assert [s.title for s in songs] == ["A", "B", "C"]

    def test_update_song(self, conn):
        _, sr, p = self._setup(conn)
        s = _make_song(p.id)
        sr.create(s)
        conn.commit()
        updated = replace(s, title="New Title", artist="New Artist")
        sr.update(updated)
        conn.commit()
        got = sr.get(s.id)
        assert got.title == "New Title"
        assert got.artist == "New Artist"

    def test_delete_song(self, conn):
        _, sr, p = self._setup(conn)
        s = _make_song(p.id)
        sr.create(s)
        conn.commit()
        sr.delete(s.id)
        conn.commit()
        assert sr.get(s.id) is None

    def test_reorder_songs(self, conn):
        _, sr, p = self._setup(conn)
        s1 = _make_song(p.id, title="First", order=0)
        s2 = _make_song(p.id, title="Second", order=1)
        s3 = _make_song(p.id, title="Third", order=2)
        sr.create(s1)
        sr.create(s2)
        sr.create(s3)
        conn.commit()
        sr.reorder(p.id, [s3.id, s1.id, s2.id])
        conn.commit()
        songs = sr.list_by_project(p.id)
        assert [s.title for s in songs] == ["Third", "First", "Second"]


# ---------------------------------------------------------------------------
# TimelineRegionRecord CRUD
# ---------------------------------------------------------------------------


class TestTimelineRegionRepository:
    def _setup(self, conn) -> tuple[TimelineRegionRepository, str]:
        project_repo = ProjectRepository(conn)
        song_repo = SongRepository(conn)
        version_repo = SongVersionRepository(conn)
        project = _make_project()
        project_repo.create(project)
        song = _make_song(project.id)
        song_repo.create(song)
        version = _make_version(song.id)
        version_repo.create(version)
        conn.commit()
        return TimelineRegionRepository(conn), version.id

    def test_create_list_update_delete_and_reorder(self, conn):
        region_repo, version_id = self._setup(conn)
        region_a = _make_timeline_region(
            version_id,
            id="region_a",
            label="Intro",
            start_seconds=0.0,
            end_seconds=1.0,
            order_index=1,
        )
        region_b = _make_timeline_region(
            version_id,
            id="region_b",
            label="Verse",
            start_seconds=1.0,
            end_seconds=2.0,
            color="#ddeeff",
            order_index=0,
            kind="song",
        )

        region_repo.create(region_a)
        region_repo.create(region_b)
        conn.commit()

        listed = region_repo.list_by_version(version_id)
        assert [region.id for region in listed] == ["region_b", "region_a"]
        assert listed[0].color == "#ddeeff"
        assert listed[0].kind == "song"

        updated = replace(
            region_a,
            label="Intro Updated",
            end_seconds=1.5,
            order_index=2,
        )
        region_repo.update(updated)
        conn.commit()

        got = region_repo.get("region_a")
        assert got is not None
        assert got.label == "Intro Updated"
        assert got.end_seconds == 1.5
        assert got.order_index == 2

        region_repo.reorder(version_id, ["region_a", "region_b"])
        conn.commit()
        reordered = region_repo.list_by_version(version_id)
        assert [region.id for region in reordered] == ["region_a", "region_b"]

        region_repo.delete("region_b")
        conn.commit()
        remaining = region_repo.list_by_version(version_id)
        assert [region.id for region in remaining] == ["region_a"]


# ---------------------------------------------------------------------------
# SongVersionRecord CRUD
# ---------------------------------------------------------------------------


class TestSongVersionRepository:
    def _setup(self, conn) -> tuple[SongVersionRepository, str]:
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        conn.commit()
        return SongVersionRepository(conn), s.id

    def test_create_and_get(self, conn):
        vr, song_id = self._setup(conn)
        v = _make_version(song_id, label="Live Mix")
        vr.create(v)
        conn.commit()
        got = vr.get(v.id)
        assert got is not None
        assert got.label == "Live Mix"
        assert got.song_id == song_id

    def test_list_by_song(self, conn):
        vr, song_id = self._setup(conn)
        vr.create(_make_version(song_id, label="V1"))
        vr.create(_make_version(song_id, label="V2"))
        conn.commit()
        versions = vr.list_by_song(song_id)
        assert len(versions) == 2

    def test_delete_version(self, conn):
        vr, song_id = self._setup(conn)
        v = _make_version(song_id)
        vr.create(v)
        conn.commit()
        vr.delete(v.id)
        conn.commit()
        assert vr.get(v.id) is None

    def test_fields_round_trip(self, conn):
        vr, song_id = self._setup(conn)
        v = _make_version(
            song_id, label="Master", audio_file="audio/master.wav",
            duration_seconds=245.7, original_sample_rate=96000,
            audio_hash="deadbeef1234",
        )
        vr.create(v)
        conn.commit()
        got = vr.get(v.id)
        assert got.audio_file == "audio/master.wav"
        assert got.duration_seconds == 245.7
        assert got.original_sample_rate == 96000
        assert got.audio_hash == "deadbeef1234"


# ---------------------------------------------------------------------------
# Layer CRUD
# ---------------------------------------------------------------------------



__all__ = [name for name in globals() if name.startswith("Test")]

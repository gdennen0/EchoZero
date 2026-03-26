"""
Persistence layer tests: schema, CRUD, cascades, ordering, invariants, round-trips.
Exercises every entity type and repository against in-memory SQLite.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.persistence.base import BaseRepository
from echozero.persistence.entities import (
    LayerRecord,
    Project,
    ProjectSettings,
    Song,
    SongPipelineConfig,
    SongVersion,
)
from echozero.persistence.repositories import (
    LayerRepository,
    PipelineConfigRepository,
    ProjectRepository,
    SongRepository,
    SongVersionRepository,
    TakeRepository,
)
from echozero.persistence.schema import (
    SCHEMA_VERSION,
    get_schema_version,
    init_db,
    set_schema_version,
)
from echozero.takes import Take, TakeSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_project(**kw) -> Project:
    defaults = dict(
        id=_uid(), name="Test Project", settings=ProjectSettings(),
        created_at=_now(), updated_at=_now(),
    )
    defaults.update(kw)
    return Project(**defaults)


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


def _make_audio_data() -> AudioData:
    return AudioData(
        sample_rate=44100, duration=120.5,
        file_path="audio/test.wav", channel_count=2,
    )


def _make_take(is_main: bool = False, **kw) -> Take:
    defaults = dict(
        id=_uid(), label="Take 1", data=_make_event_data(),
        origin="pipeline", source=None, created_at=_now(),
        is_main=is_main, notes="",
    )
    defaults.update(kw)
    return Take(**defaults)


def _make_pipeline_config(song_version_id: str, **kw) -> SongPipelineConfig:
    defaults = dict(
        id=_uid(), song_version_id=song_version_id,
        pipeline_id="onset_detection",
        bindings={"audio_file": "test.wav", "threshold": 0.3},
        created_at=_now(),
    )
    defaults.update(kw)
    return SongPipelineConfig(**defaults)


# ---------------------------------------------------------------------------
# Fixture: fresh in-memory database for every test
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_db(c)
    return c


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


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
        assert "song_pipeline_configs" in tables

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


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------


class TestProjectRepository:
    def test_create_and_get(self, conn):
        repo = ProjectRepository(conn)
        p = _make_project(name="My Show", settings=ProjectSettings(bpm=120.0))
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
        s = ProjectSettings(sample_rate=48000, bpm=140.5, bpm_confidence=0.95, timecode_fps=29.97)
        p = _make_project(settings=s)
        repo.create(p)
        conn.commit()
        got = repo.get(p.id)
        assert got.settings == s


# ---------------------------------------------------------------------------
# Song CRUD
# ---------------------------------------------------------------------------


class TestSongRepository:
    def _setup(self, conn) -> tuple[ProjectRepository, SongRepository, Project]:
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
# SongVersion CRUD
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


class TestLayerRepository:
    def _setup(self, conn) -> tuple[LayerRepository, str]:
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        vr = SongVersionRepository(conn)
        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        v = _make_version(s.id)
        vr.create(v)
        conn.commit()
        return LayerRepository(conn), v.id

    def test_create_and_get(self, conn):
        lr, vid = self._setup(conn)
        layer = _make_layer(vid, name="Bass", color="#00FF00")
        lr.create(layer)
        conn.commit()
        got = lr.get(layer.id)
        assert got is not None
        assert got.name == "Bass"
        assert got.color == "#00FF00"
        assert got.visible is True
        assert got.locked is False

    def test_list_by_version_ordered(self, conn):
        lr, vid = self._setup(conn)
        lr.create(_make_layer(vid, name="C", order=2))
        lr.create(_make_layer(vid, name="A", order=0))
        lr.create(_make_layer(vid, name="B", order=1))
        conn.commit()
        layers = lr.list_by_version(vid)
        assert [l.name for l in layers] == ["A", "B", "C"]

    def test_update_layer(self, conn):
        lr, vid = self._setup(conn)
        layer = _make_layer(vid)
        lr.create(layer)
        conn.commit()
        updated = replace(layer, name="Renamed", visible=False, locked=True)
        lr.update(updated)
        conn.commit()
        got = lr.get(layer.id)
        assert got.name == "Renamed"
        assert got.visible is False
        assert got.locked is True

    def test_delete_layer(self, conn):
        lr, vid = self._setup(conn)
        layer = _make_layer(vid)
        lr.create(layer)
        conn.commit()
        lr.delete(layer.id)
        conn.commit()
        assert lr.get(layer.id) is None

    def test_reorder_layers(self, conn):
        lr, vid = self._setup(conn)
        l1 = _make_layer(vid, name="Drums", order=0)
        l2 = _make_layer(vid, name="Bass", order=1)
        l3 = _make_layer(vid, name="Vocals", order=2)
        lr.create(l1)
        lr.create(l2)
        lr.create(l3)
        conn.commit()
        lr.reorder(vid, [l3.id, l1.id, l2.id])
        conn.commit()
        layers = lr.list_by_version(vid)
        assert [l.name for l in layers] == ["Vocals", "Drums", "Bass"]

    def test_layer_type_check_constraint(self, conn):
        lr, vid = self._setup(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO layers "
                '(id, song_version_id, name, layer_type, "order", visible, locked, created_at) '
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_uid(), vid, "Bad", "invalid_type", 0, 1, 0, _now().isoformat()),
            )

    def test_source_pipeline_round_trip(self, conn):
        lr, vid = self._setup(conn)
        pipeline = {"name": "full_analysis", "version": "1.2", "settings": {"threshold": 0.5}}
        layer = _make_layer(vid, source_pipeline=pipeline)
        lr.create(layer)
        conn.commit()
        got = lr.get(layer.id)
        assert got.source_pipeline == pipeline

    def test_source_pipeline_null(self, conn):
        lr, vid = self._setup(conn)
        layer = _make_layer(vid, source_pipeline=None)
        lr.create(layer)
        conn.commit()
        got = lr.get(layer.id)
        assert got.source_pipeline is None


# ---------------------------------------------------------------------------
# Take CRUD
# ---------------------------------------------------------------------------


class TestTakeRepository:
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

    def test_create_and_get(self, conn):
        tr, lid = self._setup(conn)
        t = _make_take(is_main=True)
        tr.create(lid, t)
        conn.commit()
        got = tr.get(t.id)
        assert got is not None
        assert got.id == t.id
        assert got.label == t.label
        assert got.is_main is True

    def test_list_by_layer(self, conn):
        tr, lid = self._setup(conn)
        tr.create(lid, _make_take(is_main=True, label="Main"))
        tr.create(lid, _make_take(is_main=False, label="Alt"))
        conn.commit()
        takes = tr.list_by_layer(lid)
        assert len(takes) == 2

    def test_update_take(self, conn):
        tr, lid = self._setup(conn)
        t = _make_take(is_main=True)
        tr.create(lid, t)
        conn.commit()
        updated = replace(t, label="Renamed", notes="good one")
        tr.update(updated)
        conn.commit()
        got = tr.get(t.id)
        assert got.label == "Renamed"
        assert got.notes == "good one"

    def test_delete_take(self, conn):
        tr, lid = self._setup(conn)
        t = _make_take()
        tr.create(lid, t)
        conn.commit()
        tr.delete(t.id)
        conn.commit()
        assert tr.get(t.id) is None

    def test_get_main(self, conn):
        tr, lid = self._setup(conn)
        tr.create(lid, _make_take(is_main=False, label="Alt"))
        main = _make_take(is_main=True, label="Main")
        tr.create(lid, main)
        conn.commit()
        got = tr.get_main(lid)
        assert got is not None
        assert got.label == "Main"
        assert got.is_main is True

    def test_get_main_returns_none_when_no_main(self, conn):
        tr, lid = self._setup(conn)
        tr.create(lid, _make_take(is_main=False))
        conn.commit()
        assert tr.get_main(lid) is None

    def test_event_data_round_trip(self, conn):
        tr, lid = self._setup(conn)
        ed = _make_event_data()
        t = _make_take(is_main=True, data=ed)
        tr.create(lid, t)
        conn.commit()
        got = tr.get(t.id)
        assert isinstance(got.data, EventData)
        assert len(got.data.layers) == 1
        assert got.data.layers[0].name == "onsets"
        assert len(got.data.layers[0].events) == 1
        assert got.data.layers[0].events[0].time == 1.0

    def test_audio_data_round_trip(self, conn):
        tr, lid = self._setup(conn)
        ad = _make_audio_data()
        t = _make_take(is_main=True, data=ad)
        tr.create(lid, t)
        conn.commit()
        got = tr.get(t.id)
        assert isinstance(got.data, AudioData)
        assert got.data.sample_rate == 44100
        assert got.data.duration == 120.5
        assert got.data.channel_count == 2

    def test_take_source_round_trip(self, conn):
        tr, lid = self._setup(conn)
        source = TakeSource(
            block_id="blk1", block_type="onset_detector",
            settings_snapshot={"threshold": 0.3}, run_id="run_abc",
        )
        t = _make_take(is_main=True, source=source)
        tr.create(lid, t)
        conn.commit()
        got = tr.get(t.id)
        assert got.source is not None
        assert got.source.block_id == "blk1"
        assert got.source.settings_snapshot == {"threshold": 0.3}

    def test_take_source_null_round_trip(self, conn):
        tr, lid = self._setup(conn)
        t = _make_take(is_main=True, source=None)
        tr.create(lid, t)
        conn.commit()
        got = tr.get(t.id)
        assert got.source is None

    def test_origin_check_constraint(self, conn):
        tr, lid = self._setup(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO takes "
                "(id, layer_id, label, origin, is_main, is_archived, data_json, created_at, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_uid(), lid, "Bad", "invalid_origin", 0, 0, "{}", _now().isoformat(), ""),
            )


# ---------------------------------------------------------------------------
# PipelineConfig CRUD
# ---------------------------------------------------------------------------


class TestPipelineConfigRepository:
    def _setup(self, conn) -> tuple[PipelineConfigRepository, str]:
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        vr = SongVersionRepository(conn)
        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        v = _make_version(s.id)
        vr.create(v)
        conn.commit()
        return PipelineConfigRepository(conn), v.id

    def test_create_and_get(self, conn):
        pcr, vid = self._setup(conn)
        cfg = _make_pipeline_config(vid, pipeline_id="onset_detection")
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got is not None
        assert got.id == cfg.id
        assert got.pipeline_id == "onset_detection"
        assert got.song_version_id == vid

    def test_bindings_round_trip(self, conn):
        pcr, vid = self._setup(conn)
        bindings = {"audio_file": "test.wav", "threshold": 0.5, "method": "default"}
        cfg = _make_pipeline_config(vid, bindings=bindings)
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got.bindings == bindings

    def test_list_by_version(self, conn):
        pcr, vid = self._setup(conn)
        pcr.create(_make_pipeline_config(vid, pipeline_id="onset_detection"))
        pcr.create(_make_pipeline_config(vid, pipeline_id="full_analysis"))
        conn.commit()
        configs = pcr.list_by_version(vid)
        assert len(configs) == 2

    def test_delete_config(self, conn):
        pcr, vid = self._setup(conn)
        cfg = _make_pipeline_config(vid)
        pcr.create(cfg)
        conn.commit()
        pcr.delete(cfg.id)
        conn.commit()
        assert pcr.get(cfg.id) is None

    def test_get_nonexistent_returns_none(self, conn):
        pcr, vid = self._setup(conn)
        assert pcr.get("nonexistent") is None

    def test_cascade_delete_from_version(self, conn):
        pcr, vid = self._setup(conn)
        vr = SongVersionRepository(conn)
        cfg = _make_pipeline_config(vid)
        pcr.create(cfg)
        conn.commit()
        vr.delete(vid)
        conn.commit()
        assert pcr.get(cfg.id) is None

    def test_datetime_round_trip(self, conn):
        pcr, vid = self._setup(conn)
        now = _now()
        cfg = _make_pipeline_config(vid, created_at=now)
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got.created_at == now

    def test_empty_bindings(self, conn):
        pcr, vid = self._setup(conn)
        cfg = _make_pipeline_config(vid, bindings={})
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got.bindings == {}

    def test_complex_bindings(self, conn):
        pcr, vid = self._setup(conn)
        bindings = {
            "audio_file": "/path/to/file.wav",
            "threshold": 0.3,
            "enabled": True,
            "tags": ["kick", "snare"],
            "nested": {"key": "value"},
        }
        cfg = _make_pipeline_config(vid, bindings=bindings)
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got.bindings == bindings


# ---------------------------------------------------------------------------
# FK cascade deletes
# ---------------------------------------------------------------------------


class TestCascadeDeletes:
    def test_delete_project_cascades_everything(self, conn):
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        vr = SongVersionRepository(conn)
        lr = LayerRepository(conn)
        tr = TakeRepository(conn)

        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        v = _make_version(s.id)
        vr.create(v)
        layer = _make_layer(v.id)
        lr.create(layer)
        take = _make_take(is_main=True)
        tr.create(layer.id, take)
        conn.commit()

        # Verify everything exists
        assert sr.get(s.id) is not None
        assert vr.get(v.id) is not None
        assert lr.get(layer.id) is not None
        assert tr.get(take.id) is not None

        # Delete the project
        pr.delete(p.id)
        conn.commit()

        # Verify everything is gone
        assert sr.get(s.id) is None
        assert vr.get(v.id) is None
        assert lr.get(layer.id) is None
        assert tr.get(take.id) is None

    def test_delete_song_cascades_to_versions_layers_takes(self, conn):
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        vr = SongVersionRepository(conn)
        lr = LayerRepository(conn)
        tr = TakeRepository(conn)

        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        v = _make_version(s.id)
        vr.create(v)
        layer = _make_layer(v.id)
        lr.create(layer)
        take = _make_take(is_main=True)
        tr.create(layer.id, take)
        conn.commit()

        sr.delete(s.id)
        conn.commit()

        assert vr.get(v.id) is None
        assert lr.get(layer.id) is None
        assert tr.get(take.id) is None

    def test_delete_version_cascades_to_layers_takes(self, conn):
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        vr = SongVersionRepository(conn)
        lr = LayerRepository(conn)
        tr = TakeRepository(conn)

        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        v = _make_version(s.id)
        vr.create(v)
        layer = _make_layer(v.id)
        lr.create(layer)
        take = _make_take(is_main=True)
        tr.create(layer.id, take)
        conn.commit()

        vr.delete(v.id)
        conn.commit()

        assert lr.get(layer.id) is None
        assert tr.get(take.id) is None

    def test_delete_layer_cascades_to_takes(self, conn):
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        vr = SongVersionRepository(conn)
        lr = LayerRepository(conn)
        tr = TakeRepository(conn)

        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        v = _make_version(s.id)
        vr.create(v)
        layer = _make_layer(v.id)
        lr.create(layer)
        take = _make_take(is_main=True)
        tr.create(layer.id, take)
        conn.commit()

        lr.delete(layer.id)
        conn.commit()

        assert tr.get(take.id) is None


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_songs_order_in_project(self, conn):
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        p = _make_project()
        pr.create(p)
        conn.commit()

        titles = ["Encore", "Opener", "Bridge", "Closer"]
        for i, title in enumerate(titles):
            sr.create(_make_song(p.id, title=title, order=i))
        conn.commit()

        songs = sr.list_by_project(p.id)
        assert [s.title for s in songs] == titles

    def test_layers_order_in_version(self, conn):
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

        names = ["Vocals", "Drums", "Bass", "Keys"]
        for i, name in enumerate(names):
            lr.create(_make_layer(v.id, name=name, order=i))
        conn.commit()

        layers = lr.list_by_version(v.id)
        assert [l.name for l in layers] == names


# ---------------------------------------------------------------------------
# Take main invariant
# ---------------------------------------------------------------------------


class TestTakeMainInvariant:
    def _setup_layer(self, conn) -> tuple[TakeRepository, str]:
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

    def test_one_main_per_layer(self, conn):
        tr, lid = self._setup_layer(conn)
        main = _make_take(is_main=True, label="Main")
        alt = _make_take(is_main=False, label="Alt")
        tr.create(lid, main)
        tr.create(lid, alt)
        conn.commit()

        got_main = tr.get_main(lid)
        assert got_main.label == "Main"

    def test_promote_demotes_old_main(self, conn):
        tr, lid = self._setup_layer(conn)
        t1 = _make_take(is_main=True, label="First Main")
        t2 = _make_take(is_main=False, label="New Main")
        tr.create(lid, t1)
        tr.create(lid, t2)
        conn.commit()

        # Demote old, promote new
        tr.update(replace(t1, is_main=False))
        tr.update(replace(t2, is_main=True))
        conn.commit()

        got_main = tr.get_main(lid)
        assert got_main.label == "New Main"

        # Old main is no longer main
        got_old = tr.get(t1.id)
        assert got_old.is_main is False

    def test_multiple_layers_independent_mains(self, conn):
        pr = ProjectRepository(conn)
        sr = SongRepository(conn)
        vr = SongVersionRepository(conn)
        lr = LayerRepository(conn)
        tr = TakeRepository(conn)

        p = _make_project()
        pr.create(p)
        s = _make_song(p.id)
        sr.create(s)
        v = _make_version(s.id)
        vr.create(v)

        l1 = _make_layer(v.id, name="Drums", order=0)
        l2 = _make_layer(v.id, name="Bass", order=1)
        lr.create(l1)
        lr.create(l2)

        t1 = _make_take(is_main=True, label="Drums Main")
        t2 = _make_take(is_main=True, label="Bass Main")
        tr.create(l1.id, t1)
        tr.create(l2.id, t2)
        conn.commit()

        assert tr.get_main(l1.id).label == "Drums Main"
        assert tr.get_main(l2.id).label == "Bass Main"


# ---------------------------------------------------------------------------
# Full round-trip
# ---------------------------------------------------------------------------


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
            settings=ProjectSettings(sample_rate=48000, bpm=128.0),
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
        with pytest.raises(sqlite3.IntegrityError):
            pr.create(p)

    def test_song_with_missing_project_fk_raises(self, conn):
        sr = SongRepository(conn)
        s = _make_song("nonexistent_project_id")
        with pytest.raises(sqlite3.IntegrityError):
            sr.create(s)

    def test_version_with_missing_song_fk_raises(self, conn):
        vr = SongVersionRepository(conn)
        v = _make_version("nonexistent_song_id")
        with pytest.raises(sqlite3.IntegrityError):
            vr.create(v)

    def test_layer_with_missing_version_fk_raises(self, conn):
        lr = LayerRepository(conn)
        layer = _make_layer("nonexistent_version_id")
        with pytest.raises(sqlite3.IntegrityError):
            lr.create(layer)

    def test_take_with_missing_layer_fk_raises(self, conn):
        tr = TakeRepository(conn)
        t = _make_take()
        with pytest.raises(sqlite3.IntegrityError):
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
        with pytest.raises(sqlite3.IntegrityError):
            sr.create(s)

    def test_project_with_null_optional_settings(self, conn):
        pr = ProjectRepository(conn)
        p = _make_project(settings=ProjectSettings())
        pr.create(p)
        conn.commit()
        got = pr.get(p.id)
        assert got.settings.bpm is None
        assert got.settings.bpm_confidence is None
        assert got.settings.timecode_fps is None

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

"""Layer and take persistence support cases.
Exists to keep layered ordering and invariant coverage separate from core and round-trip support tests.
Connects the compatibility wrapper to the bounded persistence layers slice.
"""

from tests.persistence_shared_support import *  # noqa: F401,F403

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
# PipelineConfigRecord CRUD
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
        cfg = _make_pipeline_config(vid, template_id="onset_detection")
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got is not None
        assert got.id == cfg.id
        assert got.template_id == "onset_detection"
        assert got.song_version_id == vid
        assert got.name == cfg.name

    def test_knob_values_round_trip(self, conn):
        pcr, vid = self._setup(conn)
        knob_values = {"threshold": 0.5, "method": "default"}
        cfg = _make_pipeline_config(vid, knob_values=knob_values)
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got.knob_values == knob_values

    def test_graph_json_round_trip(self, conn):
        pcr, vid = self._setup(conn)
        graph_json = '{"blocks": [{"id": "load"}], "connections": []}'
        cfg = _make_pipeline_config(vid, graph_json=graph_json)
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got.graph_json == graph_json

    def test_list_by_version(self, conn):
        pcr, vid = self._setup(conn)
        pcr.create(_make_pipeline_config(vid, template_id="onset_detection"))
        pcr.create(_make_pipeline_config(vid, template_id="full_analysis"))
        conn.commit()
        configs = pcr.list_by_version(vid)
        assert len(configs) == 2

    def test_list_by_template(self, conn):
        pcr, vid = self._setup(conn)
        pcr.create(_make_pipeline_config(vid, template_id="onset_detection"))
        pcr.create(_make_pipeline_config(vid, template_id="full_analysis"))
        pcr.create(_make_pipeline_config(vid, template_id="onset_detection"))
        conn.commit()
        configs = pcr.list_by_template("onset_detection")
        assert len(configs) == 2

    def test_update(self, conn):
        pcr, vid = self._setup(conn)
        cfg = _make_pipeline_config(vid, knob_values={"threshold": 0.3})
        pcr.create(cfg)
        conn.commit()
        updated = replace(cfg, knob_values={"threshold": 0.7}, name="Updated")
        pcr.update(updated)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got.knob_values == {"threshold": 0.7}
        assert got.name == "Updated"

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
        cfg = _make_pipeline_config(vid, created_at=now, updated_at=now)
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got.created_at == now
        assert got.updated_at == now

    def test_empty_knob_values(self, conn):
        pcr, vid = self._setup(conn)
        cfg = _make_pipeline_config(vid, knob_values={})
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got.knob_values == {}

    def test_complex_knob_values(self, conn):
        pcr, vid = self._setup(conn)
        knob_values = {
            "threshold": 0.3,
            "enabled": True,
            "tags": ["kick", "snare"],
            "nested": {"key": "value"},
        }
        cfg = _make_pipeline_config(vid, knob_values=knob_values)
        pcr.create(cfg)
        conn.commit()
        got = pcr.get(cfg.id)
        assert got.knob_values == knob_values


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



__all__ = [name for name in globals() if name.startswith("Test")]

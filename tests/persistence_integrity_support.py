"""Integrity and migration persistence support cases.
Exists to keep duplicate-id, null-guard, error-wrapping, and migration coverage separate from CRUD support tests.
Connects the compatibility wrapper to the bounded persistence integrity slice.
"""

from tests.persistence_shared_support import *  # noqa: F401,F403

class TestDuplicateIds:
    def _setup_version(self, conn) -> tuple[str, str, str]:
        """Create project -> song -> version and return (project_id, song_id, version_id)."""
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
        return p.id, s.id, v.id

    def test_duplicate_song_version_id_raises(self, conn):
        _, song_id, _ = self._setup_version(conn)
        vr = SongVersionRepository(conn)
        v = _make_version(song_id)
        vr.create(v)
        conn.commit()
        with pytest.raises(PersistenceError):
            vr.create(v)

    def test_duplicate_layer_id_raises(self, conn):
        _, _, vid = self._setup_version(conn)
        lr = LayerRepository(conn)
        layer = _make_layer(vid)
        lr.create(layer)
        conn.commit()
        with pytest.raises(PersistenceError):
            lr.create(layer)

    def test_duplicate_take_id_raises(self, conn):
        _, _, vid = self._setup_version(conn)
        lr = LayerRepository(conn)
        tr = TakeRepository(conn)
        layer = _make_layer(vid)
        lr.create(layer)
        conn.commit()
        t = _make_take(is_main=True)
        tr.create(layer.id, t)
        conn.commit()
        with pytest.raises(PersistenceError):
            tr.create(layer.id, t)

    def test_duplicate_pipeline_config_id_raises(self, conn):
        _, _, vid = self._setup_version(conn)
        pcr = PipelineConfigRepository(conn)
        cfg = _make_pipeline_config(vid)
        pcr.create(cfg)
        conn.commit()
        with pytest.raises(PersistenceError):
            pcr.create(cfg)


# ---------------------------------------------------------------------------
# data_json null guard test (item 26)
# ---------------------------------------------------------------------------


class TestDataJsonNullGuard:
    def test_null_data_json_raises_persistence_error(self, conn):
        """Take with NULL data_json: _from_row raises PersistenceError, but
        list_by_layer skips the corrupt row instead of crashing the whole listing.
        P8 fix: one bad take must not poison the entire layer listing."""
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
        conn.commit()

        # Manually insert a take with NULL data_json
        corrupt_id = _uid()
        conn.execute(
            "INSERT INTO takes "
            "(id, layer_id, label, origin, is_main, is_archived, "
            "source_json, data_json, created_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (corrupt_id, layer.id, "Bad Take", "pipeline", 0, 0,
             None, None, _now().isoformat(), ""),
        )
        conn.commit()

        takes = conn.execute(
            "SELECT id FROM takes WHERE layer_id = ?", (layer.id,)
        ).fetchall()
        assert len(takes) == 1

        # list_by_layer should skip the corrupt row and return an empty list,
        # NOT raise — this is the P8 resilience fix.
        result = tr.list_by_layer(layer.id)
        assert result == []

        # _from_row itself still raises PersistenceError on a NULL data_json row
        # (the error is just caught at the listing level now).
        row = conn.execute(
            "SELECT id, layer_id, label, origin, is_main, is_archived, "
            "source_json, data_json, created_at, notes "
            "FROM takes WHERE id = ?",
            (corrupt_id,),
        ).fetchone()
        with pytest.raises(PersistenceError, match="Take has no data"):
            tr._from_row(row)


# ---------------------------------------------------------------------------
# PersistenceError wrapping test (item 27)
# ---------------------------------------------------------------------------


class TestPersistenceErrorWrapping:
    def test_sqlite3_error_becomes_persistence_error(self, conn):
        """sqlite3.IntegrityError through a repo method becomes PersistenceError."""
        pr = ProjectRepository(conn)
        p = _make_project()
        pr.create(p)
        conn.commit()
        with pytest.raises(PersistenceError) as exc_info:
            pr.create(p)
        # Original sqlite3 error is preserved as __cause__
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, sqlite3.IntegrityError)

    def test_bad_sql_becomes_persistence_error(self, conn):
        """sqlite3.OperationalError through _execute becomes PersistenceError."""
        from echozero.persistence.base import BaseRepository

        class TestRepo(BaseRepository[str]):
            def _from_row(self, row):
                return str(row[0])

        repo = TestRepo(conn)
        with pytest.raises(PersistenceError) as exc_info:
            repo._execute("SELECT * FROM nonexistent_table_xyz")
        assert isinstance(exc_info.value.__cause__, sqlite3.OperationalError)


# ---------------------------------------------------------------------------
# Migration infrastructure test (item 23)
# ---------------------------------------------------------------------------


class TestMigrationInfrastructure:
    def test_apply_migrations_with_mock_migration(self):
        """Mock a migration and verify it runs."""
        from echozero.persistence.schema import (
            _MIGRATIONS,
            SCHEMA_VERSION,
            apply_migrations,
            get_schema_version,
            set_schema_version,
        )
        # Use a fresh in-memory DB
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        # Create minimal schema with just _meta at version 0
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT);"
        )

        # Set version to SCHEMA_VERSION - 1 to simulate needing one migration
        fake_old_version = SCHEMA_VERSION  # current version
        fake_new_version = fake_old_version + 1

        set_schema_version(conn, fake_old_version)
        conn.commit()

        # Create a dummy table as migration target
        migration_ran = []

        def mock_migration(c):
            c.execute(
                "CREATE TABLE IF NOT EXISTS _migration_test (id TEXT PRIMARY KEY)"
            )
            migration_ran.append(True)

        # Temporarily patch _MIGRATIONS and SCHEMA_VERSION
        import echozero.persistence.schema as schema_mod
        original_version = schema_mod.SCHEMA_VERSION
        original_migrations = dict(_MIGRATIONS)

        try:
            schema_mod.SCHEMA_VERSION = fake_new_version
            _MIGRATIONS[fake_new_version] = mock_migration

            apply_migrations(conn)

            assert len(migration_ran) == 1
            assert get_schema_version(conn) == fake_new_version

            # Verify the migration actually created the table
            tables = {
                row['name']
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert '_migration_test' in tables
        finally:
            # Restore
            schema_mod.SCHEMA_VERSION = original_version
            _MIGRATIONS.clear()
            _MIGRATIONS.update(original_migrations)
            conn.close()

__all__ = [name for name in globals() if name.startswith("Test")]

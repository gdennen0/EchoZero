"""
Schema: SQLite DDL, version tracking, and migration infrastructure for EchoZero projects.
Exists because the persistence layer needs a stable, versioned schema that can evolve
across releases without losing user data. All tables, indexes, and constraints live here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

SCHEMA_VERSION = 5

_DDL = """\
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
    graph_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS songs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    artist TEXT DEFAULT '',
    "order" INTEGER NOT NULL DEFAULT 0,
    active_version_id TEXT
);

CREATE TABLE IF NOT EXISTS song_versions (
    id TEXT PRIMARY KEY,
    song_id TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    audio_file TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    original_sample_rate INTEGER NOT NULL,
    audio_hash TEXT NOT NULL,
    rebuild_plan_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS layers (
    id TEXT PRIMARY KEY,
    song_version_id TEXT NOT NULL REFERENCES song_versions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    layer_type TEXT NOT NULL DEFAULT 'analysis'
        CHECK(layer_type IN ('analysis', 'structure', 'manual')),
    color TEXT,
    "order" INTEGER NOT NULL DEFAULT 0,
    visible INTEGER NOT NULL DEFAULT 1,
    locked INTEGER NOT NULL DEFAULT 0,
    parent_layer_id TEXT REFERENCES layers(id),
    source_pipeline TEXT,
    state_flags_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS takes (
    id TEXT PRIMARY KEY,
    layer_id TEXT NOT NULL REFERENCES layers(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    origin TEXT NOT NULL CHECK(origin IN ('pipeline', 'user', 'merge', 'sync')),
    is_main INTEGER NOT NULL DEFAULT 0,
    is_archived INTEGER NOT NULL DEFAULT 0,
    source_json TEXT,
    data_json TEXT,
    created_at TEXT NOT NULL,
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS pipeline_configs (
    id TEXT PRIMARY KEY,
    song_version_id TEXT NOT NULL REFERENCES song_versions(id) ON DELETE CASCADE,
    template_id TEXT NOT NULL,
    name TEXT NOT NULL,
    graph_json TEXT NOT NULL,
    outputs_json TEXT NOT NULL DEFAULT '[]',
    knob_values_json TEXT NOT NULL DEFAULT '{}',
    block_overrides_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_songs_project ON songs(project_id);
CREATE INDEX IF NOT EXISTS idx_versions_song ON song_versions(song_id);
CREATE INDEX IF NOT EXISTS idx_layers_version ON layers(song_version_id);
CREATE INDEX IF NOT EXISTS idx_takes_layer ON takes(layer_id);
CREATE INDEX IF NOT EXISTS idx_configs_version ON pipeline_configs(song_version_id);
CREATE INDEX IF NOT EXISTS idx_configs_template ON pipeline_configs(template_id);

CREATE TABLE IF NOT EXISTS song_default_pipeline_configs (
    id TEXT PRIMARY KEY,
    song_id TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    template_id TEXT NOT NULL,
    name TEXT NOT NULL,
    graph_json TEXT NOT NULL,
    outputs_json TEXT NOT NULL DEFAULT '[]',
    knob_values_json TEXT NOT NULL DEFAULT '{}',
    block_overrides_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_song_default_configs_song ON song_default_pipeline_configs(song_id);
CREATE INDEX IF NOT EXISTS idx_song_default_configs_template ON song_default_pipeline_configs(template_id);
"""


def get_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        return 0
    return int(row["value"])


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO _meta (key, value) VALUES ('schema_version', ?)",
        (str(version),),
    )


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    old_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='song_pipeline_configs'"
    ).fetchone()

    conn.executescript("""\
        CREATE TABLE IF NOT EXISTS pipeline_configs (
            id TEXT PRIMARY KEY,
            song_version_id TEXT NOT NULL REFERENCES song_versions(id) ON DELETE CASCADE,
            template_id TEXT NOT NULL,
            name TEXT NOT NULL,
            graph_json TEXT NOT NULL,
            outputs_json TEXT NOT NULL DEFAULT '[]',
            knob_values_json TEXT NOT NULL DEFAULT '{}',
            block_overrides_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_configs_version ON pipeline_configs(song_version_id);
        CREATE INDEX IF NOT EXISTS idx_configs_template ON pipeline_configs(template_id);
    """)

    if old_table is None:
        return

    rows = conn.execute(
        "SELECT id, song_version_id, pipeline_id, bindings, created_at "
        "FROM song_pipeline_configs"
    ).fetchall()
    for row in rows:
        conn.execute(
            "INSERT INTO pipeline_configs "
            "(id, song_version_id, template_id, name, graph_json, outputs_json, "
            "knob_values_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, '{}', '[]', ?, ?, ?)",
            (
                row["id"],
                row["song_version_id"],
                row["pipeline_id"],
                row["pipeline_id"],
                row["bindings"],
                row["created_at"],
                row["created_at"],
            ),
        )

    conn.execute("DROP TABLE IF EXISTS song_pipeline_configs")


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    layers_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='layers'"
    ).fetchone()
    if layers_table is None:
        return

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(layers)").fetchall()}
    if "state_flags_json" not in columns:
        conn.execute("ALTER TABLE layers ADD COLUMN state_flags_json TEXT NOT NULL DEFAULT '{}' ")
    if "provenance_json" not in columns:
        conn.execute("ALTER TABLE layers ADD COLUMN provenance_json TEXT NOT NULL DEFAULT '{}' ")


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    versions_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='song_versions'"
    ).fetchone()
    if versions_table is None:
        return

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(song_versions)").fetchall()}
    if "rebuild_plan_json" not in columns:
        conn.execute(
            "ALTER TABLE song_versions ADD COLUMN rebuild_plan_json TEXT NOT NULL DEFAULT '{}' "
        )


_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    2: _migrate_v1_to_v2,
    3: _migrate_v2_to_v3,
    4: _migrate_v3_to_v4,
    5: lambda conn: conn.executescript("""\
        CREATE TABLE IF NOT EXISTS song_default_pipeline_configs (
            id TEXT PRIMARY KEY,
            song_id TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
            template_id TEXT NOT NULL,
            name TEXT NOT NULL,
            graph_json TEXT NOT NULL,
            outputs_json TEXT NOT NULL DEFAULT '[]',
            knob_values_json TEXT NOT NULL DEFAULT '{}',
            block_overrides_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_song_default_configs_song ON song_default_pipeline_configs(song_id);
        CREATE INDEX IF NOT EXISTS idx_song_default_configs_template ON song_default_pipeline_configs(template_id);
    """),
}


def apply_migrations(conn: sqlite3.Connection) -> None:
    current = get_schema_version(conn)
    for target in range(current + 1, SCHEMA_VERSION + 1):
        migrate_fn = _MIGRATIONS.get(target)
        if migrate_fn is not None:
            migrate_fn(conn)
        set_schema_version(conn, target)
    conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL)
    current = get_schema_version(conn)
    if current == 0:
        set_schema_version(conn, SCHEMA_VERSION)
        conn.commit()
    else:
        apply_migrations(conn)

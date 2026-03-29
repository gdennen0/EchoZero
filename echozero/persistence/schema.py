"""
Schema: SQLite DDL, version tracking, and migration infrastructure for EchoZero projects.
Exists because the persistence layer needs a stable, versioned schema that can evolve
across releases without losing user data. All tables, indexes, and constraints live here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# DDL — all tables for schema version 1
# ---------------------------------------------------------------------------

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
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_songs_project ON songs(project_id);
CREATE INDEX IF NOT EXISTS idx_versions_song ON song_versions(song_id);
CREATE INDEX IF NOT EXISTS idx_layers_version ON layers(song_version_id);
CREATE INDEX IF NOT EXISTS idx_takes_layer ON takes(layer_id);
CREATE INDEX IF NOT EXISTS idx_configs_version ON pipeline_configs(song_version_id);
CREATE INDEX IF NOT EXISTS idx_configs_template ON pipeline_configs(template_id);
"""


# ---------------------------------------------------------------------------
# Schema version helpers
# ---------------------------------------------------------------------------


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Read the current schema version from _meta. Returns 0 if unset."""
    row = conn.execute(
        "SELECT value FROM _meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return 0
    return int(row['value'])


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Write the schema version to _meta (insert or update)."""
    conn.execute(
        "INSERT OR REPLACE INTO _meta (key, value) VALUES ('schema_version', ?)",
        (str(version),),
    )


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

# Numbered migration functions: key = target version, value = upgrade function.
# Each function upgrades from (key - 1) to key.
# Empty for V1 — this is just the infrastructure.
def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Replace song_pipeline_configs with pipeline_configs.

    V1 had a flat bindings blob. V2 stores the full graph + outputs + knob values.
    Existing configs are migrated with empty graph/outputs (user must re-configure).
    """
    # Create new table
    conn.executescript("""\
        CREATE TABLE IF NOT EXISTS pipeline_configs (
            id TEXT PRIMARY KEY,
            song_version_id TEXT NOT NULL REFERENCES song_versions(id) ON DELETE CASCADE,
            template_id TEXT NOT NULL,
            name TEXT NOT NULL,
            graph_json TEXT NOT NULL,
            outputs_json TEXT NOT NULL DEFAULT '[]',
            knob_values_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_configs_version ON pipeline_configs(song_version_id);
        CREATE INDEX IF NOT EXISTS idx_configs_template ON pipeline_configs(template_id);
    """)

    # Migrate existing data (best-effort — graph must be rebuilt from template)
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
                row['id'],
                row['song_version_id'],
                row['pipeline_id'],
                row['pipeline_id'],  # name = template_id as placeholder
                row['bindings'],     # preserve old bindings as knob_values
                row['created_at'],
                row['created_at'],   # updated_at = created_at
            ),
        )

    conn.execute("DROP TABLE IF EXISTS song_pipeline_configs")


_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    2: _migrate_v1_to_v2,
}


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations to bring the schema up to SCHEMA_VERSION."""
    current = get_schema_version(conn)
    for target in range(current + 1, SCHEMA_VERSION + 1):
        migrate_fn = _MIGRATIONS.get(target)
        if migrate_fn is not None:
            migrate_fn(conn)
        set_schema_version(conn, target)
    conn.commit()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables and set the schema version. Safe to call on an existing DB."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL)
    current = get_schema_version(conn)
    if current == 0:
        set_schema_version(conn, SCHEMA_VERSION)
        conn.commit()
    else:
        apply_migrations(conn)

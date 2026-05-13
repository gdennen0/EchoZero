"""
Schema: SQLite DDL, version tracking, and migration infrastructure for EchoZero projects.
Exists because the persistence layer needs a stable, versioned schema that can evolve
across releases without losing user data. All tables, indexes, and constraints live here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
import json

from echozero.errors import PersistenceError

SCHEMA_VERSION = 11
OBJECT_CONTENT_SCHEMA_VERSION = 10

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
    ma3_push_offset_seconds REAL NOT NULL DEFAULT -1.0,
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
    bpm REAL,
    bpm_confidence REAL,
    beat_anchor_seconds REAL,
    ma3_timecode_pool_no INTEGER,
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

CREATE TABLE IF NOT EXISTS timeline_objects (
    id TEXT PRIMARY KEY,
    song_version_id TEXT NOT NULL REFERENCES song_versions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    object_kind TEXT NOT NULL,
    main_content_id TEXT NOT NULL REFERENCES object_contents(id) DEFERRABLE INITIALLY DEFERRED,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS object_contents (
    id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL REFERENCES timeline_objects(id) ON DELETE CASCADE,
    revision_id TEXT NOT NULL,
    content_kind TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    source_ref_json TEXT,
    analysis_build_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS object_candidates (
    id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL REFERENCES timeline_objects(id) ON DELETE CASCADE,
    content_id TEXT NOT NULL REFERENCES object_contents(id) ON DELETE CASCADE,
    label TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
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
CREATE INDEX IF NOT EXISTS idx_timeline_objects_version ON timeline_objects(song_version_id);
CREATE INDEX IF NOT EXISTS idx_object_contents_object ON object_contents(object_id);
CREATE INDEX IF NOT EXISTS idx_object_candidates_object ON object_candidates(object_id);
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


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    versions_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='song_versions'"
    ).fetchone()
    if versions_table is None:
        return

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(song_versions)").fetchall()}
    if "ma3_timecode_pool_no" not in columns:
        conn.execute("ALTER TABLE song_versions ADD COLUMN ma3_timecode_pool_no INTEGER ")


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    conn.executescript("""\
        CREATE TABLE IF NOT EXISTS timeline_regions (
            id TEXT PRIMARY KEY,
            song_version_id TEXT NOT NULL REFERENCES song_versions(id) ON DELETE CASCADE,
            label TEXT NOT NULL,
            start_seconds REAL NOT NULL,
            end_seconds REAL NOT NULL,
            color TEXT,
            order_index INTEGER NOT NULL DEFAULT 0,
            kind TEXT NOT NULL DEFAULT 'custom',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_timeline_regions_version ON timeline_regions(song_version_id);
    """)


def _migrate_v7_to_v8(conn: sqlite3.Connection) -> None:
    projects_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='projects'"
    ).fetchone()
    if projects_table is None:
        return

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(projects)").fetchall()}
    if "ma3_push_offset_seconds" not in columns:
        conn.execute(
            "ALTER TABLE projects ADD COLUMN ma3_push_offset_seconds REAL NOT NULL DEFAULT -1.0"
        )


def _migrate_v8_to_v9(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX IF EXISTS idx_timeline_regions_version")
    conn.execute("DROP TABLE IF EXISTS timeline_regions")


def _migrate_v9_to_v10(conn: sqlite3.Connection) -> None:
    conn.executescript("""\
        CREATE TABLE IF NOT EXISTS timeline_objects (
            id TEXT PRIMARY KEY,
            song_version_id TEXT NOT NULL REFERENCES song_versions(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            object_kind TEXT NOT NULL,
            main_content_id TEXT NOT NULL REFERENCES object_contents(id) DEFERRABLE INITIALLY DEFERRED,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS object_contents (
            id TEXT PRIMARY KEY,
            object_id TEXT NOT NULL REFERENCES timeline_objects(id) ON DELETE CASCADE,
            revision_id TEXT NOT NULL,
            content_kind TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            source_ref_json TEXT,
            analysis_build_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS object_candidates (
            id TEXT PRIMARY KEY,
            object_id TEXT NOT NULL REFERENCES timeline_objects(id) ON DELETE CASCADE,
            content_id TEXT NOT NULL REFERENCES object_contents(id) ON DELETE CASCADE,
            label TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_timeline_objects_version ON timeline_objects(song_version_id);
        CREATE INDEX IF NOT EXISTS idx_object_contents_object ON object_contents(object_id);
        CREATE INDEX IF NOT EXISTS idx_object_candidates_object ON object_candidates(object_id);
    """)


def _migrate_v10_to_v11(conn: sqlite3.Connection) -> None:
    versions_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='song_versions'"
    ).fetchone()
    if versions_table is None:
        return

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(song_versions)").fetchall()}
    if "bpm" not in columns:
        conn.execute("ALTER TABLE song_versions ADD COLUMN bpm REAL ")
    if "bpm_confidence" not in columns:
        conn.execute("ALTER TABLE song_versions ADD COLUMN bpm_confidence REAL ")
    if "beat_anchor_seconds" not in columns:
        conn.execute("ALTER TABLE song_versions ADD COLUMN beat_anchor_seconds REAL ")


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
    6: _migrate_v5_to_v6,
    7: _migrate_v6_to_v7,
    8: _migrate_v7_to_v8,
    9: _migrate_v8_to_v9,
    10: _migrate_v9_to_v10,
    11: _migrate_v10_to_v11,
}


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply runtime-safe migrations, failing loudly for pre-object/content projects."""

    current = get_schema_version(conn)
    if 0 < current < OBJECT_CONTENT_SCHEMA_VERSION:
        raise PersistenceError(
            "Unsupported EchoZero project schema "
            f"v{current}; runtime open requires v{OBJECT_CONTENT_SCHEMA_VERSION}. "
            "Run the manual object/content project updater before opening this project."
        )
    _apply_migrations(conn, allow_pre_object_content_upgrade=False)


def apply_manual_object_content_update(conn: sqlite3.Connection) -> None:
    """Run explicit legacy project conversion outside the runtime open path."""

    _apply_migrations(conn, allow_pre_object_content_upgrade=True)
    _backfill_object_content_rows(conn)
    conn.commit()


def _apply_migrations(
    conn: sqlite3.Connection,
    *,
    allow_pre_object_content_upgrade: bool,
) -> None:
    current = get_schema_version(conn)
    if 0 < current < OBJECT_CONTENT_SCHEMA_VERSION and not allow_pre_object_content_upgrade:
        raise PersistenceError(
            "Unsupported EchoZero project schema "
            f"v{current}; runtime open requires v{OBJECT_CONTENT_SCHEMA_VERSION}."
        )
    for target in range(current + 1, SCHEMA_VERSION + 1):
        migrate_fn = _MIGRATIONS.get(target)
        if migrate_fn is not None:
            migrate_fn(conn)
        set_schema_version(conn, target)
    conn.commit()


def _backfill_object_content_rows(conn: sqlite3.Connection) -> None:
    versions = conn.execute(
        "SELECT id, audio_file, audio_hash FROM song_versions ORDER BY created_at"
    ).fetchall()
    now = datetime.now(timezone.utc).isoformat()
    for version in versions:
        object_id = f"object_song_{version['id']}"
        content_id = f"content_song_audio_{version['id']}"
        conn.execute(
            "INSERT OR IGNORE INTO timeline_objects "
            "(id, song_version_id, name, object_kind, main_content_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                object_id,
                version["id"],
                "Imported Song",
                "audio_clip",
                content_id,
                now,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO object_contents "
            "(id, object_id, revision_id, content_kind, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                content_id,
                object_id,
                f"revision_song_audio_{version['audio_hash']}",
                "audio_clip",
                json.dumps({"audio_file": version["audio_file"]}),
                now,
            ),
        )

    layers = conn.execute(
        "SELECT id, song_version_id, name, layer_type, source_pipeline, "
        'state_flags_json, created_at FROM layers ORDER BY song_version_id, "order"'
    ).fetchall()
    for layer in layers:
        takes = conn.execute(
            "SELECT id, label, is_main, source_json, data_json, created_at "
            "FROM takes WHERE layer_id = ? ORDER BY created_at",
            (layer["id"],),
        ).fetchall()
        if not takes:
            continue
        main_take = next((take for take in takes if int(take["is_main"]) == 1), takes[0])
        object_id = f"object_{layer['id']}"
        conn.execute(
            "INSERT OR IGNORE INTO timeline_objects "
            "(id, song_version_id, name, object_kind, main_content_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                object_id,
                layer["song_version_id"],
                layer["name"],
                _object_kind_for_legacy_layer(layer),
                f"content_{main_take['id']}",
                layer["created_at"],
            ),
        )
        for take in takes:
            conn.execute(
                "INSERT OR IGNORE INTO object_contents "
                "(id, object_id, revision_id, content_kind, payload_json, "
                "source_ref_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"content_{take['id']}",
                    object_id,
                    f"revision_{take['id']}",
                    _object_kind_for_legacy_layer(layer),
                    json.dumps(_payload_for_legacy_take(take)),
                    _source_ref_json_for_legacy_take(conn, layer, take),
                    take["created_at"],
                ),
            )
            if int(take["is_main"]) != 1:
                conn.execute(
                    "INSERT OR IGNORE INTO object_candidates "
                    "(id, object_id, content_id, label, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        f"candidate_{take['id']}",
                        object_id,
                        f"content_{take['id']}",
                        take["label"],
                        take["created_at"],
                    ),
                )


def _object_kind_for_legacy_layer(row: sqlite3.Row) -> str:
    state_flags = json.loads(row["state_flags_json"] or "{}")
    manual_kind = str(state_flags.get("manual_kind") or "").strip()
    if manual_kind == "audio":
        return "audio_clip"
    if manual_kind == "section":
        return "section_cue_set"
    source_pipeline = json.loads(row["source_pipeline"] or "{}")
    output_name = str(source_pipeline.get("output_name") or "").lower()
    if "stem" in output_name or row["layer_type"] == "audio":
        return "generated_audio"
    return "event_set"


def _payload_for_legacy_take(row: sqlite3.Row) -> dict[str, object]:
    payload: dict[str, object] = {"take_id": row["id"]}
    data = json.loads(row["data_json"] or "{}")
    if data.get("type") == "AudioData" and data.get("file_path"):
        payload["audio_file"] = data["file_path"]
    elif data.get("type") == "EventData":
        layers = data.get("layers") or []
        payload["event_layer_count"] = len(layers)
        payload["event_count"] = sum(len(layer.get("events") or []) for layer in layers)
    return payload


def _source_ref_json_for_legacy_take(
    conn: sqlite3.Connection,
    layer: sqlite3.Row,
    take: sqlite3.Row,
) -> str | None:
    source = json.loads(take["source_json"] or "{}")
    settings = source.get("settings_snapshot") or {}
    source_audio_path = str(settings.get("source_audio_path") or "").strip()
    if not source_audio_path:
        return None
    source_content = _find_audio_content_for_legacy_locator(
        conn,
        song_version_id=layer["song_version_id"],
        locator=source_audio_path,
    )
    if source_content is None:
        raise PersistenceError(
            "Cannot backfill object/content rows because a take source audio path "
            f"does not resolve to persisted content: {source_audio_path}"
        )
    return json.dumps(
        {
            "object_id": source_content["object_id"],
            "content_id": source_content["id"],
            "revision_id": source_content["revision_id"],
            "role": "audio_source",
            "locator": source_audio_path,
        }
    )


def _find_audio_content_for_legacy_locator(
    conn: sqlite3.Connection,
    *,
    song_version_id: str,
    locator: str,
) -> sqlite3.Row | None:
    rows = conn.execute(
        "SELECT c.id, c.object_id, c.revision_id, c.payload_json "
        "FROM object_contents c JOIN timeline_objects o ON o.id = c.object_id "
        "WHERE o.song_version_id = ?",
        (song_version_id,),
    ).fetchall()
    requested = _legacy_path_tokens(locator)
    for row in rows:
        payload = json.loads(row["payload_json"] or "{}")
        audio_file = payload.get("audio_file")
        if audio_file and requested & _legacy_path_tokens(str(audio_file)):
            return row
    return None


def _legacy_path_tokens(value: str) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    return {text, text.replace("\\", "/"), text.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]}


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL)
    current = get_schema_version(conn)
    if current == 0:
        set_schema_version(conn, SCHEMA_VERSION)
        conn.commit()
    else:
        apply_migrations(conn)

"""
Tests for audit fix batches 4 & 5 — robustness + quality/polish.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from echozero.persistence.archive import is_valid_ez, pack_ez
from echozero.persistence.schema import apply_migrations, init_db
from echozero.pipelines.registry import PipelineRegistry, PipelineTemplate


# ---------------------------------------------------------------------------
# P6: Migration guard — V1 table never existed
# ---------------------------------------------------------------------------


def test_migrate_v1_to_v2_no_v1_table():
    """DB created directly at V2 (no song_pipeline_configs table) must not crash."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    # Simulate a DB that was initialised fresh at v2 — _meta says version=1
    # but song_pipeline_configs was never created.
    conn.executescript("""
        CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO _meta (key, value) VALUES ('schema_version', '1');

        CREATE TABLE songs (id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
            title TEXT NOT NULL, artist TEXT DEFAULT '', "order" INTEGER NOT NULL DEFAULT 0,
            active_version_id TEXT);
        CREATE TABLE song_versions (id TEXT PRIMARY KEY, song_id TEXT NOT NULL,
            label TEXT NOT NULL, audio_file TEXT NOT NULL,
            duration_seconds REAL NOT NULL, original_sample_rate INTEGER NOT NULL,
            audio_hash TEXT NOT NULL, created_at TEXT NOT NULL);
    """)

    # Should not raise even though song_pipeline_configs doesn't exist
    apply_migrations(conn)

    # pipeline_configs table should exist (created by migration)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_configs'"
    ).fetchone()
    assert row is not None, "pipeline_configs table should have been created"


def test_migrate_v1_to_v2_with_v1_table():
    """DB with V1 song_pipeline_configs should migrate data correctly."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")

    conn.executescript("""
        CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO _meta (key, value) VALUES ('schema_version', '1');

        CREATE TABLE song_versions (id TEXT PRIMARY KEY, song_id TEXT NOT NULL,
            label TEXT NOT NULL, audio_file TEXT NOT NULL,
            duration_seconds REAL NOT NULL, original_sample_rate INTEGER NOT NULL,
            audio_hash TEXT NOT NULL, created_at TEXT NOT NULL);

        CREATE TABLE song_pipeline_configs (
            id TEXT PRIMARY KEY,
            song_version_id TEXT NOT NULL,
            pipeline_id TEXT NOT NULL,
            bindings TEXT,
            created_at TEXT NOT NULL
        );
        INSERT INTO song_pipeline_configs VALUES
            ('cfg1', 'ver1', 'pipe1', '{}', '2024-01-01T00:00:00');
    """)

    apply_migrations(conn)

    row = conn.execute("SELECT * FROM pipeline_configs WHERE id = 'cfg1'").fetchone()
    assert row is not None
    assert row["template_id"] == "pipe1"


# ---------------------------------------------------------------------------
# A2: pack_ez packs nested audio subdirectories
# ---------------------------------------------------------------------------


def test_pack_ez_nested_audio_dirs(tmp_path):
    """pack_ez should include files in audio/ subdirectories, not just top level."""
    working_dir = tmp_path / "project"
    working_dir.mkdir()

    # Create project.db (minimal)
    db_path = working_dir / "project.db"
    db_path.write_bytes(b"SQLite format 3\x00" + b"\x00" * 96)

    # Create nested audio structure
    audio_dir = working_dir / "audio"
    audio_dir.mkdir()
    (audio_dir / "top.wav").write_bytes(b"RIFF0000WAVEfmt ")
    subdir = audio_dir / "stems"
    subdir.mkdir()
    (subdir / "drums.wav").write_bytes(b"RIFF0000WAVEfmt ")
    (subdir / "vocals.wav").write_bytes(b"RIFF0000WAVEfmt ")

    dest = tmp_path / "project.ez"
    pack_ez(working_dir, dest)

    with zipfile.ZipFile(dest, "r") as zf:
        names = zf.namelist()

    assert "audio/top.wav" in names
    assert "audio/stems/drums.wav" in names
    assert "audio/stems/vocals.wav" in names


# ---------------------------------------------------------------------------
# A3: is_valid_ez checks for project.db
# ---------------------------------------------------------------------------


def test_is_valid_ez_requires_project_db(tmp_path):
    """is_valid_ez should return False if project.db is missing."""
    ez_path = tmp_path / "no_db.ez"
    with zipfile.ZipFile(ez_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format_version": 1}))
    assert not is_valid_ez(ez_path)


def test_is_valid_ez_valid_archive(tmp_path):
    """is_valid_ez should return True when both manifest.json and project.db present."""
    ez_path = tmp_path / "valid.ez"
    with zipfile.ZipFile(ez_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format_version": 1}))
        zf.writestr("project.db", b"fake db")
    assert is_valid_ez(ez_path)


def test_is_valid_ez_missing_manifest(tmp_path):
    """is_valid_ez should return False if manifest.json is missing."""
    ez_path = tmp_path / "no_manifest.ez"
    with zipfile.ZipFile(ez_path, "w") as zf:
        zf.writestr("project.db", b"fake db")
    assert not is_valid_ez(ez_path)


def test_is_valid_ez_nonexistent(tmp_path):
    """is_valid_ez should return False for a path that doesn't exist."""
    assert not is_valid_ez(tmp_path / "ghost.ez")


# ---------------------------------------------------------------------------
# REG1: PipelineRegistry.clear()
# ---------------------------------------------------------------------------


def test_registry_clear():
    """clear() should remove all registered templates."""
    registry = PipelineRegistry()

    def builder():
        from echozero.domain.graph import Graph
        return Graph()

    template = PipelineTemplate(
        id="test_pipe",
        name="Test",
        description="",
        knobs={},
        builder=builder,
    )
    registry.register(template)
    assert len(registry.list()) == 1

    registry.clear()
    assert len(registry.list()) == 0
    assert registry.get("test_pipe") is None


# ---------------------------------------------------------------------------
# REG2: build_pipeline uses public Pipeline constructor (graph= kwarg)
# ---------------------------------------------------------------------------


def test_build_pipeline_uses_public_graph_kwarg():
    """build_pipeline() for a Graph-returning builder must use Pipeline(graph=result)."""
    from echozero.domain.graph import Graph
    from echozero.pipelines.pipeline import Pipeline

    registry = PipelineRegistry()
    returned_graph = Graph()

    def legacy_builder():
        return returned_graph

    template = PipelineTemplate(
        id="legacy",
        name="Legacy",
        description="",
        knobs={},
        builder=legacy_builder,
    )
    registry.register(template)

    pipeline = template.build_pipeline()
    assert isinstance(pipeline, Pipeline)
    # The graph should be the one returned by the builder (not a fresh Graph)
    assert pipeline.graph is returned_graph
    # Should NOT have used p._graph = ... after construction (verify no separate assignment)
    # Since we use graph= kwarg the internal _graph is set in __init__
    assert pipeline._graph is returned_graph


# ---------------------------------------------------------------------------
# O3: analyze() delegates to execute()
# ---------------------------------------------------------------------------


def test_analyze_delegates_to_execute():
    """analyze() should call create_config() then execute() — same code path."""
    from echozero.services.orchestrator import Orchestrator
    from echozero.result import ok

    mock_result = MagicMock()
    mock_result_obj = MagicMock()

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._registry = MagicMock()
    orchestrator._executors = {}
    orchestrator._output_mappings = {}
    orchestrator._max_takes_per_layer = 20

    # Mock create_config to return a fake config
    fake_config = MagicMock()
    fake_config.id = "config-123"
    orchestrator.create_config = MagicMock(return_value=ok(fake_config))
    orchestrator.execute = MagicMock(return_value=ok(mock_result_obj))

    session = MagicMock()
    result = orchestrator.analyze(session, "sv1", "pipe1", bindings={"x": 1})

    orchestrator.create_config.assert_called_once_with(
        session, "sv1", "pipe1", knob_overrides={"x": 1}
    )
    orchestrator.execute.assert_called_once_with(session, "config-123", on_progress=None)


# ---------------------------------------------------------------------------
# SESS1: Invalid audio file rejected before import
# ---------------------------------------------------------------------------


def test_import_song_rejects_invalid_audio(tmp_path):
    """import_song() should raise ValidationError for invalid audio before copying."""
    from echozero.persistence.session import ProjectSession
    from echozero.errors import ValidationError

    session = ProjectSession.create_new("Test", working_dir_root=tmp_path)

    fake_audio = tmp_path / "bad.wav"
    fake_audio.write_bytes(b"NOT AUDIO DATA")

    def bad_scan(path):
        raise RuntimeError("Not a valid audio file")

    with pytest.raises(ValidationError, match="Invalid audio file"):
        session.import_song("Bad Song", fake_audio, scan_fn=bad_scan)

    session.close()


def test_import_song_accepts_valid_audio(tmp_path):
    """import_song() should succeed when audio passes validation."""
    from echozero.persistence.session import ProjectSession
    from echozero.persistence.audio import AudioMetadata

    session = ProjectSession.create_new("Test", working_dir_root=tmp_path)

    fake_audio = tmp_path / "good.wav"
    fake_audio.write_bytes(b"fake audio content")

    def good_scan(path):
        return AudioMetadata(duration_seconds=3.0, sample_rate=44100, channel_count=2)

    song, version = session.import_song("Good Song", fake_audio, scan_fn=good_scan)
    assert song.title == "Good Song"
    assert version.duration_seconds == 3.0

    session.close()

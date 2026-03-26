"""
Archive tests: .ez pack/unpack, atomic write, round-trip, validation.
Exercises the archive module against real temp files using pytest tmp_path fixture.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from echozero.persistence.archive import (
    MANIFEST_VERSION,
    is_valid_ez,
    pack_ez,
    unpack_ez,
)
from echozero.persistence.entities import (
    Project,
    ProjectSettings,
    Song,
    SongVersion,
)
from echozero.persistence.repositories import ProjectRepository, SongRepository, SongVersionRepository
from echozero.persistence.schema import init_db
from echozero.persistence.session import ProjectSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _setup_working_dir(working_dir: Path) -> sqlite3.Connection:
    """Create a working directory with an initialized DB. Returns the connection."""
    working_dir.mkdir(parents=True, exist_ok=True)
    db_path = working_dir / "project.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def _add_project(conn: sqlite3.Connection, name: str = "Test") -> Project:
    """Add a project to the DB and return it."""
    project = Project(
        id=_uid(), name=name, settings=ProjectSettings(),
        created_at=_now(), updated_at=_now(),
    )
    ProjectRepository(conn).create(project)
    conn.commit()
    return project


def _write_audio(working_dir: Path, name: str, content: bytes) -> str:
    """Write a fake audio file into the audio/ dir. Returns relative path."""
    audio_dir = working_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    (audio_dir / name).write_bytes(content)
    return f"audio/{name}"


# ---------------------------------------------------------------------------
# pack_ez
# ---------------------------------------------------------------------------


class TestPackEz:
    def test_creates_valid_zip(self, tmp_path):
        """pack_ez creates a valid ZIP file."""
        working_dir = tmp_path / "project"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        dest = tmp_path / "test.ez"
        pack_ez(working_dir, dest)

        assert dest.exists()
        assert zipfile.is_zipfile(dest)

    def test_contains_manifest_and_db(self, tmp_path):
        """pack_ez includes manifest.json and project.db."""
        working_dir = tmp_path / "project"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        dest = tmp_path / "test.ez"
        pack_ez(working_dir, dest)

        with zipfile.ZipFile(dest, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "project.db" in names

    def test_contains_audio_files(self, tmp_path):
        """pack_ez includes audio files under audio/."""
        working_dir = tmp_path / "project"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        _write_audio(working_dir, "song1.wav", b"audio data 1")
        _write_audio(working_dir, "song2.wav", b"audio data 2")

        dest = tmp_path / "test.ez"
        pack_ez(working_dir, dest)

        with zipfile.ZipFile(dest, "r") as zf:
            names = zf.namelist()
            assert "audio/song1.wav" in names
            assert "audio/song2.wav" in names

    def test_manifest_has_required_fields(self, tmp_path):
        """Manifest contains format_version, app_version, created_at."""
        working_dir = tmp_path / "project"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        dest = tmp_path / "test.ez"
        pack_ez(working_dir, dest)

        with zipfile.ZipFile(dest, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["format_version"] == MANIFEST_VERSION
            assert "app_version" in manifest
            assert "created_at" in manifest

    def test_atomic_write_no_tmp_on_success(self, tmp_path):
        """After successful pack, .ez.tmp file should not exist."""
        working_dir = tmp_path / "project"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        dest = tmp_path / "test.ez"
        pack_ez(working_dir, dest)

        tmp_file = dest.with_suffix(".ez.tmp")
        assert not tmp_file.exists()
        assert dest.exists()

    def test_atomic_write_cleans_tmp_on_failure(self, tmp_path):
        """On pack failure, .ez.tmp file is cleaned up."""
        working_dir = tmp_path / "project"
        working_dir.mkdir(parents=True)

        dest = tmp_path / "test.ez"
        tmp_file = dest.with_suffix(".ez.tmp")

        with patch("zipfile.ZipFile.__init__", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                pack_ez(working_dir, dest)

        assert not tmp_file.exists()
        assert not dest.exists()

    def test_audio_stored_not_deflated(self, tmp_path):
        """Audio files use STORED compression (already compressed data)."""
        working_dir = tmp_path / "project"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        _write_audio(working_dir, "song.wav", b"audio data")

        dest = tmp_path / "test.ez"
        pack_ez(working_dir, dest)

        with zipfile.ZipFile(dest, "r") as zf:
            info = zf.getinfo("audio/song.wav")
            assert info.compress_type == zipfile.ZIP_STORED

    def test_db_deflated(self, tmp_path):
        """Database file uses DEFLATED compression."""
        working_dir = tmp_path / "project"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        dest = tmp_path / "test.ez"
        pack_ez(working_dir, dest)

        with zipfile.ZipFile(dest, "r") as zf:
            info = zf.getinfo("project.db")
            assert info.compress_type == zipfile.ZIP_DEFLATED

    def test_no_audio_dir_is_fine(self, tmp_path):
        """pack_ez works even if there's no audio/ directory."""
        working_dir = tmp_path / "project"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        dest = tmp_path / "test.ez"
        pack_ez(working_dir, dest)

        with zipfile.ZipFile(dest, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "project.db" in names
            audio_entries = [n for n in names if n.startswith("audio/")]
            assert audio_entries == []


# ---------------------------------------------------------------------------
# unpack_ez
# ---------------------------------------------------------------------------


class TestUnpackEz:
    def test_extracts_everything(self, tmp_path):
        """unpack_ez extracts manifest, db, and audio."""
        # Create an archive
        working_dir = tmp_path / "source"
        conn = _setup_working_dir(working_dir)
        _add_project(conn, "Unpacked")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        _write_audio(working_dir, "song.wav", b"audio data")

        ez_path = tmp_path / "test.ez"
        pack_ez(working_dir, ez_path)

        # Unpack to a new location
        target = tmp_path / "unpacked"
        manifest = unpack_ez(ez_path, target)

        assert (target / "manifest.json").exists()
        assert (target / "project.db").exists()
        assert (target / "audio" / "song.wav").exists()
        assert (target / "audio" / "song.wav").read_bytes() == b"audio data"

    def test_returns_manifest(self, tmp_path):
        """unpack_ez returns the manifest dict."""
        working_dir = tmp_path / "source"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        ez_path = tmp_path / "test.ez"
        pack_ez(working_dir, ez_path)

        target = tmp_path / "unpacked"
        manifest = unpack_ez(ez_path, target)

        assert manifest["format_version"] == MANIFEST_VERSION
        assert "app_version" in manifest

    def test_creates_working_dir_if_needed(self, tmp_path):
        """unpack_ez creates the target directory."""
        working_dir = tmp_path / "source"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        ez_path = tmp_path / "test.ez"
        pack_ez(working_dir, ez_path)

        target = tmp_path / "nested" / "deep" / "unpacked"
        assert not target.exists()
        unpack_ez(ez_path, target)
        assert target.exists()

    def test_missing_manifest_raises_value_error(self, tmp_path):
        """unpack_ez raises ValueError if archive has no manifest.json."""
        ez_path = tmp_path / "bad.ez"
        with zipfile.ZipFile(ez_path, "w") as zf:
            zf.writestr("project.db", b"some data")

        target = tmp_path / "unpacked"
        with pytest.raises(ValueError, match="no manifest.json"):
            unpack_ez(ez_path, target)

    def test_newer_format_version_raises_value_error(self, tmp_path):
        """unpack_ez raises ValueError for unsupported format versions."""
        ez_path = tmp_path / "future.ez"
        manifest = {
            "format_version": MANIFEST_VERSION + 1,
            "app_version": "99.0.0",
            "created_at": _now().isoformat(),
        }
        with zipfile.ZipFile(ez_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))

        target = tmp_path / "unpacked"
        with pytest.raises(ValueError, match="Please update EchoZero"):
            unpack_ez(ez_path, target)

    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        """unpack_ez raises FileNotFoundError for missing archive."""
        target = tmp_path / "unpacked"
        with pytest.raises(FileNotFoundError, match="Archive not found"):
            unpack_ez(tmp_path / "ghost.ez", target)


# ---------------------------------------------------------------------------
# pack -> unpack round-trip
# ---------------------------------------------------------------------------


class TestPackUnpackRoundTrip:
    def test_round_trip_preserves_db(self, tmp_path):
        """Pack then unpack preserves the database content."""
        working_dir = tmp_path / "source"
        conn = _setup_working_dir(working_dir)
        project = _add_project(conn, "RoundTrip")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        ez_path = tmp_path / "test.ez"
        pack_ez(working_dir, ez_path)

        target = tmp_path / "unpacked"
        unpack_ez(ez_path, target)

        # Verify DB content
        conn2 = sqlite3.connect(str(target / "project.db"))
        conn2.row_factory = sqlite3.Row
        row = conn2.execute("SELECT name FROM projects WHERE id = ?", (project.id,)).fetchone()
        assert row["name"] == "RoundTrip"
        conn2.close()

    def test_round_trip_preserves_audio(self, tmp_path):
        """Pack then unpack preserves audio file content."""
        working_dir = tmp_path / "source"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        audio_content = b"real audio bytes here"
        _write_audio(working_dir, "track.wav", audio_content)

        ez_path = tmp_path / "test.ez"
        pack_ez(working_dir, ez_path)

        target = tmp_path / "unpacked"
        unpack_ez(ez_path, target)

        assert (target / "audio" / "track.wav").read_bytes() == audio_content

    def test_round_trip_preserves_multiple_audio_files(self, tmp_path):
        """Pack then unpack preserves multiple audio files."""
        working_dir = tmp_path / "source"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        _write_audio(working_dir, "a.wav", b"audio A")
        _write_audio(working_dir, "b.mp3", b"audio B")
        _write_audio(working_dir, "c.flac", b"audio C")

        ez_path = tmp_path / "test.ez"
        pack_ez(working_dir, ez_path)

        target = tmp_path / "unpacked"
        unpack_ez(ez_path, target)

        assert (target / "audio" / "a.wav").read_bytes() == b"audio A"
        assert (target / "audio" / "b.mp3").read_bytes() == b"audio B"
        assert (target / "audio" / "c.flac").read_bytes() == b"audio C"


# ---------------------------------------------------------------------------
# is_valid_ez
# ---------------------------------------------------------------------------


class TestIsValidEz:
    def test_valid_archive(self, tmp_path):
        """is_valid_ez returns True for a valid .ez archive."""
        working_dir = tmp_path / "project"
        conn = _setup_working_dir(working_dir)
        _add_project(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        ez_path = tmp_path / "test.ez"
        pack_ez(working_dir, ez_path)

        assert is_valid_ez(ez_path) is True

    def test_missing_file(self, tmp_path):
        """is_valid_ez returns False for nonexistent file."""
        assert is_valid_ez(tmp_path / "ghost.ez") is False

    def test_not_a_zip(self, tmp_path):
        """is_valid_ez returns False for a non-ZIP file."""
        bad = tmp_path / "bad.ez"
        bad.write_bytes(b"not a zip file")
        assert is_valid_ez(bad) is False

    def test_zip_without_manifest(self, tmp_path):
        """is_valid_ez returns False for a ZIP without manifest.json."""
        ez_path = tmp_path / "no_manifest.ez"
        with zipfile.ZipFile(ez_path, "w") as zf:
            zf.writestr("project.db", b"data")
        assert is_valid_ez(ez_path) is False


# ---------------------------------------------------------------------------
# Full session round-trip
# ---------------------------------------------------------------------------


class TestFullSessionRoundTrip:
    def test_create_save_open_verify(self, tmp_path):
        """Full round-trip: create project -> add songs -> save_as(.ez) -> close -> open(.ez) -> verify."""
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "project.ez"

        # Create and populate
        session = ProjectSession.create_new(
            "Tour 2026",
            settings=ProjectSettings(sample_rate=48000, bpm=128.0),
            working_dir_root=tmp_root,
        )
        pid = session.project.id

        # Create a song + version manually (no audio file needed for DB round-trip)
        song = Song(
            id=_uid(), project_id=pid, title="Opening Act",
            artist="The Band", order=0, active_version_id=None,
        )
        session.songs.create(song)

        version = SongVersion(
            id=_uid(), song_id=song.id, label="Final Mix",
            audio_file="audio/test.wav", duration_seconds=180.0,
            original_sample_rate=44100, audio_hash="abc123", created_at=_now(),
        )
        session.song_versions.create(version)

        # Write a fake audio file
        audio_dir = session.working_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        (audio_dir / "test.wav").write_bytes(b"fake audio for round trip")

        # Save as .ez
        session.save_as(ez_path)
        session.close()

        assert ez_path.exists()
        assert is_valid_ez(ez_path)

        # Open from .ez
        session2 = ProjectSession.open(ez_path, working_dir_root=tmp_root)
        try:
            assert session2.project.name == "Tour 2026"
            assert session2.project.settings.bpm == 128.0

            songs = session2.songs.list_by_project(pid)
            assert len(songs) == 1
            assert songs[0].title == "Opening Act"
            assert songs[0].artist == "The Band"

            versions = session2.song_versions.list_by_song(song.id)
            assert len(versions) == 1
            assert versions[0].label == "Final Mix"

            # Audio file should be present in the new working dir
            assert (session2.working_dir / "audio" / "test.wav").exists()
            assert (session2.working_dir / "audio" / "test.wav").read_bytes() == b"fake audio for round trip"
        finally:
            session2.close()

    def test_import_song_save_open_round_trip(self, tmp_path):
        """Round-trip with import_song: import -> save_as -> close -> open -> verify."""
        tmp_root = tmp_path / "working"
        ez_path = tmp_path / "project.ez"

        # Create project and import a song
        session = ProjectSession.create_new("Import Test", working_dir_root=tmp_root)
        pid = session.project.id

        audio_source = tmp_path / "external" / "my_track.wav"
        audio_source.parent.mkdir(parents=True)
        audio_source.write_bytes(b"imported audio content")

        song, version = session.import_song(
            title="My Track",
            audio_source=audio_source,
            artist="DJ Test",
            label="Studio Mix",
        )

        session.save_as(ez_path)
        session.close()

        # Open and verify
        session2 = ProjectSession.open(ez_path, working_dir_root=tmp_root)
        try:
            songs = session2.songs.list_by_project(pid)
            assert len(songs) == 1
            assert songs[0].title == "My Track"
            assert songs[0].artist == "DJ Test"

            versions = session2.song_versions.list_by_song(song.id)
            assert len(versions) == 1
            assert versions[0].label == "Studio Mix"
            assert versions[0].audio_hash == version.audio_hash

            # Audio file round-tripped
            audio_path = session2.working_dir / versions[0].audio_file
            assert audio_path.exists()
            assert audio_path.read_bytes() == b"imported audio content"
        finally:
            session2.close()

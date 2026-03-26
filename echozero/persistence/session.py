"""
ProjectSession: Lifecycle manager for an EchoZero project's working directory and SQLite DB.
Exists because projects need a single entry point for open/save/close with autosave, dirty
tracking, crash recovery, and convenient access to all repositories. The working directory
pattern enables crash recovery — closing does NOT delete the working dir, so stale dirs
can be detected and recovered on next open.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from echozero.domain.graph import Graph
from echozero.event_bus import EventBus
from echozero.persistence.dirty import DirtyTracker
from echozero.persistence.entities import Project, ProjectSettings, Song, SongVersion
from echozero.persistence.repositories import (
    LayerRepository,
    PipelineConfigRepository,
    ProjectRepository,
    SongRepository,
    SongVersionRepository,
    TakeRepository,
)
from echozero.persistence.schema import init_db
from echozero.serialization import deserialize_graph, serialize_graph

logger = logging.getLogger(__name__)

WORKING_DIR_ROOT: Path = Path.home() / ".echozero" / "working"


def _setup_connection(conn: sqlite3.Connection) -> None:
    """Configure a connection with WAL mode, foreign keys, and Row factory."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row


def _working_dir_for_path(ez_path: Path, root: Path | None = None) -> Path:
    """Derive a working directory from an .ez file path using sha256[:16]."""
    canonical = str(ez_path.resolve())
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return (root or WORKING_DIR_ROOT) / digest


def _working_dir_for_id(project_id: str) -> Path:
    """Derive a working directory from a project ID (new projects)."""
    return WORKING_DIR_ROOT / project_id


class ProjectSession:
    """Main project lifecycle manager — owns DB connection, repos, dirty tracker, autosave."""

    def __init__(
        self,
        project: Project,
        working_dir: Path,
        db: sqlite3.Connection,
        dirty_tracker: DirtyTracker,
        event_bus: EventBus | None = None,
    ) -> None:
        self.project = project
        self.working_dir = working_dir
        self.db = db
        self.dirty_tracker = dirty_tracker
        self._event_bus = event_bus
        self._closed = False
        self._lock = threading.RLock()
        self._autosave_timer: threading.Timer | None = None
        self._autosave_interval: float = 0.0
        self._autosave_lock = threading.Lock()

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def create_new(
        cls,
        name: str,
        settings: ProjectSettings | None = None,
        event_bus: EventBus | None = None,
        working_dir_root: Path | None = None,
    ) -> ProjectSession:
        """Create a brand new project. Sets up working dir + DB."""
        project_id = uuid.uuid4().hex
        root = working_dir_root or WORKING_DIR_ROOT
        working_dir = root / project_id
        working_dir.mkdir(parents=True, exist_ok=True)

        db_path = working_dir / "project.db"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _setup_connection(conn)
        init_db(conn)

        now = datetime.now(timezone.utc)
        project = Project(
            id=project_id,
            name=name,
            settings=settings or ProjectSettings(),
            created_at=now,
            updated_at=now,
        )

        dirty_tracker = DirtyTracker(event_bus)
        session = cls(project, working_dir, conn, dirty_tracker, event_bus)

        # Persist the initial project row
        ProjectRepository(conn).create(project)
        conn.commit()

        return session

    @classmethod
    def open_db(
        cls,
        working_dir: Path,
        event_bus: EventBus | None = None,
    ) -> ProjectSession:
        """Open directly from a working directory that already has a project.db."""
        db_path = working_dir / "project.db"
        if not db_path.exists():
            raise FileNotFoundError(f"No project.db found in {working_dir}")

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _setup_connection(conn)

        repo = ProjectRepository(conn)
        projects = repo.list()
        if not projects:
            conn.close()
            raise ValueError(f"No project found in database at {db_path}")
        project = projects[0]

        dirty_tracker = DirtyTracker(event_bus)
        return cls(project, working_dir, conn, dirty_tracker, event_bus)

    @classmethod
    def open(
        cls,
        ez_path: Path,
        event_bus: EventBus | None = None,
        working_dir_root: Path | None = None,
    ) -> ProjectSession:
        """Open an existing project from an .ez file path.

        If the working directory already exists with a project.db (recovery scenario),
        opens it directly. Otherwise, unpacks the .ez archive first.
        """
        root = working_dir_root or WORKING_DIR_ROOT
        working_dir = _working_dir_for_path(ez_path, root)

        if not (working_dir / "project.db").exists():
            # Fresh open — unpack the archive
            from echozero.persistence.archive import unpack_ez

            unpack_ez(ez_path, working_dir)

        return cls.open_db(working_dir, event_bus)

    # -- Transaction control ------------------------------------------------

    @contextmanager
    def transaction(self):
        """Execute multiple operations atomically. Commits on success, rolls back on exception."""
        with self._lock:
            self._check_closed()
            try:
                yield self
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    def commit(self) -> None:
        """Explicitly commit pending changes."""
        with self._lock:
            self._check_closed()
            self.db.commit()

    # -- Save / close -------------------------------------------------------

    def save(self) -> None:
        """Flush any pending changes to the SQLite DB. Clears dirty flag."""
        with self._lock:
            self._check_closed()
            self.db.commit()
            self.dirty_tracker.clear()

    def save_as(self, ez_path: Path) -> None:
        """Save project to .ez archive. Atomic write."""
        self._check_closed()
        with self._lock:
            from echozero.persistence.archive import pack_ez

            # Commit any pending changes first
            self.db.commit()
            # WAL checkpoint before packing to ensure DB is fully written
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            pack_ez(self.working_dir, ez_path)
            self.dirty_tracker.clear()

    def close(self) -> None:
        """Close the DB connection. Does NOT delete the working dir (crash recovery)."""
        self.stop_autosave()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.dirty_tracker._unsubscribe()
            try:
                self.db.close()
            except Exception as exc:
                logger.debug("Error closing database: %r", exc)

    def _check_closed(self) -> None:
        """Raise if the session has been closed."""
        if self._closed:
            raise RuntimeError("ProjectSession is closed")

    def is_dirty(self) -> bool:
        """Whether there are unsaved changes."""
        return self.dirty_tracker.is_dirty()

    # -- Context manager ----------------------------------------------------

    def __enter__(self) -> ProjectSession:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- Repository accessors -----------------------------------------------

    @property
    def projects(self) -> ProjectRepository:
        """Access the project repository."""
        self._check_closed()
        return ProjectRepository(self.db)

    @property
    def songs(self) -> SongRepository:
        """Access the song repository."""
        self._check_closed()
        return SongRepository(self.db)

    @property
    def song_versions(self) -> SongVersionRepository:
        """Access the song version repository."""
        self._check_closed()
        return SongVersionRepository(self.db)

    @property
    def layers(self) -> LayerRepository:
        """Access the layer repository."""
        self._check_closed()
        return LayerRepository(self.db)

    @property
    def takes(self) -> TakeRepository:
        """Access the take repository."""
        self._check_closed()
        return TakeRepository(self.db)

    @property
    def pipeline_configs(self) -> PipelineConfigRepository:
        """Access the pipeline config repository."""
        self._check_closed()
        return PipelineConfigRepository(self.db)

    # -- Graph persistence --------------------------------------------------

    def save_graph(self, graph: Graph) -> None:
        """Serialize graph to JSON and store in the project row."""
        with self._lock:
            self._check_closed()
            graph_json = json.dumps(serialize_graph(graph))
            self.db.execute(
                "UPDATE projects SET graph_json = ? WHERE id = ?",
                (graph_json, self.project.id),
            )
            self.dirty_tracker.mark_dirty(self.project.id)

    def load_graph(self) -> Graph | None:
        """Load graph from project row, or None if not set."""
        with self._lock:
            self._check_closed()
            row = self.db.execute(
                "SELECT graph_json FROM projects WHERE id = ?",
                (self.project.id,),
            ).fetchone()
            if row is None or row['graph_json'] is None:
                return None
            return deserialize_graph(json.loads(row['graph_json']))

    # -- Audio import -------------------------------------------------------

    def import_song(
        self,
        title: str,
        audio_source: Path,
        artist: str = "",
        label: str = "Original",
    ) -> tuple[Song, SongVersion]:
        """Import an audio file as a new song. Copies audio, creates Song + SongVersion."""
        self._check_closed()
        from echozero.persistence.audio import import_audio

        with self._lock:
            # Import audio (content-addressed copy)
            audio_rel_path, audio_hash = import_audio(audio_source, self.working_dir)

            song_id = uuid.uuid4().hex
            version_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc)

            song = Song(
                id=song_id,
                project_id=self.project.id,
                title=title,
                artist=artist,
                order=len(self.songs.list_by_project(self.project.id)),
                active_version_id=version_id,
            )

            version = SongVersion(
                id=version_id,
                song_id=song_id,
                label=label,
                audio_file=audio_rel_path,
                duration_seconds=0.0,
                original_sample_rate=0,
                audio_hash=audio_hash,
                created_at=now,
            )

            self.songs.create(song)
            self.song_versions.create(version)
            self.db.commit()
            self.dirty_tracker.mark_dirty(song_id)

            return song, version

    # -- Autosave -----------------------------------------------------------

    def start_autosave(self, interval_seconds: float = 30.0) -> None:
        """Start background autosave timer."""
        self._check_closed()
        self._autosave_interval = interval_seconds
        self._schedule_autosave()

    def stop_autosave(self) -> None:
        """Stop background autosave timer."""
        with self._autosave_lock:
            if self._autosave_timer is not None:
                self._autosave_timer.cancel()
                self._autosave_timer = None

    def _schedule_autosave(self) -> None:
        """Schedule the next autosave tick."""
        with self._autosave_lock:
            if self._closed:
                return
            self._autosave_timer = threading.Timer(
                self._autosave_interval, self._autosave_tick
            )
            self._autosave_timer.daemon = True
            self._autosave_timer.start()

    def _autosave_tick(self) -> None:
        """Called by the timer — commit if dirty, then reschedule."""
        try:
            if not self._closed and self.dirty_tracker.is_dirty():
                with self._lock:
                    self.db.commit()
                    self.dirty_tracker.clear()
                logger.debug("Autosave: committed changes for project %s", self.project.id)
        except Exception as exc:
            logger.warning("Autosave failed: %r", exc)
        finally:
            if not self._closed:
                self._schedule_autosave()

    # -- Crash recovery -----------------------------------------------------

    @staticmethod
    def check_recovery(
        ez_path: Path,
        working_dir_root: Path | None = None,
    ) -> bool:
        """Check if a working dir exists with a project.db for this .ez file."""
        root = working_dir_root or WORKING_DIR_ROOT
        working_dir = root / hashlib.sha256(
            str(ez_path.resolve()).encode()
        ).hexdigest()[:16]
        return (working_dir / "project.db").exists()

    @staticmethod
    def recover(
        ez_path: Path,
        event_bus: EventBus | None = None,
        working_dir_root: Path | None = None,
    ) -> ProjectSession:
        """Open the existing working dir for recovery instead of unpacking fresh."""
        root = working_dir_root or WORKING_DIR_ROOT
        working_dir = root / hashlib.sha256(
            str(ez_path.resolve()).encode()
        ).hexdigest()[:16]
        return ProjectSession.open_db(working_dir, event_bus)

    @staticmethod
    def discard_recovery(
        ez_path: Path,
        working_dir_root: Path | None = None,
    ) -> None:
        """Delete the stale working dir."""
        import shutil

        root = working_dir_root or WORKING_DIR_ROOT
        working_dir = root / hashlib.sha256(
            str(ez_path.resolve()).encode()
        ).hexdigest()[:16]
        if working_dir.exists():
            shutil.rmtree(working_dir)

"""Project storage session manager for EchoZero projects.
Exists to own project open/save/close lifecycle, autosave state, and repository access.
Connects working-directory persistence and SQLite repositories to application project flows.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from echozero.domain.graph import Graph
from echozero.event_bus import EventBus
from echozero.persistence.dirty import DirtyTracker
from echozero.persistence.entities import (
    PipelineConfigRecord,
    ProjectRecord,
    ProjectSettingsRecord,
    SongDefaultPipelineConfigRecord,
    SongRecord,
    SongVersionRecord,
)
from echozero.persistence.repositories import (
    LayerRepository,
    PipelineConfigRepository,
    ProjectRepository,
    SongDefaultPipelineConfigRepository,
    SongRepository,
    SongVersionRepository,
    TakeRepository,
    TimelineRegionRepository,
)
from echozero.persistence.schema import init_db
from echozero.persistence.session_runtime_mixin import ProjectStorageRuntimeMixin
from echozero.persistence.session_versioning_mixin import ProjectStorageVersioningMixin
from echozero.serialization import deserialize_graph, serialize_graph

logger = logging.getLogger(__name__)


def _default_working_dir_root() -> Path:
    """Resolve canonical working-dir root for local app data."""
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "EchoZero" / "working"
    return Path.home() / ".echozero" / "working"


WORKING_DIR_ROOT: Path = _default_working_dir_root()


def _is_pid_alive(pid: int) -> bool:
    """Check if a process is still running. Cross-platform."""
    if os.name == "nt":
        # Windows: use ctypes OpenProcess with SYNCHRONIZE access
        SYNCHRONIZE = 0x100000
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _acquire_project_lock(working_dir: Path) -> Path:
    """Acquire a lockfile. Raises RuntimeError if project is already open."""
    lock_path = working_dir / "project.lock"
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
            if _is_pid_alive(old_pid):
                raise RuntimeError(
                    f"ProjectRecord is already open by process {old_pid}. "
                    f"Close it first, or delete {lock_path} if the process crashed."
                )
            else:
                # Process is dead — stale lock
                logger.warning("Removing stale lock from process %d", old_pid)
        except (ValueError, OSError):
            # Corrupt lock file — remove it
            pass
    lock_path.write_text(str(os.getpid()))
    return lock_path


def _release_project_lock(lock_path: Path | None) -> None:
    """Release the lockfile."""
    if lock_path is not None and lock_path.exists():
        try:
            lock_path.unlink()
        except OSError:
            pass


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


class ProjectStorage(ProjectStorageVersioningMixin, ProjectStorageRuntimeMixin):
    """Main project lifecycle manager — owns DB connection, repos, dirty tracker, autosave."""

    def __init__(
        self,
        project: ProjectRecord,
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
        self._lockfile_path: Path | None = None
        self._in_transaction: bool = False

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def create_new(
        cls,
        name: str,
        settings: ProjectSettingsRecord | None = None,
        event_bus: EventBus | None = None,
        working_dir_root: Path | None = None,
    ) -> ProjectStorage:
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
        project = ProjectRecord(
            id=project_id,
            name=name,
            settings=settings or ProjectSettingsRecord(),
            created_at=now,
            updated_at=now,
        )

        dirty_tracker = DirtyTracker(event_bus)
        session = cls(project, working_dir, conn, dirty_tracker, event_bus)
        session._lockfile_path = _acquire_project_lock(working_dir)

        # Persist the initial project row
        ProjectRepository(conn).create(project)
        conn.commit()

        return session

    @classmethod
    def open_db(
        cls,
        working_dir: Path,
        event_bus: EventBus | None = None,
    ) -> ProjectStorage:
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
        session = cls(project, working_dir, conn, dirty_tracker, event_bus)
        session._lockfile_path = _acquire_project_lock(working_dir)
        return session

    @classmethod
    def open(
        cls,
        ez_path: Path,
        event_bus: EventBus | None = None,
        working_dir_root: Path | None = None,
    ) -> ProjectStorage:
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
    def transaction(self) -> Iterator[ProjectStorage]:
        """Execute multiple operations atomically. Commits on success, rolls back on exception."""
        with self._lock:
            self._check_closed()
            self._in_transaction = True
            try:
                yield self
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            finally:
                self._in_transaction = False

    @contextmanager
    def locked(self) -> Iterator[ProjectStorage]:
        """Serialize direct DB access without starting a transaction."""
        with self._lock:
            self._check_closed()
            yield self

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
        with self._lock:
            self._check_closed()
            from echozero.persistence.archive import pack_ez

            # Commit any pending changes first
            self.db.commit()
            # WAL checkpoint before packing to ensure DB is fully written
            result = self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result and result[0] != 0:
                logger.warning(
                    "WAL checkpoint returned non-zero status %d — "
                    "archive may not include latest changes",
                    result[0],
                )
            pack_ez(self.working_dir, ez_path)
            self.dirty_tracker.clear()

    def close(self) -> None:
        """Close the DB connection. Does NOT delete the working dir (crash recovery)."""
        self.stop_autosave()
        with self._lock:
            if self._closed:
                return
            _release_project_lock(self._lockfile_path)
            self._lockfile_path = None
            self._closed = True
            self.dirty_tracker._unsubscribe()
            try:
                self.db.close()
            except Exception as exc:
                logger.debug("Error closing database: %r", exc)

    def _check_closed(self) -> None:
        """Raise if the session has been closed."""
        if self._closed:
            raise RuntimeError("ProjectStorage is closed")

    def is_dirty(self) -> bool:
        """Whether there are unsaved changes."""
        return bool(self.dirty_tracker.is_dirty())

    # -- Context manager ----------------------------------------------------

    def __enter__(self) -> ProjectStorage:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
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

    @property
    def timeline_regions(self) -> TimelineRegionRepository:
        """Access the timeline region repository."""
        self._check_closed()
        return TimelineRegionRepository(self.db)

    @property
    def song_default_pipeline_configs(self) -> SongDefaultPipelineConfigRepository:
        """Access the song default pipeline config repository."""
        self._check_closed()
        return SongDefaultPipelineConfigRepository(self.db)

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
            if row is None or row["graph_json"] is None:
                return None
            return deserialize_graph(json.loads(row["graph_json"]))

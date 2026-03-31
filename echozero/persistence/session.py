"""
ProjectStorage: Lifecycle manager for an EchoZero project's working directory and SQLite DB.
Exists because projects need a single entry point for open/save/close with autosave, dirty
tracking, crash recovery, and convenient access to all repositories. The working directory
pattern enables crash recovery — closing does NOT delete the working dir, so stale dirs
can be detected and recovered on next open.
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from echozero.domain.graph import Graph
from echozero.event_bus import EventBus
from echozero.persistence.dirty import DirtyTracker
from echozero.persistence.entities import ProjectRecord, ProjectSettingsRecord, SongRecord, SongVersionRecord
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


class ProjectStorage:
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
    def transaction(self):
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
        return self.dirty_tracker.is_dirty()

    # -- Context manager ----------------------------------------------------

    def __enter__(self) -> ProjectStorage:
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

    def _create_version(
        self,
        song_id: str,
        audio_source: Path,
        label: str,
        scan_fn=None,
    ) -> SongVersionRecord:
        """Shared version factory: import audio, scan metadata, create SongVersionRecord.

        This is the single path for all audio → SongVersionRecord creation.
        Both import_song and add_song_version call this.

        Args:
            song_id: SongRecord this version belongs to.
            audio_source: Path to the audio file on disk.
            label: Human-readable version label.
            scan_fn: Optional injectable for testing (skips real audio scanning).

        Returns:
            The newly created and persisted SongVersionRecord with real metadata.
        """
        from echozero.persistence.audio import import_audio, scan_audio_metadata
        from echozero.errors import ValidationError

        # Validate audio file BEFORE copying into the project
        try:
            scan_audio_metadata(audio_source, scan_fn=scan_fn)
        except Exception as exc:
            raise ValidationError(
                f"Invalid audio file '{audio_source.name}': {exc}"
            ) from exc

        audio_rel_path, audio_hash = import_audio(audio_source, self.working_dir)

        # Scan real metadata from the copied file
        full_audio_path = self.working_dir / audio_rel_path
        metadata = scan_audio_metadata(full_audio_path, scan_fn=scan_fn)

        version = SongVersionRecord(
            id=uuid.uuid4().hex,
            song_id=song_id,
            label=label,
            audio_file=audio_rel_path,
            duration_seconds=metadata.duration_seconds,
            original_sample_rate=metadata.sample_rate,
            audio_hash=audio_hash,
            created_at=datetime.now(timezone.utc),
        )
        self.song_versions.create(version)
        return version

    def import_song(
        self,
        title: str,
        audio_source: Path,
        artist: str = "",
        label: str = "Original",
        default_templates: list[str] | None = None,
        scan_fn=None,
    ) -> tuple[SongRecord, SongVersionRecord]:
        """Import an audio file as a new song with default pipeline configs.

        Creates SongRecord + SongVersionRecord + default PipelineConfigs from registered templates.

        Args:
            title: SongRecord title.
            audio_source: Path to the audio file.
            artist: Artist name (optional).
            label: Version label (default "Original").
            default_templates: Template IDs to create configs for. If None, uses all
                registered templates. Pass [] to skip default config creation.
            scan_fn: Optional injectable for audio scanning (testing).

        Returns:
            (SongRecord, SongVersionRecord) tuple.
        """
        self._check_closed()

        with self._lock:
            song_id = uuid.uuid4().hex

            # SongRecord must exist before version (FK constraint)
            song = SongRecord(
                id=song_id,
                project_id=self.project.id,
                title=title,
                artist=artist,
                order=len(self.songs.list_by_project(self.project.id)),
                active_version_id=None,  # set after version creation
            )
            self.songs.create(song)

            # Create version (imports audio, scans metadata, persists)
            version = self._create_version(song_id, audio_source, label, scan_fn)

            # Point song to the new version
            from dataclasses import replace as _replace
            updated_song = _replace(song, active_version_id=version.id)
            self.songs.update(updated_song)

            # Create default pipeline configs from templates
            self._apply_default_templates(version.id, default_templates)

            self.db.commit()
            self.dirty_tracker.mark_dirty(song_id)

            return updated_song, version

    def _apply_default_templates(
        self,
        song_version_id: str,
        template_ids: list[str] | None = None,
    ) -> None:
        """Create PipelineConfigs from registered templates for a new version.

        Args:
            song_version_id: The version to attach configs to.
            template_ids: Specific template IDs. None = all registered. [] = none.
        """
        from echozero.pipelines.registry import get_registry
        from echozero.persistence.entities import PipelineConfigRecord

        registry = get_registry()

        if template_ids is None:
            templates = registry.list()
        else:
            templates = [t for tid in template_ids if (t := registry.get(tid)) is not None]

        for template in templates:
            # Build pipeline with default knob values
            pipeline = template.build_pipeline()

            config = PipelineConfigRecord.from_pipeline(
                pipeline,
                template_id=template.id,
                song_version_id=song_version_id,
                knob_values={k: v.default for k, v in template.knobs.items()},
                name=template.name,
            )
            self.pipeline_configs.create(config)

    def add_song_version(
        self,
        song_id: str,
        audio_source: Path,
        label: str | None = None,
        activate: bool = True,
        scan_fn=None,
    ) -> SongVersionRecord:
        """Add a new version of an existing song and copy all pipeline configs.

        This is the D278 "Update Track" flow:
        1. Import + scan the new audio file.
        2. Create a new SongVersionRecord with real metadata.
        3. Copy all PipelineConfigs from the current active version → new version
           (same graph, same knobs, same block overrides — but new IDs).
        4. Optionally set the new version as active.

        Args:
            song_id: The existing song to add a version to.
            audio_source: Path to the new audio file.
            label: Human-readable label (e.g. "Festival Edit"). Auto-generated if None.
            activate: If True, set the new version as the song's active version.
            scan_fn: Optional injectable for audio scanning (testing).

        Returns:
            The newly created SongVersionRecord.

        Raises:
            ValueError: If song_id not found or song has no active version.
        """
        self._check_closed()
        from dataclasses import replace as _replace

        with self._lock:
            song = self.songs.get(song_id)
            if song is None:
                raise ValueError(f"SongRecord not found: {song_id}")

            source_version_id = song.active_version_id
            if source_version_id is None:
                raise ValueError(
                    f"SongRecord '{song_id}' has no active version to copy configs from"
                )

            # Auto-generate label
            if label is None:
                existing = self.song_versions.list_by_song(song_id)
                label = f"v{len(existing) + 1}"

            # Create version (imports audio, scans metadata)
            version = self._create_version(song_id, audio_source, label, scan_fn)

            # Copy all pipeline configs from source version
            now = datetime.now(timezone.utc)
            for config in self.pipeline_configs.list_by_version(source_version_id):
                new_config = _replace(
                    config,
                    id=uuid.uuid4().hex,
                    song_version_id=version.id,
                    created_at=now,
                    updated_at=now,
                )
                self.pipeline_configs.create(new_config)

            if activate:
                self.songs.update(_replace(song, active_version_id=version.id))

            self.db.commit()
            self.dirty_tracker.mark_dirty(song_id)

            return version

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
                    if self._in_transaction:
                        # Don't commit during an active transaction — it would
                        # commit a partial set of changes.
                        return
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
    ) -> ProjectStorage:
        """Open the existing working dir for recovery instead of unpacking fresh."""
        root = working_dir_root or WORKING_DIR_ROOT
        working_dir = root / hashlib.sha256(
            str(ez_path.resolve()).encode()
        ).hexdigest()[:16]
        return ProjectStorage.open_db(working_dir, event_bus)

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
            # Release any stale lockfile before removing
            _release_project_lock(working_dir / "project.lock")
            shutil.rmtree(working_dir)

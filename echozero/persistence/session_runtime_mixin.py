"""Autosave and crash-recovery helpers for project storage.
Exists to keep timer-driven persistence and recovery path helpers out of the ProjectStorage root.
Connects working-directory lifecycle state to autosave and recovery workflows.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sqlite3
import threading
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING, Protocol, cast

from echozero.event_bus import EventBus
from echozero.persistence.dirty import DirtyTracker
from echozero.persistence.entities import ProjectRecord

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from echozero.persistence.session import ProjectStorage


class _ProjectStorageRuntimeHost(Protocol):
    project: ProjectRecord
    db: sqlite3.Connection
    dirty_tracker: DirtyTracker
    _closed: bool
    _in_transaction: bool
    _lock: RLock
    _autosave_timer: threading.Timer | None
    _autosave_interval: float
    _autosave_lock: Lock

    def _check_closed(self) -> None: ...


class ProjectStorageRuntimeMixin:
    def start_autosave(self, interval_seconds: float = 30.0) -> None:
        host = cast(_ProjectStorageRuntimeHost, self)
        host._check_closed()
        host._autosave_interval = interval_seconds
        self._schedule_autosave()

    def stop_autosave(self) -> None:
        host = cast(_ProjectStorageRuntimeHost, self)
        with host._autosave_lock:
            if host._autosave_timer is not None:
                host._autosave_timer.cancel()
                host._autosave_timer = None

    def _schedule_autosave(self) -> None:
        host = cast(_ProjectStorageRuntimeHost, self)
        with host._autosave_lock:
            if host._closed:
                return
            host._autosave_timer = threading.Timer(host._autosave_interval, self._autosave_tick)
            host._autosave_timer.daemon = True
            host._autosave_timer.start()

    def _autosave_tick(self) -> None:
        host = cast(_ProjectStorageRuntimeHost, self)
        try:
            if not host._closed and host.dirty_tracker.is_dirty():
                with host._lock:
                    if host._in_transaction:
                        return
                    host.db.commit()
                    host.dirty_tracker.clear()
                logger.debug("Autosave: committed changes for project %s", host.project.id)
        except Exception as exc:
            logger.warning("Autosave failed: %r", exc)
        finally:
            if not host._closed:
                self._schedule_autosave()

    @staticmethod
    def check_recovery(
        ez_path: Path,
        working_dir_root: Path | None = None,
    ) -> bool:
        from echozero.persistence.session import WORKING_DIR_ROOT

        root = working_dir_root or WORKING_DIR_ROOT
        working_dir = root / hashlib.sha256(str(ez_path.resolve()).encode()).hexdigest()[:16]
        return (working_dir / "project.db").exists()

    @staticmethod
    def recover(
        ez_path: Path,
        event_bus: EventBus | None = None,
        working_dir_root: Path | None = None,
    ) -> ProjectStorage:
        from echozero.persistence.session import ProjectStorage, WORKING_DIR_ROOT

        root = working_dir_root or WORKING_DIR_ROOT
        working_dir = root / hashlib.sha256(str(ez_path.resolve()).encode()).hexdigest()[:16]
        return ProjectStorage.open_db(working_dir, event_bus)

    @staticmethod
    def discard_recovery(
        ez_path: Path,
        working_dir_root: Path | None = None,
    ) -> None:
        from echozero.persistence.session import WORKING_DIR_ROOT, _release_project_lock

        root = working_dir_root or WORKING_DIR_ROOT
        working_dir = root / hashlib.sha256(str(ez_path.resolve()).encode()).hexdigest()[:16]
        if working_dir.exists():
            _release_project_lock(working_dir / "project.lock")
            shutil.rmtree(working_dir)


__all__ = ["ProjectStorageRuntimeMixin"]

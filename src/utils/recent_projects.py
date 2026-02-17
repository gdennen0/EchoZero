"""
Manage the recent-projects list.

This is a lightweight JSON-based store instead of keeping a table in SQLite.
"""
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from src.utils.paths import get_recent_projects_path
from src.features.projects.domain import Project


class RecentProjectsStore:
    """
    JSON-backed store for recent project metadata.
    """

    def __init__(self, path: Optional[Path] = None, max_entries: int = 20):
        self.path = Path(path) if path else get_recent_projects_path()
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries = self._load()

    def _load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2)

    def _normalize_entry(self, entry: Dict) -> Dict:
        entry = dict(entry)
        if isinstance(entry.get("last_accessed"), datetime):
            entry["last_accessed"] = entry["last_accessed"].isoformat()
        return entry

    def update(self, project: Project, project_file: Optional[str] = None) -> None:
        entry = {
            "project_id": project.id,
            "name": project.name,
            "version": project.version,
            "save_directory": self._serialize_path(project.save_directory),
            "project_file": self._serialize_path(project_file),
            "last_accessed": datetime.utcnow().isoformat()
        }

        with self._lock:
            self._entries = [
                e for e in self._entries if e.get("project_id") != project.id
            ]
            self._entries.insert(0, entry)
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[: self.max_entries]
            self._save()

    def get(self, identifier: str) -> Optional[Dict]:
        with self._lock:
            for entry in self._entries:
                if entry.get("project_id") == identifier or entry.get("name") == identifier:
                    return dict(entry)
        return None

    def list_recent(self, limit: int = 10) -> List[Dict]:
        with self._lock:
            return [dict(entry) for entry in self._entries[:limit]]

    @staticmethod
    def _serialize_path(value):
        if isinstance(value, Path):
            return str(value)
        return value


"""JSON-backed storage for machine-local EchoZero app preferences.
Exists because app settings need durable local persistence outside project archives.
Connects typed app preferences to an editable install-local JSON store.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from echozero.application.settings.models import (
    AppPreferences,
    app_preferences_from_dict,
    app_preferences_to_dict,
)


def default_app_settings_path() -> Path:
    """Resolve the canonical filesystem path for local app settings."""
    explicit_path = os.getenv("ECHOZERO_APP_SETTINGS_PATH")
    if explicit_path:
        return Path(explicit_path).expanduser()
    install_root = _default_install_root()
    return install_root / "config" / "app-settings.json"


def _legacy_user_app_settings_path() -> Path:
    """Resolve the legacy user-profile app settings path for backward compatibility."""
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "EchoZero" / "app-settings.json"
    return Path.home() / ".echozero" / "app-settings.json"


def _default_install_root() -> Path:
    """Resolve the install root used for editable install-local settings JSON files."""
    explicit_root = os.getenv("ECHOZERO_INSTALL_ROOT")
    if explicit_root:
        return Path(explicit_root).expanduser()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


class JsonAppSettingsStore:
    """Load and save app preferences as JSON on the local machine."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_app_settings_path()
        self._legacy_path: Path | None = (
            None if path is not None else _legacy_user_app_settings_path()
        )

    def load(self) -> AppPreferences:
        for candidate in self._load_candidates():
            loaded = self._load_candidate(candidate)
            if loaded is None:
                continue
            if candidate != self.path and not self._write_candidate(self.path, loaded):
                self.path = candidate
            return loaded
        return AppPreferences()

    def save(self, preferences: AppPreferences) -> None:
        if self._write_candidate(self.path, preferences):
            return
        if (
            self._legacy_path is not None
            and self._legacy_path != self.path
            and self._write_candidate(self._legacy_path, preferences)
        ):
            self.path = self._legacy_path
            return
        raise OSError(f"Unable to persist app settings JSON to {self.path}")

    def _load_candidates(self) -> tuple[Path, ...]:
        if self._legacy_path is None or self._legacy_path == self.path:
            return (self.path,)
        return (self.path, self._legacy_path)

    @staticmethod
    def _load_candidate(path: Path) -> AppPreferences | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return app_preferences_from_dict(payload)

    @staticmethod
    def _write_candidate(path: Path, preferences: AppPreferences) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(app_preferences_to_dict(preferences), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            return False
        return True

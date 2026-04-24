"""JSON-backed storage for machine-local EchoZero app preferences.
Exists because app settings need durable local persistence outside project archives.
Connects typed app preferences to a simple filesystem-backed store under app data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from echozero.application.settings.models import (
    AppPreferences,
    app_preferences_from_dict,
    app_preferences_to_dict,
)


def default_app_settings_path() -> Path:
    """Resolve the canonical filesystem path for local app settings."""
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "EchoZero" / "app-settings.json"
    return Path.home() / ".echozero" / "app-settings.json"


class JsonAppSettingsStore:
    """Load and save app preferences as JSON on the local machine."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_app_settings_path()

    def load(self) -> AppPreferences:
        if not self.path.exists():
            return AppPreferences()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return AppPreferences()
        if not isinstance(payload, dict):
            return AppPreferences()
        return app_preferences_from_dict(payload)

    def save(self, preferences: AppPreferences) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(app_preferences_to_dict(preferences), indent=2, sort_keys=True),
            encoding="utf-8",
        )

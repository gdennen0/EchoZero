"""Infrastructure settings helpers for local EchoZero preferences.
Exists to keep machine-local settings persistence behind one filesystem boundary.
Connects app-settings services to JSON-backed local storage helpers.
"""

from echozero.infrastructure.settings.json_store import (
    JsonAppSettingsStore,
    default_app_settings_path,
)

__all__ = [
    "JsonAppSettingsStore",
    "default_app_settings_path",
]

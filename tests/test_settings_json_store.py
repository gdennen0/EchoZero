from __future__ import annotations

import json
from pathlib import Path

from echozero.application.settings.models import (
    AppPreferences,
    AudioOutputPreferences,
    app_preferences_to_dict,
)
from echozero.infrastructure.settings.json_store import (
    JsonAppSettingsStore,
    default_app_settings_path,
)


def test_default_app_settings_path_uses_install_root_env(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ECHOZERO_APP_SETTINGS_PATH", raising=False)
    monkeypatch.setenv("ECHOZERO_INSTALL_ROOT", str(tmp_path))

    path = default_app_settings_path()

    assert path == tmp_path / "config" / "app-settings.json"


def test_default_app_settings_path_honors_explicit_path_env(monkeypatch, tmp_path) -> None:
    explicit_path = tmp_path / "custom" / "prefs.json"
    monkeypatch.setenv("ECHOZERO_APP_SETTINGS_PATH", str(explicit_path))
    monkeypatch.setenv("ECHOZERO_INSTALL_ROOT", str(tmp_path / "ignored"))

    path = default_app_settings_path()

    assert path == explicit_path


def test_json_store_loads_legacy_path_and_migrates_to_install_path(monkeypatch, tmp_path) -> None:
    install_root = tmp_path / "install"
    legacy_root = tmp_path / "legacy"
    monkeypatch.delenv("ECHOZERO_APP_SETTINGS_PATH", raising=False)
    monkeypatch.setenv("ECHOZERO_INSTALL_ROOT", str(install_root))
    monkeypatch.setenv("LOCALAPPDATA", str(legacy_root))

    legacy_path = legacy_root / "EchoZero" / "app-settings.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_preferences = AppPreferences(
        audio_output=AudioOutputPreferences(sample_rate=48000),
    )
    legacy_path.write_text(
        json.dumps(app_preferences_to_dict(legacy_preferences), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    store = JsonAppSettingsStore()
    loaded = store.load()

    assert loaded.audio_output.sample_rate == 48000
    install_path = Path(install_root) / "config" / "app-settings.json"
    assert install_path.exists()
    migrated_payload = json.loads(install_path.read_text(encoding="utf-8"))
    assert migrated_payload["audio_output"]["sample_rate"] == 48000

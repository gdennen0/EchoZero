"""Behavior-oriented preferences-dialog tests.
Exists to prove the canonical app-settings dialog can restore defaults and save local config edits.
Connects the Qt preferences surface to the reusable AppSettingsService seam.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from echozero.application.settings import (
    AppPreferences,
    AppSettingsService,
    AudioLatencyProfile,
    AudioOutputPreferences,
    MA3OscPreferences,
    OscSendPreferences,
    SettingsOption,
)
from echozero.ui.qt.preferences_dialog import PreferencesDialog


class _MemoryStore:
    """In-memory app-settings store for dialog tests."""

    path = Path("/tmp/echozero-test-preferences-dialog.json")

    def __init__(self, preferences: AppPreferences | None = None) -> None:
        self._preferences = preferences or AppPreferences()

    def load(self) -> AppPreferences:
        return self._preferences

    def save(self, preferences: AppPreferences) -> None:
        self._preferences = preferences


def _device_options() -> tuple[SettingsOption, ...]:
    return (
        SettingsOption(value="", label="System Default"),
        SettingsOption(value="7", label="Studio Output"),
    )


def test_preferences_dialog_restore_defaults_resets_form_values() -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(
        _MemoryStore(
            AppPreferences(
                audio_output=AudioOutputPreferences(
                    output_device="7",
                    sample_rate=48000,
                    output_channels=2,
                    latency_profile=AudioLatencyProfile.LOW,
                ),
                ma3_osc=MA3OscPreferences(
                    send=OscSendPreferences(enabled=True, port=9000),
                ),
            )
        ),
        audio_device_options_provider=_device_options,
    )
    dialog = PreferencesDialog(service)

    try:
        dialog._on_restore_defaults()

        restored = dialog._form.values()
        expected = {
            key: value
            for key, value in service.default_values().items()
            if key in restored
        }
        assert restored == expected
    finally:
        dialog.close()
        app.processEvents()


def test_preferences_dialog_save_persists_json_settings_and_reports_restart(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    info_messages: list[str] = []
    dialog = PreferencesDialog(service)

    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *_args: info_messages.append(str(_args[2])) or QMessageBox.StandardButton.Ok,
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args: (_ for _ in ()).throw(AssertionError("warning dialog not expected")),
    )

    try:
        dialog._form.set_values(
            {
                "audio.output_device": "7",
                "audio.sample_rate": 48000,
                "audio.output_channels": 2,
                "osc_send.enabled": True,
                "osc_send.port": 9000,
            }
        )

        dialog._on_save()

        assert service.preferences().audio_output.output_device == "7"
        assert service.preferences().audio_output.sample_rate == 48000
        assert service.preferences().ma3_osc.send.enabled is True
        assert service.preferences().ma3_osc.send.port == 9000
        assert info_messages == [
            "Restart EchoZero to apply saved audio output settings.\n"
            "Restart EchoZero to apply saved OSC settings."
        ]
        assert dialog.result() != 0
    finally:
        dialog.close()
        app.processEvents()

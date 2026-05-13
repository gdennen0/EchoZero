"""Behavior-oriented preferences-dialog tests.
Exists to prove the canonical app-settings dialog can restore defaults and save local config edits.
Connects the Qt preferences surface to the reusable AppSettingsService seam.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtWidgets import QCheckBox, QComboBox, QGridLayout

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
        SettingsOption(value="", label="System Default", metadata={"max_output_channels": 2}),
        SettingsOption(value="7", label="Studio Output", metadata={"max_output_channels": 8}),
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
            key: value for key, value in service.default_values().items() if key in restored
        }
        assert restored == expected
    finally:
        dialog.close()
        app.processEvents()


def test_preferences_dialog_save_persists_json_settings_and_calls_saved_hook(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    saved = {"called": False}
    dialog = PreferencesDialog(service, on_saved=lambda _result: saved.__setitem__("called", True))

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
            }
        )

        dialog._on_save()

        assert service.preferences().audio_output.output_device == "7"
        assert service.preferences().audio_output.sample_rate == 48000
        assert saved["called"] is True
        assert dialog.result() != 0
    finally:
        dialog.close()
        app.processEvents()


def test_preferences_dialog_master_output_buses_are_device_channel_checkboxes(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    dialog = PreferencesDialog(service)

    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args: (_ for _ in ()).throw(AssertionError("warning dialog not expected")),
    )

    try:
        output_device = dialog._form._inputs["audio.output_device"]
        assert isinstance(output_device, QComboBox)
        output_device.setCurrentIndex(output_device.findData("7"))
        app.processEvents()

        device_output_names = sorted(
            checkbox.text()
            for checkbox in dialog.findChildren(QCheckBox)
            if checkbox.text().startswith("Output ")
        )
        assert device_output_names == [
            "Output 1",
            "Output 2",
            "Output 3",
            "Output 4",
            "Output 5",
            "Output 6",
            "Output 7",
            "Output 8",
        ]

        output_channels = dialog._form._inputs["audio.output_channels"]
        assert isinstance(output_channels, QComboBox)
        output_channels.setCurrentIndex(output_channels.findData(4))
        app.processEvents()

        output_checkboxes = {
            checkbox.text(): checkbox
            for checkbox in dialog.findChildren(QCheckBox)
            if checkbox.text().startswith("Output ")
        }
        assert sorted(output_checkboxes) == [
            "Output 1",
            "Output 2",
            "Output 3",
            "Output 4",
            "Output 5",
            "Output 6",
            "Output 7",
            "Output 8",
        ]
        output_checkboxes["Output 2"].setChecked(True)

        assert dialog._form.values()["audio.master_output_bus"] == (
            "outputs_1_1,outputs_2_2"
        )

        dialog._on_save()

        assert service.preferences().audio_output.output_channels == 4
        assert service.preferences().audio_output.master_output_bus == (
            "outputs_1_1,outputs_2_2"
        )
    finally:
        dialog.close()
        app.processEvents()


def test_preferences_dialog_audio_section_shows_advanced_fields_without_toggle() -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    dialog = PreferencesDialog(service)

    try:
        toggle_texts = [
            checkbox.text()
            for checkbox in dialog.findChildren(QCheckBox)
            if checkbox.text() == "Show advanced settings"
        ]
        assert toggle_texts == []

        content_layout = dialog._form._content_layout
        audio_grid = content_layout.itemAt(2).layout()
        advanced_container = content_layout.itemAt(3).widget()
        advanced_grid = advanced_container.layout().itemAt(1).layout()

        assert isinstance(audio_grid, QGridLayout)
        assert isinstance(advanced_grid, QGridLayout)
        assert advanced_container.isHidden() is False
    finally:
        dialog.close()
        app.processEvents()

"""Behavior-oriented OSC settings dialog tests.
Exists to prove the dedicated OSC dialog saves endpoint settings and reports probe health.
Connects the Qt OSC settings surface to reusable AppSettingsService persistence behavior.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from echozero.application.settings import (
    AppPreferences,
    AppSettingsService,
    MA3OscPreferences,
    OscSendPreferences,
    SettingsOption,
)
from echozero.ui.qt.osc_settings_dialog import OscSettingsDialog
from echozero.ui.qt.osc_settings_panel import OscSettingsPanel, _OscProbeConfig
from echozero.testing.ma3.simulator import _SimulatedMA3OSCServer


class _MemoryStore:
    """In-memory app-settings store for OSC dialog tests."""

    path = Path("/tmp/echozero-test-osc-settings-dialog.json")

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


def test_osc_settings_dialog_restore_defaults_resets_form_values() -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(
        _MemoryStore(
            AppPreferences(
                ma3_osc=MA3OscPreferences(
                    send=OscSendPreferences(enabled=True, host="10.0.0.5", port=9000),
                ),
            )
        ),
        audio_device_options_provider=_device_options,
    )
    dialog = OscSettingsDialog(service)

    try:
        dialog._on_restore_defaults()

        restored = dialog._form.values()
        expected = {
            key: value
            for key, value in service.default_values().items()
            if key.startswith("osc_")
        }
        assert restored == expected
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_dialog_save_persists_settings_and_reports_restart(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    info_messages: list[str] = []
    dialog = OscSettingsDialog(service)

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
                "osc_receive.enabled": True,
                "osc_receive.host": "127.0.0.1",
                "osc_receive.port": 7100,
                "osc_send.enabled": True,
                "osc_send.host": "127.0.0.1",
                "osc_send.port": 9000,
            }
        )

        dialog._on_save()

        assert service.preferences().ma3_osc.receive.enabled is True
        assert service.preferences().ma3_osc.receive.port == 7100
        assert service.preferences().ma3_osc.send.enabled is True
        assert service.preferences().ma3_osc.send.port == 9000
        assert info_messages == ["Restart EchoZero to apply saved OSC settings."]
        assert dialog.result() != 0
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_dialog_panel_reports_ready_after_status_check(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    dialog = OscSettingsDialog(service)

    monkeypatch.setattr(
        dialog._panel,
        "_probe_receive_endpoint",
        lambda _config: (True, "Receive OK (127.0.0.1:7100)."),
    )
    monkeypatch.setattr(
        dialog._panel,
        "_probe_send_endpoint",
        lambda _config: (True, "Send OK (127.0.0.1:9000)."),
    )

    try:
        dialog._form.set_values(
            {
                "osc_receive.enabled": True,
                "osc_receive.host": "127.0.0.1",
                "osc_receive.port": 7100,
                "osc_send.enabled": True,
                "osc_send.host": "127.0.0.1",
                "osc_send.port": 9000,
            }
        )

        dialog._panel._on_check_status()

        assert dialog._panel._status_value.text() == "Ready"
        assert "Receive OK" in dialog._panel._status_detail.text()
        assert "Send OK" in dialog._panel._status_detail.text()
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_dialog_panel_ping_updates_connection_and_latency(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    dialog = OscSettingsDialog(service)

    monkeypatch.setattr(
        dialog._panel,
        "_run_ping",
        lambda _config: (True, "Ping response received (status=ok).", 12.34),
    )

    try:
        dialog._form.set_values(
            {
                "osc_receive.enabled": True,
                "osc_send.enabled": True,
            }
        )

        dialog._panel._on_ping()

        assert dialog._panel._status_value.text() == "Connected"
        assert dialog._panel._ping_value.text() == "12.3 ms"
        assert "status=ok" in dialog._panel._status_detail.text()
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_panel_ping_uses_routable_target_for_wildcard_receive_host() -> None:
    server = _SimulatedMA3OSCServer().start()
    try:
        success, detail, latency_ms = OscSettingsPanel._run_ping(
            _OscProbeConfig(
                receive_enabled=True,
                receive_host="0.0.0.0",
                receive_port=0,
                send_enabled=True,
                send_host=server.endpoint[0],
                send_port=server.endpoint[1],
            )
        )

        assert success is True
        assert latency_ms is not None and latency_ms >= 0.0
        assert "status=ok" in detail
        target_command = next(
            (command for command in server.commands if command.startswith("EZ.SetTarget(")),
            "",
        )
        assert target_command
        assert "0.0.0.0" not in target_command
    finally:
        server.stop()

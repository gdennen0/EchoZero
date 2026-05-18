"""Behavior-oriented OSC settings dialog tests.
Exists to prove the dedicated OSC dialog saves endpoint settings and reports probe health.
Connects the Qt OSC settings surface to reusable AppSettingsService persistence behavior.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QApplication, QComboBox, QMessageBox

from echozero.application.settings import (
    AppPreferences,
    AppSettingsService,
    MA3OscPreferences,
    OscReceivePreferences,
    OscSendPreferences,
    SettingsOption,
)
from echozero.application.sync.ma3_connection_check import (
    MA3OscConnectionCheckRequest,
    MA3OscConnectionCheckResult,
    MA3OscConnectionCheckService,
    MA3OscConnectionState,
    MA3OscEndpointCheck,
)
from echozero.ui.qt.osc_settings_dialog import OscSettingsDialog
from echozero.ui.qt.osc_settings_panel import OscSettingsPanel
from echozero.testing.ma3.simulator import _SimulatedMA3OSCServer


def _successful_hardware_ping(host: str, _timeout_seconds: float) -> MA3OscEndpointCheck:
    return MA3OscEndpointCheck(
        stage="Hardware Ping",
        ok=True,
        detail=f"Ping response received from {host}.",
    )


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


def _receive_address_options() -> tuple[SettingsOption, ...]:
    return (
        SettingsOption(value="127.0.0.1", label="Localhost (127.0.0.1)"),
        SettingsOption(value="0.0.0.0", label="All Interfaces (0.0.0.0)"),
        SettingsOption(value="10.0.0.25", label="en0 (10.0.0.25)"),
    )


class _FakeConnectionChecker:
    """Connection-check double for panel rendering tests."""

    def __init__(self, result: MA3OscConnectionCheckResult) -> None:
        self.result = result
        self.values: dict[str, object] | None = None
        self.live_bridge = None

    def request_from_values(self, values: dict[str, object]):
        self.values = dict(values)
        return object()

    def ping(self, _request, *, live_bridge=None):
        self.live_bridge = live_bridge
        return self.result


class _StateInspectingConnectionChecker:
    """Connection-check double that verifies visible state at ping start."""

    def __init__(self, panel: OscSettingsPanel, result: MA3OscConnectionCheckResult) -> None:
        self.panel = panel
        self.result = result
        self.saw_reset_state = False

    def request_from_values(self, values: dict[str, object]):
        return object()

    def ping(self, _request, *, live_bridge=None):
        self.saw_reset_state = (
            self.panel._status_value.text() == "Checking"
            and self.panel._ping_value.text() == "Not measured"
            and self.panel._copy_report_button.isEnabled() is False
            and self.panel._check_status_button.isEnabled() is False
        )
        return self.result


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
            key: value for key, value in service.default_values().items() if key.startswith("osc_")
        }
        assert restored == expected
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_dialog_receive_address_is_dropdown_of_network_options() -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(
        _MemoryStore(
            AppPreferences(
                ma3_osc=MA3OscPreferences(receive=OscReceivePreferences(host="10.0.0.25")),
            )
        ),
        audio_device_options_provider=_device_options,
        osc_receive_address_options_provider=_receive_address_options,
    )
    dialog = OscSettingsDialog(service)

    try:
        widget = dialog._form._inputs["osc_receive.host"]

        assert isinstance(widget, QComboBox)
        assert widget.currentData() == "10.0.0.25"
        assert widget.findData("0.0.0.0") >= 0
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_dialog_save_persists_settings_and_calls_saved_hook(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    saved = {"called": False}
    dialog = OscSettingsDialog(service, on_saved=lambda _result: saved.__setitem__("called", True))

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
        assert saved["called"] is True
        assert dialog.result() != 0
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_dialog_panel_reports_connected_after_connection_check() -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    checker = _FakeConnectionChecker(
        MA3OscConnectionCheckResult(
            state=MA3OscConnectionState.CONNECTED,
            detail="Ping response received (status=ok).",
            recommended_action="MA3 round trip is connected.",
            latency_ms=12.34,
            checks=(
                MA3OscEndpointCheck(
                    stage="Receive Listener",
                    ok=True,
                    detail="Receive Listener OK (127.0.0.1:7100).",
                ),
                MA3OscEndpointCheck(
                    stage="Command Send",
                    ok=True,
                    detail="Command Send OK (127.0.0.1:9000).",
                ),
                MA3OscEndpointCheck(
                    stage="Hardware Ping",
                    ok=True,
                    detail="Ping response received from 127.0.0.1.",
                ),
                MA3OscEndpointCheck(
                    stage="MA3 Reply",
                    ok=True,
                    detail="Ping response received (status=ok).",
                ),
            ),
        )
    )
    dialog = OscSettingsDialog(service)
    dialog._panel._connection_checker = checker

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

        dialog._panel._on_run_connection_check()

        assert dialog._panel._status_value.text() == "Connected"
        assert dialog._panel._ping_value.text() == "12.3 ms"
        assert "OK Receive Listener" in dialog._panel._status_detail.text()
        assert "OK Command Send" in dialog._panel._status_detail.text()
        assert "OK Hardware Ping" in dialog._panel._status_detail.text()
        assert "Next: MA3 round trip is connected." in dialog._panel._status_detail.text()
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_dialog_panel_resets_stale_success_when_check_starts() -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    dialog = OscSettingsDialog(service)
    connected_result = MA3OscConnectionCheckResult(
        state=MA3OscConnectionState.CONNECTED,
        detail="Ping response received (status=ok).",
        latency_ms=8.0,
    )
    dialog._panel._apply_connection_result(connected_result)
    dialog._panel._last_check_request = object()
    dialog._panel._last_check_result = connected_result
    dialog._panel._copy_report_button.setEnabled(True)
    checker = _StateInspectingConnectionChecker(dialog._panel, connected_result)
    dialog._panel._connection_checker = checker

    try:
        dialog._panel._on_run_connection_check()

        assert checker.saw_reset_state is True
        assert dialog._panel._status_value.text() == "Connected"
        assert dialog._panel._ping_value.text() == "8.0 ms"
        assert dialog._panel._copy_report_button.isEnabled() is True
        assert dialog._panel._check_status_button.isEnabled() is True
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_dialog_panel_reports_unreachable_hardware_ping() -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    checker = _FakeConnectionChecker(
        MA3OscConnectionCheckResult(
            state=MA3OscConnectionState.HARDWARE_UNREACHABLE,
            detail="Hardware ping failed: No ping response from 192.0.2.55.",
            recommended_action="Verify the MA3 host is powered on and reachable on the network.",
            checks=(
                MA3OscEndpointCheck(
                    stage="Hardware Ping",
                    ok=False,
                    detail="No ping response from 192.0.2.55.",
                ),
            ),
        )
    )
    dialog = OscSettingsDialog(service)
    dialog._panel._connection_checker = checker

    try:
        dialog._panel._on_run_connection_check()

        assert dialog._panel._status_value.text() == "Device Unreachable"
        assert "FAIL Hardware Ping" in dialog._panel._status_detail.text()
        assert dialog._panel._ping_value.text() == "Not measured"
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_dialog_keeps_settings_and_feedback_in_separate_panes() -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    dialog = OscSettingsDialog(service, monitor_provider=lambda: [])

    try:
        dialog.resize(700, 620)
        dialog.show()
        app.processEvents()

        form_bottom = dialog._form.geometry().bottom()
        panel_top = dialog._panel.geometry().top()
        assert form_bottom < panel_top
        assert 148 <= dialog._form.height() <= 250
        assert dialog._panel.height() > dialog._form.height()
        assert dialog._panel._monitor_output.maximumHeight() == 140
        assert dialog._panel._monitor_output.lineWrapMode().name == "WidgetWidth"
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_dialog_panel_uses_live_bridge_when_settings_are_clean() -> None:
    app = QApplication.instance() or QApplication([])
    service = AppSettingsService(_MemoryStore(), audio_device_options_provider=_device_options)
    bridge = object()
    checker = _FakeConnectionChecker(
        MA3OscConnectionCheckResult(
            state=MA3OscConnectionState.CONNECTED,
            detail="Ping response received (status=ok).",
        )
    )
    dialog = OscSettingsDialog(
        service,
        live_bridge_provider=lambda: bridge,
    )
    dialog._panel._connection_checker = checker

    try:
        dialog._panel._on_run_connection_check()

        assert checker.live_bridge is bridge
    finally:
        dialog.close()
        app.processEvents()


def test_osc_settings_panel_ping_uses_routable_target_for_wildcard_receive_host() -> None:
    server = _SimulatedMA3OSCServer().start()
    try:
        result = MA3OscConnectionCheckService(
            terminal_ping=_successful_hardware_ping,
        ).ping(
            MA3OscConnectionCheckRequest(
                receive_enabled=True,
                receive_host="0.0.0.0",
                receive_port=0,
                send_enabled=True,
                send_host=server.endpoint[0],
                send_port=server.endpoint[1],
            )
        )

        assert result.is_connected is True
        assert result.latency_ms is not None and result.latency_ms >= 0.0
        assert "status=ok" in result.detail
        target_command = next(
            (command for command in server.commands if command.startswith("EZ.SetTarget(")),
            "",
        )
        assert target_command
        assert "0.0.0.0" not in target_command
    finally:
        server.stop()


def test_osc_settings_panel_ping_wraps_commands_for_ma3_cmd_path(monkeypatch) -> None:
    sent_commands: list[str] = []

    def capture_command(_host: str, _port: int, command: str) -> None:
        sent_commands.append(command)

    monkeypatch.setattr(
        "echozero.application.sync.ma3_connection_check._send_udp_command",
        capture_command,
    )

    MA3OscConnectionCheckService(terminal_ping=_successful_hardware_ping).ping(
        MA3OscConnectionCheckRequest(
            receive_enabled=True,
            receive_host="127.0.0.1",
            receive_port=7100,
            send_enabled=True,
            send_host="127.0.0.1",
            send_port=9000,
        )
    )

    assert sent_commands == [
        "Lua \"EZ.SetTarget('127.0.0.1', 7100)\"",
        'Lua "EZ.Ping()"',
    ]


def test_osc_settings_panel_connection_check_fails_before_osc_when_hardware_missing() -> None:
    def failed_hardware_ping(host: str, _timeout_seconds: float) -> MA3OscEndpointCheck:
        return MA3OscEndpointCheck(
            stage="Hardware Ping",
            ok=False,
            detail=f"No ping response from {host}.",
        )

    result = MA3OscConnectionCheckService(terminal_ping=failed_hardware_ping).ping(
        MA3OscConnectionCheckRequest(
            receive_enabled=True,
            receive_host="127.0.0.1",
            receive_port=7100,
            send_enabled=True,
            send_host="192.0.2.55",
            send_port=9000,
        )
    )

    assert result.state is MA3OscConnectionState.HARDWARE_UNREACHABLE
    assert result.latency_ms is None
    assert result.checks[-1].stage == "Hardware Ping"
    assert result.checks[-1].ok is False
    assert "Hardware ping failed" in result.detail
    assert "MA3 Reply" not in result.diagnostic_report(
        MA3OscConnectionCheckRequest(
            receive_enabled=True,
            receive_host="127.0.0.1",
            receive_port=7100,
            send_enabled=True,
            send_host="192.0.2.55",
            send_port=9000,
        )
    )


def test_osc_settings_panel_monitor_refresh_renders_recent_messages() -> None:
    app = QApplication.instance() or QApplication([])
    rows = [
        {
            "timestamp": 1_717_590_000,
            "message_type": "transport",
            "change": "scrubbed",
            "fields": {"tc": 112, "to_seconds": 29.1, "delta_seconds": 1.25},
        },
        {
            "timestamp": 1_717_590_001,
            "message_type": "transport",
            "change": "jumped_previous_section",
            "fields": {"tc": 112, "tg": 1, "track": 4, "to_seconds": 26.8},
        },
    ]
    panel = OscSettingsPanel(
        values_provider=lambda: {},
        monitor_provider=lambda: rows,
    )

    try:
        panel._refresh_monitor()
        body = panel._monitor_output.toPlainText()
        assert "transport.scrubbed" in body
        assert "to_seconds=29.1" in body
        assert "transport.jumped_previous_section" in body
        assert "track=4" in body
    finally:
        panel.close()
        app.processEvents()


def test_osc_settings_panel_clear_log_clears_provider_and_display() -> None:
    app = QApplication.instance() or QApplication([])
    rows = [
        {
            "timestamp": 1_717_590_000,
            "message_type": "transport",
            "change": "scrubbed",
            "fields": {"tc": 112},
        },
    ]
    panel = OscSettingsPanel(
        values_provider=lambda: {},
        monitor_provider=lambda: rows,
        clear_monitor=rows.clear,
    )

    try:
        panel._refresh_monitor()
        assert "transport.scrubbed" in panel._monitor_output.toPlainText()

        panel._monitor_clear.click()

        assert rows == []
        assert panel._monitor_output.toPlainText() == "No inbound OSC messages yet."
    finally:
        panel.close()
        app.processEvents()


def test_osc_settings_panel_scan_selects_discovered_ma3_destination() -> None:
    app = QApplication.instance() or QApplication([])
    server = _SimulatedMA3OSCServer().start()
    values = {
        "osc_receive.host": "127.0.0.1",
        "osc_receive.port": 0,
        "osc_send.enabled": False,
        "osc_send.host": server.endpoint[0],
        "osc_send.port": server.endpoint[1],
    }
    panel = OscSettingsPanel(
        values_provider=lambda: dict(values),
        values_applier=values.update,
    )

    try:
        panel._on_scan_ma3_devices()

        assert values["osc_send.enabled"] is True
        assert values["osc_send.host"] == server.endpoint[0]
        assert values["osc_send.port"] == server.endpoint[1]
        assert panel._status_value.text() == "MA3 Found"
    finally:
        panel.close()
        server.stop()
        app.processEvents()

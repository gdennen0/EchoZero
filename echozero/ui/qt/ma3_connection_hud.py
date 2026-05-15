"""MA3 connection overlay for quick OSC endpoint recovery.
Exists to let operators recover send and receive settings from the MA3 send flow.
Connects AppSettingsService values to lightweight test-and-save controls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from echozero.application.settings import AppSettingsService
from echozero.application.sync.ma3_connection_check import (
    MA3OscConnectionCheckService,
    MA3OscConnectionState,
)
from echozero.ui.style.qt import ensure_qt_theme_installed

if TYPE_CHECKING:
    from echozero.application.settings.models import AppSettingsUpdateResult


class MA3ConnectionHUD(QDialog):
    """Reusable MA3 OSC troubleshooting overlay used by MA3 send paths."""

    def __init__(
        self,
        settings_service: AppSettingsService,
        *,
        on_saved: Callable[["AppSettingsUpdateResult"], None] | None = None,
        connection_checker: MA3OscConnectionCheckService | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("ma3ConnectionHUD")
        ensure_qt_theme_installed()
        self._settings_service = settings_service
        self._on_saved = on_saved
        self._connection_checker = connection_checker or MA3OscConnectionCheckService()
        self._last_check_request = None
        self._last_check_result = None
        self.setWindowTitle("MA3 OSC Connection")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._status = QLabel(self)
        self._status.setObjectName("ma3ConnectionHudStatus")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        receive_group = QGroupBox("Receive Endpoint", self)
        receive_form = QFormLayout(receive_group)
        receive_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        receive_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._receive_enabled = QCheckBox("Enable", receive_group)
        self._receive_host = QLineEdit(receive_group)
        self._receive_port = QSpinBox(receive_group)
        self._receive_port.setRange(0, 65_535)
        self._receive_port.setKeyboardTracking(False)
        receive_form.addRow(self._receive_enabled)
        receive_form.addRow("Host", self._receive_host)
        receive_form.addRow("Port", self._receive_port)
        self._receive_enabled.toggled.connect(self._sync_host_fields_enabled)
        layout.addWidget(receive_group)

        send_group = QGroupBox("Send Endpoint", self)
        send_form = QFormLayout(send_group)
        send_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        send_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._send_enabled = QCheckBox("Enable", send_group)
        self._send_host = QLineEdit(send_group)
        self._send_port = QSpinBox(send_group)
        self._send_port.setRange(0, 65_535)
        self._send_port.setKeyboardTracking(False)
        self._send_enabled.toggled.connect(self._sync_host_fields_enabled)
        send_form.addRow(self._send_enabled)
        send_form.addRow("Host", self._send_host)
        send_form.addRow("Port", self._send_port)
        layout.addWidget(send_group)

        self._message = QLabel(self)
        self._message.setWordWrap(True)
        self._message.setObjectName("ma3ConnectionHudMessage")
        layout.addWidget(self._message)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self._copy_report_button = QPushButton("Copy Diagnostic Report", self)
        self._copy_report_button.setProperty("appearance", "subtle")
        self._copy_report_button.setEnabled(False)
        self._copy_report_button.clicked.connect(self._copy_diagnostic_report)
        action_row.addWidget(self._copy_report_button)
        self._test_button = QPushButton("Run Connection Check", self)
        self._test_button.clicked.connect(self._test_connection)
        action_row.addWidget(self._test_button)
        layout.addLayout(action_row)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._apply_button = self._require_button(QDialogButtonBox.StandardButton.Ok)
        self._apply_button.setText("Apply")
        self._buttons.rejected.connect(self.reject)
        self._apply_button.clicked.connect(self._on_apply)
        layout.addWidget(self._buttons)

        self._load_from_settings()
        self._sync_host_fields_enabled()

    def _load_from_settings(self) -> None:
        preferences = self._settings_service.preferences()
        self._receive_enabled.setChecked(preferences.ma3_osc.receive.enabled)
        self._receive_host.setText(preferences.ma3_osc.receive.host)
        self._receive_port.setValue(preferences.ma3_osc.receive.port)
        self._send_enabled.setChecked(preferences.ma3_osc.send.enabled)
        self._send_host.setText(preferences.ma3_osc.send.host)
        self._send_port.setValue(preferences.ma3_osc.send.port or 0)
        self._set_message("Inspect and adjust MA3 OSC connection values.", severity="neutral")

    def _sync_host_fields_enabled(self, *_args: object) -> None:
        self._receive_host.setEnabled(self._receive_enabled.isChecked())
        self._receive_port.setEnabled(self._receive_enabled.isChecked())
        self._send_host.setEnabled(self._send_enabled.isChecked())
        self._send_port.setEnabled(self._send_enabled.isChecked())
        if not self._send_enabled.isChecked() and self._send_port.value() == 0:
            self._send_port.setValue(10_000)

    def _set_message(self, message: str, *, severity: str = "neutral") -> None:
        tone = "ok" if severity == "success" else severity
        self._message.setProperty("tone", tone if tone in {"ok", "error", "neutral"} else "neutral")
        style = self._message.style()
        if style is not None:
            style.unpolish(self._message)
            style.polish(self._message)
        self._message.update()
        self._message.setText(message)

    def _message_payload(self) -> dict[str, object]:
        return {
            "osc_receive.enabled": self._receive_enabled.isChecked(),
            "osc_receive.host": self._receive_host.text().strip() or "127.0.0.1",
            "osc_receive.port": int(self._receive_port.value()),
            "osc_send.enabled": self._send_enabled.isChecked(),
            "osc_send.host": self._send_host.text().strip() or "127.0.0.1",
            "osc_send.port": (
                int(self._send_port.value()) if self._send_enabled.isChecked() else None
            ),
        }

    def _test_connection(self) -> None:
        request = self._connection_checker.request_from_values(self._message_payload())
        result = self._connection_checker.ping(request)
        self._last_check_request = request
        self._last_check_result = result
        self._copy_report_button.setEnabled(True)
        severity = "success" if result.state is MA3OscConnectionState.CONNECTED else "error"
        if result.state is MA3OscConnectionState.DISABLED:
            severity = "neutral"
        self._set_message(self._format_result_detail(result), severity=severity)

    def _copy_diagnostic_report(self) -> None:
        if self._last_check_request is None or self._last_check_result is None:
            return
        QApplication.clipboard().setText(
            self._last_check_result.diagnostic_report(self._last_check_request)
        )

    @staticmethod
    def _format_result_detail(result) -> str:
        lines = [result.detail] if result.detail else []
        for check in result.checks:
            prefix = "OK" if check.ok else "FAIL"
            lines.append(f"{prefix} {check.stage}: {check.detail}")
        if result.recommended_action:
            lines.append(f"Next: {result.recommended_action}")
        return "\n".join(lines)

    def _on_apply(self) -> None:
        if not (self._receive_enabled.isChecked() or self._send_enabled.isChecked()):
            self._set_message("Enable MA3 receive and/or send before saving.", severity="error")
            return
        payload = self._message_payload()
        try:
            result: "AppSettingsUpdateResult" = self._settings_service.apply_updates(payload)
        except Exception as exc:
            self._set_message(f"Unable to save MA3 OSC settings: {exc}", severity="error")
            return
        if self._on_saved is not None:
            try:
                self._on_saved(result)
            except Exception as exc:
                self._set_message(f"Unable to apply MA3 OSC settings: {exc}", severity="error")
                return
        self.accept()

    def _require_button(self, standard_button: QDialogButtonBox.StandardButton) -> QPushButton:
        button = self._buttons.button(standard_button)
        if button is None:
            raise RuntimeError(f"Missing HUD button for standard button {standard_button!r}")
        return button

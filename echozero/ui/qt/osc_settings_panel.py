"""OSC connection panel for machine-local EchoZero settings surfaces.
Exists because operators need one place to validate OSC endpoint health before saving config.
Connects live probe + ping checks to editable OSC settings values from the preferences form.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from echozero.application.sync.ma3_connection_check import (
    MA3OscLiveBridge,
    MA3OscConnectionCheckResult,
    MA3OscConnectionCheckService,
    MA3OscConnectionState,
)

_MONITOR_REFRESH_MS = 500


class OscSettingsPanel(QWidget):
    """Display OSC connection status and run one-shot ping checks from form values."""

    def __init__(
        self,
        *,
        values_provider: Callable[[], Mapping[str, object]],
        monitor_provider: Callable[[], list[Mapping[str, object]]] | None = None,
        clear_monitor: Callable[[], None] | None = None,
        connection_checker: MA3OscConnectionCheckService | None = None,
        live_bridge_provider: Callable[[], MA3OscLiveBridge | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._values_provider = values_provider
        self._monitor_provider = monitor_provider
        self._clear_monitor = clear_monitor
        self._connection_checker = connection_checker or MA3OscConnectionCheckService()
        self._live_bridge_provider = live_bridge_provider
        self._has_dirty_settings = False
        self._last_check_request = None
        self._last_check_result: MA3OscConnectionCheckResult | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        group = QGroupBox("OSC Connection", self)
        group.setProperty("section", True)
        group.setProperty("compact", True)
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(8, 10, 8, 8)
        group_layout.setSpacing(5)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        self._status_value = QLabel("Unknown", group)
        self._status_detail = QLabel(
            "Run Connection Check to validate the MA3 OSC round trip.",
            group,
        )
        self._status_detail.setWordWrap(True)
        self._status_detail.setMinimumHeight(68)
        self._status_detail.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        self._ping_value = QLabel("Not measured", group)

        form.addRow("Status", self._status_value)
        form.addRow("Last Ping", self._ping_value)
        group_layout.addLayout(form)
        group_layout.addWidget(self._status_detail)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        self._check_status_button = QPushButton("Run Check", group)
        self._check_status_button.setProperty("appearance", "subtle")
        self._check_status_button.setToolTip("Run the full MA3 OSC round-trip connection check.")
        self._check_status_button.clicked.connect(self._on_run_connection_check)
        actions.addWidget(self._check_status_button)
        self._copy_report_button = QPushButton("Copy Report", group)
        self._copy_report_button.setProperty("appearance", "subtle")
        self._copy_report_button.setToolTip("Copy the full MA3 OSC diagnostic report.")
        self._copy_report_button.setEnabled(False)
        self._copy_report_button.clicked.connect(self._copy_diagnostic_report)
        actions.addWidget(self._copy_report_button)
        group_layout.addLayout(actions)

        layout.addWidget(group, 1)
        layout.addWidget(self._build_monitor_group(), 2)
        self._set_status("unknown", "Unknown", "Run Connection Check to validate OSC endpoints.")
        self._sync_monitor_state()

    def _build_monitor_group(self) -> QGroupBox:
        group = QGroupBox("Recent Incoming OSC", self)
        group.setProperty("section", True)
        group.setProperty("compact", True)
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(5)

        self._monitor_status = QLabel(
            "Monitor unavailable in this shell.",
            group,
        )
        self._monitor_status.setWordWrap(True)
        layout.addWidget(self._monitor_status)

        self._monitor_output = QPlainTextEdit(group)
        self._monitor_output.setReadOnly(True)
        self._monitor_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._monitor_output.setMinimumHeight(110)
        self._monitor_output.setMaximumHeight(140)
        self._monitor_output.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        self._monitor_output.setPlainText("No inbound OSC messages yet.")
        layout.addWidget(self._monitor_output)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        self._monitor_auto = QCheckBox("Auto Refresh", group)
        self._monitor_auto.setChecked(True)
        self._monitor_auto.toggled.connect(self._sync_monitor_state)
        actions.addWidget(self._monitor_auto)
        actions.addStretch(1)
        self._monitor_clear = QPushButton("Clear Log", group)
        self._monitor_clear.setProperty("appearance", "subtle")
        self._monitor_clear.clicked.connect(self._clear_monitor_log)
        actions.addWidget(self._monitor_clear)
        self._monitor_refresh = QPushButton("Refresh", group)
        self._monitor_refresh.setProperty("appearance", "subtle")
        self._monitor_refresh.clicked.connect(self._refresh_monitor)
        actions.addWidget(self._monitor_refresh)
        layout.addLayout(actions)

        self._monitor_timer = QTimer(group)
        self._monitor_timer.setInterval(_MONITOR_REFRESH_MS)
        self._monitor_timer.timeout.connect(self._refresh_monitor)
        return group

    def mark_settings_dirty(self) -> None:
        """Reset connection health to unknown after OSC form edits."""

        self._ping_value.setText("Not measured")
        self._has_dirty_settings = True
        self._last_check_request = None
        self._last_check_result = None
        self._copy_report_button.setEnabled(False)
        self._set_status(
            "unknown",
            "Pending Check",
            "OSC settings changed. Run Connection Check to refresh health.",
        )

    def _set_status(self, tone: str, title: str, detail: str) -> None:
        self._status_value.setText(title)
        self._status_value.setProperty("statusLabel", True)
        self._set_tone(self._status_value, tone)
        self._set_tone(self._status_detail, tone)
        self._status_detail.setText(detail)

    def _sync_monitor_state(self) -> None:
        provider_available = callable(self._monitor_provider)
        self._monitor_auto.setEnabled(provider_available)
        self._monitor_refresh.setEnabled(provider_available)
        self._monitor_clear.setEnabled(provider_available and callable(self._clear_monitor))
        if not provider_available:
            self._monitor_timer.stop()
            self._set_tone(self._monitor_status, "warn")
            self._monitor_status.setText(
                "Live monitor unavailable: this surface has no active MA3 bridge hook."
            )
            self._monitor_output.setPlainText("No live OSC stream available in this shell.")
            return

        self._set_tone(self._monitor_status, "ok")
        self._monitor_status.setText("Showing the latest inbound OSC messages seen by EZ.")
        self._refresh_monitor()
        if self._monitor_auto.isChecked():
            self._monitor_timer.start()
        else:
            self._monitor_timer.stop()

    def _clear_monitor_log(self) -> None:
        if callable(self._clear_monitor):
            self._clear_monitor()
        self._monitor_output.setPlainText("No inbound OSC messages yet.")
        self._refresh_monitor()

    def _refresh_monitor(self) -> None:
        if not callable(self._monitor_provider):
            return
        try:
            rows = list(self._monitor_provider())
        except Exception as exc:
            self._set_tone(self._monitor_status, "error")
            self._monitor_status.setText(f"Monitor read failed: {exc}")
            return

        lines = self._format_monitor_lines(rows)
        if not lines:
            self._monitor_output.setPlainText("No inbound OSC messages yet.")
            return
        self._monitor_output.setPlainText("\n".join(lines))
        self._monitor_output.verticalScrollBar().setValue(
            self._monitor_output.verticalScrollBar().maximum()
        )

    @staticmethod
    def _format_monitor_lines(rows: list[Mapping[str, object]]) -> list[str]:
        lines: list[str] = []
        for row in rows[-12:]:
            timestamp = OscSettingsPanel._format_monitor_timestamp(row.get("timestamp"))
            message_type = str(row.get("message_type") or "unknown").strip() or "unknown"
            change = str(row.get("change") or "unknown").strip() or "unknown"
            fields = row.get("fields")
            field_summary = OscSettingsPanel._monitor_field_summary(
                fields if isinstance(fields, Mapping) else {}
            )
            lines.append(f"{timestamp} {message_type}.{change} {field_summary}".rstrip())
        return lines

    @staticmethod
    def _format_monitor_timestamp(raw_timestamp: object) -> str:
        try:
            resolved = float(raw_timestamp)
        except (TypeError, ValueError):
            return "--:--:--"
        if resolved <= 0:
            return "--:--:--"
        try:
            return datetime.fromtimestamp(resolved).strftime("%H:%M:%S")
        except (OverflowError, OSError, ValueError):
            return "--:--:--"

    @staticmethod
    def _monitor_field_summary(fields: Mapping[str, object]) -> str:
        keys = (
            "tc",
            "tg",
            "track",
            "to_seconds",
            "from_seconds",
            "delta_seconds",
            "status",
            "error",
        )
        parts: list[str] = []
        for key in keys:
            if key not in fields:
                continue
            value = fields.get(key)
            if value in {None, ""}:
                continue
            parts.append(f"{key}={value}")
        return " ".join(parts)

    def _on_run_connection_check(self) -> None:
        request = self._connection_checker.request_from_values(dict(self._values_provider()))
        live_bridge = self._live_bridge_for_current_values()
        result = self._connection_checker.ping(request, live_bridge=live_bridge)
        self._last_check_request = request
        self._last_check_result = result
        self._copy_report_button.setEnabled(True)
        self._apply_connection_result(result)

    def _live_bridge_for_current_values(self) -> MA3OscLiveBridge | None:
        if self._has_dirty_settings or self._live_bridge_provider is None:
            return None
        try:
            return self._live_bridge_provider()
        except Exception:
            return None

    def _apply_connection_result(self, result: MA3OscConnectionCheckResult) -> None:
        if result.latency_ms is None:
            self._ping_value.setText("Not measured")
        else:
            self._ping_value.setText(f"{result.latency_ms:.1f} ms")
        tone, title = self._status_tone_and_title(result.state)
        self._set_status(tone, title, self._format_result_detail(result))

    def _copy_diagnostic_report(self) -> None:
        if self._last_check_request is None or self._last_check_result is None:
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(self._last_check_result.diagnostic_report(self._last_check_request))

    @staticmethod
    def _set_tone(label: QLabel, tone: str) -> None:
        label.setProperty("tone", tone if tone in {"ok", "warn", "error", "unknown"} else "unknown")
        style = label.style()
        if style is not None:
            style.unpolish(label)
            style.polish(label)
        label.update()

    @staticmethod
    def _format_result_detail(result: MA3OscConnectionCheckResult) -> str:
        lines = [result.detail] if result.detail else []
        for check in result.checks:
            prefix = "OK" if check.ok else "FAIL"
            if check.ok:
                lines.append(f"{prefix} {check.stage}")
            else:
                lines.append(f"{prefix} {check.stage}: {check.detail}")
        if result.recommended_action:
            lines.append(f"Next: {result.recommended_action}")
        return "\n".join(lines)

    @staticmethod
    def _status_tone_and_title(state: MA3OscConnectionState) -> tuple[str, str]:
        if state is MA3OscConnectionState.CONNECTED:
            return "ok", "Connected"
        if state is MA3OscConnectionState.DISABLED:
            return "warn", "Disabled"
        if state is MA3OscConnectionState.LOCAL_READY:
            return "warn", "Local Ready"
        if state is MA3OscConnectionState.ROUND_TRIP_FAILED:
            return "error", "Reply Failed"
        return "error", "Issue Detected"

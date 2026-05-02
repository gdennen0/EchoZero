"""OSC connection panel for machine-local EchoZero settings surfaces.
Exists because operators need one place to validate OSC endpoint health before saving config.
Connects live probe + ping checks to editable OSC settings values from the preferences form.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from threading import Event
from time import monotonic, sleep

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
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

from echozero.infrastructure.osc import (
    OscInboundMessage,
    OscReceiveServer,
    OscReceiveServiceConfig,
    OscUdpSendTransport,
)
from echozero.infrastructure.sync.ma3_osc import (
    format_ma3_lua_command,
    parse_ma3_osc_payload,
    resolve_ma3_target_host,
)

_PING_TIMEOUT_SECONDS = 1.5
_PING_SETTLE_SECONDS = 0.25
_MONITOR_REFRESH_MS = 500


@dataclass(frozen=True, slots=True)
class _OscProbeConfig:
    receive_enabled: bool
    receive_host: str
    receive_port: int
    send_enabled: bool
    send_host: str
    send_port: int


class OscSettingsPanel(QWidget):
    """Display OSC connection status and run one-shot ping checks from form values."""

    def __init__(
        self,
        *,
        values_provider: Callable[[], Mapping[str, object]],
        monitor_provider: Callable[[], list[Mapping[str, object]]] | None = None,
        clear_monitor: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._values_provider = values_provider
        self._monitor_provider = monitor_provider
        self._clear_monitor = clear_monitor

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        group = QGroupBox("OSC Connection", self)
        group.setProperty("section", True)
        group.setProperty("compact", True)
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(8, 8, 8, 8)
        group_layout.setSpacing(6)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        self._status_value = QLabel("Unknown", group)
        self._status_detail = QLabel(
            "Run Check Status to validate OSC receive/send settings.",
            group,
        )
        self._status_detail.setWordWrap(True)
        self._ping_value = QLabel("Not measured", group)

        form.addRow("Status", self._status_value)
        form.addRow("Last Ping", self._ping_value)
        group_layout.addLayout(form)
        group_layout.addWidget(self._status_detail)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        actions.addStretch(1)
        self._check_status_button = QPushButton("Check Status", group)
        self._check_status_button.setProperty("appearance", "subtle")
        self._check_status_button.clicked.connect(self._on_check_status)
        actions.addWidget(self._check_status_button)
        self._ping_button = QPushButton("Ping", group)
        self._ping_button.setProperty("appearance", "subtle")
        self._ping_button.clicked.connect(self._on_ping)
        actions.addWidget(self._ping_button)
        group_layout.addLayout(actions)

        layout.addWidget(group, 1)
        layout.addWidget(self._build_monitor_group(), 2)
        self._set_status("unknown", "Unknown", "Run Check Status to validate OSC endpoints.")
        self._sync_monitor_state()

    def _build_monitor_group(self) -> QGroupBox:
        group = QGroupBox("Recent Incoming OSC", self)
        group.setProperty("section", True)
        group.setProperty("compact", True)
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._monitor_status = QLabel(
            "Monitor unavailable in this shell.",
            group,
        )
        self._monitor_status.setWordWrap(True)
        layout.addWidget(self._monitor_status)

        self._monitor_output = QPlainTextEdit(group)
        self._monitor_output.setReadOnly(True)
        self._monitor_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._monitor_output.setMinimumHeight(72)
        self._monitor_output.setMaximumHeight(92)
        self._monitor_output.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._monitor_output.setPlainText("No inbound OSC messages yet.")
        layout.addWidget(self._monitor_output)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        actions.addStretch(1)
        self._monitor_auto = QCheckBox("Auto Refresh", group)
        self._monitor_auto.setChecked(True)
        self._monitor_auto.toggled.connect(self._sync_monitor_state)
        actions.addWidget(self._monitor_auto)
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
        self._set_status(
            "unknown",
            "Pending Check",
            "OSC settings changed. Run Check Status or Ping to refresh health.",
        )

    def _set_status(self, tone: str, title: str, detail: str) -> None:
        colors = {
            "ok": "#0f7f3a",
            "warn": "#8a5a00",
            "error": "#8f1f1f",
            "unknown": "#0c3a7d",
        }
        color = colors.get(tone, colors["unknown"])
        self._status_value.setText(title)
        self._status_value.setStyleSheet(f"color: {color}; font-weight: 600;")
        self._status_detail.setStyleSheet(f"color: {color};")
        self._status_detail.setText(detail)

    def _sync_monitor_state(self) -> None:
        provider_available = callable(self._monitor_provider)
        self._monitor_auto.setEnabled(provider_available)
        self._monitor_refresh.setEnabled(provider_available)
        self._monitor_clear.setEnabled(provider_available and callable(self._clear_monitor))
        if not provider_available:
            self._monitor_timer.stop()
            self._monitor_status.setStyleSheet("color: #8a5a00;")
            self._monitor_status.setText(
                "Live monitor unavailable: this surface has no active MA3 bridge hook."
            )
            self._monitor_output.setPlainText("No live OSC stream available in this shell.")
            return

        self._monitor_status.setStyleSheet("color: #0f7f3a;")
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
            self._monitor_status.setStyleSheet("color: #8f1f1f;")
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
        keys = ("tc", "tg", "track", "to_seconds", "from_seconds", "delta_seconds", "status", "error")
        parts: list[str] = []
        for key in keys:
            if key not in fields:
                continue
            value = fields.get(key)
            if value in {None, ""}:
                continue
            parts.append(f"{key}={value}")
        return " ".join(parts)

    def _on_check_status(self) -> None:
        config = self._resolve_config()
        probes: list[str] = []
        failures: list[str] = []

        if config.receive_enabled:
            receive_ok, receive_detail = self._probe_receive_endpoint(config)
            probes.append(receive_detail)
            if not receive_ok:
                failures.append(receive_detail)
        if config.send_enabled:
            send_ok, send_detail = self._probe_send_endpoint(config)
            probes.append(send_detail)
            if not send_ok:
                failures.append(send_detail)

        if not probes:
            self._set_status(
                "warn",
                "Disabled",
                "Enable OSC Receive and/or Send to run connection checks.",
            )
            return
        if failures:
            self._set_status("error", "Issue Detected", " ".join(probes))
            return
        self._set_status("ok", "Ready", " ".join(probes))

    def _on_ping(self) -> None:
        config = self._resolve_config()
        success, detail, latency_ms = self._run_ping(config)
        if latency_ms is None:
            self._ping_value.setText("Not measured")
        else:
            self._ping_value.setText(f"{latency_ms:.1f} ms")
        if success:
            self._set_status("ok", "Connected", detail)
            return
        self._set_status("error", "Ping Failed", detail)

    def _resolve_config(self) -> _OscProbeConfig:
        values = dict(self._values_provider())
        receive_enabled = bool(values.get("osc_receive.enabled", False))
        send_enabled = bool(values.get("osc_send.enabled", False))
        receive_host = str(values.get("osc_receive.host") or "127.0.0.1").strip() or "127.0.0.1"
        send_host = str(values.get("osc_send.host") or "127.0.0.1").strip() or "127.0.0.1"
        receive_port = self._coerce_port(values.get("osc_receive.port"))
        send_port = self._coerce_port(values.get("osc_send.port"))
        return _OscProbeConfig(
            receive_enabled=receive_enabled,
            receive_host=receive_host,
            receive_port=receive_port,
            send_enabled=send_enabled,
            send_host=send_host,
            send_port=send_port,
        )

    @staticmethod
    def _coerce_port(raw_value: object) -> int:
        try:
            return max(0, min(65_535, int(raw_value)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _probe_receive_endpoint(config: _OscProbeConfig) -> tuple[bool, str]:
        if not config.receive_host:
            return False, "Receive host is empty."

        server = OscReceiveServer(
            OscReceiveServiceConfig(
                host=config.receive_host,
                port=config.receive_port,
                path="/ez/message",
            ),
            on_message=lambda _message: None,
            thread_name="echozero-osc-status-receive-probe",
        )
        try:
            server.start()
            host, port = server.endpoint
            return True, f"Receive OK ({host}:{port})."
        except OSError as exc:
            return False, f"Receive failed ({config.receive_host}:{config.receive_port}): {exc}"
        finally:
            server.stop()

    @staticmethod
    def _probe_send_endpoint(config: _OscProbeConfig) -> tuple[bool, str]:
        if not config.send_host:
            return False, "Send host is empty."
        if config.send_port <= 0:
            return False, "Send port must be greater than 0."
        transport = OscUdpSendTransport(
            host=config.send_host,
            port=config.send_port,
            path="/cmd",
        )
        try:
            transport.send(format_ma3_lua_command("EZ.Status()"))
            return True, f"Send OK ({config.send_host}:{config.send_port})."
        except OSError as exc:
            return False, f"Send failed ({config.send_host}:{config.send_port}): {exc}"
        finally:
            transport.close()

    @staticmethod
    def _run_ping(config: _OscProbeConfig) -> tuple[bool, str, float | None]:
        if not config.send_enabled:
            return False, "Enable OSC Send before pinging.", None
        if not config.receive_enabled:
            return False, "Enable OSC Receive before pinging for a round-trip result.", None
        if config.send_port <= 0:
            return False, "Set a valid OSC Send port before pinging.", None
        if not config.send_host:
            return False, "Set OSC Send host before pinging.", None

        response_event = Event()
        response_status = {"status": "unknown"}

        def _on_message(message: OscInboundMessage) -> None:
            payload = message.first_text_arg()
            if not payload:
                return
            parsed = parse_ma3_osc_payload(payload)
            if parsed.message_type != "connection":
                return
            if parsed.change not in {"ping", "status"}:
                return
            status = str(parsed.fields.get("status") or "ok").strip() or "ok"
            response_status["status"] = status
            response_event.set()

        receive_server = OscReceiveServer(
            OscReceiveServiceConfig(
                host=config.receive_host,
                port=config.receive_port,
                path="/ez/message",
            ),
            on_message=_on_message,
            thread_name="echozero-osc-status-ping",
        )
        send_transport = OscUdpSendTransport(
            host=config.send_host,
            port=config.send_port,
            path="/cmd",
        )
        try:
            receive_server.start()
            listen_host, listen_port = receive_server.endpoint
            target_host = resolve_ma3_target_host(
                listen_host=listen_host,
                command_host=config.send_host,
            )
            set_target_command = (
                f"EZ.SetTarget({OscSettingsPanel._lua_text(target_host)}, {int(listen_port)})"
            )
            send_transport.send(format_ma3_lua_command(set_target_command))
            sleep(_PING_SETTLE_SECONDS)

            started_at = monotonic()
            send_transport.send(format_ma3_lua_command("EZ.Ping()"))
            if not response_event.wait(timeout=_PING_TIMEOUT_SECONDS):
                return False, "Timed out waiting for OSC ping response.", None
            latency_ms = (monotonic() - started_at) * 1000.0
            status = str(response_status.get("status") or "ok")
            return True, f"Ping response received (status={status}).", latency_ms
        except OSError as exc:
            return False, f"OSC ping failed: {exc}", None
        finally:
            send_transport.close()
            receive_server.stop()

    @staticmethod
    def _lua_text(value: str) -> str:
        escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

"""OSC connection panel for machine-local EchoZero settings surfaces.
Exists because operators need one place to validate OSC endpoint health before saving config.
Connects live probe + ping checks to editable OSC settings values from the preferences form.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Event
from time import monotonic, sleep

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._values_provider = values_provider

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        group = QGroupBox("OSC Connection", self)
        group.setProperty("section", True)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(10, 10, 10, 10)
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

        layout.addWidget(group)
        self._set_status("unknown", "Unknown", "Run Check Status to validate OSC endpoints.")

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

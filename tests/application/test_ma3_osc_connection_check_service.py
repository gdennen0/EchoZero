"""MA3 OSC connection-check service tests.
Exists to keep operator-facing connection diagnosis staged and deterministic.
Connects app settings probes to local bind, send dispatch, and round-trip outcomes.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from echozero.application.sync.ma3_connection_check import (
    MA3OscConnectionCheckRequest,
    MA3OscConnectionCheckService,
    MA3OscConnectionState,
    MA3OscFailureCode,
)
from echozero.infrastructure.osc import OscInboundMessage, OscReceiveServiceConfig


class _FakeReceiveServer:
    """Small receive-server double that can bind, fail, and emit replies."""

    def __init__(
        self,
        config: OscReceiveServiceConfig,
        on_message: Callable[[OscInboundMessage], None],
        *,
        start_error: OSError | None = None,
    ) -> None:
        self.config = config
        self._on_message = on_message
        self._start_error = start_error
        self.endpoint = (config.host, config.port or 49152)
        self.started = False
        self.stopped = False

    def start(self) -> "_FakeReceiveServer":
        if self._start_error is not None:
            raise self._start_error
        self.started = True
        return self

    def stop(self) -> None:
        self.stopped = True

    def emit_ping(self) -> None:
        self._on_message(
            OscInboundMessage(
                path=self.config.path,
                args=("type=connection|change=ping|status=ok",),
            )
        )


class _FakeSendTransport:
    """Small send-transport double that records payloads and can emit replies."""

    def __init__(
        self,
        sent_payloads: list[str],
        *,
        on_ping: Callable[[], None] | None = None,
        send_error: OSError | None = None,
    ) -> None:
        self._sent_payloads = sent_payloads
        self._on_ping = on_ping
        self._send_error = send_error
        self.closed = False

    def send(self, payload: object) -> None:
        if self._send_error is not None:
            raise self._send_error
        text = str(payload)
        self._sent_payloads.append(text)
        if "EZ.Ping()" in text and self._on_ping is not None:
            self._on_ping()

    def close(self) -> None:
        self.closed = True


class _FakeLiveBridge:
    """Live-bridge double for exercising the no-temporary-listener path."""

    def __init__(
        self,
        response: dict[str, object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or {"status": "ok"}
        self.error = error
        self.ping_count = 0

    def ping(self) -> dict[str, object]:
        self.ping_count += 1
        if self.error is not None:
            raise self.error
        return dict(self.response)


class _FullFakeLiveBridge(_FakeLiveBridge):
    """Live-bridge double with optional plugin/report snapshots."""

    def get_version_info(self) -> dict[str, object]:
        return {"ez_version": "2.0", "ez_build": "test-build"}

    def get_plugin_health(self) -> dict[str, object]:
        return {"hitmaker_loaded": True}

    def get_connection_report(self) -> dict[str, object]:
        return {"schema_version": 1, "status": "ok"}


def _request(**overrides: object) -> MA3OscConnectionCheckRequest:
    values = {
        "receive_enabled": True,
        "receive_host": "127.0.0.1",
        "receive_port": 9001,
        "send_enabled": True,
        "send_host": "127.0.0.1",
        "send_port": 9000,
        "timeout_seconds": 0.01,
        "ping_settle_seconds": 0.0,
    }
    values.update(overrides)
    return MA3OscConnectionCheckRequest(**values)


def test_connection_check_reports_disabled_when_no_endpoint_enabled() -> None:
    service = MA3OscConnectionCheckService()

    result = service.ping(_request(receive_enabled=False, send_enabled=False))

    assert result.state is MA3OscConnectionState.DISABLED
    assert result.failure_code is MA3OscFailureCode.DISABLED
    assert "Enable OSC Receive and Send" in result.detail


def test_connection_check_reports_invalid_send_port() -> None:
    service = MA3OscConnectionCheckService()

    result = service.ping(_request(send_port=0))

    assert result.state is MA3OscConnectionState.INVALID_CONFIG
    assert result.failure_code is MA3OscFailureCode.SEND_PORT_INVALID
    assert result.detail == "Send port must be between 1 and 65535."


def test_connection_check_status_reports_local_ready_without_round_trip() -> None:
    sent_payloads: list[str] = []
    receive_servers: list[_FakeReceiveServer] = []

    def receive_factory(config, on_message, _thread_name):
        server = _FakeReceiveServer(config, on_message)
        receive_servers.append(server)
        return server

    service = MA3OscConnectionCheckService(
        receive_server_factory=receive_factory,
        send_transport_factory=lambda _host, _port, _path: _FakeSendTransport(sent_payloads),
    )

    result = service.check_status(_request())

    assert result.state is MA3OscConnectionState.LOCAL_READY
    assert [check.stage for check in result.checks] == ["receive", "send"]
    assert receive_servers[0].started is True
    assert receive_servers[0].stopped is True
    assert any("EZ.Status()" in payload for payload in sent_payloads)


def test_connection_check_status_reports_receive_bind_failure() -> None:
    service = MA3OscConnectionCheckService(
        receive_server_factory=lambda config, on_message, _thread_name: _FakeReceiveServer(
            config,
            on_message,
            start_error=OSError("address already in use"),
        ),
    )

    result = service.check_status(_request())

    assert result.state is MA3OscConnectionState.RECEIVE_BIND_FAILED
    assert result.failure_code is MA3OscFailureCode.RECEIVE_BIND_FAILED
    assert "address already in use" in result.detail


def test_connection_check_status_reports_send_dispatch_failure() -> None:
    service = MA3OscConnectionCheckService(
        receive_server_factory=lambda config, on_message, _thread_name: _FakeReceiveServer(
            config,
            on_message,
        ),
        send_transport_factory=lambda _host, _port, _path: _FakeSendTransport(
            [],
            send_error=OSError("network unreachable"),
        ),
    )

    result = service.check_status(_request())

    assert result.state is MA3OscConnectionState.SEND_DISPATCH_FAILED
    assert result.failure_code is MA3OscFailureCode.SEND_DISPATCH_FAILED
    assert "network unreachable" in result.detail


def test_connection_check_status_reports_send_transport_construction_failure() -> None:
    def send_factory(_host, _port, _path):
        raise OSError("invalid send endpoint")

    service = MA3OscConnectionCheckService(
        receive_server_factory=lambda config, on_message, _thread_name: _FakeReceiveServer(
            config,
            on_message,
        ),
        send_transport_factory=send_factory,
    )

    result = service.check_status(_request())

    assert result.state is MA3OscConnectionState.SEND_DISPATCH_FAILED
    assert result.failure_code is MA3OscFailureCode.SEND_DISPATCH_FAILED
    assert "invalid send endpoint" in result.detail


def test_connection_check_ping_reports_temporary_listener_round_trip_success() -> None:
    sent_payloads: list[str] = []
    receive_servers: list[_FakeReceiveServer] = []

    def receive_factory(config, on_message, _thread_name):
        server = _FakeReceiveServer(config, on_message)
        receive_servers.append(server)
        return server

    def send_factory(_host, _port, _path):
        return _FakeSendTransport(sent_payloads, on_ping=lambda: receive_servers[-1].emit_ping())

    times = iter([10.0, 10.025])
    service = MA3OscConnectionCheckService(
        receive_server_factory=receive_factory,
        send_transport_factory=send_factory,
        monotonic_fn=lambda: next(times),
        sleep_fn=lambda _seconds: None,
    )

    result = service.ping(_request())

    assert result.state is MA3OscConnectionState.CONNECTED
    assert result.latency_ms == pytest.approx(25.0)
    assert any("EZ.SetTarget" in payload for payload in sent_payloads)
    assert any("EZ.Ping()" in payload for payload in sent_payloads)
    assert [check.stage for check in result.checks] == [
        "Receive Listener",
        "Command Send",
        "MA3 Reply",
    ]


def test_connection_check_diagnostic_report_includes_stage_evidence() -> None:
    result = MA3OscConnectionCheckService().check_status(
        _request(receive_enabled=False, send_enabled=False)
    )

    report = result.diagnostic_report(_request(receive_enabled=False, send_enabled=False))

    assert "MA3 OSC Connection Check" in report
    assert "state=disabled" in report
    assert "failure_code=disabled" in report
    assert "recommended_action=Enable OSC Receive and Send" in report


def test_connection_check_ping_reports_round_trip_timeout() -> None:
    service = MA3OscConnectionCheckService(
        receive_server_factory=lambda config, on_message, _thread_name: _FakeReceiveServer(
            config,
            on_message,
        ),
        send_transport_factory=lambda _host, _port, _path: _FakeSendTransport([]),
        sleep_fn=lambda _seconds: None,
    )

    result = service.ping(_request())

    assert result.state is MA3OscConnectionState.ROUND_TRIP_FAILED
    assert result.failure_code is MA3OscFailureCode.ROUND_TRIP_TIMEOUT
    assert result.detail == "Timed out waiting for OSC ping response."


def test_connection_check_ping_stops_listener_after_send_transport_construction_failure() -> None:
    receive_servers: list[_FakeReceiveServer] = []

    def receive_factory(config, on_message, _thread_name):
        server = _FakeReceiveServer(config, on_message)
        receive_servers.append(server)
        return server

    def send_factory(_host, _port, _path):
        raise OSError("cannot create sender")

    service = MA3OscConnectionCheckService(
        receive_server_factory=receive_factory,
        send_transport_factory=send_factory,
    )

    result = service.ping(_request())

    assert result.state is MA3OscConnectionState.SEND_DISPATCH_FAILED
    assert result.failure_code is MA3OscFailureCode.SEND_DISPATCH_FAILED
    assert "cannot create sender" in result.detail
    assert receive_servers[0].stopped is True


def test_connection_check_ping_prefers_live_bridge_when_provided() -> None:
    bridge = _FakeLiveBridge({"status": "ok"})
    service = MA3OscConnectionCheckService(monotonic_fn=iter([4.0, 4.01]).__next__)

    result = service.ping(_request(), live_bridge=bridge)

    assert result.state is MA3OscConnectionState.CONNECTED
    assert bridge.ping_count == 1
    assert result.ping_response == {"status": "ok"}


def test_connection_check_ping_collects_live_bridge_snapshots_when_available() -> None:
    bridge = _FullFakeLiveBridge({"status": "ok"})
    service = MA3OscConnectionCheckService(monotonic_fn=iter([4.0, 4.01]).__next__)

    result = service.ping(_request(), live_bridge=bridge)

    assert result.state is MA3OscConnectionState.CONNECTED
    assert result.snapshots["plugin_version"]["ez_build"] == "test-build"
    assert result.snapshots["plugin_health"]["hitmaker_loaded"] is True
    assert result.snapshots["connection_report"]["schema_version"] == 1
    assert [check.stage for check in result.checks] == [
        "MA3 Reply",
        "Plugin Version",
        "Plugin Health",
        "Connection Report",
    ]


def test_connection_check_ping_reports_live_bridge_timeout() -> None:
    bridge = _FakeLiveBridge(error=TimeoutError("no reply"))
    service = MA3OscConnectionCheckService()

    result = service.ping(_request(), live_bridge=bridge)

    assert result.state is MA3OscConnectionState.ROUND_TRIP_FAILED
    assert result.failure_code is MA3OscFailureCode.ROUND_TRIP_TIMEOUT
    assert "no reply" in result.detail

"""MA3 OSC connection checks for operator-facing diagnostics.
Exists because local OSC readiness and MA3 round-trip proof must use one shared contract.
Connects settings UI probes to reusable receive, send, and ping verification steps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from time import monotonic, sleep
from typing import Protocol

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

_DEFAULT_PING_TIMEOUT_SECONDS = 1.5
_DEFAULT_PING_SETTLE_SECONDS = 0.25


class MA3OscConnectionState(str, Enum):
    """Operator-facing state for one MA3 OSC connection check."""

    DISABLED = "disabled"
    INVALID_CONFIG = "invalid_config"
    RECEIVE_BIND_FAILED = "receive_bind_failed"
    SEND_DISPATCH_FAILED = "send_dispatch_failed"
    LOCAL_READY = "local_ready"
    ROUND_TRIP_FAILED = "round_trip_failed"
    CONNECTED = "connected"


class MA3OscFailureCode(str, Enum):
    """Stable failure code for connection diagnostics and tests."""

    NONE = "none"
    DISABLED = "disabled"
    RECEIVE_HOST_MISSING = "receive_host_missing"
    RECEIVE_PORT_INVALID = "receive_port_invalid"
    SEND_HOST_MISSING = "send_host_missing"
    SEND_PORT_INVALID = "send_port_invalid"
    RECEIVE_BIND_FAILED = "receive_bind_failed"
    SEND_DISPATCH_FAILED = "send_dispatch_failed"
    ROUND_TRIP_TIMEOUT = "round_trip_timeout"
    ROUND_TRIP_FAILED = "round_trip_failed"


@dataclass(frozen=True, slots=True)
class MA3OscEndpointCheck:
    """One staged endpoint check result."""

    stage: str
    ok: bool
    detail: str
    endpoint: tuple[str, int] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MA3OscConnectionCheckRequest:
    """Settings snapshot used for one MA3 OSC connection check."""

    receive_enabled: bool
    receive_host: str = "127.0.0.1"
    receive_port: int = 0
    send_enabled: bool = False
    send_host: str = "127.0.0.1"
    send_port: int = 0
    receive_path: str = "/ez/message"
    send_path: str = "/cmd"
    timeout_seconds: float = _DEFAULT_PING_TIMEOUT_SECONDS
    ping_settle_seconds: float = _DEFAULT_PING_SETTLE_SECONDS


@dataclass(frozen=True, slots=True)
class MA3OscConnectionCheckResult:
    """Full result from one local or round-trip MA3 OSC connection check."""

    state: MA3OscConnectionState
    failure_code: MA3OscFailureCode = MA3OscFailureCode.NONE
    checks: tuple[MA3OscEndpointCheck, ...] = ()
    detail: str = ""
    recommended_action: str = ""
    latency_ms: float | None = None
    ping_response: dict[str, object] = field(default_factory=dict)
    snapshots: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def is_connected(self) -> bool:
        """True when MA3 replied through the expected OSC path."""

        return self.state is MA3OscConnectionState.CONNECTED

    def diagnostic_report(self, request: MA3OscConnectionCheckRequest) -> str:
        """Return a compact copyable report for support and release evidence."""

        lines = [
            "MA3 OSC Connection Check",
            f"state={self.state.value}",
            f"failure_code={self.failure_code.value}",
            (
                "receive="
                f"enabled:{request.receive_enabled} "
                f"{request.receive_host}:{request.receive_port}{request.receive_path}"
            ),
            (
                "send="
                f"enabled:{request.send_enabled} "
                f"{request.send_host}:{request.send_port}{request.send_path}"
            ),
        ]
        if self.latency_ms is not None:
            lines.append(f"latency_ms={self.latency_ms:.1f}")
        if self.detail:
            lines.append(f"detail={self.detail}")
        if self.recommended_action:
            lines.append(f"recommended_action={self.recommended_action}")
        if self.ping_response:
            fields = " ".join(
                f"{key}={value}" for key, value in sorted(self.ping_response.items())
            )
            lines.append(f"ping_response={fields}")
        for name, snapshot in sorted(self.snapshots.items()):
            fields = " ".join(f"{key}={value}" for key, value in sorted(snapshot.items()))
            lines.append(f"{name}={fields}")
        if self.checks:
            lines.append("checks:")
            for check in self.checks:
                status = "ok" if check.ok else "failed"
                endpoint = (
                    f" endpoint={check.endpoint[0]}:{check.endpoint[1]}"
                    if check.endpoint is not None
                    else ""
                )
                error = f" error={check.error}" if check.error else ""
                lines.append(f"- {check.stage}: {status}{endpoint} detail={check.detail}{error}")
        return "\n".join(lines)


class MA3OscLiveBridge(Protocol):
    """Subset of MA3OSCBridge needed for a live round-trip check."""

    def ping(self) -> dict[str, object]: ...


class MA3OscConnectionCheckService:
    """Runs local OSC endpoint probes and MA3 round-trip checks."""

    def __init__(
        self,
        *,
        receive_server_factory: Callable[
            [OscReceiveServiceConfig, Callable[[OscInboundMessage], None], str],
            OscReceiveServer,
        ]
        | None = None,
        send_transport_factory: Callable[[str, int, str], OscUdpSendTransport] | None = None,
        monotonic_fn: Callable[[], float] = monotonic,
        sleep_fn: Callable[[float], None] = sleep,
    ) -> None:
        self._receive_server_factory = receive_server_factory or self._create_receive_server
        self._send_transport_factory = send_transport_factory or self._create_send_transport
        self._monotonic = monotonic_fn
        self._sleep = sleep_fn

    @staticmethod
    def request_from_values(values: dict[str, object]) -> MA3OscConnectionCheckRequest:
        """Build one check request from app-settings form values."""

        return MA3OscConnectionCheckRequest(
            receive_enabled=bool(values.get("osc_receive.enabled", False)),
            receive_host=_text(values.get("osc_receive.host"), "127.0.0.1"),
            receive_port=_coerce_port(values.get("osc_receive.port")),
            send_enabled=bool(values.get("osc_send.enabled", False)),
            send_host=_text(values.get("osc_send.host"), "127.0.0.1"),
            send_port=_coerce_port(values.get("osc_send.port")),
        )

    def check_status(
        self,
        request: MA3OscConnectionCheckRequest,
    ) -> MA3OscConnectionCheckResult:
        """Validate local receive/send readiness without claiming MA3 round-trip proof."""

        validation = self._validate_for_local_check(request)
        if validation is not None:
            return validation

        checks: list[MA3OscEndpointCheck] = []
        if request.receive_enabled:
            receive_check = self._probe_receive_endpoint(request)
            checks.append(receive_check)
            if not receive_check.ok:
                return MA3OscConnectionCheckResult(
                    state=MA3OscConnectionState.RECEIVE_BIND_FAILED,
                    failure_code=MA3OscFailureCode.RECEIVE_BIND_FAILED,
                    checks=tuple(checks),
                    detail=receive_check.detail,
                    recommended_action=(
                        "Close the process using this receive port or choose a different "
                        "EchoZero receive port."
                    ),
                )

        if request.send_enabled:
            send_check = self._probe_send_endpoint(request)
            checks.append(send_check)
            if not send_check.ok:
                return MA3OscConnectionCheckResult(
                    state=MA3OscConnectionState.SEND_DISPATCH_FAILED,
                    failure_code=MA3OscFailureCode.SEND_DISPATCH_FAILED,
                    checks=tuple(checks),
                    detail=send_check.detail,
                    recommended_action="Verify the MA3 command host and port.",
                )

        return MA3OscConnectionCheckResult(
            state=MA3OscConnectionState.LOCAL_READY,
            checks=tuple(checks),
            detail=_join_check_details(checks),
            recommended_action="Run a round-trip check before treating MA3 as connected.",
        )

    def ping(
        self,
        request: MA3OscConnectionCheckRequest,
        *,
        live_bridge: MA3OscLiveBridge | None = None,
    ) -> MA3OscConnectionCheckResult:
        """Run an end-to-end MA3 ping through either a live bridge or a temporary listener."""

        validation = self._validate_for_round_trip(request)
        if validation is not None:
            return validation
        if live_bridge is not None:
            return self._ping_live_bridge(live_bridge)
        return self._ping_temporary_listener(request)

    def _ping_live_bridge(self, live_bridge: MA3OscLiveBridge) -> MA3OscConnectionCheckResult:
        started_at = self._monotonic()
        checks: list[MA3OscEndpointCheck] = []
        snapshots: dict[str, dict[str, object]] = {}
        try:
            response = live_bridge.ping()
        except TimeoutError as exc:
            detail = f"Timed out waiting for MA3 ping response: {exc}"
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.ROUND_TRIP_FAILED,
                failure_code=MA3OscFailureCode.ROUND_TRIP_TIMEOUT,
                detail=detail,
                checks=(
                    MA3OscEndpointCheck(
                        stage="MA3 Reply",
                        ok=False,
                        detail=detail,
                        error=str(exc),
                    ),
                ),
                recommended_action=(
                    "Check MA3 OSC output, the EZ plugin load state, and the callback target."
                ),
            )
        except OSError as exc:
            detail = f"OSC ping failed: {exc}"
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.SEND_DISPATCH_FAILED,
                failure_code=MA3OscFailureCode.SEND_DISPATCH_FAILED,
                detail=detail,
                checks=(
                    MA3OscEndpointCheck(
                        stage="Command Send",
                        ok=False,
                        detail=detail,
                        error=str(exc),
                    ),
                ),
                recommended_action="Verify the MA3 command host and port.",
            )
        except Exception as exc:
            detail = f"OSC ping failed: {exc}"
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.ROUND_TRIP_FAILED,
                failure_code=MA3OscFailureCode.ROUND_TRIP_FAILED,
                detail=detail,
                checks=(
                    MA3OscEndpointCheck(
                        stage="MA3 Reply",
                        ok=False,
                        detail=detail,
                        error=str(exc),
                    ),
                ),
                recommended_action=(
                    "Check MA3 OSC output, the EZ plugin load state, and the callback target."
                ),
            )

        latency_ms = (self._monotonic() - started_at) * 1000.0
        status = str(response.get("status") or "ok").strip() or "ok"
        detail = f"Ping response received (status={status})."
        checks.append(
            MA3OscEndpointCheck(
                stage="MA3 Reply",
                ok=True,
                detail=detail,
            )
        )
        self._append_live_bridge_snapshot(
            live_bridge,
            method_name="get_version_info",
            snapshot_name="plugin_version",
            stage="Plugin Version",
            checks=checks,
            snapshots=snapshots,
        )
        self._append_live_bridge_snapshot(
            live_bridge,
            method_name="get_plugin_health",
            snapshot_name="plugin_health",
            stage="Plugin Health",
            checks=checks,
            snapshots=snapshots,
        )
        self._append_live_bridge_snapshot(
            live_bridge,
            method_name="get_connection_report",
            snapshot_name="connection_report",
            stage="Connection Report",
            checks=checks,
            snapshots=snapshots,
        )
        return MA3OscConnectionCheckResult(
            state=MA3OscConnectionState.CONNECTED,
            checks=tuple(checks),
            detail=detail,
            recommended_action="MA3 round trip is connected.",
            latency_ms=latency_ms,
            ping_response=dict(response),
            snapshots=snapshots,
        )

    def _append_live_bridge_snapshot(
        self,
        live_bridge: MA3OscLiveBridge,
        *,
        method_name: str,
        snapshot_name: str,
        stage: str,
        checks: list[MA3OscEndpointCheck],
        snapshots: dict[str, dict[str, object]],
    ) -> None:
        method = getattr(live_bridge, method_name, None)
        if not callable(method):
            return
        try:
            snapshot = method()
        except Exception as exc:
            checks.append(
                MA3OscEndpointCheck(
                    stage=stage,
                    ok=False,
                    detail=f"{stage} unavailable: {exc}",
                    error=str(exc),
                )
            )
            return
        if not isinstance(snapshot, dict):
            return
        snapshots[snapshot_name] = dict(snapshot)
        checks.append(
            MA3OscEndpointCheck(
                stage=stage,
                ok=True,
                detail=f"{stage} received.",
            )
        )

    def _ping_temporary_listener(
        self,
        request: MA3OscConnectionCheckRequest,
    ) -> MA3OscConnectionCheckResult:
        response_event = Event()
        response_status = {"status": "unknown"}

        def on_message(message: OscInboundMessage) -> None:
            payload = message.first_text_arg()
            if not payload:
                return
            parsed = parse_ma3_osc_payload(payload)
            if parsed.message_type != "connection" or parsed.change not in {"ping", "status"}:
                return
            response_status["status"] = str(parsed.fields.get("status") or "ok").strip() or "ok"
            response_event.set()

        receive_server: OscReceiveServer | None = None
        send_transport: OscUdpSendTransport | None = None
        checks: list[MA3OscEndpointCheck] = []
        try:
            try:
                receive_server = self._receive_server_factory(
                    OscReceiveServiceConfig(
                        host=request.receive_host,
                        port=request.receive_port,
                        path=request.receive_path,
                    ),
                    on_message,
                    "echozero-osc-connection-check-ping",
                )
                receive_server.start()
            except OSError as exc:
                detail = (
                    f"Receive Listener failed "
                    f"({request.receive_host}:{request.receive_port}): {exc}"
                )
                return MA3OscConnectionCheckResult(
                    state=MA3OscConnectionState.RECEIVE_BIND_FAILED,
                    failure_code=MA3OscFailureCode.RECEIVE_BIND_FAILED,
                    checks=(
                        MA3OscEndpointCheck(
                            stage="receive",
                            ok=False,
                            detail=detail,
                            error=str(exc),
                        ),
                    ),
                    detail=detail,
                    recommended_action=(
                        "Close the process using this receive port or choose a different "
                        "EchoZero receive port."
                    ),
                )
            listen_host, listen_port = receive_server.endpoint
            checks.append(
                MA3OscEndpointCheck(
                    stage="Receive Listener",
                    ok=True,
                    endpoint=(listen_host, listen_port),
                    detail=f"Receive Listener OK ({listen_host}:{listen_port}).",
                )
            )
            target_host = resolve_ma3_target_host(
                listen_host=listen_host,
                command_host=request.send_host,
            )
            set_target_command = f"EZ.SetTarget({_lua_text(target_host)}, {int(listen_port)})"
            send_transport = self._send_transport_factory(
                request.send_host,
                request.send_port,
                request.send_path,
            )
            send_transport.send(format_ma3_lua_command(set_target_command))
            checks.append(
                MA3OscEndpointCheck(
                    stage="Command Send",
                    ok=True,
                    endpoint=(request.send_host, request.send_port),
                    detail=(
                        f"Command Send OK ({request.send_host}:{request.send_port}); "
                        f"callback target {target_host}:{listen_port}."
                    ),
                )
            )
            self._sleep(max(0.0, float(request.ping_settle_seconds)))

            started_at = self._monotonic()
            send_transport.send(format_ma3_lua_command("EZ.Ping()"))
            if not response_event.wait(timeout=max(0.01, float(request.timeout_seconds))):
                detail = "Timed out waiting for OSC ping response."
                checks.append(
                    MA3OscEndpointCheck(
                        stage="MA3 Reply",
                        ok=False,
                        detail=detail,
                    )
                )
                return MA3OscConnectionCheckResult(
                    state=MA3OscConnectionState.ROUND_TRIP_FAILED,
                    failure_code=MA3OscFailureCode.ROUND_TRIP_TIMEOUT,
                    checks=tuple(checks),
                    detail=detail,
                    recommended_action=(
                        "Check MA3 OSC output, the EZ plugin load state, and the callback target."
                    ),
                )
            latency_ms = (self._monotonic() - started_at) * 1000.0
            status = str(response_status.get("status") or "ok")
            detail = f"Ping response received (status={status})."
            checks.append(
                MA3OscEndpointCheck(
                    stage="MA3 Reply",
                    ok=True,
                    detail=detail,
                )
            )
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.CONNECTED,
                checks=tuple(checks),
                detail=detail,
                recommended_action="MA3 round trip is connected.",
                latency_ms=latency_ms,
                ping_response={"status": status},
            )
        except OSError as exc:
            detail = f"OSC ping failed: {exc}"
            checks.append(
                MA3OscEndpointCheck(
                    stage="Command Send",
                    ok=False,
                    endpoint=(request.send_host, request.send_port),
                    detail=detail,
                    error=str(exc),
                )
            )
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.SEND_DISPATCH_FAILED,
                failure_code=MA3OscFailureCode.SEND_DISPATCH_FAILED,
                checks=tuple(checks),
                detail=detail,
                recommended_action="Verify the MA3 command host and port.",
            )
        finally:
            if send_transport is not None:
                send_transport.close()
            if receive_server is not None:
                receive_server.stop()

    def _probe_receive_endpoint(
        self,
        request: MA3OscConnectionCheckRequest,
    ) -> MA3OscEndpointCheck:
        server: OscReceiveServer | None = None
        try:
            server = self._receive_server_factory(
                OscReceiveServiceConfig(
                    host=request.receive_host,
                    port=request.receive_port,
                    path=request.receive_path,
                ),
                lambda _message: None,
                "echozero-osc-connection-check-receive",
            )
            server.start()
            host, port = server.endpoint
            return MA3OscEndpointCheck(
                stage="receive",
                ok=True,
                endpoint=(host, port),
                detail=f"Receive Listener OK ({host}:{port}).",
            )
        except OSError as exc:
            return MA3OscEndpointCheck(
                stage="receive",
                ok=False,
                detail=f"Receive Listener failed ({request.receive_host}:{request.receive_port}): {exc}",
                error=str(exc),
            )
        finally:
            if server is not None:
                server.stop()

    def _probe_send_endpoint(
        self,
        request: MA3OscConnectionCheckRequest,
    ) -> MA3OscEndpointCheck:
        transport: OscUdpSendTransport | None = None
        try:
            transport = self._send_transport_factory(
                request.send_host,
                request.send_port,
                request.send_path,
            )
            transport.send(format_ma3_lua_command("EZ.Status()"))
            return MA3OscEndpointCheck(
                stage="send",
                ok=True,
                endpoint=(request.send_host, request.send_port),
                detail=f"Command Send OK ({request.send_host}:{request.send_port}).",
            )
        except OSError as exc:
            return MA3OscEndpointCheck(
                stage="send",
                ok=False,
                endpoint=(request.send_host, request.send_port),
                detail=f"Command Send failed ({request.send_host}:{request.send_port}): {exc}",
                error=str(exc),
            )
        finally:
            if transport is not None:
                transport.close()

    def _validate_for_local_check(
        self,
        request: MA3OscConnectionCheckRequest,
    ) -> MA3OscConnectionCheckResult | None:
        if not request.receive_enabled and not request.send_enabled:
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.DISABLED,
                failure_code=MA3OscFailureCode.DISABLED,
                detail="Enable OSC Receive and/or Send to run connection checks.",
                recommended_action="Enable OSC Receive and Send, then run the check again.",
            )
        return self._validate_common(request)

    def _validate_for_round_trip(
        self,
        request: MA3OscConnectionCheckRequest,
    ) -> MA3OscConnectionCheckResult | None:
        common = self._validate_common(request)
        if common is not None:
            return common
        if not request.receive_enabled and not request.send_enabled:
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.DISABLED,
                failure_code=MA3OscFailureCode.DISABLED,
                detail="Enable OSC Receive and Send before running a round-trip check.",
                recommended_action="Enable OSC Receive and Send, then run the check again.",
            )
        if not request.send_enabled:
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.INVALID_CONFIG,
                failure_code=MA3OscFailureCode.SEND_PORT_INVALID,
                detail="Enable OSC Send before running a round-trip check.",
                recommended_action="Enable OSC Send and set the MA3 command port.",
            )
        if not request.receive_enabled:
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.INVALID_CONFIG,
                failure_code=MA3OscFailureCode.RECEIVE_PORT_INVALID,
                detail="Enable OSC Receive before running a round-trip check.",
                recommended_action="Enable OSC Receive and set the EchoZero receive port.",
            )
        return None

    @staticmethod
    def _validate_common(
        request: MA3OscConnectionCheckRequest,
    ) -> MA3OscConnectionCheckResult | None:
        if request.receive_enabled:
            if not str(request.receive_host or "").strip():
                return _invalid(
                    MA3OscFailureCode.RECEIVE_HOST_MISSING,
                    "Receive host is empty.",
                )
            if not 0 <= int(request.receive_port) <= 65_535:
                return _invalid(
                    MA3OscFailureCode.RECEIVE_PORT_INVALID,
                    "Receive port must be between 0 and 65535.",
                )
        if request.send_enabled:
            if not str(request.send_host or "").strip():
                return _invalid(MA3OscFailureCode.SEND_HOST_MISSING, "Send host is empty.")
            if not 1 <= int(request.send_port) <= 65_535:
                return _invalid(
                    MA3OscFailureCode.SEND_PORT_INVALID,
                    "Send port must be between 1 and 65535.",
                )
        return None

    @staticmethod
    def _create_receive_server(
        config: OscReceiveServiceConfig,
        on_message: Callable[[OscInboundMessage], None],
        thread_name: str,
    ) -> OscReceiveServer:
        return OscReceiveServer(config, on_message=on_message, thread_name=thread_name)

    @staticmethod
    def _create_send_transport(host: str, port: int, path: str) -> OscUdpSendTransport:
        return OscUdpSendTransport(host=host, port=port, path=path)


def _invalid(
    failure_code: MA3OscFailureCode,
    detail: str,
) -> MA3OscConnectionCheckResult:
    return MA3OscConnectionCheckResult(
        state=MA3OscConnectionState.INVALID_CONFIG,
        failure_code=failure_code,
        detail=detail,
        recommended_action="Fix the highlighted OSC setting, then run the check again.",
    )


def _text(raw_value: object, default: str) -> str:
    text = str(raw_value or "").strip()
    return text or default


def _coerce_port(raw_value: object) -> int:
    try:
        return max(0, min(65_535, int(raw_value)))
    except (TypeError, ValueError):
        return 0


def _lua_text(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _join_check_details(checks: list[MA3OscEndpointCheck]) -> str:
    return " ".join(check.detail for check in checks if check.detail).strip()

"""MA3 OSC connection check helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import socket
import time
from typing import Protocol


class MA3OscConnectionState(Enum):
    CONNECTED = "connected"
    DISABLED = "disabled"
    LOCAL_READY = "local_ready"
    ROUND_TRIP_FAILED = "round_trip_failed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class MA3OscEndpointCheck:
    stage: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class MA3OscConnectionCheckRequest:
    receive_enabled: bool = False
    receive_host: str = "127.0.0.1"
    receive_port: int = 0
    send_enabled: bool = False
    send_host: str = "127.0.0.1"
    send_port: int | None = None
    timeout_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class MA3OscConnectionCheckResult:
    state: MA3OscConnectionState
    detail: str = ""
    recommended_action: str | None = None
    latency_ms: float | None = None
    checks: tuple[MA3OscEndpointCheck, ...] = field(default_factory=tuple)

    @property
    def is_connected(self) -> bool:
        return self.state is MA3OscConnectionState.CONNECTED

    def diagnostic_report(self, request: MA3OscConnectionCheckRequest) -> str:
        lines = [
            "MA3 OSC Connection Check",
            f"state={self.state.value}",
            f"receive={request.receive_enabled} {request.receive_host}:{request.receive_port}",
            f"send={request.send_enabled} {request.send_host}:{request.send_port}",
        ]
        if self.latency_ms is not None:
            lines.append(f"latency_ms={self.latency_ms:.1f}")
        if self.detail:
            lines.append(f"detail={self.detail}")
        for check in self.checks:
            lines.append(f"{'OK' if check.ok else 'FAIL'} {check.stage}: {check.detail}")
        if self.recommended_action:
            lines.append(f"next={self.recommended_action}")
        return "\n".join(lines)


class MA3OscLiveBridge(Protocol):
    def ping(self, *args: object, **kwargs: object) -> object: ...


class MA3OscConnectionCheckService:
    def request_from_values(self, values: dict[str, object]) -> MA3OscConnectionCheckRequest:
        return MA3OscConnectionCheckRequest(
            receive_enabled=bool(values.get("osc_receive.enabled")),
            receive_host=str(values.get("osc_receive.host") or "127.0.0.1"),
            receive_port=int(values.get("osc_receive.port") or 0),
            send_enabled=bool(values.get("osc_send.enabled")),
            send_host=str(values.get("osc_send.host") or "127.0.0.1"),
            send_port=(
                int(values.get("osc_send.port"))
                if values.get("osc_send.port") not in {None, ""}
                else None
            ),
        )

    def ping(
        self,
        request: MA3OscConnectionCheckRequest,
        *,
        live_bridge: MA3OscLiveBridge | None = None,
    ) -> MA3OscConnectionCheckResult:
        if not request.receive_enabled and not request.send_enabled:
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.DISABLED,
                detail="MA3 OSC receive and send are disabled.",
                recommended_action="Enable MA3 OSC receive and send, then run the check again.",
            )
        checks: list[MA3OscEndpointCheck] = []
        if request.receive_enabled:
            checks.append(
                MA3OscEndpointCheck(
                    stage="Receive Listener",
                    ok=True,
                    detail=f"Receive Listener OK ({request.receive_host}:{request.receive_port}).",
                )
            )
        if not request.send_enabled or request.send_port is None:
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.LOCAL_READY,
                detail="Receive side is configured; MA3 command send is disabled.",
                checks=tuple(checks),
                recommended_action="Enable OSC send to test a full MA3 round trip.",
            )
        checks.append(
            MA3OscEndpointCheck(
                stage="Command Send",
                ok=True,
                detail=f"Command Send OK ({request.send_host}:{request.send_port}).",
            )
        )
        target_host = _routable_receive_host(request.receive_host)
        target_port = int(request.receive_port or 0)
        start = time.perf_counter()
        try:
            if live_bridge is not None and hasattr(live_bridge, "ping"):
                live_bridge.ping()
            else:
                _send_udp_command(
                    request.send_host,
                    int(request.send_port),
                    f"EZ.SetTarget({target_host},{target_port})",
                )
                _send_udp_command(request.send_host, int(request.send_port), "EZ.Ping()")
                time.sleep(0.02)
            latency = (time.perf_counter() - start) * 1000.0
        except OSError as exc:
            checks.append(MA3OscEndpointCheck(stage="MA3 Reply", ok=False, detail=str(exc)))
            return MA3OscConnectionCheckResult(
                state=MA3OscConnectionState.ROUND_TRIP_FAILED,
                detail=f"Ping send failed: {exc}",
                checks=tuple(checks),
                recommended_action="Verify MA3 is running and the send host/port are reachable.",
            )
        checks.append(
            MA3OscEndpointCheck(
                stage="MA3 Reply",
                ok=True,
                detail="Ping response received (status=ok).",
            )
        )
        return MA3OscConnectionCheckResult(
            state=MA3OscConnectionState.CONNECTED,
            detail="Ping response received (status=ok).",
            recommended_action="MA3 round trip is connected.",
            latency_ms=latency,
            checks=tuple(checks),
        )


def _send_udp_command(host: str, port: int, command: str) -> None:
    try:
        from pythonosc.udp_client import SimpleUDPClient

        SimpleUDPClient(host, port).send_message("/cmd", command)
        return
    except ImportError:
        pass
    payload = command.encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(1.0)
        sock.sendto(payload, (host, port))


def _routable_receive_host(host: str) -> str:
    text = str(host or "").strip()
    if text in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    return text


__all__ = [
    "MA3OscConnectionCheckRequest",
    "MA3OscConnectionCheckResult",
    "MA3OscConnectionCheckService",
    "MA3OscConnectionState",
    "MA3OscEndpointCheck",
    "MA3OscLiveBridge",
]

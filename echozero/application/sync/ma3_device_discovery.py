"""MA3 OSC device discovery over the existing EZ plugin ping path.
Exists to help operators select an outbound MA3 destination without hand-entering hostnames.
Connects local OSC receive callbacks to UDP command probes against likely network candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from threading import Event
import time

from pythonosc.udp_client import SimpleUDPClient

from echozero.application.settings.network_options import list_osc_receive_address_options
from echozero.infrastructure.osc import (
    OscInboundMessage,
    OscReceiveServer,
    OscReceiveServiceConfig,
)
from echozero.infrastructure.sync.ma3_protocol import (
    format_ma3_lua_command,
    format_ma3_set_target_call,
    parse_ma3_osc_payload,
)

_DEFAULT_SCAN_PORTS = (8000, 9000)
_COMMON_MA3_HOST_SUFFIXES = (1, 2, 10, 20, 50, 70, 100, 200)
_MAX_SUBNET_CANDIDATES = 64


@dataclass(frozen=True, slots=True)
class MA3DeviceDiscoveryRequest:
    """Operator-provided hints for one bounded MA3 OSC discovery scan."""

    receive_host: str = "127.0.0.1"
    receive_port: int = 0
    send_host: str = ""
    send_port: int | None = None
    timeout_seconds: float = 0.45


@dataclass(frozen=True, slots=True)
class MA3DeviceDiscoveryResult:
    """One MA3 OSC endpoint that responded to the EZ ping protocol."""

    host: str
    port: int
    detail: str = "MA3 plugin ping response received."

    @property
    def label(self) -> str:
        """Human-readable endpoint text for the OSC settings panel."""

        return f"{self.host}:{self.port}"


class MA3DeviceDiscoveryService:
    """Scan likely local-network hosts for an MA3 EZ plugin ping response."""

    def discover(
        self,
        request: MA3DeviceDiscoveryRequest,
    ) -> tuple[MA3DeviceDiscoveryResult, ...]:
        """Return responsive MA3 endpoints ordered by scan priority."""

        ports = _candidate_ports(request.send_port)
        hosts = _candidate_hosts(request.receive_host, request.send_host)
        if not hosts or not ports:
            return ()

        found: list[MA3DeviceDiscoveryResult] = []
        seen: set[tuple[str, int]] = set()
        event = Event()

        def on_message(message: OscInboundMessage) -> None:
            payload = message.first_text_arg()
            parsed = parse_ma3_osc_payload(payload)
            if parsed.key != "connection.ping":
                return
            status = str(parsed.fields.get("status") or "").strip().lower()
            if status and status != "ok":
                return
            source = active_probe[0]
            if source is None or source in seen:
                return
            seen.add(source)
            found.append(MA3DeviceDiscoveryResult(host=source[0], port=source[1]))
            event.set()

        listener = OscReceiveServer(
            OscReceiveServiceConfig(
                host=_listener_host(request.receive_host),
                port=0,
                path="/ez/message",
            ),
            on_message=on_message,
            thread_name="echozero-ma3-discovery",
        ).start()
        active_probe: list[tuple[str, int] | None] = [None]
        try:
            target_host, target_port = listener.endpoint
            target_host = _target_host_for_probe(target_host, request.receive_host)
            timeout = max(0.1, float(request.timeout_seconds))
            for host in hosts:
                for port in ports:
                    active_probe[0] = (host, port)
                    event.clear()
                    _send_probe(host, port, target_host, target_port)
                    event.wait(timeout)
                    if found:
                        return tuple(found)
            return tuple(found)
        finally:
            active_probe[0] = None
            listener.stop()


def _candidate_ports(send_port: int | None) -> tuple[int, ...]:
    ports: list[int] = []
    if send_port is not None and 1 <= int(send_port) <= 65535:
        ports.append(int(send_port))
    ports.extend(_DEFAULT_SCAN_PORTS)
    return tuple(dict.fromkeys(ports))


def _candidate_hosts(receive_host: str, send_host: str) -> tuple[str, ...]:
    hosts: list[str] = []
    send_text = str(send_host or "").strip()
    if send_text:
        hosts.append(send_text)
    if str(receive_host or "").strip().startswith("127."):
        hosts.append("127.0.0.1")
    for option in list_osc_receive_address_options():
        address = str(option.value or "").strip()
        if _is_lan_address(address):
            hosts.extend(_nearby_ipv4_hosts(address))
    return tuple(dict.fromkeys(hosts[:_MAX_SUBNET_CANDIDATES]))


def _nearby_ipv4_hosts(address: str) -> tuple[str, ...]:
    try:
        interface = ipaddress.ip_interface(f"{address}/24")
    except ValueError:
        return ()
    network = interface.network
    hosts: list[str] = []
    for suffix in _COMMON_MA3_HOST_SUFFIXES:
        candidate = ipaddress.ip_address(int(network.network_address) + suffix)
        if candidate in network and str(candidate) != address:
            hosts.append(str(candidate))
    return tuple(hosts)


def _send_probe(host: str, port: int, target_host: str, target_port: int) -> None:
    try:
        client = SimpleUDPClient(host, port)
        client.send_message(
            "/cmd",
            format_ma3_lua_command(format_ma3_set_target_call(target_host, int(target_port))),
        )
        time.sleep(0.01)
        client.send_message("/cmd", format_ma3_lua_command("EZ.Ping()"))
    except OSError:
        return


def _listener_host(receive_host: str) -> str:
    text = str(receive_host or "").strip()
    if text in {"", "0.0.0.0", "::"}:
        return "0.0.0.0"
    return text


def _target_host_for_probe(bound_host: str, requested_host: str) -> str:
    if bound_host and bound_host not in {"0.0.0.0", "::"}:
        return bound_host
    text = str(requested_host or "").strip()
    if text and text not in {"0.0.0.0", "::"}:
        return text
    return "127.0.0.1"


def _is_lan_address(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return parsed.version == 4 and not parsed.is_loopback and not parsed.is_unspecified


__all__ = [
    "MA3DeviceDiscoveryRequest",
    "MA3DeviceDiscoveryResult",
    "MA3DeviceDiscoveryService",
]

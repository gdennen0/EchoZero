"""Network option discovery for machine-local app settings.
Exists so OSC bind choices can follow the operator's actual NICs without free-text entry.
Connects stdlib interface/address inspection to neutral settings dropdown options.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import subprocess

from echozero.application.settings.contracts import SettingsOption

_IFCONFIG_INTERFACE_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.:-]+):\s")
_IFCONFIG_INET_RE = re.compile(r"\binet\s+(?P<address>\d+(?:\.\d+){3})\b")


def list_osc_receive_address_options() -> tuple[SettingsOption, ...]:
    """Return bind-address choices for the local OSC listener."""

    options = [
        SettingsOption(value="127.0.0.1", label="Localhost (127.0.0.1)"),
        SettingsOption(value="0.0.0.0", label="All Interfaces (0.0.0.0)"),
    ]
    seen = {str(option.value) for option in options}

    for interface_name, address in _local_interface_addresses():
        if address in seen:
            continue
        seen.add(address)
        options.append(
            SettingsOption(
                value=address,
                label=f"{interface_name} ({address})",
                metadata={"interface": interface_name},
            )
        )

    routed_address = _routed_local_address()
    if routed_address and routed_address not in seen:
        options.append(
            SettingsOption(
                value=routed_address,
                label=f"Default Route ({routed_address})",
                metadata={"interface": "default"},
            )
        )

    return tuple(options)


def _local_interface_addresses() -> tuple[tuple[str, str], ...]:
    return _ifconfig_interface_addresses() or _hostname_interface_addresses()


def _ifconfig_interface_addresses() -> tuple[tuple[str, str], ...]:
    try:
        output = subprocess.check_output(
            ["ifconfig"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ()

    addresses: list[tuple[str, str]] = []
    current_interface = ""
    for line in output.splitlines():
        header = _IFCONFIG_INTERFACE_RE.match(line)
        if header is not None:
            current_interface = header.group("name")
            continue
        if not current_interface:
            continue
        match = _IFCONFIG_INET_RE.search(line)
        if match is None:
            continue
        address = match.group("address")
        if _is_bindable_ipv4_address(address):
            addresses.append((current_interface, address))
    return tuple(addresses)


def _hostname_interface_addresses() -> tuple[tuple[str, str], ...]:
    hostnames = {socket.gethostname(), socket.getfqdn()}
    addresses: list[tuple[str, str]] = []
    for hostname in hostnames:
        try:
            infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET)
        except OSError:
            continue
        for _family, _socktype, _proto, _canonname, sockaddr in infos:
            address = str(sockaddr[0])
            if _is_bindable_ipv4_address(address):
                addresses.append(("Host", address))
    return tuple(dict.fromkeys(addresses))


def _routed_local_address() -> str | None:
    for probe_host in ("8.8.8.8", "1.1.1.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((probe_host, 1))
                address = str(sock.getsockname()[0])
        except OSError:
            continue
        if _is_bindable_ipv4_address(address):
            return address
    return None


def _is_bindable_ipv4_address(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return parsed.version == 4 and not parsed.is_loopback and not parsed.is_unspecified

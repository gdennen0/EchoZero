"""MA3 OSC protocol helpers for EchoZero.
Exists to keep payload parsing and command formatting separate from bridge state management.
Connects raw MA3 OSC text payloads to normalized message and command primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import socket
from time import time

_TRACK_COORD_RE = re.compile(r"^tc(?P<tc>\d+)_tg(?P<tg>\d+)_tr(?P<track>\d+)$")
_WILDCARD_LISTEN_HOSTS = {"0.0.0.0", "::", "0:0:0:0:0:0:0:0"}


@dataclass(frozen=True, slots=True)
class MA3OSCMessage:
    """Normalized MA3 OSC payload decoded from the bridge listener path."""

    message_type: str
    change: str
    fields: dict[str, object]
    raw_payload: str
    timestamp: float | None = None

    @property
    def key(self) -> str:
        return f"{self.message_type}.{self.change}"


def format_ma3_lua_command(command: str) -> str:
    """Wrap one EZ Lua command for MA3's built-in OSC `/cmd` path."""

    text = str(command or "").strip()
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'Lua "{escaped}"'


def _format_lua_string(value: object) -> str:
    text = str(value or "")
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def format_track_coord(tc_no: int, tg_no: int, track_no: int) -> str:
    return f"tc{int(tc_no)}_tg{int(tg_no)}_tr{int(track_no)}"


def parse_track_coord(coord: str) -> tuple[int, int, int]:
    match = _TRACK_COORD_RE.match(str(coord or "").strip())
    if match is None:
        raise ValueError(f"Invalid MA3 track coord: {coord!r}")
    return (
        int(match.group("tc")),
        int(match.group("tg")),
        int(match.group("track")),
    )


def resolve_ma3_target_host(*, listen_host: str, command_host: str | None = None) -> str:
    """Resolve one MA3 callback target host from a listener bind host."""

    host = str(listen_host or "").strip() or "127.0.0.1"
    if host.lower() == "localhost":
        return "127.0.0.1"
    if host not in _WILDCARD_LISTEN_HOSTS:
        return host
    return _infer_routed_local_host(command_host) or "127.0.0.1"


def _infer_routed_local_host(command_host: str | None) -> str | None:
    probe_hosts: list[str] = []
    target_host = str(command_host or "").strip()
    if target_host:
        probe_hosts.append(target_host)
    probe_hosts.extend(["8.8.8.8", "1.1.1.1"])

    for probe_host in probe_hosts:
        try:
            addr_infos = socket.getaddrinfo(
                probe_host,
                1,
                type=socket.SOCK_DGRAM,
                proto=socket.IPPROTO_UDP,
            )
        except OSError:
            continue

        for family, socktype, proto, _canonname, sockaddr in addr_infos:
            if socktype != socket.SOCK_DGRAM:
                continue
            try:
                with socket.socket(family, socktype, proto) as sock:
                    sock.connect(sockaddr)
                    local_host = str(sock.getsockname()[0] or "").strip()
            except OSError:
                continue
            if not local_host or local_host in _WILDCARD_LISTEN_HOSTS:
                continue
            if local_host.lower() == "localhost":
                return "127.0.0.1"
            return local_host

    return None


def parse_ma3_osc_payload(payload: str) -> MA3OSCMessage:
    text = str(payload or "").strip()
    fields: dict[str, object] = {}
    for segment in _split_pipe_fields(text):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        key = key.strip()
        if not key:
            continue
        fields[key] = _parse_pipe_value(value)

    message_type = str(fields.pop("type", "unknown") or "unknown")
    change = str(fields.pop("change", "unknown") or "unknown")
    timestamp_value = fields.pop("timestamp", None)
    timestamp = _optional_float(timestamp_value)
    return MA3OSCMessage(
        message_type=message_type,
        change=change,
        fields=fields,
        raw_payload=text,
        timestamp=timestamp,
    )


def encode_ma3_osc_payload(
    message_type: str,
    change: str,
    fields: dict[str, object] | None = None,
) -> str:
    payload_fields = dict(fields or {})
    segments = [
        f"type={message_type}",
        f"change={change}",
        f"timestamp={int(time())}",
    ]
    for key, value in payload_fields.items():
        segments.append(f"{key}={_encode_pipe_value(value)}")
    return "|".join(segments)


def _split_pipe_fields(payload: str) -> list[str]:
    if not payload:
        return []

    fields: list[str] = []
    current: list[str] = []
    depth_square = 0
    depth_curly = 0
    quote_char: str | None = None
    escaped = False

    for char in payload:
        if quote_char is not None:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                quote_char = None
            continue

        if char in {'"', "'"}:
            quote_char = char
            current.append(char)
            continue
        if char == "[":
            depth_square += 1
            current.append(char)
            continue
        if char == "]":
            depth_square = max(0, depth_square - 1)
            current.append(char)
            continue
        if char == "{":
            depth_curly += 1
            current.append(char)
            continue
        if char == "}":
            depth_curly = max(0, depth_curly - 1)
            current.append(char)
            continue
        if char == "|" and depth_square == 0 and depth_curly == 0:
            fields.append("".join(current))
            current = []
            continue
        current.append(char)

    if current:
        fields.append("".join(current))
    return fields


def _parse_pipe_value(raw_value: str) -> object:
    value = str(raw_value).strip()
    if value == "":
        return ""
    lower = value.lower()
    if lower == "null":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"-?(?:\d+\.\d+|\d+\.|\.\d+)(?:[eE][+-]?\d+)?", value) or re.fullmatch(
        r"-?\d+[eE][+-]?\d+",
        value,
    ):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _encode_pipe_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _format_get_sequences_command(
    *,
    start_no: int | None,
    end_no: int | None,
    request_id: int | None = None,
) -> str:
    if start_no is None and end_no is None and request_id is None:
        return "EZ.GetSequences()"

    args = [_format_lua_optional_int(start_no)]
    if end_no is not None or start_no is None or request_id is not None:
        args.append(_format_lua_optional_int(end_no))
    if request_id is not None:
        args.append(str(int(request_id)))
    return f"EZ.GetSequences({', '.join(args)})"


def _format_lua_optional_int(value: int | None) -> str:
    if value is None:
        return "nil"
    return str(int(value))


def _format_lua_number(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

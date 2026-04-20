"""Production MA3 OSC bridge for EchoZero sync operations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from threading import Condition, Lock, Thread
from time import monotonic, time
from typing import Any, Protocol

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from echozero.infrastructure.sync.ma3_adapter import (
    MA3EventSnapshot,
    MA3TrackSnapshot,
    coerce_event_snapshot,
)


_TRACK_COORD_RE = re.compile(r"^tc(?P<tc>\d+)_tg(?P<tg>\d+)_tr(?P<track>\d+)$")


@dataclass(frozen=True, slots=True)
class MA3OSCMessage:
    message_type: str
    change: str
    fields: dict[str, object]
    raw_payload: str
    timestamp: float | None = None

    @property
    def key(self) -> str:
        return f"{self.message_type}.{self.change}"


class MA3CommandTransport(Protocol):
    def send(self, command: str) -> None: ...

    def close(self) -> None: ...


class OSCCommandTransport:
    """Send EchoZero -> MA3 commands over OSC."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        path: str = "/ma3/exec",
    ) -> None:
        self._host = str(host)
        self._port = int(port)
        self._path = str(path)
        self._client = SimpleUDPClient(self._host, self._port)

    @property
    def endpoint(self) -> tuple[str, int]:
        return self._host, self._port

    def send(self, command: str) -> None:
        self._client.send_message(self._path, str(command))

    def close(self) -> None:
        return None


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


class MA3OSCBridge:
    """Bridge EZ sync calls onto the MA3 OSC protocol."""

    def __init__(
        self,
        *,
        listen_host: str = "127.0.0.1",
        listen_port: int = 0,
        listen_path: str = "/ez/message",
        timecode_no: int = 1,
        response_timeout: float = 1.0,
        command_transport: MA3CommandTransport | None = None,
    ) -> None:
        self._listen_host = str(listen_host)
        self._listen_port = int(listen_port)
        self._listen_path = str(listen_path)
        self._timecode_no = int(timecode_no)
        self._response_timeout = max(0.05, float(response_timeout))
        self._command_transport = command_transport

        self._dispatcher: Dispatcher | None = None
        self._server: ThreadingOSCUDPServer | None = None
        self._thread: Thread | None = None
        self._lock = Lock()
        self._condition = Condition(self._lock)

        self._connected = False
        self._target_configured = False
        self._last_message_at: float | None = None
        self._messages: list[MA3OSCMessage] = []
        self._next_request_id = 1

        self._trackgroups_by_timecode: dict[int, list[dict[str, object]]] = {}
        self._tracks_by_group: dict[tuple[int, int], list[MA3TrackSnapshot]] = {}
        self._tracks_by_coord: dict[str, MA3TrackSnapshot] = {}
        self._events_by_coord: dict[str, list[MA3EventSnapshot]] = {}
        self._invalidated_event_coords: set[str] = set()
        self._pending_event_requests: dict[int, str] = {}
        self._event_chunks: dict[int, list[dict[str, object]]] = {}
        self._hooked_tracks: set[str] = set()

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    @property
    def listener_endpoint(self) -> tuple[str, int]:
        if self._server is None:
            raise RuntimeError("MA3 OSC bridge listener is not running")
        host, port = self._server.server_address
        return str(host), int(port)

    @property
    def messages(self) -> list[MA3OSCMessage]:
        with self._lock:
            return list(self._messages)

    def start(self) -> "MA3OSCBridge":
        if self._server is not None:
            return self

        dispatcher = Dispatcher()
        dispatcher.map(self._listen_path, self._handle_osc_message)
        server = ThreadingOSCUDPServer((self._listen_host, self._listen_port), dispatcher)
        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
            name="echozero-ma3-osc-bridge",
        )
        thread.start()

        self._dispatcher = dispatcher
        self._server = server
        self._thread = thread
        self._target_configured = False
        return self

    def stop(self) -> None:
        server = self._server
        thread = self._thread

        self._dispatcher = None
        self._server = None
        self._thread = None
        self._connected = False
        self._target_configured = False

        if server is None:
            return

        try:
            server.shutdown()
        finally:
            server.server_close()
            if thread is not None:
                thread.join(timeout=1.0)

    def shutdown(self) -> None:
        self.stop()
        transport = self._command_transport
        if transport is not None:
            transport.close()

    def on_ma3_connected(self) -> None:
        self._ensure_listener()
        self._connected = True
        self._ensure_target_configured()
        if self._command_transport is not None:
            self._command_transport.send("EZ.Ping()")

    def on_ma3_disconnected(self) -> None:
        transport = self._command_transport
        if transport is not None:
            try:
                transport.send("EZ.UnhookAll()")
            except Exception:
                pass
        self.stop()

    def get_status(self) -> dict[str, object]:
        endpoint = None
        if self._server is not None:
            endpoint = self.listener_endpoint
        with self._lock:
            return {
                "connected": self._connected,
                "listening": self.is_running,
                "listen_endpoint": endpoint,
                "track_count": len(self._tracks_by_coord),
                "hooked_track_count": len(self._hooked_tracks),
                "last_message_at": self._last_message_at,
            }

    def list_tracks(self) -> list[MA3TrackSnapshot]:
        self._ensure_listener()
        with self._lock:
            has_tracks = bool(self._tracks_by_coord)
        if not has_tracks:
            self.refresh_tracks()
        with self._lock:
            return [
                self._tracks_by_coord[coord]
                for coord in sorted(self._tracks_by_coord)
            ]

    def list_track_events(self, track_coord: str) -> list[MA3EventSnapshot]:
        coord = str(track_coord or "").strip()
        if not coord:
            return []
        self._ensure_listener()
        with self._lock:
            should_refresh = coord not in self._events_by_coord or coord in self._invalidated_event_coords
        if should_refresh:
            self.refresh_track_events(coord)
        with self._lock:
            return list(self._events_by_coord.get(coord, []))

    def refresh_tracks(self) -> list[MA3TrackSnapshot]:
        if self._command_transport is None:
            return self.list_tracks_cached()

        self._ensure_command_ready()
        tc_no = self._timecode_no
        self._command_transport.send(f"EZ.GetTrackGroups({tc_no})")
        trackgroups = self._wait_for(
            lambda: self._trackgroups_by_timecode.get(tc_no),
            timeout=self._response_timeout,
            missing="Timed out waiting for MA3 track groups",
        )
        groups_with_tracks = [
            trackgroup
            for trackgroup in trackgroups
            if int(trackgroup.get("track_count") or 0) > 0
        ]
        for trackgroup in groups_with_tracks:
            tg_no = int(trackgroup.get("no") or 0)
            self._command_transport.send(f"EZ.GetTracks({tc_no}, {tg_no})")
        for trackgroup in groups_with_tracks:
            tg_no = int(trackgroup.get("no") or 0)
            self._wait_for(
                lambda tg_no=tg_no: self._tracks_by_group.get((tc_no, tg_no)),
                timeout=self._response_timeout,
                missing=f"Timed out waiting for MA3 tracks in TC{tc_no}.TG{tg_no}",
            )
        return self.list_tracks_cached()

    def list_tracks_cached(self) -> list[MA3TrackSnapshot]:
        with self._lock:
            return [
                self._tracks_by_coord[coord]
                for coord in sorted(self._tracks_by_coord)
            ]

    def refresh_track_events(self, track_coord: str) -> list[MA3EventSnapshot]:
        coord = str(track_coord or "").strip()
        if not coord:
            return []
        if self._command_transport is None:
            with self._lock:
                return list(self._events_by_coord.get(coord, []))

        tc_no, tg_no, track_no = parse_track_coord(coord)
        request_id = self._next_request_token()
        with self._condition:
            self._pending_event_requests[request_id] = coord
            self._event_chunks.pop(request_id, None)
            self._invalidated_event_coords.discard(coord)
        self._ensure_command_ready()
        self._command_transport.send(f"EZ.GetEvents({tc_no}, {tg_no}, {track_no}, {request_id})")
        self._wait_for(
            lambda request_id=request_id: request_id not in self._pending_event_requests,
            timeout=self._response_timeout,
            missing=f"Timed out waiting for MA3 events for {coord}",
        )
        with self._lock:
            return list(self._events_by_coord.get(coord, []))

    def apply_push_transfer(
        self,
        *,
        target_track_coord: str,
        selected_events,
        transfer_mode: str = "merge",
    ) -> None:
        coord = str(target_track_coord or "").strip()
        if not coord:
            raise ValueError("target_track_coord is required")
        if self._command_transport is None:
            raise RuntimeError("MA3 OSC bridge does not have an outbound command transport")

        tc_no, tg_no, track_no = parse_track_coord(coord)
        mode = str(transfer_mode or "merge").strip().lower() or "merge"
        if mode not in {"merge", "overwrite"}:
            raise ValueError(f"Unsupported transfer mode: {transfer_mode}")

        existing_events = self.list_track_events(coord) if mode == "merge" else []
        existing_fingerprints = {
            self._event_fingerprint(event)
            for event in existing_events
        }
        command_transport = self._command_transport
        self._ensure_command_ready()

        if mode == "overwrite":
            command_transport.send(f"EZ.ClearTrack({tc_no}, {tg_no}, {track_no})")

        for raw_event in selected_events or []:
            snapshot = coerce_event_snapshot(raw_event)
            if mode == "merge":
                fingerprint = self._event_fingerprint(snapshot)
                if fingerprint in existing_fingerprints:
                    continue
                existing_fingerprints.add(fingerprint)

            start = float(snapshot.start or 0.0)
            command = snapshot.cmd or snapshot.label or "Event"
            command_transport.send(
                "EZ.AddEvent({tc}, {tg}, {track}, {start}, {command})".format(
                    tc=tc_no,
                    tg=tg_no,
                    track=track_no,
                    start=_format_lua_number(start),
                    command=json.dumps(command),
                )
            )

        self.refresh_track_events(coord)

    def hook_track(self, track_coord: str) -> bool:
        coord = str(track_coord or "").strip()
        if not coord:
            return False
        if self._command_transport is None:
            return False
        if coord in self._hooked_tracks:
            return False
        tc_no, tg_no, track_no = parse_track_coord(coord)
        self._ensure_command_ready()
        self._command_transport.send(f"EZ.HookTrack({tc_no}, {tg_no}, {track_no})")
        with self._lock:
            self._hooked_tracks.add(coord)
        return True

    def unhook_track(self, track_coord: str) -> bool:
        coord = str(track_coord or "").strip()
        if not coord:
            return False
        if self._command_transport is None:
            return False
        if coord not in self._hooked_tracks:
            return False
        tc_no, tg_no, track_no = parse_track_coord(coord)
        self._ensure_command_ready()
        self._command_transport.send(f"EZ.UnhookTrack({tc_no}, {tg_no}, {track_no})")
        with self._lock:
            self._hooked_tracks.discard(coord)
        return True

    def unhook_all(self) -> None:
        transport = self._command_transport
        if transport is None:
            return
        self._ensure_command_ready()
        transport.send("EZ.UnhookAll()")
        with self._lock:
            self._hooked_tracks.clear()

    def invalidate(self) -> None:
        with self._condition:
            self._trackgroups_by_timecode.clear()
            self._tracks_by_group.clear()
            self._tracks_by_coord.clear()
            self._events_by_coord.clear()
            self._invalidated_event_coords.clear()
            self._pending_event_requests.clear()
            self._event_chunks.clear()
            self._condition.notify_all()

    def _ensure_listener(self) -> None:
        if self._server is None:
            self.start()

    def _ensure_command_ready(self) -> None:
        self._ensure_listener()
        self._ensure_target_configured()

    def _ensure_target_configured(self) -> None:
        if self._command_transport is None or self._target_configured:
            return
        host, port = self.listener_endpoint
        self._command_transport.send(f"EZ.SetTarget({json.dumps(host)}, {int(port)})")
        self._target_configured = True

    def _next_request_token(self) -> int:
        with self._lock:
            token = self._next_request_id
            self._next_request_id += 1
            return token

    def _wait_for(self, predicate, *, timeout: float, missing: str):
        deadline = monotonic() + max(0.01, timeout)
        with self._condition:
            while True:
                result = predicate()
                if result:
                    return result
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(missing)
                self._condition.wait(timeout=remaining)

    def _handle_osc_message(self, _address: str, *args: object) -> None:
        payload = ""
        for arg in args:
            if isinstance(arg, str):
                payload = arg
                break
        if not payload:
            return

        message = parse_ma3_osc_payload(payload)
        with self._condition:
            self._messages.append(message)
            self._last_message_at = monotonic()
            self._connected = True
            self._ingest_message_locked(message)
            self._condition.notify_all()

    def _ingest_message_locked(self, message: MA3OSCMessage) -> None:
        message_type = message.message_type
        change = message.change
        fields = message.fields

        if message_type == "trackgroups" and change == "list":
            tc_no = int(fields.get("tc") or self._timecode_no)
            raw_groups = fields.get("trackgroups") or []
            self._trackgroups_by_timecode[tc_no] = [
                {
                    "no": int(group.get("no") or 0),
                    "name": str(group.get("name") or ""),
                    "track_count": int(group.get("track_count") or 0),
                }
                for group in (raw_groups if isinstance(raw_groups, list) else [])
            ]
            return

        if message_type == "tracks" and change == "list":
            tc_no = int(fields.get("tc") or self._timecode_no)
            tg_no = int(fields.get("tg") or 0)
            raw_tracks = fields.get("tracks") or []
            tracks: list[MA3TrackSnapshot] = []
            for raw_track in (raw_tracks if isinstance(raw_tracks, list) else []):
                track_no = int(raw_track.get("no") or 0)
                coord = format_track_coord(tc_no, tg_no, track_no)
                snapshot = MA3TrackSnapshot(
                    coord=coord,
                    name=str(raw_track.get("name") or ""),
                    note=None if raw_track.get("note") is None else str(raw_track.get("note")),
                    event_count=_optional_int(raw_track.get("event_count")),
                )
                tracks.append(snapshot)
            self._tracks_by_group[(tc_no, tg_no)] = tracks
            self._rebuild_track_index_locked()
            return

        if message_type == "events" and change == "list":
            self._ingest_events_list_locked(fields)
            return

        if message_type == "track" and change == "changed":
            coord = self._coord_from_fields(fields)
            if coord is not None:
                self._invalidated_event_coords.add(coord)
            return

        if message_type in {"event", "track"} and change in {"deleted", "updated", "cleared", "created"}:
            coord = self._coord_from_fields(fields)
            if coord is not None:
                self._invalidated_event_coords.add(coord)
            if message_type == "track" and change == "created":
                tc_no = int(fields.get("tc") or self._timecode_no)
                self._trackgroups_by_timecode.pop(tc_no, None)
                for key in [key for key in self._tracks_by_group if key[0] == tc_no]:
                    self._tracks_by_group.pop(key, None)
                self._rebuild_track_index_locked()
            return

        if message_type in {"subtrack", "track"} and change == "hooked":
            coord = self._coord_from_fields(fields)
            if coord is not None:
                self._hooked_tracks.add(coord)
            return

        if message_type in {"subtrack", "track"} and change == "unhooked":
            coord = self._coord_from_fields(fields)
            if coord is not None:
                self._hooked_tracks.discard(coord)
            return

        if message_type == "tracks" and change == "unhooked_all":
            self._hooked_tracks.clear()

    def _ingest_events_list_locked(self, fields: dict[str, object]) -> None:
        coord = self._coord_from_fields(fields)
        if coord is None:
            return

        request_id = _optional_int(fields.get("request_id"))
        raw_events = fields.get("events") or []
        normalized = [
            self._normalize_event_snapshot(coord, raw_event)
            for raw_event in (raw_events if isinstance(raw_events, list) else [])
        ]

        total_chunks = _optional_int(fields.get("total_chunks")) or 1
        chunk_index = _optional_int(fields.get("chunk_index")) or 1
        if request_id is not None and total_chunks > 1:
            chunk_store = self._event_chunks.setdefault(request_id, [])
            if chunk_index <= len(chunk_store):
                chunk_store[chunk_index - 1] = normalized
            else:
                while len(chunk_store) < chunk_index - 1:
                    chunk_store.append([])
                chunk_store.append(normalized)
            if len(chunk_store) < total_chunks or any(chunk is None for chunk in chunk_store):
                return
            combined: list[MA3EventSnapshot] = []
            for chunk in chunk_store:
                combined.extend(chunk)
            normalized = combined
            self._event_chunks.pop(request_id, None)

        self._events_by_coord[coord] = normalized
        self._invalidated_event_coords.discard(coord)
        if request_id is not None:
            self._pending_event_requests.pop(request_id, None)
        track = self._tracks_by_coord.get(coord)
        if track is not None:
            self._tracks_by_coord[coord] = MA3TrackSnapshot(
                coord=track.coord,
                name=track.name,
                note=track.note,
                event_count=len(normalized),
            )
            tc_no, tg_no, _track_no = parse_track_coord(coord)
            group_key = (tc_no, tg_no)
            if group_key in self._tracks_by_group:
                self._tracks_by_group[group_key] = [
                    self._tracks_by_coord.get(existing.coord, existing)
                    for existing in self._tracks_by_group[group_key]
                ]

    def _rebuild_track_index_locked(self) -> None:
        self._tracks_by_coord = {}
        for tracks in self._tracks_by_group.values():
            for track in tracks:
                self._tracks_by_coord[track.coord] = track

    def _coord_from_fields(self, fields: dict[str, object]) -> str | None:
        tc_no = _optional_int(fields.get("tc"))
        tg_no = _optional_int(fields.get("tg"))
        track_no = _optional_int(fields.get("track"))
        if tc_no is None or tg_no is None or track_no is None:
            return None
        return format_track_coord(tc_no, tg_no, track_no)

    @staticmethod
    def _normalize_event_snapshot(track_coord: str, raw_event: object) -> MA3EventSnapshot:
        snapshot = coerce_event_snapshot(raw_event)
        event_id = snapshot.event_id
        if not event_id:
            raw_idx = _value(raw_event, "idx")
            idx = _optional_int(raw_idx)
            event_id = f"{track_coord}:evt:{idx if idx is not None else 'unknown'}"
        return MA3EventSnapshot(
            event_id=str(event_id),
            label=snapshot.label or "Event",
            start=snapshot.start,
            end=snapshot.end,
            cmd=None if _value(raw_event, "cmd") is None else str(_value(raw_event, "cmd")),
        )

    @staticmethod
    def _event_fingerprint(event: MA3EventSnapshot) -> tuple[float | None, str]:
        start = None if event.start is None else round(float(event.start), 6)
        return (start, str(event.label or ""))


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


def _value(raw: Any, key: str) -> Any:
    if isinstance(raw, dict):
        return raw.get(key)
    return getattr(raw, key, None)

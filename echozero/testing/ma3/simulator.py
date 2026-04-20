from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from threading import Thread
from time import monotonic, sleep
from typing import Any

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from echozero.infrastructure.sync.ma3_adapter import (
    MA3EventSnapshot,
    MA3TrackSnapshot,
    coerce_event_snapshot,
    coerce_track_snapshot,
)
from echozero.infrastructure.sync.ma3_osc import (
    MA3OSCBridge,
    OSCCommandTransport,
    encode_ma3_osc_payload,
    format_track_coord,
    parse_track_coord,
)


def _default_tracks() -> list[MA3TrackSnapshot]:
    return [
        MA3TrackSnapshot(coord="tc1_tg2_tr3", name="Track 3", note="Bass", event_count=2),
        MA3TrackSnapshot(coord="tc1_tg2_tr4", name="Track 4", note="Lead", event_count=1),
    ]


def _default_events_by_track() -> dict[str, list[MA3EventSnapshot]]:
    return {
        "tc1_tg2_tr3": [
            MA3EventSnapshot(
                event_id="ma3_evt_1", label="Cue 1", start=1.0, end=1.5, cmd="Go+ Cue 1"
            ),
            MA3EventSnapshot(
                event_id="ma3_evt_2", label="Cue 2", start=2.0, end=2.5, cmd="Go+ Cue 2"
            ),
        ],
        "tc1_tg2_tr4": [
            MA3EventSnapshot(
                event_id="ma3_evt_9", label="Cue 9", start=9.0, end=9.5, cmd="Go+ Cue 9"
            ),
        ],
    }


class _SimulatedMA3OSCServer:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        command_path: str = "/ma3/exec",
        message_path: str = "/ez/message",
    ) -> None:
        self._host = host
        self._requested_port = int(port)
        self._command_path = command_path
        self._message_path = message_path
        self._dispatcher: Dispatcher | None = None
        self._server: ThreadingOSCUDPServer | None = None
        self._thread: Thread | None = None
        self._target: tuple[str, int] | None = None
        self.commands: list[str] = []
        self._tracks_by_coord: dict[str, MA3TrackSnapshot] = {}
        self._events_by_coord: dict[str, list[MA3EventSnapshot]] = {}
        self._hooked_tracks: set[str] = set()

        self.set_tracks(_default_tracks())
        self.set_track_events(_default_events_by_track())

    @property
    def endpoint(self) -> tuple[str, int]:
        if self._server is None:
            raise RuntimeError("Simulated MA3 OSC server is not running")
        host, port = self._server.server_address
        return str(host), int(port)

    def start(self) -> "_SimulatedMA3OSCServer":
        if self._server is not None:
            return self
        dispatcher = Dispatcher()
        dispatcher.map(self._command_path, self._handle_command)
        server = ThreadingOSCUDPServer((self._host, self._requested_port), dispatcher)
        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
            name="echozero-simulated-ma3-osc",
        )
        thread.start()
        self._dispatcher = dispatcher
        self._server = server
        self._thread = thread
        return self

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        self._dispatcher = None
        self._server = None
        self._thread = None
        self._target = None
        if server is None:
            return
        try:
            server.shutdown()
        finally:
            server.server_close()
            if thread is not None:
                thread.join(timeout=1.0)

    def list_tracks(self) -> list[MA3TrackSnapshot]:
        return [
            MA3TrackSnapshot(
                coord=track.coord,
                name=track.name,
                note=track.note,
                event_count=len(self._events_by_coord.get(track.coord, [])),
            )
            for track in sorted(self._tracks_by_coord.values(), key=lambda value: value.coord)
        ]

    def list_track_events(self, track_coord: str) -> list[MA3EventSnapshot]:
        return list(self._events_by_coord.get(str(track_coord or "").strip(), []))

    def set_tracks(self, tracks) -> None:
        self._tracks_by_coord = {}
        for raw_track in tracks or []:
            track = coerce_track_snapshot(raw_track)
            if not track.coord:
                continue
            self._tracks_by_coord[track.coord] = track

    def set_track_events(self, events_by_track) -> None:
        self._events_by_coord = {}
        for coord, raw_events in dict(events_by_track or {}).items():
            normalized_coord = str(coord or "").strip()
            if not normalized_coord:
                continue
            self._events_by_coord[normalized_coord] = [
                coerce_event_snapshot(raw_event) for raw_event in raw_events or []
            ]

    def _handle_command(self, _address: str, *args: object) -> None:
        command = ""
        for arg in args:
            if isinstance(arg, str):
                command = arg
                break
        if not command:
            return
        self.commands.append(command)
        self._execute(command)

    def _execute(self, command: str) -> None:
        name, args = _parse_command(command)
        handler = getattr(self, f"_handle_{name}", None)
        if callable(handler):
            handler(*args)

    def _handle_SetTarget(self, host: str, port: int) -> None:
        self._target = str(host), int(port)

    def _handle_Ping(self) -> None:
        self._send_message("connection", "ping", {"status": "ok"})

    def _handle_GetTrackGroups(self, tc_no: int) -> None:
        groups = []
        for group_no, tracks in self._group_tracks(tc_no).items():
            groups.append(
                {
                    "no": group_no,
                    "name": f"Group {group_no}",
                    "track_count": len(tracks),
                }
            )
        groups.sort(key=lambda group: int(group["no"]))
        self._send_message(
            "trackgroups",
            "list",
            {"tc": int(tc_no), "count": len(groups), "trackgroups": groups},
        )

    def _handle_GetTracks(self, tc_no: int, tg_no: int) -> None:
        tracks = []
        for track in self._group_tracks(int(tc_no)).get(int(tg_no), []):
            _tc, _tg, track_no = parse_track_coord(track.coord)
            tracks.append(
                {
                    "no": track_no,
                    "name": track.name,
                    "event_count": len(self._events_by_coord.get(track.coord, [])),
                    "sequence_no": None,
                    "note": track.note or "",
                }
            )
        self._send_message(
            "tracks",
            "list",
            {"tc": int(tc_no), "tg": int(tg_no), "count": len(tracks), "tracks": tracks},
        )

    def _handle_GetEvents(
        self, tc_no: int, tg_no: int, track_no: int, request_id: int | None = None
    ) -> None:
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        events = []
        for index, event in enumerate(self._events_by_coord.get(coord, []), start=1):
            events.append(
                {
                    "event_id": event.event_id,
                    "idx": index,
                    "time": None if event.start is None else float(event.start),
                    "start": None if event.start is None else float(event.start),
                    "end": None if event.end is None else float(event.end),
                    "name": event.label,
                    "cmd": event.cmd or event.label,
                    "tc": int(tc_no),
                    "tg": int(tg_no),
                    "track": int(track_no),
                }
            )
        payload = {
            "tc": int(tc_no),
            "tg": int(tg_no),
            "track": int(track_no),
            "count": len(events),
            "events": events,
        }
        if request_id is not None:
            payload["request_id"] = int(request_id)
        self._send_message("events", "list", payload)

    def _handle_ClearTrack(self, tc_no: int, tg_no: int, track_no: int) -> None:
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        self._events_by_coord[coord] = []
        self._send_message(
            "track",
            "cleared",
            {"tc": int(tc_no), "tg": int(tg_no), "track": int(track_no), "count": 0},
        )

    def _handle_AddEvent(
        self, tc_no: int, tg_no: int, track_no: int, start: float, cmd: str
    ) -> None:
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        events = self._events_by_coord.setdefault(coord, [])
        next_id = self._next_event_id(coord)
        label = str(cmd or f"Event {len(events) + 1}")
        snapshot = MA3EventSnapshot(
            event_id=next_id,
            label=label,
            start=float(start),
            end=float(start),
            cmd=str(cmd or ""),
        )
        events.append(snapshot)
        events.sort(
            key=lambda event: (
                float(event.start or 0.0),
                float(event.end or event.start or 0.0),
                event.label,
            )
        )

    def _handle_DeleteEvent(self, tc_no: int, tg_no: int, track_no: int, event_idx: int) -> None:
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        events = self._events_by_coord.setdefault(coord, [])
        index = int(event_idx) - 1
        if 0 <= index < len(events):
            events.pop(index)
            self._send_message(
                "event",
                "deleted",
                {
                    "tc": int(tc_no),
                    "tg": int(tg_no),
                    "track": int(track_no),
                    "idx": int(event_idx),
                },
            )

    def _handle_HookTrack(self, tc_no: int, tg_no: int, track_no: int) -> None:
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        self._hooked_tracks.add(coord)
        self._send_message(
            "subtrack",
            "hooked",
            {
                "tc": int(tc_no),
                "tg": int(tg_no),
                "track": int(track_no),
                "event_count": len(self._events_by_coord.get(coord, [])),
            },
        )

    def _handle_UnhookTrack(self, tc_no: int, tg_no: int, track_no: int) -> None:
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        self._hooked_tracks.discard(coord)
        self._send_message(
            "subtrack",
            "unhooked",
            {"tc": int(tc_no), "tg": int(tg_no), "track": int(track_no), "count": 1},
        )

    def _handle_UnhookAll(self) -> None:
        count = len(self._hooked_tracks)
        self._hooked_tracks.clear()
        self._send_message("tracks", "unhooked_all", {"count": count})

    def _group_tracks(self, tc_no: int) -> dict[int, list[MA3TrackSnapshot]]:
        groups: dict[int, list[MA3TrackSnapshot]] = {}
        for track in self._tracks_by_coord.values():
            track_tc, tg_no, _track_no = parse_track_coord(track.coord)
            if track_tc != int(tc_no):
                continue
            groups.setdefault(tg_no, []).append(track)
        for tracks in groups.values():
            tracks.sort(key=lambda track: parse_track_coord(track.coord)[2])
        return groups

    def _next_event_id(self, coord: str) -> str:
        existing_ids = {event.event_id for event in self._events_by_coord.get(coord, [])}
        prefix = f"{coord}:evt"
        index = 1
        while True:
            candidate = f"{prefix}:{index}"
            if candidate not in existing_ids:
                return candidate
            index += 1

    def _send_message(self, message_type: str, change: str, payload: dict[str, object]) -> None:
        target = self._target
        if target is None:
            return
        client = SimpleUDPClient(target[0], target[1])
        client.send_message(
            self._message_path, encode_ma3_osc_payload(message_type, change, payload)
        )


@dataclass(slots=True)
class SimulatedMA3Bridge:
    connected: bool = False
    connect_calls: int = 0
    disconnect_calls: int = 0
    emitted_events: list[dict[str, Any]] = field(default_factory=list)
    _pending_events: deque[dict[str, Any]] = field(default_factory=deque)
    _server: _SimulatedMA3OSCServer | None = field(default=None, init=False, repr=False)
    _bridge: MA3OSCBridge | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        server = _SimulatedMA3OSCServer().start()
        bridge = MA3OSCBridge(
            listen_host="127.0.0.1",
            listen_port=0,
            timecode_no=1,
            command_transport=OSCCommandTransport(*server.endpoint),
        )
        self._server = server
        self._bridge = bridge

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    @property
    def commands(self) -> list[str]:
        server = self._require_server()
        return list(server.commands)

    def on_ma3_connected(self) -> None:
        self.connect_calls += 1
        self.connected = True
        self._require_bridge().on_ma3_connected()

    def on_ma3_disconnected(self) -> None:
        self.disconnect_calls += 1
        self.connected = False
        server = self._require_server()
        prior_command_count = len(server.commands)
        self._require_bridge().on_ma3_disconnected()
        deadline = monotonic() + 0.1
        while len(server.commands) == prior_command_count and monotonic() < deadline:
            sleep(0.005)

    def emit(self, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {"kind": kind, "payload": dict(payload or {})}
        self.emitted_events.append(event)
        return event

    def push_event(self, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {"kind": kind, "payload": dict(payload or {})}
        self._pending_events.append(event)
        return event

    def pop_event(self) -> dict[str, Any] | None:
        if not self._pending_events:
            return None
        return self._pending_events.popleft()

    def pending_events(self) -> list[dict[str, Any]]:
        return list(self._pending_events)

    def get_status(self) -> dict[str, Any]:
        return self._require_bridge().get_status()

    def list_tracks(self) -> list[MA3TrackSnapshot]:
        return self._require_bridge().list_tracks()

    def list_track_events(self, track_coord: str) -> list[MA3EventSnapshot]:
        return self._require_bridge().list_track_events(track_coord)

    def set_tracks(self, tracks) -> None:
        self._require_server().set_tracks(tracks)
        self._require_bridge().invalidate()

    def set_track_events(self, events_by_track) -> None:
        self._require_server().set_track_events(events_by_track)
        self._require_bridge().invalidate()

    def apply_push_transfer(
        self,
        *,
        target_track_coord: str,
        selected_events,
        transfer_mode: str = "merge",
    ) -> None:
        self._require_bridge().apply_push_transfer(
            target_track_coord=target_track_coord,
            selected_events=selected_events,
            transfer_mode=transfer_mode,
        )
        self.emit(
            "transfer.push_applied",
            {
                "target_track_coord": str(target_track_coord),
                "transfer_mode": str(transfer_mode or "merge"),
                "selected_count": len(list(selected_events or [])),
            },
        )

    def shutdown(self) -> None:
        bridge = self._bridge
        server = self._server
        self._bridge = None
        self._server = None
        if bridge is not None:
            bridge.shutdown()
        if server is not None:
            server.stop()

    def _require_bridge(self) -> MA3OSCBridge:
        if self._bridge is None:
            raise RuntimeError("SimulatedMA3Bridge is shut down")
        return self._bridge

    def _require_server(self) -> _SimulatedMA3OSCServer:
        if self._server is None:
            raise RuntimeError("SimulatedMA3Bridge is shut down")
        return self._server


def _parse_command(command: str) -> tuple[str, list[object]]:
    text = str(command or "").strip()
    if not text.startswith("EZ."):
        raise ValueError(f"Unsupported simulated MA3 command: {command!r}")
    open_index = text.find("(")
    close_index = text.rfind(")")
    if open_index < 0 or close_index < open_index:
        raise ValueError(f"Unsupported simulated MA3 command: {command!r}")
    name = text[3:open_index].strip()
    args_text = text[open_index + 1 : close_index].strip()
    if not args_text:
        return name, []
    return name, [_parse_command_arg(token) for token in _split_command_args(args_text)]


def _split_command_args(args_text: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    quote_char: str | None = None
    escaped = False
    for char in args_text:
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
        if char == ",":
            args.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        args.append("".join(current).strip())
    return args


def _parse_command_arg(token: str) -> object:
    text = str(token or "").strip()
    if not text:
        return ""
    if text.startswith('"') and text.endswith('"'):
        return json.loads(text)
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text

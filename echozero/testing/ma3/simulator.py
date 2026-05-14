"""ma3-simulator: In-process MA3 OSC simulator for sync tests and demos.
Exists to emulate MA3 command and state traffic for deterministic integration coverage.
Never treat this module as the production MA3 boundary or canonical proof surface.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass, field
from threading import Thread
from time import monotonic, sleep
from typing import Any

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from echozero.application.shared.cue_numbers import (
    CueNumber,
    cue_number_text,
    parse_positive_cue_number,
)
from echozero.infrastructure.osc import OscUdpSendTransport
from echozero.infrastructure.sync.ma3_adapter import (
    MA3EventSnapshot,
    MA3PresetSnapshot,
    MA3SequenceRangeSnapshot,
    MA3SequenceSnapshot,
    MA3TimecodeSnapshot,
    MA3TrackGroupSnapshot,
    MA3TrackSnapshot,
    coerce_event_snapshot,
    coerce_track_snapshot,
)
from echozero.infrastructure.sync.ma3_osc import (
    MA3OSCBridge,
    encode_ma3_osc_payload,
    format_track_coord,
    parse_track_coord,
)

_SEQUENCE_CHUNK_SIZE = 40
_TRACK_CHUNK_SIZE = 40
_CUE_COMMAND_RE = re.compile(r"(?i)\b(?:go\+|goto)\s+cue\s+(\d+(?:\.\d+)?)\b")
_PRESET_POOL_NAMES = {
    1: "Dimmer",
    2: "Position",
    4: "Color",
    5: "Beam",
    6: "Focus",
    21: "Phasers",
    22: "Optical",
}


def _default_recipe_line_rows() -> list[dict[str, object]]:
    return [
        {
            "seq_number": 12,
            "seq_name": "Song A",
            "actual_cue_number": 1.0,
            "part_number": "0.1",
            "feature_group": "Dimmer",
            "recipe_mode": "absolute",
            "matched_group": "Drums",
            "line_index": 1,
            "selection_key": "Drums:Dimmer",
            "source_cue_number": 1.0,
            "source_part_number": "0.1",
            "preset_ref": "Preset 21.221",
        },
        {
            "seq_number": 12,
            "seq_name": "Song A",
            "actual_cue_number": 1.0,
            "part_number": "0.2",
            "feature_group": "Dimmer",
            "recipe_mode": "absolute",
            "matched_group": "Drums",
            "line_index": 2,
            "selection_key": "Drums:Dimmer",
            "source_cue_number": 1.0,
            "source_part_number": "0.2",
            "preset_ref": "Preset 21.221",
        },
        {
            "seq_number": 12,
            "seq_name": "Song A",
            "actual_cue_number": 2.0,
            "part_number": "0.1",
            "feature_group": "Dimmer",
            "recipe_mode": "absolute",
            "matched_group": "Drums",
            "line_index": 1,
            "selection_key": "Drums:Dimmer",
            "source_cue_number": 2.0,
            "source_part_number": "0.1",
            "preset_ref": "Preset 21.222",
        },
        {
            "seq_number": 12,
            "seq_name": "Song A",
            "actual_cue_number": 2.0,
            "part_number": "0.2",
            "feature_group": "Color",
            "recipe_mode": "relative",
            "matched_group": "Drums",
            "line_index": 2,
            "selection_key": "Drums:Color",
            "source_cue_number": 2.0,
            "source_part_number": "0.2",
            "preset_ref": "Preset 4.44",
        },
        {
            "seq_number": 12,
            "seq_name": "Song A",
            "actual_cue_number": 2.0,
            "part_number": "0.3",
            "feature_group": "Beam",
            "recipe_mode": "absolute",
            "matched_group": "Drums",
            "line_index": 3,
            "selection_key": "Drums:Beam",
            "source_cue_number": 2.0,
            "source_part_number": "0.3",
            "preset_ref": "Preset 5.23",
        },
        {
            "seq_number": 12,
            "seq_name": "Song A",
            "actual_cue_number": 4.0,
            "part_number": "0.1",
            "feature_group": "Dimmer",
            "recipe_mode": "absolute",
            "matched_group": "Drums",
            "line_index": 1,
            "selection_key": "Drums:Dimmer",
            "source_cue_number": 4.0,
            "source_part_number": "0.1",
            "preset_ref": "Preset 21.333",
        },
    ]


def _cue_number_from_command(command: str) -> CueNumber | None:
    match = _CUE_COMMAND_RE.search(str(command or "").strip())
    if match is None:
        return None
    return parse_positive_cue_number(match.group(1))


def _event_label_from_command(command: str, cue_number: CueNumber | None) -> str:
    cue_number_label = cue_number_text(cue_number)
    if cue_number_label is not None:
        return f"Cue {cue_number_label}"
    text = str(command or "").strip()
    return text or "Event"


def _cue_ref_from_number(cue_number: CueNumber | None) -> str | None:
    cue_number_label = cue_number_text(cue_number)
    if cue_number_label is None:
        return None
    return str(cue_number_label)


def _default_tracks() -> list[MA3TrackSnapshot]:
    return [
        MA3TrackSnapshot(
            coord="tc1_tg2_tr3",
            name="Track 3",
            number=3,
            note="Bass",
            event_count=2,
            sequence_no=12,
        ),
        MA3TrackSnapshot(
            coord="tc1_tg2_tr4",
            name="Track 4",
            number=4,
            note="Lead",
            event_count=1,
            sequence_no=None,
        ),
    ]


def _default_events_by_track() -> dict[str, list[MA3EventSnapshot]]:
    return {
        "tc1_tg2_tr3": [
            MA3EventSnapshot(
                event_id="ma3_evt_1",
                label="Cue 1",
                start=1.0,
                end=1.5,
                cmd="Go+ Cue 1",
                cue_number=1,
                cue_ref="1",
            ),
            MA3EventSnapshot(
                event_id="ma3_evt_2",
                label="Cue 2",
                start=2.0,
                end=2.5,
                cmd="Go+ Cue 2",
                cue_number=2,
                cue_ref="2",
            ),
        ],
        "tc1_tg2_tr4": [
            MA3EventSnapshot(
                event_id="ma3_evt_9",
                label="Cue 9",
                start=9.0,
                end=9.5,
                cmd="Go+ Cue 9",
                cue_number=9,
                cue_ref="9",
            ),
        ],
    }


def _default_sequences() -> list[MA3SequenceSnapshot]:
    return [
        MA3SequenceSnapshot(number=12, name="Song A", cue_count=2),
        MA3SequenceSnapshot(number=15, name="Lead Stack", cue_count=1),
    ]


class _SimulatedMA3OSCServer:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        command_path: str = "/cmd",
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
        self._sequence_by_coord: dict[str, int | None] = {}
        self._sequences_by_number: dict[int, MA3SequenceSnapshot] = {}
        self._sequence_cues_by_sequence_no: dict[int, list[dict[str, object]]] = {}
        self._current_song_label: str | None = "Song A"
        self._timecode_name_by_no: dict[int, str] = {1: "Song A"}
        self._track_group_name_by_key: dict[tuple[int, int], str] = {(1, 2): "Group 2"}
        self._time_range_idx_by_coord: dict[str, int] = {}
        self._cmd_subtrack_ready_by_coord: dict[str, bool] = {}
        self._cmd_subtrack_create_blocked: set[str] = set()
        self._clear_delay_seconds_by_coord: dict[str, float] = {}
        self._presets_by_key: dict[tuple[int, int], MA3PresetSnapshot] = {}
        self._preset_details_by_key: dict[tuple[int, int], dict[str, object]] = {}
        self._recipe_line_rows: list[dict[str, object]] = _default_recipe_line_rows()
        self._drop_ping_reply_count = 0
        self._hooked_tracks: set[str] = set()
        self._hook_fail_coords: set[str] = set()
        self._ez_version = "2.0"
        self._ez_build = "2026-04-30.hitmaker-health-1"
        self._hitmaker_loaded = True
        self._hitmaker_version = "1.1.0"
        self._hitmaker_build = "2026-04-30.hitmaker-health-1"
        self._hitmaker_supports_event_type_create = True
        self._hitmaker_supports_go_hit = True
        self._hitmaker_supports_version_info = True

        self.set_tracks(_default_tracks())
        self.set_track_events(_default_events_by_track())
        self.set_sequences(_default_sequences())

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

    def list_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[MA3TrackSnapshot]:
        tracks = [
            MA3TrackSnapshot(
                coord=track.coord,
                name=track.name,
                number=track.number or parse_track_coord(track.coord)[2],
                timecode_name=self._timecode_name_by_no.get(parse_track_coord(track.coord)[0]),
                note=track.note,
                event_count=len(self._events_by_coord.get(track.coord, [])),
                sequence_no=self._sequence_by_coord.get(track.coord),
            )
            for track in sorted(self._tracks_by_coord.values(), key=lambda value: value.coord)
        ]
        if timecode_no is not None:
            tracks = [
                track for track in tracks if parse_track_coord(track.coord)[0] == int(timecode_no)
            ]
        if track_group_no is not None:
            tracks = [
                track
                for track in tracks
                if parse_track_coord(track.coord)[1] == int(track_group_no)
            ]
        return tracks

    def list_timecodes(self) -> list[MA3TimecodeSnapshot]:
        return [
            MA3TimecodeSnapshot(number=tc_no, name=name or None)
            for tc_no, name in sorted(self._timecode_name_by_no.items())
        ]

    def list_track_groups(self, *, timecode_no: int) -> list[MA3TrackGroupSnapshot]:
        return [
            MA3TrackGroupSnapshot(
                number=group_no,
                name=self._track_group_name_by_key.get(
                    (int(timecode_no), group_no), f"Group {group_no}"
                ),
                track_count=len(tracks),
            )
            for group_no, tracks in sorted(self._group_tracks(int(timecode_no)).items())
        ]

    def list_track_events(self, track_coord: str) -> list[MA3EventSnapshot]:
        return list(self._events_by_coord.get(str(track_coord or "").strip(), []))

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[MA3SequenceSnapshot]:
        sequences = sorted(self._sequences_by_number.values(), key=lambda item: item.number)
        if start_no is not None:
            sequences = [sequence for sequence in sequences if sequence.number >= int(start_no)]
        if end_no is not None:
            sequences = [sequence for sequence in sequences if sequence.number <= int(end_no)]
        return sequences

    def list_sequence_cues(self, *, sequence_no: int) -> list[dict[str, object]]:
        return list(self._sequence_cues_by_sequence_no.get(int(sequence_no), []))

    def get_current_song_sequence_range(self) -> MA3SequenceRangeSnapshot | None:
        return self._resolve_current_song_range()

    def set_tracks(self, tracks) -> None:
        prior_group_names = dict(self._track_group_name_by_key)
        self._tracks_by_coord = {}
        self._track_group_name_by_key = {}
        for raw_track in tracks or []:
            track = coerce_track_snapshot(raw_track)
            if not track.coord:
                continue
            self._tracks_by_coord[track.coord] = track
            tc_no, tg_no, _track_no = parse_track_coord(track.coord)
            if track.timecode_name:
                self._timecode_name_by_no[tc_no] = track.timecode_name
            else:
                self._timecode_name_by_no.setdefault(tc_no, f"Timecode {tc_no}")
            group_key = (tc_no, tg_no)
            self._track_group_name_by_key[group_key] = prior_group_names.get(
                group_key,
                f"Group {tg_no}",
            )
            self._sequence_by_coord[track.coord] = track.sequence_no
            self._time_range_idx_by_coord.setdefault(track.coord, 1)
            self._cmd_subtrack_ready_by_coord.setdefault(track.coord, True)

    def set_timecodes(self, timecodes) -> None:
        mapping: dict[int, str] = {}
        if isinstance(timecodes, dict):
            iterator = timecodes.items()
        else:
            iterator = timecodes or []
        for raw_key, raw_value in iterator:
            if isinstance(raw_key, dict):
                no = int(raw_key.get("no") or 0)
                name = str(raw_key.get("name") or "")
            else:
                no = int(raw_key or 0)
                name = str(raw_value or "")
            if no > 0:
                mapping[no] = name
        if mapping:
            self._timecode_name_by_no = mapping

    def set_track_events(self, events_by_track) -> None:
        self._events_by_coord = {}
        for coord, raw_events in dict(events_by_track or {}).items():
            normalized_coord = str(coord or "").strip()
            if not normalized_coord:
                continue
            normalized_events = [
                coerce_event_snapshot(raw_event) for raw_event in raw_events or []
            ]
            self._events_by_coord[normalized_coord] = normalized_events
            if normalized_events:
                self._cmd_subtrack_ready_by_coord[normalized_coord] = True

    def set_sequences(self, sequences) -> None:
        self._sequences_by_number = {}
        self._sequence_cues_by_sequence_no = {}
        for raw_sequence in sequences or []:
            if isinstance(raw_sequence, MA3SequenceSnapshot):
                sequence = raw_sequence
            else:
                number = getattr(raw_sequence, "number", None)
                if number in {None, ""} and isinstance(raw_sequence, dict):
                    number = raw_sequence.get("number", raw_sequence.get("no"))
                cue_count = getattr(raw_sequence, "cue_count", None)
                if cue_count in {None, ""} and isinstance(raw_sequence, dict):
                    cue_count = raw_sequence.get("cue_count")
                name = getattr(raw_sequence, "name", None)
                if name in {None, ""} and isinstance(raw_sequence, dict):
                    name = raw_sequence.get("name")
                sequence = MA3SequenceSnapshot(
                    number=int(number or 0),
                    name=str(name or ""),
                    cue_count=None if cue_count in {None, ""} else int(cue_count),
                )
            if sequence.number > 0:
                self._sequences_by_number[sequence.number] = sequence
                cue_count = int(sequence.cue_count or 0)
                self._sequence_cues_by_sequence_no[sequence.number] = [
                    {"no": index, "name": f"Cue {index}"} for index in range(1, cue_count + 1)
                ]

    def set_current_song_label(self, song_label: str | None) -> None:
        self._current_song_label = None if song_label in {None, ""} else str(song_label)
        if self._current_song_label:
            self._timecode_name_by_no[1] = self._current_song_label

    def set_plugin_health(
        self,
        *,
        ez_version: str | None = None,
        ez_build: str | None = None,
        hitmaker_loaded: bool | None = None,
        hitmaker_version: str | None = None,
        hitmaker_build: str | None = None,
        hitmaker_supports_event_type_create: bool | None = None,
        hitmaker_supports_go_hit: bool | None = None,
        hitmaker_supports_version_info: bool | None = None,
    ) -> None:
        if ez_version is not None:
            self._ez_version = str(ez_version)
        if ez_build is not None:
            self._ez_build = str(ez_build)
        if hitmaker_loaded is not None:
            self._hitmaker_loaded = bool(hitmaker_loaded)
        if hitmaker_version is not None:
            self._hitmaker_version = str(hitmaker_version)
        if hitmaker_build is not None:
            self._hitmaker_build = str(hitmaker_build)
        if hitmaker_supports_event_type_create is not None:
            self._hitmaker_supports_event_type_create = bool(hitmaker_supports_event_type_create)
        if hitmaker_supports_go_hit is not None:
            self._hitmaker_supports_go_hit = bool(hitmaker_supports_go_hit)
        if hitmaker_supports_version_info is not None:
            self._hitmaker_supports_version_info = bool(hitmaker_supports_version_info)

    def set_track_write_ready(self, track_coord: str, *, ready: bool) -> None:
        coord = str(track_coord or "").strip()
        if not coord:
            return
        self._cmd_subtrack_ready_by_coord[coord] = bool(ready)

    def set_cmd_subtrack_create_blocked(self, track_coord: str, *, blocked: bool) -> None:
        coord = str(track_coord or "").strip()
        if not coord:
            return
        if blocked:
            self._cmd_subtrack_create_blocked.add(coord)
        else:
            self._cmd_subtrack_create_blocked.discard(coord)

    def set_clear_delay(self, track_coord: str, *, seconds: float) -> None:
        coord = str(track_coord or "").strip()
        if not coord:
            return
        delay_seconds = max(0.0, float(seconds))
        if delay_seconds <= 0.0:
            self._clear_delay_seconds_by_coord.pop(coord, None)
            return
        self._clear_delay_seconds_by_coord[coord] = delay_seconds

    def set_drop_ping_reply_count(self, count: int) -> None:
        self._drop_ping_reply_count = max(0, int(count))

    def set_hook_failure(self, track_coord: str, *, should_fail: bool = True) -> None:
        coord = str(track_coord or "").strip()
        if not coord:
            return
        if should_fail:
            self._hook_fail_coords.add(coord)
        else:
            self._hook_fail_coords.discard(coord)

    def _handle_command(self, _address: str, *args: object) -> None:
        command = ""
        for arg in args:
            if isinstance(arg, str):
                command = arg
                break
        if not command:
            return
        normalized = _unwrap_lua_command(command)
        self.commands.append(normalized)
        self._execute(normalized)

    def _execute(self, command: str) -> None:
        if self._handle_raw_label_command(command):
            return
        name, args = _parse_command(command)
        handler = getattr(self, f"_handle_{name}", None)
        if callable(handler):
            handler(*args)

    def _handle_raw_label_command(self, command: str) -> bool:
        match = re.fullmatch(
            r"(?i)\s*label\s+sequence\s+(\d+)\s+cue\s+"
            r"(\d+(?:\.\d+)?)\s+\"((?:\\.|[^\"])*)\"\s*",
            str(command or ""),
        )
        if match is None:
            return False
        sequence_no = int(match.group(1))
        cue_number = parse_positive_cue_number(match.group(2))
        if cue_number is None:
            return True
        cue_name = match.group(3).replace('\\"', '"').replace("\\\\", "\\")
        cues = self._sequence_cues_by_sequence_no.setdefault(sequence_no, [])
        for cue in cues:
            if parse_positive_cue_number(cue.get("no")) == cue_number:
                cue["name"] = cue_name
                break
        else:
            cues.append({"no": cue_number, "name": cue_name})
            cues.sort(key=lambda cue: float(parse_positive_cue_number(cue.get("no")) or 0))
        return True

    def _handle_SetTarget(self, host: str, port: int) -> None:
        self._target = str(host), int(port)

    def _handle_Ping(self) -> None:
        if self._drop_ping_reply_count > 0:
            self._drop_ping_reply_count -= 1
            return
        self._send_message("connection", "ping", {"status": "ok"})

    def _handle_Version(self) -> None:
        self._send_message(
            "plugin",
            "version",
            {
                "ez_version": self._ez_version,
                "ez_build": self._ez_build,
                "text": f"[EZ] Version: {self._ez_version} (build {self._ez_build})",
            },
        )

    def _handle_GetPluginHealth(self) -> None:
        self._send_message(
            "plugin",
            "health",
            {
                "ez_version": self._ez_version,
                "ez_build": self._ez_build,
                "hitmaker_loaded": self._hitmaker_loaded,
                "hitmaker_version": self._hitmaker_version,
                "hitmaker_build": self._hitmaker_build,
                "hitmaker_supports_event_type_create": self._hitmaker_supports_event_type_create,
                "hitmaker_supports_go_hit": self._hitmaker_supports_go_hit,
                "hitmaker_supports_version_info": self._hitmaker_supports_version_info,
            },
        )

    def _handle_ConnectionReport(self, request_id: int | None = None) -> None:
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": "ok",
            "ez_version": self._ez_version,
            "ez_build": self._ez_build,
            "hitmaker_loaded": self._hitmaker_loaded,
            "hitmaker_version": self._hitmaker_version,
            "hitmaker_build": self._hitmaker_build,
            "target_ip": self._target[0] if self._target is not None else "",
            "target_port": self._target[1] if self._target is not None else 0,
            "socket_ok": True,
            "osc_module_loaded": True,
            "hooks": 0,
            "hook_keys": [],
            "capabilities": {
                "ping": True,
                "status": True,
                "version": True,
                "plugin_health": True,
                "connection_report": True,
                "hook_track": True,
            },
            "send": {
                "send_sequence": len(self.commands),
                "send_ok_count": len(self.commands),
                "send_fail_count": 0,
                "last_send_error": "",
            },
        }
        if request_id is not None:
            payload["request_id"] = int(request_id)
        self._send_message("connection", "report", payload)

    def _handle_GetDataPoolObjects(
        self, path: str | None = None, request_id: int | None = None
    ) -> None:
        normalized_path = self._normalize_datapool_path(path)
        children = self._datapool_children_for_path(normalized_path)
        if children is None:
            payload = {"path": normalized_path, "error": "path_not_found"}
            if request_id is not None:
                payload["request_id"] = int(request_id)
            self._send_message("datapool", "error", payload)
            return
        payload = {
            "path": normalized_path,
            "count": len(children),
            "total_children": len(children),
            "truncated": False,
            "children": children,
        }
        if request_id is not None:
            payload["request_id"] = int(request_id)
        self._send_message("datapool", "children", payload)

    def _handle_DescribeDataPoolObject(
        self, path: str | None = None, request_id: int | None = None
    ) -> None:
        normalized_path = self._normalize_datapool_path(path)
        entry = self._datapool_object_for_path(normalized_path)
        if entry is None:
            payload = {"path": normalized_path, "error": "path_not_found"}
            if request_id is not None:
                payload["request_id"] = int(request_id)
            self._send_message("datapool", "error", payload)
            return
        payload = {"path": normalized_path, "object": entry}
        if request_id is not None:
            payload["request_id"] = int(request_id)
        self._send_message("datapool", "object", payload)

    def _handle_GetTimecodes(self) -> None:
        timecodes = [
            {"no": tc_no, "name": name}
            for tc_no, name in sorted(self._timecode_name_by_no.items())
        ]
        self._send_message(
            "timecodes",
            "list",
            {"count": len(timecodes), "timecodes": timecodes},
        )

    def _handle_GetTrackGroups(self, tc_no: int) -> None:
        groups = []
        for group_no, tracks in self._group_tracks(tc_no).items():
            groups.append(
                {
                    "no": group_no,
                    "name": self._track_group_name_by_key.get(
                        (int(tc_no), int(group_no)),
                        f"Group {group_no}",
                    ),
                    "track_count": len(tracks),
                }
            )
        groups.sort(key=lambda group: int(group["no"]))
        self._send_message(
            "trackgroups",
            "list",
            {"tc": int(tc_no), "count": len(groups), "trackgroups": groups},
        )

    def _handle_CreateTimecode(self, preferred_name: str | None = None) -> None:
        next_timecode_no = max(self._timecode_name_by_no, default=0) + 1
        name = str(preferred_name or "").strip() or f"Timecode {next_timecode_no}"
        self._timecode_name_by_no[next_timecode_no] = name
        self._send_message(
            "timecode",
            "created",
            {
                "no": int(next_timecode_no),
                "name": name,
            },
        )

    def _handle_CreateTrackGroup(self, tc_no: int, preferred_name: str | None = None) -> None:
        requested_timecode_no = int(tc_no)
        if requested_timecode_no not in self._timecode_name_by_no:
            self._send_message(
                "trackgroup",
                "error",
                {"tc": requested_timecode_no, "error": "Timecode does not exist"},
            )
            return

        desired_name = str(preferred_name or "").strip()
        groups = self._group_tracks(requested_timecode_no)
        for group_no in sorted(groups):
            existing_name = self._track_group_name_by_key.get(
                (requested_timecode_no, int(group_no)),
                f"Group {int(group_no)}",
            )
            if desired_name and existing_name.lower() == desired_name.lower():
                self._send_message(
                    "trackgroup",
                    "exists",
                    {"tc": requested_timecode_no, "tg": int(group_no), "name": existing_name},
                )
                return

        next_group_no = 1
        while next_group_no in groups:
            next_group_no += 1
        group_name = desired_name or f"Group {next_group_no}"
        self._track_group_name_by_key[(requested_timecode_no, next_group_no)] = group_name
        self._send_message(
            "trackgroup",
            "created",
            {"tc": requested_timecode_no, "tg": next_group_no, "name": group_name},
        )

    def _handle_CreateTrack(self, tc_no: int, tg_no: int, track_name: str | None = None) -> None:
        requested_timecode_no = int(tc_no)
        requested_track_group_no = int(tg_no)
        desired_name = str(track_name or "").strip()
        if not desired_name:
            self._send_track_error(
                requested_timecode_no,
                requested_track_group_no,
                0,
                "Track name required",
            )
            return
        if requested_timecode_no not in self._timecode_name_by_no:
            self._send_track_error(
                requested_timecode_no,
                requested_track_group_no,
                0,
                "Timecode does not exist",
            )
            return

        groups = self._group_tracks(requested_timecode_no)
        if requested_track_group_no not in groups:
            self._send_track_error(
                requested_timecode_no,
                requested_track_group_no,
                0,
                "Track group does not exist",
            )
            return

        group_tracks = groups[requested_track_group_no]
        for track in group_tracks:
            if str(track.name or "").strip().lower() == desired_name.lower():
                _track_tc_no, _track_group_no, track_no = parse_track_coord(track.coord)
                self._send_message(
                    "track",
                    "exists",
                    {
                        "tc": requested_timecode_no,
                        "tg": requested_track_group_no,
                        "track": int(track_no),
                        "name": str(track.name or desired_name),
                    },
                )
                return

        next_track_no = (
            max(parse_track_coord(track.coord)[2] for track in group_tracks) + 1
            if group_tracks
            else 1
        )
        coord = format_track_coord(requested_timecode_no, requested_track_group_no, next_track_no)
        timecode_name = self._timecode_name_by_no.get(requested_timecode_no)
        self._track_group_name_by_key.setdefault(
            (requested_timecode_no, requested_track_group_no),
            f"Group {requested_track_group_no}",
        )
        self._tracks_by_coord[coord] = MA3TrackSnapshot(
            coord=coord,
            name=desired_name,
            number=next_track_no,
            timecode_name=timecode_name,
            note="",
            event_count=0,
            sequence_no=None,
        )
        self._events_by_coord.setdefault(coord, [])
        self._sequence_by_coord.setdefault(coord, None)
        self._time_range_idx_by_coord.setdefault(coord, 1)
        self._cmd_subtrack_ready_by_coord.setdefault(coord, False)
        self._send_message(
            "track",
            "created",
            {
                "tc": requested_timecode_no,
                "tg": requested_track_group_no,
                "track": next_track_no,
                "name": desired_name,
            },
        )

    def _handle_CreateStaticPreset(
        self,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        value_command: str,
    ) -> None:
        requested_preset_type_no = int(preset_type_no)
        requested_preset_no = int(preset_no)
        normalized_mode = str(store_mode or "").strip()
        normalized_name = str(preset_name or "").strip()
        normalized_selection = str(selection_command or "").strip()
        normalized_value = str(value_command or "").strip()
        if requested_preset_type_no < 1 or requested_preset_no < 1:
            self._send_preset_error(
                requested_preset_type_no,
                requested_preset_no,
                "Preset type and number are required",
            )
            return
        if not normalized_name:
            self._send_preset_error(
                requested_preset_type_no, requested_preset_no, "Preset name required"
            )
            return
        if not normalized_selection:
            self._send_preset_error(
                requested_preset_type_no,
                requested_preset_no,
                "Selection command required",
            )
            return
        if not normalized_value:
            self._send_preset_error(
                requested_preset_type_no,
                requested_preset_no,
                "Value command required",
            )
            return

        key = (requested_preset_type_no, requested_preset_no)
        existing = self._presets_by_key.get(key)
        if existing is not None:
            self._send_preset_snapshot("exists", existing)
            return

        created = MA3PresetSnapshot(
            preset_type=requested_preset_type_no,
            number=requested_preset_no,
            name=normalized_name,
            store_mode=normalized_mode,
            kind="static",
            step_count=1,
        )
        self._presets_by_key[key] = created
        self._preset_details_by_key[key] = {
            "selection_command": normalized_selection,
            "value_command": normalized_value,
        }
        self._send_preset_snapshot("created", created)

    def _handle_CreatePhaserPreset(
        self,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        step_spec: str,
        speed_bpm: float | None = None,
    ) -> None:
        requested_preset_type_no = int(preset_type_no)
        requested_preset_no = int(preset_no)
        normalized_mode = str(store_mode or "").strip()
        normalized_name = str(preset_name or "").strip()
        normalized_selection = str(selection_command or "").strip()
        normalized_step_spec = str(step_spec or "").strip()
        if requested_preset_type_no < 1 or requested_preset_no < 1:
            self._send_preset_error(
                requested_preset_type_no,
                requested_preset_no,
                "Preset type and number are required",
            )
            return
        if not normalized_name:
            self._send_preset_error(
                requested_preset_type_no, requested_preset_no, "Preset name required"
            )
            return
        if not normalized_selection:
            self._send_preset_error(
                requested_preset_type_no,
                requested_preset_no,
                "Selection command required",
            )
            return
        step_groups = [group for group in normalized_step_spec.split(";") if group.strip()]
        if len(step_groups) < 2:
            self._send_preset_error(
                requested_preset_type_no,
                requested_preset_no,
                "Phaser presets require at least two steps",
            )
            return

        key = (requested_preset_type_no, requested_preset_no)
        existing = self._presets_by_key.get(key)
        if existing is not None:
            self._send_preset_snapshot("exists", existing)
            return

        created = MA3PresetSnapshot(
            preset_type=requested_preset_type_no,
            number=requested_preset_no,
            name=normalized_name,
            store_mode=normalized_mode,
            kind="phaser",
            step_count=len(step_groups),
        )
        self._presets_by_key[key] = created
        self._preset_details_by_key[key] = {
            "selection_command": normalized_selection,
            "step_spec": normalized_step_spec,
            "speed_bpm": None if speed_bpm is None else float(speed_bpm),
        }
        self._send_preset_snapshot("created", created)

    def _handle_CreateRecipePreset(
        self,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        source_preset_ref: str,
        selection_mode: str = "Strict",
    ) -> None:
        requested_preset_type_no = int(preset_type_no)
        requested_preset_no = int(preset_no)
        normalized_mode = str(store_mode or "").strip()
        normalized_name = str(preset_name or "").strip()
        normalized_selection = str(selection_command or "").strip()
        normalized_source_ref = str(source_preset_ref or "").strip()
        normalized_selection_mode = str(selection_mode or "").strip()
        if requested_preset_type_no < 1 or requested_preset_no < 1:
            self._send_preset_error(
                requested_preset_type_no,
                requested_preset_no,
                "Preset type and number are required",
            )
            return
        if not normalized_name:
            self._send_preset_error(
                requested_preset_type_no, requested_preset_no, "Preset name required"
            )
            return
        if not normalized_selection:
            self._send_preset_error(
                requested_preset_type_no,
                requested_preset_no,
                "Selection command required",
            )
            return
        if not normalized_source_ref:
            self._send_preset_error(
                requested_preset_type_no,
                requested_preset_no,
                "Source preset ref required",
            )
            return
        if not normalized_selection_mode:
            self._send_preset_error(
                requested_preset_type_no,
                requested_preset_no,
                "Selection mode required",
            )
            return

        key = (requested_preset_type_no, requested_preset_no)
        existing = self._presets_by_key.get(key)
        if existing is not None:
            self._send_preset_snapshot("exists", existing)
            return

        created = MA3PresetSnapshot(
            preset_type=requested_preset_type_no,
            number=requested_preset_no,
            name=normalized_name,
            store_mode=normalized_mode,
            kind="recipe",
            step_count=1,
        )
        self._presets_by_key[key] = created
        self._preset_details_by_key[key] = {
            "selection_command": normalized_selection,
            "source_preset_ref": normalized_source_ref,
            "selection_mode": normalized_selection_mode,
        }
        self._send_preset_snapshot("created", created)

    def _handle_EditStaticPreset(
        self,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        value_command: str,
    ) -> None:
        self._presets_by_key.pop((int(preset_type_no), int(preset_no)), None)
        self._preset_details_by_key.pop((int(preset_type_no), int(preset_no)), None)
        self._handle_CreateStaticPreset(
            preset_type_no,
            preset_no,
            store_mode,
            preset_name,
            selection_command,
            value_command,
        )
        key = (int(preset_type_no), int(preset_no))
        updated = self._presets_by_key.get(key)
        if updated is not None:
            self._send_preset_snapshot("updated", updated)

    def _handle_EditPhaserPreset(
        self,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        step_spec: str,
        speed_bpm: float | None = None,
    ) -> None:
        self._presets_by_key.pop((int(preset_type_no), int(preset_no)), None)
        self._preset_details_by_key.pop((int(preset_type_no), int(preset_no)), None)
        self._handle_CreatePhaserPreset(
            preset_type_no,
            preset_no,
            store_mode,
            preset_name,
            selection_command,
            step_spec,
            speed_bpm,
        )
        key = (int(preset_type_no), int(preset_no))
        updated = self._presets_by_key.get(key)
        if updated is not None:
            self._send_preset_snapshot("updated", updated)

    def _handle_EditRecipePreset(
        self,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        source_preset_ref: str,
        selection_mode: str = "Strict",
    ) -> None:
        self._presets_by_key.pop((int(preset_type_no), int(preset_no)), None)
        self._preset_details_by_key.pop((int(preset_type_no), int(preset_no)), None)
        self._handle_CreateRecipePreset(
            preset_type_no,
            preset_no,
            store_mode,
            preset_name,
            selection_command,
            source_preset_ref,
            selection_mode,
        )
        key = (int(preset_type_no), int(preset_no))
        updated = self._presets_by_key.get(key)
        if updated is not None:
            self._send_preset_snapshot("updated", updated)

    def _handle_ListPresets(self, preset_type_no: int, request_id: int | None = None) -> None:
        requested_preset_type_no = int(preset_type_no)
        pool = [
            {
                "preset_type": snapshot.preset_type,
                "number": snapshot.number,
                "name": snapshot.name,
                "store_mode": snapshot.store_mode,
                "kind": snapshot.kind,
                "step_count": snapshot.step_count,
                "path": f"PresetPools/{snapshot.preset_type}/{snapshot.number}",
            }
            for _preset_no, snapshot in self._preset_snapshots_for_type(requested_preset_type_no)
        ]
        self._send_message(
            "presets",
            "list",
            {
                "preset_type": requested_preset_type_no,
                "presets": pool,
                "request_id": None if request_id is None else int(request_id),
            },
        )

    def _handle_DescribePreset(
        self,
        preset_type_no: int,
        preset_no: int,
        request_id: int | None = None,
    ) -> None:
        requested_preset_type_no = int(preset_type_no)
        requested_preset_no = int(preset_no)
        payload = self._datapool_object_for_path(
            f"PresetPools/{requested_preset_type_no}/{requested_preset_no}"
        )
        if payload is None:
            self._send_preset_error(
                requested_preset_type_no, requested_preset_no, "Preset not found"
            )
            return
        self._send_message(
            "preset",
            "described",
            {
                "preset_type": requested_preset_type_no,
                "number": requested_preset_no,
                "object": payload,
                "request_id": None if request_id is None else int(request_id),
            },
        )

    def _handle_PreviewReplacePresetWhenGroup(
        self,
        preset_type_no: int,
        source_preset_ref: str,
        dest_preset_ref: str,
        group_filter_csv: str,
        sequence_numbers_csv: str,
        request_id: int | None = None,
    ) -> None:
        findings = self._preset_replace_findings(
            source_preset_ref=str(source_preset_ref),
            dest_preset_ref=str(dest_preset_ref),
            group_filter_csv=str(group_filter_csv),
            sequence_numbers_csv=str(sequence_numbers_csv),
        )
        self._send_message(
            "preset_replace",
            "preview",
            {
                "preset_type": int(preset_type_no),
                "source_preset_ref": str(source_preset_ref),
                "dest_preset_ref": str(dest_preset_ref),
                "count": len(findings),
                "findings": findings,
                "request_id": None if request_id is None else int(request_id),
            },
        )

    def _handle_ReplacePresetWhenGroup(
        self,
        preset_type_no: int,
        source_preset_ref: str,
        dest_preset_ref: str,
        group_filter_csv: str,
        sequence_numbers_csv: str,
        request_id: int | None = None,
    ) -> None:
        findings = self._preset_replace_findings(
            source_preset_ref=str(source_preset_ref),
            dest_preset_ref=str(dest_preset_ref),
            group_filter_csv=str(group_filter_csv),
            sequence_numbers_csv=str(sequence_numbers_csv),
        )
        for finding in findings:
            for row in self._recipe_line_rows:
                if (
                    int(row.get("seq_number") or 0) == int(finding.get("seqNumber") or 0)
                    and str(row.get("part_number") or "") == str(finding.get("partNumber") or "")
                    and float(row.get("actual_cue_number") or 0.0)
                    == float(finding.get("actualCueNumber") or 0.0)
                ):
                    row["preset_ref"] = str(dest_preset_ref)
        self._send_message(
            "preset_replace",
            "applied",
            {
                "preset_type": int(preset_type_no),
                "source_preset_ref": str(source_preset_ref),
                "dest_preset_ref": str(dest_preset_ref),
                "count": len(findings),
                "replaced_count": len(findings),
                "findings": findings,
                "request_id": None if request_id is None else int(request_id),
            },
        )

    def _handle_AnalyzeCueRecipeState(
        self,
        sequence_no: int,
        cue_no: str,
        request_id: int | None = None,
    ) -> None:
        payload = self._analyze_cue_recipe_state(
            sequence_no=int(sequence_no),
            cue_number=float(cue_no),
        )
        payload["request_id"] = None if request_id is None else int(request_id)
        self._send_message("recipe_cue", "analysis", payload)

    def _handle_PreviewRecipeCueOnly(
        self,
        sequence_no: int,
        source_cue_no: str,
        target_cue_no: str,
        request_id: int | None = None,
    ) -> None:
        payload = self._preview_recipe_cue_only(
            sequence_no=int(sequence_no),
            source_cue_no=float(source_cue_no),
            target_cue_no=float(target_cue_no),
        )
        payload["request_id"] = None if request_id is None else int(request_id)
        self._send_message("recipe_cue", "cue_only_preview", payload)

    def _handle_ApplyRecipeCueOnly(
        self,
        sequence_no: int,
        source_cue_no: str,
        target_cue_no: str,
        request_id: int | None = None,
    ) -> None:
        payload = self._apply_recipe_cue_only(
            sequence_no=int(sequence_no),
            source_cue_no=float(source_cue_no),
            target_cue_no=float(target_cue_no),
        )
        payload["request_id"] = None if request_id is None else int(request_id)
        self._send_message("recipe_cue", "cue_only_applied", payload)

    def _handle_CopyCueWithStatus(
        self,
        sequence_no: int,
        source_cue_no: str,
        dest_cue_no: str,
        request_id: int | None = None,
    ) -> None:
        payload = self._copy_cue_with_status(
            sequence_no=int(sequence_no),
            source_cue_no=float(source_cue_no),
            dest_cue_no=float(dest_cue_no),
        )
        payload["request_id"] = None if request_id is None else int(request_id)
        self._send_message("recipe_cue", "copied_with_status", payload)

    def _handle_PreviewCopyCueWithStatus(
        self,
        sequence_no: int,
        source_cue_no: str,
        dest_cue_no: str,
        request_id: int | None = None,
    ) -> None:
        payload = self._preview_copy_cue_with_status(
            sequence_no=int(sequence_no),
            source_cue_no=float(source_cue_no),
            dest_cue_no=float(dest_cue_no),
        )
        payload["request_id"] = None if request_id is None else int(request_id)
        self._send_message("recipe_cue", "copy_with_status_preview", payload)

    def _handle_GetTracks(self, tc_no: int, tg_no: int, request_id: int | None = None) -> None:
        tracks = []
        for track in self._group_tracks(int(tc_no)).get(int(tg_no), []):
            _tc, _tg, track_no = parse_track_coord(track.coord)
            tracks.append(
                {
                    "no": track_no,
                    "name": track.name,
                    "event_count": len(self._events_by_coord.get(track.coord, [])),
                    "sequence_no": self._sequence_by_coord.get(track.coord),
                    "note": track.note or "",
                }
            )
        total = len(tracks)
        if total > _TRACK_CHUNK_SIZE:
            total_chunks = math.ceil(total / _TRACK_CHUNK_SIZE)
            for chunk_index in range(total_chunks):
                start_index = chunk_index * _TRACK_CHUNK_SIZE
                end_index = start_index + _TRACK_CHUNK_SIZE
                payload = {
                    "tc": int(tc_no),
                    "tg": int(tg_no),
                    "count": total,
                    "offset": start_index + 1,
                    "chunk_index": chunk_index + 1,
                    "total_chunks": total_chunks,
                    "tracks": tracks[start_index:end_index],
                }
                if request_id is not None:
                    payload["request_id"] = int(request_id)
                self._send_message("tracks", "list", payload)
            return

        payload = {
            "tc": int(tc_no),
            "tg": int(tg_no),
            "count": total,
            "tracks": tracks,
        }
        if request_id is not None:
            payload["request_id"] = int(request_id)
        self._send_message("tracks", "list", payload)

    def _handle_GetSequences(
        self,
        start_no: int | None = None,
        end_no: int | None = None,
        request_id: int | None = None,
    ) -> None:
        sequences = [
            {
                "no": sequence.number,
                "name": sequence.name,
                "cue_count": sequence.cue_count,
            }
            for sequence in self.list_sequences(start_no=start_no, end_no=end_no)
        ]
        total = len(sequences)
        if total > _SEQUENCE_CHUNK_SIZE:
            total_chunks = math.ceil(total / _SEQUENCE_CHUNK_SIZE)
            for chunk_index in range(total_chunks):
                start_index = chunk_index * _SEQUENCE_CHUNK_SIZE
                end_index = start_index + _SEQUENCE_CHUNK_SIZE
                payload = {
                    "count": total,
                    "offset": start_index + 1,
                    "chunk_index": chunk_index + 1,
                    "total_chunks": total_chunks,
                    "sequences": sequences[start_index:end_index],
                }
                if request_id is not None:
                    payload["request_id"] = int(request_id)
                self._send_message("sequences", "list", payload)
            return

        payload = {"count": total, "sequences": sequences}
        if request_id is not None:
            payload["request_id"] = int(request_id)
        self._send_message("sequences", "list", payload)

    def _handle_GetSequenceCues(self, sequence_no: int, request_id: int | None = None) -> None:
        requested_sequence_no = int(sequence_no)
        sequence = self._sequences_by_number.get(requested_sequence_no)
        if sequence is None:
            payload = {"sequence_no": requested_sequence_no, "error": "sequence_not_found"}
            if request_id is not None:
                payload["request_id"] = int(request_id)
            self._send_message("sequence_cues", "error", payload)
            return
        cues = list(self._sequence_cues_by_sequence_no.get(requested_sequence_no, []))
        payload = {
            "sequence_no": requested_sequence_no,
            "count": len(cues),
            "chunk_index": 1,
            "total_chunks": 1,
            "cues": cues,
        }
        if request_id is not None:
            payload["request_id"] = int(request_id)
        self._send_message("sequence_cues", "list", payload)

    def _handle_GetCurrentSongSequenceRange(self) -> None:
        sequence_range = self._resolve_current_song_range()
        if sequence_range is None:
            return
        self._send_message(
            "sequence_range",
            "current_song",
            {
                "song_label": sequence_range.song_label,
                "start": sequence_range.start,
                "end": sequence_range.end,
            },
        )

    def _handle_GetEvents(
        self, tc_no: int, tg_no: int, track_no: int, request_id: int | None = None
    ) -> None:
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        events = []
        for index, event in enumerate(self._events_by_coord.get(coord, []), start=1):
            cue_ref = str(event.cue_ref or "").strip() or _cue_ref_from_number(event.cue_number)
            events.append(
                {
                    "event_id": event.event_id,
                    "idx": index,
                    "time": None if event.start is None else float(event.start),
                    "start": None if event.start is None else float(event.start),
                    "end": None if event.end is None else float(event.end),
                    "name": event.label,
                    "cmd": event.cmd or event.label,
                    "cue_number": event.cue_number,
                    "cue_ref": cue_ref,
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
        delay_seconds = max(0.0, self._clear_delay_seconds_by_coord.get(coord, 0.0))
        if delay_seconds > 0.0:
            Thread(
                target=self._complete_delayed_clear,
                args=(coord, int(tc_no), int(tg_no), int(track_no), delay_seconds),
                daemon=True,
                name="echozero-simulated-ma3-clear",
            ).start()
            return
        self._clear_track(coord, int(tc_no), int(tg_no), int(track_no))

    def _handle_AddEvent(
        self,
        tc_no: int,
        tg_no: int,
        track_no: int,
        start: float,
        cmd: str,
        event_name: str | None = None,
        cue_no: CueNumber | None = None,
        cue_label: str | None = None,
        channel_no: int | None = None,
    ) -> None:
        del channel_no
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        if not self._cmd_subtrack_ready_by_coord.get(coord, True):
            self._send_message(
                "event",
                "error",
                {
                    "tc": int(tc_no),
                    "tg": int(tg_no),
                    "track": int(track_no),
                    "error": "No CmdSubTrack - Attempting to Aquire() CmdSubTrack",
                },
            )
            return
        events = self._events_by_coord.setdefault(coord, [])
        next_id = self._next_event_id(coord)
        command = str(cmd or "")
        resolved_cue_number = parse_positive_cue_number(cue_no)
        cue_number = (
            resolved_cue_number
            if resolved_cue_number is not None
            else _cue_number_from_command(command)
        )
        explicit_event_name = str(event_name or "").strip()
        explicit_cue_label = str(cue_label or "").strip()
        cue_ref = _cue_ref_from_number(cue_number)
        label = _event_label_from_command(command, cue_number)
        if (
            explicit_event_name
            and explicit_cue_label
            and explicit_event_name != explicit_cue_label
        ):
            suffix_index = explicit_event_name.rfind(explicit_cue_label)
            if suffix_index >= 0:
                inferred_cue_ref = explicit_event_name[:suffix_index].strip(" :-")
                cue_ref = inferred_cue_ref or cue_ref
                label = explicit_cue_label
            else:
                label = explicit_event_name
        elif explicit_event_name:
            label = explicit_event_name
        elif explicit_cue_label:
            label = explicit_cue_label
        snapshot = MA3EventSnapshot(
            event_id=next_id,
            label=label,
            start=float(start),
            end=float(start),
            cmd=command,
            cue_number=cue_number,
            cue_ref=cue_ref,
        )
        events.append(snapshot)
        events.sort(
            key=lambda event: (
                float(event.start or 0.0),
                float(event.end or event.start or 0.0),
                event.label,
            )
        )

    def _complete_delayed_clear(
        self,
        coord: str,
        tc_no: int,
        tg_no: int,
        track_no: int,
        delay_seconds: float,
    ) -> None:
        sleep(delay_seconds)
        self._clear_track(coord, tc_no, tg_no, track_no)

    def _clear_track(self, coord: str, tc_no: int, tg_no: int, track_no: int) -> None:
        self._events_by_coord[coord] = []
        self._send_message(
            "track",
            "cleared",
            {"tc": int(tc_no), "tg": int(tg_no), "track": int(track_no), "count": 0},
        )

    def _handle_AssignTrackSequence(
        self,
        tc_no: int,
        tg_no: int,
        track_no: int,
        sequence_no: int,
    ) -> None:
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        if coord not in self._tracks_by_coord:
            self._send_track_error(tc_no, tg_no, track_no, "Track does not exist")
            return
        if int(sequence_no) not in self._sequences_by_number:
            self._send_track_error(
                tc_no, tg_no, track_no, f"Sequence {int(sequence_no)} does not exist"
            )
            return
        self._sequence_by_coord[coord] = int(sequence_no)
        self._send_message(
            "track",
            "assigned",
            {
                "tc": int(tc_no),
                "tg": int(tg_no),
                "track": int(track_no),
                "seq": int(sequence_no),
            },
        )

    def _handle_CreateSequenceNextAvailable(self, preferred_name: str | None = None) -> None:
        sequence = self._create_sequence(
            preferred_name=preferred_name,
            mode="next_available",
            allocator=self._next_sequence_after_highest,
        )
        self._send_message(
            "sequence",
            "created",
            {"no": sequence.number, "name": sequence.name, "mode": "next_available"},
        )

    def _handle_CreateSequenceInCurrentSongRange(self, preferred_name: str | None = None) -> None:
        sequence_range = self._resolve_current_song_range()
        if sequence_range is None:
            return
        sequence = self._create_sequence(
            preferred_name=preferred_name,
            mode="current_song_range",
            allocator=lambda: self._next_available_sequence_no(
                start=sequence_range.start,
                end=sequence_range.end,
            ),
        )
        self._send_message(
            "sequence",
            "created",
            {"no": sequence.number, "name": sequence.name, "mode": "current_song_range"},
        )

    def _handle_PrepareTrackForEvents(
        self,
        tc_no: int,
        tg_no: int,
        track_no: int,
    ) -> None:
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        if coord not in self._tracks_by_coord:
            self._send_track_error(tc_no, tg_no, track_no, "Track does not exist")
            return
        sequence_no = self._sequence_by_coord.get(coord)
        if sequence_no is None:
            self._send_track_error(tc_no, tg_no, track_no, "Track has no assigned sequence")
            return
        if coord in self._cmd_subtrack_create_blocked:
            self._send_track_error(
                tc_no, tg_no, track_no, "Track prep could not create CmdSubTrack"
            )
            return
        self._cmd_subtrack_ready_by_coord[coord] = True
        time_range_idx = self._time_range_idx_by_coord.setdefault(coord, 1)
        self._send_message(
            "track",
            "prepared",
            {
                "tc": int(tc_no),
                "tg": int(tg_no),
                "track": int(track_no),
                "seq": int(sequence_no),
                "time_range_idx": int(time_range_idx),
                "cmd_subtrack_ready": True,
            },
        )

    def _handle_CreateCmdSubTrack(
        self,
        tc_no: int,
        tg_no: int,
        track_no: int,
        _time_range_idx: int,
    ) -> None:
        coord = format_track_coord(int(tc_no), int(tg_no), int(track_no))
        if coord in self._cmd_subtrack_create_blocked:
            return
        self._cmd_subtrack_ready_by_coord[coord] = True

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
        if coord in self._hook_fail_coords:
            self._send_message(
                "hooks",
                "error",
                {
                    "action": "hook_failed",
                    "reason": "simulated_hook_failure",
                    "tc": int(tc_no),
                    "tg": int(tg_no),
                    "track": int(track_no),
                },
            )
            return
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
        for group_tc_no, group_no in self._track_group_name_by_key:
            if int(group_tc_no) == int(tc_no):
                groups.setdefault(int(group_no), [])
        for tracks in groups.values():
            tracks.sort(key=lambda track: parse_track_coord(track.coord)[2])
        return groups

    @staticmethod
    def _normalize_datapool_path(path: str | None) -> str:
        text = str(path or "").strip().strip("/")
        if text.startswith("DataPool/"):
            text = text[len("DataPool/") :]
        elif text == "DataPool":
            text = ""
        return text

    @staticmethod
    def _path_tokens(path: str) -> list[str]:
        if not path:
            return []
        return [token for token in path.split("/") if token]

    def _datapool_children_for_path(self, path: str) -> list[dict[str, object]] | None:
        tokens = self._path_tokens(path)
        if not tokens:
            return [
                self._make_datapool_entry(
                    path="Timecodes",
                    name="Timecodes",
                    class_name="DataPoolCategory",
                    child_count=len(self._timecode_name_by_no),
                ),
                self._make_datapool_entry(
                    path="Sequences",
                    name="Sequences",
                    class_name="DataPoolCategory",
                    child_count=len(self._sequences_by_number),
                ),
                self._make_datapool_entry(
                    path="PresetPools",
                    name="PresetPools",
                    class_name="DataPoolCategory",
                    child_count=len(self._preset_pool_types()),
                ),
            ]
        if tokens == ["Timecodes"]:
            return [
                self._make_datapool_entry(
                    path=f"Timecodes/{tc_no}",
                    name=name,
                    class_name="Timecode",
                    no=int(tc_no),
                    browse_token=str(tc_no),
                    child_count=len(self._group_tracks(int(tc_no))),
                )
                for tc_no, name in sorted(self._timecode_name_by_no.items())
            ]
        if len(tokens) == 2 and tokens[0] == "Timecodes":
            tc_no = int(tokens[1])
            groups = self._group_tracks(tc_no)
            return [
                self._make_datapool_entry(
                    path=f"Timecodes/{tc_no}/{group_no}",
                    name=self._track_group_name_by_key.get(
                        (tc_no, int(group_no)), f"Group {group_no}"
                    ),
                    class_name="TrackGroup",
                    no=int(group_no),
                    browse_token=str(group_no),
                    child_count=len(tracks),
                )
                for group_no, tracks in sorted(groups.items())
            ]
        if len(tokens) == 3 and tokens[0] == "Timecodes":
            tc_no = int(tokens[1])
            tg_no = int(tokens[2])
            tracks = self._group_tracks(tc_no).get(tg_no, [])
            return [
                self._make_datapool_entry(
                    path=f"Timecodes/{tc_no}/{tg_no}/{track.number}",
                    name=track.name,
                    class_name="Track",
                    no=int(track.number),
                    browse_token=str(track.number),
                    child_count=len(self._events_by_coord.get(track.coord, [])),
                )
                for track in tracks
            ]
        if tokens == ["Sequences"]:
            return [
                self._make_datapool_entry(
                    path=f"Sequences/{sequence.number}",
                    name=sequence.name,
                    class_name="Sequence",
                    no=int(sequence.number),
                    browse_token=str(sequence.number),
                    child_count=int(sequence.cue_count),
                )
                for sequence in sorted(
                    self._sequences_by_number.values(), key=lambda item: item.number
                )
            ]
        if tokens == ["PresetPools"]:
            return [
                self._make_datapool_entry(
                    path=f"PresetPools/{preset_type_no}",
                    name=self._preset_pool_name(preset_type_no),
                    class_name="PresetPool",
                    no=int(preset_type_no),
                    browse_token=str(preset_type_no),
                    child_count=len(self._preset_numbers_for_type(preset_type_no)),
                )
                for preset_type_no in self._preset_pool_types()
            ]
        if len(tokens) == 2 and tokens[0] == "PresetPools":
            preset_type_no = int(tokens[1])
            return [
                self._make_datapool_entry(
                    path=f"PresetPools/{preset_type_no}/{preset_no}",
                    name=snapshot.name,
                    class_name="Preset",
                    no=int(preset_no),
                    browse_token=str(preset_no),
                    child_count=1 if self._preset_has_recipe_child(snapshot) else 0,
                )
                for preset_no, snapshot in self._preset_snapshots_for_type(preset_type_no)
            ]
        if len(tokens) == 3 and tokens[0] == "PresetPools":
            preset_type_no = int(tokens[1])
            preset_no = int(tokens[2])
            snapshot = self._presets_by_key.get((preset_type_no, preset_no))
            if snapshot is None or not self._preset_has_recipe_child(snapshot):
                return []
            return [
                self._make_datapool_entry(
                    path=f"PresetPools/{preset_type_no}/{preset_no}/Recipe 1",
                    name="Recipe 1",
                    class_name="Recipe",
                    browse_token="Recipe 1",
                    child_count=0,
                )
            ]
        return None

    def _datapool_object_for_path(self, path: str) -> dict[str, object] | None:
        tokens = self._path_tokens(path)
        if not tokens:
            return {
                **self._make_datapool_entry(
                    path="", name="DataPool", class_name="DataPool", child_count=3
                ),
                "properties": self._make_datapool_properties(
                    ("NAME", "DataPool"),
                    ("COUNT", 3),
                ),
                "property_items": self._make_datapool_properties(
                    ("NAME", "DataPool"),
                    ("COUNT", 3),
                ),
                "preview_children": self._datapool_children_for_path("") or [],
                "dump": "DataPool()",
            }
        if tokens == ["Timecodes"]:
            children = self._datapool_children_for_path(path) or []
            return {
                **self._make_datapool_entry(
                    path="Timecodes",
                    name="Timecodes",
                    class_name="DataPoolCategory",
                    child_count=len(children),
                ),
                "properties": self._make_datapool_properties(
                    ("NAME", "Timecodes"),
                    ("COUNT", len(children)),
                ),
                "property_items": self._make_datapool_properties(
                    ("NAME", "Timecodes"),
                    ("COUNT", len(children)),
                ),
                "preview_children": children[:12],
                "dump": "DataPool().Timecodes",
            }
        if len(tokens) == 2 and tokens[0] == "Timecodes":
            tc_no = int(tokens[1])
            name = self._timecode_name_by_no.get(tc_no)
            if name is None:
                return None
            children = self._datapool_children_for_path(path) or []
            return {
                **self._make_datapool_entry(
                    path=path,
                    name=name,
                    class_name="Timecode",
                    no=tc_no,
                    browse_token=str(tc_no),
                    child_count=len(children),
                ),
                "properties": self._make_datapool_properties(
                    ("NAME", name),
                    ("NO", tc_no),
                    ("COUNT", len(children)),
                ),
                "property_items": self._make_datapool_properties(
                    ("NAME", name),
                    ("NO", tc_no),
                    ("COUNT", len(children)),
                ),
                "preview_children": children[:12],
                "dump": f"DataPool().Timecodes[{tc_no}]",
            }
        if len(tokens) == 3 and tokens[0] == "Timecodes":
            tc_no = int(tokens[1])
            tg_no = int(tokens[2])
            name = self._track_group_name_by_key.get((tc_no, tg_no), f"Group {tg_no}")
            children = self._datapool_children_for_path(path) or []
            return {
                **self._make_datapool_entry(
                    path=path,
                    name=name,
                    class_name="TrackGroup",
                    no=tg_no,
                    browse_token=str(tg_no),
                    child_count=len(children),
                ),
                "properties": self._make_datapool_properties(
                    ("NAME", name),
                    ("NO", tg_no),
                    ("COUNT", len(children)),
                ),
                "property_items": self._make_datapool_properties(
                    ("NAME", name),
                    ("NO", tg_no),
                    ("COUNT", len(children)),
                ),
                "preview_children": children[:12],
                "dump": f"DataPool().Timecodes[{tc_no}][{tg_no}]",
            }
        if len(tokens) == 4 and tokens[0] == "Timecodes":
            tc_no = int(tokens[1])
            tg_no = int(tokens[2])
            track_no = int(tokens[3])
            coord = format_track_coord(tc_no, tg_no, track_no)
            track = self._tracks_by_coord.get(coord)
            if track is None:
                return None
            preview_children: list[dict[str, object]] = []
            for index, event in enumerate(self._events_by_coord.get(coord, []), start=1):
                preview_children.append(
                    self._make_datapool_entry(
                        path=f"{path}/Event{index}",
                        name=event.label,
                        class_name="CmdEvent",
                        no=index,
                        browse_token=f"Event{index}",
                        child_count=0,
                    )
                )
            return {
                **self._make_datapool_entry(
                    path=path,
                    name=track.name,
                    class_name="Track",
                    no=track.number,
                    browse_token=str(track.number),
                    child_count=len(preview_children),
                ),
                "properties": self._make_datapool_properties(
                    ("NAME", track.name),
                    ("NO", track.number),
                    ("NOTE", track.note or ""),
                    ("SEQUENCE", track.sequence_no),
                ),
                "property_items": self._make_datapool_properties(
                    ("NAME", track.name),
                    ("NO", track.number),
                    ("NOTE", track.note or ""),
                    ("SEQUENCE", track.sequence_no),
                ),
                "preview_children": preview_children[:12],
                "dump": (
                    f"DataPool().Timecodes[{tc_no}][{tg_no}][{track_no}]\n"
                    f"class=Track\nname={track.name}\nno={track.number}\n"
                    f"note={track.note or ''}\nsequence={track.sequence_no or ''}"
                ),
            }
        if tokens == ["Sequences"]:
            children = self._datapool_children_for_path(path) or []
            return {
                **self._make_datapool_entry(
                    path="Sequences",
                    name="Sequences",
                    class_name="DataPoolCategory",
                    child_count=len(children),
                ),
                "properties": self._make_datapool_properties(
                    ("NAME", "Sequences"),
                    ("COUNT", len(children)),
                ),
                "property_items": self._make_datapool_properties(
                    ("NAME", "Sequences"),
                    ("COUNT", len(children)),
                ),
                "preview_children": children[:12],
                "dump": "DataPool().Sequences",
            }
        if len(tokens) == 2 and tokens[0] == "Sequences":
            sequence_no = int(tokens[1])
            sequence = self._sequences_by_number.get(sequence_no)
            if sequence is None:
                return None
            return {
                **self._make_datapool_entry(
                    path=path,
                    name=sequence.name,
                    class_name="Sequence",
                    no=sequence.number,
                    browse_token=str(sequence.number),
                    child_count=int(sequence.cue_count),
                ),
                "properties": self._make_datapool_properties(
                    ("NAME", sequence.name),
                    ("NO", sequence.number),
                    ("CUE_COUNT", int(sequence.cue_count)),
                ),
                "property_items": self._make_datapool_properties(
                    ("NAME", sequence.name),
                    ("NO", sequence.number),
                    ("CUE_COUNT", int(sequence.cue_count)),
                ),
                "preview_children": [],
                "dump": (
                    f"DataPool().Sequences[{sequence.number}]\n"
                    f"class=Sequence\nname={sequence.name}\nno={sequence.number}\n"
                    f"cue_count={sequence.cue_count}"
                ),
            }
        if tokens == ["PresetPools"]:
            children = self._datapool_children_for_path(path) or []
            property_items = self._make_datapool_properties(
                ("NAME", "PresetPools"),
                ("COUNT", len(children)),
            )
            return {
                **self._make_datapool_entry(
                    path="PresetPools",
                    name="PresetPools",
                    class_name="DataPoolCategory",
                    child_count=len(children),
                ),
                "properties": property_items,
                "property_items": property_items,
                "preview_children": children[:12],
                "dump": "DataPool().PresetPools",
            }
        if len(tokens) == 2 and tokens[0] == "PresetPools":
            preset_type_no = int(tokens[1])
            children = self._datapool_children_for_path(path) or []
            property_items = self._make_datapool_properties(
                ("NAME", self._preset_pool_name(preset_type_no)),
                ("NO", preset_type_no),
                ("COUNT", len(children)),
            )
            return {
                **self._make_datapool_entry(
                    path=path,
                    name=self._preset_pool_name(preset_type_no),
                    class_name="PresetPool",
                    no=preset_type_no,
                    browse_token=str(preset_type_no),
                    child_count=len(children),
                ),
                "properties": property_items,
                "property_items": property_items,
                "preview_children": children[:12],
                "dump": f"DataPool().PresetPools[{preset_type_no}]",
            }
        if len(tokens) == 3 and tokens[0] == "PresetPools":
            preset_type_no = int(tokens[1])
            preset_no = int(tokens[2])
            snapshot = self._presets_by_key.get((preset_type_no, preset_no))
            if snapshot is None:
                return None
            property_items = self._preset_property_items(snapshot)
            preview_children = self._datapool_children_for_path(path) or []
            children = [
                self._datapool_object_for_path(str(child.get("path") or ""))
                for child in preview_children
                if str(child.get("path") or "").strip()
            ]
            return {
                **self._make_datapool_entry(
                    path=path,
                    name=snapshot.name,
                    class_name="Preset",
                    no=snapshot.number,
                    browse_token=str(snapshot.number),
                    child_count=len(preview_children),
                ),
                "preset_type": snapshot.preset_type,
                "store_mode": snapshot.store_mode,
                "kind": snapshot.kind,
                "step_count": snapshot.step_count,
                "properties": self._preset_properties_dict(property_items),
                "property_items": property_items,
                "preview_children": preview_children[:12],
                "children": [child for child in children if isinstance(child, dict)],
                "dump": f"DataPool().PresetPools[{preset_type_no}][{preset_no}]",
            }
        if len(tokens) == 4 and tokens[0] == "PresetPools":
            preset_type_no = int(tokens[1])
            preset_no = int(tokens[2])
            snapshot = self._presets_by_key.get((preset_type_no, preset_no))
            if (
                snapshot is None
                or tokens[3] != "Recipe 1"
                or not self._preset_has_recipe_child(snapshot)
            ):
                return None
            property_items = self._preset_recipe_property_items(snapshot)
            return {
                **self._make_datapool_entry(
                    path=path,
                    name="Recipe 1",
                    class_name="Recipe",
                    browse_token="Recipe 1",
                    child_count=0,
                ),
                "properties": self._preset_properties_dict(property_items),
                "property_items": property_items,
                "preview_children": [],
                "dump": f"DataPool().PresetPools[{preset_type_no}][{preset_no}]:Children()[1]",
            }
        return None

    def _preset_pool_types(self) -> list[int]:
        return sorted({preset_type_no for preset_type_no, _preset_no in self._presets_by_key})

    def _preset_numbers_for_type(self, preset_type_no: int) -> list[int]:
        return sorted(
            preset_no
            for (raw_type_no, preset_no) in self._presets_by_key
            if int(raw_type_no) == int(preset_type_no)
        )

    def _preset_snapshots_for_type(
        self, preset_type_no: int
    ) -> list[tuple[int, MA3PresetSnapshot]]:
        return [
            (preset_no, self._presets_by_key[(raw_type_no, preset_no)])
            for raw_type_no, preset_no in sorted(self._presets_by_key)
            if int(raw_type_no) == int(preset_type_no)
        ]

    def _preset_pool_name(self, preset_type_no: int) -> str:
        return _PRESET_POOL_NAMES.get(int(preset_type_no), f"Preset Pool {int(preset_type_no)}")

    @staticmethod
    def _preset_has_recipe_child(snapshot: MA3PresetSnapshot) -> bool:
        return snapshot.kind in {"phaser", "recipe"}

    @staticmethod
    def _preset_properties_dict(property_items: list[dict[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for item in property_items:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            result[name.lower()] = item.get("value")
        return result

    def _preset_property_items(self, snapshot: MA3PresetSnapshot) -> list[dict[str, object]]:
        details = self._preset_details_by_key.get((snapshot.preset_type, snapshot.number), {})
        items = self._make_datapool_properties(
            ("NAME", snapshot.name),
            ("NO", snapshot.number),
            ("PRESETTYPE", snapshot.preset_type),
            ("PRESETMODE", snapshot.store_mode),
            ("KIND", snapshot.kind),
            ("STEPCOUNT", snapshot.step_count),
        )
        if "selection_command" in details:
            items.extend(
                self._make_datapool_properties(("SELECTIONCOMMAND", details["selection_command"]))
            )
        return items

    def _preset_recipe_property_items(
        self, snapshot: MA3PresetSnapshot
    ) -> list[dict[str, object]]:
        details = self._preset_details_by_key.get((snapshot.preset_type, snapshot.number), {})
        items = self._make_datapool_properties(
            ("NAME", "Recipe 1"),
            ("PARENT_PRESET", f"{snapshot.preset_type}.{snapshot.number}"),
            ("KIND", snapshot.kind),
        )
        if snapshot.kind == "phaser":
            step_spec = str(details.get("step_spec") or "")
            for step_index, step_group in enumerate(
                [group for group in step_spec.split(";") if group], start=1
            ):
                items.extend(self._make_datapool_properties((f"STEP_{step_index}", step_group)))
            if details.get("speed_bpm") is not None:
                items.extend(self._make_datapool_properties(("SPEEDFROMX", details["speed_bpm"])))
        if snapshot.kind == "recipe":
            if details.get("source_preset_ref") is not None:
                items.extend(
                    self._make_datapool_properties(("PRESET", details["source_preset_ref"]))
                )
            if details.get("selection_mode") is not None:
                items.extend(
                    self._make_datapool_properties(("SELECTIONMODE", details["selection_mode"]))
                )
        return items

    def _preset_replace_findings(
        self,
        *,
        source_preset_ref: str,
        dest_preset_ref: str,
        group_filter_csv: str,
        sequence_numbers_csv: str,
    ) -> list[dict[str, object]]:
        group_filter = {
            item.strip() for item in str(group_filter_csv or "").split(",") if item.strip()
        }
        sequence_numbers = {
            int(item.strip())
            for item in str(sequence_numbers_csv or "").split(",")
            if item.strip().isdigit()
        }
        findings: list[dict[str, object]] = []
        for row in self._recipe_line_rows:
            if str(row.get("preset_ref") or "") != str(source_preset_ref):
                continue
            if group_filter and str(row.get("matched_group") or "") not in group_filter:
                continue
            if sequence_numbers and int(row.get("seq_number") or 0) not in sequence_numbers:
                continue
            findings.append(
                {
                    "description": (
                        f'Seq {int(row["seq_number"])} "{row["seq_name"]}" Cue '
                        f'{float(row["actual_cue_number"]):g} Part {row["part_number"]} '
                        f'[Group: {row["matched_group"]}]: {source_preset_ref} -> {dest_preset_ref}'
                    ),
                    "seqNumber": int(row["seq_number"]),
                    "actualCueNumber": float(row["actual_cue_number"]),
                    "partNumber": str(row["part_number"]),
                    "matched_group": str(row["matched_group"]),
                }
            )
        return findings

    def _cue_recipe_rows(
        self,
        *,
        sequence_no: int,
        cue_number: float,
    ) -> list[dict[str, object]]:
        rows = [
            dict(row)
            for row in self._recipe_line_rows
            if int(row.get("seq_number") or 0) == int(sequence_no)
            and float(row.get("actual_cue_number") or 0.0) == float(cue_number)
        ]
        rows.sort(
            key=lambda row: (
                float(row.get("actual_cue_number") or 0.0),
                int(row.get("line_index") or 0),
                str(row.get("part_number") or ""),
            )
        )
        return rows

    @staticmethod
    def _recipe_state_key(row: dict[str, object]) -> str:
        explicit = str(row.get("selection_key") or "").strip()
        if explicit:
            return explicit
        return f'{str(row.get("matched_group") or "").strip()}:{str(row.get("feature_group") or "").strip()}'

    @staticmethod
    def _dedupe_texts(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _recipe_analysis_flags(
        self,
        *,
        rows: list[dict[str, object]],
    ) -> tuple[bool, list[str], list[str]]:
        warnings: list[str] = []
        unsupported_reasons: list[str] = []
        for row in rows:
            selection_key = self._recipe_state_key(row)
            recipe_mode = str(row.get("recipe_mode") or "").strip().lower()
            if not selection_key or selection_key == ":":
                unsupported_reasons.append(
                    "One or more recipe lines are missing a stable selection key."
                )
            if recipe_mode not in {"absolute", "relative"}:
                unsupported_reasons.append(
                    "One or more recipe lines are missing relative/absolute mode metadata."
                )
        if rows:
            warnings.append(
                "Analysis is limited to detected cue recipe lines and does not model direct stored values or cooked MA output."
            )
        return (
            (not unsupported_reasons),
            self._dedupe_texts(warnings),
            self._dedupe_texts(unsupported_reasons),
        )

    def _effective_recipe_contributors(
        self,
        *,
        sequence_no: int,
        cue_number: float,
    ) -> list[dict[str, object]]:
        relevant_rows = [
            dict(row)
            for row in self._recipe_line_rows
            if int(row.get("seq_number") or 0) == int(sequence_no)
            and float(row.get("actual_cue_number") or 0.0) <= float(cue_number)
        ]
        relevant_rows.sort(
            key=lambda row: (
                float(row.get("actual_cue_number") or 0.0),
                int(row.get("line_index") or 0),
                str(row.get("part_number") or ""),
            )
        )
        contributors_by_key: dict[str, list[dict[str, object]]] = {}
        for row in relevant_rows:
            key = self._recipe_state_key(row)
            mode = str(row.get("recipe_mode") or "absolute").strip().lower()
            if mode == "relative":
                bucket = contributors_by_key.setdefault(key, [])
                bucket.append(dict(row))
                continue
            contributors_by_key[key] = [dict(row)]
        flattened: list[dict[str, object]] = []
        for bucket in contributors_by_key.values():
            flattened.extend(dict(item) for item in bucket)
        flattened.sort(
            key=lambda row: (
                float(row.get("actual_cue_number") or 0.0),
                int(row.get("line_index") or 0),
                str(row.get("part_number") or ""),
            )
        )
        return flattened

    def _analyze_cue_recipe_state(
        self,
        *,
        sequence_no: int,
        cue_number: float,
    ) -> dict[str, object]:
        local_rows = self._cue_recipe_rows(sequence_no=sequence_no, cue_number=cue_number)
        contributor_rows = self._effective_recipe_contributors(
            sequence_no=sequence_no,
            cue_number=cue_number,
        )
        state_keys = sorted({self._recipe_state_key(row) for row in contributor_rows})
        supported, warnings, unsupported_reasons = self._recipe_analysis_flags(
            rows=contributor_rows
        )
        return {
            "sequence_no": int(sequence_no),
            "cue_no": float(cue_number),
            "supported": supported,
            "status": "ready" if supported else "unsupported",
            "warnings": warnings,
            "unsupported_reasons": unsupported_reasons,
            "local_line_count": len(local_rows),
            "contributor_count": len(contributor_rows),
            "state_keys": state_keys,
            "local_lines": local_rows,
            "contributors": contributor_rows,
        }

    def _replace_cue_recipe_rows(
        self,
        *,
        sequence_no: int,
        cue_number: float,
        replacement_rows: list[dict[str, object]],
    ) -> None:
        survivors = [
            dict(row)
            for row in self._recipe_line_rows
            if not (
                int(row.get("seq_number") or 0) == int(sequence_no)
                and float(row.get("actual_cue_number") or 0.0) == float(cue_number)
            )
        ]
        normalized_rows: list[dict[str, object]] = []
        for line_index, row in enumerate(replacement_rows, start=1):
            cloned = dict(row)
            cloned["seq_number"] = int(sequence_no)
            cloned["actual_cue_number"] = float(cue_number)
            cloned["line_index"] = line_index
            cloned["part_number"] = f"0.{line_index}"
            cloned["source_cue_number"] = float(row.get("source_cue_number") or cue_number)
            cloned["source_part_number"] = str(
                row.get("source_part_number") or row.get("part_number") or f"0.{line_index}"
            )
            normalized_rows.append(cloned)
        survivors.extend(normalized_rows)
        self._recipe_line_rows = survivors

    @staticmethod
    def _contributor_signature(rows: list[dict[str, object]]) -> set[tuple[str, str, str, str]]:
        return {
            (
                str(row.get("selection_key") or ""),
                str(row.get("recipe_mode") or ""),
                str(row.get("preset_ref") or ""),
                str(row.get("matched_group") or ""),
            )
            for row in rows
        }

    def _preview_recipe_cue_only(
        self,
        *,
        sequence_no: int,
        source_cue_no: float,
        target_cue_no: float,
    ) -> dict[str, object]:
        source_analysis = self._analyze_cue_recipe_state(
            sequence_no=sequence_no,
            cue_number=source_cue_no,
        )
        next_cue_no = float(target_cue_no) + 1.0
        incoming_rows = self._cue_recipe_rows(sequence_no=sequence_no, cue_number=source_cue_no)
        before_next = self._effective_recipe_contributors(
            sequence_no=sequence_no, cue_number=next_cue_no
        )
        after_target_rows = [dict(row) for row in incoming_rows]

        original_rows = list(self._recipe_line_rows)
        self._replace_cue_recipe_rows(
            sequence_no=sequence_no,
            cue_number=target_cue_no,
            replacement_rows=after_target_rows,
        )
        after_next = self._effective_recipe_contributors(
            sequence_no=sequence_no, cue_number=next_cue_no
        )
        self._recipe_line_rows = original_rows

        affected_keys = {self._recipe_state_key(row) for row in before_next + after_next}
        restore_rows = [
            dict(row) for row in before_next if self._recipe_state_key(row) in affected_keys
        ]
        changed_keys = sorted(
            key
            for key in affected_keys
            if self._contributor_signature(
                [row for row in before_next if self._recipe_state_key(row) == key]
            )
            != self._contributor_signature(
                [row for row in after_next if self._recipe_state_key(row) == key]
            )
        )
        restore_rows = [
            row for row in restore_rows if self._recipe_state_key(row) in set(changed_keys)
        ]
        warnings = list(source_analysis.get("warnings") or [])
        unsupported_reasons = list(source_analysis.get("unsupported_reasons") or [])
        if float(source_cue_no) == float(target_cue_no):
            unsupported_reasons.append("Source cue and target cue must be different.")
        if not incoming_rows:
            unsupported_reasons.append("Source cue does not expose local recipe lines.")
        if before_next:
            warnings.append(
                "Cue-only preview only restores detected recipe contributors in the following cue; direct stored values are not modeled."
            )
        unsupported_reasons = self._dedupe_texts(unsupported_reasons)
        warnings = self._dedupe_texts(warnings)
        return {
            "sequence_no": int(sequence_no),
            "source_cue_no": float(source_cue_no),
            "target_cue_no": float(target_cue_no),
            "next_cue_no": float(next_cue_no),
            "supported": not unsupported_reasons,
            "status": "ready" if not unsupported_reasons else "unsupported",
            "warnings": warnings,
            "unsupported_reasons": unsupported_reasons,
            "stored_lines": [dict(row) for row in incoming_rows],
            "restore_lines": restore_rows,
            "changed_keys": changed_keys,
        }

    def _apply_recipe_cue_only(
        self,
        *,
        sequence_no: int,
        source_cue_no: float,
        target_cue_no: float,
    ) -> dict[str, object]:
        preview = self._preview_recipe_cue_only(
            sequence_no=sequence_no,
            source_cue_no=source_cue_no,
            target_cue_no=target_cue_no,
        )
        self._replace_cue_recipe_rows(
            sequence_no=sequence_no,
            cue_number=float(target_cue_no),
            replacement_rows=[dict(row) for row in preview["stored_lines"]],
        )
        existing_next_rows = self._cue_recipe_rows(
            sequence_no=sequence_no,
            cue_number=float(preview["next_cue_no"]),
        )
        changed_keys = set(preview["changed_keys"])
        merged_next_rows = [
            dict(row)
            for row in existing_next_rows
            if self._recipe_state_key(row) not in changed_keys
        ]
        merged_next_rows.extend(dict(row) for row in preview["restore_lines"])
        self._replace_cue_recipe_rows(
            sequence_no=sequence_no,
            cue_number=float(preview["next_cue_no"]),
            replacement_rows=merged_next_rows,
        )
        return preview

    def _copy_cue_with_status(
        self,
        *,
        sequence_no: int,
        source_cue_no: float,
        dest_cue_no: float,
    ) -> dict[str, object]:
        contributors = self._effective_recipe_contributors(
            sequence_no=sequence_no,
            cue_number=source_cue_no,
        )
        self._replace_cue_recipe_rows(
            sequence_no=sequence_no,
            cue_number=dest_cue_no,
            replacement_rows=[dict(row) for row in contributors],
        )
        return {
            "sequence_no": int(sequence_no),
            "source_cue_no": float(source_cue_no),
            "dest_cue_no": float(dest_cue_no),
            "copied_lines": [dict(row) for row in contributors],
            "copied_line_count": len(contributors),
        }

    def _preview_copy_cue_with_status(
        self,
        *,
        sequence_no: int,
        source_cue_no: float,
        dest_cue_no: float,
    ) -> dict[str, object]:
        analysis = self._analyze_cue_recipe_state(
            sequence_no=sequence_no,
            cue_number=source_cue_no,
        )
        warnings = list(analysis.get("warnings") or [])
        unsupported_reasons = list(analysis.get("unsupported_reasons") or [])
        if float(source_cue_no) == float(dest_cue_no):
            unsupported_reasons.append("Source cue and destination cue must be different.")
        if int(analysis.get("contributor_count") or 0) > int(
            analysis.get("local_line_count") or 0
        ):
            warnings.append(
                "Status preview includes tracked contributors from earlier cues, not only local recipe lines."
            )
        warnings = self._dedupe_texts(warnings)
        unsupported_reasons = self._dedupe_texts(unsupported_reasons)
        return {
            "sequence_no": int(sequence_no),
            "source_cue_no": float(source_cue_no),
            "dest_cue_no": float(dest_cue_no),
            "supported": not unsupported_reasons,
            "status": "ready" if not unsupported_reasons else "unsupported",
            "warnings": warnings,
            "unsupported_reasons": unsupported_reasons,
            "copied_lines": [dict(row) for row in analysis.get("contributors") or []],
            "copied_line_count": int(analysis.get("contributor_count") or 0),
            "local_line_count": int(analysis.get("local_line_count") or 0),
            "contributor_count": int(analysis.get("contributor_count") or 0),
        }

    @staticmethod
    def _make_datapool_entry(
        *,
        path: str,
        name: str,
        class_name: str,
        no: int | None = None,
        browse_token: str | None = None,
        child_count: int = 0,
    ) -> dict[str, object]:
        entry: dict[str, object] = {
            "path": path,
            "name": str(name),
            "class": class_name,
            "child_count": int(child_count),
        }
        if no is not None:
            entry["no"] = int(no)
        if browse_token is not None:
            entry["browse_token"] = str(browse_token)
        return entry

    @staticmethod
    def _make_datapool_properties(*items: tuple[str, object]) -> list[dict[str, object]]:
        properties: list[dict[str, object]] = []
        for name, value in items:
            properties.append(
                {
                    "name": str(name),
                    "value": value,
                    "property_type": type(value).__name__ if value is not None else "nil",
                    "read_only": False,
                }
            )
        return properties

    def _create_sequence(
        self,
        *,
        preferred_name: str | None,
        mode: str,
        allocator,
    ) -> MA3SequenceSnapshot:
        sequence_no = allocator()
        if sequence_no is None:
            raise RuntimeError(f"Unable to allocate MA3 sequence for mode {mode}")
        sequence = MA3SequenceSnapshot(
            number=int(sequence_no),
            name=str(preferred_name or f"Sequence {int(sequence_no)}"),
            cue_count=0,
        )
        self._sequences_by_number[sequence.number] = sequence
        return sequence

    def _next_available_sequence_no(
        self,
        *,
        start: int = 1,
        end: int | None = None,
    ) -> int | None:
        used = set(self._sequences_by_number)
        current = max(1, int(start))
        if end is None:
            while current in used:
                current += 1
            return current
        for number in range(current, int(end) + 1):
            if number not in used:
                return number
        return None

    def _next_sequence_after_highest(self) -> int:
        if not self._sequences_by_number:
            return 1
        return max(self._sequences_by_number) + 1

    def _resolve_current_song_range(self) -> MA3SequenceRangeSnapshot | None:
        song_label = self._current_song_label
        if not song_label:
            return None
        anchor = None
        for sequence in self._sequences_by_number.values():
            if sequence.name == song_label:
                anchor = sequence
                break
        if anchor is None:
            return None
        return MA3SequenceRangeSnapshot(
            song_label=song_label,
            start=anchor.number,
            end=anchor.number + 99,
        )

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

    def _send_track_error(self, tc_no: int, tg_no: int, track_no: int, error: str) -> None:
        self._send_message(
            "track",
            "error",
            {
                "tc": int(tc_no),
                "tg": int(tg_no),
                "track": int(track_no),
                "error": str(error),
            },
        )

    def _send_preset_error(self, preset_type_no: int, preset_no: int, error: str) -> None:
        self._send_message(
            "preset",
            "error",
            {
                "preset_type": int(preset_type_no),
                "number": int(preset_no),
                "error": str(error),
            },
        )

    def _send_preset_snapshot(self, change: str, snapshot: MA3PresetSnapshot) -> None:
        self._send_message(
            "preset",
            change,
            {
                "preset_type": snapshot.preset_type,
                "number": snapshot.number,
                "name": snapshot.name,
                "store_mode": snapshot.store_mode,
                "kind": snapshot.kind,
                "step_count": snapshot.step_count,
            },
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
            command_transport=OscUdpSendTransport(*server.endpoint, path="/cmd"),
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
        deadline = monotonic() + 0.25
        while monotonic() < deadline:
            recent_commands = server.commands[prior_command_count:]
            if "EZ.UnhookAll()" in recent_commands:
                break
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

    def list_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[MA3TrackSnapshot]:
        return self._require_bridge().list_tracks(
            timecode_no=timecode_no,
            track_group_no=track_group_no,
        )

    def refresh_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[MA3TrackSnapshot]:
        return self._require_bridge().refresh_tracks(
            timecode_no=timecode_no,
            track_group_no=track_group_no,
        )

    def list_timecodes(self) -> list[MA3TimecodeSnapshot]:
        return self._require_bridge().list_timecodes()

    def list_track_groups(self, *, timecode_no: int) -> list[MA3TrackGroupSnapshot]:
        return self._require_bridge().list_track_groups(timecode_no=timecode_no)

    def refresh_track_groups(self, *, timecode_no: int) -> list[MA3TrackGroupSnapshot]:
        return self._require_bridge().refresh_track_groups(timecode_no=timecode_no)

    def list_track_events(self, track_coord: str) -> list[MA3EventSnapshot]:
        return self._require_bridge().list_track_events(track_coord)

    def refresh_track_events(self, track_coord: str) -> list[MA3EventSnapshot]:
        return self._require_bridge().refresh_track_events(track_coord)

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[MA3SequenceSnapshot]:
        return self._require_bridge().list_sequences(start_no=start_no, end_no=end_no)

    def list_sequence_cues(self, *, sequence_no: int) -> list[dict[str, object]]:
        return self._require_bridge().list_sequence_cues(sequence_no=sequence_no)

    def get_current_song_sequence_range(self) -> MA3SequenceRangeSnapshot | None:
        return self._require_bridge().get_current_song_sequence_range()

    def list_presets(self, *, preset_type_no: int) -> list[dict[str, object]]:
        return self._require_bridge().list_presets(preset_type_no=preset_type_no)

    def describe_preset(self, *, preset_type_no: int, preset_no: int) -> dict[str, object]:
        return self._require_bridge().describe_preset(
            preset_type_no=preset_type_no,
            preset_no=preset_no,
        )

    def preview_replace_preset_when_group(
        self,
        *,
        preset_type_no: int,
        source_preset_ref: str,
        dest_preset_ref: str,
        group_filter_csv: str,
        sequence_numbers_csv: str,
    ) -> dict[str, object]:
        return self._require_bridge().preview_replace_preset_when_group(
            preset_type_no=preset_type_no,
            source_preset_ref=source_preset_ref,
            dest_preset_ref=dest_preset_ref,
            group_filter_csv=group_filter_csv,
            sequence_numbers_csv=sequence_numbers_csv,
        )

    def replace_preset_when_group(
        self,
        *,
        preset_type_no: int,
        source_preset_ref: str,
        dest_preset_ref: str,
        group_filter_csv: str,
        sequence_numbers_csv: str,
    ) -> dict[str, object]:
        return self._require_bridge().replace_preset_when_group(
            preset_type_no=preset_type_no,
            source_preset_ref=source_preset_ref,
            dest_preset_ref=dest_preset_ref,
            group_filter_csv=group_filter_csv,
            sequence_numbers_csv=sequence_numbers_csv,
        )

    def analyze_cue_recipe_state(
        self,
        *,
        sequence_no: int,
        cue_no: CueNumber | float | int | str,
    ) -> dict[str, object]:
        return self._require_bridge().analyze_cue_recipe_state(
            sequence_no=sequence_no,
            cue_no=cue_no,
        )

    def preview_recipe_cue_only(
        self,
        *,
        sequence_no: int,
        source_cue_no: CueNumber | float | int | str,
        target_cue_no: CueNumber | float | int | str,
    ) -> dict[str, object]:
        return self._require_bridge().preview_recipe_cue_only(
            sequence_no=sequence_no,
            source_cue_no=source_cue_no,
            target_cue_no=target_cue_no,
        )

    def apply_recipe_cue_only(
        self,
        *,
        sequence_no: int,
        source_cue_no: CueNumber | float | int | str,
        target_cue_no: CueNumber | float | int | str,
    ) -> dict[str, object]:
        return self._require_bridge().apply_recipe_cue_only(
            sequence_no=sequence_no,
            source_cue_no=source_cue_no,
            target_cue_no=target_cue_no,
        )

    def copy_cue_with_status(
        self,
        *,
        sequence_no: int,
        source_cue_no: CueNumber | float | int | str,
        dest_cue_no: CueNumber | float | int | str,
    ) -> dict[str, object]:
        return self._require_bridge().copy_cue_with_status(
            sequence_no=sequence_no,
            source_cue_no=source_cue_no,
            dest_cue_no=dest_cue_no,
        )

    def preview_copy_cue_with_status(
        self,
        *,
        sequence_no: int,
        source_cue_no: CueNumber | float | int | str,
        dest_cue_no: CueNumber | float | int | str,
    ) -> dict[str, object]:
        return self._require_bridge().preview_copy_cue_with_status(
            sequence_no=sequence_no,
            source_cue_no=source_cue_no,
            dest_cue_no=dest_cue_no,
        )

    def ping(self) -> dict[str, object]:
        return self._require_bridge().ping()

    def get_version_info(self) -> dict[str, object]:
        return self._require_bridge().get_version_info()

    def get_plugin_health(self) -> dict[str, object]:
        return self._require_bridge().get_plugin_health()

    def get_connection_report(self) -> dict[str, object]:
        return self._require_bridge().get_connection_report()

    def hook_track(self, track_coord: str) -> bool:
        return self._require_bridge().hook_track(track_coord)

    def set_hook_failure(self, track_coord: str, *, should_fail: bool = True) -> None:
        self._require_server().set_hook_failure(track_coord, should_fail=should_fail)

    def set_tracks(self, tracks) -> None:
        self._require_server().set_tracks(tracks)
        self._require_bridge().invalidate()

    def set_track_events(self, events_by_track) -> None:
        self._require_server().set_track_events(events_by_track)
        self._require_bridge().invalidate()

    def set_sequences(self, sequences) -> None:
        self._require_server().set_sequences(sequences)
        self._require_bridge().invalidate()

    def set_current_song_label(self, song_label: str | None) -> None:
        self._require_server().set_current_song_label(song_label)
        self._require_bridge().invalidate()

    def set_plugin_health(
        self,
        *,
        ez_version: str | None = None,
        ez_build: str | None = None,
        hitmaker_loaded: bool | None = None,
        hitmaker_version: str | None = None,
        hitmaker_build: str | None = None,
        hitmaker_supports_event_type_create: bool | None = None,
        hitmaker_supports_go_hit: bool | None = None,
        hitmaker_supports_version_info: bool | None = None,
    ) -> None:
        self._require_server().set_plugin_health(
            ez_version=ez_version,
            ez_build=ez_build,
            hitmaker_loaded=hitmaker_loaded,
            hitmaker_version=hitmaker_version,
            hitmaker_build=hitmaker_build,
            hitmaker_supports_event_type_create=hitmaker_supports_event_type_create,
            hitmaker_supports_go_hit=hitmaker_supports_go_hit,
            hitmaker_supports_version_info=hitmaker_supports_version_info,
        )
        self._require_bridge().invalidate()

    def set_track_write_ready(self, track_coord: str, *, ready: bool) -> None:
        self._require_server().set_track_write_ready(track_coord, ready=ready)
        self._require_bridge().invalidate()

    def set_cmd_subtrack_create_blocked(self, track_coord: str, *, blocked: bool) -> None:
        self._require_server().set_cmd_subtrack_create_blocked(track_coord, blocked=blocked)
        self._require_bridge().invalidate()

    def set_clear_delay(self, track_coord: str, *, seconds: float) -> None:
        self._require_server().set_clear_delay(track_coord, seconds=seconds)
        self._require_bridge().invalidate()

    def set_drop_ping_reply_count(self, count: int) -> None:
        self._require_server().set_drop_ping_reply_count(count)
        self._require_bridge().invalidate()

    def assign_track_sequence(
        self,
        *,
        target_track_coord: str,
        sequence_no: int,
    ) -> None:
        self._require_bridge().assign_track_sequence(
            target_track_coord=target_track_coord,
            sequence_no=sequence_no,
        )

    def create_sequence_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot:
        return self._require_bridge().create_sequence_next_available(preferred_name=preferred_name)

    def create_sequence_in_current_song_range(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot:
        return self._require_bridge().create_sequence_in_current_song_range(
            preferred_name=preferred_name
        )

    def create_sequence_for_event_type(
        self,
        *,
        event_type: str,
        sequence_type: str = "go_hit",
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot:
        del sequence_type
        resolved_name = preferred_name or str(event_type or "Hit").strip().title()
        return self._require_bridge().create_sequence_in_current_song_range(
            preferred_name=resolved_name,
        )

    def create_timecode_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3TimecodeSnapshot:
        return self._require_bridge().create_timecode_next_available(preferred_name=preferred_name)

    def create_track_group_next_available(
        self,
        *,
        timecode_no: int,
        preferred_name: str | None = None,
    ) -> MA3TrackGroupSnapshot:
        return self._require_bridge().create_track_group_next_available(
            timecode_no=timecode_no,
            preferred_name=preferred_name,
        )

    def create_track(
        self,
        *,
        timecode_no: int,
        track_group_no: int,
        preferred_name: str | None = None,
    ) -> MA3TrackSnapshot:
        return self._require_bridge().create_track(
            timecode_no=timecode_no,
            track_group_no=track_group_no,
            preferred_name=preferred_name,
        )

    def create_static_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        value_command: str,
    ) -> MA3PresetSnapshot:
        return self._require_bridge().create_static_preset(
            preset_type_no=preset_type_no,
            preset_no=preset_no,
            store_mode=store_mode,
            preset_name=preset_name,
            selection_command=selection_command,
            value_command=value_command,
        )

    def create_phaser_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        step_preset_refs: tuple[str, ...] | list[str] | tuple[list[str], ...] | list[list[str]],
        speed_bpm: float | None = None,
    ) -> MA3PresetSnapshot:
        return self._require_bridge().create_phaser_preset(
            preset_type_no=preset_type_no,
            preset_no=preset_no,
            store_mode=store_mode,
            preset_name=preset_name,
            selection_command=selection_command,
            step_preset_refs=step_preset_refs,
            speed_bpm=speed_bpm,
        )

    def create_recipe_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        source_preset_ref: str,
        selection_mode: str = "Strict",
    ) -> MA3PresetSnapshot:
        return self._require_bridge().create_recipe_preset(
            preset_type_no=preset_type_no,
            preset_no=preset_no,
            store_mode=store_mode,
            preset_name=preset_name,
            selection_command=selection_command,
            source_preset_ref=source_preset_ref,
            selection_mode=selection_mode,
        )

    def edit_static_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        value_command: str,
    ) -> MA3PresetSnapshot:
        return self._require_bridge().edit_static_preset(
            preset_type_no=preset_type_no,
            preset_no=preset_no,
            store_mode=store_mode,
            preset_name=preset_name,
            selection_command=selection_command,
            value_command=value_command,
        )

    def edit_phaser_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        step_preset_refs: tuple[str, ...] | list[str] | tuple[list[str], ...] | list[list[str]],
        speed_bpm: float | None = None,
    ) -> MA3PresetSnapshot:
        return self._require_bridge().edit_phaser_preset(
            preset_type_no=preset_type_no,
            preset_no=preset_no,
            store_mode=store_mode,
            preset_name=preset_name,
            selection_command=selection_command,
            step_preset_refs=step_preset_refs,
            speed_bpm=speed_bpm,
        )

    def edit_recipe_preset(
        self,
        *,
        preset_type_no: int,
        preset_no: int,
        store_mode: str,
        preset_name: str,
        selection_command: str,
        source_preset_ref: str,
        selection_mode: str = "Strict",
    ) -> MA3PresetSnapshot:
        return self._require_bridge().edit_recipe_preset(
            preset_type_no=preset_type_no,
            preset_no=preset_no,
            store_mode=store_mode,
            preset_name=preset_name,
            selection_command=selection_command,
            source_preset_ref=source_preset_ref,
            selection_mode=selection_mode,
        )

    def prepare_track_for_events(self, *, target_track_coord: str) -> None:
        self._require_bridge().prepare_track_for_events(target_track_coord=target_track_coord)

    def send_console_command(self, command: str) -> None:
        self._require_bridge().send_console_command(command)

    def reload_plugins(self) -> None:
        self._require_bridge().reload_plugins()

    def apply_push_transfer(
        self,
        *,
        target_track_coord: str,
        ma3_channel_no: int | None = None,
        selected_events,
        transfer_mode: str = "merge",
        start_offset_seconds: float = 0.0,
    ) -> None:
        self._require_bridge().apply_push_transfer(
            target_track_coord=target_track_coord,
            ma3_channel_no=ma3_channel_no,
            selected_events=selected_events,
            transfer_mode=transfer_mode,
            start_offset_seconds=start_offset_seconds,
        )
        self.emit(
            "transfer.push_applied",
            {
                "target_track_coord": str(target_track_coord),
                "ma3_channel_no": ma3_channel_no,
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


def _unwrap_lua_command(command: str) -> str:
    text = str(command or "").strip()
    if text.startswith('Lua "') and text.endswith('"'):
        inner = text[5:-1]
        return inner.replace('\\"', '"').replace("\\\\", "\\")
    if text.startswith("Lua '") and text.endswith("'"):
        inner = text[5:-1]
        return inner.replace("\\'", "'").replace("\\\\", "\\")
    return text


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
    if text.lower() in {"nil", "null"}:
        return None
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

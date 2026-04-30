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
            ),
            MA3EventSnapshot(
                event_id="ma3_evt_2",
                label="Cue 2",
                start=2.0,
                end=2.5,
                cmd="Go+ Cue 2",
                cue_number=2,
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
        self._current_song_label: str | None = "Song A"
        self._timecode_name_by_no: dict[int, str] = {1: "Song A"}
        self._track_group_name_by_key: dict[tuple[int, int], str] = {(1, 2): "Group 2"}
        self._time_range_idx_by_coord: dict[str, int] = {}
        self._cmd_subtrack_ready_by_coord: dict[str, bool] = {}
        self._cmd_subtrack_create_blocked: set[str] = set()
        self._clear_delay_seconds_by_coord: dict[str, float] = {}
        self._hooked_tracks: set[str] = set()

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
                track
                for track in tracks
                if parse_track_coord(track.coord)[0] == int(timecode_no)
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
                name=self._track_group_name_by_key.get((int(timecode_no), group_no), f"Group {group_no}"),
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

    def set_current_song_label(self, song_label: str | None) -> None:
        self._current_song_label = None if song_label in {None, ""} else str(song_label)
        if self._current_song_label:
            self._timecode_name_by_no[1] = self._current_song_label

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
        name, args = _parse_command(command)
        handler = getattr(self, f"_handle_{name}", None)
        if callable(handler):
            handler(*args)

    def _handle_SetTarget(self, host: str, port: int) -> None:
        self._target = str(host), int(port)

    def _handle_Ping(self) -> None:
        self._send_message("connection", "ping", {"status": "ok"})

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
    ) -> None:
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
        explicit_label = str(event_name or cue_label or "").strip()
        label = explicit_label or _event_label_from_command(command, cue_number)
        snapshot = MA3EventSnapshot(
            event_id=next_id,
            label=label,
            start=float(start),
            end=float(start),
            cmd=command,
            cue_number=cue_number,
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
            self._send_track_error(tc_no, tg_no, track_no, f"Sequence {int(sequence_no)} does not exist")
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
            self._send_track_error(tc_no, tg_no, track_no, "Track prep could not create CmdSubTrack")
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

    def list_timecodes(self) -> list[MA3TimecodeSnapshot]:
        return self._require_bridge().list_timecodes()

    def list_track_groups(self, *, timecode_no: int) -> list[MA3TrackGroupSnapshot]:
        return self._require_bridge().list_track_groups(timecode_no=timecode_no)

    def list_track_events(self, track_coord: str) -> list[MA3EventSnapshot]:
        return self._require_bridge().list_track_events(track_coord)

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[MA3SequenceSnapshot]:
        return self._require_bridge().list_sequences(start_no=start_no, end_no=end_no)

    def get_current_song_sequence_range(self) -> MA3SequenceRangeSnapshot | None:
        return self._require_bridge().get_current_song_sequence_range()

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

    def set_track_write_ready(self, track_coord: str, *, ready: bool) -> None:
        self._require_server().set_track_write_ready(track_coord, ready=ready)
        self._require_bridge().invalidate()

    def set_cmd_subtrack_create_blocked(self, track_coord: str, *, blocked: bool) -> None:
        self._require_server().set_cmd_subtrack_create_blocked(track_coord, blocked=blocked)
        self._require_bridge().invalidate()

    def set_clear_delay(self, track_coord: str, *, seconds: float) -> None:
        self._require_server().set_clear_delay(track_coord, seconds=seconds)
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
        return self._require_bridge().create_sequence_next_available(
            preferred_name=preferred_name
        )

    def create_sequence_in_current_song_range(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot:
        return self._require_bridge().create_sequence_in_current_song_range(
            preferred_name=preferred_name
        )

    def create_timecode_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3TimecodeSnapshot:
        return self._require_bridge().create_timecode_next_available(
            preferred_name=preferred_name
        )

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

    def prepare_track_for_events(self, *, target_track_coord: str) -> None:
        self._require_bridge().prepare_track_for_events(target_track_coord=target_track_coord)

    def apply_push_transfer(
        self,
        *,
        target_track_coord: str,
        selected_events,
        transfer_mode: str = "merge",
        start_offset_seconds: float = 0.0,
    ) -> None:
        self._require_bridge().apply_push_transfer(
            target_track_coord=target_track_coord,
            selected_events=selected_events,
            transfer_mode=transfer_mode,
            start_offset_seconds=start_offset_seconds,
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

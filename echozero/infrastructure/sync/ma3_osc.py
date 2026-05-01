"""MA3 OSC transport bridge for EchoZero sync operations.
Exists to encode, decode, and coordinate live MA3 OSC command and state traffic.
Connects the MA3 sync adapter boundary to reusable OSC transport services.
"""

from __future__ import annotations

from threading import Condition, Lock
from time import monotonic, sleep
from typing import Any, Protocol

from echozero.application.shared.cue_numbers import CueNumber, cue_number_text, parse_positive_cue_number
from echozero.infrastructure.osc import (
    OscInboundMessage,
    OscReceiveServer,
    OscReceiveServiceConfig,
    OscSendTransport,
)
from echozero.infrastructure.sync.ma3_adapter import (
    MA3EventSnapshot,
    MA3SequenceRangeSnapshot,
    MA3SequenceSnapshot,
    MA3TimecodeSnapshot,
    MA3TrackGroupSnapshot,
    MA3TrackSnapshot,
    coerce_event_snapshot,
    transport_event_label,
)
from echozero.infrastructure.sync.ma3_protocol import (
    MA3OSCMessage,
    _format_get_sequences_command,
    _format_lua_number,
    _format_lua_optional_int,
    _format_lua_string,
    _optional_float,
    _optional_int,
    encode_ma3_osc_payload,
    format_ma3_lua_command,
    format_track_coord,
    parse_ma3_osc_payload,
    parse_track_coord,
    resolve_ma3_target_host,
)

_TARGET_CONFIG_SETTLE_SECONDS = 0.25
_WRITE_ERROR_GRACE_SECONDS = 0.35
_CMD_SUBTRACK_RETRY_SETTLE_SECONDS = 0.2
_PUSH_WRITE_BATCH_SIZE = 200
_PUSH_WRITE_BATCH_SETTLE_SECONDS = 0.01


class MA3CommandTransport(Protocol):
    """Outbound command transport expected by the MA3 OSC bridge."""

    def send(self, command: str) -> None: ...

    def close(self) -> None: ...


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
        command_transport: OscSendTransport | MA3CommandTransport | None = None,
    ) -> None:
        self._listen_host = str(listen_host)
        self._listen_port = int(listen_port)
        self._listen_path = str(listen_path)
        self._timecode_no = int(timecode_no)
        self._response_timeout = max(0.05, float(response_timeout))
        self._command_transport = command_transport

        self._listener: OscReceiveServer | None = None
        self._lock = Lock()
        self._condition = Condition(self._lock)

        self._connected = False
        self._target_configured = False
        self._last_message_at: float | None = None
        self._messages: list[MA3OSCMessage] = []
        self._next_request_id = 1

        self._timecodes_by_number: dict[int, str] = {}
        self._trackgroups_by_timecode: dict[int, list[dict[str, object]]] = {}
        self._tracks_by_group: dict[tuple[int, int], list[MA3TrackSnapshot]] = {}
        self._tracks_by_coord: dict[str, MA3TrackSnapshot] = {}
        self._pending_track_requests: dict[int, tuple[int, int]] = {}
        self._track_chunks: dict[int, list[list[MA3TrackSnapshot]]] = {}
        self._events_by_coord: dict[str, list[MA3EventSnapshot]] = {}
        self._invalidated_event_coords: set[str] = set()
        self._pending_event_requests: dict[int, str] = {}
        self._event_chunks: dict[int, list[dict[str, object]]] = {}
        self._pending_sequence_requests: dict[int, tuple[int | None, int | None]] = {}
        self._sequence_chunks: dict[int, list[list[MA3SequenceSnapshot]]] = {}
        self._sequence_cues_by_sequence_no: dict[int, list[dict[str, object]]] = {}
        self._pending_sequence_cue_requests: dict[int, int] = {}
        self._sequence_cue_chunks: dict[int, list[list[dict[str, object]]]] = {}
        self._hooked_tracks: set[str] = set()
        self._sequences_by_number: dict[int, MA3SequenceSnapshot] = {}
        self._current_song_sequence_range: MA3SequenceRangeSnapshot | None = None
        self._plugin_health: dict[str, object] | None = None
        self._transport_updates: list[dict[str, object]] = []

    @property
    def is_running(self) -> bool:
        return self._listener is not None and self._listener.is_running

    @property
    def listener_endpoint(self) -> tuple[str, int]:
        if self._listener is None:
            raise RuntimeError("MA3 OSC bridge listener is not running")
        return self._listener.endpoint

    @property
    def messages(self) -> list[MA3OSCMessage]:
        with self._lock:
            return list(self._messages)

    def start(self) -> "MA3OSCBridge":
        if self._listener is not None:
            return self
        self._listener = OscReceiveServer(
            OscReceiveServiceConfig(
                host=self._listen_host,
                port=self._listen_port,
                path=self._listen_path,
            ),
            on_message=self._handle_osc_message,
            thread_name="echozero-ma3-osc-bridge",
        ).start()
        self._target_configured = False
        return self

    def stop(self) -> None:
        listener = self._listener
        self._listener = None
        self._connected = False
        self._target_configured = False

        if listener is None:
            return
        listener.stop()

    def reconfigure(
        self,
        *,
        listen_host: str,
        listen_port: int,
        listen_path: str | None = None,
        command_transport: OscSendTransport | MA3CommandTransport | None,
    ) -> None:
        should_restart = self.is_running
        self._listen_host = str(listen_host)
        self._listen_port = int(listen_port)
        if listen_path is not None:
            self._listen_path = str(listen_path)
        if command_transport is not None:
            if self._command_transport is not None and self._command_transport is not command_transport:
                self._command_transport.close()
            self._command_transport = command_transport
        else:
            self._command_transport = command_transport
        self._target_configured = False

        if not should_restart:
            return
        self.stop()
        self.start()

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
            self._send_command("EZ.Ping()")

    def on_ma3_disconnected(self) -> None:
        transport = self._command_transport
        if transport is not None:
            try:
                self._send_command("EZ.UnhookAll()")
            except Exception:
                pass
        self.stop()

    def get_status(self) -> dict[str, object]:
        endpoint = None
        if self._listener is not None and self._listener.is_running:
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

    def list_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[MA3TrackSnapshot]:
        self._ensure_listener()
        if track_group_no is not None and timecode_no is None:
            raise ValueError("list_tracks track_group_no requires timecode_no")
        with self._lock:
            if track_group_no is not None:
                has_requested_tracks = (int(timecode_no), int(track_group_no)) in self._tracks_by_group
            else:
                has_tracks = bool(self._tracks_by_coord)
                has_requested_tracks = (
                    has_tracks
                    if timecode_no is None
                    else any(
                        parse_track_coord(coord)[0] == int(timecode_no)
                        for coord in self._tracks_by_coord
                    )
                )
        if not has_requested_tracks:
            self.refresh_tracks(timecode_no=timecode_no, track_group_no=track_group_no)
        return self.list_tracks_cached(timecode_no=timecode_no, track_group_no=track_group_no)

    def list_timecodes(self) -> list[MA3TimecodeSnapshot]:
        self._ensure_listener()
        if self._command_transport is not None and not self._timecodes_cached():
            self._refresh_timecodes()
        return [
            MA3TimecodeSnapshot(number=tc_no, name=name)
            for tc_no, name in self._timecodes_cached()
        ]

    def list_track_groups(self, *, timecode_no: int) -> list[MA3TrackGroupSnapshot]:
        self._ensure_listener()
        requested_timecode_no = int(timecode_no)
        with self._lock:
            has_groups = requested_timecode_no in self._trackgroups_by_timecode
        if not has_groups and self._command_transport is not None:
            self.refresh_track_groups(timecode_no=requested_timecode_no)
        return self.list_track_groups_cached(timecode_no=requested_timecode_no)

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

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[MA3SequenceSnapshot]:
        if self._command_transport is None:
            return self.list_sequences_cached(start_no=start_no, end_no=end_no)

        request_id = self._next_request_token()
        with self._condition:
            self._pending_sequence_requests[request_id] = (start_no, end_no)
            self._sequence_chunks.pop(request_id, None)
            self._sequences_by_number.clear()
        self._ensure_command_ready()
        self._send_command(
            _format_get_sequences_command(
                start_no=start_no,
                end_no=end_no,
                request_id=request_id,
            )
        )
        self._wait_for(
            lambda request_id=request_id: request_id not in self._pending_sequence_requests,
            timeout=self._response_timeout,
            missing="Timed out waiting for MA3 sequences",
        )
        return self.list_sequences_cached(start_no=start_no, end_no=end_no)

    def list_sequences_cached(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[MA3SequenceSnapshot]:
        with self._lock:
            sequences = sorted(self._sequences_by_number.values(), key=lambda item: item.number)
        if start_no is not None:
            sequences = [sequence for sequence in sequences if sequence.number >= int(start_no)]
        if end_no is not None:
            sequences = [sequence for sequence in sequences if sequence.number <= int(end_no)]
        return sequences

    def list_sequence_cues(self, *, sequence_no: int) -> list[dict[str, object]]:
        requested_sequence_no = _optional_int(sequence_no)
        if requested_sequence_no is None or requested_sequence_no <= 0:
            return []
        if self._command_transport is None:
            with self._lock:
                return list(self._sequence_cues_by_sequence_no.get(int(requested_sequence_no), []))

        request_id = self._next_request_token()
        with self._condition:
            self._pending_sequence_cue_requests[request_id] = int(requested_sequence_no)
            self._sequence_cue_chunks.pop(request_id, None)
            self._sequence_cues_by_sequence_no.pop(int(requested_sequence_no), None)
        self._ensure_command_ready()
        self._send_command(f"EZ.GetSequenceCues({int(requested_sequence_no)}, {request_id})")
        self._wait_for(
            lambda request_id=request_id: request_id not in self._pending_sequence_cue_requests,
            timeout=self._response_timeout,
            missing=f"Timed out waiting for MA3 sequence cues for sequence {requested_sequence_no}",
        )
        with self._lock:
            return list(self._sequence_cues_by_sequence_no.get(int(requested_sequence_no), []))

    def get_current_song_sequence_range(self) -> MA3SequenceRangeSnapshot | None:
        if self._command_transport is None:
            with self._lock:
                return self._current_song_sequence_range

        self._ensure_command_ready()
        after_index = self._message_count()
        self._send_command("EZ.GetCurrentSongSequenceRange()")
        self._wait_for_message(
            after_index=after_index,
            predicate=lambda message: message.key == "sequence_range.current_song",
            timeout=self._response_timeout,
            missing="Timed out waiting for MA3 current-song sequence range",
        )
        with self._lock:
            return self._current_song_sequence_range

    def get_plugin_health(self) -> dict[str, object]:
        if self._command_transport is None:
            with self._lock:
                return dict(self._plugin_health or {})

        self._ensure_command_ready()
        after_index = self._message_count()
        self._send_command("EZ.GetPluginHealth()")
        message = self._wait_for_message(
            after_index=after_index,
            predicate=lambda message: message.key == "plugin.health",
            timeout=self._response_timeout,
            missing="Timed out waiting for MA3 plugin health snapshot",
        )
        with self._lock:
            self._plugin_health = dict(message.fields)
            return dict(self._plugin_health)

    def consume_transport_update(self) -> dict[str, object] | None:
        """Consume one queued MA3 transport update from inbound OSC state."""

        with self._lock:
            if not self._transport_updates:
                return None
            return dict(self._transport_updates.pop(0))

    def consume_latest_transport_update(self) -> dict[str, object] | None:
        """Consume the newest queued MA3 transport update and clear older ones."""

        with self._lock:
            if not self._transport_updates:
                return None
            latest = dict(self._transport_updates[-1])
            self._transport_updates.clear()
            return latest

    def assign_track_sequence(
        self,
        *,
        target_track_coord: str,
        sequence_no: int,
    ) -> None:
        coord = str(target_track_coord or "").strip()
        if not coord:
            raise ValueError("target_track_coord is required")
        if self._command_transport is None:
            raise RuntimeError("MA3 OSC bridge does not have an outbound command transport")

        tc_no, tg_no, track_no = parse_track_coord(coord)
        self._ensure_command_ready()
        after_index = self._message_count()
        self._send_command(
            f"EZ.AssignTrackSequence({tc_no}, {tg_no}, {track_no}, {int(sequence_no)})"
        )
        self._wait_for_track_result(
            coord=coord,
            after_index=after_index,
            success_keys={"track.assigned"},
            timeout=self._response_timeout,
            missing=f"Timed out assigning MA3 sequence {sequence_no} to {coord}",
        )

    def create_sequence_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot:
        return self._create_sequence(
            command_name="CreateSequenceNextAvailable",
            preferred_name=preferred_name,
        )

    def create_sequence_in_current_song_range(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot:
        return self._create_sequence(
            command_name="CreateSequenceInCurrentSongRange",
            preferred_name=preferred_name,
        )

    def create_sequence_for_event_type(
        self,
        *,
        event_type: str,
        sequence_type: str = "go_hit",
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot:
        normalized_event_type = str(event_type or "").strip()
        if not normalized_event_type:
            raise ValueError("event_type is required")
        normalized_sequence_type = str(sequence_type or "").strip().lower() or "go_hit"
        normalized_preferred_name = (
            None
            if preferred_name is None
            else str(preferred_name).strip() or None
        )
        if self._command_transport is None:
            raise RuntimeError("MA3 OSC bridge does not have an outbound command transport")

        current_song_range = self.get_current_song_sequence_range()
        start_no = None if current_song_range is None else int(current_song_range.start)
        end_no = None if current_song_range is None else int(current_song_range.end)
        known_sequences = self.list_sequences(start_no=start_no, end_no=end_no)
        known_by_number = {sequence.number: sequence for sequence in known_sequences}

        sequence_name_arg = "nil"
        if normalized_preferred_name is not None:
            sequence_name_arg = _format_lua_string(normalized_preferred_name)
        command = (
            "HitMaker.create_sequence_for_event_type({"
            f"event_type={_format_lua_string(normalized_event_type)}, "
            f"sequence_type={_format_lua_string(normalized_sequence_type)}, "
            f"sequence_name={sequence_name_arg}, "
            f"name={sequence_name_arg}, "
            "skip_dialog=true, "
            "assign_to_timecode=false, "
            "clearFirst=false, "
            "doNotAssign=false, "
            "autoAssign=true, "
            "tapOnExec=false"
            "})"
        )
        self._ensure_command_ready()
        self._send_command(command)
        sleep(0.05)

        refreshed_sequences = self.list_sequences(start_no=start_no, end_no=end_no)
        created_sequences = [
            sequence
            for sequence in refreshed_sequences
            if sequence.number not in known_by_number
        ]
        if created_sequences:
            return sorted(created_sequences, key=lambda sequence: sequence.number)[-1]

        if normalized_preferred_name is not None:
            named_matches = [
                sequence
                for sequence in refreshed_sequences
                if sequence.name.strip().lower() == normalized_preferred_name.lower()
            ]
            if named_matches:
                return sorted(named_matches, key=lambda sequence: sequence.number)[-1]

        if start_no is not None and end_no is not None:
            refreshed_by_number = {sequence.number: sequence for sequence in refreshed_sequences}
            for sequence_no in range(start_no, end_no + 1):
                if sequence_no in known_by_number:
                    continue
                created_sequence = refreshed_by_number.get(sequence_no)
                if created_sequence is not None:
                    return created_sequence

        raise RuntimeError(
            "HitMaker sequence creation did not produce a discoverable sequence for "
            f"event type '{normalized_event_type}'"
        )

    def create_timecode_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3TimecodeSnapshot:
        if self._command_transport is None:
            raise RuntimeError("MA3 OSC bridge does not have an outbound command transport")

        known_timecodes = {tc_no for tc_no, _name in self._timecodes_cached()}
        self._ensure_command_ready()
        after_index = self._message_count()
        if preferred_name is None:
            self._send_command("EZ.CreateTimecode()")
        else:
            self._send_command(f"EZ.CreateTimecode({_format_lua_string(preferred_name)})")
        message = self._wait_for_timecode_result(
            after_index=after_index,
            timeout=self._response_timeout,
            missing="Timed out waiting for MA3 timecode creation",
        )
        if message.key == "timecode.error":
            error_text = str(message.fields.get("error") or "").strip()
            raise RuntimeError(error_text or "MA3 timecode creation failed")

        created_no = _optional_int(message.fields.get("no"))
        if created_no is None:
            created_no = _optional_int(message.fields.get("number"))
        if created_no is None:
            created_no = _optional_int(message.fields.get("tc"))

        timecodes = self._refresh_timecodes()
        if created_no is None:
            for tc_no, _name in timecodes:
                if tc_no not in known_timecodes:
                    created_no = tc_no
                    break
        if created_no is None:
            raise RuntimeError("MA3 timecode creation did not return a valid timecode number")

        resolved_name = next(
            (name for tc_no, name in timecodes if tc_no == int(created_no)),
            None,
        )
        if resolved_name in {None, ""}:
            field_name = message.fields.get("name")
            resolved_name = None if field_name in {None, ""} else str(field_name)
        return MA3TimecodeSnapshot(number=int(created_no), name=resolved_name)

    def create_track_group_next_available(
        self,
        *,
        timecode_no: int,
        preferred_name: str | None = None,
    ) -> MA3TrackGroupSnapshot:
        requested_timecode_no = _optional_int(timecode_no)
        if requested_timecode_no is None or requested_timecode_no < 1:
            raise ValueError("timecode_no is required")
        if self._command_transport is None:
            raise RuntimeError("MA3 OSC bridge does not have an outbound command transport")

        known_groups = {
            group.number
            for group in self.list_track_groups_cached(timecode_no=requested_timecode_no)
        }
        self._ensure_command_ready()
        after_index = self._message_count()
        if preferred_name is None:
            self._send_command(f"EZ.CreateTrackGroup({requested_timecode_no})")
        else:
            self._send_command(
                "EZ.CreateTrackGroup({tc}, {name})".format(
                    tc=requested_timecode_no,
                    name=_format_lua_string(preferred_name),
                )
            )
        message = self._wait_for_track_group_result(
            timecode_no=requested_timecode_no,
            after_index=after_index,
            timeout=self._response_timeout,
            missing=(
                f"Timed out waiting for MA3 track-group creation in TC{requested_timecode_no}"
            ),
        )
        if message.key == "trackgroup.error":
            error_text = str(message.fields.get("error") or "").strip()
            raise RuntimeError(error_text or "MA3 track-group creation failed")

        created_no = _optional_int(message.fields.get("tg"))
        if created_no is None:
            created_no = _optional_int(message.fields.get("no"))
        if created_no is None:
            created_no = _optional_int(message.fields.get("number"))

        groups = self.refresh_track_groups(timecode_no=requested_timecode_no)
        if created_no is None:
            for group in groups:
                if group.number not in known_groups:
                    created_no = group.number
                    break
        if created_no is None:
            raise RuntimeError("MA3 track-group creation did not return a valid group number")

        for group in groups:
            if group.number == int(created_no):
                return group

        field_name = message.fields.get("name")
        return MA3TrackGroupSnapshot(
            number=int(created_no),
            name=str(field_name or f"Group {int(created_no)}"),
            track_count=None,
        )

    def create_track(
        self,
        *,
        timecode_no: int,
        track_group_no: int,
        preferred_name: str | None = None,
    ) -> MA3TrackSnapshot:
        requested_timecode_no = _optional_int(timecode_no)
        requested_track_group_no = _optional_int(track_group_no)
        if requested_timecode_no is None or requested_timecode_no < 1:
            raise ValueError("timecode_no is required")
        if requested_track_group_no is None or requested_track_group_no < 1:
            raise ValueError("track_group_no is required")
        if self._command_transport is None:
            raise RuntimeError("MA3 OSC bridge does not have an outbound command transport")

        existing_tracks = self.list_tracks_cached(
            timecode_no=requested_timecode_no,
            track_group_no=requested_track_group_no,
        )
        existing_coords = {track.coord for track in existing_tracks}
        desired_name = str(preferred_name or "").strip()
        if not desired_name:
            desired_name = f"Track {len(existing_tracks) + 1}"

        self._ensure_command_ready()
        after_index = self._message_count()
        self._send_command(
            "EZ.CreateTrack({tc}, {tg}, {name})".format(
                tc=requested_timecode_no,
                tg=requested_track_group_no,
                name=_format_lua_string(desired_name),
            )
        )
        message = self._wait_for_track_create_result(
            timecode_no=requested_timecode_no,
            track_group_no=requested_track_group_no,
            after_index=after_index,
            timeout=self._response_timeout,
            missing=(
                "Timed out waiting for MA3 track creation in "
                f"TC{requested_timecode_no}.TG{requested_track_group_no}"
            ),
        )
        if message.key == "track.error":
            error_text = str(message.fields.get("error") or "").strip()
            raise RuntimeError(error_text or "MA3 track creation failed")

        tracks = self.refresh_tracks(
            timecode_no=requested_timecode_no,
            track_group_no=requested_track_group_no,
        )
        created_track_no = _optional_int(message.fields.get("track"))
        if created_track_no is not None:
            created_coord = format_track_coord(
                requested_timecode_no,
                requested_track_group_no,
                created_track_no,
            )
            for track in tracks:
                if track.coord == created_coord:
                    return track

        created_tracks = [track for track in tracks if track.coord not in existing_coords]
        if created_tracks:
            return sorted(
                created_tracks,
                key=lambda track: parse_track_coord(track.coord)[2],
            )[-1]

        target_name = str(message.fields.get("name") or desired_name).strip().lower()
        named_tracks = [
            track
            for track in tracks
            if str(track.name or "").strip().lower() == target_name
        ]
        if named_tracks:
            return sorted(named_tracks, key=lambda track: parse_track_coord(track.coord)[2])[-1]

        if tracks:
            return sorted(tracks, key=lambda track: parse_track_coord(track.coord)[2])[-1]
        raise RuntimeError("MA3 track creation succeeded but no track was returned")

    def prepare_track_for_events(self, *, target_track_coord: str) -> None:
        coord = str(target_track_coord or "").strip()
        if not coord:
            raise ValueError("target_track_coord is required")
        if self._command_transport is None:
            raise RuntimeError("MA3 OSC bridge does not have an outbound command transport")

        tc_no, tg_no, track_no = parse_track_coord(coord)
        self._ensure_command_ready()
        after_index = self._message_count()
        self._send_command(f"EZ.PrepareTrackForEvents({tc_no}, {tg_no}, {track_no})")
        message = self._wait_for_track_result(
            coord=coord,
            after_index=after_index,
            success_keys={"track.prepared"},
            timeout=self._response_timeout,
            missing=f"Timed out preparing MA3 track {coord} for events",
        )
        if message.fields.get("cmd_subtrack_ready") is False:
            raise RuntimeError(f"MA3 track {coord} did not report cmd-subtrack readiness")

    def send_console_command(self, command: str) -> None:
        text = str(command or "").strip()
        if not text:
            raise ValueError("command is required")
        transport = self._command_transport
        if transport is None:
            raise RuntimeError("MA3 OSC bridge does not have an outbound command transport")
        transport.send(text)

    def reload_plugins(self) -> None:
        self.send_console_command("RP")

    def _create_sequence(
        self,
        *,
        command_name: str,
        preferred_name: str | None,
    ) -> MA3SequenceSnapshot:
        if self._command_transport is None:
            raise RuntimeError("MA3 OSC bridge does not have an outbound command transport")

        self._ensure_command_ready()
        after_index = self._message_count()
        if preferred_name is None:
            self._send_command(f"EZ.{command_name}()")
        else:
            self._send_command(f"EZ.{command_name}({_format_lua_string(preferred_name)})")
        message = self._wait_for_message(
            after_index=after_index,
            predicate=lambda item: item.key == "sequence.created",
            timeout=self._response_timeout,
            missing=f"Timed out waiting for MA3 {command_name}",
        )
        return self._sequence_snapshot_from_fields(message.fields)

    def refresh_track_groups(self, *, timecode_no: int) -> list[MA3TrackGroupSnapshot]:
        if self._command_transport is None:
            return self.list_track_groups_cached(timecode_no=timecode_no)

        requested_timecode_no = int(timecode_no)
        self._ensure_command_ready()
        self._refresh_timecodes()
        with self._lock:
            self._trackgroups_by_timecode.pop(requested_timecode_no, None)
        after_index = self._message_count()
        self._send_command(f"EZ.GetTrackGroups({requested_timecode_no})")
        self._wait_for_message(
            after_index=after_index,
            predicate=lambda message, tc_no=requested_timecode_no: (
                message.key == "trackgroups.list"
                and int(message.fields.get("tc") or 0) == tc_no
            ),
            timeout=self._response_timeout,
            missing=f"Timed out waiting for MA3 track groups in TC{requested_timecode_no}",
        )
        return self.list_track_groups_cached(timecode_no=requested_timecode_no)

    def refresh_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[MA3TrackSnapshot]:
        if self._command_transport is None:
            return self.list_tracks_cached(timecode_no=timecode_no, track_group_no=track_group_no)

        requested_track_group_no = _optional_int(track_group_no)
        if requested_track_group_no is not None:
            requested_timecode_no = _optional_int(timecode_no)
            if requested_timecode_no is None:
                raise ValueError("refresh_tracks track_group_no requires timecode_no")
            self.refresh_track_groups(timecode_no=requested_timecode_no)
            with self._lock:
                self._tracks_by_group.pop((requested_timecode_no, requested_track_group_no), None)
                self._rebuild_track_index_locked()
            request_id = self._next_request_token()
            with self._condition:
                self._pending_track_requests[request_id] = (
                    requested_timecode_no,
                    requested_track_group_no,
                )
                self._track_chunks.pop(request_id, None)
            self._send_command(
                f"EZ.GetTracks({requested_timecode_no}, {requested_track_group_no}, {request_id})"
            )
            self._wait_for(
                lambda request_id=request_id: request_id not in self._pending_track_requests,
                timeout=self._response_timeout,
                missing=(
                    "Timed out waiting for MA3 tracks in "
                    f"TC{requested_timecode_no}.TG{requested_track_group_no}"
                ),
            )
            return self.list_tracks_cached(
                timecode_no=requested_timecode_no,
                track_group_no=requested_track_group_no,
            )

        self._ensure_command_ready()
        timecodes = self._refresh_timecodes()
        requested_timecode_no = _optional_int(timecode_no)
        if requested_timecode_no is not None:
            timecodes = [
                (tc_no, name)
                for tc_no, name in timecodes
                if tc_no == requested_timecode_no
            ] or [
                (requested_timecode_no, self._timecodes_by_number.get(requested_timecode_no))
            ]
        valid_timecodes = {tc_no for tc_no, _name in timecodes}
        with self._lock:
            if requested_timecode_no is None:
                for tc_no in [tc for tc in self._trackgroups_by_timecode if tc not in valid_timecodes]:
                    self._trackgroups_by_timecode.pop(tc_no, None)
                for key in [key for key in self._tracks_by_group if key[0] not in valid_timecodes]:
                    self._tracks_by_group.pop(key, None)
            else:
                self._trackgroups_by_timecode.pop(requested_timecode_no, None)
                for key in [key for key in self._tracks_by_group if key[0] == requested_timecode_no]:
                    self._tracks_by_group.pop(key, None)
            self._rebuild_track_index_locked()

        for tc_no, _timecode_name in timecodes:
            after_index = self._message_count()
            self._send_command(f"EZ.GetTrackGroups({tc_no})")
            self._wait_for_message(
                after_index=after_index,
                predicate=lambda message, tc_no=tc_no: (
                    message.key == "trackgroups.list"
                    and int(message.fields.get("tc") or 0) == tc_no
                ),
                timeout=self._response_timeout,
                missing=f"Timed out waiting for MA3 track groups in TC{tc_no}",
            )
            with self._lock:
                trackgroups = list(self._trackgroups_by_timecode.get(tc_no, []))
            groups_with_tracks = [
                trackgroup
                for trackgroup in trackgroups
                if int(trackgroup.get("track_count") or 0) > 0
            ]
            for trackgroup in groups_with_tracks:
                tg_no = int(trackgroup.get("no") or 0)
                request_id = self._next_request_token()
                with self._condition:
                    self._pending_track_requests[request_id] = (tc_no, tg_no)
                    self._track_chunks.pop(request_id, None)
                self._send_command(f"EZ.GetTracks({tc_no}, {tg_no}, {request_id})")
                self._wait_for(
                    lambda request_id=request_id: request_id not in self._pending_track_requests,
                    timeout=self._response_timeout,
                    missing=f"Timed out waiting for MA3 tracks in TC{tc_no}.TG{tg_no}",
                )
        return self.list_tracks_cached(timecode_no=requested_timecode_no)

    def list_tracks_cached(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[MA3TrackSnapshot]:
        with self._lock:
            if track_group_no is not None and timecode_no is not None:
                return list(self._tracks_by_group.get((int(timecode_no), int(track_group_no)), []))
            coords = self._sorted_track_coords_locked()
            if timecode_no is not None:
                coords = [
                    coord
                    for coord in coords
                    if _coord_timecode_no(coord) == int(timecode_no)
                ]
            return [self._tracks_by_coord[coord] for coord in coords]

    def list_track_groups_cached(self, *, timecode_no: int) -> list[MA3TrackGroupSnapshot]:
        with self._lock:
            groups = list(self._trackgroups_by_timecode.get(int(timecode_no), []))
        return [
            MA3TrackGroupSnapshot(
                number=int(group.get("no") or 0),
                name=str(group.get("name") or ""),
                track_count=_optional_int(group.get("track_count")),
            )
            for group in groups
            if int(group.get("no") or 0) > 0
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
        self._send_command(f"EZ.GetEvents({tc_no}, {tg_no}, {track_no}, {request_id})")
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
        ma3_channel_no: int | None = None,
        selected_events,
        transfer_mode: str = "merge",
        start_offset_seconds: float = 0.0,
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
        resolved_channel_no = 1
        if ma3_channel_no is not None:
            try:
                resolved_channel_no = max(1, int(ma3_channel_no))
            except (TypeError, ValueError):
                resolved_channel_no = 1
        start_offset = _optional_float(start_offset_seconds)
        resolved_start_offset = 0.0 if start_offset is None else float(start_offset)

        existing_events = self.list_track_events(coord)
        existing_fingerprints = (
            {self._event_fingerprint(event) for event in existing_events}
            if mode == "merge"
            else set()
        )
        self._ensure_command_ready()

        if mode == "overwrite" and existing_events:
            clear_start_index = self._message_count()
            self._send_command(f"EZ.ClearTrack({tc_no}, {tg_no}, {track_no})")
            self._wait_for_track_result(
                coord=coord,
                after_index=clear_start_index,
                success_keys={"track.cleared"},
                timeout=self._response_timeout,
                missing=f"Timed out clearing MA3 track {coord} before overwrite",
            )

        first_event = True
        sent_event_count = 0
        for raw_event in selected_events or []:
            snapshot = coerce_event_snapshot(raw_event)
            command = self._event_command_text(snapshot)
            event_name = transport_event_label(raw_event)
            event_name = str(event_name or "").strip() or None
            cue_number = snapshot.cue_number
            cue_label = self._event_cue_name(snapshot, event_name=event_name)
            if mode == "merge":
                fingerprint = self._event_fingerprint(snapshot)
                if fingerprint in existing_fingerprints:
                    continue
                existing_fingerprints.add(fingerprint)

            start = max(0.0, float(snapshot.start or 0.0) + resolved_start_offset)
            send_start_index = self._message_count()
            self._send_add_event_command(
                tc_no=tc_no,
                tg_no=tg_no,
                track_no=track_no,
                start=start,
                channel_no=resolved_channel_no,
                command=command,
                event_name=event_name,
                cue_number=cue_number,
                cue_label=cue_label,
            )
            if first_event:
                self._recover_missing_cmd_subtrack_if_needed(
                    coord=coord,
                    tc_no=tc_no,
                    tg_no=tg_no,
                    track_no=track_no,
                    start=start,
                    channel_no=resolved_channel_no,
                    command=command,
                    event_name=event_name,
                    cue_number=cue_number,
                    cue_label=cue_label,
                    after_index=send_start_index,
                )
                first_event = False
            sent_event_count += 1
            if (
                _PUSH_WRITE_BATCH_SIZE > 0
                and sent_event_count % _PUSH_WRITE_BATCH_SIZE == 0
            ):
                sleep(_PUSH_WRITE_BATCH_SETTLE_SECONDS)

        self.refresh_track_events(coord)

    def _send_add_event_command(
        self,
        *,
        tc_no: int,
        tg_no: int,
        track_no: int,
        start: float,
        command: str,
        channel_no: int = 1,
        event_name: str | None = None,
        cue_number: CueNumber | None = None,
        cue_label: str | None = None,
    ) -> None:
        event_name_arg = (
            "nil"
            if event_name is None
            else _format_lua_string(event_name)
        )
        cue_number_arg = "nil" if cue_number is None else str(cue_number_text(cue_number))
        cue_label_arg = (
            "nil"
            if cue_label is None
            else _format_lua_string(cue_label)
        )
        channel_no_arg = str(max(1, int(channel_no)))
        self._send_command(
            "EZ.AddEvent({tc}, {tg}, {track}, {start}, {command}, {event_name}, {cue_number}, {cue_label}, {channel_no})".format(
                tc=tc_no,
                tg=tg_no,
                track=track_no,
                start=_format_lua_number(start),
                command=_format_lua_string(command),
                event_name=event_name_arg,
                cue_number=cue_number_arg,
                cue_label=cue_label_arg,
                channel_no=channel_no_arg,
            )
        )

    def _recover_missing_cmd_subtrack_if_needed(
        self,
        *,
        coord: str,
        tc_no: int,
        tg_no: int,
        track_no: int,
        start: float,
        channel_no: int,
        command: str,
        event_name: str | None,
        cue_number: CueNumber | None,
        cue_label: str | None,
        after_index: int,
    ) -> None:
        error = self._wait_for_track_event_error(
            coord,
            after_index=after_index,
            timeout=_WRITE_ERROR_GRACE_SECONDS,
        )
        if error is None:
            return

        error_text = str(error.fields.get("error") or "").strip()
        if "No CmdSubTrack" not in error_text:
            raise RuntimeError(f"MA3 AddEvent failed for {coord}: {error_text or error.raw_payload}")

        self._send_command(f"EZ.CreateCmdSubTrack({tc_no}, {tg_no}, {track_no}, 1)")
        sleep(_CMD_SUBTRACK_RETRY_SETTLE_SECONDS)

        retry_start_index = self._message_count()
        self._send_add_event_command(
            tc_no=tc_no,
            tg_no=tg_no,
            track_no=track_no,
            start=start,
            channel_no=channel_no,
            command=command,
            event_name=event_name,
            cue_number=cue_number,
            cue_label=cue_label,
        )
        retry_error = self._wait_for_track_event_error(
            coord,
            after_index=retry_start_index,
            timeout=_WRITE_ERROR_GRACE_SECONDS,
        )
        if retry_error is None:
            return

        retry_text = str(retry_error.fields.get("error") or "").strip()
        if "No CmdSubTrack" in retry_text:
            raise RuntimeError(
                "MA3 track {coord} is not write-ready. Assign a sequence to the track in MA3, "
                "then retry the push.".format(coord=coord)
            )
        raise RuntimeError(
            f"MA3 AddEvent failed for {coord} after CreateCmdSubTrack: "
            f"{retry_text or retry_error.raw_payload}"
        )

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
        self._send_command(f"EZ.HookTrack({tc_no}, {tg_no}, {track_no})")
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
        self._send_command(f"EZ.UnhookTrack({tc_no}, {tg_no}, {track_no})")
        with self._lock:
            self._hooked_tracks.discard(coord)
        return True

    def unhook_all(self) -> None:
        transport = self._command_transport
        if transport is None:
            return
        self._ensure_command_ready()
        self._send_command("EZ.UnhookAll()")
        with self._lock:
            self._hooked_tracks.clear()

    def invalidate(self) -> None:
        with self._condition:
            self._timecodes_by_number.clear()
            self._trackgroups_by_timecode.clear()
            self._tracks_by_group.clear()
            self._tracks_by_coord.clear()
            self._events_by_coord.clear()
            self._invalidated_event_coords.clear()
            self._pending_track_requests.clear()
            self._track_chunks.clear()
            self._pending_event_requests.clear()
            self._event_chunks.clear()
            self._pending_sequence_requests.clear()
            self._sequence_chunks.clear()
            self._sequence_cues_by_sequence_no.clear()
            self._pending_sequence_cue_requests.clear()
            self._sequence_cue_chunks.clear()
            self._sequences_by_number.clear()
            self._current_song_sequence_range = None
            self._condition.notify_all()

    def _ensure_listener(self) -> None:
        if self._listener is None:
            self.start()

    def _ensure_command_ready(self) -> None:
        self._ensure_listener()
        self._ensure_target_configured()

    def _ensure_target_configured(self) -> None:
        if self._command_transport is None or self._target_configured:
            return
        host, port = self.listener_endpoint
        target_host = resolve_ma3_target_host(
            listen_host=host,
            command_host=self._command_host_hint(),
        )
        self._send_command(f"EZ.SetTarget({_format_lua_string(target_host)}, {int(port)})")
        # Real MA3 applies a new OSC target asynchronously; the next response-bound
        # command can race the update unless we give the console a brief settle window.
        sleep(_TARGET_CONFIG_SETTLE_SECONDS)
        self._target_configured = True

    def _command_host_hint(self) -> str | None:
        transport = self._command_transport
        if transport is None:
            return None
        endpoint = getattr(transport, "endpoint", None)
        if not isinstance(endpoint, tuple) or not endpoint:
            return None
        host = str(endpoint[0] or "").strip()
        return host or None

    def _send_command(self, command: str) -> None:
        transport = self._command_transport
        if transport is None:
            return
        transport.send(format_ma3_lua_command(command))

    def _message_count(self) -> int:
        with self._lock:
            return len(self._messages)

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

    def _wait_for_message(
        self,
        *,
        after_index: int,
        predicate,
        timeout: float,
        missing: str,
    ) -> MA3OSCMessage:
        deadline = monotonic() + max(0.01, timeout)
        with self._condition:
            while True:
                for message in self._messages[after_index:]:
                    if predicate(message):
                        return message
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(missing)
                self._condition.wait(timeout=remaining)

    def _handle_osc_message(self, inbound: OscInboundMessage) -> None:
        payload = inbound.first_text_arg()
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

        if message_type == "timecodes" and change == "list":
            raw_timecodes = fields.get("timecodes") or []
            self._timecodes_by_number = {
                int(raw_timecode.get("no") or 0): str(raw_timecode.get("name") or "")
                for raw_timecode in (raw_timecodes if isinstance(raw_timecodes, list) else [])
                if int(raw_timecode.get("no") or 0) > 0
            }
            self._tracks_by_group = {
                key: [self._annotate_track_timecode_locked(track) for track in tracks]
                for key, tracks in self._tracks_by_group.items()
            }
            self._rebuild_track_index_locked()
            return

        if message_type == "timecode" and change in {"created", "exists"}:
            tc_no = _optional_int(fields.get("no"))
            if tc_no is None:
                tc_no = _optional_int(fields.get("number"))
            if tc_no is None:
                tc_no = _optional_int(fields.get("tc"))
            if tc_no is not None and tc_no > 0:
                raw_name = fields.get("name")
                self._timecodes_by_number[int(tc_no)] = str(raw_name or "")
            return

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

        if message_type == "trackgroup" and change in {"created", "exists", "deleted"}:
            tc_no = int(fields.get("tc") or self._timecode_no)
            self._trackgroups_by_timecode.pop(tc_no, None)
            for key in [key for key in self._tracks_by_group if key[0] == tc_no]:
                self._tracks_by_group.pop(key, None)
            self._rebuild_track_index_locked()
            return

        if message_type == "tracks" and change == "list":
            self._ingest_tracks_list_locked(fields)
            return

        if message_type == "sequences" and change == "list":
            self._ingest_sequences_list_locked(fields)
            return

        if message_type == "sequence_cues" and change == "list":
            self._ingest_sequence_cues_list_locked(fields)
            return

        if message_type == "sequence_cues" and change == "error":
            request_id = _optional_int(fields.get("request_id"))
            sequence_no = _optional_int(fields.get("sequence_no"))
            if request_id is not None:
                pending_sequence_no = self._pending_sequence_cue_requests.pop(request_id, None)
                self._sequence_cue_chunks.pop(request_id, None)
                if sequence_no is None:
                    sequence_no = pending_sequence_no
            if sequence_no is not None:
                self._sequence_cues_by_sequence_no.setdefault(int(sequence_no), [])
            return

        if message_type == "sequence_range" and change == "current_song":
            self._current_song_sequence_range = self._sequence_range_snapshot_from_fields(fields)
            return

        if message_type == "plugin" and change == "health":
            self._plugin_health = dict(fields)
            return

        if message_type == "transport":
            update = _coerce_transport_update_payload(message)
            if update is not None:
                self._transport_updates.append(update)
            return

        if message_type == "sequence" and change == "created":
            sequence = self._sequence_snapshot_from_fields(fields)
            if sequence.number > 0:
                self._sequences_by_number[sequence.number] = sequence
                self._sequence_cues_by_sequence_no.pop(sequence.number, None)
            return

        if message_type == "events" and change == "list":
            self._ingest_events_list_locked(fields)
            return

        if message_type == "track" and change == "assigned":
            coord = self._coord_from_fields(fields)
            if coord is not None:
                self._update_track_sequence_locked(coord, _optional_int(fields.get("seq")))
            return

        if message_type == "track" and change == "prepared":
            coord = self._coord_from_fields(fields)
            if coord is not None:
                self._update_track_sequence_locked(coord, _optional_int(fields.get("seq")))
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
                number=track.number,
                timecode_name=track.timecode_name,
                note=track.note,
                event_count=len(normalized),
                sequence_no=track.sequence_no,
            )
            tc_no, tg_no, _track_no = parse_track_coord(coord)
            group_key = (tc_no, tg_no)
            if group_key in self._tracks_by_group:
                self._tracks_by_group[group_key] = [
                    self._tracks_by_coord.get(existing.coord, existing)
                    for existing in self._tracks_by_group[group_key]
                ]

    def _ingest_tracks_list_locked(self, fields: dict[str, object]) -> None:
        tc_no = int(fields.get("tc") or self._timecode_no)
        tg_no = int(fields.get("tg") or 0)
        request_id = _optional_int(fields.get("request_id"))
        raw_tracks = fields.get("tracks") or []
        normalized: list[MA3TrackSnapshot] = []
        for raw_track in (raw_tracks if isinstance(raw_tracks, list) else []):
            track_no = int(raw_track.get("no") or 0)
            coord = format_track_coord(tc_no, tg_no, track_no)
            normalized.append(
                MA3TrackSnapshot(
                    coord=coord,
                    name=str(raw_track.get("name") or ""),
                    number=track_no,
                    timecode_name=self._timecodes_by_number.get(tc_no),
                    note=None if raw_track.get("note") is None else str(raw_track.get("note")),
                    event_count=_optional_int(raw_track.get("event_count")),
                    sequence_no=_optional_int(raw_track.get("sequence_no")),
                )
            )

        total_chunks = _optional_int(fields.get("total_chunks")) or 1
        chunk_index = _optional_int(fields.get("chunk_index")) or 1
        if request_id is not None and total_chunks > 1:
            chunk_store = self._track_chunks.setdefault(request_id, [])
            if chunk_index <= len(chunk_store):
                chunk_store[chunk_index - 1] = normalized
            else:
                while len(chunk_store) < chunk_index - 1:
                    chunk_store.append([])
                chunk_store.append(normalized)
            if len(chunk_store) < total_chunks or any(chunk is None for chunk in chunk_store):
                return
            combined: list[MA3TrackSnapshot] = []
            for chunk in chunk_store:
                combined.extend(chunk)
            normalized = combined
            self._track_chunks.pop(request_id, None)

        self._tracks_by_group[(tc_no, tg_no)] = normalized
        self._rebuild_track_index_locked()
        if request_id is not None:
            self._pending_track_requests.pop(request_id, None)

    def _ingest_sequences_list_locked(self, fields: dict[str, object]) -> None:
        request_id = _optional_int(fields.get("request_id"))
        raw_sequences = fields.get("sequences") or []
        normalized = [
            self._sequence_snapshot_from_fields(raw_sequence)
            for raw_sequence in (raw_sequences if isinstance(raw_sequences, list) else [])
        ]

        total_chunks = _optional_int(fields.get("total_chunks")) or 1
        chunk_index = _optional_int(fields.get("chunk_index")) or 1
        if request_id is not None and total_chunks > 1:
            chunk_store = self._sequence_chunks.setdefault(request_id, [])
            if chunk_index <= len(chunk_store):
                chunk_store[chunk_index - 1] = normalized
            else:
                while len(chunk_store) < chunk_index - 1:
                    chunk_store.append([])
                chunk_store.append(normalized)
            if len(chunk_store) < total_chunks or any(chunk is None for chunk in chunk_store):
                return
            combined: list[MA3SequenceSnapshot] = []
            for chunk in chunk_store:
                combined.extend(chunk)
            normalized = combined
            self._sequence_chunks.pop(request_id, None)

        self._sequences_by_number = {
            sequence.number: sequence
            for sequence in normalized
            if sequence.number > 0
        }
        if request_id is not None:
            self._pending_sequence_requests.pop(request_id, None)

    def _ingest_sequence_cues_list_locked(self, fields: dict[str, object]) -> None:
        request_id = _optional_int(fields.get("request_id"))
        sequence_no = _optional_int(fields.get("sequence_no"))
        if sequence_no is None:
            sequence_no = _optional_int(fields.get("seq"))
        if sequence_no is None and request_id is not None:
            sequence_no = self._pending_sequence_cue_requests.get(request_id)
        if sequence_no is None:
            return

        raw_cues = fields.get("cues") or []
        normalized: list[dict[str, object]] = []
        for raw_cue in (raw_cues if isinstance(raw_cues, list) else []):
            if not isinstance(raw_cue, dict):
                continue
            cue_number = parse_positive_cue_number(
                raw_cue.get("cue_number")
                or raw_cue.get("cue_no")
                or raw_cue.get("no")
                or raw_cue.get("cueNo")
                or raw_cue.get("cue_ref")
                or raw_cue.get("cueRef")
            )
            if cue_number is None:
                continue
            cue_ref = cue_number_text(cue_number) or str(cue_number)
            normalized.append(
                {
                    "sequence_no": int(sequence_no),
                    "cue_number": cue_number,
                    "cue_ref": cue_ref,
                    "name": str(
                        raw_cue.get("name")
                        or raw_cue.get("cue_name")
                        or raw_cue.get("cueName")
                        or ""
                    ).strip(),
                }
            )

        total_chunks = _optional_int(fields.get("total_chunks")) or 1
        chunk_index = _optional_int(fields.get("chunk_index")) or 1
        if request_id is not None and total_chunks > 1:
            chunk_store = self._sequence_cue_chunks.setdefault(request_id, [])
            if chunk_index <= len(chunk_store):
                chunk_store[chunk_index - 1] = normalized
            else:
                while len(chunk_store) < chunk_index - 1:
                    chunk_store.append([])
                chunk_store.append(normalized)
            if len(chunk_store) < total_chunks or any(chunk is None for chunk in chunk_store):
                return
            combined: list[dict[str, object]] = []
            for chunk in chunk_store:
                combined.extend(chunk)
            normalized = combined
            self._sequence_cue_chunks.pop(request_id, None)

        self._sequence_cues_by_sequence_no[int(sequence_no)] = normalized
        if request_id is not None:
            self._pending_sequence_cue_requests.pop(request_id, None)

    def _rebuild_track_index_locked(self) -> None:
        self._tracks_by_coord = {}
        for tracks in self._tracks_by_group.values():
            for track in tracks:
                self._tracks_by_coord[track.coord] = self._annotate_track_timecode_locked(track)

    def _annotate_track_timecode_locked(self, track: MA3TrackSnapshot) -> MA3TrackSnapshot:
        try:
            tc_no, _tg_no, _track_no = parse_track_coord(track.coord)
        except ValueError:
            return track
        timecode_name = self._timecodes_by_number.get(tc_no)
        if track.timecode_name == timecode_name:
            return track
        return MA3TrackSnapshot(
            coord=track.coord,
            name=track.name,
            number=track.number,
            timecode_name=timecode_name,
            note=track.note,
            event_count=track.event_count,
            sequence_no=track.sequence_no,
        )

    def _update_track_sequence_locked(self, coord: str, sequence_no: int | None) -> None:
        track = self._tracks_by_coord.get(coord)
        if track is None:
            return
        updated = MA3TrackSnapshot(
            coord=track.coord,
            name=track.name,
            number=track.number,
            timecode_name=track.timecode_name,
            note=track.note,
            event_count=track.event_count,
            sequence_no=sequence_no,
        )
        self._tracks_by_coord[coord] = updated
        tc_no, tg_no, _track_no = parse_track_coord(coord)
        group_key = (tc_no, tg_no)
        if group_key in self._tracks_by_group:
            self._tracks_by_group[group_key] = [
                updated if existing.coord == coord else existing
                for existing in self._tracks_by_group[group_key]
            ]

    @staticmethod
    def _sequence_snapshot_from_fields(fields: dict[str, object]) -> MA3SequenceSnapshot:
        number = _optional_int(fields.get("number"))
        if number is None:
            number = _optional_int(fields.get("no"))
        cue_count = _optional_int(fields.get("cue_count"))
        if cue_count is None:
            cue_count = _optional_int(fields.get("cueCount"))
        return MA3SequenceSnapshot(
            number=int(number or 0),
            name=str(fields.get("name") or ""),
            cue_count=cue_count,
        )

    @staticmethod
    def _sequence_range_snapshot_from_fields(
        fields: dict[str, object],
    ) -> MA3SequenceRangeSnapshot | None:
        start = _optional_int(fields.get("start"))
        end = _optional_int(fields.get("end"))
        if start is None or end is None:
            return None
        song_label = fields.get("song_label")
        if song_label in {None, ""}:
            song_label = fields.get("songLabel")
        return MA3SequenceRangeSnapshot(
            song_label=None if song_label in {None, ""} else str(song_label),
            start=start,
            end=end,
        )

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
            cue_number=snapshot.cue_number,
            cue_ref=snapshot.cue_ref,
            color=snapshot.color,
            notes=snapshot.notes,
            payload_ref=snapshot.payload_ref,
        )

    @staticmethod
    def _event_command_text(event: MA3EventSnapshot) -> str:
        command = str(event.cmd or "").strip()
        if command:
            return command
        cue_number = cue_number_text(event.cue_number)
        if cue_number is not None:
            return f"Go+ Cue {cue_number}"
        label = str(event.label or "").strip()
        return label or "Event"

    @staticmethod
    def _event_cue_name(
        event: MA3EventSnapshot,
        *,
        event_name: str | None,
    ) -> str | None:
        """Choose the MA3 cue name payload while preserving readable operator labels."""

        label = str(event.label or "").strip()
        if label:
            return label
        return event_name

    @classmethod
    def _event_fingerprint(cls, event: MA3EventSnapshot) -> tuple[float | None, str]:
        start = None if event.start is None else round(float(event.start), 6)
        cue_number = cue_number_text(event.cue_number)
        if cue_number is not None:
            cue_ref = str(event.cue_ref or "").strip()
            if cue_ref:
                return (start, f"cue:{cue_number}:{cue_ref.casefold()}")
            return (start, f"cue:{cue_number}")
        command = " ".join(cls._event_command_text(event).lower().split())
        return (start, command)

    def _refresh_timecodes(self) -> list[tuple[int, str | None]]:
        if self._command_transport is None:
            return self._timecodes_cached()

        self._ensure_command_ready()
        after_index = self._message_count()
        self._send_command("EZ.GetTimecodes()")
        try:
            self._wait_for_message(
                after_index=after_index,
                predicate=lambda message: message.key == "timecodes.list",
                timeout=self._response_timeout,
                missing="Timed out waiting for MA3 timecodes",
            )
        except TimeoutError:
            cached = self._timecodes_cached()
            if cached:
                return cached
            return [(self._timecode_no, None)]
        cached = self._timecodes_cached()
        return cached if cached else [(self._timecode_no, None)]

    def _timecodes_cached(self) -> list[tuple[int, str | None]]:
        with self._lock:
            return [
                (tc_no, self._timecodes_by_number.get(tc_no) or None)
                for tc_no in sorted(self._timecodes_by_number)
            ]

    def _sorted_track_coords_locked(self) -> list[str]:
        def _sort_key(coord: str) -> tuple[int, int, int, str]:
            try:
                tc_no, tg_no, track_no = parse_track_coord(coord)
            except ValueError:
                return (self._timecode_no, 0, 0, coord)
            return (tc_no, tg_no, track_no, coord)

        return sorted(self._tracks_by_coord, key=_sort_key)

    def _wait_for_timecode_result(
        self,
        *,
        after_index: int,
        timeout: float,
        missing: str,
    ) -> MA3OSCMessage:
        deadline = monotonic() + max(0.01, timeout)
        with self._condition:
            while True:
                for message in self._messages[after_index:]:
                    if message.message_type != "timecode":
                        continue
                    if message.change in {"created", "exists", "error"}:
                        return message
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(missing)
                self._condition.wait(timeout=remaining)

    def _wait_for_track_group_result(
        self,
        *,
        timecode_no: int,
        after_index: int,
        timeout: float,
        missing: str,
    ) -> MA3OSCMessage:
        deadline = monotonic() + max(0.01, timeout)
        with self._condition:
            while True:
                for message in self._messages[after_index:]:
                    if message.message_type != "trackgroup":
                        continue
                    if message.change not in {"created", "exists", "error"}:
                        continue
                    message_timecode_no = _optional_int(message.fields.get("tc"))
                    if message_timecode_no != int(timecode_no):
                        continue
                    return message
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(missing)
                self._condition.wait(timeout=remaining)

    def _wait_for_track_create_result(
        self,
        *,
        timecode_no: int,
        track_group_no: int,
        after_index: int,
        timeout: float,
        missing: str,
    ) -> MA3OSCMessage:
        deadline = monotonic() + max(0.01, timeout)
        with self._condition:
            while True:
                for message in self._messages[after_index:]:
                    if message.message_type != "track":
                        continue
                    if message.change not in {"created", "exists", "error"}:
                        continue
                    message_timecode_no = _optional_int(message.fields.get("tc"))
                    message_track_group_no = _optional_int(message.fields.get("tg"))
                    if message_timecode_no != int(timecode_no):
                        continue
                    if message_track_group_no != int(track_group_no):
                        continue
                    return message
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(missing)
                self._condition.wait(timeout=remaining)

    def _wait_for_track_event_error(
        self,
        coord: str,
        *,
        after_index: int,
        timeout: float,
    ) -> MA3OSCMessage | None:
        deadline = monotonic() + max(0.01, timeout)
        with self._condition:
            while True:
                for message in self._messages[after_index:]:
                    if (
                        message.message_type == "event"
                        and message.change == "error"
                        and self._coord_from_fields(message.fields) == coord
                    ):
                        return message
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def _wait_for_track_result(
        self,
        *,
        coord: str,
        after_index: int,
        success_keys: set[str],
        timeout: float,
        missing: str,
    ) -> MA3OSCMessage:
        deadline = monotonic() + max(0.01, timeout)
        with self._condition:
            while True:
                for message in self._messages[after_index:]:
                    message_coord = self._coord_from_fields(message.fields)
                    if message_coord != coord:
                        continue
                    if message.key in success_keys:
                        return message
                    if message.key == "track.error":
                        error_text = str(message.fields.get("error") or "").strip()
                        raise RuntimeError(
                            error_text or f"MA3 track operation failed for {coord}"
                        )
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(missing)
                self._condition.wait(timeout=remaining)

def _coord_timecode_no(coord: str) -> int | None:
    try:
        tc_no, _tg_no, _track_no = parse_track_coord(coord)
    except ValueError:
        return None
    return tc_no


def _value(raw: Any, key: str) -> Any:
    if isinstance(raw, dict):
        return raw.get(key)
    return getattr(raw, key, None)


def _coerce_transport_update_payload(message: MA3OSCMessage) -> dict[str, object] | None:
    fields = dict(message.fields)
    playhead_seconds = _first_transport_float(
        fields,
        keys=("to_seconds", "position", "playhead", "seconds"),
    )
    is_playing = _first_transport_bool(
        fields,
        keys=("is_playing", "playing"),
    )
    if is_playing is None:
        is_playing = _transport_state_to_bool(fields.get("state"))

    if playhead_seconds is None and is_playing is None:
        return None

    payload: dict[str, object] = {
        "change": str(message.change or ""),
        "fields": fields,
    }
    if playhead_seconds is not None:
        payload["playhead_seconds"] = playhead_seconds
    if is_playing is not None:
        payload["is_playing"] = bool(is_playing)
    return payload


def _first_transport_float(fields: dict[str, object], *, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _optional_float(fields.get(key))
        if value is not None:
            return max(0.0, float(value))
    return None


def _first_transport_bool(fields: dict[str, object], *, keys: tuple[str, ...]) -> bool | None:
    for key in keys:
        if key in fields:
            parsed = _coerce_bool(fields.get(key))
            if parsed is not None:
                return parsed
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _transport_state_to_bool(value: object) -> bool | None:
    state = str(value or "").strip().lower()
    if state in {"play", "playing", "run", "running", "go"}:
        return True
    if state in {"pause", "paused", "stop", "stopped"}:
        return False
    return None

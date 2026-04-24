"""Concrete SyncService adapters for application/runtime wiring."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, Protocol

from echozero.application.shared.enums import SyncMode
from echozero.application.sync.models import SyncState
from echozero.application.sync.service import SyncService
from echozero.application.transport.models import TransportState
from echozero.infrastructure.sync.ma3_adapter import (
    event_snapshot_payload,
    sequence_range_snapshot_payload,
    sequence_snapshot_payload,
    timecode_snapshot_payload,
    trackgroup_snapshot_payload,
    track_snapshot_payload,
)


class MA3SyncBridge(Protocol):
    """Minimal bridge contract expected from MA3/show-manager runtime layer."""

    def on_ma3_connected(self) -> None: ...

    def on_ma3_disconnected(self) -> None: ...

    def list_tracks(self, *, timecode_no: int | None = None) -> list[object]: ...

    def list_timecodes(self) -> list[object]: ...

    def list_track_groups(self, *, timecode_no: int) -> list[object]: ...

    def list_track_events(self, track_coord: str) -> list[object]: ...

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[object]: ...

    def get_current_song_sequence_range(self) -> object | None: ...

    def assign_track_sequence(
        self,
        *,
        target_track_coord: str,
        sequence_no: int,
    ) -> None: ...

    def create_sequence_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> object: ...

    def create_sequence_in_current_song_range(
        self,
        *,
        preferred_name: str | None = None,
    ) -> object: ...

    def create_timecode_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> object: ...

    def create_track_group_next_available(
        self,
        *,
        timecode_no: int,
        preferred_name: str | None = None,
    ) -> object: ...

    def create_track(
        self,
        *,
        timecode_no: int,
        track_group_no: int,
        preferred_name: str | None = None,
    ) -> object: ...

    def prepare_track_for_events(self, *, target_track_coord: str) -> None: ...

    def send_console_command(self, command: str) -> None: ...

    def reload_plugins(self) -> None: ...

    def apply_push_transfer(
        self,
        *,
        target_track_coord: str,
        selected_events: list[object],
        transfer_mode: str = "merge",
    ) -> None: ...


class InMemorySyncService(SyncService):
    """Simple stateful implementation used by demos/tests."""

    def __init__(self, state: SyncState | None = None):
        self._state = state or SyncState()

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._state.mode = mode
        return self._state

    def connect(self) -> SyncState:
        self._state.connected = True
        self._state.health = "healthy"
        return self._state

    def disconnect(self) -> SyncState:
        self._state.connected = False
        self._state.mode = SyncMode.NONE
        self._state.health = "offline"
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        if not self._state.connected:
            return transport
        if self._state.offset_ms == 0.0:
            return transport

        shifted = max(0.0, transport.playhead + (self._state.offset_ms / 1000.0))
        return replace(transport, playhead=shifted)


class MA3SyncAdapter(SyncService):
    """App-layer SyncService adapter over MA3/show-manager bridge callbacks."""

    def __init__(
        self,
        bridge: MA3SyncBridge,
        *,
        target_ref: str = "show_manager",
        state: SyncState | None = None,
    ):
        self._bridge = bridge
        self._state = state or SyncState(mode=SyncMode.MA3, connected=False, target_ref=target_ref)

    def get_state(self) -> SyncState:
        return self._state

    def set_mode(self, mode: SyncMode) -> SyncState:
        self._state.mode = mode
        return self._state

    def connect(self) -> SyncState:
        if self._state.mode == SyncMode.NONE:
            self._state.mode = SyncMode.MA3

        try:
            self._bridge.on_ma3_connected()
        except Exception:
            self._state.connected = False
            self._state.health = "error"
            raise

        self._state.connected = True
        self._state.health = "healthy"
        return self._state

    def disconnect(self) -> SyncState:
        try:
            self._bridge.on_ma3_disconnected()
        finally:
            self._state.connected = False
            self._state.mode = SyncMode.NONE
            self._state.health = "offline"
        return self._state

    def align_transport(self, transport: TransportState) -> TransportState:
        if not self._state.connected:
            return transport
        if self._state.offset_ms == 0.0:
            return transport

        shifted = max(0.0, transport.playhead + (self._state.offset_ms / 1000.0))
        return replace(transport, playhead=shifted)

    def list_push_track_options(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[dict[str, object]]:
        return self._list_bridge_tracks(
            timecode_no=timecode_no,
            track_group_no=track_group_no,
        )

    def list_pull_track_options(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[dict[str, object]]:
        return self._list_bridge_tracks(
            timecode_no=timecode_no,
            track_group_no=track_group_no,
        )

    def list_timecodes(self) -> list[dict[str, object]]:
        method = self._bridge_method("list_timecodes", "list_ma3_timecodes")
        if method is None:
            return []
        return [
            timecode_snapshot_payload(raw_timecode)
            for raw_timecode in method() or []
        ]

    def list_track_groups(self, *, timecode_no: int) -> list[dict[str, object]]:
        method = self._bridge_method("list_track_groups", "list_ma3_track_groups")
        if method is None:
            return []
        return [
            trackgroup_snapshot_payload(raw_trackgroup)
            for raw_trackgroup in method(timecode_no=timecode_no) or []
        ]

    def list_pull_source_events(self, source_track_coord: str) -> list[dict[str, object]]:
        method = self._bridge_method(
            "list_track_events", "list_ma3_track_events", "get_available_ma3_events"
        )
        if method is None:
            return []
        return [
            event_snapshot_payload(raw_event) for raw_event in method(source_track_coord) or []
        ]

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[dict[str, object]]:
        method = self._bridge_method("list_sequences", "list_ma3_sequences")
        if method is None:
            return []
        return [
            sequence_snapshot_payload(raw_sequence)
            for raw_sequence in method(start_no=start_no, end_no=end_no) or []
        ]

    def get_current_song_sequence_range(self) -> dict[str, object] | None:
        method = self._bridge_method(
            "get_current_song_sequence_range",
            "get_ma3_current_song_sequence_range",
        )
        if method is None:
            return None
        return sequence_range_snapshot_payload(method())

    def assign_track_sequence(
        self,
        *,
        target_track_coord: str,
        sequence_no: int,
    ) -> None:
        method = self._bridge_method("assign_track_sequence")
        if method is None:
            raise RuntimeError("MA3 bridge does not support sequence assignment")
        method(target_track_coord=target_track_coord, sequence_no=sequence_no)

    def create_sequence_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> dict[str, object]:
        method = self._bridge_method("create_sequence_next_available")
        if method is None:
            raise RuntimeError("MA3 bridge does not support sequence creation")
        return sequence_snapshot_payload(method(preferred_name=preferred_name))

    def create_sequence_in_current_song_range(
        self,
        *,
        preferred_name: str | None = None,
    ) -> dict[str, object]:
        method = self._bridge_method("create_sequence_in_current_song_range")
        if method is None:
            raise RuntimeError("MA3 bridge does not support current-song sequence creation")
        return sequence_snapshot_payload(method(preferred_name=preferred_name))

    def create_timecode_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> dict[str, object]:
        method = self._bridge_method("create_timecode_next_available")
        if method is None:
            raise RuntimeError("MA3 bridge does not support timecode creation")
        return timecode_snapshot_payload(
            self._call_with_supported_kwargs(
                method,
                preferred_name=preferred_name,
                name=preferred_name,
            )
        )

    def create_track_group_next_available(
        self,
        *,
        timecode_no: int,
        preferred_name: str | None = None,
    ) -> dict[str, object]:
        method = self._bridge_method("create_track_group_next_available")
        if method is None:
            raise RuntimeError("MA3 bridge does not support track-group creation")
        return trackgroup_snapshot_payload(
            self._call_with_supported_kwargs(
                method,
                timecode_no=timecode_no,
                tc_no=timecode_no,
                preferred_name=preferred_name,
                name=preferred_name,
            )
        )

    def create_track(
        self,
        *,
        timecode_no: int,
        track_group_no: int,
        preferred_name: str | None = None,
    ) -> dict[str, object]:
        method = self._bridge_method("create_track")
        if method is None:
            raise RuntimeError("MA3 bridge does not support track creation")
        return track_snapshot_payload(
            self._call_with_supported_kwargs(
                method,
                timecode_no=timecode_no,
                tc_no=timecode_no,
                track_group_no=track_group_no,
                tg_no=track_group_no,
                preferred_name=preferred_name,
                track_name=preferred_name,
                name=preferred_name,
            )
        )

    def prepare_track_for_events(self, *, target_track_coord: str) -> None:
        method = self._bridge_method("prepare_track_for_events")
        if method is None:
            raise RuntimeError("MA3 bridge does not support track preparation")
        method(target_track_coord=target_track_coord)

    def send_console_command(self, command: str) -> None:
        method = self._bridge_method("send_console_command")
        if method is None:
            raise RuntimeError("MA3 bridge does not support raw console commands")
        method(command)

    def reload_plugins(self) -> None:
        method = self._bridge_method("reload_plugins")
        if method is not None:
            method()
            return
        self.send_console_command("RP")

    def apply_push_transfer(
        self,
        *,
        target_track_coord,
        selected_events,
        transfer_mode: str = "merge",
    ) -> None:
        callback = self._bridge_method("apply_push_transfer", "execute_push_transfer")
        if callback is None:
            raise RuntimeError("MA3 bridge does not support push apply")

        kwargs: dict[str, Any] = {
            "target_track_coord": target_track_coord,
            "selected_events": selected_events,
        }
        try:
            parameters = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "transfer_mode" in parameters:
            kwargs["transfer_mode"] = transfer_mode
        elif "mode" in parameters:
            kwargs["mode"] = transfer_mode
        callback(**kwargs)

    def _list_bridge_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[dict[str, object]]:
        method = self._bridge_method("list_tracks", "list_ma3_tracks", "get_available_ma3_tracks")
        if method is None:
            return []
        tracks = [
            track_snapshot_payload(raw_track)
            for raw_track in self._call_with_supported_kwargs(
                method,
                timecode_no=timecode_no,
                track_group_no=track_group_no,
            ) or []
        ]
        if timecode_no is not None:
            tracks = [
                track
                for track in tracks
                if _track_coord_timecode_no(track.get("coord")) == int(timecode_no)
            ]
        if track_group_no is not None:
            tracks = [
                track
                for track in tracks
                if _track_coord_track_group_no(track.get("coord")) == int(track_group_no)
            ]
        return tracks

    def _bridge_method(self, *names: str):
        for name in names:
            method = getattr(self._bridge, name, None)
            if callable(method):
                return method
        return None

    @staticmethod
    def _call_with_supported_kwargs(method, **kwargs):
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}
        if not parameters:
            return method()
        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in parameters and value is not None
        }
        return method(**supported_kwargs)


def _track_coord_timecode_no(raw_coord: object) -> int | None:
    coord = str(raw_coord or "").strip().lower()
    if not coord.startswith("tc"):
        return None
    tc_text = coord[2:].split("_", 1)[0]
    try:
        resolved = int(tc_text)
    except ValueError:
        return None
    return resolved if resolved > 0 else None


def _track_coord_track_group_no(raw_coord: object) -> int | None:
    coord = str(raw_coord or "").strip().lower()
    if "_tg" not in coord:
        return None
    group_text = coord.split("_tg", 1)[1].split("_", 1)[0]
    try:
        resolved = int(group_text)
    except ValueError:
        return None
    return resolved if resolved > 0 else None

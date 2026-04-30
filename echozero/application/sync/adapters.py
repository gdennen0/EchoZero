"""Concrete SyncService adapters for application/runtime wiring."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from echozero.application.shared.enums import SyncMode
from echozero.application.sync.ma3_push_service import (
    MA3CatalogService,
    MA3OperationRunner,
    MA3OperationSnapshot,
    MA3ProtocolClient,
    MA3PushService,
    _translate_ma3_error,
)
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
    """Strict MA3 bridge contract expected by the app sync adapter."""

    def on_ma3_connected(self) -> None: ...

    def on_ma3_disconnected(self) -> None: ...

    def list_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[object]: ...

    def refresh_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[object]: ...

    def list_timecodes(self) -> list[object]: ...

    def list_track_groups(self, *, timecode_no: int) -> list[object]: ...

    def refresh_track_groups(self, *, timecode_no: int) -> list[object]: ...

    def list_track_events(self, track_coord: str) -> list[object]: ...

    def refresh_track_events(self, track_coord: str) -> list[object]: ...

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
        start_offset_seconds: float = 0.0,
    ) -> None: ...


class _BridgeProtocolClient(MA3ProtocolClient):
    """Strict protocol client that delegates directly to the MA3 bridge."""

    def __init__(self, bridge: MA3SyncBridge):
        self._bridge = bridge

    def list_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[object]:
        return self._bridge.list_tracks(timecode_no=timecode_no, track_group_no=track_group_no)

    def refresh_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[object]:
        return self._bridge.refresh_tracks(timecode_no=timecode_no, track_group_no=track_group_no)

    def list_timecodes(self) -> list[object]:
        return self._bridge.list_timecodes()

    def list_track_groups(self, *, timecode_no: int) -> list[object]:
        return self._bridge.list_track_groups(timecode_no=timecode_no)

    def refresh_track_groups(self, *, timecode_no: int) -> list[object]:
        return self._bridge.refresh_track_groups(timecode_no=timecode_no)

    def list_track_events(self, track_coord: str) -> list[object]:
        return self._bridge.list_track_events(track_coord)

    def refresh_track_events(self, track_coord: str) -> list[object]:
        return self._bridge.refresh_track_events(track_coord)

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[object]:
        return self._bridge.list_sequences(start_no=start_no, end_no=end_no)

    def get_current_song_sequence_range(self) -> object | None:
        return self._bridge.get_current_song_sequence_range()

    def assign_track_sequence(self, *, target_track_coord: str, sequence_no: int) -> None:
        self._bridge.assign_track_sequence(
            target_track_coord=target_track_coord,
            sequence_no=sequence_no,
        )

    def create_sequence_next_available(self, *, preferred_name: str | None = None) -> object:
        return self._bridge.create_sequence_next_available(preferred_name=preferred_name)

    def create_sequence_in_current_song_range(
        self,
        *,
        preferred_name: str | None = None,
    ) -> object:
        return self._bridge.create_sequence_in_current_song_range(preferred_name=preferred_name)

    def create_timecode_next_available(self, *, preferred_name: str | None = None) -> object:
        return self._bridge.create_timecode_next_available(preferred_name=preferred_name)

    def create_track_group_next_available(
        self,
        *,
        timecode_no: int,
        preferred_name: str | None = None,
    ) -> object:
        return self._bridge.create_track_group_next_available(
            timecode_no=timecode_no,
            preferred_name=preferred_name,
        )

    def create_track(
        self,
        *,
        timecode_no: int,
        track_group_no: int,
        preferred_name: str | None = None,
    ) -> object:
        return self._bridge.create_track(
            timecode_no=timecode_no,
            track_group_no=track_group_no,
            preferred_name=preferred_name,
        )

    def prepare_track_for_events(self, *, target_track_coord: str) -> None:
        self._bridge.prepare_track_for_events(target_track_coord=target_track_coord)

    def send_console_command(self, command: str) -> None:
        self._bridge.send_console_command(command)

    def reload_plugins(self) -> None:
        try:
            self._bridge.reload_plugins()
        except AttributeError:
            self._bridge.send_console_command("RP")

    def apply_push_transfer(
        self,
        *,
        target_track_coord: str,
        selected_events: list[object],
        transfer_mode: str,
        start_offset_seconds: float,
    ) -> None:
        try:
            self._bridge.apply_push_transfer(
                target_track_coord=target_track_coord,
                selected_events=selected_events,
                transfer_mode=transfer_mode,
                start_offset_seconds=start_offset_seconds,
            )
        except AttributeError as exc:
            raise RuntimeError("MA3 bridge does not support push apply") from exc


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
    """App-layer SyncService adapter over strict MA3 protocol services."""

    def __init__(
        self,
        bridge: MA3SyncBridge,
        *,
        target_ref: str = "show_manager",
        state: SyncState | None = None,
    ):
        self._bridge = bridge
        self._state = state or SyncState(mode=SyncMode.MA3, connected=False, target_ref=target_ref)
        self._client = _BridgeProtocolClient(bridge)
        self._catalog = MA3CatalogService(self._client)
        self._push_service = MA3PushService(client=self._client, catalog=self._catalog)
        self._operations = MA3OperationRunner(max_workers=2)

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
        return [
            track_snapshot_payload(item)
            for item in self._catalog.list_tracks(
                timecode_no=timecode_no,
                track_group_no=track_group_no,
            )
        ]

    def list_pull_track_options(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[dict[str, object]]:
        return [
            track_snapshot_payload(item)
            for item in self._catalog.list_tracks(
                timecode_no=timecode_no,
                track_group_no=track_group_no,
            )
        ]

    def list_timecodes(self) -> list[dict[str, object]]:
        return [timecode_snapshot_payload(item) for item in self._catalog.list_timecodes()]

    def list_track_groups(self, *, timecode_no: int) -> list[dict[str, object]]:
        return [
            trackgroup_snapshot_payload(item)
            for item in self._catalog.list_track_groups(timecode_no=timecode_no)
        ]

    def list_pull_source_events(self, source_track_coord: str) -> list[dict[str, object]]:
        return [
            event_snapshot_payload(item)
            for item in self._catalog.list_track_events(source_track_coord)
        ]

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[dict[str, object]]:
        return [
            sequence_snapshot_payload(item)
            for item in self._catalog.list_sequences(start_no=start_no, end_no=end_no)
        ]

    def get_current_song_sequence_range(self) -> dict[str, object] | None:
        snapshot = self._catalog.get_current_song_sequence_range(refresh=True)
        if snapshot is None:
            return None
        return sequence_range_snapshot_payload(snapshot)

    def assign_track_sequence(
        self,
        *,
        target_track_coord: str,
        sequence_no: int,
    ) -> None:
        self._call(self._client.assign_track_sequence, target_track_coord=target_track_coord, sequence_no=sequence_no)
        self._catalog.refresh_tracks_scope(target_track_coord=target_track_coord)

    def create_sequence_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> dict[str, object]:
        raw = self._call(self._client.create_sequence_next_available, preferred_name=preferred_name)
        self._catalog.list_sequences(refresh=True)
        return sequence_snapshot_payload(raw)

    def create_sequence_in_current_song_range(
        self,
        *,
        preferred_name: str | None = None,
    ) -> dict[str, object]:
        raw = self._call(
            self._client.create_sequence_in_current_song_range,
            preferred_name=preferred_name,
        )
        self._catalog.list_sequences(refresh=True)
        return sequence_snapshot_payload(raw)

    def create_timecode_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> dict[str, object]:
        raw = self._call(self._client.create_timecode_next_available, preferred_name=preferred_name)
        self._catalog.list_timecodes(refresh=True)
        return timecode_snapshot_payload(raw)

    def create_track_group_next_available(
        self,
        *,
        timecode_no: int,
        preferred_name: str | None = None,
    ) -> dict[str, object]:
        raw = self._call(
            self._client.create_track_group_next_available,
            timecode_no=timecode_no,
            preferred_name=preferred_name,
        )
        self._catalog.list_track_groups(timecode_no=timecode_no, refresh=True)
        return trackgroup_snapshot_payload(raw)

    def create_track(
        self,
        *,
        timecode_no: int,
        track_group_no: int,
        preferred_name: str | None = None,
    ) -> dict[str, object]:
        raw = self._call(
            self._client.create_track,
            timecode_no=timecode_no,
            track_group_no=track_group_no,
            preferred_name=preferred_name,
        )
        created = track_snapshot_payload(raw)
        self._catalog.refresh_tracks_scope(
            target_track_coord=str(created.get("coord") or ""),
            timecode_no=timecode_no,
            track_group_no=track_group_no,
        )
        return created

    def prepare_track_for_events(self, *, target_track_coord: str) -> None:
        self._call(self._client.prepare_track_for_events, target_track_coord=target_track_coord)
        self._catalog.refresh_tracks_scope(target_track_coord=target_track_coord)

    def send_console_command(self, command: str) -> None:
        self._call(self._client.send_console_command, command)

    def reload_plugins(self) -> None:
        self._call(self._client.reload_plugins)

    def start_push(
        self,
        *,
        target_track_coord: str,
        selected_events: list[object],
        transfer_mode: str = "merge",
        start_offset_seconds: float = 0.0,
    ) -> str:
        return self._operations.start(
            kind="ma3.push",
            message=f"Sending {len(list(selected_events or []))} event(s) to MA3",
            callback=lambda: self._push_service.push(
                target_track_coord=target_track_coord,
                selected_events=list(selected_events or []),
                transfer_mode=transfer_mode,
                start_offset_seconds=start_offset_seconds,
            ),
        )

    def get_operation(self, operation_id: str) -> dict[str, object] | None:
        snapshot = self._operations.get(operation_id)
        if snapshot is None:
            return None
        return _operation_payload(snapshot)

    def cancel_operation(self, operation_id: str) -> bool:
        return self._operations.cancel(operation_id)

    def apply_push_transfer(
        self,
        *,
        target_track_coord,
        selected_events,
        transfer_mode: str = "merge",
        start_offset_seconds: float = 0.0,
    ) -> None:
        operation_id = self.start_push(
            target_track_coord=str(target_track_coord),
            selected_events=list(selected_events or []),
            transfer_mode=transfer_mode,
            start_offset_seconds=float(start_offset_seconds),
        )
        snapshot = self._operations.wait(operation_id, timeout=60.0)
        if snapshot.status == "error":
            raise RuntimeError(snapshot.error or snapshot.message or "MA3 push failed")

    def refresh_push_track_options(
        self,
        *,
        target_track_coord: str | None = None,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> None:
        self._catalog.refresh_tracks_scope(
            target_track_coord=target_track_coord,
            timecode_no=timecode_no,
            track_group_no=track_group_no,
        )

    @staticmethod
    def _call(callback, *args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except Exception as exc:
            raise _translate_ma3_error(exc) from exc


def _operation_payload(snapshot: MA3OperationSnapshot) -> dict[str, object]:
    return {
        "operation_id": snapshot.operation_id,
        "status": snapshot.status,
        "message": snapshot.message,
        "kind": snapshot.kind,
        "started_at": snapshot.started_at,
        "completed_at": snapshot.completed_at,
        "result": snapshot.result,
        "error": snapshot.error,
    }

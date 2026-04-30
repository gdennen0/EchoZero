"""MA3 push services: strict protocol boundary, async operations, and scoped catalog refresh.
Exists to keep MA3 push execution non-blocking for the UI while preserving deterministic sync semantics.
Connects the app sync adapter to MA3 bridge request/response operations without reflective fallbacks.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any, Callable, Literal, Protocol, cast

from echozero.infrastructure.sync.ma3_adapter import (
    MA3EventSnapshot,
    MA3SequenceRangeSnapshot,
    MA3SequenceSnapshot,
    MA3TimecodeSnapshot,
    MA3TrackGroupSnapshot,
    MA3TrackSnapshot,
    coerce_event_snapshot,
    coerce_sequence_range_snapshot,
    coerce_sequence_snapshot,
    coerce_timecode_snapshot,
    coerce_track_snapshot,
    coerce_trackgroup_snapshot,
)
from echozero.infrastructure.sync.ma3_protocol import parse_track_coord

MA3OperationStatus = Literal["running", "success", "error", "cancelled"]


class MA3SyncError(RuntimeError):
    """Base class for deterministic MA3 sync failures."""


class MA3LookupTimeoutError(MA3SyncError):
    """Raised when MA3 lookup/read operations time out."""


class MA3WriteTimeoutError(MA3SyncError):
    """Raised when MA3 write/update operations time out."""


class MA3ProtocolMismatchError(MA3SyncError):
    """Raised when MA3 payload/contract shape is invalid for required operation."""


class MA3ProtocolClient(Protocol):
    """Strict MA3 protocol surface with no reflective capability discovery."""

    def list_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[MA3TrackSnapshot]: ...

    def refresh_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[MA3TrackSnapshot]: ...

    def list_timecodes(self) -> list[MA3TimecodeSnapshot]: ...

    def list_track_groups(self, *, timecode_no: int) -> list[MA3TrackGroupSnapshot]: ...

    def refresh_track_groups(self, *, timecode_no: int) -> list[MA3TrackGroupSnapshot]: ...

    def list_track_events(self, track_coord: str) -> list[MA3EventSnapshot]: ...

    def refresh_track_events(self, track_coord: str) -> list[MA3EventSnapshot]: ...

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
    ) -> list[MA3SequenceSnapshot]: ...

    def get_current_song_sequence_range(self) -> MA3SequenceRangeSnapshot | None: ...

    def assign_track_sequence(self, *, target_track_coord: str, sequence_no: int) -> None: ...

    def create_sequence_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot: ...

    def create_sequence_in_current_song_range(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3SequenceSnapshot: ...

    def create_timecode_next_available(
        self,
        *,
        preferred_name: str | None = None,
    ) -> MA3TimecodeSnapshot: ...

    def create_track_group_next_available(
        self,
        *,
        timecode_no: int,
        preferred_name: str | None = None,
    ) -> MA3TrackGroupSnapshot: ...

    def create_track(
        self,
        *,
        timecode_no: int,
        track_group_no: int,
        preferred_name: str | None = None,
    ) -> MA3TrackSnapshot: ...

    def prepare_track_for_events(self, *, target_track_coord: str) -> None: ...

    def send_console_command(self, command: str) -> None: ...

    def reload_plugins(self) -> None: ...

    def apply_push_transfer(
        self,
        *,
        target_track_coord: str,
        selected_events: list[object],
        transfer_mode: str,
        start_offset_seconds: float,
    ) -> None: ...


@dataclass(slots=True, frozen=True)
class MA3OperationSnapshot:
    """Read model for one MA3 async operation."""

    operation_id: str
    status: MA3OperationStatus
    message: str
    kind: str
    started_at: float
    completed_at: float | None = None
    result: dict[str, object] | None = None
    error: str | None = None


@dataclass(slots=True)
class _MA3OperationRecord:
    operation_id: str
    kind: str
    started_at: float
    status: MA3OperationStatus = "running"
    message: str = ""
    completed_at: float | None = None
    result: dict[str, object] | None = None
    error: str | None = None
    cancellation_requested: bool = False
    future: Future[dict[str, object]] | None = None


class MA3OperationRunner:
    """Runs MA3 operations in a worker lane and stores queryable status snapshots."""

    def __init__(self, *, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="echozero-ma3-op",
        )
        self._lock = Lock()
        self._next_operation_no = 1
        self._operations: dict[str, _MA3OperationRecord] = {}

    def start(
        self,
        *,
        kind: str,
        message: str,
        callback: Callable[[], dict[str, object]],
    ) -> str:
        started_at = monotonic()
        with self._lock:
            operation_id = f"ma3-op-{self._next_operation_no}"
            self._next_operation_no += 1
            record = _MA3OperationRecord(
                operation_id=operation_id,
                kind=str(kind),
                started_at=started_at,
                message=str(message or "Running"),
            )
            self._operations[operation_id] = record

        future = self._executor.submit(self._run_callback, operation_id, callback)
        with self._lock:
            record.future = future
        return operation_id

    def get(self, operation_id: str) -> MA3OperationSnapshot | None:
        with self._lock:
            record = self._operations.get(str(operation_id or ""))
            if record is None:
                return None
            return MA3OperationSnapshot(
                operation_id=record.operation_id,
                status=record.status,
                message=record.message,
                kind=record.kind,
                started_at=record.started_at,
                completed_at=record.completed_at,
                result=record.result,
                error=record.error,
            )

    def cancel(self, operation_id: str) -> bool:
        with self._lock:
            record = self._operations.get(str(operation_id or ""))
            if record is None:
                return False
            record.cancellation_requested = True
            future = record.future

        if future is None:
            return False
        cancelled = future.cancel()
        if not cancelled:
            return False
        with self._lock:
            if record.status == "running":
                record.status = "cancelled"
                record.message = "Cancelled"
                record.completed_at = monotonic()
        return True

    def wait(self, operation_id: str, *, timeout: float) -> MA3OperationSnapshot:
        with self._lock:
            record = self._operations.get(str(operation_id or ""))
            if record is None:
                raise ValueError(f"Unknown MA3 operation: {operation_id}")
            future = record.future
        if future is not None:
            try:
                future.result(timeout=max(0.01, float(timeout)))
            except TimeoutError as exc:
                raise MA3WriteTimeoutError(
                    f"Timed out waiting for MA3 operation {operation_id}"
                ) from exc
        snapshot = self.get(operation_id)
        if snapshot is None:
            raise ValueError(f"Unknown MA3 operation: {operation_id}")
        return snapshot

    def _run_callback(
        self,
        operation_id: str,
        callback: Callable[[], dict[str, object]],
    ) -> dict[str, object]:
        try:
            result = callback()
        except Exception as exc:  # pragma: no cover - exercised by operation surface tests
            resolved_exc = _translate_ma3_error(exc)
            with self._lock:
                record = self._operations.get(operation_id)
                if record is not None and record.status == "running":
                    record.status = "error"
                    record.error = str(resolved_exc)
                    record.message = str(resolved_exc) or resolved_exc.__class__.__name__
                    record.completed_at = monotonic()
            raise resolved_exc

        with self._lock:
            record = self._operations.get(operation_id)
            if record is not None and record.status == "running":
                if record.cancellation_requested:
                    record.status = "cancelled"
                    record.message = "Cancelled"
                    record.completed_at = monotonic()
                else:
                    record.status = "success"
                    record.result = dict(result)
                    record.message = str(result.get("message") or "Completed")
                    record.completed_at = monotonic()
        return result


class MA3CatalogService:
    """Scoped/cached MA3 catalog reads with deterministic invalidation."""

    def __init__(self, client: MA3ProtocolClient):
        self._client = client
        self._lock = Lock()
        self._timecodes: list[MA3TimecodeSnapshot] | None = None
        self._track_groups_by_timecode: dict[int, list[MA3TrackGroupSnapshot]] = {}
        self._tracks_by_scope: dict[tuple[int | None, int | None], list[MA3TrackSnapshot]] = {}
        self._events_by_coord: dict[str, list[MA3EventSnapshot]] = {}
        self._sequences_by_range: dict[tuple[int | None, int | None], list[MA3SequenceSnapshot]] = {}
        self._current_song_sequence_range: MA3SequenceRangeSnapshot | None = None

    def list_timecodes(self, *, refresh: bool = False) -> list[MA3TimecodeSnapshot]:
        with self._lock:
            cached = self._timecodes
        if cached is None or refresh:
            fresh = [coerce_timecode_snapshot(item) for item in self._client.list_timecodes()]
            with self._lock:
                self._timecodes = list(fresh)
            return fresh
        return list(cached)

    def list_track_groups(
        self,
        *,
        timecode_no: int,
        refresh: bool = False,
    ) -> list[MA3TrackGroupSnapshot]:
        resolved_timecode_no = int(timecode_no)
        with self._lock:
            cached = self._track_groups_by_timecode.get(resolved_timecode_no)
        if cached is None or refresh:
            source = (
                self._client.refresh_track_groups(timecode_no=resolved_timecode_no)
                if refresh
                else self._client.list_track_groups(timecode_no=resolved_timecode_no)
            )
            fresh = [coerce_trackgroup_snapshot(item) for item in source]
            with self._lock:
                self._track_groups_by_timecode[resolved_timecode_no] = list(fresh)
            return fresh
        return list(cached)

    def list_tracks(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
        refresh: bool = False,
    ) -> list[MA3TrackSnapshot]:
        key = (
            None if timecode_no is None else int(timecode_no),
            None if track_group_no is None else int(track_group_no),
        )
        with self._lock:
            cached = self._tracks_by_scope.get(key)
        if cached is None or refresh:
            source = (
                self._client.refresh_tracks(timecode_no=key[0], track_group_no=key[1])
                if refresh
                else self._client.list_tracks(timecode_no=key[0], track_group_no=key[1])
            )
            fresh = [coerce_track_snapshot(item) for item in source]
            with self._lock:
                self._tracks_by_scope[key] = list(fresh)
            return fresh
        return list(cached)

    def list_track_events(
        self,
        track_coord: str,
        *,
        refresh: bool = False,
    ) -> list[MA3EventSnapshot]:
        coord = str(track_coord or "").strip()
        if not coord:
            return []
        with self._lock:
            cached = self._events_by_coord.get(coord)
        if cached is None or refresh:
            source = (
                self._client.refresh_track_events(coord)
                if refresh
                else self._client.list_track_events(coord)
            )
            fresh = [coerce_event_snapshot(item) for item in source]
            with self._lock:
                self._events_by_coord[coord] = list(fresh)
            return fresh
        return list(cached)

    def list_sequences(
        self,
        *,
        start_no: int | None = None,
        end_no: int | None = None,
        refresh: bool = False,
    ) -> list[MA3SequenceSnapshot]:
        key = (
            None if start_no is None else int(start_no),
            None if end_no is None else int(end_no),
        )
        with self._lock:
            cached = self._sequences_by_range.get(key)
        if cached is None or refresh:
            source = self._client.list_sequences(start_no=key[0], end_no=key[1])
            fresh = [coerce_sequence_snapshot(item) for item in source]
            with self._lock:
                self._sequences_by_range[key] = list(fresh)
            return fresh
        return list(cached)

    def get_current_song_sequence_range(
        self,
        *,
        refresh: bool = False,
    ) -> MA3SequenceRangeSnapshot | None:
        with self._lock:
            cached = self._current_song_sequence_range
        if cached is None or refresh:
            fresh = coerce_sequence_range_snapshot(self._client.get_current_song_sequence_range())
            with self._lock:
                self._current_song_sequence_range = fresh
            return fresh
        return cached

    def refresh_tracks_scope(
        self,
        *,
        target_track_coord: str | None = None,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[MA3TrackSnapshot]:
        resolved_timecode_no = int(timecode_no) if timecode_no is not None else None
        resolved_track_group_no = int(track_group_no) if track_group_no is not None else None
        coord = str(target_track_coord or "").strip()
        if coord:
            parsed = parse_track_coord(coord)
            resolved_timecode_no = parsed[0]
            resolved_track_group_no = parsed[1]
        with self._lock:
            self._tracks_by_scope.pop((resolved_timecode_no, resolved_track_group_no), None)
            if resolved_track_group_no is None:
                self._tracks_by_scope.pop((resolved_timecode_no, None), None)
        return self.list_tracks(
            timecode_no=resolved_timecode_no,
            track_group_no=resolved_track_group_no,
            refresh=True,
        )

    def invalidate_track_events(self, target_track_coord: str) -> None:
        coord = str(target_track_coord or "").strip()
        if not coord:
            return
        with self._lock:
            self._events_by_coord.pop(coord, None)


class MA3PushService:
    """Push orchestration service over strict protocol and scoped catalog refresh."""

    def __init__(
        self,
        *,
        client: MA3ProtocolClient,
        catalog: MA3CatalogService,
    ):
        self._client = client
        self._catalog = catalog

    def push(
        self,
        *,
        target_track_coord: str,
        selected_events: list[object],
        transfer_mode: str,
        start_offset_seconds: float,
    ) -> dict[str, object]:
        coord = str(target_track_coord or "").strip()
        if not coord:
            raise MA3ProtocolMismatchError("MA3 push requires target_track_coord")
        mode = str(transfer_mode or "merge").strip().lower()
        if mode not in {"merge", "overwrite"}:
            raise MA3ProtocolMismatchError(f"Unsupported MA3 push mode: {transfer_mode}")
        submitted_events = list(selected_events or [])
        submitted_count = len(submitted_events)

        self._client.apply_push_transfer(
            target_track_coord=coord,
            selected_events=submitted_events,
            transfer_mode=mode,
            start_offset_seconds=float(start_offset_seconds),
        )

        self._catalog.invalidate_track_events(coord)
        refreshed_events = self._catalog.list_track_events(coord, refresh=True)
        refreshed_tracks = self._catalog.refresh_tracks_scope(target_track_coord=coord)
        refreshed_track = next((track for track in refreshed_tracks if track.coord == coord), None)
        target_event_count = len(refreshed_events)

        if mode == "overwrite" and target_event_count != submitted_count:
            raise MA3SyncError(
                "MA3 overwrite verification failed for {coord}: wrote {actual} of {expected} event(s). "
                "Reduce push size or retry after MA3 track readiness checks.".format(
                    coord=coord,
                    actual=target_event_count,
                    expected=submitted_count,
                )
            )

        return {
            "message": f"Sent {submitted_count} event(s) to {coord}",
            "target_track_coord": coord,
            "transfer_mode": mode,
            "target_event_count": target_event_count,
            "target_sequence_no": None if refreshed_track is None else refreshed_track.sequence_no,
        }


def _translate_ma3_error(exc: Exception) -> Exception:
    if isinstance(exc, MA3SyncError):
        return exc
    if isinstance(exc, ValueError):
        return MA3ProtocolMismatchError(str(exc))

    text = str(exc).strip()
    lowered = text.lower()
    if "timed out" in lowered:
        if any(
            key in lowered
            for key in (
                "clear",
                "before overwrite",
                "addevent",
                "assigning",
                "write-ready",
                "createcmdsubtrack",
            )
        ):
            return MA3WriteTimeoutError(text or exc.__class__.__name__)
        return MA3LookupTimeoutError(text or exc.__class__.__name__)
    return MA3SyncError(text or exc.__class__.__name__)


__all__ = [
    "MA3CatalogService",
    "MA3LookupTimeoutError",
    "MA3OperationRunner",
    "MA3OperationSnapshot",
    "MA3ProtocolClient",
    "MA3ProtocolMismatchError",
    "MA3PushService",
    "MA3SyncError",
    "MA3WriteTimeoutError",
    "_translate_ma3_error",
]

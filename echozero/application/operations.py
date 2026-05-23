"""
Application operation contracts for cross-lane work.
Exists so long-running app, runtime, and sync workflows share one state shape.
Connects legacy progress records to operation snapshots consumed by UI and automation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class OperationKind(StrEnum):
    """Canonical operation categories exposed at the application boundary."""

    PLAYBACK = "playback"
    TRANSPORT = "transport"
    PIPELINE = "pipeline"
    SYNC = "sync"
    IMPORT = "import"
    REVIEW = "review"
    DIAGNOSTICS = "diagnostics"
    EXPORT = "export"


class OperationLane(StrEnum):
    """Execution lanes used to keep operation ownership explicit."""

    UI = "ui"
    APP = "app"
    REALTIME = "realtime"
    TRANSPORT = "transport"
    PREPARE = "prepare"
    PERSIST = "persist"
    SYNC = "sync"
    AUTOMATION = "automation"


class OperationStatus(StrEnum):
    """Public lifecycle states shared by app-visible operations."""

    QUEUED = "queued"
    PREPARING = "preparing"
    READY = "ready"
    APPLYING = "applying"
    RUNNING = "running"
    PERSISTING = "persisting"
    APPLIED = "applied"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_OPERATION_STATUSES = frozenset(
    {
        OperationStatus.QUEUED,
        OperationStatus.PREPARING,
        OperationStatus.READY,
        OperationStatus.APPLYING,
        OperationStatus.RUNNING,
        OperationStatus.PERSISTING,
    }
)
FINAL_OPERATION_STATUSES = frozenset(
    {
        OperationStatus.APPLIED,
        OperationStatus.FAILED,
        OperationStatus.CANCELLED,
    }
)


@dataclass(slots=True, frozen=True)
class OperationSubject:
    """Describes the project object an operation is acting on."""

    song_id: str | None = None
    song_version_id: str | None = None
    layer_id: str | None = None
    take_id: str | None = None
    object_id: str | None = None
    object_type: str | None = None
    label: str = ""


@dataclass(slots=True, frozen=True)
class OperationState:
    """Immutable app-visible state for one operation."""

    operation_id: str
    kind: OperationKind
    lane: OperationLane
    status: OperationStatus
    subject: OperationSubject = field(default_factory=OperationSubject)
    command_name: str = ""
    message: str = ""
    progress: float | None = None
    generation_id: str | None = None
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float | None = None
    can_cancel: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", str(self.operation_id or "").strip())
        object.__setattr__(self, "kind", _coerce_kind(self.kind))
        object.__setattr__(self, "lane", _coerce_lane(self.lane))
        object.__setattr__(self, "status", _coerce_status(self.status))
        object.__setattr__(self, "command_name", str(self.command_name or "").strip())
        object.__setattr__(self, "message", str(self.message or "").strip())
        object.__setattr__(self, "progress", clamp_progress(self.progress))
        started_at = normalize_timestamp(self.started_at)
        updated_at = normalize_timestamp(self.updated_at) or started_at
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "finished_at", normalize_timestamp(self.finished_at))


@dataclass(slots=True, frozen=True)
class OperationSnapshot:
    """Point-in-time operation state for application surfaces."""

    revision: int
    active_operations: tuple[OperationState, ...] = ()
    recent_final_operations: tuple[OperationState, ...] = ()


def clamp_progress(progress: float | None) -> float | None:
    """Clamp a progress fraction into the public 0..1 range."""

    if progress is None:
        return None
    return max(0.0, min(1.0, float(progress)))


def normalize_timestamp(value: float | int | None) -> float | None:
    """Return a non-negative timestamp or `None` when no timestamp exists."""

    if value is None:
        return None
    return max(0.0, float(value))


def current_timestamp() -> float:
    """Return the default wall-clock timestamp for operation records."""

    return time.time()


def is_active_status(status: OperationStatus | str) -> bool:
    """Return whether `status` represents unfinished operation work."""

    return _coerce_status(status) in ACTIVE_OPERATION_STATUSES


def is_final_status(status: OperationStatus | str) -> bool:
    """Return whether `status` represents terminal operation work."""

    return _coerce_status(status) in FINAL_OPERATION_STATUSES


def operation_status_from_progress_status(status: object) -> OperationStatus:
    """Map legacy operation-progress statuses to public operation statuses."""

    normalized = str(status or "").strip().lower()
    mapping = {
        "queued": OperationStatus.QUEUED,
        "resolving": OperationStatus.PREPARING,
        "preparing": OperationStatus.PREPARING,
        "ready": OperationStatus.READY,
        "applying": OperationStatus.APPLYING,
        "running": OperationStatus.RUNNING,
        "persisting": OperationStatus.PERSISTING,
        "completed": OperationStatus.APPLIED,
        "applied": OperationStatus.APPLIED,
        "failed": OperationStatus.FAILED,
        "cancelled": OperationStatus.CANCELLED,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported operation status: {status!r}")
    return mapping[normalized]


def operation_state_from_progress_state(state: Any) -> OperationState:
    """Translate one legacy pipeline progress record into `OperationState`."""

    diagnostics = {
        "legacy_status": str(getattr(state, "status", "") or ""),
        "workflow_id": str(getattr(state, "workflow_id", "") or ""),
        "output_layer_ids": tuple(str(item) for item in getattr(state, "output_layer_ids", ())),
    }
    return OperationState(
        operation_id=str(getattr(state, "operation_id", "") or ""),
        kind=_coerce_kind(getattr(state, "kind", OperationKind.PIPELINE)),
        lane=OperationLane.PREPARE,
        status=operation_status_from_progress_status(getattr(state, "status", "")),
        subject=OperationSubject(
            song_id=_optional_text(getattr(state, "song_id", None)),
            song_version_id=_optional_text(getattr(state, "song_version_id", None)),
            layer_id=_optional_text(getattr(state, "source_layer_id", None)),
            object_id=_optional_text(getattr(state, "object_id", None)),
            object_type=_optional_text(getattr(state, "object_type", None)),
            label=str(getattr(state, "display_label", "") or ""),
        ),
        command_name=str(getattr(state, "action_id", "") or ""),
        message=str(getattr(state, "message", "") or ""),
        progress=getattr(state, "fraction_complete", None),
        started_at=float(getattr(state, "started_at", 0.0) or 0.0),
        updated_at=float(
            getattr(state, "updated_at", None)
            or getattr(state, "finished_at", None)
            or getattr(state, "started_at", 0.0)
            or 0.0
        ),
        finished_at=getattr(state, "finished_at", None),
        can_cancel=bool(getattr(state, "can_cancel", False)),
        diagnostics=diagnostics,
        error=_optional_text(getattr(state, "error", None)),
    )


def operation_status_from_ma3_status(status: object) -> OperationStatus:
    """Map MA3 async operation statuses to public operation statuses."""

    normalized = str(status or "").strip().lower()
    mapping = {
        "running": OperationStatus.RUNNING,
        "success": OperationStatus.APPLIED,
        "error": OperationStatus.FAILED,
        "cancelled": OperationStatus.CANCELLED,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported MA3 operation status: {status!r}")
    return mapping[normalized]


def operation_state_from_ma3_operation_snapshot(
    snapshot: Any,
    *,
    observed_at: float | None = None,
) -> OperationState:
    """Translate one MA3 async operation snapshot into `OperationState`.

    MA3 runner timestamps are monotonic, so public wall-clock timestamps use the
    observation time and retain raw runner times as diagnostics.
    """

    timestamp = normalize_timestamp(observed_at) or current_timestamp()
    status = operation_status_from_ma3_status(getattr(snapshot, "status", ""))
    finished_at = timestamp if is_final_status(status) else None
    return OperationState(
        operation_id=str(getattr(snapshot, "operation_id", "") or ""),
        kind=OperationKind.SYNC,
        lane=OperationLane.SYNC,
        status=status,
        subject=OperationSubject(label=str(getattr(snapshot, "kind", "") or "MA3 operation")),
        command_name=str(getattr(snapshot, "kind", "") or ""),
        message=str(getattr(snapshot, "message", "") or ""),
        started_at=timestamp,
        updated_at=timestamp,
        finished_at=finished_at,
        can_cancel=status is OperationStatus.RUNNING,
        diagnostics={
            "legacy_status": str(getattr(snapshot, "status", "") or ""),
            "legacy_kind": str(getattr(snapshot, "kind", "") or ""),
            "legacy_started_at_monotonic": getattr(snapshot, "started_at", None),
            "legacy_completed_at_monotonic": getattr(snapshot, "completed_at", None),
            "result": getattr(snapshot, "result", None),
        },
        error=_optional_text(getattr(snapshot, "error", None)),
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_kind(value: OperationKind | str) -> OperationKind:
    return value if isinstance(value, OperationKind) else OperationKind(str(value))


def _coerce_lane(value: OperationLane | str) -> OperationLane:
    return value if isinstance(value, OperationLane) else OperationLane(str(value))


def _coerce_status(value: OperationStatus | str) -> OperationStatus:
    if isinstance(value, OperationStatus):
        return value
    return operation_status_from_progress_status(value)


__all__ = [
    "ACTIVE_OPERATION_STATUSES",
    "FINAL_OPERATION_STATUSES",
    "OperationKind",
    "OperationLane",
    "OperationSnapshot",
    "OperationState",
    "OperationStatus",
    "OperationSubject",
    "clamp_progress",
    "current_timestamp",
    "is_active_status",
    "is_final_status",
    "normalize_timestamp",
    "operation_state_from_ma3_operation_snapshot",
    "operation_state_from_progress_state",
    "operation_status_from_ma3_status",
    "operation_status_from_progress_status",
]

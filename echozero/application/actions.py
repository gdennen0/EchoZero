"""
Application action gateway contracts for EchoZero command routing.
Exists so UI, dialogs, timeline controls, and runtime-adjacent commands share one low-latency entrypoint.
Connects typed commands to operation-state snapshots while legacy app paths migrate incrementally.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from echozero.application.operations import (
    OperationKind,
    OperationLane,
    OperationSnapshot,
    OperationState,
    OperationStatus,
    OperationSubject,
)


class ActionPriority(StrEnum):
    """Gateway priority bands for queued and synchronous command users."""

    REALTIME = "realtime"
    USER_BLOCKING = "user_blocking"
    NORMAL = "normal"
    BACKGROUND = "background"


class ActionCoalescingMode(StrEnum):
    """How pending commands with the same coalescing key should be handled."""

    NONE = "none"
    KEEP_LATEST = "keep_latest"
    REPLACE_PENDING = "replace_pending"


class ActionResultStatus(StrEnum):
    """Public acceptance/result states for gateway commands."""

    ACCEPTED = "accepted"
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True, frozen=True)
class ActionCoalescing:
    """Coalescing policy for high-frequency UI or transport commands."""

    mode: ActionCoalescingMode = ActionCoalescingMode.NONE
    key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _coerce_coalescing_mode(self.mode))
        object.__setattr__(self, "key", str(self.key or "").strip())


@dataclass(slots=True, frozen=True)
class ActionCommand:
    """One app-visible command submitted by UI, automation, dialogs, or runtime adapters."""

    command_type: str
    command_id: str = ""
    lane: OperationLane = OperationLane.APP
    priority: ActionPriority = ActionPriority.NORMAL
    source: str = "app"
    payload: dict[str, object] = field(default_factory=dict)
    subject: OperationSubject = field(default_factory=OperationSubject)
    coalescing: ActionCoalescing = field(default_factory=ActionCoalescing)
    operation_kind: OperationKind = OperationKind.PIPELINE
    generation_id: str | None = None
    can_cancel: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)
    created_monotonic_seconds: float = 0.0

    def __post_init__(self) -> None:
        command_type = str(self.command_type or "").strip()
        if not command_type:
            raise ValueError("ActionCommand requires a command_type.")
        object.__setattr__(self, "command_type", command_type)
        object.__setattr__(self, "command_id", str(self.command_id or _new_action_id()).strip())
        object.__setattr__(self, "lane", _coerce_operation_lane(self.lane))
        object.__setattr__(self, "priority", _coerce_action_priority(self.priority))
        object.__setattr__(self, "source", str(self.source or "app").strip() or "app")
        object.__setattr__(self, "payload", dict(self.payload or {}))
        object.__setattr__(self, "coalescing", _coerce_coalescing(self.coalescing))
        object.__setattr__(self, "operation_kind", _coerce_operation_kind(self.operation_kind))
        created = float(self.created_monotonic_seconds or 0.0)
        object.__setattr__(
            self,
            "created_monotonic_seconds",
            created if created > 0.0 else time.monotonic(),
        )

    @property
    def coalescing_key(self) -> str:
        """Return the effective coalescing key for this command."""

        return self.coalescing.key


@dataclass(slots=True, frozen=True)
class ActionRoute:
    """Default routing metadata for one command type."""

    command_type: str
    lane: OperationLane = OperationLane.APP
    priority: ActionPriority = ActionPriority.NORMAL
    coalescing: ActionCoalescing = field(default_factory=ActionCoalescing)
    operation_kind: OperationKind = OperationKind.PIPELINE

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_type", str(self.command_type or "").strip())
        object.__setattr__(self, "lane", _coerce_operation_lane(self.lane))
        object.__setattr__(self, "priority", _coerce_action_priority(self.priority))
        object.__setattr__(self, "coalescing", _coerce_coalescing(self.coalescing))
        object.__setattr__(self, "operation_kind", _coerce_operation_kind(self.operation_kind))


@dataclass(slots=True, frozen=True)
class ActionResult:
    """Result of accepting, applying, or rejecting one gateway command."""

    command: ActionCommand
    status: ActionResultStatus
    operation_id: str = ""
    message: str = ""
    value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _coerce_result_status(self.status))
        object.__setattr__(self, "operation_id", str(self.operation_id or "").strip())
        object.__setattr__(self, "message", str(self.message or "").strip())

    @property
    def ok(self) -> bool:
        """Return whether the command was accepted or applied."""

        return self.status in {ActionResultStatus.ACCEPTED, ActionResultStatus.APPLIED}

    @property
    def command_id(self) -> str:
        """Return the normalized command id."""

        return self.command.command_id

    @classmethod
    def accepted(
        cls,
        command: ActionCommand,
        *,
        operation_id: str = "",
        value: Any = None,
    ) -> "ActionResult":
        """Build an accepted result for one command."""

        return cls(
            command=command,
            status=ActionResultStatus.ACCEPTED,
            operation_id=operation_id,
            value=value,
        )

    @classmethod
    def rejected(cls, command: ActionCommand, message: str) -> "ActionResult":
        """Build a rejected result for one command."""

        return cls(command=command, status=ActionResultStatus.REJECTED, message=message)


class ActionGateway:
    """Low-latency action entrypoint with operation-state diagnostics."""

    def __init__(
        self,
        *,
        routes: dict[str, ActionRoute] | None = None,
        recent_limit: int = 32,
    ) -> None:
        self._routes = dict(routes or {})
        self._revision = 0
        self._active: dict[str, OperationState] = {}
        self._recent: deque[OperationState] = deque(maxlen=max(1, int(recent_limit)))
        self._latest_by_coalescing_key: dict[str, str] = {}
        self._final_by_operation_id: dict[str, OperationState] = {}
        self._coalesced_count = 0

    @property
    def coalesced_count(self) -> int:
        """Return the number of accepted commands superseded by later intent."""

        return int(self._coalesced_count)

    def accept(self, command: ActionCommand) -> ActionResult:
        """Record a command as accepted without requiring immediate execution."""

        routed = route_action_command(command, self._routes)
        self._mark_coalesced(routed)
        operation = self._operation_from_command(routed, OperationStatus.QUEUED)
        self._active[operation.operation_id] = operation
        self._bump_revision()
        return ActionResult.accepted(routed, operation_id=operation.operation_id)

    def complete(self, command_id: str, *, message: str = "") -> OperationState:
        """Mark an accepted command as applied."""

        return self._finish(command_id, OperationStatus.APPLIED, message=message)

    def fail(self, command_id: str, error: str, *, message: str = "") -> OperationState:
        """Mark an accepted command as failed."""

        return self._finish(command_id, OperationStatus.FAILED, message=message, error=error)

    def snapshot(self) -> OperationSnapshot:
        """Return active and recent gateway operations."""

        return OperationSnapshot(
            revision=int(self._revision),
            active_operations=tuple(self._active.values()),
            recent_final_operations=tuple(self._recent),
        )

    def _mark_coalesced(self, command: ActionCommand) -> None:
        key = command.coalescing_key
        if not key or command.coalescing.mode is ActionCoalescingMode.NONE:
            return
        previous_id = self._latest_by_coalescing_key.get(key)
        self._latest_by_coalescing_key[key] = command.command_id
        if not previous_id or previous_id not in self._active:
            return
        previous = self._active.pop(previous_id)
        finished = _replace_operation(
            previous,
            status=OperationStatus.CANCELLED,
            message="Superseded by newer command intent.",
            finished_at=time.time(),
        )
        self._remember_final(finished)
        self._coalesced_count += 1
        self._bump_revision()

    def _finish(
        self,
        command_id: str,
        status: OperationStatus,
        *,
        message: str = "",
        error: str | None = None,
    ) -> OperationState:
        operation = self._active.pop(str(command_id).strip(), None)
        if operation is None:
            final = self._final_by_operation_id.get(str(command_id).strip())
            if final is not None:
                return final
            raise KeyError(f"Unknown action command id: {command_id}")
        finished = _replace_operation(
            operation,
            status=status,
            message=message,
            error=error,
            finished_at=time.time(),
        )
        self._remember_final(finished)
        self._bump_revision()
        return finished

    def _remember_final(self, operation: OperationState) -> None:
        self._recent.append(operation)
        self._final_by_operation_id[operation.operation_id] = operation
        recent_ids = {state.operation_id for state in self._recent}
        for operation_id in list(self._final_by_operation_id):
            if operation_id not in recent_ids:
                self._final_by_operation_id.pop(operation_id, None)

    def _operation_from_command(
        self,
        command: ActionCommand,
        status: OperationStatus,
    ) -> OperationState:
        return OperationState(
            operation_id=command.command_id,
            kind=command.operation_kind,
            lane=command.lane,
            status=status,
            subject=command.subject,
            command_name=command.command_type,
            generation_id=command.generation_id,
            started_at=time.time(),
            updated_at=time.time(),
            can_cancel=command.can_cancel,
            diagnostics={
                "source": command.source,
                "priority": command.priority.value,
                "coalescing_key": command.coalescing_key,
                **dict(command.diagnostics),
            },
        )

    def _bump_revision(self) -> None:
        self._revision += 1


def route_action_command(
    command: ActionCommand,
    routes: dict[str, ActionRoute],
) -> ActionCommand:
    """Apply registered route defaults to one command."""

    route = routes.get(command.command_type)
    if route is None:
        return command
    coalescing = (
        command.coalescing
        if command.coalescing.mode is not ActionCoalescingMode.NONE
        else route.coalescing
    )
    return replace(
        command,
        lane=route.lane,
        priority=route.priority,
        coalescing=coalescing,
        operation_kind=route.operation_kind,
    )


def bounded_operation_snapshot(
    snapshot: OperationSnapshot | None,
    *,
    active_limit: int = 16,
    recent_limit: int = 16,
) -> OperationSnapshot:
    """Return a snapshot capped to the latest active and final operations."""

    if snapshot is None:
        return OperationSnapshot(revision=0)
    active = sorted(
        snapshot.active_operations,
        key=lambda state: (float(state.updated_at), float(state.started_at)),
    )
    recent = sorted(
        snapshot.recent_final_operations,
        key=lambda state: (
            float(state.finished_at or 0.0),
            float(state.updated_at),
            float(state.started_at),
        ),
    )
    return OperationSnapshot(
        revision=snapshot.revision,
        active_operations=tuple(active[-max(0, int(active_limit)) :])
        if active_limit > 0
        else (),
        recent_final_operations=tuple(recent[-max(0, int(recent_limit)) :])
        if recent_limit > 0
        else (),
    )


def _replace_operation(
    operation: OperationState,
    *,
    status: OperationStatus,
    message: str = "",
    error: str | None = None,
    finished_at: float | None = None,
) -> OperationState:
    return OperationState(
        operation_id=operation.operation_id,
        kind=operation.kind,
        lane=operation.lane,
        status=status,
        subject=operation.subject,
        command_name=operation.command_name,
        message=message or operation.message,
        progress=operation.progress,
        generation_id=operation.generation_id,
        started_at=operation.started_at,
        updated_at=time.time(),
        finished_at=finished_at,
        can_cancel=operation.can_cancel,
        diagnostics=dict(operation.diagnostics),
        error=error,
    )


def _coerce_action_priority(value: ActionPriority | str) -> ActionPriority:
    return value if isinstance(value, ActionPriority) else ActionPriority(str(value))


def _coerce_coalescing_mode(
    value: ActionCoalescingMode | str,
) -> ActionCoalescingMode:
    if isinstance(value, ActionCoalescingMode):
        return value
    return ActionCoalescingMode(str(value))


def _coerce_coalescing(value: ActionCoalescing | None) -> ActionCoalescing:
    if value is None:
        return ActionCoalescing()
    return value if isinstance(value, ActionCoalescing) else ActionCoalescing()


def _coerce_operation_lane(value: OperationLane | str) -> OperationLane:
    return value if isinstance(value, OperationLane) else OperationLane(str(value))


def _coerce_operation_kind(value: OperationKind | str) -> OperationKind:
    return value if isinstance(value, OperationKind) else OperationKind(str(value))


def _coerce_result_status(value: ActionResultStatus | str) -> ActionResultStatus:
    return value if isinstance(value, ActionResultStatus) else ActionResultStatus(str(value))


def _new_action_id() -> str:
    return f"action-{uuid.uuid4().hex}"


__all__ = [
    "ActionCoalescing",
    "ActionCoalescingMode",
    "ActionCommand",
    "ActionGateway",
    "ActionPriority",
    "ActionResult",
    "ActionResultStatus",
    "ActionRoute",
    "bounded_operation_snapshot",
    "route_action_command",
]

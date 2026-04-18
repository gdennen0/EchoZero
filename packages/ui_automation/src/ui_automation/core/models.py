from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class AutomationObjectFact:
    label: str
    value: str


@dataclass(slots=True, frozen=True)
class AutomationObject:
    object_id: str
    object_type: str
    label: str
    target_id: str | None = None
    facts: tuple[AutomationObjectFact, ...] = ()
    actions: tuple["AutomationAction", ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AutomationBounds:
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True, frozen=True)
class AutomationTarget:
    kind: str
    target_id: str
    parent_id: str | None = None
    label: str | None = None
    bounds: AutomationBounds | None = None
    time_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AutomationAction:
    action_id: str
    label: str
    enabled: bool = True
    group: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    target_id: str | None = None


@dataclass(slots=True, frozen=True)
class AutomationHitTarget:
    target_id: str
    kind: str
    bounds: AutomationBounds
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AutomationSnapshot:
    app: str
    selection: tuple[str, ...] = ()
    focused_target_id: str | None = None
    focused_object_id: str | None = None
    sync: dict[str, Any] = field(default_factory=dict)
    targets: tuple[AutomationTarget, ...] = ()
    actions: tuple[AutomationAction, ...] = ()
    objects: tuple[AutomationObject, ...] = ()
    hit_targets: tuple[AutomationHitTarget, ...] = ()
    artifacts: dict[str, Any] = field(default_factory=dict)

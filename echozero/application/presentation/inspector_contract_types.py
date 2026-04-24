"""Inspector contract types for the canonical presentation layer.
Exists to keep contract data structures stable and reusable across builder modules.
Connects inspector assembly, tests, and Qt surfaces through one typed contract vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class InspectorObjectIdentity:
    object_id: str
    object_type: str
    label: str


@dataclass(slots=True, frozen=True)
class InspectorFactRow:
    label: str
    value: str


@dataclass(slots=True, frozen=True)
class InspectorSection:
    section_id: str
    label: str
    rows: tuple[InspectorFactRow, ...] = ()


@dataclass(slots=True, frozen=True)
class InspectorAction:
    action_id: str
    label: str
    enabled: bool = True
    kind: str = "intent"
    group: str = "default"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class InspectorContextSection:
    section_id: str
    label: str
    actions: tuple[InspectorAction, ...] = ()


@dataclass(slots=True, frozen=True)
class TimelineInspectorHitTarget:
    kind: str
    layer_id: object | None = None
    take_id: object | None = None
    event_id: object | None = None
    time_seconds: float | None = None


@dataclass(slots=True, frozen=True)
class InspectorContract:
    title: str
    identity: InspectorObjectIdentity | None = None
    sections: tuple[InspectorSection, ...] = ()
    context_sections: tuple[InspectorContextSection, ...] = ()
    empty_state: str = "No timeline object selected."

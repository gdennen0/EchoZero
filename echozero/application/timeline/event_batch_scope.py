"""EventBatchScope: Typed scope descriptors for timeline batch event operations.
Exists to keep selection-vs-layer-vs-take targeting out of widget-local action params and ad hoc orchestrator branches.
Connects inspector/context actions and canonical timeline intents to one shared batch-operation scope vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from echozero.application.shared.ids import LayerId, RegionId, TakeId
from echozero.application.timeline.models import EventRef

_SCOPE_MODES = frozenset(
    {
        "selected_events",
        "take",
        "layer_main",
        "selected_layers_main",
        "region",
    }
)


@dataclass(frozen=True, slots=True)
class EventBatchScope:
    """Describes which timeline events one batch operation should target."""

    mode: str
    layer_id: LayerId | None = None
    take_id: TakeId | None = None
    region_id: RegionId | None = None

    def __post_init__(self) -> None:
        mode = (self.mode or "").strip().lower()
        if mode not in _SCOPE_MODES:
            raise ValueError(f"Unsupported event batch scope mode: {self.mode!r}")
        object.__setattr__(self, "mode", mode)
        if mode == "take":
            if self.layer_id is None or self.take_id is None:
                raise ValueError("EventBatchScope(mode='take') requires layer_id and take_id")
            return
        if mode == "layer_main":
            if self.layer_id is None:
                raise ValueError("EventBatchScope(mode='layer_main') requires layer_id")
            return
        if mode == "region":
            if self.region_id is None:
                raise ValueError("EventBatchScope(mode='region') requires region_id")
            if self.layer_id is not None or self.take_id is not None:
                raise ValueError("EventBatchScope(mode='region') does not accept layer_id or take_id")
            return
        if self.take_id is not None:
            raise ValueError(f"EventBatchScope(mode={mode!r}) does not accept take_id")
        if self.region_id is not None:
            raise ValueError(f"EventBatchScope(mode={mode!r}) does not accept region_id")


@dataclass(frozen=True, slots=True)
class ResolvedEventBatchScope:
    """Resolved event refs plus selection anchors for one batch operation."""

    scope: EventBatchScope
    event_refs: tuple[EventRef, ...]
    event_ref_groups: tuple[tuple[EventRef, ...], ...]
    anchor_layer_id: LayerId | None
    anchor_take_id: TakeId | None
    selected_layer_ids: tuple[LayerId, ...]
    label: str

    @property
    def is_empty(self) -> bool:
        return not self.event_refs


def event_batch_scope_params(scope: EventBatchScope) -> dict[str, object]:
    """Serialize a typed batch scope into inspector action params."""

    params: dict[str, object] = {"scope_mode": scope.mode}
    if scope.layer_id is not None:
        params["scope_layer_id"] = str(scope.layer_id)
    if scope.take_id is not None:
        params["scope_take_id"] = str(scope.take_id)
    if scope.region_id is not None:
        params["scope_region_id"] = str(scope.region_id)
    return params


def event_batch_scope_from_params(params: Mapping[str, object]) -> EventBatchScope | None:
    """Deserialize inspector action params into a typed batch scope."""

    raw_mode = params.get("scope_mode")
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        return None
    raw_layer_id = params.get("scope_layer_id")
    raw_take_id = params.get("scope_take_id")
    raw_region_id = params.get("scope_region_id")
    layer_id = (
        LayerId(raw_layer_id.strip())
        if isinstance(raw_layer_id, str) and raw_layer_id.strip()
        else None
    )
    take_id = (
        TakeId(raw_take_id.strip())
        if isinstance(raw_take_id, str) and raw_take_id.strip()
        else None
    )
    region_id = (
        RegionId(raw_region_id.strip())
        if isinstance(raw_region_id, str) and raw_region_id.strip()
        else None
    )
    return EventBatchScope(
        mode=raw_mode,
        layer_id=layer_id,
        take_id=take_id,
        region_id=region_id,
    )

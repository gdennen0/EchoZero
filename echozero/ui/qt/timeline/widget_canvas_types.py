"""Timeline canvas shared types.
Exists to keep paint and interaction helpers aligned on one type vocabulary.
Connects the canvas root and helper mixins through shared event/rect contracts.
"""

from __future__ import annotations

from typing import TypedDict

from PyQt6.QtCore import QPointF, QRectF, Qt

from echozero.application.shared.ids import EventId, LayerId, SectionCueId, TakeId

TakeRect = tuple[QRectF, LayerId, TakeId]
TakeActionRect = tuple[QRectF, LayerId, TakeId, str]
EventRect = tuple[QRectF, LayerId, TakeId | None, EventId]
EventLaneRect = tuple[QRectF, LayerId, TakeId | None]
FixEventRect = tuple[QRectF, LayerId, TakeId | None, str, float, float, bool]
SectionLabelRect = tuple[QRectF, SectionCueId]
SectionBoundaryRect = tuple[QRectF, SectionCueId]


class LayerDragCandidate(TypedDict):
    source_layer_id: LayerId
    anchor_y: float
    target_after_layer_id: LayerId | None
    insert_at_start: bool


class EventDragCandidate(TypedDict):
    anchor_x: float
    anchor_y: float
    source_layer_id: LayerId
    source_take_id: TakeId | None
    anchor_event_id: EventId
    anchor_event_start: float
    copy_on_drag: bool


class SelectionDragCandidate(TypedDict):
    anchor_pos: QPointF
    origin_layer_id: LayerId
    origin_take_id: TakeId | None
    modifiers: Qt.KeyboardModifier
    edit_mode: str
    fix_action: str


class DrawCandidate(TypedDict):
    layer_id: LayerId
    take_id: TakeId | None
    anchor_time: float


class LayerResizeCandidate(TypedDict):
    layer_id: LayerId
    anchor_y: float
    anchor_height: int

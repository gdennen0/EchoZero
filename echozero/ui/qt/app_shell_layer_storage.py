"""Layer storage helpers for the Qt app shell.
Exists to isolate runtime-layer creation and persistence DTO translation.
Connects in-memory timeline layers to ProjectStorage records and takes.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Iterable

from echozero.application.shared.enums import LayerKind
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.shared.ids import LayerId
from echozero.application.timeline.models import (
    Layer,
    LayerPresentationHints,
    LayerProvenance,
    LayerStatus,
    LayerSyncState,
    Take as RuntimeTake,
    Timeline,
)
from echozero.domain.types import Event as DomainEvent, EventData, Layer as DomainLayer
from echozero.persistence.entities import LayerRecord
from echozero.takes import Take as PersistedTake


def build_manual_layer(
    *,
    timeline: Timeline,
    layer_kind: LayerKind,
    layer_title: str,
) -> Layer:
    return Layer(
        id=LayerId(f"layer_{uuid.uuid4().hex[:12]}"),
        timeline_id=timeline.id,
        name=layer_title,
        kind=layer_kind,
        order_index=next_runtime_layer_order_index(timeline),
        status=LayerStatus(),
        provenance=LayerProvenance(),
        presentation_hints=LayerPresentationHints(),
        sync=LayerSyncState(),
    )


def next_runtime_layer_order_index(timeline: Timeline) -> int:
    if not timeline.layers:
        return 0
    return max(int(layer.order_index) for layer in timeline.layers) + 1


def next_persisted_manual_layer_order(existing_layers: Iterable[LayerRecord]) -> int:
    order_values = [int(layer.order) for layer in existing_layers]
    if not order_values:
        return 0
    return max(0, max(order_values) + 1)


def build_manual_layer_record(
    layer: Layer,
    *,
    song_version_id: str,
    persisted_order: int,
) -> LayerRecord:
    return LayerRecord(
        id=str(layer.id),
        song_version_id=song_version_id,
        name=layer.name,
        layer_type="manual",
        color=layer.presentation_hints.color,
        order=persisted_order,
        visible=layer.presentation_hints.visible,
        locked=layer.presentation_hints.locked,
        parent_layer_id=None,
        source_pipeline=None,
        created_at=datetime.now(timezone.utc),
        state_flags={
            "manual_kind": layer.kind.value,
            "take_lanes_expanded": bool(layer.presentation_hints.expanded),
        },
        provenance={},
    )


def manual_layer_take_data(layer: Layer) -> EventData:
    main_take = layer.takes[0] if layer.takes else None
    return runtime_take_data(layer, main_take)


def runtime_take_data(layer: Layer, take: RuntimeTake | None) -> EventData:
    if not is_event_like_layer_kind(layer.kind) or take is None:
        return EventData(layers=())
    domain_events = tuple(
        DomainEvent(
            id=str(event.id),
            time=float(event.start),
            duration=float(event.duration),
            classifications={"label": event.label} if event.label else {},
            metadata={
                key: value
                for key, value in {
                    "cue_number": event.cue_number,
                    "payload_ref": event.payload_ref,
                    "color": event.color,
                    "muted": event.muted,
                }.items()
                if value not in (None, False)
            },
            origin="user",
            source_event_id=event.source_event_id,
            parent_event_id=event.parent_event_id,
        )
        for event in take.events
    )
    return EventData(
        layers=(
            DomainLayer(
                id=str(layer.id),
                name=layer.name,
                events=domain_events,
            ),
        )
    )


def runtime_layer_record(
    layer: Layer,
    *,
    existing: LayerRecord,
) -> LayerRecord:
    state_flags = dict(existing.state_flags)
    state_flags["stale"] = bool(layer.status.stale)
    state_flags["manually_modified"] = bool(layer.status.manually_modified)
    if layer.status.stale_reason:
        state_flags["stale_reason"] = layer.status.stale_reason
    else:
        state_flags.pop("stale_reason", None)
    if existing.layer_type == "manual":
        state_flags["manual_kind"] = layer.kind.value
    state_flags["take_lanes_expanded"] = bool(layer.presentation_hints.expanded)

    provenance = dict(existing.provenance)
    if layer.provenance.source_layer_id is not None:
        provenance["source_layer_id"] = str(layer.provenance.source_layer_id)
    else:
        provenance.pop("source_layer_id", None)
    if layer.provenance.source_song_version_id is not None:
        provenance["source_song_version_id"] = str(layer.provenance.source_song_version_id)
    else:
        provenance.pop("source_song_version_id", None)
    if layer.provenance.source_run_id:
        provenance["source_run_id"] = layer.provenance.source_run_id
    else:
        provenance.pop("source_run_id", None)
    if layer.provenance.pipeline_id:
        provenance["pipeline_id"] = layer.provenance.pipeline_id
    else:
        provenance.pop("pipeline_id", None)
    if layer.provenance.output_name:
        provenance["output_name"] = layer.provenance.output_name
    else:
        provenance.pop("output_name", None)

    source_pipeline = dict(existing.source_pipeline) if existing.source_pipeline is not None else None
    if layer.provenance.pipeline_id or layer.provenance.output_name:
        source_pipeline = {
            **(source_pipeline or {}),
            **({"pipeline_id": layer.provenance.pipeline_id} if layer.provenance.pipeline_id else {}),
            **({"output_name": layer.provenance.output_name} if layer.provenance.output_name else {}),
        }

    return replace(
        existing,
        name=layer.name,
        color=layer.presentation_hints.color,
        order=max(0, int(layer.order_index) - 1),
        visible=layer.presentation_hints.visible,
        locked=layer.presentation_hints.locked,
        source_pipeline=source_pipeline,
        state_flags=state_flags,
        provenance=provenance,
    )


def persisted_take_from_runtime_take(
    layer: Layer,
    take: RuntimeTake,
    *,
    existing: PersistedTake | None,
    is_main: bool,
) -> PersistedTake:
    if existing is not None:
        if not is_event_like_layer_kind(layer.kind):
            return replace(existing, label=take.name or existing.label, is_main=is_main)
        return replace(
            existing,
            data=runtime_take_data(layer, take),
            is_main=is_main,
        )
    return PersistedTake(
        id=str(take.id),
        label="Take 1" if is_main else (take.name or "Take"),
        data=runtime_take_data(layer, take),
        origin="user",
        source=None,
        created_at=datetime.now(timezone.utc),
        is_main=is_main,
        is_archived=False,
        notes="",
    )

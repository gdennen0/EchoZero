from __future__ import annotations

from datetime import datetime, timezone

from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, SongVersionId, TakeId, TimelineId
from echozero.application.timeline.models import (
    Event,
    Layer,
    LayerPresentationHints,
    LayerProvenance,
    LayerStatus,
    Take,
    Timeline,
)
from echozero.domain.types import Event as DomainEvent, EventData, Layer as DomainLayer
from echozero.persistence.entities import LayerRecord
from echozero.takes import Take as PersistedTake
from echozero.ui.qt.app_shell_layer_storage import (
    build_manual_layer,
    build_manual_layer_record,
    manual_layer_take_data,
    persisted_take_from_runtime_take,
    runtime_layer_record,
    runtime_take_data,
)
from echozero.ui.qt.app_shell_project_timeline_storage import events_from_take


def test_build_manual_layer_uses_next_runtime_order_and_defaults():
    timeline = Timeline(
        id=TimelineId("timeline_runtime"),
        song_version_id=SongVersionId("song_version_runtime"),
        layers=[
            Layer(
                id=LayerId("layer_existing"),
                timeline_id=TimelineId("timeline_runtime"),
                name="Existing",
                kind=LayerKind.EVENT,
                order_index=4,
            )
        ],
    )

    layer = build_manual_layer(
        timeline=timeline,
        layer_kind=LayerKind.AUDIO,
        layer_title="Drums",
    )

    assert str(layer.id).startswith("layer_")
    assert layer.timeline_id == TimelineId("timeline_runtime")
    assert layer.name == "Drums"
    assert layer.kind is LayerKind.AUDIO
    assert layer.order_index == 5
    assert layer.presentation_hints == LayerPresentationHints()


def test_build_manual_layer_record_tracks_manual_kind_and_ui_state():
    layer = Layer(
        id=LayerId("layer_runtime"),
        timeline_id=TimelineId("timeline_runtime"),
        name="Manual Layer",
        kind=LayerKind.EVENT,
        order_index=2,
        presentation_hints=LayerPresentationHints(
            color="#123456",
            visible=False,
            locked=True,
        ),
    )

    record = build_manual_layer_record(
        layer,
        song_version_id="song_version_runtime",
        persisted_order=7,
    )

    assert record.id == "layer_runtime"
    assert record.song_version_id == "song_version_runtime"
    assert record.order == 7
    assert record.color == "#123456"
    assert record.visible is False
    assert record.locked is True
    assert record.state_flags["manual_kind"] == "event"
    assert record.state_flags["take_lanes_expanded"] is False


def test_runtime_take_data_serializes_event_metadata():
    layer = _event_layer()

    data = runtime_take_data(layer, layer.takes[0])

    assert len(data.layers) == 1
    serialized = data.layers[0].events[0]
    assert serialized.id == "evt_1"
    assert serialized.time == 1.5
    assert serialized.duration == 0.5
    assert serialized.classifications == {"label": "Kick"}
    assert serialized.metadata == {
        "cue_number": 1,
        "payload_ref": "payload://kick",
        "color": "#ff0000",
        "muted": True,
    }
    assert serialized.source_event_id == "src_evt"
    assert serialized.parent_event_id == "parent_evt"


def test_runtime_take_data_serializes_marker_layer_events():
    layer = _event_layer()
    layer.kind = LayerKind.MARKER

    data = runtime_take_data(layer, layer.takes[0])

    assert len(data.layers) == 1
    assert data.layers[0].events[0].metadata["cue_number"] == 1
    assert data.layers[0].events[0].classifications == {"label": "Kick"}


def test_events_from_take_restores_cue_number_from_metadata():
    take = PersistedTake.create(
        data=EventData(
            layers=(
                DomainLayer(
                    id="layer_runtime",
                    name="Events",
                    events=(
                        DomainEvent(
                            id="evt_1",
                            time=1.5,
                            duration=0.5,
                            classifications={"label": "Kick"},
                            metadata={"cue_number": 9},
                            origin="user",
                        ),
                    ),
                ),
            )
        ),
        label="Take 1",
        origin="user",
        is_main=True,
    )

    events = events_from_take(take)

    assert len(events) == 1
    assert events[0].cue_number == 9


def test_manual_layer_take_data_uses_first_take_only():
    layer = _event_layer()

    data = manual_layer_take_data(layer)

    assert data.layers[0].events[0].id == "evt_1"


def test_runtime_layer_record_updates_state_flags_and_provenance():
    layer = _event_layer()
    layer.status = LayerStatus(stale=True, manually_modified=True, stale_reason="source changed")
    layer.provenance = LayerProvenance(
        source_layer_id=LayerId("source_audio"),
        source_song_version_id=SongVersionId("song_version_source"),
        source_run_id="run_123",
        pipeline_id="pipeline_drums",
        output_name="drums",
    )
    layer.presentation_hints = LayerPresentationHints(
        color="#00ff00",
        visible=False,
        locked=True,
        expanded=True,
    )
    existing = LayerRecord(
        id="layer_runtime",
        song_version_id="song_version_runtime",
        name="Old Layer",
        layer_type="manual",
        color="#111111",
        order=1,
        visible=True,
        locked=False,
        parent_layer_id=None,
        source_pipeline={"block_id": "old"},
        created_at=datetime.now(timezone.utc),
        state_flags={},
        provenance={},
    )

    updated = runtime_layer_record(layer, existing=existing)

    assert updated.name == "Events"
    assert updated.color == "#00ff00"
    assert updated.order == 0
    assert updated.visible is False
    assert updated.locked is True
    assert updated.state_flags["stale"] is True
    assert updated.state_flags["manually_modified"] is True
    assert updated.state_flags["stale_reason"] == "source changed"
    assert updated.state_flags["manual_kind"] == "event"
    assert updated.state_flags["take_lanes_expanded"] is True
    assert updated.provenance == {
        "source_layer_id": "source_audio",
        "source_song_version_id": "song_version_source",
        "source_run_id": "run_123",
        "pipeline_id": "pipeline_drums",
        "output_name": "drums",
    }
    assert updated.source_pipeline == {
        "block_id": "old",
        "pipeline_id": "pipeline_drums",
        "output_name": "drums",
    }


def test_persisted_take_from_runtime_take_updates_existing_non_event_take_label():
    layer = Layer(
        id=LayerId("layer_audio"),
        timeline_id=TimelineId("timeline_runtime"),
        name="Audio",
        kind=LayerKind.AUDIO,
        order_index=0,
        takes=[Take(id=TakeId("take_audio"), layer_id=LayerId("layer_audio"), name="Drums Stem")],
    )
    existing = PersistedTake.create(
        data=manual_layer_take_data(_event_layer()),
        label="Old Label",
        origin="user",
        is_main=False,
    )

    updated = persisted_take_from_runtime_take(
        layer,
        layer.takes[0],
        existing=existing,
        is_main=True,
    )

    assert updated.label == "Drums Stem"
    assert updated.is_main is True


def _event_layer() -> Layer:
    return Layer(
        id=LayerId("layer_runtime"),
        timeline_id=TimelineId("timeline_runtime"),
        name="Events",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[
            Take(
                id=TakeId("take_main"),
                layer_id=LayerId("layer_runtime"),
                name="Take 1",
                events=[
                    Event(
                        id=EventId("evt_1"),
                        take_id=TakeId("take_main"),
                        start=1.5,
                        end=2.0,
                        label="Kick",
                        color="#ff0000",
                        muted=True,
                        payload_ref="payload://kick",
                        source_event_id="src_evt",
                        parent_event_id="parent_evt",
                    )
                ],
            )
        ],
    )

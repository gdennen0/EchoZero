from __future__ import annotations

from datetime import datetime, timezone

from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, SectionCueId, SongVersionId, TakeId, TimelineId
from echozero.application.timeline.models import (
    Event,
    Layer,
    LayerPresentationHints,
    LayerProvenance,
    SectionCue,
    LayerStatus,
    Take,
    Timeline,
    derive_section_cues_from_layers,
    derive_section_regions,
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
from echozero.ui.qt.app_shell_project_timeline_storage import section_cues_from_take


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
    layer.mixer.output_bus = "outputs_3_4"
    layer.mixer.mute = True
    layer.mixer.solo = True

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
    assert record.state_flags["output_bus"] == "outputs_3_4"
    assert record.state_flags["mute"] is True
    assert record.state_flags["solo"] is True


def test_runtime_take_data_serializes_event_metadata():
    layer = _event_layer()

    data = runtime_take_data(layer, layer.takes[0])

    assert len(data.layers) == 1
    serialized = data.layers[0].events[0]
    assert serialized.id == "evt_1"
    assert serialized.time == 1.5
    assert serialized.duration == 0.5
    assert serialized.classifications == {
        "class": "kick",
        "confidence": 0.91,
        "label": "Kick",
    }
    assert serialized.metadata == {
        "review": {
            "schema": "echozero.event_review.v1",
            "promotion_state": "demoted",
            "review_state": "unreviewed",
        },
        "detection": {
            "schema": "echozero.event_detection.v1",
            "classifier_score": 0.91,
            "positive_threshold": 0.95,
            "threshold_passed": False,
            "source_model": "kick-specialized.manifest.json",
        },
        "cue_number": 1,
        "cue_ref": "Q1A",
        "payload_ref": "payload://kick",
        "color": "#ff0000",
        "notes": "Front fill",
        "muted": True,
    }
    assert serialized.origin == "binary_drum_classify:kick"
    assert serialized.source_event_id == "src_evt"
    assert serialized.parent_event_id == "parent_evt"


def test_runtime_take_data_serializes_marker_layer_events():
    layer = _event_layer()
    layer.kind = LayerKind.EVENT

    data = runtime_take_data(layer, layer.takes[0])

    assert len(data.layers) == 1
    assert data.layers[0].events[0].metadata["cue_number"] == 1
    assert data.layers[0].events[0].classifications["label"] == "Kick"


def test_runtime_take_data_serializes_section_layer_events():
    layer = _event_layer()
    layer.kind = LayerKind.SECTION
    layer.name = "Sections"
    layer.takes[0].events[0].label = "Verse"
    layer.takes[0].events[0].cue_ref = "Q7A"
    layer.takes[0].events[0].notes = "Keep exact"

    data = runtime_take_data(layer, layer.takes[0])

    assert len(data.layers) == 1
    assert data.layers[0].events[0].classifications["label"] == "Verse"
    assert data.layers[0].events[0].metadata["cue_ref"] == "Q7A"
    assert data.layers[0].events[0].metadata["notes"] == "Keep exact"


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


def test_events_from_take_restores_float_cue_number_from_metadata():
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
                            metadata={"cue_number": 9.5},
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
    assert events[0].cue_number == 9.5
    assert events[0].cue_ref == "9.5"


def test_events_from_take_restores_shared_cue_metadata():
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
                            classifications={"label": "Verse"},
                            metadata={
                                "cue_number": 9,
                                "cue_ref": "Q9A",
                                "notes": "Keep exact",
                                "payload_ref": "payload://verse",
                                "color": "#ffaa00",
                            },
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
    assert events[0].cue_ref == "Q9A"
    assert events[0].notes == "Keep exact"
    assert events[0].payload_ref == "payload://verse"
    assert events[0].color == "#ffaa00"


def test_events_from_take_restores_review_and_detection_metadata():
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
                            classifications={
                                "class": "kick",
                                "confidence": 0.31,
                                "label": "Kick",
                            },
                            metadata={
                                "review": {
                                    "schema": "echozero.event_review.v1",
                                    "promotion_state": "demoted",
                                    "review_state": "signed_off",
                                },
                                "detection": {
                                    "schema": "echozero.event_detection.v1",
                                    "classifier_score": 0.31,
                                    "positive_threshold": 0.7,
                                    "threshold_passed": False,
                                },
                            },
                            origin="binary_drum_classify:kick",
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
    assert events[0].origin == "binary_drum_classify:kick"
    assert events[0].classifications["class"] == "kick"
    assert events[0].promotion_state == "demoted"
    assert events[0].review_state == "signed_off"
    assert events[0].detection_metadata["classifier_score"] == 0.31


def test_section_cues_from_take_preserves_cue_refs_and_derives_regions_by_time():
    take = PersistedTake.create(
        data=EventData(
            layers=(
                DomainLayer(
                    id="layer_sections",
                    name="Sections",
                    events=(
                        DomainEvent(
                            id="evt_verse",
                            time=12.0,
                            duration=0.0,
                            classifications={"label": "Verse"},
                            metadata={"cue_ref": "Q7"},
                            origin="user",
                        ),
                        DomainEvent(
                            id="evt_chorus",
                            time=41.0,
                            duration=0.0,
                            classifications={"label": "Chorus"},
                            metadata={"cue_ref": "Q3"},
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

    cues = section_cues_from_take(take)
    regions = derive_section_regions(cues, timeline_end=95.0)

    assert [(cue.cue_ref, cue.start, cue.name) for cue in cues] == [
        ("Q7", 12.0, "Verse"),
        ("Q3", 41.0, "Chorus"),
    ]
    assert [(region.cue_ref, region.start, region.end) for region in regions] == [
        ("Q7", 12.0, 41.0),
        ("Q3", 41.0, 95.0),
    ]


def test_derive_section_cues_from_layers_uses_section_layer_main_take():
    section_layer = Layer(
        id=LayerId("layer_sections"),
        timeline_id=TimelineId("timeline_runtime"),
        name="Sections",
        kind=LayerKind.SECTION,
        order_index=1,
        takes=[
            Take(
                id=TakeId("take_sections"),
                layer_id=LayerId("layer_sections"),
                name="Main",
                events=[
                    Event(
                        id=EventId("section_intro"),
                        take_id=TakeId("take_sections"),
                        start=4.0,
                        end=4.08,
                        label="Intro",
                        cue_ref="Q4",
                    ),
                    Event(
                        id=EventId("section_drop"),
                        take_id=TakeId("take_sections"),
                        start=18.0,
                        end=18.08,
                        label="Drop",
                        cue_ref="Q1B",
                    ),
                ],
            )
        ],
    )
    marker_layer = Layer(
        id=LayerId("layer_marker"),
        timeline_id=TimelineId("timeline_runtime"),
        name="Markers",
        kind=LayerKind.EVENT,
        order_index=2,
        takes=[
            Take(
                id=TakeId("take_marker"),
                layer_id=LayerId("layer_marker"),
                name="Main",
                events=[
                    Event(
                        id=EventId("marker_misc"),
                        take_id=TakeId("take_marker"),
                        start=9.0,
                        end=9.08,
                        label="Misc",
                        cue_ref="Q99",
                    )
                ],
            )
        ],
    )

    cues = derive_section_cues_from_layers([marker_layer, section_layer])

    assert [(cue.cue_ref, cue.start, cue.name) for cue in cues] == [
        ("Q4", 4.0, "Intro"),
        ("Q1B", 18.0, "Drop"),
    ]


def test_manual_layer_take_data_uses_first_take_only():
    layer = _event_layer()

    data = manual_layer_take_data(layer)

    assert data.layers[0].events[0].id == "evt_1"


def test_runtime_layer_record_updates_state_flags_and_provenance():
    layer = _event_layer()
    layer.status = LayerStatus(stale=True, manually_modified=True, stale_reason="source changed")
    layer.mixer.output_bus = "outputs_1_2"
    layer.mixer.mute = True
    layer.mixer.solo = True
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
        state_flags={"output_bus": "outputs_3_4"},
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
    assert updated.state_flags["output_bus"] == "outputs_1_2"
    assert updated.state_flags["mute"] is True
    assert updated.state_flags["solo"] is True
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


def test_runtime_layer_record_clears_output_bus_state_flag_when_layer_has_no_route():
    layer = _event_layer()
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
        source_pipeline=None,
        created_at=datetime.now(timezone.utc),
        state_flags={"output_bus": "outputs_3_4", "mute": True, "solo": True},
        provenance={},
    )

    updated = runtime_layer_record(layer, existing=existing)

    assert "output_bus" not in updated.state_flags
    assert "mute" not in updated.state_flags
    assert "solo" not in updated.state_flags


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
                        origin="binary_drum_classify:kick",
                        classifications={
                            "class": "kick",
                            "confidence": 0.91,
                        },
                        metadata={
                            "review": {
                                "schema": "echozero.event_review.v1",
                                "promotion_state": "demoted",
                                "review_state": "unreviewed",
                            },
                            "detection": {
                                "schema": "echozero.event_detection.v1",
                                "classifier_score": 0.91,
                                "positive_threshold": 0.95,
                                "threshold_passed": False,
                                "source_model": "kick-specialized.manifest.json",
                            },
                        },
                        label="Kick",
                        cue_ref="Q1A",
                        color="#ff0000",
                        notes="Front fill",
                        muted=True,
                        payload_ref="payload://kick",
                        source_event_id="src_evt",
                        parent_event_id="parent_evt",
                    )
                ],
            )
        ],
    )

from echozero.application.session.models import (
    BatchTransferPlanRowState,
    BatchTransferPlanState,
    ManualPullFlowState,
    ManualPushFlowState,
    Session,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, ProjectId, SessionId, SongId, SongVersionId, TakeId, TimelineId
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.models import Event, Layer, LayerProvenance, LayerStatus, Take, Timeline


def _event(event_id: str, take_id: str, start: float) -> Event:
    return Event(
        id=EventId(event_id),
        take_id=TakeId(take_id),
        start=start,
        end=start + 0.2,
        label=event_id,
    )


def test_assembler_uses_main_take_as_truth_even_when_alt_take_selected():
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_1"),
        name="Main",
        events=[_event("main_a", "take_main", 1.0), _event("main_b", "take_main", 2.0)],
    )
    alt_take = Take(
        id=TakeId("take_alt"),
        layer_id=LayerId("layer_1"),
        name="Take 2",
        events=[_event("alt_a", "take_alt", 4.0)],
    )

    layer = Layer(
        id=LayerId("layer_1"),
        timeline_id=TimelineId("timeline_1"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[main_take, alt_take],
    )
    timeline = Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[layer],
    )
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_take_id = alt_take.id

    session = Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=timeline.id,
    )

    assembled = TimelineAssembler().assemble(timeline, session)
    assembled_layer = assembled.layers[0]

    assert assembled_layer.main_take_id == main_take.id
    assert [str(e.event_id) for e in assembled_layer.events] == ["main_a", "main_b"]
    assert len(assembled_layer.takes) == 1
    assert assembled_layer.takes[0].take_id == alt_take.id


def test_assembler_maps_application_backed_status_and_provenance_to_presentation():
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_1"),
        name="Main",
        source_ref="fallback-source",
        events=[_event("main_a", "take_main", 1.0)],
    )
    layer = Layer(
        id=LayerId("layer_1"),
        timeline_id=TimelineId("timeline_1"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[main_take],
        status=LayerStatus(
            stale=True,
            manually_modified=True,
            stale_reason="Upstream main changed",
        ),
        provenance=LayerProvenance(
            source_layer_id=LayerId("layer_source"),
            source_song_version_id=SongVersionId("version_source"),
            source_run_id="run_1234",
            pipeline_id="stem_separation",
            output_name="drums",
        ),
    )
    timeline = Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[layer],
    )
    session = Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=timeline.id,
    )

    assembled = TimelineAssembler().assemble(timeline, session)
    status = assembled.layers[0].status

    assert status.stale is True
    assert status.manually_modified is True
    assert status.stale_reason == "Upstream main changed"
    assert status.source_label == "stem_separation · drums"
    assert status.source_layer_id == "layer_source"
    assert status.source_song_version_id == "version_source"
    assert status.pipeline_id == "stem_separation"
    assert status.output_name == "drums"
    assert status.source_run_id == "run_1234"


def test_assembler_uses_take_source_ref_when_structured_provenance_is_missing():
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_1"),
        name="Main",
        source_ref="imported-track",
        events=[_event("main_a", "take_main", 1.0)],
    )
    layer = Layer(
        id=LayerId("layer_1"),
        timeline_id=TimelineId("timeline_1"),
        name="Song",
        kind=LayerKind.AUDIO,
        order_index=0,
        takes=[main_take],
    )
    timeline = Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[layer],
    )
    session = Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=timeline.id,
    )

    assembled = TimelineAssembler().assemble(timeline, session)

    assert assembled.layers[0].status.source_label == "imported-track"


def test_assembler_maps_sync_target_and_batch_transfer_plan_to_presentation():
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_1"),
        name="Main",
        events=[_event("main_a", "take_main", 1.0)],
    )
    layer = Layer(
        id=LayerId("layer_1"),
        timeline_id=TimelineId("timeline_1"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[main_take],
    )
    layer.sync.ma3_track_coord = "tc1_tg2_tr3"
    timeline = Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[layer],
    )
    session = Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=timeline.id,
        manual_push_flow=ManualPushFlowState(push_mode_active=True),
        manual_pull_flow=ManualPullFlowState(
            workspace_active=True,
            selected_source_track_coords=["tc1_tg2_tr5"],
            active_source_track_coord="tc1_tg2_tr5",
            source_track_coord="tc1_tg2_tr5",
            selected_ma3_event_ids=["ma3_evt_1"],
            selected_ma3_event_ids_by_track={"tc1_tg2_tr5": ["ma3_evt_1"]},
            target_layer_id_by_source_track={"tc1_tg2_tr5": LayerId("layer_1")},
        ),
        batch_transfer_plan=BatchTransferPlanState(
            plan_id="plan_123",
            operation_type="mixed",
            rows=[
                BatchTransferPlanRowState(
                    row_id="row_1",
                    direction="push",
                    source_label="Kick",
                    target_label="Track 3",
                    source_layer_id=LayerId("layer_1"),
                    target_track_coord="tc1_tg2_tr3",
                    selected_event_ids=[EventId("main_a"), EventId("main_b")],
                    selected_count=2,
                    status="ready",
                ),
                BatchTransferPlanRowState(
                    row_id="row_2",
                    direction="pull",
                    source_label="Track 5",
                    target_label="Snare",
                    source_track_coord="tc1_tg2_tr5",
                    target_layer_id=LayerId("layer_1"),
                    selected_ma3_event_ids=["ma3_evt_1"],
                    selected_count=1,
                    status="blocked",
                    issue="Target layer required",
                ),
            ],
            ready_count=1,
            blocked_count=1,
        ),
    )

    assembled = TimelineAssembler().assemble(timeline, session)

    assert assembled.layers[0].sync_target_label == "tc1_tg2_tr3"
    assert assembled.layers[0].push_target_label == "Track 3"
    assert assembled.layers[0].push_selection_count == 2
    assert assembled.layers[0].push_row_status == "ready"
    assert assembled.layers[0].pull_target_label == "Snare"
    assert assembled.layers[0].pull_selection_count == 1
    assert assembled.layers[0].pull_row_status == "blocked"
    assert assembled.batch_transfer_plan is not None
    assert assembled.batch_transfer_plan.plan_id == "plan_123"
    assert assembled.batch_transfer_plan.operation_type == "mixed"
    assert assembled.batch_transfer_plan.ready_count == 1
    assert assembled.batch_transfer_plan.blocked_count == 1
    assert assembled.batch_transfer_plan.rows[0].row_id == "row_1"
    assert assembled.batch_transfer_plan.rows[0].direction == "push"
    assert assembled.batch_transfer_plan.rows[0].source_layer_id == LayerId("layer_1")
    assert assembled.batch_transfer_plan.rows[0].selected_count == 2
    assert assembled.batch_transfer_plan.rows[1].source_track_coord == "tc1_tg2_tr5"
    assert assembled.batch_transfer_plan.rows[1].target_layer_id == LayerId("layer_1")
    assert assembled.batch_transfer_plan.rows[1].selected_ma3_event_ids == ["ma3_evt_1"]
    assert assembled.batch_transfer_plan.rows[1].issue == "Target layer required"

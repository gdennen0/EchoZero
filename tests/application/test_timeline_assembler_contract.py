from echozero.application.session.models import (
    BatchTransferPlanRowState,
    BatchTransferPlanState,
    ManualPullFlowState,
    ManualPushFlowState,
    Session,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import (
    EventId,
    LayerId,
    ProjectId,
    RegionId,
    SectionCueId,
    SessionId,
    SongId,
    SongVersionId,
    TakeId,
    TimelineId,
)
from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.models import (
    Event,
    EventRef,
    Layer,
    LayerProvenance,
    SectionCue,
    LayerStatus,
    Take,
    Timeline,
    TimelineRegion,
    derive_section_regions,
)
from echozero.application.timeline.pipeline_run_service import PipelineRunState


def _event(event_id: str, take_id: str, start: float) -> Event:
    return Event(
        id=EventId(event_id),
        take_id=TakeId(take_id),
        start=start,
        end=start + 0.2,
        label=event_id,
    )


def test_assembler_filters_pipeline_banner_to_active_song_version():
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
        pipeline_runs={
            "run_version_1": PipelineRunState(
                run_id="run_version_1",
                action_id="timeline.extract_stems",
                workflow_id="layer.audio.extract_stems",
                display_label="Extract Stems",
                object_id="source_audio",
                object_type="layer",
                source_layer_id="source_audio",
                song_id="song_1",
                song_version_id="version_1",
                status="running",
                message="Executing pipeline",
                percent=0.2,
                started_at=10.0,
            ),
            "run_version_2": PipelineRunState(
                run_id="run_version_2",
                action_id="timeline.extract_stems",
                workflow_id="layer.audio.extract_stems",
                display_label="Extract Stems",
                object_id="source_audio",
                object_type="layer",
                source_layer_id="source_audio",
                song_id="song_1",
                song_version_id="version_2",
                status="running",
                message="Executing pipeline",
                percent=0.6,
                started_at=20.0,
            ),
        },
    )

    assembled = TimelineAssembler().assemble(timeline, session)

    assert assembled.pipeline_run_banner is not None
    assert assembled.pipeline_run_banner.run_id == "run_version_1"
    assert assembled.pipeline_run_banner.percent == 0.2


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
    timeline.playback_target.layer_id = layer.id
    timeline.playback_target.take_id = main_take.id

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
    assert assembled.active_playback_layer_id == layer.id
    assert assembled.active_playback_take_id == main_take.id
    assert assembled_layer.is_selected is True
    assert assembled_layer.is_playback_active is True
    assert len(assembled_layer.takes) == 1
    assert assembled_layer.takes[0].take_id == alt_take.id
    assert assembled_layer.takes[0].is_selected is True
    assert assembled_layer.takes[0].is_playback_active is False


def test_assembler_preserves_independent_selection_and_playback_target_fields():
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_1"),
        name="Main",
        events=[_event("main_a", "take_main", 1.0)],
    )
    alt_take = Take(
        id=TakeId("take_alt"),
        layer_id=LayerId("layer_1"),
        name="Alt",
        events=[_event("alt_a", "take_alt", 2.0)],
    )
    other_take = Take(
        id=TakeId("take_other_main"),
        layer_id=LayerId("layer_2"),
        name="Main",
        events=[_event("other_a", "take_other_main", 3.0)],
    )
    selected_layer = Layer(
        id=LayerId("layer_1"),
        timeline_id=TimelineId("timeline_1"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[main_take, alt_take],
    )
    playback_layer = Layer(
        id=LayerId("layer_2"),
        timeline_id=TimelineId("timeline_1"),
        name="Snare",
        kind=LayerKind.EVENT,
        order_index=1,
        takes=[other_take],
    )
    timeline = Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[selected_layer, playback_layer],
    )
    timeline.selection.selected_layer_id = selected_layer.id
    timeline.selection.selected_layer_ids = [selected_layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.playback_target.layer_id = playback_layer.id
    timeline.playback_target.take_id = other_take.id

    session = Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=timeline.id,
    )

    assembled = TimelineAssembler().assemble(timeline, session)

    assert assembled.selected_layer_id == selected_layer.id
    assert assembled.selected_take_id == alt_take.id
    assert assembled.active_playback_layer_id == playback_layer.id
    assert assembled.active_playback_take_id == other_take.id
    assert assembled.layers[0].is_selected is True
    assert assembled.layers[0].is_playback_active is False
    assert assembled.layers[1].is_selected is False
    assert assembled.layers[1].is_playback_active is True


def test_assembler_projects_playback_output_channel_count():
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_1"),
        name="Main",
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
    session.playback_state.output_channels = 4

    assembled = TimelineAssembler().assemble(timeline, session)

    assert assembled.playback_output_channels == 4


def test_assembler_projects_event_cue_numbers_into_presentation():
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_marker"),
        name="Main",
        events=[
            Event(
                id=EventId("marker_1"),
                take_id=TakeId("take_main"),
                start=1.0,
                end=1.08,
                cue_number=17,
                label="Verse",
            )
        ],
    )
    layer = Layer(
        id=LayerId("layer_marker"),
        timeline_id=TimelineId("timeline_1"),
        name="Markers",
        kind=LayerKind.MARKER,
        order_index=0,
        takes=[main_take],
    )
    timeline = Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[layer],
    )
    session = Session(
        id=SessionId("session_marker"),
        project_id=ProjectId("project_marker"),
        active_song_id=SongId("song_marker"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=timeline.id,
    )

    assembled = TimelineAssembler().assemble(timeline, session)

    assert len(assembled.layers) == 1
    assert assembled.layers[0].events[0].cue_number == 17
    assert assembled.layers[0].events[0].label == "Verse"


def test_derive_section_regions_uses_time_order_and_preserves_nonsequential_cue_refs():
    cues = [
        SectionCue(
            id=SectionCueId("section_cue_q7"),
            start=12.0,
            cue_ref="Q7",
            name="Verse",
        ),
        SectionCue(
            id=SectionCueId("section_cue_q3"),
            start=41.0,
            cue_ref="Q3",
            name="Chorus",
        ),
        SectionCue(
            id=SectionCueId("section_cue_q9"),
            start=70.0,
            cue_ref="Q9",
            name="Breakdown",
        ),
    ]

    regions = derive_section_regions(cues, timeline_end=100.0)

    assert [(region.cue_ref, region.start, region.end) for region in regions] == [
        ("Q7", 12.0, 41.0),
        ("Q3", 41.0, 70.0),
        ("Q9", 70.0, 100.0),
    ]


def test_assembler_projects_section_cues_and_derived_regions_without_fake_prefirst_gap():
    timeline = Timeline(
        id=TimelineId("timeline_sections"),
        song_version_id=SongVersionId("version_sections"),
        end=95.0,
        section_cues=[
            SectionCue(
                id=SectionCueId("section_cue_q7"),
                start=12.0,
                cue_ref="Q7",
                name="Verse",
                color="#ffcc00",
                notes="Starts after intro",
                payload_ref="ma3://cue/Q7",
            ),
            SectionCue(
                id=SectionCueId("section_cue_q3"),
                start=41.0,
                cue_ref="Q3",
                name="Chorus",
            ),
        ],
    )
    session = Session(
        id=SessionId("session_sections"),
        project_id=ProjectId("project_sections"),
        active_song_id=SongId("song_sections"),
        active_song_version_id=SongVersionId("version_sections"),
        active_timeline_id=timeline.id,
    )

    assembled = TimelineAssembler().assemble(timeline, session)

    assert [(cue.cue_ref, cue.start, cue.name) for cue in assembled.section_cues] == [
        ("Q7", 12.0, "Verse"),
        ("Q3", 41.0, "Chorus"),
    ]
    assert [(region.cue_ref, region.start, region.end) for region in assembled.section_regions] == [
        ("Q7", 12.0, 41.0),
        ("Q3", 41.0, 95.0),
    ]
    assert all(region.start >= 12.0 for region in assembled.section_regions)


def test_assembler_projects_regions_in_sorted_order_with_selection_state():
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
    timeline = Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[layer],
        regions=[
            TimelineRegion(
                id=RegionId("region_b"),
                start=4.0,
                end=5.0,
                label="Chorus",
                color="#aaccee",
                order_index=1,
                kind="song",
            ),
            TimelineRegion(
                id=RegionId("region_a"),
                start=1.0,
                end=2.0,
                label="Verse",
                order_index=0,
                kind="structure",
            ),
        ],
    )
    timeline.selection.selected_region_id = RegionId("region_b")

    session = Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=timeline.id,
    )

    assembled = TimelineAssembler().assemble(timeline, session)

    assert [region.region_id for region in assembled.regions] == [
        RegionId("region_a"),
        RegionId("region_b"),
    ]
    assert assembled.selected_region_id == RegionId("region_b")
    assert assembled.regions[0].is_selected is False
    assert assembled.regions[1].is_selected is True
    assert assembled.regions[1].color == "#aaccee"
    assert assembled.regions[0].kind == "structure"


def test_assembler_marks_take_lane_selection_and_playback_target_independently():
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_1"),
        name="Main",
        events=[_event("main_a", "take_main", 1.0)],
    )
    selected_take = Take(
        id=TakeId("take_selected"),
        layer_id=LayerId("layer_1"),
        name="Selected",
        events=[_event("selected_a", "take_selected", 2.0)],
    )
    active_take = Take(
        id=TakeId("take_active"),
        layer_id=LayerId("layer_1"),
        name="Active",
        events=[_event("active_a", "take_active", 3.0)],
    )
    layer = Layer(
        id=LayerId("layer_1"),
        timeline_id=TimelineId("timeline_1"),
        name="Kick",
        kind=LayerKind.EVENT,
        order_index=0,
        takes=[main_take, selected_take, active_take],
    )
    timeline = Timeline(
        id=TimelineId("timeline_1"),
        song_version_id=SongVersionId("version_1"),
        layers=[layer],
    )
    timeline.selection.selected_layer_id = layer.id
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = selected_take.id
    timeline.playback_target.layer_id = layer.id
    timeline.playback_target.take_id = active_take.id

    session = Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=timeline.id,
    )

    assembled = TimelineAssembler().assemble(timeline, session)
    assembled_layer = assembled.layers[0]
    selected_lane = next(take for take in assembled_layer.takes if take.take_id == selected_take.id)
    active_lane = next(take for take in assembled_layer.takes if take.take_id == active_take.id)

    assert assembled_layer.is_selected is True
    assert assembled_layer.is_playback_active is True
    assert selected_lane.is_selected is True
    assert selected_lane.is_playback_active is False
    assert active_lane.is_selected is False
    assert active_lane.is_playback_active is True


def test_assembler_adds_selection_to_main_action_when_take_has_selected_events():
    main_take = Take(
        id=TakeId("take_main"),
        layer_id=LayerId("layer_1"),
        name="Main",
        events=[_event("main_a", "take_main", 1.0)],
    )
    alt_take = Take(
        id=TakeId("take_alt"),
        layer_id=LayerId("layer_1"),
        name="Alt",
        events=[_event("alt_a", "take_alt", 2.0)],
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
    timeline.selection.selected_layer_ids = [layer.id]
    timeline.selection.selected_take_id = alt_take.id
    timeline.selection.selected_event_refs = [
        EventRef(layer_id=layer.id, take_id=alt_take.id, event_id=alt_take.events[0].id)
    ]
    timeline.selection.selected_event_ids = [alt_take.events[0].id]

    session = Session(
        id=SessionId("session_1"),
        project_id=ProjectId("project_1"),
        active_song_id=SongId("song_1"),
        active_song_version_id=SongVersionId("version_1"),
        active_timeline_id=timeline.id,
    )

    assembled = TimelineAssembler().assemble(timeline, session)
    take_lane = assembled.layers[0].takes[0]

    assert {action.action_id for action in take_lane.actions} >= {
        "add_selection_to_main",
        "overwrite_main",
        "merge_main",
        "delete_take",
    }


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
            import_mode="new_take",
            import_mode_by_source_track={"tc1_tg2_tr5": "new_take"},
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
                    import_mode="new_take",
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
    assert assembled.batch_transfer_plan.rows[1].import_mode == "new_take"
    assert assembled.batch_transfer_plan.rows[1].selected_ma3_event_ids == ["ma3_evt_1"]
    assert assembled.batch_transfer_plan.rows[1].issue == "Target layer required"


def test_assembler_coerces_legacy_string_sync_mode_to_enum():
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
    layer.sync.mode = "ma3"
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

    assert assembled.layers[0].sync_mode is not None
    assert str(assembled.layers[0].sync_mode.value) == "ma3"

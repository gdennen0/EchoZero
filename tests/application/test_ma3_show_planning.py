"""MA3 show planning contract tests.
Exists to pin clean-sheet song identity, CSV import, and direct automator events.
Connects new MA3 show platform primitives to focused application coverage.
"""

from pathlib import Path

from echozero.application.ma3_show import (
    EchoZeroShowPlan,
    MA3ExecutorPlacement,
    MA3MasterOutputAssignment,
    MA3ObjectMapping,
    MA3SongManagerSnapshot,
    SequenceBlockPolicy,
    Setlist,
    ShowSong,
    SongSection,
    allocate_song_sequence_mappings,
    automator_event_to_ma3_snapshot,
    build_automator_events,
    build_show_plan,
    command_for_sequence_cue,
    diff_show_plan_snapshot,
    import_setlist_csv,
    preflight_show_plan,
)


def test_sequence_mapping_is_canonical_and_executor_is_view_only() -> None:
    mapping = MA3ObjectMapping(
        main_sequence_no=1100,
        sequence_range_start=1100,
        sequence_range_end=1199,
        executor_placements=(MA3ExecutorPlacement(page_no=3, executor_no=215),),
    )

    moved = MA3ObjectMapping(
        main_sequence_no=1100,
        sequence_range_start=1100,
        sequence_range_end=1199,
        executor_placements=(MA3ExecutorPlacement(page_no=7, executor_no=101),),
    )

    assert mapping.canonical_identity == "sequence:1100"
    assert moved.canonical_identity == mapping.canonical_identity
    assert moved.executor_placements[0].executor_no == 101


def test_sequence_block_policy_allocates_song_ranges() -> None:
    setlist = Setlist(
        id="setlist-1",
        name="Show",
        songs=(
            ShowSong(id="song-a", title="Song A", order=0),
            ShowSong(id="song-b", title="Song B", order=1),
        ),
    )

    planned = allocate_song_sequence_mappings(
        setlist,
        policy=SequenceBlockPolicy(first_sequence_no=1100, block_size=100),
    )

    assert planned.songs[0].ma3_mapping is not None
    assert planned.songs[0].ma3_mapping.main_sequence_no == 1100
    assert planned.songs[0].ma3_mapping.sequence_range_end == 1199
    assert planned.songs[1].ma3_mapping is not None
    assert planned.songs[1].ma3_mapping.main_sequence_no == 1200


def test_automator_events_use_direct_sequence_cue_commands() -> None:
    setlist = Setlist(
        id="setlist-1",
        name="Show",
        songs=(
            ShowSong(
                id="song-a",
                title="Song A",
                order=0,
                ma3_mapping=MA3ObjectMapping(
                    main_sequence_no=1100,
                    sequence_range_start=1100,
                    sequence_range_end=1199,
                ),
                sections=(
                    SongSection(
                        id="verse",
                        label="Verse",
                        start_seconds=12.5,
                        cue_ref="5",
                    ),
                ),
            ),
        ),
    )

    events = build_automator_events(setlist)

    assert command_for_sequence_cue(1100, "5") == "Goto Sequence 1100 Cue 5"
    assert len(events) == 1
    assert events[0].target_sequence_no == 1100
    assert events[0].command == "Goto Sequence 1100 Cue 5"
    assert "Record Timecode" not in events[0].command


def test_csv_import_builds_setlist_sections_and_lossless_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "show.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Order,Song,Artist,BPM,Section,Start Time,Cue,Notes,Designer Column",
                "1,Open,Artist A,128,Intro,0,1,Start note,keep-me",
                "1,Open,Artist A,128,Chorus,64,2,Chorus note,keep-me-too",
                "2,Close,Artist B,95,,,,,tail",
            ]
        ),
        encoding="utf-8",
    )

    result = import_setlist_csv(csv_path, setlist_id="setlist-csv")

    assert result.warnings == ()
    assert [song.title for song in result.setlist.songs] == ["Open", "Close"]
    assert result.setlist.songs[0].bpm == 128
    assert len(result.setlist.songs[0].sections) == 2
    assert result.setlist.songs[0].sections[1].cue_ref == "2"
    rows = result.setlist.songs[0].metadata["spreadsheet_rows"]
    assert rows[0]["Designer Column"] == "keep-me"


def test_master_outputs_are_single_channel_assignments() -> None:
    master = MA3MasterOutputAssignment(
        id="vox-master",
        label="Vox Master",
        universe_no=1,
        channel_no=42,
    )
    planned = allocate_song_sequence_mappings(
        Setlist(id="setlist-1", name="Show", songs=(ShowSong(id="song-a", title="A"),))
    )

    payload = EchoZeroShowPlan(
        project_id="project-1",
        setlist=planned,
        master_outputs=(master,),
    ).to_payload()

    assert master.channel_address == "universe:1/channel:42"
    assert payload["master_outputs"][0]["channel_no"] == 42
    assert "executor" not in payload["master_outputs"][0]


def test_show_plan_payload_is_json_compatible() -> None:
    planned = allocate_song_sequence_mappings(
        Setlist(id="setlist-1", name="Show", songs=(ShowSong(id="song-a", title="A"),))
    )
    events = build_automator_events(
        Setlist(
            id=planned.id,
            name=planned.name,
            songs=(
                ShowSong(
                    id="song-a",
                    title="A",
                    ma3_mapping=planned.songs[0].ma3_mapping,
                    sections=(SongSection(id="cue-1", label="Cue 1", start_seconds=0, cue_ref="1"),),
                ),
            ),
        )
    )

    payload = EchoZeroShowPlan(
        project_id="project-1",
        setlist=planned,
        automator_events=events,
    ).to_payload()

    assert payload["project_id"] == "project-1"
    assert payload["setlist"]["songs"][0]["ma3_mapping"]["main_sequence_no"] == 1100
    assert payload["automator_events"][0]["command"] == "Goto Sequence 1100 Cue 1"


def test_show_plan_preflight_rejects_cursor_or_record_timecode_commands() -> None:
    setlist = Setlist(
        id="setlist-1",
        name="Show",
        songs=(
            ShowSong(
                id="song-a",
                title="A",
                ma3_mapping=MA3ObjectMapping(
                    main_sequence_no=1100,
                    sequence_range_start=1100,
                    sequence_range_end=1199,
                ),
                sections=(SongSection(id="cue-1", label="Cue 1", start_seconds=0, cue_ref="1"),),
            ),
        ),
    )
    plan = build_show_plan(project_id="project-1", setlist=setlist)
    bad_event = plan.automator_events[0].__class__(
        id="bad",
        song_id="song-a",
        section_id="cue-1",
        start_seconds=0,
        target_sequence_no=1100,
        cue_ref="1",
        command="Cursor Action Record Timecode",
        label="Bad",
    )
    bad_plan = EchoZeroShowPlan(
        project_id=plan.project_id,
        setlist=plan.setlist,
        automator_events=(bad_event,),
    )

    preflight = preflight_show_plan(bad_plan)

    assert not preflight.is_ready
    assert any("Record Timecode" in issue for issue in preflight.issues)
    assert any("cursor" in issue.lower() for issue in preflight.issues)


def test_snapshot_diff_uses_sequence_identity_not_executor_identity() -> None:
    setlist = Setlist(
        id="setlist-1",
        name="Show",
        songs=(
            ShowSong(
                id="song-a",
                title="A",
                ma3_mapping=MA3ObjectMapping(
                    main_sequence_no=1100,
                    sequence_range_start=1100,
                    sequence_range_end=1199,
                    executor_placements=(MA3ExecutorPlacement(page_no=1, executor_no=215),),
                ),
                sections=(SongSection(id="cue-1", label="Cue 1", start_seconds=0, cue_ref="1"),),
            ),
        ),
    )
    plan = build_show_plan(project_id="project-1", setlist=setlist)
    moved_executor_snapshot = MA3SongManagerSnapshot(
        project_id="project-1",
        songs=(
            {
                "main_sequence_no": 1100,
                "executor_placements": [{"page_no": 2, "executor_no": 101}],
            },
        ),
        automator_events=({"id": "song-a:cue-1:automator"},),
    )

    diff = diff_show_plan_snapshot(plan, moved_executor_snapshot)

    assert diff.missing_sequences == ()
    assert diff.missing_automator_events == ()
    assert diff.placement_changes == ("Sequence 1100 placement 1.215",)


def test_automator_event_adapter_feeds_existing_direct_ma3_event_dto() -> None:
    setlist = Setlist(
        id="setlist-1",
        name="Show",
        songs=(
            ShowSong(
                id="song-a",
                title="A",
                ma3_mapping=MA3ObjectMapping(
                    main_sequence_no=1100,
                    sequence_range_start=1100,
                    sequence_range_end=1199,
                ),
                sections=(SongSection(id="cue-1", label="Cue 1", start_seconds=4, cue_ref="1"),),
            ),
        ),
    )
    event = build_show_plan(project_id="project-1", setlist=setlist).automator_events[0]

    snapshot = automator_event_to_ma3_snapshot(event)

    assert snapshot.event_id == "song-a:cue-1:automator"
    assert snapshot.start == 4
    assert snapshot.cmd == "Goto Sequence 1100 Cue 1"
    assert snapshot.payload_ref == "echozero://automator-events/song-a:cue-1:automator"

from echozero.application.session.models import ManualPullEventOption
from echozero.application.sync.diff_service import SyncDiffRow, SyncDiffService, SyncDiffSummary
from echozero.application.timeline.models import Event


def test_push_preview_rows_are_deterministic_and_added_only():
    service = SyncDiffService()

    summary, rows = service.build_push_preview_rows(
        selected_events=[
            Event(id="evt_b", take_id="take_1", start=2.0, end=2.5, label="Beta"),
            Event(id="evt_a", take_id="take_1", start=1.0, end=1.25, label="Alpha"),
            Event(id="evt_c", take_id="take_1", start=2.0, end=2.5, label="Alpha"),
        ],
        target_track_name="Track 3",
        target_track_coord="tc1_tg2_tr3",
    )

    assert summary == SyncDiffSummary(
        added_count=3,
        removed_count=0,
        modified_count=0,
        unchanged_count=0,
        row_count=3,
    )
    assert rows == [
        SyncDiffRow(
            row_id="evt_a",
            action="add",
            start=1.0,
            end=1.25,
            label="Alpha",
            before="Not present in MA3 target",
            after="Track 3 (tc1_tg2_tr3)",
        ),
        SyncDiffRow(
            row_id="evt_c",
            action="add",
            start=2.0,
            end=2.5,
            label="Alpha",
            before="Not present in MA3 target",
            after="Track 3 (tc1_tg2_tr3)",
        ),
        SyncDiffRow(
            row_id="evt_b",
            action="add",
            start=2.0,
            end=2.5,
            label="Beta",
            before="Not present in MA3 target",
            after="Track 3 (tc1_tg2_tr3)",
        ),
    ]


def test_pull_preview_rows_are_deterministic_and_use_resolved_ranges():
    service = SyncDiffService()

    summary, rows = service.build_pull_preview_rows(
        selected_events=[
            ManualPullEventOption(event_id="ma3_evt_3", label="Cue 3"),
            ManualPullEventOption(event_id="ma3_evt_1", label="Cue 1", start=3.0),
            ManualPullEventOption(event_id="ma3_evt_2", label="Cue 2", end=1.0),
        ],
        target_layer_name="Target Layer",
    )

    assert summary == SyncDiffSummary(
        added_count=3,
        removed_count=0,
        modified_count=0,
        unchanged_count=0,
        row_count=3,
    )
    assert rows == [
        SyncDiffRow(
            row_id="ma3_evt_3",
            action="add",
            start=0.0,
            end=0.25,
            label="Cue 3",
            before="Not present in EZ target layer",
            after="Target Layer",
        ),
        SyncDiffRow(
            row_id="ma3_evt_2",
            action="add",
            start=0.75,
            end=1.0,
            label="Cue 2",
            before="Not present in EZ target layer",
            after="Target Layer",
        ),
        SyncDiffRow(
            row_id="ma3_evt_1",
            action="add",
            start=3.0,
            end=3.25,
            label="Cue 1",
            before="Not present in EZ target layer",
            after="Target Layer",
        ),
    ]

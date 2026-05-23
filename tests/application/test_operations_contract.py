"""Shared operation contract tests.
Exists to lock the cross-lane operation state model before runtime migrations.
Connects legacy pipeline progress records to the new public operation contract.
"""

from echozero.application.operations import (
    OperationKind,
    OperationLane,
    OperationState,
    OperationStatus,
    OperationSubject,
    clamp_progress,
    is_active_status,
    is_final_status,
    operation_state_from_ma3_operation_snapshot,
    operation_state_from_progress_state,
    operation_status_from_ma3_status,
    operation_status_from_progress_status,
)
from echozero.application.sync.ma3_push_service import MA3OperationSnapshot
from echozero.application.timeline.operation_progress_service import OperationProgressState


def test_operation_state_normalizes_status_and_progress() -> None:
    state = OperationState(
        operation_id=" op_1 ",
        kind=OperationKind.TRANSPORT,
        lane=OperationLane.TRANSPORT,
        status="running",
        progress=2.0,
        started_at=-2.0,
    )

    assert state.operation_id == "op_1"
    assert state.status is OperationStatus.RUNNING
    assert state.progress == 1.0
    assert state.started_at == 0.0


def test_operation_status_helpers_classify_lifecycle_states() -> None:
    assert is_active_status(OperationStatus.QUEUED)
    assert is_active_status("preparing")
    assert not is_active_status("applied")

    assert is_final_status(OperationStatus.APPLIED)
    assert is_final_status("failed")
    assert not is_final_status("running")


def test_operation_progress_clamps_or_preserves_indeterminate() -> None:
    assert clamp_progress(-0.25) == 0.0
    assert clamp_progress(1.25) == 1.0
    assert clamp_progress(None) is None


def test_legacy_pipeline_statuses_map_to_public_statuses() -> None:
    assert operation_status_from_progress_status("resolving") is OperationStatus.PREPARING
    assert operation_status_from_progress_status("completed") is OperationStatus.APPLIED
    assert operation_status_from_progress_status("persisting") is OperationStatus.PERSISTING


def test_ma3_statuses_map_to_public_statuses() -> None:
    assert operation_status_from_ma3_status("running") is OperationStatus.RUNNING
    assert operation_status_from_ma3_status("success") is OperationStatus.APPLIED
    assert operation_status_from_ma3_status("error") is OperationStatus.FAILED
    assert operation_status_from_ma3_status("cancelled") is OperationStatus.CANCELLED


def test_operation_state_rejects_unknown_kind() -> None:
    try:
        OperationState(
            operation_id="op_1",
            kind="not-a-kind",
            lane=OperationLane.APP,
            status=OperationStatus.RUNNING,
        )
    except ValueError:
        return
    raise AssertionError("unknown operation kind should fail fast")


def test_pipeline_progress_state_converts_to_operation_state() -> None:
    legacy = OperationProgressState(
        operation_id="operation_1",
        kind="pipeline",
        action_id="timeline.extract_stems",
        workflow_id="workflow:extract",
        display_label="Extract Stems",
        object_id="layer_source",
        object_type="layer",
        source_layer_id="layer_source",
        song_id="song_1",
        song_version_id="version_1",
        status="completed",
        message="Complete",
        fraction_complete=1.0,
        started_at=10.0,
        finished_at=12.0,
        output_layer_ids=("layer_output",),
    )

    state = operation_state_from_progress_state(legacy)

    assert state.kind is OperationKind.PIPELINE
    assert state.lane is OperationLane.PREPARE
    assert state.status is OperationStatus.APPLIED
    assert state.subject == OperationSubject(
        song_id="song_1",
        song_version_id="version_1",
        layer_id="layer_source",
        object_id="layer_source",
        object_type="layer",
        label="Extract Stems",
    )
    assert state.command_name == "timeline.extract_stems"
    assert state.diagnostics["legacy_status"] == "completed"
    assert state.diagnostics["output_layer_ids"] == ("layer_output",)


def test_ma3_operation_snapshot_converts_additively_without_monotonic_public_time() -> None:
    snapshot = MA3OperationSnapshot(
        operation_id="ma3-op-1",
        status="success",
        message="Sent",
        kind="ma3.push",
        started_at=4.0,
        completed_at=5.0,
        result={"ok": True},
    )

    state = operation_state_from_ma3_operation_snapshot(snapshot, observed_at=1000.0)

    assert state.kind is OperationKind.SYNC
    assert state.lane is OperationLane.SYNC
    assert state.status is OperationStatus.APPLIED
    assert state.started_at == 1000.0
    assert state.finished_at == 1000.0
    assert state.command_name == "ma3.push"
    assert state.diagnostics["legacy_started_at_monotonic"] == 4.0
    assert state.diagnostics["legacy_completed_at_monotonic"] == 5.0

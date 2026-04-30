"""Operation progress contract tests.
Exists to verify the canonical application progress shapes and invariants.
Connects producer updates and operation state visibility rules through one typed contract.
"""

from echozero.application.progress import (
    ACTIVE_OPERATION_PROGRESS_STATUSES,
    FINAL_OPERATION_PROGRESS_STATUSES,
    OperationProgressUpdate,
)


def test_operation_progress_update_clamps_fraction_to_zero_and_one() -> None:
    low = OperationProgressUpdate(
        stage="executing_pipeline",
        message="running",
        fraction_complete=-0.5,
    )
    high = OperationProgressUpdate(
        stage="executing_pipeline",
        message="running",
        fraction_complete=2.0,
    )

    assert low.fraction_complete == 0.0
    assert high.fraction_complete == 1.0


def test_operation_progress_update_accepts_indeterminate_fraction() -> None:
    update = OperationProgressUpdate(
        stage="loading_configuration",
        message="Loading",
        fraction_complete=None,
    )

    assert update.fraction_complete is None


def test_operation_progress_status_sets_have_expected_membership() -> None:
    assert "running" in ACTIVE_OPERATION_PROGRESS_STATUSES
    assert "persisting" in ACTIVE_OPERATION_PROGRESS_STATUSES
    assert "failed" not in ACTIVE_OPERATION_PROGRESS_STATUSES

    assert "completed" in FINAL_OPERATION_PROGRESS_STATUSES
    assert "failed" in FINAL_OPERATION_PROGRESS_STATUSES
    assert "running" not in FINAL_OPERATION_PROGRESS_STATUSES

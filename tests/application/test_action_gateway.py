"""Action gateway contract tests.
Exists to lock typed command dispatch metadata before gateway implementation work.
Connects route defaults, coalescing keys, and bounded operation snapshots.
"""

from echozero.application.actions import (
    ActionCoalescing,
    ActionCoalescingMode,
    ActionCommand,
    ActionGateway,
    ActionPriority,
    ActionResult,
    ActionResultStatus,
    ActionRoute,
    bounded_operation_snapshot,
    route_action_command,
)
from echozero.application.operations import (
    OperationKind,
    OperationLane,
    OperationSnapshot,
    OperationState,
    OperationStatus,
)


def test_action_command_normalizes_core_dispatch_metadata() -> None:
    command = ActionCommand(
        command_type=" timeline.run_action ",
        lane="prepare",
        priority="user_blocking",
        source="",
        payload={"object_id": "layer_1"},
    )

    assert command.command_type == "timeline.run_action"
    assert command.lane is OperationLane.PREPARE
    assert command.priority is ActionPriority.USER_BLOCKING
    assert command.source == "app"
    assert command.payload == {"object_id": "layer_1"}


def test_action_command_requires_command_type() -> None:
    try:
        ActionCommand(command_type=" ")
    except ValueError:
        return
    raise AssertionError("blank command_type should fail fast")


def test_route_action_command_applies_lane_priority_and_coalescing_defaults() -> None:
    command = ActionCommand(command_type="timeline.selection.changed")
    route = ActionRoute(
        command_type="timeline.selection.changed",
        lane=OperationLane.UI,
        priority=ActionPriority.REALTIME,
        coalescing=ActionCoalescing(
            ActionCoalescingMode.KEEP_LATEST,
            "timeline.selection",
        ),
    )

    routed = route_action_command(command, {route.command_type: route})

    assert routed.lane is OperationLane.UI
    assert routed.priority is ActionPriority.REALTIME
    assert routed.coalescing_key == "timeline.selection"


def test_command_specific_coalescing_overrides_route_default() -> None:
    command = ActionCommand(
        command_type="timeline.viewport.changed",
        coalescing=ActionCoalescing(
            ActionCoalescingMode.REPLACE_PENDING,
            "timeline.viewport:main",
        ),
    )
    route = ActionRoute(
        command_type="timeline.viewport.changed",
        coalescing=ActionCoalescing(
            ActionCoalescingMode.KEEP_LATEST,
            "timeline.viewport",
        ),
    )

    routed = route_action_command(command, {route.command_type: route})

    assert routed.coalescing.mode is ActionCoalescingMode.REPLACE_PENDING
    assert routed.coalescing_key == "timeline.viewport:main"


def test_action_result_exposes_success_states_and_normalizes_snapshot() -> None:
    command = ActionCommand(command_type="timeline.refresh", command_id=" cmd_1 ")
    accepted = ActionResult.accepted(command, operation_id=" operation_1 ")
    rejected = ActionResult.rejected(command, "No active project")

    assert accepted.ok
    assert accepted.command_id == "cmd_1"
    assert accepted.operation_id == "operation_1"
    assert rejected.status is ActionResultStatus.REJECTED
    assert not rejected.ok


def test_bounded_operation_snapshot_keeps_latest_active_and_recent_finals() -> None:
    snapshot = OperationSnapshot(
        revision=7,
        active_operations=(
            _operation("active_old", OperationStatus.RUNNING, started_at=1.0, updated_at=2.0),
            _operation("active_mid", OperationStatus.RUNNING, started_at=2.0, updated_at=3.0),
            _operation("active_new", OperationStatus.RUNNING, started_at=3.0, updated_at=4.0),
        ),
        recent_final_operations=(
            _operation("final_old", OperationStatus.APPLIED, started_at=1.0, finished_at=5.0),
            _operation("final_new", OperationStatus.FAILED, started_at=2.0, finished_at=6.0),
        ),
    )

    bounded = bounded_operation_snapshot(snapshot, active_limit=2, recent_limit=1)

    assert bounded.revision == 7
    assert [state.operation_id for state in bounded.active_operations] == [
        "active_mid",
        "active_new",
    ]
    assert [state.operation_id for state in bounded.recent_final_operations] == ["final_new"]


def test_bounded_operation_snapshot_accepts_zero_limits_and_none() -> None:
    empty = bounded_operation_snapshot(None)
    bounded = bounded_operation_snapshot(
        OperationSnapshot(
            revision=3,
            active_operations=(_operation("active", OperationStatus.RUNNING),),
            recent_final_operations=(_operation("final", OperationStatus.APPLIED),),
        ),
        active_limit=0,
        recent_limit=0,
    )

    assert empty.revision == 0
    assert bounded.active_operations == ()
    assert bounded.recent_final_operations == ()


def test_action_gateway_coalesced_command_completion_is_idempotent() -> None:
    gateway = ActionGateway()
    first = ActionCommand(
        command_type="timeline.seek",
        command_id="seek_1",
        coalescing=ActionCoalescing(ActionCoalescingMode.KEEP_LATEST, "transport.seek"),
    )
    second = ActionCommand(
        command_type="timeline.seek",
        command_id="seek_2",
        coalescing=ActionCoalescing(ActionCoalescingMode.KEEP_LATEST, "transport.seek"),
    )

    gateway.accept(first)
    gateway.accept(second)
    first_final = gateway.complete(first.command_id)
    second_final = gateway.complete(second.command_id)

    assert first_final.status is OperationStatus.CANCELLED
    assert second_final.status is OperationStatus.APPLIED
    assert gateway.coalesced_count == 1


def _operation(
    operation_id: str,
    status: OperationStatus,
    *,
    started_at: float = 1.0,
    updated_at: float = 1.0,
    finished_at: float | None = None,
) -> OperationState:
    return OperationState(
        operation_id=operation_id,
        kind=OperationKind.PIPELINE,
        lane=OperationLane.APP,
        status=status,
        started_at=started_at,
        updated_at=updated_at,
        finished_at=finished_at,
    )

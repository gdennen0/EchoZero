"""Playback coordination contract tests.
Exists to lock transport commands, snapshots, and generation requests before coordinator work.
Connects compact playback projections to the public nonblocking playback seam.
"""

from echozero.application.playback.coordination import (
    PlaybackGenerationState,
    PlaybackGenerationStatus,
    PlaybackGraphRequest,
    TransportCommand,
    TransportCommandAction,
    TransportSnapshot,
)
from echozero.application.playback.audible_intent import audible_intent_from_sync_payload
from echozero.application.playback.models import PlaybackTimingSnapshot
from echozero.application.playback.output_matrix import resolve_output_matrix
from echozero.application.playback.sync_delta import (
    playback_mix_signature,
    playback_structure_signature,
)
from echozero.application.playback.sync_projection import PlaybackSyncPayload
from echozero.ui.qt.timeline.demo_app import build_demo_app


def test_transport_command_normalizes_action_and_position() -> None:
    command = TransportCommand(action="seek", position_seconds=-3.0, source="")

    assert command.action is TransportCommandAction.SEEK
    assert command.position_seconds == 0.0
    assert command.source == "app"
    assert command.monotonic_seconds > 0.0


def test_transport_snapshot_round_trips_legacy_timing_snapshot() -> None:
    timing = PlaybackTimingSnapshot(
        audible_time_seconds=1.25,
        clock_time_seconds=1.5,
        snapshot_monotonic_seconds=10.0,
        is_playing=True,
        sample_position=60000,
        display_label="00:00:01:07",
    )

    snapshot = TransportSnapshot.from_timing_snapshot(timing, generation_id="gen_1")
    restored = snapshot.to_timing_snapshot()

    assert snapshot.generation_id == "gen_1"
    assert restored.audible_time_seconds == 1.25
    assert restored.clock_time_seconds == 1.5
    assert restored.display_label == "00:00:01:07"
    assert restored.is_playing is True


def test_playback_graph_request_uses_compact_sync_payload_signatures() -> None:
    payload = PlaybackSyncPayload.from_presentation(build_demo_app().presentation())
    request = PlaybackGraphRequest.from_sync_payload(payload, request_id="request_1")

    assert request.request_id == "request_1"
    assert request.payload is payload
    assert request.source_signature == playback_structure_signature(payload)
    assert request.mix_signature == playback_mix_signature(payload)


def test_playback_payload_strips_selection_from_audible_intent() -> None:
    presentation = build_demo_app().presentation()
    selected_payload = PlaybackSyncPayload.from_presentation(presentation)

    assert selected_payload.selected_layer_id is None
    assert selected_payload.selected_take_id is None
    assert audible_intent_from_sync_payload(selected_payload).structure_signature == (
        playback_structure_signature(selected_payload)
    )


def test_output_matrix_preserves_invalid_route_intent_as_diagnostics() -> None:
    matrix = resolve_output_matrix(
        {"layer_a": "outputs_1_1,outputs_7_8"},
        hardware_channels=2,
    )

    assert matrix.healthy is False
    assert [assignment.token for assignment in matrix.assignments] == ["outputs_1_1"]
    assert [(issue.owner_id, issue.token, issue.reason) for issue in matrix.issues] == [
        ("layer_a", "outputs_7_8", "outside-hardware")
    ]
    assert matrix.diagnostics_label == (
        "routes-exceed-hardware;routes-degraded:layer_a:outputs_7_8->outside-hardware"
    )


def test_output_matrix_matches_master_and_no_output_route_semantics() -> None:
    matrix = resolve_output_matrix(
        {
            "mastered": "master,outputs_4_4",
            "silent": "none",
        },
        hardware_channels=4,
        default_route="outputs_1_2",
    )

    assert [assignment.token for assignment in matrix.assignments] == [
        "outputs_1_2",
        "outputs_4_4",
    ]
    assert matrix.issues == ()


def test_playback_generation_state_defaults_to_queued() -> None:
    state = PlaybackGenerationState(generation_id="gen_1")

    assert state.status is PlaybackGenerationStatus.QUEUED
    assert state.source_signature == ()
    assert state.diagnostics == {}

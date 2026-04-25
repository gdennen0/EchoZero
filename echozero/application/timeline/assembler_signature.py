"""
Timeline assembler cache signatures for the canonical presentation boundary.
Exists to isolate signature shaping from the public TimelineAssembler facade.
Connects timeline/session mutation surfaces to cached layer presentation reuse.
"""

from echozero.application.session.models import Session
from echozero.application.timeline.assembler_state import (
    AssemblerSignature,
    TimelineAssemblyState,
)
from echozero.application.timeline.models import Event, Layer, Timeline

__all__ = [
    "build_layer_signature",
    "build_session_transfer_signature",
]


def build_layer_signature(
    timeline: Timeline,
    ordered_layers: list[Layer],
    state: TimelineAssemblyState,
    session: Session,
) -> AssemblerSignature:
    """Build the cache signature for layer presentation reuse."""

    layer_sigs: list[AssemblerSignature] = []
    for layer in ordered_layers:
        take_sigs = tuple(
            (
                str(take.id),
                idx == 0,
                take.name,
                take.source_ref,
                _events_signature(take.events),
            )
            for idx, take in enumerate(layer.takes)
        )
        layer_sigs.append(
            (
                str(layer.id),
                int(layer.order_index),
                bool(layer.presentation_hints.expanded),
                bool(layer.presentation_hints.visible),
                bool(layer.presentation_hints.locked),
                layer.presentation_hints.color,
                bool(layer.status.stale),
                bool(layer.status.manually_modified),
                layer.status.stale_reason,
                (
                    str(layer.provenance.source_layer_id)
                    if layer.provenance.source_layer_id is not None
                    else None
                ),
                (
                    str(layer.provenance.source_song_version_id)
                    if layer.provenance.source_song_version_id is not None
                    else None
                ),
                layer.provenance.source_run_id,
                layer.provenance.pipeline_id,
                layer.provenance.output_name,
                bool(layer.mixer.mute),
                bool(layer.mixer.solo),
                float(layer.mixer.gain_db),
                float(layer.mixer.pan),
                str(layer.sync.mode),
                bool(layer.sync.connected),
                layer.sync.target_ref,
                layer.sync.show_manager_block_id,
                layer.sync.ma3_track_coord,
                str(layer.sync.live_sync_state.value),
                layer.sync.live_sync_pause_reason,
                bool(layer.sync.live_sync_divergent),
                take_sigs,
            )
        )

    return (
        str(timeline.id),
        tuple(layer_sigs),
        str(state.selected_layer_id) if state.selected_layer_id is not None else None,
        tuple(str(layer_id) for layer_id in state.selected_layer_ids),
        str(state.selected_take_id) if state.selected_take_id is not None else None,
        (
            str(state.active_playback_layer_id)
            if state.active_playback_layer_id is not None
            else None
        ),
        str(state.active_playback_take_id) if state.active_playback_take_id is not None else None,
        tuple(sorted(state.selected_event_ref_keys)),
        tuple(sorted(str(event_id) for event_id in state.selected_event_ids_set)),
        build_session_transfer_signature(session),
    )


def build_session_transfer_signature(session: Session) -> AssemblerSignature:
    """Build the session-side signature that affects layer assembly."""

    plan = session.batch_transfer_plan
    plan_sig: AssemblerSignature | None = None
    if plan is not None:
        plan_sig = (
            plan.plan_id,
            plan.operation_type,
            tuple(
                (
                    row.row_id,
                    row.direction,
                    str(row.source_layer_id) if row.source_layer_id is not None else None,
                    row.source_track_coord,
                    row.target_track_coord,
                    str(row.target_layer_id) if row.target_layer_id is not None else None,
                    row.import_mode,
                    tuple(str(event_id) for event_id in row.selected_event_ids),
                    tuple(row.selected_ma3_event_ids),
                    row.selected_count,
                    row.status,
                    row.issue,
                    row.target_label,
                )
                for row in plan.rows
            ),
        )

    return (
        session.manual_push_flow.push_mode_active,
        tuple(str(layer_id) for layer_id in session.manual_push_flow.selected_layer_ids),
        session.manual_push_flow.transfer_mode,
        session.manual_pull_flow.workspace_active,
        tuple(
            (timecode.number, timecode.name)
            for timecode in session.manual_pull_flow.available_timecodes
        ),
        session.manual_pull_flow.selected_timecode_no,
        tuple(
            (group.number, group.name, group.track_count)
            for group in session.manual_pull_flow.available_track_groups
        ),
        session.manual_pull_flow.selected_track_group_no,
        tuple(
            (
                track.coord,
                track.name,
                track.number,
                track.timecode_name,
                track.note,
                track.event_count,
            )
            for track in session.manual_pull_flow.available_tracks
        ),
        session.manual_pull_flow.active_source_track_coord,
        session.manual_pull_flow.source_track_coord,
        tuple(session.manual_pull_flow.selected_source_track_coords),
        tuple(
            (
                event.event_id,
                event.label,
                event.start,
                event.end,
                event.cue_number,
            )
            for event in session.manual_pull_flow.available_events
        ),
        tuple(
            (coord, tuple(event_ids))
            for coord, event_ids in sorted(
                session.manual_pull_flow.selected_ma3_event_ids_by_track.items()
            )
        ),
        tuple(session.manual_pull_flow.selected_ma3_event_ids),
        session.manual_pull_flow.import_mode,
        tuple(
            (coord, mode)
            for coord, mode in sorted(
                session.manual_pull_flow.import_mode_by_source_track.items()
            )
        ),
        tuple(
            (coord, str(layer_id))
            for coord, layer_id in sorted(
                session.manual_pull_flow.target_layer_id_by_source_track.items()
            )
        ),
        plan_sig,
    )


def _events_signature(events: list[Event]) -> AssemblerSignature:
    return (
        id(events),
        tuple(
            (
                str(event.id),
                str(event.take_id),
                float(event.start),
                float(event.end),
                str(event.label),
                int(event.cue_number),
                bool(event.muted),
            )
            for event in events
        ),
    )

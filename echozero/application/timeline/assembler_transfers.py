"""
Timeline transfer and banner presentation builders for the canonical app contract.
Exists to keep top-level transfer flows and run status banners out of the assembler facade.
Connects session-side transfer state to TimelinePresentation support surfaces.
"""

from typing import TYPE_CHECKING

from echozero.application.presentation.models import (
    BatchTransferPlanPresentation,
    BatchTransferPlanRowPresentation,
    ManualPullDiffPreviewPresentation,
    ManualPullEventOptionPresentation,
    ManualPullFlowPresentation,
    ManualPullTimecodeOptionPresentation,
    ManualPullTrackGroupOptionPresentation,
    ManualPullTargetOptionPresentation,
    ManualPullTrackOptionPresentation,
    ManualPushDiffPreviewPresentation,
    ManualPushFlowPresentation,
    ManualPushTimecodeOptionPresentation,
    ManualPushTrackGroupOptionPresentation,
    ManualPushSequenceOptionPresentation,
    ManualPushSequenceRangePresentation,
    ManualPushTrackOptionPresentation,
    PipelineRunBannerPresentation,
    SyncDiffRowPresentation,
    SyncDiffSummaryPresentation,
    TransferPresetPresentation,
)
from echozero.application.session.models import (
    ManualPullDiffPreview,
    ManualPushDiffPreview,
    Session,
)
from echozero.application.sync.diff_service import SyncDiffRow, SyncDiffSummary

if TYPE_CHECKING:
    from echozero.application.timeline.pipeline_run_service import PipelineRunState

__all__ = [
    "assemble_batch_transfer_plan",
    "assemble_manual_pull_flow",
    "assemble_manual_push_flow",
    "assemble_pipeline_run_banner",
    "assemble_transfer_presets",
]


def assemble_manual_push_flow(session: Session) -> ManualPushFlowPresentation:
    """Build the push dialog state visible to the timeline UI."""

    flow = session.manual_push_flow
    return ManualPushFlowPresentation(
        dialog_open=flow.dialog_open,
        push_mode_active=flow.push_mode_active,
        selected_layer_ids=list(flow.selected_layer_ids),
        available_timecodes=[
            ManualPushTimecodeOptionPresentation(
                number=timecode.number,
                name=timecode.name,
            )
            for timecode in flow.available_timecodes
        ],
        selected_timecode_no=flow.selected_timecode_no,
        available_track_groups=[
            ManualPushTrackGroupOptionPresentation(
                number=group.number,
                name=group.name,
                track_count=group.track_count,
            )
            for group in flow.available_track_groups
        ],
        selected_track_group_no=flow.selected_track_group_no,
        available_tracks=[
            ManualPushTrackOptionPresentation(
                coord=track.coord,
                name=track.name,
                number=track.number,
                timecode_name=track.timecode_name,
                note=track.note,
                event_count=track.event_count,
                sequence_no=track.sequence_no,
            )
            for track in flow.available_tracks
        ],
        available_sequences=[
            ManualPushSequenceOptionPresentation(
                number=sequence.number,
                name=sequence.name,
            )
            for sequence in flow.available_sequences
        ],
        current_song_sequence_range=_assemble_manual_push_sequence_range(
            flow.current_song_sequence_range
        ),
        target_track_coord=flow.target_track_coord,
        transfer_mode=flow.transfer_mode,
        diff_gate_open=flow.diff_gate_open,
        diff_preview=_assemble_manual_push_diff_preview(flow.diff_preview),
    )


def assemble_manual_pull_flow(session: Session) -> ManualPullFlowPresentation:
    """Build the pull workspace state visible to the timeline UI."""

    flow = session.manual_pull_flow
    return ManualPullFlowPresentation(
        dialog_open=flow.dialog_open,
        workspace_active=flow.workspace_active,
        available_timecodes=[
            ManualPullTimecodeOptionPresentation(
                number=timecode.number,
                name=timecode.name,
            )
            for timecode in flow.available_timecodes
        ],
        selected_timecode_no=flow.selected_timecode_no,
        available_track_groups=[
            ManualPullTrackGroupOptionPresentation(
                number=group.number,
                name=group.name,
                track_count=group.track_count,
            )
            for group in flow.available_track_groups
        ],
        selected_track_group_no=flow.selected_track_group_no,
        available_tracks=[
            ManualPullTrackOptionPresentation(
                coord=track.coord,
                name=track.name,
                number=track.number,
                timecode_name=track.timecode_name,
                note=track.note,
                event_count=track.event_count,
            )
            for track in flow.available_tracks
        ],
        selected_source_track_coords=list(flow.selected_source_track_coords),
        active_source_track_coord=flow.active_source_track_coord,
        source_track_coord=flow.source_track_coord,
        available_events=[
            ManualPullEventOptionPresentation(
                event_id=event.event_id,
                label=event.label,
                start=event.start,
                end=event.end,
                cue_ref=event.cue_ref,
                color=event.color,
                notes=event.notes,
                payload_ref=event.payload_ref,
            )
            for event in flow.available_events
        ],
        selected_ma3_event_ids=list(flow.selected_ma3_event_ids),
        selected_ma3_event_ids_by_track={
            coord: list(event_ids)
            for coord, event_ids in flow.selected_ma3_event_ids_by_track.items()
        },
        import_mode=flow.import_mode,
        import_mode_by_source_track=dict(flow.import_mode_by_source_track),
        available_target_layers=[
            ManualPullTargetOptionPresentation(
                layer_id=target.layer_id,
                name=target.name,
                kind=target.kind,
            )
            for target in flow.available_target_layers
        ],
        target_layer_id=flow.target_layer_id,
        target_layer_id_by_source_track=dict(flow.target_layer_id_by_source_track),
        diff_gate_open=flow.diff_gate_open,
        diff_preview=_assemble_manual_pull_diff_preview(flow.diff_preview),
    )


def assemble_batch_transfer_plan(session: Session) -> BatchTransferPlanPresentation | None:
    """Build the batch transfer plan summary visible to the timeline UI."""

    plan = session.batch_transfer_plan
    if plan is None:
        return None
    return BatchTransferPlanPresentation(
        plan_id=plan.plan_id,
        operation_type=plan.operation_type,
        rows=[
            BatchTransferPlanRowPresentation(
                row_id=row.row_id,
                direction=row.direction,
                source_label=row.source_label,
                target_label=row.target_label,
                source_layer_id=row.source_layer_id,
                source_track_coord=row.source_track_coord,
                target_track_coord=row.target_track_coord,
                target_layer_id=row.target_layer_id,
                import_mode=row.import_mode,
                selected_event_ids=list(row.selected_event_ids),
                selected_ma3_event_ids=list(row.selected_ma3_event_ids),
                selected_count=row.selected_count,
                status=row.status,
                issue=row.issue,
            )
            for row in plan.rows
        ],
        draft_count=plan.draft_count,
        ready_count=plan.ready_count,
        blocked_count=plan.blocked_count,
        applied_count=plan.applied_count,
        failed_count=plan.failed_count,
    )


def assemble_transfer_presets(session: Session) -> list[TransferPresetPresentation]:
    """Build saved transfer preset mappings for the timeline UI."""

    return [
        TransferPresetPresentation(
            preset_id=preset.preset_id,
            name=preset.name,
            push_target_mapping_by_layer_id=dict(preset.push_target_mapping_by_layer_id),
            pull_target_mapping_by_source_track=dict(preset.pull_target_mapping_by_source_track),
        )
        for preset in session.transfer_presets
    ]


def assemble_pipeline_run_banner(
    session: Session,
    *,
    song_id: str | None = None,
    song_version_id: str | None = None,
) -> PipelineRunBannerPresentation | None:
    """Build the latest active or failed pipeline run banner."""

    runs = [
        state
        for state in session.pipeline_runs.values()
        if _pipeline_run_matches_context(
            state,
            song_id=song_id,
            song_version_id=song_version_id,
        )
    ]
    active = [
        state
        for state in runs
        if state.status in {"queued", "resolving", "running", "persisting"}
    ]
    if active:
        state = max(active, key=lambda value: value.started_at)
        return PipelineRunBannerPresentation(
            run_id=state.run_id,
            title=state.display_label,
            status=state.status,
            message=state.message,
            percent=state.percent,
            is_error=False,
        )

    failed = [state for state in runs if state.status == "failed"]
    if not failed:
        return None
    state = max(
        failed,
        key=lambda value: value.finished_at if value.finished_at is not None else value.started_at,
    )
    return PipelineRunBannerPresentation(
        run_id=state.run_id,
        title=state.display_label,
        status=state.status,
        message=state.error or state.message,
        percent=state.percent,
        is_error=True,
    )


def _pipeline_run_matches_context(
    state: "PipelineRunState",
    *,
    song_id: str | None,
    song_version_id: str | None,
) -> bool:
    if song_version_id is not None:
        return state.song_version_id == song_version_id
    if song_id is not None:
        return state.song_id == song_id
    return True


def _assemble_manual_push_diff_preview(
    diff_preview: ManualPushDiffPreview | None,
) -> ManualPushDiffPreviewPresentation | None:
    if diff_preview is None:
        return None
    return ManualPushDiffPreviewPresentation(
        selected_count=diff_preview.selected_count,
        target_track_coord=diff_preview.target_track_coord,
        target_track_name=diff_preview.target_track_name,
        target_track_note=diff_preview.target_track_note,
        target_track_event_count=diff_preview.target_track_event_count,
        diff_summary=_assemble_sync_diff_summary(diff_preview.diff_summary),
        diff_rows=_assemble_sync_diff_rows(diff_preview.diff_rows),
    )


def _assemble_manual_pull_diff_preview(
    diff_preview: ManualPullDiffPreview | None,
) -> ManualPullDiffPreviewPresentation | None:
    if diff_preview is None:
        return None
    return ManualPullDiffPreviewPresentation(
        selected_count=diff_preview.selected_count,
        source_track_coord=diff_preview.source_track_coord,
        source_track_name=diff_preview.source_track_name,
        source_track_note=diff_preview.source_track_note,
        source_track_event_count=diff_preview.source_track_event_count,
        target_layer_id=diff_preview.target_layer_id,
        target_layer_name=diff_preview.target_layer_name,
        import_mode=diff_preview.import_mode,
        diff_summary=_assemble_sync_diff_summary(diff_preview.diff_summary),
        diff_rows=_assemble_sync_diff_rows(diff_preview.diff_rows),
    )


def _assemble_manual_push_sequence_range(
    sequence_range,
) -> ManualPushSequenceRangePresentation | None:
    if sequence_range is None:
        return None
    return ManualPushSequenceRangePresentation(
        start=sequence_range.start,
        end=sequence_range.end,
        song_label=sequence_range.song_label,
    )


def _assemble_sync_diff_summary(
    summary: SyncDiffSummary | None,
) -> SyncDiffSummaryPresentation | None:
    if summary is None:
        return None
    return SyncDiffSummaryPresentation(
        added_count=summary.added_count,
        removed_count=summary.removed_count,
        modified_count=summary.modified_count,
        unchanged_count=summary.unchanged_count,
        row_count=summary.row_count,
    )


def _assemble_sync_diff_rows(rows: list[SyncDiffRow]) -> list[SyncDiffRowPresentation]:
    return [
        SyncDiffRowPresentation(
            row_id=row.row_id,
            action=row.action,
            start=row.start,
            end=row.end,
            label=row.label,
            before=row.before,
            after=row.after,
        )
        for row in rows
    ]

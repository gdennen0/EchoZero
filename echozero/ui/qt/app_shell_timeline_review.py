"""Timeline-review helpers for the Qt app shell runtime.
Exists because fix-mode corrections should emit canonical review signals without touching engine code.
Connects explicit timeline review intents to Foundry review-signal persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.session.models import Session
from echozero.application.timeline.app import TimelineApplication
from echozero.application.timeline.intents import (
    CommitBoundaryCorrectedEventReview,
    CommitMissedEventsReview,
    CommitMissedEventReview,
    CommitRejectedEventsReview,
    CommitRejectedEventReview,
    CommitRelabeledEventReview,
    CommitVerifiedEventsReview,
    CommitVerifiedEventReview,
    CreateEvent,
    TrimEvent,
    UpdateEventLabel,
)
from echozero.application.timeline.models import (
    Event as TimelineEvent,
    Layer as TimelineLayer,
    Take as TimelineTake,
)
from echozero.foundry.domain.review import (
    ExplicitReviewCommit,
    ReviewCommitContext,
    ReviewDecisionKind,
    ReviewOutcome,
    ReviewPolarity,
    ReviewSignal,
    ReviewSurface,
    build_review_decision,
    build_review_provenance,
)
from echozero.foundry.services.review_event_state import (
    normalize_review_label,
    updated_review_metadata,
)
from echozero.foundry.services.dataset_service import DatasetService
from echozero.foundry.services.review_signal_service import ReviewSignalService
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell_project_timeline_storage import resolve_project_audio_path


class TimelineReviewShell(Protocol):
    _app: TimelineApplication
    project_storage: ProjectStorage

    @property
    def session(self) -> Session: ...

    def _sync_storage_backed_timeline(self) -> None: ...

    def _sync_storage_backed_layers(self, layer_ids: list[LayerId]) -> None: ...


def commit_missed_event_review(
    shell: TimelineReviewShell,
    intent: CommitMissedEventReview,
    *,
    sync_runtime: bool = True,
    signal_service: ReviewSignalService | None = None,
    review_context: ReviewCommitContext | None = None,
    apply_project_writeback: bool = True,
    materialize_dataset: bool = True,
) -> TimelinePresentation:
    """Create one missing event and emit one canonical timeline review signal."""

    created = shell._app.dispatch(
        CreateEvent(
            layer_id=intent.layer_id,
            take_id=intent.take_id,
            time_range=intent.time_range,
            label=intent.label,
            cue_number=intent.cue_number,
            source_event_id=intent.source_event_id,
            payload_ref=intent.payload_ref,
            color=intent.color,
        )
    )
    created_event_id = _selected_created_event_id(created)
    if created_event_id is None:
        raise ValueError("CommitMissedEventReview could not resolve the created event id")
    runtime_event = _require_runtime_event(
        shell,
        layer_id=intent.layer_id,
        event_id=created_event_id,
        take_id=intent.take_id,
    )

    project = shell.project_storage.project
    active_song_id = shell.session.active_song_id
    active_song_version_id = shell.session.active_song_version_id
    if active_song_id is None or active_song_version_id is None:
        raise ValueError("CommitMissedEventReview requires an active song and song version")
    version = shell.project_storage.song_versions.get(str(active_song_version_id))
    if version is None:
        raise ValueError(f"Active song version not found: {active_song_version_id}")
    song = shell.project_storage.songs.get(str(active_song_id))
    if song is None:
        raise ValueError(f"Active song not found: {active_song_id}")

    layer = _require_layer(created, intent.layer_id)
    source_audio_path = _resolve_source_audio_path(
        presentation=created,
        shell=shell,
        layer=layer,
        version_audio_file=version.audio_file,
        take_id=intent.take_id,
    )
    review_label = normalize_review_label(layer.title or intent.label)
    event_ref = _build_ref("event", created_event_id)
    source_event_ref = _build_ref("event", intent.source_event_id or created_event_id)
    take_ref = _build_ref("take", intent.take_id) if intent.take_id is not None else None
    review_note = (
        intent.review_note
        or f"Operator added the missed {review_label} event from timeline fix mode."
    )
    source_provenance = {
        "kind": "ez_timeline_fix_review",
        "project_ref": _build_ref("project", project.id),
        "project_name": project.name,
        "song_ref": _build_ref("song", active_song_id),
        "song_title": song.title,
        "version_ref": _build_ref("version", active_song_version_id),
        "version_label": version.label,
        "layer_ref": _build_ref("layer", intent.layer_id),
        "layer_name": layer.title,
        "take_ref": take_ref,
        "event_ref": event_ref,
        "source_event_ref": source_event_ref,
        "audio_ref": source_audio_path,
        "source_audio_ref": source_audio_path,
    }
    provenance = build_review_provenance(
        source_provenance,
        surface=ReviewSurface.TIMELINE_FIX_MODE,
        workflow="timeline_event_edit",
        operator_action="add_missing_event",
    )
    decision = build_review_decision(
        ReviewOutcome.INCORRECT,
        corrected_label=review_label,
        review_note=review_note,
        decision_kind=ReviewDecisionKind.MISSED_EVENT_ADDED,
        corrected_start_ms=float(intent.time_range.start) * 1000.0,
        corrected_end_ms=float(intent.time_range.end) * 1000.0,
        created_event_ref=event_ref,
        provenance=provenance,
    )
    if decision is None:
        raise ValueError("CommitMissedEventReview could not build a review decision")
    context = review_context or ReviewCommitContext(
        session_id=f"timeline_fix_{project.id}_{active_song_version_id}",
        session_name=f"Timeline Fix Review - {project.name}",
        source_ref=str(Path(shell.project_storage.working_dir).resolve()),
        metadata={
            "queue_source_kind": "timeline_fix_mode",
            "review_surface": ReviewSurface.TIMELINE_FIX_MODE.value,
        },
    )
    commit = ExplicitReviewCommit(
        item_id=f"timeline_fix:{active_song_version_id}:{intent.layer_id}:{created_event_id}",
        audio_path=source_audio_path,
        predicted_label=review_label,
        target_class=review_label,
        polarity=ReviewPolarity.POSITIVE,
        source_provenance=source_provenance,
        review_outcome=ReviewOutcome.INCORRECT,
        review_decision=decision,
        corrected_label=review_label,
        review_note=review_note,
    )
    service = signal_service or ReviewSignalService(Path(shell.project_storage.working_dir).resolve())
    service.record_explicit_review(
        context,
        commit,
        apply_project_writeback=apply_project_writeback,
        materialize_dataset=materialize_dataset,
    )
    _apply_runtime_review_state(
        runtime_event,
        promotion_state="promoted",
        review_state="corrected",
        review_outcome=ReviewOutcome.INCORRECT,
        decision_kind=ReviewDecisionKind.MISSED_EVENT_ADDED,
        original_label=None,
        corrected_label=review_label,
        review_note=review_note,
        reviewed_at=datetime.now(UTC),
        original_start_ms=None,
        original_end_ms=None,
        corrected_start_ms=float(runtime_event.start) * 1000.0,
        corrected_end_ms=float(runtime_event.end) * 1000.0,
        created_event_ref=event_ref,
        surface=ReviewSurface.TIMELINE_FIX_MODE,
        workflow="timeline_event_edit",
        operator_action="add_missing_event",
        set_origin="manual_added",
        display_label=runtime_event.label or intent.label,
    )
    if sync_runtime:
        shell._sync_storage_backed_timeline()
    return shell.presentation()


def commit_verified_review(
    shell: TimelineReviewShell,
    intent: CommitVerifiedEventReview,
    *,
    sync_runtime: bool = True,
    signal_service: ReviewSignalService | None = None,
    review_context: ReviewCommitContext | None = None,
    resolved_target: ReviewEventContext | None = None,
    resolved_review_context: tuple[object, object, object, object, object] | None = None,
    apply_project_writeback: bool = True,
    materialize_dataset: bool = True,
) -> TimelinePresentation:
    """Record one explicit verified event review signal without changing event truth."""

    if resolved_target is None:
        target = _resolve_review_event_context(
            shell,
            layer_id=intent.layer_id,
            event_id=intent.event_id,
            take_id=intent.take_id,
        )
    else:
        target = resolved_target
    if resolved_review_context is None:
        project, active_song_id, active_song_version_id, version, song = _require_review_context(shell)
    else:
        project, active_song_id, active_song_version_id, version, song = resolved_review_context
    review_label = normalize_review_label(target.event.label)
    review_note = intent.review_note or f"Operator verified the {target.event.label} event from timeline review."
    source_provenance = _review_source_provenance(
        project=project,
        song=song,
        version=version,
        active_song_id=active_song_id,
        active_song_version_id=active_song_version_id,
        target=target,
    )
    provenance = build_review_provenance(
        source_provenance,
        surface=ReviewSurface.TIMELINE_FIX_MODE,
        workflow="timeline_event_review",
        operator_action="verify_event",
    )
    decision = build_review_decision(
        ReviewOutcome.CORRECT,
        corrected_label=None,
        review_note=review_note,
        decision_kind=ReviewDecisionKind.VERIFIED,
        provenance=provenance,
    )
    if decision is None:
        raise ValueError("CommitVerifiedEventReview could not build a review decision")
    context = review_context or _review_commit_context(shell, project, active_song_version_id)
    commit = ExplicitReviewCommit(
        item_id=_event_review_item_id(
            active_song_version_id=active_song_version_id,
            layer_id=target.layer.layer_id,
            event_id=target.event.event_id,
        ),
        audio_path=target.source_audio_path,
        predicted_label=review_label,
        target_class=review_label,
        polarity=ReviewPolarity.POSITIVE,
        source_provenance=source_provenance,
        review_outcome=ReviewOutcome.CORRECT,
        review_decision=decision,
        review_note=review_note,
    )
    service = signal_service or ReviewSignalService(Path(shell.project_storage.working_dir).resolve())
    service.record_explicit_review(
        context,
        commit,
        apply_project_writeback=apply_project_writeback,
        materialize_dataset=materialize_dataset,
    )
    _apply_runtime_review_state(
        _require_runtime_event(
            shell,
            layer_id=target.layer.layer_id,
            event_id=target.event.event_id,
            take_id=target.take.take_id if target.take is not None else None,
        ),
        promotion_state="promoted",
        review_state="signed_off",
        review_outcome=ReviewOutcome.CORRECT,
        decision_kind=ReviewDecisionKind.VERIFIED,
        original_label=review_label,
        corrected_label=None,
        review_note=review_note,
        reviewed_at=datetime.now(UTC),
        original_start_ms=float(target.event.start) * 1000.0,
        original_end_ms=float(target.event.end) * 1000.0,
        corrected_start_ms=None,
        corrected_end_ms=None,
        created_event_ref=None,
        surface=ReviewSurface.TIMELINE_FIX_MODE,
        workflow="timeline_event_review",
        operator_action="verify_event",
    )
    if sync_runtime:
        shell._sync_storage_backed_timeline()
    return shell.presentation()


def commit_rejected_review(
    shell: TimelineReviewShell,
    intent: CommitRejectedEventReview,
    *,
    sync_runtime: bool = True,
    signal_service: ReviewSignalService | None = None,
    review_context: ReviewCommitContext | None = None,
    resolved_target: ReviewEventContext | None = None,
    resolved_review_context: tuple[object, object, object, object, object] | None = None,
    apply_project_writeback: bool = True,
    materialize_dataset: bool = True,
) -> TimelinePresentation:
    """Demote one false-positive event and emit one canonical rejection signal."""

    if resolved_target is None:
        target = _resolve_review_event_context(
            shell,
            layer_id=intent.layer_id,
            event_id=intent.event_id,
            take_id=intent.take_id,
        )
    else:
        target = resolved_target
    if resolved_review_context is None:
        project, active_song_id, active_song_version_id, version, song = _require_review_context(shell)
    else:
        project, active_song_id, active_song_version_id, version, song = resolved_review_context
    review_label = normalize_review_label(target.event.label)
    review_note = intent.review_note or f"Operator rejected the {target.event.label} event from timeline review."
    source_provenance = _review_source_provenance(
        project=project,
        song=song,
        version=version,
        active_song_id=active_song_id,
        active_song_version_id=active_song_version_id,
        target=target,
    )
    provenance = build_review_provenance(
        source_provenance,
        surface=ReviewSurface.TIMELINE_FIX_MODE,
        workflow="timeline_event_review",
        operator_action="reject_event",
    )
    decision = build_review_decision(
        ReviewOutcome.INCORRECT,
        corrected_label=None,
        review_note=review_note,
        decision_kind=ReviewDecisionKind.REJECTED,
        provenance=provenance,
    )
    if decision is None:
        raise ValueError("CommitRejectedEventReview could not build a review decision")
    context = review_context or _review_commit_context(shell, project, active_song_version_id)
    commit = ExplicitReviewCommit(
        item_id=_event_review_item_id(
            active_song_version_id=active_song_version_id,
            layer_id=target.layer.layer_id,
            event_id=target.event.event_id,
        ),
        audio_path=target.source_audio_path,
        predicted_label=review_label,
        target_class=review_label,
        polarity=ReviewPolarity.NEGATIVE,
        source_provenance=source_provenance,
        review_outcome=ReviewOutcome.INCORRECT,
        review_decision=decision,
        review_note=review_note,
    )
    service = signal_service or ReviewSignalService(Path(shell.project_storage.working_dir).resolve())
    service.record_explicit_review(
        context,
        commit,
        apply_project_writeback=apply_project_writeback,
        materialize_dataset=materialize_dataset,
    )
    _apply_runtime_review_state(
        _require_runtime_event(
            shell,
            layer_id=target.layer.layer_id,
            event_id=target.event.event_id,
            take_id=target.take.take_id if target.take is not None else None,
        ),
        promotion_state="demoted",
        review_state="corrected",
        review_outcome=ReviewOutcome.INCORRECT,
        decision_kind=ReviewDecisionKind.REJECTED,
        original_label=review_label,
        corrected_label=None,
        review_note=review_note,
        reviewed_at=datetime.now(UTC),
        original_start_ms=float(target.event.start) * 1000.0,
        original_end_ms=float(target.event.end) * 1000.0,
        corrected_start_ms=None,
        corrected_end_ms=None,
        created_event_ref=None,
        surface=ReviewSurface.TIMELINE_FIX_MODE,
        workflow="timeline_event_review",
        operator_action="reject_event",
    )
    if sync_runtime:
        shell._sync_storage_backed_timeline()
    return shell.presentation()


def commit_missed_events_review(
    shell: TimelineReviewShell,
    intent: CommitMissedEventsReview,
) -> TimelinePresentation:
    """Create multiple missing events and emit review signals in one undoable operation."""

    project, _song_id, active_song_version_id, _version, _song = _require_review_context(shell)
    review_context = ReviewCommitContext(
        session_id=f"timeline_fix_{project.id}_{active_song_version_id}",
        session_name=f"Timeline Fix Review - {project.name}",
        source_ref=str(Path(shell.project_storage.working_dir).resolve()),
        metadata={
            "queue_source_kind": "timeline_fix_mode",
            "review_surface": ReviewSurface.TIMELINE_FIX_MODE.value,
        },
    )
    service = ReviewSignalService(Path(shell.project_storage.working_dir).resolve())
    touched_layer_ids = {entry.layer_id for entry in intent.intents}
    for entry in intent.intents:
        commit_missed_event_review(
            shell,
            entry,
            sync_runtime=False,
            signal_service=service,
            review_context=review_context,
            apply_project_writeback=False,
            materialize_dataset=False,
        )
    shell._sync_storage_backed_layers(list(touched_layer_ids))
    _refresh_project_review_export(shell)
    return shell.presentation()


def commit_verified_events_review(
    shell: TimelineReviewShell,
    intent: CommitVerifiedEventsReview,
) -> TimelinePresentation:
    """Verify multiple events in one undoable operation."""

    project, active_song_id, active_song_version_id, version, song = _require_review_context(shell)
    review_context = _review_commit_context(shell, project, active_song_version_id)
    service = ReviewSignalService(Path(shell.project_storage.working_dir).resolve())
    presentation = shell.presentation()
    resolved_context = (project, active_song_id, active_song_version_id, version, song)
    touched_layer_ids: set[LayerId] = set()
    for event_ref in intent.event_refs:
        target = _resolve_review_event_context_on_presentation(
            shell,
            presentation=presentation,
            layer_id=event_ref.layer_id,
            event_id=event_ref.event_id,
            take_id=event_ref.take_id,
            version_audio_file=version.audio_file,
        )
        touched_layer_ids.add(target.layer.layer_id)
        commit_verified_review(
            shell,
            CommitVerifiedEventReview(
                layer_id=event_ref.layer_id,
                event_id=event_ref.event_id,
                take_id=event_ref.take_id,
                review_note=intent.review_note,
            ),
            sync_runtime=False,
            signal_service=service,
            review_context=review_context,
            resolved_target=target,
            resolved_review_context=resolved_context,
            apply_project_writeback=False,
            materialize_dataset=False,
        )
    shell._sync_storage_backed_layers(list(touched_layer_ids))
    _refresh_project_review_export(shell)
    return shell.presentation()


def commit_rejected_events_review(
    shell: TimelineReviewShell,
    intent: CommitRejectedEventsReview,
) -> TimelinePresentation:
    """Reject multiple events in one undoable operation."""

    project, active_song_id, active_song_version_id, version, song = _require_review_context(shell)
    review_context = _review_commit_context(shell, project, active_song_version_id)
    service = ReviewSignalService(Path(shell.project_storage.working_dir).resolve())
    presentation = shell.presentation()
    resolved_context = (project, active_song_id, active_song_version_id, version, song)
    touched_layer_ids: set[LayerId] = set()
    for event_ref in intent.event_refs:
        target = _resolve_review_event_context_on_presentation(
            shell,
            presentation=presentation,
            layer_id=event_ref.layer_id,
            event_id=event_ref.event_id,
            take_id=event_ref.take_id,
            version_audio_file=version.audio_file,
        )
        touched_layer_ids.add(target.layer.layer_id)
        commit_rejected_review(
            shell,
            CommitRejectedEventReview(
                layer_id=event_ref.layer_id,
                event_id=event_ref.event_id,
                take_id=event_ref.take_id,
                review_note=intent.review_note,
            ),
            sync_runtime=False,
            signal_service=service,
            review_context=review_context,
            resolved_target=target,
            resolved_review_context=resolved_context,
            apply_project_writeback=False,
            materialize_dataset=False,
        )
    shell._sync_storage_backed_layers(list(touched_layer_ids))
    _refresh_project_review_export(shell)
    return shell.presentation()


def commit_relabel_review(
    shell: TimelineReviewShell,
    intent: CommitRelabeledEventReview,
) -> TimelinePresentation:
    """Relabel one existing event and emit one canonical relabel review signal."""

    target = _resolve_review_event_context(
        shell,
        layer_id=intent.layer_id,
        event_id=intent.event_id,
        take_id=intent.take_id,
    )
    project, active_song_id, active_song_version_id, version, song = _require_review_context(shell)
    corrected_label = str(intent.corrected_label).strip()
    original_label = normalize_review_label(target.event.label)
    normalized_corrected_label = normalize_review_label(corrected_label)
    review_note = intent.review_note or f"Operator relabeled the {target.event.label} event as {corrected_label}."
    source_provenance = _review_source_provenance(
        project=project,
        song=song,
        version=version,
        active_song_id=active_song_id,
        active_song_version_id=active_song_version_id,
        target=target,
    )
    provenance = build_review_provenance(
        source_provenance,
        surface=ReviewSurface.TIMELINE_FIX_MODE,
        workflow="timeline_event_review",
        operator_action="relabel_event",
    )
    decision = build_review_decision(
        ReviewOutcome.INCORRECT,
        corrected_label=normalized_corrected_label,
        review_note=review_note,
        decision_kind=ReviewDecisionKind.RELABELED,
        provenance=provenance,
    )
    if decision is None:
        raise ValueError("CommitRelabeledEventReview could not build a review decision")
    shell._app.dispatch(
        UpdateEventLabel(
            event_id=target.event.event_id,
            label=corrected_label,
            layer_id=target.layer.layer_id,
            take_id=target.take.take_id if target.take is not None else None,
        )
    )
    context = _review_commit_context(shell, project, active_song_version_id)
    commit = ExplicitReviewCommit(
        item_id=_event_review_item_id(
            active_song_version_id=active_song_version_id,
            layer_id=target.layer.layer_id,
            event_id=target.event.event_id,
        ),
        audio_path=target.source_audio_path,
        predicted_label=original_label,
        target_class=normalized_corrected_label,
        polarity=ReviewPolarity.POSITIVE,
        source_provenance=source_provenance,
        review_outcome=ReviewOutcome.INCORRECT,
        review_decision=decision,
        corrected_label=normalized_corrected_label,
        review_note=review_note,
    )
    ReviewSignalService(Path(shell.project_storage.working_dir).resolve()).record_explicit_review(
        context,
        commit,
    )
    _apply_runtime_review_state(
        _require_runtime_event(
            shell,
            layer_id=target.layer.layer_id,
            event_id=target.event.event_id,
            take_id=target.take.take_id if target.take is not None else None,
        ),
        promotion_state="promoted",
        review_state="corrected",
        review_outcome=ReviewOutcome.INCORRECT,
        decision_kind=ReviewDecisionKind.RELABELED,
        original_label=original_label,
        corrected_label=normalized_corrected_label,
        review_note=review_note,
        reviewed_at=datetime.now(UTC),
        original_start_ms=float(target.event.start) * 1000.0,
        original_end_ms=float(target.event.end) * 1000.0,
        corrected_start_ms=None,
        corrected_end_ms=None,
        created_event_ref=None,
        surface=ReviewSurface.TIMELINE_FIX_MODE,
        workflow="timeline_event_review",
        operator_action="relabel_event",
        display_label=corrected_label,
    )
    shell._sync_storage_backed_timeline()
    return shell.presentation()


def commit_boundary_corrected_review(
    shell: TimelineReviewShell,
    intent: CommitBoundaryCorrectedEventReview,
) -> TimelinePresentation:
    """Trim one event boundary and emit one canonical boundary-correction signal."""

    target = _resolve_review_event_context(
        shell,
        layer_id=intent.layer_id,
        event_id=intent.event_id,
        take_id=intent.take_id,
    )
    project, active_song_id, active_song_version_id, version, song = _require_review_context(shell)
    review_label = normalize_review_label(target.event.label)
    corrected_start_ms = float(intent.corrected_range.start) * 1000.0
    corrected_end_ms = float(intent.corrected_range.end) * 1000.0
    review_note = (
        intent.review_note
        or (
            "Operator corrected the "
            f"{target.event.label} boundary to "
            f"{float(intent.corrected_range.start):.2f}s-{float(intent.corrected_range.end):.2f}s."
        )
    )
    source_provenance = _review_source_provenance(
        project=project,
        song=song,
        version=version,
        active_song_id=active_song_id,
        active_song_version_id=active_song_version_id,
        target=target,
    )
    provenance = build_review_provenance(
        source_provenance,
        surface=ReviewSurface.TIMELINE_FIX_MODE,
        workflow="timeline_event_review",
        operator_action="correct_event_boundary",
    )
    decision = build_review_decision(
        ReviewOutcome.INCORRECT,
        corrected_label=review_label,
        review_note=review_note,
        decision_kind=ReviewDecisionKind.BOUNDARY_CORRECTED,
        original_start_ms=float(target.event.start) * 1000.0,
        original_end_ms=float(target.event.end) * 1000.0,
        corrected_start_ms=corrected_start_ms,
        corrected_end_ms=corrected_end_ms,
        provenance=provenance,
    )
    if decision is None:
        raise ValueError("CommitBoundaryCorrectedEventReview could not build a review decision")
    shell._app.dispatch(
        TrimEvent(
            event_id=target.event.event_id,
            new_range=intent.corrected_range,
        )
    )
    context = _review_commit_context(shell, project, active_song_version_id)
    commit = ExplicitReviewCommit(
        item_id=_event_review_item_id(
            active_song_version_id=active_song_version_id,
            layer_id=target.layer.layer_id,
            event_id=target.event.event_id,
        ),
        audio_path=target.source_audio_path,
        predicted_label=review_label,
        target_class=review_label,
        polarity=ReviewPolarity.POSITIVE,
        source_provenance=source_provenance,
        review_outcome=ReviewOutcome.INCORRECT,
        review_decision=decision,
        corrected_label=review_label,
        review_note=review_note,
    )
    ReviewSignalService(Path(shell.project_storage.working_dir).resolve()).record_explicit_review(
        context,
        commit,
    )
    _apply_runtime_review_state(
        _require_runtime_event(
            shell,
            layer_id=target.layer.layer_id,
            event_id=target.event.event_id,
            take_id=target.take.take_id if target.take is not None else None,
        ),
        promotion_state="promoted",
        review_state="corrected",
        review_outcome=ReviewOutcome.INCORRECT,
        decision_kind=ReviewDecisionKind.BOUNDARY_CORRECTED,
        original_label=review_label,
        corrected_label=review_label,
        review_note=review_note,
        reviewed_at=datetime.now(UTC),
        original_start_ms=float(target.event.start) * 1000.0,
        original_end_ms=float(target.event.end) * 1000.0,
        corrected_start_ms=corrected_start_ms,
        corrected_end_ms=corrected_end_ms,
        created_event_ref=None,
        surface=ReviewSurface.TIMELINE_FIX_MODE,
        workflow="timeline_event_review",
        operator_action="correct_event_boundary",
    )
    shell._sync_storage_backed_timeline()
    return shell.presentation()


def apply_review_signal_to_runtime(
    shell: TimelineReviewShell,
    signal: ReviewSignal,
) -> dict[str, object]:
    """Apply one persisted phone/project review signal through the live runtime."""

    decision = signal.review_decision
    if signal.review_outcome == ReviewOutcome.PENDING or decision is None:
        return {"status": "skipped", "reason": "pending_review"}

    layer_id = _ref_id(signal.source_provenance.get("layer_ref"), prefix="layer")
    event_id = _review_signal_event_id(signal)
    take_id = _ref_id(signal.source_provenance.get("take_ref"), prefix="take")
    if layer_id is None:
        return {"status": "skipped", "reason": "missing_layer_provenance"}
    if event_id is None:
        return {"status": "skipped", "reason": "missing_event_provenance"}

    runtime_event = _require_runtime_event(
        shell,
        layer_id=layer_id,
        event_id=event_id,
        take_id=take_id,
    )
    review_note = decision.review_note or signal.review_note
    original_label = normalize_review_label(signal.predicted_label)
    corrected_label = (
        normalize_review_label(decision.corrected_label)
        if decision.corrected_label is not None
        else None
    )
    reviewed_at = signal.reviewed_at or datetime.now(UTC)
    operator_action = (
        decision.provenance.operator_action
        if decision.provenance is not None and decision.provenance.operator_action
        else "phone_review"
    )
    workflow = (
        decision.provenance.workflow
        if decision.provenance is not None and decision.provenance.workflow
        else "manual_review"
    )
    surface = (
        decision.provenance.surface
        if decision.provenance is not None
        else ReviewSurface.PHONE_REVIEW
    )

    if decision.kind == ReviewDecisionKind.RELABELED:
        corrected_display = decision.corrected_label or signal.corrected_label or runtime_event.label
        shell._app.dispatch(
            UpdateEventLabel(
                event_id=event_id,
                label=corrected_display,
                layer_id=layer_id,
                take_id=take_id,
            )
        )
        runtime_event = _require_runtime_event(
            shell,
            layer_id=layer_id,
            event_id=event_id,
            take_id=take_id,
        )
        _apply_runtime_review_state(
            runtime_event,
            promotion_state="promoted",
            review_state="corrected",
            review_outcome=signal.review_outcome,
            decision_kind=decision.kind,
            original_label=original_label,
            corrected_label=corrected_label,
            review_note=review_note,
            reviewed_at=reviewed_at,
            original_start_ms=decision.original_start_ms or (float(runtime_event.start) * 1000.0),
            original_end_ms=decision.original_end_ms or (float(runtime_event.end) * 1000.0),
            corrected_start_ms=decision.corrected_start_ms,
            corrected_end_ms=decision.corrected_end_ms,
            created_event_ref=decision.created_event_ref,
            surface=surface,
            workflow=workflow,
            operator_action=operator_action,
            display_label=corrected_display,
        )
    elif decision.kind == ReviewDecisionKind.BOUNDARY_CORRECTED:
        if decision.corrected_start_ms is None or decision.corrected_end_ms is None:
            return {"status": "skipped", "reason": "missing_boundary_correction"}
        shell._app.dispatch(
            TrimEvent(
                event_id=event_id,
                new_range=TimeRange(
                    float(decision.corrected_start_ms) / 1000.0,
                    float(decision.corrected_end_ms) / 1000.0,
                ),
            )
        )
        runtime_event = _require_runtime_event(
            shell,
            layer_id=layer_id,
            event_id=event_id,
            take_id=take_id,
        )
        _apply_runtime_review_state(
            runtime_event,
            promotion_state="promoted",
            review_state="corrected",
            review_outcome=signal.review_outcome,
            decision_kind=decision.kind,
            original_label=original_label,
            corrected_label=corrected_label or original_label,
            review_note=review_note,
            reviewed_at=reviewed_at,
            original_start_ms=decision.original_start_ms,
            original_end_ms=decision.original_end_ms,
            corrected_start_ms=decision.corrected_start_ms,
            corrected_end_ms=decision.corrected_end_ms,
            created_event_ref=decision.created_event_ref,
            surface=surface,
            workflow=workflow,
            operator_action=operator_action,
        )
    elif decision.kind == ReviewDecisionKind.REJECTED:
        _apply_runtime_review_state(
            runtime_event,
            promotion_state="demoted",
            review_state="corrected",
            review_outcome=signal.review_outcome,
            decision_kind=decision.kind,
            original_label=original_label,
            corrected_label=None,
            review_note=review_note,
            reviewed_at=reviewed_at,
            original_start_ms=decision.original_start_ms or (float(runtime_event.start) * 1000.0),
            original_end_ms=decision.original_end_ms or (float(runtime_event.end) * 1000.0),
            corrected_start_ms=None,
            corrected_end_ms=None,
            created_event_ref=decision.created_event_ref,
            surface=surface,
            workflow=workflow,
            operator_action=operator_action,
        )
    elif decision.kind == ReviewDecisionKind.MISSED_EVENT_ADDED:
        display_label = decision.corrected_label or signal.corrected_label or runtime_event.label
        _apply_runtime_review_state(
            runtime_event,
            promotion_state="promoted",
            review_state="corrected",
            review_outcome=signal.review_outcome,
            decision_kind=decision.kind,
            original_label=original_label,
            corrected_label=corrected_label or original_label,
            review_note=review_note,
            reviewed_at=reviewed_at,
            original_start_ms=decision.original_start_ms,
            original_end_ms=decision.original_end_ms,
            corrected_start_ms=decision.corrected_start_ms,
            corrected_end_ms=decision.corrected_end_ms,
            created_event_ref=decision.created_event_ref,
            surface=surface,
            workflow=workflow,
            operator_action=operator_action,
            set_origin="manual_added",
            display_label=display_label,
        )
    else:
        _apply_runtime_review_state(
            runtime_event,
            promotion_state="promoted",
            review_state="signed_off",
            review_outcome=signal.review_outcome,
            decision_kind=decision.kind,
            original_label=original_label,
            corrected_label=None,
            review_note=review_note,
            reviewed_at=reviewed_at,
            original_start_ms=decision.original_start_ms or (float(runtime_event.start) * 1000.0),
            original_end_ms=decision.original_end_ms or (float(runtime_event.end) * 1000.0),
            corrected_start_ms=None,
            corrected_end_ms=None,
            created_event_ref=decision.created_event_ref,
            surface=surface,
            workflow=workflow,
            operator_action=operator_action,
        )

    shell._sync_storage_backed_timeline()
    return {
        "status": "applied_via_runtime_bridge",
        "layer_id": layer_id,
        "event_id": event_id,
        "decision_kind": decision.kind.value,
    }


def _selected_created_event_id(presentation: TimelinePresentation) -> str | None:
    if presentation.selected_event_refs:
        return str(presentation.selected_event_refs[0].event_id)
    if presentation.selected_event_ids:
        return str(presentation.selected_event_ids[0])
    return None


def _require_layer(presentation: TimelinePresentation, layer_id: object) -> LayerPresentation:
    target = str(layer_id)
    for layer in presentation.layers:
        if str(layer.layer_id) == target:
            return layer
    raise ValueError(f"CommitMissedEventReview layer not found: {layer_id}")


@dataclass(slots=True)
class ReviewEventContext:
    layer: LayerPresentation
    take: TakeLanePresentation | None
    event: EventPresentation
    source_audio_path: str

    @property
    def event_ref(self) -> str:
        return _build_ref("event", self.event.event_id)

    @property
    def take_ref(self) -> str | None:
        return None if self.take is None else _build_ref("take", self.take.take_id)


def _resolve_review_event_context(
    shell: TimelineReviewShell,
    *,
    layer_id: object,
    event_id: object,
    take_id: object | None,
) -> ReviewEventContext:
    version = _require_review_context(shell)[3]
    presentation = shell.presentation()
    return _resolve_review_event_context_on_presentation(
        shell,
        presentation=presentation,
        layer_id=layer_id,
        event_id=event_id,
        take_id=take_id,
        version_audio_file=version.audio_file,
    )


def _resolve_review_event_context_on_presentation(
    shell: TimelineReviewShell,
    *,
    presentation: TimelinePresentation,
    layer_id: object,
    event_id: object,
    take_id: object | None,
    version_audio_file: str,
) -> ReviewEventContext:
    layer = _require_layer(presentation, layer_id)
    target_take, event = _require_event_on_layer(layer, event_id, take_id)
    source_audio_path = _resolve_source_audio_path(
        presentation=presentation,
        shell=shell,
        layer=layer,
        version_audio_file=version_audio_file,
        take_id=target_take.take_id if target_take is not None else None,
    )
    return ReviewEventContext(
        layer=layer,
        take=target_take,
        event=event,
        source_audio_path=source_audio_path,
    )


def _require_runtime_event(
    shell: TimelineReviewShell,
    *,
    layer_id: object,
    event_id: object,
    take_id: object | None,
) -> TimelineEvent:
    target_layer = str(layer_id)
    target_event = str(event_id)
    target_take = None if take_id is None else str(take_id)
    for layer in shell._app.timeline.layers:
        if str(layer.id) != target_layer:
            continue
        resolved = _event_from_runtime_layer(layer, event_id=target_event, take_id=target_take)
        if resolved is not None:
            return resolved
        break
    raise ValueError(f"Timeline review runtime event not found: {event_id}")


def _event_from_runtime_layer(
    layer: TimelineLayer,
    *,
    event_id: str,
    take_id: str | None,
) -> TimelineEvent | None:
    takes = list(layer.takes)
    if take_id:
        takes = [take for take in takes if str(take.id) == take_id]
    for take in takes:
        for event in take.events:
            if str(event.id) == event_id:
                return event
    return None


def _apply_runtime_review_state(
    event: TimelineEvent,
    *,
    promotion_state: str,
    review_state: str,
    review_outcome: ReviewOutcome,
    decision_kind: ReviewDecisionKind,
    original_label: str | None,
    corrected_label: str | None,
    review_note: str | None,
    reviewed_at: datetime,
    original_start_ms: float | None,
    original_end_ms: float | None,
    corrected_start_ms: float | None,
    corrected_end_ms: float | None,
    created_event_ref: str | None,
    surface: ReviewSurface,
    workflow: str,
    operator_action: str,
    set_origin: str | None = None,
    display_label: str | None = None,
) -> None:
    if set_origin is not None:
        event.origin = set_origin
    event.metadata = updated_review_metadata(
        event.metadata,
        promotion_state=promotion_state,
        review_state=review_state,
        review_outcome=review_outcome,
        decision_kind=decision_kind,
        original_label=original_label,
        corrected_label=corrected_label,
        review_note=review_note,
        reviewed_at=reviewed_at,
        original_start_ms=original_start_ms,
        original_end_ms=original_end_ms,
        corrected_start_ms=corrected_start_ms,
        corrected_end_ms=corrected_end_ms,
        created_event_ref=created_event_ref,
        surface=surface,
        workflow=workflow,
        operator_action=operator_action,
    )
    event.classifications = _runtime_event_classifications(
        event.classifications,
        display_label=display_label or event.label,
    )


def _runtime_event_classifications(
    classifications: dict[str, object],
    *,
    display_label: str,
) -> dict[str, object]:
    next_classifications = dict(classifications or {})
    normalized_label = normalize_review_label(display_label)
    next_classifications["class"] = normalized_label
    next_classifications["label"] = str(display_label).strip() or normalized_label
    return next_classifications


def _require_event_on_layer(
    layer: LayerPresentation,
    event_id: object,
    take_id: object | None,
) -> tuple[TakeLanePresentation | None, EventPresentation]:
    target_event_id = str(event_id)
    if take_id is None or str(take_id).strip() == "" or str(take_id) == str(layer.main_take_id):
        main_event = next(
            (event for event in layer.events if str(event.event_id) == target_event_id),
            None,
        )
        if main_event is not None:
            return None, main_event
    else:
        target_take_id = str(take_id)
        for take in layer.takes:
            if str(take.take_id) != target_take_id:
                continue
            for event in take.events:
                if str(event.event_id) == target_event_id:
                    return take, event
            raise ValueError(
                f"Timeline review event not found on take {take_id}: {event_id}"
            )
    for take in layer.takes:
        for event in take.events:
            if str(event.event_id) == target_event_id:
                return take, event
    raise ValueError(f"Timeline review event not found: {event_id}")


def _review_commit_context(
    shell: TimelineReviewShell,
    project,
    active_song_version_id: object,
) -> ReviewCommitContext:
    project_id = project.id
    project_name = project.name
    return ReviewCommitContext(
        session_id=f"timeline_review_{project_id}_{active_song_version_id}",
        session_name=f"Timeline Review - {project_name}",
        source_ref=str(Path(shell.project_storage.working_dir).resolve()),
        metadata={
            "queue_source_kind": "timeline_review_mode",
            "review_surface": ReviewSurface.TIMELINE_FIX_MODE.value,
        },
    )


def _require_review_context(
    shell: TimelineReviewShell,
) -> tuple[object, object, object, object, object]:
    project = shell.project_storage.project
    active_song_id = shell.session.active_song_id
    active_song_version_id = shell.session.active_song_version_id
    if active_song_id is None or active_song_version_id is None:
        raise ValueError("Timeline review requires an active song and song version")
    version = shell.project_storage.song_versions.get(str(active_song_version_id))
    if version is None:
        raise ValueError(f"Active song version not found: {active_song_version_id}")
    song = shell.project_storage.songs.get(str(active_song_id))
    if song is None:
        raise ValueError(f"Active song not found: {active_song_id}")
    return project, active_song_id, active_song_version_id, version, song


def _refresh_project_review_export(shell: TimelineReviewShell) -> None:
    project_dir = Path(shell.project_storage.working_dir).resolve()
    service = DatasetService(project_dir)
    try:
        service.export_project_review_dataset(project_dir, queue_source_kind="ez_project")
    except ValueError:
        return


def _review_source_provenance(
    *,
    project,
    song,
    version,
    active_song_id: object,
    active_song_version_id: object,
    target: ReviewEventContext,
) -> dict[str, object]:
    return {
        "kind": "ez_timeline_review",
        "project_ref": _build_ref("project", project.id),
        "project_name": project.name,
        "song_ref": _build_ref("song", active_song_id),
        "song_title": song.title,
        "version_ref": _build_ref("version", active_song_version_id),
        "version_label": version.label,
        "layer_ref": _build_ref("layer", target.layer.layer_id),
        "layer_name": target.layer.title,
        "take_ref": target.take_ref,
        "event_ref": target.event_ref,
        "audio_ref": target.source_audio_path,
        "source_audio_ref": target.source_audio_path,
        "event_label": target.event.label,
        "event_start_ms": float(target.event.start) * 1000.0,
        "event_end_ms": float(target.event.end) * 1000.0,
    }


def _event_review_item_id(
    *,
    active_song_version_id: object,
    layer_id: object,
    event_id: object,
) -> str:
    return f"timeline_review:{active_song_version_id}:{layer_id}:{event_id}"


def _resolve_source_audio_path(
    *,
    presentation: TimelinePresentation,
    shell: TimelineReviewShell,
    layer: LayerPresentation,
    version_audio_file: str,
    take_id: object | None,
) -> str:
    if take_id is not None:
        target_take_id = str(take_id)
        for take in layer.takes:
            if str(take.take_id) != target_take_id:
                continue
            if take.source_audio_path:
                return take.source_audio_path
            if take.playback_source_ref:
                return take.playback_source_ref
    if layer.source_audio_path:
        return layer.source_audio_path
    if layer.playback_source_ref:
        return layer.playback_source_ref
    for candidate in presentation.layers:
        if str(candidate.layer_id) == "source_audio":
            if candidate.source_audio_path:
                return candidate.source_audio_path
            if candidate.playback_source_ref:
                return candidate.playback_source_ref
    return str(resolve_project_audio_path(shell.project_storage, version_audio_file))
def _build_ref(prefix: str, value: object) -> str:
    return f"{prefix}:{value}"


def _ref_id(value: object, *, prefix: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith(f"{prefix}:"):
        return text.split(":", 1)[1].strip() or None
    return text


def _review_signal_event_id(signal: ReviewSignal) -> str | None:
    decision = signal.review_decision
    if decision is not None and decision.kind == ReviewDecisionKind.MISSED_EVENT_ADDED:
        return _ref_id(decision.created_event_ref, prefix="event")
    return _ref_id(signal.source_provenance.get("event_ref"), prefix="event")

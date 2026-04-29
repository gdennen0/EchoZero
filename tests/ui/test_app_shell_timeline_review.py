"""App-shell timeline-review tests for the canonical Stage Zero runtime.
Exists because timeline fix-mode review commits must be proven through AppShellRuntime.
Connects explicit missed-event review intents to durable Foundry signal materialization.
"""

from __future__ import annotations

from pathlib import Path

from echozero.application.shared.ids import EventId, LayerId
from echozero.application.shared.ranges import TimeRange
from echozero.application.timeline.intents import (
    CommitBoundaryCorrectedEventReview,
    CommitMissedEventsReview,
    CommitMissedEventReview,
    CommitRejectedEventsReview,
    CommitRejectedEventReview,
    CommitRelabeledEventReview,
    CommitVerifiedEventsReview,
    CommitVerifiedEventReview,
)
from echozero.application.timeline.models import EventRef
from echozero.foundry.domain.review import ReviewDecisionKind
from echozero.foundry.persistence import ReviewSignalRepository
from echozero.testing.analysis_mocks import build_mock_analysis_service, write_test_wav
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell


def _build_timeline_review_runtime(tmp_path: Path) -> tuple[AppShellRuntime, LayerId, str, float, float]:
    runtime = build_app_shell(
        working_dir_root=tmp_path / "working",
        analysis_service=build_mock_analysis_service(),
    )
    assert isinstance(runtime, AppShellRuntime)

    audio_path = write_test_wav(tmp_path / "fixtures" / "timeline-review.wav")
    runtime.add_song_from_path("Timeline Review Song", audio_path)
    after_stems = runtime.extract_stems(LayerId("source_audio"))
    drums_layer = next(layer for layer in after_stems.layers if layer.title == "Drums")
    classified = runtime.extract_classified_drums(drums_layer.layer_id)
    kick_layer = next(layer for layer in classified.layers if layer.title == "Kick")
    first_event = kick_layer.events[0]
    return runtime, kick_layer.layer_id, str(first_event.event_id), float(first_event.start), float(
        first_event.end
    )


def _runtime_event(runtime: AppShellRuntime, *, layer_id: LayerId, event_id: str):
    for layer in runtime._app.timeline.layers:
        if layer.id != layer_id:
            continue
        for take in layer.takes:
            for event in take.events:
                if str(event.id) == event_id:
                    return event
    raise AssertionError(f"Runtime event not found: {event_id}")


def test_app_shell_runtime_commit_missed_event_review_creates_signal_and_updates_runtime_state(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        before_layer = next(layer for layer in runtime.presentation().layers if layer.layer_id == layer_id)
        before_event_count = len(before_layer.events)
        before_event_ids = {str(event.event_id) for event in before_layer.events}

        reviewed = runtime.dispatch(
            CommitMissedEventReview(
                layer_id=layer_id,
                take_id=None,
                time_range=TimeRange(0.2, 0.38),
                label="Kick",
                source_event_id=event_id,
                payload_ref=event_id,
            )
        )

        updated_layer = next(layer for layer in reviewed.layers if layer.layer_id == layer_id)
        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[0]

        assert len(updated_layer.events) == before_event_count + 1
        created_event = next(
            event for event in updated_layer.events if str(event.event_id) not in before_event_ids
        )
        runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=str(created_event.event_id))
        assert signal.review_decision is not None
        assert signal.review_decision.kind == ReviewDecisionKind.MISSED_EVENT_ADDED
        assert signal.source_provenance["project_writeback"]["reason"] == "non_project_session"
        assert signal.source_provenance["dataset_materialization"]["status"] == "materialized"
        assert runtime_event.origin == "manual_added"
        assert runtime_event.metadata["review"]["promotion_state"] == "promoted"
        assert runtime_event.metadata["review"]["review_state"] == "corrected"
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_verified_event_review_creates_signal(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        reviewed = runtime.dispatch(
            CommitVerifiedEventReview(
                layer_id=layer_id,
                event_id=event_id,
                review_note="operator verified the detected hit",
            )
        )

        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[0]
        runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=event_id)

        assert any(layer.layer_id == layer_id for layer in reviewed.layers)
        assert signal.review_decision is not None
        assert signal.review_decision.kind == ReviewDecisionKind.VERIFIED
        assert signal.source_provenance["dataset_materialization"]["status"] == "materialized"
        assert runtime_event.metadata["review"]["promotion_state"] == "promoted"
        assert runtime_event.metadata["review"]["review_state"] == "signed_off"
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_verified_events_review_batches_signals(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer = next(layer for layer in runtime.presentation().layers if layer.layer_id == layer_id)
        event_ids = [str(event.event_id) for event in layer.events]
        if len(event_ids) < 2:
            runtime.dispatch(
                CommitMissedEventReview(
                    layer_id=layer_id,
                    take_id=layer.main_take_id,
                    time_range=TimeRange(0.2, 0.34),
                    label="Kick",
                    source_event_id=event_id,
                    payload_ref=event_id,
                )
            )
            layer = next(layer for layer in runtime.presentation().layers if layer.layer_id == layer_id)
            event_ids = [str(event.event_id) for event in layer.events]
        assert len(event_ids) >= 2
        target_ids = event_ids[:2]

        runtime.dispatch(
            CommitVerifiedEventsReview(
                event_refs=[
                    EventRef(
                        layer_id=layer_id,
                        take_id=layer.main_take_id,
                        event_id=EventId(target_ids[0]),
                    ),
                    EventRef(
                        layer_id=layer_id,
                        take_id=layer.main_take_id,
                        event_id=EventId(target_ids[1]),
                    ),
                ],
                review_note="batch verify",
            )
        )

        signal_repo = ReviewSignalRepository(runtime.project_storage.working_dir)
        verified_signals = [
            signal
            for signal in signal_repo.list()
            if signal.review_decision is not None
            and signal.review_decision.kind == ReviewDecisionKind.VERIFIED
            and signal.review_note == "batch verify"
        ]
        assert len(verified_signals) == 2
        for signal in verified_signals:
            dataset_materialization = signal.source_provenance.get("dataset_materialization", {})
            assert isinstance(dataset_materialization, dict)
            assert dataset_materialization.get("status") == "deferred"
        for target_id in target_ids:
            runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=target_id)
            assert runtime_event.metadata["review"]["promotion_state"] == "promoted"
            assert runtime_event.metadata["review"]["review_state"] == "signed_off"
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_rejected_event_review_demotes_event_and_creates_signal(
    tmp_path: Path,
):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        before_count = len(
            next(layer for layer in runtime.presentation().layers if layer.layer_id == layer_id).events
        )

        reviewed = runtime.dispatch(
            CommitRejectedEventReview(
                layer_id=layer_id,
                event_id=event_id,
                review_note="operator rejected the false positive",
            )
        )

        updated_layer = next(layer for layer in reviewed.layers if layer.layer_id == layer_id)
        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[0]
        runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=event_id)

        assert len(updated_layer.events) == before_count
        assert signal.review_decision is not None
        assert signal.review_decision.kind == ReviewDecisionKind.REJECTED
        assert signal.source_provenance["dataset_materialization"]["status"] == "materialized"
        assert runtime_event.metadata["review"]["promotion_state"] == "demoted"
        assert runtime_event.metadata["review"]["review_state"] == "corrected"
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_rejected_events_review_batches_signals(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer = next(layer for layer in runtime.presentation().layers if layer.layer_id == layer_id)
        event_ids = [str(event.event_id) for event in layer.events]
        if len(event_ids) < 2:
            runtime.dispatch(
                CommitMissedEventReview(
                    layer_id=layer_id,
                    take_id=layer.main_take_id,
                    time_range=TimeRange(0.2, 0.34),
                    label="Kick",
                    source_event_id=event_id,
                    payload_ref=event_id,
                )
            )
            layer = next(layer for layer in runtime.presentation().layers if layer.layer_id == layer_id)
            event_ids = [str(event.event_id) for event in layer.events]
        assert len(event_ids) >= 2
        target_ids = event_ids[:2]

        runtime.dispatch(
            CommitRejectedEventsReview(
                event_refs=[
                    EventRef(
                        layer_id=layer_id,
                        take_id=layer.main_take_id,
                        event_id=EventId(target_ids[0]),
                    ),
                    EventRef(
                        layer_id=layer_id,
                        take_id=layer.main_take_id,
                        event_id=EventId(target_ids[1]),
                    ),
                ],
                review_note="batch reject",
            )
        )

        signal_repo = ReviewSignalRepository(runtime.project_storage.working_dir)
        rejected_signals = [
            signal
            for signal in signal_repo.list()
            if signal.review_decision is not None
            and signal.review_decision.kind == ReviewDecisionKind.REJECTED
        ]
        assert len(rejected_signals) >= 2
        for target_id in target_ids:
            runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=target_id)
            assert runtime_event.metadata["review"]["promotion_state"] == "demoted"
            assert runtime_event.metadata["review"]["review_state"] == "corrected"
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_missed_events_review_batches_signals_and_creates_events(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        before_layer = next(layer for layer in runtime.presentation().layers if layer.layer_id == layer_id)
        before_event_ids = {str(event.event_id) for event in before_layer.events}

        reviewed = runtime.dispatch(
            CommitMissedEventsReview(
                intents=[
                    CommitMissedEventReview(
                        layer_id=layer_id,
                        take_id=before_layer.main_take_id,
                        time_range=TimeRange(0.25, 0.37),
                        label="Kick",
                        source_event_id=event_id,
                        payload_ref=event_id,
                    ),
                    CommitMissedEventReview(
                        layer_id=layer_id,
                        take_id=before_layer.main_take_id,
                        time_range=TimeRange(0.45, 0.57),
                        label="Kick",
                        source_event_id="synthetic_onset_b",
                        payload_ref="synthetic_onset_b",
                    ),
                ]
            )
        )

        updated_layer = next(layer for layer in reviewed.layers if layer.layer_id == layer_id)
        new_event_ids = [
            str(event.event_id)
            for event in updated_layer.events
            if str(event.event_id) not in before_event_ids
        ]
        assert len(new_event_ids) == 2

        signal_repo = ReviewSignalRepository(runtime.project_storage.working_dir)
        missed_signals = [
            signal
            for signal in signal_repo.list()
            if signal.review_decision is not None
            and signal.review_decision.kind == ReviewDecisionKind.MISSED_EVENT_ADDED
        ]
        assert len(missed_signals) >= 2
        for created_id in new_event_ids:
            runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=created_id)
            assert runtime_event.origin == "manual_added"
            assert runtime_event.metadata["review"]["promotion_state"] == "promoted"
            assert runtime_event.metadata["review"]["review_state"] == "corrected"
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_relabel_event_review_updates_label_and_creates_signal(
    tmp_path: Path,
):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        reviewed = runtime.dispatch(
            CommitRelabeledEventReview(
                layer_id=layer_id,
                event_id=event_id,
                corrected_label="Snare",
                review_note="operator relabeled the detected hit",
            )
        )

        updated_layer = next(layer for layer in reviewed.layers if layer.layer_id == layer_id)
        updated_event = next(event for event in updated_layer.events if str(event.event_id) == event_id)
        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[0]
        runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=event_id)

        assert updated_event.label == "Snare"
        assert signal.review_decision is not None
        assert signal.review_decision.kind == ReviewDecisionKind.RELABELED
        assert signal.corrected_label == "snare"
        assert signal.source_provenance["dataset_materialization"]["status"] == "materialized"
        assert runtime_event.metadata["review"]["original_label"] == "kick"
        assert runtime_event.metadata["review"]["corrected_label"] == "snare"
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_boundary_corrected_event_review_updates_timing_and_creates_signal(
    tmp_path: Path,
):
    runtime, layer_id, event_id, start, end = _build_timeline_review_runtime(tmp_path)
    try:
        corrected_range = TimeRange(start + 0.04, end + 0.06)
        reviewed = runtime.dispatch(
            CommitBoundaryCorrectedEventReview(
                layer_id=layer_id,
                event_id=event_id,
                corrected_range=corrected_range,
                review_note="operator corrected the event boundary",
            )
        )

        updated_layer = next(layer for layer in reviewed.layers if layer.layer_id == layer_id)
        updated_event = next(event for event in updated_layer.events if str(event.event_id) == event_id)
        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[0]
        runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=event_id)

        assert updated_event.start == corrected_range.start
        assert updated_event.end == corrected_range.end
        assert signal.review_decision is not None
        assert signal.review_decision.kind == ReviewDecisionKind.BOUNDARY_CORRECTED
        assert signal.review_decision.corrected_start_ms == corrected_range.start * 1000.0
        assert signal.review_decision.corrected_end_ms == corrected_range.end * 1000.0
        assert signal.source_provenance["dataset_materialization"]["status"] == "materialized"
        assert runtime_event.metadata["review"]["review_state"] == "corrected"
        assert runtime_event.metadata["review"]["corrected_start_ms"] == corrected_range.start * 1000.0
    finally:
        runtime.shutdown()

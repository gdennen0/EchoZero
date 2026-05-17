"""App-shell timeline-review tests for the canonical Stage Zero runtime.
Exists because timeline fix-mode review commits must be proven through AppShellRuntime.
Connects explicit missed-event review intents to durable Foundry signal materialization.
"""

from __future__ import annotations

from copy import deepcopy
import json
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
from echozero.foundry.services.review_audio_clip_service import ReviewAudioClipService
from echozero.foundry.services.review_pipeline_controller import ReviewPipelineController
from echozero.testing.analysis_mocks import build_mock_analysis_service, write_test_wav
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.app_shell_history import RuntimeScopedHistorySnapshot


def _build_timeline_review_runtime(
    tmp_path: Path,
) -> tuple[AppShellRuntime, LayerId, str, float, float]:
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
    return (
        runtime,
        kick_layer.layer_id,
        str(first_event.event_id),
        float(first_event.start),
        float(first_event.end),
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


def _flush_deferred_review_persistence(runtime: AppShellRuntime) -> None:
    runtime.flush_deferred_review_persistence()


class _PlayingRuntimeAudio:
    def is_playing(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass


def _ensure_two_review_target_ids(
    runtime: AppShellRuntime,
    *,
    layer_id: LayerId,
    seed_event_id: str,
) -> tuple[object, list[str]]:
    layer = next(layer for layer in runtime.presentation().layers if layer.layer_id == layer_id)
    event_ids = [str(event.event_id) for event in layer.events]
    if len(event_ids) < 2:
        runtime.dispatch(
            CommitMissedEventReview(
                layer_id=layer_id,
                take_id=layer.main_take_id,
                time_range=TimeRange(0.2, 0.34),
                label="Kick",
                source_event_id=seed_event_id,
                payload_ref=seed_event_id,
            )
        )
        layer = next(
            layer for layer in runtime.presentation().layers if layer.layer_id == layer_id
        )
        event_ids = [str(event.event_id) for event in layer.events]
    assert len(event_ids) >= 2
    return layer, event_ids[:2]


def test_timeline_fix_mode_routes_review_commits_through_shared_pipeline_controller(
    tmp_path: Path,
    monkeypatch,
):
    captured_commands = []
    original_commit = ReviewPipelineController.commit

    def _capture_commit(self, command):
        captured_commands.append(command)
        return original_commit(self, command)

    monkeypatch.setattr(ReviewPipelineController, "commit", _capture_commit)
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        runtime.dispatch(
            CommitVerifiedEventReview(
                layer_id=layer_id,
                event_id=event_id,
                review_note="controller seam check",
            )
        )
        _flush_deferred_review_persistence(runtime)
    finally:
        runtime.shutdown()

    assert len(captured_commands) == 1
    command = captured_commands[0]
    assert command.context.session_id.startswith("timeline_review_")
    assert command.context.metadata["queue_source_kind"] == "timeline_review_mode"
    assert command.commit.item_id.startswith("timeline_review:")
    assert command.commit.review_decision is not None
    assert command.commit.review_decision.kind == ReviewDecisionKind.VERIFIED
    assert command.commit.source_provenance["project_ref"].startswith("project:")
    assert command.commit.source_provenance["song_ref"].startswith("song:")
    assert command.commit.source_provenance["layer_ref"].startswith("layer:")
    assert command.commit.source_provenance["event_ref"].startswith("event:")
    assert "projectRef" not in command.commit.source_provenance
    assert "songRef" not in command.commit.source_provenance


def test_app_shell_runtime_commit_missed_event_review_creates_signal_and_updates_runtime_state(
    tmp_path: Path,
):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        before_layer = next(
            layer for layer in runtime.presentation().layers if layer.layer_id == layer_id
        )
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
        _flush_deferred_review_persistence(runtime)

        updated_layer = next(layer for layer in reviewed.layers if layer.layer_id == layer_id)
        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[0]

        assert len(updated_layer.events) == before_event_count + 1
        created_event = next(
            event for event in updated_layer.events if str(event.event_id) not in before_event_ids
        )
        runtime_event = _runtime_event(
            runtime, layer_id=layer_id, event_id=str(created_event.event_id)
        )
        assert signal.review_decision is not None
        assert signal.review_decision.kind == ReviewDecisionKind.MISSED_EVENT_ADDED
        assert signal.source_provenance["project_writeback"]["reason"] == "non_project_session"
        assert signal.source_provenance["dataset_materialization"]["status"] == "deferred"
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
        _flush_deferred_review_persistence(runtime)

        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[0]
        runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=event_id)

        assert any(layer.layer_id == layer_id for layer in reviewed.layers)
        assert signal.review_decision is not None
        assert signal.review_decision.kind == ReviewDecisionKind.VERIFIED
        assert signal.source_provenance["dataset_materialization"]["status"] == "deferred"
        assert runtime_event.metadata["review"]["promotion_state"] == "promoted"
        assert runtime_event.metadata["review"]["review_state"] == "signed_off"
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_verified_events_review_batches_signals(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer, target_ids = _ensure_two_review_target_ids(
            runtime,
            layer_id=layer_id,
            seed_event_id=event_id,
        )

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
        _flush_deferred_review_persistence(runtime)

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


def test_app_shell_runtime_commit_verified_events_review_supports_undo(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer, target_ids = _ensure_two_review_target_ids(
            runtime,
            layer_id=layer_id,
            seed_event_id=event_id,
        )
        before_review_metadata = {
            target_id: deepcopy(
                _runtime_event(runtime, layer_id=layer_id, event_id=target_id).metadata.get(
                    "review"
                )
            )
            for target_id in target_ids
        }

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
                review_note="batch verify undo",
            )
        )

        assert runtime.can_undo() is True
        assert runtime.undo_label() == "Verify Events"

        runtime.undo()

        for target_id in target_ids:
            runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=target_id)
            assert runtime_event.metadata.get("review") == before_review_metadata[target_id]
    finally:
        runtime.shutdown()


def test_app_shell_defers_storage_sync_for_review_batch_while_playing(
    tmp_path: Path,
    monkeypatch,
):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer, target_ids = _ensure_two_review_target_ids(
            runtime,
            layer_id=layer_id,
            seed_event_id=event_id,
        )
        runtime.runtime_audio = _PlayingRuntimeAudio()
        synced_timeline: list[str] = []
        synced_layers: list[list[LayerId]] = []
        monkeypatch.setattr(
            runtime,
            "_sync_storage_backed_timeline",
            lambda: synced_timeline.append("timeline"),
        )
        monkeypatch.setattr(
            runtime,
            "_sync_storage_backed_layers",
            lambda layer_ids: synced_layers.append(list(layer_ids)),
        )

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
                review_note="batch reject while playing",
            )
        )

        assert synced_timeline == []
        assert synced_layers == []
        for target_id in target_ids:
            runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=target_id)
            assert runtime_event.metadata["review"]["promotion_state"] == "demoted"

        runtime._flush_deferred_storage_sync()

        assert synced_timeline == []
        assert synced_layers == [[layer_id]]
    finally:
        runtime.shutdown()


def test_app_shell_review_batch_uses_scoped_history_snapshot(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer, target_ids = _ensure_two_review_target_ids(
            runtime,
            layer_id=layer_id,
            seed_event_id=event_id,
        )

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
            )
        )

        entry = runtime._history._undo[-1]
        assert isinstance(entry.before, RuntimeScopedHistorySnapshot)
        assert isinstance(entry.after, RuntimeScopedHistorySnapshot)
        assert set(entry.before.layers) == {layer_id}
        assert set(entry.after.layers) == {layer_id}

        runtime.undo()
        for target_id in target_ids:
            runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=target_id)
            assert runtime_event.metadata.get("review", {}).get("promotion_state") != "demoted"
    finally:
        runtime.shutdown()


def test_app_shell_review_batch_assembles_presentation_once_after_commit(
    tmp_path: Path,
    monkeypatch,
):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer, target_ids = _ensure_two_review_target_ids(
            runtime,
            layer_id=layer_id,
            seed_event_id=event_id,
        )
        presentation_count = 0
        original_presentation = runtime.presentation

        def _counted_presentation():
            nonlocal presentation_count
            presentation_count += 1
            return original_presentation()

        monkeypatch.setattr(runtime, "presentation", _counted_presentation)

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
            )
        )

        assert presentation_count == 2
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_rejected_event_review_demotes_event_and_creates_signal(
    tmp_path: Path,
):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        before_count = len(
            next(
                layer for layer in runtime.presentation().layers if layer.layer_id == layer_id
            ).events
        )

        reviewed = runtime.dispatch(
            CommitRejectedEventReview(
                layer_id=layer_id,
                event_id=event_id,
                review_note="operator rejected the false positive",
            )
        )
        _flush_deferred_review_persistence(runtime)

        updated_layer = next(layer for layer in reviewed.layers if layer.layer_id == layer_id)
        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[0]
        runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=event_id)

        assert len(updated_layer.events) == before_count
        assert signal.review_decision is not None
        assert signal.review_decision.kind == ReviewDecisionKind.REJECTED
        assert signal.source_provenance["dataset_materialization"]["status"] == "deferred"
        assert runtime_event.metadata["review"]["promotion_state"] == "demoted"
        assert runtime_event.metadata["review"]["review_state"] == "corrected"
    finally:
        runtime.shutdown()


def test_app_shell_runtime_rejected_review_updates_selected_event_presentation_immediately(
    tmp_path: Path,
):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer = next(layer for layer in runtime._app.timeline.layers if layer.id == layer_id)
        take = layer.takes[0]
        runtime._app.timeline.selection.selected_layer_id = layer_id
        runtime._app.timeline.selection.selected_layer_ids = [layer_id]
        runtime._app.timeline.selection.selected_take_id = take.id
        runtime._app.timeline.selection.selected_event_refs = [
            EventRef(
                layer_id=layer_id,
                take_id=take.id,
                event_id=EventId(event_id),
            )
        ]
        runtime._app.timeline.selection.selected_event_ids = [EventId(event_id)]

        reviewed = runtime.dispatch(
            CommitRejectedEventReview(
                layer_id=layer_id,
                event_id=event_id,
                review_note="selected event immediate demote",
            )
        )

        updated_layer = next(layer for layer in reviewed.layers if layer.layer_id == layer_id)
        updated_event = next(event for event in updated_layer.events if str(event.event_id) == event_id)

        assert updated_event.is_selected is True
        assert "demoted" in updated_event.badges
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_rejected_events_review_batches_signals(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer, target_ids = _ensure_two_review_target_ids(
            runtime,
            layer_id=layer_id,
            seed_event_id=event_id,
        )

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
        _flush_deferred_review_persistence(runtime)

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


def test_app_shell_runtime_batch_rejected_review_reuses_clip_audio_cache(
    tmp_path: Path,
    monkeypatch,
):
    export_root = tmp_path / "review-sample-export"
    monkeypatch.setenv("ECHOZERO_REVIEW_SAMPLE_EXPORT_ROOT", str(export_root))
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer, target_ids = _ensure_two_review_target_ids(
            runtime,
            layer_id=layer_id,
            seed_event_id=event_id,
        )
        read_count = 0
        soundfile_module = ReviewAudioClipService._soundfile()

        class CountingSoundfile:
            @staticmethod
            def read(*args, **kwargs):
                nonlocal read_count
                read_count += 1
                return soundfile_module.read(*args, **kwargs)

            @staticmethod
            def write(*args, **kwargs):
                return soundfile_module.write(*args, **kwargs)

        monkeypatch.setattr(
            ReviewAudioClipService,
            "_soundfile",
            staticmethod(lambda: CountingSoundfile),
        )

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
                review_note="batch reject cache",
            )
        )
        _flush_deferred_review_persistence(runtime)

        assert read_count == 1
        exported = sorted((export_root / "kick").glob("*.wav"))
        assert len(exported) >= 2
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_rejected_events_review_supports_undo(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        layer, target_ids = _ensure_two_review_target_ids(
            runtime,
            layer_id=layer_id,
            seed_event_id=event_id,
        )
        before_review_metadata = {
            target_id: deepcopy(
                _runtime_event(runtime, layer_id=layer_id, event_id=target_id).metadata.get(
                    "review"
                )
            )
            for target_id in target_ids
        }

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
                review_note="batch reject undo",
            )
        )

        assert runtime.can_undo() is True
        assert runtime.undo_label() == "Reject Events"

        runtime.undo()

        for target_id in target_ids:
            runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=target_id)
            assert runtime_event.metadata.get("review") == before_review_metadata[target_id]
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_missed_events_review_batches_signals_and_creates_events(
    tmp_path: Path,
):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        before_layer = next(
            layer for layer in runtime.presentation().layers if layer.layer_id == layer_id
        )
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
        _flush_deferred_review_persistence(runtime)

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


def test_app_shell_runtime_commit_missed_events_review_supports_undo(tmp_path: Path):
    runtime, layer_id, event_id, _start, _end = _build_timeline_review_runtime(tmp_path)
    try:
        before_layer = next(
            layer for layer in runtime.presentation().layers if layer.layer_id == layer_id
        )
        before_event_ids = {str(event.event_id) for event in before_layer.events}

        runtime.dispatch(
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

        assert runtime.can_undo() is True
        assert runtime.undo_label() == "Add Missed Events"

        undone = runtime.undo()
        undone_layer = next(layer for layer in undone.layers if layer.layer_id == layer_id)
        undone_event_ids = {str(event.event_id) for event in undone_layer.events}
        assert undone_event_ids == before_event_ids
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_relabel_event_review_updates_label_and_creates_signal(
    tmp_path: Path,
    monkeypatch,
):
    export_root = tmp_path / "review-sample-export"
    monkeypatch.setenv("ECHOZERO_REVIEW_SAMPLE_EXPORT_ROOT", str(export_root))
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
        _flush_deferred_review_persistence(runtime)

        updated_layer = next(layer for layer in reviewed.layers if layer.layer_id == layer_id)
        updated_event = next(
            event for event in updated_layer.events if str(event.event_id) == event_id
        )
        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[0]
        runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=event_id)

        assert updated_event.label == "Snare"
        assert signal.review_decision is not None
        assert signal.review_decision.kind == ReviewDecisionKind.RELABELED
        assert signal.corrected_label == "snare"
        assert signal.source_provenance["dataset_materialization"]["status"] == "deferred"
        assert runtime_event.metadata["review"]["original_label"] == "kick"
        assert runtime_event.metadata["review"]["corrected_label"] == "snare"
        exported = sorted((export_root / "snare").glob("*.wav"))
        assert exported
        manifest_path = export_root / "manifest.jsonl"
        assert manifest_path.exists()
        manifest_rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert manifest_rows
        assert any(row["decision_kind"] == "relabeled" for row in manifest_rows)
        assert any(row["class_label"] == "snare" for row in manifest_rows)
        relabeled_row = next(row for row in manifest_rows if row["decision_kind"] == "relabeled")
        assert Path(relabeled_row["clip_path"]).is_absolute() is False
        assert Path(relabeled_row["source_audio_path"]).is_absolute() is False
        assert (export_root / relabeled_row["clip_path"]).exists()
    finally:
        runtime.shutdown()


def test_app_shell_runtime_commit_boundary_corrected_event_review_updates_timing_and_creates_signal(
    tmp_path: Path,
    monkeypatch,
):
    export_root = tmp_path / "review-sample-export"
    monkeypatch.setenv("ECHOZERO_REVIEW_SAMPLE_EXPORT_ROOT", str(export_root))
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
        _flush_deferred_review_persistence(runtime)

        updated_layer = next(layer for layer in reviewed.layers if layer.layer_id == layer_id)
        updated_event = next(
            event for event in updated_layer.events if str(event.event_id) == event_id
        )
        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[0]
        runtime_event = _runtime_event(runtime, layer_id=layer_id, event_id=event_id)

        assert updated_event.start == corrected_range.start
        assert updated_event.end == corrected_range.end
        assert signal.review_decision is not None
        assert signal.review_decision.kind == ReviewDecisionKind.BOUNDARY_CORRECTED
        assert signal.review_decision.corrected_start_ms == corrected_range.start * 1000.0
        assert signal.review_decision.corrected_end_ms == corrected_range.end * 1000.0
        assert signal.source_provenance["dataset_materialization"]["status"] == "deferred"
        assert runtime_event.metadata["review"]["review_state"] == "corrected"
        assert (
            runtime_event.metadata["review"]["corrected_start_ms"]
            == corrected_range.start * 1000.0
        )
        exported = sorted((export_root / "kick").glob("*.wav"))
        assert exported
        manifest_rows = [
            json.loads(line)
            for line in (export_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(row["decision_kind"] == "boundary_corrected" for row in manifest_rows)
        corrected_row = next(
            row for row in manifest_rows if row["decision_kind"] == "boundary_corrected"
        )
        assert Path(corrected_row["clip_path"]).is_absolute() is False
        assert Path(corrected_row["source_audio_path"]).is_absolute() is False
        assert (export_root / corrected_row["clip_path"]).exists()
    finally:
        runtime.shutdown()

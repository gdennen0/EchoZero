"""Focused proof for the durable Foundry review-signal seam.
Exists because review sessions must stay queue state while explicit reviews persist canonically.
Connects review-session commits to the dedicated review-signal repository contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.foundry.domain import (
    ExplicitReviewCommit,
    ReviewCommitCommand,
    ReviewCommitContext,
    ReviewDecisionKind,
    ReviewOutcome,
    ReviewPolarity,
    build_review_decision,
)
from echozero.foundry.persistence import (
    ReviewSessionRepository,
    ReviewSignalRepository,
)
from echozero.foundry.services.review_session_service import ReviewSessionService
from echozero.foundry.services.review_extraction_service import ReviewExtractionService
from echozero.foundry.services.review_commit_mapper import (
    normalize_review_payload,
    normalize_source_provenance,
)
from echozero.foundry.services.review_pipeline_controller import ReviewPipelineController
from echozero.foundry.services.review_signal_service import ReviewSignalService
from echozero.persistence.session import ProjectStorage
from tests.foundry.test_review_project_queue_builder import _build_project_review_fixture
from tests.foundry.audio_fixtures import write_percussion_dataset


def test_review_import_does_not_emit_durable_signal_rows(tmp_path: Path):
    service = ReviewSessionService(tmp_path)

    session = service.import_session_file(_write_review_items_json(tmp_path))

    assert ReviewSessionRepository(tmp_path).get(session.id) is not None
    assert ReviewSignalRepository(tmp_path).list() == []


def test_explicit_review_writes_one_durable_signal_record(tmp_path: Path):
    service = ReviewSessionService(tmp_path)

    session = service.import_session_file(_write_review_items_json(tmp_path))
    reviewed = service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="tom",
        review_note="operator heard a tom, not a kick",
    )
    signal = ReviewSignalRepository(tmp_path).get(
        ReviewSignalService.build_signal_id(session.id, session.items[0].item_id)
    )

    assert reviewed.items[0].review_decision is not None
    assert signal is not None
    assert signal.review_outcome == ReviewOutcome.INCORRECT
    assert signal.review_decision is not None
    assert signal.review_decision.kind == ReviewDecisionKind.RELABELED
    assert signal.review_decision.provenance is not None
    assert signal.review_decision.provenance.queue_session_ref == session.id
    assert signal.review_decision.provenance.project_ref == "project:fixture"
    assert signal.corrected_label == "tom"
    assert signal.review_note == "operator heard a tom, not a kick"
    assert signal.source_provenance["project_writeback"]["reason"] == "non_project_session"
    assert signal.source_provenance["dataset_materialization"]["status"] == "deferred"


def test_shared_explicit_review_commit_api_supports_non_session_producers(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples, sample_count=1)
    service = ReviewSignalService(tmp_path)

    signal = service.record_explicit_review(
        ReviewCommitContext(
            session_id="rev_timeline",
            session_name="Timeline Corrections",
            source_ref=str(samples.resolve()),
            metadata={"queue_source_kind": "timeline_review"},
        ),
        ExplicitReviewCommit(
            item_id="timeline:event:k1",
            audio_path=str(samples / "kick" / "k1.wav"),
            predicted_label="kick",
            target_class="kick",
            polarity=ReviewPolarity.POSITIVE,
            score=0.91,
            source_provenance={
                "project_ref": "project:fixture",
                "song_ref": "song:arcade",
                "layer_ref": "layer:kick",
                "event_ref": "event:kick-01",
                "model_ref": "bundle:fixture-v1",
            },
            review_outcome=ReviewOutcome.INCORRECT,
            review_decision=build_review_decision(
                ReviewOutcome.INCORRECT,
                corrected_label="tom",
                review_note="timeline correction",
            ),
            corrected_label="tom",
            review_note="timeline correction",
        ),
    )
    persisted = ReviewSignalRepository(tmp_path).get(signal.id)

    assert signal.id == ReviewSignalService.build_signal_id("rev_timeline", "timeline:event:k1")
    assert persisted is not None
    assert persisted.review_decision is not None
    assert persisted.review_decision.kind == ReviewDecisionKind.RELABELED
    assert persisted.source_provenance["project_writeback"]["reason"] == "non_project_session"
    assert persisted.source_provenance["dataset_materialization"]["status"] == "deferred"


def test_explicit_re_review_updates_existing_signal_record(tmp_path: Path):
    service = ReviewSessionService(tmp_path)

    session = service.import_session_file(_write_review_items_json(tmp_path))
    service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="tom",
        review_note="first pass relabel",
    )
    signal_id = ReviewSignalService.build_signal_id(session.id, session.items[0].item_id)
    first_signal = ReviewSignalRepository(tmp_path).get(signal_id)
    service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.CORRECT,
        corrected_label=None,
        review_note=None,
    )
    updated_signal = ReviewSignalRepository(tmp_path).get(signal_id)

    assert first_signal is not None
    assert updated_signal is not None
    assert len(ReviewSignalRepository(tmp_path).list()) == 1
    assert updated_signal.created_at == first_signal.created_at
    assert updated_signal.updated_at >= first_signal.updated_at
    assert updated_signal.review_outcome == ReviewOutcome.CORRECT
    assert updated_signal.review_decision is not None
    assert updated_signal.review_decision.kind == ReviewDecisionKind.VERIFIED
    assert updated_signal.corrected_label is None
    assert updated_signal.review_note is None


def test_review_signal_repository_caches_deserialized_rows_across_repeated_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service = ReviewSessionService(tmp_path)
    session = service.import_session_file(_write_review_items_json(tmp_path))
    service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.CORRECT,
        corrected_label=None,
        review_note=None,
    )

    deserialize_calls = 0
    original_deserialize = ReviewSignalRepository._deserialize

    def _deserialize_spy(row):
        nonlocal deserialize_calls
        deserialize_calls += 1
        return original_deserialize(row)

    monkeypatch.setattr(
        ReviewSignalRepository,
        "_deserialize",
        staticmethod(_deserialize_spy),
    )

    repository = ReviewSignalRepository(tmp_path)
    first = repository.list_for_session(session.id)
    second = repository.list_for_session(session.id)
    by_id = repository.get(first[0].id)
    all_rows = repository.list()

    assert len(first) == 1
    assert len(second) == 1
    assert by_id is not None
    assert len(all_rows) == 1
    assert deserialize_calls == 1


def test_timeline_and_phone_provenance_keys_normalize_to_same_contract():
    phone_source = {
        "projectRef": "project:fixture",
        "songRef": "song:arcade",
        "layerRef": "layer:kick",
        "eventRef": "event:kick-01",
        "sourceEventRef": "event:kick-01",
        "modelRef": "bundle:fixture-v1",
    }
    timeline_source = {
        "project_ref": "project:fixture",
        "song_ref": "song:arcade",
        "layer_ref": "layer:kick",
        "event_ref": "event:kick-01",
        "source_event_ref": "event:kick-01",
        "model_ref": "bundle:fixture-v1",
    }

    normalized_phone = normalize_source_provenance(phone_source)
    normalized_timeline = normalize_source_provenance(timeline_source)

    for key in (
        "project_ref",
        "song_ref",
        "layer_ref",
        "event_ref",
        "source_event_ref",
        "model_ref",
    ):
        assert normalized_phone[key] == normalized_timeline[key]


def test_timeline_and_phone_review_payload_keys_normalize_to_same_contract():
    phone_payload = {
        "itemId": "ri_equiv",
        "outcome": "incorrect",
        "correctedLabel": "tom",
        "reviewNote": "equivalent decision",
        "decisionKind": "relabeled",
        "originalStartMs": 120.0,
        "originalEndMs": 180.0,
        "correctedStartMs": 125.0,
        "correctedEndMs": 175.0,
        "createdEventRef": "event:created-01",
        "operatorAction": "relabel_event",
    }
    timeline_payload = {
        "item_id": "ri_equiv",
        "outcome": "incorrect",
        "corrected_label": "tom",
        "review_note": "equivalent decision",
        "decision_kind": "relabeled",
        "original_start_ms": 120.0,
        "original_end_ms": 180.0,
        "corrected_start_ms": 125.0,
        "corrected_end_ms": 175.0,
        "created_event_ref": "event:created-01",
        "operator_action": "relabel_event",
    }

    normalized_phone = normalize_review_payload(phone_payload)
    normalized_timeline = normalize_review_payload(timeline_payload)

    for key in (
        "item_id",
        "outcome",
        "corrected_label",
        "review_note",
        "decision_kind",
        "original_start_ms",
        "original_end_ms",
        "corrected_start_ms",
        "corrected_end_ms",
        "created_event_ref",
        "operator_action",
    ):
        assert normalized_phone[key] == normalized_timeline[key]


def test_timeline_and_phone_equivalent_commits_persist_equivalent_signal_payloads(tmp_path: Path):
    controller = ReviewPipelineController(tmp_path)
    context = ReviewCommitContext(
        session_id="rev_equiv",
        session_name="Equivalence Session",
        source_ref=str(tmp_path.resolve()),
        metadata={"queue_source_kind": "equivalence_test"},
    )
    decision = build_review_decision(
        ReviewOutcome.INCORRECT,
        corrected_label="tom",
        review_note="equivalent decision",
        decision_kind=ReviewDecisionKind.RELABELED,
    )
    assert decision is not None

    controller.commit(
        ReviewCommitCommand(
            context=context,
            commit=ExplicitReviewCommit(
                item_id="ri_equiv",
                audio_path=str((tmp_path / "kick.wav").resolve()),
                predicted_label="kick",
                target_class="kick",
                polarity=ReviewPolarity.POSITIVE,
                source_provenance={
                    "projectRef": "project:fixture",
                    "songRef": "song:arcade",
                    "layerRef": "layer:kick",
                    "eventRef": "event:kick-01",
                    "sourceEventRef": "event:kick-src-01",
                    "modelRef": "bundle:fixture-v1",
                },
                review_outcome=ReviewOutcome.INCORRECT,
                review_decision=decision,
                corrected_label="tom",
                review_note="equivalent decision",
            ),
        )
    )
    first = ReviewSignalRepository(tmp_path).get(
        ReviewSignalService.build_signal_id(context.session_id, "ri_equiv")
    )
    assert first is not None

    controller.commit(
        ReviewCommitCommand(
            context=context,
            commit=ExplicitReviewCommit(
                item_id="ri_equiv",
                audio_path=str((tmp_path / "kick.wav").resolve()),
                predicted_label="kick",
                target_class="kick",
                polarity=ReviewPolarity.POSITIVE,
                source_provenance={
                    "project_ref": "project:fixture",
                    "song_ref": "song:arcade",
                    "layer_ref": "layer:kick",
                    "event_ref": "event:kick-01",
                    "source_event_ref": "event:kick-src-01",
                    "model_ref": "bundle:fixture-v1",
                },
                review_outcome=ReviewOutcome.INCORRECT,
                review_decision=decision,
                corrected_label="tom",
                review_note="equivalent decision",
            ),
        )
    )
    second = ReviewSignalRepository(tmp_path).get(
        ReviewSignalService.build_signal_id(context.session_id, "ri_equiv")
    )
    assert second is not None

    normalized_first = normalize_source_provenance(second.source_provenance)
    normalized_second = normalize_source_provenance(first.source_provenance)
    for key in (
        "project_ref",
        "song_ref",
        "layer_ref",
        "event_ref",
        "source_event_ref",
        "model_ref",
    ):
        assert normalized_first.get(key) == normalized_second.get(key)


def test_project_backed_review_relabel_writes_back_and_defers_local_dataset_extraction(tmp_path: Path):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.create_project_session(
        working_dir,
        name="Kick Corrections",
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )
    reviewed = service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="tom",
        review_note="operator confirmed this was a tom hit",
    )
    signal_id = ReviewSignalService.build_signal_id(session.id, session.items[0].item_id)
    signal = ReviewSignalRepository(tmp_path).get(signal_id)

    assert reviewed.items[0].review_decision is not None
    assert signal is not None
    assert signal.source_provenance["project_writeback"]["status"] == "applied"
    assert signal.source_provenance["dataset_materialization"]["status"] == "deferred"

    with ProjectStorage.open_db(working_dir) as project:
        take = project.takes.get_main("layer_alpha_kick")
        assert take is not None
        event = next(
            candidate
            for candidate in take.data.layers[0].events
            if candidate.id == "evt_alpha_kick_01"
        )
        assert event.classifications["class"] == "tom"
        assert event.metadata["foundry_review"]["decision_kind"] == "relabeled"
        assert event.metadata["foundry_review"]["signal_id"] == signal_id
        assert event.metadata["review"]["promotion_state"] == "promoted"
        assert event.metadata["review"]["review_state"] == "corrected"
        assert event.metadata["review"]["corrected_label"] == "tom"


def test_project_backed_review_reject_demotes_event_in_canonical_truth(tmp_path: Path):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.create_project_session(
        working_dir,
        name="Kick Rejections",
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )
    reviewed = service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label=None,
        review_note="operator rejected the false positive kick",
    )
    signal_id = ReviewSignalService.build_signal_id(session.id, session.items[0].item_id)
    signal = ReviewSignalRepository(tmp_path).get(signal_id)

    assert reviewed.items[0].review_decision is not None
    assert reviewed.items[0].review_decision.kind == ReviewDecisionKind.REJECTED
    assert signal is not None
    assert signal.source_provenance["project_writeback"]["status"] == "applied"

    with ProjectStorage.open_db(working_dir) as project:
        take = project.takes.get_main("layer_alpha_kick")
        assert take is not None
        event = next(
            candidate
            for candidate in take.data.layers[0].events
            if candidate.id == "evt_alpha_kick_01"
        )
        assert event.metadata["review"]["promotion_state"] == "demoted"
        assert event.metadata["review"]["review_state"] == "corrected"
        assert event.metadata["review"]["decision_kind"] == "rejected"


def test_boundary_corrected_review_skips_dataset_materialization_without_source_audio_provenance(
    tmp_path: Path,
):
    service = ReviewSessionService(tmp_path)
    session = service.import_session_file(_write_review_items_json(tmp_path))

    service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="kick",
        review_note="retimed boundary without project-backed provenance",
        decision_kind=ReviewDecisionKind.BOUNDARY_CORRECTED,
        original_start_ms=100.0,
        original_end_ms=150.0,
        corrected_start_ms=110.0,
        corrected_end_ms=165.0,
    )
    signal = ReviewSignalRepository(tmp_path).get(
        ReviewSignalService.build_signal_id(session.id, session.items[0].item_id)
    )

    assert signal is not None
    assert signal.source_provenance["project_writeback"]["reason"] == "non_project_session"
    assert signal.source_provenance["dataset_materialization"]["status"] == "deferred"


def test_explicit_review_hot_path_defers_extraction_by_default(tmp_path: Path):
    service = ReviewSignalService(tmp_path)
    session = ReviewSessionService(tmp_path).import_session_file(_write_review_items_json(tmp_path))

    signal = service.record_explicit_review(
        ReviewCommitContext(
            session_id=session.id,
            session_name=session.name,
            source_ref=session.source_ref,
            metadata=dict(session.metadata),
        ),
        ExplicitReviewCommit(
            item_id=session.items[0].item_id,
            audio_path=session.items[0].audio_path,
            predicted_label=session.items[0].predicted_label,
            target_class=session.items[0].target_class,
            polarity=session.items[0].polarity,
            score=session.items[0].score,
            source_provenance=dict(session.items[0].source_provenance),
            review_outcome=ReviewOutcome.CORRECT,
            review_decision=build_review_decision(
                ReviewOutcome.CORRECT,
                corrected_label=None,
                review_note="hot-path default",
            ),
            corrected_label=None,
            review_note=None,
        ),
    )

    assert signal.source_provenance["dataset_materialization"]["status"] == "deferred"


def test_explicit_review_commit_defers_extraction_to_explicit_pipeline(tmp_path: Path):
    service = ReviewSignalService(tmp_path)
    session = ReviewSessionService(tmp_path).import_session_file(_write_review_items_json(tmp_path))

    signal = service.record_explicit_review(
        ReviewCommitContext(
            session_id=session.id,
            session_name=session.name,
            source_ref=session.source_ref,
            metadata=dict(session.metadata),
        ),
        ExplicitReviewCommit(
            item_id=session.items[0].item_id,
            audio_path=session.items[0].audio_path,
            predicted_label=session.items[0].predicted_label,
            target_class=session.items[0].target_class,
            polarity=session.items[0].polarity,
            score=session.items[0].score,
            source_provenance=dict(session.items[0].source_provenance),
            review_outcome=ReviewOutcome.CORRECT,
            review_decision=build_review_decision(ReviewOutcome.CORRECT, corrected_label=None, review_note=None),
            corrected_label=None,
            review_note=None,
        ),
    )

    dataset_state = signal.source_provenance["dataset_materialization"]
    assert dataset_state["status"] == "deferred"
    assert dataset_state["reason"] == "explicit_extraction_only"
    assert dataset_state["next_step"] == "use_review_extraction_service"


def test_review_session_updates_route_through_shared_review_pipeline_controller(tmp_path: Path):
    class _PipelineSpy:
        def __init__(self):
            self.commands: list[ReviewCommitCommand] = []

        def commit(self, command: ReviewCommitCommand):
            self.commands.append(command)
            return ReviewPipelineController(tmp_path).commit(command)

    spy = _PipelineSpy()
    service = ReviewSessionService(tmp_path, pipeline_controller=spy)  # type: ignore[arg-type]
    session = service.import_session_file(_write_review_items_json(tmp_path))

    service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="tom",
        review_note="controller contract check",
    )

    assert len(spy.commands) == 1
    command = spy.commands[0]
    assert command.context.session_id == session.id
    assert command.commit.item_id == session.items[0].item_id
    assert command.commit.review_decision is not None
    assert command.commit.review_decision.kind == ReviewDecisionKind.RELABELED


def test_review_extraction_service_materializes_project_review_signal_on_explicit_request(tmp_path: Path):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    service = ReviewSessionService(tmp_path)

    session = service.create_project_session(
        working_dir,
        name="Kick Corrections",
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )
    reviewed = service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="tom",
        review_note="deferred extraction then explicit materialization",
    )
    reviewed_item = reviewed.items[0]
    signal_id = ReviewSignalService.build_signal_id(session.id, reviewed_item.item_id)
    signal = ReviewSignalRepository(tmp_path).get(signal_id)

    assert signal is not None
    assert signal.source_provenance["dataset_materialization"]["status"] == "deferred"

    extraction = ReviewExtractionService(tmp_path)
    result = extraction.extract_review_signal(
        ReviewCommitContext(
            session_id=reviewed.id,
            session_name=reviewed.name,
            source_ref=reviewed.source_ref,
            metadata=dict(reviewed.metadata),
        ),
        signal,
    )

    assert result["status"] == "materialized"
    assert str(result["dataset_id"]).startswith("ds_")
    assert str(result["version_id"]).startswith("dsv_")


def _write_review_items_json(tmp_path: Path) -> Path:
    samples = tmp_path / "samples"
    write_percussion_dataset(samples, sample_count=1)
    payload = [
        {
            "item_id": "ri_kick",
            "audio_path": str(samples / "kick" / "k1.wav"),
            "predicted_label": "kick",
            "target_class": "kick",
            "polarity": "positive",
            "score": 0.97,
            "source_provenance": {
                "project_ref": "project:fixture",
                "song_ref": "song:arcade",
                "version_ref": "version:main",
                "layer_ref": "layer:kick",
                "event_ref": "event:kick-01",
                "source_event_ref": "event:kick-source-01",
                "model_ref": "bundle:fixture-v1",
            },
        }
    ]
    path = tmp_path / "review_items.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

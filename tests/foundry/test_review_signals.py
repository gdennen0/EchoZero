"""Focused proof for the durable Foundry review-signal seam.
Exists because review sessions must stay queue state while explicit reviews persist canonically.
Connects review-session commits to the dedicated review-signal repository contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from echozero.foundry.domain import ReviewDecisionKind, ReviewOutcome
from echozero.foundry.persistence import (
    DatasetRepository,
    DatasetVersionRepository,
    ReviewSessionRepository,
    ReviewSignalRepository,
)
from echozero.foundry.services.review_session_service import ReviewSessionService
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
    service.set_item_outcome(session.id, session.items[0].item_id, ReviewOutcome.CORRECT)
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


def test_project_backed_review_relabel_writes_back_and_materializes_dataset_samples(tmp_path: Path):
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
    assert signal.source_provenance["dataset_materialization"]["status"] == "materialized"

    dataset = DatasetRepository(tmp_path).list()[0]
    version = DatasetVersionRepository(tmp_path).list_for_dataset(dataset.id)[-1]

    assert dataset.source_kind == "review_signal"
    assert version.stats["sample_count"] == 2
    assert sorted(sample.label for sample in version.samples) == ["kick", "tom"]
    assert {sample.source_provenance["review_polarity"] for sample in version.samples} == {
        "negative",
        "positive",
    }

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
    assert signal.source_provenance["dataset_materialization"]["status"] == "materialized"
    assert signal.source_provenance["dataset_materialization"]["sample_count"] == 1
    assert any(
        detail.get("reason") == "missing_materialized_correction"
        for detail in signal.source_provenance["dataset_materialization"]["details"]
    )
    dataset = DatasetRepository(tmp_path).list()[0]
    version = DatasetVersionRepository(tmp_path).list_for_dataset(dataset.id)[-1]
    assert [sample.source_provenance["review_polarity"] for sample in version.samples] == ["negative"]


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

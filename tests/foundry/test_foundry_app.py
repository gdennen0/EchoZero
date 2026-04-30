from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.services.review_session_service import ReviewSessionService
from echozero.foundry.services.review_signal_service import ReviewSignalService
from echozero.foundry.domain.review import ReviewOutcome
from tests.foundry.audio_fixtures import write_percussion_dataset
from tests.foundry.test_review_project_queue_builder import _build_project_review_fixture


def test_foundry_app_end_to_end_run_to_artifact(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.25, test_split=0.25, seed=9, balance_strategy="none")

    run_spec = {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version.id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 19},
    }

    run = app.create_run(version.id, run_spec)
    run = app.start_run(run.id)
    artifacts = app.artifacts._artifact_repo.list_for_run(run.id)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    report = app.validate_artifact(artifact.id)

    assert run.status.value == "completed"
    assert (run.exports_dir(tmp_path) / "model.pth").exists()
    assert app.eval._repo.list_for_run(run.id)
    assert report.ok is True
    assert any(item.kind == "run_created" for item in app.activity.items)
    assert any(item.kind == "artifact_validated" for item in app.activity.items)


def test_foundry_app_extract_review_signal_is_explicit(tmp_path: Path):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    reviews = ReviewSessionService(working_dir)
    session = reviews.create_project_session(
        working_dir,
        name="Kick Corrections",
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )
    reviewed = reviews.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="tom",
        review_note="explicit materialization check",
    )
    signal_id = ReviewSignalService.build_signal_id(reviewed.id, reviewed.items[0].item_id)

    app = FoundryApp(working_dir)
    result = app.extract_review_signal(session_id=reviewed.id, signal_id=signal_id)

    assert result["status"] == "materialized"
    assert str(result["dataset_id"]).startswith("ds_")
    assert str(result["version_id"]).startswith("dsv_")
    assert app.list_runs() == []


def test_foundry_app_create_run_rejects_mismatched_dataset_version_contract(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples, sample_count=1)
    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)

    run_spec = {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": "dsv_wrong_contract",
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 19},
    }

    try:
        app.create_run(version.id, run_spec)
    except ValueError as exc:
        assert "datasetVersionId" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for mismatched dataset version contract")

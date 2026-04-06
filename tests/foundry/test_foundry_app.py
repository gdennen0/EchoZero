from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
from tests.foundry.audio_fixtures import write_percussion_dataset


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

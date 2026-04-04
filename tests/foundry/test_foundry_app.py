from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp


def _write_samples(root: Path) -> None:
    (root / "kick").mkdir(parents=True, exist_ok=True)
    (root / "snare").mkdir(parents=True, exist_ok=True)
    (root / "kick" / "k1.wav").write_bytes(b"RIFF" + b"\x00" * 32)
    (root / "snare" / "s1.wav").write_bytes(b"RIFF" + b"\x00" * 32)


def test_foundry_app_end_to_end_run_to_artifact(tmp_path: Path):
    samples = tmp_path / "samples"
    _write_samples(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)

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
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.001},
    }

    run = app.create_run(version.id, run_spec)
    app.start_run(run.id)
    app.runs.complete_run(run.id, metrics={"f1": 0.88})

    artifact = app.finalize_artifact(
        run.id,
        {
            "weightsPath": "exports/model.pth",
            "classes": ["kick", "snare"],
            "classificationMode": "multiclass",
            "inferencePreprocessing": {
                "sampleRate": 22050,
                "maxLength": 22050,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
        },
    )
    report = app.validate_artifact(artifact.id)

    assert report.ok is True
    assert any(item.kind == "run_created" for item in app.activity.items)
    assert any(item.kind == "artifact_validated" for item in app.activity.items)

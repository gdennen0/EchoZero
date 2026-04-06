from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.persistence import ModelArtifactRepository
from echozero.foundry.services import ArtifactService, DatasetService, TrainRunService


def _write_samples(root: Path) -> None:
    (root / "kick").mkdir(parents=True, exist_ok=True)
    (root / "snare").mkdir(parents=True, exist_ok=True)
    (root / "kick" / "k1.wav").write_bytes(b"RIFF" + b"\x01" * 32)
    (root / "kick" / "k2.wav").write_bytes(b"RIFF" + b"\x02" * 32)
    (root / "kick" / "k3.wav").write_bytes(b"RIFF" + b"\x03" * 32)
    (root / "snare" / "s1.wav").write_bytes(b"RIFF" + b"\x10" * 32)
    (root / "snare" / "s2.wav").write_bytes(b"RIFF" + b"\x11" * 32)
    (root / "snare" / "s3.wav").write_bytes(b"RIFF" + b"\x12" * 32)


def test_foundry_smoke_tiny_dataset_run_to_artifact_compatibility(tmp_path: Path):
    samples = tmp_path / "samples"
    _write_samples(samples)

    app = FoundryApp(tmp_path)
    dataset_service = DatasetService(tmp_path)
    run_service = TrainRunService(tmp_path)
    artifact_service = ArtifactService(tmp_path)

    dataset = dataset_service.create_dataset("Tiny Drums", source_ref=str(samples))
    version = dataset_service.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=21, balance_strategy="none")
    version = dataset_service.get_version(version.id)
    assert version is not None

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
        "training": {
            "epochs": 1,
            "batchSize": 2,
            "learningRate": 0.001,
        },
    }

    run = run_service.create_run(dataset_version_id=version.id, run_spec=run_spec)
    run = run_service.start_run(run.id)
    run = run_service.complete_run(run.id, metrics={"accuracy": 1.0})

    artifact = artifact_service.finalize_artifact(
        run_id=run.id,
        manifest={
            "weightsPath": "exports/model.pth",
            "classes": version.class_map,
            "classificationMode": "multiclass",
            "inferencePreprocessing": {
                "sampleRate": 22050,
                "maxLength": 22050,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "thresholdPolicy": None,
            "evalSummary": {"accuracy": 1.0},
        },
    )

    persisted_artifact = ModelArtifactRepository(tmp_path).get(artifact.id)
    report = artifact_service.validate_compatibility(artifact.id, consumer="PyTorchAudioClassify")

    assert dataset.source_ref == str(samples)
    assert version.sample_count == 6
    assert run.status.value == "completed"
    assert artifact.path.exists()
    assert persisted_artifact is not None
    assert persisted_artifact.consumer_hints["consumer"] == "PyTorchAudioClassify"
    assert report.ok is True
    assert report.errors == []


def test_artifact_compatibility_rejects_runtime_mismatch(tmp_path: Path):
    samples = tmp_path / "samples"
    _write_samples(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Mismatch Drums", source_ref=str(samples))
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=33, balance_strategy="none")

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
    app.runs.complete_run(run.id, metrics={"macro_f1": 0.8})
    artifact = app.finalize_artifact(
        run.id,
        {
            "weightsPath": "exports/model.pth",
            "classes": version.class_map,
            "classificationMode": "multiclass",
            "runtime": {"consumer": "OtherRuntime", "backend": "pytorch", "device": "cpu"},
            "inferencePreprocessing": run_spec["data"],
        },
    )

    report = app.validate_artifact(artifact.id)

    assert report.ok is False
    assert "manifest.runtime.consumer must match the validated consumer" in report.errors

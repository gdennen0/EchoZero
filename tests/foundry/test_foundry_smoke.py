from __future__ import annotations

from pathlib import Path

from echozero.foundry.services import ArtifactService, TrainRunService


def test_foundry_smoke_run_to_artifact_compatibility(tmp_path: Path):
    run_service = TrainRunService(tmp_path)
    artifact_service = ArtifactService(tmp_path)

    run_spec = {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": "dsv_smoke",
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

    run = run_service.create_run(dataset_version_id="dsv_smoke", run_spec=run_spec)
    run = run_service.start_run(run.id)

    artifact = artifact_service.finalize_artifact(
        run_id=run.id,
        manifest={
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
            "thresholdPolicy": None,
            "evalSummary": {"accuracy": 0.9},
        },
    )

    report = artifact_service.validate_compatibility(artifact.id, consumer="PyTorchAudioClassify")

    assert run.status.value == "running"
    assert artifact.path.exists()
    assert report.ok is True
    assert report.errors == []

from __future__ import annotations

from pathlib import Path

from echozero.foundry.persistence import ModelArtifactRepository, TrainRunRepository
from echozero.foundry.services import ArtifactService, DatasetService, TrainRunService


def _write_samples(root: Path) -> None:
    (root / "kick").mkdir(parents=True, exist_ok=True)
    (root / "snare").mkdir(parents=True, exist_ok=True)
    (root / "kick" / "k1.wav").write_bytes(b"RIFF" + b"\x00" * 16)
    (root / "snare" / "s1.wav").write_bytes(b"RIFF" + b"\x00" * 16)


def test_train_run_and_model_artifact_persist_round_trip(tmp_path: Path):
    samples = tmp_path / "samples"
    _write_samples(samples)

    datasets = DatasetService(tmp_path)
    dataset = datasets.create_dataset("Persistence Drums")
    version = datasets.ingest_from_folder(dataset.id, samples)

    spec = {
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

    runs = TrainRunService(tmp_path)
    run = runs.create_run(version.id, spec, backend="pytorch", device="cpu")
    runs.start_run(run.id)

    artifact = ArtifactService(tmp_path).finalize_artifact(
        run.id,
        {
            "weightsPath": "exports/model.pth",
            "classes": version.class_map,
            "classificationMode": "multiclass",
            "inferencePreprocessing": spec["data"] | {},
        },
    )

    persisted_run = TrainRunRepository(tmp_path).get(run.id)
    persisted_artifact = ModelArtifactRepository(tmp_path).get(artifact.id)
    listed_artifacts = ModelArtifactRepository(tmp_path).list()

    assert persisted_run is not None
    assert persisted_run.dataset_version_id == version.id
    assert persisted_run.spec_hash == run.spec_hash
    assert persisted_artifact is not None
    assert persisted_artifact.run_id == run.id
    assert persisted_artifact.manifest["artifactId"] == artifact.id
    assert [item.id for item in listed_artifacts] == [artifact.id]

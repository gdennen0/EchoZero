from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.persistence import EvalReportRepository, ModelArtifactRepository, TrainRunRepository
from echozero.foundry.services import ArtifactService, EvalService, TrainRunService
from tests.foundry.audio_fixtures import write_percussion_dataset


def test_train_run_and_model_artifact_persist_round_trip(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Persistence Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.25, test_split=0.25, seed=17, balance_strategy="none")
    version = app.datasets.get_version(version.id)
    assert version is not None

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
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 23},
    }

    runs = app.runs
    run = runs.create_run(version.id, spec, backend="pytorch", device="cpu")
    run = runs.start_run(run.id)
    artifacts = ModelArtifactRepository(tmp_path).list_for_run(run.id)
    assert len(artifacts) == 1
    artifact = artifacts[0]

    persisted_run = TrainRunRepository(tmp_path).get(run.id)
    persisted_artifact = ModelArtifactRepository(tmp_path).get(artifact.id)
    listed_artifacts = ModelArtifactRepository(tmp_path).list()

    assert persisted_run is not None
    assert persisted_run.dataset_version_id == version.id
    assert persisted_run.spec_hash == run.spec_hash
    assert persisted_artifact is not None
    assert persisted_artifact.run_id == run.id
    assert persisted_artifact.manifest["artifactId"] == artifact.id
    assert persisted_artifact.manifest["datasetVersionId"] == version.id
    assert persisted_artifact.manifest["specHash"] == run.spec_hash
    assert persisted_artifact.manifest["taxonomy"] == version.taxonomy
    assert [item.id for item in listed_artifacts] == [artifact.id]
    assert EvalReportRepository(tmp_path).list_for_run(run.id)


def test_eval_report_persist_round_trip_with_baseline_metrics(tmp_path: Path):
    repo = EvalReportRepository(tmp_path)
    service = EvalService(repo)

    report = service.record_eval(
        "run_eval",
        classification_mode="multiclass",
        dataset_version_id="dsv_eval",
        split_name="test",
        metrics={"loss": 0.12, "macro_f1": 0.89},
        aggregate_metrics={"accuracy": 0.91, "macro_f1": 0.89},
        per_class_metrics={
            "kick": {"precision": 0.93, "recall": 0.88, "f1": 0.9, "support": 12},
            "snare": {"precision": 0.89, "recall": 0.91, "f1": 0.9, "support": 11},
        },
        baseline={"family": "cnn_melspec", "checkpoint_epoch": 4},
        threshold_policy={"metric": "f1", "threshold": 0.5, "source": "default"},
        confusion={"labels": ["kick", "snare"], "matrix": [[11, 1], [1, 10]]},
    )

    persisted = repo.get(report.id)

    assert persisted is not None
    assert persisted.dataset_version_id == "dsv_eval"
    assert persisted.split_name == "test"
    assert persisted.aggregate_metrics["macro_f1"] == 0.89
    assert persisted.per_class_metrics["kick"]["support"] == 12
    assert persisted.summary["primary_metric"] == "macro_f1"

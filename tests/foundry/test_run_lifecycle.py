from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import TrainRunStatus
from echozero.foundry.services import TrainRunService
from tests.foundry.audio_fixtures import write_percussion_dataset


def _prepared_version(root: Path):
    samples = root / "samples"
    write_percussion_dataset(samples)
    app = FoundryApp(root)
    dataset = app.datasets.create_dataset("Run Lifecycle Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=5, balance_strategy="none")
    return app.datasets.get_version(version.id)


def _run_spec(version_id: str) -> dict:
    return {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version_id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {"epochs": 2, "batchSize": 2, "learningRate": 0.01, "seed": 17},
    }


def test_run_lifecycle_executes_training_and_writes_artifacts(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    app = FoundryApp(tmp_path)
    svc = app.runs
    run = svc.create_run(version.id, _run_spec(version.id))
    assert run.status == TrainRunStatus.QUEUED
    assert run.checkpoints_dir(tmp_path).exists()
    assert run.exports_dir(tmp_path).exists()
    assert run.logs_dir(tmp_path).exists()

    run = svc.start_run(run.id)
    assert run.status == TrainRunStatus.COMPLETED

    reloaded = TrainRunService(tmp_path).get_run(run.id)
    assert reloaded is not None
    assert reloaded.status == TrainRunStatus.COMPLETED

    checkpoints = sorted(run.checkpoints_dir(tmp_path).glob("epoch_*.json"))
    assert len(checkpoints) == 2
    assert (run.exports_dir(tmp_path) / "model.pth").exists()
    assert (run.exports_dir(tmp_path) / "metrics.json").exists()
    assert (run.exports_dir(tmp_path) / "run_summary.json").exists()
    assert app.eval._repo.list_for_run(run.id)
    assert app.artifacts._artifact_repo.list_for_run(run.id)

    events_path = run.event_log_path(tmp_path)
    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    event_types = [line["type"] for line in lines]
    assert "RUN_CREATED" in event_types
    assert "RUN_PREPARING" in event_types
    assert "RUN_STARTED" in event_types
    assert "CHECKPOINT_SAVED" in event_types
    assert "RUN_EVALUATING" in event_types
    assert "RUN_EXPORTING" in event_types
    assert "RUN_COMPLETED" in event_types


def test_invalid_transition_raises(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    svc = TrainRunService(tmp_path)
    run = svc.create_run(version.id, _run_spec(version.id))

    with pytest.raises(ValueError):
        svc.complete_run(run.id)


def test_create_run_requires_dataset_planning_and_matching_spec(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)
    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Validation Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)

    svc = TrainRunService(tmp_path)
    with pytest.raises(ValueError, match="split assignments"):
        svc.create_run(version.id, _run_spec(version.id))

    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=3, balance_strategy="none")
    with pytest.raises(ValueError, match="datasetVersionId"):
        svc.create_run(version.id, _run_spec("dsv_other"))

from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import TrainRunStatus
from echozero.foundry.services import TrainRunService


def _write_samples(root: Path) -> None:
    (root / "kick").mkdir(parents=True, exist_ok=True)
    (root / "snare").mkdir(parents=True, exist_ok=True)
    (root / "kick" / "k1.wav").write_bytes(b"RIFF" + b"\x01" * 32)
    (root / "kick" / "k2.wav").write_bytes(b"RIFF" + b"\x02" * 32)
    (root / "kick" / "k3.wav").write_bytes(b"RIFF" + b"\x03" * 32)
    (root / "snare" / "s1.wav").write_bytes(b"RIFF" + b"\x11" * 32)
    (root / "snare" / "s2.wav").write_bytes(b"RIFF" + b"\x12" * 32)
    (root / "snare" / "s3.wav").write_bytes(b"RIFF" + b"\x13" * 32)


def _prepared_version(root: Path):
    samples = root / "samples"
    _write_samples(samples)
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
        "training": {"epochs": 2, "batchSize": 2, "learningRate": 0.001},
    }


def test_run_lifecycle_and_checkpoint(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    svc = TrainRunService(tmp_path)
    run = svc.create_run(version.id, _run_spec(version.id))
    assert run.status == TrainRunStatus.QUEUED
    assert run.checkpoints_dir(tmp_path).exists()
    assert run.exports_dir(tmp_path).exists()
    assert run.logs_dir(tmp_path).exists()

    run = svc.start_run(run.id)
    assert run.status == TrainRunStatus.RUNNING

    ckpt = svc.save_checkpoint(run.id, epoch=1, metric_snapshot={"loss": 0.1})
    assert ckpt.exists()

    run = svc.set_stage(run.id, TrainRunStatus.EVALUATING)
    assert run.status == TrainRunStatus.EVALUATING

    run = svc.complete_run(run.id, metrics={"f1": 0.91})
    assert run.status == TrainRunStatus.COMPLETED

    reloaded = TrainRunService(tmp_path).get_run(run.id)
    assert reloaded is not None
    assert reloaded.status == TrainRunStatus.COMPLETED

    events_path = run.event_log_path(tmp_path)
    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    event_types = [line["type"] for line in lines]
    assert "RUN_CREATED" in event_types
    assert "RUN_STARTED" in event_types
    assert "CHECKPOINT_SAVED" in event_types
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
    _write_samples(samples)
    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Validation Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)

    svc = TrainRunService(tmp_path)
    with pytest.raises(ValueError, match="split assignments"):
        svc.create_run(version.id, _run_spec(version.id))

    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=3, balance_strategy="none")
    with pytest.raises(ValueError, match="datasetVersionId"):
        svc.create_run(version.id, _run_spec("dsv_other"))

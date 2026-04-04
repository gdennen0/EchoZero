from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.foundry.domain import TrainRunStatus
from echozero.foundry.services import TrainRunService


def _run_spec() -> dict:
    return {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": "dsv_life",
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
    svc = TrainRunService(tmp_path)
    run = svc.create_run("dsv_life", _run_spec())
    assert run.status == TrainRunStatus.QUEUED

    run = svc.start_run(run.id)
    assert run.status == TrainRunStatus.RUNNING

    ckpt = svc.save_checkpoint(run.id, epoch=1, metric_snapshot={"loss": 0.1})
    assert ckpt.exists()

    run = svc.set_stage(run.id, TrainRunStatus.EVALUATING)
    assert run.status == TrainRunStatus.EVALUATING

    run = svc.complete_run(run.id, metrics={"f1": 0.91})
    assert run.status == TrainRunStatus.COMPLETED

    events_path = run.run_dir(tmp_path) / "events.jsonl"
    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    event_types = [line["type"] for line in lines]
    assert "RUN_CREATED" in event_types
    assert "RUN_STARTED" in event_types
    assert "CHECKPOINT_SAVED" in event_types
    assert "RUN_COMPLETED" in event_types


def test_invalid_transition_raises(tmp_path: Path):
    svc = TrainRunService(tmp_path)
    run = svc.create_run("dsv_life", _run_spec())

    with pytest.raises(ValueError):
        svc.complete_run(run.id)

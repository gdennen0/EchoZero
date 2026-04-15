from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import TrainRunStatus
from echozero.foundry.services import TrainRunService
from echozero.foundry.services import baseline_trainer as baseline_trainer_module
from echozero.foundry.services import cnn_trainer as cnn_trainer_module
from tests.foundry.audio_fixtures import write_percussion_dataset


def _prepared_version(root: Path):
    samples = root / "samples"
    write_percussion_dataset(samples)
    app = FoundryApp(root)
    dataset = app.datasets.create_dataset("Trainer Stability Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=19, balance_strategy="none")
    return app.datasets.get_version(version.id)


def _run_spec(version_id: str, *, model_type: str = "cnn") -> dict:
    payload = {
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
        "training": {
            "epochs": 1,
            "batchSize": 2,
            "learningRate": 0.001,
            "seed": 31,
            "deterministic": True,
        },
    }
    if model_type:
        payload["model"] = {"type": model_type}
    return payload


def test_create_run_requires_seed_for_determinism_controls(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None

    run_spec = _run_spec(version.id)
    del run_spec["training"]["seed"]

    with pytest.raises(ValueError, match="run_spec.training missing keys: seed"):
        TrainRunService(tmp_path).create_run(version.id, run_spec)


def test_cnn_run_persists_reproducibility_fingerprint(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None

    app = FoundryApp(tmp_path)
    run = app.runs.create_run(version.id, _run_spec(version.id))
    run = app.runs.start_run(run.id)

    assert run.status == TrainRunStatus.COMPLETED

    exports = run.exports_dir(tmp_path)
    metrics_payload = json.loads((exports / "metrics.json").read_text(encoding="utf-8"))
    run_summary = json.loads((exports / "run_summary.json").read_text(encoding="utf-8"))

    reproducibility = metrics_payload["reproducibility"]
    assert reproducibility["seed"] == 31
    assert reproducibility["deterministic"] is True
    assert isinstance(reproducibility["configFingerprint"], str)
    assert len(reproducibility["configFingerprint"]) == 64
    assert run_summary["reproducibility"]["configFingerprint"] == reproducibility["configFingerprint"]


def test_cnn_non_finite_watchdog_fails_run_with_explicit_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    version = _prepared_version(tmp_path)
    assert version is not None

    app = FoundryApp(tmp_path)
    run = app.runs.create_run(version.id, _run_spec(version.id))

    original_forward = cnn_trainer_module._SimpleCnn.forward

    def _non_finite_forward(self, x):
        logits = original_forward(self, x)
        return torch.full_like(logits, float("nan"))

    monkeypatch.setattr(cnn_trainer_module._SimpleCnn, "forward", _non_finite_forward)

    run = app.runs.start_run(run.id)
    assert run.status == TrainRunStatus.FAILED

    events = [
        json.loads(line)
        for line in run.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines()
    ]
    failed_events = [event for event in events if event["type"] == "RUN_FAILED"]
    assert failed_events
    assert "non-finite value detected in cnn/logits" in failed_events[-1]["payload"]["error"]


def test_baseline_non_finite_audio_watchdog_fails_run_with_explicit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    version = _prepared_version(tmp_path)
    assert version is not None

    app = FoundryApp(tmp_path)
    run = app.runs.create_run(version.id, _run_spec(version.id, model_type="baseline_sgd"))

    def _bad_audio(path: Path, *, sample_rate: int, max_length: int):
        del path, sample_rate
        return torch.full((max_length,), float("nan"), dtype=torch.float32).numpy()

    monkeypatch.setattr(baseline_trainer_module.BaselineTrainer, "_load_audio", staticmethod(_bad_audio))

    run = app.runs.start_run(run.id)
    assert run.status == TrainRunStatus.FAILED

    events = [
        json.loads(line)
        for line in run.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines()
    ]
    failed_events = [event for event in events if event["type"] == "RUN_FAILED"]
    assert failed_events
    assert "not finite" in failed_events[-1]["payload"]["error"].lower()

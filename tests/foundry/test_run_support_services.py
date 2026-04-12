from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import TrainRun, TrainRunStatus
from echozero.foundry.services import RunNotificationService, RunSpecValidator, RunTelemetryService
from tests.foundry.audio_fixtures import write_percussion_dataset


def _prepared_version(root: Path):
    samples = root / "samples"
    write_percussion_dataset(samples)
    app = FoundryApp(root)
    dataset = app.datasets.create_dataset("Support Services Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=13, balance_strategy="none")
    saved_version = app.datasets.get_version(version.id)
    assert saved_version is not None
    return saved_version


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
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 17},
    }


def test_run_spec_validator_validates_against_dataset_version(tmp_path: Path):
    version = _prepared_version(tmp_path)
    validator = RunSpecValidator(FoundryApp(tmp_path).runs._dataset_versions)

    validator.validate(version.id, _run_spec(version.id))

    bad_spec = _run_spec(version.id)
    bad_spec["data"]["sampleRate"] = 16000
    with pytest.raises(ValueError, match="sampleRate"):
        validator.validate(version.id, bad_spec)


def test_run_telemetry_service_writes_snapshots_and_latest_payload(tmp_path: Path):
    service = RunTelemetryService(tmp_path)
    run = TrainRun(
        id="run_fixture",
        dataset_version_id="dsv_fixture",
        status=TrainRunStatus.RUNNING,
        spec={},
        spec_hash="fixture",
    )
    run.run_dir(tmp_path).mkdir(parents=True, exist_ok=True)

    service.write_status_snapshot(run.id, status=run.status.value, event_type="RUN_STARTED")
    service.write_progress_snapshot(run.id, epoch=2, metric_snapshot={"train_loss": 0.42})
    service.append_run_telemetry(run, epoch=2, metric_snapshot={"train_loss": 0.42, "eta_seconds": 9})

    status_payload = json.loads(
        (tmp_path / "foundry" / "tracking" / "snapshots" / f"{run.id}_latest_status.json").read_text(
            encoding="utf-8"
        )
    )
    progress_payload = json.loads(
        (tmp_path / "foundry" / "tracking" / "snapshots" / f"{run.id}_latest_progress.json").read_text(
            encoding="utf-8"
        )
    )
    latest_payload = json.loads((run.run_dir(tmp_path) / "telemetry.latest.json").read_text(encoding="utf-8"))

    assert status_payload["event_type"] == "RUN_STARTED"
    assert progress_payload["epoch"] == 2
    assert latest_payload["epoch"] == 2
    assert latest_payload["loss"] == 0.42
    assert "cpu_percent" in latest_payload


def test_run_notification_service_dedupes_and_persists_state(tmp_path: Path):
    service = RunNotificationService(tmp_path)
    sent: list[str] = []
    run = TrainRun(
        id="run_fixture",
        dataset_version_id="dsv_fixture",
        status=TrainRunStatus.RUNNING,
        spec={"model": {"type": "cnn"}},
        spec_hash="fixture",
    )

    service.emit_notification_cadence(
        run,
        event_type="RUN_STARTED",
        list_runs=lambda: [run],
        notify=sent.append,
    )
    service.emit_notification_cadence(
        run,
        event_type="RUN_STARTED",
        list_runs=lambda: [run],
        notify=sent.append,
    )

    state = service.read_notification_state()
    assert len(sent) == 1
    assert "Foundry run started" in sent[0]
    assert f"start:{run.id}" in state["sent"]

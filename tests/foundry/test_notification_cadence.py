from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import TrainRunStatus
from tests.foundry.audio_fixtures import write_percussion_dataset


def _prepared_version(root: Path):
    samples = root / "samples"
    write_percussion_dataset(samples)
    app = FoundryApp(root)
    dataset = app.datasets.create_dataset("Notification Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=9, balance_strategy="none")
    return app, version


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


def test_notification_cadence_tracks_start_milestone_and_first_failure(tmp_path: Path):
    app, version = _prepared_version(tmp_path)
    sent: list[str] = []
    app.runs._notify_openclaw = lambda text: sent.append(text)  # type: ignore[attr-defined]

    run0 = app.runs.create_run(version.id, _run_spec(version.id))
    run0.status = TrainRunStatus.RUNNING
    app.runs._emit_notification_cadence(run0, event_type="RUN_STARTED")  # type: ignore[attr-defined]
    app.runs.cancel_run(run0.id, reason="fixture")

    run1 = app.runs.create_run(version.id, _run_spec(version.id))
    app.runs.set_stage(run1.id, TrainRunStatus.PREPARING)
    app.runs.fail_run(run1.id, "fixture")

    run2 = app.runs.create_run(version.id, _run_spec(version.id))
    app.runs.start_run(run2.id)

    run3 = app.runs.create_run(version.id, _run_spec(version.id))
    app.runs.cancel_run(run3.id, reason="fixture")

    assert any("Foundry run started" in message for message in sent)
    assert len([message for message in sent if "first failure" in message.lower()]) == 1
    assert any("milestone: 3 runs" in message.lower() for message in sent)
    assert any("final digest" in message.lower() for message in sent)

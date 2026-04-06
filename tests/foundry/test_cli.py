from __future__ import annotations

import json
from pathlib import Path

from echozero.foundry.cli import main
from echozero.foundry.persistence import TrainRunRepository
from tests.foundry.audio_fixtures import write_percussion_dataset


def test_cli_dataset_ingest_and_run(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    assert main(["--root", str(tmp_path), "create-dataset", "Drums"]) == 0
    out = capsys.readouterr().out
    dataset_id = json.loads(out)["id"]

    assert main(["--root", str(tmp_path), "ingest-folder", dataset_id, str(samples)]) == 0
    out = capsys.readouterr().out
    version_id = json.loads(out)["version_id"]

    assert main(["--root", str(tmp_path), "plan-version", version_id, "--val", "0.2", "--test", "0.2", "--seed", "13"]) == 0
    capsys.readouterr()

    run_spec = json.dumps(
        {
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
            "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 41},
        }
    )
    assert main(["--root", str(tmp_path), "create-run", version_id, run_spec]) == 0
    out = capsys.readouterr().out
    run_id = json.loads(out)["run_id"]

    assert main(["--root", str(tmp_path), "start-run", run_id]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "completed"
    assert out["eval_report_ids"]
    assert out["artifact_ids"]


def test_cli_train_folder_happy_path(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    assert main(
        [
            "--root",
            str(tmp_path),
            "train-folder",
            "Practical Drums",
            str(samples),
            "--val",
            "0.25",
            "--test",
            "0.25",
            "--epochs",
            "2",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["eval_report_ids"]
    assert payload["artifact_ids"]


def test_cli_train_folder_next_level_profile(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    assert main(
        [
            "--root",
            str(tmp_path),
            "train-folder",
            "Next Level Drums",
            str(samples),
            "--val",
            "0.25",
            "--test",
            "0.25",
            "--epochs",
            "2",
            "--next-level",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["eval_report_ids"]
    assert payload["artifact_ids"]


def test_cli_train_folder_stronger_profile_and_synthetic_mix(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    assert main(
        [
            "--root",
            str(tmp_path),
            "train-folder",
            "Stronger Drums",
            str(samples),
            "--val",
            "0.25",
            "--test",
            "0.25",
            "--epochs",
            "5",
            "--trainer-profile",
            "stronger_v1",
            "--early-stopping-patience",
            "2",
            "--min-epochs",
            "2",
            "--average-weights",
            "--synthetic-mix-enabled",
            "--synthetic-mix-ratio",
            "0.25",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    run = TrainRunRepository(tmp_path).get(payload["run_id"])
    assert run is not None
    assert run.spec["training"]["trainerProfile"] == "stronger_v1"
    assert run.spec["training"]["syntheticMix"]["enabled"] is True
    assert run.spec["training"]["syntheticMix"]["ratio"] == 0.25

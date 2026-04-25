from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_cli_train_folder_promotion_flags_persist_into_run_spec(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    assert main(
        [
            "--root",
            str(tmp_path),
            "train-folder",
            "Promotion Drums",
            str(samples),
            "--val",
            "0.25",
            "--test",
            "0.25",
            "--epochs",
            "2",
            "--gate-macro-f1-floor",
            "0.8",
            "--gate-max-regression-vs-reference",
            "0.05",
            "--gate-per-class-recall-floor",
            "kick=0.7",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    run = TrainRunRepository(tmp_path).get(payload["run_id"])
    assert run is not None
    assert run.spec["promotion"]["gate_policy"]["macro_f1_floor"] == 0.8
    assert run.spec["promotion"]["gate_policy"]["max_regression_vs_reference"] == 0.05
    assert run.spec["promotion"]["gate_policy"]["per_class_recall_floors"] == {"kick": 0.7}


def test_cli_ui_launches_foundry_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured_root: list[Path] = []

    def fake_run_foundry_ui(root: Path) -> int:
        captured_root.append(root)
        return 0

    monkeypatch.setattr("echozero.foundry.cli.run_foundry_ui", fake_run_foundry_ui)

    assert main(["--root", str(tmp_path), "ui"]) == 0
    assert captured_root == [tmp_path]


def test_cli_installs_runtime_bundle_from_artifact_id(tmp_path: Path, capsys) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    exports_dir = tmp_path / "workspace" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    weights_path = exports_dir / "model.pth"
    manifest_path = exports_dir / "art_live.manifest.json"
    models_dir = tmp_path / "models"

    torch.save(
        {
            "classes": ["snare", "other"],
            "classification_mode": "binary",
            "preprocessing": {
                "sampleRate": 22050,
                "maxLength": 22050,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "schema": "foundry.crnn_model.v1",
            "trainer": "crnn_melspec_v1",
            "model_state_dict": {},
        },
        weights_path,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "artifactId": "art_live",
                "runId": "run_live",
                "weightsPath": "model.pth",
                "classes": ["snare", "other"],
                "classificationMode": "binary",
                "sharedContractFingerprint": "bad-fingerprint",
                "inferencePreprocessing": {
                    "sampleRate": 22050,
                    "maxLength": 22050,
                    "nFft": 2048,
                    "hopLength": 512,
                    "nMels": 128,
                    "fmax": 8000,
                },
                "runtime": {"consumer": "PyTorchAudioClassify"},
            }
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--root",
            str(tmp_path),
            "install-runtime-bundle",
            "art_live",
            "--models-dir",
            str(models_dir),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"] == "snare"
    assert payload["bundle_name"] == "binary-drum-snare"
    assert (models_dir / "binary_drum_bundles.json").exists()

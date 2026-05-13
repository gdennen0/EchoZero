from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from echozero.foundry.cli import main
from echozero.foundry.domain import DatasetVersion, EvalReport, ModelArtifact, TrainRun, TrainRunStatus
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

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "plan-version",
                version_id,
                "--val",
                "0.2",
                "--test",
                "0.2",
                "--seed",
                "13",
            ]
        )
        == 0
    )
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

    assert (
        main(
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
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["eval_report_ids"]
    assert payload["artifact_ids"]


def test_cli_sample_library_record_and_train_flow(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    assert main(["--root", str(tmp_path), "create-dataset", "Library Drums"]) == 0
    dataset_id = json.loads(capsys.readouterr().out)["id"]

    assert main(["--root", str(tmp_path), "ingest-folder", dataset_id, str(samples)]) == 0
    version_id = json.loads(capsys.readouterr().out)["version_id"]

    assert main(["--root", str(tmp_path), "record-sample-library", version_id]) == 0
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["version_id"] == version_id
    assert recorded["recorded_count"] == 8
    assert len(recorded["library_sample_ids"]) == 8

    assert main(["--root", str(tmp_path), "sample-library-summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["sample_count"] == 8
    assert summary["approved_count"] == 8
    assert summary["approved_class_counts"] == {"kick": 4, "snare": 4}

    assert (
        main(["--root", str(tmp_path), "train-sample-library", "Library Drums", "--epochs", "1"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["dataset_version_id"].startswith("dsv_")
    assert payload["eval_report_ids"]
    assert payload["artifact_ids"]
    assert payload["refresh_version_id"] is None
    assert payload["refreshed_sample_count"] == 0


def test_cli_train_sample_library_can_refresh_from_dataset_version(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    assert main(["--root", str(tmp_path), "create-dataset", "Library Drums"]) == 0
    dataset_id = json.loads(capsys.readouterr().out)["id"]

    assert main(["--root", str(tmp_path), "ingest-folder", dataset_id, str(samples)]) == 0
    version_id = json.loads(capsys.readouterr().out)["version_id"]

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "train-sample-library",
                "Library Drums",
                "--epochs",
                "1",
                "--refresh-version-id",
                version_id,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["refresh_version_id"] == version_id
    assert payload["refresh_state"] == "approved"
    assert payload["refreshed_sample_count"] == 8
    assert payload["eval_report_ids"]
    assert payload["artifact_ids"]


def test_cli_train_folder_next_level_profile(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    assert (
        main(
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
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["eval_report_ids"]
    assert payload["artifact_ids"]


def test_cli_train_folder_stronger_profile_and_synthetic_mix(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    assert (
        main(
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
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    run = TrainRunRepository(tmp_path).get(payload["run_id"])
    assert run is not None
    assert run.spec["training"]["trainerProfile"] == "stronger_v1"
    assert run.spec["training"]["syntheticMix"]["enabled"] is True
    assert run.spec["training"]["syntheticMix"]["ratio"] == 0.25


def test_cli_train_folder_promotion_flags_persist_into_run_spec(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    assert (
        main(
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
        )
        == 0
    )
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

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "install-runtime-bundle",
                "art_live",
                "--models-dir",
                str(models_dir),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"] == "snare"
    assert payload["bundle_name"] == "binary-drum-snare"
    assert (models_dir / "binary_drum_bundles.json").exists()


def test_cli_train_grouped_binary_models_runs_multiple_grouped_specs(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version_by_id: dict[str, DatasetVersion] = {}
    captured_run_specs: dict[str, dict[str, object]] = {}

    @dataclass
    class _InstalledBundle:
        label: str
        bundle_name: str
        bundle_dir: Path
        manifest_path: Path
        weights_path: Path
        artifact_id: str
        run_id: str

    class _FakeDatasets:
        def derive_binary_dataset_version(
            self,
            source_version_id: str,
            *,
            positive_label: str,
            negative_label: str = "other",
            positive_aliases: tuple[str, ...] = (),
        ) -> DatasetVersion:
            assert source_version_id == "dsv_review"
            assert negative_label == "other"
            version = DatasetVersion(
                id=f"dsv_{positive_label}",
                dataset_id=f"ds_{positive_label}",
                version=1,
                manifest_hash=f"hash-{positive_label}",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=[positive_label, "other"],
                split_plan={},
                created_at=datetime.now(UTC),
            )
            version_by_id[version.id] = version
            return version

        def get_version(self, version_id: str) -> DatasetVersion | None:
            return version_by_id.get(version_id)

    class _FakeRuntimeBundles:
        def install_binary_drum_artifact(
            self,
            artifact_ref: str,
            *,
            bundle_label: str | None = None,
            bundle_name: str | None = None,
            models_dir: Path | None = None,
        ) -> _InstalledBundle:
            assert bundle_label is not None
            assert bundle_name is not None
            assert models_dir == tmp_path / "models"
            bundle_dir = models_dir / bundle_name
            return _InstalledBundle(
                label=bundle_label,
                bundle_name=bundle_name,
                bundle_dir=bundle_dir,
                manifest_path=bundle_dir / f"{bundle_label}.manifest.json",
                weights_path=bundle_dir / "model.pth",
                artifact_id=artifact_ref,
                run_id=f"run_{bundle_label}",
            )

    class _FakeApp:
        def __init__(self, root: Path) -> None:
            assert root == tmp_path
            self.datasets = _FakeDatasets()
            self.runtime_bundles = _FakeRuntimeBundles()

        def plan_version(
            self,
            version_id: str,
            *,
            validation_split: float,
            test_split: float,
            seed: int,
            balance_strategy: str,
        ) -> dict[str, object]:
            version = version_by_id[version_id]
            version.split_plan = {"assignments": {"sm1": "train"}}
            return {"version_id": version_id}

        def create_run(self, dataset_version_id: str, run_spec: dict[str, object]) -> TrainRun:
            captured_run_specs[dataset_version_id] = run_spec
            return TrainRun(
                id=f"run_{dataset_version_id}",
                dataset_version_id=dataset_version_id,
                status=TrainRunStatus.QUEUED,
                spec=run_spec,
                spec_hash=f"hash-{dataset_version_id}",
            )

        def start_run(self, run_id: str) -> TrainRun:
            return TrainRun(
                id=run_id,
                dataset_version_id=run_id.removeprefix("run_"),
                status=TrainRunStatus.COMPLETED,
                spec={},
                spec_hash=f"hash-{run_id}",
            )

    class _FakeEvalReportRepository:
        def __init__(self, root: Path) -> None:
            assert root == tmp_path

        def list_for_run(self, run_id: str) -> list[EvalReport]:
            return [
                EvalReport(
                    id=f"eval_{run_id}",
                    run_id=run_id,
                    classification_mode="binary",
                    metrics={"macro_f1": 0.9},
                )
            ]

    class _FakeArtifactRepository:
        def __init__(self, root: Path) -> None:
            assert root == tmp_path

        def list_for_run(self, run_id: str) -> list[ModelArtifact]:
            label = run_id.removeprefix("run_dsv_")
            return [
                ModelArtifact(
                    id=f"art_{label}",
                    run_id=run_id,
                    artifact_version="v1",
                    path=tmp_path / f"{label}.manifest.json",
                    sha256=f"sha-{label}",
                    manifest={},
                    created_at=datetime.now(UTC),
                )
            ]

    monkeypatch.setattr("echozero.foundry.cli.FoundryApp", _FakeApp)
    monkeypatch.setattr("echozero.foundry.cli.EvalReportRepository", _FakeEvalReportRepository)
    monkeypatch.setattr("echozero.foundry.cli.ModelArtifactRepository", _FakeArtifactRepository)

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "train-grouped-binary-models",
                "dsv_review",
                "--model",
                "clap",
                "--model",
                "cymbal=hi_hat,crash,ride",
                "--install-runtime",
                "--models-dir",
                str(tmp_path / "models"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_version_id"] == "dsv_review"
    assert [item["target_label"] for item in payload["models"]] == ["clap", "cymbal"]
    assert payload["models"][0]["source_labels"] == ["clap"]
    assert payload["models"][1]["source_labels"] == ["hi_hat", "crash", "ride"]
    assert payload["models"][0]["installed_bundle"]["label"] == "clap"
    assert payload["models"][1]["installed_bundle"]["label"] == "cymbal"
    clap_run_spec = captured_run_specs["dsv_clap"]
    assert clap_run_spec["classificationMode"] == "binary"
    assert clap_run_spec["model"] == {"type": "crnn"}
    assert clap_run_spec["training"]["trainerProfile"] == "stronger_v1"
    assert clap_run_spec["training"]["optimizer"] == "adamw"
    assert clap_run_spec["training"]["averageWeights"] is True

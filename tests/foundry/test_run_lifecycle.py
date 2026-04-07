from __future__ import annotations

import json
from threading import Event
from pathlib import Path

import pytest

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import DatasetSample, TrainRunStatus
from echozero.foundry.persistence import DatasetVersionRepository, ModelArtifactRepository, TrainRunRepository
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


def _run_spec(version_id: str, *, model_type: str | None = None) -> dict:
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
        "training": {"epochs": 2, "batchSize": 2, "learningRate": 0.01, "seed": 17},
    }
    if model_type is not None:
        payload["model"] = {"type": model_type}
    return payload


def _mark_train_samples_synthetic(root: Path, version_id: str) -> tuple[object, list[str]]:
    repo = DatasetVersionRepository(root)
    version = repo.get(version_id)
    assert version is not None

    train_ids = set(version.split_plan.get("train_ids", []))
    synthetic_ids: list[str] = []
    seen_labels: set[str] = set()
    for sample in version.samples:
        if sample.sample_id not in train_ids:
            continue
        if sample.label in seen_labels:
            synthetic_ids.append(sample.sample_id)
            continue
        seen_labels.add(sample.label)

    synthetic_id_set = set(synthetic_ids)
    version.samples = [
        DatasetSample(
            sample_id=sample.sample_id,
            audio_ref=sample.audio_ref,
            label=sample.label,
            duration_ms=sample.duration_ms,
            content_hash=sample.content_hash,
            source_provenance=sample.source_provenance,
            is_synthetic=sample.sample_id in synthetic_id_set,
            synthetic_provenance=(
                {
                    "generator": "fixture.synthetic",
                    "strategy": "test_marked_train_sample",
                    "source_sample_id": sample.sample_id,
                }
                if sample.sample_id in synthetic_id_set
                else {}
            ),
            quality_flags=sample.quality_flags,
            split_assignment=sample.split_assignment,
            curation_state=sample.curation_state,
        )
        for sample in version.samples
    ]
    version.stats = {
        **version.stats,
        "real_sample_count": sum(1 for sample in version.samples if not sample.is_synthetic),
        "synthetic_sample_count": sum(1 for sample in version.samples if sample.is_synthetic),
    }
    version.manifest = {
        **version.manifest,
        "synthetic_sample_ids": synthetic_ids,
        "real_sample_ids": [sample.sample_id for sample in version.samples if not sample.is_synthetic],
    }
    return repo.save(version), synthetic_ids


def _write_export_payloads(
    root: Path,
    run_id: str,
    *,
    macro_f1: float,
    accuracy: float,
    per_class_recall: dict[str, float],
    synthetic_mix_enabled: bool = False,
    synthetic_macro_f1: float | None = None,
) -> None:
    exports_dir = root / "foundry" / "runs" / run_id / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    (exports_dir / "model.pth").write_bytes(b"fixture-model")
    metrics_payload = {
        "schema": "foundry.training_metrics.v1",
        "runId": run_id,
        "finalEval": {
            "metrics": {
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "sample_count": 4,
            },
            "per_class_metrics": {
                label: {
                    "precision": recall,
                    "recall": recall,
                    "f1": recall,
                    "support": 2,
                }
                for label, recall in per_class_recall.items()
            },
        },
        "trainerOptions": {
            "syntheticMix": {
                "enabled": synthetic_mix_enabled,
                "ratio": 0.5 if synthetic_mix_enabled else 0.0,
                "cap": None,
                "availableSyntheticCount": 2 if synthetic_mix_enabled else 0,
                "actualSyntheticCount": 1 if synthetic_mix_enabled else 0,
                "realTrainCount": 2,
                "totalTrainCount": 3 if synthetic_mix_enabled else 2,
            }
        },
    }
    if synthetic_macro_f1 is not None:
        metrics_payload["syntheticEval"] = {
            "metrics": {
                "macro_f1": synthetic_macro_f1,
                "accuracy": synthetic_macro_f1,
                "sample_count": 2,
            }
        }
    (exports_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (exports_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "runId": run_id,
                "status": "completed",
                "modelPath": "model.pth",
                "metricsPath": "metrics.json",
                "primaryMetric": macro_f1,
                "accuracy": accuracy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _artifact_manifest_payload() -> dict:
    return {
        "weightsPath": "model.pth",
        "classes": ["kick", "snare"],
        "classificationMode": "multiclass",
        "inferencePreprocessing": {
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "evalSummary": {
            "splitName": "test",
            "accuracy": 0.0,
            "macroF1": 0.0,
        },
        "trainingSummary": {
            "trainer": "baseline_sgd_melspec_v1_5",
            "metricsPath": "metrics.json",
            "runSummaryPath": "run_summary.json",
        },
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


def test_crnn_run_lifecycle_executes_training_and_writes_artifacts(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    app = FoundryApp(tmp_path)
    run = app.runs.create_run(version.id, _run_spec(version.id, model_type="crnn"))

    run = app.runs.start_run(run.id)
    assert run.status == TrainRunStatus.COMPLETED

    exports = run.exports_dir(tmp_path)
    assert (exports / "model.pth").exists()
    assert (exports / "metrics.json").exists()
    metrics = json.loads((exports / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["trainerOptions"]["trainerProfile"] == "crnn_v1"


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


def test_next_level_training_options_persist_into_eval_baseline(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    app = FoundryApp(tmp_path)

    spec = _run_spec(version.id)
    spec["training"].update(
        {
            "classWeighting": "balanced",
            "rebalanceStrategy": "oversample",
            "augmentTrain": True,
            "augmentCopies": 2,
            "augmentNoiseStd": 0.03,
            "augmentGainJitter": 0.15,
        }
    )

    run = app.runs.create_run(version.id, spec)
    run = app.runs.start_run(run.id)
    assert run.status == TrainRunStatus.COMPLETED

    reports = app.eval._repo.list_for_run(run.id)
    assert reports
    baseline = reports[0].baseline
    assert baseline["family"] == "baseline_sgd_melspec_v1_5"
    assert baseline["class_weighting"] == "balanced"
    assert baseline["rebalance_strategy"] == "oversample"
    assert baseline["augment_train"] is True


def test_synthetic_disabled_excludes_synthetic_from_training_path(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    version, synthetic_ids = _mark_train_samples_synthetic(tmp_path, version.id)
    assert synthetic_ids

    app = FoundryApp(tmp_path)
    run = app.runs.create_run(version.id, _run_spec(version.id))
    run = app.runs.start_run(run.id)
    assert run.status == TrainRunStatus.COMPLETED

    run_summary = json.loads((run.exports_dir(tmp_path) / "run_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((run.exports_dir(tmp_path) / "metrics.json").read_text(encoding="utf-8"))
    artifact = ModelArtifactRepository(tmp_path).list_for_run(run.id)[0]

    assert run_summary["syntheticMix"]["enabled"] is False
    assert run_summary["syntheticMix"]["availableSyntheticCount"] == len(synthetic_ids)
    assert run_summary["syntheticMix"]["actualSyntheticCount"] == 0
    assert metrics["trainerOptions"]["syntheticMix"]["actualSyntheticCount"] == 0
    assert artifact.manifest["trainingSummary"]["syntheticMix"]["actualSyntheticCount"] == 0


def test_synthetic_enabled_with_ratio_bounds_inclusion(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    version, synthetic_ids = _mark_train_samples_synthetic(tmp_path, version.id)
    assert len(synthetic_ids) >= 1

    app = FoundryApp(tmp_path)
    spec = _run_spec(version.id)
    spec["training"]["syntheticMix"] = {"enabled": True, "ratio": 0.5}

    run = app.runs.create_run(version.id, spec)
    run = app.runs.start_run(run.id)
    assert run.status == TrainRunStatus.COMPLETED

    run_summary = json.loads((run.exports_dir(tmp_path) / "run_summary.json").read_text(encoding="utf-8"))
    synthetic_mix = run_summary["syntheticMix"]

    assert synthetic_mix["enabled"] is True
    assert synthetic_mix["availableSyntheticCount"] == len(synthetic_ids)
    assert synthetic_mix["actualSyntheticCount"] == 1
    assert synthetic_mix["actualSyntheticCount"] <= int(synthetic_mix["realTrainCount"] * synthetic_mix["ratio"])


def test_stronger_profile_persists_best_checkpoint_and_training_controls(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    app = FoundryApp(tmp_path)

    spec = _run_spec(version.id)
    spec["training"].update(
        {
            "epochs": 5,
            "trainerProfile": "stronger_v1",
            "optimizer": "sgd_optimal",
            "averageWeights": True,
            "regularizationAlpha": 0.00005,
            "earlyStoppingPatience": 2,
            "minEpochs": 2,
        }
    )

    run = app.runs.create_run(version.id, spec)
    run = app.runs.start_run(run.id)
    assert run.status == TrainRunStatus.COMPLETED

    run_summary = json.loads((run.exports_dir(tmp_path) / "run_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((run.exports_dir(tmp_path) / "metrics.json").read_text(encoding="utf-8"))
    reports = app.eval._repo.list_for_run(run.id)

    assert run_summary["trainerProfile"] == "stronger_v1"
    assert run_summary["completedEpochs"] <= 5
    assert 1 <= run_summary["bestCheckpointEpoch"] <= run_summary["completedEpochs"]
    assert metrics["trainerOptions"]["optimizer"] == "sgd_optimal"
    assert metrics["trainerOptions"]["averageWeights"] is True
    assert metrics["trainerOptions"]["earlyStoppingPatience"] == 2
    assert reports[0].baseline["profile"] == "stronger_v1"
    assert reports[0].baseline["checkpoint_epoch"] == run_summary["bestCheckpointEpoch"]


def test_create_run_rejects_invalid_stronger_profile_controls(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None

    svc = TrainRunService(tmp_path)
    bad_profile = _run_spec(version.id)
    bad_profile["training"]["trainerProfile"] = "not_real"
    with pytest.raises(ValueError, match="trainerProfile"):
        svc.create_run(version.id, bad_profile)

    bad_min_epochs = _run_spec(version.id)
    bad_min_epochs["training"]["minEpochs"] = 3
    with pytest.raises(ValueError, match="minEpochs"):
        svc.create_run(version.id, bad_min_epochs)


def test_promotion_gate_failure_persists_reasons_in_exports_and_manifest(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    app = FoundryApp(tmp_path)
    spec = _run_spec(version.id)
    spec["training"]["syntheticMix"] = {"enabled": True, "ratio": 0.5}
    spec["promotion"] = {
        "gate_policy": {
            "macro_f1_floor": 0.9,
            "max_real_vs_synth_gap": 0.05,
            "per_class_recall_floors": {"kick": 0.8},
        }
    }

    run = app.runs.create_run(version.id, spec)
    run.status = TrainRunStatus.EXPORTING
    TrainRunRepository(tmp_path).save(run)
    _write_export_payloads(
        tmp_path,
        run.id,
        macro_f1=0.7,
        accuracy=0.75,
        per_class_recall={"kick": 0.6, "snare": 0.8},
        synthetic_mix_enabled=True,
        synthetic_macro_f1=0.5,
    )

    artifact = app.artifacts.finalize_artifact(run.id, _artifact_manifest_payload())
    metrics_payload = json.loads((run.exports_dir(tmp_path) / "metrics.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run.exports_dir(tmp_path) / "run_summary.json").read_text(encoding="utf-8"))

    assert artifact.manifest["promotionGate"]["passed"] is False
    assert len(artifact.manifest["promotionGate"]["reasons"]) == 3
    assert metrics_payload["promotionGate"] == artifact.manifest["promotionGate"]
    assert run_summary["promotionGate"] == artifact.manifest["promotionGate"]


def test_reference_comparison_summary_persists_in_exports_and_manifest(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    app = FoundryApp(tmp_path)

    reference_run = app.runs.create_run(version.id, _run_spec(version.id))
    reference_run.status = TrainRunStatus.EXPORTING
    TrainRunRepository(tmp_path).save(reference_run)
    _write_export_payloads(
        tmp_path,
        reference_run.id,
        macro_f1=0.82,
        accuracy=0.81,
        per_class_recall={"kick": 0.8, "snare": 0.78},
    )
    reference_artifact = app.artifacts.finalize_artifact(reference_run.id, _artifact_manifest_payload())

    current_spec = _run_spec(version.id)
    current_spec["promotion"] = {
        "reference_artifact_id": reference_artifact.id,
        "gate_policy": {
            "macro_f1_floor": 0.8,
            "max_regression_vs_reference": 0.05,
        },
    }
    current_run = app.runs.create_run(version.id, current_spec)
    current_run.status = TrainRunStatus.EXPORTING
    TrainRunRepository(tmp_path).save(current_run)
    _write_export_payloads(
        tmp_path,
        current_run.id,
        macro_f1=0.8,
        accuracy=0.79,
        per_class_recall={"kick": 0.79, "snare": 0.77},
    )

    artifact = app.artifacts.finalize_artifact(current_run.id, _artifact_manifest_payload())
    metrics_payload = json.loads((current_run.exports_dir(tmp_path) / "metrics.json").read_text(encoding="utf-8"))
    run_summary = json.loads((current_run.exports_dir(tmp_path) / "run_summary.json").read_text(encoding="utf-8"))

    assert artifact.manifest["promotionGate"]["passed"] is True
    assert artifact.manifest["referenceComparison"]["referenceArtifactId"] == reference_artifact.id
    assert artifact.manifest["referenceComparison"]["delta"]["macroF1"] == pytest.approx(-0.02, abs=1e-6)
    assert metrics_payload["referenceComparison"] == artifact.manifest["referenceComparison"]
    assert run_summary["referenceComparison"] == artifact.manifest["referenceComparison"]



def test_start_run_cancels_when_cancel_event_is_set_before_execution(tmp_path: Path):
    version = _prepared_version(tmp_path)
    assert version is not None
    app = FoundryApp(tmp_path)
    run = app.runs.create_run(version.id, _run_spec(version.id))

    cancel_event = Event()
    cancel_event.set()
    run = app.runs.start_run(run.id, cancel_event=cancel_event)

    assert run.status == TrainRunStatus.CANCELED
    events = [
        json.loads(line)["type"]
        for line in run.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines()
    ]
    assert "RUN_PREPARING" in events
    assert "RUN_CANCELED" in events
    assert "RUN_COMPLETED" not in events


def test_start_run_transitions_to_failed_when_export_step_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    version = _prepared_version(tmp_path)
    assert version is not None
    app = FoundryApp(tmp_path)
    run = app.runs.create_run(version.id, _run_spec(version.id))

    def _boom(*args, **kwargs):
        raise RuntimeError("export explosion")

    monkeypatch.setattr(app.runs._artifacts, "finalize_artifact", _boom)

    run = app.runs.start_run(run.id)
    assert run.status == TrainRunStatus.FAILED

    events = [
        json.loads(line)["type"]
        for line in run.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines()
    ]
    assert "RUN_EXPORTING" in events
    assert "RUN_FAILED" in events
    assert "RUN_COMPLETED" not in events

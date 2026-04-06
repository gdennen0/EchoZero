from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import DatasetSample, TrainRunStatus
from echozero.foundry.persistence import DatasetVersionRepository, ModelArtifactRepository
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

from __future__ import annotations

import json
from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import DatasetSample
from echozero.foundry.persistence import (
    DatasetVersionRepository,
    EvalReportRepository,
    ModelArtifactRepository,
    TrainRunRepository,
)
from echozero.foundry.services import ArtifactService, EvalService, TrainRunService
from echozero.foundry.services.dataset_service import DatasetService
from echozero.foundry.services.split_balance_service import SplitBalanceService
from tests.foundry.audio_fixtures import write_percussion_dataset


def test_train_run_and_model_artifact_persist_round_trip(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Persistence Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.25, test_split=0.25, seed=17, balance_strategy="none")
    version = app.datasets.get_version(version.id)
    assert version is not None

    spec = {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version.id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 23},
    }

    runs = app.runs
    run = runs.create_run(version.id, spec, backend="pytorch", device="cpu")
    run = runs.start_run(run.id)
    artifacts = ModelArtifactRepository(tmp_path).list_for_run(run.id)
    assert len(artifacts) == 1
    artifact = artifacts[0]

    persisted_run = TrainRunRepository(tmp_path).get(run.id)
    persisted_artifact = ModelArtifactRepository(tmp_path).get(artifact.id)
    listed_artifacts = ModelArtifactRepository(tmp_path).list()

    assert persisted_run is not None
    assert persisted_run.dataset_version_id == version.id
    assert persisted_run.spec_hash == run.spec_hash
    assert persisted_artifact is not None
    assert persisted_artifact.run_id == run.id
    assert persisted_artifact.manifest["artifactId"] == artifact.id
    assert persisted_artifact.manifest["datasetVersionId"] == version.id
    assert persisted_artifact.manifest["specHash"] == run.spec_hash
    assert persisted_artifact.manifest["taxonomy"] == version.taxonomy
    assert [item.id for item in listed_artifacts] == [artifact.id]
    assert EvalReportRepository(tmp_path).list_for_run(run.id)


def test_eval_report_persist_round_trip_with_baseline_metrics(tmp_path: Path):
    repo = EvalReportRepository(tmp_path)
    service = EvalService(repo)

    report = service.record_eval(
        "run_eval",
        classification_mode="multiclass",
        dataset_version_id="dsv_eval",
        split_name="test",
        metrics={"loss": 0.12, "macro_f1": 0.89},
        aggregate_metrics={"accuracy": 0.91, "macro_f1": 0.89},
        per_class_metrics={
            "kick": {"precision": 0.93, "recall": 0.88, "f1": 0.9, "support": 12},
            "snare": {"precision": 0.89, "recall": 0.91, "f1": 0.9, "support": 11},
        },
        baseline={"family": "cnn_melspec", "checkpoint_epoch": 4},
        threshold_policy={"metric": "f1", "threshold": 0.5, "source": "default"},
        confusion={"labels": ["kick", "snare"], "matrix": [[11, 1], [1, 10]]},
    )

    persisted = repo.get(report.id)

    assert persisted is not None
    assert persisted.dataset_version_id == "dsv_eval"
    assert persisted.split_name == "test"
    assert persisted.aggregate_metrics["macro_f1"] == 0.89
    assert persisted.per_class_metrics["kick"]["support"] == 12
    assert persisted.summary["primary_metric"] == "macro_f1"


def test_synthetic_provenance_persisted_and_surfaced(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Synthetic Provenance Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.25, test_split=0.25, seed=17, balance_strategy="none")

    version_repo = DatasetVersionRepository(tmp_path)
    version = version_repo.get(version.id)
    assert version is not None

    synthetic_id = version.split_plan["train_ids"][0]
    version.samples = [
        DatasetSample(
            sample_id=sample.sample_id,
            audio_ref=sample.audio_ref,
            label=sample.label,
            duration_ms=sample.duration_ms,
            content_hash=sample.content_hash,
            source_provenance=sample.source_provenance,
            group_id=sample.group_id,
            is_synthetic=sample.sample_id == synthetic_id,
            synthetic_provenance=(
                {
                    "generator": "fixture.synthetic",
                    "recipe": "noise_plus_envelope",
                    "source_sample_id": synthetic_id,
                }
                if sample.sample_id == synthetic_id
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
    version.manifest_hash = DatasetService.compute_manifest_hash(version.samples)
    version.manifest = {
        **version.manifest,
        "synthetic_sample_ids": [synthetic_id],
        "real_sample_ids": [sample.sample_id for sample in version.samples if not sample.is_synthetic],
    }
    version.split_plan = SplitBalanceService().plan_splits(
        version,
        validation_split=float(version.split_plan.get("validation_split", 0.25)),
        test_split=float(version.split_plan.get("test_split", 0.25)),
        seed=int(version.split_plan.get("seed", 42)),
    )
    version_repo.save(version)

    persisted_version = version_repo.get(version.id)
    assert persisted_version is not None
    persisted_sample = next(sample for sample in persisted_version.samples if sample.sample_id == synthetic_id)
    assert persisted_sample.is_synthetic is True
    assert persisted_sample.synthetic_provenance["generator"] == "fixture.synthetic"
    assert persisted_version.manifest["synthetic_sample_ids"] == [synthetic_id]
    assert persisted_version.stats["synthetic_sample_count"] == 1

    spec = {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version.id,
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
            "learningRate": 0.01,
            "seed": 23,
            "syntheticMix": {"enabled": True, "ratio": 1.0, "cap": 1},
        },
    }

    run = app.runs.create_run(version.id, spec, backend="pytorch", device="cpu")
    run = app.runs.start_run(run.id)
    artifact = ModelArtifactRepository(tmp_path).list_for_run(run.id)[0]

    assert artifact.manifest["syntheticProvenance"]["syntheticSampleIds"] == [synthetic_id]
    assert artifact.manifest["syntheticProvenance"]["syntheticSampleCount"] == 1
    assert artifact.manifest["trainingSummary"]["syntheticMix"]["actualSyntheticCount"] == 1


def test_run_list_includes_untracked_run_directories(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)
    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Directory Drift Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.15, test_split=0.10, seed=42, balance_strategy="none")
    run = app.runs.create_run(
        version.id,
        {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "multiclass",
            "data": {
                "datasetVersionId": version.id,
                "sampleRate": 22050,
                "maxLength": 22050,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 23},
        },
    )

    runs_root = tmp_path / "foundry" / "runs"
    orphan_with_spec = runs_root / "run_orphan_with_spec"
    orphan_with_spec.mkdir(parents=True, exist_ok=True)
    (orphan_with_spec / "spec.json").write_text(
        json.dumps(
            {
                "schema": "foundry.train_run_spec.v1",
                "classificationMode": "multiclass",
                "data": {
                    "datasetVersionId": version.id,
                },
                "training": {
                    "epochs": 3,
                    "batchSize": 4,
                    "learningRate": 0.01,
                    "seed": 77,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (orphan_with_spec / "events.jsonl").write_text(
        json.dumps({"type": "RUN_COMPLETED", "payload": {"status": "completed"}}) + "\n",
        encoding="utf-8",
    )
    orphan_without_spec = runs_root / "run_orphan_without_spec"
    orphan_without_spec.mkdir(parents=True, exist_ok=True)

    run_ids = {item.id for item in app.runs.list_runs()}
    assert run.id in run_ids
    assert "run_orphan_with_spec" in run_ids
    assert "run_orphan_without_spec" in run_ids

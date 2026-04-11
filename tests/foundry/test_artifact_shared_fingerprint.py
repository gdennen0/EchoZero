from __future__ import annotations

import json
from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.persistence import ModelArtifactRepository
from echozero.inference_eval import create_foundry_adapter
from tests.foundry.audio_fixtures import write_percussion_dataset


def _prepare_run(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Fingerprint Drums", source_ref=str(samples))
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=43, balance_strategy="none")

    run_spec = {
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
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 47},
    }

    run = app.create_run(version.id, run_spec)
    run = app.start_run(run.id)
    return app, version, run


def _artifact_manifest_payload(version: object) -> dict:
    return {
        "weightsPath": "model.pth",
        "classes": list(version.class_map),
        "classificationMode": "multiclass",
        "runtime": {"consumer": "PyTorchAudioClassify", "backend": "pytorch", "device": "cpu"},
        "inferencePreprocessing": {
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
    }


def test_finalize_artifact_persists_shared_contract_fingerprint(tmp_path: Path) -> None:
    app, version, run = _prepare_run(tmp_path)

    artifact = app.finalize_artifact(run.id, _artifact_manifest_payload(version))
    expected = create_foundry_adapter().contract_fingerprint_from_run_spec(run.spec, class_map=version.class_map)

    assert artifact.manifest["sharedContractFingerprint"] == expected

    persisted = ModelArtifactRepository(tmp_path).get(artifact.id)
    assert persisted is not None
    assert persisted.manifest["sharedContractFingerprint"] == expected


def test_validate_compatibility_rejects_shared_contract_fingerprint_mismatch(tmp_path: Path) -> None:
    app, version, run = _prepare_run(tmp_path)
    artifact = app.finalize_artifact(run.id, _artifact_manifest_payload(version))

    manifest = dict(artifact.manifest)
    manifest["sharedContractFingerprint"] = "bad-fingerprint"
    artifact.path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    updated_artifact = type(artifact)(
        id=artifact.id,
        run_id=artifact.run_id,
        artifact_version=artifact.artifact_version,
        path=artifact.path,
        sha256=artifact.sha256,
        manifest=manifest,
        consumer_hints=artifact.consumer_hints,
        created_at=artifact.created_at,
    )
    ModelArtifactRepository(tmp_path).save(updated_artifact)

    report = app.validate_artifact(artifact.id)

    assert report.ok is False
    assert (
        "manifest.sharedContractFingerprint must match the originating shared contract fingerprint"
        in report.errors
    )
    assert {
        "code": "shared_contract_fingerprint_mismatch",
        "path": "manifest.sharedContractFingerprint",
        "message": "manifest.sharedContractFingerprint must match the originating shared contract fingerprint",
        "severity": "error",
    } in report.error_details


def test_validate_compatibility_allows_legacy_manifest_without_shared_contract_fingerprint(tmp_path: Path) -> None:
    app, version, run = _prepare_run(tmp_path)
    artifact = app.finalize_artifact(run.id, _artifact_manifest_payload(version))

    manifest = dict(artifact.manifest)
    manifest.pop("sharedContractFingerprint", None)
    artifact.path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    updated_artifact = type(artifact)(
        id=artifact.id,
        run_id=artifact.run_id,
        artifact_version=artifact.artifact_version,
        path=artifact.path,
        sha256=artifact.sha256,
        manifest=manifest,
        consumer_hints=artifact.consumer_hints,
        created_at=artifact.created_at,
    )
    ModelArtifactRepository(tmp_path).save(updated_artifact)

    report = app.validate_artifact(artifact.id)

    assert report.ok is True
    assert report.errors == []
    assert report.error_details == []


def test_validate_compatibility_includes_structured_details_for_manifest_and_runtime_issues(tmp_path: Path) -> None:
    app, version, run = _prepare_run(tmp_path)
    artifact = app.finalize_artifact(run.id, _artifact_manifest_payload(version))

    manifest = dict(artifact.manifest)
    manifest["runtime"] = {"consumer": "OtherProcessor", "backend": "pytorch", "device": "cpu"}
    manifest["inferencePreprocessing"] = {
        "sampleRate": 22050,
        "maxLength": 22050,
        "nFft": 2048,
        "nMels": 128,
        "fmax": 8000,
    }
    artifact.path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    updated_artifact = type(artifact)(
        id=artifact.id,
        run_id=artifact.run_id,
        artifact_version=artifact.artifact_version,
        path=artifact.path,
        sha256=artifact.sha256,
        manifest=manifest,
        consumer_hints=artifact.consumer_hints,
        created_at=artifact.created_at,
    )
    ModelArtifactRepository(tmp_path).save(updated_artifact)

    report = app.validate_artifact(artifact.id)

    assert "manifest.inferencePreprocessing missing keys: hopLength" in report.errors
    assert "manifest.runtime.consumer must match the validated consumer" in report.errors
    assert {
        "code": "missing_preprocessing_keys",
        "path": "manifest.inferencePreprocessing",
        "message": "manifest.inferencePreprocessing missing keys: hopLength",
        "severity": "error",
    } in report.error_details
    assert {
        "code": "runtime_consumer_mismatch",
        "path": "manifest.runtime.consumer",
        "message": "manifest.runtime.consumer must match the validated consumer",
        "severity": "error",
    } in report.error_details

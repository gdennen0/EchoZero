from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.errors import ValidationError
from echozero.foundry.app import FoundryApp
from echozero.foundry.persistence import ModelArtifactRepository
from echozero.inference_eval.runtime_preflight import checkpoint_contract_fingerprint, run_runtime_preflight
from tests.foundry.audio_fixtures import write_percussion_dataset


def _prepare_run(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Parity Drums", source_ref=str(samples))
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=43, balance_strategy="none")

    run_spec = {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "model": {"type": "cnn"},
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


def _checkpoint_from_run(run: object, version: object) -> dict[str, object]:
    run_data = run.spec.get("data", {})
    return {
        "model_state_dict": {},
        "classes": list(version.class_map),
        "classification_mode": run.spec.get("classificationMode", "multiclass"),
        "preprocessing": dict(run_data),
        "model_type": (run.spec.get("model") or {}).get("type", "baseline_sgd"),
    }


def test_train_artifact_runtime_contract_parity_guard(tmp_path: Path) -> None:
    app, version, run = _prepare_run(tmp_path)
    artifacts = ModelArtifactRepository(tmp_path).list_for_run(run.id)
    artifacts.sort(key=lambda item: item.created_at)
    artifact = artifacts[-1]

    model_path = artifact.path.parent / "model.pth"
    model_path.write_bytes(b"weights")

    checkpoint = _checkpoint_from_run(run, version)
    assert artifact.manifest["sharedContractFingerprint"] == checkpoint_contract_fingerprint(checkpoint)

    run_runtime_preflight(model_path, checkpoint)

    mutated_manifest = dict(artifact.manifest)
    mutated_manifest["sharedContractFingerprint"] = "bad-fingerprint"
    artifact.path.write_text(json.dumps(mutated_manifest, indent=2), encoding="utf-8")

    with pytest.raises(
        ValidationError,
        match=(
            "Runtime bundle preflight failed for model\\.pth: "
            "manifest\\.sharedContractFingerprint must match the checkpoint-derived shared contract fingerprint"
        ),
    ):
        run_runtime_preflight(model_path, checkpoint)


def test_train_artifact_runtime_compatibility_issue_code_parity(tmp_path: Path) -> None:
    app, version, run = _prepare_run(tmp_path)
    artifacts = ModelArtifactRepository(tmp_path).list_for_run(run.id)
    artifacts.sort(key=lambda item: item.created_at)
    artifact = artifacts[-1]

    model_path = artifact.path.parent / "model.pth"
    model_path.write_bytes(b"weights")
    checkpoint = _checkpoint_from_run(run, version)

    mutated_manifest = dict(artifact.manifest)
    mutated_manifest["runtime"] = {
        "consumer": "OtherProcessor",
        "backend": "pytorch",
        "device": "cpu",
    }
    mutated_preprocessing = dict(mutated_manifest["inferencePreprocessing"])
    mutated_preprocessing.pop("hopLength", None)
    mutated_manifest["inferencePreprocessing"] = mutated_preprocessing
    mutated_manifest.pop("sharedContractFingerprint", None)
    artifact.path.write_text(json.dumps(mutated_manifest, indent=2), encoding="utf-8")
    updated_artifact = type(artifact)(
        id=artifact.id,
        run_id=artifact.run_id,
        artifact_version=artifact.artifact_version,
        path=artifact.path,
        sha256=artifact.sha256,
        manifest=mutated_manifest,
        consumer_hints=artifact.consumer_hints,
        created_at=artifact.created_at,
    )
    ModelArtifactRepository(tmp_path).save(updated_artifact)

    report = app.validate_artifact(artifact.id)
    foundry_codes = {detail["code"] for detail in report.error_details}

    with pytest.raises(ValidationError) as exc_info:
        run_runtime_preflight(model_path, checkpoint)

    runtime_diagnostics = getattr(exc_info.value, "validation_diagnostics", None)
    assert runtime_diagnostics is not None
    runtime_codes = {detail["code"] for detail in runtime_diagnostics["errors"]}

    expected_codes = {
        "missing_preprocessing_keys",
        "runtime_consumer_mismatch",
        "missing_shared_contract_fingerprint",
    }
    assert expected_codes <= foundry_codes
    assert expected_codes <= runtime_codes

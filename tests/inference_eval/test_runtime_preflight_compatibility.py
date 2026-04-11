from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.errors import ValidationError
from echozero.inference_eval.runtime_preflight import (
    checkpoint_contract_fingerprint,
    run_runtime_preflight,
)


def _checkpoint() -> dict[str, object]:
    return {
        "model_state_dict": {},
        "classes": ["kick", "snare"],
        "classification_mode": "multiclass",
        "preprocessing": {
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "trainer": "cnn_melspec_v1",
    }


def _manifest(model_name: str, *, fingerprint: str | None = None) -> dict[str, object]:
    if fingerprint is None:
        fingerprint = checkpoint_contract_fingerprint(_checkpoint())
    return {
        "schema": "foundry.artifact_manifest.v1",
        "weightsPath": model_name,
        "sharedContractFingerprint": fingerprint,
        "classes": ["kick", "snare"],
        "classificationMode": "multiclass",
        "runtime": {"consumer": "PyTorchAudioClassify"},
        "inferencePreprocessing": {
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
    }


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_runtime_preflight_keeps_human_readable_summary_and_attaches_structured_diagnostics(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")

    manifest = _manifest(model_path.name)
    manifest["runtime"] = {"consumer": "OtherProcessor"}
    manifest["inferencePreprocessing"].pop("hopLength")
    _write_manifest(tmp_path / "art_test.manifest.json", manifest)

    with pytest.raises(ValidationError) as exc_info:
        run_runtime_preflight(model_path, _checkpoint())

    assert str(exc_info.value) == (
        "Runtime bundle preflight failed for model.pth: "
        "manifest.inferencePreprocessing missing keys: hopLength; "
        "manifest.runtime.consumer must match the validated consumer"
    )

    diagnostics = getattr(exc_info.value, "validation_diagnostics", None)
    assert diagnostics is not None
    assert diagnostics["errors"][0]["path"] == "manifest.inferencePreprocessing"
    assert diagnostics["errors"][0]["code"] == "missing_preprocessing_keys"
    assert diagnostics["errors"][1]["path"] == "manifest.runtime.consumer"
    assert diagnostics["errors"][1]["code"] == "runtime_consumer_mismatch"


def test_runtime_preflight_rejects_missing_manifest_for_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")

    with pytest.raises(ValidationError) as exc_info:
        run_runtime_preflight(model_path, _checkpoint())

    message = str(exc_info.value)
    assert (
        "artifact manifest is required for runtime preflight; no *.manifest.json files were found"
        in message
    )
    assert f"requested model path: {model_path.resolve()}" in message

    diagnostics = getattr(exc_info.value, "validation_diagnostics", None)
    assert diagnostics is not None
    assert diagnostics["errors"][0]["code"] == "missing_model_manifest"
    assert diagnostics["errors"][0]["path"] == "manifest"


def test_runtime_preflight_rejects_directory_manifests_that_do_not_resolve_to_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")

    wrong_manifest = _manifest("other_model.pth")
    _write_manifest(tmp_path / "other.manifest.json", wrong_manifest)

    with pytest.raises(ValidationError) as exc_info:
        run_runtime_preflight(model_path, _checkpoint())

    message = str(exc_info.value)
    assert (
        "no manifest in model directory resolves manifest.weightsPath to the requested model path"
        in message
    )
    assert f"requested model path: {model_path.resolve()}" in message
    assert f"other.manifest.json -> {(tmp_path / 'other_model.pth').resolve()}" in message


def test_runtime_preflight_rejects_ambiguous_manifest_resolution(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")

    payload = _manifest(model_path.name)
    _write_manifest(tmp_path / "a.manifest.json", payload)
    _write_manifest(tmp_path / "b.manifest.json", payload)

    with pytest.raises(ValidationError) as exc_info:
        run_runtime_preflight(model_path, _checkpoint())

    message = str(exc_info.value)
    assert "multiple manifests resolve to the requested model path" in message
    assert "keep exactly one matching artifact manifest" in message
    assert f"requested model path: {model_path.resolve()}" in message
    assert f"a.manifest.json -> {(tmp_path / model_path.name).resolve()}" in message
    assert f"b.manifest.json -> {(tmp_path / model_path.name).resolve()}" in message


def test_runtime_preflight_rejects_manifest_validation_when_checkpoint_metadata_is_missing(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    legacy_checkpoint = {
        "model_state_dict": {},
        "preprocessing": {
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
        },
        "trainer": "cnn_melspec_v1",
    }
    _write_manifest(
        tmp_path / "art_test.manifest.json",
        _manifest(model_path.name, fingerprint=checkpoint_contract_fingerprint(legacy_checkpoint)),
    )

    with pytest.raises(ValidationError) as exc_info:
        run_runtime_preflight(model_path, legacy_checkpoint)

    message = str(exc_info.value)
    assert "checkpoint.classes must be present when validating against an artifact manifest" in message
    assert "checkpoint preprocessing missing keys required for manifest verification: fmax" in message
    assert "checkpoint classification mode must be present when validating against an artifact manifest" in message

    diagnostics = getattr(exc_info.value, "validation_diagnostics", None)
    assert diagnostics is not None
    error_codes = [entry["code"] for entry in diagnostics["errors"]]
    assert "missing_checkpoint_classes" in error_codes
    assert "missing_checkpoint_preprocessing_keys" in error_codes
    assert "missing_checkpoint_classification_mode" in error_codes


def test_runtime_preflight_rejects_missing_shared_contract_fingerprint(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")
    manifest_payload = _manifest(model_path.name)
    manifest_payload.pop("sharedContractFingerprint", None)
    _write_manifest(tmp_path / "art_test.manifest.json", manifest_payload)

    with pytest.raises(ValidationError) as exc_info:
        run_runtime_preflight(model_path, _checkpoint())

    diagnostics = getattr(exc_info.value, "validation_diagnostics", None)
    assert diagnostics is not None
    assert {
        "code": "missing_shared_contract_fingerprint",
        "path": "manifest.sharedContractFingerprint",
        "message": "manifest.sharedContractFingerprint is required",
        "severity": "error",
    } in diagnostics["errors"]

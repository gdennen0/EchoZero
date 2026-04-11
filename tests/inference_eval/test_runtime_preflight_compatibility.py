from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.errors import ValidationError
from echozero.inference_eval.runtime_preflight import run_runtime_preflight


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


def _manifest(model_name: str) -> dict[str, object]:
    return {
        "schema": "foundry.artifact_manifest.v1",
        "weightsPath": model_name,
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


def test_runtime_preflight_rejects_directory_manifests_that_do_not_resolve_to_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")

    wrong_manifest = _manifest("other_model.pth")
    _write_manifest(tmp_path / "other.manifest.json", wrong_manifest)

    with pytest.raises(
        ValidationError,
        match=(
            "Runtime bundle preflight failed for model\\.pth: "
            "no manifest in model directory resolves manifest\\.weightsPath to the requested model path"
        ),
    ):
        run_runtime_preflight(model_path, _checkpoint())


def test_runtime_preflight_rejects_ambiguous_manifest_resolution(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"weights")

    payload = _manifest(model_path.name)
    _write_manifest(tmp_path / "a.manifest.json", payload)
    _write_manifest(tmp_path / "b.manifest.json", payload)

    with pytest.raises(
        ValidationError,
        match=(
            "Runtime bundle preflight failed for model\\.pth: "
            "multiple manifests resolve to the requested model path; "
            "keep exactly one matching artifact manifest"
        ),
    ):
        run_runtime_preflight(model_path, _checkpoint())

from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.foundry.services.runtime_bundle_install_service import RuntimeBundleInstallService
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles


def _write_binary_artifact(tmp_path: Path) -> Path:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    exports_dir = tmp_path / "workspace" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    weights_path = exports_dir / "model.pth"
    manifest_path = exports_dir / "art_live.manifest.json"

    torch.save(
        {
            "classes": ["snare", "other"],
            "classification_mode": "binary",
            "preprocessing": {
                "datasetVersionId": "dsv_live",
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
    return manifest_path


def test_install_binary_drum_artifact_creates_stable_runtime_bundle(tmp_path: Path) -> None:
    manifest_path = _write_binary_artifact(tmp_path)

    models_dir = tmp_path / "models"
    installed = RuntimeBundleInstallService(tmp_path).install_binary_drum_artifact(
        str(manifest_path),
        models_dir=models_dir,
    )

    assert installed.label == "snare"
    assert installed.bundle_name == "binary-drum-snare"
    assert installed.manifest_path.exists()
    assert installed.weights_path.exists()
    payload = json.loads(installed.manifest_path.read_text(encoding="utf-8"))
    assert payload["sharedContractFingerprint"] != "bad-fingerprint"
    assert (models_dir / "binary_drum_bundles.json").exists()

    resolved = resolve_installed_binary_drum_bundles(labels=("snare",), models_dir=models_dir)
    assert resolved["snare"].manifest_path == installed.manifest_path.resolve()


def test_install_binary_drum_artifact_resolves_workspace_artifact_id(tmp_path: Path) -> None:
    _write_binary_artifact(tmp_path)

    models_dir = tmp_path / "models"
    installed = RuntimeBundleInstallService(tmp_path).install_binary_drum_artifact(
        "art_live",
        models_dir=models_dir,
    )

    assert installed.artifact_id == "art_live"
    assert installed.manifest_path.exists()

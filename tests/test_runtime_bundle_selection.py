from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles
from echozero.runtime_models.bundle_compat import backfill_manifest_fingerprint


def _write_bundle(root: Path, name: str, classes: list[str]) -> Path:
    bundle_dir = root / name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.pth").write_bytes(b"weights")
    manifest_path = bundle_dir / f"{name}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "weightsPath": "model.pth",
                "classes": classes,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_resolve_installed_binary_drum_bundles_returns_kick_and_snare(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "kick_bundle", ["kick", "other"])
    _write_bundle(tmp_path, "snare_bundle", ["snare", "other"])

    bundles = resolve_installed_binary_drum_bundles(models_dir=tmp_path)

    assert set(bundles.keys()) == {"kick", "snare"}
    assert bundles["kick"].manifest_path.name == "kick_bundle.manifest.json"
    assert bundles["snare"].manifest_path.name == "snare_bundle.manifest.json"


def test_resolve_installed_binary_drum_bundles_rejects_ambiguous_label(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "kick_bundle_a", ["kick", "other"])
    _write_bundle(tmp_path, "kick_bundle_b", ["kick", "other"])
    _write_bundle(tmp_path, "snare_bundle", ["snare", "other"])

    with pytest.raises(Exception, match="Multiple installed runtime bundles matched 'kick'"):
        resolve_installed_binary_drum_bundles(models_dir=tmp_path)


def test_bundle_resolution_does_not_mutate_manifest(tmp_path: Path) -> None:
    manifest_path = _write_bundle(tmp_path, "kick_bundle", ["kick", "other"])
    _write_bundle(tmp_path, "snare_bundle", ["snare", "other"])
    before = manifest_path.read_text(encoding="utf-8")

    resolve_installed_binary_drum_bundles(models_dir=tmp_path)

    assert manifest_path.read_text(encoding="utf-8") == before


def test_backfill_manifest_fingerprint_is_explicit_upgrade_step(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "kick_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    weights_path = bundle_dir / "model.pth"
    manifest_path = bundle_dir / "kick_bundle.manifest.json"

    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    torch.save(
        {
            "classes": ["kick", "other"],
            "classification_mode": "multiclass",
            "preprocessing": {
                "sampleRate": 22050,
                "maxLength": 22050,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "schema": "foundry.cnn_model.v1",
            "trainer": "cnn_melspec_v1",
            "model_state_dict": {},
        },
        weights_path,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "weightsPath": "model.pth",
                "classes": ["kick", "other"],
                "classificationMode": "multiclass",
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

    assert backfill_manifest_fingerprint(manifest_path, weights_path) is True
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["sharedContractFingerprint"]

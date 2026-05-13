"""Tests for central model registry installs and app-local model state.
Exists to prove v1-alpha model distribution without bundling weights into the app.
Connects manifest downloads, staged validation, and runtime bundle indexing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from echozero.models.distribution import (
    ModelInstallState,
    default_registry_manifest_source,
    discover_registry_models,
    import_local_model_bundle,
    install_model_from_registry,
    list_installed_models,
    load_registry_manifest,
    model_state_for_entry,
    save_registry_manifest_source,
    validate_installed_model,
)
from echozero.models.runtime_bundle_index import load_binary_drum_bundle_index
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_remote_model(remote_dir: Path, *, model_id: str = "default-kick") -> Path:
    bundle_dir = remote_dir / model_id
    bundle_dir.mkdir(parents=True)
    weights = bundle_dir / "model.pth"
    weights.write_bytes(b"kick weights")
    manifest = bundle_dir / "kick.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "weightsPath": "model.pth",
                "classes": ["kick", "other"],
                "classificationMode": "binary",
                "runtime": {"consumer": "BinaryDrumClassify"},
            }
        ),
        encoding="utf-8",
    )
    registry_manifest = remote_dir / "models.json"
    registry_manifest.write_text(
        json.dumps(
            {
                "schema": "echozero.model_registry.v1",
                "models": [
                    {
                        "model_id": model_id,
                        "type": "binary_drum",
                        "label": "Default Kick",
                        "version": "1.0.0-alpha.0",
                        "classes": ["kick", "other"],
                        "runtime": {"consumer": "BinaryDrumClassify"},
                        "compatibility_fingerprint": "fingerprint-kick",
                        "files": [
                            {
                                "path": "kick.manifest.json",
                                "url": f"{model_id}/kick.manifest.json",
                                "sha256": _sha256(manifest),
                                "size_bytes": manifest.stat().st_size,
                                "role": "manifest",
                            },
                            {
                                "path": "model.pth",
                                "url": f"{model_id}/model.pth",
                                "sha256": _sha256(weights),
                                "size_bytes": weights.stat().st_size,
                                "role": "weights",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return registry_manifest


def test_load_registry_manifest_resolves_relative_file_urls(tmp_path: Path) -> None:
    manifest_path = _write_remote_model(tmp_path / "remote")

    entries = load_registry_manifest(manifest_path)

    assert len(entries) == 1
    assert entries[0].model_id == "default-kick"
    assert entries[0].files[0].url.startswith("file://")


def test_install_model_from_registry_stages_validates_and_promotes_runtime_bundle(
    tmp_path: Path,
) -> None:
    manifest_path = _write_remote_model(tmp_path / "remote")
    models_dir = tmp_path / "models"

    record = install_model_from_registry(
        model_id="default-kick",
        manifest_source=manifest_path,
        models_dir=models_dir,
    )

    assert record.bundle_dir == "default-kick-1.0.0-alpha.0"
    assert (models_dir / record.bundle_dir / "model.pth").read_bytes() == b"kick weights"
    assert not (models_dir / ".staging" / record.bundle_dir).exists()
    assert list_installed_models(models_dir) == (record,)
    assert validate_installed_model(record, models_dir=models_dir) is True

    index = load_binary_drum_bundle_index(models_dir)
    assert index["kick"].bundle_dir == record.bundle_dir
    assert index["kick"].manifest_file == "kick.manifest.json"
    assert resolve_installed_binary_drum_bundles(labels=("kick",), models_dir=models_dir)[
        "kick"
    ].weights_path == (models_dir / record.bundle_dir / "model.pth").resolve()


def test_install_rejects_hash_mismatch_without_promoting_partial_bundle(tmp_path: Path) -> None:
    manifest_path = _write_remote_model(tmp_path / "remote")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["models"][0]["files"][1]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    models_dir = tmp_path / "models"

    with pytest.raises(ValueError, match="hash mismatch"):
        install_model_from_registry(
            model_id="default-kick",
            manifest_source=manifest_path,
            models_dir=models_dir,
        )

    assert not (models_dir / "default-kick-1.0.0-alpha.0").exists()
    assert list_installed_models(models_dir) == ()


def test_model_state_reports_missing_ready_outdated_and_invalid(tmp_path: Path) -> None:
    manifest_path = _write_remote_model(tmp_path / "remote")
    entry = load_registry_manifest(manifest_path)[0]
    models_dir = tmp_path / "models"

    assert model_state_for_entry(entry, models_dir=models_dir) is ModelInstallState.MISSING
    record = install_model_from_registry(
        model_id="default-kick",
        manifest_source=manifest_path,
        models_dir=models_dir,
    )
    assert model_state_for_entry(entry, models_dir=models_dir) is ModelInstallState.READY

    updated_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated_payload["models"][0]["version"] = "1.0.0-alpha.1"
    manifest_path.write_text(json.dumps(updated_payload), encoding="utf-8")
    updated_entry = load_registry_manifest(manifest_path)[0]
    assert (
        model_state_for_entry(updated_entry, models_dir=models_dir)
        is ModelInstallState.OUTDATED
    )

    (models_dir / record.bundle_dir / "model.pth").unlink()
    assert model_state_for_entry(entry, models_dir=models_dir) is ModelInstallState.INVALID


def test_discover_registry_models_uses_configured_source_and_reports_states(
    tmp_path: Path,
) -> None:
    manifest_path = _write_remote_model(tmp_path / "remote")
    models_dir = tmp_path / "models"
    save_registry_manifest_source(str(manifest_path), models_dir=models_dir)

    missing = discover_registry_models(models_dir=models_dir)

    assert default_registry_manifest_source(models_dir) == str(manifest_path)
    assert len(missing) == 1
    assert missing[0].entry.model_id == "default-kick"
    assert missing[0].state is ModelInstallState.MISSING

    install_model_from_registry(
        model_id="default-kick",
        manifest_source=manifest_path,
        models_dir=models_dir,
    )

    ready = discover_registry_models(models_dir=models_dir)

    assert ready[0].state is ModelInstallState.READY


def test_import_local_model_bundle_updates_runtime_index(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "snare.manifest.json").write_text(
        json.dumps({"weightsPath": "model.pth", "classes": ["snare", "other"]}),
        encoding="utf-8",
    )
    (bundle / "model.pth").write_bytes(b"snare weights")

    record = import_local_model_bundle(
        bundle_path=bundle,
        model_id="local-snare",
        model_type="binary_drum",
        label="Local Snare",
        version="1.0.0-alpha.0",
        classes=("snare", "other"),
        runtime_consumer="BinaryDrumClassify",
        models_dir=tmp_path / "models",
    )

    assert record.manifest_file == "snare.manifest.json"
    index = load_binary_drum_bundle_index(tmp_path / "models")
    assert index["snare"].bundle_dir == "local-snare-1.0.0-alpha.0"

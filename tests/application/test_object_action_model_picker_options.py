"""Tests for runtime model picker option labels in object-action settings."""

from __future__ import annotations

import json
from pathlib import Path

from echozero.application.timeline import object_action_model_picker_options
from echozero.models.distribution import import_local_model_bundle
from echozero.pipelines.params import KnobWidget, knob


def _write_manifest(path: Path, *, created_at: str | None = None) -> None:
    payload: dict[str, object] = {
        "weightsPath": "model.pth",
        "classes": ["kick", "other"],
    }
    if created_at is not None:
        payload["createdAt"] = created_at
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_runtime_model_option_label_includes_release_date_from_manifest_created_at(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "binary-drum-kick" / "kick.manifest.json"
    _write_manifest(manifest_path, created_at="2026-04-24T17:35:10Z")

    label = object_action_model_picker_options.runtime_model_option_label(
        path=manifest_path.resolve(),
        models_root=tmp_path.resolve(),
    )

    assert label == "binary-drum-kick/kick.manifest.json · Released 2026-04-24"


def test_runtime_model_option_label_omits_release_date_when_manifest_has_no_date(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "binary-drum-kick" / "kick.manifest.json"
    _write_manifest(manifest_path, created_at=None)

    label = object_action_model_picker_options.runtime_model_option_label(
        path=manifest_path.resolve(),
        models_root=tmp_path.resolve(),
    )

    assert label == "binary-drum-kick/kick.manifest.json"


def test_build_runtime_model_picker_options_marks_current_manifest_with_release_date(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installed_manifest = tmp_path / "binary-drum-kick" / "kick.manifest.json"
    _write_manifest(installed_manifest, created_at="2026-04-24T17:35:10Z")

    custom_manifest = tmp_path.parent / "custom-kick.manifest.json"
    _write_manifest(custom_manifest, created_at="2026-03-10T09:00:00+00:00")

    monkeypatch.setattr(
        object_action_model_picker_options,
        "resolve_installed_models_root",
        lambda: tmp_path.resolve(),
    )

    options = object_action_model_picker_options.build_runtime_model_picker_options(
        knob=knob(
            "",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".manifest.json", ".pth"),
        ),
        value=str(custom_manifest),
    )

    labels = {option.value: option.label for option in options}
    assert (
        labels[str(installed_manifest.resolve())]
        == "binary-drum-kick/kick.manifest.json · Released 2026-04-24"
    )
    assert (
        labels[str(custom_manifest)] == "Current: custom-kick.manifest.json · Released 2026-03-10"
    )


def test_runtime_model_picker_labels_central_model_status_and_skips_staging(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "source-bundle"
    _write_manifest(bundle / "kick.manifest.json", created_at="2026-04-24T17:35:10Z")
    (bundle / "model.pth").write_bytes(b"weights")
    record = import_local_model_bundle(
        bundle_path=bundle,
        model_id="default-kick",
        model_type="binary_drum",
        label="Default Kick",
        version="1.0.0-alpha.0",
        classes=("kick", "other"),
        runtime_consumer="BinaryDrumClassify",
        models_dir=tmp_path,
    )
    staged_manifest = tmp_path / ".staging" / "pending" / "pending.manifest.json"
    _write_manifest(staged_manifest, created_at="2026-04-25T17:35:10Z")

    discovered = object_action_model_picker_options.discover_runtime_model_paths(
        models_root=tmp_path,
        knob=knob("", widget=KnobWidget.MODEL_PICKER),
    )

    assert staged_manifest.resolve() not in discovered
    installed_manifest = tmp_path / record.bundle_dir / "kick.manifest.json"
    assert installed_manifest.resolve() in discovered
    assert object_action_model_picker_options.runtime_model_option_label(
        path=installed_manifest,
        models_root=tmp_path,
    ) == (
        f"{record.bundle_dir}/kick.manifest.json · Released 2026-04-24 · "
        "Installed 1.0.0-alpha.0"
    )

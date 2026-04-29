"""App-shell proof for the one-button project specialized-model flow.
Exists because EZ must own the runtime entrypoint while Foundry performs the bounded training/promote work.
Connects the app-shell method to global runtime bundle promotion and pending-config default refresh.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from echozero.foundry.services.project_specialized_model_service import (
    ProjectSpecializedModelResult,
    SpecializedModelPromotion,
)
from echozero.models.runtime_bundle_index import IndexedBinaryDrumBundle, save_binary_drum_bundle_index
from echozero.testing.analysis_mocks import build_mock_analysis_service, write_test_wav
from echozero.ui.qt.app_shell import build_app_shell


def _write_bundle(root: Path, bundle_name: str, label: str) -> Path:
    bundle_dir = root / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.pth").write_bytes(b"fixture-model")
    manifest_path = bundle_dir / f"{label}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classes": [label, "other"],
                "weightsPath": "model.pth",
                "classificationMode": "binary",
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_app_shell_runtime_create_project_specialized_drum_models_updates_pending_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    temp_root = tmp_path
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )
    global_models_root = temp_root / "global-models"
    global_kick_manifest = _write_bundle(global_models_root, "global-kick", "kick")
    global_snare_manifest = _write_bundle(global_models_root, "global-snare", "snare")
    save_binary_drum_bundle_index(
        global_models_root,
        {
            "kick": IndexedBinaryDrumBundle(
                label="kick",
                bundle_dir="global-kick",
                manifest_file=global_kick_manifest.name,
                weights_file="model.pth",
            ),
            "snare": IndexedBinaryDrumBundle(
                label="snare",
                bundle_dir="global-snare",
                manifest_file=global_snare_manifest.name,
                weights_file="model.pth",
            ),
        },
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.ensure_installed_models_dir",
        lambda: global_models_root,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.upgrade_installed_runtime_bundles",
        lambda _models_dir: None,
    )
    captured: dict[str, object] = {}
    promoted_paths: dict[str, Path] = {}

    class _FakeService:
        def create_project_specialized_drum_models(
            self,
            *,
            project_ref: str,
            labels: tuple[str, ...] = ("kick", "snare"),
        ) -> ProjectSpecializedModelResult:
            captured["project_ref"] = project_ref
            captured["labels"] = labels
            promoted_kick_manifest = _write_bundle(global_models_root, "binary-drum-kick-art-kick", "kick")
            promoted_snare_manifest = _write_bundle(global_models_root, "binary-drum-snare-art-snare", "snare")
            promoted_paths["kick"] = promoted_kick_manifest
            promoted_paths["snare"] = promoted_snare_manifest
            save_binary_drum_bundle_index(
                global_models_root,
                {
                    "kick": IndexedBinaryDrumBundle(
                        label="kick",
                        bundle_dir="binary-drum-kick-art-kick",
                        manifest_file=promoted_kick_manifest.name,
                        weights_file="model.pth",
                    ),
                    "snare": IndexedBinaryDrumBundle(
                        label="snare",
                        bundle_dir="binary-drum-snare-art-snare",
                        manifest_file=promoted_snare_manifest.name,
                        weights_file="model.pth",
                    ),
                },
            )
            return ProjectSpecializedModelResult(
                project_ref=project_ref,
                review_dataset_id="ds_review",
                review_dataset_version_id="dsv_review",
                promotions=(
                    SpecializedModelPromotion(
                        label="kick",
                        dataset_version_id="dsv_kick",
                        run_id="run_kick",
                        artifact_id="art_kick",
                        manifest_path=promoted_kick_manifest,
                        weights_path=promoted_kick_manifest.parent / "model.pth",
                    ),
                    SpecializedModelPromotion(
                        label="snare",
                        dataset_version_id="dsv_snare",
                        run_id="run_snare",
                        artifact_id="art_snare",
                        manifest_path=promoted_snare_manifest,
                        weights_path=promoted_snare_manifest.parent / "model.pth",
                    ),
                ),
            )

    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_specialized_model._build_service",
        lambda _shell: _FakeService(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "specialized-flow.wav")
        runtime.add_song_from_path("Specialized Flow", audio_path)
        runtime.describe_object_action(
            "timeline.extract_song_drum_events",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )

        result = runtime.create_project_specialized_drum_models()

        assert captured["project_ref"] == f"project:{runtime.project_storage.project.id}"
        assert captured["labels"] == ("kick", "snare")
        assert result.review_dataset_version_id == "dsv_review"

        refreshed = runtime.describe_object_action(
            "timeline.extract_song_drum_events",
            {"layer_id": "source_audio"},
            object_id="source_audio",
            object_type="layer",
        )
        assert any(
            field.key == "kick_model_path" and field.value == str(promoted_paths["kick"])
            for field in refreshed.editable_fields
        )
        assert any(
            field.key == "snare_model_path" and field.value == str(promoted_paths["snare"])
            for field in refreshed.editable_fields
        )

        song_version_id = str(runtime.session.active_song_version_id)
        config = next(
            candidate
            for candidate in runtime.project_storage.pipeline_configs.list_by_version(song_version_id)
            if candidate.template_id == "extract_song_drum_events"
        )
        assert config.knob_values["kick_model_path"] == str(promoted_paths["kick"])
        assert config.knob_values["snare_model_path"] == str(promoted_paths["snare"])
        assert config.knob_values["kick_model_path"] != str(global_kick_manifest)
        assert config.knob_values["snare_model_path"] != str(global_snare_manifest)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_create_project_specialized_drum_models_handles_first_global_promotion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    temp_root = tmp_path
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )
    global_models_root = temp_root / "global-models"
    promoted_kick_manifest = _write_bundle(global_models_root, "binary-drum-kick-art-kick", "kick")
    promoted_snare_manifest = _write_bundle(global_models_root, "binary-drum-snare-art-snare", "snare")
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.ensure_installed_models_dir",
        lambda: global_models_root,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.upgrade_installed_runtime_bundles",
        lambda _models_dir: None,
    )

    class _FakeService:
        def create_project_specialized_drum_models(
            self,
            *,
            project_ref: str,
            labels: tuple[str, ...] = ("kick", "snare"),
        ) -> ProjectSpecializedModelResult:
            assert labels == ("kick", "snare")
            return ProjectSpecializedModelResult(
                project_ref=project_ref,
                review_dataset_id="ds_review",
                review_dataset_version_id="dsv_review",
                promotions=(
                    SpecializedModelPromotion(
                        label="kick",
                        dataset_version_id="dsv_kick",
                        run_id="run_kick",
                        artifact_id="art_kick",
                        manifest_path=promoted_kick_manifest,
                        weights_path=promoted_kick_manifest.parent / "model.pth",
                    ),
                    SpecializedModelPromotion(
                        label="snare",
                        dataset_version_id="dsv_snare",
                        run_id="run_snare",
                        artifact_id="art_snare",
                        manifest_path=promoted_snare_manifest,
                        weights_path=promoted_snare_manifest.parent / "model.pth",
                    ),
                ),
            )

    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_specialized_model._build_service",
        lambda _shell: _FakeService(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "first-specialized-flow.wav")
        runtime.add_song_from_path("First Specialized Flow", audio_path)

        result = runtime.create_project_specialized_drum_models()

        assert result.review_dataset_version_id == "dsv_review"
        song_version_id = str(runtime.session.active_song_version_id)
        config = next(
            candidate
            for candidate in runtime.project_storage.pipeline_configs.list_by_version(song_version_id)
            if candidate.template_id == "extract_song_drum_events"
        )
        assert config.knob_values["kick_model_path"] == str(promoted_kick_manifest)
        assert config.knob_values["snare_model_path"] == str(promoted_snare_manifest)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_create_project_specialized_snare_model_updates_only_snare_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    temp_root = tmp_path
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )
    global_models_root = temp_root / "global-models"
    existing_snare_manifest = _write_bundle(global_models_root, "global-snare", "snare")
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.ensure_installed_models_dir",
        lambda: global_models_root,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.upgrade_installed_runtime_bundles",
        lambda _models_dir: None,
    )
    promoted_paths: dict[str, Path] = {}

    class _FakeService:
        def create_project_specialized_drum_models(
            self,
            *,
            project_ref: str,
            labels: tuple[str, ...] = ("kick", "snare"),
        ) -> ProjectSpecializedModelResult:
            assert project_ref.startswith("project:")
            assert labels == ("snare",)
            promoted_snare_manifest = _write_bundle(global_models_root, "binary-drum-snare-art-snare", "snare")
            promoted_paths["snare"] = promoted_snare_manifest
            return ProjectSpecializedModelResult(
                project_ref=project_ref,
                review_dataset_id="ds_review",
                review_dataset_version_id="dsv_review",
                promotions=(
                    SpecializedModelPromotion(
                        label="snare",
                        dataset_version_id="dsv_snare",
                        run_id="run_snare",
                        artifact_id="art_snare",
                        manifest_path=promoted_snare_manifest,
                        weights_path=promoted_snare_manifest.parent / "model.pth",
                    ),
                ),
            )

    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_specialized_model._build_service",
        lambda _shell: _FakeService(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "snare-only-specialized-flow.wav")
        runtime.add_song_from_path("Snare-Only Specialized Flow", audio_path)
        runtime.save_object_action_settings(
            "timeline.extract_song_drum_events",
            {
                "layer_id": "source_audio",
                "snare_model_path": str(existing_snare_manifest),
            },
            object_id="source_audio",
            object_type="layer",
            scope="version",
        )

        result = runtime.create_project_specialized_snare_model()

        assert result.review_dataset_version_id == "dsv_review"
        song_version_id = str(runtime.session.active_song_version_id)
        config = next(
            candidate
            for candidate in runtime.project_storage.pipeline_configs.list_by_version(song_version_id)
            if candidate.template_id == "extract_song_drum_events"
        )
        assert str(config.knob_values.get("kick_model_path", "")).strip() == ""
        assert config.knob_values["snare_model_path"] == str(promoted_paths["snare"])
        assert config.knob_values["snare_model_path"] != str(existing_snare_manifest)
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_create_project_specialized_drum_models_preserves_custom_pinned_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    temp_root = tmp_path
    runtime = build_app_shell(
        working_dir_root=temp_root / "working",
        analysis_service=build_mock_analysis_service(),
    )
    global_models_root = temp_root / "global-models"
    custom_models_root = temp_root / "custom-models"
    default_kick_manifest = _write_bundle(global_models_root, "global-kick", "kick")
    default_snare_manifest = _write_bundle(global_models_root, "global-snare", "snare")
    custom_kick_manifest = _write_bundle(custom_models_root, "custom-kick", "kick")
    custom_snare_manifest = _write_bundle(custom_models_root, "custom-snare", "snare")
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.ensure_installed_models_dir",
        lambda: global_models_root,
    )
    monkeypatch.setattr(
        "echozero.application.timeline.object_action_settings_service.upgrade_installed_runtime_bundles",
        lambda _models_dir: None,
    )
    promoted_paths: dict[str, Path] = {}

    class _FakeService:
        def create_project_specialized_drum_models(
            self,
            *,
            project_ref: str,
            labels: tuple[str, ...] = ("kick", "snare"),
        ) -> ProjectSpecializedModelResult:
            assert labels == ("kick", "snare")
            promoted_kick_manifest = _write_bundle(global_models_root, "binary-drum-kick-art-kick", "kick")
            promoted_snare_manifest = _write_bundle(global_models_root, "binary-drum-snare-art-snare", "snare")
            promoted_paths["kick"] = promoted_kick_manifest
            promoted_paths["snare"] = promoted_snare_manifest
            save_binary_drum_bundle_index(
                global_models_root,
                {
                    "kick": IndexedBinaryDrumBundle(
                        label="kick",
                        bundle_dir="binary-drum-kick-art-kick",
                        manifest_file=promoted_kick_manifest.name,
                        weights_file="model.pth",
                    ),
                    "snare": IndexedBinaryDrumBundle(
                        label="snare",
                        bundle_dir="binary-drum-snare-art-snare",
                        manifest_file=promoted_snare_manifest.name,
                        weights_file="model.pth",
                    ),
                },
            )
            return ProjectSpecializedModelResult(
                project_ref=project_ref,
                review_dataset_id="ds_review",
                review_dataset_version_id="dsv_review",
                promotions=(
                    SpecializedModelPromotion(
                        label="kick",
                        dataset_version_id="dsv_kick",
                        run_id="run_kick",
                        artifact_id="art_kick",
                        manifest_path=promoted_kick_manifest,
                        weights_path=promoted_kick_manifest.parent / "model.pth",
                    ),
                    SpecializedModelPromotion(
                        label="snare",
                        dataset_version_id="dsv_snare",
                        run_id="run_snare",
                        artifact_id="art_snare",
                        manifest_path=promoted_snare_manifest,
                        weights_path=promoted_snare_manifest.parent / "model.pth",
                    ),
                ),
            )

    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_specialized_model._build_service",
        lambda _shell: _FakeService(),
    )

    try:
        audio_path = write_test_wav(temp_root / "fixtures" / "custom-specialized-flow.wav")
        runtime.add_song_from_path("Custom Specialized Flow", audio_path)
        runtime.save_object_action_settings(
            "timeline.extract_song_drum_events",
            {
                "layer_id": "source_audio",
                "kick_model_path": str(custom_kick_manifest),
                "snare_model_path": str(custom_snare_manifest),
            },
            object_id="source_audio",
            object_type="layer",
            scope="version",
        )

        runtime.create_project_specialized_drum_models()

        song_version_id = str(runtime.session.active_song_version_id)
        config = next(
            candidate
            for candidate in runtime.project_storage.pipeline_configs.list_by_version(song_version_id)
            if candidate.template_id == "extract_song_drum_events"
        )
        assert promoted_paths["kick"].exists()
        assert promoted_paths["snare"].exists()
        assert config.knob_values["kick_model_path"] == str(custom_kick_manifest)
        assert config.knob_values["snare_model_path"] == str(custom_snare_manifest)
        assert config.knob_values["kick_model_path"] != str(default_kick_manifest)
        assert config.knob_values["snare_model_path"] != str(default_snare_manifest)
        assert config.knob_values["kick_model_path"] != str(promoted_paths["kick"])
        assert config.knob_values["snare_model_path"] != str(promoted_paths["snare"])
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)

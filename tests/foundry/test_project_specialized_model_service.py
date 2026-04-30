"""Focused proof for project-derived specialized model promotion orchestration.
Exists because EZ's one-button project training flow must stay bounded and deterministic under test.
Connects latest review dataset resolution to derived datasets, training, install, and global bundle adoption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from echozero.foundry.domain import CompatibilityReport, Dataset, DatasetVersion, ModelArtifact, TrainRun, TrainRunStatus
from echozero.foundry.services.project_specialized_model_service import ProjectSpecializedModelService
from echozero.models.runtime_bundle_index import (
    IndexedBinaryDrumBundle,
    load_binary_drum_bundle_index,
    save_binary_drum_bundle_index,
)
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles


def _write_bundle(root: Path, bundle_name: str, label: str) -> Path:
    bundle_dir = root / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
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
    (bundle_dir / "model.pth").write_bytes(b"fixture-model")
    return manifest_path


def test_project_specialized_model_service_runs_expected_flow_and_installs_global_bundles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_models_root = tmp_path / "global-models"
    monkeypatch.setattr(
        "echozero.foundry.services.project_specialized_model_service.ensure_installed_models_dir",
        lambda: global_models_root,
    )
    review_dataset = Dataset(
        id="ds_review",
        name="Review Samples",
        source_kind="project_review_export",
        metadata={"project_ref": "project:alpha", "queue_source_kind": "ez_project"},
    )
    review_version = DatasetVersion(
        id="dsv_review_latest",
        dataset_id=review_dataset.id,
        version=2,
        manifest_hash="hash-review",
        sample_rate=22050,
        audio_standard="mono_wav_pcm16",
        class_map=["kick", "snare", "tom"],
        created_at=datetime.now(UTC),
    )
    call_log: list[tuple[str, str]] = []

    class _FakeDatasets:
        def get_dataset(self, dataset_id: str) -> Dataset | None:
            if dataset_id == review_dataset.id:
                return review_dataset
            return None

        def get_version(self, version_id: str) -> DatasetVersion | None:
            if version_id == review_version.id:
                return review_version
            return DatasetVersion(
                id=version_id,
                dataset_id=f"ds_{version_id}",
                version=1,
                manifest_hash=f"hash-{version_id}",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=["kick", "other"] if "kick" in version_id else ["snare", "other"],
                split_plan={
                    "assignments": {"sm1": "train", "sm2": "val", "sm3": "test"},
                    "train_ids": ["sm1"],
                    "val_ids": ["sm2"],
                    "test_ids": ["sm3"],
                },
                created_at=datetime.now(UTC),
            )

        def derive_binary_dataset_version(self, source_version_id: str, *, positive_label: str) -> DatasetVersion:
            call_log.append(("derive", f"{source_version_id}:{positive_label}"))
            return DatasetVersion(
                id=f"dsv_{positive_label}",
                dataset_id=f"ds_{positive_label}",
                version=1,
                manifest_hash=f"hash-{positive_label}",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=[positive_label, "other"],
                split_plan={},
                created_at=datetime.now(UTC),
            )

    @dataclass
    class _InstalledBundle:
        label: str
        bundle_name: str
        bundle_dir: Path
        manifest_path: Path
        weights_path: Path

    class _FakeRuntimeBundles:
        def install_binary_drum_artifact(
            self,
            artifact_ref: str,
            *,
            models_dir: Path | None = None,
            bundle_name: str | None = None,
            bundle_label: str | None = None,
        ) -> _InstalledBundle:
            label = bundle_label or artifact_ref.split("_")[-1]
            call_log.append(("install", f"{artifact_ref}:{label}"))
            assert models_dir == global_models_root
            assert bundle_name is not None
            manifest_path = _write_bundle(models_dir, str(bundle_name), label)
            return _InstalledBundle(
                label=label,
                bundle_name=str(bundle_name),
                bundle_dir=manifest_path.parent,
                manifest_path=manifest_path,
                weights_path=manifest_path.parent / "model.pth",
            )

    class _FakeApp:
        datasets = _FakeDatasets()
        runtime_bundles = _FakeRuntimeBundles()

        def extract_project_review_dataset(
            self,
            project_path: str | Path,
            *,
            project_ref: str | None = None,
            song_id: str | None = None,
            song_version_id: str | None = None,
            layer_id: str | None = None,
            queue_source_kind: str = "ez_project",
        ) -> DatasetVersion:
            call_log.append(("export_review", f"{project_ref}:{queue_source_kind}"))
            assert Path(project_path) == tmp_path
            assert song_id is None
            assert song_version_id is None
            assert layer_id is None
            return review_version

        def plan_version(
            self,
            version_id: str,
            *,
            validation_split: float,
            test_split: float,
            seed: int,
            balance_strategy: str,
        ) -> dict[str, object]:
            call_log.append(("plan", version_id))
            return {"version_id": version_id}

        def create_run(self, dataset_version_id: str, run_spec: dict[str, object]) -> TrainRun:
            call_log.append(("create_run", dataset_version_id))
            return TrainRun(
                id=f"run_{dataset_version_id}",
                dataset_version_id=dataset_version_id,
                status=TrainRunStatus.QUEUED,
                spec=run_spec,
                spec_hash=f"hash-{dataset_version_id}",
            )

        def start_run(self, run_id: str) -> TrainRun:
            call_log.append(("start_run", run_id))
            return TrainRun(
                id=run_id,
                dataset_version_id=run_id.removeprefix("run_"),
                status=TrainRunStatus.COMPLETED,
                spec={},
                spec_hash=f"hash-{run_id}",
            )

        def list_artifacts_for_run(self, run_id: str) -> list[ModelArtifact]:
            label = "kick" if "kick" in run_id else "snare"
            return [
                ModelArtifact(
                    id=f"art_{label}",
                    run_id=run_id,
                    artifact_version="v1",
                    path=tmp_path / f"{label}.manifest.json",
                    sha256=f"sha-{label}",
                    manifest={},
                )
            ]

        def validate_artifact(self, artifact_id: str) -> CompatibilityReport:
            call_log.append(("validate", artifact_id))
            return CompatibilityReport(
                artifact_id=artifact_id,
                consumer="PyTorchAudioClassify",
                ok=True,
            )

    service = ProjectSpecializedModelService(
        tmp_path,
        foundry_app_factory=lambda _root: _FakeApp(),
    )

    result = service.create_project_specialized_drum_models(project_ref="project:alpha")

    assert result.review_dataset_id == review_dataset.id
    assert result.review_dataset_version_id == review_version.id
    assert [promotion.label for promotion in result.promotions] == ["kick", "snare"]
    assert call_log == [
        ("export_review", "project:alpha:ez_project"),
        ("derive", "dsv_review_latest:kick"),
        ("plan", "dsv_kick"),
        ("create_run", "dsv_kick"),
        ("start_run", "run_dsv_kick"),
        ("validate", "art_kick"),
        ("install", "art_kick:kick"),
        ("derive", "dsv_review_latest:snare"),
        ("plan", "dsv_snare"),
        ("create_run", "dsv_snare"),
        ("start_run", "run_dsv_snare"),
        ("validate", "art_snare"),
        ("install", "art_snare:snare"),
    ]
    bundles = resolve_installed_binary_drum_bundles(models_dir=global_models_root)
    assert bundles["kick"].manifest_path == result.promotions[0].manifest_path.resolve()
    assert bundles["snare"].manifest_path == result.promotions[1].manifest_path.resolve()
    assert result.promotions[0].manifest_path.parent.parent == global_models_root
    assert result.promotions[1].manifest_path.parent.parent == global_models_root


def test_project_specialized_model_service_can_promote_only_snare(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_models_root = tmp_path / "global-models"
    monkeypatch.setattr(
        "echozero.foundry.services.project_specialized_model_service.ensure_installed_models_dir",
        lambda: global_models_root,
    )
    review_dataset = Dataset(
        id="ds_review",
        name="Review Samples",
        source_kind="project_review_export",
        metadata={"project_ref": "project:alpha", "queue_source_kind": "ez_project"},
    )
    review_version = DatasetVersion(
        id="dsv_review_latest",
        dataset_id=review_dataset.id,
        version=2,
        manifest_hash="hash-review",
        sample_rate=22050,
        audio_standard="mono_wav_pcm16",
        class_map=["kick", "snare", "tom"],
        created_at=datetime.now(UTC),
    )
    call_log: list[tuple[str, str]] = []

    class _FakeDatasets:
        def get_dataset(self, dataset_id: str) -> Dataset | None:
            return review_dataset if dataset_id == review_dataset.id else None

        def get_version(self, version_id: str) -> DatasetVersion | None:
            if version_id == review_version.id:
                return review_version
            return DatasetVersion(
                id=version_id,
                dataset_id=f"ds_{version_id}",
                version=1,
                manifest_hash=f"hash-{version_id}",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=["snare", "other"],
                split_plan={"assignments": {"sm1": "train"}},
                created_at=datetime.now(UTC),
            )

        def derive_binary_dataset_version(self, source_version_id: str, *, positive_label: str) -> DatasetVersion:
            call_log.append(("derive", f"{source_version_id}:{positive_label}"))
            return DatasetVersion(
                id=f"dsv_{positive_label}",
                dataset_id=f"ds_{positive_label}",
                version=1,
                manifest_hash=f"hash-{positive_label}",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=[positive_label, "other"],
                split_plan={"assignments": {"sm1": "train"}},
                created_at=datetime.now(UTC),
            )

    @dataclass
    class _InstalledBundle:
        label: str
        bundle_name: str
        bundle_dir: Path
        manifest_path: Path
        weights_path: Path

    class _FakeRuntimeBundles:
        def install_binary_drum_artifact(
            self,
            artifact_ref: str,
            *,
            models_dir: Path | None = None,
            bundle_name: str | None = None,
            bundle_label: str | None = None,
        ) -> _InstalledBundle:
            label = bundle_label or artifact_ref.split("_")[-1]
            call_log.append(("install", f"{artifact_ref}:{label}"))
            assert models_dir == global_models_root
            assert bundle_name is not None
            manifest_path = _write_bundle(models_dir, str(bundle_name), label)
            return _InstalledBundle(
                label=label,
                bundle_name=str(bundle_name),
                bundle_dir=manifest_path.parent,
                manifest_path=manifest_path,
                weights_path=manifest_path.parent / "model.pth",
            )

    class _FakeApp:
        datasets = _FakeDatasets()
        runtime_bundles = _FakeRuntimeBundles()

        def extract_project_review_dataset(
            self,
            project_path: str | Path,
            *,
            project_ref: str | None = None,
            song_id: str | None = None,
            song_version_id: str | None = None,
            layer_id: str | None = None,
            queue_source_kind: str = "ez_project",
        ) -> DatasetVersion:
            call_log.append(("export_review", f"{project_ref}:{queue_source_kind}"))
            assert Path(project_path) == tmp_path
            assert song_id is None
            assert song_version_id is None
            assert layer_id is None
            return review_version

        def create_run(self, dataset_version_id: str, run_spec: dict[str, object]) -> TrainRun:
            call_log.append(("create_run", dataset_version_id))
            return TrainRun(
                id=f"run_{dataset_version_id}",
                dataset_version_id=dataset_version_id,
                status=TrainRunStatus.QUEUED,
                spec=run_spec,
                spec_hash=f"hash-{dataset_version_id}",
            )

        def start_run(self, run_id: str) -> TrainRun:
            call_log.append(("start_run", run_id))
            return TrainRun(
                id=run_id,
                dataset_version_id=run_id.removeprefix("run_"),
                status=TrainRunStatus.COMPLETED,
                spec={},
                spec_hash=f"hash-{run_id}",
            )

        def list_artifacts_for_run(self, run_id: str) -> list[ModelArtifact]:
            return [
                ModelArtifact(
                    id="art_snare",
                    run_id=run_id,
                    artifact_version="v1",
                    path=tmp_path / "snare.manifest.json",
                    sha256="sha-snare",
                    manifest={},
                )
            ]

        def validate_artifact(self, artifact_id: str) -> CompatibilityReport:
            call_log.append(("validate", artifact_id))
            return CompatibilityReport(
                artifact_id=artifact_id,
                consumer="PyTorchAudioClassify",
                ok=True,
            )

    service = ProjectSpecializedModelService(
        tmp_path,
        foundry_app_factory=lambda _root: _FakeApp(),
    )

    result = service.create_project_specialized_drum_models(
        project_ref="project:alpha",
        labels=("snare",),
    )

    assert [promotion.label for promotion in result.promotions] == ["snare"]
    assert call_log == [
        ("export_review", "project:alpha:ez_project"),
        ("derive", "dsv_review_latest:snare"),
        ("create_run", "dsv_snare"),
        ("start_run", "run_dsv_snare"),
        ("validate", "art_snare"),
        ("install", "art_snare:snare"),
    ]
    bundles = resolve_installed_binary_drum_bundles(
        models_dir=global_models_root,
        labels=("snare",),
    )
    assert set(bundles.keys()) == {"snare"}
    assert bundles["snare"].manifest_path == result.promotions[0].manifest_path.resolve()


def test_project_specialized_model_service_restores_previous_global_index_on_later_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_models_root = tmp_path / "global-models"
    monkeypatch.setattr(
        "echozero.foundry.services.project_specialized_model_service.ensure_installed_models_dir",
        lambda: global_models_root,
    )
    old_kick_manifest = _write_bundle(global_models_root, "global-kick", "kick")
    old_snare_manifest = _write_bundle(global_models_root, "global-snare", "snare")
    previous_index = {
        "kick": IndexedBinaryDrumBundle(
            label="kick",
            bundle_dir="global-kick",
            manifest_file=old_kick_manifest.name,
            weights_file="model.pth",
        ),
        "snare": IndexedBinaryDrumBundle(
            label="snare",
            bundle_dir="global-snare",
            manifest_file=old_snare_manifest.name,
            weights_file="model.pth",
        ),
    }
    save_binary_drum_bundle_index(global_models_root, previous_index)
    review_dataset = Dataset(
        id="ds_review",
        name="Review Samples",
        source_kind="project_review_export",
        metadata={"project_ref": "project:alpha", "queue_source_kind": "ez_project"},
    )
    review_version = DatasetVersion(
        id="dsv_review_latest",
        dataset_id=review_dataset.id,
        version=1,
        manifest_hash="hash-review",
        sample_rate=22050,
        audio_standard="mono_wav_pcm16",
        class_map=["kick", "snare"],
        created_at=datetime.now(UTC),
    )

    class _FakeDatasets:
        def get_dataset(self, dataset_id: str) -> Dataset | None:
            return review_dataset if dataset_id == review_dataset.id else None

        def get_version(self, version_id: str) -> DatasetVersion | None:
            if version_id == review_version.id:
                return review_version
            return DatasetVersion(
                id=version_id,
                dataset_id=f"ds_{version_id}",
                version=1,
                manifest_hash=f"hash-{version_id}",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=["kick", "other"] if "kick" in version_id else ["snare", "other"],
                split_plan={"assignments": {"sm1": "train"}},
                created_at=datetime.now(UTC),
            )

        def derive_binary_dataset_version(self, source_version_id: str, *, positive_label: str) -> DatasetVersion:
            return DatasetVersion(
                id=f"dsv_{positive_label}",
                dataset_id=f"ds_{positive_label}",
                version=1,
                manifest_hash=f"hash-{positive_label}",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=[positive_label, "other"],
                split_plan={"assignments": {"sm1": "train"}},
                created_at=datetime.now(UTC),
            )

    @dataclass
    class _InstalledBundle:
        label: str
        bundle_name: str
        bundle_dir: Path
        manifest_path: Path
        weights_path: Path

    class _FakeRuntimeBundles:
        def install_binary_drum_artifact(
            self,
            artifact_ref: str,
            *,
            models_dir: Path | None = None,
            bundle_name: str | None = None,
            bundle_label: str | None = None,
        ) -> _InstalledBundle:
            assert models_dir == global_models_root
            assert bundle_name is not None
            label = bundle_label or artifact_ref.split("_")[-1]
            manifest_path = _write_bundle(models_dir, str(bundle_name), label)
            updated_index = load_binary_drum_bundle_index(models_dir)
            updated_index[label] = IndexedBinaryDrumBundle(
                label=label,
                bundle_dir=str(bundle_name),
                manifest_file=manifest_path.name,
                weights_file="model.pth",
                artifact_id=artifact_ref,
            )
            save_binary_drum_bundle_index(models_dir, updated_index)
            return _InstalledBundle(
                label=label,
                bundle_name=str(bundle_name),
                bundle_dir=manifest_path.parent,
                manifest_path=manifest_path,
                weights_path=manifest_path.parent / "model.pth",
            )

    class _FakeApp:
        datasets = _FakeDatasets()
        runtime_bundles = _FakeRuntimeBundles()

        def extract_project_review_dataset(
            self,
            project_path: str | Path,
            *,
            project_ref: str | None = None,
            song_id: str | None = None,
            song_version_id: str | None = None,
            layer_id: str | None = None,
            queue_source_kind: str = "ez_project",
        ) -> DatasetVersion:
            assert Path(project_path) == tmp_path
            assert project_ref == "project:alpha"
            assert queue_source_kind == "ez_project"
            assert song_id is None
            assert song_version_id is None
            assert layer_id is None
            return review_version

        def create_run(self, dataset_version_id: str, run_spec: dict[str, object]) -> TrainRun:
            return TrainRun(
                id=f"run_{dataset_version_id}",
                dataset_version_id=dataset_version_id,
                status=TrainRunStatus.QUEUED,
                spec=run_spec,
                spec_hash=f"hash-{dataset_version_id}",
            )

        def start_run(self, run_id: str) -> TrainRun:
            return TrainRun(
                id=run_id,
                dataset_version_id=run_id.removeprefix("run_"),
                status=TrainRunStatus.COMPLETED,
                spec={},
                spec_hash=f"hash-{run_id}",
            )

        def list_artifacts_for_run(self, run_id: str) -> list[ModelArtifact]:
            label = "kick" if "kick" in run_id else "snare"
            return [
                ModelArtifact(
                    id=f"art_{label}",
                    run_id=run_id,
                    artifact_version="v1",
                    path=tmp_path / f"{label}.manifest.json",
                    sha256=f"sha-{label}",
                    manifest={},
                )
            ]

        def validate_artifact(self, artifact_id: str) -> CompatibilityReport:
            if artifact_id == "art_snare":
                return CompatibilityReport(
                    artifact_id=artifact_id,
                    consumer="PyTorchAudioClassify",
                    ok=False,
                    errors=("snare-invalid",),
                )
            return CompatibilityReport(
                artifact_id=artifact_id,
                consumer="PyTorchAudioClassify",
                ok=True,
            )

    service = ProjectSpecializedModelService(
        tmp_path,
        foundry_app_factory=lambda _root: _FakeApp(),
    )

    with pytest.raises(RuntimeError, match="failed validation"):
        service.create_project_specialized_drum_models(project_ref="project:alpha")

    restored_index = load_binary_drum_bundle_index(global_models_root)
    assert restored_index == previous_index
    bundles = resolve_installed_binary_drum_bundles(models_dir=global_models_root)
    assert bundles["kick"].manifest_path == old_kick_manifest.resolve()
    assert bundles["snare"].manifest_path == old_snare_manifest.resolve()
    assert not (global_models_root / "binary-drum-kick-art-kick").exists()

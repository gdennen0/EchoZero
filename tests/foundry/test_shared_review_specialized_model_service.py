"""Focused proof for shared-review artist drum model orchestration.
Exists because Noah Kahan models must train from the shared review pool, not project exports.
Connects folder dataset ingestion, beefy CRNN specs, warm-starts, and runtime bundle naming.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from echozero.foundry.domain import (
    CompatibilityReport,
    Dataset,
    DatasetVersion,
    ModelArtifact,
    TrainRun,
    TrainRunStatus,
)
from echozero.foundry.services.shared_review_specialized_model_service import (
    SharedReviewSpecializedModelService,
)
from echozero.models.runtime_bundle_index import (
    IndexedBinaryDrumBundle,
    save_binary_drum_bundle_index,
)


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


def test_shared_review_specialized_service_builds_noah_kahan_beefy_warm_started_bundles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    global_models_root = tmp_path / "global-models"
    old_kick_manifest = _write_bundle(global_models_root, "foundry-kick-ovr-crnn-v2", "kick")
    old_snare_manifest = _write_bundle(
        global_models_root, "binary-drum-snare-olivia-monster", "snare"
    )
    save_binary_drum_bundle_index(
        global_models_root,
        {
            "kick": IndexedBinaryDrumBundle(
                label="kick",
                bundle_dir="foundry-kick-ovr-crnn-v2",
                manifest_file="kick.manifest.json",
                weights_file="model.pth",
            ),
            "snare": IndexedBinaryDrumBundle(
                label="snare",
                bundle_dir="binary-drum-snare-olivia-monster",
                manifest_file="snare.manifest.json",
                weights_file="model.pth",
            ),
        },
    )
    monkeypatch.setattr(
        "echozero.foundry.services.shared_review_specialized_model_service.ensure_installed_models_dir",
        lambda: global_models_root,
    )
    source_dataset = Dataset(
        id="ds_shared",
        name="Noah Kahan Shared Review Samples",
        source_kind="shared_review_samples",
    )
    source_version = DatasetVersion(
        id="dsv_shared",
        dataset_id=source_dataset.id,
        version=1,
        manifest_hash="hash-shared",
        sample_rate=22050,
        audio_standard="mono_wav_pcm16",
        class_map=["kick", "snare"],
        created_at=datetime.now(UTC),
    )
    created_specs: dict[str, dict[str, object]] = {}

    class _FakeDatasets:
        def ingest_shared_review_sample_folders(
            self,
            root_path: str | Path,
            *,
            dataset_name: str,
            labels: tuple[str, ...],
        ) -> DatasetVersion:
            assert Path(root_path) == tmp_path / "review_samples"
            assert dataset_name == "Noah Kahan Shared Review Samples"
            assert labels == ("clap", "kick", "snare")
            return source_version

        def get_dataset(self, dataset_id: str) -> Dataset | None:
            return source_dataset if dataset_id == source_dataset.id else None

        def get_version(self, version_id: str) -> DatasetVersion | None:
            label = version_id.removeprefix("dsv_")
            return DatasetVersion(
                id=version_id,
                dataset_id=f"ds_{label}",
                version=1,
                manifest_hash=f"hash-{label}",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=[label, "other"],
                split_plan={"assignments": {"sm1": "train"}, "train_ids": ["sm1"]},
                created_at=datetime.now(UTC),
            )

        def derive_binary_dataset_version(
            self,
            source_version_id: str,
            *,
            positive_label: str,
        ) -> DatasetVersion:
            assert source_version_id == source_version.id
            return DatasetVersion(
                id=f"dsv_{positive_label}",
                dataset_id=f"ds_{positive_label}",
                version=1,
                manifest_hash=f"hash-{positive_label}",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=[positive_label, "other"],
                split_plan={"assignments": {"sm1": "train"}, "train_ids": ["sm1"]},
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
            assert bundle_label is not None
            assert bundle_name is not None
            manifest_path = _write_bundle(models_dir, bundle_name, bundle_label)
            return _InstalledBundle(
                label=bundle_label,
                bundle_name=bundle_name,
                bundle_dir=manifest_path.parent,
                manifest_path=manifest_path,
                weights_path=manifest_path.parent / "model.pth",
            )

    class _FakeApp:
        datasets = _FakeDatasets()
        runtime_bundles = _FakeRuntimeBundles()

        def plan_version(self, *args, **kwargs) -> dict[str, object]:
            raise AssertionError("derived fixtures are already planned")

        def create_run(self, dataset_version_id: str, run_spec: dict[str, object]) -> TrainRun:
            created_specs[dataset_version_id] = run_spec
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
            return CompatibilityReport(
                artifact_id=artifact_id,
                consumer="PyTorchAudioClassify",
                ok=True,
            )

    service = SharedReviewSpecializedModelService(
        tmp_path,
        foundry_app_factory=lambda _root: _FakeApp(),
    )
    review_root = tmp_path / "review_samples"
    for class_name in ("kick", "snare", "clap"):
        class_dir = review_root / class_name
        class_dir.mkdir(parents=True)
        (class_dir / f"{class_name}.wav").write_bytes(b"fixture")

    result = service.create_artist_drum_models(
        artist_name="Noah Kahan",
        review_sample_root=review_root,
        initial_model_paths={
            "kick": old_kick_manifest,
            "snare": old_snare_manifest,
        },
    )

    assert [promotion.label for promotion in result.promotions] == ["kick", "snare"]
    assert result.promotions[0].manifest_path.parent.name == "binary-drum-kick-noah-kahan-art-kick"
    assert (
        result.promotions[1].manifest_path.parent.name == "binary-drum-snare-noah-kahan-art-snare"
    )
    kick_spec = created_specs["dsv_kick"]
    snare_spec = created_specs["dsv_snare"]
    assert kick_spec["model"] == {"type": "crnn", "initialWeightsPath": str(old_kick_manifest)}
    assert snare_spec["model"] == {
        "type": "crnn",
        "initialWeightsPath": str(old_snare_manifest),
    }
    assert kick_spec["training"]["profileName"] == "beefy"
    assert kick_spec["training"]["epochs"] == 12
    assert kick_spec["training"]["optimizer"] == "adamw"
    installed_manifest = json.loads(result.promotions[0].manifest_path.read_text(encoding="utf-8"))
    assert installed_manifest["specialization"]["targetIdentity"] == "Noah Kahan"
    assert installed_manifest["specialization"]["label"] == "kick"
    assert installed_manifest["displayIdentity"]["trainingProfile"] == "beefy"


def test_crnn_trainer_loads_initial_weights_from_runtime_manifest(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from echozero.foundry.services.crnn_trainer import CrnnTrainer
    from echozero.runtime_models.architectures import CrnnRuntimeModel

    source_model = CrnnRuntimeModel(num_classes=2, mel_bins=128)
    weights_path = tmp_path / "model.pth"
    manifest_path = tmp_path / "kick.manifest.json"
    torch.save(
        {
            "classes": ["kick", "other"],
            "preprocessing": {"nMels": 128},
            "model_state_dict": source_model.state_dict(),
        },
        weights_path,
    )
    manifest_path.write_text(
        json.dumps({"classes": ["kick", "other"], "weightsPath": "model.pth"}),
        encoding="utf-8",
    )
    target_model = CrnnRuntimeModel(num_classes=2, mel_bins=128)
    run = TrainRun(
        id="run_warm",
        dataset_version_id="dsv_kick",
        status=TrainRunStatus.QUEUED,
        spec={"model": {"type": "crnn", "initialWeightsPath": str(manifest_path)}},
        spec_hash="hash-warm",
    )

    summary = CrnnTrainer(tmp_path)._load_initial_weights(
        target_model,
        run=run,
        class_names=["kick", "other"],
        n_mels=128,
    )

    assert summary == {
        "kind": "warm_start",
        "sourcePath": str(weights_path.resolve()),
        "classes": ["kick", "other"],
    }
    for key, value in source_model.state_dict().items():
        assert torch.equal(target_model.state_dict()[key], value)

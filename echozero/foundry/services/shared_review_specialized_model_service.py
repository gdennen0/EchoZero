"""
SharedReviewSpecializedModelService: Train artist drum models from shared review samples.
Exists so client-specialized Foundry builds can reuse the runtime bundle path without project samples.
Connects class-folder review sample pools, one-vs-rest CRNN runs, and installed drum bundles.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from echozero.foundry import FoundryApp
from echozero.foundry.domain import Dataset, DatasetVersion, TrainRunStatus
from echozero.foundry.review_samples import ReviewSampleTrainingRole
from echozero.models.paths import ensure_installed_models_dir
from echozero.models.runtime_bundle_index import (
    load_binary_drum_bundle_index,
    save_binary_drum_bundle_index,
)
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles


@dataclass(frozen=True, slots=True)
class SharedReviewSpecializedPromotion:
    """Outcome for one shared-review positive-label model promotion."""

    label: str
    dataset_version_id: str
    run_id: str
    artifact_id: str
    manifest_path: Path
    weights_path: Path
    initial_model_path: Path | None


@dataclass(frozen=True, slots=True)
class SharedReviewSpecializedModelResult:
    """Compact result for shared-review artist model creation."""

    artist_name: str
    source_dataset_id: str
    source_dataset_version_id: str
    review_sample_root: Path
    promotions: tuple[SharedReviewSpecializedPromotion, ...]


class SharedReviewSpecializedModelService:
    """Train and promote artist-scoped binary drum models from shared review folders."""

    _default_labels = ("kick", "snare")

    def __init__(
        self,
        root: Path,
        *,
        foundry_app_factory: Callable[[Path], FoundryApp] = FoundryApp,
    ) -> None:
        self._root = Path(root)
        self._foundry_app_factory = foundry_app_factory

    def create_artist_drum_models(
        self,
        *,
        artist_name: str = "Noah Kahan",
        review_sample_root: Path | None = None,
        labels: tuple[str, ...] = _default_labels,
        source_labels: tuple[str, ...] | None = None,
        initial_model_paths: dict[str, Path] | None = None,
        warm_start: bool = True,
    ) -> SharedReviewSpecializedModelResult:
        """Train, validate, and install artist drum bundles from shared review samples."""
        app = self._foundry_app_factory(self._root)
        sample_root = self.resolve_review_sample_root(review_sample_root)
        ingest_labels = source_labels or self.discover_review_sample_labels(sample_root)
        source_version = app.datasets.ingest_shared_review_sample_folders(
            sample_root,
            dataset_name=f"{artist_name} Shared Review Samples",
            labels=ingest_labels,
        )
        source_dataset = app.datasets.get_dataset(source_version.dataset_id)
        if source_dataset is None:
            raise RuntimeError(
                f"Shared review dataset metadata is incomplete for '{source_version.id}'."
            )

        selected_labels = self._resolve_requested_labels(
            labels,
            available_labels=tuple(source_version.class_map),
        )
        models_dir = ensure_installed_models_dir().resolve()
        previous_index = load_binary_drum_bundle_index(models_dir)
        initial_models = self._normalize_initial_model_paths(initial_model_paths)
        if not initial_models:
            initial_models = self._resolve_initial_model_paths(
                selected_labels,
                models_dir=models_dir,
                enabled=warm_start,
            )
        promotions: list[SharedReviewSpecializedPromotion] = []
        installed_bundle_dirs: list[Path] = []

        try:
            for label in selected_labels:
                derived = app.datasets.derive_binary_dataset_version(
                    source_version.id,
                    positive_label=label,
                )
                if not derived.split_plan.get("assignments"):
                    app.plan_version(
                        derived.id,
                        validation_split=0.15,
                        test_split=0.10,
                        seed=42,
                        balance_strategy="none",
                    )
                    refreshed = app.datasets.get_version(derived.id)
                    if refreshed is None:
                        raise RuntimeError(f"Derived dataset version disappeared: {derived.id}")
                    derived = refreshed

                run = app.create_run(
                    derived.id,
                    self._build_beefy_binary_run_spec(
                        derived,
                        initial_model_path=initial_models.get(label),
                    ),
                )
                completed_run = app.start_run(run.id)
                if completed_run.status is not TrainRunStatus.COMPLETED:
                    raise RuntimeError(
                        f"Shared review model run for '{label}' did not complete successfully: "
                        f"{completed_run.status.value}"
                    )

                artifacts = app.list_artifacts_for_run(completed_run.id)
                if not artifacts:
                    raise RuntimeError(
                        f"No artifact was finalized for shared review run '{completed_run.id}'."
                    )
                artifact = sorted(artifacts, key=lambda candidate: candidate.created_at)[-1]
                compatibility = app.validate_artifact(artifact.id)
                if not compatibility.ok:
                    raise RuntimeError(
                        f"Shared review artifact '{artifact.id}' failed validation: "
                        f"{compatibility.errors[0]}"
                    )
                installed = app.runtime_bundles.install_binary_drum_artifact(
                    artifact.id,
                    models_dir=models_dir,
                    bundle_label=label,
                    bundle_name=self._bundle_name(
                        artist_name=artist_name,
                        label=label,
                        artifact_id=artifact.id,
                    ),
                )
                self._annotate_installed_manifest(
                    installed.manifest_path,
                    artist_name=artist_name,
                    label=label,
                    source_dataset=source_dataset,
                    source_version=source_version,
                    initial_model_path=initial_models.get(label),
                )
                installed_bundle_dirs.append(installed.bundle_dir)
                promotions.append(
                    SharedReviewSpecializedPromotion(
                        label=label,
                        dataset_version_id=derived.id,
                        run_id=completed_run.id,
                        artifact_id=artifact.id,
                        manifest_path=installed.manifest_path,
                        weights_path=installed.weights_path,
                        initial_model_path=initial_models.get(label),
                    )
                )
        except Exception:
            for bundle_dir in installed_bundle_dirs:
                shutil.rmtree(bundle_dir, ignore_errors=True)
            save_binary_drum_bundle_index(models_dir, previous_index)
            raise

        return SharedReviewSpecializedModelResult(
            artist_name=artist_name,
            source_dataset_id=source_dataset.id,
            source_dataset_version_id=source_version.id,
            review_sample_root=sample_root,
            promotions=tuple(promotions),
        )

    @staticmethod
    def resolve_review_sample_root(review_sample_root: Path | None = None) -> Path:
        """Resolve the shared local review-sample export root."""
        if review_sample_root is not None:
            return Path(review_sample_root).expanduser().resolve()
        default_root = "~/.echozero/data/tmp/review_samples"
        return (
            Path(os.environ.get("ECHOZERO_REVIEW_SAMPLE_EXPORT_ROOT", default_root))
            .expanduser()
            .resolve()
        )

    @staticmethod
    def discover_review_sample_labels(review_sample_root: Path) -> tuple[str, ...]:
        """List class folders that contain at least one audio sample."""
        labels: set[str] = set()
        candidate_dirs: list[Path] = []
        for role in ReviewSampleTrainingRole:
            role_dir = review_sample_root / role.value
            if role_dir.is_dir():
                candidate_dirs.extend(path for path in role_dir.iterdir() if path.is_dir())
        candidate_dirs.extend(
            path
            for path in review_sample_root.iterdir()
            if path.is_dir() and path.name not in {role.value for role in ReviewSampleTrainingRole}
        )
        for class_dir in sorted(candidate_dirs):
            if any(
                path.is_file()
                and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}
                for path in class_dir.rglob("*")
            ):
                labels.add(class_dir.name.strip().lower())
        if not labels:
            raise ValueError(f"No shared review sample class folders found: {review_sample_root}")
        return tuple(sorted(labels))

    @classmethod
    def _resolve_requested_labels(
        cls,
        labels: tuple[str, ...],
        *,
        available_labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        available = {
            str(raw_label).strip().lower()
            for raw_label in available_labels
            if str(raw_label).strip()
        }
        normalized_labels: list[str] = []
        seen: set[str] = set()
        for raw_label in labels:
            label = str(raw_label).strip().lower()
            if not label or label in seen:
                continue
            if label not in available:
                supported = ", ".join(sorted(available)) or "(none)"
                raise ValueError(
                    f"Unsupported shared review model label '{label}'. "
                    f"Available review labels: {supported}."
                )
            normalized_labels.append(label)
            seen.add(label)
        if not normalized_labels:
            raise ValueError("At least one shared review model label is required.")
        return tuple(normalized_labels)

    @staticmethod
    def _build_beefy_binary_run_spec(
        version: DatasetVersion,
        *,
        initial_model_path: Path | None,
    ) -> dict[str, object]:
        model: dict[str, object] = {"type": "crnn"}
        if initial_model_path is not None:
            model["initialWeightsPath"] = str(initial_model_path)
        return {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "binary",
            "model": model,
            "data": {
                "datasetVersionId": version.id,
                "sampleRate": version.sample_rate,
                "maxLength": version.sample_rate,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "training": {
                "epochs": 12,
                "batchSize": 4,
                "learningRate": 0.001,
                "seed": 42,
                "deterministic": True,
                "trainerProfile": "stronger_v1",
                "optimizer": "adamw",
                "regularizationAlpha": 0.00005,
                "weightDecay": 0.0001,
                "averageWeights": True,
                "earlyStoppingPatience": 4,
                "minEpochs": 4,
                "classWeighting": "balanced",
                "rebalanceStrategy": "oversample",
                "augmentTrain": True,
                "augmentNoiseStd": 0.03,
                "augmentGainJitter": 0.15,
                "augmentCopies": 2,
                "syntheticMix": {"enabled": True, "ratio": 0.35, "cap": 400},
                "profileName": "beefy",
            },
        }

    @staticmethod
    def _resolve_initial_model_paths(
        labels: tuple[str, ...],
        *,
        models_dir: Path,
        enabled: bool,
    ) -> dict[str, Path]:
        if not enabled:
            return {}
        bundles = resolve_installed_binary_drum_bundles(labels=labels, models_dir=models_dir)
        return {
            label: bundle.manifest_path
            for label, bundle in bundles.items()
            if bundle.manifest_path.exists()
        }

    @staticmethod
    def _normalize_initial_model_paths(
        initial_model_paths: dict[str, Path] | None,
    ) -> dict[str, Path]:
        if not initial_model_paths:
            return {}
        return {
            str(label).strip().lower(): Path(path).expanduser().resolve()
            for label, path in initial_model_paths.items()
            if str(label).strip()
        }

    @classmethod
    def _bundle_name(
        cls,
        *,
        artist_name: str,
        label: str,
        artifact_id: str,
    ) -> str:
        artist_slug = cls._slug(artist_name)
        artifact_slug = cls._slug(artifact_id)
        return f"binary-drum-{label}-{artist_slug}-{artifact_slug}"

    @staticmethod
    def _annotate_installed_manifest(
        manifest_path: Path,
        *,
        artist_name: str,
        label: str,
        source_dataset: Dataset,
        source_version: DatasetVersion,
        initial_model_path: Path | None,
    ) -> None:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["specialization"] = {
            "schema": "foundry.runtime_specialization.v1",
            "targetIdentity": artist_name,
            "label": label,
            "sourceKind": "shared_review_samples",
            "sourceDatasetId": source_dataset.id,
            "sourceDatasetVersionId": source_version.id,
            "trainingProfile": "beefy",
            "initialModelPath": None if initial_model_path is None else str(initial_model_path),
        }
        display_identity = dict(payload.get("displayIdentity") or {})
        display_identity["targetIdentity"] = artist_name
        display_identity["label"] = label
        display_identity["trainingProfile"] = "beefy"
        payload["displayIdentity"] = display_identity
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _slug(value: str) -> str:
        return value.strip().lower().replace("_", "-").replace(" ", "-")

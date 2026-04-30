"""
ProjectSpecializedModelService: Promote review-driven drum classifiers for one EZ project.
Exists to keep the one-button Stage Zero specialized-model flow out of Qt and inside Foundry services.
Connects project review datasets, bounded training runs, and global runtime bundle installation.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from echozero.foundry import FoundryApp
from echozero.foundry.domain import Dataset, DatasetVersion, TrainRunStatus
from echozero.models.paths import ensure_installed_models_dir
from echozero.models.runtime_bundle_index import (
    load_binary_drum_bundle_index,
    save_binary_drum_bundle_index,
)


@dataclass(frozen=True, slots=True)
class SpecializedModelPromotion:
    """Outcome for one positive-label specialized classifier promotion."""

    label: str
    dataset_version_id: str
    run_id: str
    artifact_id: str
    manifest_path: Path
    weights_path: Path


@dataclass(frozen=True, slots=True)
class ProjectSpecializedModelResult:
    """Compact result for the EZ specialized-drum-model creation flow."""

    project_ref: str
    review_dataset_id: str
    review_dataset_version_id: str
    promotions: tuple[SpecializedModelPromotion, ...]


class ProjectSpecializedModelService:
    """Train and promote project-derived kick/snare one-vs-rest models into global runtime bundles."""

    _default_labels = ("kick", "snare")

    def __init__(
        self,
        root: Path,
        *,
        foundry_app_factory: Callable[[Path], FoundryApp] = FoundryApp,
    ) -> None:
        self._root = Path(root)
        self._foundry_app_factory = foundry_app_factory

    def create_project_specialized_drum_models(
        self,
        *,
        project_ref: str,
        labels: tuple[str, ...] = _default_labels,
    ) -> ProjectSpecializedModelResult:
        """Train, validate, and install project-derived kick/snare runtime bundles."""
        app = self._foundry_app_factory(self._root)
        selected_labels = self._resolve_requested_labels(labels)
        review_dataset, review_version = self._extract_project_review_dataset(
            app,
            project_ref=project_ref,
        )
        models_dir = ensure_installed_models_dir().resolve()
        previous_index = load_binary_drum_bundle_index(models_dir)
        promotions: list[SpecializedModelPromotion] = []
        installed_bundle_dirs: list[Path] = []

        try:
            for label in selected_labels:
                derived = app.datasets.derive_binary_dataset_version(
                    review_version.id,
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

                run = app.create_run(derived.id, self._build_binary_run_spec(derived))
                completed_run = app.start_run(run.id)
                if completed_run.status is not TrainRunStatus.COMPLETED:
                    raise RuntimeError(
                        f"Specialized model run for '{label}' did not complete successfully: "
                        f"{completed_run.status.value}"
                    )

                artifacts = app.list_artifacts_for_run(completed_run.id)
                if not artifacts:
                    raise RuntimeError(f"No artifact was finalized for specialized run '{completed_run.id}'.")
                artifact = sorted(artifacts, key=lambda candidate: candidate.created_at)[-1]
                compatibility = app.validate_artifact(artifact.id)
                if not compatibility.ok:
                    raise RuntimeError(
                        f"Specialized artifact '{artifact.id}' failed validation: "
                        f"{compatibility.errors[0]}"
                    )
                installed = app.runtime_bundles.install_binary_drum_artifact(
                    artifact.id,
                    models_dir=models_dir,
                    bundle_name=self._bundle_name(label=label, artifact_id=artifact.id),
                )
                installed_bundle_dirs.append(installed.bundle_dir)
                promotions.append(
                    SpecializedModelPromotion(
                        label=label,
                        dataset_version_id=derived.id,
                        run_id=completed_run.id,
                        artifact_id=artifact.id,
                        manifest_path=installed.manifest_path,
                        weights_path=installed.weights_path,
                    )
                )
        except Exception:
            for bundle_dir in installed_bundle_dirs:
                shutil.rmtree(bundle_dir, ignore_errors=True)
            save_binary_drum_bundle_index(models_dir, previous_index)
            raise

        return ProjectSpecializedModelResult(
            project_ref=project_ref,
            review_dataset_id=review_dataset.id,
            review_dataset_version_id=review_version.id,
            promotions=tuple(promotions),
        )

    @classmethod
    def _resolve_requested_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        normalized_labels: list[str] = []
        seen: set[str] = set()
        for raw_label in labels:
            label = str(raw_label).strip().lower()
            if not label or label in seen:
                continue
            if label not in cls._default_labels:
                supported = ", ".join(cls._default_labels)
                raise ValueError(
                    f"Unsupported specialized model label '{label}'. Supported labels: {supported}."
                )
            normalized_labels.append(label)
            seen.add(label)
        if not normalized_labels:
            raise ValueError("At least one specialized model label is required.")
        return tuple(normalized_labels)

    def _extract_project_review_dataset(
        self,
        app: FoundryApp,
        *,
        project_ref: str,
    ) -> tuple[Dataset, DatasetVersion]:
        version = app.extract_project_review_dataset(
            self._root,
            project_ref=project_ref,
        )
        dataset = app.datasets.get_dataset(version.dataset_id)
        if dataset is None or version is None:
            raise RuntimeError(
                f"Exported project review dataset metadata is incomplete for project '{project_ref}'."
            )
        return dataset, version

    @staticmethod
    def _build_binary_run_spec(version: DatasetVersion) -> dict[str, object]:
        return {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "binary",
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
                "epochs": 4,
                "batchSize": 4,
                "learningRate": 0.01,
                "seed": 42,
                "classWeighting": "balanced",
                "rebalanceStrategy": "oversample",
                "augmentTrain": True,
                "augmentNoiseStd": 0.03,
                "augmentGainJitter": 0.15,
                "augmentCopies": 2,
                "trainerProfile": "baseline_v1",
                "optimizer": "sgd_constant",
                "regularizationAlpha": 0.0001,
                "averageWeights": False,
            },
        }

    @staticmethod
    def _bundle_name(*, label: str, artifact_id: str) -> str:
        slug = artifact_id.strip().lower().replace("_", "-")
        return f"binary-drum-{label}-{slug}"

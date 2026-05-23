"""Candidate run planning for model evolution.
Exists to standardize one-vs-rest training specs, profiles, negatives, and lineage.
Connects event-span datasets to Foundry train runs without changing legacy flows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from echozero.foundry.domain import DatasetVersion
from echozero.foundry.model_evolution.lineage import ModelLineage
from echozero.foundry.services.dataset_service import DatasetService


@dataclass(frozen=True, slots=True)
class EvolutionTrainingProfile:
    """Named model evolution training profile."""

    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    trainer_profile: str
    optimizer: str
    seed: int = 42
    deterministic: bool = True
    regularization_alpha: float = 0.00005
    weight_decay: float = 0.0001
    average_weights: bool = True
    early_stopping_patience: int = 4
    min_epochs: int = 4
    augment_train: bool = True
    augment_noise_std: float = 0.03
    augment_gain_jitter: float = 0.15
    augment_copies: int = 2
    synthetic_mix_ratio: float = 0.35
    synthetic_mix_cap: int = 400

    @classmethod
    def named(cls, name: str) -> "EvolutionTrainingProfile":
        """Return a bounded explicit profile by name."""
        normalized = name.strip().lower()
        if normalized == "quick_check":
            return cls(
                name="quick_check",
                epochs=2,
                batch_size=4,
                learning_rate=0.001,
                trainer_profile="stronger_v1",
                optimizer="adamw",
                early_stopping_patience=2,
                min_epochs=1,
                augment_copies=1,
                synthetic_mix_ratio=0.0,
                synthetic_mix_cap=0,
            )
        if normalized == "release_candidate":
            return cls(
                name="release_candidate",
                epochs=18,
                batch_size=4,
                learning_rate=0.001,
                trainer_profile="stronger_v1",
                optimizer="adamw",
                early_stopping_patience=5,
                min_epochs=6,
                synthetic_mix_ratio=0.35,
                synthetic_mix_cap=600,
            )
        if normalized in {"beefy", "client_beefy"}:
            return cls(
                name="beefy",
                epochs=12,
                batch_size=4,
                learning_rate=0.001,
                trainer_profile="stronger_v1",
                optimizer="adamw",
            )
        raise ValueError(
            "Unsupported model evolution profile. "
            "Use one of: quick_check, beefy, release_candidate."
        )


@dataclass(frozen=True, slots=True)
class CandidateModelPlan:
    """Planned one-vs-rest candidate model run."""

    label: str
    identity: str
    dataset_version_id: str
    source_dataset_version_id: str
    lineage: ModelLineage
    profile: EvolutionTrainingProfile
    run_spec: dict[str, object]
    positive_count: int
    negative_count: int
    negative_source_counts: dict[str, int]


class ModelEvolutionPlanner:
    """Builds binary drum candidate datasets and run specs."""

    def __init__(self, *, dataset_service: DatasetService) -> None:
        self._datasets = dataset_service

    def plan_binary_drum_candidates(
        self,
        source_version: DatasetVersion,
        *,
        labels: tuple[str, ...],
        identity: str,
        profile: EvolutionTrainingProfile,
        lineage_by_label: dict[str, ModelLineage],
    ) -> tuple[CandidateModelPlan, ...]:
        """Build derived datasets and candidate run specs for requested labels."""
        selected = self._resolve_labels(labels, available_labels=tuple(source_version.class_map))
        plans: list[CandidateModelPlan] = []
        for label in selected:
            derived = self._datasets.derive_binary_dataset_version(
                source_version.id,
                positive_label=label,
            )
            counts = Counter(sample.label for sample in derived.samples)
            negative_source_counts = self.negative_source_counts(
                source_version,
                positive_label=label,
            )
            lineage = lineage_by_label.get(label) or ModelLineage(label=label, kind="from_scratch")
            plans.append(
                CandidateModelPlan(
                    label=label,
                    identity=identity,
                    dataset_version_id=derived.id,
                    source_dataset_version_id=source_version.id,
                    lineage=lineage,
                    profile=profile,
                    run_spec=self.build_binary_drum_run_spec(
                        derived,
                        label=label,
                        identity=identity,
                        profile=profile,
                        lineage=lineage,
                        source_version=source_version,
                        negative_source_counts=negative_source_counts,
                    ),
                    positive_count=int(counts.get(label, 0)),
                    negative_count=int(counts.get("other", 0)),
                    negative_source_counts=negative_source_counts,
                )
            )
        return tuple(plans)

    @staticmethod
    def build_binary_drum_run_spec(
        version: DatasetVersion,
        *,
        label: str,
        identity: str,
        profile: EvolutionTrainingProfile,
        lineage: ModelLineage,
        source_version: DatasetVersion,
        negative_source_counts: dict[str, int],
    ) -> dict[str, object]:
        """Build a CRNN binary drum run spec with explicit evolution metadata."""
        model: dict[str, object] = {"type": "crnn"}
        if lineage.initial_model_path is not None:
            model["initialWeightsPath"] = str(lineage.initial_model_path)
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
                "epochs": profile.epochs,
                "batchSize": profile.batch_size,
                "learningRate": profile.learning_rate,
                "seed": profile.seed,
                "deterministic": profile.deterministic,
                "trainerProfile": profile.trainer_profile,
                "optimizer": profile.optimizer,
                "regularizationAlpha": profile.regularization_alpha,
                "weightDecay": profile.weight_decay,
                "averageWeights": profile.average_weights,
                "earlyStoppingPatience": profile.early_stopping_patience,
                "minEpochs": profile.min_epochs,
                "classWeighting": "balanced",
                "rebalanceStrategy": "oversample",
                "augmentTrain": profile.augment_train,
                "augmentNoiseStd": profile.augment_noise_std,
                "augmentGainJitter": profile.augment_gain_jitter,
                "augmentCopies": profile.augment_copies,
                "syntheticMix": {
                    "enabled": profile.synthetic_mix_ratio > 0.0,
                    "ratio": profile.synthetic_mix_ratio,
                    "cap": profile.synthetic_mix_cap,
                },
                "profileName": profile.name,
            },
            "evolution": {
                "schema": "foundry.model_evolution_run.v1",
                "targetIdentity": identity,
                "label": label,
                "trainingProfile": profile.name,
                "sourceDatasetVersionId": source_version.id,
                "candidateDatasetVersionId": version.id,
                "lineage": lineage.to_payload(),
                "positiveCount": sum(1 for sample in version.samples if sample.label == label),
                "negativeCount": sum(1 for sample in version.samples if sample.label == "other"),
                "negativeSourceCounts": dict(sorted(negative_source_counts.items())),
            },
        }

    @staticmethod
    def negative_source_counts(
        source_version: DatasetVersion,
        *,
        positive_label: str,
    ) -> dict[str, int]:
        """Count all one-vs-rest negative source labels for a target class."""
        normalized_positive = positive_label.strip().lower()
        counts = Counter(
            sample.label.strip().lower()
            for sample in source_version.samples
            if sample.label.strip().lower() != normalized_positive
        )
        return dict(sorted(counts.items()))

    @staticmethod
    def _resolve_labels(
        labels: tuple[str, ...],
        *,
        available_labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        available = {
            str(raw_label).strip().lower()
            for raw_label in available_labels
            if str(raw_label).strip()
        }
        selected: list[str] = []
        seen: set[str] = set()
        for raw_label in labels:
            label = str(raw_label).strip().lower()
            if not label or label in seen:
                continue
            if label not in available:
                supported = ", ".join(sorted(available)) or "(none)"
                raise ValueError(
                    f"Unsupported model evolution label '{label}'. "
                    f"Available labels: {supported}."
                )
            selected.append(label)
            seen.add(label)
        if not selected:
            raise ValueError("At least one model evolution label is required.")
        return tuple(selected)

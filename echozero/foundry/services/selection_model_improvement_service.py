"""
SelectionModelImprovementService trains one candidate model from reviewed EZ selections.
Exists because operators need one simple "improve from selection" flow without hand-driving Foundry.
Connects selected review signals, bounded binary datasets, and candidate runs on the local machine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from echozero.foundry import FoundryApp
from echozero.foundry.domain import DatasetSample, ModelArtifact, TrainRunStatus
from echozero.foundry.domain.review import ReviewCommitContext, ReviewSignal
from echozero.foundry.persistence.review_signal_repository import ReviewSignalRepository
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles

_NEGATIVE_LABEL = "other"


@dataclass(frozen=True, slots=True)
class ImproveModelBaseOption:
    """One operator-facing base-model choice for comparison during candidate training."""

    option_id: str
    label: str
    artifact_id: str | None = None
    is_current: bool = False


@dataclass(frozen=True, slots=True)
class ImproveModelTrainingRequest:
    """One bounded local-training request from selected reviewed examples."""

    target_label: str
    selected_signal_ids: tuple[str, ...]
    candidate_name: str
    scope_mode: str = "song_layer"
    strength: str = "balanced"
    include_related_examples: bool = True
    base_model_option_id: str | None = None


@dataclass(frozen=True, slots=True)
class ImproveModelTrainingResult:
    """Compact outcome for one candidate model trained from selection."""

    target_label: str
    candidate_name: str
    scope_mode: str
    strength: str
    selected_signal_count: int
    anchor_sample_count: int
    related_sample_count: int
    dataset_id: str
    dataset_version_id: str
    run_id: str
    artifact_id: str
    base_artifact_id: str | None
    compared_to_base_model: bool


@dataclass(frozen=True, slots=True)
class _StrengthPreset:
    anchor_multiplier: int
    max_related_per_class: int
    epochs: int
    learning_rate: float
    regularization_alpha: float
    average_weights: bool


class SelectionModelImprovementService:
    """Build and train one binary candidate model from selected review signals."""

    _strength_presets = {
        "light": _StrengthPreset(
            anchor_multiplier=2,
            max_related_per_class=24,
            epochs=4,
            learning_rate=0.005,
            regularization_alpha=0.0002,
            average_weights=False,
        ),
        "balanced": _StrengthPreset(
            anchor_multiplier=4,
            max_related_per_class=16,
            epochs=6,
            learning_rate=0.008,
            regularization_alpha=0.00012,
            average_weights=True,
        ),
        "strong": _StrengthPreset(
            anchor_multiplier=6,
            max_related_per_class=8,
            epochs=8,
            learning_rate=0.01,
            regularization_alpha=0.00008,
            average_weights=True,
        ),
    }
    _supported_scope_modes = frozenset({"selected_events", "song_layer", "song", "project"})

    def __init__(
        self,
        root: Path,
        *,
        foundry_app_factory: Callable[[Path], FoundryApp] = FoundryApp,
        review_signal_repository: ReviewSignalRepository | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._foundry_app_factory = foundry_app_factory
        self._review_signals = review_signal_repository or ReviewSignalRepository(self._root)

    def list_base_model_options(self, *, target_label: str) -> tuple[ImproveModelBaseOption, ...]:
        """Return current installed and local artifact options for one binary target label."""
        normalized_label = str(target_label).strip().lower()
        if not normalized_label:
            return ()
        app = self._foundry_app_factory(self._root)
        matching_artifacts = [
            artifact
            for artifact in app.list_artifacts()
            if _artifact_matches_binary_label(artifact, target_label=normalized_label)
        ]
        matching_artifacts.sort(key=lambda artifact: artifact.created_at, reverse=True)
        artifact_by_id = {artifact.id: artifact for artifact in matching_artifacts}
        options: list[ImproveModelBaseOption] = []
        current_artifact = _resolve_current_artifact(
            target_label=normalized_label,
            artifacts=matching_artifacts,
        )
        seen_artifact_ids: set[str] = set()
        if current_artifact is not None:
            options.append(
                ImproveModelBaseOption(
                    option_id=f"artifact:{current_artifact.id}",
                    label=f"Current installed {normalized_label} model",
                    artifact_id=current_artifact.id,
                    is_current=True,
                )
            )
            seen_artifact_ids.add(current_artifact.id)
        for artifact in matching_artifacts:
            if artifact.id in seen_artifact_ids:
                continue
            options.append(
                ImproveModelBaseOption(
                    option_id=f"artifact:{artifact.id}",
                    label=_artifact_option_label(artifact),
                    artifact_id=artifact.id,
                    is_current=False,
                )
            )
            seen_artifact_ids.add(artifact.id)
        return tuple(options)

    def train_candidate_model(
        self,
        request: ImproveModelTrainingRequest,
    ) -> ImproveModelTrainingResult:
        """Materialize one bounded binary dataset and train a local candidate artifact."""
        preset = self._resolve_strength_preset(request.strength)
        scope_mode = self._resolve_scope_mode(request.scope_mode)
        selected_signals = self._load_selected_signals(request.selected_signal_ids)
        normalized_label = str(request.target_label).strip().lower()
        if not normalized_label:
            raise ValueError("Improve Model requires a non-empty target label.")
        _require_selection_targets(selected_signals, target_label=normalized_label)

        app = self._foundry_app_factory(self._root)
        anchor_samples = self._build_anchor_samples(
            app,
            selected_signals=selected_signals,
            target_label=normalized_label,
            anchor_multiplier=preset.anchor_multiplier,
        )
        if not anchor_samples:
            raise ValueError("The selected reviewed events did not produce any training samples.")
        related_samples = self._build_related_samples(
            app,
            selected_signals=selected_signals,
            target_label=normalized_label,
            max_related_per_class=preset.max_related_per_class,
            include_related_examples=request.include_related_examples,
            anchor_samples=anchor_samples,
            scope_mode=scope_mode,
        )
        dataset_samples = [*anchor_samples, *related_samples]
        class_labels = {sample.label for sample in dataset_samples}
        if class_labels != {normalized_label, _NEGATIVE_LABEL}:
            raise ValueError(
                "Improve Model requires both positive and negative reviewed signal. "
                "Select at least one reviewed positive or enable related examples."
            )

        dataset = app.datasets.create_dataset(
            f"{request.candidate_name} Selection Dataset",
            source_kind="selection_model_improvement",
            source_ref=str(self._root),
            metadata={
                "schema": "foundry.selection_model_improvement_dataset.v1",
                "candidate_name": request.candidate_name,
                "target_label": normalized_label,
                "scope_mode": scope_mode,
                "strength": request.strength,
                "selected_signal_ids": list(request.selected_signal_ids),
                "include_related_examples": bool(request.include_related_examples),
            },
        )
        version = app.datasets.create_version_from_samples(
            dataset.id,
            samples=dataset_samples,
            taxonomy=_binary_taxonomy(normalized_label),
            label_policy=_binary_label_policy(normalized_label),
            manifest={
                "schema": "foundry.selection_model_improvement_manifest.v1",
                "candidate_name": request.candidate_name,
                "target_label": normalized_label,
                "scope_mode": scope_mode,
                "selected_signal_ids": list(request.selected_signal_ids),
                "deterministic_order": [sample.sample_id for sample in dataset_samples],
                "content_hash_algorithm": "sha256",
            },
            stats={
                "sample_count": len(dataset_samples),
                "anchor_sample_count": len(anchor_samples),
                "related_sample_count": len(related_samples),
                "class_counts": {
                    normalized_label: sum(1 for sample in dataset_samples if sample.label == normalized_label),
                    _NEGATIVE_LABEL: sum(1 for sample in dataset_samples if sample.label == _NEGATIVE_LABEL),
                },
            },
            lineage={
                "kind": "selection_model_improvement",
                "target_label": normalized_label,
                "scope_mode": scope_mode,
                "selected_signal_ids": list(request.selected_signal_ids),
            },
        )
        app.plan_version(
            version.id,
            validation_split=0.15,
            test_split=0.10,
            seed=42,
            balance_strategy="hybrid",
        )
        base_artifact_id = _resolve_base_artifact_id(
            option_id=request.base_model_option_id,
            options=self.list_base_model_options(target_label=normalized_label),
        )
        run = app.create_run(
            version.id,
            self._build_run_spec(
                dataset_version_id=version.id,
                sample_rate=version.sample_rate,
                target_label=normalized_label,
                preset=preset,
                reference_artifact_id=base_artifact_id,
            ),
        )
        completed_run = app.start_run(run.id)
        if completed_run.status is not TrainRunStatus.COMPLETED:
            raise RuntimeError(
                f"Improve Model candidate run did not complete successfully: {completed_run.status.value}"
            )
        artifacts = app.list_artifacts_for_run(completed_run.id)
        if not artifacts:
            raise RuntimeError(f"No candidate artifact was produced for run '{completed_run.id}'.")
        artifacts.sort(key=lambda artifact: artifact.created_at)
        artifact = artifacts[-1]
        compatibility = app.validate_artifact(artifact.id)
        if not compatibility.ok:
            raise RuntimeError(
                f"Candidate artifact '{artifact.id}' failed validation: {compatibility.errors[0]}"
            )
        return ImproveModelTrainingResult(
            target_label=normalized_label,
            candidate_name=request.candidate_name,
            scope_mode=scope_mode,
            strength=request.strength,
            selected_signal_count=len(selected_signals),
            anchor_sample_count=len(anchor_samples),
            related_sample_count=len(related_samples),
            dataset_id=dataset.id,
            dataset_version_id=version.id,
            run_id=completed_run.id,
            artifact_id=artifact.id,
            base_artifact_id=base_artifact_id,
            compared_to_base_model=base_artifact_id is not None,
        )

    def _build_anchor_samples(
        self,
        app: FoundryApp,
        *,
        selected_signals: list[ReviewSignal],
        target_label: str,
        anchor_multiplier: int,
    ) -> list[DatasetSample]:
        anchor_samples: list[DatasetSample] = []
        for signal in selected_signals:
            materialization = app.datasets.materialize_review_signal(self._selection_session(signal), signal)
            if materialization.get("status") != "materialized":
                continue
            version_id = str(materialization.get("version_id", "")).strip()
            version = app.datasets.get_version(version_id)
            if version is None:
                continue
            selected_sample_ids = {
                str(sample_id)
                for sample_id in materialization.get("materialized_signal_samples", [])
                if str(sample_id).strip()
            }
            for sample in version.samples:
                if sample.sample_id not in selected_sample_ids:
                    continue
                binary_label = _binary_sample_label(sample=sample, target_label=target_label)
                for index in range(anchor_multiplier):
                    anchor_samples.append(
                        _clone_sample(
                            sample,
                            new_label=binary_label,
                            sample_prefix="sel",
                            suffix=f"{signal.id}_{index}",
                            extra_quality_flags=["selected_anchor"],
                            extra_provenance={"selection_anchor": True, "selection_signal_id": signal.id},
                        )
                    )
        return anchor_samples

    def _build_related_samples(
        self,
        app: FoundryApp,
        *,
        selected_signals: list[ReviewSignal],
        target_label: str,
        max_related_per_class: int,
        include_related_examples: bool,
        anchor_samples: list[DatasetSample],
        scope_mode: str,
    ) -> list[DatasetSample]:
        if not include_related_examples or scope_mode == "selected_events":
            return []
        project_ref = _single_provenance_value(selected_signals, "project_ref")
        if project_ref is None:
            return []
        song_id, song_version_id, layer_id = _scope_filters(
            selected_signals,
            scope_mode=scope_mode,
        )
        review_version = app.extract_project_review_dataset(
            self._root,
            project_ref=project_ref,
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
            queue_source_kind="timeline_review_mode",
        )
        derived_version = app.datasets.derive_binary_dataset_version(
            review_version.id,
            positive_label=target_label,
            negative_label=_NEGATIVE_LABEL,
        )
        anchor_hashes = {sample.content_hash for sample in anchor_samples if sample.content_hash}
        related_positive = [
            sample
            for sample in derived_version.samples
            if sample.label == target_label and sample.content_hash not in anchor_hashes
        ]
        related_negative = [
            sample
            for sample in derived_version.samples
            if sample.label == _NEGATIVE_LABEL and sample.content_hash not in anchor_hashes
        ]
        related_samples: list[DatasetSample] = []
        for label, pool in ((target_label, related_positive), (_NEGATIVE_LABEL, related_negative)):
            for index, sample in enumerate(pool[:max_related_per_class]):
                related_samples.append(
                    _clone_sample(
                        sample,
                        new_label=label,
                        sample_prefix="rel",
                        suffix=f"{label}_{index}",
                        extra_quality_flags=["related_context"],
                        extra_provenance={"selection_related": True},
                    )
                )
        return related_samples

    def _build_run_spec(
        self,
        *,
        dataset_version_id: str,
        sample_rate: int,
        target_label: str,
        preset: _StrengthPreset,
        reference_artifact_id: str | None,
    ) -> dict[str, object]:
        spec: dict[str, object] = {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "binary",
            "data": {
                "datasetVersionId": dataset_version_id,
                "sampleRate": sample_rate,
                "maxLength": sample_rate,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "training": {
                "epochs": preset.epochs,
                "batchSize": 4,
                "learningRate": preset.learning_rate,
                "seed": 42,
                "classWeighting": "balanced",
                "rebalanceStrategy": "oversample",
                "augmentTrain": True,
                "augmentNoiseStd": 0.03,
                "augmentGainJitter": 0.12,
                "augmentCopies": 2,
                "trainerProfile": "baseline_v1",
                "optimizer": "sgd_constant",
                "regularizationAlpha": preset.regularization_alpha,
                "averageWeights": preset.average_weights,
            },
            "metadata": {
                "targetLabel": target_label,
                "improveModelFlow": True,
            },
        }
        if reference_artifact_id:
            spec["promotion"] = {"reference_artifact_id": reference_artifact_id}
        return spec

    def _load_selected_signals(self, signal_ids: tuple[str, ...]) -> list[ReviewSignal]:
        selected_signals: list[ReviewSignal] = []
        seen: set[str] = set()
        for raw_signal_id in signal_ids:
            signal_id = str(raw_signal_id).strip()
            if not signal_id or signal_id in seen:
                continue
            signal = self._review_signals.get(signal_id)
            if signal is None:
                raise ValueError(f"Selected review signal not found: {signal_id}")
            selected_signals.append(signal)
            seen.add(signal_id)
        if not selected_signals:
            raise ValueError("Select at least one reviewed event to improve a model.")
        return selected_signals

    @classmethod
    def _resolve_strength_preset(cls, strength: str) -> _StrengthPreset:
        normalized = str(strength).strip().lower() or "balanced"
        try:
            return cls._strength_presets[normalized]
        except KeyError as exc:
            supported = ", ".join(sorted(cls._strength_presets))
            raise ValueError(
                f"Unsupported Improve Model strength '{strength}'. Supported values: {supported}."
            ) from exc

    @classmethod
    def _resolve_scope_mode(cls, scope_mode: str) -> str:
        normalized = str(scope_mode).strip().lower() or "song_layer"
        if normalized not in cls._supported_scope_modes:
            supported = ", ".join(sorted(cls._supported_scope_modes))
            raise ValueError(
                f"Unsupported Improve Model scope '{scope_mode}'. Supported values: {supported}."
            )
        return normalized

    def _selection_session(self, signal: ReviewSignal):
        reviewed_at = signal.reviewed_at or datetime.now(UTC)
        queue_source_kind = "manual_review"
        source_kind = str(signal.source_provenance.get("kind", "")).strip()
        if source_kind.startswith("ez_timeline_fix"):
            queue_source_kind = "timeline_fix_mode"
        elif source_kind.startswith("ez_timeline_review"):
            queue_source_kind = "timeline_review_mode"
        context = ReviewCommitContext(
            session_id=signal.session_id,
            session_name=f"Improve Model Selection - {signal.session_id}",
            source_ref=str(self._root),
            metadata={"queue_source_kind": queue_source_kind},
        )
        return context.as_review_session(reviewed_at=reviewed_at)


def _artifact_matches_binary_label(artifact: ModelArtifact, *, target_label: str) -> bool:
    classes = artifact.manifest.get("classes")
    if not isinstance(classes, list):
        return False
    normalized = [str(value).strip().lower() for value in classes]
    return len(normalized) == 2 and normalized == [target_label, _NEGATIVE_LABEL]


def _artifact_option_label(artifact: ModelArtifact) -> str:
    created = artifact.created_at.strftime("%Y-%m-%d")
    return f"Local candidate {artifact.id} ({created})"


def _resolve_current_artifact(
    *,
    target_label: str,
    artifacts: list[ModelArtifact],
) -> ModelArtifact | None:
    try:
        current_bundle = resolve_installed_binary_drum_bundles(labels=(target_label,))[target_label]
    except Exception:
        return None
    current_manifest_path = current_bundle.manifest_path.resolve()
    for artifact in artifacts:
        source_manifest = artifact.manifest.get("sourceManifestPath")
        artifact_path = artifact.path.resolve()
        if artifact_path == current_manifest_path:
            return artifact
        if isinstance(source_manifest, str) and Path(source_manifest).expanduser().resolve() == current_manifest_path:
            return artifact
    return None


def _resolve_base_artifact_id(
    *,
    option_id: str | None,
    options: tuple[ImproveModelBaseOption, ...],
) -> str | None:
    if option_id is None:
        return None
    return next(
        (
            option.artifact_id
            for option in options
            if option.option_id == option_id and option.artifact_id is not None
        ),
        None,
    )


def _require_selection_targets(signals: list[ReviewSignal], *, target_label: str) -> None:
    selection_targets = {
        _signal_target_label(signal)
        for signal in signals
    }
    if selection_targets != {target_label}:
        joined = ", ".join(sorted(selection_targets))
        raise ValueError(
            "Improve Model V1 requires the selected reviewed events to resolve to one target label. "
            f"Found: {joined}"
        )


def _signal_target_label(signal: ReviewSignal) -> str:
    for candidate in (signal.target_class, signal.corrected_label, signal.predicted_label):
        text = str(candidate).strip().lower()
        if text:
            return text
    raise ValueError(f"Review signal '{signal.id}' is missing a usable target label.")


def _single_provenance_value(signals: list[ReviewSignal], key: str) -> str | None:
    values = {
        str(signal.source_provenance.get(key, "")).strip()
        for signal in signals
        if str(signal.source_provenance.get(key, "")).strip()
    }
    if len(values) != 1:
        return None
    return next(iter(values))


def _single_ref_id(signals: list[ReviewSignal], key: str) -> str | None:
    value = _single_provenance_value(signals, key)
    if value is None or ":" not in value:
        return None
    _prefix, raw_id = value.split(":", 1)
    text = raw_id.strip()
    return text or None


def _scope_filters(
    signals: list[ReviewSignal],
    *,
    scope_mode: str,
) -> tuple[str | None, str | None, str | None]:
    if scope_mode == "project":
        return None, None, None
    if scope_mode == "song":
        song_id = _require_single_ref_id(signals, "song_ref", scope_label="song")
        return song_id, None, None
    if scope_mode == "song_layer":
        song_id = _require_single_ref_id(signals, "song_ref", scope_label="song/layer")
        layer_id = _require_single_ref_id(signals, "layer_ref", scope_label="song/layer")
        song_version_id = _single_ref_id(signals, "version_ref")
        return song_id, song_version_id, layer_id
    return None, None, None


def _require_single_ref_id(
    signals: list[ReviewSignal],
    key: str,
    *,
    scope_label: str,
) -> str:
    values = {
        _single_ref_id([signal], key)
        for signal in signals
    }
    normalized = {value for value in values if value is not None}
    if len(normalized) != 1:
        raise ValueError(
            f"Improve Model scope '{scope_label}' requires the selected reviewed events "
            f"to belong to one shared {key.removesuffix('_ref')}."
        )
    return next(iter(normalized))


def _binary_taxonomy(target_label: str) -> dict[str, object]:
    return {
        "schema": "foundry.taxonomy.v1",
        "namespace": "percussion.one_shot",
        "version": 1,
        "labels": [
            {"id": target_label, "display_name": target_label.replace("_", " "), "aliases": []},
            {"id": _NEGATIVE_LABEL, "display_name": "other", "aliases": []},
        ],
    }


def _binary_label_policy(target_label: str) -> dict[str, object]:
    return {
        "schema": "foundry.label_policy.v1",
        "classification_mode": "binary",
        "unit": "one_shot",
        "allowed_labels": [target_label, _NEGATIVE_LABEL],
        "unknown_label": None,
    }


def _binary_sample_label(*, sample: DatasetSample, target_label: str) -> str:
    polarity = str(sample.source_provenance.get("review_polarity", "")).strip().lower()
    if polarity == "negative":
        return _NEGATIVE_LABEL
    sample_label = str(sample.label).strip().lower()
    return target_label if sample_label == target_label else _NEGATIVE_LABEL


def _clone_sample(
    sample: DatasetSample,
    *,
    new_label: str,
    sample_prefix: str,
    suffix: str,
    extra_quality_flags: list[str],
    extra_provenance: dict[str, object],
) -> DatasetSample:
    sample_id = f"{sample_prefix}_{uuid4().hex[:12]}"
    return replace(
        sample,
        sample_id=sample_id,
        label=new_label,
        quality_flags=[*sample.quality_flags, *extra_quality_flags],
        source_provenance={
            **dict(sample.source_provenance),
            **extra_provenance,
            "selection_clone_suffix": suffix,
        },
    )

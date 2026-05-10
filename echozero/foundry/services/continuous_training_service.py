"""Continuous local training kickoff over the sample library.
Exists to hide Foundry dataset/run plumbing behind one small library-first entrypoint.
Connects approved local samples to dataset-version materialization and immediate runs.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import (
    CurationState,
    DatasetSample,
    DatasetVersion,
    LibrarySampleState,
    SampleLibraryRecord,
)
from echozero.foundry.domain.entities import TrainRun
from echozero.foundry.services.dataset_service import DatasetService
from echozero.foundry.services.sample_library_service import SampleLibraryService
from echozero.foundry.services.split_balance_service import SplitBalanceService
from echozero.foundry.services.training_orchestrator import TrainingOrchestrator


class ContinuousTrainingService:
    """Materialize approved library samples into a dataset version and run it."""

    def __init__(
        self,
        root: Path,
        *,
        dataset_service: DatasetService,
        library_service: SampleLibraryService,
        orchestrator: TrainingOrchestrator,
        split_balance_service: SplitBalanceService | None = None,
    ) -> None:
        self._root = root
        self._dataset_service = dataset_service
        self._library_service = library_service
        self._orchestrator = orchestrator
        self._split_balance = split_balance_service or SplitBalanceService()

    def create_dataset_version(
        self,
        *,
        name: str,
        scope: str = "local.default",
    ) -> DatasetVersion:
        """Build one dataset version from the approved sample library."""
        approved = self._library_service.list_samples_by_state(LibrarySampleState.APPROVED)
        if not approved:
            raise ValueError("No approved library samples are available for training")
        dataset = self._dataset_service.create_dataset(
            f"{name} Training Library",
            source_kind="sample_library",
            source_ref=scope,
            metadata={"scope": scope, "approved_sample_count": len(approved)},
        )
        dataset_samples = [self._build_dataset_sample(record) for record in approved]
        version = self._dataset_service.create_version_from_samples(
            dataset.id,
            samples=dataset_samples,
            manifest={
                "schema": "foundry.sample_library_dataset_manifest.v1",
                "source_kind": "sample_library",
                "scope": scope,
                "deterministic_order": [sample.sample_id for sample in dataset_samples],
                "content_hash_algorithm": "sha256",
                "content_groups": {
                    sample.content_hash: [sample.sample_id]
                    for sample in dataset_samples
                    if sample.content_hash
                },
            },
            lineage={"kind": "sample_library", "scope": scope},
        )
        split_plan = self._split_balance.plan_splits(
            version,
            validation_split=0.15,
            test_split=0.10,
            seed=42,
        )
        balance_plan = self._split_balance.plan_balance(version, strategy="hybrid")
        version = self._dataset_service.update_version_plans(
            version.id,
            split_plan=split_plan,
            balance_plan=balance_plan,
        )
        return version

    def kickoff_run(
        self,
        *,
        name: str,
        epochs: int = 4,
        scope: str = "local.default",
    ) -> TrainRun:
        """Create a dataset version from approved library samples and start a run."""
        version = self.create_dataset_version(name=name, scope=scope)
        run_spec = self._build_run_spec(version.id, sample_rate=version.sample_rate, epochs=epochs)
        run = self._orchestrator.create_run(version.id, run_spec)
        return self._orchestrator.start_run(run.id)

    @staticmethod
    def _build_dataset_sample(record: SampleLibraryRecord) -> DatasetSample:
        return DatasetSample(
            sample_id=f"sm_{uuid4().hex[:12]}",
            audio_ref=record.audio_ref,
            label=record.label,
            duration_ms=record.duration_ms,
            content_hash=record.content_hash,
            source_provenance=dict(record.provenance),
            group_id=f"library:{record.id}",
            is_synthetic=False,
            synthetic_provenance={},
            quality_flags=list(record.quality_flags),
            split_assignment=None,
            curation_state=CurationState.ACCEPTED,
        )

    @staticmethod
    def _build_run_spec(dataset_version_id: str, *, sample_rate: int, epochs: int) -> dict[str, object]:
        return {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "multiclass",
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
                "epochs": epochs,
                "batchSize": 4,
                "learningRate": 0.01,
                "seed": 42,
                "trainerProfile": "baseline_v1",
                "optimizer": "sgd_constant",
                "classWeighting": "balanced",
                "rebalanceStrategy": "oversample",
                "augmentTrain": True,
                "augmentNoiseStd": 0.03,
                "augmentGainJitter": 0.15,
                "augmentCopies": 2,
            },
        }

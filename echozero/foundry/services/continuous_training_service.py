"""Continuous local training orchestration over the sample library.
Exists to let operators train challengers from a growing library instead of raw Foundry primitives.
Connects library capture, snapshot creation, recipe presets, and current Foundry training services.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from echozero.foundry.domain import (
    CurationState,
    DatasetSample,
    LibrarySampleState,
    ModelCandidateRecord,
    SampleLibraryRecord,
    TrainingRecipeName,
)
from echozero.foundry.persistence import (
    EvalReportRepository,
    ModelArtifactRepository,
    ModelCandidateRepository,
)
from echozero.foundry.services.champion_service import ChampionService
from echozero.foundry.services.dataset_service import DatasetService
from echozero.foundry.services.sample_library_service import SampleLibraryService
from echozero.foundry.services.snapshot_service import SnapshotService
from echozero.foundry.services.split_balance_service import SplitBalanceService
from echozero.foundry.services.training_orchestrator import TrainingOrchestrator
from echozero.foundry.services.training_recipe_service import TrainingRecipeService


class ContinuousTrainingService:
    """Train local challengers from the sample library with simple recipes."""

    def __init__(
        self,
        root: Path,
        *,
        dataset_service: DatasetService,
        library_service: SampleLibraryService,
        snapshot_service: SnapshotService,
        recipe_service: TrainingRecipeService | None = None,
        orchestrator: TrainingOrchestrator,
        split_balance_service: SplitBalanceService | None = None,
        candidate_repository: ModelCandidateRepository | None = None,
        artifact_repository: ModelArtifactRepository | None = None,
        eval_repository: EvalReportRepository | None = None,
        champion_service: ChampionService | None = None,
    ) -> None:
        self._root = root
        self._dataset_service = dataset_service
        self._library_service = library_service
        self._snapshot_service = snapshot_service
        self._recipe_service = recipe_service or TrainingRecipeService()
        self._orchestrator = orchestrator
        self._split_balance = split_balance_service or SplitBalanceService()
        self._candidate_repository = candidate_repository or ModelCandidateRepository(root)
        self._artifact_repository = artifact_repository or ModelArtifactRepository(root)
        self._eval_repository = eval_repository or EvalReportRepository(root)
        self._champion_service = champion_service or ChampionService(root)

    def train_challenger(
        self,
        *,
        name: str,
        recipe_name: TrainingRecipeName = TrainingRecipeName.BALANCED,
        scope: str = "local.default",
    ) -> ModelCandidateRecord:
        """Create a snapshot, train one challenger, and persist its metadata."""
        approved = self._library_service.list_samples_by_state(LibrarySampleState.APPROVED)
        if not approved:
            raise ValueError("No approved library samples are available for training")
        snapshot = self._snapshot_service.create_snapshot(
            name=f"{name} Snapshot",
            samples=approved,
            provenance={"scope": scope, "recipe": recipe_name.value},
            filters={"state": LibrarySampleState.APPROVED.value},
        )
        dataset = self._dataset_service.create_dataset(
            f"{name} Training Library",
            source_kind="sample_library_snapshot",
            source_ref=snapshot.id,
            metadata={"scope": scope, "snapshot_id": snapshot.id},
        )
        dataset_samples = [self._build_dataset_sample(record) for record in approved]
        version = self._dataset_service.create_version_from_samples(
            dataset.id,
            samples=dataset_samples,
            manifest={
                "schema": "foundry.library_snapshot_dataset_manifest.v1",
                "snapshot_id": snapshot.id,
                "source_kind": "sample_library_snapshot",
                "deterministic_order": [sample.sample_id for sample in dataset_samples],
                "content_hash_algorithm": "sha256",
                "content_groups": {
                    sample.content_hash: [sample.sample_id]
                    for sample in dataset_samples
                    if sample.content_hash
                },
            },
            lineage={"kind": "sample_library_snapshot", "snapshot_id": snapshot.id, "scope": scope},
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
        run_spec = self._recipe_service.build_run_spec(
            version.id,
            recipe_name=recipe_name,
            sample_rate=version.sample_rate,
        )
        run = self._orchestrator.create_run(version.id, run_spec)
        run = self._orchestrator.start_run(run.id)
        artifacts = sorted(self._artifact_repository.list_for_run(run.id), key=lambda item: item.created_at)
        reports = sorted(self._eval_repository.list_for_run(run.id), key=lambda item: item.created_at)
        artifact = artifacts[-1] if artifacts else None
        report = reports[-1] if reports else None
        candidate = ModelCandidateRecord(
            id=f"cand_{uuid4().hex[:12]}",
            snapshot_id=snapshot.id,
            recipe_name=recipe_name.value,
            dataset_id=dataset.id,
            dataset_version_id=version.id,
            run_id=run.id,
            artifact_id=artifact.id if artifact else None,
            eval_report_id=report.id if report else None,
            status=run.status.value,
            metrics=dict(report.aggregate_metrics if report else {}),
            comparison=self._build_comparison(scope=scope, report=report),
        )
        return self._candidate_repository.save(candidate)

    def promote_candidate(
        self,
        candidate_id: str,
        *,
        scope: str = "local.default",
    ):
        """Promote a completed candidate into the active local champion slot."""
        candidate = self._candidate_repository.get(candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate not found: {candidate_id}")
        return self._champion_service.promote_candidate(
            candidate,
            scope=scope,
            notes=candidate.comparison,
        )

    def list_candidates(self) -> list[ModelCandidateRecord]:
        """Return persisted challenger records."""
        return self._candidate_repository.list()

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

    def _build_comparison(self, *, scope: str, report) -> dict[str, object]:
        current = self._champion_service.get_champion(scope)
        comparison: dict[str, object] = {"scope": scope, "has_champion": current is not None}
        if report is None:
            return comparison
        comparison["candidate_macro_f1"] = report.aggregate_metrics.get("macro_f1")
        comparison["candidate_accuracy"] = report.aggregate_metrics.get("accuracy")
        return comparison

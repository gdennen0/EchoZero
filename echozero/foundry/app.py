from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

from echozero.domain.events import (
    FoundryArtifactFinalizedEvent,
    FoundryArtifactValidatedEvent,
    FoundryRunCreatedEvent,
    FoundryRunStartedEvent,
    create_event_id,
)
from echozero.event_bus import EventBus
from echozero.foundry.domain import CompatibilityReport, Dataset, EvalReport, ModelArtifact, TrainRun
from echozero.foundry.persistence import (
    DatasetRepository,
    DatasetVersionRepository,
    EvalReportRepository,
    ModelArtifactRepository,
    ReviewSessionRepository,
)
from echozero.foundry.presentation import FoundryActivityFeed
from echozero.foundry.services.query_service import (
    FoundryQueryService,
    ProjectReviewDatasetVersionRef,
)
from echozero.foundry.services import (
    ArtifactService,
    DatasetService,
    EvalService,
    ReviewSessionService,
    RuntimeBundleInstallService,
    SplitBalanceService,
    TrainRunService,
)


class FoundryApp:
    """Standalone Foundry v1 composition root.

    This app intentionally lives outside Stage Zero UI concerns.
    """

    def __init__(self, root: Path):
        self.root = root
        self.event_bus = EventBus()

        self._dataset_repo = DatasetRepository(root)
        self._dataset_version_repo = DatasetVersionRepository(root)
        self._artifact_repo = ModelArtifactRepository(root)
        self._eval_repo = EvalReportRepository(root)
        self._review_repo = ReviewSessionRepository(root)

        self.datasets = DatasetService(
            root,
            dataset_repo=self._dataset_repo,
            version_repo=self._dataset_version_repo,
        )
        self.split_balance = SplitBalanceService()
        self.eval = EvalService(self._eval_repo)
        self.artifacts = ArtifactService(root, artifact_repository=self._artifact_repo)
        self.reviews = ReviewSessionService(root, repository=self._review_repo)
        self.runtime_bundles = RuntimeBundleInstallService(root, artifact_repository=self._artifact_repo)
        self.runs = TrainRunService(root, eval_service=self.eval, artifact_service=self.artifacts)
        self.queries = FoundryQueryService(
            dataset_repo=self._dataset_repo,
            version_repo=self._dataset_version_repo,
            artifact_repo=self._artifact_repo,
            eval_repo=self._eval_repo,
            run_service=self.runs,
            root=root,
        )

        self.activity = FoundryActivityFeed(self.event_bus)

    def create_run(
        self,
        dataset_version_id: str,
        run_spec: dict[str, Any],
        *,
        backend: str = "pytorch",
        device: str = "cpu",
    ) -> TrainRun:
        run = self.runs.create_run(dataset_version_id, run_spec, backend=backend, device=device)
        self.event_bus.publish(
            FoundryRunCreatedEvent(
                event_id=create_event_id(),
                timestamp=run.created_at.timestamp(),
                correlation_id=run.id,
                run_id=run.id,
                dataset_version_id=run.dataset_version_id,
                status=run.status.value,
            )
        )
        return run

    def start_run(self, run_id: str, *, cancel_event: Event | None = None) -> TrainRun:
        run = self.runs.start_run(run_id, cancel_event=cancel_event)
        self.event_bus.publish(
            FoundryRunStartedEvent(
                event_id=create_event_id(),
                timestamp=run.updated_at.timestamp(),
                correlation_id=run.id,
                run_id=run.id,
                status=run.status.value,
            )
        )
        return run

    def finalize_artifact(self, run_id: str, manifest: dict[str, Any]) -> ModelArtifact:
        artifact = self.artifacts.finalize_artifact(run_id, manifest)
        self.event_bus.publish(
            FoundryArtifactFinalizedEvent(
                event_id=create_event_id(),
                timestamp=artifact.created_at.timestamp(),
                correlation_id=artifact.run_id,
                artifact_id=artifact.id,
                run_id=artifact.run_id,
            )
        )
        return artifact

    def validate_artifact(
        self,
        artifact_id: str,
        *,
        consumer: str = "PyTorchAudioClassify",
    ) -> CompatibilityReport:
        report = self.artifacts.validate_compatibility(artifact_id, consumer=consumer)
        self.event_bus.publish(
            FoundryArtifactValidatedEvent(
                event_id=create_event_id(),
                timestamp=report.checked_at.timestamp(),
                correlation_id=artifact_id,
                artifact_id=artifact_id,
                consumer=consumer,
                ok=report.ok,
                error_count=len(report.errors),
                warning_count=len(report.warnings),
            )
        )
        return report

    def plan_version(
        self,
        version_id: str,
        *,
        validation_split: float = 0.15,
        test_split: float = 0.10,
        seed: int = 42,
        balance_strategy: str = "none",
    ) -> dict[str, Any]:
        version = self.datasets.get_version(version_id)
        if version is None:
            raise ValueError(f"DatasetVersion not found: {version_id}")

        split_plan = self.split_balance.plan_splits(
            version,
            validation_split=validation_split,
            test_split=test_split,
            seed=seed,
        )
        balance_plan = self.split_balance.plan_balance(version, strategy=balance_strategy)

        self.datasets.update_version_plans(
            version.id,
            split_plan=split_plan,
            balance_plan=balance_plan,
        )

        return {
            "version_id": version.id,
            "split_plan": split_plan,
            "balance_plan": balance_plan,
        }

    def export_project_review_dataset(
        self,
        project_path: str | Path,
        *,
        project_ref: str | None = None,
        song_id: str | None = None,
        song_version_id: str | None = None,
        layer_id: str | None = None,
        queue_source_kind: str = "ez_project",
    ) -> DatasetVersion:
        return self.datasets.export_project_review_dataset(
            project_path,
            project_ref=project_ref,
            song_id=song_id,
            song_version_id=song_version_id,
            layer_id=layer_id,
            queue_source_kind=queue_source_kind,
        )

    # ------------------------------------------------------------------
    # Query boundary for Foundry UI
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[Dataset]:
        return self.queries.list_datasets()

    def list_runs(self) -> list[TrainRun]:
        return self.queries.list_runs()

    def list_artifacts(self) -> list[ModelArtifact]:
        return self.queries.list_artifacts()

    def list_artifacts_for_run(self, run_id: str) -> list[ModelArtifact]:
        return self.queries.list_artifacts_for_run(run_id)

    def get_artifact(self, artifact_id: str) -> ModelArtifact | None:
        return self.queries.get_artifact(artifact_id)

    def list_eval_reports_for_run(self, run_id: str) -> list[EvalReport]:
        return self.queries.list_eval_reports_for_run(run_id)

    def list_project_review_dataset_versions(
        self,
        *,
        project_ref: str,
        queue_source_kind: str | None = "ez_project",
    ) -> list[ProjectReviewDatasetVersionRef]:
        return self.queries.list_project_review_dataset_versions(
            project_ref=project_ref,
            queue_source_kind=queue_source_kind,
        )

    def get_latest_project_review_dataset_version(
        self,
        *,
        project_ref: str,
        queue_source_kind: str | None = "ez_project",
    ) -> ProjectReviewDatasetVersionRef | None:
        return self.queries.get_latest_project_review_dataset_version(
            project_ref=project_ref,
            queue_source_kind=queue_source_kind,
        )

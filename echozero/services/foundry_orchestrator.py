"""Foundry orchestration layer: app-facing wrapper around Foundry services.

Keeps Project-facing API Result[T]-based and enforces compatibility gate semantics
for artifact promotion paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from echozero.domain.events import (
    FoundryArtifactFinalizedEvent,
    FoundryArtifactValidatedEvent,
    FoundryRunCreatedEvent,
    FoundryRunStartedEvent,
    create_event_id,
)
from echozero.errors import ValidationError
from echozero.event_bus import EventBus
from echozero.foundry.domain import CompatibilityReport, Dataset, DatasetVersion, EvalReport, ModelArtifact, TrainRun
from echozero.foundry.persistence import EvalReportRepository
from echozero.foundry.services import (
    ArtifactService,
    DatasetService,
    EvalService,
    SplitBalanceService,
    TrainRunService,
)
from echozero.result import Result, err, is_err, ok


class FoundryOrchestrator:
    def __init__(self, root: Path, event_bus: EventBus | None = None):
        self._datasets = DatasetService(root)
        self._split_balance = SplitBalanceService()
        self._train_runs = TrainRunService(root)
        self._artifacts = ArtifactService(root)
        self._eval = EvalService(EvalReportRepository(root))
        self._event_bus = event_bus

    # Dataset lane ---------------------------------------------------------

    def create_dataset(self, name: str, source_kind: str = "folder_import") -> Result[Dataset]:
        try:
            return ok(self._datasets.create_dataset(name=name, source_kind=source_kind))
        except Exception as exc:
            return err(exc)

    def ingest_dataset_folder(self, dataset_id: str, folder_path: str | Path) -> Result[DatasetVersion]:
        try:
            return ok(self._datasets.ingest_from_folder(dataset_id, folder_path))
        except Exception as exc:
            return err(exc)

    def plan_dataset_version(
        self,
        version_id: str,
        *,
        validation_split: float = 0.15,
        test_split: float = 0.10,
        seed: int = 42,
        balance_strategy: str = "none",
    ) -> Result[DatasetVersion]:
        try:
            version = self._datasets.get_version(version_id)
            if version is None:
                return err(ValidationError(f"DatasetVersion not found: {version_id}"))

            split_plan = self._split_balance.plan_splits(
                version,
                validation_split=validation_split,
                test_split=test_split,
                seed=seed,
            )
            balance_plan = self._split_balance.plan_balance(version, strategy=balance_strategy)
            updated = self._datasets.update_version_plans(
                version_id,
                split_plan=split_plan,
                balance_plan=balance_plan,
            )
            return ok(updated)
        except Exception as exc:
            return err(exc)

    # Train run lane -------------------------------------------------------

    def create_run(
        self,
        dataset_version_id: str,
        run_spec: dict[str, Any],
        *,
        backend: str = "pytorch",
        device: str = "cpu",
    ) -> Result[TrainRun]:
        try:
            run = self._train_runs.create_run(
                dataset_version_id=dataset_version_id,
                run_spec=run_spec,
                backend=backend,
                device=device,
            )
            self._publish(
                FoundryRunCreatedEvent(
                    event_id=create_event_id(),
                    timestamp=run.created_at.timestamp(),
                    correlation_id=run.id,
                    run_id=run.id,
                    dataset_version_id=run.dataset_version_id,
                    status=run.status.value,
                )
            )
            return ok(run)
        except Exception as exc:
            return err(exc)

    def start_run(self, run_id: str) -> Result[TrainRun]:
        try:
            run = self._train_runs.start_run(run_id)
            self._publish(
                FoundryRunStartedEvent(
                    event_id=create_event_id(),
                    timestamp=run.updated_at.timestamp(),
                    correlation_id=run.id,
                    run_id=run.id,
                    status=run.status.value,
                )
            )
            return ok(run)
        except Exception as exc:
            return err(exc)

    def cancel_run(self, run_id: str, reason: str = "user") -> Result[TrainRun]:
        try:
            return ok(self._train_runs.cancel_run(run_id, reason=reason))
        except Exception as exc:
            return err(exc)

    def resume_run(self, run_id: str) -> Result[TrainRun]:
        try:
            return ok(self._train_runs.resume_run(run_id))
        except Exception as exc:
            return err(exc)

    def complete_run(self, run_id: str, metrics: dict | None = None) -> Result[TrainRun]:
        try:
            return ok(self._train_runs.complete_run(run_id, metrics=metrics))
        except Exception as exc:
            return err(exc)

    def fail_run(self, run_id: str, error: str) -> Result[TrainRun]:
        try:
            return ok(self._train_runs.fail_run(run_id, error=error))
        except Exception as exc:
            return err(exc)

    def save_checkpoint(self, run_id: str, epoch: int, metric_snapshot: dict | None = None) -> Result[Path]:
        try:
            return ok(self._train_runs.save_checkpoint(run_id, epoch=epoch, metric_snapshot=metric_snapshot))
        except Exception as exc:
            return err(exc)

    def get_run(self, run_id: str) -> Result[TrainRun]:
        try:
            run = self._train_runs.get_run(run_id)
            if run is None:
                return err(ValidationError(f"TrainRun not found: {run_id}"))
            return ok(run)
        except Exception as exc:
            return err(exc)

    # Eval + artifact lane -------------------------------------------------

    def record_eval(
        self,
        run_id: str,
        *,
        classification_mode: str,
        metrics: dict,
        threshold_policy: dict | None = None,
        confusion: dict | None = None,
    ) -> Result[EvalReport]:
        try:
            return ok(
                self._eval.record_eval(
                    run_id,
                    classification_mode=classification_mode,
                    metrics=metrics,
                    threshold_policy=threshold_policy,
                    confusion=confusion,
                )
            )
        except Exception as exc:
            return err(exc)

    def finalize_artifact(self, run_id: str, manifest: dict[str, Any]) -> Result[ModelArtifact]:
        try:
            artifact = self._artifacts.finalize_artifact(run_id, manifest)
            self._publish(
                FoundryArtifactFinalizedEvent(
                    event_id=create_event_id(),
                    timestamp=artifact.created_at.timestamp(),
                    correlation_id=artifact.run_id,
                    artifact_id=artifact.id,
                    run_id=artifact.run_id,
                )
            )
            return ok(artifact)
        except Exception as exc:
            return err(exc)

    def validate_artifact(
        self,
        artifact_id: str,
        *,
        consumer: str = "PyTorchAudioClassify",
    ) -> Result[CompatibilityReport]:
        try:
            report = self._artifacts.validate_compatibility(artifact_id, consumer=consumer)
            self._publish(
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
            return ok(report)
        except Exception as exc:
            return err(exc)

    def finalize_artifact_checked(
        self,
        run_id: str,
        manifest: dict[str, Any],
        *,
        consumer: str = "PyTorchAudioClassify",
    ) -> Result[ModelArtifact]:
        finalized = self.finalize_artifact(run_id, manifest)
        if is_err(finalized):
            return finalized

        artifact = finalized.value
        report_result = self.validate_artifact(artifact.id, consumer=consumer)
        if is_err(report_result):
            return report_result

        report = report_result.value
        if not report.ok:
            return err(
                ValidationError(
                    "Artifact compatibility gate failed: " + "; ".join(report.errors)
                )
            )
        return ok(artifact)

    def _publish(self, event) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(event)

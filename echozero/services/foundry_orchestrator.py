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
from echozero.foundry.domain import CompatibilityReport, ModelArtifact, TrainRun
from echozero.foundry.services import ArtifactService, TrainRunService
from echozero.result import Result, err, is_err, ok


class FoundryOrchestrator:
    def __init__(self, root: Path, event_bus: EventBus | None = None):
        self._train_runs = TrainRunService(root)
        self._artifacts = ArtifactService(root)
        self._event_bus = event_bus

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

    def get_run(self, run_id: str) -> Result[TrainRun]:
        try:
            run = self._train_runs.get_run(run_id)
            if run is None:
                return err(ValidationError(f"TrainRun not found: {run_id}"))
            return ok(run)
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
        """Finalize artifact and enforce compatibility gate before promotion.

        Returns Err(ValidationError) when compatibility checks fail.
        """
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

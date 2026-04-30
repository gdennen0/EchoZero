"""Training orchestration boundary over dataset-version inputs.
Exists because model tuning should be isolated from review mutation/extraction flows.
Connects dataset version ids to run lifecycle operations.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

from echozero.foundry.domain import TrainRun
from echozero.foundry.services.train_run_service import TrainRunService


class TrainingOrchestrator:
    """Orchestrates training run lifecycle from dataset version identifiers."""

    def __init__(
        self,
        root: Path,
        *,
        run_service: TrainRunService | None = None,
    ) -> None:
        self._runs = run_service or TrainRunService(root)

    def create_run(
        self,
        dataset_version_id: str,
        run_spec: dict[str, Any],
        *,
        backend: str = "pytorch",
        device: str = "cpu",
    ) -> TrainRun:
        """Create one queued training run for a dataset version."""
        self._validate_dataset_contract(dataset_version_id, run_spec)
        return self._runs.create_run(
            dataset_version_id,
            run_spec,
            backend=backend,
            device=device,
        )

    def start_run(self, run_id: str, *, cancel_event: Event | None = None) -> TrainRun:
        """Start one queued training run."""
        return self._runs.start_run(run_id, cancel_event=cancel_event)

    @staticmethod
    def _validate_dataset_contract(dataset_version_id: str, run_spec: dict[str, Any]) -> None:
        if not isinstance(dataset_version_id, str) or not dataset_version_id.strip():
            raise ValueError("dataset_version_id is required")
        data = run_spec.get("data")
        if not isinstance(data, dict):
            return
        embedded_id = data.get("datasetVersionId")
        if embedded_id is None:
            return
        if not isinstance(embedded_id, str) or not embedded_id.strip():
            raise ValueError("run_spec.data.datasetVersionId must be a non-empty string when provided")
        if embedded_id != dataset_version_id:
            raise ValueError(
                "dataset_version_id and run_spec.data.datasetVersionId must match for training orchestration"
            )

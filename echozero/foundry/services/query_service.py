from __future__ import annotations

from echozero.foundry.domain import Dataset, EvalReport, ModelArtifact, TrainRun
from echozero.foundry.persistence import (
    DatasetRepository,
    EvalReportRepository,
    ModelArtifactRepository,
)
from echozero.foundry.services.train_run_service import TrainRunService


class FoundryQueryService:
    def __init__(
        self,
        *,
        dataset_repo: DatasetRepository,
        artifact_repo: ModelArtifactRepository,
        eval_repo: EvalReportRepository,
        run_service: TrainRunService,
    ):
        self._dataset_repo = dataset_repo
        self._artifact_repo = artifact_repo
        self._eval_repo = eval_repo
        self._run_service = run_service

    def list_datasets(self) -> list[Dataset]:
        return self._dataset_repo.list()

    def list_runs(self) -> list[TrainRun]:
        return self._run_service.list_runs()

    def list_artifacts(self) -> list[ModelArtifact]:
        return self._artifact_repo.list()

    def list_artifacts_for_run(self, run_id: str) -> list[ModelArtifact]:
        return self._artifact_repo.list_for_run(run_id)

    def get_artifact(self, artifact_id: str) -> ModelArtifact | None:
        return self._artifact_repo.get(artifact_id)

    def list_eval_reports_for_run(self, run_id: str) -> list[EvalReport]:
        return self._eval_repo.list_for_run(run_id)

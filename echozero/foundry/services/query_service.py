"""FoundryQueryService: read-only state lookups for Foundry UI and app bridges.
Exists because callers need one stable query boundary instead of reaching into repositories.
Connects persisted Foundry state to UI/runtime surfaces, including project-scoped review datasets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from echozero.foundry.domain import Dataset, DatasetVersion, EvalReport, ModelArtifact, TrainRun
from echozero.foundry.persistence import (
    DatasetRepository,
    DatasetVersionRepository,
    EvalReportRepository,
    ModelArtifactRepository,
)
from echozero.foundry.services.train_run_service import TrainRunService


@dataclass(slots=True, frozen=True)
class ProjectReviewDatasetVersionRef:
    """One persisted review-dataset version resolved for one EZ project."""

    dataset_id: str
    dataset_name: str
    version_id: str
    version_number: int
    sample_count: int
    queue_source_kind: str
    project_ref: str
    dataset_folder_path: Path
    version_artifact_path: Path
    created_at: datetime


class FoundryQueryService:
    """Reads persisted Foundry state without mutating repositories."""

    def __init__(
        self,
        *,
        dataset_repo: DatasetRepository,
        version_repo: DatasetVersionRepository,
        artifact_repo: ModelArtifactRepository,
        eval_repo: EvalReportRepository,
        run_service: TrainRunService,
        root: Path | None = None,
    ):
        self._dataset_repo = dataset_repo
        self._version_repo = version_repo
        self._artifact_repo = artifact_repo
        self._eval_repo = eval_repo
        self._run_service = run_service
        self._root = _resolve_root(root=root, dataset_repo=dataset_repo)

    def list_datasets(self) -> list[Dataset]:
        """Return every persisted dataset."""

        return self._dataset_repo.list()

    def list_runs(self) -> list[TrainRun]:
        """Return every persisted training run."""

        return self._run_service.list_runs()

    def list_artifacts(self) -> list[ModelArtifact]:
        """Return every persisted artifact manifest."""

        return self._artifact_repo.list()

    def list_artifacts_for_run(self, run_id: str) -> list[ModelArtifact]:
        """Return artifact manifests for one run."""

        return self._artifact_repo.list_for_run(run_id)

    def get_artifact(self, artifact_id: str) -> ModelArtifact | None:
        """Return one persisted artifact manifest when present."""

        return self._artifact_repo.get(artifact_id)

    def list_eval_reports_for_run(self, run_id: str) -> list[EvalReport]:
        """Return evaluation reports for one run."""

        return self._eval_repo.list_for_run(run_id)

    def list_project_review_dataset_versions(
        self,
        *,
        project_ref: str,
        queue_source_kind: str | None = "ez_project",
    ) -> list[ProjectReviewDatasetVersionRef]:
        """Return persisted review-dataset versions for one EZ project."""

        normalized_project_ref = str(project_ref).strip()
        normalized_queue_kind = (
            str(queue_source_kind).strip() if queue_source_kind is not None else None
        )
        if not normalized_project_ref:
            return []

        matches: list[ProjectReviewDatasetVersionRef] = []
        for dataset in self._dataset_repo.list():
            if not _is_project_review_dataset(dataset):
                continue
            metadata = dataset.metadata or {}
            if str(metadata.get("project_ref", "")).strip() != normalized_project_ref:
                continue
            if normalized_queue_kind is not None:
                candidate_queue_kind = str(metadata.get("queue_source_kind", "")).strip()
                if candidate_queue_kind != normalized_queue_kind:
                    continue
            for version in self._version_repo.list_for_dataset(dataset.id):
                matches.append(self._build_project_review_dataset_version_ref(dataset, version))
        return sorted(
            matches,
            key=lambda item: (item.created_at, item.version_number, item.version_id),
            reverse=True,
        )

    def get_latest_project_review_dataset_version(
        self,
        *,
        project_ref: str,
        queue_source_kind: str | None = "ez_project",
    ) -> ProjectReviewDatasetVersionRef | None:
        """Return the newest persisted review-dataset version for one EZ project."""

        matches = self.list_project_review_dataset_versions(
            project_ref=project_ref,
            queue_source_kind=queue_source_kind,
        )
        return matches[0] if matches else None

    def _build_project_review_dataset_version_ref(
        self,
        dataset: Dataset,
        version: DatasetVersion,
    ) -> ProjectReviewDatasetVersionRef:
        metadata = dataset.metadata or {}
        return ProjectReviewDatasetVersionRef(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            version_id=version.id,
            version_number=version.version,
            sample_count=len(version.samples),
            queue_source_kind=str(metadata.get("queue_source_kind", "")).strip(),
            project_ref=str(metadata.get("project_ref", "")).strip(),
            dataset_folder_path=self._resolve_dataset_folder_path(version),
            version_artifact_path=self._root / "foundry" / "state" / "dataset_versions.json",
            created_at=version.created_at,
        )

    def _resolve_dataset_folder_path(self, version: DatasetVersion) -> Path:
        sample_dirs: list[Path] = []
        for sample in version.samples:
            audio_ref = str(sample.audio_ref).strip()
            if not audio_ref:
                continue
            candidate = Path(audio_ref).expanduser()
            if not candidate.is_absolute():
                candidate = (self._root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            sample_dirs.append(candidate.parent)
        if not sample_dirs:
            return (self._root / "foundry" / "cache").resolve()
        try:
            common_dir = Path(os.path.commonpath([str(path) for path in sample_dirs]))
        except ValueError:
            return (self._root / "foundry" / "cache").resolve()
        return common_dir.resolve()


def _resolve_root(*, root: Path | None, dataset_repo: DatasetRepository) -> Path:
    if root is not None:
        return Path(root).resolve()
    dataset_state_path = getattr(dataset_repo, "_path", None)
    if dataset_state_path is None:
        raise ValueError("FoundryQueryService requires a root path or a dataset repository with _path.")
    return Path(dataset_state_path).resolve().parents[2]


def _is_project_review_dataset(dataset: Dataset) -> bool:
    metadata = dataset.metadata or {}
    schema = str(metadata.get("schema", "")).strip()
    if schema in {"foundry.project_review_dataset.v1", "foundry.review_dataset.v1"}:
        return True
    return dataset.source_kind in {"project_review_export", "review_signal"}

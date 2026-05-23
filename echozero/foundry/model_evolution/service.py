"""Model evolution orchestration service.
Exists to give the EZ app one clean path for improving or creating models from fixed Events.
Connects event-span datasets, all-negative planning, lineage, and Foundry run creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from echozero.foundry import FoundryApp
from echozero.foundry.domain import DatasetVersion, TrainRun
from echozero.foundry.model_evolution.lineage import ModelLineageResolver
from echozero.foundry.model_evolution.planner import (
    CandidateModelPlan,
    EvolutionTrainingProfile,
    ModelEvolutionPlanner,
)
from echozero.foundry.model_evolution.sample_materializer import (
    RuntimeWindowMaterializer,
    RuntimeWindowPolicy,
)
from echozero.foundry.model_evolution.truth import FixedEventTruth
from echozero.models.paths import ensure_installed_models_dir


@dataclass(frozen=True, slots=True)
class ModelEvolutionRunRequest:
    """App-facing request to create candidate model evolution runs."""

    identity: str
    truths: tuple[FixedEventTruth, ...]
    labels: tuple[str, ...] = ("kick", "snare")
    profile_name: str = "beefy"
    source_scope: str = "fixed_events"
    warm_start: bool = True
    initial_model_paths: dict[str, Path] | None = None
    window_policy: RuntimeWindowPolicy | None = None


@dataclass(frozen=True, slots=True)
class ModelEvolutionRunResult:
    """Created model evolution dataset and candidate run records."""

    identity: str
    source_dataset_version: DatasetVersion
    candidate_plans: tuple[CandidateModelPlan, ...]
    runs: tuple[TrainRun, ...]


class ModelEvolutionService:
    """Coordinates the first production slice of app-driven model evolution."""

    def __init__(
        self,
        root: Path,
        *,
        foundry_app_factory: Callable[[Path], FoundryApp] = FoundryApp,
        models_dir_factory: Callable[[], Path] = ensure_installed_models_dir,
    ) -> None:
        self._root = Path(root)
        self._foundry_app_factory = foundry_app_factory
        self._models_dir_factory = models_dir_factory

    def create_candidate_runs(
        self,
        request: ModelEvolutionRunRequest,
    ) -> ModelEvolutionRunResult:
        """Materialize fixed Events and create candidate model training runs."""
        app = self._foundry_app_factory(self._root)
        profile = EvolutionTrainingProfile.named(request.profile_name)
        materializer = RuntimeWindowMaterializer(
            self._root,
            dataset_service=app.datasets,
        )
        source_version = materializer.materialize_dataset(
            list(request.truths),
            dataset_name=f"{request.identity} Model Evolution Samples",
            policy=request.window_policy,
            source_scope=request.source_scope,
        )
        models_dir = self._models_dir_factory().resolve()
        lineage = ModelLineageResolver(models_dir=models_dir).resolve_installed_binary_drum_lineage(
            request.labels,
            enabled=request.warm_start,
            explicit_initial_model_paths=request.initial_model_paths,
        )
        planner = ModelEvolutionPlanner(dataset_service=app.datasets)
        candidate_plans = planner.plan_binary_drum_candidates(
            source_version,
            labels=request.labels,
            identity=request.identity,
            profile=profile,
            lineage_by_label=lineage,
        )
        runs: list[TrainRun] = []
        for plan in candidate_plans:
            self._ensure_planned(app, plan.dataset_version_id)
            run = app.create_run(plan.dataset_version_id, plan.run_spec)
            runs.append(run)
        return ModelEvolutionRunResult(
            identity=request.identity,
            source_dataset_version=source_version,
            candidate_plans=candidate_plans,
            runs=tuple(runs),
        )

    @staticmethod
    def _ensure_planned(app: FoundryApp, dataset_version_id: str) -> None:
        version = app.datasets.get_version(dataset_version_id)
        if version is None:
            raise RuntimeError(f"Candidate dataset version disappeared: {dataset_version_id}")
        if version.split_plan.get("assignments"):
            return
        app.plan_version(
            version.id,
            validation_split=0.15,
            test_split=0.10,
            seed=42,
            balance_strategy="none",
        )

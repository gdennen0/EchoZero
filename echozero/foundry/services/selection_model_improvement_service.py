"""Selection-based model-improvement service contracts.
Exists so selected fixed Events can create Foundry model-evolution candidate runs.
Connects the timeline UX to event-span training datasets and lineage-aware run specs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from echozero.foundry.model_evolution import (
    FixedEventTruth,
    ModelEvolutionRunRequest,
    ModelEvolutionService,
)


@dataclass(frozen=True, slots=True)
class ImproveModelTrainingRequest:
    target_label: str = "selection"
    target_identity: str = "EchoZero Core"
    labels: tuple[str, ...] = ("kick", "snare")
    profile_name: str = "beefy"
    base_model_path: str | None = None
    output_dir: str | None = None
    truths: tuple[FixedEventTruth, ...] = ()


@dataclass(frozen=True, slots=True)
class ImproveModelSelectionSummary:
    target_label: str = "selection"
    target_identity: str = "EchoZero Core"
    selected_event_count: int = 0
    reviewed_event_count: int = 0
    base_model_path: str | None = None
    labels: tuple[str, ...] = ()
    label_counts: dict[str, int] = field(default_factory=dict)
    source_audio_count: int = 0
    truths: tuple[FixedEventTruth, ...] = ()


@dataclass(frozen=True, slots=True)
class ImproveModelTrainingResult:
    target_label: str
    run_id: str
    artifact_id: str
    anchor_sample_count: int = 0
    related_sample_count: int = 0
    compared_to_base_model: bool = False
    target_identity: str = "EchoZero Core"
    run_ids: tuple[str, ...] = ()
    dataset_version_id: str | None = None
    candidate_count: int = 0
    profile_name: str = "beefy"


class SelectionModelImprovementService:
    """Creates model-evolution candidate runs from selected fixed Events."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        model_evolution_service: ModelEvolutionService | None = None,
    ) -> None:
        self._root = Path(root or ".")
        self._model_evolution = model_evolution_service

    def summarize_selection(
        self,
        event_refs: list[object],
        *,
        truths: tuple[FixedEventTruth, ...] = (),
        target_identity: str = "EchoZero Core",
    ) -> ImproveModelSelectionSummary:
        label_counts = Counter(truth.normalized_label for truth in truths)
        labels = tuple(label for label in ("kick", "snare") if label_counts.get(label, 0) > 0)
        if not labels:
            labels = tuple(label for label, _count in sorted(label_counts.items()))
        return ImproveModelSelectionSummary(
            target_label="selection",
            target_identity=target_identity,
            selected_event_count=len(event_refs),
            reviewed_event_count=len(truths),
            labels=labels,
            label_counts=dict(sorted(label_counts.items())),
            source_audio_count=len(
                {
                    str(truth.source_audio_path.expanduser().resolve())
                    for truth in truths
                }
            ),
            truths=truths,
        )

    def train_from_selection(
        self, request: ImproveModelTrainingRequest
    ) -> ImproveModelTrainingResult:
        if request.truths:
            service = self._model_evolution or ModelEvolutionService(self._root)
            result = service.create_candidate_runs(
                ModelEvolutionRunRequest(
                    identity=request.target_identity or "EchoZero Core",
                    truths=request.truths,
                    labels=request.labels or ("kick", "snare"),
                    profile_name=request.profile_name or "beefy",
                    source_scope="timeline_selection",
                    warm_start=True,
                )
            )
            run_ids = tuple(run.id for run in result.runs)
            return ImproveModelTrainingResult(
                target_label=",".join(plan.label for plan in result.candidate_plans),
                target_identity=result.identity,
                run_id=run_ids[0] if run_ids else "",
                run_ids=run_ids,
                artifact_id="candidate-runs-created",
                anchor_sample_count=len(result.source_dataset_version.samples),
                related_sample_count=sum(plan.negative_count for plan in result.candidate_plans),
                compared_to_base_model=True,
                dataset_version_id=result.source_dataset_version.id,
                candidate_count=len(result.candidate_plans),
                profile_name=request.profile_name,
            )
        return ImproveModelTrainingResult(
            target_label=request.target_label,
            run_id="selection-improvement-preview",
            artifact_id=Path(request.output_dir or "artifacts").name,
            compared_to_base_model=bool(request.base_model_path),
        )


__all__ = [
    "ImproveModelSelectionSummary",
    "ImproveModelTrainingRequest",
    "ImproveModelTrainingResult",
    "SelectionModelImprovementService",
]

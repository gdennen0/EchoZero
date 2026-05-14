"""Selection-based model-improvement service contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImproveModelTrainingRequest:
    target_label: str = "selection"
    base_model_path: str | None = None
    output_dir: str | None = None


@dataclass(frozen=True, slots=True)
class ImproveModelSelectionSummary:
    target_label: str = "selection"
    selected_event_count: int = 0
    reviewed_event_count: int = 0
    base_model_path: str | None = None


@dataclass(frozen=True, slots=True)
class ImproveModelTrainingResult:
    target_label: str
    run_id: str
    artifact_id: str
    anchor_sample_count: int = 0
    related_sample_count: int = 0
    compared_to_base_model: bool = False


class SelectionModelImprovementService:
    def summarize_selection(self, event_refs: list[object]) -> ImproveModelSelectionSummary:
        return ImproveModelSelectionSummary(selected_event_count=len(event_refs))

    def train_from_selection(
        self, request: ImproveModelTrainingRequest
    ) -> ImproveModelTrainingResult:
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

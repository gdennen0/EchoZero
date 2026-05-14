"""App-shell hooks for selection-based model improvement."""

from __future__ import annotations

from echozero.foundry.services.selection_model_improvement_service import (
    ImproveModelTrainingRequest,
    SelectionModelImprovementService,
)


class AppShellSelectionModelImprovementMixin:
    def summarize_improve_model_selection(self, event_refs: list[object]) -> object:
        service = SelectionModelImprovementService()
        return service.summarize_selection(event_refs)

    def train_improved_model_from_selection(self, request: ImproveModelTrainingRequest) -> object:
        service = SelectionModelImprovementService()
        return service.train_from_selection(request)


__all__ = ["AppShellSelectionModelImprovementMixin"]

"""
App-shell helpers for selection-based candidate model training from EZ review signals.
Exists because timeline selections need one direct runtime path into Foundry's improve-model flow.
Connects selected reviewed events to base-model choices and candidate-run kickoff without widget-owned logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from echozero.application.session.models import Session
from echozero.application.timeline.models import EventRef
from echozero.foundry.services.selection_model_improvement_service import (
    ImproveModelBaseOption,
    ImproveModelTrainingRequest,
    ImproveModelTrainingResult,
    SelectionModelImprovementService,
)
from echozero.foundry.persistence.review_signal_repository import ReviewSignalRepository
from echozero.foundry.domain.review import ReviewSignal
from echozero.foundry.services.review_signal_service import ReviewSignalService
from echozero.persistence.session import ProjectStorage


@dataclass(frozen=True, slots=True)
class ImproveModelSelectionSummary:
    """Operator-facing summary for one reviewed-event selection."""

    target_label: str
    selected_signal_ids: tuple[str, ...]
    reviewed_event_count: int
    positive_signal_count: int
    negative_signal_count: int
    default_scope_mode: str
    base_model_options: tuple[ImproveModelBaseOption, ...]


class SelectionModelImprovementShell(Protocol):
    project_storage: ProjectStorage

    @property
    def session(self) -> Session: ...


class AppShellSelectionModelImprovementMixin:
    """Expose one EZ runtime path for improve-model-from-selection workflows."""

    def summarize_improve_model_selection(
        self: SelectionModelImprovementShell,
        event_refs: list[EventRef],
    ) -> ImproveModelSelectionSummary:
        """Resolve selected reviewed events into one target-label training summary."""
        project_id = self.project_storage.project.id
        active_song_version_id = self.session.active_song_version_id
        if active_song_version_id is None:
            raise ValueError("Improve Model requires an active song version.")
        signal_repo = ReviewSignalRepository(Path(self.project_storage.working_dir).resolve())
        selected_signals = _selected_review_signals(
            signal_repo,
            event_refs=event_refs,
            project_id=str(project_id),
            active_song_version_id=str(active_song_version_id),
        )
        if not selected_signals:
            raise ValueError(
                "No explicit review signal was found for the selected events. "
                "Verify or reject the events first, then try Improve Model again."
            )
        target_labels = {_selection_target_label(signal) for signal in selected_signals}
        if len(target_labels) != 1:
            joined = ", ".join(sorted(target_labels))
            raise ValueError(
                "Improve Model V1 requires the selected reviewed events to map to one target label. "
                f"Found: {joined}"
            )
        target_label = next(iter(target_labels))
        positive_signal_count = sum(1 for signal in selected_signals if _counts_as_positive(signal))
        negative_signal_count = sum(1 for signal in selected_signals if _counts_as_negative(signal))
        service = SelectionModelImprovementService(Path(self.project_storage.working_dir).resolve())
        signal_ids = tuple(signal.id for signal in selected_signals)
        return ImproveModelSelectionSummary(
            target_label=target_label,
            selected_signal_ids=signal_ids,
            reviewed_event_count=len(selected_signals),
            positive_signal_count=positive_signal_count,
            negative_signal_count=negative_signal_count,
            default_scope_mode=_default_scope_mode(selected_signals),
            base_model_options=service.list_base_model_options(target_label=target_label),
        )

    def train_improved_model_from_selection(
        self: SelectionModelImprovementShell,
        request: ImproveModelTrainingRequest,
    ) -> ImproveModelTrainingResult:
        """Kick off and complete one candidate training run from selected reviewed events."""
        service = SelectionModelImprovementService(Path(self.project_storage.working_dir).resolve())
        return service.train_candidate_model(request)


def _selected_review_signals(
    repo: ReviewSignalRepository,
    *,
    event_refs: list[EventRef],
    project_id: str,
    active_song_version_id: str,
) -> list[ReviewSignal]:
    selected_signals: list[ReviewSignal] = []
    seen: set[str] = set()
    for event_ref in event_refs:
        item_id = _event_review_item_id(
            active_song_version_id=active_song_version_id,
            layer_id=str(event_ref.layer_id),
            event_id=str(event_ref.event_id),
        )
        candidate_signal_ids = (
            ReviewSignalService.build_signal_id(
                f"timeline_review_{project_id}_{active_song_version_id}",
                item_id,
            ),
            ReviewSignalService.build_signal_id(
                f"timeline_fix_{project_id}_{active_song_version_id}",
                item_id,
            ),
        )
        for signal_id in candidate_signal_ids:
            signal = repo.get(signal_id)
            if signal is None or signal.id in seen:
                continue
            selected_signals.append(signal)
            seen.add(signal.id)
            break
    return selected_signals


def _selection_target_label(signal: ReviewSignal) -> str:
    for candidate in (signal.target_class, signal.corrected_label, signal.predicted_label):
        text = str(candidate).strip().lower()
        if text:
            return text
    raise ValueError(f"Review signal '{signal.id}' is missing a usable target label.")


def _counts_as_positive(signal: ReviewSignal) -> bool:
    decision = signal.review_decision
    if decision is None:
        return False
    return bool(decision.training_eligibility.allows_positive_signal)


def _counts_as_negative(signal: ReviewSignal) -> bool:
    decision = signal.review_decision
    if decision is None:
        return False
    return bool(decision.training_eligibility.allows_negative_signal)


def _event_review_item_id(
    *,
    active_song_version_id: str,
    layer_id: str,
    event_id: str,
) -> str:
    return f"timeline_review:{active_song_version_id}:{layer_id}:{event_id}"


def _default_scope_mode(signals: list[ReviewSignal]) -> str:
    layer_values = {
        str(signal.source_provenance.get("layer_ref", "")).strip()
        for signal in signals
        if str(signal.source_provenance.get("layer_ref", "")).strip()
    }
    if len(layer_values) == 1:
        return "song_layer"
    song_values = {
        str(signal.source_provenance.get("song_ref", "")).strip()
        for signal in signals
        if str(signal.source_provenance.get("song_ref", "")).strip()
    }
    if len(song_values) == 1:
        return "song"
    return "project"

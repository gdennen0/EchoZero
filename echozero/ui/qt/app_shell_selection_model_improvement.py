"""App-shell hooks for selection-based model improvement.
Exists so selected timeline Events can create Foundry model-evolution candidate runs.
Connects Stage Zero selection state to fixed Event truth and event-span training.
"""

from __future__ import annotations

from pathlib import Path

from echozero.foundry.model_evolution import FixedEventTruth
from echozero.foundry.services.selection_model_improvement_service import (
    ImproveModelTrainingRequest,
    SelectionModelImprovementService,
)
from echozero.ui.qt.app_shell_timeline_state import resolve_event_clip_preview
from echozero.foundry.services.review_event_state import normalize_review_label


class AppShellSelectionModelImprovementMixin:
    def summarize_improve_model_selection(self, event_refs: list[object]) -> object:
        selected_refs = list(event_refs)
        if not selected_refs:
            selected_refs = _fixed_event_refs_from_timeline(self._app.timeline)
        presentation = self.presentation()
        truths = tuple(_truths_from_event_refs(presentation, selected_refs))
        service = SelectionModelImprovementService(self.project_storage.working_dir)
        return service.summarize_selection(
            selected_refs,
            truths=truths,
            target_identity=_default_target_identity(self),
        )

    def train_improved_model_from_selection(self, request: ImproveModelTrainingRequest) -> object:
        service = SelectionModelImprovementService(self.project_storage.working_dir)
        return service.train_from_selection(request)


def _truths_from_event_refs(presentation, event_refs: list[object]) -> list[FixedEventTruth]:
    truths: list[FixedEventTruth] = []
    for event_ref in event_refs:
        layer_id = getattr(event_ref, "layer_id", None)
        take_id = getattr(event_ref, "take_id", None)
        event_id = getattr(event_ref, "event_id", None)
        if layer_id is None or event_id is None:
            continue
        target = _find_event(presentation, layer_id=layer_id, take_id=take_id, event_id=event_id)
        if target is None:
            continue
        layer, _take, event = target
        label = _event_training_label(layer, event)
        if not label:
            continue
        preview = resolve_event_clip_preview(
            presentation,
            layer_id=layer_id,
            take_id=take_id,
            event_id=event_id,
        )
        truths.append(
            FixedEventTruth(
                truth_id=f"timeline_selection:{layer_id}:{take_id}:{event_id}",
                label=label,
                source_audio_path=Path(preview.source_ref),
                event_start_seconds=float(event.start),
                event_end_seconds=float(event.end),
                anchor_seconds=float(event.start),
                layer_id=str(layer_id),
                event_id=str(event_id),
                metadata={
                    "selection_source": "timeline",
                    "layer_title": str(getattr(layer, "title", "")),
                    "event_label": str(getattr(event, "label", "")),
                    "event_duration_seconds": float(event.end) - float(event.start),
                },
            )
        )
    return truths


def _fixed_event_refs_from_timeline(timeline) -> list[object]:
    refs: list[object] = []
    for layer in getattr(timeline, "layers", []) or []:
        for take in getattr(layer, "takes", []) or []:
            for event in getattr(take, "events", []) or []:
                if str(getattr(event, "review_state", "")).strip().lower() not in {
                    "corrected",
                    "signed_off",
                }:
                    continue
                refs.append(
                    _EventRefLike(
                        layer_id=getattr(layer, "id", None),
                        take_id=getattr(take, "id", None),
                        event_id=getattr(event, "id", None),
                    )
                )
    return refs


class _EventRefLike:
    def __init__(self, *, layer_id: object, take_id: object, event_id: object) -> None:
        self.layer_id = layer_id
        self.take_id = take_id
        self.event_id = event_id


def _find_event(presentation, *, layer_id: object, take_id: object, event_id: object):
    for layer in presentation.layers:
        if str(layer.layer_id) != str(layer_id):
            continue
        if take_id in (None, layer.main_take_id):
            for event in layer.events:
                if str(event.event_id) == str(event_id):
                    return layer, None, event
        for take in layer.takes:
            if str(take.take_id) != str(take_id):
                continue
            for event in take.events:
                if str(event.event_id) == str(event_id):
                    return layer, take, event
    return None


def _event_training_label(layer, event) -> str:
    classifications = dict(getattr(event, "classifications", {}) or {})
    candidates = (
        classifications.get("class"),
        classifications.get("label"),
        getattr(event, "label", ""),
        getattr(layer, "title", ""),
    )
    for candidate in candidates:
        label = normalize_review_label(str(candidate or ""))
        if label and label != "event":
            return label
    return ""


def _default_target_identity(shell: object) -> str:
    project = getattr(getattr(shell, "project_storage", None), "project", None)
    name = str(getattr(project, "name", "") or "").strip()
    return name or "EchoZero Core"


__all__ = ["AppShellSelectionModelImprovementMixin"]

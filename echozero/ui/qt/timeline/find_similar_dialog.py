"""Find-similar event comparison dialog."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.models import EventRef


@dataclass(frozen=True, slots=True)
class _Candidate:
    layer_id: LayerId
    take_id: TakeId
    event_id: EventId


class FindSimilarSoundsDialog(QDialog):
    """Small comparison dialog for shape-envelope and timbre event matching."""

    def __init__(
        self,
        *,
        presentation: TimelinePresentation,
        layer_id: LayerId,
        take_id: TakeId,
        event_id: EventId,
        default_scope_mode: str = "take",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._presentation = presentation
        self._layer_id = layer_id
        self._take_id = take_id
        self._event_id = event_id
        self._default_scope_mode = _coerce_scope_mode(default_scope_mode)
        self.setWindowTitle("Compare Events")

        layout = QVBoxLayout(self)
        self._summary = QLabel(self._summary_text(), self)
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem("Shape Envelope", "shape_envelope")
        self._mode_combo.addItem("Timbre Fingerprint", "timbre_fingerprint")
        layout.addWidget(self._mode_combo)

        self._scope_combo = QComboBox(self)
        self._scope_combo.addItem("Current Take", "take")
        self._scope_combo.addItem("Current Layer", "layer")
        self._scope_combo.addItem("Selected Layers · Main Takes", "selected_layers_main")
        index = self._scope_combo.findData(self._default_scope_mode)
        if index >= 0:
            self._scope_combo.setCurrentIndex(index)
        layout.addWidget(self._scope_combo)

        self._strength_combo = QComboBox(self)
        self._strength_combo.addItem("Very Strict", "very_strict")
        self._strength_combo.addItem("Strict", "strict")
        self._strength_combo.addItem("Balanced", "balanced")
        self._strength_combo.addItem("Loose", "loose")
        balanced = self._strength_combo.findData("balanced")
        if balanced >= 0:
            self._strength_combo.setCurrentIndex(balanced)
        layout.addWidget(self._strength_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_payload(self) -> dict[str, object]:
        # The real similarity computation is application-layer intent work.  The
        # dialog returns the anchor payload shape expected by legacy action tests
        # and leaves scored selection to SelectSimilarEvents callers.
        candidate = EventRef(self._layer_id, self._take_id, self._event_id)
        selected_layer_ids = self._selected_layer_ids_for_scope(str(self._scope_combo.currentData()))
        return {
            "event_ids": [self._event_id],
            "event_refs": [candidate],
            "anchor_layer_id": self._layer_id,
            "anchor_take_id": self._take_id,
            "selected_layer_ids": selected_layer_ids,
            "comparison_mode": str(self._mode_combo.currentData()),
            "scope_mode": str(self._scope_combo.currentData()),
            "match_strength": str(self._strength_combo.currentData()),
        }

    def _summary_text(self) -> str:
        candidates = self._candidate_events(self._default_scope_mode)
        layer_count = len({candidate.layer_id for candidate in candidates})
        take_count = len({candidate.take_id for candidate in candidates})
        return (
            f"Compare the selected event against {len(candidates)} candidate events across "
            f"{take_count} takes and {layer_count} layers."
        )

    def _candidate_events(self, scope_mode: str) -> tuple[_Candidate, ...]:
        layer = _find_layer(self._presentation, self._layer_id)
        if layer is None:
            return ()
        if scope_mode == "selected_layers_main":
            selected_layer_ids = self._selected_layer_ids_for_scope(scope_mode)
            layers = [candidate for candidate in self._presentation.layers if candidate.layer_id in selected_layer_ids]
            return tuple(
                _Candidate(candidate_layer.layer_id, take.take_id, event.event_id)
                for candidate_layer in layers
                for take in candidate_layer.takes
                if take.take_id == candidate_layer.main_take_id
                for event in take.events
            )
        if scope_mode == "layer":
            return tuple(
                _Candidate(layer.layer_id, take.take_id, event.event_id)
                for take in layer.takes
                for event in take.events
            )
        return tuple(
            _Candidate(layer.layer_id, take.take_id, event.event_id)
            for take in layer.takes
            if take.take_id == self._take_id
            for event in take.events
        )

    def _selected_layer_ids_for_scope(self, scope_mode: str) -> list[LayerId]:
        if scope_mode == "selected_layers_main":
            ids = [layer_id for layer_id in self._presentation.selected_layer_ids if layer_id]
            return ids or [self._layer_id]
        return [self._layer_id]


class EventComparisonDialog(FindSimilarSoundsDialog):
    """New name for the find-similar dialog."""


class FindSimilarShapesDialog(FindSimilarSoundsDialog):
    """Compatibility alias for older tests/imports."""


def _find_layer(presentation: TimelinePresentation, layer_id: LayerId) -> LayerPresentation | None:
    return next((layer for layer in presentation.layers if layer.layer_id == layer_id), None)


def _coerce_scope_mode(value: str) -> str:
    normalized = (value or "take").strip().lower()
    if normalized not in {"take", "layer", "selected_layers_main"}:
        return "take"
    return normalized


__all__ = ["EventComparisonDialog", "FindSimilarShapesDialog", "FindSimilarSoundsDialog"]

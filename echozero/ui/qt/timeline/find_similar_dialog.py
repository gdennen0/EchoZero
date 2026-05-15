"""Find-similar event comparison dialog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.models import EventPresentation, LayerPresentation, TakeLanePresentation, TimelinePresentation
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.event_similarity_audio import (
    align_shape_to_reference,
    audio_shape_preview,
    compare_shape_similarity,
    read_mono_audio_slice,
)
from echozero.application.timeline.models import EventRef


@dataclass(frozen=True, slots=True)
class _Candidate:
    layer_id: LayerId
    take_id: TakeId
    event_id: EventId
    label: str
    start: float
    end: float
    audio_path: str | None


@dataclass(frozen=True, slots=True)
class ShapePreviewRow:
    """One visual shape row shown while configuring find-similar matching."""

    event_ref: EventRef
    label: str
    shape: tuple[float, ...]
    score: float | None = None
    is_anchor: bool = False


class EventShapeComparisonPreviewWidget(QWidget):
    """Paint the anchor shape and the candidate event shapes considered by the dialog."""

    def __init__(self, rows: tuple[ShapePreviewRow, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows = rows
        self.setMinimumHeight(max(120, 44 + 42 * max(1, len(rows))))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

    @property
    def rows(self) -> tuple[ShapePreviewRow, ...]:
        return self._rows

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        if QGuiApplication.instance() is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.fillRect(rect, QColor("#111827"))
        painter.setPen(QPen(QColor("#6b7280"), 1.0))
        painter.drawRoundedRect(QRectF(rect), 8.0, 8.0)
        if not self._rows:
            painter.setPen(QColor("#d1d5db"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Audio shape preview unavailable")
            painter.end()
            return

        painter.setPen(QColor("#f9fafb"))
        painter.drawText(rect.adjusted(12, 8, -12, -8), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "Shape comparison preview")
        top = rect.top() + 34
        row_height = 38
        label_width = min(190, max(110, rect.width() // 3))
        graph_left = rect.left() + label_width + 16
        graph_width = max(40, rect.right() - graph_left - 12)
        for index, row in enumerate(self._rows):
            y = top + index * row_height
            row_rect = QRectF(rect.left() + 8, y, rect.width() - 16, row_height - 6)
            painter.fillRect(row_rect, QColor("#1f2937") if row.is_anchor else QColor("#172033"))
            painter.setPen(QColor("#fbbf24") if row.is_anchor else QColor("#cbd5e1"))
            score = "anchor" if row.is_anchor else ("--" if row.score is None else f"{row.score:.2f}")
            label = f"{row.label} · {score}"
            painter.drawText(row_rect.adjusted(8, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
            graph_rect = QRectF(graph_left, y + 6, graph_width, row_height - 18)
            painter.setPen(QPen(QColor("#374151"), 1.0))
            painter.drawLine(QPointF(graph_rect.left(), graph_rect.center().y()), QPointF(graph_rect.right(), graph_rect.center().y()))
            self._draw_shape(painter, graph_rect, row.shape, QColor("#f59e0b") if row.is_anchor else QColor("#38bdf8"))
        painter.end()

    def _draw_shape(self, painter: QPainter, rect: QRectF, shape: tuple[float, ...], color: QColor) -> None:
        if not shape:
            painter.setPen(QColor("#9ca3af"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "no samples")
            return
        values = list(shape)
        peak = max(values) if values else 0.0
        if peak > 1e-9:
            values = [value / peak for value in values]
        path = QPainterPath()
        for index, value in enumerate(values):
            x = rect.left() + (rect.width() * index / max(1, len(values) - 1))
            y = rect.bottom() - (rect.height() * max(0.0, min(1.0, float(value))))
            point = QPointF(x, y)
            if index == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        painter.setPen(QPen(color, 2.0))
        painter.drawPath(path)


class FindSimilarSoundsDialog(QDialog):
    """Comparison dialog for shape-envelope and timbre event matching."""

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
        self._scope_combo.currentIndexChanged.connect(self._refresh_preview)
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

        self._preview_scroll = QScrollArea(self)
        self._preview_scroll.setWidgetResizable(True)
        self._preview_scroll.setMinimumHeight(180)
        self._preview_widget = EventShapeComparisonPreviewWidget((), self._preview_scroll)
        self._preview_scroll.setWidget(self._preview_widget)
        layout.addWidget(self._preview_scroll, stretch=1)
        self._refresh_preview()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_payload(self) -> dict[str, object]:
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

    def _refresh_preview(self) -> None:
        rows = self._build_shape_preview_rows(str(self._scope_combo.currentData()))
        self._preview_widget = EventShapeComparisonPreviewWidget(rows, self._preview_scroll)
        self._preview_scroll.setWidget(self._preview_widget)

    def _build_shape_preview_rows(self, scope_mode: str) -> tuple[ShapePreviewRow, ...]:
        candidates = self._candidate_events(scope_mode)
        anchor = next(
            (
                candidate
                for candidate in candidates
                if candidate.layer_id == self._layer_id
                and candidate.take_id == self._take_id
                and candidate.event_id == self._event_id
            ),
            None,
        )
        if anchor is None:
            anchor = self._find_event_candidate(self._layer_id, self._take_id, self._event_id)
        if anchor is None:
            return ()
        anchor_shape = _shape_for_candidate(anchor)
        if not anchor_shape:
            return ()
        rows = [
            ShapePreviewRow(
                event_ref=EventRef(anchor.layer_id, anchor.take_id, anchor.event_id),
                label=f"Current · {anchor.label}",
                shape=anchor_shape,
                score=1.0,
                is_anchor=True,
            )
        ]
        for candidate in candidates:
            if (
                candidate.layer_id == anchor.layer_id
                and candidate.take_id == anchor.take_id
                and candidate.event_id == anchor.event_id
            ):
                continue
            candidate_shape = _shape_for_candidate(candidate)
            if not candidate_shape:
                continue
            aligned = align_shape_to_reference(anchor_shape, candidate_shape)
            rows.append(
                ShapePreviewRow(
                    event_ref=EventRef(candidate.layer_id, candidate.take_id, candidate.event_id),
                    label=candidate.label,
                    shape=aligned,
                    score=compare_shape_similarity(anchor_shape, candidate_shape),
                )
            )
        return tuple(rows)

    def _find_event_candidate(self, layer_id: LayerId, take_id: TakeId, event_id: EventId) -> _Candidate | None:
        layer = _find_layer(self._presentation, layer_id)
        if layer is None:
            return None
        for candidate in _layer_candidates(layer):
            if candidate.take_id == take_id and candidate.event_id == event_id:
                return candidate
        return None

    def _candidate_events(self, scope_mode: str) -> tuple[_Candidate, ...]:
        layer = _find_layer(self._presentation, self._layer_id)
        if layer is None:
            return ()
        if scope_mode == "selected_layers_main":
            selected_layer_ids = self._selected_layer_ids_for_scope(scope_mode)
            layers = [candidate for candidate in self._presentation.layers if candidate.layer_id in selected_layer_ids]
            return tuple(
                candidate
                for candidate_layer in layers
                for candidate in _layer_main_candidates(candidate_layer)
            )
        if scope_mode == "layer":
            return tuple(candidate for candidate in _layer_candidates(layer))
        return tuple(
            candidate
            for candidate in _layer_candidates(layer)
            if candidate.take_id == self._take_id
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


def _layer_candidates(layer: LayerPresentation) -> tuple[_Candidate, ...]:
    candidates: list[_Candidate] = []
    main_take_id = layer.main_take_id or TakeId("main")
    candidates.extend(
        _candidate_from_event(
            layer=layer,
            take_id=main_take_id,
            event=event,
            take=None,
        )
        for event in layer.events
    )
    candidates.extend(
        _candidate_from_event(layer=layer, take_id=take.take_id, event=event, take=take)
        for take in layer.takes
        for event in take.events
    )
    return tuple(candidates)


def _layer_main_candidates(layer: LayerPresentation) -> tuple[_Candidate, ...]:
    main_take_id = layer.main_take_id
    if main_take_id is None:
        return tuple(_layer_candidates(layer))
    return tuple(candidate for candidate in _layer_candidates(layer) if candidate.take_id == main_take_id)


def _candidate_from_event(
    *,
    layer: LayerPresentation,
    take_id: TakeId,
    event: EventPresentation,
    take: TakeLanePresentation | None,
) -> _Candidate:
    return _Candidate(
        layer_id=layer.layer_id,
        take_id=take_id,
        event_id=event.event_id,
        label=event.label or str(event.event_id),
        start=float(event.start),
        end=float(event.end),
        audio_path=(take.source_audio_path if take is not None and take.source_audio_path else layer.source_audio_path),
    )


def _shape_for_candidate(candidate: _Candidate) -> tuple[float, ...]:
    if not candidate.audio_path:
        return ()
    path = Path(candidate.audio_path)
    if not path.exists():
        return ()
    sliced = read_mono_audio_slice(path, start_seconds=candidate.start, end_seconds=candidate.end)
    if sliced is None:
        return ()
    samples, _sample_rate = sliced
    return audio_shape_preview(samples, sample_count=64)


def _coerce_scope_mode(value: str) -> str:
    normalized = (value or "take").strip().lower()
    if normalized not in {"take", "layer", "selected_layers_main"}:
        return "take"
    return normalized


__all__ = [
    "EventComparisonDialog",
    "EventShapeComparisonPreviewWidget",
    "FindSimilarShapesDialog",
    "FindSimilarSoundsDialog",
    "ShapePreviewRow",
]

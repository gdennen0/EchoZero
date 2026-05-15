"""Find-similar event comparison dialog.
Exists so users can tune shape-matching before selecting related timeline events.
Connects timeline presentation events to visual audio-shape comparison previews.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSlider,
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
    is_match: bool = False


class EventShapeComparisonPreviewWidget(QWidget):
    """Paint the anchor shape and the candidate event shapes considered by the dialog."""

    def __init__(
        self,
        rows: tuple[ShapePreviewRow, ...],
        *,
        smoothing: int = 3,
        control_points: int = 24,
        fuzziness: int = 35,
        threshold: float = 0.78,
        scan_total: int | None = None,
        scan_limit: int | None = None,
        match_count: int | None = None,
        action_label: str = "Select matched events",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._rows = rows
        self._smoothing = smoothing
        self._control_points = control_points
        self._fuzziness = fuzziness
        self._threshold = threshold
        self._scan_total = scan_total if scan_total is not None else max(0, len(rows) - 1)
        self._scan_limit = scan_limit if scan_limit is not None else self._scan_total
        self._match_count = match_count if match_count is not None else sum(1 for row in rows[1:] if row.is_match)
        self._action_label = action_label
        self.setMinimumHeight(max(320, 76 + 54 * max(1, len(rows) - 1)))
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
        painter.fillRect(rect, QColor("#07111f"))
        painter.setPen(QPen(QColor("#1d4ed8"), 1.2))
        painter.drawRoundedRect(QRectF(rect), 12.0, 12.0)
        if not self._rows:
            painter.setPen(QColor("#93c5fd"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "NO AUDIO SHAPE TELEMETRY")
            painter.end()
            return

        anchor = self._rows[0]
        scanned_count = max(0, int(self._scan_limit))
        total_count = max(scanned_count, int(self._scan_total))
        left_width = min(360.0, max(285.0, rect.width() * 0.36))
        gap = 12.0
        left_panel = QRectF(rect.left() + 12, rect.top() + 12, left_width, rect.height() - 24)
        right_panel = QRectF(left_panel.right() + gap, rect.top() + 12, rect.right() - left_panel.right() - gap - 12, rect.height() - 24)

        painter.fillRect(left_panel, QColor("#0b1f35"))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(left_panel, 10.0, 10.0)
        painter.setPen(QColor("#dbeafe"))
        painter.drawText(left_panel.adjusted(14, 10, -14, -10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "SELECTED CLIP")
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(left_panel.adjusted(14, 32, -14, -10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, anchor.label)
        self._draw_shape(painter, QRectF(left_panel.left() + 14, left_panel.top() + 62, left_panel.width() - 28, 128), anchor.shape, QColor("#f59e0b"), width=3.2)

        progress_rect = QRectF(left_panel.left() + 14, left_panel.top() + 204, left_panel.width() - 28, 92)
        painter.fillRect(progress_rect, QColor("#07111f"))
        painter.setPen(QColor("#bae6fd"))
        painter.drawText(progress_rect.adjusted(10, 4, -10, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "LIVE ITERATION")
        current_row = self._rows[-1] if len(self._rows) > 1 and scanned_count else None
        current_verdict = "PASS" if current_row is not None and current_row.is_match else "FAIL"
        current_color = QColor("#22c55e") if current_verdict == "PASS" else QColor("#ef4444")
        painter.setPen(QColor("#e0f2fe"))
        painter.drawText(
            progress_rect.adjusted(10, 24, -10, -4),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            f"{scanned_count}/{total_count} scanned · {self._match_count} passes · {self._action_label}",
        )
        if current_row is not None:
            badge = QRectF(progress_rect.left() + 10, progress_rect.top() + 48, 58, 22)
            painter.fillRect(badge, current_color)
            painter.setPen(QColor("#020617"))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, current_verdict)
            painter.setPen(QColor("#dbeafe"))
            score = "--" if current_row.score is None else f"{current_row.score:.2f}"
            painter.drawText(
                progress_rect.adjusted(78, 48, -10, -4),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                f"current: {current_row.label} · {score}",
            )
        segment_top = progress_rect.top() + 76
        visible_segments = max(1, min(total_count, 28))
        segment_width = max(5.0, (progress_rect.width() - 20) / visible_segments)
        scanned_segments = round(visible_segments * (scanned_count / max(1, total_count)))
        for index in range(visible_segments):
            segment = QRectF(progress_rect.left() + 10 + index * segment_width + 1, segment_top, segment_width - 2, 8)
            painter.fillRect(segment, QColor("#0ea5e9") if index < scanned_segments else QColor("#1e293b"))

        meter_top = progress_rect.bottom() + 16
        for index, (label, value, color) in enumerate(
            (
                (f"smooth {self._smoothing}", min(1.0, self._smoothing / 12.0), QColor("#22d3ee")),
                (f"points {self._control_points}", min(1.0, self._control_points / 64.0), QColor("#a78bfa")),
                (f"fuzz {self._fuzziness}%", min(1.0, self._fuzziness / 100.0), QColor("#f59e0b")),
                (f"min {self._threshold:.2f}", min(1.0, self._threshold), QColor("#34d399")),
            )
        ):
            y = meter_top + index * 34
            painter.setPen(QColor("#93c5fd"))
            painter.drawText(QRectF(left_panel.left() + 14, y, 88, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label.upper())
            bar = QRectF(left_panel.left() + 112, y + 5, left_panel.width() - 132, 8)
            painter.fillRect(bar, QColor("#1e293b"))
            painter.fillRect(QRectF(bar.left(), bar.top(), bar.width() * value, bar.height()), color)

        painter.fillRect(right_panel, QColor("#081827"))
        painter.setPen(QPen(QColor("#164e63"), 1.0))
        painter.drawRoundedRect(right_panel, 10.0, 10.0)
        painter.setPen(QColor("#93c5fd"))
        painter.drawText(right_panel.adjusted(14, 10, -14, -10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "CANDIDATE OVERLAYS · GOLD ANCHOR / CYAN EVENT")

        top = right_panel.top() + 38
        row_height = 48
        label_width = min(210, max(135, int(right_panel.width() * 0.31)))
        graph_left = right_panel.left() + label_width + 24
        graph_width = max(120, right_panel.right() - graph_left - 14)
        for index, row in enumerate(self._rows[1:]):
            y = top + index * row_height
            row_rect = QRectF(right_panel.left() + 10, y, right_panel.width() - 20, row_height - 7)
            is_current = index == max(0, min(len(self._rows) - 2, scanned_count - 1))
            verdict_color = QColor("#22c55e") if row.is_match else QColor("#ef4444")
            verdict_fill = QColor(verdict_color)
            verdict_fill.setAlpha(44 if row.is_match else 34)
            painter.fillRect(row_rect, QColor("#0d2740") if index % 2 == 0 else QColor("#0b2137"))
            painter.fillRect(row_rect, verdict_fill)
            painter.setPen(QPen(verdict_color if is_current else QColor("#155e75"), 2.4 if is_current else 0.8))
            painter.drawRoundedRect(row_rect, 7.0, 7.0)
            score = "--" if row.score is None else f"{row.score:.2f}"
            verdict = "PASS" if row.is_match else "FAIL"
            badge_rect = QRectF(row_rect.left() + 8, row_rect.top() + 9, 48, row_rect.height() - 18)
            painter.fillRect(badge_rect, verdict_color)
            painter.setPen(QColor("#020617"))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, verdict)
            painter.setPen(QColor("#bbf7d0") if row.is_match else QColor("#fecaca"))
            painter.drawText(row_rect.adjusted(66, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"{row.label} · {score}")
            graph_rect = QRectF(graph_left, y + 8, graph_width, row_height - 24)
            graph_overlay = QColor(verdict_color)
            graph_overlay.setAlpha(26)
            painter.fillRect(graph_rect.adjusted(-4, -4, 4, 4), graph_overlay)
            painter.setPen(QPen(verdict_color, 1.0))
            painter.drawRoundedRect(graph_rect.adjusted(-4, -4, 4, 4), 5.0, 5.0)
            painter.setPen(QPen(QColor("#1e3a8a"), 1.0))
            painter.drawLine(QPointF(graph_rect.left(), graph_rect.center().y()), QPointF(graph_rect.right(), graph_rect.center().y()))
            self._draw_shape(painter, graph_rect, anchor.shape, QColor("#f59e0b"), width=2.2)
            self._draw_shape(painter, graph_rect, row.shape, QColor("#22d3ee"), width=2.0)
        painter.end()

    def _draw_shape(
        self,
        painter: QPainter,
        rect: QRectF,
        shape: tuple[float, ...],
        color: QColor,
        *,
        width: float = 2.0,
    ) -> None:
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
        painter.setPen(QPen(color, width))
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
        self._scan_limit: int | None = None
        self.setWindowTitle("Compare Events")
        self.setStyleSheet(
            "QDialog { background: #020617; color: #dbeafe; }"
            "QLabel { color: #dbeafe; }"
            "QComboBox { background: #0f172a; color: #e0f2fe; border: 1px solid #2563eb; padding: 6px; }"
            "QScrollArea { border: 1px solid #1d4ed8; background: #020617; }"
            "QPushButton { background: #1d4ed8; color: white; padding: 6px 14px; }"
        )

        layout = QVBoxLayout(self)
        control_grid = QGridLayout()
        self._summary = QLabel(self._summary_text(), self)
        self._summary.setWordWrap(True)
        control_grid.addWidget(self._summary, 0, 0, 1, 6)

        self._mode_label = QLabel("Shape telemetry mode", self)
        control_grid.addWidget(self._mode_label, 1, 0)
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem("Shape Envelope", "shape_envelope")
        self._mode_combo.addItem("Timbre Fingerprint", "timbre_fingerprint")
        control_grid.addWidget(self._mode_combo, 1, 1)

        self._scope_label = QLabel("Scan field", self)
        control_grid.addWidget(self._scope_label, 1, 2)
        self._scope_combo = QComboBox(self)
        self._scope_combo.addItem("Current Take", "take")
        self._scope_combo.addItem("Current Layer", "layer")
        self._scope_combo.addItem("Selected Layers · Main Takes", "selected_layers_main")
        index = self._scope_combo.findData(self._default_scope_mode)
        if index >= 0:
            self._scope_combo.setCurrentIndex(index)
        self._scope_combo.currentIndexChanged.connect(self._refresh_preview)
        control_grid.addWidget(self._scope_combo, 1, 3)

        self._strength_label = QLabel("Match sensitivity", self)
        control_grid.addWidget(self._strength_label, 1, 4)
        self._strength_combo = QComboBox(self)
        self._strength_combo.addItem("Very Strict", "very_strict")
        self._strength_combo.addItem("Strict", "strict")
        self._strength_combo.addItem("Balanced", "balanced")
        self._strength_combo.addItem("Loose", "loose")
        balanced = self._strength_combo.findData("balanced")
        if balanced >= 0:
            self._strength_combo.setCurrentIndex(balanced)
        control_grid.addWidget(self._strength_combo, 1, 5)

        self._smoothing_label = QLabel("Smooth: 3", self)
        control_grid.addWidget(self._smoothing_label, 2, 0)
        self._smoothing_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._smoothing_slider.setRange(0, 12)
        self._smoothing_slider.setValue(3)
        self._smoothing_slider.valueChanged.connect(self._refresh_preview)
        self._smoothing_slider.valueChanged.connect(lambda value: self._smoothing_label.setText(f"Smooth: {value}"))
        control_grid.addWidget(self._smoothing_slider, 2, 1)

        self._points_label = QLabel("Points: 24", self)
        control_grid.addWidget(self._points_label, 2, 2)
        self._points_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._points_slider.setRange(8, 64)
        self._points_slider.setValue(24)
        self._points_slider.valueChanged.connect(self._refresh_preview)
        self._points_slider.valueChanged.connect(lambda value: self._points_label.setText(f"Points: {value}"))
        control_grid.addWidget(self._points_slider, 2, 3)

        self._fuzziness_label = QLabel("Fuzz: 35%", self)
        control_grid.addWidget(self._fuzziness_label, 2, 4)
        self._fuzziness_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._fuzziness_slider.setRange(0, 100)
        self._fuzziness_slider.setValue(35)
        self._fuzziness_slider.valueChanged.connect(self._refresh_preview)
        self._fuzziness_slider.valueChanged.connect(lambda value: self._fuzziness_label.setText(f"Fuzz: {value}%"))
        control_grid.addWidget(self._fuzziness_slider, 2, 5)

        self._outcome_label = QLabel("Matched events", self)
        control_grid.addWidget(self._outcome_label, 3, 0)
        self._outcome_combo = QComboBox(self)
        self._outcome_combo.addItem("Select matched events", "select")
        self._outcome_combo.addItem("Promote matched events", "promote")
        self._outcome_combo.addItem("Demote matched events", "demote")
        self._outcome_combo.addItem("Create new layer from matches", "create_layer")
        self._outcome_combo.currentIndexChanged.connect(self._refresh_preview)
        control_grid.addWidget(self._outcome_combo, 3, 1, 1, 2)

        self._layer_name_label = QLabel("New layer", self)
        control_grid.addWidget(self._layer_name_label, 3, 3)
        self._layer_name_edit = QLineEdit("Similar Events", self)
        control_grid.addWidget(self._layer_name_edit, 3, 4, 1, 2)
        layout.addLayout(control_grid)

        self._preview_scroll = QScrollArea(self)
        self._preview_scroll.setWidgetResizable(True)
        self._preview_scroll.setMinimumHeight(180)
        self._preview_widget = EventShapeComparisonPreviewWidget((), parent=self._preview_scroll)
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
        selected_layer_ids = self._selected_layer_ids_for_scope(str(self._scope_combo.currentData()))
        rows = self._build_shape_preview_rows(str(self._scope_combo.currentData()), scan_limit=None)
        matched_rows = [row for row in rows if row.is_anchor or row.is_match]
        if not matched_rows:
            matched_rows = [
                ShapePreviewRow(
                    event_ref=EventRef(self._layer_id, self._take_id, self._event_id),
                    label="Current",
                    shape=(),
                    score=1.0,
                    is_anchor=True,
                    is_match=True,
                )
            ]
        event_refs = [row.event_ref for row in matched_rows]
        return {
            "event_ids": [row.event_ref.event_id for row in matched_rows],
            "event_refs": event_refs,
            "matched_event_refs": event_refs,
            "match_count": len(event_refs),
            "anchor_layer_id": self._layer_id,
            "anchor_take_id": self._take_id,
            "selected_layer_ids": selected_layer_ids,
            "comparison_mode": str(self._mode_combo.currentData()),
            "scope_mode": str(self._scope_combo.currentData()),
            "match_strength": str(self._strength_combo.currentData()),
            "match_threshold": self._current_threshold(),
            "shape_smoothing": int(self._smoothing_slider.value()),
            "shape_control_points": int(self._points_slider.value()),
            "shape_fuzziness": int(self._fuzziness_slider.value()),
            "outcome_action": str(self._outcome_combo.currentData()),
            "new_layer_title": self._layer_name_edit.text().strip() or "Similar Events",
        }

    def _summary_text(self) -> str:
        candidates = self._candidate_events(self._default_scope_mode)
        comparison_count = sum(
            1
            for candidate in candidates
            if not (
                candidate.layer_id == self._layer_id
                and candidate.take_id == self._take_id
                and candidate.event_id == self._event_id
            )
        )
        layer_count = len({candidate.layer_id for candidate in candidates})
        take_count = len({candidate.take_id for candidate in candidates})
        event_word = "event" if comparison_count == 1 else "events"
        take_word = "take" if take_count == 1 else "takes"
        layer_word = "layer" if layer_count == 1 else "layers"
        return (
            f"Compare the selected event against {comparison_count} candidate {event_word} across "
            f"{take_count} {take_word} and {layer_count} {layer_word}."
        )

    def _refresh_preview(self) -> None:
        scope_mode = str(self._scope_combo.currentData())
        total_candidates = max(0, len(self._candidate_events(scope_mode)) - 1)
        rows = self._build_shape_preview_rows(scope_mode, scan_limit=self._scan_limit)
        self._preview_widget = EventShapeComparisonPreviewWidget(
            rows,
            smoothing=int(self._smoothing_slider.value()),
            control_points=int(self._points_slider.value()),
            fuzziness=int(self._fuzziness_slider.value()),
            threshold=self._current_threshold(),
            scan_total=total_candidates,
            scan_limit=max(0, len(rows) - 1) if self._scan_limit is None else min(self._scan_limit, total_candidates),
            match_count=sum(1 for row in rows[1:] if row.is_match),
            action_label=self._outcome_combo.currentText(),
            parent=self._preview_scroll,
        )
        self._preview_scroll.setWidget(self._preview_widget)

    def _build_shape_preview_rows(self, scope_mode: str, *, scan_limit: int | None = None) -> tuple[ShapePreviewRow, ...]:
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
        raw_anchor_shape = _shape_for_candidate(anchor)
        if not raw_anchor_shape:
            return ()
        anchor_shape = _shape_with_params(
            raw_anchor_shape,
            smoothing=int(self._smoothing_slider.value()),
            control_points=int(self._points_slider.value()),
        )
        rows = [
            ShapePreviewRow(
                event_ref=EventRef(anchor.layer_id, anchor.take_id, anchor.event_id),
                label=f"Current · {anchor.label}",
                shape=anchor_shape,
                score=1.0,
                is_anchor=True,
            )
        ]
        scanned = 0
        for candidate in candidates:
            if (
                candidate.layer_id == anchor.layer_id
                and candidate.take_id == anchor.take_id
                and candidate.event_id == anchor.event_id
            ):
                continue
            if scan_limit is not None and scanned >= scan_limit:
                break
            scanned += 1
            raw_candidate_shape = _shape_for_candidate(candidate)
            if not raw_candidate_shape:
                continue
            candidate_shape = _shape_with_params(
                raw_candidate_shape,
                smoothing=int(self._smoothing_slider.value()),
                control_points=int(self._points_slider.value()),
            )
            aligned = align_shape_to_reference(anchor_shape, candidate_shape)
            score = compare_shape_similarity(anchor_shape, candidate_shape)
            rows.append(
                ShapePreviewRow(
                    event_ref=EventRef(candidate.layer_id, candidate.take_id, candidate.event_id),
                    label=candidate.label,
                    shape=aligned,
                    score=score,
                    is_match=score >= self._current_threshold(),
                )
            )
        return tuple(rows)

    def set_scan_preview_limit(self, value: int | None) -> None:
        """Set a visible scan-progress cap for live app demos and automation captures."""

        self._scan_limit = None if value is None else max(0, int(value))
        self._refresh_preview()

    def _current_threshold(self) -> float:
        base = {
            "very_strict": 0.95,
            "strict": 0.90,
            "balanced": 0.78,
            "loose": 0.65,
        }.get(str(self._strength_combo.currentData()), 0.78)
        fuzz_relief = max(0.0, min(1.0, int(self._fuzziness_slider.value()) / 100.0)) * 0.18
        return max(0.0, min(1.0, base - fuzz_relief))

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


def _shape_with_params(
    shape: tuple[float, ...],
    *,
    smoothing: int,
    control_points: int,
) -> tuple[float, ...]:
    values = list(shape)
    if not values:
        return ()
    if smoothing > 0 and len(values) > 2:
        radius = max(1, int(smoothing))
        smoothed = []
        for index in range(len(values)):
            lo = max(0, index - radius)
            hi = min(len(values), index + radius + 1)
            smoothed.append(sum(values[lo:hi]) / max(1, hi - lo))
        values = smoothed
    point_count = max(4, int(control_points))
    if point_count != len(values):
        source_max = max(1, len(values) - 1)
        values = [
            _linear_sample(values, index * source_max / max(1, point_count - 1))
            for index in range(point_count)
        ]
    peak = max(values) if values else 0.0
    return tuple(float(value / peak) if peak > 1e-9 else 0.0 for value in values)


def _linear_sample(values: list[float], position: float) -> float:
    left = int(position)
    right = min(len(values) - 1, left + 1)
    blend = position - left
    return float(values[left] * (1.0 - blend) + values[right] * blend)


def _shape_for_candidate(candidate: _Candidate) -> tuple[float, ...]:
    if not candidate.audio_path:
        return ()
    path = Path(candidate.audio_path)
    if not path.exists():
        return ()
    end_seconds = candidate.end if candidate.end > candidate.start else candidate.start + 0.12
    sliced = read_mono_audio_slice(path, start_seconds=candidate.start, end_seconds=end_seconds)
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

"""Find-similar event comparison dialog.
Exists so users can tune shape-matching before selecting related timeline events.
Connects timeline presentation events to visual audio-shape comparison previews.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
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
from echozero.application.timeline.event_comparison_service import (
    TimbreFingerprintSettings,
    build_timbre_fingerprint_preview,
    compare_timbre_fingerprint_similarity,
    normalize_comparison_mode,
)
from echozero.application.timeline.event_similarity_mini_model import (
    AudioEventTrainingSample,
    list_timbre_mini_models,
    load_timbre_mini_model,
    train_timbre_mini_model,
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

from echozero.ui.style.qt import ensure_qt_theme_installed


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
        ensure_qt_theme_installed()
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
        painter.fillRect(rect, QColor("#101010"))
        painter.setPen(QPen(QColor("#8f8a84"), 1.2))
        painter.drawRoundedRect(QRectF(rect), 3.0, 3.0)
        if not self._rows:
            painter.setPen(QColor("#aaa49e"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "NO AUDIO COMPARISON TELEMETRY")
            painter.end()
            return

        anchor = self._rows[0]
        scanned_count = max(0, int(self._scan_limit))
        total_count = max(scanned_count, int(self._scan_total))
        left_width = min(360.0, max(285.0, rect.width() * 0.36))
        gap = 12.0
        left_panel = QRectF(rect.left() + 12, rect.top() + 12, left_width, rect.height() - 24)
        right_panel = QRectF(left_panel.right() + gap, rect.top() + 12, rect.right() - left_panel.right() - gap - 12, rect.height() - 24)

        painter.fillRect(left_panel, QColor("#171719"))
        painter.setPen(QPen(QColor("#8f8a84"), 1.0))
        painter.drawRoundedRect(left_panel, 3.0, 3.0)
        painter.setPen(QColor("#f6f3ee"))
        painter.drawText(left_panel.adjusted(14, 10, -14, -10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "SELECTED CLIP")
        painter.setPen(QColor("#d8d2cb"))
        painter.drawText(left_panel.adjusted(14, 32, -14, -10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, anchor.label)
        self._draw_shape(painter, QRectF(left_panel.left() + 14, left_panel.top() + 62, left_panel.width() - 28, 128), anchor.shape, QColor("#d8d2cb"), width=3.2)

        progress_rect = QRectF(left_panel.left() + 14, left_panel.top() + 204, left_panel.width() - 28, 92)
        painter.fillRect(progress_rect, QColor("#101010"))
        painter.setPen(QColor("#e8e2dc"))
        painter.drawText(progress_rect.adjusted(10, 4, -10, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "LIVE ITERATION")
        current_row = self._rows[-1] if len(self._rows) > 1 and scanned_count else None
        current_color = QColor("#8f8a84") if current_row is not None and current_row.is_match else QColor("#8f3a2f")
        painter.setPen(QColor("#f6f3ee"))
        painter.drawText(
            progress_rect.adjusted(10, 24, -10, -4),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            f"{scanned_count}/{total_count} scanned · {self._match_count} passes · {self._action_label}",
        )
        if current_row is not None:
            badge = QRectF(progress_rect.left() + 10, progress_rect.top() + 48, 22, 22)
            painter.fillRect(badge, current_color)
            painter.setPen(QPen(current_color, 2.0))
            painter.drawRoundedRect(badge, 2.0, 2.0)
            painter.setPen(QColor("#f6f3ee"))
            score = "--" if current_row.score is None else f"{current_row.score:.2f}"
            painter.drawText(
                progress_rect.adjusted(42, 48, -10, -4),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                f"current: {current_row.label} · {score}",
            )
        segment_top = progress_rect.top() + 76
        visible_segments = max(1, min(total_count, 28))
        segment_width = max(5.0, (progress_rect.width() - 20) / visible_segments)
        scanned_segments = round(visible_segments * (scanned_count / max(1, total_count)))
        for index in range(visible_segments):
            segment = QRectF(progress_rect.left() + 10 + index * segment_width + 1, segment_top, segment_width - 2, 8)
            painter.fillRect(segment, QColor("#8f8a84") if index < scanned_segments else QColor("#202022"))

        meter_top = progress_rect.bottom() + 16
        for index, (label, value, color) in enumerate(
            (
                (f"smooth {self._smoothing}", min(1.0, self._smoothing / 12.0), QColor("#7fd1ae")),
                (f"points {self._control_points}", min(1.0, self._control_points / 64.0), QColor("#685f67")),
                (f"fuzz {self._fuzziness}%", min(1.0, self._fuzziness / 100.0), QColor("#d8d2cb")),
                (f"min {self._threshold:.2f}", min(1.0, self._threshold), QColor("#8f8a84")),
            )
        ):
            y = meter_top + index * 34
            painter.setPen(QColor("#aaa49e"))
            painter.drawText(QRectF(left_panel.left() + 14, y, 88, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label.upper())
            bar = QRectF(left_panel.left() + 112, y + 5, left_panel.width() - 132, 8)
            painter.fillRect(bar, QColor("#202022"))
            painter.fillRect(QRectF(bar.left(), bar.top(), bar.width() * value, bar.height()), color)

        painter.fillRect(right_panel, QColor("#171719"))
        painter.setPen(QPen(QColor("#4a4749"), 1.0))
        painter.drawRoundedRect(right_panel, 3.0, 3.0)
        painter.setPen(QColor("#aaa49e"))
        painter.drawText(right_panel.adjusted(14, 10, -14, -10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "CANDIDATE CURVES · GOLD REFERENCE / CYAN EVENT")

        top = right_panel.top() + 38
        row_height = 48
        label_width = min(210, max(135, int(right_panel.width() * 0.31)))
        graph_left = right_panel.left() + label_width + 24
        graph_width = max(120, right_panel.right() - graph_left - 14)
        for index, row in enumerate(self._rows[1:]):
            y = top + index * row_height
            row_rect = QRectF(right_panel.left() + 10, y, right_panel.width() - 20, row_height - 7)
            is_current = index == max(0, min(len(self._rows) - 2, scanned_count - 1))
            verdict_color = QColor("#8f8a84") if row.is_match else QColor("#8f3a2f")
            verdict_fill = QColor(verdict_color)
            verdict_fill.setAlpha(44 if row.is_match else 34)
            painter.fillRect(row_rect, QColor("#202022") if index % 2 == 0 else QColor("#171719"))
            painter.fillRect(row_rect, verdict_fill)
            painter.setPen(QPen(verdict_color if is_current else QColor("#685f67"), 2.4 if is_current else 0.8))
            painter.drawRoundedRect(row_rect, 2.0, 2.0)
            score = "--" if row.score is None else f"{row.score:.2f}"
            badge_rect = QRectF(row_rect.left() + 10, row_rect.top() + 13, 14, 14)
            painter.fillRect(badge_rect, verdict_color)
            painter.setPen(QPen(verdict_color.lighter(135), 1.2))
            painter.drawRoundedRect(badge_rect, 2.0, 2.0)
            painter.setPen(QColor("#8f8a84") if row.is_match else QColor("#c86f5f"))
            painter.drawText(row_rect.adjusted(34, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"{row.label} · {score}")
            graph_rect = QRectF(graph_left, y + 8, graph_width, row_height - 24)
            graph_overlay = QColor(verdict_color)
            graph_overlay.setAlpha(26)
            painter.fillRect(graph_rect.adjusted(-4, -4, 4, 4), graph_overlay)
            painter.setPen(QPen(verdict_color, 1.0))
            painter.drawRoundedRect(graph_rect.adjusted(-4, -4, 4, 4), 2.0, 2.0)
            painter.setPen(QPen(QColor("#4a4749"), 1.0))
            painter.drawLine(QPointF(graph_rect.left(), graph_rect.center().y()), QPointF(graph_rect.right(), graph_rect.center().y()))
            self._draw_shape(painter, graph_rect, anchor.shape, QColor("#d8d2cb"), width=2.2)
            self._draw_shape(painter, graph_rect, row.shape, QColor("#7fd1ae"), width=2.0)
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
            painter.setPen(QColor("#aaa49e"))
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
        ensure_qt_theme_installed()
        self._presentation = presentation
        self._layer_id = layer_id
        self._take_id = take_id
        self._event_id = event_id
        self._default_scope_mode = _coerce_scope_mode(default_scope_mode)
        self._scan_limit: int | None = None
        self._audio_slice_cache: dict[str, tuple[np.ndarray, int]] = {}
        self._preview_cache: dict[tuple[str, str, str, str, float, float, int], tuple[float, ...]] = {}
        self._model_score_cache: dict[tuple[str, str, str, str], float] = {}
        self.setWindowTitle("Compare Events")

        layout = QVBoxLayout(self)
        control_grid = QGridLayout()
        self._summary = QLabel(self._summary_text(), self)
        self._summary.setWordWrap(True)
        control_grid.addWidget(self._summary, 0, 0, 1, 6)

        self._mode_label = QLabel("Comparison method", self)
        control_grid.addWidget(self._mode_label, 1, 0)
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem("Shape Envelope", "shape_envelope")
        self._mode_combo.addItem("Timbre Fingerprint", "timbre_fingerprint")
        self._mode_combo.addItem("Saved Mini-model", "timbre_mini_model")
        self._mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self._mode_combo.currentIndexChanged.connect(self._sync_method_labels)
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
        self._strength_combo.currentIndexChanged.connect(self._refresh_preview)
        control_grid.addWidget(self._strength_combo, 1, 5)

        self._smoothing_label = QLabel("Smooth: 3", self)
        control_grid.addWidget(self._smoothing_label, 2, 0)
        self._smoothing_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._smoothing_slider.setRange(0, 12)
        self._smoothing_slider.setValue(3)
        self._smoothing_slider.valueChanged.connect(self._refresh_preview)
        self._smoothing_slider.valueChanged.connect(lambda _value: self._sync_method_labels())
        control_grid.addWidget(self._smoothing_slider, 2, 1)

        self._points_label = QLabel("Points: 24", self)
        control_grid.addWidget(self._points_label, 2, 2)
        self._points_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._points_slider.setRange(8, 64)
        self._points_slider.setValue(24)
        self._points_slider.valueChanged.connect(self._refresh_preview)
        self._points_slider.valueChanged.connect(lambda _value: self._sync_method_labels())
        control_grid.addWidget(self._points_slider, 2, 3)

        self._fuzziness_label = QLabel("Fuzz: 35%", self)
        control_grid.addWidget(self._fuzziness_label, 2, 4)
        self._fuzziness_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._fuzziness_slider.setRange(0, 100)
        self._fuzziness_slider.setValue(35)
        self._fuzziness_slider.valueChanged.connect(self._refresh_preview)
        self._fuzziness_slider.valueChanged.connect(lambda _value: self._sync_method_labels())
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

        self._model_picker_label = QLabel("Saved model", self)
        control_grid.addWidget(self._model_picker_label, 4, 0)
        self._model_combo = QComboBox(self)
        self._populate_model_combo()
        self._model_combo.currentIndexChanged.connect(self._refresh_preview)
        self._model_combo.currentIndexChanged.connect(
            lambda _index: self._sync_model_status_label(self._train_model_checkbox.isChecked())
        )
        control_grid.addWidget(self._model_combo, 4, 1, 1, 2)

        self._train_model_checkbox = QCheckBox("Save local mini-model from matches", self)
        self._train_model_checkbox.setToolTip(
            "Saves a lightweight timbre prototype using the anchor and matched positives."
        )
        self._train_model_checkbox.toggled.connect(self._sync_model_status_label)
        control_grid.addWidget(self._train_model_checkbox, 5, 0, 1, 3)
        self._model_status_label = QLabel("Mini-model: off", self)
        control_grid.addWidget(self._model_status_label, 5, 3, 1, 3)
        layout.addLayout(control_grid)

        self._preview_scroll = QScrollArea(self)
        self._preview_scroll.setWidgetResizable(True)
        self._preview_scroll.setMinimumHeight(180)
        self._preview_widget = EventShapeComparisonPreviewWidget((), parent=self._preview_scroll)
        self._preview_scroll.setWidget(self._preview_widget)
        layout.addWidget(self._preview_scroll, stretch=1)
        self._sync_method_labels()
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
        requested_mode = normalize_comparison_mode(str(self._mode_combo.currentData()))
        selected_model_path = self._selected_model_path()
        comparison_mode = requested_mode
        model_error: str | None = None
        if requested_mode == "timbre_mini_model" and selected_model_path is None:
            # Keep OK safe when the operator previews saved-model mode before any
            # local model exists. Build the payload/matches with the same raw
            # timbre method we will dispatch, rather than changing mode after
            # rows have already been scored.
            comparison_mode = "timbre_fingerprint"
            model_error = "No saved mini-model selected"
        rows = self._build_preview_rows(
            str(self._scope_combo.currentData()),
            scan_limit=None,
            comparison_mode=comparison_mode,
        )
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
        payload: dict[str, object] = {
            "event_ids": [row.event_ref.event_id for row in matched_rows],
            "event_refs": event_refs,
            "matched_event_refs": event_refs,
            "match_count": len(event_refs),
            "anchor_layer_id": self._layer_id,
            "anchor_take_id": self._take_id,
            "selected_layer_ids": selected_layer_ids,
            "comparison_mode": comparison_mode,
            "scope_mode": str(self._scope_combo.currentData()),
            "match_strength": str(self._strength_combo.currentData()),
            "match_threshold": self._current_threshold(),
            "shape_smoothing": int(self._smoothing_slider.value()),
            "shape_control_points": int(self._points_slider.value()),
            "shape_fuzziness": int(self._fuzziness_slider.value()),
            "outcome_action": str(self._outcome_combo.currentData()),
            "new_layer_title": self._layer_name_edit.text().strip() or "Similar Events",
        }
        if comparison_mode == "timbre_mini_model" and selected_model_path is not None:
            payload["mini_model_path"] = str(selected_model_path)
            payload["comparison_options"] = {"artifact_path": str(selected_model_path)}
        if model_error is not None:
            payload["mini_model_error"] = model_error
        if self._train_model_checkbox.isChecked():
            training_payload = self._train_mini_model_payload(matched_rows)
            if "mini_model_path" in payload and "mini_model_path" in training_payload:
                training_payload["trained_mini_model_path"] = training_payload.pop("mini_model_path")
            payload.update(training_payload)
        return payload

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
        rows = self._build_preview_rows(scope_mode, scan_limit=self._scan_limit)
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
        return self._build_preview_rows(scope_mode, scan_limit=scan_limit, comparison_mode="shape_envelope")

    def _build_preview_rows(
        self,
        scope_mode: str,
        *,
        scan_limit: int | None = None,
        comparison_mode: str | None = None,
    ) -> tuple[ShapePreviewRow, ...]:
        mode = normalize_comparison_mode(comparison_mode or str(self._mode_combo.currentData()))
        preview_mode = "timbre_fingerprint" if mode == "timbre_mini_model" else mode
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
        if mode == "timbre_mini_model":
            selected_model = self._selected_model_payload()
            raw_anchor_preview = (
                tuple(float(value) for value in selected_model["centroid"])
                if selected_model is not None
                else ()
            )
        else:
            raw_anchor_preview = self._preview_for_candidate(anchor, mode=preview_mode)
        if not raw_anchor_preview:
            return ()
        anchor_shape = _display_curve_with_params(
            raw_anchor_preview,
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
            raw_candidate_preview = self._preview_for_candidate(candidate, mode=preview_mode)
            if not raw_candidate_preview:
                continue
            candidate_shape = _display_curve_with_params(
                raw_candidate_preview,
                smoothing=int(self._smoothing_slider.value()),
                control_points=int(self._points_slider.value()),
            )
            if mode == "timbre_mini_model":
                aligned = candidate_shape
                score = self._score_candidate_with_selected_model(candidate)
            elif mode == "timbre_fingerprint":
                aligned = candidate_shape
                score = compare_timbre_fingerprint_similarity(raw_anchor_preview, raw_candidate_preview)
            else:
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

    def _preview_for_candidate(self, candidate: _Candidate, *, mode: str) -> tuple[float, ...]:
        normalized = "timbre_fingerprint" if mode == "timbre_mini_model" else normalize_comparison_mode(mode)
        cache_key = (
            normalized,
            str(candidate.layer_id),
            str(candidate.take_id),
            str(candidate.event_id),
            float(candidate.start),
            float(candidate.end),
            64,
        )
        cached = self._preview_cache.get(cache_key)
        if cached is not None:
            return cached
        preview = _preview_for_candidate(candidate, mode=normalized, audio_cache=self._audio_slice_cache)
        self._preview_cache[cache_key] = preview
        return preview

    def _selected_model_path(self) -> Path | None:
        if not hasattr(self, "_model_combo"):
            return None
        value = self._model_combo.currentData()
        if not value:
            return None
        return Path(str(value))

    def _selected_model_payload(self) -> dict[str, object] | None:
        path = self._selected_model_path()
        if path is None:
            return None
        try:
            return load_timbre_mini_model(path)
        except (OSError, ValueError):
            return None

    def _score_candidate_with_selected_model(self, candidate: _Candidate) -> float:
        path = self._selected_model_path()
        if path is None:
            self._model_status_label.setText("Mini-model: choose a saved model")
            return 0.0
        cache_key = (
            str(path),
            str(candidate.layer_id),
            str(candidate.take_id),
            str(candidate.event_id),
        )
        cached = self._model_score_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            payload = load_timbre_mini_model(path)
            settings_payload = payload.get("settings", {}) if isinstance(payload.get("settings"), dict) else {}
            fingerprint = build_timbre_fingerprint_preview(
                audio_path=candidate.audio_path,
                start_seconds=candidate.start,
                end_seconds=candidate.end if candidate.end > candidate.start else candidate.start + 0.12,
                settings=TimbreFingerprintSettings(
                    sample_count=int(settings_payload.get("sample_count", 64)),
                    padding_ms=float(settings_payload.get("padding_ms", 20.0)),
                ),
                audio_cache=self._audio_slice_cache,
            )
            score = 0.0 if fingerprint is None else compare_timbre_fingerprint_similarity(
                tuple(float(value) for value in payload["centroid"]),
                fingerprint,
            )
        except (OSError, ValueError):
            score = 0.0
            self._model_status_label.setText("Mini-model: saved model unavailable")
        self._model_score_cache[cache_key] = score
        return score

    def _train_mini_model_payload(self, rows: list[ShapePreviewRow]) -> dict[str, object]:
        samples = self._training_samples_for_rows(rows)
        if not samples:
            self._model_status_label.setText("Mini-model: no matched audio samples")
            return {"mini_model_requested": True, "mini_model_error": "No matched audio samples"}
        try:
            result = train_timbre_mini_model(anchor_sample=samples[0], positive_samples=samples)
        except ValueError as exc:
            self._model_status_label.setText("Mini-model: training skipped")
            return {"mini_model_requested": True, "mini_model_error": str(exc)}
        self._model_status_label.setText(f"Mini-model: saved {result.artifact_path.name}")
        if Path(result.artifact_path).exists():
            self._populate_model_combo()
            index = self._model_combo.findData(str(result.artifact_path))
            if index >= 0:
                self._model_combo.setCurrentIndex(index)
        return {
            "mini_model_requested": True,
            "mini_model_path": str(result.artifact_path),
            "mini_model_sample_count": result.positive_sample_count,
        }

    def _training_samples_for_rows(self, rows: list[ShapePreviewRow]) -> list[AudioEventTrainingSample]:
        candidates = list(self._candidate_events(str(self._scope_combo.currentData())))
        anchor = self._find_event_candidate(self._layer_id, self._take_id, self._event_id)
        if anchor is not None:
            candidates.append(anchor)
        candidate_by_ref = {
            (str(candidate.layer_id), str(candidate.take_id), str(candidate.event_id)): candidate
            for candidate in candidates
        }
        samples: list[AudioEventTrainingSample] = []
        for row in rows:
            key = (str(row.event_ref.layer_id), str(row.event_ref.take_id), str(row.event_ref.event_id))
            candidate = candidate_by_ref.get(key)
            if candidate is None or not candidate.audio_path:
                continue
            samples.append(_training_sample_from_candidate(candidate))
        return samples

    def set_scan_preview_limit(self, value: int | None) -> None:
        """Set a visible scan-progress cap for live app demos and automation captures."""

        self._scan_limit = None if value is None else max(0, int(value))
        self._refresh_preview()

    def _populate_model_combo(self) -> None:
        self._model_combo.clear()
        self._model_combo.addItem("No saved mini-model", None)
        try:
            entries = list_timbre_mini_models()
        except OSError:
            entries = ()
        for entry in entries:
            count = entry.positive_sample_count
            label = f"{entry.label} · {count} sample{'s' if count != 1 else ''}"
            self._model_combo.addItem(label, str(entry.artifact_path))

    def _sync_method_labels(self) -> None:
        mode = normalize_comparison_mode(str(self._mode_combo.currentData()))
        is_shape = mode == "shape_envelope"
        is_model = mode == "timbre_mini_model"
        self._smoothing_label.setText(f"{'Smooth' if is_shape else 'Display smooth'}: {self._smoothing_slider.value()}")
        self._points_label.setText(f"{'Points' if is_shape else 'Display points'}: {self._points_slider.value()}")
        self._fuzziness_label.setText(f"{'Fuzz' if is_shape else 'Score tolerance'}: {self._fuzziness_slider.value()}%")
        self._model_picker_label.setVisible(is_model)
        self._model_combo.setVisible(is_model)
        if is_model and self._model_combo.currentData() is None and self._model_combo.count() > 1:
            self._model_combo.setCurrentIndex(1)
        self._sync_model_status_label(self._train_model_checkbox.isChecked())

    def _sync_model_status_label(self, checked: bool) -> None:
        mode = normalize_comparison_mode(str(self._mode_combo.currentData()))
        if mode == "timbre_mini_model":
            selected = self._selected_model_path()
            if selected is None:
                status = "Mini-model: choose a saved model"
            else:
                status = f"Mini-model: using {selected.name}"
        else:
            status = "Mini-model: will save on OK" if checked else "Mini-model: off"
        if checked and mode != "timbre_mini_model":
            status = "Mini-model: will save on OK"
        elif checked and mode == "timbre_mini_model":
            status += " · will also save matches on OK"
        self._model_status_label.setText(status)

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


def _display_curve_with_params(
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


def _shape_with_params(
    shape: tuple[float, ...],
    *,
    smoothing: int,
    control_points: int,
) -> tuple[float, ...]:
    return _display_curve_with_params(
        shape,
        smoothing=smoothing,
        control_points=control_points,
    )


def _linear_sample(values: list[float], position: float) -> float:
    left = int(position)
    right = min(len(values) - 1, left + 1)
    blend = position - left
    return float(values[left] * (1.0 - blend) + values[right] * blend)


def _shape_for_candidate(candidate: _Candidate, *, audio_cache: dict | None = None) -> tuple[float, ...]:
    if not candidate.audio_path:
        return ()
    path = Path(candidate.audio_path)
    if not path.exists():
        return ()
    end_seconds = candidate.end if candidate.end > candidate.start else candidate.start + 0.12
    sliced = _cached_audio_slice(
        audio_cache,
        str(path),
        start_seconds=candidate.start,
        end_seconds=end_seconds,
    )
    if sliced is None:
        return ()
    samples, _sample_rate = sliced
    return audio_shape_preview(samples, sample_count=64)


def _preview_for_candidate(
    candidate: _Candidate,
    *,
    mode: str,
    audio_cache: dict | None = None,
) -> tuple[float, ...]:
    if mode == "timbre_fingerprint":
        return _timbre_fingerprint_for_candidate(candidate, audio_cache=audio_cache)
    return _shape_for_candidate(candidate, audio_cache=audio_cache)


def _timbre_fingerprint_for_candidate(
    candidate: _Candidate,
    *,
    audio_cache: dict | None = None,
) -> tuple[float, ...]:
    if not candidate.audio_path:
        return ()
    path = Path(candidate.audio_path)
    if not path.exists():
        return ()
    end_seconds = candidate.end if candidate.end > candidate.start else candidate.start + 0.12
    fingerprint = build_timbre_fingerprint_preview(
        audio_path=str(path),
        start_seconds=candidate.start,
        end_seconds=end_seconds,
        settings=TimbreFingerprintSettings(sample_count=64, padding_ms=20.0),
        audio_cache=audio_cache,
    )
    return fingerprint or ()


def _cached_audio_slice(
    audio_cache: dict | None,
    audio_path: str,
    *,
    start_seconds: float,
    end_seconds: float,
) -> tuple[np.ndarray, int] | None:
    cache_key = f"{audio_path}|{float(start_seconds):.6f}|{float(end_seconds):.6f}"
    if audio_cache is not None and cache_key in audio_cache:
        return audio_cache[cache_key]
    sliced = read_mono_audio_slice(audio_path, start_seconds=start_seconds, end_seconds=end_seconds)
    if sliced is not None and audio_cache is not None:
        audio_cache[cache_key] = sliced
    return sliced


def _training_sample_from_candidate(candidate: _Candidate) -> AudioEventTrainingSample:
    return AudioEventTrainingSample(
        event_ref=EventRef(candidate.layer_id, candidate.take_id, candidate.event_id),
        label=candidate.label,
        audio_path=candidate.audio_path,
        start_seconds=candidate.start,
        end_seconds=candidate.end if candidate.end > candidate.start else candidate.start + 0.12,
    )


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

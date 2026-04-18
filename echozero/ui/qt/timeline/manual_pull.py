"""Manual pull timeline dialog surfaces for MA3 import flows.
Exists to keep the main timeline widget free of popup-specific rendering and selection logic.
Connects manual pull event picking to the timeline transfer action router.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF, QWheelEvent
from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QScrollBar, QVBoxLayout, QWidget

from echozero.ui.qt.timeline.blocks.ruler import visible_ruler_seconds
from echozero.ui.qt.timeline.style import TIMELINE_STYLE


def format_manual_pull_seconds(value: float) -> str:
    """Format seconds for the manual pull dialog timeline."""

    return f"{value:.2f}s"


@dataclass(slots=True, frozen=True)
class ManualPullTimelineSelectionResult:
    """User selections returned from the manual pull timeline dialog."""

    selected_event_ids: list[str]
    target_layer_id: object
    import_mode: str = "new_take"


class ManualPullTimelineCanvas(QWidget):
    """Scrollable event-selection canvas used inside the manual pull dialog."""

    selection_changed = pyqtSignal(object)
    zoom_changed = pyqtSignal(float)

    def __init__(self, events, selected_event_ids: list[str] | None = None, parent=None):
        super().__init__(parent)
        self._events = list(events)
        self._selected_event_ids = list(selected_event_ids or [])
        self._anchor_index: int | None = self._selected_index() if self._selected_event_ids else None
        self._rects: list[QRectF] = []
        self._left_padding = 16.0
        self._right_padding = 16.0
        self._top_padding = 18.0
        self._bar_height = 24.0
        self._base_pixels_per_second = 140.0
        self._zoom_factor = 1.0
        self.setMinimumHeight(150)
        self._sync_timeline_geometry()

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    @property
    def pixels_per_second(self) -> float:
        return max(1.0, self._base_pixels_per_second * self._zoom_factor)

    def zoom_in(self) -> None:
        self.set_zoom_factor(self._zoom_factor * 1.15)

    def zoom_out(self) -> None:
        self.set_zoom_factor(self._zoom_factor / 1.15)

    def reset_zoom(self) -> None:
        self.set_zoom_factor(1.0)

    def set_zoom_factor(self, value: float) -> None:
        bounded = max(0.35, min(4.0, float(value)))
        if abs(bounded - self._zoom_factor) < 1e-6:
            return
        self._zoom_factor = bounded
        self._sync_timeline_geometry()
        self.zoom_changed.emit(self._zoom_factor)

    def selected_event_ids(self) -> list[str]:
        ordered_ids = [event.event_id for event in self._events if event.event_id in self._selected_event_ids]
        return ordered_ids

    def set_selected_event_ids(self, event_ids: list[str]) -> None:
        self._selected_event_ids = list(dict.fromkeys(event_ids))
        self._anchor_index = self._selected_index()
        self.selection_changed.emit(self.selected_event_ids())
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        index = self._index_at(event.position())
        if index is None:
            super().mousePressEvent(event)
            return

        modifiers = event.modifiers()
        has_toggle = bool(modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        event_id = self._events[index].event_id

        if has_shift and self._anchor_index is not None:
            start_index = min(self._anchor_index, index)
            end_index = max(self._anchor_index, index)
            self._selected_event_ids = [candidate.event_id for candidate in self._events[start_index : end_index + 1]]
        elif has_toggle:
            selected = list(self._selected_event_ids)
            if event_id in selected:
                selected.remove(event_id)
            else:
                selected.append(event_id)
            self._selected_event_ids = selected
            self._anchor_index = index
        else:
            self._selected_event_ids = [event_id]
            self._anchor_index = index

        self.selection_changed.emit(self.selected_event_ids())
        self.update()
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return

        has_primary = bool(event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        if has_primary:
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
            return

        scroll_area = self._find_scroll_area()
        if scroll_area is not None:
            bar = scroll_area.horizontalScrollBar()
            bar.setValue(bar.value() - delta)
            event.accept()
            return

        super().wheelEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#11161b"))

        track_pen = QPen(QColor("#2b3642"))
        track_pen.setWidth(1)
        painter.setPen(track_pen)
        baseline_y = self.height() * 0.5
        painter.drawLine(int(self._left_padding), int(baseline_y), max(int(self._left_padding), self.width() - int(self._right_padding)), int(baseline_y))

        self._rects = self._compute_event_rects()
        metrics = painter.fontMetrics()
        for index, (event_model, rect) in enumerate(zip(self._events, self._rects)):
            is_selected = event_model.event_id in self._selected_event_ids
            fill = QColor("#5cb2ff" if is_selected else "#475569")
            stroke = QColor("#d7ebff" if is_selected else "#90a2b5")
            painter.setPen(QPen(stroke, 1.5))
            painter.setBrush(fill)
            painter.drawPolygon(self._diamond_polygon(rect))

            next_left = self._rects[index + 1].left() if index + 1 < len(self._rects) else self.width() - self._right_padding
            label_left = rect.center().x() + (min(rect.width() - 6.0, rect.height()) * 0.5) + 8.0
            label_width = max(72.0, next_left - label_left - 8.0)
            label_rect = QRectF(
                label_left,
                rect.top() - 2.0,
                max(0.0, min(label_width, (self.width() - self._right_padding) - label_left)),
                rect.height() + 4.0,
            )
            painter.setPen(QColor("#08111a" if is_selected else "#eef4ff"))
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                metrics.elidedText(event_model.label, Qt.TextElideMode.ElideRight, int(label_rect.width())),
            )

            if event_model.start is not None:
                painter.setPen(QColor("#c9d6e2"))
                footer = format_manual_pull_seconds(event_model.start)
                if event_model.end is not None:
                    footer = f"{footer}-{format_manual_pull_seconds(event_model.end)}"
                painter.drawText(
                    QRectF(rect.left(), rect.bottom() + 4.0, rect.width(), 14.0),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    footer,
                )

    def _find_scroll_area(self) -> QScrollArea | None:
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parent()
        return None

    def _selected_index(self) -> int | None:
        if not self._selected_event_ids:
            return None
        first_id = self._selected_event_ids[-1]
        for index, event in enumerate(self._events):
            if event.event_id == first_id:
                return index
        return None

    def _index_at(self, pos) -> int | None:
        for index, rect in enumerate(self._rects or self._compute_event_rects()):
            if rect.contains(pos):
                return index
        return None

    @staticmethod
    def _is_timed_event(event) -> bool:
        return event.start is not None

    def _timed_events(self) -> list:
        return [event for event in self._events if self._is_timed_event(event)]

    @staticmethod
    def _one_shot_seconds() -> float:
        return 0.30

    def _timed_event_axis(self) -> tuple[list, float, float]:
        timed = self._timed_events()
        if not timed:
            return [], 0.0, self._one_shot_seconds()
        min_start = min(event.start for event in timed)
        max_start = max(event.start for event in timed)
        span = max(self._one_shot_seconds(), (max_start - min_start) + self._one_shot_seconds())
        return timed, min_start, span

    def _timeline_bounds(self) -> tuple[float, float]:
        timed, min_start, _span = self._timed_event_axis()
        if not timed:
            count = max(1, len(self._events))
            return 0.0, max(1.0, count * 0.75)
        one_shot_seconds = self._one_shot_seconds()
        max_end = max(max(event.end if event.end is not None else event.start, event.start + one_shot_seconds) for event in timed)
        return min_start, max(max_end, min_start + one_shot_seconds)

    def _sync_timeline_geometry(self) -> None:
        start, end = self._timeline_bounds()
        span = max(1.0, end - start)
        content_width = int((span * self.pixels_per_second) + self._left_padding + self._right_padding + 80.0)
        self.setMinimumWidth(max(640, content_width))
        self.updateGeometry()
        self.update()

    def _compute_event_rects(self) -> list[QRectF]:
        if not self._events:
            return []

        content_width = max(120.0, self.width() - self._left_padding - self._right_padding)
        one_shot_seconds = self._one_shot_seconds()
        one_shot_width = max(44.0, self.pixels_per_second * one_shot_seconds)
        y = max(self._top_padding, (self.height() * 0.5) - (self._bar_height * 0.5))

        timed, min_start, span = self._timed_event_axis()
        rects: list[QRectF] = []
        if timed:
            max_x = self._left_padding + content_width - one_shot_width
            for index, event in enumerate(self._events):
                if not self._is_timed_event(event):
                    slot_width = content_width / max(1, len(self._events))
                    x = self._left_padding + (index * slot_width)
                else:
                    start_ratio = (event.start - min_start) / span
                    x = self._left_padding + (start_ratio * content_width)
                x = min(x, max_x)
                rects.append(QRectF(x, y, one_shot_width, self._bar_height))
            return rects

        slot_width = content_width / max(1, len(self._events))
        for index, _event in enumerate(self._events):
            x = self._left_padding + (index * slot_width) + 4.0
            rects.append(QRectF(x, y, max(one_shot_width, slot_width - 8.0), self._bar_height))
        return rects

    def _diamond_polygon(self, rect: QRectF) -> QPolygonF:
        diamond_width = min(rect.width() - 6.0, rect.height())
        diamond_height = max(10.0, rect.height() - 4.0)
        half_width = diamond_width * 0.5
        half_height = diamond_height * 0.5
        center = rect.center()
        return QPolygonF(
            [
                QPointF(center.x(), center.y() - half_height),
                QPointF(center.x() + half_width, center.y()),
                QPointF(center.x(), center.y() + half_height),
                QPointF(center.x() - half_width, center.y()),
            ]
        )


class ManualPullTimelineRuler(QWidget):
    """Ruler surface paired with the manual pull timeline canvas."""

    def __init__(self, canvas: ManualPullTimelineCanvas, scroll_bar: QScrollBar, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self._scroll_bar = scroll_bar
        self.setMinimumHeight(24)
        self.setMaximumHeight(24)
        self._canvas.zoom_changed.connect(lambda _zoom: self.update())
        self._scroll_bar.valueChanged.connect(lambda _value: self.update())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        rect = QRectF(0, 0, self.width(), self.height())
        painter.fillRect(rect, QColor(TIMELINE_STYLE.ruler.background_hex))
        painter.fillRect(QRectF(0, rect.bottom() - 1, rect.width(), 1), QColor(TIMELINE_STYLE.ruler.divider_hex))

        scroll_x = float(self._scroll_bar.value())
        marks = visible_ruler_seconds(
            scroll_x=scroll_x,
            pixels_per_second=self._canvas.pixels_per_second,
            content_width=max(1.0, rect.width()),
            content_start_x=0.0,
        )
        for second, x in marks:
            painter.setPen(QPen(QColor(TIMELINE_STYLE.ruler.tick_hex), 1))
            painter.drawLine(int(x), int(rect.bottom()) - 8, int(x), int(rect.bottom()))
            painter.setPen(QColor(TIMELINE_STYLE.ruler.label_hex))
            painter.drawText(int(x) + 3, int(rect.top()) + 12, f"{second}")


class ManualPullTimelineDialog(QDialog):
    """Dialog for selecting MA3 source events and the EZ import target."""

    def __init__(
        self,
        *,
        source_track_label: str,
        events,
        selected_event_ids: list[str] | None,
        available_targets,
        selected_target_layer_id,
        selected_import_mode: str = "new_take",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Import from MA3")
        self.resize(980, 440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        layout.addWidget(QLabel(f"Source track: {source_track_label}", self))

        help_label = QLabel(
            "Select source events on the timeline and choose the destination EZ layer below. "
            "Click to select. Ctrl/Cmd toggles. Shift selects a range. Ctrl/Cmd + wheel zooms; wheel scrolls timeline.",
            self,
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom", self))
        self._zoom_out_btn = QPushButton("-", self)
        self._zoom_in_btn = QPushButton("+", self)
        self._zoom_reset_btn = QPushButton("Reset", self)
        self._zoom_value_label = QLabel("100%", self)
        zoom_row.addWidget(self._zoom_out_btn)
        zoom_row.addWidget(self._zoom_in_btn)
        zoom_row.addWidget(self._zoom_reset_btn)
        zoom_row.addWidget(self._zoom_value_label)
        zoom_row.addStretch(1)
        layout.addLayout(zoom_row)

        self._canvas = ManualPullTimelineCanvas(events, selected_event_ids=selected_event_ids, parent=self)
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll_area.setWidget(self._canvas)
        self._scroll_area.setMinimumHeight(180)

        self._timeline_scroll = self._scroll_area.horizontalScrollBar()
        self._ruler = ManualPullTimelineRuler(self._canvas, self._timeline_scroll, self)
        layout.addWidget(self._ruler)
        layout.addWidget(self._scroll_area)

        self._selection_label = QLabel(self)
        layout.addWidget(self._selection_label)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target EZ layer", self))
        self._target_combo = QComboBox(self)
        for target in available_targets:
            self._target_combo.addItem(target.name, target.layer_id)
        if selected_target_layer_id is not None:
            for index in range(self._target_combo.count()):
                if self._target_combo.itemData(index) == selected_target_layer_id:
                    self._target_combo.setCurrentIndex(index)
                    break
        target_row.addWidget(self._target_combo, 1)
        layout.addLayout(target_row)

        import_row = QHBoxLayout()
        import_row.addWidget(QLabel("Import mode", self))
        self._import_mode_combo = QComboBox(self)
        self._import_mode_combo.addItem("Import as New Take", "new_take")
        self._import_mode_combo.addItem("Import to Main", "main")
        for index in range(self._import_mode_combo.count()):
            if self._import_mode_combo.itemData(index) == selected_import_mode:
                self._import_mode_combo.setCurrentIndex(index)
                break
        import_row.addWidget(self._import_mode_combo, 1)
        layout.addLayout(import_row)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._zoom_out_btn.clicked.connect(self._zoom_out)
        self._zoom_in_btn.clicked.connect(self._zoom_in)
        self._zoom_reset_btn.clicked.connect(self._reset_zoom)
        self._canvas.selection_changed.connect(self._refresh_state)
        self._target_combo.currentIndexChanged.connect(self._refresh_state)
        self._canvas.zoom_changed.connect(self._refresh_zoom_label)
        self._refresh_zoom_label(self._canvas.zoom_factor)
        self._refresh_state()

    def selected_event_ids(self) -> list[str]:
        """Return the currently selected MA3 event ids in display order."""

        return self._canvas.selected_event_ids()

    def selected_target_layer_id(self):
        """Return the currently selected EZ target layer id."""

        return self._target_combo.currentData()

    def selected_import_mode(self) -> str:
        """Return the import mode chosen for the current pull selection."""

        return str(self._import_mode_combo.currentData() or "new_take")

    def accept(self) -> None:
        """Reject submission until the dialog has both event and layer selections."""

        if not self.selected_event_ids():
            QMessageBox.warning(self, "Import from MA3", "Select at least one source event.")
            return
        if self.selected_target_layer_id() is None:
            QMessageBox.warning(self, "Import from MA3", "Select a target EZ layer.")
            return
        super().accept()

    def _zoom_in(self) -> None:
        self._canvas.zoom_in()
        self._ruler.update()

    def _zoom_out(self) -> None:
        self._canvas.zoom_out()
        self._ruler.update()

    def _reset_zoom(self) -> None:
        self._canvas.reset_zoom()
        self._ruler.update()

    def _refresh_zoom_label(self, zoom_factor: float) -> None:
        self._zoom_value_label.setText(f"{int(round(zoom_factor * 100))}%")

    def _refresh_state(self, *_args) -> None:
        selected_count = len(self.selected_event_ids())
        noun = "event" if selected_count == 1 else "events"
        self._selection_label.setText(f"Selected: {selected_count} {noun}")
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(bool(self.selected_event_ids()) and self.selected_target_layer_id() is not None)

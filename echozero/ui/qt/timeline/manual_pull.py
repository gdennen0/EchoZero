"""Manual pull timeline dialog surfaces for MA3 import flows.
Exists to keep the main timeline widget free of popup-specific rendering and selection logic.
Connects manual pull event picking to the timeline transfer action router.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt, QSignalBlocker, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF, QWheelEvent
from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QScrollBar, QVBoxLayout, QWidget

from echozero.application.shared.ids import LayerId
from echozero.ui.qt.timeline.manual_pull_source_browser import (
    ManualPullSourceBrowser,
    ManualPullTimecodePicker,
)
from echozero.ui.qt.timeline.blocks.ruler import visible_ruler_seconds
from echozero.ui.qt.timeline.style import TIMELINE_STYLE


def format_manual_pull_seconds(value: float) -> str:
    """Format seconds for the manual pull dialog timeline."""

    return f"{value:.2f}s"


def _is_new_layer_pull_target(target_layer_id: LayerId | None) -> bool:
    target_text = str(target_layer_id or "").strip()
    return target_text.startswith("__manual_pull__:create_new_layer")


def _pull_import_mode_for_target(target_layer_id: LayerId | None) -> str:
    return "main" if _is_new_layer_pull_target(target_layer_id) else "new_take"


@dataclass(slots=True, frozen=True)
class ManualPullTimelineSelectionResult:
    """User selections returned from the manual pull timeline dialog."""

    selected_event_ids: list[str]
    target_layer_id: LayerId | None
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

    def set_events(self, events, *, selected_event_ids: list[str] | None = None) -> None:
        self._events = list(events)
        self._selected_event_ids = list(dict.fromkeys(selected_event_ids or []))
        self._anchor_index = self._selected_index() if self._selected_event_ids else None
        self._rects = []
        self._sync_timeline_geometry()
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
        self._import_mode_combo.setEnabled(False)
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
        self._target_combo.currentIndexChanged.connect(self._handle_target_changed)
        self._canvas.zoom_changed.connect(self._refresh_zoom_label)
        self._sync_import_mode_with_target(fallback_mode=selected_import_mode)
        self._refresh_zoom_label(self._canvas.zoom_factor)
        self._refresh_state()

    def selected_event_ids(self) -> list[str]:
        """Return the currently selected MA3 event ids in display order."""

        return self._canvas.selected_event_ids()

    def selected_target_layer_id(self) -> LayerId | None:
        """Return the currently selected EZ target layer id."""

        value = self._target_combo.currentData()
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return LayerId(stripped)
        return None

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

    def _handle_target_changed(self) -> None:
        self._sync_import_mode_with_target()
        self._refresh_state()

    def _sync_import_mode_with_target(self, *, fallback_mode: str | None = None) -> None:
        target_layer_id = self.selected_target_layer_id()
        import_mode = (
            _pull_import_mode_for_target(target_layer_id)
            if target_layer_id is not None
            else str(fallback_mode or "new_take")
        )
        with QSignalBlocker(self._import_mode_combo):
            for index in range(self._import_mode_combo.count()):
                if self._import_mode_combo.itemData(index) == import_mode:
                    self._import_mode_combo.setCurrentIndex(index)
                    break

    def _refresh_state(self, *_args) -> None:
        selected_count = len(self.selected_event_ids())
        noun = "event" if selected_count == 1 else "events"
        self._selection_label.setText(f"Selected: {selected_count} {noun}")
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(bool(self.selected_event_ids()) and self.selected_target_layer_id() is not None)


class ManualPullWorkspaceDialog(QDialog):
    """Workspace for browsing MA3 timecodes, track groups, and tracks before import."""

    timecode_selected = pyqtSignal(int)
    track_group_selected = pyqtSignal(int)
    track_selected = pyqtSignal(str)
    target_layer_selected = pyqtSignal(object)
    import_mode_selected = pyqtSignal(str)
    event_selection_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Event Layer from MA3")
        self.resize(1120, 560)
        self._syncing = False
        self._selected_timecode_no: int | None = None
        self._selected_track_group_no: int | None = None
        self._selected_source_track_coord_value: str | None = None
        self._track_labels_by_coord: dict[str, str] = {}
        self._track_group_labels_by_no: dict[int, str] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        intro = QLabel(
            "Pick a timecode pool first, then click from a full view of its MA3 track groups "
            "and tracks. The timeline below always previews the currently selected source track.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        timecode_row = QHBoxLayout()
        timecode_row.addWidget(QLabel("Timecode pool", self))
        self._timecode_picker = ManualPullTimecodePicker(self)
        timecode_row.addWidget(self._timecode_picker, 1)
        layout.addLayout(timecode_row)

        source_header = QLabel("Track groups and tracks", self)
        layout.addWidget(source_header)

        self._source_browser = ManualPullSourceBrowser(self)
        self._source_browser_scroll = QScrollArea(self)
        self._source_browser_scroll.setWidgetResizable(True)
        self._source_browser_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._source_browser_scroll.setWidget(self._source_browser)
        self._source_browser_scroll.setMinimumHeight(210)
        layout.addWidget(self._source_browser_scroll)

        self._source_summary = QLabel("Source: Select an MA3 track", self)
        self._source_summary.setWordWrap(True)
        layout.addWidget(self._source_summary)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom", self))
        self._zoom_out_btn = QPushButton("-", self)
        self._zoom_in_btn = QPushButton("+", self)
        self._zoom_reset_btn = QPushButton("Reset", self)
        self._zoom_value_label = QLabel("100%", self)
        self._select_all_btn = QPushButton("Select All", self)
        self._clear_selection_btn = QPushButton("Clear", self)
        zoom_row.addWidget(self._zoom_out_btn)
        zoom_row.addWidget(self._zoom_in_btn)
        zoom_row.addWidget(self._zoom_reset_btn)
        zoom_row.addWidget(self._zoom_value_label)
        zoom_row.addStretch(1)
        zoom_row.addWidget(self._select_all_btn)
        zoom_row.addWidget(self._clear_selection_btn)
        layout.addLayout(zoom_row)

        self._canvas = ManualPullTimelineCanvas([], selected_event_ids=[], parent=self)
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll_area.setWidget(self._canvas)
        self._scroll_area.setMinimumHeight(200)
        self._timeline_scroll = self._scroll_area.horizontalScrollBar()
        self._ruler = ManualPullTimelineRuler(self._canvas, self._timeline_scroll, self)
        layout.addWidget(self._ruler)
        layout.addWidget(self._scroll_area)

        self._selection_label = QLabel("Selected: 0 events", self)
        layout.addWidget(self._selection_label)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Import destination", self))
        self._target_combo = QComboBox(self)
        target_row.addWidget(self._target_combo, 1)
        layout.addLayout(target_row)

        import_row = QHBoxLayout()
        import_row.addWidget(QLabel("Import mode", self))
        self._import_mode_combo = QComboBox(self)
        self._import_mode_combo.addItem("Import as New Take", "new_take")
        self._import_mode_combo.addItem("Import to Main", "main")
        self._import_mode_combo.setEnabled(False)
        import_row.addWidget(self._import_mode_combo, 1)
        layout.addLayout(import_row)

        self._destination_summary = QLabel(self)
        self._destination_summary.setWordWrap(True)
        layout.addWidget(self._destination_summary)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Import")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._timecode_picker.timecode_selected.connect(self._emit_timecode_changed)
        self._source_browser.track_group_selected.connect(self._emit_track_group_changed)
        self._source_browser.track_selected.connect(self._emit_track_changed)
        self._target_combo.currentIndexChanged.connect(self._emit_target_changed)
        self._canvas.selection_changed.connect(self._handle_event_selection_changed)
        self._canvas.zoom_changed.connect(self._refresh_zoom_label)
        self._zoom_out_btn.clicked.connect(self._zoom_out)
        self._zoom_in_btn.clicked.connect(self._zoom_in)
        self._zoom_reset_btn.clicked.connect(self._reset_zoom)
        self._select_all_btn.clicked.connect(self._select_all_events)
        self._clear_selection_btn.clicked.connect(self._clear_event_selection)
        self._refresh_zoom_label(self._canvas.zoom_factor)
        self._refresh_state()

    def set_flow(self, flow) -> None:
        self._syncing = True
        try:
            self._selected_timecode_no = flow.selected_timecode_no
            self._selected_track_group_no = flow.selected_track_group_no
            self._selected_source_track_coord_value = (
                flow.source_track_coord or flow.active_source_track_coord
            )
            self._track_labels_by_coord = {
                str(track.coord): self._track_label(track) for track in flow.available_tracks
            }
            self._track_group_labels_by_no = {
                int(group.number): self._track_group_label(group)
                for group in flow.available_track_groups
            }
            self._timecode_picker.set_timecodes(
                flow.available_timecodes,
                flow.selected_timecode_no,
            )
            self._source_browser.set_source_options(
                track_groups=flow.available_track_groups,
                tracks=flow.available_tracks,
                selected_track_group_no=flow.selected_track_group_no,
                active_track_coord=self._selected_source_track_coord_value,
                selected_track_coords=list(flow.selected_source_track_coords),
            )

            with QSignalBlocker(self._target_combo):
                self._target_combo.clear()
                for target in flow.available_target_layers:
                    self._target_combo.addItem(target.name, target.layer_id)
                self._set_combo_value(self._target_combo, flow.target_layer_id)

            self._sync_import_mode_with_target(fallback_mode=flow.import_mode)

            self._canvas.set_events(
                flow.available_events,
                selected_event_ids=list(flow.selected_ma3_event_ids),
            )
        finally:
            self._syncing = False
        self._refresh_state()

    def selected_event_ids(self) -> list[str]:
        return self._canvas.selected_event_ids()

    def selected_target_layer_id(self) -> LayerId | None:
        value = self._target_combo.currentData()
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return LayerId(stripped)
        return None

    def selected_source_track_coord(self) -> str | None:
        return self._selected_source_track_coord_value

    def selected_import_mode(self) -> str:
        return str(self._import_mode_combo.currentData() or "new_take")

    def accept(self) -> None:
        if self.selected_source_track_coord() is None:
            QMessageBox.warning(self, "Import Event Layer from MA3", "Select a source MA3 track.")
            return
        if not self.selected_event_ids():
            QMessageBox.warning(self, "Import Event Layer from MA3", "Select at least one source event.")
            return
        if self.selected_target_layer_id() is None:
            QMessageBox.warning(self, "Import Event Layer from MA3", "Select an EZ destination.")
            return
        super().accept()

    def _emit_timecode_changed(self, timecode_no: int) -> None:
        if self._syncing:
            return
        self._selected_timecode_no = int(timecode_no)
        self.timecode_selected.emit(int(timecode_no))

    def _emit_track_group_changed(self, track_group_no: int) -> None:
        if self._syncing:
            return
        self._selected_track_group_no = int(track_group_no)
        self.track_group_selected.emit(int(track_group_no))

    def _emit_track_changed(self, source_track_coord: str) -> None:
        if self._syncing:
            return
        coord = str(source_track_coord).strip()
        if not coord:
            return
        self._selected_source_track_coord_value = coord
        derived_group_no = self._track_group_no(coord)
        if derived_group_no is not None:
            self._selected_track_group_no = derived_group_no
        self.track_selected.emit(coord)
        self._refresh_state()

    def _emit_target_changed(self) -> None:
        self._sync_import_mode_with_target()
        if self._syncing:
            return
        self.target_layer_selected.emit(self.selected_target_layer_id())
        self._refresh_state()

    def _handle_event_selection_changed(self, selected_event_ids: list[str]) -> None:
        self._refresh_state()
        if self._syncing:
            return
        self.event_selection_changed.emit(list(selected_event_ids))

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

    def _select_all_events(self) -> None:
        self._canvas.set_selected_event_ids([event.event_id for event in self._canvas._events])

    def _clear_event_selection(self) -> None:
        self._canvas.set_selected_event_ids([])

    def _sync_import_mode_with_target(self, *, fallback_mode: str | None = None) -> None:
        target_layer_id = self.selected_target_layer_id()
        import_mode = (
            _pull_import_mode_for_target(target_layer_id)
            if target_layer_id is not None
            else str(fallback_mode or "new_take")
        )
        with QSignalBlocker(self._import_mode_combo):
            self._set_combo_value(self._import_mode_combo, import_mode)

    def _refresh_state(self) -> None:
        selected_count = len(self.selected_event_ids())
        noun = "event" if selected_count == 1 else "events"
        self._selection_label.setText(f"Selected: {selected_count} {noun}")
        source_track_coord = self.selected_source_track_coord()
        if source_track_coord is None:
            self._source_summary.setText("Source: Select an MA3 track to preview its events.")
        else:
            source_label = self._track_labels_by_coord.get(source_track_coord, source_track_coord)
            group_label = (
                self._track_group_labels_by_no.get(self._selected_track_group_no)
                if self._selected_track_group_no is not None
                else None
            )
            if group_label:
                self._source_summary.setText(f"Source: {group_label} / {source_label}")
            else:
                self._source_summary.setText(f"Source: {source_label}")
        target_label = self._target_combo.currentText().strip()
        target_value = self.selected_target_layer_id()
        import_mode_label = (
            "main"
            if self.selected_import_mode() == "main"
            else "a new take"
        )
        if target_value is None:
            self._destination_summary.setText("Destination: Select an EZ target")
        elif _is_new_layer_pull_target(target_value):
            self._destination_summary.setText(
                f"Destination: Create a new EZ event layer and import to {import_mode_label}"
            )
        else:
            self._destination_summary.setText(
                f"Destination: Import to {import_mode_label} in {target_label}"
            )
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(
                self.selected_source_track_coord() is not None
                and bool(self.selected_event_ids())
                and self.selected_target_layer_id() is not None
            )

    @staticmethod
    def _set_combo_value(combo: QComboBox, value) -> None:
        if value is None:
            return
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return

    @staticmethod
    def _timecode_label(timecode) -> str:
        if timecode.name:
            return f"TC{timecode.number} {timecode.name}"
        return f"TC{timecode.number}"

    @staticmethod
    def _track_group_label(track_group) -> str:
        suffix = ""
        if track_group.track_count is not None:
            noun = "track" if track_group.track_count == 1 else "tracks"
            suffix = f" [{track_group.track_count} {noun}]"
        name = track_group.name or f"Group {track_group.number}"
        return f"TG{track_group.number} {name}{suffix}"

    @staticmethod
    def _track_label(track) -> str:
        track_no = track.number
        prefix = f"TR{track_no}" if track_no is not None else "TR?"
        name = track.name or prefix
        parts = [f"{prefix} {name}", f"({track.coord})"]
        if track.event_count is not None:
            noun = "event" if track.event_count == 1 else "events"
            parts.append(f"[{track.event_count} {noun}]")
        return " ".join(parts)

    @staticmethod
    def _track_group_no(raw_coord: str | None) -> int | None:
        coord = str(raw_coord or "").strip().lower()
        if "_tg" not in coord:
            return None
        track_group_text = coord.split("_tg", 1)[1].split("_", 1)[0]
        try:
            parsed = int(track_group_text)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 1 else None

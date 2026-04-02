"""Interactive Stage Zero timeline shell for the new application architecture."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton

from echozero.application.presentation.models import (
    TimelinePresentation,
    LayerPresentation,
    EventPresentation,
)
from echozero.application.timeline.intents import Pause, Play, Seek, SelectLayer, SelectTake, ToggleTakeSelector


class TimelineCanvas(QWidget):
    layer_clicked = pyqtSignal(object)
    take_toggle_clicked = pyqtSignal(object)
    take_selected = pyqtSignal(object, object)
    seek_requested = pyqtSignal(float)

    def __init__(self, presentation: TimelinePresentation, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self.setMinimumHeight(max(240, len(self.presentation.layers) * 72 + 60))
        self.setMinimumWidth(1200)
        self._row_height = 72
        self._header_width = 280
        self._event_height = 24
        self._take_chip_rects: list[tuple[QRectF, object, object]] = []
        self._take_toggle_rects: list[tuple[QRectF, object]] = []

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self.setMinimumHeight(max(240, len(self.presentation.layers) * 72 + 60))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#14161b"))

        self._take_chip_rects.clear()
        self._take_toggle_rects.clear()

        self._draw_header_background(painter)
        self._draw_grid(painter)
        self._draw_layers(painter)
        self._draw_playhead(painter)

    def mousePressEvent(self, event):
        pos = event.position()

        for rect, layer_id, take_id in self._take_chip_rects:
            if rect.contains(pos):
                self.take_selected.emit(layer_id, take_id)
                return

        for rect, layer_id in self._take_toggle_rects:
            if rect.contains(pos):
                self.take_toggle_clicked.emit(layer_id)
                return

        if pos.x() < self._header_width:
            layer_index = int(pos.y() // self._row_height)
            if 0 <= layer_index < len(self.presentation.layers):
                self.layer_clicked.emit(self.presentation.layers[layer_index].layer_id)
            return

        pps = max(1.0, self.presentation.pixels_per_second)
        seek_time = max(0.0, (pos.x() - self._header_width + self.presentation.scroll_x) / pps)
        self.seek_requested.emit(seek_time)

    def _draw_header_background(self, painter: QPainter) -> None:
        painter.fillRect(0, 0, self._header_width, self.height(), QColor("#1b1f27"))
        painter.fillRect(self._header_width, 0, 1, self.height(), QColor("#2c3340"))

    def _draw_grid(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#232936"), 1))
        pps = max(1.0, self.presentation.pixels_per_second)
        seconds = int(max(10, (self.width() - self._header_width) / pps))
        for second in range(seconds + 2):
            x = self._header_width + int(second * pps) - int(self.presentation.scroll_x)
            painter.drawLine(x, 0, x, self.height())
            painter.setPen(QColor("#657084"))
            painter.drawText(x + 4, 14, f"{second}s")
            painter.setPen(QPen(QColor("#232936"), 1))

    def _draw_layers(self, painter: QPainter) -> None:
        for index, layer in enumerate(self.presentation.layers):
            top = index * self._row_height
            self._draw_layer_row(painter, layer, top)

    def _draw_layer_row(self, painter: QPainter, layer: LayerPresentation, top: int) -> None:
        bg = QColor("#1c2530") if layer.is_selected else QColor("#171b22")
        header_bg = QColor("#212a35") if layer.is_selected else QColor("#1d222b")
        painter.fillRect(0, top, self.width(), self._row_height - 1, bg)
        painter.fillRect(0, top, self._header_width, self._row_height - 1, header_bg)
        painter.fillRect(0, top + self._row_height - 1, self.width(), 1, QColor("#252c38"))

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#f0f3f8"))
        painter.drawText(14, top + 22, layer.title)

        sub_font = QFont()
        sub_font.setPointSize(8)
        painter.setFont(sub_font)
        painter.setPen(QColor("#9aa4b2"))
        subtitle = layer.subtitle or layer.take_summary.compact_label
        painter.drawText(14, top + 39, subtitle)

        badge_x = 14
        badge_y = top + 46
        for badge in layer.badges:
            rect = QRectF(badge_x, badge_y, 50, 16)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#2c6bed")))
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(QColor("white"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, badge[:6])
            badge_x += 56

        if layer.take_summary.can_expand:
            toggle_rect = QRectF(190, top + 11, 78, 20)
            self._take_toggle_rects.append((toggle_rect, layer.layer_id))
            painter.setPen(QColor("#3b4554"))
            painter.setBrush(QBrush(QColor("#151922")))
            painter.drawRoundedRect(toggle_rect, 6, 6)
            painter.setPen(QColor("#d7dce4"))
            caret = "▾" if layer.is_expanded else "▸"
            painter.drawText(
                toggle_rect,
                Qt.AlignmentFlag.AlignCenter,
                f"{caret} {layer.take_summary.total_take_count} takes",
            )

        if layer.is_expanded and layer.take_summary.available_take_names:
            chip_x = 14
            chip_y = top + 52
            for index, take_name in enumerate(layer.take_summary.available_take_names):
                take_id = None
                if index < len(layer.take_summary.available_take_names):
                    # take ids are not exposed directly in summary names; active id is enough for demo shell
                    take_id = layer.active_take_id if take_name == layer.take_summary.active_take_name else take_name
                rect = QRectF(chip_x, chip_y, max(58, 18 + len(take_name) * 7), 16)
                self._take_chip_rects.append((rect, layer.layer_id, take_id))
                active = take_name == layer.take_summary.active_take_name
                painter.setPen(QColor("#4f5d73" if active else "#384354"))
                painter.setBrush(QBrush(QColor("#2b3442" if active else "#1a2029")))
                painter.drawRoundedRect(rect, 6, 6)
                painter.setPen(QColor("#f0f3f8" if active else "#b7c0cc"))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, take_name)
                chip_x += rect.width() + 8

        for event in layer.events:
            self._draw_event(painter, event, top)

    def _draw_event(self, painter: QPainter, event: EventPresentation, top: int) -> None:
        pps = max(1.0, self.presentation.pixels_per_second)
        x = self._header_width + (event.start * pps) - self.presentation.scroll_x
        width = max(10.0, (event.duration * pps))
        y = top + 24

        color = QColor(event.color or "#57a0ff")
        if event.is_selected:
            color = color.lighter(130)

        painter.setPen(QPen(color.darker(160), 1))
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(QRectF(x, y, width, self._event_height), 5, 5)

        painter.setPen(QColor("#0b1220"))
        painter.drawText(
            QRectF(x + 6, y, max(0, width - 12), self._event_height),
            Qt.AlignmentFlag.AlignVCenter,
            event.label,
        )

    def _draw_playhead(self, painter: QPainter) -> None:
        x = self._header_width + (
            self.presentation.playhead * self.presentation.pixels_per_second
        ) - self.presentation.scroll_x
        painter.setPen(QPen(QColor("#ff5f57"), 2))
        painter.drawLine(int(x), 0, int(x), self.height())


class TimelineWidget(QWidget):
    def __init__(
        self,
        presentation: TimelinePresentation,
        on_intent: Callable[[object], TimelinePresentation] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.presentation = presentation
        self._on_intent = on_intent
        self.setWindowTitle("EchoZero Timeline Preview")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(12)
        header.setStyleSheet("background: #10141a; color: #f0f3f8;")

        self._title = QLabel()
        self._title.setStyleSheet("font-size: 14px; font-weight: 700;")
        self._status = QLabel()
        self._status.setStyleSheet("color: #9aa4b2;")

        self._transport_button = QPushButton()
        self._transport_button.clicked.connect(self._toggle_play_pause)
        self._transport_button.setStyleSheet(
            "QPushButton { background: #1d6cf2; color: white; border: none; padding: 6px 12px; border-radius: 6px; }"
            "QPushButton:hover { background: #2f7cff; }"
        )

        header_layout.addWidget(self._title)
        header_layout.addStretch(1)
        header_layout.addWidget(self._status)
        header_layout.addWidget(self._transport_button)
        layout.addWidget(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("background: #14161b; border: none;")
        self._canvas = TimelineCanvas(self.presentation)
        self._canvas.layer_clicked.connect(self._select_layer)
        self._canvas.take_toggle_clicked.connect(self._toggle_take_selector)
        self._canvas.take_selected.connect(self._select_take)
        self._canvas.seek_requested.connect(self._seek)
        self._scroll.setWidget(self._canvas)
        layout.addWidget(self._scroll)

        self.set_presentation(self.presentation)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self._title.setText(self.presentation.title)
        self._status.setText(
            f"{'Playing' if self.presentation.is_playing else 'Stopped'}  •  "
            f"Layers: {len(self.presentation.layers)}  •  "
            f"Playhead: {self.presentation.playhead:.2f}s"
        )
        self._transport_button.setText("Pause" if self.presentation.is_playing else "Play")
        self._canvas.set_presentation(presentation)

    def _dispatch(self, intent: object) -> None:
        if self._on_intent is None:
            return
        updated = self._on_intent(intent)
        if updated is not None:
            self.set_presentation(updated)

    def _toggle_play_pause(self) -> None:
        self._dispatch(Pause() if self.presentation.is_playing else Play())

    def _seek(self, position: float) -> None:
        self._dispatch(Seek(position))

    def _select_layer(self, layer_id) -> None:
        self._dispatch(SelectLayer(layer_id))

    def _toggle_take_selector(self, layer_id) -> None:
        self._dispatch(ToggleTakeSelector(layer_id))

    def _select_take(self, layer_id, take_id) -> None:
        if take_id is None:
            return
        self._dispatch(SelectTake(layer_id, take_id))

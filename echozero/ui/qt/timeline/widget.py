"""Stage Zero timeline shell composed from reusable blocks."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea

from echozero.application.presentation.models import TimelinePresentation, LayerPresentation, TakeLanePresentation
from echozero.application.timeline.intents import Pause, Play, Seek, SelectEvent, SelectLayer, SelectTake, ToggleTakeSelector
from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock, EventLanePresentation
from echozero.ui.qt.timeline.blocks.layer_header import HeaderSlots, LayerHeaderBlock
from echozero.ui.qt.timeline.blocks.layouts import MainRowLayout, TakeRowLayout
from echozero.ui.qt.timeline.blocks.ruler import RulerBlock, RulerLayout
from echozero.ui.qt.timeline.blocks.take_row import TakeRowBlock
from echozero.ui.qt.timeline.blocks.transport_bar import TransportLayout
from echozero.ui.qt.timeline.blocks.transport_bar_block import TransportBarBlock
from echozero.ui.qt.timeline.blocks.waveform_lane import WaveformLaneBlock, WaveformLanePresentation


class TimelineCanvas(QWidget):
    layer_clicked = pyqtSignal(object)
    take_toggle_clicked = pyqtSignal(object)
    take_selected = pyqtSignal(object, object)
    event_selected = pyqtSignal(object, object)
    seek_requested = pyqtSignal(float)

    def __init__(self, presentation: TimelinePresentation, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self._header_width = 320
        self._ruler_height = 28
        self._main_row_height = 72
        self._take_row_height = 44
        self._event_height = 22
        self._take_rects: list[tuple[QRectF, object, object]] = []
        self._toggle_rects: list[tuple[QRectF, object]] = []
        self._event_rects: list[tuple[QRectF, object, object]] = []
        self._ruler_block = RulerBlock()
        self._header_block = LayerHeaderBlock()
        self._waveform_block = WaveformLaneBlock()
        self._event_lane_block = EventLaneBlock()
        self._take_row_block = TakeRowBlock()
        self.setMinimumWidth(1440)
        self._recompute_height()

    def _recompute_height(self) -> None:
        height = self._ruler_height + 8
        for layer in self.presentation.layers:
            height += self._main_row_height
            if layer.is_expanded:
                height += len(layer.takes) * self._take_row_height
        self.setMinimumHeight(max(320, height + 12))

    def _any_solo(self) -> bool:
        return any(layer.soloed for layer in self.presentation.layers)

    def _layer_dimmed(self, layer: LayerPresentation) -> bool:
        if layer.muted:
            return True
        if self._any_solo() and not layer.soloed:
            return True
        return False

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self._recompute_height()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#12151b'))
        self._take_rects.clear()
        self._toggle_rects.clear()
        self._event_rects.clear()
        self._ruler_block.paint(painter, RulerLayout(QRectF(0, 0, self.width(), self._ruler_height), self._header_width), self.presentation)
        self._draw_layers(painter)
        self._draw_playhead(painter)

    def mousePressEvent(self, event):
        pos = event.position()
        for rect, layer_id, take_id in self._take_rects:
            if rect.contains(pos):
                self.take_selected.emit(layer_id, take_id)
                return
        for rect, layer_id, event_id in self._event_rects:
            if rect.contains(pos):
                self.event_selected.emit(layer_id, event_id)
                return
        for rect, layer_id in self._toggle_rects:
            if rect.contains(pos):
                self.take_toggle_clicked.emit(layer_id)
                return
        self._emit_seek(pos.x())

    def _emit_seek(self, x: float) -> None:
        pps = max(1.0, self.presentation.pixels_per_second)
        seek_time = max(0.0, (x - self._header_width + self.presentation.scroll_x) / pps)
        self.seek_requested.emit(seek_time)

    def _draw_layers(self, painter: QPainter) -> None:
        y = self._ruler_height + 8
        for layer in self.presentation.layers:
            self._draw_main_row(painter, layer, y)
            y += self._main_row_height
            if layer.is_expanded:
                for take in layer.takes:
                    self._draw_take_row(painter, layer, take, y)
                    y += self._take_row_height

    def _draw_main_row(self, painter: QPainter, layer: LayerPresentation, top: int) -> None:
        dimmed = self._layer_dimmed(layer)
        layout = MainRowLayout.create(top=top, width=self.width(), header_width=self._header_width, row_height=self._main_row_height)
        row_bg = QColor('#1a212b' if layer.is_selected else '#161b22')
        if dimmed:
            row_bg = QColor('#12161c')
        painter.fillRect(layout.row_rect, row_bg)
        painter.fillRect(0, top + self._main_row_height - 1, self.width(), 1, QColor('#252c38'))

        slots = HeaderSlots(
            rect=layout.header_rect,
            title_rect=layout.title_rect,
            subtitle_rect=layout.subtitle_rect,
            status_rect=layout.status_rect,
            controls_rect=layout.controls_rect,
            toggle_rect=layout.toggle_rect,
            badges_origin_x=layout.badges_origin_x,
            badges_y=layout.badges_y,
        )
        self._toggle_rects.append((slots.toggle_rect, layer.layer_id))
        self._header_block.paint(painter, slots, layer, dimmed=dimmed)

        painter.save()
        painter.setClipRect(layout.content_rect)
        try:
            if layer.kind.name == 'AUDIO':
                self._waveform_block.paint(
                    painter,
                    top,
                    WaveformLanePresentation(
                        color_hex=layer.color or '#9b87f5',
                        row_height=self._main_row_height,
                        pixels_per_second=self.presentation.pixels_per_second,
                        scroll_x=self.presentation.scroll_x,
                        header_width=self._header_width,
                        width=self.width(),
                        dimmed=dimmed,
                    ),
                )
            else:
                self._event_rects.extend(
                    self._event_lane_block.paint(
                        painter,
                        top + 24,
                        EventLanePresentation(
                            layer_id=layer.layer_id,
                            events=layer.events,
                            pixels_per_second=self.presentation.pixels_per_second,
                            scroll_x=self.presentation.scroll_x,
                            header_width=self._header_width,
                            event_height=self._event_height,
                            dimmed=dimmed,
                        ),
                    )
                )
        finally:
            painter.restore()

    def _draw_take_row(self, painter: QPainter, layer: LayerPresentation, take: TakeLanePresentation, top: int) -> None:
        dimmed = self._layer_dimmed(layer)
        layout = TakeRowLayout.create(top=top, width=self.width(), header_width=self._header_width, row_height=self._take_row_height)
        hit_targets = self._take_row_block.paint_header(painter, layout, layer, take, dimmed=dimmed)
        self._take_rects.append(hit_targets.take_rect)

        painter.save()
        painter.setClipRect(layout.content_rect)
        try:
            if take.kind.name == 'AUDIO':
                self._waveform_block.paint(
                    painter,
                    top,
                    WaveformLanePresentation(
                        color_hex=layer.color or '#9b87f5',
                        row_height=self._take_row_height,
                        pixels_per_second=self.presentation.pixels_per_second,
                        scroll_x=self.presentation.scroll_x,
                        header_width=self._header_width,
                        width=self.width(),
                        dimmed=True or dimmed,
                    ),
                )
            else:
                self._event_rects.extend(
                    self._event_lane_block.paint(
                        painter,
                        top + 10,
                        EventLanePresentation(
                            layer_id=layer.layer_id,
                            events=take.events,
                            pixels_per_second=self.presentation.pixels_per_second,
                            scroll_x=self.presentation.scroll_x,
                            header_width=self._header_width,
                            event_height=self._event_height,
                            dimmed=True or dimmed,
                        ),
                    )
                )
        finally:
            painter.restore()

    def _draw_playhead(self, painter: QPainter) -> None:
        x = self._header_width + (self.presentation.playhead * self.presentation.pixels_per_second) - self.presentation.scroll_x
        painter.setPen(QPen(QColor('#ff5f57'), 2))
        painter.drawLine(int(x), 0, int(x), self.height())


class TransportBar(QWidget):
    def __init__(self, presentation: TimelinePresentation, on_intent: Callable[[object], TimelinePresentation] | None = None, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self._on_intent = on_intent
        self._block = TransportBarBlock()
        self.setMinimumHeight(44)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._block.paint(painter, TransportLayout.create(width=self.width(), height=self.height()), self.presentation)


class TimelineWidget(QWidget):
    def __init__(self, presentation: TimelinePresentation, on_intent: Callable[[object], TimelinePresentation] | None = None, parent=None):
        super().__init__(parent)
        self.presentation = presentation
        self._on_intent = on_intent
        self.setWindowTitle('EchoZero Timeline Preview')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._transport = TransportBar(self.presentation, on_intent=self._on_intent)
        layout.addWidget(self._transport)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet('background: #12151b; border: none;')
        self._canvas = TimelineCanvas(self.presentation)
        self._canvas.layer_clicked.connect(self._select_layer)
        self._canvas.take_toggle_clicked.connect(self._toggle_take_selector)
        self._canvas.take_selected.connect(self._select_take)
        self._canvas.event_selected.connect(self._select_event)
        self._canvas.seek_requested.connect(self._seek)
        self._scroll.setWidget(self._canvas)
        layout.addWidget(self._scroll)

        self.set_presentation(self.presentation)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self._transport.set_presentation(presentation)
        self._canvas.set_presentation(presentation)

    def _dispatch(self, intent: object) -> None:
        if self._on_intent is None:
            return
        updated = self._on_intent(intent)
        if updated is not None:
            self.set_presentation(updated)

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

    def _select_event(self, layer_id, event_id) -> None:
        if event_id is None:
            return
        self._dispatch(SelectEvent(layer_id, event_id))

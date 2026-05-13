"""Visual Lab preview widgets.
Exists to render catalog entries without entering the canonical EchoZero app path.
Factories in the preview runner compose these widgets into individual and composite samples.
"""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from echozero.application.presentation.inspector_contract import (
    build_timeline_inspector_contract,
)
from echozero.application.presentation.models import LayerPresentation, TimelinePresentation
from echozero.application.settings import AppPreferences, SettingsOption, build_app_settings_page
from echozero.ui.qt.settings_page_form import SettingsPageForm
from echozero.ui.qt.song_browser_panel import SongBrowserPanel
from echozero.ui.qt.timeline.blocks.layer_header import HeaderSlots, LayerHeaderBlock
from echozero.ui.qt.timeline.object_info_panel import ObjectInfoPanel
from echozero.ui.qt.timeline.object_info_panel_preview import EventPreviewWaveform
from echozero.ui.qt.timeline.widget import TimelineWidget
from echozero.ui.qt.timeline.widget_canvas import TimelineCanvas
from echozero.ui.qt.timeline.widget_controls import (
    TimelineEditorModeBar,
    TimelineRuler,
    TransportBar,
)

from dev.visual_lab.tokens import VisualLabTokens
from dev.visual_lab.waveforms import build_fun_event_preview_state


class CatalogFrame(QWidget):
    """Simple lab-only frame for a single object preview."""

    def __init__(
        self,
        tokens: VisualLabTokens,
        title: str,
        child: QWidget,
        *,
        width: int = 760,
        height: int = 260,
    ) -> None:
        super().__init__()
        self.setObjectName("visual_lab_catalog_frame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            tokens.metrics.padding_px,
            tokens.metrics.padding_px,
            tokens.metrics.padding_px,
            tokens.metrics.padding_px,
        )
        layout.setSpacing(tokens.metrics.gap_px)
        label = QLabel(title)
        label.setObjectName("visual_lab_object_title")
        layout.addWidget(label)
        layout.addWidget(child, stretch=1)
        self.setMinimumSize(width, height)


class TimelinePreviewWidget(QWidget):
    """Painted timeline/layer preview for Visual Lab scenes."""

    def __init__(self, tokens: VisualLabTokens, presentation: TimelinePresentation) -> None:
        super().__init__()
        self.tokens = tokens
        self.presentation = presentation
        total_height = sum(_row_height(tokens, layer) for layer in presentation.layers)
        self.setMinimumHeight(total_height)
        self.setMinimumWidth(
            tokens.metrics.timeline_header_width_px + tokens.metrics.timeline_width_px
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.tokens.palette.panel))

        y = 0
        for layer in self.presentation.layers:
            height = _row_height(self.tokens, layer)
            self._paint_row(painter, layer, QRectF(0, y, self.width(), height))
            y += height

        playhead_x = (
            self.tokens.metrics.timeline_header_width_px
            + self.presentation.playhead * self.presentation.pixels_per_second
        )
        painter.setPen(QPen(QColor(self.tokens.palette.accent_secondary), 2))
        painter.drawLine(QPointF(playhead_x, 0), QPointF(playhead_x, self.height()))

    def _paint_row(self, painter: QPainter, layer: LayerPresentation, rect: QRectF) -> None:
        palette = self.tokens.palette
        metrics = self.tokens.metrics
        is_child = layer.parent_layer_id is not None
        fill = palette.row_child if is_child else palette.row
        if layer.is_selected:
            fill = palette.row_selected
        if layer.muted:
            fill = palette.row_muted

        painter.fillRect(rect, QColor(fill))
        painter.setPen(QPen(QColor(palette.border), metrics.border_width_px))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())

        header_rect = QRectF(0, rect.y(), metrics.timeline_header_width_px, rect.height())
        timeline_rect = QRectF(
            metrics.timeline_header_width_px,
            rect.y(),
            self.width() - metrics.timeline_header_width_px,
            rect.height(),
        )
        painter.fillRect(
            header_rect, QColor(palette.panel_raised if not is_child else palette.panel)
        )
        self._paint_header(painter, layer, header_rect, is_child=is_child)
        self._paint_grid(painter, timeline_rect)
        self._paint_waveform(painter, layer, timeline_rect)
        self._paint_events(painter, layer, timeline_rect)

    def _paint_header(
        self, painter: QPainter, layer: LayerPresentation, rect: QRectF, *, is_child: bool
    ) -> None:
        palette = self.tokens.palette
        left = rect.left() + self.tokens.metrics.padding_px + (18 if is_child else 0)
        painter.setPen(QColor(palette.text if not layer.muted else palette.text_muted))
        painter.setFont(
            _font(self.tokens, self.tokens.fonts.label_px, self.tokens.fonts.weight_bold)
        )
        painter.drawText(QRectF(left, rect.top() + 9, rect.width() - left, 18), layer.title)
        painter.setPen(QColor(palette.text_muted))
        painter.setFont(_font(self.tokens, self.tokens.fonts.small_px))
        painter.drawText(QRectF(left, rect.top() + 29, rect.width() - left, 16), layer.subtitle)

        chips = _status_chips(self.tokens, layer)
        x = rect.right() - self.tokens.metrics.padding_px
        for label, color in reversed(chips):
            chip_width = max(44, len(label) * 7 + 16)
            chip_rect = QRectF(
                x - chip_width,
                rect.top() + rect.height() - self.tokens.metrics.status_chip_height_px - 8,
                chip_width,
                self.tokens.metrics.status_chip_height_px,
            )
            painter.setBrush(QColor(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                chip_rect,
                self.tokens.metrics.control_radius_px,
                self.tokens.metrics.control_radius_px,
            )
            painter.setPen(QColor("#081018"))
            painter.drawText(chip_rect, Qt.AlignmentFlag.AlignCenter, label)
            x -= chip_width + 6

    def _paint_grid(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(self.tokens.palette.grid), 1))
        step = self.presentation.pixels_per_second * 4
        x = rect.left()
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step

    def _paint_waveform(self, painter: QPainter, layer: LayerPresentation, rect: QRectF) -> None:
        if layer.kind.value != "audio":
            return
        mid = rect.center().y()
        height = min(self.tokens.metrics.waveform_height_px, rect.height() - 14)
        painter.setPen(QPen(QColor(self.tokens.palette.waveform), 1.5))
        painter.setBrush(QColor(self.tokens.palette.waveform_fill))
        x = rect.left() + 8
        while x < rect.right() - 8:
            peak = ((int(x) // 17) % 7 + 2) / 9
            top = mid - height * peak / 2
            bottom = mid + height * peak / 2
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            x += 4

    def _paint_events(self, painter: QPainter, layer: LayerPresentation, rect: QRectF) -> None:
        for timeline_event in layer.events:
            x = rect.left() + timeline_event.start * self.presentation.pixels_per_second
            width = max(4.0, timeline_event.duration * self.presentation.pixels_per_second)
            event_rect = QRectF(x, rect.top() + 9, width, max(18, rect.height() - 18))
            fill = QColor(layer.color or self.tokens.palette.accent)
            if timeline_event.muted or layer.muted:
                fill = QColor(self.tokens.palette.status_muted)
            border = QColor(
                self.tokens.palette.text if timeline_event.is_selected else fill.darker(140)
            )
            painter.setBrush(fill)
            painter.setPen(QPen(border, 2 if timeline_event.is_selected else 1))
            painter.drawRoundedRect(
                event_rect,
                self.tokens.metrics.control_radius_px,
                self.tokens.metrics.control_radius_px,
            )
            if event_rect.width() > 34:
                painter.setPen(QColor("#081018"))
                painter.setFont(_font(self.tokens, self.tokens.fonts.small_px))
                painter.drawText(
                    event_rect.adjusted(6, 0, -4, 0),
                    Qt.AlignmentFlag.AlignVCenter,
                    timeline_event.label,
                )


class TimelineCanvasPreviewWidget(TimelineCanvas):
    """Preview the production timeline canvas with assembled current models."""

    def __init__(self, tokens: VisualLabTokens, presentation: TimelinePresentation) -> None:
        del tokens
        super().__init__(presentation)
        self.setMinimumWidth(860)


class TransportBarPreviewWidget(TransportBar):
    """Preview the production transport bar with assembled current models."""

    def __init__(self, tokens: VisualLabTokens, presentation: TimelinePresentation) -> None:
        del tokens
        super().__init__(presentation)
        self.setMinimumWidth(720)


class TimelineShellPreviewWidget(TimelineWidget):
    """Preview the production timeline shell with current-model lab state."""

    def __init__(self, tokens: VisualLabTokens, presentation: TimelinePresentation) -> None:
        del tokens
        super().__init__(presentation)
        self.setMinimumSize(1120, 680)


class EditorModeBarPreviewWidget(TimelineEditorModeBar):
    """Preview the production top timeline toolbar in isolation."""

    def __init__(self, tokens: VisualLabTokens) -> None:
        del tokens
        super().__init__()
        self.setMinimumWidth(960)
        self.setFixedHeight(max(52, self.sizeHint().height()))


class TimelineRulerPreviewWidget(TimelineRuler):
    """Preview the production timeline ruler chrome in isolation."""

    def __init__(self, tokens: VisualLabTokens, presentation: TimelinePresentation) -> None:
        del tokens
        super().__init__(presentation)
        self.setMinimumWidth(860)


class SongBrowserPanelPreviewWidget(SongBrowserPanel):
    """Preview the production setlist/object browser panel."""

    def __init__(self, tokens: VisualLabTokens, presentation: TimelinePresentation) -> None:
        del tokens
        super().__init__(presentation)
        self.setMinimumSize(320, 520)


class ObjectInfoPanelPreviewWidget(ObjectInfoPanel):
    """Preview the production object info palette with current inspector state."""

    def __init__(self, tokens: VisualLabTokens, presentation: TimelinePresentation) -> None:
        del tokens
        super().__init__()
        self.setMinimumSize(360, 560)
        self.set_contract(presentation, build_timeline_inspector_contract(presentation))


class SettingsFormPreviewWidget(SettingsPageForm):
    """Preview the production neutral settings form with current settings models."""

    def __init__(self, tokens: VisualLabTokens) -> None:
        del tokens
        super().__init__()
        page = build_app_settings_page(
            AppPreferences(),
            audio_device_options_provider=lambda: (
                SettingsOption(value="", label="System Default"),
                SettingsOption(value="visual-lab-output", label="Visual Lab Output 1/2"),
            ),
            include_hidden=True,
        )
        self.set_page(page)
        self.setMinimumSize(620, 520)


class WaveformPreviewWidget(EventPreviewWaveform):
    """Preview the current-shape synthetic waveform state used by Visual Lab."""

    def __init__(self, tokens: VisualLabTokens) -> None:
        del tokens
        super().__init__()
        self.set_preview(build_fun_event_preview_state())
        self.setMinimumSize(520, 96)


class TransportPreviewWidget(QWidget):
    """Compact transport/status/control preview using Visual Lab tokens."""

    def __init__(self, tokens: VisualLabTokens, presentation: TimelinePresentation) -> None:
        super().__init__()
        self.tokens = tokens
        self.presentation = presentation
        self.setFixedHeight(tokens.metrics.transport_height_px)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.tokens.palette.panel_raised))
        painter.setPen(
            QPen(QColor(self.tokens.palette.border), self.tokens.metrics.border_width_px)
        )
        painter.drawRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5),
            self.tokens.metrics.corner_radius_px,
            self.tokens.metrics.corner_radius_px,
        )

        painter.setPen(QColor(self.tokens.palette.text))
        painter.setFont(_font(self.tokens, 24, self.tokens.fonts.weight_bold))
        painter.drawText(QRectF(18, 14, 130, 36), self.presentation.current_time_label)
        painter.setFont(_font(self.tokens, self.tokens.fonts.small_px))
        painter.setPen(QColor(self.tokens.palette.text_muted))
        painter.drawText(QRectF(20, 50, 160, 18), f"End {self.presentation.end_time_label}")

        labels = ("PLAY", "STOP", "SYNC", "SNAP")
        x = 190
        for index, label in enumerate(labels):
            active = index in {0, 2}
            fill = self.tokens.palette.accent if active else self.tokens.palette.panel
            text = "#081018" if active else self.tokens.palette.text
            rect = QRectF(x, 22, 72, 34)
            painter.setBrush(QColor(fill))
            painter.setPen(QPen(QColor(self.tokens.palette.border), 1))
            painter.drawRoundedRect(rect, 5, 5)
            painter.setPen(QColor(text))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
            x += 82

        status = "MA3 connected / stale cue layer pending review"
        painter.setPen(QColor(self.tokens.palette.status_stale))
        painter.setFont(
            _font(self.tokens, self.tokens.fonts.label_px, self.tokens.fonts.weight_medium)
        )
        painter.drawText(
            QRectF(x + 14, 22, self.width() - x - 30, 34),
            Qt.AlignmentFlag.AlignVCenter,
            status,
        )


class LayerHeaderPreviewWidget(QWidget):
    """Preview the production layer-header paint block in isolation."""

    def __init__(
        self,
        tokens: VisualLabTokens,
        layer: LayerPresentation,
        *,
        has_child_layers: bool = False,
    ) -> None:
        super().__init__()
        self.tokens = tokens
        self.layer = layer
        self.has_child_layers = has_child_layers
        self.setMinimumSize(tokens.metrics.timeline_header_width_px, 78)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.tokens.palette.panel))
        rect = QRectF(0, 8, self.width(), 58)
        slots = HeaderSlots(
            rect=rect,
            title_rect=QRectF(12, rect.top() + 5, rect.width() - 90, 20),
            subtitle_rect=QRectF(12, rect.top() + 28, rect.width() - 24, 16),
            status_rect=QRectF(12, rect.bottom() - 20, 130, 16),
            controls_rect=QRectF(rect.right() - 132, rect.top() + 8, 120, 20),
            active_rect=QRectF(rect.right() - 64, rect.top() + 8, 52, 18),
            toggle_rect=QRectF(rect.right() - 28, rect.bottom() - 25, 18, 18),
            metadata_rect=QRectF(12, rect.bottom() - 20, rect.width() - 24, 16),
        )
        LayerHeaderBlock().paint(
            painter,
            slots,
            self.layer,
            has_child_layers=self.has_child_layers,
        )


class StatusChipPreviewWidget(QWidget):
    """Preview lab status chips/tokens as independent objects."""

    def __init__(self, tokens: VisualLabTokens, chips: Sequence[tuple[str, str]]) -> None:
        super().__init__()
        self.tokens = tokens
        self.chips = tuple(chips)
        self.setMinimumSize(420, 96)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.tokens.palette.panel))
        x = 18.0
        y = 32.0
        for label, color in self.chips:
            width = max(58, len(label) * 8 + 22)
            rect = QRectF(x, y, width, self.tokens.metrics.status_chip_height_px + 4)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(rect, 5, 5)
            painter.setPen(QColor("#081018"))
            painter.setFont(_font(self.tokens, self.tokens.fonts.small_px))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
            x += width + 10


class ControlPrimitivePreviewWidget(QWidget):
    """Preview basic button/card primitives for lab-only catalog scaffolding."""

    def __init__(self, tokens: VisualLabTokens) -> None:
        super().__init__()
        self.tokens = tokens
        self.setMinimumSize(460, 150)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.tokens.palette.panel))
        card = QRectF(18, 18, 424, 112)
        painter.setPen(QPen(QColor(self.tokens.palette.border), 1))
        painter.setBrush(QColor(self.tokens.palette.panel_raised))
        painter.drawRoundedRect(
            card, self.tokens.metrics.corner_radius_px, self.tokens.metrics.corner_radius_px
        )
        labels = (("Primary", True), ("Quiet", False), ("Danger", False))
        x = card.left() + 18
        for label, active in labels:
            fill = self.tokens.palette.accent if active else self.tokens.palette.panel
            if label == "Danger":
                fill = self.tokens.palette.danger
            rect = QRectF(x, card.top() + 42, 94, 34)
            painter.setPen(QPen(QColor(self.tokens.palette.border), 1))
            painter.setBrush(QColor(fill))
            painter.drawRoundedRect(
                rect, self.tokens.metrics.control_radius_px, self.tokens.metrics.control_radius_px
            )
            painter.setPen(
                QColor("#081018" if active or label == "Danger" else self.tokens.palette.text)
            )
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
            x += 108


def _row_height(tokens: VisualLabTokens, layer: LayerPresentation) -> int:
    if layer.parent_layer_id is not None:
        return tokens.metrics.stem_row_height_px
    return tokens.metrics.audio_row_height_px


def _status_chips(tokens: VisualLabTokens, layer: LayerPresentation) -> list[tuple[str, str]]:
    chips: list[tuple[str, str]] = []
    if layer.muted:
        chips.append(("muted", tokens.palette.status_muted))
    if layer.status.stale:
        chips.append(("stale", tokens.palette.status_stale))
    if layer.sync_connected:
        chips.append(("sync", tokens.palette.status_sync))
    elif layer.status.sync_label:
        chips.append((layer.status.sync_label, tokens.palette.status_ok))
    return chips[:2]


def _font(tokens: VisualLabTokens, pixel_size: int, weight: int | None = None) -> QFont:
    font = QFont(tokens.fonts.family)
    font.setPixelSize(pixel_size)
    if weight is not None:
        font.setWeight(weight)
    return font

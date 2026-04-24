"""Reusable timeline chrome widgets.
Exists to keep ruler, transport, and editor-mode widgets out of the main timeline shell.
Connects timeline presentation and basic transport intents to compact Qt controls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.timeline.intents import Pause, Play, Stop, TimelineIntent
from echozero.ui.FEEL import (
    LAYER_HEADER_WIDTH_PX,
    RULER_HEIGHT_PX,
    TIMELINE_EDITOR_BAR_PADDING_X_PX,
    TIMELINE_EDITOR_BAR_PADDING_Y_PX,
    TIMELINE_EDITOR_GROUP_PADDING_X_PX,
    TIMELINE_EDITOR_GROUP_PADDING_Y_PX,
    TIMELINE_EDITOR_GROUP_SPACING_PX,
    TIMELINE_TRANSPORT_HEIGHT_PX,
)
from echozero.ui.qt.timeline.blocks.ruler import (
    RulerBlock,
    RulerLayout,
    seek_time_for_x,
    timeline_x_for_time,
)
from echozero.ui.qt.timeline.blocks.transport_bar import TransportLayout
from echozero.ui.qt.timeline.blocks.transport_bar_block import TransportBarBlock
from echozero.ui.qt.timeline.time_grid import TimelineGridMode


class TimelineEditorModeBar(QWidget):
    """Compact editor-mode strip for timeline tools and shell actions."""

    edit_mode_changed = pyqtSignal(str)
    snap_toggled = pyqtSignal(bool)
    grid_mode_changed = pyqtSignal(str)
    settings_requested = pyqtSignal()
    regions_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("timelineEditorModeBar")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(
            TIMELINE_EDITOR_BAR_PADDING_X_PX,
            TIMELINE_EDITOR_BAR_PADDING_Y_PX,
            TIMELINE_EDITOR_BAR_PADDING_X_PX,
            TIMELINE_EDITOR_BAR_PADDING_Y_PX,
        )
        layout.setSpacing(TIMELINE_EDITOR_GROUP_SPACING_PX)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}
        mode_group = QWidget(self)
        mode_group.setObjectName("timelineEditorModeGroup")
        mode_group.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        mode_layout = self._create_group_layout(mode_group)
        mode_label = QLabel("Edit", mode_group)
        mode_label.setProperty("timelineToolbarLabel", True)
        mode_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        mode_layout.addWidget(mode_label, 0, Qt.AlignmentFlag.AlignVCenter)
        for mode, label in (
            ("select", "Select"),
            ("move", "Move"),
            ("draw", "Draw"),
            ("region", "Region"),
            ("erase", "Erase"),
        ):
            button = QPushButton(label, mode_group)
            button.setProperty("timelineModeButton", True)
            button.setCheckable(True)
            button.clicked.connect(
                lambda _checked=False, mode_name=mode: self.edit_mode_changed.emit(mode_name)
            )
            mode_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
            self._button_group.addButton(button)
            self._mode_buttons[mode] = button
        layout.addWidget(mode_group, 0, Qt.AlignmentFlag.AlignVCenter)

        assist_group = QWidget(self)
        assist_group.setObjectName("timelineEditorAssistGroup")
        assist_group.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        assist_layout = self._create_group_layout(assist_group)
        assist_label = QLabel("Guides", assist_group)
        assist_label.setProperty("timelineToolbarLabel", True)
        assist_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        assist_layout.addWidget(assist_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._snap_button = QPushButton("Snap", assist_group)
        self._snap_button.setObjectName("timelineEditorSnapButton")
        self._snap_button.setCheckable(True)
        self._snap_button.clicked.connect(lambda checked: self.snap_toggled.emit(bool(checked)))
        assist_layout.addWidget(self._snap_button, 0, Qt.AlignmentFlag.AlignVCenter)

        self._grid_button = QPushButton("Grid: Auto", assist_group)
        self._grid_button.setObjectName("timelineEditorGridButton")
        self._grid_button.clicked.connect(self._cycle_grid_mode)
        assist_layout.addWidget(self._grid_button, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(assist_group, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)

        shell_group = QWidget(self)
        shell_group.setObjectName("timelineEditorShellGroup")
        shell_group.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        shell_layout = self._create_group_layout(shell_group)
        shell_label = QLabel("Shell", shell_group)
        shell_label.setProperty("timelineToolbarLabel", True)
        shell_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        shell_layout.addWidget(shell_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self._settings_button = QPushButton("Settings", shell_group)
        self._settings_button.setObjectName("timelineEditorSettingsButton")
        self._settings_button.clicked.connect(self.settings_requested.emit)
        shell_layout.addWidget(self._settings_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self._regions_button = QPushButton("Regions", shell_group)
        self._regions_button.setObjectName("timelineEditorRegionsButton")
        self._regions_button.clicked.connect(self.regions_requested.emit)
        shell_layout.addWidget(self._regions_button, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(shell_group, 0, Qt.AlignmentFlag.AlignVCenter)

        self._grid_modes: tuple[TimelineGridMode, ...] = (TimelineGridMode.AUTO,)
        self._grid_mode = TimelineGridMode.AUTO

    def set_state(
        self,
        *,
        edit_mode: str,
        snap_enabled: bool,
        grid_mode: str,
        beat_available: bool,
    ) -> None:
        self._grid_modes = (
            (TimelineGridMode.AUTO, TimelineGridMode.BEAT, TimelineGridMode.OFF)
            if beat_available
            else (TimelineGridMode.AUTO, TimelineGridMode.OFF)
        )
        self._grid_mode = TimelineGridMode(str(grid_mode))
        if self._grid_mode not in self._grid_modes:
            self._grid_mode = self._grid_modes[0]

        for mode_name, button in self._mode_buttons.items():
            button.blockSignals(True)
            button.setChecked(mode_name == edit_mode)
            button.blockSignals(False)
        self._snap_button.blockSignals(True)
        self._snap_button.setChecked(bool(snap_enabled))
        self._snap_button.blockSignals(False)
        self._grid_button.setText(f"Grid: {self._grid_mode.value.title()}")

    def _cycle_grid_mode(self) -> None:
        if not self._grid_modes:
            return
        try:
            current_index = self._grid_modes.index(self._grid_mode)
        except ValueError:
            current_index = 0
        next_mode = self._grid_modes[(current_index + 1) % len(self._grid_modes)]
        self._grid_mode = next_mode
        self._grid_button.setText(f"Grid: {next_mode.value.title()}")
        self.grid_mode_changed.emit(next_mode.value)

    def _create_group_layout(self, parent: QWidget) -> QHBoxLayout:
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(
            TIMELINE_EDITOR_GROUP_PADDING_X_PX,
            TIMELINE_EDITOR_GROUP_PADDING_Y_PX,
            TIMELINE_EDITOR_GROUP_PADDING_X_PX,
            TIMELINE_EDITOR_GROUP_PADDING_Y_PX,
        )
        layout.setSpacing(TIMELINE_EDITOR_GROUP_SPACING_PX)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        return layout


class TransportBar(QWidget):
    def __init__(
        self,
        presentation: TimelinePresentation,
        on_intent: Callable[[TimelineIntent], TimelinePresentation | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.presentation = presentation
        self._on_intent = on_intent
        self._block = TransportBarBlock()
        self._control_rects: dict[str, QRectF] = {}
        self.setMinimumHeight(TIMELINE_TRANSPORT_HEIGHT_PX)
        self.setMaximumHeight(TIMELINE_TRANSPORT_HEIGHT_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        if event is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._control_rects = cast(
            dict[str, QRectF],
            self._block.paint(
                painter,
                TransportLayout.create(width=self.width(), height=self.height()),
                self.presentation,
            ),
        )

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        pos = event.position()
        if (play_rect := self._control_rects.get("play")) is not None and play_rect.contains(pos):
            self._dispatch(Pause() if self.presentation.is_playing else Play())
            return
        if (stop_rect := self._control_rects.get("stop")) is not None and stop_rect.contains(pos):
            self._dispatch(Stop())
            return
        super().mousePressEvent(event)

    def _dispatch(self, intent: TimelineIntent) -> None:
        if self._on_intent is None:
            return
        updated = self._on_intent(intent)
        if updated is not None:
            self.set_presentation(updated)


class TimelineRuler(QWidget):
    seek_requested = pyqtSignal(float)
    region_span_requested = pyqtSignal(float, float)

    def __init__(
        self,
        presentation: TimelinePresentation,
        *,
        header_width: float = float(LAYER_HEADER_WIDTH_PX),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.presentation = presentation
        self._header_width = header_width
        self._block = RulerBlock()
        self._dragging = False
        self._edit_mode = "select"
        self._drag_anchor_time: float | None = None
        self._drag_current_time: float | None = None
        self.setMinimumHeight(RULER_HEIGHT_PX)
        self.setMaximumHeight(RULER_HEIGHT_PX)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        self.presentation = presentation
        self.update()

    def set_editor_mode(self, mode: str) -> None:
        previous_mode = self._edit_mode
        normalized = (mode or "select").strip().lower()
        self._edit_mode = normalized
        if previous_mode != normalized and (
            previous_mode == "region" or normalized == "region"
        ):
            self._dragging = False
            self._drag_anchor_time = None
            self._drag_current_time = None
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        if event is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._block.paint(
            painter,
            RulerLayout(QRectF(0, 0, self.width(), self.height()), self._header_width),
            self.presentation,
        )
        self._paint_region_drag_preview(painter)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.position().x() >= self._header_width
        ):
            self._dragging = True
            current_time = self._seek_time_at_x(event.position().x())
            if self._edit_mode == "region":
                self._drag_anchor_time = current_time
                self._drag_current_time = current_time
                self.update()
            else:
                self.seek_requested.emit(current_time)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if self._dragging and event.buttons() & Qt.MouseButton.LeftButton:
            current_time = self._seek_time_at_x(event.position().x())
            if self._edit_mode == "region":
                self._drag_current_time = current_time
                self.update()
            else:
                self.seek_requested.emit(current_time)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging and self._edit_mode == "region":
                anchor = self._drag_anchor_time
                if anchor is not None:
                    current_time = self._seek_time_at_x(event.position().x())
                    self._drag_current_time = current_time
                    start_seconds = min(anchor, current_time)
                    end_seconds = max(anchor, current_time)
                    if (end_seconds - start_seconds) >= 0.02:
                        self.region_span_requested.emit(start_seconds, end_seconds)
            self._dragging = False
            self._drag_anchor_time = None
            self._drag_current_time = None
            self.update()
        super().mouseReleaseEvent(event)

    def _seek_time_at_x(self, x: float) -> float:
        return seek_time_for_x(
            x,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )

    def _paint_region_drag_preview(self, painter: QPainter) -> None:
        if (
            self._edit_mode != "region"
            or not self._dragging
            or self._drag_anchor_time is None
            or self._drag_current_time is None
        ):
            return

        start_seconds = min(self._drag_anchor_time, self._drag_current_time)
        end_seconds = max(self._drag_anchor_time, self._drag_current_time)
        if (end_seconds - start_seconds) < 1e-6:
            return

        start_x = timeline_x_for_time(
            start_seconds,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        end_x = timeline_x_for_time(
            end_seconds,
            scroll_x=self.presentation.scroll_x,
            pixels_per_second=self.presentation.pixels_per_second,
            content_start_x=self._header_width,
        )
        left = max(float(self._header_width), min(start_x, end_x))
        right = min(float(self.width()), max(start_x, end_x))
        width = max(0.0, right - left)
        if width <= 0.0:
            return

        preview_rect = QRectF(left, 1.0, width, max(1.0, float(self.height()) - 2.0))
        fill = QColor(self._block.playhead_color_hex)
        fill.setAlpha(46)
        border = QColor(self._block.playhead_color_hex)
        border.setAlpha(140)
        painter.fillRect(preview_rect, fill)
        painter.setPen(QPen(border, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(preview_rect.adjusted(0.5, 0.5, -0.5, -0.5))


__all__ = ["TimelineEditorModeBar", "TimelineRuler", "TransportBar"]

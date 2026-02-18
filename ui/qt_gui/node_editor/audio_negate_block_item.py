"""
Audio Negate Block Item

Custom BlockItem subclass with embedded negation controls
(mode selector, crossfade knob, attenuation/gain knobs)
rendered directly inside the node editor via QGraphicsProxyWidget.

Wider than the default node to fit knobs with labels.
The node height scales dynamically based on the current mode's
visible knob rows.
"""
import math
from typing import Optional, Dict, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMenu, QGraphicsProxyWidget, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor,
    QFont, QFontMetrics, QMouseEvent, QPaintEvent,
)

from ui.qt_gui.node_editor.block_item import BlockItem
from ui.qt_gui.design_system import Colors, Sizes, Spacing, Typography, border_radius
from src.utils.message import Log

if TYPE_CHECKING:
    from PyQt6.QtGui import QUndoStack
    from src.features.blocks.domain import Block
    from src.application.api.application_facade import ApplicationFacade


# ======================================================================
# Layout constants
# ======================================================================

NEGATE_MODE_OPTIONS = [
    ("silence",   "Silence"),
    ("attenuate", "Attenuate"),
    ("subtract",  "Subtract"),
]

MODE_BTN_HEIGHT = 22
KNOB_ROW_HEIGHT = 76
KNOB_ROW_SPACING = 4
WIDGET_VERTICAL_MARGIN = 10   # top + bottom padding inside the widget

# How many knob rows each mode shows
_MODE_ROW_COUNT = {
    "silence": 1,      # crossfade
    "attenuate": 2,    # crossfade + attenuation
    "subtract": 3,     # crossfade + gain + onset
}


def _negate_control_height(visible_rows: int) -> int:
    """Calculate the required control height for a given number of visible rows."""
    if visible_rows <= 0:
        return MODE_BTN_HEIGHT + WIDGET_VERTICAL_MARGIN
    rows = visible_rows * KNOB_ROW_HEIGHT + max(0, visible_rows - 1) * KNOB_ROW_SPACING
    return MODE_BTN_HEIGHT + WIDGET_VERTICAL_MARGIN + rows


# ======================================================================
# RotaryKnob -- reusable painted round knob widget
# ======================================================================

class RotaryKnob(QWidget):
    """
    A round turnable knob for selecting a value.

    Features:
    - Circular arc track with filled indicator arc
    - Pointer line at current position
    - Value label below the knob
    - Title label above the knob
    - Linear value mapping
    - Click-drag to change value (vertical drag)
    - Mouse wheel support
    """

    valueChanged = pyqtSignal(float)

    ARC_START_DEG = 225.0
    ARC_SPAN_DEG = 270.0

    def __init__(
        self,
        label: str = "",
        min_val: float = 0.0,
        max_val: float = 100.0,
        default: float = 50.0,
        suffix: str = "",
        decimals: int = 1,
        accent_color: QColor = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._label = label
        self._min = min_val
        self._max = max_val
        self._value = default
        self._suffix = suffix
        self._decimals = decimals
        self._accent = accent_color or Colors.ACCENT_ORANGE

        self._dragging = False
        self._drag_start_y = 0.0
        self._drag_start_ratio = 0.0

        self._knob_diameter = 36
        self._total_height = 72
        self.setFixedSize(self._knob_diameter + 18, self._total_height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def value(self) -> float:
        return self._value

    def setValue(self, val: float):
        val = max(self._min, min(val, self._max))
        if abs(val - self._value) > 1e-4:
            self._value = val
            self.update()
            self.valueChanged.emit(self._value)

    def _value_to_ratio(self, val: float) -> float:
        rng = self._max - self._min
        return (val - self._min) / rng if rng != 0 else 0.0

    def _ratio_to_value(self, ratio: float) -> float:
        ratio = max(0.0, min(1.0, ratio))
        return self._min + ratio * (self._max - self._min)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        d = self._knob_diameter
        cx = w / 2.0

        title_font = Typography.default_font()
        title_font.setPixelSize(8)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(Colors.TEXT_SECONDARY))
        title_rect = QRectF(0, 0, w, 12)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self._label)

        cy = 12 + d / 2.0 + 2
        ratio = self._value_to_ratio(self._value)

        track_rect = QRectF(cx - d / 2, cy - d / 2, d, d)
        track_pen = QPen(Colors.BG_DARK, 3.0)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(
            track_rect,
            int(self.ARC_START_DEG * 16),
            int(-self.ARC_SPAN_DEG * 16),
        )

        if ratio > 0.005:
            fill_pen = QPen(self._accent, 3.0)
            fill_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(fill_pen)
            sweep = -self.ARC_SPAN_DEG * ratio
            painter.drawArc(
                track_rect,
                int(self.ARC_START_DEG * 16),
                int(sweep * 16),
            )

        body_r = d / 2.0 - 4
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(Colors.BG_LIGHT))
        painter.drawEllipse(QPointF(cx, cy), body_r, body_r)

        angle_deg = self.ARC_START_DEG - self.ARC_SPAN_DEG * ratio
        angle_rad = math.radians(angle_deg)
        ptr_inner = body_r * 0.3
        ptr_outer = body_r * 0.85
        pointer_pen = QPen(Colors.TEXT_PRIMARY, 2.0)
        pointer_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pointer_pen)
        painter.drawLine(
            QPointF(cx + ptr_inner * math.cos(angle_rad), cy - ptr_inner * math.sin(angle_rad)),
            QPointF(cx + ptr_outer * math.cos(angle_rad), cy - ptr_outer * math.sin(angle_rad)),
        )

        text_y = cy + d / 2.0 + 2
        val_text = self._format_value(self._value)
        val_font = Typography.default_font()
        val_font.setPixelSize(9)
        painter.setFont(val_font)
        painter.setPen(QPen(Colors.TEXT_SECONDARY))
        text_rect = QRectF(0, text_y, w, 14)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, val_text)

        painter.end()

    def _format_value(self, val: float) -> str:
        if self._decimals == 0:
            return f"{int(val)}{self._suffix}"
        return f"{val:.{self._decimals}f}{self._suffix}"

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_y = event.position().y()
            self._drag_start_ratio = self._value_to_ratio(self._value)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            dy = self._drag_start_y - event.position().y()
            sensitivity = 180.0
            delta_ratio = dy / sensitivity
            new_ratio = max(0.0, min(1.0, self._drag_start_ratio + delta_ratio))
            self.setValue(self._ratio_to_value(new_ratio))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        ratio = self._value_to_ratio(self._value)
        step = 0.02 if abs(delta) < 120 else 0.05
        if delta > 0:
            ratio = min(1.0, ratio + step)
        else:
            ratio = max(0.0, ratio - step)
        self.setValue(self._ratio_to_value(ratio))
        event.accept()


# ======================================================================
# AudioNegateWidget -- embedded control panel for the node
# ======================================================================

class AudioNegateWidget(QWidget):
    """
    Compact audio negation control widget embedded inside the node.

    Layout:
      [Mode selector button]
      [Crossfade knob row]       -- always visible
      [Attenuation knob row]     -- attenuate mode only
      [Subtract Gain knob row]   -- subtract mode only
      [Onset Emphasis knob row]  -- subtract mode only

    Emits visible_rows_changed when the mode changes so the parent
    BlockItem can resize the node dynamically.
    """

    visible_rows_changed = pyqtSignal(int)  # number of visible knob rows

    def __init__(self, block_id: str, facade: "ApplicationFacade", parent=None):
        super().__init__(parent)
        self.block_id = block_id
        self.facade = facade

        self._current_mode = "silence"
        self._save_timer: Optional[QTimer] = None

        self.setFixedWidth(Sizes.NEGATE_BLOCK_WIDTH - 12)
        self._build_ui()
        # Do NOT call _load_from_metadata here; the parent BlockItem
        # will call it after connecting to visible_rows_changed.

    @property
    def visible_row_count(self) -> int:
        return _MODE_ROW_COUNT.get(self._current_mode, 1)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.setSpacing(4)

        self.mode_btn = QPushButton("Silence")
        self.mode_btn.setObjectName("negateModeBtn")
        self.mode_btn.setFixedHeight(MODE_BTN_HEIGHT)
        self.mode_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.mode_btn.clicked.connect(self._show_mode_menu)
        layout.addWidget(self.mode_btn)

        # Knob rows (no scroll area -- height is dynamic)
        self._knob_container = QWidget()
        self._knob_layout = QVBoxLayout(self._knob_container)
        self._knob_layout.setContentsMargins(0, 2, 0, 0)
        self._knob_layout.setSpacing(KNOB_ROW_SPACING)

        # Row 1: Crossfade (always visible)
        self.fade_knob = RotaryKnob(
            label="FADE", min_val=0.0, max_val=100.0, default=10.0,
            suffix="ms", decimals=1, accent_color=Colors.ACCENT_BLUE,
        )
        self.fade_knob.valueChanged.connect(self._on_fade_changed)
        self._fade_row = self._make_knob_row("Crossfade", self.fade_knob)
        self._knob_layout.addWidget(self._fade_row)

        # Row 2: Attenuation (attenuate mode only)
        self.atten_knob = RotaryKnob(
            label="ATTEN", min_val=-60.0, max_val=0.0, default=-20.0,
            suffix="dB", decimals=1, accent_color=Colors.ACCENT_RED,
        )
        self.atten_knob.valueChanged.connect(self._on_attenuation_changed)
        self._atten_row = self._make_knob_row("Reduction", self.atten_knob)
        self._knob_layout.addWidget(self._atten_row)

        # Row 3: Subtract Gain (subtract mode only)
        self.gain_knob = RotaryKnob(
            label="GAIN", min_val=1.0, max_val=10.0, default=1.0,
            suffix="x", decimals=1, accent_color=Colors.ACCENT_ORANGE,
        )
        self.gain_knob.valueChanged.connect(self._on_gain_changed)
        self._gain_row = self._make_knob_row("Sub. Gain", self.gain_knob)
        self._knob_layout.addWidget(self._gain_row)

        # Row 4: Onset Emphasis (subtract mode only)
        self.onset_knob = RotaryKnob(
            label="ONSET", min_val=1.0, max_val=5.0, default=1.0,
            suffix="x", decimals=1, accent_color=Colors.ACCENT_YELLOW,
        )
        self.onset_knob.valueChanged.connect(self._on_onset_changed)
        self._onset_row = self._make_knob_row("Onset Emp.", self.onset_knob)
        self._knob_layout.addWidget(self._onset_row)

        layout.addWidget(self._knob_container)

        self._apply_stylesheet()
        self._update_knob_visibility()

    def _make_knob_row(self, label_text: str, knob: RotaryKnob) -> QWidget:
        """Create a single parameter row: [label on left] [knob on right]."""
        row = QWidget()
        row.setFixedHeight(KNOB_ROW_HEIGHT)
        h = QHBoxLayout(row)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(4)

        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY.name()};
            font-size: 9px;
            font-weight: 600;
            background: transparent;
        """)
        lbl.setFixedWidth(52)
        h.addWidget(lbl)
        h.addStretch()
        h.addWidget(knob, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row.setStyleSheet(f"""
            QWidget {{
                background: {Colors.BG_DARK.name()};
                border-radius: 3px;
            }}
        """)
        return row

    def _apply_stylesheet(self):
        bg = Colors.BG_DARK.name()
        bg_light = Colors.BG_LIGHT.name()
        text = Colors.TEXT_PRIMARY.name()
        text_sec = Colors.TEXT_SECONDARY.name()
        accent = Colors.BLOCK_TRANSFORM.name()
        border = Colors.BORDER.name()

        self.setStyleSheet(f"""
            QWidget {{
                background: transparent;
                color: {text};
                font-size: 9px;
            }}
            QPushButton#negateModeBtn {{
                background: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 3px;
                font-size: 10px;
                font-weight: 600;
                padding: 2px 6px;
                text-align: center;
            }}
            QPushButton#negateModeBtn:hover {{
                background: {bg_light};
                border-color: {accent};
            }}
            QLabel {{
                color: {text_sec};
                font-size: 9px;
                background: transparent;
            }}
        """)

    # ------------------------------------------------------------------
    # Mode menu
    # ------------------------------------------------------------------

    def _show_mode_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 2px;
                font-size: 10px;
            }}
            QMenu::item {{
                padding: 4px 16px 4px 8px;
                border-radius: {border_radius(3)};
                margin: 1px 2px;
            }}
            QMenu::item:selected {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
        """)

        for mode_id, display_name in NEGATE_MODE_OPTIONS:
            action = menu.addAction(display_name)
            action.setData(mode_id)
            if mode_id == self._current_mode:
                font = action.font()
                font.setBold(True)
                action.setFont(font)

        global_pos = self.mode_btn.mapToGlobal(QPoint(0, self.mode_btn.height()))
        selected = menu.exec(global_pos)
        if selected:
            new_mode = selected.data()
            if new_mode and new_mode != self._current_mode:
                self._current_mode = new_mode
                for mid, name in NEGATE_MODE_OPTIONS:
                    if mid == new_mode:
                        self.mode_btn.setText(name)
                        break
                self._update_knob_visibility()
                self._save_metadata("mode", new_mode)

    # ------------------------------------------------------------------
    # Knob visibility per mode
    # ------------------------------------------------------------------

    def _update_knob_visibility(self):
        """Show/hide knob rows based on the current mode and resize."""
        mode = self._current_mode
        self._fade_row.setVisible(True)
        self._atten_row.setVisible(mode == "attenuate")
        self._gain_row.setVisible(mode == "subtract")
        self._onset_row.setVisible(mode == "subtract")

        # Resize widget to fit visible content and notify parent
        h = _negate_control_height(self.visible_row_count)
        self.setFixedHeight(h)
        self.visible_rows_changed.emit(self.visible_row_count)

    # ------------------------------------------------------------------
    # Value change handlers
    # ------------------------------------------------------------------

    def _on_fade_changed(self, value: float):
        self._save_metadata_debounced("fade_ms", round(value, 1))

    def _on_attenuation_changed(self, value: float):
        self._save_metadata_debounced("attenuation_db", round(value, 1))

    def _on_gain_changed(self, value: float):
        self._save_metadata_debounced("subtract_gain", round(value, 1))

    def _on_onset_changed(self, value: float):
        self._save_metadata_debounced("onset_emphasis", round(value, 1))

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def _load_from_metadata(self):
        try:
            result = self.facade.describe_block(self.block_id)
            if not result.success or not result.data:
                return

            block = result.data
            meta = block.metadata or {}

            mode = meta.get("mode", "silence")
            self._current_mode = mode
            for mid, name in NEGATE_MODE_OPTIONS:
                if mid == mode:
                    self.mode_btn.setText(name)
                    break

            self.fade_knob.setValue(float(meta.get("fade_ms", 10.0)))
            self.atten_knob.setValue(float(meta.get("attenuation_db", -20.0)))
            self.gain_knob.setValue(float(meta.get("subtract_gain", 1.0)))
            self.onset_knob.setValue(float(meta.get("onset_emphasis", 1.0)))

            self._update_knob_visibility()
        except Exception as e:
            Log.warning(f"AudioNegateWidget: Error loading metadata: {e}")

    def _save_metadata(self, key: str, value):
        try:
            from src.application.commands.block_commands import UpdateBlockMetadataCommand

            cmd = UpdateBlockMetadataCommand(
                facade=self.facade,
                block_id=self.block_id,
                key=key,
                new_value=value,
                description=f"Set {key}",
            )
            self.facade.command_bus.execute(cmd)
        except Exception as e:
            Log.warning(f"AudioNegateWidget: Error saving metadata '{key}': {e}")

    def _save_metadata_debounced(self, key: str, value):
        if self._save_timer is not None:
            self._save_timer.stop()
            self._save_timer.deleteLater()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(lambda: self._save_metadata(key, value))
        self._save_timer.start(150)


# ======================================================================
# AudioNegateBlockItem -- dynamic-height node with embedded controls
# ======================================================================

class AudioNegateBlockItem(BlockItem):
    """
    BlockItem subclass that embeds an AudioNegateWidget and dynamically
    resizes when the negation mode changes (different modes show
    different numbers of knob rows).
    """

    def __init__(
        self,
        block: "Block",
        facade: "ApplicationFacade",
        undo_stack: Optional["QUndoStack"] = None,
    ):
        # Set initial control height before super().__init__ calls
        # _calculate_dimensions.
        self._negate_control_height = _negate_control_height(1)  # silence default

        super().__init__(block, facade, undo_stack)

        self._negate_widget = AudioNegateWidget(block.id, facade)
        self._negate_widget.visible_rows_changed.connect(self._on_visible_rows_changed)

        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setWidget(self._negate_widget)

        # Load metadata (triggers visible_rows_changed -> resize)
        self._negate_widget._load_from_metadata()

        self._position_proxy()
        self._subscribe_negate_events()

    # ------------------------------------------------------------------
    # Dynamic dimensions
    # ------------------------------------------------------------------

    def _calculate_dimensions(self):
        super()._calculate_dimensions()
        self._width = Sizes.NEGATE_BLOCK_WIDTH
        self._height += self._negate_control_height

    def _on_visible_rows_changed(self, row_count: int):
        """Resize the node when the mode changes visible row count."""
        new_height = _negate_control_height(row_count)
        if new_height == self._negate_control_height:
            return

        self.prepareGeometryChange()
        self._negate_control_height = new_height
        self._calculate_dimensions()
        self._position_proxy()

        for conn in self.connections:
            conn.update_position()

        self.update()

    def _position_proxy(self):
        rect = QRectF(
            -self._width / 2,
            -self._height / 2,
            self._width,
            self._height,
        )
        proxy_x = rect.left() + 6
        proxy_y = rect.bottom() - self._negate_control_height - 4
        self._proxy.setPos(proxy_x, proxy_y)

    # ------------------------------------------------------------------
    # Event subscriptions
    # ------------------------------------------------------------------

    def _subscribe_negate_events(self):
        if not self.facade or not self.facade.event_bus:
            return
        self.facade.event_bus.subscribe("BlockUpdated", self._on_negate_block_updated)

    def _unsubscribe_negate_events(self):
        if not self.facade or not self.facade.event_bus:
            return
        try:
            self.facade.event_bus.unsubscribe("BlockUpdated", self._on_negate_block_updated)
        except Exception as e:
            Log.debug(f"AudioNegateBlockItem: Error unsubscribing: {e}")

    def _on_negate_block_updated(self, event):
        if not self._is_valid():
            return
        try:
            updated_id = event.data.get("id") if hasattr(event, "data") and event.data else None
            if updated_id == self.block.id:
                QTimer.singleShot(100, self._safe_refresh_negate)
        except Exception:
            pass

    def _safe_refresh_negate(self):
        try:
            if self._is_valid() and self._negate_widget:
                self._negate_widget._load_from_metadata()
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def itemChange(self, change, value):
        from PyQt6.QtWidgets import QGraphicsItem

        if change == QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged:
            if value is None:
                self._unsubscribe_negate_events()
        return super().itemChange(change, value)

"""
Audio Filter Block Item

Custom BlockItem subclass with embedded filter controls
(filter type selector, frequency knob(s)) rendered directly
inside the node editor via QGraphicsProxyWidget.
"""
import math
from typing import Optional, Dict, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMenu, QGraphicsProxyWidget, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QPoint, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QConicalGradient,
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
# Filter type data (mirrors FILTER_TYPES in audio_filter_block.py)
# ======================================================================

FILTER_TYPE_OPTIONS = [
    ("lowpass",   "Low-Pass"),
    ("highpass",  "High-Pass"),
    ("bandpass",  "Band-Pass"),
    ("bandstop",  "Band-Stop"),
    ("lowshelf",  "Low-Shelf"),
    ("highshelf", "High-Shelf"),
    ("peak",      "Peak (Bell)"),
]

# Which filter types need a second frequency knob
_DUAL_FREQ_TYPES = {"bandpass", "bandstop"}


# ======================================================================
# RotaryKnob -- custom painted round knob widget
# ======================================================================

class RotaryKnob(QWidget):
    """
    A round turnable knob for selecting a value.

    Features:
    - Circular arc track with filled indicator arc
    - Small pointer dot at current position
    - Value label below the knob
    - Logarithmic or linear value mapping
    - Click-drag to change value (vertical drag)
    """

    valueChanged = pyqtSignal(float)

    # Arc geometry: start at 225 deg, sweep 270 deg (clockwise)
    ARC_START_DEG = 225.0
    ARC_SPAN_DEG = 270.0

    def __init__(
        self,
        label: str = "FREQ",
        min_val: float = 20.0,
        max_val: float = 20000.0,
        default: float = 1000.0,
        logarithmic: bool = True,
        suffix: str = "Hz",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._label = label
        self._min = min_val
        self._max = max_val
        self._value = default
        self._logarithmic = logarithmic
        self._suffix = suffix

        # Drag state
        self._dragging = False
        self._drag_start_y = 0.0
        self._drag_start_ratio = 0.0

        # Size
        self._knob_diameter = 40
        self._total_height = 64  # knob + label below
        self.setFixedSize(self._knob_diameter + 12, self._total_height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------------------------------------------------
    # Value mapping
    # ------------------------------------------------------------------

    def value(self) -> float:
        return self._value

    def setValue(self, val: float):
        val = max(self._min, min(val, self._max))
        if abs(val - self._value) > 0.001:
            self._value = val
            self.update()
            self.valueChanged.emit(self._value)

    def _value_to_ratio(self, val: float) -> float:
        """Map a value to 0..1 ratio."""
        if self._logarithmic and self._min > 0 and self._max > 0:
            log_min = math.log(self._min)
            log_max = math.log(self._max)
            return (math.log(max(val, self._min)) - log_min) / (log_max - log_min)
        return (val - self._min) / (self._max - self._min) if self._max != self._min else 0.0

    def _ratio_to_value(self, ratio: float) -> float:
        """Map a 0..1 ratio to a value."""
        ratio = max(0.0, min(1.0, ratio))
        if self._logarithmic and self._min > 0 and self._max > 0:
            log_min = math.log(self._min)
            log_max = math.log(self._max)
            return math.exp(log_min + ratio * (log_max - log_min))
        return self._min + ratio * (self._max - self._min)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        d = self._knob_diameter
        cx = w / 2.0
        cy = d / 2.0 + 2  # small top margin

        ratio = self._value_to_ratio(self._value)

        # --- Track arc (background) ---
        track_rect = QRectF(cx - d / 2, cy - d / 2, d, d)
        track_pen = QPen(Colors.BG_DARK, 3.0)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Qt drawArc uses 1/16th degree units, positive = counter-clockwise
        painter.drawArc(
            track_rect,
            int(self.ARC_START_DEG * 16),
            int(-self.ARC_SPAN_DEG * 16),
        )

        # --- Filled arc (value indicator) ---
        if ratio > 0.005:
            fill_pen = QPen(Colors.ACCENT_ORANGE, 3.0)
            fill_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(fill_pen)
            sweep = -self.ARC_SPAN_DEG * ratio
            painter.drawArc(
                track_rect,
                int(self.ARC_START_DEG * 16),
                int(sweep * 16),
            )

        # --- Knob body (filled circle, slightly smaller) ---
        body_r = d / 2.0 - 4
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(Colors.BG_LIGHT))
        painter.drawEllipse(QPointF(cx, cy), body_r, body_r)

        # --- Pointer line ---
        angle_deg = self.ARC_START_DEG - self.ARC_SPAN_DEG * ratio
        angle_rad = math.radians(angle_deg)
        ptr_inner = body_r * 0.3
        ptr_outer = body_r * 0.85
        px_inner = cx + ptr_inner * math.cos(angle_rad)
        py_inner = cy - ptr_inner * math.sin(angle_rad)
        px_outer = cx + ptr_outer * math.cos(angle_rad)
        py_outer = cy - ptr_outer * math.sin(angle_rad)

        pointer_pen = QPen(Colors.TEXT_PRIMARY, 2.0)
        pointer_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pointer_pen)
        painter.drawLine(QPointF(px_inner, py_inner), QPointF(px_outer, py_outer))

        # --- Value text below knob ---
        text_y = cy + d / 2.0 + 2
        val_text = self._format_value(self._value)
        font = Typography.default_font()
        font.setPixelSize(9)
        painter.setFont(font)
        painter.setPen(QPen(Colors.TEXT_SECONDARY))
        text_rect = QRectF(0, text_y, w, 14)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, val_text)

        painter.end()

    def _format_value(self, val: float) -> str:
        """Format the value for display."""
        if val >= 10000:
            return f"{val / 1000:.1f}k"
        elif val >= 1000:
            return f"{val / 1000:.2f}k"
        elif val >= 100:
            return f"{int(val)}"
        elif val >= 10:
            return f"{val:.1f}"
        else:
            return f"{val:.1f}"

    # ------------------------------------------------------------------
    # Mouse interaction (vertical drag to change value)
    # ------------------------------------------------------------------

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
            # Drag up = increase, drag down = decrease
            dy = self._drag_start_y - event.position().y()
            sensitivity = 200.0  # pixels for full range
            delta_ratio = dy / sensitivity
            new_ratio = max(0.0, min(1.0, self._drag_start_ratio + delta_ratio))
            new_value = self._ratio_to_value(new_ratio)
            self.setValue(new_value)
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
        """Allow mouse wheel to change value."""
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
# AudioFilterWidget -- embedded control panel
# ======================================================================

class AudioFilterWidget(QWidget):
    """
    Compact audio filter control widget embedded inside the node.

    Contains:
    - Filter type selector button (opens QMenu popup)
    - One or two RotaryKnob widgets for frequency control
    """

    def __init__(self, block_id: str, facade: "ApplicationFacade", parent=None):
        super().__init__(parent)
        self.block_id = block_id
        self.facade = facade

        self._current_filter_type = "lowpass"
        self._save_timer: Optional[QTimer] = None

        self.setFixedWidth(Sizes.BLOCK_WIDTH - 12)
        self._build_ui()
        self._load_from_metadata()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(3)

        # --- Filter type selector (button + menu, avoids QComboBox popup clipping) ---
        self.type_btn = QPushButton("Low-Pass")
        self.type_btn.setObjectName("filterTypeBtn")
        self.type_btn.setFixedHeight(20)
        self.type_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.type_btn.clicked.connect(self._show_filter_type_menu)
        layout.addWidget(self.type_btn)

        # --- Knob row ---
        knob_row = QHBoxLayout()
        knob_row.setContentsMargins(0, 0, 0, 0)
        knob_row.setSpacing(2)

        # Primary frequency knob (always visible)
        self.freq_knob = RotaryKnob(
            label="FREQ",
            min_val=20.0,
            max_val=20000.0,
            default=1000.0,
            logarithmic=True,
            suffix="Hz",
        )
        self.freq_knob.valueChanged.connect(self._on_freq_changed)
        knob_row.addWidget(self.freq_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        # Secondary frequency knob (bandpass/bandstop only)
        self.freq_high_knob = RotaryKnob(
            label="HIGH",
            min_val=20.0,
            max_val=20000.0,
            default=8000.0,
            logarithmic=True,
            suffix="Hz",
        )
        self.freq_high_knob.valueChanged.connect(self._on_freq_high_changed)
        knob_row.addWidget(self.freq_high_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(knob_row)

        # Label row under knobs
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(2)

        self.freq_label = QLabel("Freq")
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_row.addWidget(self.freq_label)

        self.freq_high_label = QLabel("High")
        self.freq_high_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_row.addWidget(self.freq_high_label)

        layout.addLayout(label_row)

        self._apply_stylesheet()
        self._update_knob_visibility()

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
            QPushButton#filterTypeBtn {{
                background: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 3px;
                font-size: 9px;
                padding: 2px 4px;
                text-align: center;
            }}
            QPushButton#filterTypeBtn:hover {{
                background: {bg_light};
                border-color: {accent};
            }}
            QLabel {{
                color: {text_sec};
                font-size: 8px;
                background: transparent;
            }}
        """)

    # ------------------------------------------------------------------
    # Filter type menu
    # ------------------------------------------------------------------

    def _show_filter_type_menu(self):
        """Show a QMenu popup with all filter type options."""
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

        for type_id, display_name in FILTER_TYPE_OPTIONS:
            action = menu.addAction(display_name)
            action.setData(type_id)
            if type_id == self._current_filter_type:
                font = action.font()
                font.setBold(True)
                action.setFont(font)

        # Show at the button's global position
        global_pos = self.type_btn.mapToGlobal(QPoint(0, self.type_btn.height()))
        selected = menu.exec(global_pos)
        if selected:
            new_type = selected.data()
            if new_type and new_type != self._current_filter_type:
                self._current_filter_type = new_type
                # Update button text
                for tid, name in FILTER_TYPE_OPTIONS:
                    if tid == new_type:
                        self.type_btn.setText(name)
                        break
                self._update_knob_visibility()
                self._save_metadata("filter_type", new_type)

    # ------------------------------------------------------------------
    # Knob visibility
    # ------------------------------------------------------------------

    def _update_knob_visibility(self):
        """Show/hide the second frequency knob based on filter type."""
        needs_high = self._current_filter_type in _DUAL_FREQ_TYPES
        self.freq_high_knob.setVisible(needs_high)
        self.freq_high_label.setVisible(needs_high)

    # ------------------------------------------------------------------
    # Value change handlers
    # ------------------------------------------------------------------

    def _on_freq_changed(self, value: float):
        self._save_metadata_debounced("cutoff_freq", round(value, 1))

    def _on_freq_high_changed(self, value: float):
        self._save_metadata_debounced("cutoff_freq_high", round(value, 1))

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def _load_from_metadata(self):
        """Load current settings from block metadata."""
        try:
            result = self.facade.describe_block(self.block_id)
            if not result.success or not result.data:
                return

            block = result.data
            meta = block.metadata or {}

            # Filter type
            ft = meta.get("filter_type", "lowpass")
            self._current_filter_type = ft
            for tid, name in FILTER_TYPE_OPTIONS:
                if tid == ft:
                    self.type_btn.setText(name)
                    break

            # Frequencies
            self.freq_knob.setValue(float(meta.get("cutoff_freq", 1000.0)))
            self.freq_high_knob.setValue(float(meta.get("cutoff_freq_high", 8000.0)))

            self._update_knob_visibility()
        except Exception as e:
            Log.warning(f"AudioFilterWidget: Error loading metadata: {e}")

    def _save_metadata(self, key: str, value):
        """Save a single metadata key immediately."""
        try:
            self.facade.update_block_metadata(self.block_id, {key: value})
        except Exception as e:
            Log.warning(f"AudioFilterWidget: Error saving metadata '{key}': {e}")

    def _save_metadata_debounced(self, key: str, value):
        """Save metadata with debouncing (avoids excessive writes during knob drag)."""
        if self._save_timer is not None:
            self._save_timer.stop()
            self._save_timer.deleteLater()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(lambda: self._save_metadata(key, value))
        self._save_timer.start(150)  # 150ms debounce


# ======================================================================
# AudioFilterBlockItem -- custom node with embedded filter controls
# ======================================================================

class AudioFilterBlockItem(BlockItem):
    """
    BlockItem subclass that embeds an AudioFilterWidget via
    QGraphicsProxyWidget inside the node body.

    The widget sits below the port zone and provides filter type
    selection and frequency knobs directly in the node.
    """

    def __init__(
        self,
        block: "Block",
        facade: "ApplicationFacade",
        undo_stack: Optional["QUndoStack"] = None,
    ):
        super().__init__(block, facade, undo_stack)

        # Build and embed the filter widget
        self._filter_widget = AudioFilterWidget(block.id, facade)
        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setWidget(self._filter_widget)

        # Position the proxy inside the block body (below ports)
        self._position_proxy()

        # Subscribe to metadata change events to stay in sync
        self._subscribe_filter_events()

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def _calculate_dimensions(self):
        """Extend base dimensions to accommodate the filter controls."""
        super()._calculate_dimensions()
        self._height += Sizes.FILTER_CONTROL_HEIGHT

    def _position_proxy(self):
        """Position the proxy widget inside the block, below the port zone."""
        rect = QRectF(
            -self._width / 2,
            -self._height / 2,
            self._width,
            self._height,
        )
        proxy_x = rect.left() + 6
        proxy_y = rect.bottom() - Sizes.FILTER_CONTROL_HEIGHT - 4
        self._proxy.setPos(proxy_x, proxy_y)

    # ------------------------------------------------------------------
    # Event subscriptions
    # ------------------------------------------------------------------

    def _subscribe_filter_events(self):
        """Subscribe to events that indicate block metadata changed."""
        if not self.facade or not self.facade.event_bus:
            return
        self.facade.event_bus.subscribe("BlockUpdated", self._on_filter_block_updated)

    def _unsubscribe_filter_events(self):
        if not self.facade or not self.facade.event_bus:
            return
        try:
            self.facade.event_bus.unsubscribe("BlockUpdated", self._on_filter_block_updated)
        except Exception as e:
            Log.debug(f"AudioFilterBlockItem: Error unsubscribing: {e}")

    def _on_filter_block_updated(self, event):
        """If this block's metadata changed externally, refresh the widget."""
        if not self._is_valid():
            return
        try:
            updated_id = event.data.get("id") if hasattr(event, "data") and event.data else None
            if updated_id == self.block.id:
                QTimer.singleShot(100, self._safe_refresh_filter)
        except Exception:
            pass

    def _safe_refresh_filter(self):
        try:
            if self._is_valid() and self._filter_widget:
                self._filter_widget._load_from_metadata()
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def itemChange(self, change, value):
        from PyQt6.QtWidgets import QGraphicsItem

        if change == QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged:
            if value is None:
                self._unsubscribe_filter_events()
        return super().itemChange(change, value)

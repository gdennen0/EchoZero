"""
EQ Bands Block Item

Custom BlockItem subclass with embedded knob controls
(3 per band: Low freq, High freq, Gain) rendered directly
inside the node editor via QGraphicsProxyWidget.

Each band row shows [LOW] [HIGH] [GAIN] knobs side-by-side.
Multiple bands stack vertically. The node dynamically resizes
its height when bands are added or removed.
"""
import math
from typing import Optional, List, Dict, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QGraphicsProxyWidget, QSizePolicy,
    QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor,
    QFont, QMouseEvent, QPaintEvent,
)

from ui.qt_gui.node_editor.block_item import BlockItem
from ui.qt_gui.design_system import Colors, Sizes, Spacing, Typography
from src.utils.message import Log

if TYPE_CHECKING:
    from PyQt6.QtGui import QUndoStack
    from src.features.blocks.domain import Block
    from src.application.api.application_facade import ApplicationFacade


# ======================================================================
# Layout constants
# ======================================================================

BAND_ROW_HEIGHT = 56       # Height of one band row (3 knobs)
BAND_ROW_SPACING = 2       # Spacing between band rows
BUTTON_ROW_HEIGHT = 20     # Height of add/remove button row
WIDGET_MARGIN = 8          # Total vertical margins in the widget


def _control_height_for_bands(count: int) -> int:
    """Calculate the required control height for a given number of bands."""
    if count <= 0:
        return BUTTON_ROW_HEIGHT + WIDGET_MARGIN
    rows_height = count * BAND_ROW_HEIGHT + (count - 1) * BAND_ROW_SPACING
    return rows_height + BUTTON_ROW_HEIGHT + WIDGET_MARGIN


# ======================================================================
# EQKnob -- compact knob supporting both log (freq) and linear (gain)
# ======================================================================

class EQKnob(QWidget):
    """
    Compact rotary knob for EQ band parameters.

    Supports logarithmic mapping for frequency knobs and linear
    mapping for gain knobs. The arc fill style adapts: frequency
    knobs fill from the left; gain knobs fill from center (0 dB).
    """

    valueChanged = pyqtSignal(float)

    ARC_START_DEG = 225.0
    ARC_SPAN_DEG = 270.0

    def __init__(
        self,
        label: str = "",
        min_val: float = 20.0,
        max_val: float = 20000.0,
        default: float = 1000.0,
        logarithmic: bool = False,
        center_fill: bool = False,
        accent: QColor = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._label = label
        self._min = min_val
        self._max = max_val
        self._value = default
        self._logarithmic = logarithmic
        self._center_fill = center_fill
        self._accent = accent or Colors.ACCENT_ORANGE

        self._dragging = False
        self._drag_start_y = 0.0
        self._drag_start_ratio = 0.0

        self._knob_diameter = 30
        self._total_height = 54  # label + knob + value
        self.setFixedSize(self._knob_diameter + 12, self._total_height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def value(self) -> float:
        return self._value

    def setValue(self, val: float):
        val = max(self._min, min(val, self._max))
        if abs(val - self._value) > 0.01:
            self._value = val
            self.update()
            self.valueChanged.emit(self._value)

    def _value_to_ratio(self, val: float) -> float:
        if self._logarithmic and self._min > 0 and self._max > 0:
            log_min = math.log(self._min)
            log_max = math.log(self._max)
            return (math.log(max(val, self._min)) - log_min) / (log_max - log_min)
        rng = self._max - self._min
        return (val - self._min) / rng if rng != 0 else 0.0

    def _ratio_to_value(self, ratio: float) -> float:
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

        # -- Title label --
        title_font = QFont()
        title_font.setPixelSize(7)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(Colors.TEXT_SECONDARY))
        title_rect = QRectF(0, 0, w, 10)
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            self._label,
        )

        cy = 10 + d / 2.0 + 1
        ratio = self._value_to_ratio(self._value)

        # -- Track arc --
        track_rect = QRectF(cx - d / 2, cy - d / 2, d, d)
        track_pen = QPen(Colors.BG_DARK, 2.5)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(
            track_rect,
            int(self.ARC_START_DEG * 16),
            int(-self.ARC_SPAN_DEG * 16),
        )

        # -- Filled arc --
        if self._center_fill:
            center_ratio = 0.5
            if abs(ratio - center_ratio) > 0.005:
                if self._value >= 0:
                    arc_color = QColor(80, 200, 100)
                else:
                    arc_color = QColor(220, 140, 60)
                fill_pen = QPen(arc_color, 2.5)
                fill_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(fill_pen)
                start_angle = self.ARC_START_DEG - self.ARC_SPAN_DEG * center_ratio
                end_angle = self.ARC_START_DEG - self.ARC_SPAN_DEG * ratio
                painter.drawArc(
                    track_rect,
                    int(start_angle * 16),
                    int((end_angle - start_angle) * 16),
                )
        else:
            if ratio > 0.005:
                fill_pen = QPen(self._accent, 2.5)
                fill_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(fill_pen)
                painter.drawArc(
                    track_rect,
                    int(self.ARC_START_DEG * 16),
                    int(-self.ARC_SPAN_DEG * ratio * 16),
                )

        # -- Knob body --
        body_r = d / 2.0 - 3
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(Colors.BG_LIGHT))
        painter.drawEllipse(QPointF(cx, cy), body_r, body_r)

        # -- Pointer line --
        angle_deg = self.ARC_START_DEG - self.ARC_SPAN_DEG * ratio
        angle_rad = math.radians(angle_deg)
        ptr_inner = body_r * 0.3
        ptr_outer = body_r * 0.85
        painter.setPen(QPen(Colors.TEXT_PRIMARY, 1.5, cap=Qt.PenCapStyle.RoundCap))
        painter.drawLine(
            QPointF(cx + ptr_inner * math.cos(angle_rad), cy - ptr_inner * math.sin(angle_rad)),
            QPointF(cx + ptr_outer * math.cos(angle_rad), cy - ptr_outer * math.sin(angle_rad)),
        )

        # -- Value text --
        text_y = cy + d / 2.0 + 1
        val_font = QFont()
        val_font.setPixelSize(8)
        painter.setFont(val_font)

        if self._center_fill:
            if abs(self._value) < 0.1:
                painter.setPen(QPen(Colors.TEXT_SECONDARY))
                val_text = "0dB"
            elif self._value > 0:
                painter.setPen(QPen(QColor(80, 200, 100)))
                val_text = f"+{self._value:.1f}"
            else:
                painter.setPen(QPen(QColor(220, 140, 60)))
                val_text = f"{self._value:.1f}"
        else:
            painter.setPen(QPen(Colors.TEXT_SECONDARY))
            val_text = self._format_freq(self._value)

        text_rect = QRectF(0, text_y, w, 12)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            val_text,
        )
        painter.end()

    @staticmethod
    def _format_freq(val: float) -> str:
        if val >= 10000:
            return f"{val / 1000:.1f}k"
        elif val >= 1000:
            return f"{val / 1000:.1f}k"
        elif val >= 100:
            return f"{int(val)}"
        else:
            return f"{val:.0f}"

    # ------------------------------------------------------------------
    # Mouse interaction
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
            dy = self._drag_start_y - event.position().y()
            sensitivity = 160.0
            delta_ratio = dy / sensitivity
            new_ratio = max(0.0, min(1.0, self._drag_start_ratio + delta_ratio))
            new_val = self._ratio_to_value(new_ratio)
            if self._center_fill and abs(new_val) < 0.5:
                new_val = 0.0
            self.setValue(new_val)
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

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click to reset: gain to 0dB, freq to midpoint."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._center_fill:
                self.setValue(0.0)
            else:
                self.setValue(self._ratio_to_value(0.5))
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


# ======================================================================
# BandRow -- one row of 3 knobs for a single EQ band
# ======================================================================

class BandRow(QWidget):
    """
    A single EQ band row: [LOW freq] [HIGH freq] [GAIN dB]
    """

    changed = pyqtSignal()

    def __init__(
        self,
        index: int,
        freq_low: float = 60.0,
        freq_high: float = 250.0,
        gain_db: float = 0.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._index = index
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self.low_knob = EQKnob(
            label="LOW",
            min_val=20.0, max_val=20000.0, default=freq_low,
            logarithmic=True, accent=Colors.ACCENT_BLUE,
        )
        self.low_knob.valueChanged.connect(self._on_changed)
        layout.addWidget(self.low_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.high_knob = EQKnob(
            label="HIGH",
            min_val=20.0, max_val=20000.0, default=freq_high,
            logarithmic=True, accent=Colors.ACCENT_ORANGE,
        )
        self.high_knob.valueChanged.connect(self._on_changed)
        layout.addWidget(self.high_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.gain_knob = EQKnob(
            label="GAIN",
            min_val=-24.0, max_val=24.0, default=gain_db,
            logarithmic=False, center_fill=True,
        )
        self.gain_knob.valueChanged.connect(self._on_changed)
        layout.addWidget(self.gain_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setFixedHeight(BAND_ROW_HEIGHT)

    def _on_changed(self, _val: float):
        if not self._updating:
            self.changed.emit()

    def get_band_data(self) -> Dict[str, float]:
        return {
            "freq_low": round(self.low_knob.value(), 1),
            "freq_high": round(self.high_knob.value(), 1),
            "gain_db": round(self.gain_knob.value(), 1),
        }

    def set_band_data(self, freq_low: float, freq_high: float, gain_db: float):
        self._updating = True
        self.low_knob.setValue(freq_low)
        self.high_knob.setValue(freq_high)
        self.gain_knob.setValue(gain_db)
        self._updating = False


# ======================================================================
# EQBandsWidget -- main embedded widget
# ======================================================================

class EQBandsWidget(QWidget):
    """
    Compact EQ band editor embedded inside the node.

    Stacks rows of 3 knobs (Low, High, Gain) per band with
    add/remove buttons. Emits band_count_changed when the number
    of bands changes so the parent BlockItem can resize.
    """

    band_count_changed = pyqtSignal(int)   # new band count

    def __init__(self, block_id: str, facade: "ApplicationFacade", parent=None):
        super().__init__(parent)
        self.block_id = block_id
        self.facade = facade

        self._band_rows: List[BandRow] = []
        self._save_timer: Optional[QTimer] = None
        self._loading = False

        self._build_ui()
        # Do NOT call _load_from_metadata here; the parent BlockItem
        # will call it after connecting to band_count_changed.

    @property
    def band_count(self) -> int:
        return len(self._band_rows)

    def _build_ui(self):
        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(2, 2, 2, 4)
        self._root_layout.setSpacing(2)

        # Band container (no scroll -- height is dynamic)
        self._band_container = QWidget()
        self._band_layout = QVBoxLayout(self._band_container)
        self._band_layout.setContentsMargins(0, 0, 0, 0)
        self._band_layout.setSpacing(BAND_ROW_SPACING)
        self._root_layout.addWidget(self._band_container)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(2)

        self.add_btn = QPushButton("+")
        self.add_btn.setObjectName("eqAddBtn")
        self.add_btn.setFixedSize(22, 16)
        self.add_btn.setToolTip("Add band")
        self.add_btn.clicked.connect(self._on_add_band)
        btn_row.addWidget(self.add_btn)

        self.remove_btn = QPushButton("-")
        self.remove_btn.setObjectName("eqRemoveBtn")
        self.remove_btn.setFixedSize(22, 16)
        self.remove_btn.setToolTip("Remove last band")
        self.remove_btn.clicked.connect(self._on_remove_band)
        btn_row.addWidget(self.remove_btn)

        btn_row.addStretch()
        self._root_layout.addLayout(btn_row)

        self._apply_stylesheet()

    def _apply_stylesheet(self):
        bg = Colors.BG_DARK.name()
        bg_light = Colors.BG_LIGHT.name()
        text = Colors.TEXT_PRIMARY.name()
        accent = Colors.BLOCK_TRANSFORM.name()
        border = Colors.BORDER.name()

        self.setStyleSheet(f"""
            QWidget {{
                background: transparent;
                color: {text};
                font-size: 9px;
            }}
            QPushButton#eqAddBtn, QPushButton#eqRemoveBtn {{
                background: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 2px;
                font-size: 10px;
                font-weight: bold;
                padding: 0px;
            }}
            QPushButton#eqAddBtn:hover, QPushButton#eqRemoveBtn:hover {{
                background: {bg_light};
                border-color: {accent};
            }}
        """)

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _resize_to_fit(self):
        """Resize the widget to exactly fit its content and notify parent."""
        h = _control_height_for_bands(len(self._band_rows))
        self.setFixedHeight(h)
        self.setFixedWidth(Sizes.BLOCK_WIDTH - 12)
        self.band_count_changed.emit(len(self._band_rows))

    # ------------------------------------------------------------------
    # Band row management
    # ------------------------------------------------------------------

    def _clear_rows(self):
        for row in self._band_rows:
            row.changed.disconnect()
            self._band_layout.removeWidget(row)
            row.deleteLater()
        self._band_rows.clear()

    def _add_row(self, freq_low: float, freq_high: float, gain_db: float) -> BandRow:
        index = len(self._band_rows)
        row = BandRow(index, freq_low, freq_high, gain_db)
        row.changed.connect(self._on_band_changed)

        if index % 2 == 1:
            row.setStyleSheet("background: rgba(255,255,255,6); border-radius: 3px;")

        self._band_layout.addWidget(row)
        self._band_rows.append(row)
        return row

    def _rebuild_from_data(self, bands: List[Dict]):
        self._loading = True
        self._clear_rows()
        for band in bands:
            self._add_row(
                float(band.get("freq_low", 60.0)),
                float(band.get("freq_high", 250.0)),
                float(band.get("gain_db", 0.0)),
            )
        self.remove_btn.setEnabled(len(bands) > 0)
        self._loading = False
        self._resize_to_fit()

    # ------------------------------------------------------------------
    # Data sync
    # ------------------------------------------------------------------

    def _load_from_metadata(self):
        try:
            result = self.facade.describe_block(self.block_id)
            if not result.success or not result.data:
                return
            block = result.data
            meta = block.metadata or {}
            bands = meta.get("bands", [])
            if not isinstance(bands, list):
                bands = []
            self._rebuild_from_data(bands)
        except Exception as e:
            Log.warning(f"EQBandsWidget: Error loading metadata: {e}")

    def _collect_bands(self) -> List[Dict]:
        return [row.get_band_data() for row in self._band_rows]

    def _save_bands(self):
        try:
            from src.application.commands.block_commands import UpdateBlockMetadataCommand

            cmd = UpdateBlockMetadataCommand(
                facade=self.facade,
                block_id=self.block_id,
                key="bands",
                new_value=self._collect_bands(),
                description="Update EQ bands",
            )
            self.facade.command_bus.execute(cmd)
        except Exception as e:
            Log.warning(f"EQBandsWidget: Error saving bands: {e}")

    def _save_debounced(self):
        if self._save_timer is not None:
            self._save_timer.stop()
            self._save_timer.deleteLater()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_bands)
        self._save_timer.start(200)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_band_changed(self):
        if not self._loading:
            self._save_debounced()

    def _on_add_band(self):
        if self._band_rows:
            last = self._band_rows[-1].get_band_data()
            new_low = last["freq_high"]
            new_high = min(new_low * 2.0, 20000.0)
            if new_low >= 19999.0:
                new_low, new_high = 1000.0, 4000.0
        else:
            new_low, new_high = 60.0, 250.0

        self._add_row(new_low, new_high, 0.0)
        self.remove_btn.setEnabled(True)
        self._resize_to_fit()
        self._save_bands()

    def _on_remove_band(self):
        if self._band_rows:
            row = self._band_rows.pop()
            row.changed.disconnect()
            self._band_layout.removeWidget(row)
            row.deleteLater()
            self.remove_btn.setEnabled(len(self._band_rows) > 0)
            self._resize_to_fit()
            self._save_bands()


# ======================================================================
# EQBandsBlockItem -- dynamic-height node with embedded EQ knobs
# ======================================================================

class EQBandsBlockItem(BlockItem):
    """
    BlockItem subclass that embeds an EQBandsWidget and dynamically
    resizes when bands are added or removed.
    """

    def __init__(
        self,
        block: "Block",
        facade: "ApplicationFacade",
        undo_stack: Optional["QUndoStack"] = None,
    ):
        # _eq_control_height is used by _calculate_dimensions (called
        # in super().__init__), so set a default before super init.
        self._eq_control_height = _control_height_for_bands(0)

        super().__init__(block, facade, undo_stack)

        self._eq_widget = EQBandsWidget(block.id, facade)
        self._eq_widget.band_count_changed.connect(self._on_band_count_changed)

        self._proxy = QGraphicsProxyWidget(self)
        self._proxy.setWidget(self._eq_widget)

        # Now load data (triggers band_count_changed -> resize)
        self._eq_widget._load_from_metadata()

        self._position_proxy()
        self._subscribe_eq_events()

    # ------------------------------------------------------------------
    # Dynamic dimensions
    # ------------------------------------------------------------------

    def _calculate_dimensions(self):
        """Calculate dimensions including the current EQ control height."""
        super()._calculate_dimensions()
        self._height += self._eq_control_height

    def _on_band_count_changed(self, count: int):
        """Resize the node when band count changes."""
        new_height = _control_height_for_bands(count)
        if new_height == self._eq_control_height:
            return

        self.prepareGeometryChange()
        self._eq_control_height = new_height
        self._calculate_dimensions()
        self._position_proxy()

        # Update connections since port positions changed
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
        proxy_y = rect.bottom() - self._eq_control_height - 4
        self._proxy.setPos(proxy_x, proxy_y)

    # ------------------------------------------------------------------
    # Event subscriptions
    # ------------------------------------------------------------------

    def _subscribe_eq_events(self):
        if not self.facade or not self.facade.event_bus:
            return
        self.facade.event_bus.subscribe("BlockUpdated", self._on_eq_block_updated)

    def _unsubscribe_eq_events(self):
        if not self.facade or not self.facade.event_bus:
            return
        try:
            self.facade.event_bus.unsubscribe("BlockUpdated", self._on_eq_block_updated)
        except Exception as e:
            Log.debug(f"EQBandsBlockItem: Error unsubscribing: {e}")

    def _on_eq_block_updated(self, event):
        if not self._is_valid():
            return
        try:
            updated_id = event.data.get("id") if hasattr(event, "data") and event.data else None
            if updated_id == self.block.id:
                QTimer.singleShot(100, self._safe_refresh_eq)
        except Exception:
            pass

    def _safe_refresh_eq(self):
        try:
            if self._is_valid() and self._eq_widget:
                self._eq_widget._load_from_metadata()
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def itemChange(self, change, value):
        from PyQt6.QtWidgets import QGraphicsItem

        if change == QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged:
            if value is None:
                self._unsubscribe_eq_events()
        return super().itemChange(change, value)

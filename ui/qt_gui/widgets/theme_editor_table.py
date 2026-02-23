"""
Theme Editor Table

Table-based widget for editing all design system theme values.
Each row = one setting. Columns = Name | Value | Preview (for colors).
Supports color (swatch + QColorDialog), int, str, and bool types.
"""
from PyQt6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QWidget, QVBoxLayout,
    QColorDialog, QSpinBox, QLineEdit, QCheckBox, QComboBox,
    QStyledItemDelegate, QStyleOptionViewItem, QApplication,
    QHeaderView,
)
from PyQt6.QtCore import Qt, pyqtSignal, QModelIndex
from PyQt6.QtGui import QColor, QPainter

from ui.qt_gui.design_system import Colors, border_radius
from ui.qt_gui.theme_registry import ThemeRegistry


# Human-readable labels for theme fields
FIELD_LABELS = {
    "bg_dark": "Background Dark",
    "bg_medium": "Background Medium",
    "bg_light": "Background Light",
    "border": "Border",
    "hover": "Hover",
    "selected": "Selected",
    "text_primary": "Text Primary",
    "text_secondary": "Text Secondary",
    "text_disabled": "Text Disabled",
    "accent_blue": "Accent Blue",
    "accent_green": "Accent Green",
    "accent_red": "Accent Red",
    "accent_yellow": "Accent Yellow",
    "block_load": "Block Load",
    "block_analyze": "Block Analyze",
    "block_transform": "Block Transform",
    "block_export": "Block Export",
    "block_editor": "Block Editor",
    "block_visualize": "Block Visualize",
    "block_utility": "Block Utility",
    "connection_normal": "Connection Normal",
    "connection_hover": "Connection Hover",
    "connection_selected": "Connection Selected",
    "port_input": "Port Input",
    "port_output": "Port Output",
    "port_audio": "Port Audio",
    "port_event": "Port Event",
    "port_manipulator": "Port Manipulator",
    "port_generic": "Port Generic",
    "accent_orange": "Accent Orange",
    "accent_purple": "Accent Purple",
    "status_success": "Status Success",
    "status_warning": "Status Warning",
    "status_error": "Status Error",
    "status_info": "Status Info",
    "status_inactive": "Status Inactive",
    "danger_bg": "Danger Background",
    "danger_fg": "Danger Foreground",
    "overlay_subtle": "Overlay Subtle",
    "overlay_feint": "Overlay Feint",
    "overlay_dim": "Overlay Dim",
    "overlay_very_subtle": "Overlay Very Subtle",
    "text_on_light": "Text On Light",
    "text_on_dark": "Text On Dark",
    "filter_shelf": "Filter Shelf",
    "filter_peak": "Filter Peak",
    "grid_line": "Grid Line",
    "ui_font_family": "UI Font Family",
    "ui_font_size": "UI Font Size",
    "sharp_corners": "Sharp Corners",
}


def _get_colors_attr_name(theme_attr: str) -> str:
    """Map theme attribute name to Colors class attribute."""
    mapping = {
        "bg_dark": "BG_DARK", "bg_medium": "BG_MEDIUM", "bg_light": "BG_LIGHT",
        "border": "BORDER", "hover": "HOVER", "selected": "SELECTED",
        "text_primary": "TEXT_PRIMARY", "text_secondary": "TEXT_SECONDARY",
        "text_disabled": "TEXT_DISABLED",
        "accent_blue": "ACCENT_BLUE", "accent_green": "ACCENT_GREEN",
        "accent_red": "ACCENT_RED", "accent_yellow": "ACCENT_YELLOW",
        "block_load": "BLOCK_LOAD", "block_analyze": "BLOCK_ANALYZE",
        "block_transform": "BLOCK_TRANSFORM", "block_export": "BLOCK_EXPORT",
        "block_editor": "BLOCK_EDITOR", "block_visualize": "BLOCK_VISUALIZE",
        "block_utility": "BLOCK_UTILITY",
        "connection_normal": "CONNECTION_NORMAL", "connection_hover": "CONNECTION_HOVER",
        "connection_selected": "CONNECTION_SELECTED",
        "port_input": "PORT_INPUT", "port_output": "PORT_OUTPUT",
        "port_audio": "PORT_AUDIO", "port_event": "PORT_EVENT",
        "port_manipulator": "PORT_MANIPULATOR", "port_generic": "PORT_GENERIC",
        "overlay_subtle": "OVERLAY_SUBTLE", "overlay_feint": "OVERLAY_FEINT",
        "overlay_dim": "OVERLAY_DIM", "overlay_very_subtle": "OVERLAY_VERY_SUBTLE",
        "text_on_light": "TEXT_ON_LIGHT", "text_on_dark": "TEXT_ON_DARK",
        "filter_shelf": "FILTER_SHELF", "filter_peak": "FILTER_PEAK",
        "grid_line": "GRID_LINE",
    }
    return mapping.get(theme_attr, "BG_DARK")


class ColorPreviewDelegate(QStyledItemDelegate):
    """Delegate for Preview column: paints color swatch."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        super().paint(painter, option, index)
        hex_val = index.data(Qt.ItemDataRole.EditRole) or index.data(Qt.ItemDataRole.DisplayRole)
        if hex_val:
            try:
                color = QColor(hex_val)
                if color.isValid():
                    rect = option.rect.adjusted(2, 2, -2, -2)
                    swatch_w = min(24, rect.width(), rect.height())
                    swatch_rect = rect.adjusted(0, (rect.height() - swatch_w) // 2, 0, 0)
                    swatch_rect.setWidth(swatch_w)
                    swatch_rect.setHeight(swatch_w)
                    painter.fillRect(swatch_rect, color)
                    painter.setPen(QColor(Colors.BORDER))
                    painter.drawRect(swatch_rect)
            except Exception:
                pass


class ThemeEditorTable(QTableWidget):
    """
    Table widget for editing theme values.
    Rows = settings, columns = Name | Value | Preview.
    Emits value_changed when user edits.
    """

    value_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Setting", "Value", "Preview"])
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(0, 200)
        self.setColumnWidth(2, 60)
        self._row_meta = []  # [(key, type), ...]
        self._populate_rows()
        self.setItemDelegateForColumn(2, ColorPreviewDelegate(self))
        self.setMinimumHeight(400)
        self._block_signals = False
        self.itemChanged.connect(self._on_item_changed)
        self.cellDoubleClicked.connect(self._on_cell_double_clicked)

    def _populate_rows(self):
        """Populate rows from COLOR_FIELDS + font/size/sharp_corners."""
        self.setRowCount(0)
        self._row_meta.clear()

        for field in ThemeRegistry.COLOR_FIELDS:
            self._add_color_row(field)

        self._add_int_row("ui_font_size", "UI Font Size", 0, 48, 13)
        self._add_bool_row("sharp_corners", "Sharp Corners")

    def _add_color_row(self, key: str):
        row = self.rowCount()
        self.insertRow(row)
        label = FIELD_LABELS.get(key, key.replace("_", " ").title())
        name_item = QTableWidgetItem(label)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, 0, name_item)

        colors_attr = _get_colors_attr_name(key)
        hex_val = getattr(Colors, colors_attr, None)
        if hex_val is not None:
            hex_val = hex_val.name()
        else:
            hex_val = "#1c1c20"
        value_item = QTableWidgetItem(hex_val)
        self.setItem(row, 1, value_item)

        preview_item = QTableWidgetItem(hex_val)
        preview_item.setFlags(preview_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        preview_item.setData(Qt.ItemDataRole.EditRole, hex_val)
        self.setItem(row, 2, preview_item)
        self._row_meta.append((key, "color"))

    def _add_int_row(self, key: str, label: str, min_val: int, max_val: int, default: int):
        row = self.rowCount()
        self.insertRow(row)
        name_item = QTableWidgetItem(label)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, 0, name_item)
        value_item = QTableWidgetItem(str(default))
        self.setItem(row, 1, value_item)
        preview_item = QTableWidgetItem("")
        preview_item.setFlags(preview_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, 2, preview_item)
        self._row_meta.append((key, "int"))

    def _add_bool_row(self, key: str, label: str):
        row = self.rowCount()
        self.insertRow(row)
        name_item = QTableWidgetItem(label)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, 0, name_item)
        value_item = QTableWidgetItem("true")
        self.setItem(row, 1, value_item)
        preview_item = QTableWidgetItem("")
        preview_item.setFlags(preview_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, 2, preview_item)
        self._row_meta.append((key, "bool"))

    def _on_cell_double_clicked(self, row: int, col: int):
        if row >= len(self._row_meta):
            return
        key, cell_type = self._row_meta[row]
        if cell_type == "color":
            value_item = self.item(row, 1)
            hex_val = value_item.text() if value_item else "#1c1c20"
            color = QColorDialog.getColor(QColor(hex_val), self, f"Choose color for {key}")
            if color.isValid():
                self._block_signals = True
                try:
                    if value_item:
                        value_item.setText(color.name())
                    self._update_preview_cell(row)
                    self.value_changed.emit()
                finally:
                    self._block_signals = False

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._block_signals:
            return
        col = item.column()
        if col == 1:
            self._update_preview_cell(item.row())
            self.value_changed.emit()

    def _update_preview_cell(self, row: int):
        if row >= len(self._row_meta):
            return
        key, cell_type = self._row_meta[row]
        if cell_type == "color":
            value_item = self.item(row, 1)
            if value_item:
                hex_val = value_item.text().strip()
                if not hex_val.startswith("#"):
                    hex_val = "#" + hex_val
                preview_item = self.item(row, 2)
                if preview_item:
                    preview_item.setData(Qt.ItemDataRole.EditRole, hex_val)
                    preview_item.setData(Qt.ItemDataRole.DisplayRole, hex_val)
                self.viewport().update(self.visualRect(self.model().index(row, 2)))

    def get_values(self) -> dict:
        """Return current table state as dict for Colors.apply_theme_from_dict()."""
        result = {}
        for row in range(self.rowCount()):
            if row >= len(self._row_meta):
                continue
            key, cell_type = self._row_meta[row]
            value_item = self.item(row, 1)
            if not value_item:
                continue
            raw = value_item.text().strip()
            if cell_type == "color":
                result[key] = raw if raw.startswith("#") else f"#{raw}"
            elif cell_type == "int":
                try:
                    result[key] = int(raw)
                except ValueError:
                    result[key] = 13
            elif cell_type == "bool":
                result[key] = raw.lower() in ("true", "1", "yes", "on")
        return result

    def set_values(self, values: dict):
        """Load values from theme or preset into table."""
        self._block_signals = True
        try:
            for row in range(self.rowCount()):
                if row >= len(self._row_meta):
                    continue
                key, cell_type = self._row_meta[row]
                if key not in values:
                    continue
                val = values[key]
                value_item = self.item(row, 1)
                if not value_item:
                    continue
                if cell_type == "color":
                    if isinstance(val, QColor):
                        value_item.setText(val.name())
                    else:
                        value_item.setText(str(val))
                elif cell_type == "int":
                    value_item.setText(str(int(val)))
                elif cell_type == "bool":
                    value_item.setText("true" if val else "false")
                self._update_preview_cell(row)
        finally:
            self._block_signals = False

    def set_colors_from_theme(self, theme):
        """Load color values from a Theme object."""
        vals = ThemeRegistry.theme_to_dict(theme) if theme else {}
        self.set_values(vals)

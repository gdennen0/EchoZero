"""
Ordered Table View - Reusable Table View Component

A standardized, reusable table view component for managing ordered lists with:
- +/- buttons for row reordering (via callback to consumer)
- Empty row handling
- CRUD operations
- Automatic order_index management

This component can be reused across the application for any ordered list UI.
"""

from typing import Optional, Callable, Any, Dict, List
from PyQt6.QtWidgets import (
    QTableView, QAbstractItemView, QHeaderView, QWidget, QVBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt, QModelIndex, pyqtSignal
from PyQt6.QtGui import (
    QStandardItemModel, QStandardItem, QMouseEvent
)

from ui.qt_gui.design_system import Colors, border_radius
from src.utils.message import Log


class OrderedTableModel(QStandardItemModel):
    """
    Base model for ordered table views.

    Stores row IDs in column 0's UserRole data for tracking.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def _is_empty_row_id(self, row_id: Any) -> bool:
        """Check if a row ID indicates an empty row"""
        if isinstance(row_id, str):
            return row_id.startswith("temp_") or row_id == "" or row_id == "empty"
        return False

    def get_row_id(self, row: int) -> Optional[Any]:
        """Get the ID stored in the first column of a row"""
        item = self.item(row, 0)
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None

    def find_row_by_id(self, row_id: Any) -> int:
        """Find row index by ID"""
        for row in range(self.rowCount()):
            if self.get_row_id(row) == row_id:
                return row
        return -1


class OrderedTableView(QTableView):
    """
    Reusable table view component for ordered lists.

    Features:
    - +/- buttons for row reordering (delegates to move_handler callback)
    - Empty row protection
    - Selection handling

    The +/- buttons call the move_handler with (row_id, direction) where
    direction is "up" or "down". The consumer is responsible for updating
    the underlying data and refreshing the UI.

    Usage:
        view = OrderedTableView()
        view.set_model(OrderedTableModel())
        view.set_move_handler(lambda row_id, direction: ...)  # Required for reordering
        view.set_empty_row_handler(lambda row: ...)  # Optional
    """

    # Signals
    empty_row_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: Optional[OrderedTableModel] = None
        self._empty_row_handler: Optional[Callable[[int], None]] = None
        self._move_handler: Optional[Callable[[str, str], None]] = None  # (row_id, "up"|"down")
        self._move_buttons_column = -1

        # Configure view
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)

        # Drag-and-drop disabled -- widget cells (combo boxes, buttons) don't
        # survive Qt's serialise-remove-insert cycle for InternalMove.
        self.setDragEnabled(False)
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)

    def set_model(self, model: OrderedTableModel):
        """Set the model"""
        self._model = model
        super().setModel(model)

    def set_move_buttons_column(self, column: int):
        """Set which column should contain the +/- move buttons"""
        self._move_buttons_column = column

    def set_empty_row_handler(self, handler: Callable[[int], None]):
        """Set handler for empty row clicks"""
        self._empty_row_handler = handler

    def set_move_handler(self, handler: Callable[[str, str], None]):
        """Set handler called when +/- buttons are clicked.

        handler(row_id, direction)  where direction is "up" or "down".
        The handler should update the underlying data and refresh the UI.
        """
        self._move_handler = handler

    # -- kept for backward compat but no longer wires into model --
    def set_order_update_handler(self, handler: Callable[[int, int], None]):
        """Legacy -- kept for API compat. Prefer set_move_handler."""
        pass

    def set_refresh_handler(self, handler: Callable[[], None]):
        """Legacy -- kept for API compat. Move handler should refresh."""
        pass

    # ---- move buttons ------------------------------------------------

    def add_move_buttons_to_row(self, row: int):
        """Add +/- buttons to a specific row"""
        if not self._model or self._move_buttons_column < 0:
            return

        if row < 0 or row >= self._model.rowCount():
            return

        row_id = self._model.get_row_id(row)
        if not row_id or self._model._is_empty_row_id(row_id):
            return

        # Remove existing widget
        existing_index = self._model.index(row, self._move_buttons_column)
        existing_widget = self.indexWidget(existing_index)
        if existing_widget:
            existing_widget.deleteLater()

        button_container = QWidget()
        button_container.setProperty("row_id", row_id)
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)

        button_style = f"""
            QPushButton {{
                background: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                font-size: 10px;
                font-weight: bold;
                border-radius: {border_radius(2)};
            }}
            QPushButton:hover {{
                background: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QPushButton:pressed {{
                background: {Colors.HOVER.name()};
            }}
        """

        up_btn = QPushButton("+")
        up_btn.setStyleSheet(button_style)
        up_btn.clicked.connect(lambda _checked, rid=row_id: self._on_move_up(rid))
        button_layout.addWidget(up_btn, 1)

        down_btn = QPushButton("\u2212")  # minus sign
        down_btn.setStyleSheet(button_style)
        down_btn.clicked.connect(lambda _checked, rid=row_id: self._on_move_down(rid))
        button_layout.addWidget(down_btn, 1)

        index = self._model.index(row, self._move_buttons_column)
        self.setIndexWidget(index, button_container)

    def _on_move_up(self, row_id: str):
        """Delegate move-up to the consumer via callback"""
        if not self._model:
            return
        row = self._model.find_row_by_id(row_id)
        if row <= 0:
            return
        target_id = self._model.get_row_id(row - 1)
        if not target_id or self._model._is_empty_row_id(target_id):
            return
        if self._move_handler:
            self._move_handler(row_id, "up")

    def _on_move_down(self, row_id: str):
        """Delegate move-down to the consumer via callback"""
        if not self._model:
            return
        row = self._model.find_row_by_id(row_id)
        if row < 0 or row >= self._model.rowCount() - 1:
            return
        target_id = self._model.get_row_id(row + 1)
        if not target_id or self._model._is_empty_row_id(target_id):
            return
        if self._move_handler:
            self._move_handler(row_id, "down")

    def refresh_all_move_buttons(self):
        """Refresh move buttons for all rows"""
        if not self._model or self._move_buttons_column < 0:
            return
        for row in range(self._model.rowCount()):
            idx = self._model.index(row, self._move_buttons_column)
            w = self.indexWidget(idx)
            if w:
                w.deleteLater()
        for row in range(self._model.rowCount()):
            self.add_move_buttons_to_row(row)

    # ---- mouse / empty row handling ----------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press - detect empty row clicks"""
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                row = index.row()
                if self._model:
                    row_id = self._model.get_row_id(row)
                    if row_id and self._model._is_empty_row_id(row_id):
                        if self._empty_row_handler:
                            self._empty_row_handler(row)
                        self.empty_row_clicked.emit(row)
                        event.accept()
                        return

        super().mousePressEvent(event)

    # ---- empty row helpers -------------------------------------------

    def add_empty_row(self, row: int, columns: int = 1, empty_id: str = "empty"):
        """Add an empty row at the specified position."""
        if not self._model:
            return

        row_items = []
        for col in range(columns):
            item = QStandardItem("")
            flags = Qt.ItemFlag.ItemIsEnabled
            item.setFlags(flags)
            if col == 0:
                item.setData(empty_id, Qt.ItemDataRole.UserRole)
            row_items.append(item)

        if row >= self._model.rowCount():
            self._model.appendRow(row_items)
        else:
            self._model.insertRow(row, row_items)

    def remove_empty_rows(self):
        """Remove all empty rows from the table"""
        if not self._model:
            return
        rows_to_remove = []
        for row in range(self._model.rowCount() - 1, -1, -1):
            row_id = self._model.get_row_id(row)
            if row_id and self._model._is_empty_row_id(row_id):
                rows_to_remove.append(row)
        for row in rows_to_remove:
            self._model.removeRow(row)

    def ensure_empty_row_at_end(self, columns: int = 1):
        """Ensure there's exactly one empty row at the end"""
        if not self._model:
            return
        self.remove_empty_rows()
        self.add_empty_row(self._model.rowCount(), columns)

    def is_empty_row(self, row: int) -> bool:
        """Check if a row is an empty row"""
        if not self._model:
            return False
        row_id = self._model.get_row_id(row)
        return self._model._is_empty_row_id(row_id) if row_id else False

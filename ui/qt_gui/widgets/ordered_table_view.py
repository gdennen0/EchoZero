"""
Ordered Table View - Reusable Table View Component

A standardized, reusable table view component for managing ordered lists with:
- +/- buttons for row reordering
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
    
    Handles automatic order_index updates when rows are reordered.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_moving = False
        self._on_order_changed: Optional[Callable[[int, int], None]] = None  # Callback: (row, new_order_index)
    
    def set_order_changed_callback(self, callback: Callable[[int, int], None]):
        """Set callback to be called when order changes"""
        self._on_order_changed = callback
    
    def moveRow(self, sourceParent: QModelIndex, sourceRow: int, destinationParent: QModelIndex, destinationChild: int) -> bool:
        """
        Override moveRow to validate and handle empty rows, then delegate to parent implementation.
        
        Simplified approach: let QStandardItemModel.handle the actual move logic.
        """
        if self._is_moving:
            return False
        
        # Validate indices
        if sourceRow < 0 or sourceRow >= self.rowCount():
            return False
        
        if destinationChild < 0 or destinationChild > self.rowCount():
            return False
        
        # Can't move to the same position
        if sourceRow == destinationChild:
            return False
        
        # Check if source row is valid (not empty row)
        source_id_item = self.item(sourceRow, 0)
        if not source_id_item:
            return False
        
        source_id = source_id_item.data(Qt.ItemDataRole.UserRole)
        if not source_id or self._is_empty_row_id(source_id):
            # Empty row - don't allow moving
            return False
        
        # Check if target position is valid (not empty row)
        # When moving down, check the row that will be before our inserted row
        # When moving up, check the row at destinationChild
        if destinationChild > sourceRow:
            # Moving down: check row at destinationChild (will be after our inserted row)
            check_row = destinationChild
        else:
            # Moving up: check row at destinationChild (where we're inserting)
            check_row = destinationChild
        
        if check_row < self.rowCount():
            target_id_item = self.item(check_row, 0)
            if target_id_item:
                target_id = target_id_item.data(Qt.ItemDataRole.UserRole)
                if target_id and self._is_empty_row_id(target_id):
                    # Empty row - don't allow moving to/next to empty row
                    return False
        
        # QStandardItemModel.moveRow() is virtual and does nothing by default - we must implement it
        # Use beginMoveRows/endMoveRows and manually move items
        self._is_moving = True
        try:
            # Calculate destination for beginMoveRows
            # Qt's beginMoveRows has specific requirements for the destination parameter.
            # When moving down, Qt may require destinationChild + 1 to avoid validation issues.
            # Let's try: when moving down and destinationChild == sourceRow + 1, use destinationChild + 1
            # Otherwise, use destinationChild as-is
            if destinationChild > sourceRow:
                # Moving down: try destinationChild + 1 to see if that passes validation
                # This represents "after the row at destinationChild after removal"
                actual_dest = destinationChild + 1
            else:
                # Moving up: destinationChild is correct
                actual_dest = destinationChild
            
            # Notify views that rows are about to move
            if not self.beginMoveRows(sourceParent, sourceRow, sourceRow, destinationParent, actual_dest):
                return False
            
            # Get all items from source row (this removes the row)
            source_row_items = self.takeRow(sourceRow)
            
            # Calculate insertion index: after takeRow, indices have shifted
            # When moving down (destinationChild > sourceRow):
            #   - We told beginMoveRows destination = destinationChild + 1 (to pass validation)
            #   - But after removing sourceRow, indices shift down by 1
            #   - So we insert at destinationChild (the target position)
            #   - Example: [A(0), B(1), C(2)] → move A(0) to position 1
            #   - After removing A: [B(0), C(1)] → insert at 1 → [B(0), A(1), C(2)] ✓
            # When moving up (destinationChild < sourceRow):
            #   - We told beginMoveRows destination = destinationChild
            #   - After removing sourceRow, destinationChild is still correct
            #   - Example: [A(0), B(1), C(2)] → move C(2) to position 1
            #   - After removing C: [A(0), B(1)] → insert at 1 → [A(0), C(1), B(2)] ✓
            insert_index = destinationChild
            
            # Insert at calculated position
            self.insertRow(insert_index, source_row_items)
            
            # Notify views that move is complete
            self.endMoveRows()
            
            if self._on_order_changed:
                # Final position after move is where we inserted
                final_pos = insert_index
                self._on_order_changed(final_pos, final_pos)
            
            return True
        except Exception as e:
            Log.error(f"OrderedTableModel: Error in moveRow: {e}")
            return False
        finally:
            self._is_moving = False
    
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
    - +/- buttons for row reordering
    - Empty row protection
    - Automatic order_index management
    - Selection handling
    
    Usage:
        view = OrderedTableView()
        view.set_model(OrderedTableModel())
        view.set_empty_row_handler(lambda row: ...)  # Optional
        view.set_order_update_handler(lambda row, new_index: ...)  # Required
    """
    
    # Signals
    empty_row_clicked = pyqtSignal(int)  # Emitted when empty row is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: Optional[OrderedTableModel] = None
        self._empty_row_indicator = "empty"  # ID used for empty rows
        self._empty_row_handler: Optional[Callable[[int], None]] = None
        self._order_update_handler: Optional[Callable[[int, int], None]] = None
        self._move_buttons_column = -1  # Column index for move buttons (set by user)
        
        # Configure view
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        
        # Disable drag-and-drop
        self.setDragEnabled(False)
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
    
    def set_model(self, model: OrderedTableModel):
        """Set the model and configure callbacks"""
        self._model = model
        super().setModel(model)
        
        # Set up order change callback
        if self._order_update_handler:
            model.set_order_changed_callback(self._order_update_handler)
    
    def set_move_buttons_column(self, column: int):
        """Set which column should contain the +/- move buttons"""
        self._move_buttons_column = column
    
    def set_empty_row_handler(self, handler: Callable[[int], None]):
        """Set handler for empty row clicks"""
        self._empty_row_handler = handler
    
    def set_order_update_handler(self, handler: Callable[[int, int], None]):
        """Set handler for order updates (required for reordering to work)"""
        self._order_update_handler = handler
        if self._model:
            self._model.set_order_changed_callback(handler)
    
    def set_refresh_handler(self, handler: Callable[[], None]):
        """Set handler to refresh entire UI after row moves (for widget repositioning)"""
        self._refresh_handler = handler
    
    def add_move_buttons_to_row(self, row: int):
        """Add +/- buttons to a specific row"""
        if not self._model or self._move_buttons_column < 0:
            return
        
        if row < 0 or row >= self._model.rowCount():
            return
        
        # Check if this is an empty row
        row_id = self._model.get_row_id(row)
        if self._model._is_empty_row_id(row_id):
            return
        
        # Remove existing widget if any (to prevent duplicates)
        existing_index = self._model.index(row, self._move_buttons_column)
        existing_widget = self.indexWidget(existing_index)
        if existing_widget:
            existing_widget.deleteLater()
        
        # Create container widget for buttons
        button_container = QWidget()
        button_container.setProperty("row_id", row_id)  # Store row ID for dynamic lookup
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)  # No padding - expand to edges
        button_layout.setSpacing(0)  # No spacing between buttons
        
        # Shared button style
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
        
        # Up button (+) - takes first half
        # Use dynamic row lookup instead of capturing row index
        up_btn = QPushButton("+")
        up_btn.setStyleSheet(button_style)
        up_btn.clicked.connect(lambda checked, rid=row_id: self._move_row_up_by_id(rid))
        button_layout.addWidget(up_btn, 1)  # Stretch factor 1 for equal distribution
        
        # Down button (-) - takes second half
        # Use dynamic row lookup instead of capturing row index
        down_btn = QPushButton("−")  # Use minus sign (U+2212) for better visual
        down_btn.setStyleSheet(button_style)
        down_btn.clicked.connect(lambda checked, rid=row_id: self._move_row_down_by_id(rid))
        button_layout.addWidget(down_btn, 1)  # Stretch factor 1 for equal distribution
        
        # Set widget in the move buttons column
        index = self._model.index(row, self._move_buttons_column)
        self.setIndexWidget(index, button_container)
    
    def _move_row_up_by_id(self, row_id: Any):
        """Move a row up by one position using row ID for dynamic lookup"""
        if not self._model:
            return
        
        # Find current row by ID
        row = self._model.find_row_by_id(row_id)
        if row < 0:
            Log.debug(f"OrderedTableView: Cannot find row with ID {row_id}")
            return
        
        self._move_row_up(row)
    
    def _move_row_down_by_id(self, row_id: Any):
        """Move a row down by one position using row ID for dynamic lookup"""
        if not self._model:
            return
        
        # Find current row by ID
        row = self._model.find_row_by_id(row_id)
        if row < 0:
            Log.debug(f"OrderedTableView: Cannot find row with ID {row_id}")
            return
        
        self._move_row_down(row)
    
    def _move_row_up(self, row: int):
        """Move a row up by one position"""
        if not self._model or row <= 0:
            Log.debug(f"OrderedTableView: Cannot move row {row} up - invalid row or at top")
            return
        
        # Check if source row is valid
        source_id = self._model.get_row_id(row)
        if not source_id or self._model._is_empty_row_id(source_id):
            Log.debug(f"OrderedTableView: Cannot move row {row} up - invalid or empty source row")
            return
        
        # Check if target row is valid (not empty)
        target_id = self._model.get_row_id(row - 1)
        if not target_id or self._model._is_empty_row_id(target_id):
            Log.debug(f"OrderedTableView: Cannot move row {row} up - invalid or empty target row")
            return
        
        # Move row up - use beginMoveRows/endMoveRows for proper model notification
        source_parent = QModelIndex()
        dest_parent = QModelIndex()
        Log.debug(f"OrderedTableView: Moving row {row} up to {row - 1}")
        
        # Call moveRow which will trigger model signals
        result = self._model.moveRow(source_parent, row, dest_parent, row - 1)
        if result:
            Log.debug(f"OrderedTableView: Move successful, updating UI")
            # After move, the row that was at 'row' is now at 'row - 1'
            # Update order numbers for all rows
            self._update_order_numbers()
            # Always call refresh handler to rebuild all widgets in correct positions
            # This is necessary because setIndexWidget widgets don't move automatically
            if self._refresh_handler:
                Log.debug(f"OrderedTableView: Calling refresh handler to rebuild widgets")
                self._refresh_handler()
            else:
                # Fallback: refresh all move buttons
                self.refresh_all_move_buttons()
                # Force view update
                self.viewport().update()
                # Emit dataChanged to ensure view refreshes
                top_left = self._model.index(0, 0)
                bottom_right = self._model.index(self._model.rowCount() - 1, self._model.columnCount() - 1)
                self._model.dataChanged.emit(top_left, bottom_right)
        else:
            Log.debug(f"OrderedTableView: Move failed - moveRow returned False")
    
    def _move_row_down(self, row: int):
        """Move a row down by one position"""
        if not self._model or row < 0 or row >= self._model.rowCount() - 1:
            Log.debug(f"OrderedTableView: Cannot move row {row} down - invalid row or at bottom")
            return
        
        # Check if source row is valid
        source_id = self._model.get_row_id(row)
        if not source_id or self._model._is_empty_row_id(source_id):
            Log.debug(f"OrderedTableView: Cannot move row {row} down - invalid or empty source row")
            return
        
        # Check if target row is valid (not empty)
        target_id = self._model.get_row_id(row + 1)
        if not target_id or self._model._is_empty_row_id(target_id):
            Log.debug(f"OrderedTableView: Cannot move row {row} down - invalid or empty target row")
            return
        
        # Move row down - use beginMoveRows/endMoveRows for proper model notification
        source_parent = QModelIndex()
        dest_parent = QModelIndex()
        Log.debug(f"OrderedTableView: Moving row {row} down to {row + 1}")
        
        # Call moveRow which will trigger model signals
        result = self._model.moveRow(source_parent, row, dest_parent, row + 1)
        if result:
            Log.debug(f"OrderedTableView: Move successful, updating UI")
            # After move, the row that was at 'row' is now at 'row + 1'
            # Update order numbers for all rows
            self._update_order_numbers()
            # Always call refresh handler to rebuild all widgets in correct positions
            # This is necessary because setIndexWidget widgets don't move automatically
            if self._refresh_handler:
                Log.debug(f"OrderedTableView: Calling refresh handler to rebuild widgets")
                self._refresh_handler()
            else:
                # Fallback: refresh all move buttons
                self.refresh_all_move_buttons()
                # Force view update
                self.viewport().update()
                # Emit dataChanged to ensure view refreshes
                top_left = self._model.index(0, 0)
                bottom_right = self._model.index(self._model.rowCount() - 1, self._model.columnCount() - 1)
                self._model.dataChanged.emit(top_left, bottom_right)
        else:
            Log.debug(f"OrderedTableView: Move failed - moveRow returned False")
    
    def _update_order_numbers(self):
        """Update the order number (#) column after rows are moved"""
        if not self._model:
            return
        
        for row in range(self._model.rowCount()):
            order_item = self._model.item(row, 0)
            if order_item:
                # Skip empty rows
                row_id = order_item.data(Qt.ItemDataRole.UserRole)
                if self._model._is_empty_row_id(row_id):
                    continue
                # Update order number (1-based)
                order_item.setText(str(row + 1))
    
    def _refresh_move_buttons_for_row(self, row: int):
        """Refresh move buttons for a specific row"""
        if self._move_buttons_column >= 0:
            self.add_move_buttons_to_row(row)
    
    def refresh_all_move_buttons(self):
        """Refresh move buttons for all rows"""
        if not self._model or self._move_buttons_column < 0:
            return
        
        # Clear all existing move button widgets first
        for row in range(self._model.rowCount()):
            index = self._model.index(row, self._move_buttons_column)
            existing_widget = self.indexWidget(index)
            if existing_widget:
                existing_widget.deleteLater()
        
        # Add buttons for all valid rows
        for row in range(self._model.rowCount()):
            self.add_move_buttons_to_row(row)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press - detect empty row clicks"""
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                row = index.row()
                if self._model:
                    row_id = self._model.get_row_id(row)
                    if self._model._is_empty_row_id(row_id):
                        # Empty row clicked
                        if self._empty_row_handler:
                            self._empty_row_handler(row)
                        self.empty_row_clicked.emit(row)
                        event.accept()
                        return
        
        super().mousePressEvent(event)
    
    def add_empty_row(self, row: int, columns: int = 1, empty_id: str = "empty"):
        """
        Add an empty row at the specified position.
        
        Args:
            row: Row index to insert at
            columns: Number of columns
            empty_id: ID to use for empty row identification
        """
        if not self._model:
            return
        
        # Create row items
        row_items = []
        for col in range(columns):
            item = QStandardItem("")
            flags = Qt.ItemFlag.ItemIsEnabled
            item.setFlags(flags)
            if col == 0:
                # Store empty row ID in first column
                item.setData(empty_id, Qt.ItemDataRole.UserRole)
            row_items.append(item)
        
        # Insert row
        if row >= self._model.rowCount():
            self._model.appendRow(row_items)
        else:
            self._model.insertRow(row, row_items)
    
    def remove_empty_rows(self):
        """Remove all empty rows from the table"""
        if not self._model:
            return
        
        rows_to_remove = []
        for row in range(self._model.rowCount() - 1, -1, -1):  # Iterate backwards
            row_id = self._model.get_row_id(row)
            if self._model._is_empty_row_id(row_id):
                rows_to_remove.append(row)
        
        for row in rows_to_remove:
            self._model.removeRow(row)
    
    def ensure_empty_row_at_end(self, columns: int = 1):
        """Ensure there's exactly one empty row at the end"""
        if not self._model:
            return
        
        # Remove all empty rows first
        self.remove_empty_rows()
        
        # Add one at the end
        self.add_empty_row(self._model.rowCount(), columns)
    
    def is_empty_row(self, row: int) -> bool:
        """Check if a row is an empty row"""
        if not self._model:
            return False
        row_id = self._model.get_row_id(row)
        return self._model._is_empty_row_id(row_id) if row_id else False

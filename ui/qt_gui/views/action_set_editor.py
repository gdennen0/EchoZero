"""
Action Set Editor

Clean, minimal UI for creating and editing action sets.
Actions are displayed in order and executed sequentially.
"""
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox,
    QMessageBox, QInputDialog, QHeaderView,
    QAbstractItemView, QSizePolicy, QMenu, QLineEdit,
    QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QModelIndex, QPoint
from PyQt6.QtGui import QWheelEvent, QStandardItem

from ui.qt_gui.widgets.ordered_table_view import OrderedTableView, OrderedTableModel

from src.application.api.application_facade import ApplicationFacade
from src.features.projects.domain import ActionSet, ActionItem
from src.application.commands import (
    AddActionItemCommand,
    UpdateActionItemCommand,
    DeleteActionItemCommand,
)
from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory
from src.utils.message import Log



class NonScrollableComboBox(QComboBox):
    """QComboBox that ignores wheel events to prevent accidental selection changes"""
    
    def wheelEvent(self, event: QWheelEvent):
        """Ignore wheel events to prevent accidental selection changes"""
        event.ignore()


class ActionItemModel(OrderedTableModel):
    """
    Custom model for action items that extends OrderedTableModel.
    
    Handles action-specific logic while leveraging base drag-and-drop functionality.
    """
    
    def __init__(self, editor: 'ActionSetEditor', parent=None):
        super().__init__(parent)
        self.editor = editor
    
    def moveRow(self, sourceParent: QModelIndex, sourceRow: int, destinationParent: QModelIndex, destinationChild: int) -> bool:
        """Override to call _update_order_indices after move"""
        result = super().moveRow(sourceParent, sourceRow, destinationParent, destinationChild)
        if result:
            # Update order indices after move completes (use timer to avoid recursion)
            QTimer.singleShot(0, self._update_order_indices)
        return result
    
    def _update_order_indices(self):
        """Update order_index values based on current row positions"""
        if not self.editor.current_action_set or self.editor._is_refreshing:
            return
        
        # Get all valid actions
        valid_actions = [
            a for a in self.editor.current_action_set.actions 
            if a.action_name and ((a.action_type == "block" and a.block_id) or (a.action_type == "project"))
        ]
        
        # Update order_index based on current model row positions
        # Capture old state BEFORE modifying to ensure proper undo/redo
        updates_needed = []
        for table_row in range(self.rowCount()):
            action_id = self.get_row_id(table_row)
            if not action_id or self._is_empty_row_id(action_id):
                continue
            
            action = self.editor._get_action_by_id(action_id)
            if action and action.order_index != table_row:
                # Capture old state before modifying
                from src.features.projects.domain import ActionItem
                old_action = ActionItem(
                    id=action.id,
                    action_set_id=action.action_set_id,
                    project_id=action.project_id,
                    action_type=action.action_type,
                    block_id=action.block_id,
                    block_name=action.block_name,
                    action_name=action.action_name,
                    action_description=action.action_description,
                    action_args=action.action_args.copy() if action.action_args else {},
                    order_index=action.order_index,  # OLD order_index
                    created_at=action.created_at,
                    modified_at=action.modified_at,
                    metadata=action.metadata.copy() if action.metadata else {}
                )
                
                # Update order_index in action object
                action.order_index = table_row
                
                # Create updated action with new order_index
                updated_action = ActionItem(
                    id=action.id,
                    action_set_id=action.action_set_id,
                    project_id=action.project_id,
                    action_type=action.action_type,
                    block_id=action.block_id,
                    block_name=action.block_name,
                    action_name=action.action_name,
                    action_description=action.action_description,
                    action_args=action.action_args.copy() if action.action_args else {},
                    order_index=table_row,  # NEW order_index
                    created_at=action.created_at,
                    modified_at=action.modified_at,
                    metadata=action.metadata.copy() if action.metadata else {}
                )
                
                updates_needed.append((updated_action, old_action))
        
        # Batch update via commands with explicit old state for proper undo/redo
        if updates_needed:
            if len(updates_needed) > 1:
                self.editor.facade.command_bus.begin_macro(f"Reorder {len(updates_needed)} actions")
            
            for updated_action, old_action in updates_needed:
                # Pass old_action explicitly to ensure undo restores correct order
                cmd = UpdateActionItemCommand(self.editor.facade, updated_action, old_action)
                self.editor.facade.command_bus.execute(cmd)
            
            if len(updates_needed) > 1:
                self.editor.facade.command_bus.end_macro()


class ActionSetEditor(ThemeAwareMixin, QWidget):
    """
    Action Set Editor - clean minimal UI for creating action sets.
    
    Features:
    - Empty list to start
    - + button to add new block actions
    - Actions displayed in order
    - Save/load action sets
    - Edit/remove actions
    """
    
    # Signal emitted when action set changes
    action_set_changed = pyqtSignal()
    
    def __init__(self, facade: ApplicationFacade, parent=None):
        super().__init__(parent)
        self.facade = facade
        self.current_project_id: Optional[str] = None
        self.current_action_set: Optional[ActionSet] = None
        self.actions_by_block: Dict[str, List[Dict[str, Any]]] = {}
        self._is_refreshing = False  # Guard to prevent refresh loops
        
        Log.debug("ActionSetEditor: Initializing ActionSetEditor widget")
        self._setup_ui()
        
        # Subscribe to undo stack changes to refresh UI
        self._subscribe_to_undo_stack()
        
        self._init_theme_aware()
        
        # Try to get project ID and discover actions immediately
        # Note: Parent widget (SetlistView) calls load_project() when project changes
        project_id = self.facade.current_project_id
        if project_id:
            Log.debug(f"ActionSetEditor: Found project_id in facade: {project_id}")
            self.load_project(project_id)
        else:
            Log.debug("ActionSetEditor: No project_id found - parent will call load_project() when ready")
    
    def _setup_ui(self):
        """Setup the UI layout"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        main_layout.setSpacing(Spacing.SM)
        
        # Header with title and add button
        header_layout = QHBoxLayout()
        
        title = QLabel("Action Set")
        title.setFont(Typography.heading_font())
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Add action button (top right, just + sign)
        self.add_action_btn = QPushButton("+")
        self.add_action_btn.setStyleSheet(self._add_button_style())
        self.add_action_btn.setFixedSize(28, 28)
        self.add_action_btn.clicked.connect(self._on_add_action_row)
        header_layout.addWidget(self.add_action_btn)
        
        main_layout.addLayout(header_layout)
        
        # Description
        description = QLabel(
            "Create a sequence of actions to execute. Actions run in order, "
            "waiting for each to complete before proceeding."
        )
        description.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        description.setWordWrap(True)
        main_layout.addWidget(description)
        
        # Actions table with columns: # | Block | Action | Parameters | Move | Edit | Delete
        # Create model first
        self.actions_model = ActionItemModel(self)
        self.actions_model.setColumnCount(7)
        self.actions_model.setHorizontalHeaderLabels(["#", "Block", "Action", "Parameters", "", "", ""])
        
        # Create ordered table view (reusable component)
        self.actions_table = OrderedTableView()
        self.actions_table.set_model(self.actions_model)
        
        # Set move buttons column (column 4)
        self.actions_table.set_move_buttons_column(4)
        
        # Set up order update handler (required for reordering)
        self.actions_table.set_order_update_handler(self._on_order_changed)
        
        # Set up refresh handler to rebuild widgets after row moves
        # This is needed because setIndexWidget widgets don't move automatically
        self.actions_table.set_refresh_handler(self._refresh_actions_list)
        
        # Set up empty row handler
        self.actions_table.set_empty_row_handler(self._on_empty_row_clicked)
        
        # Configure header
        header = self.actions_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Order number
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Block
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Action
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)  # Parameters
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)  # Move buttons
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)  # Edit button
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)  # Delete button
        self.actions_table.setColumnWidth(0, 40)  # Order number
        self.actions_table.setColumnWidth(4, 30)  # Move buttons
        self.actions_table.setColumnWidth(5, 50)  # Edit button
        self.actions_table.setColumnWidth(6, 60)  # Delete button
        
        self.actions_table.verticalHeader().setDefaultSectionSize(36)  # Set default row height
        self.actions_table.setStyleSheet(StyleFactory.table())
        # Ensure proper clipping to prevent combo box dropdowns from extending beyond table bounds
        self.actions_table.setViewportMargins(0, 0, 0, 0)
        
        # Enable context menu for right-click insert
        self.actions_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.actions_table.customContextMenuRequested.connect(self._on_context_menu_requested)
        
        main_layout.addWidget(self.actions_table, 1)  # Stretch
        
        # Action set management toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(6)
        
        # Load set dropdown
        load_label = QLabel("Load:")
        load_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        toolbar_layout.addWidget(load_label)
        
        self.load_set_combo = NonScrollableComboBox()
        self.load_set_combo.setStyleSheet(StyleFactory.combo())
        self.load_set_combo.setMinimumWidth(180)
        self.load_set_combo.setEditable(False)
        self.load_set_combo.setMaxVisibleItems(1000)  # Show all items without scrolling
        self.load_set_combo.currentIndexChanged.connect(self._on_load_set_selected)
        toolbar_layout.addWidget(self.load_set_combo)
        
        toolbar_layout.addSpacing(12)
        
        # Action buttons
        self.save_set_btn = QPushButton("Save Set")
        self.save_set_btn.setStyleSheet(StyleFactory.button())
        self.save_set_btn.clicked.connect(self._on_save_set)
        toolbar_layout.addWidget(self.save_set_btn)
        
        self.new_set_btn = QPushButton("New Set")
        self.new_set_btn.setStyleSheet(StyleFactory.button())
        self.new_set_btn.clicked.connect(self._on_new_set)
        toolbar_layout.addWidget(self.new_set_btn)
        
        toolbar_layout.addStretch()
        
        # Remove action button
        self.remove_action_btn = QPushButton("Remove")
        self.remove_action_btn.setStyleSheet(StyleFactory.button())
        self.remove_action_btn.clicked.connect(self._on_remove_action)
        self.remove_action_btn.setEnabled(False)
        toolbar_layout.addWidget(self.remove_action_btn)
        
        main_layout.addLayout(toolbar_layout)
        
        # Populate load combo
        self._populate_load_combo()
        
        # Connect selection change
        self.actions_table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        
        # Connect double-click on cells (for editing parameters)
        self.actions_table.doubleClicked.connect(self._on_cell_double_clicked)
        
        # Initialize with empty set
        self._new_action_set()
    
    def _on_order_changed(self, row: int, new_index: int):
        """Handle order change after drag-and-drop"""
        if self._is_refreshing or not self.current_action_set:
            return
        
        # Update order_index values for all affected rows
        self.actions_model._update_order_indices()
    
    def _on_empty_row_clicked(self, row: int):
        """Handle empty row click"""
        self._on_add_action_row()
    
    def _new_action_set(self):
        """Create a new empty action set"""
        self.current_action_set = ActionSet(
            id="",
            name="Untitled",
            description="",
            actions=[],
            project_id=self.current_project_id
        )
        # Always refresh to show empty row - it will be populated when load_project() is called
        self._refresh_actions_list()
        self.action_set_changed.emit()
    
    def _refresh_actions_list(self):
        """Refresh the actions table display with inline editing"""
        self._is_refreshing = True
        try:
            # Clear all widgets before removing rows to ensure proper cleanup
            row_count = self.actions_model.rowCount()
            for row in range(row_count):
                for col in range(self.actions_model.columnCount()):
                    index = self.actions_model.index(row, col)
                    widget = self.actions_table.indexWidget(index)
                    if widget:
                        widget.deleteLater()
            
            self.actions_model.removeRows(0, row_count)
        finally:
            self._is_refreshing = False
        
        # Get project_id from facade if not set
        if not self.current_project_id:
            self.current_project_id = self.facade.get_current_project_id()
        
        # Always try to discover actions if we have a project_id and don't have block data yet
        if (not self.actions_by_block or len(self.actions_by_block) == 0) and self.current_project_id:
            result = self.facade.discover_setlist_actions(self.current_project_id)
            if result.success:
                self.actions_by_block = result.data or {}
            else:
                Log.warning(f"ActionSetEditor: Failed to discover actions: {result.message}")
                self.actions_by_block = {}
        
        if not self.current_action_set:
            # Show empty row for adding (even if no project_id yet - it will be populated when load_project() is called)
            self._add_empty_row(0)
            return
        
        # Filter and sort actions by order_index
        valid_actions = [
            a for a in self.current_action_set.actions 
            if a.action_name and ((a.action_type == "block" and a.block_id) or (a.action_type == "project"))
        ]
        # Sort by order_index to ensure correct order
        valid_actions.sort(key=lambda x: x.order_index)
        
        if not self.actions_by_block:
            # Can't populate combos without block data - show empty row
            self._add_empty_row(0)
            return
        
        # Add rows for existing actions + 1 empty row at bottom
        self._is_refreshing = True
        try:
            for idx, action in enumerate(valid_actions):
                # Store action ID in order item for reliable tracking
                action_id = action.id if action.id else f"temp_{idx}"
                # Order number column
                order_item = QStandardItem(str(idx + 1))
                # Enable drag for row reordering
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsDragEnabled
                order_item.setFlags(flags)
                order_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                order_item.setForeground(self._get_text_color(Colors.TEXT_SECONDARY))
                order_item.setData(action_id, Qt.ItemDataRole.UserRole)  # Store ID for tracking
                
                # Create row with empty items for other columns (will be filled with widgets)
                row_items = [order_item]
                for col in range(1, 7):  # 7 columns now (added move buttons column)
                    item = QStandardItem()
                    item.setFlags(flags)
                    row_items.append(item)
                
                self.actions_model.appendRow(row_items)
                
                # Add move buttons to this row
                self.actions_table.add_move_buttons_to_row(idx)
                
                # Block column - combo box for selection
                block_combo = NonScrollableComboBox()
                block_combo.setStyleSheet(StyleFactory.combo())
                block_combo.setFixedHeight(36)  # Match table row height
                block_combo.setMaxVisibleItems(1000)  # Show all items without scrolling
                self._populate_block_combo(block_combo)
                # Select current block (or "project" for project actions)
                block_id_to_match = "project" if action.action_type == "project" else action.block_id
                for i in range(block_combo.count()):
                    if block_combo.itemData(i) == block_id_to_match:
                        block_combo.setCurrentIndex(i)
                        break
                # Use action_id instead of row index for reliable tracking
                block_combo.currentIndexChanged.connect(
                    lambda checked, action_id=action_id: self._on_block_changed_by_id(action_id)
                )
                block_combo.setProperty("action_id", action_id)  # Store ID in widget
                self.actions_table.setIndexWidget(self.actions_model.index(idx, 1), block_combo)
                
                # Action column - combo box for selection
                action_combo = NonScrollableComboBox()
                action_combo.setStyleSheet(StyleFactory.combo())
                action_combo.setFixedHeight(36)  # Match table row height
                action_combo.setMaxVisibleItems(1000)  # Show all items without scrolling
                # Use "project" for project actions, block_id for block actions
                block_id_for_combo = "project" if action.action_type == "project" else action.block_id
                self._populate_action_combo(action_combo, block_id_for_combo)
                # Select current action
                for i in range(action_combo.count()):
                    act_data = action_combo.itemData(i)
                    if act_data and act_data.get("name") == action.action_name:
                        action_combo.setCurrentIndex(i)
                        break
                # Use action_id instead of row index for reliable tracking
                action_combo.currentIndexChanged.connect(
                    lambda checked, action_id=action_id: self._on_action_changed_by_id(action_id)
                )
                action_combo.setProperty("action_id", action_id)  # Store ID in widget
                self.actions_table.setIndexWidget(self.actions_model.index(idx, 2), action_combo)
                
                # Parameters column - show params (double-click to edit)
                params_text = self._format_params(action.action_args)
                params_item = QStandardItem(params_text)
                params_item.setToolTip(f"Double-click to edit parameters\n\n{str(action.action_args)}")
                params_item.setForeground(self._get_text_color(Colors.ACCENT_BLUE if action.action_args else Colors.TEXT_SECONDARY))
                self.actions_model.setItem(idx, 3, params_item)
                
                # Edit button column (now column 5)
                edit_btn = QPushButton("Edit")
                edit_btn.setStyleSheet(StyleFactory.button("small"))
                edit_btn.clicked.connect(lambda checked, action_id=action_id: self._on_edit_params_by_id(action_id))
                edit_btn.setProperty("action_id", action_id)  # Store ID in widget
                self.actions_table.setIndexWidget(self.actions_model.index(idx, 5), edit_btn)
                
                # Delete button column (now column 6)
                delete_btn = QPushButton("Delete")
                delete_btn.setStyleSheet(StyleFactory.button("small"))
                delete_btn.clicked.connect(lambda checked, action_id=action_id: self._on_delete_action_by_id(action_id))
                delete_btn.setProperty("action_id", action_id)  # Store ID in widget
                self.actions_table.setIndexWidget(self.actions_model.index(idx, 6), delete_btn)
            
            # Always add empty row at bottom for adding new actions
            self._add_empty_row(len(valid_actions))
        finally:
            self._is_refreshing = False
            # Ensure exactly one empty row with the styled label exists at end
            QTimer.singleShot(0, self._ensure_styled_empty_row)
    
    def _add_empty_row(self, row: int):
        """Add an empty row with 'Create new action...' label"""
        
        # Use OrderedTableView's built-in empty row handling
        self.actions_table.add_empty_row(row, 7, "empty")  # 7 columns now (added move buttons column)
        
        # Get the actual row index (may have been adjusted)
        actual_row = self.actions_model.rowCount() - 1 if row >= self.actions_model.rowCount() else row
        
        # Create a clickable label spanning the main columns
        create_label = QPushButton("+ Create new action...")
        create_label.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {Colors.TEXT_SECONDARY.name()};
                text-align: left;
                padding: 4px 8px;
                font-style: italic;
            }}
            QPushButton:hover {{
                color: {Colors.ACCENT_BLUE.name()};
                text-decoration: underline;
            }}
        """)
        create_label.setCursor(Qt.CursorShape.PointingHandCursor)
        create_label.clicked.connect(self._on_add_action_row)
        # Make button expand to fill available width
        create_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        create_label.setMinimumWidth(0)
        # Set widget on column 1, it will visually span due to styling
        self.actions_table.setIndexWidget(self.actions_model.index(actual_row, 1), create_label)
    
    def _ensure_styled_empty_row(self):
        """Ensure exactly one empty row with the styled '+ Create new action...' label exists at end."""
        if not self.actions_model:
            return
        
        # Remove all existing empty rows
        self.actions_table.remove_empty_rows()
        
        # Add a styled empty row at the end
        self._add_empty_row(self.actions_model.rowCount())
    
    def _populate_block_combo(self, combo: QComboBox, include_placeholder: bool = False):
        """Populate block combo box"""
        if include_placeholder:
            combo.clear()
            combo.addItem("-- Select Block --", None)
        elif combo.count() == 0 or combo.itemData(0) is None:
            # Clear placeholder if it exists
            combo.clear()
        
        # Check if actions_by_block exists and has items
        if not self.actions_by_block or len(self.actions_by_block) == 0:
            return
        
        items_added = 0
        # Add project actions first
        if "project" in self.actions_by_block:
            project_data = self.actions_by_block["project"]
            if isinstance(project_data, dict):
                display_name = "Project (Project Actions)"
                combo.addItem(display_name, "project")
                items_added += 1
        
        # Then add all blocks
        for block_id, block_data in self.actions_by_block.items():
            if block_id == "project":
                continue  # Already added
            if isinstance(block_data, dict):
                block_name = block_data.get("block_name", f"Block {block_id[:8]}...")
                block_type = block_data.get("block_type", "Unknown")
                # Add ALL blocks, even if they have no actions
                display_name = f"{block_name} ({block_type})"
                combo.addItem(display_name, block_id)
                items_added += 1
            else:
                # Fallback for unexpected data structure
                display_name = f"Block {block_id[:8]}... (Unknown)"
                combo.addItem(display_name, block_id)
                items_added += 1
    
    def _populate_action_combo(self, combo: QComboBox, block_id: str):
        """Populate action combo box for a specific block"""
        combo.clear()
        if not block_id or block_id not in self.actions_by_block:
            return
        
        block_data = self.actions_by_block.get(block_id, {})
        if isinstance(block_data, dict) and "actions" in block_data:
            actions = block_data.get("actions", [])
        else:
            actions = block_data if isinstance(block_data, list) else []
        
        for action in actions:
            action_name = action.get("name", "Unknown")
            combo.addItem(action_name, action)
    
    def _subscribe_to_undo_stack(self):
        """Subscribe to undo stack changes to refresh UI after commands"""
        if self.facade and self.facade.command_bus:
            undo_stack = self.facade.command_bus.get_stack()
            if undo_stack:
                undo_stack.indexChanged.connect(self._on_undo_stack_changed)
                Log.debug("ActionSetEditor: Subscribed to undo stack changes")
            else:
                Log.warning("ActionSetEditor: Cannot subscribe to undo stack - get_stack() returned None")
    
    def _on_undo_stack_changed(self, index: int):
        """Refresh UI when undo stack changes (after commands execute)"""
        if not self._is_refreshing:
            Log.debug(f"ActionSetEditor: Undo stack changed to index {index}, refreshing from database")
            self._refresh_from_database()
    
    def _refresh_from_database(self):
        """Reload action items from database and refresh UI"""
        if not self.current_project_id:
            return
        
        self._is_refreshing = True
        try:
            # Reload from database
            self._load_action_items_from_project()
            # Refresh UI
            self._refresh_actions_list()
            Log.debug(f"ActionSetEditor: Refreshed from database - action set has {len(self.current_action_set.actions) if self.current_action_set else 0} action(s)")
        finally:
            self._is_refreshing = False
    
    def _get_action_by_id(self, action_id: str) -> Optional[ActionItem]:
        """Find action item by ID"""
        if not self.current_action_set:
            return None
        for action in self.current_action_set.actions:
            if action.id == action_id:
                return action
        return None
    
    def _get_row_by_action_id(self, action_id: str) -> Optional[int]:
        """Find table row index by action ID"""
        for row in range(self.actions_model.rowCount()):
            order_item = self.actions_model.item(row, 0)
            if order_item:
                stored_id = order_item.data(Qt.ItemDataRole.UserRole)
                if stored_id == action_id:
                    return row
        return None
    
    def _on_block_changed_by_id(self, action_id: str):
        """Handle block change using action ID"""
        row = self._get_row_by_action_id(action_id)
        if row is None:
            return
        self._on_block_changed_inline(row, action_id)
    
    def _on_action_changed_by_id(self, action_id: str):
        """Handle action change using action ID"""
        row = self._get_row_by_action_id(action_id)
        if row is None:
            return
        self._on_action_changed_inline(row, action_id)
    
    def _on_edit_params_by_id(self, action_id: str):
        """Handle edit params using action ID"""
        row = self._get_row_by_action_id(action_id)
        if row is None:
            return
        self._on_edit_params(row)
    
    def _on_delete_action_by_id(self, action_id: str):
        """Handle delete using action ID"""
        row = self._get_row_by_action_id(action_id)
        if row is None:
            return
        self._on_delete_action(row, action_id)
    
    def _on_block_changed_inline(self, row: int, action_id: Optional[str] = None):
        """Handle block change in inline combo"""
        block_combo = self.actions_table.indexWidget(self.actions_model.index(row, 1))
        if not block_combo:
            return
        
        block_id = block_combo.itemData(block_combo.currentIndex())
        if not block_id:
            # No block selected - disable action combo
            action_combo = self.actions_table.indexWidget(self.actions_model.index(row, 2))
            if action_combo:
                action_combo.clear()
                action_combo.addItem("-- Select Action --", None)
                action_combo.setEnabled(False)
            edit_btn = self.actions_table.indexWidget(self.actions_model.index(row, 4))
            if edit_btn:
                edit_btn.setEnabled(False)
            delete_btn = self.actions_table.indexWidget(self.actions_model.index(row, 5))
            if delete_btn:
                delete_btn.setEnabled(False)
            return
        
        # Update action combo
        action_combo = self.actions_table.indexWidget(self.actions_model.index(row, 2))
        if action_combo:
            self._populate_action_combo(action_combo, block_id)
            action_combo.setEnabled(True)
            if action_combo.count() > 0:
                action_combo.setCurrentIndex(0)
                # Trigger action change to update params
                self._on_action_changed_inline(row)
        
        # Update or create action in action set
        if not self.current_action_set:
            self._new_action_set()
        
        # Determine action type and block info
        is_project = (block_id == "project")
        action_type = "project" if is_project else "block"
        
        block_data = self.actions_by_block.get(block_id, {})
        if isinstance(block_data, dict) and "actions" in block_data:
            block_name = block_data.get("block_name", "Project" if is_project else f"Block {block_id[:8]}...")
        else:
            block_name = "Project" if is_project else f"Block {block_id[:8]}..."
        
        # Get valid actions (filter by action_type and appropriate identifier)
        if is_project:
            valid_actions = [a for a in self.current_action_set.actions if a.action_type == "project" and a.action_name]
        else:
            valid_actions = [a for a in self.current_action_set.actions if a.action_type == "block" and a.block_id == block_id and a.action_name]
        
        if row < len(valid_actions):
            # Update existing action using command
            action = valid_actions[row]
            # Create updated action item
            updated_action = ActionItem(
                id=action.id,
                action_set_id=action.action_set_id,
                project_id=action.project_id,
                action_type=action_type,
                block_id=None if is_project else block_id,
                block_name=block_name,
                action_name=action.action_name,
                action_description=action.action_description,
                action_args=action.action_args.copy() if action.action_args else {},
                order_index=action.order_index,
                created_at=action.created_at,
                modified_at=action.modified_at,
                metadata=action.metadata.copy() if action.metadata else {}
            )
            # Use command to update (this will persist to database)
            if updated_action.id and self.current_action_set and self.current_action_set.id:
                cmd = UpdateActionItemCommand(self.facade, updated_action)
                self.facade.command_bus.execute(cmd)
                Log.debug(f"ActionSetEditor: Updated action item block via command")
        else:
            # Create new action using command - set order_index based on table position
            all_valid_actions = [
                a for a in self.current_action_set.actions 
                if a.action_name and ((a.action_type == "block" and a.block_id) or (a.action_type == "project"))
            ]
            new_action = ActionItem(
                action_type=action_type,
                block_id=None if is_project else block_id,
                block_name=block_name,
                action_name="",
                action_description="",
                action_args={},
                order_index=len(all_valid_actions)  # Append at end
            )
            # Use command to add (this will persist to database)
            if self.current_action_set and self.current_action_set.id:
                cmd = AddActionItemCommand(self.facade, self.current_action_set.id, new_action)
                self.facade.command_bus.execute(cmd)
                Log.debug(f"ActionSetEditor: Added action item block via command")
        
        # UI will refresh automatically via undo stack change handler
        self.action_set_changed.emit()
    
    def _on_action_changed_inline(self, row: int):
        """Handle action change in inline combo"""
        action_combo = self.actions_table.indexWidget(self.actions_model.index(row, 2))
        if not action_combo:
            return
        
        action_data = action_combo.itemData(action_combo.currentIndex())
        if not action_data:
            # No action selected - disable edit button
            edit_btn = self.actions_table.indexWidget(self.actions_model.index(row, 4))
            if edit_btn:
                edit_btn.setEnabled(False)
            params_item = QStandardItem("(add action to configure)")
            params_item.setForeground(self._get_text_color(Colors.TEXT_SECONDARY))
            self.actions_model.setItem(row, 3, params_item)
            return
        
        # Enable edit and delete buttons
        edit_btn = self.actions_table.indexWidget(self.actions_model.index(row, 4))
        if edit_btn:
            edit_btn.setEnabled(True)
        delete_btn = self.actions_table.indexWidget(self.actions_model.index(row, 5))
        if delete_btn:
            delete_btn.setEnabled(True)
        
        # Update action set
        if not self.current_action_set:
            self._new_action_set()
        
        # Get block_id to determine action type
        block_combo = self.actions_table.indexWidget(self.actions_model.index(row, 1))
        block_id = None
        if block_combo:
            block_id = block_combo.itemData(block_combo.currentIndex())
        
        is_project = (block_id == "project")
        
        # Get valid actions (filter by action_type and appropriate identifier)
        if is_project:
            valid_actions = [a for a in self.current_action_set.actions if a.action_type == "project" and a.action_name]
        else:
            valid_actions = [a for a in self.current_action_set.actions if a.action_type == "block" and a.block_id == block_id and a.action_name]
        
        if row < len(valid_actions):
            # Update existing action using command
            action = valid_actions[row]
            # Create updated action item
            updated_action = ActionItem(
                id=action.id,
                action_set_id=action.action_set_id,
                project_id=action.project_id,
                action_type=action.action_type,
                block_id=action.block_id,
                block_name=action.block_name,
                action_name=action_data.get("name", ""),
                action_description=action_data.get("description", ""),
                action_args={},  # Clear params - user needs to configure them
                order_index=action.order_index,
                created_at=action.created_at,
                modified_at=action.modified_at,
                metadata=action.metadata.copy() if action.metadata else {}
            )
            # Use command to update (this will persist to database)
            if updated_action.id and self.current_action_set and self.current_action_set.id:
                cmd = UpdateActionItemCommand(self.facade, updated_action)
                self.facade.command_bus.execute(cmd)
                Log.debug(f"ActionSetEditor: Updated action item name via command")
        else:
            # Update the action we just created - find it and update via command
            if block_id:
                block_data = self.actions_by_block.get(block_id, {})
                if isinstance(block_data, dict) and "actions" in block_data:
                    block_name = block_data.get("block_name", "Project" if is_project else f"Block {block_id[:8]}...")
                else:
                    block_name = "Project" if is_project else f"Block {block_id[:8]}..."
                
                # Find the action we just created (has no action_name yet)
                for action in self.current_action_set.actions:
                    if is_project:
                        if action.action_type == "project" and not action.action_name:
                            # Update via command if it has an ID (was persisted)
                            if action.id and self.current_action_set and self.current_action_set.id:
                                updated_action = ActionItem(
                                    id=action.id,
                                    action_set_id=action.action_set_id,
                                    project_id=action.project_id,
                                    action_type=action.action_type,
                                    block_id=action.block_id,
                                    block_name=action.block_name,
                                    action_name=action_data.get("name", ""),
                                    action_description=action_data.get("description", ""),
                                    action_args=action.action_args.copy() if action.action_args else {},
                                    order_index=action.order_index,
                                    created_at=action.created_at,
                                    modified_at=action.modified_at,
                                    metadata=action.metadata.copy() if action.metadata else {}
                                )
                                cmd = UpdateActionItemCommand(self.facade, updated_action)
                                self.facade.command_bus.execute(cmd)
                                Log.debug(f"ActionSetEditor: Updated newly created action item name via command")
                            break
                    else:
                        if action.action_type == "block" and action.block_id == block_id and not action.action_name:
                            # Update via command if it has an ID (was persisted)
                            if action.id and self.current_action_set and self.current_action_set.id:
                                updated_action = ActionItem(
                                    id=action.id,
                                    action_set_id=action.action_set_id,
                                    project_id=action.project_id,
                                    action_type=action.action_type,
                                    block_id=action.block_id,
                                    block_name=action.block_name,
                                    action_name=action_data.get("name", ""),
                                    action_description=action_data.get("description", ""),
                                    action_args=action.action_args.copy() if action.action_args else {},
                                    order_index=action.order_index,
                                    created_at=action.created_at,
                                    modified_at=action.modified_at,
                                    metadata=action.metadata.copy() if action.metadata else {}
                                )
                                cmd = UpdateActionItemCommand(self.facade, updated_action)
                                self.facade.command_bus.execute(cmd)
                                Log.debug(f"ActionSetEditor: Updated newly created action item name via command")
                            break
        
        # Update params display
        params_item = QStandardItem("(double-click to configure)")
        params_item.setForeground(self._get_text_color(Colors.TEXT_SECONDARY))
        params_item.setToolTip("Double-click to configure parameters")
        self.actions_model.setItem(row, 3, params_item)
        
        # UI will refresh automatically via undo stack change handler
        self.action_set_changed.emit()
        
        # Ensure empty row exists at bottom
        current_rows = self.actions_model.rowCount()
        valid_actions = [a for a in self.current_action_set.actions if a.block_id and a.action_name]
        if current_rows <= len(valid_actions):
            self._add_empty_row(len(valid_actions))
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format action parameters for display"""
        if not params:
            return "(none)"
        
        parts = []
        for key, value in params.items():
            if isinstance(value, str) and len(value) > 30:
                value = value[:27] + "..."
            parts.append(f"{key}={value}")
        
        result = ", ".join(parts)
        if len(result) > 60:
            result = result[:57] + "..."
        return result
    
    def _on_selection_changed(self, selected, deselected):
        """Handle selection change"""
        has_selection = len(self.actions_table.selectionModel().selectedRows()) > 0
        self.remove_action_btn.setEnabled(has_selection)
    
    def _on_cell_double_clicked(self, index: QModelIndex):
        """Handle double-click on table cells"""
        # Double-click on Parameters column opens edit dialog
        if index.column() == 3:
            self._on_edit_params(index.row())
    
    def _on_context_menu_requested(self, position: QPoint):
        """Handle right-click context menu request"""
        if not self.current_project_id:
            return
        
        # Get the row at the click position
        index = self.actions_table.indexAt(position)
        row = index.row()
        
        # If clicked between rows or on empty space, row will be -1
        # Calculate insertion position based on Y coordinate
        if row < 0:
            # Clicked between rows - find which row to insert before
            y_pos = position.y()
            row_height = self.actions_table.verticalHeader().defaultSectionSize()
            
            # Check if we're in the viewport
            if y_pos >= 0:
                # Calculate approximate row based on Y position
                row = max(0, min(int(y_pos / row_height), self.actions_model.rowCount() - 1))
            else:
                # Clicked above table - insert at beginning
                row = 0
        
        # Get valid actions to determine if this is a valid insertion point
        valid_actions = [
            a for a in (self.current_action_set.actions if self.current_action_set else [])
            if a.action_name and ((a.action_type == "block" and a.block_id) or (a.action_type == "project"))
        ]
        valid_actions.sort(key=lambda x: x.order_index)
        
        # Clamp row to valid range (can insert at end too)
        row = min(row, len(valid_actions))
        
        # Create context menu
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                border-radius: {border_radius(3)};
            }}
            QMenu::item:selected {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
        """)
        
        # Add "Insert Action" option with position indicator
        if row < len(valid_actions):
            insert_text = f"Insert Action at position {row + 1}"
        else:
            insert_text = "Insert Action at end"
        insert_action = menu.addAction(insert_text)
        insert_action.triggered.connect(lambda checked=False, r=row: self._on_insert_action_at_row(r))
        
        # Show menu at cursor position
        menu.exec(self.actions_table.mapToGlobal(position))
    
    def _on_insert_action_at_row(self, row: int):
        """Insert a new action at the specified row position"""
        if not self.current_project_id:
            QMessageBox.warning(self, "No Project", "Please open or create a project first.")
            return
        
        # Discover available actions
        if not self.actions_by_block:
            result = self.facade.discover_setlist_actions(self.current_project_id)
            if result.success:
                self.actions_by_block = result.data
            else:
                QMessageBox.warning(self, "Error", f"Failed to discover actions: {result.message}")
                return
        
        # Ensure action set exists
        if not self.current_action_set:
            self._new_action_set()
        
        # Open add action dialog
        from ui.qt_gui.views.add_action_dialog import AddActionDialog
        
        dialog = AddActionDialog(
            self.facade,
            self.current_project_id,
            self.actions_by_block,
            parent=self
        )
        
        if dialog.exec():
            action_item = dialog.get_action_item()
            if action_item:
                # Set project_id
                action_item.project_id = self.current_project_id
                
                # Ensure action set has an ID
                if not self.current_action_set or not self.current_action_set.id:
                    Log.warning(f"ActionSetEditor: Cannot add action item - current_action_set.id is None")
                    return
                
                # Calculate order_index based on insertion position
                # Get valid actions to determine correct order_index
                valid_actions = [
                    a for a in self.current_action_set.actions 
                    if a.action_name and ((a.action_type == "block" and a.block_id) or (a.action_type == "project"))
                ]
                valid_actions.sort(key=lambda x: x.order_index)
                
                # Set order_index to insert at the specified row
                # If row is beyond current actions, append at end
                if row < len(valid_actions):
                    # Get the order_index of the item currently at this row position
                    target_order_index = valid_actions[row].order_index
                    # Insert at this position - shift existing items
                    action_item.order_index = target_order_index
                    # Update order_index for items at or after this position
                    # Use a macro to batch ALL updates including the add
                    self.facade.command_bus.begin_macro(f"Insert action at position {row + 1}")
                    # First, add the new item with the correct order_index
                    cmd_add = AddActionItemCommand(self.facade, self.current_action_set.id, action_item)
                    self.facade.command_bus.execute(cmd_add)
                    # Then, increment all items with order_index >= target_order_index
                    for action in valid_actions[row:]:
                        action.order_index += 1
                        cmd = UpdateActionItemCommand(self.facade, action)
                        self.facade.command_bus.execute(cmd)
                    self.facade.command_bus.end_macro()
                    Log.debug(f"ActionSetEditor: Inserted action item '{action_item.action_name}' at position {row + 1} via command")
                else:
                    # Append at end
                    action_item.order_index = len(valid_actions)
                    # Use command to add (this will persist to database)
                    cmd = AddActionItemCommand(self.facade, self.current_action_set.id, action_item)
                    self.facade.command_bus.execute(cmd)
                    Log.debug(f"ActionSetEditor: Added action item '{action_item.action_name}' at end via command")
                
                # Force immediate refresh (undo stack signal may not fire immediately)
                # Use QTimer to ensure refresh happens after command completes
                QTimer.singleShot(0, self._refresh_from_database)
                
                # UI will refresh automatically via undo stack change handler
                self.action_set_changed.emit()
    
    def _on_add_action_row(self):
        """Add a new action via dialog (alternative to inline editing)"""
        if not self.current_project_id:
            QMessageBox.warning(self, "No Project", "Please open or create a project first.")
            return
        
        # Discover available actions
        if not self.actions_by_block:
            result = self.facade.discover_setlist_actions(self.current_project_id)
            if result.success:
                self.actions_by_block = result.data
            else:
                QMessageBox.warning(self, "Error", f"Failed to discover actions: {result.message}")
                return
        
        # Ensure action set exists
        if not self.current_action_set:
            self._new_action_set()
        
        # Open add action dialog
        from ui.qt_gui.views.add_action_dialog import AddActionDialog
        
        dialog = AddActionDialog(
            self.facade,
            self.current_project_id,
            self.actions_by_block,
            parent=self
        )
        
        if dialog.exec():
            action_item = dialog.get_action_item()
            if action_item:
                # Set project_id
                action_item.project_id = self.current_project_id
                
                # Set order_index based on current table row count (will be appended at end)
                valid_actions = [
                    a for a in self.current_action_set.actions 
                    if a.action_name and ((a.action_type == "block" and a.block_id) or (a.action_type == "project"))
                ]
                action_item.order_index = len(valid_actions)
                
                # Use command to add (this will persist to database)
                if self.current_action_set.id:
                    cmd = AddActionItemCommand(self.facade, self.current_action_set.id, action_item)
                    self.facade.command_bus.execute(cmd)
                    Log.debug(f"ActionSetEditor: Added action item '{action_item.action_name}' via command")
                    
                    # Force immediate refresh (undo stack signal may not fire immediately)
                    # Use QTimer to ensure refresh happens after command completes
                    QTimer.singleShot(0, self._refresh_from_database)
                else:
                    Log.warning(f"ActionSetEditor: Cannot add action item - current_action_set.id is None")
                
                # UI will refresh automatically via undo stack change handler
                self.action_set_changed.emit()
    
    def _on_configure_params(self, row: int):
        """Configure parameters for an action"""
        if not self.current_action_set:
            return
        
        # Get valid actions (both block and project actions)
        valid_actions = [a for a in self.current_action_set.actions if a.action_name and ((a.action_type == "block" and a.block_id) or (a.action_type == "project"))]
        if row >= len(valid_actions):
            return
        
        action = valid_actions[row]
        
        # Open add action dialog in edit mode
        from ui.qt_gui.views.add_action_dialog import AddActionDialog
        
        dialog = AddActionDialog(
            self.facade,
            self.current_project_id,
            self.actions_by_block,
            parent=self
        )
        
        # Pre-select the block (or "project" for project actions)
        block_id_to_match = "project" if action.action_type == "project" else action.block_id
        for i in range(dialog.block_combo.count()):
            if dialog.block_combo.itemData(i) == block_id_to_match:
                dialog.block_combo.setCurrentIndex(i)
                # Wait for actions to populate, then select action
                from PyQt6.QtCore import QTimer
                def select_action():
                    for j in range(dialog.action_combo.count()):
                        act_data = dialog.action_combo.itemData(j)
                        if act_data and act_data.get("name") == action.action_name:
                            dialog.action_combo.setCurrentIndex(j)
                            # Populate parameters after action is selected
                            QTimer.singleShot(100, lambda: self._populate_dialog_params(dialog, action))
                            break
                QTimer.singleShot(200, select_action)
                break
        
        if dialog.exec():
            action_item = dialog.get_action_item()
            if action_item:
                # Preserve the original ID and action_set_id if editing existing
                if action.id:
                    action_item.id = action.id
                    action_item.action_set_id = action.action_set_id
                    action_item.project_id = action.project_id
                    action_item.order_index = action.order_index
                
                # Update the action in the list
                for i, a in enumerate(self.current_action_set.actions):
                    if a == action:
                        self.current_action_set.actions[i] = action_item
                        break
                self.current_action_set.update_modified()
                
                # Use command to update (this will persist to database)
                if action_item.id:
                    cmd = UpdateActionItemCommand(self.facade, action_item)
                    self.facade.command_bus.execute(cmd)
                    Log.debug(f"ActionSetEditor: Updated action item '{action_item.action_name}' via command")
                else:
                    # New action - add it
                    if self.current_action_set and self.current_action_set.id:
                        cmd = AddActionItemCommand(self.facade, self.current_action_set.id, action_item)
                        self.facade.command_bus.execute(cmd)
                        Log.debug(f"ActionSetEditor: Added new action item '{action_item.action_name}' via command")
                
                # UI will refresh automatically via undo stack change handler
                self.action_set_changed.emit()
    
    def _populate_dialog_params(self, dialog, action: ActionItem):
        """Populate dialog parameters with existing action values"""
        for key, widget in dialog.input_widgets.items():
            if key in action.action_args:
                value = action.action_args[key]
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(float(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
    
    def _on_edit_params(self, row: int):
        """Handle edit button click to configure parameters"""
        self._on_configure_params(row)
    
    # _on_rows_moved removed - ActionItemModel.moveRow() handles this automatically
    
    def _on_delete_action(self, row: int, action_id: Optional[str] = None):
        """Delete a specific action using command (order updates automatically via table)"""
        if not self.current_action_set or not self.current_project_id:
            return
        
        # Get action by ID if provided, otherwise by row
        if action_id:
            action_to_remove = self._get_action_by_id(action_id)
        else:
            # Get valid actions sorted by order_index
            valid_actions = [
                a for a in self.current_action_set.actions 
                if a.action_name and ((a.action_type == "block" and a.block_id) or (a.action_type == "project"))
            ]
            valid_actions.sort(key=lambda x: x.order_index)
            if row >= len(valid_actions):
                return
            action_to_remove = valid_actions[row]
        
        if not action_to_remove or not action_to_remove.id:
            Log.warning("ActionSetEditor: Cannot delete action without ID")
            return
        
        # Confirm removal
        reply = QMessageBox.question(
            self, "Delete Action", 
            f"Delete action '{action_to_remove.action_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Use command to delete (this will persist to database)
            cmd = DeleteActionItemCommand(self.facade, action_to_remove.id)
            self.facade.command_bus.execute(cmd)
            
            # After deletion, update order_index for remaining rows
            # This happens automatically when we refresh from database,
            # but we can also trigger it manually by updating remaining items
            self._update_order_indices_from_table()
            
            # UI will refresh automatically via undo stack change handler
            self.action_set_changed.emit()
    
    def _update_order_indices_from_table(self):
        """Update order_index values based on current table row positions"""
        if self._is_refreshing or not self.current_action_set:
            return
        
        # Get all valid actions
        valid_actions = [
            a for a in self.current_action_set.actions 
            if a.action_name and ((a.action_type == "block" and a.block_id) or (a.action_type == "project"))
        ]
        
        # Update order_index based on table position
        # Capture old state BEFORE modifying to ensure proper undo/redo
        updates_needed = []
        for table_row in range(self.actions_model.rowCount()):
            order_item = self.actions_model.item(table_row, 0)
            if not order_item:
                continue
            
            action_id = order_item.data(Qt.ItemDataRole.UserRole)
            if not action_id or action_id.startswith("temp_"):
                continue
            
            action = self._get_action_by_id(action_id)
            if action and action.order_index != table_row:
                # Capture old state before modifying
                from src.features.projects.domain import ActionItem
                old_action = ActionItem(
                    id=action.id,
                    action_set_id=action.action_set_id,
                    project_id=action.project_id,
                    action_type=action.action_type,
                    block_id=action.block_id,
                    block_name=action.block_name,
                    action_name=action.action_name,
                    action_description=action.action_description,
                    action_args=action.action_args.copy() if action.action_args else {},
                    order_index=action.order_index,  # OLD order_index
                    created_at=action.created_at,
                    modified_at=action.modified_at,
                    metadata=action.metadata.copy() if action.metadata else {}
                )
                
                # Update order_index in action object
                action.order_index = table_row
                
                # Create updated action with new order_index
                updated_action = ActionItem(
                    id=action.id,
                    action_set_id=action.action_set_id,
                    project_id=action.project_id,
                    action_type=action.action_type,
                    block_id=action.block_id,
                    block_name=action.block_name,
                    action_name=action.action_name,
                    action_description=action.action_description,
                    action_args=action.action_args.copy() if action.action_args else {},
                    order_index=table_row,  # NEW order_index
                    created_at=action.created_at,
                    modified_at=action.modified_at,
                    metadata=action.metadata.copy() if action.metadata else {}
                )
                
                updates_needed.append((updated_action, old_action))
        
        # Batch update via commands with explicit old state for proper undo/redo
        if updates_needed:
            if len(updates_needed) > 1:
                self.facade.command_bus.begin_macro(f"Reorder {len(updates_needed)} actions")
            
            for updated_action, old_action in updates_needed:
                # Pass old_action explicitly to ensure undo restores correct order
                cmd = UpdateActionItemCommand(self.facade, updated_action, old_action)
                self.facade.command_bus.execute(cmd)
            
            if len(updates_needed) > 1:
                self.facade.command_bus.end_macro()
    
    def _on_remove_action(self):
        """Remove selected action"""
        selected_rows = self.actions_table.selectionModel().selectedRows()
        if not selected_rows or not self.current_action_set:
            return
        
        # Get indices (in reverse order to avoid index shifting)
        indices = sorted([row.row() for row in selected_rows], reverse=True)
        
        # Confirm removal
        if len(indices) == 1:
            msg = f"Remove action {indices[0] + 1}?"
        else:
            msg = f"Remove {len(indices)} actions?"
        
        reply = QMessageBox.question(
            self, "Remove Actions", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Get valid actions to map indices correctly
            valid_actions = [a for a in self.current_action_set.actions if a.block_id and a.action_name]
            # Remove in reverse order using commands
            if len(indices) > 1:
                self.facade.command_bus.begin_macro(f"Delete {len(indices)} actions")
            
            for idx in indices:
                if idx < len(valid_actions):
                    action_to_remove = valid_actions[idx]
                    
                    # Use command to delete (this will persist to database)
                    if action_to_remove.id:
                        cmd = DeleteActionItemCommand(self.facade, action_to_remove.id)
                        self.facade.command_bus.execute(cmd)
                        Log.debug(f"ActionSetEditor: Deleted action item '{action_to_remove.action_name}' via command")
            
            if len(indices) > 1:
                self.facade.command_bus.end_macro()
            
            # UI will refresh automatically via undo stack change handler
            self.action_set_changed.emit()
    
    def _on_save_set(self):
        """Save the current action set to file storage"""
        if not self.current_action_set:
            return
        
        # Filter out invalid actions
        valid_actions = [a for a in self.current_action_set.actions if a.block_id and a.action_name]
        if not valid_actions:
            QMessageBox.warning(self, "No Actions", "Please add at least one action before saving.")
            return
        
        # Get name from user
        name, ok = QInputDialog.getText(
            self, "Save Action Set", "Action Set Name:",
            text=self.current_action_set.name
        )
        
        if not ok or not name.strip():
            return
        
        # Update action set with valid actions only
        self.current_action_set.actions = valid_actions
        self.current_action_set.name = name.strip()
        # Set project_id so action set is associated with this project and saved to project file
        if self.current_project_id:
            self.current_action_set.project_id = self.current_project_id
        self.current_action_set.update_modified()
        
        # Save through facade (saves to file and DB)
        result = self.facade.save_action_set(self.current_action_set)
        if result.success:
            # Update action set ID from saved result
            if result.data and hasattr(result.data, 'id'):
                self.current_action_set.id = result.data.id
            
            # Save action items to database using commands
            if len(valid_actions) > 1:
                self.facade.command_bus.begin_macro(f"Save {len(valid_actions)} action items")
            
            for i, action_item in enumerate(valid_actions):
                action_item.action_set_id = self.current_action_set.id
                action_item.project_id = self.current_project_id or ""
                action_item.order_index = i
                
                if action_item.id:
                    # Update existing using command
                    cmd = UpdateActionItemCommand(self.facade, action_item)
                    self.facade.command_bus.execute(cmd)
                else:
                    # Create new using command
                    cmd = AddActionItemCommand(self.facade, self.current_action_set.id, action_item)
                    self.facade.command_bus.execute(cmd)
            
            if len(valid_actions) > 1:
                self.facade.command_bus.end_macro()
            
            Log.info(f"ActionSetEditor: Saved action set '{name}' with {len(valid_actions)} action item(s)")
            QMessageBox.information(self, "Saved", f"Action set '{name}' saved to action sets folder.")
            
            # Refresh load combo
            self._populate_load_combo()
            # Select the saved set in the combo
            for i in range(self.load_set_combo.count()):
                if self.load_set_combo.itemData(i) == name:
                    self.load_set_combo.setCurrentIndex(i)
                    break
            # Save as current action set for this project
            self._save_current_action_set_to_setlist(name)
        else:
            error_msg = result.message
            if result.errors:
                error_msg += f"\n\n{result.errors[0]}"
            QMessageBox.warning(self, "Failed to Save", error_msg)
    
    def _populate_load_combo(self):
        """Populate the load set combo box with available action sets from file storage"""
        self.load_set_combo.clear()
        self.load_set_combo.addItem("-- Select Action Set --", None)
        
        # List available action sets (global, not project-specific)
        result = self.facade.list_action_sets()
        if result.success:
            action_sets = result.data or []
            for action_set in action_sets:
                # Mark standard sets with a prefix
                is_standard = action_set.metadata.get("is_standard", False)
                prefix = "[Standard] " if is_standard else ""
                display_text = f"{prefix}{action_set.name} ({len(action_set.actions)} actions)"
                # Store name for loading (file-based uses names)
                self.load_set_combo.addItem(display_text, action_set.name)
    
    def _on_load_set_selected(self, index: int):
        """Handle action set selection from dropdown"""
        if index <= 0:  # First item is "-- Select Action Set --"
            return
        
        action_set_name = self.load_set_combo.itemData(index)
        if not action_set_name:
            return
        
        # Load the action set by name
        load_result = self.facade.load_action_set(action_set_name)
        if load_result.success:
            action_set = load_result.data
            
            # Load action items into project database (replace existing)
            loaded_items = []
            if self.current_project_id:
                load_into_project_result = self.facade.load_action_set_into_project(
                    action_set_name,
                    project_id=self.current_project_id,
                    replace=True  # Replace existing action items
                )
                if load_into_project_result.success:
                    loaded_items = load_into_project_result.data or []
                    # Update action items to have the correct action_set_id so they're associated with this action set
                    # Use commands for undo/redo support
                    if len(loaded_items) > 1:
                        self.facade.command_bus.begin_macro(f"Associate {len(loaded_items)} action items with action set")
                    
                    for item in loaded_items:
                        item.action_set_id = action_set.id
                        cmd = UpdateActionItemCommand(self.facade, item)
                        self.facade.command_bus.execute(cmd)
                    
                    if len(loaded_items) > 1:
                        self.facade.command_bus.end_macro()
                else:
                    Log.warning(f"ActionSetEditor: Failed to load action items into project: {load_into_project_result.message}")
            
            # Set the action set - use loaded items if available, otherwise use actions from loaded action set
            if loaded_items:
                # Use the action items we just loaded into the project
                action_set.actions = loaded_items
            self.set_action_set(action_set)
            
            # Save the selection to setlist so it persists
            self._save_current_action_set_to_setlist(action_set_name)
            Log.info(f"ActionSetEditor: Loaded action set '{action_set_name}'")
        else:
            QMessageBox.warning(self, "Error", f"Failed to load action set: {load_result.message}")
    
    def _on_new_set(self):
        """Create a new action set"""
        if self.current_action_set and len(self.current_action_set.actions) > 0:
            reply = QMessageBox.question(
                self, "New Action Set", "Create a new action set? Current changes will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        self._new_action_set()
    
    def load_project(self, project_id: str):
        """Load project and discover available actions"""
        # Guard against duplicate loads
        if hasattr(self, '_is_loading_project') and self._is_loading_project:
            if self.current_project_id == project_id:
                Log.debug(f"ActionSetEditor: Already loading project {project_id}, skipping duplicate call")
                return
        
        self._is_loading_project = True
        try:
            self.current_project_id = project_id
            
            # Re-subscribe to undo stack (command_bus may not have been available at init time)
            self._subscribe_to_undo_stack()
            
            if project_id:
                Log.debug(f"ActionSetEditor: Loading project {project_id} and discovering actions")
                result = self.facade.discover_setlist_actions(project_id)
                if result.success:
                    self.actions_by_block = result.data or {}
                    Log.debug(f"ActionSetEditor: Successfully discovered actions. Got {len(self.actions_by_block)} blocks")
                else:
                    Log.error(f"ActionSetEditor: Failed to discover actions: {result.message}")
                    self.actions_by_block = {}
            else:
                Log.debug("ActionSetEditor: No project_id provided to load_project")
                self.actions_by_block = {}
            
            # Refresh load combo with available action sets
            self._populate_load_combo()
            
            # Load action items from database for this project
            self._load_action_items_from_project()
            
            # Try to load the previously selected action set from setlist metadata (for action set name/metadata)
            self._load_current_action_set_from_setlist()
            
            # Refresh the actions table (this will show loaded action items or empty row)
            self._refresh_actions_list()
            
            # Ensure empty row is shown if no actions
            if not self.current_action_set or len(self.current_action_set.actions) == 0:
                # Make sure we have an action set to work with
                if not self.current_action_set:
                    self._new_action_set()
                # Ensure empty row is visible
                if self.actions_model.rowCount() == 0:
                    self._add_empty_row(0)
        finally:
            self._is_loading_project = False
    
    def _load_action_items_from_project(self):
        """Load action items from database for the current project"""
        if not self.current_project_id:
            Log.debug("ActionSetEditor: No project_id, skipping action items load")
            return
        
        try:
            result = self.facade.list_action_items_by_project(self.current_project_id)
            if result.success and result.data:
                all_action_items = result.data
                Log.debug(f"ActionSetEditor: Found {len(all_action_items)} total action item(s) in project database")
                
                # Create or update action set
                if not self.current_action_set:
                    self._new_action_set()
                
                # Filter action items by current action_set_id if we have one
                # This ensures we only show items belonging to the current action set
                if self.current_action_set.id:
                    action_items = [
                        item for item in all_action_items 
                        if item.action_set_id == self.current_action_set.id
                    ]
                    Log.debug(f"ActionSetEditor: Filtered to {len(action_items)} action item(s) for action set {self.current_action_set.id}")
                else:
                    # No action_set_id yet - use all items (will be filtered when action set is created)
                    action_items = all_action_items
                    # If items have an action_set_id, update our action set ID to match
                    if action_items and action_items[0].action_set_id:
                        self.current_action_set.id = action_items[0].action_set_id
                        Log.debug(f"ActionSetEditor: Set action_set_id to {self.current_action_set.id} from loaded items")
                        # Re-filter with the new ID
                        action_items = [
                            item for item in all_action_items 
                            if item.action_set_id == self.current_action_set.id
                        ]
                
                # Set the action items in the current action set
                self.current_action_set.actions = action_items
            else:
                Log.debug(f"ActionSetEditor: No action items found for project {self.current_project_id}")
                # Ensure we have an action set even if no items
                if not self.current_action_set:
                    self._new_action_set()
                else:
                    # Clear actions if none found
                    self.current_action_set.actions = []
        except Exception as e:
            Log.warning(f"ActionSetEditor: Failed to load action items from project: {e}")
            # Ensure we have an action set even on error
            if not self.current_action_set:
                self._new_action_set()
            else:
                # Clear actions on error
                self.current_action_set.actions = []
    
    def _load_current_action_set_from_setlist(self):
        """Load the previously selected action set metadata from setlist (name, description only - action items come from DB)"""
        if not self.current_project_id:
            return
        
        try:
            # Get the setlist for this project
            if hasattr(self.facade, 'setlist_service') and self.facade.setlist_service:
                setlist = self.facade.setlist_service._setlist_repo.get_by_project(self.current_project_id)
                if setlist and setlist.metadata.get("current_action_set_name"):
                    action_set_name = setlist.metadata["current_action_set_name"]
                    Log.debug(f"ActionSetEditor: Found saved action set name: {action_set_name}")
                    
                    # Load the action set from file to get metadata (name, description)
                    result = self.facade.load_action_set(action_set_name)
                    if result.success and result.data:
                        file_action_set = result.data
                        
                        # Update current action set metadata but keep the action items we loaded from DB
                        if self.current_action_set:
                            current_items = self.current_action_set.actions
                            self.current_action_set.name = file_action_set.name
                            self.current_action_set.description = file_action_set.description
                            self.current_action_set.id = file_action_set.id
                            self.current_action_set.actions = current_items  # Keep items from DB
                        
                        # Select in combo
                        for i in range(self.load_set_combo.count()):
                            if self.load_set_combo.itemData(i) == action_set_name:
                                self.load_set_combo.setCurrentIndex(i)
                                break
                    else:
                        Log.debug(f"ActionSetEditor: Could not load saved action set '{action_set_name}'")
        except Exception as e:
            Log.debug(f"ActionSetEditor: Could not load saved action set: {e}")
    
    def _save_current_action_set_to_setlist(self, action_set_name: str):
        """Save the current action set name to setlist metadata"""
        if not self.current_project_id:
            return
        
        try:
            if hasattr(self.facade, 'setlist_service') and self.facade.setlist_service:
                setlist = self.facade.setlist_service._setlist_repo.get_by_project(self.current_project_id)
                if setlist:
                    setlist.metadata["current_action_set_name"] = action_set_name
                    setlist.update_modified()
                    self.facade.setlist_service._setlist_repo.update(setlist)
                    Log.debug(f"ActionSetEditor: Saved current action set '{action_set_name}' to setlist")
        except Exception as e:
            Log.warning(f"ActionSetEditor: Could not save current action set to setlist: {e}")
    
    def get_action_set(self) -> Optional[ActionSet]:
        """Get the current action set"""
        return self.current_action_set
    
    def set_action_set(self, action_set: ActionSet):
        """Set the current action set"""
        self.current_action_set = action_set
        
        # Try to load action items from database for this project and action set
        # If found, use those (they're the "live" version for this project)
        # Otherwise, use the actions from the loaded action set (from file storage)
        db_action_items = []
        if action_set.id and self.current_project_id:
            # Try to load action items from database for this project that match this action set
            result = self.facade.list_action_items_by_project(self.current_project_id)
            if result.success and result.data:
                # Filter to items that match this action set ID
                db_action_items = [item for item in result.data if item.action_set_id == action_set.id]
                if db_action_items:
                    # Use database action items (they're the live version for this project)
                    self.current_action_set.actions = db_action_items
                    Log.debug(f"ActionSetEditor: Loaded {len(db_action_items)} action item(s) from database for set '{action_set.name}'")
        
        # If no database items found, use actions from the loaded action set (from file storage)
        if not db_action_items and action_set.actions:
            self.current_action_set.actions = action_set.actions
            Log.debug(f"ActionSetEditor: Using {len(action_set.actions)} action item(s) from loaded action set '{action_set.name}'")
        
        self._refresh_actions_list()
        self.action_set_changed.emit()
        
        # Update load combo to show current set (match by name since combo stores names)
        action_set_name = action_set.name
        for i in range(self.load_set_combo.count()):
            if self.load_set_combo.itemData(i) == action_set_name:
                self.load_set_combo.setCurrentIndex(i)
                break
    
    def _get_text_color(self, color):
        """Get QColor for text"""
        from PyQt6.QtGui import QColor
        return QColor(color.name())
    
    # Style helpers
    def _add_button_style(self) -> str:
        """Style for the + add button"""
        return f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE.name()};
                border: none;
                border-radius: {border_radius(14)};
                color: {Colors.TEXT_PRIMARY.name()};
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.ACCENT_BLUE.darker(110).name()};
            }}
        """
    
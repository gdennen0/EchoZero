"""
Node Editor Window

Main view window for visual block graph editing.
Includes embedded Properties and Quick Actions panels.
Implements IStatefulWindow for saving/restoring internal state.
"""
from typing import Optional, Dict, Any, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QPushButton,
    QMenu, QInputDialog, QSplitter, QFrame, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QUndoStack

from src.application.api.application_facade import ApplicationFacade
from src.application.commands import AddBlockCommand, DeleteBlockCommand
from src.utils.message import Log
from ui.qt_gui.node_editor.node_scene import NodeScene
from ui.qt_gui.design_system import Colors
from ui.qt_gui.core.window_state_types import IStatefulWindow


class NodeEditorWindow(QWidget, IStatefulWindow):
    """
    Node editor main view window.
    
    Simple QWidget container for the node editor.
    Kept minimal to ensure docking works correctly.
    """
    
    block_selected = pyqtSignal(str)  # Emits block ID when selected
    block_panel_requested = pyqtSignal(str)  # Request to open block panel
    
    def __init__(self, facade: ApplicationFacade, undo_stack: Optional[QUndoStack] = None, parent=None):
        super().__init__(parent)
        self.facade = facade
        self.undo_stack = undo_stack
        self.properties_panel = None
        self.actions_panel = None
        self.block_types = []
        
        self._setup_ui()
        self._load_block_types()
    
    def _setup_ui(self):
        """Setup the node editor UI with embedded side panels"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # Main content area with splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(2)
        
        # Left side: Node canvas
        from ui.qt_gui.node_editor.node_graphics_view import NodeGraphicsView
        
        self.scene = NodeScene(self.facade, undo_stack=self.undo_stack)
        self.view = NodeGraphicsView(self.scene)
        main_splitter.addWidget(self.view)
        
        # Right side: Properties and Actions panels stacked vertically
        right_panel = self._create_side_panels()
        main_splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (canvas gets more space)
        main_splitter.setSizes([700, 280])
        main_splitter.setStretchFactor(0, 1)  # Canvas stretches
        main_splitter.setStretchFactor(1, 0)  # Side panel fixed width
        
        layout.addWidget(main_splitter)
        
        # Connect signals
        self.scene.block_selected.connect(self._on_block_selected_internal)
        self.scene.block_double_clicked.connect(self.block_panel_requested.emit)
    
    def _create_toolbar(self) -> QToolBar:
        """Create the toolbar"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        
        # Add block button with menu
        self.btn_add_block = QPushButton("Add Block")
        self.btn_add_block.clicked.connect(self._show_add_block_menu)
        toolbar.addWidget(self.btn_add_block)
        
        # Remove block button
        self.btn_remove_block = QPushButton("Remove Block")
        self.btn_remove_block.clicked.connect(self._on_remove_block)
        toolbar.addWidget(self.btn_remove_block)
        
        # Duplicate block button
        self.btn_duplicate_block = QPushButton("Duplicate Block")
        self.btn_duplicate_block.clicked.connect(self._on_duplicate_block)
        toolbar.addWidget(self.btn_duplicate_block)
        
        # Connect blocks button
        self.btn_connect = QPushButton("Connect Blocks")
        self.btn_connect.clicked.connect(self._on_connect_blocks)
        toolbar.addWidget(self.btn_connect)
        
        # Disconnect blocks button
        self.btn_disconnect = QPushButton("Disconnect Blocks")
        self.btn_disconnect.clicked.connect(self._on_disconnect_blocks)
        toolbar.addWidget(self.btn_disconnect)
        
        toolbar.addSeparator()
        
        # Hierarchical layout button (top-down)
        self.btn_auto_layout = QPushButton("Auto Layout")
        self.btn_auto_layout.clicked.connect(self._on_auto_layout)
        toolbar.addWidget(self.btn_auto_layout)
        
        btn_reset_view = QPushButton("Reset View")
        btn_reset_view.clicked.connect(self._on_reset_view)
        toolbar.addWidget(btn_reset_view)
        
        return toolbar
    
    def _create_side_panels(self) -> QWidget:
        """Create the right-side panels (Properties and Actions)"""
        from ui.qt_gui.core.properties_panel import PropertiesPanel
        from ui.qt_gui.core.actions_panel import ActionsPanel
        
        # Create container with vertical splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)
        splitter.setMinimumWidth(280)
        splitter.setMaximumWidth(400)
        
        # Properties panel
        self.properties_panel = PropertiesPanel(self.facade)
        splitter.addWidget(self.properties_panel)
        
        # Actions panel
        self.actions_panel = ActionsPanel(self.facade, undo_stack=self.undo_stack)
        splitter.addWidget(self.actions_panel)
        
        # Connect actions panel signals
        self.actions_panel.block_deleted.connect(self.refresh)
        self.actions_panel.execute_block_requested.connect(self._on_execute_single_block)
        self.actions_panel.panel_requested.connect(self.block_panel_requested.emit)
        self.actions_panel.filter_dialog_requested.connect(self._on_open_filter_dialog)
        
        # Equal split
        splitter.setSizes([200, 200])
        
        return splitter
    
    def _on_block_selected_internal(self, block_id: str):
        """Handle block selection - update embedded panels and emit signal"""
        # Update embedded panels
        if self.properties_panel:
            self.properties_panel.show_block_properties(block_id)
        if self.actions_panel:
            self.actions_panel.show_block_actions(block_id)
        
        # Emit signal for external listeners
        self.block_selected.emit(block_id)
    
    def _on_execute_single_block(self, block_id: str):
        """Handle execute single block request from actions panel. Runs synchronously on main thread."""
        main_window = self.window()
        if main_window and hasattr(main_window, "_on_execute_single_block"):
            main_window._on_execute_single_block(block_id)
            return
        # Fallback when not embedded in main window (e.g. tests): run synchronously
        try:
            result = self.facade.execute_block(block_id)
            if result.success:
                self._on_execution_thread_complete(True)
            else:
                self._on_execution_thread_failed(
                    result.message or "Execution failed",
                    result.errors or [],
                )
        except Exception as e:
            self._on_execution_thread_failed(str(e), [])

    def _on_execution_thread_complete(self, success: bool):
        if success:
            Log.info("Block execution completed.")

    def _on_execution_thread_failed(self, error_message: str, detailed_errors=None):
        Log.error(f"Block execution failed: {error_message}")
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox(self.window() if self.window() else None)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Execution Error")
        msg.setText(error_message)
        if detailed_errors:
            error_strings = [str(e) for e in detailed_errors if isinstance(e, str)]
            if error_strings:
                msg.setDetailedText("\n\n".join(error_strings))
        msg.exec()
    
    def _on_open_filter_dialog(self, block_id: str):
        """Open the data filter dialog for a block"""
        from ui.qt_gui.dialogs.data_filter_dialog import DataFilterDialog
        dialog = DataFilterDialog(block_id, self.facade, parent=self)
        dialog.exec()
    
    def _load_block_types(self):
        """Load available block types from facade"""
        result = self.facade.list_block_types()
        if result.success and result.data:
            # result.data is now a flat list of block type dicts
            # Store as list of (type_id, name) tuples for menu display
            self.block_types = [(bt["type_id"], bt["name"]) for bt in result.data]
    
    def _show_add_block_menu(self):
        """Show menu with available block types"""
        if not self.block_types:
            Log.warning("No block types available")
            return
        
        menu = QMenu(self)
        
        # block_types is list of (type_id, name) tuples
        for type_id, name in self.block_types:
            action = menu.addAction(name)
            action.triggered.connect(lambda checked, bt=type_id: self._on_add_block(bt))
        
        # Show menu at button position
        menu.exec(self.btn_add_block.mapToGlobal(self.btn_add_block.rect().bottomLeft()))
    
    def _generate_unique_block_name(self, block_type: str) -> str:
        """
        Generate a unique block name for the given block type.
        
        Checks existing blocks in the current project and suggests
        the next available name (e.g., "Block1", "Block2", etc.).
        
        Args:
            block_type: Type of block (e.g., "LoadAudio", "Separator")
            
        Returns:
            Unique block name
        """
        # Get current project
        project_id = self.facade.get_current_project_id()
        if not project_id:
            # No project loaded, use default
            return f"{block_type}1"
        
        # Get existing blocks
        result = self.facade.list_blocks()
        if not result.success or not result.data:
            # No existing blocks, use default
            return f"{block_type}1"
        
        existing_names = {block.name for block in result.data}
        
        # Try simple name first
        base_name = f"{block_type}1"
        if base_name not in existing_names:
            return base_name
        
        # Find next available number
        counter = 2
        while True:
            name = f"{block_type}{counter}"
            if name not in existing_names:
                return name
            counter += 1
    
    def _validate_block_name(self, name: str) -> Tuple[bool, str]:
        """
        Validate a block name to ensure it's unique.
        
        Args:
            name: Block name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name or not name.strip():
            return False, "Block name cannot be empty"
        
        name = name.strip()
        
        # Get current project
        project_id = self.facade.get_current_project_id()
        if not project_id:
            return True, ""  # No project loaded, can't validate
        
        # Check if name already exists
        result = self.facade.list_blocks()
        if result.success and result.data:
            existing_names = {block.name for block in result.data}
            if name in existing_names:
                return False, f"A block named '{name}' already exists. Please choose a different name."
        
        return True, ""
    
    def _on_add_block(self, block_type: str):
        """Handle block type selection"""
        # Generate unique default name
        default_name = self._generate_unique_block_name(block_type)
        
        # Prompt for block name with unique default
        name, ok = QInputDialog.getText(
            self,
            "Add Block",
            f"Name for {block_type} block:",
            text=default_name
        )
        
        if ok and name:
            # Validate name
            is_valid, error_msg = self._validate_block_name(name)
            if not is_valid:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Invalid Block Name",
                    error_msg
                )
                # Retry with corrected default name
                return self._on_add_block(block_type)
            
            
            # Use CommandBus for undoable command
            cmd = AddBlockCommand(self.facade, block_type, name)
            
            
            self.facade.command_bus.execute(cmd)
            
            
            Log.info(f"Added block: {name} ({block_type})")
            self.refresh()
    
    def _on_remove_block(self):
        """Show dialog to select and remove blocks"""
        from ui.qt_gui.core.base_components import ItemSelectionDialog
        
        # Get all blocks in current project
        blocks_result = self.facade.list_blocks()
        if not blocks_result.success or not blocks_result.data:
            Log.warning("No blocks to remove")
            return
        
        blocks = blocks_result.data
        
        # Show selection dialog
        selected_blocks = ItemSelectionDialog.show_delete_dialog(
            parent=self,
            title="Remove Blocks",
            items=blocks,
            item_name_getter=lambda b: f"{b.name} ({b.type})"
        )
        
        if selected_blocks:
            # Use CommandBus for undoable commands
            if len(selected_blocks) > 1:
                self.facade.command_bus.begin_macro(f"Delete {len(selected_blocks)} Blocks")
            
            for block in selected_blocks:
                cmd = DeleteBlockCommand(self.facade, block.id)
                self.facade.command_bus.execute(cmd)
                Log.info(f"Removed block: {block.name}")
            
            if len(selected_blocks) > 1:
                self.facade.command_bus.end_macro()
            
            # Refresh the view
            self.refresh()
    
    def _on_duplicate_block(self):
        """Duplicate the selected block"""
        selected_block_id = self.get_selected_block_id()
        if not selected_block_id:
            Log.warning("No block selected for duplication")
            return
        
        from src.application.commands import DuplicateBlockCommand
        
        # Use CommandBus for undoable command
        cmd = DuplicateBlockCommand(self.facade, selected_block_id)
        self.facade.command_bus.execute(cmd)
        Log.info(f"Duplicated block: {selected_block_id}")
        self.refresh()
    
    def _on_connect_blocks(self):
        """Open connection dialog"""
        from ui.qt_gui.connection.connection_dialog import ConnectionDialog
        
        # Get selected block if any
        selected_block_id = self.get_selected_block_id()
        
        # Show dialog (pre-select source if block is selected)
        if ConnectionDialog.show_dialog(self.facade, selected_block_id, parent=self):
            Log.info("Connection created via toolbar dialog")
            self.refresh()
    
    def _on_disconnect_blocks(self):
        """Open disconnect dialog"""
        from ui.qt_gui.connection.disconnect_dialog import DisconnectDialog
        
        # Get selected block if any
        selected_block_id = self.get_selected_block_id()
        
        # Show dialog (pre-select block if selected)
        if DisconnectDialog.show_dialog(self.facade, selected_block_id, parent=self):
            Log.info("Connection disconnected via toolbar dialog")
            self.refresh()
    
    def _on_auto_layout(self):
        """Auto-arrange blocks with hierarchical (top-down) layout"""
        self.scene.auto_layout("hierarchical")
    
    def _on_reset_view(self):
        """Reset view to center"""
        self.view.reset_view()
    
    def refresh(self):
        """
        Refresh the node editor with current project state.
        
        Preserves the current viewport position and zoom level so the view
        does not snap/reset to center on every add/delete/connect operation.
        """
        # Save current viewport state before refresh
        saved_zoom = self.view.zoom_level
        saved_center = self.view.get_viewport_center()
        
        self.scene.refresh()
        
        # Restore viewport state after refresh (preserve user's pan/zoom)
        if saved_center and self.scene.block_items:
            self.view.set_zoom_level(saved_zoom)
            self.view.center_on_point(saved_center['x'], saved_center['y'])
    
    def refresh_and_center(self):
        """
        Refresh the node editor AND center the view on all blocks.
        
        Use this only for major transitions like project load, new project,
        or explicit user requests. For incremental changes (add/delete/connect),
        use refresh() which preserves the viewport position.
        """
        self.scene.refresh()
        if self.scene.block_items:
            self._center_view_on_blocks()
    
    def _center_view_on_blocks(self):
        """Center the view on all visible blocks"""
        if not self.scene.block_items:
            return
        
        # Calculate bounding box of all blocks
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for block_item in self.scene.block_items.values():
            pos = block_item.pos()
            rect = block_item.boundingRect()
            min_x = min(min_x, pos.x() + rect.left())
            min_y = min(min_y, pos.y() + rect.top())
            max_x = max(max_x, pos.x() + rect.right())
            max_y = max(max_y, pos.y() + rect.bottom())
        
        # Center on the middle of all blocks
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        self.view.centerOn(center_x, center_y)
    
    def get_selected_block_id(self):
        """Get currently selected block ID"""
        return self.scene.get_selected_block_id()
    
    
    # ==================== IStatefulWindow Implementation ====================
    
    def get_internal_state(self) -> Dict[str, Any]:
        """
        Get internal state for saving (IStatefulWindow interface).
        
        Saves zoom level and viewport position.
        """
        state = {}
        
        # Save zoom level
        if hasattr(self.view, 'zoom_level'):
            state['zoom_level'] = self.view.zoom_level
        
        # Save viewport center
        if hasattr(self.view, 'get_viewport_center'):
            center = self.view.get_viewport_center()
            if center:
                state['viewport_center'] = center
        
        # Save selected block
        selected = self.scene.selected_blocks()
        if selected:
            state['selected_block'] = selected[0]
        
        return state
    
    def restore_internal_state(self, state: Dict[str, Any]) -> None:
        """
        Restore internal state after loading (IStatefulWindow interface).
        
        Restores zoom level and viewport position.
        """
        # Restore zoom level
        if 'zoom_level' in state and hasattr(self.view, 'set_zoom_level'):
            self.view.set_zoom_level(state['zoom_level'])
        
        # Restore viewport center
        if 'viewport_center' in state and hasattr(self.view, 'center_on_point'):
            center = state['viewport_center']
            if isinstance(center, dict):
                self.view.center_on_point(center.get('x', 0), center.get('y', 0))
        
        # Restore selected block
        if 'selected_block' in state:
            self.scene.select_block(state['selected_block'])

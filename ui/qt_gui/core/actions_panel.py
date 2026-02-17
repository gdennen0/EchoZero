"""
Quick Actions Panel

Displays available quick actions for the selected block.
Actions are decoupled - each block type defines its own actions.
"""
from typing import Optional, Dict, Any, TYPE_CHECKING
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QPushButton, 
    QLabel, QMessageBox, QInputDialog, QFileDialog,
    QFrame, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal

from src.application.api.application_facade import ApplicationFacade
from src.application.commands import DeleteBlockCommand, RenameBlockCommand
from src.utils.message import Log
from src.utils.settings import app_settings
from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory

if TYPE_CHECKING:
    from PyQt6.QtGui import QUndoStack


class ActionsPanel(ThemeAwareMixin, QWidget):
    """
    Panel showing quick actions for selected blocks.
    
    Actions are loaded from the decoupled QuickAction system,
    where each block type defines its own relevant actions.
    """
    
    block_deleted = pyqtSignal()  # Emitted when a block is deleted
    execute_block_requested = pyqtSignal(str)  # Emitted when block execution requested (block_id)
    panel_requested = pyqtSignal(str)  # Emitted when panel should be opened (block_id)
    filter_dialog_requested = pyqtSignal(str)  # Emitted when filter dialog should be opened (block_id)
    
    def __init__(self, facade: ApplicationFacade, undo_stack: Optional["QUndoStack"] = None, parent=None):
        super().__init__(parent)
        self.facade = facade
        self.undo_stack = undo_stack
        self.current_block_id: Optional[str] = None
        self.current_block_type: Optional[str] = None
        
        self._setup_ui()
        
        # Subscribe to block update events to refresh when settings change
        self._subscribe_to_events()
        self._init_theme_aware()
    
    def _setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.SM)
        
        # Title
        title = QLabel("Quick Actions")
        title.setFont(Typography.heading_font())
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600;")
        layout.addWidget(title)
        
        # Scroll area for actions
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
        """)
        
        # Container for action buttons
        self.actions_container = QWidget()
        self.actions_layout = QVBoxLayout(self.actions_container)
        self.actions_layout.setSpacing(Spacing.XS)
        self.actions_layout.setContentsMargins(0, 0, 0, 0)
        
        # Placeholder
        self.placeholder = QLabel("Select a block to see quick actions")
        self.placeholder.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: 20px;")
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.actions_layout.addWidget(self.placeholder)
        
        self.actions_layout.addStretch()
        scroll.setWidget(self.actions_container)
        layout.addWidget(scroll)
    
    def show_block_actions(self, block_id: str):
        """Show quick actions for the selected block"""
        self.current_block_id = block_id
        
        # Clear existing actions
        self._clear_actions()
        
        # Get block details
        result = self.facade.describe_block(block_id)
        if not result.success or not result.data:
            self._show_placeholder("Block not found")
            return
        
        block = result.data
        self.current_block_type = block.type
        block_name = block.name
        
        # Hide placeholder
        if self.placeholder and not self.placeholder.isHidden():
            self.placeholder.hide()
        
        # Load quick actions from the decoupled system
        try:
            from src.application.blocks.quick_actions import (
                get_quick_actions, ActionCategory
            )
            actions = get_quick_actions(block.type)
            
            # Group by category
            categories = {}
            for action in actions:
                cat = action.category
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(action)
            
            # Display order: Execute, Configure, File, View, Edit
            category_order = [
                ActionCategory.EXECUTE,
                ActionCategory.CONFIGURE,
                ActionCategory.FILE,
                ActionCategory.VIEW,
                ActionCategory.EDIT
            ]
            
            first_section = True
            for cat in category_order:
                if cat not in categories:
                    continue
                
                cat_actions = categories[cat]
                
                # Add separator between sections
                if not first_section:
                    self._add_separator()
                first_section = False
                
                # Section label
                self._add_section_label(cat.value.title())
                
                # Add action buttons
                for action in cat_actions:
                    self._add_quick_action_button(action)
            
        except ImportError:
            # Fallback if quick_actions module not available
            self._add_fallback_actions(block)
    
    def _add_quick_action_button(self, action):
        """Add a button for a quick action"""
        btn = QPushButton(action.name)
        btn.setToolTip(action.description)
        
        # Style based on action type
        if action.dangerous:
            btn.setStyleSheet(StyleFactory.button("danger"))
        elif action.primary:
            btn.setStyleSheet(StyleFactory.button("primary"))
        else:
            btn.setStyleSheet(StyleFactory.button())
        
        # Connect click to handler
        btn.clicked.connect(lambda: self._execute_quick_action(action))
        
        self.actions_layout.insertWidget(self.actions_layout.count() - 1, btn)
    
    def _execute_quick_action(self, action):
        """Execute a quick action, handling input requirements"""
        if not self.current_block_id:
            return

        # Run Execute in background thread so UI stays responsive (e.g. PyTorch trainer)
        from src.application.blocks.quick_actions import is_execute_block_action
        if is_execute_block_action(action):
            self.execute_block_requested.emit(self.current_block_id)
            return

        try:
            result = action.handler(self.facade, self.current_block_id)
            
            if isinstance(result, dict):
                # Handle special return values
                if result.get("needs_input"):
                    self._handle_input_request(action, result)
                elif result.get("needs_confirmation"):
                    self._handle_confirmation(action, result)
                elif result.get("open_panel"):
                    self.panel_requested.emit(self.current_block_id)
                elif result.get("open_filter_dialog"):
                    self.filter_dialog_requested.emit(self.current_block_id)
                elif hasattr(result, 'success'):
                    if result.success:
                        Log.info(f"Action '{action.name}' completed successfully")
                    else:
                        Log.error(f"Action failed: {result.message if hasattr(result, 'message') else result}")
            elif hasattr(result, 'success'):
                if result.success:
                    Log.info(f"Action '{action.name}' completed successfully")
                    # Emit delete signal if action was dangerous and succeeded
                    if action.dangerous:
                        self.block_deleted.emit()
                else:
                    Log.error(f"Action failed: {result.message if hasattr(result, 'message') else result}")
                    
        except Exception as e:
            Log.error(f"Quick action error: {e}")
            QMessageBox.warning(self, "Action Error", str(e))
    
    def _handle_input_request(self, action, request: Dict[str, Any]):
        """Handle action that needs user input"""
        input_type = request.get("input_type")
        title = request.get("title", action.name)
        
        if input_type == "text":
            text, ok = QInputDialog.getText(
                self, title, 
                request.get("prompt", "Enter value:")
            )
            if ok and text:
                # Check if this is a rename action - use CommandBus
                if "rename" in action.name.lower():
                    cmd = RenameBlockCommand(self.facade, self.current_block_id, text)
                    self.facade.command_bus.execute(cmd)
                else:
                    action.handler(self.facade, self.current_block_id, new_name=text)
                # Refresh if this was a rename
                self.show_block_actions(self.current_block_id)
                
        elif input_type == "file":
            start_dir = app_settings.get_dialog_path("action_file")
            file_path, _ = QFileDialog.getOpenFileName(
                self, title, start_dir,
                request.get("file_filter", "All Files (*)")
            )
            if file_path:
                app_settings.set_dialog_path("action_file", file_path)
                result = action.handler(self.facade, self.current_block_id, file_path=file_path)
                if hasattr(result, 'success') and result.success:
                    QMessageBox.information(self, "Success", f"File set:\n{file_path}")
                
        elif input_type == "directory":
            start_dir = app_settings.get_dialog_path("action_dir")
            directory = QFileDialog.getExistingDirectory(self, title, start_dir)
            if directory:
                app_settings.set_dialog_path("action_dir", directory)
                result = action.handler(self.facade, self.current_block_id, directory=directory)
                if hasattr(result, 'success') and result.success:
                    QMessageBox.information(self, "Success", f"Directory set:\n{directory}")
                
        elif input_type == "choice":
            choices = request.get("choices", [])
            labels = request.get("labels", choices)
            default = request.get("default", choices[0] if choices else "")
            default_idx = choices.index(default) if default in choices else 0
            
            choice, ok = QInputDialog.getItem(
                self, title, "Select:", labels, default_idx, False
            )
            if ok:
                # Get actual value from choices
                idx = labels.index(choice) if choice in labels else 0
                value = choices[idx]
                # Determine kwarg based on action
                kwarg = self._get_action_kwarg(action, value)
                action.handler(self.facade, self.current_block_id, **{kwarg: value})
                
        elif input_type == "number":
            """
            Handle number input dialog.
            
            CRITICAL: QInputDialog.getDouble() signature in PyQt6:
            getDouble(parent, title, label, value, min, max, decimals, flags, step)
            
            ️ WARNING: flags parameter (Qt.WindowType) is REQUIRED before step parameter.
            Do NOT omit flags or swap parameter order - this will cause runtime error:
            "argument 8 has unexpected type 'float'"
            
            Supported request parameters:
            - default: float (default value, read from settings manager)
            - min: float (minimum value, REQUIRED)
            - max: float (maximum value, REQUIRED)
            - decimals: int (decimal places, REQUIRED, default: 2)
            - increment_jump: float (step size for arrows, REQUIRED, preferred name)
            - step: float (step size, fallback if increment_jump not provided)
            
            See: AgentAssets/QUICK_ACTIONS_INPUT_DIALOGS.md for complete reference
            """
            # Get parameters with sensible defaults
            default = request.get("default", 0.5)
            min_val = request.get("min", 0.0)
            max_val = request.get("max", 1.0)
            decimals = request.get("decimals", 2)
            # Support both "step" and "increment_jump" for step size
            step = request.get("increment_jump", request.get("step", 0.1))
            
            # CRITICAL: Must pass flags parameter before step
            # QInputDialog.getDouble(parent, title, label, value, min, max, decimals, flags, step)
            from PyQt6.QtCore import Qt
            value, ok = QInputDialog.getDouble(
                self,                    # parent: QWidget
                title,                   # title: str
                "Value:",                # label: str
                default,                 # value: float
                min_val,                 # min: float
                max_val,                 # max: float
                decimals,                # decimals: int
                Qt.WindowType.Dialog,    # flags: Qt.WindowType (REQUIRED - do not omit!)
                step                     # step: float (step size for increment/decrement arrows)
            )
            if ok:
                action.handler(self.facade, self.current_block_id, value=value)
    
    def _subscribe_to_events(self):
        """Subscribe to domain events for auto-refresh"""
        # Subscribe to block update events - when block metadata changes,
        # refresh quick actions panel to show updated state
        self.facade.event_bus.subscribe("BlockUpdated", self._on_block_updated)
    
    def _on_block_updated(self, event):
        """
        Handle block update event for the currently selected block.
        
        Quick action buttons are static for a given block type -- they don't
        display settings values. Current values are read on demand when the
        user clicks a button and a dialog opens. Therefore a full panel
        rebuild on every BlockUpdated event is unnecessary and causes a
        visible red flash (the Delete/Reset State buttons use ACCENT_RED
        backgrounds and linger for one frame via deleteLater during rebuild).
        
        We only need to rebuild if the block type itself changed (which
        doesn't happen in normal usage).
        """
        updated_block_id = event.data.get('id')
        if updated_block_id and updated_block_id == self.current_block_id:
            Log.debug(f"ActionsPanel: Block {updated_block_id} updated (no rebuild needed, buttons are static)")
    
    def _get_action_kwarg(self, action, value) -> str:
        """Determine the keyword argument name for an action"""
        name_lower = action.name.lower()
        if "model" in name_lower:
            return "model"
        elif "device" in name_lower:
            return "device"
        elif "stem" in name_lower:
            return "stem"
        elif "format" in name_lower:
            return "fmt"
        elif "style" in name_lower:
            return "style"
        return "value"
    
    def _handle_confirmation(self, action, request: Dict[str, Any]):
        """Handle action that needs confirmation"""
        reply = QMessageBox.question(
            self, action.name,
            request.get("message", f"Are you sure you want to {action.name.lower()}?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            confirm_arg = request.get("confirm_arg", "confirmed")
            # Check if this is a delete action - use CommandBus
            if action.dangerous and "delete" in action.name.lower():
                cmd = DeleteBlockCommand(self.facade, self.current_block_id)
                self.facade.command_bus.execute(cmd)
                self.clear()
                self.block_deleted.emit()
            else:
                result = action.handler(self.facade, self.current_block_id, **{confirm_arg: True})
                if action.dangerous:
                    self.clear()
                    self.block_deleted.emit()
    
    def _add_fallback_actions(self, block):
        """Add basic actions when QuickAction system not available"""
        # Execute
        self._add_action_button(
            "Execute Block",
            f"Run this {block.type} block",
            self._on_execute_block,
            primary=True
        )
        
        # Open Panel (if available)
        from ui.qt_gui.block_panels import is_panel_registered
        if is_panel_registered(block.type):
            self._add_action_button(
                "Open Panel",
                f"Open configuration panel",
                self._on_open_panel
            )
        
        # Connect/Disconnect actions
        self._add_separator()
        self._add_action_button(
            "Connect...",
            "Create a connection from this block",
            self._on_connect_block
        )
        self._add_action_button(
            "Disconnect...",
            "Disconnect connections from this block",
            self._on_disconnect_block
        )
        
        self._add_separator()
        
        # Rename
        self._add_action_button(
            "Rename",
            "Change the block name",
            self._on_rename_block
        )
        
        # Delete
        self._add_action_button(
            "Delete",
            "Remove this block",
            self._on_delete_block,
            dangerous=True
        )
    
    def _add_action_button(self, text: str, tooltip: str, callback, primary: bool = False, dangerous: bool = False):
        """Add an action button (fallback method)"""
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.clicked.connect(callback)
        
        if dangerous:
            btn.setStyleSheet(StyleFactory.button("danger"))
        elif primary:
            btn.setStyleSheet(StyleFactory.button("primary"))
        else:
            btn.setStyleSheet(StyleFactory.button())
        
        self.actions_layout.insertWidget(self.actions_layout.count() - 1, btn)
    
    def _add_section_label(self, text: str):
        """Add a section label"""
        label = QLabel(text)
        label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY.name()};
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding-top: 4px;
            padding-bottom: 2px;
        """)
        self.actions_layout.insertWidget(self.actions_layout.count() - 1, label)
    
    def _add_separator(self):
        """Add a visual separator"""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {Colors.BORDER.name()};")
        line.setFixedHeight(1)
        self.actions_layout.insertWidget(self.actions_layout.count() - 1, line)
    
    def _clear_actions(self):
        """Clear all action buttons but keep placeholder"""
        while self.actions_layout.count() > 1:  # Keep the stretch
            item = self.actions_layout.takeAt(0)
            widget = item.widget()
            if widget and widget != self.placeholder:
                widget.hide()  # Hide immediately to prevent visual flash
                widget.deleteLater()
            elif widget == self.placeholder:
                pass
    
    def _show_placeholder(self, message: str = "Select a block to see quick actions"):
        """Show placeholder message"""
        self._clear_actions()
        
        if not self.placeholder or not self.placeholder.parent():
            self.placeholder = QLabel(message)
            self.placeholder.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: 20px;")
            self.placeholder.setWordWrap(True)
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.placeholder.setText(message)
        
        if self.placeholder not in [self.actions_layout.itemAt(i).widget() for i in range(self.actions_layout.count())]:
            self.actions_layout.insertWidget(0, self.placeholder)
        self.placeholder.show()
    
    def clear(self):
        """Clear the panel"""
        self.current_block_id = None
        self.current_block_type = None
        self._show_placeholder()
    
    # Fallback action handlers
    def _on_execute_block(self):
        """Execute the current block"""
        if self.current_block_id:
            self.execute_block_requested.emit(self.current_block_id)
    
    def _on_open_panel(self):
        """Open panel for current block"""
        if self.current_block_id:
            self.panel_requested.emit(self.current_block_id)
    
    def _on_connect_block(self):
        """Open connect dialog for current block"""
        from ui.qt_gui.connection.connection_dialog import ConnectionDialog
        
        if self.current_block_id:
            if ConnectionDialog.show_dialog(self.facade, self.current_block_id, parent=self):
                # Refresh to show new connection
                self.show_block_actions(self.current_block_id)
    
    def _on_disconnect_block(self):
        """Open disconnect dialog for current block"""
        from ui.qt_gui.connection.disconnect_dialog import DisconnectDialog
        
        if self.current_block_id:
            if DisconnectDialog.show_dialog(self.facade, self.current_block_id, parent=self):
                # Refresh to show updated connections
                self.show_block_actions(self.current_block_id)
    
    def _on_rename_block(self):
        """Rename the current block"""
        if not self.current_block_id:
            return
        
        result = self.facade.describe_block(self.current_block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        new_name, ok = QInputDialog.getText(
            self, "Rename Block", "Enter new name:",
            text=block.name
        )
        
        if ok and new_name and new_name != block.name:
            # Use CommandBus for undoable command
            cmd = RenameBlockCommand(self.facade, self.current_block_id, new_name)
            self.facade.command_bus.execute(cmd)
            Log.info(f"Renamed block to: {new_name}")
            self.show_block_actions(self.current_block_id)
    
    def _on_delete_block(self):
        """Delete the current block"""
        if not self.current_block_id:
            return
        
        result = self.facade.describe_block(self.current_block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        
        # Use CommandBus for undoable command
        cmd = DeleteBlockCommand(self.facade, self.current_block_id)
        self.facade.command_bus.execute(cmd)
        Log.info(f"Deleted block: {block.name}")
        self.clear()
        self.block_deleted.emit()

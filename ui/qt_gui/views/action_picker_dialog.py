"""
Action Picker Dialog

Dialog for selecting a block and action to add to an action set.
"""
from typing import Dict, Any, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QDialogButtonBox
)
from PyQt6.QtCore import Qt

from src.application.api.application_facade import ApplicationFacade
from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory


class ActionPickerDialog(ThemeAwareMixin, QDialog):
    """
    Dialog for picking a block and action.
    
    Shows available blocks and their actions in dropdowns.
    """
    
    def __init__(
        self,
        facade: ApplicationFacade,
        project_id: str,
        actions_by_block: Dict[str, List[Dict[str, Any]]],
        parent=None
    ):
        super().__init__(parent)
        self.facade = facade
        self.project_id = project_id
        self.actions_by_block = actions_by_block
        
        self.selected_block_id = None
        self.selected_block_name = None
        self.selected_action_name = None
        self.selected_action_description = None
        self.selected_action_data = None
        
        self._setup_ui()
        self._init_theme_aware()
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.SM)
        
        # Title
        title = QLabel("Add Action")
        title.setFont(Typography.heading_font())
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Block selection
        block_layout = QHBoxLayout()
        block_layout.addWidget(QLabel("Block:"))
        
        self.block_combo = QComboBox()
        self.block_combo.setStyleSheet(StyleFactory.combo())
        self.block_combo.currentIndexChanged.connect(self._on_block_changed)
        block_layout.addWidget(self.block_combo, 1)
        
        layout.addLayout(block_layout)
        
        # Action selection
        action_layout = QHBoxLayout()
        action_layout.addWidget(QLabel("Action:"))
        
        self.action_combo = QComboBox()
        self.action_combo.setStyleSheet(StyleFactory.combo())
        self.action_combo.currentIndexChanged.connect(self._on_action_changed)
        action_layout.addWidget(self.action_combo, 1)
        
        layout.addLayout(action_layout)
        
        # Description
        self.description_label = QLabel("")
        self.description_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)
        
        layout.addStretch()
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Populate blocks
        self._populate_blocks()
    
    def _populate_blocks(self):
        """Populate the block combo box"""
        self.block_combo.clear()
        
        for block_id, block_data in self.actions_by_block.items():
            # Extract block info
            if isinstance(block_data, dict) and "actions" in block_data:
                block_name = block_data.get("block_name", f"Block {block_id[:8]}...")
                block_type = block_data.get("block_type", "Unknown")
            else:
                block_name = f"Block {block_id[:8]}..."
                block_type = "Unknown"
            
            display_name = f"{block_name} ({block_type})"
            self.block_combo.addItem(display_name, block_id)
        
        if self.block_combo.count() > 0:
            self.block_combo.setCurrentIndex(0)
            self._on_block_changed(0)
    
    def _on_block_changed(self, index: int):
        """Handle block selection change"""
        if index < 0:
            return
        
        block_id = self.block_combo.itemData(index)
        if not block_id:
            return
        
        self.selected_block_id = block_id
        
        # Get block data
        block_data = self.actions_by_block.get(block_id, {})
        if isinstance(block_data, dict) and "actions" in block_data:
            self.selected_block_name = block_data.get("block_name", f"Block {block_id[:8]}...")
            actions = block_data.get("actions", [])
        else:
            self.selected_block_name = f"Block {block_id[:8]}..."
            actions = block_data if isinstance(block_data, list) else []
        
        # Populate actions
        self.action_combo.clear()
        for action in actions:
            action_name = action.get("name", "Unknown")
            action_desc = action.get("description", "")
            display_name = f"{action_name}"
            self.action_combo.addItem(display_name, (action_name, action_desc, action))
        
        if self.action_combo.count() > 0:
            self.action_combo.setCurrentIndex(0)
            self._on_action_changed(0)
    
    def _on_action_changed(self, index: int):
        """Handle action selection change"""
        if index < 0:
            return
        
        action_data = self.action_combo.itemData(index)
        if not action_data:
            return
        
        action_name, action_desc, action = action_data
        self.selected_action_name = action_name
        self.selected_action_description = action_desc
        self.selected_action_data = action
        
        # Update description
        if action_desc:
            self.description_label.setText(action_desc)
        else:
            self.description_label.setText("")
    
    def get_selection(self) -> Tuple[str, str, str, str, Dict[str, Any]]:
        """
        Get the selected block and action.
        
        Returns:
            Tuple of (block_id, block_name, action_name, action_description, action_data)
        """
        return (
            self.selected_block_id or "",
            self.selected_block_name or "",
            self.selected_action_name or "",
            self.selected_action_description or "",
            self.selected_action_data or {}
        )
    


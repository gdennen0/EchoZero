"""
Setlist Action Configuration Panel

Professional UI for configuring which actions to apply during setlist processing.
Shows all blocks in current project and their available actions.
"""
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox,
    QGroupBox, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from src.application.api.application_facade import ApplicationFacade
from src.shared.domain.value_objects.execution_strategy import ExecutionStrategy
from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory
from src.utils.message import Log


class SetlistActionConfigPanel(ThemeAwareMixin, QWidget):
    """
    Action configuration panel for setlist processing.
    
    Shows all blocks in current project and their available actions.
    Allows users to enable/disable actions and configure parameters.
    """
    
    # Signal emitted when configuration changes
    configuration_changed = pyqtSignal()
    
    def __init__(self, facade: ApplicationFacade, parent=None):
        super().__init__(parent)
        self.facade = facade
        self.current_project_id: Optional[str] = None
        self.actions_by_block: Dict[str, List[Dict[str, Any]]] = {}
        self.enabled_actions: Dict[str, Dict[str, Dict[str, Any]]] = {}  # block_id -> {action_name: action_args}
        self.execution_strategy: ExecutionStrategy = ExecutionStrategy.default()
        
        self._setup_ui()
        self._init_theme_aware()
    
    def _setup_ui(self):
        """Setup the UI layout"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        main_layout.setSpacing(Spacing.SM)
        
        # Title
        title = QLabel("Action Configuration")
        title.setFont(Typography.heading_font())
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 16px; font-weight: bold;")
        main_layout.addWidget(title)
        
        # Description
        description = QLabel(
            "Configure which actions to apply to blocks during setlist processing. "
            "Actions allow you to modify block settings per song (e.g., change audio file path, adjust thresholds). "
            "The 'Set Audio File' action is automatically applied to the audio input block."
        )
        description.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        description.setWordWrap(True)
        main_layout.addWidget(description)
        
        # Dynamic values help
        help_label = QLabel(
            "<b>Dynamic Values:</b> Use placeholders in action parameters: "
            "<code>{song_audio_path}</code> (full path), "
            "<code>{song_name}</code> (name without extension), "
            "<code>{song_full_name}</code> (name with extension), "
            "<code>{song_index}</code> (order in setlist), "
            "<code>{setlist_id}</code> (setlist ID)"
        )
        help_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10px; background-color: {Colors.BG_MEDIUM.name()}; padding: 6px; border-radius: {border_radius(4)};")
        help_label.setWordWrap(True)
        main_layout.addWidget(help_label)
        
        # Execution Strategy Section
        strategy_group = self._create_strategy_section()
        main_layout.addWidget(strategy_group)
        
        # Strategy description
        strategy_desc = QLabel(
            "<b>Full Re-execution:</b> Run all blocks from scratch (recommended for most cases).<br/>"
            "<b>Actions Only:</b> Only apply actions, don't re-execute blocks (faster, but may miss changes).<br/>"
            "<b>Hybrid:</b> Re-execute some blocks, apply actions to others (advanced)."
        )
        strategy_desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10px; padding: 4px;")
        strategy_desc.setWordWrap(True)
        main_layout.addWidget(strategy_desc)
        
        # Blocks & Actions Tree
        actions_group = self._create_actions_section()
        main_layout.addWidget(actions_group, 1)  # Stretch
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(Spacing.SM)
        
        self.save_btn = QPushButton("Save Configuration")
        self.save_btn.setStyleSheet(StyleFactory.button("primary"))
        self.save_btn.clicked.connect(self._on_save)
        buttons_layout.addWidget(self.save_btn)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setStyleSheet(StyleFactory.button())
        self.reset_btn.clicked.connect(self._on_reset)
        buttons_layout.addWidget(self.reset_btn)
        
        buttons_layout.addStretch()
        
        main_layout.addLayout(buttons_layout)
    
    def _create_strategy_section(self) -> QGroupBox:
        """Create execution strategy selection section"""
        group = QGroupBox("Execution Strategy")
        group.setStyleSheet(StyleFactory.group_box())
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        layout.setSpacing(Spacing.XS)
        
        # Strategy dropdown
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Strategy:"))
        
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Full Re-execution", "Actions Only", "Hybrid"])
        self.strategy_combo.setCurrentIndex(0)  # Default: Full
        self.strategy_combo.setStyleSheet(StyleFactory.combo())
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo, 1)
        
        layout.addLayout(strategy_layout)
        
        # Strategy description
        self.strategy_desc = QLabel(
            "Full Re-execution: Execute all blocks from scratch (default, safest)"
        )
        self.strategy_desc.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        self.strategy_desc.setWordWrap(True)
        layout.addWidget(self.strategy_desc)
        
        return group
    
    def _create_actions_section(self) -> QGroupBox:
        """Create blocks and actions tree section"""
        group = QGroupBox("Blocks & Actions")
        group.setStyleSheet(StyleFactory.group_box())
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        layout.setSpacing(Spacing.XS)
        
        # Instructions
        instructions = QLabel(
            "All available actions are automatically enabled. "
            "Click 'Configure...' to set action parameters (e.g., file paths, thresholds)."
        )
        instructions.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Actions tree
        self.actions_tree = QTreeWidget()
        self.actions_tree.setHeaderLabels(["Action", "Description", "Configure"])
        self.actions_tree.setColumnWidth(0, 200)
        self.actions_tree.setColumnWidth(1, 300)
        self.actions_tree.setColumnWidth(2, 100)
        self.actions_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.actions_tree.setRootIsDecorated(True)
        self.actions_tree.setAlternatingRowColors(True)
        self.actions_tree.setStyleSheet(StyleFactory.tree())
        layout.addWidget(self.actions_tree)
        
        return group
    
    def _on_strategy_changed(self, index: int):
        """Handle execution strategy change"""
        strategies = ["full", "actions_only", "hybrid"]
        descriptions = [
            "Full Re-execution: Execute all blocks from scratch (default, safest)",
            "Actions Only: Only apply configured actions, don't re-execute blocks (faster, for minor tweaks)",
            "Hybrid: Re-execute some blocks, action-only for others (advanced, performance optimization)"
        ]
        
        # Create new ExecutionStrategy with updated mode
        from src.shared.domain.value_objects.execution_strategy import ExecutionStrategy
        self.execution_strategy = ExecutionStrategy(
            mode=strategies[index],
            re_execute_blocks=self.execution_strategy.re_execute_blocks,
            action_only_blocks=self.execution_strategy.action_only_blocks,
            skip_blocks=self.execution_strategy.skip_blocks
        )
        self.strategy_desc.setText(descriptions[index])
        self.configuration_changed.emit()
    
    def _on_save(self):
        """Save current configuration"""
        self.configuration_changed.emit()
        Log.info("SetlistActionConfigPanel: Configuration saved")
    
    def _on_reset(self):
        """Reset to default configuration"""
        self.execution_strategy = ExecutionStrategy.default()
        self.strategy_combo.setCurrentIndex(0)
        self.enabled_actions = {}
        self._refresh_actions_tree()
        self.configuration_changed.emit()
        Log.info("SetlistActionConfigPanel: Configuration reset to defaults")
    
    def load_project(self, project_id: str):
        """Load actions for a project"""
        self.current_project_id = project_id
        
        # Discover available actions
        result = self.facade.discover_setlist_actions(project_id)
        if result.success:
            self.actions_by_block = result.data
            self._refresh_actions_tree()
        else:
            Log.error(f"SetlistActionConfigPanel: Failed to discover actions: {result.message}")
            self.actions_by_block = {}
            self._refresh_actions_tree()
    
    def _refresh_actions_tree(self):
        """Refresh the actions tree with current blocks and actions"""
        self.actions_tree.clear()
        
        if not self.actions_by_block:
            # No blocks or no actions discovered
            item = QTreeWidgetItem(self.actions_tree)
            item.setText(0, "No blocks found")
            item.setText(1, "Add blocks to your project to see available actions")
            return
        
        # Group by block - now includes block name and type
        for block_id, block_data in self.actions_by_block.items():
            # Extract block info (new format includes block_name and block_type)
            if isinstance(block_data, dict) and "actions" in block_data:
                block_name = block_data.get("block_name", f"Block {block_id[:8]}...")
                block_type = block_data.get("block_type", "Unknown")
                actions = block_data.get("actions", [])
            else:
                # Fallback for old format (backward compatibility)
                block_name = f"Block {block_id[:8]}..."
                block_type = "Unknown"
                actions = block_data if isinstance(block_data, list) else []
            
            # Create block item with name and type
            block_item = QTreeWidgetItem(self.actions_tree)
            block_item.setText(0, f"{block_name} ({block_type})")
            block_item.setExpanded(True)
            
            for action in actions:
                action_item = QTreeWidgetItem(block_item)
                action_name = action.get("name", "Unknown")
                action_desc = action.get("description", "")
                
                # Action name as label (always enabled)
                action_item.setText(0, action_name)
                action_item.setText(1, action_desc)
                
                # Always enable action (add to enabled_actions if not already present)
                if block_id not in self.enabled_actions:
                    self.enabled_actions[block_id] = {}
                if action_name not in self.enabled_actions[block_id]:
                    # Initialize with empty args (can be configured)
                    self.enabled_actions[block_id][action_name] = {}
                
                # Configure button
                config_btn = QPushButton("Configure...")
                config_btn.setStyleSheet(StyleFactory.button("small"))
                config_btn.clicked.connect(
                    lambda checked, bid=block_id, aname=action_name, adata=action: self._on_configure_action(bid, aname, adata)
                )
                
                self.actions_tree.setItemWidget(action_item, 2, config_btn)
        
        self.actions_tree.expandAll()
    
    # Removed _on_action_toggled - actions are always enabled now
    
    def _on_configure_action(self, block_id: str, action_name: str, action_data: Dict[str, Any]):
        """Open action configuration dialog"""
        from ui.qt_gui.views.setlist_action_config_dialog import SetlistActionConfigDialog
        
        # Get current action args if configured
        current_args = {}
        if block_id in self.enabled_actions and action_name in self.enabled_actions[block_id]:
            current_args = self.enabled_actions[block_id][action_name]
        
        dialog = SetlistActionConfigDialog(
            self.facade,
            block_id=block_id,
            action_name=action_name,
            action_data=action_data,
            current_args=current_args,
            parent=self
        )
        
        if dialog.exec():
            # Save configured args
            if block_id not in self.enabled_actions:
                self.enabled_actions[block_id] = {}
            self.enabled_actions[block_id][action_name] = dialog.get_action_args()
            self.configuration_changed.emit()
    
    def get_configuration(self) -> tuple[ExecutionStrategy, Dict[str, Dict[str, Any]]]:
        """Get current configuration"""
        return self.execution_strategy, self.enabled_actions
    
    def set_configuration(self, strategy: ExecutionStrategy, actions: Dict[str, Dict[str, Any]]):
        """Set configuration"""
        self.execution_strategy = strategy
        self.enabled_actions = actions
        
        # Update UI
        strategy_index = {"full": 0, "actions_only": 1, "hybrid": 2}.get(strategy.mode, 0)
        self.strategy_combo.setCurrentIndex(strategy_index)
        self._refresh_actions_tree()
    
    # Style helpers
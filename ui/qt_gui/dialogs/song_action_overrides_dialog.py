"""
Song Action Overrides Dialog

Allows configuring per-song pre-actions and post-actions that run
before and after the main action set during setlist processing.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QGroupBox, QAbstractItemView, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt

from src.application.api.application_facade import ApplicationFacade
from src.application.blocks.quick_actions import get_quick_actions
from src.features.setlists.domain import SetlistSong
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from ui.qt_gui.style_factory import StyleFactory
from src.utils.message import Log


class SongActionOverridesDialog(QDialog):
    """
    Dialog for configuring per-song action overrides.
    
    Allows adding pre-actions (run before main action set) and
    post-actions (run after main action set) for a specific song.
    """
    
    def __init__(
        self,
        facade: ApplicationFacade,
        song: SetlistSong,
        actions_by_block: Dict[str, Dict[str, Any]],
        parent=None
    ):
        super().__init__(parent)
        self.facade = facade
        self.song = song
        self.actions_by_block = actions_by_block
        
        # Load existing overrides
        overrides = song.action_overrides or {}
        self.pre_actions: List[Dict[str, Any]] = list(overrides.get("pre_actions", []))
        self.post_actions: List[Dict[str, Any]] = list(overrides.get("post_actions", []))
        
        self.setWindowTitle(f"Song Actions: {Path(song.audio_path).name}")
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)
        
        self._setup_ui()
        self._load_tables()
    
    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        
        # Header
        header = QLabel(f"Configure actions for: {Path(self.song.audio_path).name}")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: bold; font-size: 13px;")
        layout.addWidget(header)
        
        info = QLabel(
            "Pre-actions run before the main action set. "
            "Post-actions run after. These are specific to this song only."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        layout.addWidget(info)
        
        # Pre-Actions Section
        pre_group = QGroupBox("Pre-Song Actions")
        pre_group.setStyleSheet(StyleFactory.group_box())
        pre_layout = QVBoxLayout(pre_group)
        
        self.pre_table = self._create_action_table()
        pre_layout.addWidget(self.pre_table)
        
        pre_btn_layout = QHBoxLayout()
        self.add_pre_btn = QPushButton("+ Add Pre-Action")
        self.add_pre_btn.setStyleSheet(StyleFactory.button())
        self.add_pre_btn.clicked.connect(lambda: self._add_action("pre"))
        pre_btn_layout.addWidget(self.add_pre_btn)
        
        self.remove_pre_btn = QPushButton("Remove Selected")
        self.remove_pre_btn.setStyleSheet(StyleFactory.button("danger"))
        self.remove_pre_btn.clicked.connect(lambda: self._remove_action("pre"))
        self.remove_pre_btn.setEnabled(False)
        pre_btn_layout.addWidget(self.remove_pre_btn)
        pre_btn_layout.addStretch()
        pre_layout.addLayout(pre_btn_layout)
        
        layout.addWidget(pre_group)
        
        # Post-Actions Section
        post_group = QGroupBox("Post-Song Actions")
        post_group.setStyleSheet(StyleFactory.group_box())
        post_layout = QVBoxLayout(post_group)
        
        self.post_table = self._create_action_table()
        post_layout.addWidget(self.post_table)
        
        post_btn_layout = QHBoxLayout()
        self.add_post_btn = QPushButton("+ Add Post-Action")
        self.add_post_btn.setStyleSheet(StyleFactory.button())
        self.add_post_btn.clicked.connect(lambda: self._add_action("post"))
        post_btn_layout.addWidget(self.add_post_btn)
        
        self.remove_post_btn = QPushButton("Remove Selected")
        self.remove_post_btn.setStyleSheet(StyleFactory.button("danger"))
        self.remove_post_btn.clicked.connect(lambda: self._remove_action("post"))
        self.remove_post_btn.setEnabled(False)
        post_btn_layout.addWidget(self.remove_post_btn)
        post_btn_layout.addStretch()
        post_layout.addLayout(post_btn_layout)
        
        layout.addWidget(post_group)
        
        # Connect selection changed signals
        self.pre_table.itemSelectionChanged.connect(
            lambda: self.remove_pre_btn.setEnabled(bool(self.pre_table.selectedItems()))
        )
        self.post_table.itemSelectionChanged.connect(
            lambda: self.remove_post_btn.setEnabled(bool(self.post_table.selectedItems()))
        )
        
        # Dialog buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet(StyleFactory.button("danger"))
        clear_btn.clicked.connect(self._clear_all)
        btn_layout.addWidget(clear_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(StyleFactory.button())
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet(StyleFactory.button("primary"))
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
    
    def _create_action_table(self) -> QTableWidget:
        """Create a table for displaying actions."""
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Block", "Action", "Parameters"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.setColumnWidth(0, 160)
        table.setColumnWidth(1, 160)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setMaximumHeight(150)
        table.setStyleSheet(StyleFactory.table())
        return table
    
    def _load_tables(self):
        """Load existing actions into tables."""
        self._populate_table(self.pre_table, self.pre_actions)
        self._populate_table(self.post_table, self.post_actions)
    
    def _populate_table(self, table: QTableWidget, actions: List[Dict[str, Any]]):
        """Populate a table from action list."""
        table.setRowCount(len(actions))
        for row, action in enumerate(actions):
            block_name = action.get("block_name", "Project")
            action_name = action.get("action_name", "Unknown")
            args = action.get("action_args", {})
            args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else "(none)"
            
            table.setItem(row, 0, QTableWidgetItem(block_name))
            table.setItem(row, 1, QTableWidgetItem(action_name))
            table.setItem(row, 2, QTableWidgetItem(args_str))
    
    def _add_action(self, phase: str):
        """Add a new action via inline combo selection."""
        # Use the AddActionDialog if available, otherwise inline
        try:
            from ui.qt_gui.views.add_action_dialog import AddActionDialog
            dialog = AddActionDialog(
                self.facade,
                self.facade.current_project_id,
                self.actions_by_block,
                parent=self
            )
            
            if dialog.exec():
                action_item = dialog.get_action_item()
                if action_item:
                    action_dict = {
                        "action_type": action_item.action_type,
                        "block_id": action_item.block_id,
                        "block_name": action_item.block_name,
                        "action_name": action_item.action_name,
                        "action_args": action_item.action_args or {},
                    }
                    
                    if phase == "pre":
                        self.pre_actions.append(action_dict)
                        self._populate_table(self.pre_table, self.pre_actions)
                    else:
                        self.post_actions.append(action_dict)
                        self._populate_table(self.post_table, self.post_actions)
        except ImportError:
            QMessageBox.warning(self, "Error", "AddActionDialog not available")
    
    def _remove_action(self, phase: str):
        """Remove selected action."""
        if phase == "pre":
            table = self.pre_table
            actions = self.pre_actions
        else:
            table = self.post_table
            actions = self.post_actions
        
        selected = table.selectedItems()
        if not selected:
            return
        
        row = selected[0].row()
        if 0 <= row < len(actions):
            actions.pop(row)
            self._populate_table(table, actions)
    
    def _clear_all(self):
        """Clear all overrides."""
        self.pre_actions.clear()
        self.post_actions.clear()
        self._populate_table(self.pre_table, self.pre_actions)
        self._populate_table(self.post_table, self.post_actions)
    
    def get_overrides(self) -> Dict[str, Any]:
        """
        Get the configured overrides as a dict suitable for SetlistSong.action_overrides.
        
        Returns:
            Dict with "pre_actions" and "post_actions" lists.
            Returns empty dict if no overrides are configured.
        """
        result = {}
        if self.pre_actions:
            result["pre_actions"] = self.pre_actions
        if self.post_actions:
            result["post_actions"] = self.post_actions
        return result
    
    # -- Styles --
    
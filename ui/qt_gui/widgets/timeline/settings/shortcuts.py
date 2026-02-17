"""
Keyboard Shortcuts Settings Dialog

Allows users to configure keyboard shortcuts for timeline operations.
"""
from typing import Dict, Optional, Callable
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QLineEdit, QKeySequenceEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QKeyEvent

from ..core.style import TimelineStyle as Colors
from ..logging import TimelineLog as Log


class ShortcutKeySequenceEdit(QKeySequenceEdit):
    """Key sequence editor that captures key combinations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumSequenceLength(1)  # Only single key combinations
    
    def keyPressEvent(self, event: QKeyEvent):
        """Capture key press and build sequence."""
        key = event.key()
        modifiers = event.modifiers()
        
        # Build sequence string
        parts = []
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            parts.append("Ctrl")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            parts.append("Alt")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            parts.append("Shift")
        if modifiers & Qt.KeyboardModifier.MetaModifier:
            parts.append("Meta")
        
        # Add key name
        key_name = QKeySequence(key).toString()
        if key_name:
            parts.append(key_name)
        
        if parts:
            sequence_str = "+".join(parts)
            # Set the sequence
            sequence = QKeySequence.fromString(sequence_str)
            if sequence.isEmpty():
                # Fallback: just use the key
                sequence = QKeySequence(key)
            self.setKeySequence(sequence)
        
        event.accept()


class ShortcutsSettingsDialog(QDialog):
    """
    Dialog for configuring keyboard shortcuts.
    
    Shows a table of available shortcuts with editable key sequences.
    """
    
    shortcuts_changed = pyqtSignal(dict)  # Emitted when shortcuts are saved
    
    # Default shortcuts
    DEFAULT_SHORTCUTS = {
        "Move Event Left": "Key_Left",
        "Move Event Right": "Key_Right",
        "Move Event Up Layer": "Ctrl+Key_Up",
        "Move Event Down Layer": "Ctrl+Key_Down",
    }
    
    def __init__(self, current_shortcuts: Optional[Dict[str, str]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        self._current_shortcuts = current_shortcuts or {}
        self._editors: Dict[str, ShortcutKeySequenceEdit] = {}
        
        self._setup_ui()
        self._load_shortcuts()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Title
        title = QLabel("Configure Keyboard Shortcuts")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Click on a shortcut field and press the key combination you want to use.\n"
            "Use Ctrl, Alt, Shift, or Meta as modifiers if desired."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: 8px;")
        layout.addWidget(instructions)
        
        # Table
        self._table = QTableWidget(len(self.DEFAULT_SHORTCUTS), 2)
        self._table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save_shortcuts)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
    
    def _load_shortcuts(self):
        """Load current shortcuts into the table."""
        row = 0
        for action_name, default_shortcut in self.DEFAULT_SHORTCUTS.items():
            # Action name
            action_item = QTableWidgetItem(action_name)
            action_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(row, 0, action_item)
            
            # Shortcut editor
            current_shortcut = self._current_shortcuts.get(action_name, default_shortcut)
            editor = ShortcutKeySequenceEdit()
            
            # Parse and set the shortcut
            try:
                # Convert our format (e.g., "Key_Left", "Ctrl+Key_Up") to QKeySequence format
                # Remove "Key_" prefix and try to parse
                shortcut_str = current_shortcut.replace("Key_", "")
                sequence = QKeySequence.fromString(shortcut_str)
                if sequence.isEmpty():
                    # Fallback: try the original string
                    sequence = QKeySequence.fromString(current_shortcut)
                editor.setKeySequence(sequence)
            except Exception as e:
                Log.warning(f"Failed to parse shortcut '{current_shortcut}': {e}")
                # Try default
                default_str = default_shortcut.replace("Key_", "")
                editor.setKeySequence(QKeySequence.fromString(default_str))
            
            self._table.setCellWidget(row, 1, editor)
            self._editors[action_name] = editor
            
            row += 1
    
    def _reset_to_defaults(self):
        """Reset all shortcuts to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Shortcuts",
            "Reset all shortcuts to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            for action_name, default_shortcut in self.DEFAULT_SHORTCUTS.items():
                editor = self._editors.get(action_name)
                if editor:
                    try:
                        # Convert default format to QKeySequence format
                        default_str = default_shortcut.replace("Key_", "")
                        sequence = QKeySequence.fromString(default_str)
                        editor.setKeySequence(sequence)
                    except Exception as e:
                        Log.warning(f"Failed to reset shortcut '{action_name}': {e}")
    
    def _save_shortcuts(self):
        """Save shortcuts and emit signal."""
        shortcuts = {}
        
        for action_name, editor in self._editors.items():
            sequence = editor.keySequence()
            if not sequence.isEmpty():
                # Convert to string representation
                shortcut_str = sequence.toString()
                # Convert back to our format (e.g., "Ctrl+Left" -> "Ctrl+Key_Left")
                # For now, store as-is and handle conversion in the view
                shortcuts[action_name] = shortcut_str
            else:
                # Use default
                shortcuts[action_name] = self.DEFAULT_SHORTCUTS[action_name]
        
        # Validate no duplicates
        values = list(shortcuts.values())
        if len(values) != len(set(values)):
            QMessageBox.warning(
                self,
                "Duplicate Shortcuts",
                "Some shortcuts are assigned to multiple actions. Please use unique shortcuts."
            )
            return
        
        self._current_shortcuts = shortcuts
        self.shortcuts_changed.emit(shortcuts)
        self.accept()
    
    def get_shortcuts(self) -> Dict[str, str]:
        """Get the current shortcut mappings."""
        return self._current_shortcuts.copy()


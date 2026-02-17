"""
Command History Dialog

Displays the undo/redo command history from the QUndoStack.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QSplitter, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QUndoStack, QColor, QBrush

from ui.qt_gui.design_system import Colors, Spacing, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory


class CommandHistoryDialog(ThemeAwareMixin, QDialog):
    """
    Dialog showing command history with ability to undo/redo to any point.
    
    Features:
    - Shows all commands in the undo stack
    - Current position indicator
    - Click to undo/redo to specific point
    - Clear history button
    """
    
    def __init__(self, undo_stack: QUndoStack, parent=None):
        super().__init__(parent)
        self.undo_stack = undo_stack
        
        self.setWindowTitle("Command History")
        self.setMinimumSize(350, 400)
        self.setModal(False)  # Non-modal so user can keep it open
        
        self._setup_ui()
        self._connect_signals()
        self._refresh_list()
        self._init_theme_aware()
    
    def _setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        
        # Header
        header = QLabel("Command History")
        header.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {Colors.TEXT_PRIMARY.name()};
            padding-bottom: 8px;
        """)
        layout.addWidget(header)
        
        # Info label
        self.info_label = QLabel()
        self.info_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY.name()};
            font-size: 11px;
        """)
        layout.addWidget(self.info_label)
        
        # Command list
        self.command_list = QListWidget()
        self.command_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 6px 8px;
                border-radius: {border_radius(3)};
            }}
            QListWidget::item:selected {{
                background-color: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {Colors.HOVER.name()};
            }}
        """)
        self.command_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.command_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(Spacing.SM)
        
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self._on_undo)
        self.btn_undo.setStyleSheet(StyleFactory.button())
        button_layout.addWidget(self.btn_undo)
        
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.clicked.connect(self._on_redo)
        self.btn_redo.setStyleSheet(StyleFactory.button())
        button_layout.addWidget(self.btn_redo)
        
        button_layout.addStretch()
        
        self.btn_clear = QPushButton("Clear History")
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_clear.setStyleSheet(StyleFactory.button("danger"))
        button_layout.addWidget(self.btn_clear)
        
        layout.addLayout(button_layout)
        
        # Apply dialog styling
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_MEDIUM.name()};
            }}
        """)
    
    def _connect_signals(self):
        """Connect to undo stack signals"""
        self.undo_stack.indexChanged.connect(self._refresh_list)
        self.undo_stack.cleanChanged.connect(self._refresh_list)
    
    def _refresh_list(self):
        """Refresh the command list from undo stack"""
        self.command_list.clear()
        
        count = self.undo_stack.count()
        current_index = self.undo_stack.index()
        
        # Update info label
        undo_count = current_index
        redo_count = count - current_index
        self.info_label.setText(f"{count} commands | {undo_count} undo available | {redo_count} redo available")
        
        # Add "Initial State" marker
        initial_item = QListWidgetItem("(Initial State)")
        initial_item.setData(Qt.ItemDataRole.UserRole, -1)
        if current_index == 0:
            initial_item.setBackground(QBrush(Colors.ACCENT_BLUE.darker(150)))
            initial_item.setForeground(QBrush(QColor("white")))
        else:
            initial_item.setForeground(QBrush(Colors.TEXT_SECONDARY))
        self.command_list.addItem(initial_item)
        
        # Add commands
        for i in range(count):
            command = self.undo_stack.command(i)
            text = command.text() if command else f"Command {i+1}"
            
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            
            # Highlight current position
            if i < current_index:
                # Commands that have been executed (can be undone)
                item.setForeground(QBrush(Colors.TEXT_PRIMARY))
            elif i == current_index - 1:
                # Most recent command (current position marker)
                pass  # Normal styling
            else:
                # Commands that have been undone (can be redone)
                item.setForeground(QBrush(Colors.TEXT_DISABLED))
            
            # Mark current position with arrow
            if i == current_index - 1:
                item.setText(f"-> {text}")
                item.setBackground(QBrush(Colors.ACCENT_BLUE.darker(150)))
                item.setForeground(QBrush(QColor("white")))
            
            self.command_list.addItem(item)
        
        # Update button states
        self.btn_undo.setEnabled(self.undo_stack.canUndo())
        self.btn_redo.setEnabled(self.undo_stack.canRedo())
        self.btn_clear.setEnabled(count > 0)
    
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle clicking on a command to undo/redo to that point"""
        target_index = item.data(Qt.ItemDataRole.UserRole)
        if target_index is None:
            return
        
        current_index = self.undo_stack.index()
        
        # For initial state, target is index 0
        if target_index == -1:
            target_index = 0
        else:
            # We want to be AFTER this command, so +1
            target_index = target_index + 1
        
        # Undo or redo to reach target
        if target_index < current_index:
            # Need to undo
            while self.undo_stack.index() > target_index and self.undo_stack.canUndo():
                self.undo_stack.undo()
        elif target_index > current_index:
            # Need to redo
            while self.undo_stack.index() < target_index and self.undo_stack.canRedo():
                self.undo_stack.redo()
    
    def _on_undo(self):
        """Undo one step"""
        if self.undo_stack.canUndo():
            self.undo_stack.undo()
    
    def _on_redo(self):
        """Redo one step"""
        if self.undo_stack.canRedo():
            self.undo_stack.redo()
    
    def _on_clear(self):
        """Clear all history"""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Clear all command history? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.undo_stack.clear()


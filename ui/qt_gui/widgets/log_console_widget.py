"""
Log Console Widget

A widget that displays log messages from the application's logging system.
Implements the add_output(level, message) interface expected by GUIConsoleHandler.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, 
    QPushButton, QCheckBox, QLabel, QApplication
)
from PyQt6.QtCore import Qt, QTimer, QThread
from PyQt6.QtGui import QTextCharFormat, QColor, QTextCursor

from ui.qt_gui.design_system import Colors, Spacing, Typography, border_radius


class LogConsoleWidget(QWidget):
    """
    Console widget that displays log messages with color coding by level.
    
    Features:
    - Color-coded log levels (DEBUG, INFO, WARNING, ERROR)
    - Auto-scroll to bottom (can be disabled)
    - Clear button
    - Level filtering checkboxes
    - Maximum line limit to prevent memory issues
    """
    
    # Maximum number of lines to keep in the console
    MAX_LINES = 10000
    
    # Color mapping for log levels
    LEVEL_COLORS = {
        "DEBUG": Colors.TEXT_DISABLED,
        "INFO": Colors.TEXT_PRIMARY,
        "WARNING": Colors.ACCENT_YELLOW,
        "ERROR": Colors.ACCENT_RED,
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._auto_scroll = True
        self._max_lines = self.MAX_LINES
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        layout.setSpacing(Spacing.SM)
        
        # Control bar
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(Spacing.SM)
        
        # Level filter checkboxes
        filter_label = QLabel("Show:")
        filter_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        controls_layout.addWidget(filter_label)
        
        self.filter_debug = QCheckBox("DEBUG")
        self.filter_debug.setChecked(True)
        self.filter_debug.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        controls_layout.addWidget(self.filter_debug)
        
        self.filter_info = QCheckBox("INFO")
        self.filter_info.setChecked(True)
        self.filter_info.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        controls_layout.addWidget(self.filter_info)
        
        self.filter_warning = QCheckBox("WARNING")
        self.filter_warning.setChecked(True)
        self.filter_warning.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        controls_layout.addWidget(self.filter_warning)
        
        self.filter_error = QCheckBox("ERROR")
        self.filter_error.setChecked(True)
        self.filter_error.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        controls_layout.addWidget(self.filter_error)
        
        controls_layout.addStretch()
        
        # Auto-scroll checkbox
        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        self.auto_scroll_checkbox.toggled.connect(self._on_auto_scroll_toggled)
        controls_layout.addWidget(self.auto_scroll_checkbox)
        
        # Clear button
        clear_button = QPushButton("Clear")
        clear_button.setMinimumWidth(60)
        clear_button.clicked.connect(self.clear)
        controls_layout.addWidget(clear_button)
        
        layout.addLayout(controls_layout)
        
        # Text area for log messages
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(Typography.mono_font())
        self.text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: {Spacing.SM}px;
            }}
        """)
        
        layout.addWidget(self.text_edit)
    
    def _on_auto_scroll_toggled(self, checked: bool):
        """Handle auto-scroll checkbox toggle"""
        self._auto_scroll = checked
        if checked:
            # Scroll to bottom when re-enabling
            self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
    
    def add_output(self, level: str, message: str):
        """
        Add a log message to the console.
        
        This method is called by GUIConsoleHandler to display log messages.
        Thread-safe: queues updates to the main thread if called from a background thread.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: The log message text
        """
        # Check if we're on the main thread
        app = QApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            # Not on main thread - queue the call to the main thread
            QTimer.singleShot(0, lambda: self._add_output_impl(level, message))
        else:
            # On main thread (or app not initialized yet) - execute directly
            self._add_output_impl(level, message)
    
    def _add_output_impl(self, level: str, message: str):
        """
        Internal implementation of add_output - always called on main thread.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: The log message text
        """
        # Check if this level should be displayed
        if not self._should_display_level(level):
            return
        
        # Get color for this level
        color = self.LEVEL_COLORS.get(level.upper(), Colors.TEXT_PRIMARY)
        
        # Create formatted text
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Format the message with color
        char_format = QTextCharFormat()
        char_format.setForeground(color)
        cursor.setCharFormat(char_format)
        
        # Add timestamp and message
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}\n"
        
        cursor.insertText(formatted_message)
        
        # Limit the number of lines to prevent memory issues
        self._limit_lines()
        
        # Auto-scroll to bottom if enabled
        if self._auto_scroll:
            self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.text_edit.ensureCursorVisible()
    
    def _should_display_level(self, level: str) -> bool:
        """Check if a log level should be displayed based on filters"""
        level_upper = level.upper()
        if level_upper == "DEBUG":
            return self.filter_debug.isChecked()
        elif level_upper == "INFO":
            return self.filter_info.isChecked()
        elif level_upper == "WARNING":
            return self.filter_warning.isChecked()
        elif level_upper == "ERROR":
            return self.filter_error.isChecked()
        return True  # Default to showing unknown levels
    
    def _limit_lines(self):
        """Limit the number of lines in the text edit to prevent memory issues"""
        document = self.text_edit.document()
        if document.blockCount() > self._max_lines:
            # Remove the oldest lines
            cursor = QTextCursor(document)
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            # Move forward by the number of lines to remove
            lines_to_remove = document.blockCount() - self._max_lines
            for _ in range(lines_to_remove):
                cursor.movePosition(QTextCursor.MoveOperation.Down, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
    
    def clear(self):
        """Clear all log messages from the console"""
        self.text_edit.clear()


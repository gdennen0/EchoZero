"""
Empty Workspace Widget

Displays a helpful message when no windows are docked to the main window.
Shows guidance for how to dock windows.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from ui.qt_gui.design_system import Colors, Spacing


class EmptyWorkspaceWidget(QWidget):
    """
    Widget shown in the central area when no docks are present.
    
    Provides visual guidance for users on how to dock windows.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI with helpful guidance."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.SM)
        
        # Unified message label with all content
        html_content = (
            "<div style='text-align: center;'>"
            f"<p style='font-size: 18px; font-weight: 500; color: {Colors.TEXT_PRIMARY.name()}; margin: 0 0 8px 0;'>"
            "Drop windows here to dock them"
            "</p>"
            f"<p style='font-size: 13px; color: {Colors.TEXT_SECONDARY.name()}; margin: 0 0 4px 0; line-height: 1.5;'>"
            "Drag floating windows to the edges to dock them.<br>"
            "Drop on another dock to create tabs."
            "</p>"
            f"<p style='font-size: 12px; color: {Colors.TEXT_DISABLED.name()}; margin: 8px 0 0 0; font-style: italic;'>"
            "Tip: Use Window menu to show/hide panels"
            "</p>"
            "</div>"
        )
        content_label = QLabel(html_content)
        content_label.setTextFormat(Qt.TextFormat.RichText)
        content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_label.setWordWrap(True)
        layout.addWidget(content_label)
        
        # Subtle background styling - no border, just a hint of distinction
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.BG_DARK.name()};
            }}
        """)
        
        # Set minimum size
        self.setMinimumSize(280, 150)

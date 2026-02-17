"""
Status Bar with Integrated Progress

A professional status bar that includes both messages and progress indicator.
Standard pattern used by VS Code, browsers, and other professional apps.
"""
from PyQt6.QtWidgets import (
    QStatusBar, QWidget, QHBoxLayout, QLabel, 
    QPushButton, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from ui.qt_gui.design_system import Colors, border_radius


class StatusBarProgress(QStatusBar):
    """
    Professional status bar with integrated progress indicator.
    
    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Ready                                    [▓▓▓░░░ 60%] [×]       │
    └─────────────────────────────────────────────────────────────────┘
    
    Features:
    - Left: Status message (always visible)
    - Right: Progress bar + percentage + cancel (visible during execution)
    - Fixed at bottom of window
    - Clean, professional look
    """
    
    cancel_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the status bar UI"""
        # Disable the size grip (removes the resize handle in corner)
        self.setSizeGripEnabled(False)
        
        # Set fixed height
        self.setFixedHeight(24)
        
        # Style the status bar
        self.setStyleSheet(f"""
            QStatusBar {{
                background-color: {Colors.BG_MEDIUM.name()};
                border-top: 1px solid {Colors.BORDER.name()};
            }}
            QStatusBar::item {{
                border: none;
            }}
        """)
        
        # Create progress widget container (right side)
        self.progress_widget = QWidget()
        self.progress_widget.setStyleSheet("background: transparent;")
        progress_layout = QHBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(0, 0, 8, 0)
        progress_layout.setSpacing(8)
        
        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY.name()}; 
            font-size: 11px;
            background: transparent;
        """)
        progress_layout.addWidget(self.progress_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(120)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self._set_progress_style(Colors.ACCENT_BLUE)
        progress_layout.addWidget(self.progress_bar)
        
        # Percentage label
        self.percent_label = QLabel("0%")
        self.percent_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY.name()}; 
            font-size: 11px;
            background: transparent;
        """)
        self.percent_label.setFixedWidth(32)
        progress_layout.addWidget(self.percent_label)
        
        # Cancel button
        self.btn_cancel = QPushButton("×")
        self.btn_cancel.setFixedSize(18, 18)
        self.btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_SECONDARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(9)};
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_RED.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border-color: {Colors.ACCENT_RED.name()};
            }}
        """)
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        progress_layout.addWidget(self.btn_cancel)
        
        # Add progress widget to status bar (permanent, right side)
        self.addPermanentWidget(self.progress_widget)
        self.progress_widget.hide()
        
        # Set default message
        self.showMessage("Ready")
    
    def _set_progress_style(self, color):
        """Set progress bar color"""
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Colors.BG_DARK.name()};
                border: none;
                border-radius: {border_radius(3)};
            }}
            QProgressBar::chunk {{
                background-color: {color.name()};
                border-radius: {border_radius(3)};
            }}
        """)
    
    def show_progress(self, message: str, current: int = 0, total: int = 100):
        """
        Show progress indicator.
        
        Args:
            message: Status message to display
            current: Current step
            total: Total steps
        """
        # Update status message
        self.showMessage(message)
        
        # Update progress label  
        self.progress_label.setText(message[:30] + "..." if len(message) > 30 else message)
        
        # Update progress bar
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.percent_label.setText(f"{percentage}%")
        else:
            self.progress_bar.setValue(0)
            self.percent_label.setText("0%")
        
        # Reset to normal style
        self.progress_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY.name()}; 
            font-size: 11px;
            background: transparent;
        """)
        self._set_progress_style(Colors.ACCENT_BLUE)
        
        # Show progress widget
        self.progress_widget.show()
    
    def hide_progress(self):
        """Hide progress indicator"""
        self.progress_widget.hide()
        self.showMessage("Ready")
    
    def set_error(self, message: str):
        """Show error state"""
        self.showMessage(f"Error: {message}")
        self.progress_label.setText("Error")
        self.progress_label.setStyleSheet(f"""
            color: {Colors.ACCENT_RED.name()}; 
            font-size: 11px;
            background: transparent;
        """)
        self._set_progress_style(Colors.ACCENT_RED)
    
    def set_filter_error(self, block_name: str, port_name: str):
        """Show filter error state with actionable message"""
        message = f"Filter error in {block_name}.{port_name} - All items filtered out"
        self.showMessage(message)
        self.progress_label.setText("Filter Error")
        self.progress_label.setStyleSheet(f"""
            color: {Colors.ACCENT_RED.name()}; 
            font-size: 11px;
            background: transparent;
        """)
        self._set_progress_style(Colors.ACCENT_RED)
    
    def set_complete(self, blocks_count: int):
        """Show completion state"""
        self.showMessage(f"Completed! ({blocks_count} blocks)")
        self.progress_bar.setValue(100)
        self.percent_label.setText("100%")
        self.progress_label.setText("Complete")
        self._set_progress_style(Colors.ACCENT_GREEN)
        
        # Auto-hide after 2 seconds
        QTimer.singleShot(2000, self.hide_progress)


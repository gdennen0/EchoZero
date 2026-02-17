"""
Splash Screen Widget

Displays application loading progress with module and sub-component information.
"""
from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette

from src.application.bootstrap_loading_progress import LoadingProgressCallback
from ui.qt_gui.design_system import border_radius, Colors


class SplashScreenSignals(QObject):
    """Qt signals for splash screen updates (thread-safe)"""
    module_started = pyqtSignal(str, int)  # module_name, total_steps
    step_updated = pyqtSignal(str, str, int, int)  # module_name, step_name, step, total
    module_completed = pyqtSignal(str)  # module_name
    error_occurred = pyqtSignal(str, str)  # module_name, error_message


class SplashScreen(QWidget):
    """
    Splash screen widget that displays loading progress.
    
    Shows current module, sub-components, and overall progress bar.
    Implements LoadingProgressCallback protocol for progress updates.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize splash screen.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        # Setup window properties
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        
        # Setup signals for thread-safe updates
        self.signals = SplashScreenSignals()
        self.signals.module_started.connect(self._on_module_started)
        self.signals.step_updated.connect(self._on_step_updated)
        self.signals.module_completed.connect(self._on_module_completed)
        self.signals.error_occurred.connect(self._on_error_occurred)
        
        # Current state
        self.current_module: Optional[str] = None
        self.current_step: Optional[str] = None
        self.module_progress = 0
        self.module_total_steps = 1
        self.overall_progress = 0.0
        
        # Setup UI
        self._setup_ui()
        
        # Apply dark theme
        self._apply_theme()
    
    def _setup_ui(self) -> None:
        """Create and layout UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Application title
        title_label = QLabel("EchoZero")
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Version/subtitle
        subtitle_label = QLabel("Audio Processing System")
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(subtitle_label)
        
        layout.addSpacing(30)
        
        # Current module label
        self.module_label = QLabel("Initializing...")
        module_font = QFont()
        module_font.setPointSize(12)
        module_font.setBold(True)
        self.module_label.setFont(module_font)
        self.module_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.module_label)
        
        # Current step label (sub-component)
        self.step_label = QLabel("")
        step_font = QFont()
        step_font.setPointSize(10)
        self.step_label.setFont(step_font)
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.step_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        self.step_label.setWordWrap(True)
        layout.addWidget(self.step_label)
        
        layout.addSpacing(20)
        
        # Overall progress bar
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setMinimum(0)
        self.overall_progress_bar.setMaximum(100)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setTextVisible(True)
        self.overall_progress_bar.setFormat("%p%")
        self.overall_progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {Colors.BORDER.name()};
                border-radius: {border_radius(5)};
                text-align: center;
                background-color: {Colors.BG_MEDIUM.name()};
                height: 25px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.ACCENT_BLUE.name()};
                border-radius: {border_radius(3)};
            }}
        """)
        layout.addWidget(self.overall_progress_bar)
        
        # Module progress bar (for current module)
        self.module_progress_bar = QProgressBar()
        self.module_progress_bar.setMinimum(0)
        self.module_progress_bar.setMaximum(100)
        self.module_progress_bar.setValue(0)
        self.module_progress_bar.setTextVisible(False)
        self.module_progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(3)};
                background-color: {Colors.BG_DARK.name()};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.STATUS_SUCCESS.name()};
                border-radius: {border_radius(2)};
            }}
        """)
        layout.addWidget(self.module_progress_bar)
        
        layout.addSpacing(20)
        
        # Loading log (recent modules/steps)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_DARK.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(3)};
                color: {Colors.TEXT_SECONDARY.name()};
                font-family: monospace;
                font-size: 9pt;
            }}
        """)
        layout.addWidget(self.log_text)
        
        # Set window size
        self.resize(500, 500)
        
        # Center on screen
        self._center_on_screen()
    
    def _center_on_screen(self) -> None:
        """Center the splash screen on the primary screen"""
        from PyQt6.QtWidgets import QApplication
        if QApplication.instance():
            screen = QApplication.primaryScreen()
            if screen:
                screen_geometry = screen.geometry()
                window_geometry = self.frameGeometry()
                window_geometry.moveCenter(screen_geometry.center())
                self.move(window_geometry.topLeft())
    
    def _apply_theme(self) -> None:
        """Apply dark theme colors"""
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        self.setPalette(palette)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
        """)
    
    def _add_log_entry(self, message: str) -> None:
        """Add an entry to the loading log"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    # LoadingProgressCallback implementation
    
    def on_module_start(self, module_name: str, total_steps: int = 1) -> None:
        """Called when a module starts loading (thread-safe via signals)"""
        self.signals.module_started.emit(module_name, total_steps)
    
    def on_module_step(self, module_name: str, step_name: str, step_number: int, total_steps: int) -> None:
        """Called when a step within a module completes (thread-safe via signals)"""
        self.signals.step_updated.emit(module_name, step_name, step_number, total_steps)
    
    def on_module_complete(self, module_name: str) -> None:
        """Called when a module finishes loading (thread-safe via signals)"""
        self.signals.module_completed.emit(module_name)
    
    def on_error(self, module_name: str, error: Exception) -> None:
        """Called when a module fails to load (thread-safe via signals)"""
        error_message = str(error)
        self.signals.error_occurred.emit(module_name, error_message)
    
    # Signal handlers (run on main thread)
    
    def _on_module_started(self, module_name: str, total_steps: int) -> None:
        """Handle module start (called on main thread)"""
        self.current_module = module_name
        self.module_total_steps = total_steps
        self.module_progress = 0
        self.module_label.setText(f"Loading: {module_name}")
        self.module_progress_bar.setMaximum(total_steps * 100)
        self.module_progress_bar.setValue(0)
        self._add_log_entry(f"→ {module_name}")
    
    def _on_step_updated(self, module_name: str, step_name: str, step_number: int, total_steps: int) -> None:
        """Handle step update (called on main thread)"""
        if module_name == self.current_module:
            self.current_step = step_name
            self.module_progress = step_number
            self.step_label.setText(step_name)
            
            # Update module progress bar
            progress_value = int((step_number / total_steps) * 100) if total_steps > 0 else 0
            self.module_progress_bar.setValue(progress_value)
            
            if step_name:
                self._add_log_entry(f"  • {step_name}")
    
    def _on_module_completed(self, module_name: str) -> None:
        """Handle module completion (called on main thread)"""
        if module_name == self.current_module:
            self.module_progress = self.module_total_steps
            self.module_progress_bar.setValue(100)
            self._add_log_entry(f"✓ {module_name} complete")
            self.current_step = None
            self.step_label.setText("")
    
    def _on_error_occurred(self, module_name: str, error_message: str) -> None:
        """Handle error (called on main thread)"""
        self._add_log_entry(f"✗ Error in {module_name}: {error_message}")
        self.step_label.setText(f"Error: {error_message}")
        self.step_label.setStyleSheet(f"color: {Colors.STATUS_ERROR.name()};")
    
    def update_overall_progress(self, progress: float) -> None:
        """
        Update overall progress bar.
        
        Args:
            progress: Progress value between 0.0 and 1.0
        """
        self.overall_progress = progress
        percentage = int(progress * 100)
        self.overall_progress_bar.setValue(percentage)
    
    def show_and_process_events(self) -> None:
        """Show splash screen and process Qt events"""
        self.show()
        from PyQt6.QtWidgets import QApplication
        if QApplication.instance():
            QApplication.instance().processEvents()

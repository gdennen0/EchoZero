"""
Qt Application Entry Point

Main Qt application class that implements UIBridge protocol.
"""
import sys
from typing import Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.application.api.application_facade import ApplicationFacade
from src.utils.message import Log
from src.application.bootstrap_loading_progress import LoadingProgressTracker


def _process_qt_events():
    """Process Qt events if QApplication exists (for splash screen updates)"""
    try:
        app = QApplication.instance()
        if app:
            app.processEvents()
    except ImportError:
        pass  # Qt not available


class QtEchoZeroApp:
    """
    Qt implementation of EchoZero UI.
    
    Implements UIBridge protocol for clean separation from core.
    """
    
    def __init__(self):
        self.app: Optional[QApplication] = None
        self.main_window = None
        self.facade: Optional[ApplicationFacade] = None
    
    def initialize(self, facade: ApplicationFacade, progress_tracker: Optional[LoadingProgressTracker] = None) -> None:
        """
        Initialize Qt application with facade reference.
        
        Args:
            facade: ApplicationFacade instance for all operations
            progress_tracker: Optional progress tracker for loading feedback
        """
        Log.info("Initializing Qt GUI")
        self.facade = facade
        
        # Module: Qt GUI Initialization
        if progress_tracker:
            progress_tracker.start_module("Qt GUI", "Initializing user interface", 5)
            _process_qt_events()
        
        # Create Qt application
        if progress_tracker:
            progress_tracker.update_step("Creating QApplication")
            _process_qt_events()
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("EZ")
        self.app.setOrganizationName("EZ")
        
        # Set application style
        if progress_tracker:
            progress_tracker.update_step("Setting up dark theme")
            _process_qt_events()
        self.app.setStyle("Fusion")

        # Store app_settings_manager in QApplication before theme setup so user preset is available
        if hasattr(self.facade, 'app_settings') and self.facade.app_settings:
            self.app.setProperty('app_settings', self.facade.app_settings)
            self._load_custom_themes(self.facade.app_settings)

        self._setup_dark_theme()

        # Initialize block type registry (happens during MainWindow creation)
        if progress_tracker:
            progress_tracker.update_step("Initializing block type registry")
            _process_qt_events()
        
        # Create main window (lazy import to avoid dependency at module level)
        if progress_tracker:
            progress_tracker.update_step("Creating main window")
            _process_qt_events()
        from ui.qt_gui.main_window import MainWindow
        self.main_window = MainWindow(self.facade)
        
        if progress_tracker:
            progress_tracker.complete_module()
            _process_qt_events()
        
        Log.info("Qt GUI initialized successfully")
    
    def run(self) -> int:
        """
        Start the Qt event loop.
        
        Returns:
            Exit code
        """
        if not self.app or not self.main_window:
            Log.error("Qt application not initialized. Call initialize() first.")
            return 1
        
        # Show and activate window (critical on macOS for proper focus and menu bar)
        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()
        
        Log.info("Starting Qt event loop")
        return self.app.exec()
    
    def shutdown(self) -> None:
        """Clean shutdown"""
        Log.info("Shutting down Qt GUI")
        if self.main_window:
            self.main_window.close()
        if self.app:
            self.app.quit()
    
    def _load_custom_themes(self, app_settings):
        """Load custom themes from the DB and register them in ThemeRegistry."""
        try:
            from ui.qt_gui.theme_registry import ThemeRegistry
            custom_themes = app_settings.get_custom_themes()
            for name, data in custom_themes.items():
                description = data.get("description", f"Custom theme: {name}")
                colors = data.get("colors", {})
                if colors:
                    ThemeRegistry.register_custom_theme(name, description, colors)
            if custom_themes:
                Log.debug(f"Loaded {len(custom_themes)} custom theme(s) from database")
        except Exception as e:
            Log.warning(f"Failed to load custom themes: {e}")
    
    def _setup_dark_theme(self):
        """
        Bootstrap palette setup before MainWindow is created.

        MainWindow._apply_theme() performs the full theme application
        (including global stylesheet).  This method ensures the
        QApplication palette uses design_system as the single source of truth.
        """
        from ui.qt_gui.design_system import get_application_palette

        self.app.setPalette(get_application_palette())


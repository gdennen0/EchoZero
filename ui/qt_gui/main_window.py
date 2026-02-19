"""
Main Window

Simple workspace with toolbar and all-dock layout.
Node Editor has embedded Properties and Quick Actions panels.
All windows are Level 1 docks that can be tabbed together.
"""
import os
import json
import sys
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QMenuBar, QMenu, QToolBar, QStatusBar,
    QMessageBox, QFileDialog, QSizePolicy, QApplication, QTabWidget,
    QLabel, QPlainTextEdit, QPushButton, QProgressBar, QFrame, QProgressDialog,
    QTabBar, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSettings, pyqtSlot
from PyQt6.QtGui import QShowEvent, QAction, QKeySequence, QUndoStack

from src.application.api.application_facade import ApplicationFacade
from src.application.commands import CommandBus
from src.utils.message import Log
from src.utils.settings import app_settings
from ui.qt_gui.design_system import get_stylesheet, Colors
from ui.qt_gui.style_factory import StyleFactory
from ui.qt_gui.core.dock_state_manager import DockStateManager
from ui.qt_gui.core.empty_workspace_widget import EmptyWorkspaceWidget
from ui.qt_gui.core.run_block_thread import RunBlockThread
from ui.qt_gui.core.save_project_thread import SaveProjectThread


class MainWindow(QMainWindow):
    """
    Simple workspace with two main dockable views.
    
    Layout:
    - Main toolbar (New/Open/Save)
    - Two main docks that can tabify: Node Editor and Batch Runner
    - Node Editor includes embedded Properties and Quick Actions panels
    - Block panels open as floating windows
    """
    
    # Qt signals for thread-safe GUI updates
    progress_start_signal = pyqtSignal(str, int, int)
    progress_update_signal = pyqtSignal(str, int, int)
    progress_complete_signal = pyqtSignal(int)
    progress_error_signal = pyqtSignal(str)
    subprocess_progress_signal = pyqtSignal(str, int)
    
    def __init__(self, facade: ApplicationFacade):
        super().__init__()
        self.facade = facade
        
        self.setWindowTitle("EchoZero")
        self.setMinimumSize(800, 600)
        
        # Apply theme and design system
        self._apply_theme()
        
        # Configure dock behavior for clean workspace layout:
        # - Force tabbed docking (no side-by-side split for main docks)
        # - Keep tabs at top for consistency across all dock areas
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks |
            QMainWindow.DockOption.AllowTabbedDocks |
            QMainWindow.DockOption.ForceTabbedDocks
        )
        
        # Disable nested docking to keep layout predictable
        self.setDockNestingEnabled(False)
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        
        # Undo/Redo stack (industry-standard Qt implementation)
        self.undo_stack = QUndoStack(self)
        max_undo = app_settings.get("max_undo_steps", 50)
        self.undo_stack.setUndoLimit(max_undo)
        
        # Create CommandBus instance and set it on the facade
        command_bus = CommandBus(self.undo_stack)
        self.facade.set_command_bus(command_bus)
        
        # Connect undo stack to UI refresh (single source of truth for undo/redo UI updates)
        self.undo_stack.indexChanged.connect(self._on_undo_stack_changed)
        
        # Main view docks (can be tabified together)
        self.node_editor_dock = None
        self.node_editor_window = None
        
        # Other components
        self.execution_thread = None
        self.save_thread = None
        self.save_progress_dialog = None
        self.execution_dock = None
        self.open_panels = {}
        self.block_panels_menu = None
        
        # Dock State Manager (simple Qt-native state management)
        # Uses Qt's saveState/restoreState for dock positions
        self.dock_manager = DockStateManager(self, facade)
        
        # Setup UI
        self._create_actions()
        self._create_menus()
        self._create_toolbars()
        self._create_minimal_central()
        self._create_dock_widgets()
        self._create_progress_bar()
        self._create_status_bar()
        
        # Initialize block panels menu
        if self.block_panels_menu:
            self._update_block_panels_menu()
        
        # Subscribe to events
        self._subscribe_to_events()
        self._update_ui_state()
        
        # Track if initialization has run (will run in showEvent)
        self._initialization_complete = False
        
        # Track if we're in the close sequence (to prevent double-saves)
        self._closing = False
        
        Log.info("Main window created (all-dock workspace)")
    
    def _create_actions(self):
        """Create menu and toolbar actions"""
        # File actions
        self.action_new_project = QAction("&New Project", self)
        self.action_new_project.setShortcut(QKeySequence.StandardKey.New)
        self.action_new_project.triggered.connect(self._on_new_project)
        
        self.action_open_project = QAction("&Open Project", self)
        self.action_open_project.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open_project.triggered.connect(self._on_open_project)
        
        self.action_save_project = QAction("&Save Project", self)
        self.action_save_project.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save_project.triggered.connect(self._on_save_project)
        
        self.action_save_project_as = QAction("Save Project &As...", self)
        self.action_save_project_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.action_save_project_as.triggered.connect(self._on_save_project_as)
        
        self.action_settings = QAction("&Settings...", self)
        self.action_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.action_settings.triggered.connect(self._on_open_settings)
        
        self.action_exit = QAction("E&xit", self)
        self.action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        self.action_exit.triggered.connect(self.close)
        
        # Edit actions - created by QUndoStack (auto-updates text and enabled state)
        self.action_undo = self.undo_stack.createUndoAction(self, "&Undo")
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        
        self.action_redo = self.undo_stack.createRedoAction(self, "&Redo")
        self.action_redo.setShortcut(QKeySequence.StandardKey.Redo)
        
        self.action_select_all = QAction("Select &All", self)
        self.action_select_all.setShortcut(QKeySequence.StandardKey.SelectAll)
        self.action_select_all.triggered.connect(self._on_select_all)
        
        # View actions - dock toggles
        self.action_view_node_editor = QAction("&Node Editor", self)
        self.action_view_node_editor.setShortcut(QKeySequence("Ctrl+1"))
        self.action_view_node_editor.setCheckable(True)
        self.action_view_node_editor.setChecked(False)  # Hidden by default

        self.action_view_setlist = QAction("&Setlist", self)
        self.action_view_setlist.setShortcut(QKeySequence("Ctrl+3"))
        self.action_view_setlist.setCheckable(True)
        self.action_view_setlist.setChecked(False)  # Hidden by default

        self.action_view_execution = QAction("E&xecution", self)
        self.action_view_execution.setCheckable(True)
        self.action_view_execution.setChecked(False)
        self.action_view_execution.setToolTip("Show execution log and process status")
        
        self.action_reset_layout = QAction("&Reset Layout", self)
        self.action_reset_layout.triggered.connect(self._reset_dock_layout)
        
        self.action_command_history = QAction("Command &History...", self)
        self.action_command_history.setShortcut(QKeySequence("Ctrl+H"))
        self.action_command_history.triggered.connect(self._on_show_command_history)
    
    def _create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.action_new_project)
        file_menu.addAction(self.action_open_project)
        file_menu.addSeparator()
        file_menu.addAction(self.action_save_project)
        file_menu.addAction(self.action_save_project_as)
        file_menu.addSeparator()
        file_menu.addAction(self.action_undo)
        file_menu.addAction(self.action_redo)
        file_menu.addSeparator()
        file_menu.addAction(self.action_settings)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)
        
        # Edit menu (standard location for undo/redo too)
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.action_undo)
        edit_menu.addAction(self.action_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_settings)
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_select_all)
        
        # Window menu
        window_menu = menubar.addMenu("&Window")
        
        # Main views
        window_menu.addAction(self.action_view_node_editor)
        window_menu.addAction(self.action_view_setlist)
        window_menu.addAction(self.action_view_execution)
        window_menu.addSeparator()
        
        # Block panels submenu (dynamically updated)
        self.block_panels_menu = window_menu.addMenu("Block &Panels")
        self._update_block_panels_menu()
        
        window_menu.addSeparator()
        window_menu.addAction(self.action_command_history)
        window_menu.addSeparator()
        window_menu.addAction(self.action_reset_layout)
        
        # Layout management submenu
        layout_menu = window_menu.addMenu("&Layouts")
        
        # Save as Default
        self.action_save_as_default = QAction("Save as &Default", self)
        self.action_save_as_default.triggered.connect(self._save_as_default_layout)
        layout_menu.addAction(self.action_save_as_default)
        
        layout_menu.addSeparator()
        
        # Export Layout
        self.action_export_layout = QAction("&Export Layout...", self)
        self.action_export_layout.triggered.connect(self._on_export_layout)
        layout_menu.addAction(self.action_export_layout)
        
        # Import Layout
        self.action_import_layout = QAction("&Import Layout...", self)
        self.action_import_layout.triggered.connect(self._on_import_layout)
        layout_menu.addAction(self.action_import_layout)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        action_about = QAction("&About", self)
        action_about.triggered.connect(self._on_about)
        help_menu.addAction(action_about)
    
    def _create_toolbars(self):
        """Create toolbar"""
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)
        
        toolbar.addAction(self.action_new_project)
        toolbar.addAction(self.action_open_project)
        toolbar.addAction(self.action_save_project)
        toolbar.addSeparator()
        toolbar.addAction(self.action_settings)
    
    def _create_minimal_central(self):
        """Create central widget with empty workspace guidance"""
        # Show helpful message when no docks are docked
        self.empty_workspace = EmptyWorkspaceWidget()
        self.setCentralWidget(self.empty_workspace)
    
    def _update_empty_workspace_visibility(self):
        """Show/hide empty workspace based on whether any docks are docked (not floating)"""
        if not hasattr(self, 'empty_workspace') or self.empty_workspace is None:
            return
        
        # Check if any dock is docked (not floating and not hidden)
        has_docked_windows = False
        
        # Check all dock widgets in the main window
        for dock in self.findChildren(QDockWidget):
            # A dock is "docked" if it's not floating and not explicitly hidden
            # Note: We check isHidden() instead of isVisible() because isVisible()
            # can return False for docks that are part of a tab group (not active tab)
            if not dock.isFloating() and not dock.isHidden():
                has_docked_windows = True
                break
        
        # Show empty workspace only when no docks are docked
        self.empty_workspace.setVisible(not has_docked_windows)
    
    def _on_dock_floating_changed(self, floating: bool):
        """Called when a dock's floating state changes"""
        # Update empty workspace visibility
        self._update_empty_workspace_visibility()
    
    def _create_dock_widgets(self):
        """Create main dockable windows (Node Editor, Setlist, Batch Runner)"""
        # Create Node Editor dock
        self.node_editor_dock = self._create_node_editor_dock()
        self.node_editor_window = self.node_editor_dock.widget()

        # Create Setlist dock
        self.setlist_dock = self._create_setlist_dock()
        self.setlist_window = self.setlist_dock.widget()
        
        # Tabify Setlist with Node Editor (same level as main views)
        if self.node_editor_dock and self.setlist_dock:
            self.tabifyDockWidget(self.node_editor_dock, self.setlist_dock)

        # Execution panel (view process and progress log)
        self.execution_dock = self._create_execution_dock()
        if self.execution_dock:
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.execution_dock)
            self.execution_dock.hide()
            self.dock_manager.register_dock("execution", self.execution_dock)

        # Connect Node Editor signals
        if self.node_editor_window:
            self.node_editor_window.block_panel_requested.connect(self.open_block_panel)
        
        # Connect Setlist signals
        if self.setlist_window and hasattr(self.setlist_window, 'setlist_view'):
            self.setlist_window.setlist_view.song_switched.connect(self._on_setlist_song_switched)
        
        # Update empty workspace visibility (docks start floating, so it should be visible)
        self._update_empty_workspace_visibility()
        
        # Connect view toggles to dock visibility
        if self.node_editor_dock:
            self.action_view_node_editor.triggered.connect(
                lambda checked: self._toggle_dock_visibility(self.node_editor_dock, checked))
            self.node_editor_dock.visibilityChanged.connect(
                self.action_view_node_editor.setChecked)
        
        if self.setlist_dock:
            self.action_view_setlist.triggered.connect(
                lambda checked: self._toggle_dock_visibility(self.setlist_dock, checked))
            self.setlist_dock.visibilityChanged.connect(
                self.action_view_setlist.setChecked)

        if self.execution_dock:
            self.action_view_execution.triggered.connect(
                lambda checked: self._toggle_dock_visibility(self.execution_dock, checked))
            self.execution_dock.visibilityChanged.connect(
                self.action_view_execution.setChecked)
        
        # Apply standard dock styling
        self._apply_dock_styling()
        
        # NOTE: Docks start floating by default
        # Layout will be applied by _restore_layout() after initialization
    
    def _create_node_editor_dock(self):
        """Create Node Editor dock"""
        from ui.qt_gui.node_editor.node_editor_window import NodeEditorWindow
        
        node_editor_window = NodeEditorWindow(self.facade, undo_stack=self.undo_stack)
        dock = self._create_dock("Node Editor", node_editor_window, "NodeEditorDock")
        
        # Add to dock system (hidden by default - layout restore will show if needed)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
        dock.hide()
        
        # Register with Dock State Manager
        self.dock_manager.register_dock("node_editor", dock)
        
        return dock
    
    def _create_setlist_dock(self):
        """Create Setlist dock"""
        from ui.qt_gui.views.setlist_window import SetlistWindow
        
        setlist_window = SetlistWindow(self.facade)
        dock = self._create_dock("Setlist", setlist_window, "SetlistDock")
        
        # Add to dock system (same level as Node Editor)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
        dock.hide()
        
        # Register with Dock State Manager
        self.dock_manager.register_dock("setlist", dock)
        
        return dock

    def _create_execution_dock(self):
        """Create Execution dock (process status and log)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._execution_status_label = QLabel("Idle")
        self._execution_status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._execution_status_label)

        row = QHBoxLayout()
        self._execution_panel_progress = QProgressBar()
        self._execution_panel_progress.setMinimum(0)
        self._execution_panel_progress.setMaximum(100)
        self._execution_panel_progress.setValue(0)
        self._execution_panel_progress.setTextVisible(True)
        row.addWidget(self._execution_panel_progress)
        self._execution_cancel_btn = QPushButton("Cancel")
        self._execution_cancel_btn.clicked.connect(self._on_cancel_execution)
        row.addWidget(self._execution_cancel_btn)
        layout.addLayout(row)

        layout.addWidget(QLabel("Log:"))
        self._execution_log_edit = QPlainTextEdit()
        self._execution_log_edit.setReadOnly(True)
        self._execution_log_edit.setMinimumHeight(120)
        self._execution_log_edit.setPlaceholderText("Progress and output appear here when a block is running.")
        layout.addWidget(self._execution_log_edit)

        dock = self._create_dock("Execution", panel, "ExecutionDock")
        return dock

    def _execution_set_running_ui(self, running: bool):
        """Update Execution panel for running vs idle state."""
        status_label = getattr(self, "_execution_status_label", None)
        progress_bar = getattr(self, "_execution_panel_progress", None)
        cancel_btn = getattr(self, "_execution_cancel_btn", None)
        if not status_label:
            return
        if running:
            name = getattr(self, "_current_run_block_name", "Block")
            status_label.setText(f"Running: {name}")
            status_label.setStyleSheet("font-weight: bold; color: #27ae60;")
            if progress_bar:
                progress_bar.setValue(0)
            if cancel_btn:
                cancel_btn.setEnabled(True)
            if self.execution_dock and not self.execution_dock.isVisible():
                self.execution_dock.show()
        else:
            status_label.setText("Idle")
            status_label.setStyleSheet("font-weight: bold;")
            if cancel_btn:
                cancel_btn.setEnabled(False)

    def _append_execution_log(self, line: str, is_error: bool = False):
        """Append a line to the execution log with timestamp and update the Execution panel."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        stamped = f"[{timestamp}] {line}"
        self._execution_log_lines = getattr(self, "_execution_log_lines", [])
        self._execution_log_lines.append(stamped)
        if is_error:
            status_label = getattr(self, "_execution_status_label", None)
            if status_label:
                status_label.setText(line[:80] + "..." if len(line) > 80 else line)
                status_label.setStyleSheet("font-weight: bold; color: #c0392b;")
        log_edit = getattr(self, "_execution_log_edit", None)
        if log_edit:
            log_edit.appendPlainText(stamped)
            log_edit.verticalScrollBar().setValue(log_edit.verticalScrollBar().maximum())

    def _create_dock(self, title: str, widget: QWidget, object_name: str) -> QDockWidget:
        """Create a dock widget with standard settings"""
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(self._main_dock_features())
        dock.setWidget(widget)
        
        # Connect to floating state changes to update empty workspace visibility
        dock.topLevelChanged.connect(self._on_dock_floating_changed)
        dock.visibilityChanged.connect(lambda: self._update_empty_workspace_visibility())
        
        return dock
    
    def _main_dock_features(self) -> QDockWidget.DockWidgetFeature:
        """Feature policy for primary workspace docks (tabbed in main window)."""
        return (
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
    
    def _panel_dock_features(self) -> QDockWidget.DockWidgetFeature:
        """Feature policy for block panels (can float or dock)."""
        return (
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
    
    def _enforce_main_dock_tabbing(self):
        """Keep primary docks as a single top-tabbed group."""
        if self.node_editor_dock and self.setlist_dock:
            self.tabifyDockWidget(self.node_editor_dock, self.setlist_dock)
            if self.node_editor_dock.isVisible():
                self.node_editor_dock.raise_()
    
    def _ensure_dock_features(self):
        """
        Self-healing: Re-apply standard features to all docks.
        
        Called after restoring saved state to ensure dock behavior is never
        broken by corrupted or outdated saved configurations.
        """
        main_dock_names = {"NodeEditorDock", "SetlistDock"}
        main_features = self._main_dock_features()
        panel_features = self._panel_dock_features()

        for dock in self.findChildren(QDockWidget):
            if dock:
                dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
                if dock.objectName() in main_dock_names:
                    dock.setFeatures(main_features)
                else:
                    dock.setFeatures(panel_features)
        
        self._enforce_main_dock_tabbing()
        self._setup_dock_tab_close_buttons()
    
    def _apply_dock_styling(self):
        """Apply global stylesheet to QApplication for instant app-wide propagation.
        
        Setting the stylesheet on QApplication (rather than MainWindow) ensures
        that floating docks, dialogs, and all child widgets inherit the theme
        automatically.  Qt re-evaluates the global stylesheet for every widget
        whenever it changes, so this single call updates the entire UI.
        """
        app = QApplication.instance()
        if app:
            app.setStyleSheet(get_stylesheet() + StyleFactory.dock_tabs())
    
    def _setup_dock_tab_close_buttons(self):
        """
        Enable close buttons on dock tab bars.
        
        Qt creates QTabBar for tabified dock widgets internally. We find these,
        enable tabsClosable, and connect tabCloseRequested to close the correct dock.
        """
        for tab_bar in self.findChildren(QTabBar):
            # Skip tab bars that belong to QTabWidget (e.g. settings dialog)
            if isinstance(tab_bar.parent(), QTabWidget):
                continue
            # Skip if already configured
            if tab_bar.property("_dock_tab_configured"):
                continue
            # Only configure tab bars with 2+ tabs (tabified dock groups)
            if tab_bar.count() < 2:
                continue
            tab_bar.setTabsClosable(True)
            tab_bar.tabCloseRequested.connect(self._on_dock_tab_close_requested)
            tab_bar.setProperty("_dock_tab_configured", True)
    
    def _on_dock_tab_close_requested(self, index: int):
        """Close the dock widget corresponding to the tab at the given index."""
        tab_bar = self.sender()
        if not isinstance(tab_bar, QTabBar) or index < 0 or index >= tab_bar.count():
            return
        # Try to find dock via parent hierarchy (Qt places tab bar and stacked content together)
        parent = tab_bar.parent()
        if parent:
            stacked = parent.findChild(QStackedWidget)
            if stacked and stacked.count() == tab_bar.count():
                content = stacked.widget(index)
                if content:
                    for dock in self.findChildren(QDockWidget):
                        if dock.widget() is content:
                            dock.close()
                            return
        # Fallback: use tabifiedDockWidgets with spatial matching
        tab_bar_center = tab_bar.mapToGlobal(tab_bar.rect().center())
        best_dock = None
        best_dist = float("inf")
        for dock in self.findChildren(QDockWidget):
            tabified = self.tabifiedDockWidgets(dock)
            if len(tabified) != tab_bar.count() or index >= len(tabified):
                continue
            # Use first dock in group for position check
            other = tabified[0]
            if other.isFloating():
                continue
            try:
                other_center = other.mapToGlobal(other.rect().center())
                dist = (tab_bar_center.x() - other_center.x()) ** 2 + (tab_bar_center.y() - other_center.y()) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_dock = tabified[index]
            except Exception:
                pass
        if best_dock:
            best_dock.close()
    
    def _toggle_dock_visibility(self, dock: QDockWidget, checked: bool):
        """Toggle dock visibility from Window menu"""
        if checked:
            # Ensure dock has correct features before showing
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setFeatures(self._main_dock_features())
            dock.show()
            dock.raise_()
            self._enforce_main_dock_tabbing()
        else:
            dock.hide()
    
    
    def _save_all_window_state(self):
        """Save complete window state using LayoutStateController"""
        # Save dock state using Qt's native saveState
        self.dock_manager.save_state()
    
    def _save_as_default_layout(self):
        """Save current layout as default - for now just saves to session"""
        # TODO: Could implement file export later if needed
        self.dock_manager.save_state()
        self.statusBar().showMessage("Default layout saved", 3000)
    
    def _on_export_layout(self):
        """Export current layout to a file"""
        # Simplified: just save and notify
        self.dock_manager.save_state()
        self.statusBar().showMessage("Layout saved to session", 3000)
    
    def _on_import_layout(self):
        """Import layout from a file"""
        # Simplified: just restore and notify
        if self.dock_manager.restore_state():
            self.statusBar().showMessage("Layout restored from session", 3000)
            self._ensure_dock_features()
        else:
            QMessageBox.warning(self, "Import Failed", "No saved layout found.")

    def _initialize_and_restore(self):
        """Complete initialization: load project, create panels, restore state"""
        try:
            # Try to autoload project
            project_loaded = self._try_autoload_project()
            
            # If no project loaded, prompt user to create/open one
            if not project_loaded:
                project_loaded = self._prompt_for_project()
            
            # Only proceed with full UI if project is loaded
            if project_loaded:
                # Create all saved panels BEFORE restoring state
                # Qt's restoreState needs all docks to exist first
                self._create_saved_panels()
                
                # Refresh node editor ONCE after everything is set up
                # This is the single refresh point during initialization
                if self.node_editor_window:
                    self.node_editor_window.refresh()
                
                # Restore session state (zoom, viewport, selected block)
                self._restore_session_state()
                
                # Restore layout state (dock positions, tabs, etc.)
                self._restore_layout()
                
                Log.info(f"Initialization complete. Docks: {self.dock_manager.get_registered_docks()}")
            else:
                # No project loaded - show minimal UI or exit
                Log.info("No project loaded, showing minimal UI")
                self._restore_layout()  # Still restore layout for consistency
                self._update_empty_workspace_visibility()
                # Disable project-dependent actions
                self._update_ui_for_no_project()
            
        except Exception as e:
            Log.error(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
    
    def _restore_layout(self):
        """Restore layout state using Qt's native restoreState"""
        # Try to restore from session
        if self.dock_manager.restore_state():
            Log.info("Layout restored from session")
            self._ensure_dock_features()
            self._enforce_main_dock_tabbing()
            self._update_empty_workspace_visibility()
            self._setup_dock_tab_close_buttons()
            return
        
        # No saved state - start with blank workspace
        Log.info("No saved layout, starting with blank workspace")
        self._set_default_layout()
        self._ensure_dock_features()
        self._enforce_main_dock_tabbing()
        self._update_empty_workspace_visibility()
        self._setup_dock_tab_close_buttons()
    
    def _set_default_layout(self):
        """Set default layout - blank workspace with all docks hidden.
        
        On first run, the main window is empty. Users show windows via Window menu.
        """
        # Hide all core docks
        if self.node_editor_dock:
            self.node_editor_dock.hide()
        if self.setlist_dock:
            self.setlist_dock.hide()
        if getattr(self, "execution_dock", None):
            self.execution_dock.hide()
        
        # Hide any open block panels
        for block_id, panel in self.open_panels.items():
            panel.hide()
    
    
    def _create_saved_panels(self):
        """Create all panels that were open when saved.
        
        IMPORTANT: Panels are created but NOT positioned here.
        Positioning happens in _restore_layout() via Qt's restoreState().
        """
        try:
            # Get saved panel IDs from dock manager
            panel_ids = self.dock_manager.get_open_panel_ids()
            
            if not panel_ids:
                Log.debug("No saved panels to restore")
                return
            
            # Get existing blocks in project
            blocks_result = self.facade.list_blocks()
            existing_block_ids = set()
            block_types = {}
            if blocks_result.success and blocks_result.data:
                existing_block_ids = {block.id for block in blocks_result.data}
                block_types = {block.id: block.type for block in blocks_result.data}
            
            # Create panels for existing blocks
            created_count = 0
            for block_id in panel_ids:
                if block_id in existing_block_ids and block_id not in self.open_panels:
                    panel = self._create_panel(block_id, block_types.get(block_id))
                    if panel:
                        created_count += 1
            
            if created_count > 0:
                Log.info(f"Created {created_count} panel(s) for state restoration")
                
        except Exception as e:
            Log.warning(f"Error creating saved panels: {e}")
    
    def _create_panel(self, block_id: str, block_type: str = None):
        """Create a single panel (internal helper)"""
        try:
            # Get block info if not provided
            if not block_type:
                result = self.facade.describe_block(block_id)
                if not result.success:
                    return None
                block_type = result.data.type
            
            from ui.qt_gui.block_panels import get_panel_class
            panel_class = get_panel_class(block_type)
            if not panel_class:
                return None
            
            # Create panel WITH MainWindow as parent (required for Qt dock system)
            panel = panel_class(block_id, self.facade, parent=self)
            panel.panel_closed.connect(self._on_panel_closed)
            
            # Set object name for state save/restore (CRITICAL for Qt restoreState)
            panel.setObjectName(f"BlockPanel_{block_id}")
            
            # Connect signals for empty workspace visibility
            panel.topLevelChanged.connect(self._on_dock_floating_changed)
            panel.visibilityChanged.connect(lambda: self._update_empty_workspace_visibility())
            
            # Add to dock system
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, panel)
            QTimer.singleShot(0, self._setup_dock_tab_close_buttons)
            
            # Register with Dock State Manager
            window_id = f"block_panel_{block_id}"
            self.dock_manager.register_dock(window_id, panel)
            
            self.open_panels[block_id] = panel
            return panel
        except Exception as e:
            Log.warning(f"Failed to create panel for {block_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _reset_dock_layout(self):
        """Reset to default blank layout"""
        self._set_default_layout()
        self._ensure_dock_features()
        self.statusBar().showMessage("Layout reset to blank workspace", 3000)

        # Update action states
        if self.node_editor_dock:
            self.action_view_node_editor.setChecked(self.node_editor_dock.isVisible())
        if self.setlist_dock:
            self.action_view_setlist.setChecked(self.setlist_dock.isVisible())
        if getattr(self, "execution_dock", None):
            self.action_view_execution.setChecked(self.execution_dock.isVisible())

    def _update_block_panels_menu(self):
        """
        Update Block Panels menu to show ALL blocks in project that can have panels.
        
        Shows blocks organized by type. Checkmark = panel is open.
        Clicking opens/focuses the panel (or closes if unchecked).
        """
        if not self.block_panels_menu:
            return
        
        self.block_panels_menu.clear()
        
        # Check if we have a project loaded
        if not self.facade.get_current_project_id():
            action = self.block_panels_menu.addAction("(No project loaded)")
            action.setEnabled(False)
            return
        
        # Get all blocks in the project
        result = self.facade.list_blocks()
        if not result.success or not result.data:
            action = self.block_panels_menu.addAction("(No blocks in project)")
            action.setEnabled(False)
            return
        
        # Get panel registry to check which blocks can have panels
        from ui.qt_gui.block_panels import is_panel_registered
        
        # Group blocks by type
        blocks_by_type = {}
        for block in result.data:
            # Only include blocks that have registered panel types
            if not is_panel_registered(block.type):
                continue
            
            if block.type not in blocks_by_type:
                blocks_by_type[block.type] = []
            blocks_by_type[block.type].append(block)
        
        if not blocks_by_type:
            action = self.block_panels_menu.addAction("(No panels available)")
            action.setEnabled(False)
            return
        
        # Create submenu for each block type
        for block_type in sorted(blocks_by_type.keys()):
            type_menu = self.block_panels_menu.addMenu(block_type)
            
            for block in sorted(blocks_by_type[block_type], key=lambda b: b.name):
                action = type_menu.addAction(block.name)
                action.setCheckable(True)
                
                # Check if panel is already open
                is_open = block.id in self.open_panels
                action.setChecked(is_open)
                
                # Connect to toggle handler
                action.triggered.connect(
                    lambda checked, bid=block.id: self._toggle_block_panel(bid, checked)
                )
    
    def _toggle_block_panel(self, block_id: str, should_open: bool):
        """Toggle a block panel open/closed from the Window menu"""
        if should_open:
            self.open_block_panel(block_id)
        else:
            # Close the panel if it's open
            if block_id in self.open_panels:
                panel = self.open_panels[block_id]
                panel.close()
    
    def _create_progress_bar(self):
        """Create integrated status bar with progress indicator"""
        from ui.qt_gui.core.progress_bar import StatusBarProgress
        
        # Create and set the integrated status bar
        self.progress_bar = StatusBarProgress(self)
        self.progress_bar.cancel_requested.connect(self._on_cancel_execution)
        self.setStatusBar(self.progress_bar)
        
        # Connect signals
        self.progress_start_signal.connect(self._update_progress_start)
        self.progress_update_signal.connect(self._update_progress)
        self.progress_complete_signal.connect(self._update_progress_complete)
        self.progress_error_signal.connect(self._update_progress_error)
        self.subprocess_progress_signal.connect(self._update_subprocess_progress)
    
    def _create_status_bar(self):
        """Status bar is now created in _create_progress_bar"""
        pass  # Integrated into progress bar
    
    # ==================== Panel Management ====================
    
    def open_block_panel(self, block_id: str):
        """Open configuration panel for a block"""
        if block_id in self.open_panels:
            panel = self.open_panels[block_id]
            panel.show()
            panel.raise_()
            panel.setFocus()
            return
        
        from PyQt6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Opening panel...")
        QApplication.processEvents()
        
        try:
            result = self.facade.describe_block(block_id)
            if not result.success:
                QMessageBox.warning(self, "Cannot Open Panel", f"Could not load block: {result.message}")
                return
            
            block = result.data
            
            from ui.qt_gui.block_panels import get_panel_class
            panel_class = get_panel_class(block.type)
            
            if not panel_class:
                QMessageBox.information(self, "Panel Not Available",
                    f"No panel available for {block.type} blocks.")
                return
            
            # Create panel WITH MainWindow as parent (required for Qt dock system)
            panel = panel_class(block_id, self.facade, parent=self)
            panel.panel_closed.connect(self._on_panel_closed)
            
            # Set object name for state save/restore
            panel.setObjectName(f"BlockPanel_{block_id}")
            
            # Connect signals for empty workspace visibility (same as main docks)
            panel.topLevelChanged.connect(self._on_dock_floating_changed)
            panel.visibilityChanged.connect(lambda: self._update_empty_workspace_visibility())
            
            # Add to dock system then set floating
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, panel)
            panel.setFeatures(self._panel_dock_features())
            panel.setFloating(True)
            panel.setGeometry(200, 200, 1200, 600)
            
            # Register with Dock State Manager
            window_id = f"block_panel_{block_id}"
            self.dock_manager.register_dock(window_id, panel)
            
            self.open_panels[block_id] = panel
            panel.show()
            panel.raise_()
            panel.activateWindow()
            
            # Update empty workspace visibility and block panels menu
            self._update_empty_workspace_visibility()
            self._update_block_panels_menu()
            
            self.statusBar().showMessage(f"Panel opened: {block.name} (drag title bar to dock)", 3000)
            
        except Exception as e:
            Log.error(f"Failed to create panel: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Panel Error", f"Failed to open panel: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
    
    def _on_panel_closed(self, block_id: str):
        if block_id in self.open_panels:
            # Unregister from Dock State Manager
            window_id = f"block_panel_{block_id}"
            self.dock_manager.unregister_dock(window_id)
            
            del self.open_panels[block_id]
            # Update block panels menu
            self._update_block_panels_menu()
            # Save layout state when panel is closed
            # BUT NOT during app close - closeEvent already saved the complete state
            if not self._closing:
                self._save_all_window_state()
    
    def _close_all_block_panels(self):
        """Close all open block panels (used when switching projects)"""
        # Make a copy of keys since we're modifying the dict
        block_ids = list(self.open_panels.keys())
        for block_id in block_ids:
            panel = self.open_panels.get(block_id)
            if panel:
                try:
                    panel.close()
                except Exception as e:
                    Log.warning(f"Error closing panel for {block_id}: {e}")
        
        # Clear any remaining references
        self.open_panels.clear()
        self._update_block_panels_menu()
    
    
    # ==================== Event Handling ====================
    
    def _subscribe_to_events(self):
        """Subscribe to application events"""
        event_bus = self.facade.event_bus
        
        # Project events
        event_bus.subscribe("project.loaded", self._on_project_changed)
        event_bus.subscribe("project.created", self._on_project_changed)
        event_bus.subscribe("ProjectCreated", self._on_project_changed)
        
        # Block events - refresh node editor when blocks change
        event_bus.subscribe("BlockAdded", self._on_block_changed)
        event_bus.subscribe("BlockRemoved", self._on_block_changed)
        event_bus.subscribe("BlockUpdated", self._on_block_updated)
        
        # Connection events - refresh node editor when connections change
        event_bus.subscribe("ConnectionCreated", self._on_connection_changed)
        event_bus.subscribe("ConnectionRemoved", self._on_connection_changed)
        event_bus.subscribe("ConnectionsChanged", self._on_connection_changed)
        
        # Execution events
        event_bus.subscribe("execution.completed", self._on_execution_completed)
        event_bus.subscribe("execution.failed", self._on_execution_failed)
        event_bus.subscribe("ExecutionStarted", self._on_progress_started)
        event_bus.subscribe("ExecutionProgress", self._on_progress_update)
        event_bus.subscribe("ExecutionCompleted", self._on_progress_completed)
        event_bus.subscribe("ExecutionFailed", self._on_progress_failed)
        event_bus.subscribe("SettingsOperationFailed", self._on_settings_operation_failed)
        
        # SubprocessProgress events - for subprocess progress tracking (demucs, etc.)
        event_bus.subscribe("SubprocessProgress", self._on_subprocess_progress)
        Log.info("MainWindow: Subscribed to SubprocessProgress events")
    
    def _update_ui_state(self):
        """Update UI based on project state"""
        project_id = self.facade.get_current_project_id()
        
        if project_id:
            # Enable project-dependent actions
            self.action_save_project.setEnabled(True)
            self.action_save_project_as.setEnabled(True)
            
            # Get project name for window title
            try:
                project = self.facade.project_service.load_project(project_id)
                project_name = project.name if project else "Untitled"
            except Exception:
                project_name = "Untitled"
            
            # Update window title with project name
            self.setWindowTitle(f"EchoZero - {project_name}")
        else:
            # No project - disable project-dependent actions
            self._update_ui_for_no_project()
    
    def _on_project_changed(self, event):
        """Handle project change event - update UI and refresh node editor"""
        self._update_ui_state()
        # Defer node editor refresh to avoid multiple refreshes during initialization
        if self.node_editor_window and not getattr(self, '_initialization_complete', False):
            # During initialization, refresh will happen in _initialize_and_restore
            Log.debug("MainWindow: Deferring node editor refresh during initialization")
        elif self.node_editor_window:
            # Project changed = major transition, center the view
            self.node_editor_window.refresh_and_center()
        # Update block panels menu when project changes
        self._update_block_panels_menu()
    
    def _on_block_changed(self, event):
        """Handle block added/removed events - refresh node editor and menus"""
        Log.debug(f"Block changed event: {event.name if hasattr(event, 'name') else type(event).__name__}")
        # Close panel for removed block to prevent FK errors and stale UI
        if getattr(event, "name", None) == "BlockRemoved":
            block_id = (getattr(event, "data", None) or {}).get("id")
            if block_id and block_id in self.open_panels:
                panel = self.open_panels[block_id]
                try:
                    panel.close()
                except Exception as e:
                    Log.warning(f"MainWindow: Error closing panel for removed block {block_id}: {e}")
        if self.node_editor_window:
            self.node_editor_window.refresh()
        self._update_ui_state()
        # Update block panels menu when blocks are added/removed
        self._update_block_panels_menu()
    
    def _on_block_updated(self, event):
        """Handle block updated events - refresh properties if showing this block"""
        Log.debug(f"Block updated event: {event.data if hasattr(event, 'data') else event}")
        # Refresh properties panel if it's showing the updated block
        # Properties panel is now embedded in NodeEditorWindow
        if self.node_editor_window and hasattr(self.node_editor_window, 'properties_panel'):
            properties_panel = self.node_editor_window.properties_panel
            if properties_panel and hasattr(event, 'data'):
                block_id = event.data.get('id')
                if block_id and hasattr(properties_panel, 'current_block_id'):
                    if properties_panel.current_block_id == block_id:
                        properties_panel.show_block_properties(block_id)
    
    def _on_connection_changed(self, event):
        """Handle connection created/removed events - refresh node editor"""
        Log.debug(f"Connection changed event: {event.name if hasattr(event, 'name') else type(event).__name__}")
        if self.node_editor_window:
            self.node_editor_window.refresh()
    
    def _on_execution_completed(self, event):
        pass
    
    def _on_execution_failed(self, event):
        Log.error(f"Execution failed: {event}")

    def _on_settings_operation_failed(self, event):
        """Surface settings failures loudly and clearly to the user."""
        data = getattr(event, "data", {}) or {}
        block_id = data.get("block_id", "unknown")
        keys = data.get("keys") or []
        key_text = ", ".join(keys) if keys else "<unknown>"
        message = data.get("message", "Unknown settings failure")
        manager = data.get("manager", "SettingsManager")

        visible = f"{manager} failed to save settings ({key_text}) for block {block_id}: {message}"
        self.statusBar().showMessage(visible, 12000)
        QMessageBox.warning(self, "Settings Save Failed", visible)
    
    # ==================== Undo/Redo UI Refresh ====================
    
    def _on_undo_stack_changed(self, index: int):
        """
        Handle undo stack index changes (single source of truth for UI refresh).
        
        This is called ONCE after any undo/redo operation completes.
        Notifies all open panels to refresh their state.
        
        No guard needed - commands are now created at drag-end, not during refresh.
        """
        Log.debug(f"MainWindow: Undo stack changed to index {index}, refreshing UI")
        
        # Refresh node editor (for block positions)
        if self.node_editor_window:
            self.node_editor_window.scene.refresh_from_database()
        
        # Refresh all open block panels (for timeline events, etc.)
        for block_id, panel in list(self.open_panels.items()):
            if hasattr(panel, 'refresh_for_undo'):
                panel.refresh_for_undo()
            elif hasattr(panel, 'refresh'):
                panel.refresh()
    
    # ==================== File Operations ====================
    
    def _on_new_project(self):
        if self.node_editor_window:
            self.node_editor_window.scene.flush_pending_position_saves()
        
        # Close all block panels from previous project
        self._close_all_block_panels()
        
        # Create untitled project (name will be set on first save)
        result = self.facade.create_project("Untitled", save_directory=None)
        if result.success:
            # Clear undo history for new project
            self.undo_stack.clear()
            self.statusBar().showMessage("Created new project", 3000)
            self._update_ui_state()
            self.node_editor_window.refresh_and_center()
            # Update menu after facade has project ID set
            self._update_block_panels_menu()
        else:
            QMessageBox.warning(self, "Error", result.message)
    
    def _on_open_project(self):
        if self.node_editor_window:
            self.node_editor_window.scene.flush_pending_position_saves()
        
        start_dir = app_settings.get_dialog_path("open_project")
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Project", start_dir, "EchoZero Projects (*.ez);;All Files (*)")
        
        if filename:
            # Close all block panels from previous project
            self._close_all_block_panels()
            
            app_settings.set_dialog_path("open_project", filename)
            result = self.facade.load_project(filename)
            if result.success:
                # Clear undo history for loaded project
                self.undo_stack.clear()
                self._update_ui_state()
                self.node_editor_window.refresh_and_center()
                # Update menu after facade has project ID set
                self._update_block_panels_menu()
                
                blocks = self.facade.list_blocks()
                count = len(blocks.data) if blocks.success and blocks.data else 0
                self.statusBar().showMessage(f"Loaded: {filename} ({count} blocks)", 3000)
            else:
                QMessageBox.warning(self, "Error", result.message)
    
    def _save_project_impl(self, file_path: str = None, project_name: str = None):
        """
        Unified project save implementation.
        
        Args:
            file_path: Optional file path for "Save As" (if None, uses current project path)
            project_name: Optional project name for "Save As" (if None, uses current project name)
        
        Returns:
            CommandResult from save operation
        """
        # Save dock layout state when project is saved (for better persistence)
        self._save_all_window_state()
        
        # Perform the save operation
        if file_path and project_name:
            # Save As
            result = self.facade.save_project_as(file_path, project_name)
        else:
            # Save (use current project)
            result = self.facade.save_project()
        
        return result

    def _finish_save_feedback(self):
        """
        Clean up shared save UI state.
        """
        if self.save_progress_dialog:
            self.save_progress_dialog.close()
            self.save_progress_dialog.deleteLater()
            self.save_progress_dialog = None
        if QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

        self.action_save_project.setEnabled(True)
        self.action_save_project_as.setEnabled(True)

    def _on_save_completed(self, result, success_message: str, success_callback=None):
        """Handle completion signal from async save thread."""
        self._finish_save_feedback()
        self.save_thread = None

        if result and getattr(result, "success", False):
            self.statusBar().showMessage(success_message)
            if callable(success_callback):
                success_callback()
        else:
            message = getattr(result, "message", "Save failed")
            QMessageBox.warning(self, "Error", message)

    def _on_save_failed(self, error_message: str, detailed_errors: list):
        """Handle exception signal from async save thread."""
        self._finish_save_feedback()
        self.save_thread = None

        if detailed_errors:
            Log.error(f"Save failed: {error_message} | details: {detailed_errors}")
        QMessageBox.warning(self, "Error", error_message or "Save failed")

    def _start_async_save(self, save_func, success_message: str, success_callback=None):
        """
        Start project save in a worker thread with modal busy feedback.
        """
        if self.save_thread and self.save_thread.isRunning():
            self.statusBar().showMessage("Save already in progress...", 3000)
            return

        self.statusBar().showMessage("Saving project...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        dialog = QProgressDialog("Saving project...", None, 0, 0, self)
        dialog.setWindowTitle("Saving")
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setRange(0, 0)  # Busy indicator
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        QApplication.processEvents()
        self.save_progress_dialog = dialog

        self.action_save_project.setEnabled(False)
        self.action_save_project_as.setEnabled(False)

        self.save_thread = SaveProjectThread(save_func, self)
        self.save_thread.save_complete.connect(
            lambda result: self._on_save_completed(result, success_message, success_callback)
        )
        self.save_thread.save_failed.connect(self._on_save_failed)
        self.save_thread.start()
    
    def _on_save_project(self):
        """Handle Save Project action (Cmd+S)"""
        # Check if project is untitled - if so, trigger Save As instead
        current_project_id = self.facade.get_current_project_id()
        if current_project_id:
            try:
                project = self.facade.project_service.load_project(current_project_id)
                if project and project.is_untitled():
                    # Project is untitled, trigger Save As dialog
                    self._on_save_project_as()
                    return
            except Exception as e:
                # If we can't check, try regular save and let it handle the error
                Log.debug(f"Could not check if project is untitled: {e}")
        
        # Regular save for saved projects
        self._start_async_save(self._save_project_impl, "Project saved")
    
    def _on_save_project_as(self):
        """Handle Save Project As action"""
        start_dir = app_settings.get_dialog_path("save_project")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", start_dir, "EchoZero Projects (*.ez);;All Files (*)")
        
        if filename:
            app_settings.set_dialog_path("save_project", filename)
            from pathlib import Path
            path = Path(filename)
            self._start_async_save(
                lambda: self._save_project_impl(str(path.parent), path.stem),
                f"Saved: {filename}",
                self._update_ui_state
            )
    
    def _on_open_settings(self):
        """Open the global settings dialog"""
        from ui.qt_gui.widgets.settings_dialog import SettingsDialog
        
        # Get app_settings_manager from facade
        app_settings_manager = self.facade.app_settings if hasattr(self.facade, 'app_settings') else None
        dialog = SettingsDialog(self, app_settings_manager=app_settings_manager)
        dialog.settings_changed.connect(self._on_settings_changed)
        dialog.visual_defaults_reset.connect(self._on_reset_visual_defaults)
        dialog.exec()
    
    def _on_reset_visual_defaults(self):
        """Reset all editor/timeline visualization settings to defaults."""
        from ui.qt_gui.widgets.timeline.core import TimelineWidget
        # Find all TimelineWidget instances and reset their settings panels
        for tw in self.findChildren(TimelineWidget):
            if hasattr(tw, '_settings_panel') and tw._settings_panel:
                tw._settings_panel._reset_to_defaults()
    
    def _on_settings_changed(self):
        """Handle settings changes - refresh UI if needed"""
        # Apply theme if it changed
        self._apply_theme()
        
        # Update undo stack limit
        max_undo = app_settings.get("max_undo_steps", 50)
        self.undo_stack.setUndoLimit(max_undo)
        
        # Refresh node editor if grid settings changed
        if self.node_editor_window:
            self.node_editor_window.refresh()
        self.statusBar().showMessage("Settings saved", 3000)
    
    def _apply_theme(self):
        """
        Apply the current theme to the entire application.
        
        Three-tier approach for instant, complete UI refresh:
        
        1. **QPalette** -- updates all widgets using default palette roles.
        2. **QApplication.setStyleSheet()** -- the global stylesheet covers
           ~80% of widgets (buttons, combos, inputs, tables, etc.).  Qt
           re-evaluates it for every widget when it changes.
        3. **theme_changed signal** -- emitted by ``Colors.apply_theme()``,
           notifies panels that hold variant/context-specific local styles
           (primary buttons, warning banners, etc.) so they re-apply them.
        
        Custom-painted widgets (node editor blocks, timeline items) refresh
        via ``update()`` since they read ``Colors.X`` at paint time.
        """
        from PyQt6.QtGui import QPalette
        
        # 1. Sync design system globals from settings.
        #    This also syncs TimelineStyle and emits theme_changed for Tier 3.
        Colors.apply_theme()
        
        # 2. Update QApplication palette (Tier 1)
        app = QApplication.instance()
        if app:
            palette = app.palette()
            palette.setColor(QPalette.ColorRole.Window, Colors.BG_DARK)
            palette.setColor(QPalette.ColorRole.WindowText, Colors.TEXT_PRIMARY)
            palette.setColor(QPalette.ColorRole.Base, Colors.BG_MEDIUM)
            palette.setColor(QPalette.ColorRole.AlternateBase, Colors.BG_LIGHT)
            palette.setColor(QPalette.ColorRole.ToolTipBase, Colors.BG_MEDIUM)
            palette.setColor(QPalette.ColorRole.ToolTipText, Colors.TEXT_PRIMARY)
            palette.setColor(QPalette.ColorRole.Text, Colors.TEXT_PRIMARY)
            palette.setColor(QPalette.ColorRole.Button, Colors.BG_MEDIUM)
            palette.setColor(QPalette.ColorRole.ButtonText, Colors.TEXT_PRIMARY)
            palette.setColor(QPalette.ColorRole.BrightText, Colors.ACCENT_RED)
            palette.setColor(QPalette.ColorRole.Link, Colors.ACCENT_BLUE)
            palette.setColor(QPalette.ColorRole.Highlight, Colors.ACCENT_BLUE)
            palette.setColor(QPalette.ColorRole.HighlightedText, Colors.TEXT_PRIMARY)
            app.setPalette(palette)
        
        # 3. Regenerate and apply global stylesheet on QApplication (Tier 2).
        #    This single call makes Qt re-evaluate every widget's style.
        self._apply_dock_styling()
        
        # 4. Force repaint for custom-painted widgets (node editor, timeline).
        if hasattr(self, 'node_editor_window') and self.node_editor_window:
            if hasattr(self.node_editor_window, 'refresh'):
                self.node_editor_window.refresh()
        
        # Flush the event loop so paint events are processed synchronously
        QApplication.processEvents()
    
    # ==================== Execution ====================
    
    def _on_execute_single_block(self, block_id: str):
        if getattr(self, "_execution_in_progress", False):
            QMessageBox.warning(self, "Already Executing", "Execution already in progress.")
            return

        block = self.facade.block_service.get_block(
            getattr(self.facade, "current_project_id", "") or "", block_id
        )
        if not block:
            block = self.facade.block_service.find_by_name(
                getattr(self.facade, "current_project_id", "") or "", block_id
            )
        self._current_run_block_name = block.name if block else block_id

        self._execution_in_progress = True
        self.statusBar().showMessage(f"Executing: {self._current_run_block_name}...")
        self._execution_log_lines = []
        self._append_execution_log(f"Starting: {self._current_run_block_name}")
        self._execution_set_running_ui(True)

        # Unified execution path: RunBlockThread always calls facade.execute_block,
        # which honors use_subprocess_runner (in-process vs subprocess) for both
        # single-block Run and setlist automation.
        self._start_run_block_thread(block_id)

    def _start_run_block_thread(self, block_id: str):
        """Run block via RunBlockThread. Facade handles in-process vs subprocess."""
        self.execution_thread = RunBlockThread(self.facade, block_id, parent=self)
        name = getattr(self, "_current_run_block_name", block_id)
        self.execution_thread.execution_started.connect(
            lambda: self.statusBar().showMessage(f"Executing: {name}...")
        )
        self.execution_thread.execution_complete.connect(self._on_thread_complete)
        self.execution_thread.execution_failed.connect(self._on_thread_failed)
        self.execution_thread.finished.connect(self._on_run_block_thread_finished)
        self.execution_thread.start()

    def _on_run_block_thread_finished(self):
        """Called when RunBlockThread finishes; clear flag and refresh UI."""
        self._execution_in_progress = False
        self._execution_set_running_ui(False)
        if getattr(self, "_execution_panel_progress", None):
            self._execution_panel_progress.setValue(100)
        self._on_thread_finished()

    def _on_thread_complete(self, success):
        msg = "Execution completed" if success else "Execution completed with errors"
        self.statusBar().showMessage(msg)
    
    def _on_thread_failed(self, error_message, detailed_errors=None):
        """
        Handle execution failure with detailed error information.
        
        Args:
            error_message: Main error message
            detailed_errors: List of detailed error messages (optional for backward compatibility)
        """
        # Create error dialog with detailed information
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Execution Error")
        msg_box.setText(error_message)
        
        # Check if this is a filter error and extract error details
        is_filter_error = False
        filter_error_details = None
        
        if detailed_errors:
            for error_detail in detailed_errors:
                if isinstance(error_detail, dict) and error_detail.get("error_type") == "FilterError":
                    is_filter_error = True
                    filter_error_details = error_detail
                    break
        
        # Add detailed errors if available
        if detailed_errors and len(detailed_errors) > 0:
            # Format detailed errors for display (skip dict entries, format strings only)
            error_strings = [
                str(e) for e in detailed_errors
                if isinstance(e, str)
            ]
            if error_strings:
                detailed_text = "\n\n".join(error_strings)
                msg_box.setDetailedText(detailed_text)
        
        # Add "Open Filter Dialog" button for filter errors
        if is_filter_error and filter_error_details:
            open_filter_btn = msg_box.addButton(
                "Open Filter Dialog", QMessageBox.ButtonRole.ActionRole
            )
            
            # Update status bar with filter error message
            block_name = filter_error_details.get("block_name", "Unknown")
            port_name = filter_error_details.get("port_name", "unknown")
            if hasattr(self.statusBar(), 'set_filter_error'):
                self.statusBar().set_filter_error(block_name, port_name)
            
            msg_box.exec()
            
            if msg_box.clickedButton() == open_filter_btn:
                block_id = filter_error_details.get("block_id")
                if block_id:
                    self._open_filter_dialog_for_block(block_id)
        else:
            msg_box.exec()
            self.statusBar().showMessage("Execution failed")
    
    def _open_filter_dialog_for_block(self, block_id: str):
        """Open filter dialog for a block"""
        from ui.qt_gui.dialogs.data_filter_dialog import DataFilterDialog
        dialog = DataFilterDialog(block_id, self.facade, parent=self)
        dialog.exec()
    
    def _on_thread_finished(self):
        self.node_editor_window.refresh()
        
        # Ensure editor panels refresh after execution completes
        # BlockUpdated events should handle this, but ensure UI updates if panel is open
        # We identify editor panels by checking for editor-specific methods
        from PyQt6.QtCore import QTimer
        
        def refresh_editor_panels():
            """Refresh any open editor panels to ensure they show latest data"""
            for block_id, panel in list(self.open_panels.items()):
                # Check if this is an editor panel by looking for editor-specific methods
                if hasattr(panel, '_load_owned_data') and hasattr(panel, '_load_audio_from_local_state'):
                    try:
                        # Force refresh by reloading owned data
                        panel._load_owned_data()
                        panel._load_audio_from_local_state()
                        if hasattr(panel, 'refresh'):
                            panel.refresh()
                        Log.debug(f"MainWindow: Refreshed editor panel for block {block_id} after execution")
                    except Exception as e:
                        Log.warning(f"MainWindow: Failed to refresh editor panel for block {block_id}: {e}")
        
        # Use QTimer to ensure refresh happens after event processing completes
        QTimer.singleShot(100, refresh_editor_panels)
    
    def _on_cancel_execution(self):
        if getattr(self, "execution_thread", None) and self.execution_thread.isRunning():
            self.execution_thread.request_cancel()
            self.progress_bar.hide_progress()
            QMessageBox.information(self, "Cancelled", "Execution will stop after current block.")
        else:
            self.progress_bar.hide_progress()

    def _is_execution_active(self) -> bool:
        """Return True when any block execution path is currently running."""
        if getattr(self, "_execution_in_progress", False):
            return True
        thread = getattr(self, "execution_thread", None)
        if thread and thread.isRunning():
            return True
        return False
    
    # ==================== Progress ====================
    
    def _on_progress_started(self, event):
        data = event.data
        total = data.get('block_count', 0)
        msg = "Executing block..." if data.get('block_id') else f"Executing ({total} blocks)..."
        self.progress_start_signal.emit(msg, 0, total)
    
    def _on_progress_update(self, event):
        data = event.data
        msg = f"Processing {data.get('block_name', 'block')}..."
        self.progress_update_signal.emit(msg, data.get('current', 0), data.get('total', 100))
    
    def _on_progress_completed(self, event):
        self.progress_complete_signal.emit(event.data.get('executed_count', 0))
    
    def _on_progress_failed(self, event):
        self.progress_error_signal.emit(event.data.get('error', 'Unknown error'))
    
    def _on_subprocess_progress(self, event):
        """Handle SubprocessProgress events from blocks (in-process thread execution)."""
        try:
            from src.utils.message import Log
            data = event.data or {}
            message = data.get('message', '...')
            percentage = data.get('percentage', 0)
            Log.debug(f"MainWindow: Received SubprocessProgress - {message} ({percentage}%)")
            self._append_execution_log(f"{percentage}% - {message}")
            if getattr(self, "_execution_panel_progress", None):
                self._execution_panel_progress.setValue(percentage)
            name = getattr(self, "_current_run_block_name", None)
            display_msg = f"{name}: {message}" if name else message
            self.subprocess_progress_signal.emit(display_msg, percentage)
        except Exception as e:
            from src.utils.message import Log
            Log.error(f"MainWindow: Error handling SubprocessProgress: {e}")
    
    def _update_progress_start(self, msg, current, total):
        try:
            self.progress_bar.show_progress(msg, current, total)
        except Exception as e:
            Log.error(f"Progress error: {e}")
    
    def _update_progress(self, msg, current, total):
        try:
            self.progress_bar.show_progress(msg, current, total)
        except Exception as e:
            Log.error(f"Progress error: {e}")
    
    def _update_progress_complete(self, count):
        try:
            self.progress_bar.set_complete(count)
        except Exception as e:
            Log.error(f"Progress error: {e}")
    
    def _update_progress_error(self, msg):
        try:
            self.progress_bar.set_error(msg)
            QTimer.singleShot(5000, self.progress_bar.hide_progress)
        except Exception as e:
            Log.error(f"Progress error: {e}")
    
    def _update_subprocess_progress(self, msg, pct):
        """Update progress bar with subprocess progress (called from Qt signal)"""
        try:
            from src.utils.message import Log
            Log.debug(f"MainWindow: Updating progress bar - {msg} ({pct}%)")
            self.progress_bar.show_progress(msg, pct, 100)
        except Exception as e:
            from src.utils.message import Log
            Log.error(f"Progress error: {e}")
    
    def _on_select_all(self):
        """Select all blocks in node editor"""
        if self.node_editor_window and hasattr(self.node_editor_window, 'scene'):
            self.node_editor_window.scene.select_all_blocks()
    
    def _on_show_command_history(self):
        """Show the command history dialog"""
        from ui.qt_gui.widgets.command_history_dialog import CommandHistoryDialog
        
        # Check if dialog already exists and is visible
        if hasattr(self, '_command_history_dialog') and self._command_history_dialog:
            self._command_history_dialog.show()
            self._command_history_dialog.raise_()
            self._command_history_dialog.activateWindow()
            return
        
        # Create new dialog
        self._command_history_dialog = CommandHistoryDialog(self.undo_stack, self)
        self._command_history_dialog.show()
    
    def _on_about(self):
        QMessageBox.about(self, "About EchoZero",
            "<h3>EchoZero</h3>"
            "<p>Modular audio processing pipeline builder</p>"
            "<p>Version 0.1.0</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>All panels dockable and floatable</li>"
            "<li>Batch processing for multiple files</li>"
            "<li>Visual node-based editor</li>"
            "<li>Drag-and-drop connections</li>"
            "</ul>")
    
    # ==================== Autoload & Session ====================
    
    def _try_autoload_project(self):
        """Attempt to autoload the most recent project on startup. Returns True if project was loaded."""
        # Check if autoload is enabled (use AppSettingsManager from facade)
        if not (hasattr(self.facade, 'app_settings') and self.facade.app_settings):
            Log.info("App settings not available, skipping autoload")
            return False
        
        if not self.facade.app_settings.restore_last_project:
            Log.info("Autoload disabled in settings, skipping")
            return False
        
        # Check if a project is already loaded (shouldn't be, but safety check)
        if self.facade.get_current_project_id():
            Log.info("Project already loaded, skipping autoload")
            return False
        
        try:
            # Get most recent project from recent projects store
            from src.utils.recent_projects import RecentProjectsStore
            import os
            
            store = RecentProjectsStore()
            recent = store.list_recent(limit=1)
            
            if not recent:
                Log.info("No recent projects to autoload")
                return False
            
            project_entry = recent[0]
            project_file = project_entry.get("project_file")
            
            if not project_file:
                Log.info("Recent project has no file path")
                return False
            
            # Verify file exists
            if not os.path.exists(project_file):
                Log.info(f"Recent project file not found: {project_file}")
                self.statusBar().showMessage("Recent project file not found", 5000)
                return False
            
            # Load the project
            Log.info(f"Autoloading project: {project_file}")
            self.statusBar().showMessage("Loading recent project...")
            
            result = self.facade.load_project(project_file)
            
            if result.success:
                # Clear undo history for loaded project
                self.undo_stack.clear()
                self._update_ui_state()
                # Don't refresh node editor here - it will be refreshed in _initialize_and_restore()
                # This prevents duplicate refreshes during startup
                self._update_block_panels_menu()

                blocks = self.facade.list_blocks()
                count = len(blocks.data) if blocks.success and blocks.data else 0
                project_name = project_entry.get("name", "Unknown")
                self.statusBar().showMessage(f"Loaded: {project_name} ({count} blocks)", 5000)
                Log.info(f"Autoloaded project: {project_name}")

                # Restore session state (open panels, zoom, etc.) after project is loaded
                # State restoration happens in _initialize_and_restore()
                # No need to call separately here
                return True  # FIX: Actually return True so _create_saved_panels gets called
            else:
                Log.warning(f"Failed to autoload project: {result.message}")
                
                # Ask user if they want to create a new project
                reply = QMessageBox.question(
                    self,
                    "Autoload Failed",
                    f"Failed to load recent project:\n\n{result.message}\n\nWould you like to create a new project?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Create a new untitled project
                    Log.info("User chose to create new project after autoload failure")
                    new_result = self.facade.create_project("Untitled", save_directory=None)
                    if new_result.success:
                        # Clear undo history for new project
                        self.undo_stack.clear()
                        self._update_ui_state()
                        if self.node_editor_window:
                            self.node_editor_window.refresh_and_center()
                        self._update_block_panels_menu()
                        self.statusBar().showMessage("Created new project", 3000)
                        Log.info("Created new untitled project after autoload failure")
                        return True  # Return True so _create_saved_panels gets called
                    else:
                        QMessageBox.warning(
                            self,
                            "Error",
                            f"Failed to create new project: {new_result.message}"
                        )
                        Log.error(f"Failed to create new project: {new_result.message}")
                else:
                    Log.info("User chose not to create new project after autoload failure")
                    self.statusBar().showMessage("No project loaded", 5000)
                
        except Exception as e:
            Log.error(f"Autoload error: {e}")
            
            # Ask user if they want to create a new project
            reply = QMessageBox.question(
                self,
                "Autoload Error",
                f"An error occurred while loading the recent project:\n\n{str(e)}\n\nWould you like to create a new project?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Create a new untitled project
                try:
                    Log.info("User chose to create new project after autoload exception")
                    new_result = self.facade.create_project("Untitled", save_directory=None)
                    if new_result.success:
                        self.undo_stack.clear()
                        self._update_ui_state()
                        if self.node_editor_window:
                            self.node_editor_window.refresh_and_center()
                        self._update_block_panels_menu()
                        self.statusBar().showMessage("Created new project", 3000)
                        Log.info("Created new untitled project after autoload exception")
                        return True
                    else:
                        QMessageBox.warning(
                            self,
                            "Error",
                            f"Failed to create new project: {new_result.message}"
                        )
                except Exception as create_error:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Failed to create new project: {str(create_error)}"
                    )
                    Log.error(f"Failed to create new project after autoload error: {create_error}")
            else:
                Log.info("User chose not to create new project after autoload exception")
                self.statusBar().showMessage("No project loaded", 5000)
        
        return False  # Return False if user declined or creation failed

    def _prompt_for_project(self) -> bool:
        """
        Prompt user to create a new project or open an existing one.
        Returns True if a project was loaded/created, False otherwise.
        """
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Welcome to EchoZero")
        dialog.setMinimumWidth(400)
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        
        # Welcome message
        welcome_label = QLabel(
            "No project is currently open.\n\n"
            "What would you like to do?"
        )
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)
        
        # Buttons
        btn_new = QPushButton("New Project")
        btn_new.setDefault(True)
        btn_new.clicked.connect(lambda: dialog.done(1))
        layout.addWidget(btn_new)
        
        btn_open = QPushButton("Open Project...")
        btn_open.clicked.connect(lambda: dialog.done(2))
        layout.addWidget(btn_open)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(lambda: dialog.done(0))
        layout.addWidget(btn_cancel)
        
        # Show dialog
        result = dialog.exec()
        
        if result == 1:  # New Project
            # Create new untitled project
            new_result = self.facade.create_project("Untitled", save_directory=None)
            if new_result.success:
                self.undo_stack.clear()
                self._update_ui_state()
                if self.node_editor_window:
                    self.node_editor_window.refresh_and_center()
                self._update_block_panels_menu()
                self.statusBar().showMessage("Created new project", 3000)
                Log.info("Created new project from welcome dialog")
                return True
            else:
                QMessageBox.warning(self, "Error", f"Failed to create project: {new_result.message}")
                return False
        
        elif result == 2:  # Open Project
            # Trigger open project dialog
            self._on_open_project()
            # Check if project was actually loaded
            if self.facade.get_current_project_id():
                return True
            return False
        
        else:  # Cancel
            Log.info("User cancelled project selection")
            return False

    def _update_ui_for_no_project(self):
        """Disable project-dependent UI elements when no project is loaded"""
        # Disable save actions
        self.action_save_project.setEnabled(False)
        self.action_save_project_as.setEnabled(False)
        
        # Disable execution-related actions if they exist
        # (Add more as needed)
        
        # Update window title
        self.setWindowTitle("EchoZero - No Project")
        
        # Show empty workspace message
        self._update_empty_workspace_visibility()

    def _save_session(self):
        """Save session-specific state (selected blocks, zoom, panels, etc.)"""
        try:
            # Save node editor state
            if self.node_editor_window:
                selected = self.node_editor_window.scene.selected_blocks()
                if selected:
                    self.facade.set_session_state("selected_block", selected[0])
                self.facade.set_session_state("zoom_level", self.node_editor_window.view.zoom_level)
                self.facade.set_session_state("viewport_center", self.node_editor_window.view.get_viewport_center())
            
            # Save block panel states (positions, geometry, floating status)
            # These are excluded from Qt's saveState and restored separately
            if self.open_panels:
                self.facade.set_session_state("open_panels", list(self.open_panels.keys()))
                
                panel_states = {}
                for block_id, panel in self.open_panels.items():
                    geom = panel.geometry()
                    panel_states[block_id] = {
                        'floating': panel.isFloating(),
                        'floating_geometry': {
                            'x': geom.x(),
                            'y': geom.y(),
                            'width': geom.width(),
                            'height': geom.height()
                        } if panel.isFloating() else None,
                        'visible': panel.isVisible()
                    }
                if panel_states:
                    self.facade.set_session_state("panel_states", panel_states)
            
            # Window layout (main windows, tab groups) is saved separately via _save_all_window_state()
            # which uses Qt's native saveState()
        except Exception as e:
            Log.error(f"Session save error: {e}")
    
    def _restore_session_state(self):
        """Restore session-specific state (selected blocks, zoom, etc.)"""
        try:
            if not self.facade.get_current_project_id():
                return
            
            # Restore node editor state
            result = self.facade.get_session_state("zoom_level")
            if result.success and result.data and self.node_editor_window:
                if isinstance(result.data, (int, float)):
                    self.node_editor_window.view.set_zoom_level(result.data)
            
            result = self.facade.get_session_state("viewport_center")
            if result.success and result.data and self.node_editor_window:
                if isinstance(result.data, dict):
                    self.node_editor_window.view.center_on_point(
                        result.data.get("x", 0), result.data.get("y", 0))
            
            result = self.facade.get_session_state("selected_block")
            if result.success and result.data and self.node_editor_window:
                self.node_editor_window.scene.select_block(result.data)
        except Exception as e:
            Log.error(f"Session state restore error: {e}")
    
    def showEvent(self, event: QShowEvent):
        """Called when window is shown - initialize and restore state"""
        super().showEvent(event)
        
        # Initialize once when window is first shown
        # This ensures UI is fully ready before we restore state
        # Using QTimer.singleShot(0) ensures it runs after the current event loop
        # This is cleaner than a fixed delay and ensures UI is ready
        if not self._initialization_complete:
            self._initialization_complete = True
            QTimer.singleShot(0, self._initialize_and_restore)
    
    def closeEvent(self, event):
        Log.info("Closing main window...")
        
        # Mark that we're closing - prevents _on_panel_closed from overwriting our save
        self._closing = True

        if self._is_execution_active():
            reply = QMessageBox.question(
                self,
                "Execution in Progress",
                "A block execution is currently running.\n\nAre you sure you want to close EchoZero?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._closing = False
                event.ignore()
                return
        
        if self.node_editor_window:
            self.node_editor_window.scene.flush_pending_position_saves()
        
        self._save_all_window_state()
        self._save_session()
        
        for panel in list(self.open_panels.values()):
            try:
                panel.close()
            except:
                pass
        
        if getattr(self, "save_thread", None) and self.save_thread.isRunning():
            self.save_thread.wait(2000)
        if getattr(self, "execution_thread", None) and self.execution_thread.isRunning():
            self.execution_thread.wait(2000)
        
        if self.setlist_window:
            self.setlist_window.cleanup()
        
        event.accept()
    
    @pyqtSlot(str)
    def _on_setlist_song_switched(self, song_id: str):
        """Handle signal when a song is switched in the Setlist view."""
        Log.info(f"MainWindow: Setlist song switched to {song_id}. Refreshing UI.")
        
        # Refresh Node Editor to show new data
        if self.node_editor_window:
            self.node_editor_window.refresh()
        
        # Refresh block panels if they are open (they'll pick up new data via events)
        for panel_id, panel_dock in self.dock_manager._docks.items():
            if panel_id.startswith("block_panel_") and panel_dock.isVisible():
                panel_widget = panel_dock.widget()
                if hasattr(panel_widget, 'refresh'):
                    panel_widget.refresh()

"""
Main Window

Professional workspace with modular dockable panels.
All panels (Node Editor, Setlist, Execution, Block Panels) are equal-citizen
CDockWidgets managed by PyQt6Ads CDockManager. Panels can be freely docked,
tabbed, split, floated, and pinned to auto-hide sidebars (VSCode style).
"""
import json
import os
import random
import time
from pathlib import Path

import PyQt6Ads as ads
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QMenuBar, QMenu, QToolBar, QStatusBar,
    QMessageBox, QFileDialog, QSizePolicy, QApplication,
    QLabel, QPushButton, QFrame, QProgressDialog,
    QInputDialog, QToolButton, QAbstractButton,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QShowEvent, QAction, QKeySequence, QUndoStack, QPixmap

from src.application.api.application_facade import ApplicationFacade
from src.application.commands import CommandBus
from src.utils.message import Log
from src.utils.settings import app_settings
from ui.qt_gui.design_system import get_stylesheet, get_application_palette, force_style_refresh, apply_ui_font, Colors
from ui.qt_gui.core.workspace_manager import WorkspaceManager
from ui.qt_gui.core.run_block_thread import RunBlockThread
from ui.qt_gui.core.save_project_thread import SaveProjectThread


def _load_welcome_phrases() -> list[str]:
    path = Path(__file__).resolve().parent.parent.parent / "assets" / "welcome_phrases.txt"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [l for l in lines if l.strip()] or ["Welcome to EchoZero."]
    except OSError:
        return ["Welcome to EchoZero."]


_WELCOME_PHRASES = _load_welcome_phrases()


class MainWindow(QMainWindow):
    """
    Professional workspace with modular dockable panels (VSCode style).
    
    All panels are equal-citizen CDockWidgets managed by PyQt6Ads CDockManager.
    Supports tabbed views, split panes, floating windows, auto-hide sidebars,
    close buttons on every tab, focus highlighting, and named perspectives.
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
        
        # Resolve application mode (production vs developer)
        from src.application.services.app_mode_manager import AppModeManager, AppMode
        self._mode_manager: AppModeManager = getattr(facade, 'app_mode_manager', None)
        if self._mode_manager is None:
            self._mode_manager = AppModeManager(AppMode.DEVELOPER)
        
        self.setWindowTitle("EZ")
        self.setMinimumSize(800, 600)
        
        # Apply theme and design system
        self._apply_theme()
        
        # PyQt6Ads CDockManager replaces Qt's native dock system entirely.
        # Config flags must be set before CDockManager is instantiated.
        ads.CDockManager.setConfigFlag(ads.CDockManager.eConfigFlag.OpaqueSplitterResize, True)
        ads.CDockManager.setConfigFlag(ads.CDockManager.eConfigFlag.FocusHighlighting, False)
        ads.CDockManager.setConfigFlag(ads.CDockManager.eConfigFlag.AllTabsHaveCloseButton, True)
        ads.CDockManager.setConfigFlag(ads.CDockManager.eConfigFlag.DockAreaHasCloseButton, True)
        ads.CDockManager.setConfigFlag(ads.CDockManager.eConfigFlag.MiddleMouseButtonClosesTab, True)
        ads.CDockManager.setConfigFlag(ads.CDockManager.eConfigFlag.DockAreaHasUndockButton, True)
        ads.CDockManager.setConfigFlag(ads.CDockManager.eConfigFlag.EqualSplitOnInsertion, True)
        ads.CDockManager.setConfigFlag(ads.CDockManager.eConfigFlag.HideSingleCentralWidgetTitleBar, True)
        self.dock_manager = ads.CDockManager(self)
        # Disable CDockManager's internal stylesheet so our app stylesheet applies.
        # Without this, ADS internal styles override our theme (tabs stay wrong color).
        self.dock_manager.setStyleSheet("")

        self._register_minimal_close_icons()

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
        
        # Workspace Manager (unified state management via CDockManager)
        self.workspace = WorkspaceManager(self, self.dock_manager, facade)
        
        # Setup UI
        self._create_actions()
        self._create_menus()
        self._create_toolbars()
        self._create_dock_widgets()
        self._create_progress_bar()
        self._create_status_bar()
        
        if self.block_panels_menu:
            self._update_block_panels_menu()
        
        # Subscribe to events
        self._subscribe_to_events()
        self._update_ui_state()
        
        # Mode visibility is applied in _initialize_and_restore after
        # the project and layout have been loaded.
        if self._mode_manager:
            self._mode_manager.mode_changed.connect(self._on_mode_changed_external)
        
        # Track if initialization has run (will run in showEvent)
        self._initialization_complete = False
        
        # Track if we're in the close sequence (to prevent double-saves)
        self._closing = False
        
        Log.info(f"Main window created (mode={self._mode_manager.mode.value})")
    
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
        self.action_settings.setMenuRole(QAction.MenuRole.PreferencesRole)  # macOS: appears in app menu
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
        
        # View toggle actions are created from CDockWidget.toggleViewAction() 
        # in _create_dock_widgets() after docks exist. Placeholders for shortcuts:
        self.action_view_node_editor = None
        self.action_view_setlist = None
        self.action_view_execution = None
        
        self.action_reset_layout = QAction("&Reset to Default Layout", self)
        self.action_reset_layout.triggered.connect(self._reset_dock_layout)
        
        self.action_command_history = QAction("Command &History...", self)
        self.action_command_history.setShortcut(QKeySequence("Ctrl+H"))
        self.action_command_history.triggered.connect(self._on_show_command_history)
        
    
    def _register_minimal_close_icons(self):
        """Replace default boxed-X dock tab close icons with a minimal thin X."""
        from PyQt6.QtGui import QIcon, QPainter, QPen, QPixmap

        dpr = self.devicePixelRatioF()
        logical = 16
        px = int(logical * dpr)
        pixmap = QPixmap(px, px)
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        pen = QPen(Colors.TEXT_SECONDARY)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        m = 4.5
        end = logical - m
        painter.drawLine(int(m), int(m), int(end), int(end))
        painter.drawLine(int(end), int(m), int(m), int(end))
        painter.end()

        icon = QIcon(pixmap)
        ip = ads.CDockManager.iconProvider()
        ip.registerCustomIcon(ads.eIcon.TabCloseIcon, icon)
        ip.registerCustomIcon(ads.eIcon.DockAreaCloseIcon, icon)

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
        
        # Developer-only: Export as Production Template
        self.action_export_template = QAction("Export as Production &Template...", self)
        self.action_export_template.triggered.connect(self._on_export_template)
        file_menu.addAction(self.action_export_template)
        
        file_menu.addSeparator()
        file_menu.addAction(self.action_undo)
        file_menu.addAction(self.action_redo)
        file_menu.addSeparator()
        file_menu.addAction(self.action_settings)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)
        
        # Edit menu (standard location for undo/redo too)
        self._edit_menu = menubar.addMenu("&Edit")
        edit_menu = self._edit_menu
        edit_menu.addAction(self.action_undo)
        edit_menu.addAction(self.action_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_settings)
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_select_all)
        
        # Window menu -- view toggle actions are inserted by _create_dock_widgets()
        window_menu = menubar.addMenu("&Window")
        self._window_menu = window_menu
        
        # Placeholder separator before block panels (view actions inserted before this)
        self._window_menu_panels_sep = window_menu.addSeparator()
        
        # Block panels submenu (dynamically updated)
        self.block_panels_menu = window_menu.addMenu("Block &Panels")
        self._update_block_panels_menu()
        
        window_menu.addSeparator()
        window_menu.addAction(self.action_command_history)
        window_menu.addSeparator()
        window_menu.addAction(self.action_reset_layout)
        
        # Layout presets submenu
        self._layout_menu = window_menu.addMenu("&Layouts")
        layout_menu = self._layout_menu
        
        action_save_preset = QAction("&Save Layout Preset...", self)
        action_save_preset.triggered.connect(self._on_save_preset)
        layout_menu.addAction(action_save_preset)
        
        self._preset_restore_menu = layout_menu.addMenu("&Restore Preset")
        self._preset_restore_menu.aboutToShow.connect(self._populate_preset_menu)
        
        layout_menu.addSeparator()
        
        action_delete_preset = QAction("&Delete Preset...", self)
        action_delete_preset.triggered.connect(self._on_delete_preset)
        layout_menu.addAction(action_delete_preset)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        action_about = QAction("&About", self)
        action_about.triggered.connect(self._on_about)
        help_menu.addAction(action_about)
    
    def _create_toolbars(self):
        """Create toolbar"""
        self._main_toolbar = self.addToolBar("Main Toolbar")
        self._main_toolbar.setMovable(False)

        self._main_toolbar.addAction(self.action_new_project)
        self._main_toolbar.addAction(self.action_open_project)
        self._main_toolbar.addAction(self.action_save_project)
        self._main_toolbar.addSeparator()
        self._main_toolbar.addAction(self.action_settings)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._main_toolbar.addWidget(spacer)

        logo_label = QLabel()
        logo_label.setContentsMargins(0, 0, 8, 0)
        logo_path = str(Path(__file__).resolve().parent.parent.parent / "assets" / "ez_logo.png")
        logo_pixmap = QPixmap(logo_path)
        if not logo_pixmap.isNull():
            scaled = logo_pixmap.scaledToHeight(11, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled)
        self._main_toolbar.addWidget(logo_label)
        
    
    def _create_dock_widgets(self):
        """Create all dockable panels via CDockManager.
        
        All real panels start hidden. A welcome dock is shown in the center
        until a project is loaded (via _show_workspace / _show_welcome).
        """
        # Welcome screen (shown when no project is loaded)
        self._welcome_dock = self._create_welcome_dock()

        # Node Editor
        self.node_editor_dock = self._create_node_editor_dock()
        self.node_editor_window = self.node_editor_dock.widget()

        # Setlist (tabbed with Node Editor in center area)
        self.setlist_dock = self._create_setlist_dock()
        self.setlist_window = self.setlist_dock.widget()

        # Execution
        self.execution_dock = self._create_execution_dock()
        if self.execution_dock:
            self.dock_manager.addDockWidget(ads.DockWidgetArea.BottomDockWidgetArea, self.execution_dock)
            self.execution_dock.toggleView(False)
            self.workspace.register_dock("execution", self.execution_dock)

        # Connect panel signals
        if self.node_editor_window:
            self.node_editor_window.block_panel_requested.connect(self.open_block_panel)
        if self.setlist_window and hasattr(self.setlist_window, 'setlist_view'):
            self.setlist_window.setlist_view.song_switched.connect(self._on_setlist_song_switched)

        # Production visibility toggles for built-in docks (title bar buttons)
        self._dock_prod_toggles = {}
        for dock in [self.node_editor_dock, self.setlist_dock, self.execution_dock]:
            if dock:
                self._add_dock_prod_toggle(dock)

        # Create view toggle actions from CDockWidget.toggleViewAction()
        if self.node_editor_dock:
            self.action_view_node_editor = self.node_editor_dock.toggleViewAction()
            self.action_view_node_editor.setShortcut(QKeySequence("Ctrl+1"))
        if self.setlist_dock:
            self.action_view_setlist = self.setlist_dock.toggleViewAction()
            self.action_view_setlist.setShortcut(QKeySequence("Ctrl+3"))
        if self.execution_dock:
            self.action_view_execution = self.execution_dock.toggleViewAction()
            self.action_view_execution.setToolTip("Show execution log and process status")

        # Insert view toggle actions into the Window menu
        if hasattr(self, '_window_menu') and self._window_menu:
            for action in [self.action_view_node_editor, self.action_view_setlist, self.action_view_execution]:
                if action:
                    self._window_menu.insertAction(self._window_menu_panels_sep, action)

        # Start in welcome state -- real panels hidden until project loads
        self._show_welcome()

        self._apply_dock_styling()
    
    def _create_node_editor_dock(self):
        """Create Node Editor dock."""
        from ui.qt_gui.node_editor.node_editor_window import NodeEditorWindow
        
        node_editor_window = NodeEditorWindow(self.facade, undo_stack=self.undo_stack)
        dock = self._create_dock("Node Editor", node_editor_window, "NodeEditorDock")
        self.dock_manager.addDockWidget(ads.DockWidgetArea.CenterDockWidgetArea, dock)
        self.workspace.register_dock("node_editor", dock)
        return dock
    
    def _create_setlist_dock(self):
        """Create Setlist dock."""
        from ui.qt_gui.views.setlist_window import SetlistWindow
        
        setlist_window = SetlistWindow(self.facade)
        dock = self._create_dock("Setlist", setlist_window, "SetlistDock")
        # Add to the same area as Node Editor so they are tabbed together
        ne_area = self.node_editor_dock.dockAreaWidget() if self.node_editor_dock else None
        if ne_area:
            self.dock_manager.addDockWidget(ads.DockWidgetArea.CenterDockWidgetArea, dock, ne_area)
        else:
            self.dock_manager.addDockWidget(ads.DockWidgetArea.CenterDockWidgetArea, dock)
        self.workspace.register_dock("setlist", dock)
        return dock

    def _create_execution_dock(self):
        """Create Execution dock (compact status bar + collapsible log)."""
        from ui.qt_gui.widgets.execution_panel import ExecutionPanel

        self._execution_panel = ExecutionPanel()
        self._execution_panel.cancel_requested.connect(self._on_cancel_execution)

        dock = self._create_dock("Execution", self._execution_panel, "ExecutionDock")
        return dock

    def _create_welcome_dock(self) -> ads.CDockWidget:
        """Create the welcome/landing screen shown when no project is loaded."""
        panel = QWidget()
        panel.setObjectName("WelcomePanel")
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = str(Path(__file__).resolve().parent.parent.parent / "assets" / "ez_logo.png")
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            scaled = pixmap.scaledToWidth(90, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled)
        else:
            logo_label.setText("EZ")
            logo_label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {Colors.TEXT_PRIMARY.name()};")
        layout.addWidget(logo_label)

        subtitle = QLabel(random.choice(_WELCOME_PHRASES))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(
            f"font-size: 14px; color: {Colors.TEXT_SECONDARY.name()}; font-style: italic;"
        )
        subtitle.setMinimumWidth(420)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self._welcome_subtitle = subtitle
        self._welcome_rotate_timer = QTimer(self)
        self._welcome_rotate_timer.timeout.connect(self._rotate_welcome_phrase)
        self._welcome_rotate_timer.start(8000)

        layout.addSpacing(12)

        btn_container = QFrame()
        btn_container.setObjectName("WelcomeButtonContainer")
        btn_container.setFixedSize(220, 82)
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(10)

        btn_new = QPushButton("New Project")
        btn_new.clicked.connect(self._on_new_project)
        btn_layout.addWidget(btn_new, stretch=1)

        btn_open = QPushButton("Open Project...")
        btn_open.clicked.connect(self._on_open_project)
        btn_layout.addWidget(btn_open, stretch=1)

        layout.addWidget(btn_container, alignment=Qt.AlignmentFlag.AlignCenter)

        dock = ads.CDockWidget("Welcome")
        dock.setObjectName("WelcomeDock")
        dock.setWidget(panel)
        dock.setFeature(ads.CDockWidget.DockWidgetFeature.DockWidgetClosable, False)
        dock.setFeature(ads.CDockWidget.DockWidgetFeature.DockWidgetMovable, False)
        dock.setFeature(ads.CDockWidget.DockWidgetFeature.DockWidgetFloatable, False)
        dock.setFeature(ads.CDockWidget.DockWidgetFeature.NoTab, True)
        self.dock_manager.addDockWidget(ads.DockWidgetArea.CenterDockWidgetArea, dock)
        return dock

    def _rotate_welcome_phrase(self):
        if hasattr(self, '_welcome_subtitle') and self._welcome_subtitle is not None:
            self._welcome_subtitle.setText(random.choice(_WELCOME_PHRASES))

    def _show_welcome(self):
        """Show welcome screen, hide all workspace docks."""
        if self._welcome_dock:
            self._welcome_dock.setFeature(ads.CDockWidget.DockWidgetFeature.NoTab, True)
            self._welcome_dock.toggleView(True)
        if hasattr(self, '_welcome_rotate_timer'):
            self._rotate_welcome_phrase()
            self._welcome_rotate_timer.start(8000)
        if self.node_editor_dock:
            self.node_editor_dock.toggleView(False)
        if self.setlist_dock:
            self.setlist_dock.toggleView(False)
        if self.execution_dock:
            self.execution_dock.toggleView(False)
    def _show_workspace(self):
        """Dismiss the welcome screen.

        Dock visibility is handled entirely by ``_apply_mode_visibility``
        which runs immediately after this in every call-site.
        """
        if self._welcome_dock:
            self._welcome_dock.toggleView(False)
        if hasattr(self, '_welcome_rotate_timer'):
            self._welcome_rotate_timer.stop()

    def _execution_set_running_ui(self, running: bool):
        """Update Execution panel for running vs idle state."""
        panel = getattr(self, "_execution_panel", None)
        if not panel:
            return
        if running:
            name = getattr(self, "_current_run_block_name", "Block")
            panel.set_running(name)
            if self.execution_dock and not self.execution_dock.isVisible():
                self.execution_dock.toggleView(True)
        else:
            panel.set_idle()

    def _append_execution_log(self, line: str, is_error: bool = False):
        """Append a line to the execution log with timestamp."""
        panel = getattr(self, "_execution_panel", None)
        if panel:
            panel.append_log(line, is_error)

    # ----- Built-in dock production visibility -----

    def _add_dock_prod_toggle(self, dock: ads.CDockWidget):
        """Add a 'PROD' toggle action to a dock's title bar."""
        key = f"prod_visible_{dock.objectName()}"
        checked = self._get_dock_prod_pref(key)

        action = QAction("PROD" if checked else "prod", dock)
        action.setCheckable(True)
        action.setChecked(checked)
        action.setToolTip("When checked, this window is available in Production Mode")
        action.triggered.connect(lambda c, k=key, a=action: self._on_dock_prod_toggled(k, c, a))
        dock.setTitleBarActions([action])
        self._dock_prod_toggles[dock.objectName()] = action

        if self._mode_manager:
            action.setVisible(self._mode_manager.is_developer)

    def _sync_dock_prod_toggle_visibility(self):
        """Show/hide all dock PROD toggles based on current mode."""
        is_dev = self._mode_manager.is_developer if self._mode_manager else True
        for action in self._dock_prod_toggles.values():
            action.setVisible(is_dev)

    def _on_dock_prod_toggled(self, pref_key: str, checked: bool, action: QAction):
        action.setText("PROD" if checked else "prod")
        prefs = getattr(self.facade, 'preferences_repo', None)
        if prefs:
            prefs.set(pref_key, checked)

    def _get_dock_prod_pref(self, pref_key: str) -> bool:
        prefs = getattr(self.facade, 'preferences_repo', None)
        if prefs:
            return bool(prefs.get(pref_key, False))
        return False

    def _is_dock_production_visible(self, dock: ads.CDockWidget) -> bool:
        """Check if a built-in dock should be visible in production mode."""
        key = f"prod_visible_{dock.objectName()}"
        return self._get_dock_prod_pref(key)

    def _create_dock(self, title: str, widget: QWidget, object_name: str) -> ads.CDockWidget:
        """Create a CDockWidget with standard features.
        
        All docks get the same feature set: movable, closable, floatable, pinnable.
        This is the single creation path for every panel in the application.
        """
        dock = ads.CDockWidget(title)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        dock.setFeature(ads.CDockWidget.DockWidgetFeature.DockWidgetClosable, True)
        dock.setFeature(ads.CDockWidget.DockWidgetFeature.DockWidgetMovable, True)
        dock.setFeature(ads.CDockWidget.DockWidgetFeature.DockWidgetFloatable, True)
        dock.setFeature(ads.CDockWidget.DockWidgetFeature.DockWidgetPinnable, True)
        dock.topLevelChanged.connect(self._on_dock_top_level_changed)
        return dock
    
    def _apply_dock_styling(self):
        """Apply global stylesheet to QApplication for instant app-wide propagation.
        
        Setting the stylesheet on QApplication (rather than MainWindow) ensures
        that floating docks, dialogs, and all child widgets inherit the theme
        automatically.  Qt re-evaluates the global stylesheet for every widget
        whenever it changes, so this single call updates the entire UI.
        """
        app = QApplication.instance()
        if app:
            apply_ui_font(app)
            app.setStyleSheet(get_stylesheet())

    # ==================== Auto-hide single-tab title bars ====================

    def _setup_auto_hide_single_tabs(self):
        """Hide dock area title bars when only one dock occupies the area.

        Connects to dock lifecycle signals so the title bar reappears
        automatically when docks are tabified together and hides again
        when a dock area is back to a single widget.
        """
        dm = self.dock_manager
        dm.dockAreaCreated.connect(self._on_area_created_for_tabs)
        dm.dockWidgetAdded.connect(self._on_dock_added_for_tabs)
        dm.dockWidgetRemoved.connect(lambda _: self._schedule_titlebar_update())
        dm.stateRestored.connect(self._schedule_titlebar_update)

        self._tb_timer = QTimer(self)
        self._tb_timer.setSingleShot(True)
        self._tb_timer.setInterval(0)
        self._tb_timer.timeout.connect(self._update_all_area_titlebars)

    def _on_area_created_for_tabs(self, area):
        self._schedule_titlebar_update()

    def _on_dock_added_for_tabs(self, dock):
        dock.viewToggled.connect(lambda _: self._schedule_titlebar_update())
        self._schedule_titlebar_update()

    def _schedule_titlebar_update(self):
        """Coalesce rapid changes into a single deferred update."""
        self._tb_timer.start()

    def _update_all_area_titlebars(self):
        """Show title bar only when a dock area contains multiple visible docks."""
        try:
            for i in range(self.dock_manager.dockAreaCount()):
                area = self.dock_manager.dockArea(i)
                if not area:
                    continue
                tb = area.titleBar()
                if not tb:
                    continue
                tb.setVisible(area.openDockWidgetsCount() > 1)
        except RuntimeError:
            pass

    def _toggle_dock_visibility(self, dock: ads.CDockWidget, checked: bool):
        """Toggle dock visibility from Window menu."""
        dock.toggleView(checked)
    
    
    def _save_all_window_state(self):
        """Save complete workspace state."""
        self.workspace.save_state()
    
    # ==================== Layout Presets ====================
    
    def _on_save_preset(self):
        """Save current layout as a named preset."""
        name, ok = QInputDialog.getText(self, "Save Layout Preset", "Preset name:")
        if ok and name.strip():
            if self.workspace.save_preset(name.strip()):
                self.statusBar().showMessage(f"Layout preset '{name.strip()}' saved", 3000)
    
    def _populate_preset_menu(self):
        """Populate the preset restore submenu with available presets."""
        self._preset_restore_menu.clear()
        presets = self.workspace.list_presets()
        if not presets:
            action = self._preset_restore_menu.addAction("(no presets)")
            action.setEnabled(False)
            return
        for name in sorted(presets):
            action = self._preset_restore_menu.addAction(name)
            action.triggered.connect(lambda checked, n=name: self._restore_preset(n))
    
    def _restore_preset(self, name: str):
        """Restore a named layout preset."""
        if self.workspace.restore_preset(name):
            self.statusBar().showMessage(f"Layout preset '{name}' restored", 3000)
    
    def _on_delete_preset(self):
        """Delete a named preset."""
        presets = self.workspace.list_presets()
        if not presets:
            QMessageBox.information(self, "No Presets", "No layout presets to delete.")
            return
        name, ok = QInputDialog.getItem(
            self, "Delete Layout Preset", "Select preset:", sorted(presets), 0, False)
        if ok and name:
            self.workspace.delete_preset(name)
            self.statusBar().showMessage(f"Layout preset '{name}' deleted", 3000)

    def _initialize_and_restore(self):
        """Complete initialization: load project, create panels, restore state."""
        try:
            is_prod = self._mode_manager.is_production if self._mode_manager else False

            project_loaded = self._try_autoload_project()
            
            if not project_loaded and is_prod:
                project_loaded = self._try_load_production_project()
            
            if project_loaded:
                self._on_project_ready()
            else:
                Log.info("No project loaded, showing welcome screen")
                self._show_welcome()
                self._update_ui_for_no_project()
                self._apply_mode_visibility()
            
        except Exception as e:
            Log.error(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_project_ready(self):
        """Transition from welcome state to full workspace after a project loads.

        Order matters: create panels first (hidden), restore saved
        positions, then apply mode visibility so non-production windows
        never flash on screen.
        """
        self._create_saved_panels()
        
        if self.node_editor_window:
            self.node_editor_window.refresh()
        
        self._restore_layout()

        self._show_workspace()
        self._apply_mode_visibility()

        self._update_ui_state()
        self._update_block_panels_menu()
        
        Log.info(f"Initialization complete. Docks: {self.workspace.get_registered_docks()}")
    
    def _restore_layout(self):
        """Restore layout from saved state, or apply default layout on first launch."""
        if self.workspace.restore_state():
            Log.info("Layout restored from saved state")
        else:
            Log.info("No saved layout, applying default")
            self._set_default_layout()
        
        self._restore_session_state()
    
    def _set_default_layout(self):
        """Apply default layout (all docks hidden; mode visibility shows the right ones)."""
        self.workspace.apply_default_layout()
    
    
    def _create_saved_panels(self):
        """Create all panels that were open when saved.
        
        IMPORTANT: Panels are created but NOT positioned here.
        Positioning happens in _restore_layout() via Qt's restoreState().
        """
        try:
            panel_ids = self.workspace.get_open_panel_ids()
            
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
        """Create a single block panel (internal helper)."""
        try:
            if not block_type:
                result = self.facade.describe_block(block_id)
                if not result.success:
                    return None
                block_type = result.data.type
            
            from ui.qt_gui.block_panels import get_panel_class
            panel_class = get_panel_class(block_type)
            if not panel_class:
                return None
            
            panel = panel_class(block_id, self.facade, parent=self)
            panel.panel_closed.connect(self._on_panel_closed)
            panel.topLevelChanged.connect(self._on_dock_top_level_changed)
            panel.setObjectName(f"BlockPanel_{block_id}")
            
            self.dock_manager.addDockWidget(ads.DockWidgetArea.RightDockWidgetArea, panel)
            # Hide immediately so the panel doesn't flash in the wrong position
            # before restoreState() places it correctly.
            panel.toggleView(False)

            # Keep menu checkmarks accurate whenever panel visibility changes.
            panel.viewToggled.connect(lambda _: self._update_block_panels_menu())

            window_id = f"block_panel_{block_id}"
            self.workspace.register_dock(window_id, panel)
            
            self.open_panels[block_id] = panel
            return panel
        except Exception as e:
            Log.warning(f"Failed to create panel for {block_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _reset_dock_layout(self):
        """Reset to default layout."""
        self._set_default_layout()
        self.statusBar().showMessage("Layout reset to default", 3000)

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
        
        # In production mode, only show blocks flagged as production_visible
        is_prod = self._mode_manager.is_production if self._mode_manager else False
        
        # Group blocks by type
        blocks_by_type = {}
        for block in result.data:
            # Only include blocks that have registered panel types
            if not is_panel_registered(block.type):
                continue
            
            if is_prod and not self._is_block_production_visible(block):
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
                
                # Panel is "open" only if it exists AND is not currently closed/hidden.
                # open_panels tracks created panels; isClosed() reflects actual ADS state.
                panel = self.open_panels.get(block.id)
                is_open = panel is not None and not panel.isClosed()
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
    
    # ==================== Mode Visibility ====================
    
    @staticmethod
    def _is_block_production_visible(block) -> bool:
        """Check if a block should be visible in production mode.

        Only blocks explicitly marked ``production_visible: True`` in their
        metadata are shown.  No defaults -- the user controls everything.
        """
        metadata = getattr(block, 'metadata', None) or {}
        return bool(metadata.get("production_visible", False))

    def _apply_mode_visibility(self):
        """Show/hide menus, actions, and toolbar items based on current mode.

        Safe to call from anywhere (init, project load, mode switch).
        Only enforces production-mode dock restrictions.  In developer
        mode it leaves dock visibility alone so the saved layout is
        respected.
        """
        is_prod = self._mode_manager.is_production if self._mode_manager else False
        has_project = bool(self.facade.get_current_project_id())
        
        # -- File menu: developer-only items --
        self.action_save_project_as.setVisible(not is_prod)
        if hasattr(self, 'action_export_template'):
            self.action_export_template.setVisible(not is_prod)
        
        # -- Edit menu: visible in both modes; only Settings in production --
        if hasattr(self, '_edit_menu'):
            self._edit_menu.menuAction().setVisible(True)
            self.action_undo.setVisible(not is_prod)
            self.action_redo.setVisible(not is_prod)
            self.action_select_all.setVisible(not is_prod)
        
        # -- Window menu: each dock's view action follows its prod toggle --
        for dock, action in [
            (self.node_editor_dock, self.action_view_node_editor),
            (self.setlist_dock, self.action_view_setlist),
            (self.execution_dock, self.action_view_execution),
        ]:
            if action and dock:
                show = not is_prod or self._is_dock_production_visible(dock)
                action.setVisible(show)
        self.action_command_history.setVisible(not is_prod)
        if hasattr(self, '_layout_menu'):
            self._layout_menu.menuAction().setVisible(not is_prod)
        
        # -- Toolbar: always visible, same actions in both modes --
        if hasattr(self, '_main_toolbar'):
            self._main_toolbar.show()
        
        # -- Dock PROD toggle buttons: visible only in developer mode --
        self._sync_dock_prod_toggle_visibility()
        
        # -- Dock widgets: only restrict in production or no-project --
        has_project = bool(self.facade.get_current_project_id())
        if not has_project:
            for dock in [self.node_editor_dock, self.setlist_dock, self.execution_dock]:
                if dock:
                    dock.toggleView(False)
        elif is_prod:
            for dock in [self.node_editor_dock, self.setlist_dock, self.execution_dock]:
                if dock:
                    dock.toggleView(self._is_dock_production_visible(dock))
            for block_id, panel in list(self.open_panels.items()):
                block_result = self.facade.describe_block(block_id)
                visible = (
                    block_result.success
                    and block_result.data
                    and self._is_block_production_visible(block_result.data)
                )
                panel.toggleView(visible)
        # Developer mode with project: dock visibility is whatever the
        # saved layout set.  No override here.

        # -- Block panels menu (filters to PROD-marked in production) --
        if self.block_panels_menu:
            self._update_block_panels_menu()
        
        # -- Window title hint --
        mode_label = "" if is_prod else " [Developer]"
        base_title = self.windowTitle().replace(" [Developer]", "")
        self.setWindowTitle(base_title + mode_label)
        
    
    def _on_mode_changed_external(self, _mode):
        """React to mode changes (e.g. from Settings dialog).

        Applies menu/action visibility, then restores built-in docks
        if switching back to developer mode (production had hidden them).
        """
        self._apply_mode_visibility()
        if self._mode_manager and self._mode_manager.is_developer:
            self._restore_developer_docks()
    
    def _switch_mode(self, target):
        """Switch application mode and apply visibility rules."""
        self._mode_manager.switch_mode(target)
        self._apply_mode_visibility()
        from src.application.services.app_mode_manager import AppMode
        if target == AppMode.DEVELOPER:
            self._restore_developer_docks()
    
    def _restore_developer_docks(self):
        """Show built-in docks that production mode may have hidden.

        Only shows the three core docks -- block panels stay in whatever
        state the user left them.
        """
        if not self.facade.get_current_project_id():
            return
        for dock in [self.node_editor_dock, self.setlist_dock, self.execution_dock]:
            if dock:
                dock.toggleView(True)
    
    
    # ==================== Panel Management ====================
    
    def open_block_panel(self, block_id: str):
        """Open configuration panel for a block.
        
        In production mode, only blocks marked ``production_visible`` can
        have their panels opened.
        """
        if block_id in self.open_panels:
            panel = self.open_panels[block_id]
            panel.toggleView(True)
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
            
            is_prod = self._mode_manager.is_production if self._mode_manager else False
            if is_prod and not self._is_block_production_visible(block):
                self.statusBar().showMessage("Panel not available in Production mode", 3000)
                return
            
            from ui.qt_gui.block_panels import get_panel_class
            panel_class = get_panel_class(block.type)
            
            if not panel_class:
                QMessageBox.information(self, "Panel Not Available",
                    f"No panel available for {block.type} blocks.")
                return
            
            panel = panel_class(block_id, self.facade, parent=self)
            panel.panel_closed.connect(self._on_panel_closed)
            panel.setObjectName(f"BlockPanel_{block_id}")

            # Keep menu checkmarks accurate whenever panel visibility changes.
            panel.viewToggled.connect(lambda _: self._update_block_panels_menu())

            floating_container = self.dock_manager.addDockWidgetFloating(panel)
            floating_container.resize(1200, 600)
            
            window_id = f"block_panel_{block_id}"
            self.workspace.register_dock(window_id, panel)
            
            self.open_panels[block_id] = panel
            panel.toggleView(True)
            
            self._update_block_panels_menu()
            self.statusBar().showMessage(f"Panel opened: {block.name}", 3000)
            
        except Exception as e:
            Log.error(f"Failed to create panel: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Panel Error", f"Failed to open panel: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
    
    def _on_panel_closed(self, block_id: str):
        if block_id in self.open_panels:
            window_id = f"block_panel_{block_id}"
            self.workspace.unregister_dock(window_id)
            
            del self.open_panels[block_id]
            self._update_block_panels_menu()
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
            self.setWindowTitle(f"EZ - {project_name}")
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
        
        self._close_all_block_panels()
        
        result = self.facade.create_project("Untitled", save_directory=None)
        if result.success:
            self.undo_stack.clear()
            self._show_workspace()
            if not self.workspace.restore_state():
                self._set_default_layout()
            self._apply_mode_visibility()
            self.statusBar().showMessage("Created new project", 3000)
            self._update_ui_state()
            self.node_editor_window.refresh_and_center()
            self._update_block_panels_menu()
        else:
            QMessageBox.warning(self, "Error", result.message)
    
    def _on_open_project(self):
        if self.node_editor_window:
            self.node_editor_window.scene.flush_pending_position_saves()
        
        start_dir = app_settings.get_dialog_path("open_project")
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Project", start_dir, "EZ Projects (*.ez);;All Files (*)")
        
        if filename:
            self._close_all_block_panels()
            
            app_settings.set_dialog_path("open_project", filename)
            result = self.facade.load_project(filename)
            if result.success:
                self.undo_stack.clear()
                self._show_workspace()
                self._create_saved_panels()
                restored = self.workspace.restore_state()
                if not restored:
                    self._set_default_layout()

                self._apply_mode_visibility()
                self._update_ui_state()
                self.node_editor_window.refresh_and_center()
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
            self, "Save Project As", start_dir, "EZ Projects (*.ez);;All Files (*)")
        
        if filename:
            app_settings.set_dialog_path("save_project", filename)
            from pathlib import Path
            path = Path(filename)
            self._start_async_save(
                lambda: self._save_project_impl(str(path.parent), path.stem),
                f"Saved: {filename}",
                self._update_ui_state
            )
    
    def _on_export_template(self):
        """Export current project as the production template."""
        from src.application.services.template_manager import TemplateManager
        
        if not self.facade.get_current_project_id():
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return
        
        # Let user choose target (default is data/production_template.ez)
        from src.utils.paths import get_app_install_dir
        install_dir = get_app_install_dir()
        default_path = str(install_dir / "data" / "production_template.ez") if install_dir else ""
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Production Template", default_path,
            "EZ Template (*.ez);;All Files (*)")
        
        if not filename:
            return
        
        from pathlib import Path
        target = Path(filename)
        
        tmgr = TemplateManager(self.facade)
        success, message = tmgr.export_as_template(target)
        
        if success:
            QMessageBox.information(self, "Template Exported", message)
            self.statusBar().showMessage("Production template exported", 5000)
        else:
            QMessageBox.warning(self, "Export Failed", message)
    
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
        # If the settings dialog already called Colors.apply_theme_from_dict(),
        # Colors.X attributes are already correct -- skip reloading from registry.
        app = QApplication.instance()
        skip_reload = app and app.property("_theme_applied_from_dict")

        if not skip_reload:
            # Sets Colors.X from registry AND emits theme_changed internally.
            Colors.apply_theme()

        if skip_reload and app:
            app.setProperty("_theme_applied_from_dict", False)

        # Palette and stylesheet are pure reads of Colors.X -- no side effects.
        if app:
            app.setPalette(get_application_palette())

        self._apply_dock_styling()

        # Emit theme_changed AFTER the global stylesheet is set so that
        # ThemeAwareMixin widgets clearing child stylesheets fall back to
        # the already-correct global stylesheet (no visual flash).
        if skip_reload:
            from ui.qt_gui.design_system import _get_theme_signals
            try:
                _get_theme_signals().theme_changed.emit()
            except RuntimeError:
                pass

        force_style_refresh(self)

        if hasattr(self, 'node_editor_window') and self.node_editor_window:
            if hasattr(self.node_editor_window, 'refresh'):
                self.node_editor_window.refresh()

        if hasattr(self, 'setlist_window') and self.setlist_window:
            self.setlist_window.setStyleSheet(
                f"background-color: {Colors.BG_DARK.name()};"
            )
            if hasattr(self.setlist_window, 'setlist_view') and self.setlist_window.setlist_view:
                self.setlist_window.setlist_view.setStyleSheet(
                    f"background-color: {Colors.BG_DARK.name()};"
                )

        self._apply_macos_titlebar()
        self._apply_macos_titlebar_to_floating_docks()

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
        panel = getattr(self, "_execution_panel", None)
        if panel:
            panel.clear_log()
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
        panel = getattr(self, "_execution_panel", None)
        if panel:
            panel.set_progress(100)
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
            data = event.data or {}
            message = data.get('message', '...')
            percentage = data.get('percentage', 0)
            Log.debug(f"MainWindow: Received SubprocessProgress - {message} ({percentage}%)")
            self._append_execution_log(f"{percentage}% - {message}")
            panel = getattr(self, "_execution_panel", None)
            if panel:
                panel.set_progress(percentage)
            name = getattr(self, "_current_run_block_name", None)
            display_msg = f"{name}: {message}" if name else message
            self.subprocess_progress_signal.emit(display_msg, percentage)
        except Exception as e:
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
        QMessageBox.about(self, "About EZ",
            "<h3>EZ</h3>"
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
    
    def _try_load_production_project(self) -> bool:
        """Load the production project from template (copy-on-first-use).
        
        Returns True if the project was successfully loaded.
        """
        from src.application.services.template_manager import TemplateManager
        
        tmgr = TemplateManager(self.facade)
        
        # Check for template updates
        if tmgr.has_working_copy() and tmgr.is_update_available():
            reply = QMessageBox.question(
                self,
                "Template Update Available",
                "A new processing template is available.\n\n"
                "Update and reset to the new template?\n"
                "(Your current song list will be lost.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                tmgr.reset_to_template()
        
        success, message = tmgr.load_production_project()
        if success:
            self.undo_stack.clear()
            self._update_ui_state()
            self._update_block_panels_menu()
            self.statusBar().showMessage("Production project loaded", 3000)
            Log.info("Production project loaded from template")
            return True
        
        Log.warning(f"Failed to load production project: {message}")
        return False
    
    def _try_autoload_project(self):
        """Attempt to autoload the most recent project on startup. Returns True if project was loaded."""
        if not (hasattr(self.facade, 'app_settings') and self.facade.app_settings):
            Log.info("App settings not available, skipping autoload")
            return False
        
        # Production mode always tries autoload; developer mode respects the setting
        is_prod = self._mode_manager.is_production if self._mode_manager else False
        if not is_prod and not self.facade.app_settings.restore_last_project:
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

    def _update_ui_for_no_project(self):
        """Disable project-dependent UI elements when no project is loaded."""
        self.action_save_project.setEnabled(False)
        self.action_save_project_as.setEnabled(False)
        self.setWindowTitle("EZ - No Project")

    def _save_session(self):
        """Save session-specific state (zoom, viewport, selected block)."""
        try:
            if self.node_editor_window:
                selected = self.node_editor_window.scene.selected_blocks()
                if selected:
                    self.workspace.save_session("selected_block", selected[0])
                self.workspace.save_session(
                    "zoom_level", self.node_editor_window.view.zoom_level)
                self.workspace.save_session(
                    "viewport_center", self.node_editor_window.view.get_viewport_center())
        except Exception as e:
            Log.error(f"Session save error: {e}")
    
    def _restore_session_state(self):
        """Restore session-specific state (zoom, viewport, selected block)."""
        try:
            if not self.facade.get_current_project_id():
                return
            
            if self.node_editor_window:
                zoom = self.workspace.get_session("zoom_level")
                if isinstance(zoom, (int, float)):
                    self.node_editor_window.view.set_zoom_level(zoom)
                
                center = self.workspace.get_session("viewport_center")
                if isinstance(center, dict):
                    self.node_editor_window.view.center_on_point(
                        center.get("x", 0), center.get("y", 0))
                
                selected = self.workspace.get_session("selected_block")
                if selected:
                    self.node_editor_window.scene.select_block(selected)
        except Exception as e:
            Log.error(f"Session state restore error: {e}")
    
    def _apply_macos_titlebar(self):
        """Apply theme to macOS native title bar (macOS only, no-op on other platforms)."""
        try:
            from ui.qt_gui.platform.macos_titlebar import apply_titlebar_theme
            apply_titlebar_theme(self)
        except Exception:
            pass

    def _apply_macos_titlebar_to_floating_docks(self):
        """Apply macOS title bar theme to all floating dock widget windows."""
        try:
            from ui.qt_gui.platform.macos_titlebar import apply_titlebar_theme
            for dock in self.dock_manager.findChildren(ads.CDockWidget):
                if dock.isFloating():
                    top_level = dock.window()
                    if top_level and top_level != self:
                        apply_titlebar_theme(top_level)
        except Exception:
            pass

    def _on_dock_top_level_changed(self, floating: bool):
        """When a dock becomes floating, apply macOS title bar theme after window is ready."""
        if not floating:
            return
        try:
            dock = self.sender()
            if dock and dock.isFloating():
                def apply():
                    try:
                        from ui.qt_gui.platform.macos_titlebar import apply_titlebar_theme
                        top_level = dock.window()
                        if top_level and top_level != self:
                            apply_titlebar_theme(top_level)
                    except Exception:
                        pass
                QTimer.singleShot(50, apply)
        except Exception:
            pass

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
            # macOS: apply native title bar theme after window is shown (winId valid)
            QTimer.singleShot(100, self._apply_macos_titlebar)
    
    def closeEvent(self, event):
        Log.info("Closing main window...")
        
        # Mark that we're closing - prevents _on_panel_closed from overwriting our save
        self._closing = True

        if self._is_execution_active():
            reply = QMessageBox.question(
                self,
                "Execution in Progress",
                "A block execution is currently running.\n\nAre you sure you want to close EZ?",
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

        if hasattr(self.facade, "project_service") and self.facade.project_service:
            self.facade.project_service.flush_dirty_snapshots()

        for panel in list(self.open_panels.values()):
            try:
                panel.closeDockWidget()
            except Exception:
                pass
        
        if getattr(self, "save_thread", None) and self.save_thread.isRunning():
            self.save_thread.wait(2000)
        if getattr(self, "execution_thread", None) and self.execution_thread.isRunning():
            self.execution_thread.wait(2000)
        
        if self.setlist_window:
            self.setlist_window.cleanup()
        
        # Required by PyQt6Ads to avoid cleanup crashes
        self.dock_manager.deleteLater()
        
        event.accept()
    
    @pyqtSlot(str)
    def _on_setlist_song_switched(self, song_id: str):
        """Handle signal when a song is switched in the Setlist view."""
        Log.info(f"MainWindow: Setlist song switched to {song_id}. Refreshing UI.")
        
        # Refresh Node Editor to show new data
        if self.node_editor_window:
            self.node_editor_window.refresh()
        
        total_block_panels = 0
        visible_block_panels = 0
        reload_targets = []
        refresh_targets = []

        for panel_id, panel_dock in self.workspace._docks.items():
            if panel_id.startswith("block_panel_"):
                total_block_panels += 1
            if panel_id.startswith("block_panel_") and (panel_dock.isVisible() or not panel_dock.isClosed()):
                visible_block_panels += 1
                panel_widget = panel_dock.widget()
                # Dispatch against the dock panel object itself.
                # For BlockPanelBase subclasses, reload/refresh live on the dock,
                # while panel_dock.widget() is usually just the inner content widget.
                if hasattr(panel_dock, "reload_for_song_switch"):
                    reload_targets.append(panel_id)
                    panel_dock.reload_for_song_switch()
                elif hasattr(panel_dock, 'refresh'):
                    refresh_targets.append(panel_id)
                    panel_dock.refresh()

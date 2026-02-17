"""
Setlist View

Full-featured setlist processing interface.
Allows users to process multiple songs through current project and switch between them.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QGroupBox, QAbstractItemView, QInputDialog,
    QMessageBox, QSplitter, QCheckBox, QStyledItemDelegate, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QColor, QPainter, QPen

from src.application.api.application_facade import ApplicationFacade
from src.features.setlists.domain import Setlist
from src.features.setlists.domain import SetlistSong
from src.shared.domain.value_objects.execution_strategy import ExecutionStrategy
from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory
from ui.qt_gui.views.action_set_editor import ActionSetEditor
from ui.qt_gui.views.setlist_error_summary_panel import SetlistErrorSummaryPanel
from ui.qt_gui.dialogs.setlist_processing_dialog import SetlistProcessingDialog
from ui.qt_gui.core.setlist_processing_thread import SetlistProcessingThread
from src.application.services import get_progress_store
from src.utils.message import Log
from src.utils.settings import app_settings


class BorderRowDelegate(QStyledItemDelegate):
    """Custom delegate to draw border around active row"""
    
    def __init__(self, parent=None, active_row=-1, table_widget=None):
        super().__init__(parent)
        self.active_row = active_row
        self.table_widget = table_widget
    
    def paint(self, painter: QPainter, option, index):
        """Paint item with border if row is active"""
        super().paint(painter, option, index)
        
        if index.row() == self.active_row and self.table_widget:
            rect = option.rect
            col = index.column()
            total_cols = self.table_widget.columnCount()
            
            # Only draw border segments on first and last columns to create complete row border
            pen = QPen(QColor(Colors.ACCENT_BLUE.name()), 2)
            painter.setPen(pen)
            
            # Top border (all columns)
            painter.drawLine(rect.left(), rect.top(), rect.right(), rect.top())
            
            # Bottom border (all columns)
            painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
            
            # Left border (first column only)
            if col == 0:
                painter.drawLine(rect.left(), rect.top(), rect.left(), rect.bottom())
            
            # Right border (last column only)
            if col == total_cols - 1:
                painter.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())


class SetlistView(ThemeAwareMixin, QWidget):
    """
    Setlist View - Process multiple songs through current project.
    
    Professional UI with:
    - Folder selection and setlist creation
    - Action configuration panel
    - Execution strategy selection
    - Song processing with progress
    - Error summary and recovery
    - Quick song switching
    """
    
    # Signal emitted when song is switched
    song_switched = pyqtSignal(str)  # song_id
    
    def __init__(self, facade: ApplicationFacade, parent=None):
        super().__init__(parent)
        self.facade = facade
        self.current_setlist_id: Optional[str] = None
        self.current_song_id: Optional[str] = None
        self.songs: List[SetlistSong] = []
        self.current_setlist: Optional[Setlist] = None
        
        # Track processing state for progress visualization
        self.song_progress_bars: Dict[str, QProgressBar] = {}  # song_id -> progress bar
        self.song_action_status: Dict[str, Dict[int, str]] = {}  # song_id -> {action_index: status}
        self.song_current_action: Dict[str, str] = {}  # song_id -> current action name
        
        # Background processing state
        self._processing_thread: Optional[SetlistProcessingThread] = None
        self._processing_dialog: Optional[SetlistProcessingDialog] = None
        self._processing_errors: List[Dict[str, str]] = []
        
        self._setup_ui()
        
        # Subscribe to project change events to refresh setlists
        self._subscribe_to_events()
        
        # Auto-load setlists for current project (if any)
        # Setlists are part of projects, so they appear automatically
        if self.facade.current_project_id:
            self._auto_load_project_setlists()
            # _auto_load_project_setlists() calls load_setlist() which calls _update_ui()
            # So we don't need to call _update_ui() again here
        else:
            # No project - just update UI to show empty state
            self._update_ui()
        
        self._init_theme_aware()
        Log.info("SetlistView: Created")
    
    def _subscribe_to_events(self):
        """Subscribe to project change events to refresh setlists"""
        if not self.facade or not self.facade.event_bus:
            return
        
        # Refresh setlists when project changes
        self.facade.event_bus.subscribe("project.loaded", self._on_project_changed)
        self.facade.event_bus.subscribe("project.created", self._on_project_changed)
        self.facade.event_bus.subscribe("ProjectCreated", self._on_project_changed)
        
        # Refresh block actions when blocks are added or removed
        self.facade.event_bus.subscribe("BlockAdded", self._on_block_changed)
        self.facade.event_bus.subscribe("BlockRemoved", self._on_block_changed)
    
    def _on_project_changed(self, event):
        """
        Handle project change - automatically load setlists for new project.
        
        Setlists are part of the project, so they appear automatically.
        No "loading" or "selecting" needed.
        """
        # Clear current setlist when project changes
        self.current_setlist_id = None
        self.current_setlist = None
        self.current_song_id = None
        self.songs = []
        
        # Automatically load setlists for the project (they're part of it!)
        self._auto_load_project_setlists()
        
        # Refresh action set editor with new project
        if self.action_set_editor and self.facade.current_project_id:
            self.action_set_editor.load_project(self.facade.current_project_id)
        
        # Refresh UI
        self._update_ui()
    
    def _on_block_changed(self, event):
        """
        Handle block added/removed - refresh block actions in action set editor.
        
        When blocks are added or removed, the available actions change,
        so we need to refresh the action set editor to show the new blocks.
        """
        # Only refresh if we have a current project and the event is for this project
        if not self.facade or not self.facade.current_project_id:
            return
        
        event_project_id = getattr(event, 'project_id', None)
        if event_project_id and event_project_id != self.facade.current_project_id:
            return
        
        # Refresh action set editor to discover new actions
        if self.action_set_editor:
            Log.debug("SetlistView: Block changed, refreshing action set editor")
            self.action_set_editor.load_project(self.facade.current_project_id)
    
    def _auto_load_project_setlists(self):
        """
        Automatically load the setlist for the current project.
        
        One setlist per project - it appears automatically.
        If no setlist exists, one will be auto-created when needed (empty state).
        """
        if not self.facade.current_project_id:
            Log.debug("SetlistView: _auto_load_project_setlists - no project_id")
            return
        
        result = self.facade.list_setlists(self.facade.current_project_id)
        
        if result.success and result.data:
            setlists = result.data
            # One setlist per project - load it automatically
            if setlists:
                setlist = setlists[0]
                self.current_setlist_id = setlist.id
                self.current_setlist = setlist
                folder_name = Path(setlist.audio_folder_path).name if setlist.audio_folder_path else "(empty)"
                Log.info(f"SetlistView: Auto-loading setlist for project: {folder_name}")
                
                # Check for saved active song in setlist metadata
                saved_active_song_id = setlist.metadata.get("active_song_id")
                if saved_active_song_id:
                    Log.debug(f"SetlistView: Found saved active song: {saved_active_song_id}")
                
                self.load_setlist(setlist.id, auto_select_song_id=saved_active_song_id)
        else:
            # No setlist exists yet - that's fine, it will be auto-created when needed
            self.current_setlist_id = None
            self.current_setlist = None
            Log.debug(f"SetlistView: No setlist for project {self.facade.current_project_id} (will be auto-created when needed)")
    
    def _setup_ui(self):
        """Setup the main UI layout"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        main_layout.setSpacing(Spacing.SM)
        
        # Title
        title = QLabel("Setlist")
        title.setFont(Typography.heading_font())
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 16px; font-weight: bold; margin-bottom: 2px;")
        main_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Process multiple songs through your current project")
        subtitle.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px; margin-bottom: 4px;")
        main_layout.addWidget(subtitle)
        
        # Songs Table (MAIN FOCUS - moved to top)
        self.songs_group = self._create_songs_section()
        main_layout.addWidget(self.songs_group, 1)  # Stretch
        
        # Unified controls section (add songs, load folder, process, etc.)
        self.controls_group = self._create_unified_controls_section()
        main_layout.addWidget(self.controls_group)
        
        # Action Set Editor (new simplified action creation flow)
        from ui.qt_gui.views.action_set_editor import ActionSetEditor
        self.action_set_editor = ActionSetEditor(self.facade)
        self.action_set_editor.action_set_changed.connect(self._on_action_set_changed)
        self.action_set_group = QGroupBox("Action Set")
        self.action_set_group.setStyleSheet(StyleFactory.group_box())
        action_layout = QVBoxLayout(self.action_set_group)
        action_layout.addWidget(self.action_set_editor)
        self.action_set_group.setVisible(True)
        main_layout.addWidget(self.action_set_group)
        
        # Error Summary Panel
        self.error_panel = SetlistErrorSummaryPanel()
        self.error_panel.retry_requested.connect(self._on_retry_song)
        main_layout.addWidget(self.error_panel)
        
        # Add stretch at the end to push everything to top
        main_layout.addStretch()
    
    def _create_setlist_creation_section(self) -> QGroupBox:
        """
        Create setlist creation section.
        
        Setlists are part of projects - when you create one, it becomes part of the current project.
        """
        group = QGroupBox("Create Setlist")
        group.setStyleSheet(StyleFactory.group_box())
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        layout.setSpacing(Spacing.SM)
        
        # Instructions
        instructions = QLabel(
            "Setlists are part of your project. Select a folder containing audio files to create a setlist. "
            "The setlist will process each file through your current project configuration."
        )
        instructions.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Folder selection
        folder_layout = QGridLayout()
        folder_layout.setColumnStretch(1, 1)
        
        folder_layout.addWidget(QLabel("Audio Folder:"), 0, 0)
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select folder containing audio files...")
        self.folder_edit.setStyleSheet(StyleFactory.input())
        folder_layout.addWidget(self.folder_edit, 0, 1)
        
        self.folder_browse_btn = QPushButton("Browse...")
        self.folder_browse_btn.setStyleSheet(StyleFactory.button())
        self.folder_browse_btn.clicked.connect(self._on_browse_folder)
        folder_layout.addWidget(self.folder_browse_btn, 0, 2)
        
        layout.addLayout(folder_layout)
        
        # Create button
        self.create_btn = QPushButton("Create/Load Setlist")
        self.create_btn.setStyleSheet(StyleFactory.button("primary"))
        self.create_btn.clicked.connect(self._on_create_setlist)
        layout.addWidget(self.create_btn)
        
        return group
    
    
    def _create_info_section(self) -> QGroupBox:
        """Create setlist info section"""
        group = QGroupBox("Setlist Info")
        group.setStyleSheet(StyleFactory.group_box())
        
        layout = QGridLayout(group)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        layout.setSpacing(Spacing.XS)
        layout.setColumnStretch(1, 1)
        
        layout.addWidget(QLabel("Audio Folder:"), 0, 0)
        self.folder_label = QLabel("No folder")
        self.folder_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(self.folder_label, 0, 1)
        
        layout.addWidget(QLabel("Project:"), 1, 0)
        self.project_label = QLabel("No project")
        self.project_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(self.project_label, 1, 1)
        
        return group
    
    def _wrap_in_collapsible_group(self, title: str, widget: QWidget) -> QGroupBox:
        """Wrap a widget in a collapsible group box"""
        group = QGroupBox(title)
        group.setStyleSheet(StyleFactory.group_box())
        group.setCheckable(True)
        group.setChecked(False)
        
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        
        return group
    
    # Removed _create_active_song_section - using checkboxes in songs table instead
    
    def _create_unified_controls_section(self) -> QGroupBox:
        """Create unified controls section (add songs, load folder, process)"""
        group = QGroupBox("Controls")
        group.setStyleSheet(StyleFactory.group_box())
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        layout.setSpacing(Spacing.SM)
        
        # Top row: Add songs and load folder
        top_row = QHBoxLayout()
        
        self.add_song_btn = QPushButton("Add Song")
        self.add_song_btn.setStyleSheet(StyleFactory.button())
        self.add_song_btn.clicked.connect(self._on_add_song)
        top_row.addWidget(self.add_song_btn)
        
        self.load_folder_btn = QPushButton("Load Folder")
        self.load_folder_btn.setStyleSheet(StyleFactory.button())
        self.load_folder_btn.clicked.connect(self._on_load_folder)
        top_row.addWidget(self.load_folder_btn)
        
        self.remove_song_btn = QPushButton("Remove Selected")
        self.remove_song_btn.setStyleSheet(StyleFactory.button())
        self.remove_song_btn.clicked.connect(self._on_remove_song)
        self.remove_song_btn.setEnabled(False)
        top_row.addWidget(self.remove_song_btn)
        
        top_row.addStretch()
        layout.addLayout(top_row)
        
        # Bottom row: Process controls
        bottom_row = QHBoxLayout()
        
        self.process_all_btn = QPushButton("Process All Songs")
        self.process_all_btn.setStyleSheet(StyleFactory.button("primary"))
        self.process_all_btn.clicked.connect(self._on_process_all)
        bottom_row.addWidget(self.process_all_btn)
        
        self.process_selected_btn = QPushButton("Process Checked Songs")
        self.process_selected_btn.setStyleSheet(StyleFactory.button())
        self.process_selected_btn.clicked.connect(self._on_process_checked)
        bottom_row.addWidget(self.process_selected_btn)
        
        self.open_checked_btn = QPushButton("Open Checked Song")
        self.open_checked_btn.setStyleSheet(StyleFactory.button())
        self.open_checked_btn.clicked.connect(self._on_open_checked_song)
        bottom_row.addWidget(self.open_checked_btn)
        
        bottom_row.addStretch()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(StyleFactory.progress_bar())
        self.progress_bar.setVisible(False)
        bottom_row.addWidget(self.progress_bar)
        
        layout.addLayout(bottom_row)
        
        return group
    
    def _create_songs_section(self) -> QGroupBox:
        """Create songs table section"""
        group = QGroupBox("Songs")
        group.setStyleSheet(StyleFactory.group_box())
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        layout.setSpacing(Spacing.SM)
        
        # Songs count label
        count_layout = QHBoxLayout()
        self.songs_count_label = QLabel("(0 total)")
        self.songs_count_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        count_layout.addWidget(self.songs_count_label)
        count_layout.addStretch()
        layout.addLayout(count_layout)
        
        # Songs table - styled like action items table
        self.songs_table = QTableWidget()
        self.songs_table.setColumnCount(7)
        self.songs_table.setHorizontalHeaderLabels(["", "#", "Name", "Status", "Progress", "Actions", "Audio Path"])
        self.songs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Checkbox
        self.songs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)  # Order
        self.songs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Name
        self.songs_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)  # Status
        self.songs_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)  # Progress
        self.songs_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)  # Actions
        self.songs_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)  # Path
        self.songs_table.setColumnWidth(0, 40)   # Checkbox - increased to prevent clipping
        self.songs_table.setColumnWidth(1, 40)   # Order
        self.songs_table.setColumnWidth(3, 120)  # Status
        self.songs_table.setColumnWidth(4, 200)  # Progress
        self.songs_table.setColumnWidth(5, 80)   # Actions
        
        # Style like action items table
        self.songs_table.setAlternatingRowColors(True)
        self.songs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.songs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)  # Non-editable
        self.songs_table.verticalHeader().setDefaultSectionSize(36)  # Match action items row height
        self.songs_table.verticalHeader().setVisible(False)  # Hide row numbers
        self.songs_table.setStyleSheet(StyleFactory.table())
        
        # Set custom delegate for border highlighting
        self.border_delegate = BorderRowDelegate(self.songs_table, active_row=-1, table_widget=self.songs_table)
        self.songs_table.setItemDelegate(self.border_delegate)
        
        # Prepare for drag-and-drop reordering
        self.songs_table.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.songs_table.setDefaultDropAction(Qt.DropAction.MoveAction)
        
        # Connect signals
        self.songs_table.itemDoubleClicked.connect(self._on_song_double_clicked)
        self.songs_table.itemSelectionChanged.connect(self._on_song_selection_changed)
        
        layout.addWidget(self.songs_table, 1)  # Stretch
        
        return group
    
    # Removed _on_song_selected, _on_previous_song, _on_next_song - using checkboxes instead
    
    def _switch_to_song(self, song_id: str):
        """
        Switch to a song by restoring its snapshot.
        
        Shows loading indicator during snapshot restore for better UX feedback.
        Handles validation and error cases gracefully.
        """
        # Validate prerequisites
        if not self.current_setlist_id:
            QMessageBox.warning(self, "No Setlist", "No setlist is currently loaded.")
            return
        
        if not song_id:
            QMessageBox.warning(self, "No Song Selected", "Please select a song to switch to.")
            return
        
        # Validate song exists
        song = next((s for s in self.songs if s.id == song_id), None)
        if not song:
            QMessageBox.warning(self, "Song Not Found", f"Song {song_id} not found in current setlist.")
            return
        
        # Validate song is processed
        if song.status != "completed":
            QMessageBox.information(
                self,
                "Song Not Processed",
                f"'{Path(song.audio_path).name}' has not been processed yet.\n\n"
                "Please process the song first before switching to it."
            )
            return
        
        # Show loading indicator
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.progress_bar.setFormat(f"Loading {Path(song.audio_path).name}...")
        
        # Disable controls during switch
        if hasattr(self, 'songs_table'):
            self.songs_table.setEnabled(False)
        
        try:
            result = self.facade.switch_active_song(
                setlist_id=self.current_setlist_id,
                song_id=song_id
            )
            
            if result.success:
                self.current_song_id = song_id
                self.song_switched.emit(song_id)
                
                # Save active song ID to setlist metadata for persistence
                if self.current_setlist_id and hasattr(self.facade, 'setlist_service') and self.facade.setlist_service:
                    try:
                        setlist = self.facade.setlist_service._setlist_repo.get(self.current_setlist_id)
                        if setlist:
                            setlist.metadata["active_song_id"] = song_id
                            setlist.update_modified()
                            self.facade.setlist_service._setlist_repo.update(setlist)
                            Log.debug(f"SetlistView: Saved active song ID to setlist metadata")
                    except Exception as e:
                        Log.warning(f"SetlistView: Failed to save active song ID: {e}")
                
                # Refresh table to highlight current song
                self._load_songs()
                
                Log.info(f"SetlistView: Switched to song {Path(song.audio_path).name} (id: {song_id})")
                
                # BlockUpdated events have been published during restore
                # The song_switched signal will trigger main_window to refresh panels
            else:
                # Extract error message from result
                error_msg = result.message
                if result.errors:
                    error_msg = result.errors[0] if isinstance(result.errors, list) else str(result.errors)
                
                Log.error(f"SetlistView: Failed to switch to song {song_id}: {error_msg}")
                QMessageBox.warning(
                    self, 
                    "Switch Failed", 
                    f"Failed to switch to song:\n\n{error_msg}\n\n"
                    "Your previous state has been preserved."
                )
        except Exception as e:
            error_msg = str(e)
            Log.error(f"SetlistView: Exception switching to song {song_id}: {error_msg}")
            QMessageBox.warning(
                self, 
                "Switch Failed", 
                f"Failed to switch to song:\n\n{error_msg}\n\n"
                "Your previous state has been preserved."
            )
        finally:
            # Hide loading indicator
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(False)
            
            # Re-enable controls
            if hasattr(self, 'songs_table'):
                self.songs_table.setEnabled(True)
    
    def _on_browse_folder(self):
        """Browse for audio folder"""
        start_dir = app_settings.get_dialog_path("setlist_folder")
        directory = QFileDialog.getExistingDirectory(
            self, "Select Audio Folder",
            start_dir,
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            app_settings.set_dialog_path("setlist_folder", directory)
            self.folder_edit.setText(directory)
    
    def _on_create_setlist(self):
        """Create setlist from folder"""
        folder_path = self.folder_edit.text().strip()
        if not folder_path:
            QMessageBox.warning(self, "No Folder", "Please select an audio folder.")
            return
        
        # Get action configuration
        strategy, actions = self.action_config_panel.get_configuration()
        
        # Create setlist
        result = self.facade.create_setlist_from_folder(
            audio_folder_path=folder_path,
            execution_strategy=strategy,
            default_actions=actions
        )
        
        if result.success:
            setlist = result.data
            self.load_setlist(setlist.id)
            QMessageBox.information(
                self, "Setlist Created",
                f"Created setlist with songs from:\n{folder_path}"
            )
        else:
            error_msg = result.message
            if result.errors:
                error_msg += f"\n\n{result.errors[0]}"
            QMessageBox.warning(
                self, "Failed to Create Setlist",
                error_msg
            )
    
    def _on_action_set_changed(self):
        """Handle action set changes"""
        # Action set changes are tracked, can be saved when processing
        pass
    
    def _on_process_all(self):
        """Process all songs in setlist using background thread to keep UI responsive"""
        if not self.current_setlist_id:
            Log.warning("SetlistView: No setlist loaded")
            return
        
        # Get action items for the dialog
        action_items_result = self.facade.list_action_items_by_project(self.facade.current_project_id)
        if not action_items_result.success or not action_items_result.data:
            QMessageBox.warning(self, "No Actions", "No action items configured. Please add actions before processing.")
            return
        
        action_items = action_items_result.data
        action_items_dict = [
            {
                "action_name": item.action_name,
                "block_name": item.block_name,
                "action_type": item.action_type
            }
            for item in action_items
        ]
        
        # Prepare songs data for dialog
        songs_data = [
            {
                "id": song.id,
                "name": Path(song.audio_path).name,
                "audio_path": song.audio_path
            }
            for song in self.songs
        ]
        
        # Create and show processing dialog
        setlist_name = "Setlist"
        if self.current_setlist and self.current_setlist.audio_folder_path:
            setlist_name = Path(self.current_setlist.audio_folder_path).name
        
        # Store dialog as instance variable to prevent garbage collection
        self._processing_dialog = SetlistProcessingDialog(
            setlist_name=setlist_name,
            songs=songs_data,
            action_items=action_items_dict,
            parent=self,
            event_bus=self.facade.event_bus  # Pass event bus for block-level progress
        )
        
        # Track errors for this processing run
        self._processing_errors = []
        
        # Get progress store to track operation ID
        progress_store = get_progress_store()
        
        # Set up callback to capture operation ID when it starts
        def on_progress_started(event_type: str, state):
            if event_type == "started" and state.operation_type == "setlist_processing":
                self._processing_dialog.set_operation_id(state.operation_id)
                # Remove callback after getting ID
                progress_store.remove_callback(on_progress_started)
        
        progress_store.add_callback(on_progress_started)
        
        # Disable buttons while processing
        self.process_all_btn.setEnabled(False)
        self.process_selected_btn.setEnabled(False)
        
        # Create background thread for processing
        self._processing_thread = SetlistProcessingThread(
            facade=self.facade,
            setlist_id=self.current_setlist_id,
            parent=self
        )
        
        # Connect thread signals to UI handlers (thread-safe Qt signals)
        self._processing_thread.song_progress.connect(self._on_thread_song_progress)
        self._processing_thread.action_progress.connect(self._on_thread_action_progress)
        self._processing_thread.error_occurred.connect(self._on_thread_error)
        self._processing_thread.processing_complete.connect(self._on_thread_complete)
        self._processing_thread.processing_failed.connect(self._on_thread_failed)
        
        # Connect dialog cancel to thread cancellation
        self._processing_dialog.cancelled.connect(self._processing_thread.request_cancel)
        
        # Show dialog and ensure it's visible
        self._processing_dialog.show()
        self._processing_dialog.raise_()
        self._processing_dialog.activateWindow()
        
        # Start background processing (non-blocking!)
        Log.info("SetlistView: Starting background setlist processing thread")
        self._processing_thread.start()
    
    def _on_thread_song_progress(self, song_id: str, status: str):
        """Handle song progress signal from background thread (runs on main thread via Qt signal)"""
        if hasattr(self, '_processing_dialog') and self._processing_dialog:
            self._processing_dialog.update_song_status(song_id, status)
    
    def _on_thread_action_progress(self, song_id: str, action_index: int, total_actions: int, action_name: str, status: str):
        """Handle action progress signal from background thread (runs on main thread via Qt signal)"""
        Log.info(f"SetlistView: _on_thread_action_progress received - song={song_id}, action={action_index}/{total_actions}, status={status}")
        
        if hasattr(self, '_processing_dialog') and self._processing_dialog:
            # Update dialog
            self._processing_dialog.update_action_status(song_id, action_index, status)
        
        # Update progress bar for this song in table
        Log.debug(f"SetlistView: Looking for progress bar for song_id={song_id}, available: {list(self.song_progress_bars.keys())}")
        if song_id in self.song_progress_bars:
            progress_bar = self.song_progress_bars[song_id]
            
            # Calculate progress percentage
            if total_actions > 0:
                if status == "running":
                    action_progress = int((action_index / total_actions) * 100)
                elif status == "completed":
                    action_progress = int(((action_index + 1) / total_actions) * 100)
                elif status == "failed":
                    action_progress = int((action_index / total_actions) * 100)
                else:
                    action_progress = 0
                
                progress_bar.setValue(min(action_progress, 100))
                
                # Update format text
                if status == "running":
                    progress_bar.setFormat(f"{action_index + 1}/{total_actions}")
                elif status == "completed":
                    progress_bar.setFormat(f"{action_index + 1}/{total_actions}")
                elif status == "failed":
                    progress_bar.setFormat(f"Failed at {action_index + 1}/{total_actions}")
            
            # Update current action label
            if song_id in self.song_current_action:
                if status == "running":
                    self.song_current_action[song_id] = action_name
                    for row in range(self.songs_table.rowCount()):
                        widget = self.songs_table.cellWidget(row, 4)
                        if widget and song_id == self.songs[row].id:
                            for child in widget.findChildren(QLabel):
                                child.setText(action_name)
                                break
                            break
                elif status == "completed" and action_index == total_actions - 1:
                    self.song_current_action[song_id] = ""
                    progress_bar.setValue(100)
                    progress_bar.setFormat("Complete")
                    for row in range(self.songs_table.rowCount()):
                        widget = self.songs_table.cellWidget(row, 4)
                        if widget and song_id == self.songs[row].id:
                            for child in widget.findChildren(QLabel):
                                child.setText("")
                                break
                            break
        
        # Track action status
        if song_id not in self.song_action_status:
            self.song_action_status[song_id] = {}
        self.song_action_status[song_id][action_index] = status
    
    def _on_thread_error(self, song_path: str, error_message: str):
        """Handle per-song error from background thread"""
        self._processing_errors.append({
            "song": song_path,
            "error": error_message
        })
    
    def _on_thread_complete(self, success: bool, results: dict):
        """Handle processing completion from background thread"""
        Log.info(f"SetlistView: Processing thread completed, success={success}")
        
        # Mark dialog as complete
        if hasattr(self, '_processing_dialog') and self._processing_dialog:
            self._processing_dialog.processing_complete()
        
        # Re-enable buttons
        self.process_all_btn.setEnabled(True)
        self.process_selected_btn.setEnabled(True)
        
        if success:
            Log.info("SetlistView: Processed all songs successfully")
            self._update_ui()
            
            # Switch to last successfully processed song
            if results:
                successful_song_ids = [song_id for song_id, song_success in results.items() if song_success]
                if successful_song_ids:
                    last_song_id = successful_song_ids[-1]
                    self._switch_to_song(last_song_id)
                    Log.info(f"SetlistView: Switched to last processed song: {last_song_id}")
            
            # Show error summary if any errors
            if self._processing_errors:
                self.error_panel.set_errors(self._processing_errors)
                QMessageBox.warning(
                    self, "Processing Complete with Errors",
                    f"Processed songs with {len(self._processing_errors)} error(s).\nSee Error Summary panel for details."
                )
        
        # Clean up thread reference
        self._processing_thread = None
    
    def _on_thread_failed(self, error_message: str, errors: list):
        """Handle fatal processing failure from background thread"""
        Log.error(f"SetlistView: Processing thread failed: {error_message}")
        
        # Mark dialog as complete
        if hasattr(self, '_processing_dialog') and self._processing_dialog:
            self._processing_dialog.processing_complete()
        
        # Re-enable buttons
        self.process_all_btn.setEnabled(True)
        self.process_selected_btn.setEnabled(True)
        
        # Show error message
        error_details = error_message
        if errors:
            error_details += f"\n\n{errors[0]}"
        QMessageBox.warning(self, "Processing Failed", error_details)
        
        # Clean up thread reference
        self._processing_thread = None
    
    def _on_process_checked(self):
        """Process all checked songs"""
        checked_songs = self._get_checked_songs()
        if not checked_songs:
            QMessageBox.warning(self, "No Songs Checked", "Please check songs to process.")
            return
        
        if not self.current_setlist_id:
            QMessageBox.warning(self, "No Setlist", "No setlist available.")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_all_btn.setEnabled(False)
        self.process_selected_btn.setEnabled(False)
        
        # Process each checked song
        total = len(checked_songs)
        last_successful_song_id = None
        
        for idx, song in enumerate(checked_songs):
            self.progress_bar.setValue(int((idx / total) * 100))
            self.progress_bar.setFormat(f"Processing {Path(song.audio_path).name}... ({idx+1}/{total})")
            
            def progress_callback(message: str, current: int, total_steps: int):
                if total_steps > 0:
                    base_percent = int((idx / total) * 100)
                    step_percent = int((current / total_steps) * 100) // total
                    self.progress_bar.setValue(base_percent + step_percent)
                    self.progress_bar.setFormat(f"{Path(song.audio_path).name}: {message}")
            
            def song_action_progress(action_index: int, total_actions: int, action_name: str, status: str):
                """Handle action-level progress for individual song processing"""
                if song.id in self.song_progress_bars:
                    progress_bar = self.song_progress_bars[song.id]
                    if total_actions > 0:
                        action_progress = int((action_index / total_actions) * 70)
                        if status == "running":
                            action_progress += int((1 / total_actions) * 35)
                        elif status == "completed":
                            action_progress += int((1 / total_actions) * 70)
                        progress_bar.setValue(action_progress)
                        if status == "running":
                            progress_bar.setFormat(f"{action_index + 1}/{total_actions}")
                        elif status == "completed":
                            progress_bar.setFormat(f"{action_index + 1}/{total_actions}")
                    
                    if status == "running" and song.id in self.song_current_action:
                        self.song_current_action[song.id] = action_name
                        for row in range(self.songs_table.rowCount()):
                            widget = self.songs_table.cellWidget(row, 4)
                            if widget and song.id == self.songs[row].id:
                                for child in widget.findChildren(QLabel):
                                    child.setText(action_name)
                                    break
                                break
            
            result = self.facade.process_song(
                setlist_id=self.current_setlist_id,
                song_id=song.id,
                progress_callback=progress_callback,
                action_progress_callback=song_action_progress
            )
            
            if result.success:
                last_successful_song_id = song.id
            else:
                Log.error(f"SetlistView: Failed to process song {song.id}: {result.message}")
        
        self.progress_bar.setVisible(False)
        self.process_all_btn.setEnabled(True)
        self.process_selected_btn.setEnabled(True)
        
        self._load_songs()  # Refresh to show updated statuses
        
        # Switch to last successfully processed song
        if last_successful_song_id:
            self._switch_to_song(last_successful_song_id)
            Log.info(f"SetlistView: Switched to last processed song: {last_successful_song_id}")
        
        QMessageBox.information(self, "Processing Complete", f"Processed {len(checked_songs)} song(s).")
    
    def _on_open_checked_song(self):
        """Open the first checked song (if processed)"""
        checked_songs = self._get_checked_songs()
        if not checked_songs:
            QMessageBox.warning(self, "No Songs Checked", "Please check a song to open.")
            return
        
        # Find first processed song
        for song in checked_songs:
            if song.status == "completed":
                self._switch_to_song(song.id)
                return
        
        # No processed songs found
        QMessageBox.information(
            self,
            "No Processed Songs",
            "None of the checked songs have been processed yet. Please process them first."
        )
    
    def _on_checkbox_changed(self, song_id: str, state: int):
        """Handle checkbox state change - activate/deactivate song"""
        if state == Qt.CheckState.Checked.value:
            # Uncheck all other checkboxes
            for row in range(self.songs_table.rowCount()):
                checkbox = self.songs_table.cellWidget(row, 0)
                if checkbox and isinstance(checkbox, QCheckBox):
                    if row != self._get_row_for_song(song_id):
                        checkbox.blockSignals(True)
                        checkbox.setChecked(False)
                        checkbox.blockSignals(False)
            
            # Switch to this song if it's processed
            song = next((s for s in self.songs if s.id == song_id), None)
            if song and song.status == "completed":
                self._switch_to_song(song_id)
            elif song:
                # Song not processed - show message and uncheck
                QMessageBox.information(
                    self,
                    "Song Not Processed",
                    f"'{Path(song.audio_path).name}' has not been processed yet.\n\n"
                    "Please process the song first before activating it."
                )
                checkbox = self.songs_table.cellWidget(self._get_row_for_song(song_id), 0)
                if checkbox:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)
        else:
            # Unchecking - do nothing (keep current song active)
            pass
    
    def _get_row_for_song(self, song_id: str) -> int:
        """Get row index for a song ID"""
        for row in range(self.songs_table.rowCount()):
            item = self.songs_table.item(row, 1)  # Order column has song_id in UserRole
            if item and item.data(Qt.ItemDataRole.UserRole) == song_id:
                return row
        return -1
    
    def _get_checked_songs(self) -> List[SetlistSong]:
        """Get list of selected songs (for processing)"""
        # For now, return all songs - can add multi-select later if needed
        selected_rows = self.songs_table.selectionModel().selectedRows()
        if selected_rows:
            checked = []
            for row_item in selected_rows:
                row = row_item.row()
                if row < len(self.songs):
                    checked.append(self.songs[row])
            return checked
        return []
    
    def _on_load_folder(self):
        """Load all audio files from a folder into the setlist"""
        if not self.facade.current_project_id:
            QMessageBox.warning(self, "No Project", "Please open or create a project first.")
            return
        
        # Browse for folder
        start_dir = app_settings.get_dialog_path("setlist_folder")
        # Get main window as parent for proper modal behavior on macOS
        parent_window = self.window() if self.window() else None
        folder_path = QFileDialog.getExistingDirectory(
            parent_window,  # Use window() to get top-level window for proper modal behavior
            "Select Folder with Audio Files",
            start_dir
        )
        
        if folder_path:
            app_settings.set_dialog_path("setlist_folder", folder_path)
            
            # Get or create setlist for current project
            setlist_result = self.facade.list_setlists(self.facade.current_project_id)
            setlist = None
            if setlist_result.success and setlist_result.data:
                setlist = setlist_result.data[0]
                self.current_setlist_id = setlist.id
                self.current_setlist = setlist
            
            # Scan folder for audio files
            import glob
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aif', '.aiff']
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(glob.glob(str(Path(folder_path) / f"*{ext}")))
                audio_files.extend(glob.glob(str(Path(folder_path) / f"*{ext.upper()}")))
            
            audio_files = sorted(audio_files)
            
            if not audio_files:
                QMessageBox.warning(self, "No Audio Files", f"No audio files found in folder: {folder_path}")
                return
            
            # If no setlist exists, create one from folder (which adds all songs)
            if not setlist:
                result = self.facade.create_setlist_from_folder(folder_path)
                if result.success:
                    setlist = result.data
                    self.current_setlist_id = setlist.id
                    self.current_setlist = setlist
                    Log.info(f"SetlistView: Created setlist from folder {folder_path}")
                    QMessageBox.information(self, "Success", f"Created setlist with {len(audio_files)} song(s) from folder.")
                else:
                    error_msg = result.message
                    if result.errors:
                        error_msg += f"\n\n{result.errors[0]}"
                    QMessageBox.warning(self, "Failed to Create Setlist", error_msg)
                    return
            else:
                # Add all files from folder to existing setlist
                added_count = 0
                for audio_file in audio_files:
                    result = self.facade.add_song_to_setlist(audio_file, self.facade.current_project_id)
                    if result.success:
                        added_count += 1
                    else:
                        Log.warning(f"SetlistView: Failed to add {Path(audio_file).name}: {result.message}")
                
                if added_count > 0:
                    QMessageBox.information(self, "Success", f"Added {added_count} song(s) from folder.")
                else:
                    QMessageBox.warning(self, "Failed", "Failed to add songs from folder. See logs for details.")
            
            # Refresh UI
            self._load_songs()
            self._update_ui()
    
    def _on_retry_song(self, song_id: str):
        """Retry processing a failed song"""
        if not self.current_setlist_id:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_all_btn.setEnabled(False)
        self.process_selected_btn.setEnabled(False)
        
        def progress_callback(message: str, current: int, total: int):
            if total > 0:
                percent = int((current / total) * 100)
                self.progress_bar.setValue(percent)
                self.progress_bar.setFormat(message)
        
        result = self.facade.process_song(
            setlist_id=self.current_setlist_id,
            song_id=song_id,
            progress_callback=progress_callback
        )
        
        self.progress_bar.setVisible(False)
        self.process_all_btn.setEnabled(True)
        self.process_selected_btn.setEnabled(True)
        
        if result.success:
            Log.info(f"SetlistView: Retried processing song {song_id}")
            self._load_songs()  # Refresh to show updated status
        else:
            error_msg = result.message
            if result.errors:
                error_msg += f"\n\n{result.errors[0]}"
            Log.error(f"SetlistView: Failed to retry song: {error_msg}")
            QMessageBox.warning(self, "Retry Failed", error_msg)
    
    def _on_configure_song_actions(self):
        """Configure pre/post actions for selected song."""
        selected_rows = self.songs_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a song to configure actions for.")
            return
        
        row = selected_rows[0].row()
        if row >= len(self.songs):
            return
        
        song = self.songs[row]
        
        # Discover available actions for the dialog
        actions_by_block = {}
        if self.facade.current_project_id:
            result = self.facade.discover_setlist_actions(self.facade.current_project_id)
            if result.success:
                actions_by_block = result.data or {}
        
        from ui.qt_gui.dialogs.song_action_overrides_dialog import SongActionOverridesDialog
        dialog = SongActionOverridesDialog(
            facade=self.facade,
            song=song,
            actions_by_block=actions_by_block,
            parent=self
        )
        
        if dialog.exec():
            overrides = dialog.get_overrides()
            song.action_overrides = overrides
            
            # Persist the change
            if self.current_setlist_id:
                try:
                    self.facade.update_setlist_song(
                        project_id=self.facade.current_project_id,
                        song_id=song.id,
                        updates={"action_overrides": overrides}
                    )
                except Exception:
                    # Fallback: direct repo update if facade method not available
                    Log.warning("SetlistView: facade.update_setlist_song not available, updating in-memory only")
            
            # Refresh the table to show override indicator
            self._load_songs()
    
    def _on_song_selection_changed(self):
        """Handle song selection change in table"""
        selected_rows = self.songs_table.selectionModel().selectedRows()
        has_selection = len(selected_rows) > 0
        
        # Enable/disable remove button based on selection
        if hasattr(self, 'remove_song_btn'):
            self.remove_song_btn.setEnabled(has_selection)
        
        # Configure actions button (if it exists)
        if hasattr(self, 'configure_actions_btn'):
            self.configure_actions_btn.setVisible(has_selection)
        
        if has_selection:
            row = selected_rows[0].row()
            if row < len(self.songs):
                self.current_song_id = self.songs[row].id
    
    def _on_add_song(self):
        """Add a song to the setlist"""
        if not self.facade.current_project_id:
            QMessageBox.warning(self, "No Project", "Please open or create a project first.")
            return
        
        # Ensure setlist is loaded for current project
        if not self.current_setlist_id:
            self._auto_load_project_setlists()
        
        # Browse for audio file
        start_dir = app_settings.get_dialog_path("setlist_song")
        # Get main window as parent for proper modal behavior on macOS
        parent_window = self.window() if self.window() else None
        filename, _ = QFileDialog.getOpenFileName(
            parent_window,  # Use window() to get top-level window for proper modal behavior
            "Select Audio File",
            start_dir,
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a *.aif *.aiff);;All Files (*)"
        )
        
        if filename:
            app_settings.set_dialog_path("setlist_song", filename)
            
            result = self.facade.add_song_to_setlist(filename, self.facade.current_project_id)
            if result.success:
                Log.info(f"SetlistView: Added song {Path(filename).name}")
                # Ensure setlist is loaded before refreshing
                if not self.current_setlist_id:
                    self._auto_load_project_setlists()
                self._load_songs()  # Refresh song list
                QMessageBox.information(self, "Success", f"Added song: {Path(filename).name}")
            else:
                error_msg = result.message
                if result.errors:
                    error_msg += f"\n\n{result.errors[0]}"
                Log.error(f"SetlistView: Failed to add song: {error_msg}")
                QMessageBox.warning(self, "Failed to Add Song", error_msg)
    
    def _on_remove_song(self):
        """Remove selected song(s) from the setlist"""
        if not self.facade.current_project_id:
            QMessageBox.warning(self, "No Project", "Please open or create a project first.")
            return
        
        selected_rows = self.songs_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a song to remove.")
            return
        
        # Confirm removal
        song_names = []
        song_ids = []
        for row_item in selected_rows:
            row = row_item.row()
            if row < len(self.songs):
                song = self.songs[row]
                song_names.append(Path(song.audio_path).name)
                song_ids.append(song.id)
        
        reply = QMessageBox.question(
            self,
            "Remove Songs",
            f"Remove {len(song_names)} song(s) from setlist?\n\n" + "\n".join(song_names[:5]) + 
            (f"\n... and {len(song_names) - 5} more" if len(song_names) > 5 else ""),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Remove songs
            removed_count = 0
            for song_id in song_ids:
                result = self.facade.remove_song_from_setlist(song_id, self.facade.current_project_id)
                if result.success:
                    removed_count += 1
                else:
                    Log.error(f"SetlistView: Failed to remove song {song_id}: {result.message}")
            
            if removed_count > 0:
                Log.info(f"SetlistView: Removed {removed_count} song(s)")
                self._load_songs()  # Refresh song list
                QMessageBox.information(self, "Success", f"Removed {removed_count} song(s)")
            else:
                QMessageBox.warning(self, "Failed", "Failed to remove songs. See logs for details.")
    
    def _on_song_double_clicked(self, item: QTableWidgetItem):
        """Handle double-click on song in table"""
        row = item.row()
        if row < len(self.songs):
            song = self.songs[row]
            
            # Only allow switching to processed songs
            if song.status != "completed":
                QMessageBox.information(
                    self,
                    "Song Not Processed",
                    f"'{Path(song.audio_path).name}' has not been processed yet.\n\n"
                    "Please process the song first before switching to it."
                )
                return
            
            self._switch_to_song(song.id)
    
    def _update_ui(self):
        """
        Update UI to reflect current setlist state.
        
        Setlists are part of the project - they appear automatically.
        """
        if not self.current_setlist_id or not self.current_setlist:
            # No setlist for current project - show empty state
            self.songs_group.setVisible(True)  # Always show songs table (empty)
            self.controls_group.setVisible(True)  # Always show controls
            self.action_set_group.setVisible(True)  # Always show action set editor
            self.error_panel.setVisible(False)
            
            # Clear current song when no setlist
            self.current_song_id = None
            
            # Load project in action set editor
            if self.facade.current_project_id:
                self.action_set_editor.load_project(self.facade.current_project_id)
            # Clear songs table
            self.songs_table.setRowCount(0)
            self.songs = []
            return
        
        # Setlist exists - show full UI
        self.songs_group.setVisible(True)
        self.controls_group.setVisible(True)
        self.action_set_group.setVisible(True)
        self.error_panel.setVisible(True)
        
        # Load songs
        self._load_songs()
        
        # Update action set editor
        if self.facade.current_project_id:
            self.action_set_editor.load_project(self.facade.current_project_id)
            # Convert setlist default_actions to ActionSet format if needed
            # For now, start with empty action set - user can create new ones
    
    def _load_songs(self):
        """Load songs for current setlist"""
        if not self.current_setlist_id:
            Log.debug("SetlistView: _load_songs - no current_setlist_id, returning")
            return
        
        try:
            # Get songs through facade
            result = self.facade.get_setlist_songs(self.current_setlist_id)
            
            if not result.success:
                Log.error(f"SetlistView: Failed to load songs: {result.message}")
                return
            
            songs = result.data
            self.songs = songs
            
            # Update delegate active row
            active_row_index = -1
            if self.current_song_id:
                for idx, song in enumerate(songs):
                    if song.id == self.current_song_id:
                        active_row_index = idx
                        break
            self.border_delegate.active_row = active_row_index
            
            # Fix song status: if a song has a snapshot, it should be "completed"
            # This handles cases where status wasn't properly saved/restored on project load
            if self.facade.current_project_id and hasattr(self.facade, 'project_service') and self.facade.project_service:
                # Get project once for snapshot checks
                try:
                    project = self.facade.project_service.load_project(self.facade.current_project_id)
                    if project:
                        for song in songs:
                            # Check if song has a snapshot (indicates it was processed)
                            snapshot = self.facade.project_service.get_snapshot(song.id, project)
                            if snapshot:
                                # Song has snapshot - should be completed
                                if song.status != "completed":
                                    Log.debug(f"SetlistView: Fixing song {song.id} status from '{song.status}' to 'completed' (has snapshot)")
                                    song.status = "completed"
                                    # Update in repository
                                    if hasattr(self.facade, 'setlist_service') and self.facade.setlist_service:
                                        self.facade.setlist_service._setlist_song_repo.update(song)
                except Exception as e:
                    Log.debug(f"SetlistView: Could not check snapshots for songs: {e}")
            
            processed_songs = [s for s in songs if s.status == "completed"]
            
            # Clear old progress bars
            self.song_progress_bars.clear()
            self.song_action_status.clear()
            self.song_current_action.clear()
            
            # Update table
            self.songs_table.setRowCount(len(songs))
            
            for row, song in enumerate(songs):
                # Checkbox (column 0) - for activating song
                checkbox = QCheckBox()
                checkbox.setStyleSheet("""
                    QCheckBox {
                        spacing: 4px;
                        padding-left: 4px;
                    }
                    QCheckBox::indicator {
                        width: 18px;
                        height: 18px;
                    }
                """)
                # Check if this is the active song
                is_active = song.id == self.current_song_id
                checkbox.setChecked(is_active)
                checkbox.stateChanged.connect(lambda state, song_id=song.id: self._on_checkbox_changed(song_id, state))
                self.songs_table.setCellWidget(row, 0, checkbox)
                
                # Order (column 1)
                order_item = QTableWidgetItem(str(song.order_index + 1))
                order_item.setData(Qt.ItemDataRole.UserRole, song.id)
                order_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                order_item.setForeground(QColor(Colors.TEXT_SECONDARY.name()))
                self.songs_table.setItem(row, 1, order_item)
                
                # Name (column 2)
                name_item = QTableWidgetItem(Path(song.audio_path).name)
                name_item.setData(Qt.ItemDataRole.UserRole, song.id)
                # No text color change - highlight will be via row background
                self.songs_table.setItem(row, 2, name_item)
                
                # Status (column 3) - improved visual
                status_item = QTableWidgetItem(self._format_status(song.status))
                status_item.setForeground(QColor(self._status_color(song.status).name()))
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.songs_table.setItem(row, 3, status_item)
                
                # Progress (column 4) - progress bar and current action
                progress_widget = QWidget()
                progress_layout = QVBoxLayout(progress_widget)
                progress_layout.setContentsMargins(4, 4, 4, 4)
                progress_layout.setSpacing(2)
                progress_layout.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
                
                # Progress bar
                progress_bar = QProgressBar()
                progress_bar.setMinimum(0)
                progress_bar.setMaximum(100)
                progress_bar.setValue(0)
                progress_bar.setTextVisible(True)
                progress_bar.setStyleSheet(StyleFactory.progress_bar())
                progress_bar.setFixedHeight(20)
                progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Current action label
                action_label = QLabel("")
                action_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10px;")
                action_label.setWordWrap(True)
                action_label.setFixedHeight(14)
                action_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
                
                progress_layout.addWidget(progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)
                progress_layout.addWidget(action_label, alignment=Qt.AlignmentFlag.AlignCenter)
                
                self.songs_table.setCellWidget(row, 4, progress_widget)
                self.song_progress_bars[song.id] = progress_bar
                
                # Initialize action status tracking
                if song.id not in self.song_action_status:
                    self.song_action_status[song.id] = {}
                if song.id not in self.song_current_action:
                    self.song_current_action[song.id] = ""
                
                # Update progress based on status
                # Only set to 100% if truly completed (has snapshot), otherwise show actual progress
                if song.status == "completed":
                    # For completed songs, check if we have a snapshot
                    # If status is "completed", assume it's valid (snapshot check happens elsewhere)
                    progress_bar.setValue(100)
                    progress_bar.setFormat("Complete")
                    action_label.setText("")
                elif song.status == "processing":
                    # Don't reset progress if already processing - keep current value
                    if progress_bar.value() == 0:
                        progress_bar.setFormat("Processing...")
                    if song.id in self.song_current_action and self.song_current_action[song.id]:
                        action_label.setText(self.song_current_action[song.id])
                elif song.status == "failed":
                    progress_bar.setValue(0)
                    progress_bar.setFormat("Failed")
                    action_label.setText("")
                else:
                    progress_bar.setValue(0)
                    progress_bar.setFormat("Pending")
                    action_label.setText("")
                
                # Actions indicator (column 5)
                has_overrides = bool(song.action_overrides)
                actions_item = QTableWidgetItem("✓" if has_overrides else "")
                actions_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                actions_item.setToolTip("Has custom action overrides" if has_overrides else "Uses default actions")
                self.songs_table.setItem(row, 5, actions_item)
                
                # Audio path (column 6)
                path_item = QTableWidgetItem(song.audio_path)
                path_item.setToolTip(song.audio_path)
                path_item.setForeground(QColor(Colors.TEXT_SECONDARY.name()))
                self.songs_table.setItem(row, 6, path_item)
                
                # Show error message in status if failed
                if song.status == "failed" and song.error_message:
                    status_item.setToolTip(song.error_message)
                
                # Highlight active song row with border (handled by delegate)
                if is_active:
                    # Update delegate to highlight this row
                    self.border_delegate.active_row = row
                    # Keep checkbox styling simple - no border needed
                    checkbox.setStyleSheet("""
                        QCheckBox {
                            spacing: 4px;
                            padding-left: 4px;
                        }
                        QCheckBox::indicator {
                            width: 18px;
                            height: 18px;
                        }
                    """)
            
            # Update table viewport to refresh delegate painting
            if hasattr(self, 'border_delegate'):
                self.songs_table.viewport().update()
            
            # Update count
            self.songs_count_label.setText(f"({len(songs)} total, {len(processed_songs)} processed)")
            
        except Exception as e:
            Log.error(f"SetlistView: Failed to load songs: {e}")
    
    
    def _format_status(self, status: str) -> str:
        """Format status for display - improved visual"""
        status_map = {
            "pending": "Pending",
            "processing": "Processing...",
            "completed": "Complete",
            "failed": "Failed"
        }
        return status_map.get(status, status)
    
    def _status_color(self, status: str):
        """Get color for status"""
        color_map = {
            "pending": Colors.TEXT_SECONDARY,
            "processing": Colors.ACCENT_YELLOW,
            "completed": Colors.ACCENT_GREEN,
            "failed": Colors.ACCENT_RED
        }
        return color_map.get(status, Colors.TEXT_SECONDARY)
    
    def load_setlist(self, setlist_id: str, auto_select_song_id: Optional[str] = None):
        """
        Load a setlist into the view.
        
        Args:
            setlist_id: Setlist identifier
            auto_select_song_id: Optional song ID to automatically select after loading
        """
        try:
            # Get setlist through facade
            result = self.facade.get_setlist(setlist_id)
            
            if not result.success:
                error_msg = result.message
                if result.errors:
                    error_msg += f"\n\n{result.errors[0]}"
                Log.error(f"SetlistView: Failed to load setlist: {error_msg}")
                QMessageBox.warning(self, "Failed to Load Setlist", error_msg)
                return
            
            setlist = result.data
            self.current_setlist_id = setlist_id
            self.current_setlist = setlist
            self._update_ui()
            
            # Auto-select song if provided and it exists
            if auto_select_song_id:
                # Wait for songs to load, then select
                songs_result = self.facade.get_setlist_songs(setlist_id)
                if songs_result.success and songs_result.data:
                    song = next((s for s in songs_result.data if s.id == auto_select_song_id), None)
                    if song and song.status == "completed":
                        # Switch to the saved active song
                        self._switch_to_song(auto_select_song_id)
                        Log.info(f"SetlistView: Auto-selected saved active song: {Path(song.audio_path).name}")
                    elif song:
                        Log.debug(f"SetlistView: Saved active song {auto_select_song_id} exists but is not completed (status: {song.status})")
                        # Fall through to auto-select last processed song
                        auto_select_song_id = None
                    else:
                        Log.debug(f"SetlistView: Saved active song {auto_select_song_id} not found in setlist")
                        # Fall through to auto-select last processed song
                        auto_select_song_id = None
            
            # If no saved active song, auto-select last processed song
            if not auto_select_song_id:
                songs_result = self.facade.get_setlist_songs(setlist_id)
                if songs_result.success and songs_result.data:
                    completed_songs = [s for s in songs_result.data if s.status == "completed"]
                    if completed_songs:
                        # Select the last completed song (by order_index)
                        last_song = max(completed_songs, key=lambda s: s.order_index)
                        self._switch_to_song(last_song.id)
                        Log.info(f"SetlistView: Auto-selected last processed song: {Path(last_song.audio_path).name}")
            
            if setlist.audio_folder_path:
                Log.info(f"SetlistView: Loaded setlist for folder '{Path(setlist.audio_folder_path).name}'")
            else:
                Log.info(f"SetlistView: Loaded setlist (no folder path)")
        except Exception as e:
            Log.error(f"SetlistView: Failed to load setlist: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load setlist: {e}")
    
    def refresh(self):
        """Refresh the view"""
        self._update_ui()
    
    # Style helpers
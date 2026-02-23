"""
Setlist View

Compact setlist processing interface optimized for narrow sidebar layout.
Allows users to process multiple songs through current project and switch between them.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMenu,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QAbstractItemView,
    QMessageBox, QStyledItemDelegate,
    QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen

from src.application.api.application_facade import ApplicationFacade
from src.features.setlists.domain import Setlist, SetlistSong
from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory
from ui.qt_gui.views.action_set_editor import ActionSetEditor
from ui.qt_gui.views.setlist_error_summary_panel import SetlistErrorSummaryPanel
from ui.qt_gui.dialogs.setlist_processing_dialog import SetlistProcessingDialog
from ui.qt_gui.core.setlist_processing_thread import SetlistProcessingThread
from src.application.services import get_progress_store
from src.utils.message import Log
from src.utils.settings import app_settings


class ActiveRowDelegate(QStyledItemDelegate):
    """Draws a left accent border on the active (currently loaded) song row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_row = -1

    def paint(self, painter: QPainter, option, index):
        super().paint(painter, option, index)
        if index.row() == self.active_row:
            painter.save()
            pen = QPen(QColor(Colors.ACCENT_BLUE.name()), 3)
            painter.setPen(pen)
            rect = option.rect
            painter.drawLine(rect.left(), rect.top(), rect.left(), rect.bottom())
            painter.restore()


class SetlistView(ThemeAwareMixin, QWidget):
    """
    Compact setlist view optimized for narrow sidebar dock.

    Layout:
    - Header row: title + song count + "+" add menu
    - Slim progress bar (hidden when idle)
    - Song list table: 3 columns (#, Name, Status)
    - Full-width "Process All" button
    - Error summary panel (auto-hidden when no errors)
    - Pipeline tab (Action Set editor, hidden in production mode)
    """

    song_switched = pyqtSignal(str)  # song_id

    def __init__(self, facade: ApplicationFacade, parent=None):
        super().__init__(parent)
        self.facade = facade
        self.current_setlist_id: Optional[str] = None
        self.current_song_id: Optional[str] = None
        self.songs: List[SetlistSong] = []
        self.current_setlist: Optional[Setlist] = None

        self._processing_thread: Optional[SetlistProcessingThread] = None
        self._processing_dialog: Optional[SetlistProcessingDialog] = None
        self._processing_errors: List[Dict[str, str]] = []

        self._setup_ui()
        self._subscribe_to_events()

        if self.facade.current_project_id:
            self._auto_load_project_setlists()
        else:
            self._update_ui()

        self._init_theme_aware()
        Log.info("SetlistView: Created")

    # ------------------------------------------------------------------
    # Event subscriptions (unchanged backend integration)
    # ------------------------------------------------------------------

    def _subscribe_to_events(self):
        """Subscribe to project change events to refresh setlists."""
        if not self.facade or not self.facade.event_bus:
            return
        self.facade.event_bus.subscribe("project.loaded", self._on_project_changed)
        self.facade.event_bus.subscribe("project.created", self._on_project_changed)
        self.facade.event_bus.subscribe("ProjectCreated", self._on_project_changed)
        self.facade.event_bus.subscribe("BlockAdded", self._on_block_changed)
        self.facade.event_bus.subscribe("BlockRemoved", self._on_block_changed)

    def _on_project_changed(self, event):
        """Handle project change -- auto-load setlists for new project."""
        self.current_setlist_id = None
        self.current_setlist = None
        self.current_song_id = None
        self.songs = []

        self._auto_load_project_setlists()

        if self.action_set_editor and self.facade.current_project_id:
            self.action_set_editor.load_project(self.facade.current_project_id)

        self._update_ui()

    def _on_block_changed(self, event):
        """Refresh action set editor when blocks are added/removed."""
        if not self.facade or not self.facade.current_project_id:
            return
        event_project_id = getattr(event, 'project_id', None)
        if event_project_id and event_project_id != self.facade.current_project_id:
            return
        if self.action_set_editor:
            Log.debug("SetlistView: Block changed, refreshing action set editor")
            self.action_set_editor.load_project(self.facade.current_project_id)

    def _auto_load_project_setlists(self):
        """Load the setlist for the current project (one per project)."""
        if not self.facade.current_project_id:
            Log.debug("SetlistView: _auto_load_project_setlists - no project_id")
            return

        result = self.facade.list_setlists(self.facade.current_project_id)
        if result.success and result.data:
            setlists = result.data
            if setlists:
                setlist = setlists[0]
                self.current_setlist_id = setlist.id
                self.current_setlist = setlist
                folder_name = Path(setlist.audio_folder_path).name if setlist.audio_folder_path else "(empty)"
                Log.info(f"SetlistView: Auto-loading setlist for project: {folder_name}")

                saved_active_song_id = setlist.metadata.get("active_song_id")
                if saved_active_song_id:
                    Log.debug(f"SetlistView: Found saved active song: {saved_active_song_id}")
                self.load_setlist(setlist.id, auto_select_song_id=saved_active_song_id)
        else:
            self.current_setlist_id = None
            self.current_setlist = None
            Log.debug(f"SetlistView: No setlist for project {self.facade.current_project_id}")

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Build the compact sidebar layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        main_layout.setSpacing(Spacing.XS)

        # -- Tab widget: Setlist tab + Pipeline tab --
        self._tab_widget = QTabWidget()

        # --- Tab 0: Setlist ---
        setlist_tab = QWidget()
        tab_layout = QVBoxLayout(setlist_tab)
        tab_layout.setContentsMargins(0, Spacing.XS, 0, 0)
        tab_layout.setSpacing(Spacing.XS)

        # Header row: title + count + add button
        header = QHBoxLayout()
        header.setSpacing(Spacing.SM)

        title = QLabel("Setlist")
        title.setFont(Typography.heading_font())
        title.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 15px; font-weight: bold;"
        )
        header.addWidget(title)

        self.songs_count_label = QLabel("")
        self.songs_count_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;"
        )
        header.addWidget(self.songs_count_label)

        header.addStretch()

        self._add_menu_btn = QPushButton("+")
        self._add_menu_btn.setFixedSize(26, 26)
        self._add_menu_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE.name()};
                border: none;
                border-radius: {border_radius(13)};
                color: {Colors.TEXT_PRIMARY.name()};
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.ACCENT_BLUE.darker(110).name()};
            }}
        """)
        self._add_menu_btn.clicked.connect(self._show_add_menu)
        header.addWidget(self._add_menu_btn)

        tab_layout.addLayout(header)

        # Slim global progress bar (hidden when idle)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: none;
                border-radius: {border_radius(3)};
            }}
            QProgressBar::chunk {{
                background-color: {Colors.ACCENT_BLUE.name()};
                border-radius: {border_radius(3)};
            }}
        """)
        self.progress_bar.setVisible(False)
        tab_layout.addWidget(self.progress_bar)

        # Song list table (3 columns: #, Name, Status)
        self.songs_table = QTableWidget()
        self.songs_table.setColumnCount(3)
        self.songs_table.setHorizontalHeaderLabels(["#", "Name", "Status"])
        self.songs_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed
        )
        self.songs_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.songs_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Fixed
        )
        self.songs_table.setColumnWidth(0, 30)
        self.songs_table.setColumnWidth(2, 80)

        self.songs_table.setAlternatingRowColors(True)
        self.songs_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.songs_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.songs_table.verticalHeader().setDefaultSectionSize(32)
        self.songs_table.verticalHeader().setVisible(False)
        self.songs_table.setStyleSheet(StyleFactory.table())

        self.songs_table.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.songs_table.setDefaultDropAction(Qt.DropAction.MoveAction)

        # Active row delegate (left accent bar)
        self._row_delegate = ActiveRowDelegate(self.songs_table)
        self.songs_table.setItemDelegate(self._row_delegate)

        # Signals
        self.songs_table.itemDoubleClicked.connect(self._on_song_double_clicked)
        self.songs_table.itemSelectionChanged.connect(
            self._on_song_selection_changed
        )
        self.songs_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.songs_table.customContextMenuRequested.connect(
            self._on_songs_context_menu
        )

        tab_layout.addWidget(self.songs_table, 1)

        # Process All button (full width, primary)
        self.process_all_btn = QPushButton("Process All")
        self.process_all_btn.setStyleSheet(StyleFactory.button("primary"))
        self.process_all_btn.clicked.connect(self._on_process_all)
        tab_layout.addWidget(self.process_all_btn)

        # Error panel
        self.error_panel = SetlistErrorSummaryPanel()
        self.error_panel.retry_requested.connect(self._on_retry_song)
        tab_layout.addWidget(self.error_panel)

        self._tab_widget.addTab(setlist_tab, "Setlist")

        # --- Tab 1: Pipeline (Action Set editor) ---
        pipeline_tab = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_tab)
        pipeline_layout.setContentsMargins(0, Spacing.SM, 0, 0)
        pipeline_layout.setSpacing(Spacing.SM)

        self.action_set_editor = ActionSetEditor(self.facade)
        self.action_set_editor.action_set_changed.connect(
            self._on_action_set_changed
        )
        pipeline_layout.addWidget(self.action_set_editor)
        pipeline_layout.addStretch()
        self._tab_widget.addTab(pipeline_tab, "Pipeline")

        # Legacy reference for existing code checking action_set_group visibility
        self.action_set_group = pipeline_tab

        main_layout.addWidget(self._tab_widget, 1)

        self._apply_mode_to_tabs()

    # ------------------------------------------------------------------
    # Add menu (+)
    # ------------------------------------------------------------------

    def _show_add_menu(self):
        """Show dropdown menu for adding songs."""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 16px;
                border-radius: {border_radius(3)};
            }}
            QMenu::item:selected {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
        """)

        add_song = menu.addAction("Add Song...")
        add_song.triggered.connect(self._on_add_song)

        load_folder = menu.addAction("Load Folder...")
        load_folder.triggered.connect(self._on_load_folder)

        menu.exec(
            self._add_menu_btn.mapToGlobal(
                self._add_menu_btn.rect().bottomLeft()
            )
        )

    # ------------------------------------------------------------------
    # Mode handling
    # ------------------------------------------------------------------

    def _apply_mode_to_tabs(self):
        """Disable the Pipeline tab in production mode."""
        mode_mgr = getattr(self.facade, 'app_mode_manager', None)
        if mode_mgr is None:
            return
        is_prod = mode_mgr.is_production
        self._tab_widget.setTabEnabled(1, not is_prod)

        if not getattr(self, '_mode_connected', False):
            mode_mgr.mode_changed.connect(self._on_mode_changed)
            self._mode_connected = True

    def _on_mode_changed(self, new_mode):
        """React to runtime mode switch."""
        from src.application.services.app_mode_manager import AppMode
        is_prod = new_mode == AppMode.PRODUCTION
        self._tab_widget.setTabEnabled(1, not is_prod)

    # ------------------------------------------------------------------
    # Song switching
    # ------------------------------------------------------------------

    def _switch_to_song(self, song_id: str):
        """Switch to a song by restoring its snapshot."""
        if not self.current_setlist_id:
            QMessageBox.warning(self, "No Setlist", "No setlist is currently loaded.")
            return

        if not song_id:
            QMessageBox.warning(
                self, "No Song Selected", "Please select a song to switch to."
            )
            return

        song = next((s for s in self.songs if s.id == song_id), None)
        if not song:
            QMessageBox.warning(
                self, "Song Not Found",
                f"Song {song_id} not found in current setlist."
            )
            return

        if song.status != "completed":
            QMessageBox.information(
                self, "Song Not Processed",
                f"'{Path(song.audio_path).name}' has not been processed yet.\n\n"
                "Process the song first before switching to it."
            )
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.songs_table.setEnabled(False)

        try:
            result = self.facade.switch_active_song(
                setlist_id=self.current_setlist_id, song_id=song_id
            )

            if result.success:
                self.current_song_id = song_id
                self.song_switched.emit(song_id)

                if (
                    self.current_setlist_id
                    and hasattr(self.facade, 'setlist_service')
                    and self.facade.setlist_service
                ):
                    try:
                        setlist = self.facade.setlist_service._setlist_repo.get(
                            self.current_setlist_id
                        )
                        if setlist:
                            setlist.metadata["active_song_id"] = song_id
                            setlist.update_modified()
                            self.facade.setlist_service._setlist_repo.update(setlist)
                    except Exception as e:
                        Log.warning(
                            f"SetlistView: Failed to save active song ID: {e}"
                        )

                self._load_songs()
                Log.info(
                    f"SetlistView: Switched to song "
                    f"{Path(song.audio_path).name} (id: {song_id})"
                )
            else:
                error_msg = result.message
                if result.errors:
                    error_msg = (
                        result.errors[0]
                        if isinstance(result.errors, list)
                        else str(result.errors)
                    )
                Log.error(
                    f"SetlistView: Failed to switch to song {song_id}: {error_msg}"
                )
                QMessageBox.warning(
                    self, "Switch Failed",
                    f"Failed to switch to song:\n\n{error_msg}\n\n"
                    "Your previous state has been preserved."
                )
        except Exception as e:
            Log.error(
                f"SetlistView: Exception switching to song {song_id}: {e}"
            )
            QMessageBox.warning(
                self, "Switch Failed",
                f"Failed to switch to song:\n\n{e}\n\n"
                "Your previous state has been preserved."
            )
        finally:
            self.progress_bar.setVisible(False)
            self.progress_bar.setRange(0, 100)
            self.songs_table.setEnabled(True)

    # ------------------------------------------------------------------
    # Action set
    # ------------------------------------------------------------------

    def _on_action_set_changed(self):
        """Handle action set changes."""
        pass

    def _get_active_action_set_id(self) -> Optional[str]:
        """Return the ID of the first action set for the current project."""
        repo = getattr(self.facade, "action_set_repo", None)
        if not repo or not self.facade.current_project_id:
            return None
        try:
            sets = repo.list_by_project(self.facade.current_project_id)
            if sets:
                return sets[0].id
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Processing: all songs
    # ------------------------------------------------------------------

    def _on_process_all(self):
        """Process all songs using a background thread."""
        if not self.current_setlist_id:
            Log.warning("SetlistView: No setlist loaded")
            return

        action_set_id = self._get_active_action_set_id()
        if action_set_id:
            action_items_result = self.facade.list_action_items(action_set_id)
        else:
            action_items_result = self.facade.list_action_items_by_project(
                self.facade.current_project_id
            )
        if not action_items_result.success or not action_items_result.data:
            QMessageBox.warning(
                self, "No Actions",
                "No action items configured. Add actions in the Pipeline tab first."
            )
            return

        action_items = action_items_result.data
        action_items_dict = [
            {
                "action_name": item.action_name,
                "block_name": item.block_name,
                "action_type": item.action_type,
            }
            for item in action_items
        ]

        songs_data = [
            {
                "id": song.id,
                "name": Path(song.audio_path).name,
                "audio_path": song.audio_path,
            }
            for song in self.songs
        ]

        setlist_name = "Setlist"
        if self.current_setlist and self.current_setlist.audio_folder_path:
            setlist_name = Path(self.current_setlist.audio_folder_path).name

        self._processing_dialog = SetlistProcessingDialog(
            setlist_name=setlist_name,
            songs=songs_data,
            action_items=action_items_dict,
            parent=self,
            event_bus=self.facade.event_bus,
        )

        self._processing_errors = []

        progress_store = get_progress_store()

        def on_progress_started(event_type: str, state):
            if event_type == "started" and state.operation_type == "setlist_processing":
                self._processing_dialog.set_operation_id(state.operation_id)
                progress_store.remove_callback(on_progress_started)

        progress_store.add_callback(on_progress_started)

        self.process_all_btn.setEnabled(False)

        self._processing_thread = SetlistProcessingThread(
            facade=self.facade,
            setlist_id=self.current_setlist_id,
            parent=self,
        )

        self._processing_thread.song_progress.connect(self._on_thread_song_progress)
        self._processing_thread.error_occurred.connect(self._on_thread_error)
        self._processing_thread.processing_complete.connect(self._on_thread_complete)
        self._processing_thread.processing_failed.connect(self._on_thread_failed)

        self._processing_dialog.cancelled.connect(
            self._processing_thread.request_cancel
        )

        self._processing_dialog.show()
        self._processing_dialog.raise_()
        self._processing_dialog.activateWindow()

        Log.info("SetlistView: Starting background setlist processing thread")
        self._processing_thread.start()

    # ------------------------------------------------------------------
    # Processing thread signal handlers
    # ------------------------------------------------------------------

    def _on_thread_song_progress(self, song_id: str, status: str):
        """Handle song-level progress from background thread."""
        if self._processing_dialog:
            self._processing_dialog.update_song_status(song_id, status)

        status_map = {
            "processing": "processing",
            "completed": "completed",
            "failed": "failed",
        }
        display_status = status_map.get(status)
        if display_status:
            self._update_song_status_in_table(song_id, display_status)

    def _on_thread_error(self, song_path: str, error_message: str):
        """Handle per-song error from background thread."""
        self._processing_errors.append({
            "song": song_path,
            "error": error_message,
        })

    def _on_thread_complete(self, success: bool, results: dict):
        """Handle processing completion from background thread."""
        Log.info(f"SetlistView: Processing thread completed, success={success}")

        if self._processing_dialog:
            self._processing_dialog.processing_complete()

        self.process_all_btn.setEnabled(True)

        if success:
            Log.info("SetlistView: Processed all songs successfully")
            self._update_ui()

            if results:
                successful = [
                    sid for sid, ok in results.items() if ok
                ]
                if successful:
                    last_song_id = successful[-1]
                    self._switch_to_song(last_song_id)
                    Log.info(
                        f"SetlistView: Switched to last processed song: "
                        f"{last_song_id}"
                    )

            if self._processing_errors:
                self.error_panel.set_errors(self._processing_errors)

        self._processing_thread = None

    def _on_thread_failed(self, error_message: str, errors: list):
        """Handle fatal processing failure from background thread."""
        Log.error(f"SetlistView: Processing thread failed: {error_message}")

        if self._processing_dialog:
            self._processing_dialog.processing_complete()

        self.process_all_btn.setEnabled(True)

        error_details = error_message
        if errors:
            error_details += f"\n\n{errors[0]}"
        QMessageBox.warning(self, "Processing Failed", error_details)

        self._processing_thread = None

    # ------------------------------------------------------------------
    # Processing: single song (context menu / retry)
    # ------------------------------------------------------------------

    def _process_single_song(self, song_id: str):
        """Process a single song using a background thread and progress dialog."""
        if not self.current_setlist_id:
            return

        song = next((s for s in self.songs if s.id == song_id), None)
        if not song:
            Log.warning(f"SetlistView: Song {song_id} not found")
            return

        action_set_id = self._get_active_action_set_id()
        if action_set_id:
            action_items_result = self.facade.list_action_items(action_set_id)
        else:
            action_items_result = self.facade.list_action_items_by_project(
                self.facade.current_project_id
            )
        if not action_items_result.success or not action_items_result.data:
            QMessageBox.warning(
                self, "No Actions",
                "No action items configured. Add actions in the Pipeline tab first."
            )
            return

        action_items_dict = [
            {
                "action_name": item.action_name,
                "block_name": item.block_name,
                "action_type": item.action_type,
            }
            for item in action_items_result.data
        ]

        songs_data = [
            {
                "id": song.id,
                "name": Path(song.audio_path).name,
                "audio_path": song.audio_path,
            }
        ]

        setlist_name = Path(song.audio_path).stem

        self._processing_dialog = SetlistProcessingDialog(
            setlist_name=setlist_name,
            songs=songs_data,
            action_items=action_items_dict,
            parent=self,
            event_bus=self.facade.event_bus,
        )

        self._processing_errors = []

        progress_store = get_progress_store()

        def on_progress_started(event_type: str, state):
            if event_type == "started" and state.operation_type == "setlist_processing":
                self._processing_dialog.set_operation_id(state.operation_id)
                progress_store.remove_callback(on_progress_started)

        progress_store.add_callback(on_progress_started)

        self.process_all_btn.setEnabled(False)

        self._processing_thread = SetlistProcessingThread(
            facade=self.facade,
            setlist_id=self.current_setlist_id,
            song_id=song_id,
            parent=self,
        )

        self._processing_thread.song_progress.connect(self._on_thread_song_progress)
        self._processing_thread.error_occurred.connect(self._on_thread_error)
        self._processing_thread.processing_complete.connect(self._on_thread_complete)
        self._processing_thread.processing_failed.connect(self._on_thread_failed)

        self._processing_dialog.cancelled.connect(
            self._processing_thread.request_cancel
        )

        self._processing_dialog.show()
        self._processing_dialog.raise_()
        self._processing_dialog.activateWindow()

        Log.info(f"SetlistView: Starting background single-song processing thread for {song_id}")
        self._processing_thread.start()

    def _on_retry_song(self, song_id: str):
        """Retry processing a failed song (from error panel)."""
        self._process_single_song(song_id)

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _on_songs_context_menu(self, position):
        """Right-click context menu for song rows."""
        index = self.songs_table.indexAt(position)
        if not index.isValid():
            return
        row = index.row()
        if row >= len(self.songs):
            return

        self.songs_table.selectRow(row)
        song = self.songs[row]

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 16px;
                border-radius: {border_radius(3)};
            }}
            QMenu::item:selected {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
            QMenu::item:disabled {{
                color: {Colors.TEXT_DISABLED.name()};
            }}
        """)

        switch_action = menu.addAction("Switch to Song")
        switch_action.setEnabled(song.status == "completed")
        switch_action.triggered.connect(lambda: self._switch_to_song(song.id))

        process_action = menu.addAction("Process Song")
        process_action.triggered.connect(
            lambda: self._process_single_song(song.id)
        )

        menu.addSeparator()

        remove_action = menu.addAction("Remove from Setlist")
        remove_action.triggered.connect(
            lambda: self._remove_single_song(song)
        )

        menu.exec(self.songs_table.mapToGlobal(position))

    # ------------------------------------------------------------------
    # Song management (add / remove / load folder)
    # ------------------------------------------------------------------

    def _on_add_song(self):
        """Add a song to the setlist via file dialog."""
        if not self.facade.current_project_id:
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.current_setlist_id:
            self._auto_load_project_setlists()

        start_dir = app_settings.get_dialog_path("setlist_song")
        parent_window = self.window() if self.window() else None
        filename, _ = QFileDialog.getOpenFileName(
            parent_window,
            "Select Audio File",
            start_dir,
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a *.aif *.aiff);;All Files (*)",
        )

        if filename:
            app_settings.set_dialog_path("setlist_song", filename)
            result = self.facade.add_song_to_setlist(
                filename, self.facade.current_project_id
            )
            if result.success:
                Log.info(f"SetlistView: Added song {Path(filename).name}")
                if not self.current_setlist_id:
                    self._auto_load_project_setlists()
                self._load_songs()
            else:
                error_msg = result.message
                if result.errors:
                    error_msg += f"\n\n{result.errors[0]}"
                QMessageBox.warning(self, "Failed to Add Song", error_msg)

    def _on_load_folder(self):
        """Load all audio files from a folder into the setlist."""
        if not self.facade.current_project_id:
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        start_dir = app_settings.get_dialog_path("setlist_folder")
        parent_window = self.window() if self.window() else None
        folder_path = QFileDialog.getExistingDirectory(
            parent_window, "Select Folder with Audio Files", start_dir
        )

        if not folder_path:
            return

        app_settings.set_dialog_path("setlist_folder", folder_path)

        setlist_result = self.facade.list_setlists(self.facade.current_project_id)
        setlist = None
        if setlist_result.success and setlist_result.data:
            setlist = setlist_result.data[0]
            self.current_setlist_id = setlist.id
            self.current_setlist = setlist

        import glob
        audio_extensions = [
            '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aif', '.aiff'
        ]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(
                glob.glob(str(Path(folder_path) / f"*{ext}"))
            )
            audio_files.extend(
                glob.glob(str(Path(folder_path) / f"*{ext.upper()}"))
            )
        audio_files = sorted(audio_files)

        if not audio_files:
            QMessageBox.warning(
                self, "No Audio Files",
                f"No audio files found in:\n{folder_path}"
            )
            return

        if not setlist:
            result = self.facade.create_setlist_from_folder(folder_path)
            if result.success:
                setlist = result.data
                self.current_setlist_id = setlist.id
                self.current_setlist = setlist
                Log.info(f"SetlistView: Created setlist from folder {folder_path}")
            else:
                error_msg = result.message
                if result.errors:
                    error_msg += f"\n\n{result.errors[0]}"
                QMessageBox.warning(self, "Failed to Create Setlist", error_msg)
                return
        else:
            added_count = 0
            for audio_file in audio_files:
                result = self.facade.add_song_to_setlist(
                    audio_file, self.facade.current_project_id
                )
                if result.success:
                    added_count += 1
                else:
                    Log.warning(
                        f"SetlistView: Failed to add "
                        f"{Path(audio_file).name}: {result.message}"
                    )

            if added_count == 0:
                QMessageBox.warning(
                    self, "Failed",
                    "Failed to add songs from folder. See logs for details."
                )

        self._load_songs()
        self._update_ui()

    def _on_remove_song(self):
        """Remove selected song(s) from the setlist."""
        if not self.facade.current_project_id:
            return

        selected_rows = self.songs_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        song_names = []
        song_ids = []
        for row_item in selected_rows:
            row = row_item.row()
            if row < len(self.songs):
                song = self.songs[row]
                song_names.append(Path(song.audio_path).name)
                song_ids.append(song.id)

        reply = QMessageBox.question(
            self, "Remove Songs",
            f"Remove {len(song_names)} song(s) from setlist?\n\n"
            + "\n".join(song_names[:5])
            + (f"\n... and {len(song_names) - 5} more"
               if len(song_names) > 5 else ""),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            removed_count = 0
            for song_id in song_ids:
                result = self.facade.remove_song_from_setlist(
                    song_id, self.facade.current_project_id
                )
                if result.success:
                    removed_count += 1
                else:
                    Log.error(
                        f"SetlistView: Failed to remove song "
                        f"{song_id}: {result.message}"
                    )
            if removed_count > 0:
                Log.info(f"SetlistView: Removed {removed_count} song(s)")
                self._load_songs()

    def _remove_single_song(self, song: SetlistSong):
        """Remove a single song via context menu."""
        reply = QMessageBox.question(
            self, "Remove Song",
            f"Remove '{Path(song.audio_path).name}' from setlist?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            result = self.facade.remove_song_from_setlist(
                song.id, self.facade.current_project_id
            )
            if result.success:
                self._load_songs()
            else:
                QMessageBox.warning(
                    self, "Failed",
                    f"Failed to remove song: {result.message}"
                )

    # ------------------------------------------------------------------
    # Song table interaction
    # ------------------------------------------------------------------

    def _on_song_double_clicked(self, item: QTableWidgetItem):
        """Handle double-click on song -- switch to it if processed."""
        row = item.row()
        if row < len(self.songs):
            song = self.songs[row]
            if song.status != "completed":
                QMessageBox.information(
                    self, "Song Not Processed",
                    f"'{Path(song.audio_path).name}' has not been processed yet.\n\n"
                    "Process the song first before switching to it."
                )
                return
            self._switch_to_song(song.id)

    def _on_song_selection_changed(self):
        """Handle row selection change (visual only, does not switch song)."""
        pass

    # ------------------------------------------------------------------
    # UI update helpers
    # ------------------------------------------------------------------

    def _update_ui(self):
        """Update UI to reflect current setlist state."""
        if not self.current_setlist_id or not self.current_setlist:
            self.current_song_id = None
            if self.facade.current_project_id:
                self.action_set_editor.load_project(self.facade.current_project_id)
            self.songs_table.setRowCount(0)
            self.songs = []
            self._update_header_count()
            return

        self._load_songs()

        if self.facade.current_project_id:
            self.action_set_editor.load_project(self.facade.current_project_id)

    def _update_header_count(self):
        """Update the inline song count in the header."""
        total = len(self.songs)
        processed = sum(1 for s in self.songs if s.status == "completed")
        if total == 0:
            self.songs_count_label.setText("")
        elif processed == 0:
            self.songs_count_label.setText(f"{total} songs")
        else:
            self.songs_count_label.setText(
                f"{total} songs, {processed} done"
            )

    def _load_songs(self):
        """Load songs for current setlist into the compact table."""
        if not self.current_setlist_id:
            return

        try:
            result = self.facade.get_setlist_songs(self.current_setlist_id)
            if not result.success:
                Log.error(f"SetlistView: Failed to load songs: {result.message}")
                return

            songs = result.data
            self.songs = songs

            # Fix statuses for songs that have snapshots but wrong status
            if (
                self.facade.current_project_id
                and hasattr(self.facade, 'project_service')
                and self.facade.project_service
            ):
                try:
                    project = self.facade.project_service.load_project(
                        self.facade.current_project_id
                    )
                    if project:
                        for song in songs:
                            snapshot = self.facade.project_service.get_snapshot(
                                song.id, project
                            )
                            if snapshot and song.status != "completed":
                                Log.debug(
                                    f"SetlistView: Fixing song {song.id} "
                                    f"status to 'completed' (has snapshot)"
                                )
                                song.status = "completed"
                                if (
                                    hasattr(self.facade, 'setlist_service')
                                    and self.facade.setlist_service
                                ):
                                    self.facade.setlist_service._setlist_song_repo.update(
                                        song
                                    )
                except Exception as e:
                    Log.debug(
                        f"SetlistView: Could not check snapshots: {e}"
                    )

            # Determine active row
            active_row_index = -1
            if self.current_song_id:
                for idx, song in enumerate(songs):
                    if song.id == self.current_song_id:
                        active_row_index = idx
                        break
            self._row_delegate.active_row = active_row_index

            # Populate table
            self.songs_table.setRowCount(len(songs))

            for row, song in enumerate(songs):
                is_active = song.id == self.current_song_id

                # Column 0: Order number
                order_item = QTableWidgetItem(str(song.order_index + 1))
                order_item.setData(Qt.ItemDataRole.UserRole, song.id)
                order_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                order_item.setForeground(
                    QColor(Colors.TEXT_SECONDARY.name())
                )
                self.songs_table.setItem(row, 0, order_item)

                # Column 1: Song name
                name_item = QTableWidgetItem(Path(song.audio_path).name)
                name_item.setData(Qt.ItemDataRole.UserRole, song.id)
                name_item.setToolTip(song.audio_path)
                if is_active:
                    font = name_item.font()
                    font.setBold(True)
                    name_item.setFont(font)
                self.songs_table.setItem(row, 1, name_item)

                # Column 2: Status indicator
                status_text = self._format_status(song.status)
                status_item = QTableWidgetItem(status_text)
                status_item.setForeground(
                    QColor(self._status_color(song.status).name())
                )
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if song.status == "failed" and song.error_message:
                    status_item.setToolTip(song.error_message)
                self.songs_table.setItem(row, 2, status_item)

            self.songs_table.viewport().update()
            self._update_header_count()

        except Exception as e:
            Log.error(f"SetlistView: Failed to load songs: {e}")

    def _get_row_for_song(self, song_id: str) -> int:
        """Get table row index for a song ID."""
        for row in range(self.songs_table.rowCount()):
            item = self.songs_table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == song_id:
                return row
        return -1

    def _update_song_status_in_table(self, song_id: str, status: str):
        """Update the status cell for a song during processing."""
        row = self._get_row_for_song(song_id)
        if row < 0:
            return
        status_item = self.songs_table.item(row, 2)
        if status_item:
            status_item.setText(self._format_status(status))
            status_item.setForeground(
                QColor(self._status_color(status).name())
            )

    # ------------------------------------------------------------------
    # Status formatting
    # ------------------------------------------------------------------

    def _format_status(self, status: str) -> str:
        """Format status for compact display with indicator symbol."""
        return {
            "pending": "\u25CB Pending",
            "processing": "\u25CF Running",
            "completed": "\u25CF Done",
            "failed": "\u2715 Failed",
        }.get(status, status)

    def _status_color(self, status: str):
        """Get color for a song status."""
        return {
            "pending": Colors.TEXT_SECONDARY,
            "processing": Colors.ACCENT_YELLOW,
            "completed": Colors.ACCENT_GREEN,
            "failed": Colors.ACCENT_RED,
        }.get(status, Colors.TEXT_SECONDARY)

    # ------------------------------------------------------------------
    # Setlist loading
    # ------------------------------------------------------------------

    def load_setlist(self, setlist_id: str, auto_select_song_id: Optional[str] = None):
        """Load a setlist into the view."""
        try:
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

            if auto_select_song_id:
                songs_result = self.facade.get_setlist_songs(setlist_id)
                if songs_result.success and songs_result.data:
                    song = next(
                        (s for s in songs_result.data
                         if s.id == auto_select_song_id),
                        None,
                    )
                    if song and song.status == "completed":
                        self._switch_to_song(auto_select_song_id)
                        Log.info(
                            f"SetlistView: Auto-selected saved active song: "
                            f"{Path(song.audio_path).name}"
                        )
                    else:
                        auto_select_song_id = None

            if not auto_select_song_id:
                songs_result = self.facade.get_setlist_songs(setlist_id)
                if songs_result.success and songs_result.data:
                    completed = [
                        s for s in songs_result.data
                        if s.status == "completed"
                    ]
                    if completed:
                        last_song = max(
                            completed, key=lambda s: s.order_index
                        )
                        self._switch_to_song(last_song.id)
                        Log.info(
                            f"SetlistView: Auto-selected last processed: "
                            f"{Path(last_song.audio_path).name}"
                        )

            if setlist.audio_folder_path:
                Log.info(
                    f"SetlistView: Loaded setlist for folder "
                    f"'{Path(setlist.audio_folder_path).name}'"
                )
            else:
                Log.info("SetlistView: Loaded setlist (no folder path)")
        except Exception as e:
            Log.error(f"SetlistView: Failed to load setlist: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load setlist: {e}")

    def refresh(self):
        """Refresh the view."""
        self._update_ui()

"""
Setlist Processing Dialog

Shows detailed progress of setlist processing with:
- List of all songs being processed
- List of all actions for each song
- Checkmarks as actions complete
- Real-time status updates
- Verbose timing and status information from ProgressEventStore
- Block-level progress from SubprocessProgress events (e.g., Demucs separation %)
"""
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QProgressBar,
    QWidget, QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QColor

from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory
from src.application.services import get_progress_store, ProgressStatus
from src.application.events import EventBus, SubprocessProgress
from src.utils.message import Log


class SetlistProcessingDialog(ThemeAwareMixin, QDialog):
    """
    Dialog showing detailed setlist processing progress.
    
    Layout:
    ┌─────────────────────────────────────────────────────┐
    │ Processing Setlist: "My Setlist"                    │
    ├─────────────────────────────────────────────────────┤
    │ Overall Progress: [▓▓▓▓▓▓░░░░] 50%                  │
    ├─────────────────────────────────────────────────────┤
    │ 📁 Song 1: audio1.wav                               │
    │   ✓ LoadAudio1 → set_file_path                      │
    │   → DetectOnsets → detect_onsets                    │
    │   ○ SeparatorBlock → separate_audio                 │
    │                                                      │
    │ 📁 Song 2: audio2.wav                               │
    │   ○ LoadAudio1 → set_file_path                      │
    │   ○ DetectOnsets → detect_onsets                    │
    │   ○ SeparatorBlock → separate_audio                 │
    ├─────────────────────────────────────────────────────┤
    │                                    [Close] [Cancel]  │
    └─────────────────────────────────────────────────────┘
    """
    
    # Signal emitted when user cancels
    cancelled = pyqtSignal()
    # Thread-safe signal for block-level progress updates from EventBus thread
    _block_progress_signal = pyqtSignal(str, int, str)  # block_name, percentage, message
    
    def __init__(self, setlist_name: str, songs: List[Dict[str, Any]], action_items: List[Dict[str, Any]], parent=None, event_bus: Optional[EventBus] = None):
        """
        Initialize processing dialog.
        
        Args:
            setlist_name: Name of the setlist being processed
            songs: List of song dicts with 'id', 'audio_path', 'name' keys
            action_items: List of action item dicts with 'action_name', 'block_name' keys
            parent: Parent widget
            event_bus: Optional event bus for subscribing to SubprocessProgress events
        """
        super().__init__(parent)
        self.setlist_name = setlist_name
        self.songs = songs
        self.action_items = action_items
        
        # Track processing state
        self.song_items: Dict[str, QTreeWidgetItem] = {}  # song_id -> song item
        self.action_items_map: Dict[str, Dict[int, QTreeWidgetItem]] = {}  # song_id -> {action_index: action item}
        self.completed_songs = 0
        self.total_songs = len(songs)
        self.cancelled_flag = False
        
        # Progress store integration for verbose information
        self._progress_store = get_progress_store()
        self._operation_id: Optional[str] = None
        self._start_time: datetime = datetime.now()  # Start timing immediately
        self._failed_songs = 0
        self._current_song_name: str = ""
        self._current_action_name: str = ""
        
        # Block-level progress tracking from SubprocessProgress events
        self._event_bus = event_bus
        self._subprocess_handler: Optional[Callable] = None  # Store handler for unsubscription
        self._current_block_name: str = ""
        self._current_block_progress: int = 0
        self._current_block_message: str = ""
        
        self.setWindowTitle(f"Processing Setlist: {setlist_name}")
        self.setMinimumSize(700, 600)  # Slightly larger for verbose info
        self.setModal(True)
        
        self._setup_ui()
        self._populate_tree()
        
        # Connect thread-safe signal for block progress updates
        self._block_progress_signal.connect(self._update_block_progress)
        
        # Subscribe to SubprocessProgress events for block-level progress
        self._subscribe_to_events()
        
        # Timer for updating verbose information
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_verbose_info)
        self._update_timer.start(250)  # Update every 250ms
        
        self._init_theme_aware()
        Log.info(f"SetlistProcessingDialog: Created for {len(songs)} songs, {len(action_items)} actions")
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        
        # Title
        title_label = QLabel(f"Processing Setlist: {self.setlist_name}")
        title_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {Colors.TEXT_PRIMARY.name()};
            padding-bottom: {Spacing.SM}px;
        """)
        layout.addWidget(title_label)
        
        # Overall progress bar
        self.overall_progress = QProgressBar()
        self.overall_progress.setMinimum(0)
        self.overall_progress.setMaximum(100)
        self.overall_progress.setValue(0)
        self.overall_progress.setTextVisible(True)
        self.overall_progress.setFormat("0%")
        self.overall_progress.setStyleSheet(StyleFactory.progress_bar())
        self.overall_progress.setFixedHeight(30)
        layout.addWidget(self.overall_progress)
        
        # Block-level progress bar (for detailed subprocess progress like Demucs)
        self._block_progress_frame = QFrame()
        self._block_progress_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: {Spacing.XS}px;
            }}
        """)
        block_progress_layout = QVBoxLayout(self._block_progress_frame)
        block_progress_layout.setSpacing(Spacing.XS)
        block_progress_layout.setContentsMargins(Spacing.SM, Spacing.XS, Spacing.SM, Spacing.XS)
        
        # Block name label
        self._block_name_label = QLabel("Block: Waiting...")
        self._block_name_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        block_progress_layout.addWidget(self._block_name_label)
        
        # Block progress bar
        self._block_progress = QProgressBar()
        self._block_progress.setMinimum(0)
        self._block_progress.setMaximum(100)
        self._block_progress.setValue(0)
        self._block_progress.setTextVisible(True)
        self._block_progress.setFormat("Waiting...")
        self._block_progress.setStyleSheet(StyleFactory.progress_bar(compact=True))
        self._block_progress.setFixedHeight(20)
        block_progress_layout.addWidget(self._block_progress)
        
        layout.addWidget(self._block_progress_frame)
        
        # Verbose status panel
        self._status_frame = QFrame()
        self._status_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: {Spacing.SM}px;
            }}
        """)
        status_layout = QHBoxLayout(self._status_frame)
        status_layout.setSpacing(Spacing.LG)
        status_layout.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.SM)
        
        # Elapsed time
        self._elapsed_label = QLabel("Elapsed: --:--")
        self._elapsed_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 12px;")
        status_layout.addWidget(self._elapsed_label)
        
        # Current operation
        self._current_op_label = QLabel("Status: Waiting...")
        self._current_op_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 12px;")
        status_layout.addWidget(self._current_op_label, stretch=1)
        
        # Completed/Failed counts
        self._stats_label = QLabel("Completed: 0 | Failed: 0")
        self._stats_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 12px;")
        status_layout.addWidget(self._stats_label)
        
        layout.addWidget(self._status_frame)
        
        # Tree widget showing songs and actions
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Progress")
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setIndentation(20)
        self.tree.setStyleSheet(StyleFactory.tree())
        self.tree.header().setVisible(False)
        layout.addWidget(self.tree, stretch=1)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(StyleFactory.button())
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet(StyleFactory.button())
        self.close_btn.setEnabled(False)  # Enabled when processing completes
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def _populate_tree(self):
        """Populate the tree with songs and actions"""
        self.tree.clear()
        self.song_items.clear()
        self.action_items_map.clear()
        
        for song in self.songs:
            song_id = song.get('id', '')
            song_name = song.get('name', Path(song.get('audio_path', '')).name)
            audio_path = song.get('audio_path', '')
            
            # Create song item
            song_item = QTreeWidgetItem(self.tree)
            song_item.setText(0, f"📁 {song_name}")
            song_item.setExpanded(True)
            song_item.setData(0, Qt.ItemDataRole.UserRole, song_id)
            self.song_items[song_id] = song_item
            
            # Initialize action items map for this song
            self.action_items_map[song_id] = {}
            
            # Add action items as children
            for idx, action_item in enumerate(self.action_items):
                action_name = action_item.get('action_name', 'Unknown')
                block_name = action_item.get('block_name', '')
                
                # Format: "BlockName → action_name" or just "action_name"
                if block_name:
                    action_text = f"{block_name} → {action_name}"
                else:
                    action_text = action_name
                
                action_tree_item = QTreeWidgetItem(song_item)
                action_tree_item.setText(0, f"○ {action_text}")
                action_tree_item.setData(0, Qt.ItemDataRole.UserRole, idx)
                self.action_items_map[song_id][idx] = action_tree_item
    
    def update_action_status(self, song_id: str, action_index: int, status: str):
        """
        Update the status of a specific action.
        
        Args:
            song_id: Song identifier
            action_index: Index of the action (0-based)
            status: Status string ("pending", "running", "completed", "failed")
        """
        if song_id not in self.action_items_map:
            return
        
        if action_index not in self.action_items_map[song_id]:
            return
        
        action_item = self.action_items_map[song_id][action_index]
        current_text = action_item.text(0)
        
        # Remove status prefix if present
        if current_text.startswith("○ ") or current_text.startswith("→ ") or current_text.startswith("✓ ") or current_text.startswith("✗ "):
            action_text = current_text[2:]
        else:
            action_text = current_text
        
        # Track current action for status display
        if status == "running":
            self._current_action_name = action_text
        
        # Update based on status
        if status == "pending":
            action_item.setText(0, f"○ {action_text}")
            action_item.setForeground(0, QColor(Colors.TEXT_SECONDARY.name()))
        elif status == "running":
            action_item.setText(0, f"→ {action_text}")
            action_item.setForeground(0, QColor(Colors.ACCENT_BLUE.name()))
        elif status == "completed":
            action_item.setText(0, f"✓ {action_text}")
            action_item.setForeground(0, QColor(Colors.SUCCESS.name() if hasattr(Colors, 'SUCCESS') else Colors.ACCENT_GREEN.name() if hasattr(Colors, 'ACCENT_GREEN') else Colors.TEXT_PRIMARY.name()))
        elif status == "failed":
            action_item.setText(0, f"✗ {action_text}")
            action_item.setForeground(0, QColor(Colors.ACCENT_RED.name()))
        
        # Ensure the item is visible
        action_item.setExpanded(True)
        self.tree.scrollToItem(action_item)
    
    def update_song_status(self, song_id: str, status: str):
        """
        Update the overall status of a song.
        
        Args:
            song_id: Song identifier
            status: Status string ("pending", "processing", "completed", "failed")
        """
        if song_id not in self.song_items:
            return
        
        song_item = self.song_items[song_id]
        current_text = song_item.text(0)
        
        # Extract song name (remove status prefix if present)
        if current_text.startswith("📁 "):
            song_name = current_text[2:]
        elif current_text.startswith(("✓ ", "✗ ", "→ ")):
            song_name = current_text[2:]
        else:
            song_name = current_text
        
        # Track current song for status display
        if status == "processing":
            self._current_song_name = song_name
            self._current_action_name = ""
        
        # Update based on status
        if status == "completed":
            song_item.setText(0, f"✓ {song_name}")
            song_item.setForeground(0, QColor(Colors.SUCCESS.name() if hasattr(Colors, 'SUCCESS') else Colors.ACCENT_GREEN.name() if hasattr(Colors, 'ACCENT_GREEN') else Colors.TEXT_PRIMARY.name()))
            self.completed_songs += 1
        elif status == "failed":
            song_item.setText(0, f"✗ {song_name}")
            song_item.setForeground(0, QColor(Colors.ACCENT_RED.name()))
            self.completed_songs += 1
            self._failed_songs += 1
        elif status == "processing":
            song_item.setText(0, f"→ {song_name}")
            song_item.setForeground(0, QColor(Colors.ACCENT_BLUE.name()))
        
        # Update overall progress
        if self.total_songs > 0:
            progress = int((self.completed_songs / self.total_songs) * 100)
            self.overall_progress.setValue(progress)
            self.overall_progress.setFormat(f"{progress}% ({self.completed_songs}/{self.total_songs} songs)")
    
    def set_operation_id(self, operation_id: str):
        """
        Set the operation ID to track in progress store.
        
        Args:
            operation_id: Operation identifier from ProgressContext
        """
        self._operation_id = operation_id
        self._start_time = datetime.now()
        Log.debug(f"SetlistProcessingDialog: Tracking operation {operation_id}")
    
    def _update_verbose_info(self):
        """Update verbose status information from progress store"""
        # Update elapsed time
        elapsed = (datetime.now() - self._start_time).total_seconds()
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self._elapsed_label.setText(f"Elapsed: {minutes:02d}:{seconds:02d}")
        
        # Read from progress store (single source of truth)
        if self._operation_id:
            state = self._progress_store.get_state(self._operation_id)
            if state:
                overall = state.get_overall_progress()
                self._stats_label.setText(
                    f"Completed: {overall['completed']} | Failed: {overall['failed']}"
                )
                
                # Find current running level for status
                current_message = "Processing..."
                for level in state.levels.values():
                    if level.status == ProgressStatus.RUNNING:
                        current_message = f"{level.name}: {level.message}"
                        break
                
                self._current_op_label.setText(f"Status: {current_message}")
                return
        
        # Store not yet available -- show waiting state
        self._stats_label.setText(
            f"Completed: {self.completed_songs - self._failed_songs} | Failed: {self._failed_songs}"
        )
        if self.completed_songs == 0:
            self._current_op_label.setText("Status: Starting...")
    
    def processing_complete(self):
        """Called when all processing is complete"""
        self._update_timer.stop()
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.overall_progress.setValue(100)
        self.overall_progress.setFormat("Complete")
        
        # Final elapsed time
        elapsed = (datetime.now() - self._start_time).total_seconds()
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self._elapsed_label.setText(f"Total: {minutes:02d}:{seconds:02d}")
        
        # Final stats
        success_count = self.completed_songs - self._failed_songs
        self._stats_label.setText(
            f"Completed: {success_count} | Failed: {self._failed_songs}"
        )
        
        # Final status message
        if self._failed_songs > 0:
            self._current_op_label.setText(f"Status: Completed with {self._failed_songs} error(s)")
        else:
            self._current_op_label.setText("Status: All songs processed successfully")
        
        Log.info(f"SetlistProcessingDialog: Processing complete - {success_count} succeeded, {self._failed_songs} failed, {elapsed:.1f}s elapsed")
    
    def _on_cancel(self):
        """Handle cancel button click"""
        self._update_timer.stop()
        self.cancelled_flag = True
        self.cancelled.emit()
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling...")
        self._current_op_label.setText("Status: Cancelling...")
        Log.info("SetlistProcessingDialog: Cancellation requested")
    
    def closeEvent(self, event):
        """Clean up timer and event subscriptions when dialog closes"""
        self._update_timer.stop()
        self._unsubscribe_from_events()
        super().closeEvent(event)
    
    def _subscribe_to_events(self):
        """Subscribe to SubprocessProgress events for block-level progress"""
        if not self._event_bus:
            Log.warning("SetlistProcessingDialog: No event bus provided, block progress will not be shown")
            return
        
        Log.info(f"SetlistProcessingDialog: Event bus available, subscribing to SubprocessProgress...")
        
        def on_subprocess_progress(event: SubprocessProgress):
            """Handle SubprocessProgress events from blocks"""
            try:
                data = event.data or {}
                block_name = data.get("block_name", "Unknown")
                percentage = data.get("percentage", 0)
                message = data.get("message", "Processing...")
                
                Log.debug(f"SetlistProcessingDialog: Received SubprocessProgress - {block_name}: {percentage}% - {message}")
                
                # Update block progress state
                self._current_block_name = block_name
                self._current_block_progress = percentage
                self._current_block_message = message
                
                # Marshal to main thread via signal (EventBus runs on background thread)
                self._block_progress_signal.emit(block_name, percentage, message)
                
            except Exception as e:
                Log.warning(f"SetlistProcessingDialog: Error handling SubprocessProgress: {e}")
        
        # Store handler for later unsubscription
        self._subprocess_handler = on_subprocess_progress
        self._event_bus.subscribe(SubprocessProgress.name, on_subprocess_progress)
        Log.info(f"SetlistProcessingDialog: Subscribed to SubprocessProgress events (event_bus id: {id(self._event_bus)})")
    
    def _unsubscribe_from_events(self):
        """Unsubscribe from events"""
        if self._event_bus and self._subprocess_handler:
            try:
                self._event_bus.unsubscribe(SubprocessProgress.name, self._subprocess_handler)
                Log.debug("SetlistProcessingDialog: Unsubscribed from SubprocessProgress events")
            except Exception as e:
                Log.warning(f"SetlistProcessingDialog: Error unsubscribing from events: {e}")
            self._subprocess_handler = None
    
    def _update_block_progress(self, block_name: str, percentage: int, message: str):
        """
        Update the block-level progress bar.
        
        Args:
            block_name: Name of the block being processed
            percentage: Progress percentage (0-100)
            message: Progress message from the block
        """
        try:
            self._block_name_label.setText(f"Block: {block_name}")
            self._block_progress.setValue(percentage)
            self._block_progress.setFormat(f"{message}")
            
            # Also update the current action status in the verbose panel
            self._current_action_name = f"{block_name}: {message}"
            
        except Exception as e:
            Log.warning(f"SetlistProcessingDialog: Error updating block progress: {e}")
    
    def is_cancelled(self) -> bool:
        """Check if processing was cancelled"""
        return self.cancelled_flag
    
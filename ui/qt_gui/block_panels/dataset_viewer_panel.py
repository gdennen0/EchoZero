"""
Dataset Viewer block panel.

Allows manual auditing of a directory of audio clips: select source dir,
step through samples, play audio (waveform + player), remove samples
into a "removed" subdirectory. Each file has a checkbox for batch selection.
Keyword search (additive) checks all files whose name contains the text.
Select current checks the focused file (additive). Batch: Remove selected.
Shortcuts: Space = play/pause, Ctrl/Cmd+D = remove, Left/Right = prev/next,
Ctrl/Cmd+Z = undo, Ctrl/Cmd+Shift+S = select current.
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QAbstractItemView,
    QSlider, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QLineEdit,
)
from PyQt6.QtCore import Qt, QTimer, QEvent
from PyQt6.QtGui import QKeySequence, QShortcut

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from ui.qt_gui.node_editor.audio_player_block_item import WaveformWidget
from src.application.settings.dataset_viewer_settings import (
    DatasetViewerSettingsManager,
)
from src.utils.message import Log
from src.utils.settings import app_settings

# Audio file extensions to list in the sample list
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}:{secs:02d}"


# Role to store absolute path on file items (QTreeWidgetItem.setData(0, PATH_ROLE, path))
PATH_ROLE = Qt.ItemDataRole.UserRole


def _list_audio_files_recursive(
    source_dir: str, removed_subdir: str
) -> list:
    """
    Return list of (relative_path, absolute_path) for all audio files under
    source_dir recursively. Skips the 'removed' subdir. Paths are in
    depth-first order (sorted by relative_path).
    """
    source = Path(source_dir)
    if not source.is_dir():
        return []
    removed_name = (removed_subdir or "removed").strip() or "removed"
    out = []

    def scan(dir_path: Path, rel_prefix: str) -> None:
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for f in entries:
            if f.name.startswith("."):
                continue
            rel = f"{rel_prefix}{f.name}" if rel_prefix else f.name
            if f.is_file():
                if f.suffix.lower() in AUDIO_EXTENSIONS:
                    out.append((rel, str(f.resolve())))
            else:
                if f.name == removed_name:
                    continue
                scan(f, rel + os.sep)

    scan(source, "")
    return out


def _build_tree(
    tree: QTreeWidget,
    source_dir: str,
    path_list: list,
) -> list:
    """
    path_list: list of (relative_path, absolute_path).
    Build tree under tree's invisible root; return list of QTreeWidgetItem
    for each file in same order as path_list (for prev/next).
    """
    source = Path(source_dir)
    root = tree.invisibleRootItem()
    # Count files recursively for each directory tuple path.
    # Example: "a/b/file.wav" increments ("a",) and ("a", "b").
    recursive_counts = {}
    for rel_path, _ in path_list:
        parts = rel_path.replace("/", os.sep).split(os.sep)
        for i in range(1, len(parts)):
            key = tuple(parts[:i])
            recursive_counts[key] = recursive_counts.get(key, 0) + 1

    # node path (tuple of segments) -> QTreeWidgetItem for that dir
    dir_nodes = {(): root}
    file_items = []

    for rel_path, abs_path in path_list:
        parts = rel_path.replace("/", os.sep).split(os.sep)
        parent = dir_nodes[()]
        for i, part in enumerate(parts[:-1]):
            key = tuple(parts[: i + 1])
            if key not in dir_nodes:
                count = recursive_counts.get(key, 0)
                node = QTreeWidgetItem(parent, [f"{part} ({count})"])
                node.setData(0, PATH_ROLE, None)
                dir_nodes[key] = node
                parent.addChild(node)
            parent = dir_nodes[key]
        # file (checkable for batch selection)
        file_item = QTreeWidgetItem(parent, [parts[-1]])
        file_item.setData(0, PATH_ROLE, abs_path)
        file_item.setFlags(file_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        file_item.setCheckState(0, Qt.CheckState.Unchecked)
        parent.addChild(file_item)
        file_items.append(file_item)

    return file_items


@register_block_panel("DatasetViewer")
class DatasetViewerPanel(BlockPanelBase):
    """Panel for DatasetViewer: source dir, sample list, waveform, player, remove/prev/next."""

    def __init__(self, block_id: str, facade, parent=None):
        super().__init__(block_id, facade, parent)

        self._settings_manager = DatasetViewerSettingsManager(
            facade, block_id, parent=self
        )
        self._settings_manager.settings_changed.connect(self._on_setting_changed)

        self._player = None
        self._is_playing = False
        self._duration = 0.0
        self._slider_dragging = False
        self._file_paths = []  # Flat list of absolute paths (depth-first order)
        self._tree_file_items = []  # QTreeWidgetItem per file, same order as _file_paths
        self._undo_stack = []  # List of (dest_path, source_path) for undo restore
        self._last_loaded_path = None  # For replay: same file -> seek 0 and play

        if self.block:
            self.refresh()

    def create_content_widget(self) -> QWidget:
        """Create Dataset Viewer UI: dir, list, waveform, player, buttons."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)

        # Source directory
        dir_group = QGroupBox("Source Directory")
        dir_layout = QVBoxLayout(dir_group)
        self.dir_path_label = QLabel("No directory selected")
        self.dir_path_label.setWordWrap(True)
        self.dir_path_label.setStyleSheet(f"""
            QLabel {{
                background-color: {Colors.BG_LIGHT.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: {Spacing.SM}px;
                color: {Colors.TEXT_SECONDARY.name()};
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        dir_layout.addWidget(self.dir_path_label)
        browse_btn = QPushButton("Browse for Directory...")
        browse_btn.clicked.connect(self._on_browse_directory)
        dir_layout.addWidget(browse_btn)
        layout.addWidget(dir_group)

        # Sample tree (recursive dirs + files with indentation)
        list_group = QGroupBox("Samples")
        list_layout = QVBoxLayout(list_group)
        self.sample_tree = QTreeWidget()
        self.sample_tree.setHeaderLabels(["Name"])
        self.sample_tree.setMinimumHeight(160)
        self.sample_tree.setIndentation(20)
        self.sample_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.sample_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sample_tree.currentItemChanged.connect(self._on_tree_selection_changed)
        self.sample_tree.itemChanged.connect(self._on_tree_item_changed)
        self.sample_tree.installEventFilter(self)
        list_layout.addWidget(self.sample_tree)

        # Keyword search: select matching (additive), clear selection
        search_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Keyword to select (e.g. crash)")
        self.search_edit.setClearButtonEnabled(True)
        search_layout.addWidget(self.search_edit, 1)
        self.select_matching_btn = QPushButton("Select matching")
        self.select_matching_btn.setToolTip("Check all files whose name contains the keyword (additive)")
        self.select_matching_btn.clicked.connect(self._select_matching)
        search_layout.addWidget(self.select_matching_btn)
        self.clear_selection_btn = QPushButton("Clear selection")
        self.clear_selection_btn.setToolTip("Uncheck all files")
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        self.select_current_btn = QPushButton("Select current")
        self.select_current_btn.setToolTip("Check the currently selected file (additive)")
        self.select_current_btn.clicked.connect(self._select_current)
        search_layout.addWidget(self.clear_selection_btn)
        search_layout.addWidget(self.select_current_btn)
        list_layout.addLayout(search_layout)

        # Navigation and batch actions
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Previous")
        self.prev_btn.clicked.connect(self._go_previous)
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._go_next)
        self.remove_btn = QPushButton("Remove (move to removed/)")
        self.remove_btn.clicked.connect(self._remove_current)
        self.remove_selected_btn = QPushButton("Remove selected")
        self.remove_selected_btn.setToolTip("Move all checked files to removed/")
        self.remove_selected_btn.clicked.connect(self._remove_selected)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip("Move last removed file back to its original location")
        self.undo_btn.clicked.connect(self._undo_remove)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(self.remove_btn)
        nav_layout.addWidget(self.remove_selected_btn)
        nav_layout.addWidget(self.undo_btn)
        list_layout.addLayout(nav_layout)
        layout.addWidget(list_group)

        # Waveform
        waveform_group = QGroupBox("Waveform")
        wf_layout = QVBoxLayout(waveform_group)
        self.waveform = WaveformWidget()
        wf_layout.addWidget(self.waveform)
        layout.addWidget(waveform_group)

        # Player: slider + time + play
        player_group = QGroupBox("Playback")
        player_layout = QVBoxLayout(player_group)
        scrubber = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setFixedHeight(16)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_slider_dragging", True))
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        scrubber.addWidget(self.slider, 1)
        self.time_label = QLabel("0:00")
        self.time_label.setFixedWidth(36)
        scrubber.addWidget(self.time_label)
        player_layout.addLayout(scrubber)
        self.play_btn = QPushButton("Play")
        self.play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_btn.clicked.connect(self._toggle_playback)
        player_layout.addWidget(self.play_btn)
        layout.addWidget(player_group)

        # Keyboard shortcuts
        self._shortcut_play = QShortcut(QKeySequence(Qt.Key.Key_Space), widget)
        self._shortcut_play.activated.connect(self._toggle_playback)
        self._shortcut_delete = QShortcut(QKeySequence("Ctrl+D"), widget)
        self._shortcut_delete.activated.connect(self._remove_current)
        self._shortcut_delete_mac = QShortcut(QKeySequence("Meta+D"), widget)
        self._shortcut_delete_mac.activated.connect(self._remove_current)
        self._shortcut_prev = QShortcut(QKeySequence(Qt.Key.Key_Left), widget)
        self._shortcut_prev.activated.connect(self._go_previous)
        self._shortcut_next = QShortcut(QKeySequence(Qt.Key.Key_Right), widget)
        self._shortcut_next.activated.connect(self._go_next)
        self._shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), widget)
        self._shortcut_undo.activated.connect(self._undo_remove)
        self._shortcut_undo_mac = QShortcut(QKeySequence("Meta+Z"), widget)
        self._shortcut_undo_mac.activated.connect(self._undo_remove)
        self._shortcut_select_current = QShortcut(QKeySequence("Ctrl+Shift+S"), widget)
        self._shortcut_select_current.activated.connect(self._select_current)
        self._shortcut_select_current_mac = QShortcut(QKeySequence("Meta+Shift+S"), widget)
        self._shortcut_select_current_mac.activated.connect(self._select_current)

        layout.addStretch()
        self._init_player()
        self._start_update_timer()
        return widget

    def eventFilter(self, obj, event):
        """Intercept Space in tree so it triggers play instead of expand/collapse."""
        if obj is self.sample_tree and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Space:
                self._toggle_playback()
                return True
        return super().eventFilter(obj, event)

    def _init_player(self):
        """Initialize SimpleAudioPlayer for playback."""
        try:
            from ui.qt_gui.widgets.timeline.playback.controller import SimpleAudioPlayer
            self._player = SimpleAudioPlayer()
        except Exception as e:
            Log.warning(f"DatasetViewerPanel: Could not init audio player: {e}")
            self._player = None

    def _start_update_timer(self):
        """Timer to update slider and time during playback."""
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(33)
        self._update_timer.timeout.connect(self._on_update_tick)

    def _on_browse_directory(self):
        """Open dialog to select source directory of audio clips."""
        current = self._settings_manager.source_dir or ""
        start = current or app_settings.get_dialog_path("dataset_viewer")
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Source Directory of Audio Clips", start
        )
        if dir_path:
            app_settings.set_dialog_path("dataset_viewer", dir_path)
            try:
                self._settings_manager.source_dir = dir_path
                self.set_status_message("Source directory set")
            except ValueError as e:
                self.set_status_message(str(e), error=True)
            self._refresh_sample_list()

    def _on_setting_changed(self, key: str):
        """React to settings change (e.g. source_dir)."""
        if key == "source_dir":
            self._undo_stack.clear()
            self._refresh_sample_list()
        self.refresh()

    def _refresh_sample_list(self, preferred_row: int = 0):
        """Reload file list recursively from source_dir and build tree."""
        if not hasattr(self, "sample_tree"):
            return
        source_dir = self._settings_manager.source_dir or ""
        removed_subdir = self._settings_manager.removed_subdir or "removed"
        path_list = _list_audio_files_recursive(source_dir, removed_subdir)
        self._file_paths = [abs_path for _, abs_path in path_list]

        self.sample_tree.blockSignals(True)
        self.sample_tree.clear()
        self._tree_file_items = []
        if path_list and source_dir:
            self._tree_file_items = _build_tree(
                self.sample_tree, source_dir, path_list
            )
            self.sample_tree.expandAll()
        self.sample_tree.blockSignals(False)

        if self._tree_file_items:
            idx = max(0, min(preferred_row, len(self._tree_file_items) - 1))
            self.sample_tree.setCurrentItem(self._tree_file_items[idx])
            path = self._tree_file_items[idx].data(0, PATH_ROLE)
            if path:
                self._on_file_selected(path)
        else:
            self.waveform.clear()
            self._stop_playback()
            self.time_label.setText("0:00")
            self.slider.setValue(0)
        self._update_nav_buttons()
        self._update_undo_button()
        self._update_remove_selected_button()

    def _current_file_index(self) -> int:
        """Index of current tree file item, or -1."""
        item = self.sample_tree.currentItem()
        if not item:
            return -1
        try:
            return self._tree_file_items.index(item)
        except ValueError:
            return -1

    def _update_nav_buttons(self):
        """Enable/disable Previous/Next/Remove based on list state."""
        n = len(self._file_paths)
        idx = self._current_file_index()
        self.prev_btn.setEnabled(n > 0 and idx > 0)
        self.next_btn.setEnabled(n > 0 and idx >= 0 and idx < n - 1)
        self.remove_btn.setEnabled(n > 0 and idx >= 0)

    def _update_undo_button(self):
        """Enable Undo button only when there is something to undo."""
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(len(self._undo_stack) > 0)

    def _update_remove_selected_button(self):
        """Enable Remove selected only when at least one file is checked."""
        if hasattr(self, "remove_selected_btn") and self._tree_file_items:
            any_checked = any(
                item.checkState(0) == Qt.CheckState.Checked
                for item in self._tree_file_items
            )
            self.remove_selected_btn.setEnabled(any_checked)
        elif hasattr(self, "remove_selected_btn"):
            self.remove_selected_btn.setEnabled(False)

    def _on_tree_item_changed(self, item, column):
        """Update batch button when user checks/unchecks an item."""
        self._update_remove_selected_button()

    def _on_tree_selection_changed(self, current, previous):
        """When tree selection changes, load file if a file item is selected."""
        if not current:
            return
        path = current.data(0, PATH_ROLE)
        if path:
            self._on_file_selected(path)
        self._update_nav_buttons()
        self._update_undo_button()
        self._update_remove_selected_button()

    def _on_file_selected(self, path: str):
        """Load waveform and player for the given file path."""
        self.waveform.load_audio(path)
        self._last_loaded_path = None  # Next play will load this file
        self._stop_playback()
        self.slider.setValue(0)
        self._update_duration_from_file(path)
        self._update_nav_buttons()
        self._update_undo_button()
        self._update_remove_selected_button()

    def _update_duration_from_file(self, file_path: str):
        """Set duration and time label from file (e.g. via soundfile)."""
        try:
            import soundfile as sf
            info = sf.info(file_path)
            self._duration = info.duration
            self.time_label.setText(_format_time(self._duration))
        except Exception:
            self._duration = 0.0
            self.time_label.setText("0:00")

    def _toggle_playback(self):
        """Play or pause current sample."""
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        """Start playback of current sample."""
        idx = self._current_file_index()
        path = self._file_paths[idx] if 0 <= idx < len(self._file_paths) else None
        if not self._player:
            self._init_player()
        if not self._player:
            self.set_status_message("Audio playback not available (Qt Multimedia failed to initialize)", error=True)
            return
        if idx < 0 or idx >= len(self._file_paths):
            return
        # Replay: same file already loaded -> seek to 0 and play (avoids load/LoadedMedia)
        if path and path == getattr(self, "_last_loaded_path", None):
            self._player.set_position(0)
            self.slider.setValue(0)
            self.waveform.set_position(0.0)
            self._player.play()
            self._is_playing = True
            self.play_btn.setText("Pause")
            self._update_timer.start()
            return
        if not self._player.load(path, play_when_ready=True):
            Log.warning(f"DatasetViewerPanel: Failed to load {path}")
            return
        self._last_loaded_path = path
        self._is_playing = True
        self.play_btn.setText("Pause")
        self._update_timer.start()
        # Start when loaded (SimpleAudioPlayer), with fallback in case LoadedMedia does not fire
        QTimer.singleShot(400, self._start_playback)

    def _start_playback(self):
        """Called after load delay; start playback if still in playing state."""
        if self._player and self._is_playing:
            self._player.play()

    def _pause(self):
        """Pause playback."""
        if self._player:
            self._player.pause()
        self._is_playing = False
        self.play_btn.setText("Play")
        self._update_timer.stop()

    def _stop_playback(self):
        """Stop and reset position."""
        if self._player:
            self._player.stop()
        self._is_playing = False
        if hasattr(self, "play_btn"):
            self.play_btn.setText("Play")
        if hasattr(self, "_update_timer"):
            self._update_timer.stop()
        if hasattr(self, "slider"):
            self.slider.setValue(0)
        if hasattr(self, "time_label"):
            self.time_label.setText("0:00")
        if hasattr(self, "waveform"):
            self.waveform.set_position(0.0)

    def _on_slider_released(self):
        """Seek to slider position."""
        self._slider_dragging = False
        if self._player and self._duration > 0:
            ratio = self.slider.value() / 1000.0
            self._player.set_position(ratio * self._duration)

    def _on_slider_moved(self, value: int):
        """Update time label and waveform playhead."""
        if self._duration > 0:
            secs = (value / 1000.0) * self._duration
            self.time_label.setText(_format_time(secs))
            self.waveform.set_position(value / 1000.0)

    def _on_update_tick(self):
        """Update slider and time from player position."""
        if not self._player or not self._is_playing:
            return
        if not self._player.is_playing():
            self._stop_playback()
            return
        pos = self._player.get_position()
        if self._duration > 0 and not self._slider_dragging:
            ratio = pos / self._duration
            self.slider.setValue(int(ratio * 1000))
            self.waveform.set_position(ratio)
        self.time_label.setText(_format_time(pos))

    def _go_previous(self):
        """Select previous sample."""
        idx = self._current_file_index()
        if idx > 0:
            self.sample_tree.setCurrentItem(self._tree_file_items[idx - 1])

    def _go_next(self):
        """Select next sample."""
        idx = self._current_file_index()
        if idx >= 0 and idx < len(self._file_paths) - 1:
            self.sample_tree.setCurrentItem(self._tree_file_items[idx + 1])

    def _select_matching(self):
        """Check all files whose name contains the search keyword (additive)."""
        keyword = self.search_edit.text().strip()
        if not keyword:
            return
        keyword_lower = keyword.lower()
        self.sample_tree.blockSignals(True)
        try:
            for item in self._tree_file_items:
                if keyword_lower in item.text(0).lower():
                    item.setCheckState(0, Qt.CheckState.Checked)
        finally:
            self.sample_tree.blockSignals(False)
        self._update_remove_selected_button()
        count = sum(1 for item in self._tree_file_items if item.checkState(0) == Qt.CheckState.Checked)
        self.set_status_message(f"Selected {count} matching (additive)")

    def _clear_selection(self):
        """Uncheck all file items."""
        self.sample_tree.blockSignals(True)
        try:
            for item in self._tree_file_items:
                item.setCheckState(0, Qt.CheckState.Unchecked)
        finally:
            self.sample_tree.blockSignals(False)
        self._update_remove_selected_button()
        self.set_status_message("Selection cleared")

    def _select_current(self):
        """Check the currently selected file item (additive)."""
        item = self.sample_tree.currentItem()
        if not item:
            return
        if item.data(0, PATH_ROLE):
            item.setCheckState(0, Qt.CheckState.Checked)
            self._update_remove_selected_button()
            self.set_status_message("Current file selected")

    def _remove_selected(self):
        """Move all checked files to source_dir/removed/."""
        source_dir = self._settings_manager.source_dir
        removed_subdir = self._settings_manager.removed_subdir or "removed"
        if not source_dir:
            self.set_status_message("No source directory set", error=True)
            return
        checked_paths = [
            item.data(0, PATH_ROLE)
            for item in self._tree_file_items
            if item.checkState(0) == Qt.CheckState.Checked
        ]
        if not checked_paths:
            return
        path_list = _list_audio_files_recursive(source_dir, removed_subdir)
        abs_to_rel = {a: r for r, a in path_list}
        removed_base = Path(source_dir) / removed_subdir
        moved = 0
        for path in checked_paths:
            rel_path = abs_to_rel.get(path) or Path(path).name
            src = Path(path)
            if not src.is_file():
                continue
            dest = removed_base / rel_path
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest.unlink()
                src.rename(dest)
                self._undo_stack.append((str(dest), path))
                moved += 1
            except Exception as e:
                Log.warning(f"DatasetViewerPanel: Failed to move {path}: {e}")
        self._update_undo_button()
        self._refresh_sample_list()
        self.set_status_message(f"Moved {moved} file(s) to {removed_subdir}/")

    def _remove_current(self):
        """Move current sample to source_dir/removed/ preserving relative path."""
        source_dir = self._settings_manager.source_dir
        removed_subdir = self._settings_manager.removed_subdir or "removed"
        if not source_dir:
            self.set_status_message("No source directory set", error=True)
            return
        idx = self._current_file_index()
        if idx < 0 or idx >= len(self._file_paths):
            return
        path = self._file_paths[idx]
        path_list = _list_audio_files_recursive(source_dir, removed_subdir)
        rel_path = None
        for r, a in path_list:
            if a == path:
                rel_path = r
                break
        if not rel_path:
            rel_path = Path(path).name
        src = Path(path)
        if not src.is_file():
            self.set_status_message(f"File not found: {path}", error=True)
            self._refresh_sample_list()
            return
        removed_base = Path(source_dir) / removed_subdir
        dest = removed_base / rel_path
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            src.rename(dest)
            self._undo_stack.append((str(dest), path))
            self._update_undo_button()
            self.set_status_message(f"Moved to {removed_subdir}/")
        except Exception as e:
            Log.warning(f"DatasetViewerPanel: Failed to move file: {e}")
            self.set_status_message(str(e), error=True)
            return
        n = len(self._file_paths)
        next_row = max(0, min(idx, n - 2)) if n > 1 else 0
        self._refresh_sample_list(preferred_row=next_row)

    def _undo_remove(self):
        """Move the last removed file back from removed/ to its original path."""
        if not self._undo_stack:
            return
        dest_path, source_path = self._undo_stack.pop()
        self._update_undo_button()
        dest = Path(dest_path)
        src = Path(source_path)
        if not dest.is_file():
            self.set_status_message("Undo: file no longer in removed/", error=True)
            return
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                src.unlink()
            dest.rename(src)
            self.set_status_message("Undo: restored to original location")
        except Exception as e:
            Log.warning(f"DatasetViewerPanel: Undo failed: {e}")
            self._undo_stack.append((dest_path, source_path))
            self._update_undo_button()
            self.set_status_message(str(e), error=True)
            return
        self._refresh_sample_list()
        try:
            idx = self._file_paths.index(source_path)
        except ValueError:
            idx = 0
        if 0 <= idx < len(self._tree_file_items):
            self.sample_tree.setCurrentItem(self._tree_file_items[idx])
            self._on_file_selected(source_path)

    def refresh(self):
        """Sync UI with block settings and refresh sample list."""
        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return
        if not self.block or not self._settings_manager.is_loaded():
            return
        try:
            source_dir = self._settings_manager.source_dir
            removed_subdir = self._settings_manager.removed_subdir
        except Exception as e:
            Log.error(f"DatasetViewerPanel: Failed to load settings: {e}")
            return
        if source_dir:
            path = Path(source_dir)
            if path.exists():
                self.dir_path_label.setText(str(path))
                self.dir_path_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {Colors.BG_LIGHT.name()};
                        border: 1px solid {Colors.ACCENT_GREEN.name()};
                        border-radius: {border_radius(4)};
                        padding: {Spacing.SM}px;
                        color: {Colors.TEXT_PRIMARY.name()};
                        font-family: monospace;
                        font-size: 11px;
                    }}
                """)
            else:
                self.dir_path_label.setText(f"{source_dir}\n(not found)")
                self.dir_path_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {Colors.BG_LIGHT.name()};
                        border: 1px solid {Colors.ACCENT_RED.name()};
                        border-radius: {border_radius(4)};
                        padding: {Spacing.SM}px;
                        color: {Colors.ACCENT_RED.name()};
                        font-family: monospace;
                        font-size: 11px;
                    }}
                """)
        else:
            self.dir_path_label.setText("No directory selected")
            self.dir_path_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {Colors.BG_LIGHT.name()};
                    border: 1px solid {Colors.BORDER.name()};
                    border-radius: {border_radius(4)};
                    padding: {Spacing.SM}px;
                    color: {Colors.TEXT_SECONDARY.name()};
                    font-family: monospace;
                    font-size: 11px;
                }}
            """)
        self._refresh_sample_list()

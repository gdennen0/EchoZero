"""Project song browser panel for the canonical Qt shell.
Exists to keep song selection, version switching, and batch setlist actions explicit.
Connects presentation-owned song state to the Stage Zero shell's left-side browser.
"""

from __future__ import annotations

from typing import cast

from PyQt6.QtCore import QEvent, QObject, QPoint, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.models import (
    SongOptionPresentation,
    SongVersionOptionPresentation,
    TimelinePresentation,
)
from echozero.ui.qt.song_browser_drop import (
    SongBrowserAudioDrop,
    SongBrowserTree,
    dropped_audio_paths,
    has_droppable_audio,
)

_SONG_KIND_ROLE = Qt.ItemDataRole.UserRole
_SONG_ID_ROLE = Qt.ItemDataRole.UserRole + 1
_SONG_LABEL_ROLE = Qt.ItemDataRole.UserRole + 2
_VERSION_ID_ROLE = Qt.ItemDataRole.UserRole + 3
_VERSION_SONG_ID_ROLE = Qt.ItemDataRole.UserRole + 4
_ROW_NUMBER_COLUMN = 0
_SONG_TITLE_COLUMN = 1
_ROW_NUMBER_COLUMN_WIDTH = 38
_PANEL_COLLAPSED_WIDTH = 56
_PANEL_DEFAULT_EXPANDED_WIDTH = 300
_PANEL_MIN_EXPANDED_WIDTH = 240
_PANEL_MAX_EXPANDED_WIDTH = 460


class SongBrowserPanel(QWidget):
    """Collapsible left-side song browser with version switching and batch actions."""

    collapsed_changed = pyqtSignal(bool)
    song_selected = pyqtSignal(str)
    song_version_selected = pyqtSignal(str)
    add_song_requested = pyqtSignal()
    add_song_version_requested = pyqtSignal(str)
    move_song_up_requested = pyqtSignal(str)
    move_song_down_requested = pyqtSignal(str)
    delete_song_requested = pyqtSignal(str)
    delete_song_version_requested = pyqtSignal(str)
    batch_move_songs_to_top_requested = pyqtSignal(object)
    batch_move_songs_to_bottom_requested = pyqtSignal(object)
    batch_delete_songs_requested = pyqtSignal(object)
    songs_reordered_requested = pyqtSignal(object)
    audio_paths_dropped = pyqtSignal(object)

    def __init__(
        self,
        presentation: TimelinePresentation,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._presentation = presentation
        self._collapsed = False
        self._expanded_width = _PANEL_DEFAULT_EXPANDED_WIDTH
        self._selected_song_ids: set[str] = set()
        self._pending_reorder_song_ids: tuple[str, ...] | None = None
        self._reorder_emit_timer = QTimer(self)
        self._reorder_emit_timer.setSingleShot(True)
        self._reorder_emit_timer.timeout.connect(self._flush_pending_song_reorder)
        self._is_populating_song_list = False
        self.setObjectName("songBrowserPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("collapsed", False)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        header = QWidget(self)
        header.setObjectName("songBrowserHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        self._title = QLabel("Setlist", header)
        self._title.setObjectName("songBrowserTitle")
        header_layout.addWidget(self._title, 1)

        self._add_button = QPushButton("+", header)
        self._add_button.setObjectName("songBrowserQuickAddButton")
        self._add_button.setProperty("appearance", "subtle")
        self._add_button.setToolTip("Add Song")
        self._add_button.clicked.connect(self.add_song_requested.emit)
        header_layout.addWidget(self._add_button)

        self._collapse_button = QToolButton(header)
        self._collapse_button.setObjectName("songBrowserCollapseButton")
        self._collapse_button.setProperty("appearance", "subtle")
        self._collapse_button.setAutoRaise(True)
        self._collapse_button.setText("<")
        self._collapse_button.setToolTip("Collapse Song Browser")
        self._collapse_button.clicked.connect(self.toggle_collapsed)
        header_layout.addWidget(self._collapse_button)
        root_layout.addWidget(header)

        self._content_stack = QStackedWidget(self)
        self._content_stack.setObjectName("songBrowserContent")
        root_layout.addWidget(self._content_stack, 1)

        self._browser_page = QWidget(self)
        self._browser_page.setObjectName("songBrowserPage")
        browser_layout = QVBoxLayout(self._browser_page)
        browser_layout.setContentsMargins(0, 0, 0, 0)
        browser_layout.setSpacing(8)

        self._active_card = QWidget(self._browser_page)
        self._active_card.setObjectName("songBrowserActiveCard")
        active_layout = QVBoxLayout(self._active_card)
        active_layout.setContentsMargins(10, 10, 10, 10)
        active_layout.setSpacing(4)
        self._active_caption = QLabel("Selected Song", self._active_card)
        self._active_caption.setObjectName("songBrowserActiveCaption")
        self._active_song_title = QLabel("No song selected", self._active_card)
        self._active_song_title.setObjectName("songBrowserActiveSongTitle")
        self._active_song_title.setWordWrap(True)
        self._active_song_version = QLabel("Version: -", self._active_card)
        self._active_song_version.setObjectName("songBrowserActiveSongVersion")
        self._active_song_version.setWordWrap(True)
        active_layout.addWidget(self._active_caption)
        active_layout.addWidget(self._active_song_title)
        active_layout.addWidget(self._active_song_version)
        browser_layout.addWidget(self._active_card)

        songs_header = QWidget(self._browser_page)
        songs_header.setObjectName("songBrowserSongsHeader")
        songs_header_layout = QHBoxLayout(songs_header)
        songs_header_layout.setContentsMargins(0, 0, 0, 0)
        songs_header_layout.setSpacing(6)
        self._songs_title = QLabel("Songs", songs_header)
        self._songs_title.setObjectName("songBrowserSectionTitle")
        songs_header_layout.addWidget(self._songs_title)
        self._songs_meta = QLabel("0 songs", songs_header)
        self._songs_meta.setObjectName("songBrowserSongsMeta")
        self._songs_meta.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        songs_header_layout.addWidget(self._songs_meta, 1)
        browser_layout.addWidget(songs_header)

        self._songs_tree = SongBrowserTree(self._resolve_tree_drop_target, self._browser_page)
        self._songs_tree.setObjectName("songBrowserSongList")
        self._songs_tree.setColumnCount(2)
        self._songs_tree.setHeaderHidden(True)
        self._songs_tree.setRootIsDecorated(False)
        self._songs_tree.setItemsExpandable(False)
        self._songs_tree.setIndentation(0)
        self._songs_tree.header().setMinimumSectionSize(0)
        self._songs_tree.header().setStretchLastSection(True)
        self._songs_tree.setColumnWidth(_ROW_NUMBER_COLUMN, _ROW_NUMBER_COLUMN_WIDTH)
        self._songs_tree.setUniformRowHeights(True)
        self._songs_tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._songs_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._songs_tree.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._songs_tree.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._songs_tree.setDragEnabled(True)
        self._songs_tree.setDropIndicatorShown(True)
        self._songs_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._songs_tree.itemClicked.connect(self._handle_song_item_clicked)
        self._songs_tree.itemSelectionChanged.connect(self._handle_song_selection_changed)
        self._songs_tree.customContextMenuRequested.connect(self._open_song_context_menu)
        self._songs_tree.audio_drop_requested.connect(self.audio_paths_dropped.emit)
        self._songs_tree.model().rowsMoved.connect(self._handle_song_rows_moved)
        browser_layout.addWidget(self._songs_tree, 2)

        # Backward-compatible alias used by existing tests/support helpers.
        self._tree = self._songs_tree

        version_header = QWidget(self._browser_page)
        version_header.setObjectName("songBrowserVersionsHeader")
        version_header_layout = QHBoxLayout(version_header)
        version_header_layout.setContentsMargins(0, 0, 0, 0)
        version_header_layout.setSpacing(6)
        self._versions_title = QLabel("Versions", version_header)
        self._versions_title.setObjectName("songBrowserSectionTitle")
        version_header_layout.addWidget(self._versions_title)
        self._add_version_button = QPushButton("+ Version", version_header)
        self._add_version_button.setObjectName("songBrowserAddVersionButton")
        self._add_version_button.setProperty("appearance", "subtle")
        self._add_version_button.setToolTip("Add version for selected song")
        self._add_version_button.clicked.connect(self._emit_add_version_for_active_song)
        version_header_layout.addWidget(self._add_version_button, 0)
        browser_layout.addWidget(version_header)

        self._version_list = QListWidget(self._browser_page)
        self._version_list.setObjectName("songBrowserVersionList")
        self._version_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._version_list.itemClicked.connect(self._handle_version_clicked)
        self._version_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._version_list.customContextMenuRequested.connect(self._open_version_context_menu)
        browser_layout.addWidget(self._version_list, 1)

        self._batch_bar = QWidget(self._browser_page)
        self._batch_bar.setObjectName("songBrowserBatchBar")
        batch_layout = QHBoxLayout(self._batch_bar)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(6)

        self._batch_meta = QLabel("No songs selected", self._batch_bar)
        self._batch_meta.setObjectName("songBrowserBatchMeta")
        batch_layout.addWidget(self._batch_meta, 1)

        self._select_all_button = QPushButton("[*]", self._batch_bar)
        self._select_all_button.setObjectName("songBrowserBatchSelectAll")
        self._select_all_button.setProperty("appearance", "subtle")
        self._select_all_button.setToolTip("Select all songs")
        self._select_all_button.clicked.connect(self._select_all_songs)
        batch_layout.addWidget(self._select_all_button)

        self._clear_selection_button = QPushButton("[ ]", self._batch_bar)
        self._clear_selection_button.setObjectName("songBrowserBatchClear")
        self._clear_selection_button.setProperty("appearance", "subtle")
        self._clear_selection_button.setToolTip("Clear song selection")
        self._clear_selection_button.clicked.connect(self._clear_song_selection)
        batch_layout.addWidget(self._clear_selection_button)

        self._batch_top_button = QPushButton("^^", self._batch_bar)
        self._batch_top_button.setObjectName("songBrowserBatchMoveTop")
        self._batch_top_button.setProperty("appearance", "subtle")
        self._batch_top_button.setToolTip("Move selected songs to top")
        self._batch_top_button.clicked.connect(self._emit_batch_move_songs_to_top)
        batch_layout.addWidget(self._batch_top_button)

        self._batch_bottom_button = QPushButton("vv", self._batch_bar)
        self._batch_bottom_button.setObjectName("songBrowserBatchMoveBottom")
        self._batch_bottom_button.setProperty("appearance", "subtle")
        self._batch_bottom_button.setToolTip("Move selected songs to bottom")
        self._batch_bottom_button.clicked.connect(self._emit_batch_move_songs_to_bottom)
        batch_layout.addWidget(self._batch_bottom_button)

        self._batch_delete_button = QPushButton("X", self._batch_bar)
        self._batch_delete_button.setObjectName("songBrowserBatchDelete")
        self._batch_delete_button.setProperty("appearance", "subtle")
        self._batch_delete_button.setToolTip("Delete selected songs")
        self._batch_delete_button.clicked.connect(self._emit_batch_delete_songs)
        batch_layout.addWidget(self._batch_delete_button)

        browser_layout.addWidget(self._batch_bar)
        self._content_stack.addWidget(self._browser_page)

        self._empty_page = QWidget(self)
        self._empty_page.setObjectName("songBrowserEmptyPage")
        empty_layout = QVBoxLayout(self._empty_page)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.setSpacing(0)

        self._empty_state = QLabel(
            "No songs in this project yet.\nUse + above or drop audio here.",
            self._empty_page,
        )
        self._empty_state.setObjectName("songBrowserEmptyState")
        self._empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_state.setWordWrap(True)
        empty_layout.addWidget(self._empty_state, 1)
        self._content_stack.addWidget(self._empty_page)

        for drop_target in (self._empty_page, self._empty_state):
            drop_target.setAcceptDrops(True)
            drop_target.installEventFilter(self)

        self.set_presentation(presentation)
        self._apply_collapsed_state()

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    @property
    def expanded_width(self) -> int:
        return self._expanded_width

    def target_width(self) -> int:
        return _PANEL_COLLAPSED_WIDTH if self._collapsed else self._expanded_width

    def toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        self._apply_collapsed_state()
        self.collapsed_changed.emit(self._collapsed)

    def remember_expanded_width(self, width: int) -> None:
        clamped_width = max(
            _PANEL_MIN_EXPANDED_WIDTH,
            min(_PANEL_MAX_EXPANDED_WIDTH, int(width)),
        )
        self._expanded_width = clamped_width
        if not self._collapsed:
            self.resize(self._expanded_width, self.height())
            self.updateGeometry()

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:
        if watched in (self._empty_page, self._empty_state) and event is not None:
            if event.type() == QEvent.Type.DragEnter:
                return self._accept_audio_drop(cast(QDragEnterEvent, event))
            if event.type() == QEvent.Type.DragMove:
                return self._accept_audio_drop(cast(QDragMoveEvent, event))
            if event.type() == QEvent.Type.Drop:
                return self._handle_empty_state_drop(cast(QDropEvent, event))
        return super().eventFilter(watched, event)

    def set_presentation(self, presentation: TimelinePresentation) -> None:
        if self._reorder_emit_timer.isActive():
            self._reorder_emit_timer.stop()
        self._pending_reorder_song_ids = None
        self._presentation = presentation
        song_count = len(presentation.available_songs)
        self._title.setText("Setlist")
        self._title.setToolTip(
            f"{song_count} song" if song_count == 1 else f"{song_count} songs"
        )

        self._sync_selected_song_ids(presentation)
        self._populate_song_list(presentation)
        self._populate_version_list(presentation)
        self._update_active_summary(presentation)
        self._update_batch_ui()

        has_songs = bool(presentation.available_songs)
        self._content_stack.setCurrentIndex(0 if has_songs else 1)
        self._apply_collapsed_state()

    def _populate_song_list(self, presentation: TimelinePresentation) -> None:
        self._is_populating_song_list = True
        try:
            self._songs_tree.blockSignals(True)
            self._songs_tree.clear()
            current_item: QTreeWidgetItem | None = None
            for index, song in enumerate(presentation.available_songs, start=1):
                song_item = QTreeWidgetItem([str(index), song.title])
                song_item.setTextAlignment(
                    _ROW_NUMBER_COLUMN,
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                )
                song_item.setData(_SONG_TITLE_COLUMN, _SONG_KIND_ROLE, "song")
                song_item.setData(_SONG_TITLE_COLUMN, _SONG_ID_ROLE, song.song_id)
                song_item.setData(_SONG_TITLE_COLUMN, _SONG_LABEL_ROLE, song.title)
                tooltip = (
                    f"Active version: {song.active_version_label}\nVersions: {song.version_count}"
                    if song.active_version_label
                    else f"Versions: {song.version_count}"
                )
                song_item.setToolTip(_ROW_NUMBER_COLUMN, tooltip)
                song_item.setToolTip(_SONG_TITLE_COLUMN, tooltip)
                if song.is_active:
                    song_font = QFont(song_item.font(_SONG_TITLE_COLUMN))
                    song_font.setBold(True)
                    song_item.setFont(_SONG_TITLE_COLUMN, song_font)
                    current_item = song_item
                self._songs_tree.addTopLevelItem(song_item)
                if song.song_id in self._selected_song_ids:
                    song_item.setSelected(True)

            if current_item is not None:
                self._songs_tree.setCurrentItem(current_item, _SONG_TITLE_COLUMN)
            self._songs_tree.blockSignals(False)
        finally:
            self._is_populating_song_list = False
        self._sync_selection_from_tree()

    def _populate_version_list(self, presentation: TimelinePresentation) -> None:
        self._version_list.blockSignals(True)
        self._version_list.clear()

        active_song = self._resolve_active_song(presentation)
        if active_song is None:
            self._add_version_button.setEnabled(False)
            self._version_list.blockSignals(False)
            return

        self._add_version_button.setEnabled(True)
        active_version_id = presentation.active_song_version_id.strip()
        active_item: QListWidgetItem | None = None
        for index, version in enumerate(active_song.versions, start=1):
            item = QListWidgetItem(self._song_version_label(version, index=index))
            item.setData(_VERSION_ID_ROLE, version.song_version_id)
            item.setData(_VERSION_SONG_ID_ROLE, active_song.song_id)
            item.setToolTip(
                "Active song version" if version.is_active else "Song version"
            )
            if version.song_version_id == active_version_id or version.is_active:
                version_font = QFont(item.font())
                version_font.setBold(True)
                item.setFont(version_font)
                active_item = item
            self._version_list.addItem(item)

        if active_item is not None:
            self._version_list.setCurrentItem(active_item)
        self._version_list.blockSignals(False)

    def _resolve_active_song(
        self,
        presentation: TimelinePresentation,
    ) -> SongOptionPresentation | None:
        active_song_id = presentation.active_song_id.strip()
        if active_song_id:
            for song in presentation.available_songs:
                if song.song_id == active_song_id:
                    return song
        for song in presentation.available_songs:
            if song.is_active:
                return song
        return presentation.available_songs[0] if presentation.available_songs else None

    def _update_active_summary(self, presentation: TimelinePresentation) -> None:
        song_title = presentation.active_song_title.strip()
        version_label = presentation.active_song_version_label.strip()
        if not song_title:
            active_song = self._resolve_active_song(presentation)
            if active_song is not None:
                song_title = active_song.title
                version_label = active_song.active_version_label

        if not song_title:
            self._active_song_title.setText("No song selected")
            self._active_song_version.setText("Version: -")
            return

        self._active_song_title.setText(song_title)
        if not version_label:
            version_label = "No version selected"

        pool_no = presentation.active_song_version_ma3_timecode_pool_no
        if pool_no is None:
            self._active_song_version.setText(f"Version: {version_label}")
            return
        self._active_song_version.setText(f"Version: {version_label} (TC{pool_no})")

    def _apply_collapsed_state(self) -> None:
        self._set_collapsed_style_state(self._collapsed)
        if self._collapsed:
            self.setMinimumWidth(_PANEL_COLLAPSED_WIDTH)
            self.setMaximumWidth(_PANEL_COLLAPSED_WIDTH)
        else:
            self.setMinimumWidth(_PANEL_MIN_EXPANDED_WIDTH)
            self.setMaximumWidth(_PANEL_MAX_EXPANDED_WIDTH)
            self.resize(self._expanded_width, self.height())
        self.updateGeometry()
        self._title.setVisible(not self._collapsed)
        self._add_button.setVisible(not self._collapsed)
        self._content_stack.setVisible(not self._collapsed)
        self._collapse_button.setText(">" if self._collapsed else "<")
        self._collapse_button.setToolTip(
            "Expand Song Browser" if self._collapsed else "Collapse Song Browser"
        )

    def _set_collapsed_style_state(self, collapsed: bool) -> None:
        current = bool(self.property("collapsed"))
        if current == collapsed:
            return
        self.setProperty("collapsed", collapsed)
        for widget in (self, self._collapse_button):
            style = widget.style()
            if style is None:
                continue
            style.unpolish(widget)
            style.polish(widget)
            widget.update()

    def _handle_song_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        song_id = item.data(_SONG_TITLE_COLUMN, _SONG_ID_ROLE)
        if not isinstance(song_id, str) or not song_id:
            return

        modifiers = QApplication.keyboardModifiers()
        multi_select_modifier = bool(
            modifiers
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.MetaModifier
                | Qt.KeyboardModifier.ShiftModifier
            )
        )
        if not multi_select_modifier:
            self.song_selected.emit(song_id)

    def _handle_song_selection_changed(self) -> None:
        self._sync_selection_from_tree()
        self._update_batch_ui()

    def _handle_version_clicked(self, item: QListWidgetItem) -> None:
        version_id = item.data(_VERSION_ID_ROLE)
        if isinstance(version_id, str) and version_id:
            self.song_version_selected.emit(version_id)

    def _emit_add_version_for_active_song(self) -> None:
        song_id = self._presentation.active_song_id.strip()
        if not song_id:
            selected = self._selected_song_ids_in_display_order()
            if len(selected) == 1:
                song_id = selected[0]
        if song_id:
            self.add_song_version_requested.emit(song_id)

    def _sync_selected_song_ids(self, presentation: TimelinePresentation) -> None:
        available_song_ids = {song.song_id for song in presentation.available_songs}
        self._selected_song_ids = {
            song_id for song_id in self._selected_song_ids if song_id in available_song_ids
        }
        if self._selected_song_ids:
            return

        active_song_id = presentation.active_song_id.strip()
        if active_song_id and active_song_id in available_song_ids:
            self._selected_song_ids = {active_song_id}
            return

        active_option = self._resolve_active_song(presentation)
        if active_option is not None:
            self._selected_song_ids = {active_option.song_id}

    def _sync_selection_from_tree(self) -> None:
        selected_song_ids: set[str] = set()
        for item in self._songs_tree.selectedItems():
            song_id = item.data(_SONG_TITLE_COLUMN, _SONG_ID_ROLE)
            if isinstance(song_id, str) and song_id:
                selected_song_ids.add(song_id)
        self._selected_song_ids = selected_song_ids

    def _selected_song_ids_in_display_order(self) -> tuple[str, ...]:
        ordered_song_ids: list[str] = []
        for row in range(self._songs_tree.topLevelItemCount()):
            item = self._songs_tree.topLevelItem(row)
            if item is None or not item.isSelected():
                continue
            song_id = item.data(_SONG_TITLE_COLUMN, _SONG_ID_ROLE)
            if isinstance(song_id, str) and song_id:
                ordered_song_ids.append(song_id)
        return tuple(ordered_song_ids)

    def _song_ids_in_display_order(self) -> tuple[str, ...]:
        ordered_song_ids: list[str] = []
        for row in range(self._songs_tree.topLevelItemCount()):
            item = self._songs_tree.topLevelItem(row)
            if item is None:
                continue
            song_id = item.data(_SONG_TITLE_COLUMN, _SONG_ID_ROLE)
            if isinstance(song_id, str) and song_id:
                ordered_song_ids.append(song_id)
        return tuple(ordered_song_ids)

    def _refresh_song_row_numbers(self) -> None:
        for index in range(1, self._songs_tree.topLevelItemCount() + 1):
            item = self._songs_tree.topLevelItem(index - 1)
            if item is None:
                continue
            item.setText(_ROW_NUMBER_COLUMN, str(index))
            item.setTextAlignment(
                _ROW_NUMBER_COLUMN,
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            )

    def _handle_song_rows_moved(self, *_args: object) -> None:
        self._refresh_song_row_numbers()
        if self._is_populating_song_list:
            return
        ordered_song_ids = self._song_ids_in_display_order()
        if not ordered_song_ids:
            return
        current_song_ids = tuple(song.song_id for song in self._presentation.available_songs)
        if ordered_song_ids == current_song_ids:
            return
        self._pending_reorder_song_ids = ordered_song_ids
        self._reorder_emit_timer.start(0)

    def _flush_pending_song_reorder(self) -> None:
        if self._is_populating_song_list:
            return
        ordered_song_ids = self._pending_reorder_song_ids
        self._pending_reorder_song_ids = None
        if not ordered_song_ids:
            return
        current_song_ids = tuple(song.song_id for song in self._presentation.available_songs)
        if ordered_song_ids == current_song_ids:
            return
        self.songs_reordered_requested.emit(ordered_song_ids)

    def _select_all_songs(self) -> None:
        if self._songs_tree.topLevelItemCount() <= 0:
            return
        self._songs_tree.blockSignals(True)
        for row in range(self._songs_tree.topLevelItemCount()):
            item = self._songs_tree.topLevelItem(row)
            if item is not None:
                item.setSelected(True)
        self._songs_tree.blockSignals(False)
        self._sync_selection_from_tree()
        self._update_batch_ui()

    def _clear_song_selection(self) -> None:
        self._songs_tree.blockSignals(True)
        self._songs_tree.clearSelection()
        self._songs_tree.blockSignals(False)
        self._sync_selection_from_tree()
        self._update_batch_ui()

    def _emit_batch_move_songs_to_top(self) -> None:
        selected_song_ids = self._selected_song_ids_in_display_order()
        if len(selected_song_ids) < 2:
            return
        self.batch_move_songs_to_top_requested.emit(selected_song_ids)

    def _emit_batch_move_songs_to_bottom(self) -> None:
        selected_song_ids = self._selected_song_ids_in_display_order()
        if len(selected_song_ids) < 2:
            return
        self.batch_move_songs_to_bottom_requested.emit(selected_song_ids)

    def _emit_batch_delete_songs(self) -> None:
        selected_song_ids = self._selected_song_ids_in_display_order()
        if not selected_song_ids:
            return
        self.batch_delete_songs_requested.emit(selected_song_ids)

    def _update_batch_ui(self) -> None:
        selected_count = len(self._selected_song_ids)
        total_count = self._songs_tree.topLevelItemCount()

        if selected_count:
            noun = "song" if selected_count == 1 else "songs"
            self._batch_meta.setText(f"{selected_count} {noun} selected")
            self._songs_meta.setText(f"{selected_count} selected of {total_count}")
        else:
            noun = "song" if total_count == 1 else "songs"
            self._batch_meta.setText("No songs selected")
            self._songs_meta.setText(f"{total_count} {noun}")

        self._clear_selection_button.setEnabled(selected_count > 0)
        self._batch_delete_button.setEnabled(selected_count > 0)
        self._batch_top_button.setEnabled(selected_count > 1 and total_count > 1)
        self._batch_bottom_button.setEnabled(selected_count > 1 and total_count > 1)
        self._select_all_button.setEnabled(total_count > 0 and selected_count < total_count)

    def _resolve_tree_drop_target(
        self,
        item: QTreeWidgetItem | None,
    ) -> tuple[str | None, str | None]:
        if item is None:
            return (None, None)
        song_id = item.data(_SONG_TITLE_COLUMN, _SONG_ID_ROLE)
        song_title = item.data(_SONG_TITLE_COLUMN, _SONG_LABEL_ROLE)
        return (
            song_id if isinstance(song_id, str) else None,
            song_title if isinstance(song_title, str) else None,
        )

    def _accept_audio_drop(
        self,
        event: QDragEnterEvent | QDragMoveEvent,
    ) -> bool:
        if has_droppable_audio(event):
            event.acceptProposedAction()
            return True
        event.ignore()
        return False

    def _handle_empty_state_drop(self, event: QDropEvent) -> bool:
        paths = dropped_audio_paths(event, include_directory_audio=True)
        if not paths:
            event.ignore()
            return False
        self._emit_new_song_drop(paths)
        event.acceptProposedAction()
        return True

    def _emit_new_song_drop(self, audio_paths: object) -> None:
        resolved_paths = tuple(audio_paths) if isinstance(audio_paths, (list, tuple)) else ()
        if not resolved_paths:
            return
        self.audio_paths_dropped.emit(
            SongBrowserAudioDrop(audio_paths=tuple(str(path) for path in resolved_paths))
        )

    def _open_song_context_menu(self, point: QPoint) -> None:
        item = self._songs_tree.itemAt(point)
        selected_song_ids = self._selected_song_ids_in_display_order()
        menu = QMenu(self)
        viewport = self._songs_tree.viewport()
        if viewport is None:
            return

        if item is None:
            menu.addAction("Add Song", lambda: self.add_song_requested.emit())
            if selected_song_ids:
                menu.addSeparator()
                menu.addAction("Delete Selected Songs", self._emit_batch_delete_songs)
            menu.exec(viewport.mapToGlobal(point))
            return

        song_id = item.data(_SONG_TITLE_COLUMN, _SONG_ID_ROLE)
        if not isinstance(song_id, str) or not song_id:
            return

        if len(selected_song_ids) > 1 and song_id in self._selected_song_ids:
            menu.addAction("Move Selected Songs to Top", self._emit_batch_move_songs_to_top)
            menu.addAction("Move Selected Songs to Bottom", self._emit_batch_move_songs_to_bottom)
            menu.addSeparator()
            menu.addAction("Delete Selected Songs", self._emit_batch_delete_songs)
            menu.exec(viewport.mapToGlobal(point))
            return

        menu.addAction("Select Song", lambda: self.song_selected.emit(song_id))
        menu.addAction("Add Version...", lambda: self.add_song_version_requested.emit(song_id))
        menu.addSeparator()
        menu.addAction("Move Up", lambda: self.move_song_up_requested.emit(song_id))
        menu.addAction("Move Down", lambda: self.move_song_down_requested.emit(song_id))
        menu.addSeparator()
        menu.addAction("Delete Song", lambda: self.delete_song_requested.emit(song_id))
        menu.exec(viewport.mapToGlobal(point))

    def _open_version_context_menu(self, point: QPoint) -> None:
        item = self._version_list.itemAt(point)
        menu = QMenu(self)
        viewport = self._version_list.viewport()
        if viewport is None:
            return

        if item is None:
            if self._presentation.active_song_id.strip():
                menu.addAction("Add Version...", self._emit_add_version_for_active_song)
            menu.exec(viewport.mapToGlobal(point))
            return

        version_id = item.data(_VERSION_ID_ROLE)
        if not isinstance(version_id, str) or not version_id:
            return

        menu.addAction(
            "Switch to Version",
            lambda: self.song_version_selected.emit(version_id),
        )
        menu.addSeparator()
        menu.addAction(
            "Delete Version",
            lambda: self.delete_song_version_requested.emit(version_id),
        )
        menu.exec(viewport.mapToGlobal(point))

    @staticmethod
    def _song_version_label(
        version: SongVersionOptionPresentation,
        *,
        index: int,
    ) -> str:
        label = f"V{index}: {version.label}"
        if version.ma3_timecode_pool_no is not None:
            label = f"{label} (TC{version.ma3_timecode_pool_no})"
        if version.is_active:
            label = f"{label} [Active]"
        return label


__all__ = [
    "SongBrowserAudioDrop",
    "SongBrowserPanel",
    "SongBrowserTree",
    "dropped_audio_paths",
]

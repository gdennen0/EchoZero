"""Project song browser panel for the canonical Qt shell.
Exists to surface setlist song/version selection without bypassing the runtime shell.
Connects presentation-owned song state to the Stage Zero shell's left-side browser.
"""

from __future__ import annotations

from typing import cast

from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
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
)

_ITEM_KIND_ROLE = Qt.ItemDataRole.UserRole
_ITEM_ID_ROLE = Qt.ItemDataRole.UserRole + 1
_ITEM_SONG_ID_ROLE = Qt.ItemDataRole.UserRole + 2
_ITEM_LABEL_ROLE = Qt.ItemDataRole.UserRole + 3
_PANEL_COLLAPSED_WIDTH = 64
_PANEL_DEFAULT_EXPANDED_WIDTH = 280
_PANEL_MIN_EXPANDED_WIDTH = 220
_PANEL_MAX_EXPANDED_WIDTH = 420


class SongBrowserPanel(QWidget):
    """Collapsible left-side song and version browser for one project timeline."""

    collapsed_changed = pyqtSignal(bool)
    song_selected = pyqtSignal(str)
    song_version_selected = pyqtSignal(str)
    add_song_requested = pyqtSignal()
    add_song_version_requested = pyqtSignal(str)
    delete_song_requested = pyqtSignal(str)
    delete_song_version_requested = pyqtSignal(str)
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
        self._expanded_song_ids: set[str] = set()
        self.setObjectName("songBrowserPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("collapsed", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QWidget(self)
        header.setObjectName("songBrowserHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        self._title = QLabel("Songs", header)
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
        self._collapse_button.setText("◂")
        self._collapse_button.setToolTip("Collapse Song Browser")
        self._collapse_button.clicked.connect(self.toggle_collapsed)
        header_layout.addWidget(self._collapse_button)
        layout.addWidget(header)

        self._tree = SongBrowserTree(self._resolve_tree_drop_target, self)
        self._tree.setObjectName("songBrowserTree")
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(False)
        self._tree.setIndentation(16)
        self._tree.setItemsExpandable(True)
        self._tree.setAnimated(True)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.itemClicked.connect(self._handle_item_clicked)
        self._tree.itemExpanded.connect(self._track_item_expanded)
        self._tree.itemCollapsed.connect(self._track_item_collapsed)
        self._tree.customContextMenuRequested.connect(self._open_context_menu)
        self._tree.audio_drop_requested.connect(self.audio_paths_dropped.emit)
        self._content_stack = QStackedWidget(self)
        self._content_stack.setObjectName("songBrowserContent")
        self._content_stack.addWidget(self._tree)

        self._empty_page = QWidget(self)
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
        layout.addWidget(self._content_stack, 1)

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
        self._presentation = presentation
        song_count = len(presentation.available_songs)
        self._title.setText("Setlist")
        self._title.setToolTip(
            f"{song_count} song" if song_count == 1 else f"{song_count} songs"
        )

        self._tree.blockSignals(True)
        self._tree.clear()
        current_item: QTreeWidgetItem | None = None
        for song in presentation.available_songs:
            song_item = self._build_song_item(song)
            self._tree.addTopLevelItem(song_item)
            is_expanded = song.song_id in self._expanded_song_ids
            if song.is_active:
                current_item = song_item
            if is_expanded:
                self._tree.expandItem(song_item)
            self._sync_song_item_label(song_item, expanded=is_expanded)
            for index in range(song_item.childCount()):
                version_item = song_item.child(index)
                if version_item is None:
                    continue
                if (
                    version_item.data(0, _ITEM_KIND_ROLE) == "version"
                    and version_item.data(0, _ITEM_ID_ROLE) == presentation.active_song_version_id
                    and is_expanded
                ):
                    current_item = version_item
        self._tree.blockSignals(False)

        has_songs = bool(presentation.available_songs)
        if current_item is not None:
            self._tree.setCurrentItem(current_item)
        self._content_stack.setCurrentIndex(0 if has_songs else 1)
        self._apply_collapsed_state()

    def _build_song_item(self, song: SongOptionPresentation) -> QTreeWidgetItem:
        song_item = QTreeWidgetItem([song.title])
        song_item.setData(0, _ITEM_KIND_ROLE, "song")
        song_item.setData(0, _ITEM_ID_ROLE, song.song_id)
        song_item.setData(0, _ITEM_LABEL_ROLE, song.title)
        song_item.setChildIndicatorPolicy(
            QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
        )
        song_item.setToolTip(
            0,
            (
                f"Active version: {song.active_version_label}\nVersions: {song.version_count}"
                if song.active_version_label
                else f"Versions: {song.version_count}"
            ),
        )
        if song.is_active:
            song_font = QFont(song_item.font(0))
            song_font.setBold(True)
            song_item.setFont(0, song_font)
        for index, version in enumerate(song.versions, start=1):
            version_item = QTreeWidgetItem([self._song_version_label(version, index=index)])
            version_item.setData(0, _ITEM_KIND_ROLE, "version")
            version_item.setData(0, _ITEM_ID_ROLE, version.song_version_id)
            version_item.setData(0, _ITEM_SONG_ID_ROLE, song.song_id)
            version_font = QFont(version_item.font(0))
            version_font.setItalic(True)
            if version.is_active:
                version_font.setBold(True)
            version_item.setFont(0, version_font)
            version_item.setToolTip(
                0,
                "Active song version" if version.is_active else "Song version",
            )
            song_item.addChild(version_item)
        return song_item

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
        self._collapse_button.setText("▸" if self._collapsed else "◂")
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

    def _handle_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        item_kind = item.data(0, _ITEM_KIND_ROLE)
        item_id = item.data(0, _ITEM_ID_ROLE)
        if not isinstance(item_kind, str) or not isinstance(item_id, str):
            return
        if item_kind == "song":
            if item.childCount() > 0:
                if item.isExpanded():
                    self._tree.collapseItem(item)
                else:
                    self._tree.expandItem(item)
            self.song_selected.emit(item_id)
            return
        if item_kind == "version":
            self.song_version_selected.emit(item_id)

    def _track_item_expanded(self, item: QTreeWidgetItem) -> None:
        song_id = item.data(0, _ITEM_ID_ROLE)
        if item.data(0, _ITEM_KIND_ROLE) == "song" and isinstance(song_id, str):
            self._expanded_song_ids.add(song_id)
            self._sync_song_item_label(item, expanded=True)

    def _track_item_collapsed(self, item: QTreeWidgetItem) -> None:
        song_id = item.data(0, _ITEM_ID_ROLE)
        if item.data(0, _ITEM_KIND_ROLE) == "song" and isinstance(song_id, str):
            self._expanded_song_ids.discard(song_id)
            self._sync_song_item_label(item, expanded=False)

    def _resolve_tree_drop_target(
        self,
        item: QTreeWidgetItem | None,
    ) -> tuple[str | None, str | None]:
        if item is None:
            return (None, None)
        item_kind = item.data(0, _ITEM_KIND_ROLE)
        if item_kind == "version":
            item = item.parent()
            item_kind = item.data(0, _ITEM_KIND_ROLE) if item is not None else None
        if item_kind != "song" or item is None:
            return (None, None)
        song_id = item.data(0, _ITEM_ID_ROLE)
        song_title = item.data(0, _ITEM_LABEL_ROLE)
        return (
            song_id if isinstance(song_id, str) else None,
            song_title if isinstance(song_title, str) else None,
        )

    def _accept_audio_drop(
        self,
        event: QDragEnterEvent | QDragMoveEvent,
    ) -> bool:
        if dropped_audio_paths(event):
            event.acceptProposedAction()
            return True
        event.ignore()
        return False

    def _handle_empty_state_drop(self, event: QDropEvent) -> bool:
        paths = dropped_audio_paths(event)
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

    def _open_context_menu(self, point: QPoint) -> None:
        item = self._tree.itemAt(point)
        menu = QMenu(self)
        viewport = self._tree.viewport()
        if viewport is None:
            return
        if item is None:
            menu.addAction("Add Song", lambda: self.add_song_requested.emit())
            menu.exec(viewport.mapToGlobal(point))
            return

        item_kind = item.data(0, _ITEM_KIND_ROLE)
        item_id = item.data(0, _ITEM_ID_ROLE)
        if not isinstance(item_kind, str) or not isinstance(item_id, str):
            return

        if item_kind == "song":
            menu.addAction("Select Song", lambda: self.song_selected.emit(item_id))
            menu.addAction("Add Version...", lambda: self.add_song_version_requested.emit(item_id))
            menu.addSeparator()
            menu.addAction("Delete Song", lambda: self.delete_song_requested.emit(item_id))
        elif item_kind == "version":
            menu.addAction(
                "Switch to Version",
                lambda: self.song_version_selected.emit(item_id),
            )
            menu.addSeparator()
            menu.addAction(
                "Delete Version",
                lambda: self.delete_song_version_requested.emit(item_id),
            )
        menu.exec(viewport.mapToGlobal(point))

    @staticmethod
    def _song_label(base_label: str, *, expanded: bool) -> str:
        return f"{'▼' if expanded else '▶'} {base_label}"

    @staticmethod
    def _song_version_label(
        version: SongVersionOptionPresentation,
        *,
        index: int,
    ) -> str:
        prefix = f"Version {index} · "
        active_suffix = " · Active" if version.is_active else ""
        if version.ma3_timecode_pool_no is not None:
            return (
                f"{prefix}{version.label}"
                f" · TC{version.ma3_timecode_pool_no}{active_suffix}"
            )
        return f"{prefix}{version.label}{active_suffix}"

    def _sync_song_item_label(self, item: QTreeWidgetItem, *, expanded: bool) -> None:
        base_label = item.data(0, _ITEM_LABEL_ROLE)
        if not isinstance(base_label, str):
            return
        item.setText(0, self._song_label(base_label, expanded=expanded))


__all__ = [
    "SongBrowserAudioDrop",
    "SongBrowserPanel",
    "SongBrowserTree",
    "dropped_audio_paths",
]

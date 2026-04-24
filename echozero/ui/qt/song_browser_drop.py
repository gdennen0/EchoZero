"""Song browser drag-and-drop widgets for the canonical Qt shell.
Exists to keep audio-drop behavior reusable and out of the larger setlist panel widget.
Connects setlist-local drop targets to the timeline shell's import routing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import QPushButton, QTreeWidget, QTreeWidgetItem, QWidget

_DROPPABLE_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg"})


def dropped_audio_paths(
    event: QDragEnterEvent | QDragMoveEvent | QDropEvent,
) -> tuple[str, ...]:
    """Extract one or more local audio file paths from a Qt drop event."""

    mime_data = event.mimeData()
    if mime_data is None or not mime_data.hasUrls():
        return ()

    paths: list[str] = []
    seen: set[str] = set()
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        local_path = Path(url.toLocalFile())
        if not local_path.is_file() or local_path.suffix.lower() not in _DROPPABLE_AUDIO_SUFFIXES:
            continue
        resolved = str(local_path)
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(resolved)
    return tuple(paths)


@dataclass(frozen=True, slots=True)
class SongBrowserAudioDrop:
    """Audio-file drop metadata captured from the setlist browser."""

    audio_paths: tuple[str, ...]
    target_song_id: str | None = None
    target_song_title: str | None = None


class SongBrowserAddButton(QPushButton):
    """Clickable add-song button that also accepts audio-file drops."""

    audio_paths_dropped = pyqtSignal(object)

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event is None:
            return
        if dropped_audio_paths(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent | None) -> None:
        if event is None:
            return
        if dropped_audio_paths(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None:
            return
        paths = dropped_audio_paths(event)
        if not paths:
            event.ignore()
            return
        self.audio_paths_dropped.emit(paths)
        event.acceptProposedAction()


class SongBrowserTree(QTreeWidget):
    """Setlist tree that accepts audio drops and resolves the hovered song target."""

    audio_drop_requested = pyqtSignal(object)

    def __init__(
        self,
        resolve_song_target: Callable[[QTreeWidgetItem | None], tuple[str | None, str | None]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._resolve_song_target = resolve_song_target
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event is None:
            return
        if dropped_audio_paths(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent | None) -> None:
        if event is None:
            return
        if dropped_audio_paths(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None:
            return
        paths = dropped_audio_paths(event)
        if not paths:
            event.ignore()
            return
        target_song_id, target_song_title = self._resolve_song_target(
            self.itemAt(event.position().toPoint())
        )
        self.audio_drop_requested.emit(
            SongBrowserAudioDrop(
                audio_paths=paths,
                target_song_id=target_song_id,
                target_song_title=target_song_title,
            )
        )
        event.acceptProposedAction()


__all__ = [
    "SongBrowserAddButton",
    "SongBrowserAudioDrop",
    "SongBrowserTree",
    "dropped_audio_paths",
]

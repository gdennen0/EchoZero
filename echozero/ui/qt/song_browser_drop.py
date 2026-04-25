"""Song browser drag-and-drop widgets for the canonical Qt shell.
Exists to keep audio-drop behavior reusable and out of the larger setlist panel widget.
Connects setlist-local drop targets to the timeline shell's import routing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import QPushButton, QTreeWidget, QTreeWidgetItem, QWidget

_DROPPABLE_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg"})
_NATURAL_TOKEN_PATTERN = re.compile(r"(\d+)")


def dropped_audio_paths(
    event: QDragEnterEvent | QDragMoveEvent | QDropEvent,
    *,
    include_directory_audio: bool = False,
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
        if local_path.is_file():
            if local_path.suffix.lower() not in _DROPPABLE_AUDIO_SUFFIXES:
                continue
            resolved = str(local_path)
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
            continue
        if include_directory_audio and local_path.is_dir():
            for audio_file in _audio_files_in_directory(local_path):
                resolved = str(audio_file)
                if resolved in seen:
                    continue
                seen.add(resolved)
                paths.append(resolved)
    return tuple(paths)


def has_droppable_audio(
    event: QDragEnterEvent | QDragMoveEvent | QDropEvent,
) -> bool:
    """True when the event has local audio files or folders that can resolve to audio."""

    mime_data = event.mimeData()
    if mime_data is None or not mime_data.hasUrls():
        return False
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        local_path = Path(url.toLocalFile())
        if local_path.is_dir():
            return True
        if local_path.is_file() and local_path.suffix.lower() in _DROPPABLE_AUDIO_SUFFIXES:
            return True
    return False


def _audio_files_in_directory(directory: Path) -> tuple[Path, ...]:
    discovered = [
        candidate
        for candidate in directory.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in _DROPPABLE_AUDIO_SUFFIXES
    ]
    discovered.sort(key=lambda path: _natural_sort_key(str(path.relative_to(directory))))
    return tuple(discovered)


def _natural_sort_key(value: str) -> tuple[object, ...]:
    tokens = _NATURAL_TOKEN_PATTERN.split(value.lower())
    key: list[object] = []
    for token in tokens:
        if not token:
            continue
        if token.isdigit():
            key.append(int(token))
        else:
            key.append(token)
    return tuple(key)


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
        if has_droppable_audio(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent | None) -> None:
        if event is None:
            return
        if has_droppable_audio(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None:
            return
        paths = dropped_audio_paths(event, include_directory_audio=True)
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
        if has_droppable_audio(event):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent | None) -> None:
        if event is None:
            return
        if has_droppable_audio(event):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None:
            return
        paths = dropped_audio_paths(event, include_directory_audio=True)
        if not paths:
            super().dropEvent(event)
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
    "has_droppable_audio",
]

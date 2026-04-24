"""
Timeline history: Framework-free undo/redo storage for canonical timeline edits.
Exists because undo belongs at the application/runtime boundary, not in Qt widgets.
Connects bounded before/after snapshots to app-shell undo, redo, and storage restore.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class UndoHistoryEntry:
    """One reversible timeline operation captured as before/after snapshots."""

    label: str
    before: Any
    after: Any
    storage_backed: bool = False


class UndoHistory:
    """Bounded linear undo/redo history with standard redo invalidation."""

    def __init__(self, *, limit: int) -> None:
        if limit <= 0:
            raise ValueError("UndoHistory limit must be greater than zero")
        self._limit = int(limit)
        self._undo: deque[UndoHistoryEntry] = deque()
        self._redo: deque[UndoHistoryEntry] = deque()

    def clear(self) -> None:
        """Drop all undo and redo state."""

        self._undo.clear()
        self._redo.clear()

    def push(self, entry: UndoHistoryEntry) -> None:
        """Append one new undoable entry and clear redo state."""

        self._undo.append(entry)
        while len(self._undo) > self._limit:
            self._undo.popleft()
        self._redo.clear()

    def undo(self) -> UndoHistoryEntry | None:
        """Return the next entry to undo, or None when history is empty."""

        if not self._undo:
            return None
        entry = self._undo.pop()
        self._redo.append(entry)
        return entry

    def redo(self) -> UndoHistoryEntry | None:
        """Return the next entry to redo, or None when redo is empty."""

        if not self._redo:
            return None
        entry = self._redo.pop()
        self._undo.append(entry)
        return entry

    def can_undo(self) -> bool:
        """Whether at least one undo step is available."""

        return bool(self._undo)

    def can_redo(self) -> bool:
        """Whether at least one redo step is available."""

        return bool(self._redo)

    def undo_label(self) -> str | None:
        """Return the next undo label when available."""

        return self._undo[-1].label if self._undo else None

    def redo_label(self) -> str | None:
        """Return the next redo label when available."""

        return self._redo[-1].label if self._redo else None

"""Background worker helpers for the Foundry window.
Exists to keep run-thread plumbing separate from the main window shell.
Connects long-running run actions to Qt thread lifecycle signals.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QObject, pyqtSignal


class _RunWorker(QObject):
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, action: Callable[[], str]):
        super().__init__()
        self._action = action

    def run(self) -> None:
        try:
            run_id = str(self._action())
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(run_id)


__all__ = ["_RunWorker"]

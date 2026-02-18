"""
Save Project Thread

Runs project save in a background QThread so the Qt UI event loop remains
responsive and save progress UI can be displayed.
"""
from typing import Callable, Any

from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.message import Log


class SaveProjectThread(QThread):
    """
    Executes a save callable in a worker thread.

    Signals:
        save_complete: emitted with the CommandResult-like object returned by save
        save_failed: emitted when an exception escapes the save callable
    """

    save_complete = pyqtSignal(object)
    save_failed = pyqtSignal(str, list)

    def __init__(self, save_func: Callable[[], Any], parent=None):
        super().__init__(parent)
        self._save_func = save_func
        if not callable(save_func):
            raise ValueError("save_func must be callable")

    def run(self):
        try:
            Log.info("SaveProjectThread: Starting save")
            result = self._save_func()
            self.save_complete.emit(result)
            Log.info("SaveProjectThread: Save finished")
        except Exception as e:
            Log.error(f"SaveProjectThread: Save failed with exception: {e}")
            import traceback
            details = [str(e), f"Traceback:\n{traceback.format_exc()}"]
            self.save_failed.emit(str(e), details)

"""
Run Block Thread

Runs block execution in a background QThread so the Qt UI stays responsive.
Uses the same process: calls facade.execute_block(block_id) in the thread.
Progress and events remain in-process; no subprocess or worker bootstrap.

Note: Running in a thread can be slower than running on the main thread because
the worker and the main (Qt) thread share Python's GIL. Progress updates are
throttled (see ProgressTracker) to reduce main-thread wakeups and contention.
For maximum training speed, the UI will be unresponsive during execution.
"""
from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.message import Log


class RunBlockThread(QThread):
    """
    Runs facade.execute_block(block_id) in a background thread.

    Single execution path: same API as synchronous execute_block, but off
    the main thread so the UI does not freeze during long-running blocks
    (e.g. PyTorch Audio Trainer). Progress and errors stay in-process.
    """

    execution_started = pyqtSignal()
    execution_complete = pyqtSignal(bool)  # success
    execution_failed = pyqtSignal(str, list)  # error message, detailed errors list

    def __init__(self, facade, block_id: str, parent=None):
        super().__init__(parent)
        self.facade = facade
        self.block_id = block_id
        if not block_id:
            raise ValueError("block_id is required for RunBlockThread")
        Log.info(f"RunBlockThread created: block_id={block_id}")

    def run(self):
        try:
            Log.info(f"RunBlockThread: Starting execution for block {self.block_id}")
            self.execution_started.emit()

            result = self.facade.execute_block(self.block_id)

            if result.success:
                Log.info("RunBlockThread: Execution completed successfully")
                self.execution_complete.emit(True)
            else:
                Log.error(f"RunBlockThread: Execution failed: {result.message or 'Unknown'}")
                detailed_errors = list(result.errors) if result.errors else []
                if getattr(result, "data", None) and isinstance(result.data, dict):
                    if result.data.get("error_type") == "FilterError":
                        detailed_errors.append(result.data)
                self.execution_failed.emit(
                    result.message or "Execution failed",
                    detailed_errors,
                )
        except Exception as e:
            Log.error(f"RunBlockThread: Exception during execution: {e}")
            import traceback
            detailed_errors = [str(e), f"Traceback:\n{traceback.format_exc()}"]
            self.execution_failed.emit(str(e), detailed_errors)

    def request_cancel(self):
        """
        Request cancellation. Not implemented; kept for API compatibility
        with code that previously used ExecutionThread.
        """
        Log.info("RunBlockThread: Cancellation requested (no-op)")

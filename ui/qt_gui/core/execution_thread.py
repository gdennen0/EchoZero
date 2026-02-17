"""
Execution Thread

Block execution runs in a background QThread via RunBlockThread (same process,
no subprocess). This module re-exports RunBlockThread as ExecutionThread for
backward compatibility.

The previous process-based implementation (persistent worker subprocess) has
been removed. Single-block execution uses RunBlockThread only; progress and
errors stay in-process.
"""
from ui.qt_gui.core.run_block_thread import RunBlockThread

# Backward compatibility: code that imported ExecutionThread gets RunBlockThread.
ExecutionThread = RunBlockThread

__all__ = ["ExecutionThread", "RunBlockThread"]

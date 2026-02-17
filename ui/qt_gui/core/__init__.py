"""
Core Qt Components

Shared Qt widgets and utilities used across the application.

Components:
- RunBlockThread: Runs single-block execution in a background thread (no subprocess).
- ExecutionThread: Alias for RunBlockThread (backward compatibility).
- StatusBarProgress: Status bar with integrated progress indicator
- PropertiesPanel: Block properties display
- ActionsPanel: Block actions display
- DockStateManager: Simple Qt-native dock state management
- IStatefulWindow: Interface for windows that save internal state
"""

# Execution: run block in thread so UI stays responsive; progress/errors in-process.
from .run_block_thread import RunBlockThread
from .execution_thread import ExecutionThread

# Widgets
from .progress_bar import StatusBarProgress
from .properties_panel import PropertiesPanel
from .actions_panel import ActionsPanel

# Window State Management
from .window_state_types import IStatefulWindow
from .dock_state_manager import DockStateManager

__all__ = [
    'RunBlockThread',
    'ExecutionThread',
    'StatusBarProgress',
    'PropertiesPanel',
    'ActionsPanel',
    # Window State
    'DockStateManager',
    'IStatefulWindow',
]

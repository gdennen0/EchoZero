"""
Core Qt Components

Shared Qt widgets and utilities used across the application.

Components:
- RunBlockThread: Runs single-block execution in a background thread (no subprocess).
- ExecutionThread: Alias for RunBlockThread (backward compatibility).
- StatusBarProgress: Status bar with integrated progress indicator
- PropertiesPanel: Block properties display
- ActionsPanel: Block actions display
- WorkspaceManager: Unified workspace state management
- IStatefulWindow: Interface for windows that save internal state
"""

# Execution: run block in thread so UI stays responsive; progress/errors in-process.
from .run_block_thread import RunBlockThread
from .execution_thread import ExecutionThread

# Widgets
from .progress_bar import StatusBarProgress
from .properties_panel import PropertiesPanel
from .actions_panel import ActionsPanel

# Workspace State Management
from .window_state_types import IStatefulWindow
from .workspace_manager import WorkspaceManager

__all__ = [
    'RunBlockThread',
    'ExecutionThread',
    'StatusBarProgress',
    'PropertiesPanel',
    'ActionsPanel',
    # Workspace State
    'WorkspaceManager',
    'IStatefulWindow',
]

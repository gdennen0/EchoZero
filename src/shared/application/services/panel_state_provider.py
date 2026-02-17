"""
Panel State Provider Protocol

Allows block panels to provide internal state to BlockStatusService
for status evaluation. This enables status conditions to access panel
state (like listener status, connection state, etc.) that isn't
stored in block metadata.

Usage:
    Panels implement get_panel_state() to return a dictionary of state.
    Status conditions can access this via facade.get_panel_state(block_id).
"""
from typing import Protocol, Dict, Any, Optional


class PanelStateProvider(Protocol):
    """
    Protocol for panels that can provide state to BlockStatusService.
    
    Panels implement this by providing a get_panel_state() method
    that returns a dictionary of state values.
    """
    
    def get_panel_state(self) -> Dict[str, Any]:
        """
        Get current panel state for status evaluation.
        
        Returns:
            Dictionary of state values (e.g., {"listening": True, "connected": False})
        """
        ...

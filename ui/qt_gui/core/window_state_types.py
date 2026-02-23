"""
Window State Types

Interface for windows that provide saveable internal state.
"""
from typing import Dict, Any


class IStatefulWindow:
    """
    Interface for windows that provide their own internal state.

    Any window/panel that wants to save internal state must implement this.
    The window is responsible for:
    - Providing its internal state via get_internal_state()
    - Restoring from internal state via restore_internal_state()
    """

    def get_internal_state(self) -> Dict[str, Any]:
        """
        Get internal state for saving.

        Returns:
            Dictionary of internal state (must be JSON-serializable)
        """
        return {}

    def restore_internal_state(self, state: Dict[str, Any]) -> None:
        """
        Restore internal state after loading.

        Args:
            state: Dictionary of internal state (from get_internal_state())
        """
        pass

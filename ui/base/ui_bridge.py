"""
UI Bridge Protocol

Defines the contract between core application and UI implementations.
The core application never depends on specific UI frameworks.
"""
from typing import Protocol, Optional, Dict, Any, List
from src.application.api.application_facade import ApplicationFacade


class UIBridge(Protocol):
    """
    Protocol that all UI implementations must follow.
    
    This creates a clean separation between core and UI.
    The ApplicationFacade is passed to the UI, never the other way around.
    """
    
    def initialize(self, facade: ApplicationFacade) -> None:
        """
        Initialize the UI with reference to ApplicationFacade.
        
        Args:
            facade: The unified application API
        """
        ...
    
    def run(self) -> int:
        """
        Start the UI event loop.
        
        Returns:
            Exit code (0 for success)
        """
        ...
    
    def shutdown(self) -> None:
        """Clean shutdown of UI resources"""
        ...


class BlockUIProvider(Protocol):
    """
    Protocol for block-specific UI components.
    
    Each block type can provide custom UI for parameter editing.
    Shared components can be reused across block UIs.
    """
    
    def get_block_type(self) -> str:
        """Return the block type this UI handles"""
        ...
    
    def create_editor_widget(self, block_data: Dict[str, Any]) -> Any:
        """
        Create a UI widget for editing block parameters.
        
        Args:
            block_data: Block metadata dictionary
            
        Returns:
            Framework-specific widget (e.g., QWidget for Qt)
        """
        ...
    
    def get_parameter_values(self) -> Dict[str, Any]:
        """
        Get current parameter values from UI.
        
        Returns:
            Dictionary of parameter name -> value
        """
        ...
    
    def set_parameter_values(self, values: Dict[str, Any]) -> None:
        """
        Update UI with parameter values.
        
        Args:
            values: Dictionary of parameter name -> value
        """
        ...


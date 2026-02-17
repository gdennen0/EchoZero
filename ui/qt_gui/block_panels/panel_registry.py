"""
Panel registry system for block panels.

Allows block-specific panel classes to register themselves
for automatic instantiation when panels are opened.
"""

from typing import Dict, Type, Optional

# Global registry mapping block types to panel classes
BLOCK_PANEL_REGISTRY: Dict[str, Type] = {}


def is_panel_registered(block_type: str) -> bool:
    """
    Check if a panel is registered for a block type.
    
    Args:
        block_type: The block type string
    
    Returns:
        True if a panel is registered, False otherwise
    """
    return block_type in BLOCK_PANEL_REGISTRY


def register_block_panel(block_type: str):
    """
    Decorator to register a block panel class for a specific block type.
    
    Usage:
        @register_block_panel("Separator")
        class SeparatorPanel(BlockPanelBase):
            ...
    
    Args:
        block_type: The block type string (e.g., "Separator", "LoadAudio")
    
    Returns:
        Decorator function that registers the class
    """
    def decorator(cls):
        BLOCK_PANEL_REGISTRY[block_type] = cls
        return cls
    return decorator


def get_panel_class(block_type: str) -> Optional[Type]:
    """
    Get the panel class registered for a block type.
    
    Args:
        block_type: The block type string
    
    Returns:
        The panel class, or None if no panel is registered for this type
    """
    return BLOCK_PANEL_REGISTRY.get(block_type)


def is_panel_registered(block_type: str) -> bool:
    """
    Check if a panel is registered for a block type.
    
    Args:
        block_type: The block type string
    
    Returns:
        True if a panel is registered, False otherwise
    """
    return block_type in BLOCK_PANEL_REGISTRY


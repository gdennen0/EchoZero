"""
Output Name Helpers

Standardized helper functions for generating semantic output names.
Ensures consistent "port:item" format across all blocks.
"""


def make_output_name(port_name: str, item_name: str) -> str:
    """
    Create a semantic output name in standard format: "port_name:item_name"
    
    Args:
        port_name: Name of the output port (e.g., "audio", "events")
        item_name: Name of the specific output item (e.g., "vocals", "main", "onsets")
    
    Returns:
        Semantic output name in format "port_name:item_name"
    
    Example:
        make_output_name("audio", "vocals") -> "audio:vocals"
        make_output_name("events", "onsets") -> "events:onsets"
    """
    return f"{port_name}:{item_name}"


def make_default_output_name(port_name: str) -> str:
    """
    Create default single-item output name for a port.
    
    Args:
        port_name: Name of the output port
    
    Returns:
        Default semantic output name: "port_name:main"
    
    Example:
        make_default_output_name("audio") -> "audio:main"
        make_default_output_name("events") -> "events:main"
    """
    return f"{port_name}:main"


def parse_output_name(output_name: str) -> tuple[str, str]:
    """
    Parse a semantic output name into (port_name, item_name).
    
    Args:
        output_name: Semantic output name in format "port_name:item_name"
    
    Returns:
        Tuple of (port_name, item_name)
    
    Raises:
        ValueError: If output_name doesn't contain ":"
    
    Example:
        parse_output_name("audio:vocals") -> ("audio", "vocals")
        parse_output_name("events:onsets") -> ("events", "onsets")
    """
    if ":" not in output_name:
        raise ValueError(f"Invalid output_name format: {output_name}. Expected 'port:item' format.")
    port_name, item_name = output_name.split(":", 1)
    return port_name, item_name




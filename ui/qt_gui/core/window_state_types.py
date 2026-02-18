"""
Window State Types

Type definitions for the unified window state management system.
Each window/panel is responsible for providing its own state data.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class DockPosition(Enum):
    """Position of a docked window"""
    FLOATING = "floating"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class TabGroupInfo:
    """Information about a tab group"""
    dock_position: DockPosition  # Where the tab group is docked
    window_ids: List[str]  # Ordered list of window IDs in this tab group
    active_index: int = 0  # Which tab is active (0-indexed)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dock_position": self.dock_position.value,
            "window_ids": self.window_ids,
            "active_index": self.active_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabGroupInfo":
        return cls(
            dock_position=DockPosition(data.get("dock_position", "top")),
            window_ids=data.get("window_ids", []),
            active_index=data.get("active_index", 0)
        )


@dataclass
class WindowGeometry:
    """Geometry of a window"""
    x: int
    y: int
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WindowGeometry":
        return cls(
            x=data.get("x", 100),
            y=data.get("y", 100),
            width=data.get("width", 800),
            height=data.get("height", 600)
        )
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class WindowState:
    """
    Complete state of a single window.
    
    Attributes:
        window_id: Unique identifier for this window instance
        window_type: Type of window (e.g., "node_editor", "block_panel")
        block_type: For block panels, the block type (e.g., "Editor", "ExportAudio")
        block_id: For block panels, the specific block ID (used for current session only)
        
        dock_position: Where the window is docked (or FLOATING)
        geometry: Window geometry (used for floating windows)
        is_visible: Whether the window is visible
        
        tab_group_id: ID of tab group this window belongs to (if any)
        
        internal_state: Window-specific internal state (provided by the window)
    """
    window_id: str
    window_type: str
    
    # For block panels
    block_type: Optional[str] = None
    block_id: Optional[str] = None
    
    # Position/layout
    dock_position: DockPosition = DockPosition.FLOATING
    geometry: Optional[WindowGeometry] = None
    is_visible: bool = True
    
    # Tab grouping
    tab_group_id: Optional[str] = None
    
    # Internal state (provided by the window itself)
    internal_state: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "window_type": self.window_type,
            "block_type": self.block_type,
            "block_id": self.block_id,
            "dock_position": self.dock_position.value,
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "is_visible": self.is_visible,
            "tab_group_id": self.tab_group_id,
            "internal_state": self.internal_state
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WindowState":
        geometry = None
        if data.get("geometry"):
            geometry = WindowGeometry.from_dict(data["geometry"])
        
        return cls(
            window_id=data.get("window_id", ""),
            window_type=data.get("window_type", ""),
            block_type=data.get("block_type"),
            block_id=data.get("block_id"),
            dock_position=DockPosition(data.get("dock_position", "floating")),
            geometry=geometry,
            is_visible=data.get("is_visible", True),
            tab_group_id=data.get("tab_group_id"),
            internal_state=data.get("internal_state", {})
        )


@dataclass
class LayoutState:
    """
    Complete layout state for the application.
    
    This is what gets saved to session state or exported as a layout file.
    """
    # Main window
    main_window_geometry: Optional[WindowGeometry] = None
    main_window_maximized: bool = False
    
    # All windows
    windows: List[WindowState] = field(default_factory=list)
    
    # Tab groups
    tab_groups: List[TabGroupInfo] = field(default_factory=list)
    
    # Settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    version: str = "1.0"
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "main_window_geometry": self.main_window_geometry.to_dict() if self.main_window_geometry else None,
            "main_window_maximized": self.main_window_maximized,
            "windows": [w.to_dict() for w in self.windows],
            "tab_groups": [tg.to_dict() for tg in self.tab_groups],
            "settings": self.settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayoutState":
        # Validate version
        version = data.get("version", "0.0")
        if version < "1.0":
            # Old format - return empty state
            return cls()
        
        main_geom = None
        if data.get("main_window_geometry"):
            main_geom = WindowGeometry.from_dict(data["main_window_geometry"])
        
        windows = [WindowState.from_dict(w) for w in data.get("windows", [])]
        tab_groups = [TabGroupInfo.from_dict(tg) for tg in data.get("tab_groups", [])]
        
        return cls(
            main_window_geometry=main_geom,
            main_window_maximized=data.get("main_window_maximized", False),
            windows=windows,
            tab_groups=tab_groups,
            settings=data.get("settings", {}),
            version=version,
            timestamp=data.get("timestamp", 0.0)
        )


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

"""
Snap Calculator

Calculates snap points for events using GridCalculator.
Always returns snapped times in seconds (single source of truth).

Design:
- Uses GridCalculator for base interval calculation
- Uses base minor interval directly for finer snapping (no adaptive density skip)
- Uses integer-based indexing for perfect alignment
- Returns seconds (internal format)
- Supports explicit snap intervals (1f, 5f, 10f, 1s modes)
"""

from typing import Optional, TYPE_CHECKING, Union
from .grid_calculator import GridCalculator
from .unit_preferences import UnitPreference
from .settings import GridSettings

if TYPE_CHECKING:
    from ..grid_system import GridSystem


class SnapCalculator:
    """
    Calculates snap points for events.
    
    Uses the base minor interval from GridCalculator for finer snapping precision.
    (Grid renderer uses adaptive density for visual spacing, but snapping uses
    the base interval for finer control.)
    
    Supports explicit snap intervals for DAW-style frame snapping (1f, 5f, etc.)
    """
    
    def __init__(
        self,
        grid_calculator: GridCalculator,
        settings: Optional[Union[GridSettings, 'GridSystem']] = None
    ):
        """
        Initialize snap calculator.
        
        Args:
            grid_calculator: GridCalculator instance to use for intervals
            settings: GridSettings or legacy GridSystem for explicit snap interval modes
        """
        self.grid_calculator = grid_calculator
        self._settings = settings
        self.snap_enabled = True
    
    @property
    def settings(self) -> Optional[GridSettings]:
        """Get settings, converting from legacy GridSystem if needed."""
        if self._settings is None:
            return None
        # Support both GridSettings and legacy GridSystem
        if isinstance(self._settings, GridSettings):
            return self._settings
        # Legacy GridSystem - access its settings
        if hasattr(self._settings, 'settings'):
            return self._settings.settings
        return None
    
    # Legacy property for backward compatibility
    @property
    def grid_system(self):
        """Legacy property for backward compatibility."""
        return self._settings
    
    @grid_system.setter
    def grid_system(self, value):
        """Legacy setter for backward compatibility."""
        self._settings = value
    
    def snap_time(
        self,
        time: float,
        pixels_per_second: float,
        unit_preference: UnitPreference,
        explicit_interval: Optional[float] = None
    ) -> float:
        """
        Snap time to base grid interval or explicit interval.
        
        Uses the base minor interval from GridCalculator for finer snapping precision.
        (Grid renderer uses adaptive density for visual spacing, but snapping uses
        the base interval for finer control.)
        
        Args:
            time: Time in seconds to snap
            pixels_per_second: Current zoom level
            unit_preference: Preferred unit alignment
            explicit_interval: If provided, snap to this interval instead of grid
                              (e.g., 1/30 for 1 frame at 30fps). If None, checks
                              settings for explicit interval mode.
        
        Returns:
            Snapped time in seconds (aligned to base grid interval or explicit interval)
        """
        if not self.snap_enabled:
            return time
        
        if pixels_per_second is None or pixels_per_second <= 0:
            return time
        
        # Check for explicit interval from settings if not passed directly
        if explicit_interval is None:
            settings = self.settings
            if settings is not None:
                explicit_interval = settings.get_snap_interval_seconds()
        
        # Use explicit interval if provided (e.g., 1f, 5f, 10f, 1s modes)
        if explicit_interval is not None and explicit_interval > 0:
            # Snap directly to explicit interval (frame-based or second-based)
            line_idx = round(time / explicit_interval)
            snapped_time = line_idx * explicit_interval
            return max(0, snapped_time)
        
        # Get base intervals from grid calculator (auto mode)
        try:
            _, minor_interval = self.grid_calculator.get_intervals(
                pixels_per_second,
                unit_preference
            )
        except Exception:
            return time
        
        # === FINER SNAPPING ===
        # Use base minor_interval directly for finer snapping precision
        # (Grid renderer uses adaptive density for visual spacing, but snapping
        # uses the base interval for finer control)
        
        if minor_interval <= 0:
            return time
        
        # === INTEGER-BASED SNAPPING ===
        # Snap to integer line index using base minor interval
        line_idx = round(time / minor_interval)
        snapped_time = line_idx * minor_interval
        
        return max(0, snapped_time)


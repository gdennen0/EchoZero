"""
Timing System

Unified timing and grid system with seconds as the single source of truth.
All time units (frames, timecode, beats, etc.) are display/calculation formats
derived from seconds.

Modules:
- TimeConverter: Convert between seconds and display units
- GridCalculator: Calculate grid intervals (in seconds)
- SnapCalculator: Calculate snap points (uses GridCalculator)
- GridRenderer: Draw grid lines (uses GridCalculator)
- GridSettings: Configuration settings for grid/snap behavior
"""

from .time_converter import TimeConverter, TimeFormat
from .unit_preferences import UnitPreference
from .grid_calculator import GridCalculator
from .snap_calculator import SnapCalculator
from .grid_renderer import GridRenderer
from .timebase_mapper import TimebaseMode, timebase_to_unit_preference
from .settings import GridSettings, VALID_SNAP_MODES, COMMON_FRAME_RATES

__all__ = [
    'TimeConverter',
    'TimeFormat',
    'UnitPreference',
    'GridCalculator',
    'SnapCalculator',
    'GridRenderer',
    'TimebaseMode',
    'timebase_to_unit_preference',
    'GridSettings',
    'VALID_SNAP_MODES',
    'COMMON_FRAME_RATES',
]


"""
Grid Renderer

Draws grid lines on the timeline using GridCalculator.
Pure rendering logic with POC-verified optimizations:
- Cosmetic pens for consistent line width at any zoom
- Batch drawing with drawLines() for performance
- Integer-based line indexing (no floating point bugs)
- Adaptive density based on pixel spacing

Design:
- Uses GridCalculator for intervals
- Handles major/minor line drawing
- Only draws visible lines (within rect)
"""

import math
from typing import List, Tuple, Optional

from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QRectF, QLineF

from .grid_calculator import GridCalculator
from .unit_preferences import UnitPreference
from ..constants import (
    MIN_MINOR_LINE_SPACING_PX,
    MIN_MAJOR_LINE_SPACING_PX,
    MAX_GRID_LINES,
)


class GridRenderer:
    """
    Draws grid lines on timeline using Qt best practices (POC-verified).
    
    Key optimizations:
    1. Cosmetic pens (consistent width at any zoom)
    2. Batch drawing with drawLines()
    3. Integer-based line counting (no floating point bugs)
    4. Adaptive density based on pixel spacing
    """
    
    def __init__(self, grid_calculator: GridCalculator):
        """
        Initialize grid renderer.
        
        Args:
            grid_calculator: GridCalculator instance to use for intervals
        """
        self.grid_calculator = grid_calculator
        self.show_grid_lines = True
    
    def draw_grid(
        self,
        painter: QPainter,
        rect: QRectF,
        pixels_per_second: float,
        unit_preference: UnitPreference,
        scene_rect: QRectF,
        major_color: Optional[QColor] = None,
        minor_color: Optional[QColor] = None
    ) -> None:
        """
        Draw grid lines in the given rectangle using batch rendering.
        
        Uses POC-verified optimizations:
        - Cosmetic pens for consistent line width
        - Batch drawing with drawLines()
        - Integer-based line indexing
        - Adaptive density
        
        Note: Grid intervals are automatically calculated from timebase/FPS.
        No manual multipliers - grid adapts to zoom level.
        
        Args:
            painter: QPainter to draw with
            rect: Visible rectangle to draw in
            pixels_per_second: Current zoom level
            unit_preference: Preferred unit alignment
            scene_rect: Full scene rectangle
            major_color: Color for major lines (uses default if None)
            minor_color: Color for minor lines (uses default if None)
        """
        if not self.show_grid_lines:
            return
        
        # Get base intervals from grid calculator (auto-calculated from timebase/FPS)
        try:
            base_major, base_minor = self.grid_calculator.get_intervals(
                pixels_per_second,
                unit_preference
            )
        except Exception:
            # Fallback to reasonable defaults
            base_minor = 0.1
            base_major = 1.0
        
        # === STEP 1: Calculate adaptive grid density ===
        # Skip lines that would be too close together
        
        minor_px = base_minor * pixels_per_second
        
        # Calculate line skip factor for minor lines
        if minor_px >= MIN_MINOR_LINE_SPACING_PX:
            minor_skip = 1
        else:
            minor_skip = math.ceil(MIN_MINOR_LINE_SPACING_PX / minor_px)
            minor_skip = self._snap_to_nice_divisor(minor_skip)
        
        # Effective display interval for minor lines
        display_minor = base_minor * minor_skip
        display_minor_px = display_minor * pixels_per_second
        
        # Calculate line skip factor for major lines
        major_px = base_major * pixels_per_second
        
        if major_px >= MIN_MAJOR_LINE_SPACING_PX:
            major_skip = 1
        else:
            major_skip = math.ceil(MIN_MAJOR_LINE_SPACING_PX / major_px)
            major_skip = self._snap_to_nice_divisor(major_skip)
        
        # Effective display interval for major lines
        display_major = base_major * major_skip
        
        # === STEP 2: Prepare cosmetic pens ===
        # Cosmetic pens have consistent width regardless of view transform
        
        if minor_color:
            minor_pen = QPen(minor_color, 1, Qt.PenStyle.DotLine)
        else:
            from ..core.style import TimelineStyle as Colors
            minor_pen = QPen(Colors.BORDER.darker(130), 1, Qt.PenStyle.DotLine)
        minor_pen.setCosmetic(True)  # Key Qt feature!
        
        if major_color:
            major_pen = QPen(major_color, 1)
        else:
            from ..core.style import TimelineStyle as Colors
            major_pen = QPen(Colors.BORDER, 1)
        major_pen.setCosmetic(True)  # Key Qt feature!
        
        # === STEP 3: Calculate visible range using INTEGER line indices ===
        # This avoids floating point modulo bugs
        
        # Calculate the visible time range
        visible_left = max(rect.left(), scene_rect.left())
        visible_right = min(rect.right(), scene_rect.right())
        
        if display_minor_px <= 0:
            return
        
        # Convert pixel range to line indices (integer math)
        first_line_idx = max(0, int(visible_left / display_minor_px))
        last_line_idx = int(visible_right / display_minor_px) + 1
        
        # Limit to reasonable number for performance
        if last_line_idx - first_line_idx > MAX_GRID_LINES:
            last_line_idx = first_line_idx + MAX_GRID_LINES
        
        # === STEP 4: Collect lines for batch drawing ===
        
        minor_lines: List[QLineF] = []
        major_lines: List[QLineF] = []
        
        # How many display_minor intervals fit in one display_major interval?
        # Use integer division for reliable major line detection
        if display_minor > 0:
            lines_per_major = max(1, round(display_major / display_minor))
        else:
            lines_per_major = 1
        
        y_top = max(rect.top(), scene_rect.top())
        y_bottom = min(rect.bottom(), scene_rect.bottom())
        
        # Duration limit
        max_x = scene_rect.right()
        
        for i in range(first_line_idx, last_line_idx + 1):
            x = i * display_minor_px
            
            if x < 0 or x > max_x:
                continue
            
            line = QLineF(x, y_top, x, y_bottom)
            
            # Integer-based major line detection (no floating point modulo!)
            is_major = (lines_per_major > 0 and i % lines_per_major == 0)
            
            if is_major:
                major_lines.append(line)
            else:
                minor_lines.append(line)
        
        # === STEP 5: Batch draw with drawLines() ===
        # Much more efficient than individual drawLine() calls
        
        if minor_lines:
            painter.setPen(minor_pen)
            painter.drawLines(minor_lines)
        
        if major_lines:
            painter.setPen(major_pen)
            painter.drawLines(major_lines)
    
    @staticmethod
    def _snap_to_nice_divisor(value: int) -> int:
        """
        Snap a skip factor to a 'nice' number for musical/temporal alignment.
        
        Nice numbers are: 1, 2, 4, 5, 8, 10, 12, 16, 20, 24, 25, 30, 32...
        These align well with musical beats, frames, and seconds.
        
        Args:
            value: Raw skip factor
            
        Returns:
            Nice skip factor >= value
        """
        nice_numbers = [1, 2, 4, 5, 8, 10, 12, 16, 20, 24, 25, 30, 32, 40, 48, 50, 60, 64, 80, 100]
        
        for nice in nice_numbers:
            if nice >= value:
                return nice
        
        # For very large values, round up to nearest 100
        return ((value + 99) // 100) * 100
    
    @staticmethod
    def _snap_to_boundary(time: float, interval: float) -> float:
        """
        Snap time to nearest unit boundary.
        
        Args:
            time: Time in seconds
            interval: Interval in seconds
        
        Returns:
            Snapped time
        """
        if interval <= 0:
            return time
        return round(time / interval) * interval

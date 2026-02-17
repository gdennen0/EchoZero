"""
PlotEvents Block Settings

Settings schema and manager for PlotEvents blocks.
"""
from dataclasses import dataclass
from typing import Optional, Union, Tuple

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.utils.message import Log


@dataclass
class PlotEventsBlockSettings(BaseSettings):
    """
    Settings schema for PlotEvents blocks.
    
    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    """
    # Plot style
    plot_style: str = "bars"  # "bars", "markers", "piano_roll"
    
    # Figure size (stored as separate width/height for easier UI management)
    figsize_width: float = 12.0  # Width in inches
    figsize_height: float = 8.0  # Height in inches
    
    # DPI
    dpi: int = 100  # Resolution (72-600)
    
    # Display options
    show_labels: bool = True
    show_grid: bool = True
    show_energy_amplitude: bool = False  # Show energy and amplitude subplots (3-panel layout)
    
    # Legacy support: map figure_size tuple to figsize_width/height
    @classmethod
    def from_dict(cls, data: dict):
        """
        Load settings from block metadata with backwards compatibility.
        
        Handles legacy "figure_size" tuple format mapping to "figsize_width"/"figsize_height".
        """
        merged = dict(data)
        
        # Map legacy "figure_size" tuple to separate width/height if present
        if "figure_size" in merged and "figsize_width" not in merged:
            figure_size = merged.get("figure_size")
            if isinstance(figure_size, (list, tuple)) and len(figure_size) == 2:
                merged["figsize_width"] = float(figure_size[0])
                merged["figsize_height"] = float(figure_size[1])
        
        return super().from_dict(merged)
    
    def to_dict(self) -> dict:
        """
        Convert to metadata format.
        
        Stores figsize_width and figsize_height (not legacy figure_size tuple).
        """
        return super().to_dict()


class PlotEventsSettingsManager(BlockSettingsManager):
    """
    Settings manager for PlotEvents blocks.
    
    Provides type-safe property accessors with validation.
    All settings changes go through this manager (single pathway).
    """
    SETTINGS_CLASS = PlotEventsBlockSettings
    
    @property
    def plot_style(self) -> str:
        """Get plot style."""
        return self._settings.plot_style
    
    @plot_style.setter
    def plot_style(self, value: str):
        """Set plot style with validation."""
        valid_styles = {"bars", "markers", "piano_roll"}
        if value not in valid_styles:
            raise ValueError(
                f"Invalid plot style: '{value}'. "
                f"Valid options: {', '.join(sorted(valid_styles))}"
            )
        
        if value != self._settings.plot_style:
            self._settings.plot_style = value
            self._save_setting('plot_style')
    
    @property
    def figsize_width(self) -> float:
        """Get figure width in inches."""
        return self._settings.figsize_width
    
    @figsize_width.setter
    def figsize_width(self, value: float):
        """Set figure width with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Figure width must be a number, got {type(value).__name__}")
        
        if value < 1.0:
            raise ValueError(f"Figure width must be >= 1.0 inches, got {value}")
        
        if value != self._settings.figsize_width:
            self._settings.figsize_width = float(value)
            self._save_setting('figsize_width')
    
    @property
    def figsize_height(self) -> float:
        """Get figure height in inches."""
        return self._settings.figsize_height
    
    @figsize_height.setter
    def figsize_height(self, value: float):
        """Set figure height with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Figure height must be a number, got {type(value).__name__}")
        
        if value < 1.0:
            raise ValueError(f"Figure height must be >= 1.0 inches, got {value}")
        
        if value != self._settings.figsize_height:
            self._settings.figsize_height = float(value)
            self._save_setting('figsize_height')
    
    @property
    def dpi(self) -> int:
        """Get DPI (resolution)."""
        return self._settings.dpi
    
    @dpi.setter
    def dpi(self, value: int):
        """Set DPI with validation."""
        if not isinstance(value, int):
            raise ValueError(f"DPI must be an integer, got {type(value).__name__}")
        
        if not 72 <= value <= 600:
            raise ValueError(f"DPI must be between 72 and 600, got {value}")
        
        if value != self._settings.dpi:
            self._settings.dpi = value
            self._save_setting('dpi')
    
    @property
    def show_labels(self) -> bool:
        """Get show labels setting."""
        return self._settings.show_labels
    
    @show_labels.setter
    def show_labels(self, value: bool):
        """Set show labels setting."""
        if not isinstance(value, bool):
            raise ValueError(f"Show labels must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.show_labels:
            self._settings.show_labels = value
            self._save_setting('show_labels')
    
    @property
    def show_grid(self) -> bool:
        """Get show grid setting."""
        return self._settings.show_grid
    
    @show_grid.setter
    def show_grid(self, value: bool):
        """Set show grid setting."""
        if not isinstance(value, bool):
            raise ValueError(f"Show grid must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.show_grid:
            self._settings.show_grid = value
            self._save_setting('show_grid')
    
    @property
    def show_energy_amplitude(self) -> bool:
        """Get show energy/amplitude subplots setting."""
        return self._settings.show_energy_amplitude
    
    @show_energy_amplitude.setter
    def show_energy_amplitude(self, value: bool):
        """Set show energy/amplitude subplots setting."""
        if not isinstance(value, bool):
            raise ValueError(f"Show energy/amplitude must be a boolean, got {type(value).__name__}")
        
        if value != self._settings.show_energy_amplitude:
            self._settings.show_energy_amplitude = value
            self._save_setting('show_energy_amplitude')

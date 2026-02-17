"""
TranscribeNote Block Settings

Settings schema and manager for TranscribeNote blocks.
"""
from dataclasses import dataclass

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.utils.message import Log


@dataclass
class TranscribeNoteBlockSettings(BaseSettings):
    """
    Settings schema for TranscribeNote blocks.
    
    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    """
    # Detection parameters
    onset_threshold: float = 0.5  # 0.0 to 1.0
    min_duration: float = 0.05  # Minimum note duration in seconds
    
    # Note range (MIDI)
    min_note: int = 21  # A0 (lowest MIDI note to detect)
    max_note: int = 108  # C8 (highest MIDI note to detect)


class TranscribeNoteSettingsManager(BlockSettingsManager):
    """
    Settings manager for TranscribeNote blocks.
    
    Provides type-safe property accessors with validation.
    All settings changes go through this manager (single pathway).
    """
    SETTINGS_CLASS = TranscribeNoteBlockSettings
    
    @property
    def onset_threshold(self) -> float:
        """Get onset detection threshold (0.0 to 1.0)."""
        return self._settings.onset_threshold
    
    @onset_threshold.setter
    def onset_threshold(self, value: float):
        """Set onset detection threshold with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Onset threshold must be a number, got {type(value).__name__}")
        
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Onset threshold must be between 0.0 and 1.0, got {value}"
            )
        
        # Update if changed (use small epsilon for float comparison)
        if abs(value - self._settings.onset_threshold) > 0.001:
            self._settings.onset_threshold = float(value)
            self._save_setting('onset_threshold')
    
    @property
    def min_duration(self) -> float:
        """Get minimum note duration (in seconds)."""
        return self._settings.min_duration
    
    @min_duration.setter
    def min_duration(self, value: float):
        """Set minimum note duration with validation."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Min duration must be a number, got {type(value).__name__}")
        
        if value < 0.0:
            raise ValueError(f"Min duration must be >= 0.0, got {value}")
        
        # Update if changed (use small epsilon for float comparison)
        if abs(value - self._settings.min_duration) > 0.0001:
            self._settings.min_duration = float(value)
            self._save_setting('min_duration')
    
    @property
    def min_note(self) -> int:
        """Get minimum MIDI note (0-127)."""
        return self._settings.min_note
    
    @min_note.setter
    def min_note(self, value: int):
        """Set minimum MIDI note with validation."""
        if not isinstance(value, int):
            raise ValueError(f"Min note must be an integer, got {type(value).__name__}")
        
        if not 0 <= value <= 127:
            raise ValueError(f"Min note must be between 0 and 127, got {value}")
        
        # Validate range (min_note <= max_note)
        if value > self._settings.max_note:
            raise ValueError(
                f"Min note ({value}) cannot be greater than max note ({self._settings.max_note})"
            )
        
        if value != self._settings.min_note:
            self._settings.min_note = value
            self._save_setting('min_note')
    
    @property
    def max_note(self) -> int:
        """Get maximum MIDI note (0-127)."""
        return self._settings.max_note
    
    @max_note.setter
    def max_note(self, value: int):
        """Set maximum MIDI note with validation."""
        if not isinstance(value, int):
            raise ValueError(f"Max note must be an integer, got {type(value).__name__}")
        
        if not 0 <= value <= 127:
            raise ValueError(f"Max note must be between 0 and 127, got {value}")
        
        # Validate range (min_note <= max_note)
        if value < self._settings.min_note:
            raise ValueError(
                f"Max note ({value}) cannot be less than min note ({self._settings.min_note})"
            )
        
        if value != self._settings.max_note:
            self._settings.max_note = value
            self._save_setting('max_note')

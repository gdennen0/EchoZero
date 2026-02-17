"""
Editor Event Filter Settings

Settings schema and manager for Editor block event filtering.
Part of the "processes" improvement area.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Set

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.utils.message import Log


@dataclass
class EditorEventFilterSettings(BaseSettings):
    """
    Settings schema for Editor block event filtering.
    
    All fields have default values for backwards compatibility.
    Settings are stored in block.metadata at the top level.
    """
    # Filter enabled state
    event_filter_enabled: bool = True
    
    # Classification filters
    enabled_classifications: List[str] = field(default_factory=list)  # Include these classifications
    excluded_classifications: List[str] = field(default_factory=list)  # Exclude these classifications
    
    # Time range filters
    min_time: Optional[float] = None  # Minimum event time in seconds
    max_time: Optional[float] = None  # Maximum event time in seconds
    
    # Duration filters
    min_duration: Optional[float] = None  # Minimum event duration in seconds
    max_duration: Optional[float] = None  # Maximum event duration in seconds
    
    # Metadata filters (stored as dict: {key: {"operator": "...", "value": ...}})
    metadata_filters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary for storage.
        
        Overrides base to_dict() to ensure all sets are converted to lists
        for JSON serialization compatibility.
        """
        data = asdict(self)
        # Convert any sets to lists (for JSON serialization)
        return self._convert_sets_to_lists(data)
    
    @staticmethod
    def _convert_sets_to_lists(obj: Any) -> Any:
        """Recursively convert sets to lists in nested structures"""
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: EditorEventFilterSettings._convert_sets_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [EditorEventFilterSettings._convert_sets_to_lists(item) for item in obj]
        return obj
    
    def to_event_filter_dict(self) -> Dict[str, Any]:
        """
        Convert settings to EventFilter dict format.
        
        Returns:
            Dictionary compatible with EventFilter.from_dict()
        """
        return {
            "enabled_classifications": self.enabled_classifications if self.enabled_classifications else None,
            "excluded_classifications": self.excluded_classifications if self.excluded_classifications else None,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "metadata_filters": self.metadata_filters if self.metadata_filters else {},
            "enabled": self.event_filter_enabled
        }
    
    @classmethod
    def from_event_filter_dict(cls, filter_dict: Dict[str, Any]) -> 'EditorEventFilterSettings':
        """
        Create settings from EventFilter dict format.
        
        Args:
            filter_dict: Dictionary from EventFilter.to_dict()
            
        Returns:
            EditorEventFilterSettings instance
        """
        # Convert sets to lists if present (for backwards compatibility)
        enabled_classifications = filter_dict.get("enabled_classifications")
        if isinstance(enabled_classifications, set):
            enabled_classifications = list(enabled_classifications)
        elif enabled_classifications is None:
            enabled_classifications = []
        
        excluded_classifications = filter_dict.get("excluded_classifications")
        if isinstance(excluded_classifications, set):
            excluded_classifications = list(excluded_classifications)
        elif excluded_classifications is None:
            excluded_classifications = []
        
        return cls(
            event_filter_enabled=filter_dict.get("enabled", True),
            enabled_classifications=enabled_classifications,
            excluded_classifications=excluded_classifications,
            min_time=filter_dict.get("min_time"),
            max_time=filter_dict.get("max_time"),
            min_duration=filter_dict.get("min_duration"),
            max_duration=filter_dict.get("max_duration"),
            metadata_filters=filter_dict.get("metadata_filters", {})
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EditorEventFilterSettings':
        """
        Create settings from dictionary with set-to-list conversion.
        
        Handles sets in the data (for backwards compatibility) by converting to lists.
        """
        # Convert all sets to lists recursively before creating instance
        converted_data = cls._convert_sets_to_lists(data)
        
        # Use parent from_dict but with converted data
        from dataclasses import fields
        defaults = cls()
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in converted_data.items() if k in valid_keys}
        merged = asdict(defaults)
        merged.update(filtered_data)
        
        return cls(**merged)


class EditorEventFilterSettingsManager(BlockSettingsManager):
    """
    Settings manager for Editor block event filtering.
    
    Provides type-safe property accessors with validation.
    Integrates with EventFilterManager for filter application.
    """
    SETTINGS_CLASS = EditorEventFilterSettings
    
    def __init__(self, facade, block_id: str, parent=None):
        super().__init__(facade, block_id, parent)
    
    @property
    def event_filter_enabled(self) -> bool:
        """Whether event filtering is enabled"""
        return self._settings.event_filter_enabled
    
    @event_filter_enabled.setter
    def event_filter_enabled(self, value: bool):
        if value != self._settings.event_filter_enabled:
            self._settings.event_filter_enabled = value
            self._save_setting('event_filter_enabled')
    
    @property
    def enabled_classifications(self) -> List[str]:
        """List of classifications to include"""
        return self._settings.enabled_classifications
    
    @enabled_classifications.setter
    def enabled_classifications(self, value: List[str]):
        # Convert set to list if needed (for JSON serialization)
        if isinstance(value, set):
            value = list(value)
        if value != self._settings.enabled_classifications:
            self._settings.enabled_classifications = value
            self._save_setting('enabled_classifications')
    
    @property
    def excluded_classifications(self) -> List[str]:
        """List of classifications to exclude"""
        return self._settings.excluded_classifications
    
    @excluded_classifications.setter
    def excluded_classifications(self, value: List[str]):
        # Convert set to list if needed (for JSON serialization)
        if isinstance(value, set):
            value = list(value)
        if value != self._settings.excluded_classifications:
            self._settings.excluded_classifications = value
            self._save_setting('excluded_classifications')
    
    @property
    def min_time(self) -> Optional[float]:
        """Minimum event time in seconds"""
        return self._settings.min_time
    
    @min_time.setter
    def min_time(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("min_time must be >= 0")
        if value != self._settings.min_time:
            self._settings.min_time = value
            self._save_setting('min_time')
    
    @property
    def max_time(self) -> Optional[float]:
        """Maximum event time in seconds"""
        return self._settings.max_time
    
    @max_time.setter
    def max_time(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_time must be >= 0")
        if self._settings.min_time is not None and value is not None and value < self._settings.min_time:
            raise ValueError("max_time must be >= min_time")
        if value != self._settings.max_time:
            self._settings.max_time = value
            self._save_setting('max_time')
    
    @property
    def min_duration(self) -> Optional[float]:
        """Minimum event duration in seconds"""
        return self._settings.min_duration
    
    @min_duration.setter
    def min_duration(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("min_duration must be >= 0")
        if value != self._settings.min_duration:
            self._settings.min_duration = value
            self._save_setting('min_duration')
    
    @property
    def max_duration(self) -> Optional[float]:
        """Maximum event duration in seconds"""
        return self._settings.max_duration
    
    @max_duration.setter
    def max_duration(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("max_duration must be >= 0")
        if self._settings.min_duration is not None and value is not None and value < self._settings.min_duration:
            raise ValueError("max_duration must be >= min_duration")
        if value != self._settings.max_duration:
            self._settings.max_duration = value
            self._save_setting('max_duration')
    
    @property
    def metadata_filters(self) -> Dict[str, Dict[str, Any]]:
        """Metadata filters: {key: {"operator": "...", "value": ...}}"""
        return self._settings.metadata_filters
    
    @metadata_filters.setter
    def metadata_filters(self, value: Dict[str, Dict[str, Any]]):
        if value != self._settings.metadata_filters:
            self._settings.metadata_filters = value
            self._save_setting('metadata_filters')
    
    def get_event_filter_dict(self) -> Dict[str, Any]:
        """
        Get event filter configuration as dict for EventFilterManager.
        
        Returns:
            Dictionary compatible with EventFilter.from_dict()
        """
        return self._settings.to_event_filter_dict()
    
    def set_from_event_filter_dict(self, filter_dict: Dict[str, Any]):
        """
        Set all settings from EventFilter dict format.
        
        Args:
            filter_dict: Dictionary from EventFilter.to_dict()
        """
        new_settings = EditorEventFilterSettings.from_event_filter_dict(filter_dict)
        
        # Update all fields (triggers saves)
        self.event_filter_enabled = new_settings.event_filter_enabled
        self.enabled_classifications = new_settings.enabled_classifications
        self.excluded_classifications = new_settings.excluded_classifications
        self.min_time = new_settings.min_time
        self.max_time = new_settings.max_time
        self.min_duration = new_settings.min_duration
        self.max_duration = new_settings.max_duration
        self.metadata_filters = new_settings.metadata_filters
        
        # Force immediate save (bypass debounce for dialog apply)
        self.force_save()


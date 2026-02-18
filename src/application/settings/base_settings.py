"""
Base Settings Manager

Provides a standardized foundation for all settings managers in the application.
All settings managers should inherit from BaseSettingsManager.

Features:
- Dataclass-based schema with type safety
- Automatic persistence via PreferencesRepository
- Auto-save on change with debouncing
- Signal emission for UI reactivity
- Backwards-compatible loading (handles missing fields)
- Namespaced storage keys
- Field validation with ValidationResult

Usage:
    1. Create a dataclass for your settings schema
    2. Inherit from BaseSettingsManager
    3. Implement required properties and methods
    4. Add validation metadata to fields (optional)
    
See: AgentAssets/SETTINGS_STANDARD.md for complete guide
"""
from dataclasses import dataclass, asdict, fields, field
from typing import Optional, Dict, Any, Type, List, Callable, Union, TYPE_CHECKING
from enum import Enum
import re
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

if TYPE_CHECKING:
    from src.infrastructure.persistence.sqlite.preferences_repository_impl import PreferencesRepository


# =============================================================================
# Validation Framework
# =============================================================================

@dataclass
class ValidationResult:
    """
    Result of validating settings.
    
    Attributes:
        valid: True if all validations passed
        errors: List of error messages (validation failures)
        warnings: List of warning messages (non-blocking issues)
    
    Example:
        result = settings.validate()
        if not result.valid:
            for error in result.errors:
                print(f"Error: {error}")
    """
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str):
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str):
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another ValidationResult into this one."""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
    
    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.valid


@dataclass
class FieldValidator:
    """
    Validation rules for a settings field.
    
    Use in field metadata to define validation rules:
    
    Example:
        @dataclass
        class MySettings(BaseSettings):
            volume: int = field(default=50, metadata={
                'validator': FieldValidator(min_value=0, max_value=100)
            })
            theme: str = field(default='dark', metadata={
                'validator': FieldValidator(choices=['dark', 'light', 'system'])
            })
            email: str = field(default='', metadata={
                'validator': FieldValidator(pattern=r'^[\\w.-]+@[\\w.-]+\\.\\w+$')
            })
    """
    # Range validation (for numbers)
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    
    # Choice validation (for enums/strings)
    choices: Optional[List[Any]] = None
    
    # Pattern validation (for strings)
    pattern: Optional[str] = None
    pattern_message: Optional[str] = None  # Custom error message for pattern
    
    # Length validation (for strings/lists)
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    # Required validation
    required: bool = False  # If True, value cannot be None or empty
    
    # Custom validation function
    # Signature: (value, field_name) -> Optional[str] (returns error message or None)
    custom: Optional[Callable[[Any, str], Optional[str]]] = None
    
    # Whether to allow None values
    allow_none: bool = True
    
    def validate(self, value: Any, field_name: str) -> ValidationResult:
        """
        Validate a value against this validator's rules.
        
        Args:
            value: The value to validate
            field_name: Name of the field (for error messages)
            
        Returns:
            ValidationResult with any errors/warnings
        """
        result = ValidationResult()
        
        # Handle None values
        if value is None:
            if not self.allow_none:
                result.add_error(f"{field_name}: Cannot be None")
            elif self.required:
                result.add_error(f"{field_name}: Required field cannot be empty")
            return result
        
        # Required check (for empty strings/lists)
        if self.required:
            if isinstance(value, str) and not value.strip():
                result.add_error(f"{field_name}: Required field cannot be empty")
                return result
            if isinstance(value, (list, dict)) and len(value) == 0:
                result.add_error(f"{field_name}: Required field cannot be empty")
                return result
        
        # Range validation
        if self.min_value is not None and isinstance(value, (int, float)):
            if value < self.min_value:
                result.add_error(f"{field_name}: Value {value} is below minimum {self.min_value}")
        
        if self.max_value is not None and isinstance(value, (int, float)):
            if value > self.max_value:
                result.add_error(f"{field_name}: Value {value} is above maximum {self.max_value}")
        
        # Choices validation
        if self.choices is not None:
            # Handle Enum values
            check_value = value.value if isinstance(value, Enum) else value
            valid_choices = [c.value if isinstance(c, Enum) else c for c in self.choices]
            if check_value not in valid_choices:
                result.add_error(f"{field_name}: Value '{value}' not in allowed choices: {self.choices}")
        
        # Pattern validation
        if self.pattern is not None and isinstance(value, str):
            if not re.match(self.pattern, value):
                msg = self.pattern_message or f"Value does not match required pattern"
                result.add_error(f"{field_name}: {msg}")
        
        # Length validation
        if hasattr(value, '__len__'):
            if self.min_length is not None and len(value) < self.min_length:
                result.add_error(f"{field_name}: Length {len(value)} is below minimum {self.min_length}")
            if self.max_length is not None and len(value) > self.max_length:
                result.add_error(f"{field_name}: Length {len(value)} is above maximum {self.max_length}")
        
        # Custom validation
        if self.custom is not None:
            error = self.custom(value, field_name)
            if error:
                result.add_error(error)
        
        return result


def validated_field(
    default: Any = None,
    *,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    choices: Optional[List[Any]] = None,
    pattern: Optional[str] = None,
    pattern_message: Optional[str] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    required: bool = False,
    allow_none: bool = True,
    custom: Optional[Callable[[Any, str], Optional[str]]] = None,
    **kwargs
):
    """
    Create a dataclass field with validation metadata.
    
    This is a convenience function that wraps dataclasses.field() with
    a FieldValidator in the metadata.
    
    Example:
        @dataclass
        class MySettings(BaseSettings):
            volume: int = validated_field(50, min_value=0, max_value=100)
            theme: str = validated_field('dark', choices=['dark', 'light'])
            name: str = validated_field('', required=True, min_length=1)
    
    Args:
        default: Default value for the field
        min_value: Minimum allowed value (for numbers)
        max_value: Maximum allowed value (for numbers)
        choices: List of allowed values
        pattern: Regex pattern to match (for strings)
        pattern_message: Custom error message for pattern validation
        min_length: Minimum length (for strings/lists)
        max_length: Maximum length (for strings/lists)
        required: If True, value cannot be None or empty
        allow_none: If False, None values are not allowed
        custom: Custom validation function
        **kwargs: Additional arguments passed to dataclasses.field()
    
    Returns:
        A dataclass field with validation metadata
    """
    validator = FieldValidator(
        min_value=min_value,
        max_value=max_value,
        choices=choices,
        pattern=pattern,
        pattern_message=pattern_message,
        min_length=min_length,
        max_length=max_length,
        required=required,
        allow_none=allow_none,
        custom=custom,
    )
    
    metadata = kwargs.pop('metadata', {})
    metadata['validator'] = validator
    
    return field(default=default, metadata=metadata, **kwargs)


@dataclass
class BaseSettings:
    """
    Base class for all settings dataclasses.
    
    Subclasses should define fields with default values for backwards compatibility.
    Fields can optionally include validation using validated_field() or FieldValidator.
    
    Example:
        @dataclass
        class MySettings(BaseSettings):
            theme: str = "dark"
            font_size: int = 12
            
        # With validation:
        @dataclass
        class MyValidatedSettings(BaseSettings):
            volume: int = validated_field(50, min_value=0, max_value=100)
            theme: str = validated_field('dark', choices=['dark', 'light', 'system'])
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseSettings':
        """
        Create settings from dictionary.
        
        Handles missing keys by using defaults - ensures backwards compatibility
        when new settings are added.
        """
        # Get default instance
        defaults = cls()
        
        # Only use keys that exist in the dataclass
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        # Merge with defaults
        merged = asdict(defaults)
        merged.update(filtered_data)
        
        return cls(**merged)
    
    def validate(self) -> ValidationResult:
        """
        Validate all settings fields against their validators.
        
        This method checks each field that has a 'validator' in its metadata
        and returns a ValidationResult with any errors or warnings.
        
        Fields without validators are skipped (assumed valid).
        
        Returns:
            ValidationResult with valid=True if all validations pass,
            otherwise valid=False with error messages.
        
        Example:
            settings = MySettings(volume=150)  # Invalid: above max
            result = settings.validate()
            if not result.valid:
                for error in result.errors:
                    print(f"Validation error: {error}")
        """
        result = ValidationResult()
        
        for f in fields(self):
            value = getattr(self, f.name)
            
            # Check for validator in field metadata
            validator = f.metadata.get('validator') if f.metadata else None
            
            if validator is not None:
                if isinstance(validator, FieldValidator):
                    field_result = validator.validate(value, f.name)
                    result.merge(field_result)
            else:
                # Basic type checking for fields without explicit validators
                # Skip None values (they're allowed by default)
                if value is not None:
                    # Check if value matches the field's type annotation
                    # This is a simple check - doesn't handle complex types
                    expected_type = f.type
                    if expected_type and not isinstance(expected_type, str):
                        # Handle Optional types
                        origin = getattr(expected_type, '__origin__', None)
                        if origin is Union:
                            # For Optional[X] (Union[X, None]), get the non-None type
                            args = getattr(expected_type, '__args__', ())
                            expected_types = tuple(t for t in args if t is not type(None))
                            if expected_types and not isinstance(value, expected_types):
                                result.add_warning(
                                    f"{f.name}: Expected type {expected_types}, got {type(value).__name__}"
                                )
        
        return result
    
    def validate_field(self, field_name: str) -> ValidationResult:
        """
        Validate a single field by name.
        
        Args:
            field_name: Name of the field to validate
            
        Returns:
            ValidationResult for just that field
        
        Raises:
            AttributeError: If field doesn't exist
        """
        value = getattr(self, field_name)
        
        for f in fields(self):
            if f.name == field_name:
                validator = f.metadata.get('validator') if f.metadata else None
                if validator is not None and isinstance(validator, FieldValidator):
                    return validator.validate(value, field_name)
                return ValidationResult()  # No validator = valid
        
        raise AttributeError(f"Field '{field_name}' not found in {self.__class__.__name__}")
    
    def is_valid(self) -> bool:
        """
        Quick check if settings are valid.
        
        Returns:
            True if all validations pass, False otherwise.
        """
        return self.validate().valid
    
    @classmethod
    def get_field_validators(cls) -> Dict[str, FieldValidator]:
        """
        Get all field validators defined for this settings class.
        
        Returns:
            Dictionary mapping field names to their validators.
            Only includes fields that have validators.
        """
        validators = {}
        # Create a temporary instance to access fields
        for f in fields(cls):
            validator = f.metadata.get('validator') if f.metadata else None
            if validator is not None and isinstance(validator, FieldValidator):
                validators[f.name] = validator
        return validators


class BaseSettingsManager(QObject):
    """
    Abstract base class for all settings managers.
    
    Provides:
    - Automatic persistence to PreferencesRepository
    - Auto-save with configurable debouncing
    - Signal emission when settings change
    - Type-safe property access pattern
    - Validation support via BaseSettings.validate()
    
    Subclasses must:
    1. Define NAMESPACE class attribute
    2. Define SETTINGS_CLASS class attribute
    3. Implement property accessors for type-safe access
    
    Example:
        class MySettingsManager(BaseSettingsManager[MySettings]):
            NAMESPACE = "my_component"
            SETTINGS_CLASS = MySettings
            
            @property
            def theme(self) -> str:
                return self._settings.theme
            
            @theme.setter
            def theme(self, value: str):
                if value != self._settings.theme:
                    self._settings.theme = value
                    self._save_setting('theme')
    """
    
    # Signals
    settings_changed = pyqtSignal(str)  # Setting name that changed
    settings_loaded = pyqtSignal()
    validation_failed = pyqtSignal(object)  # ValidationResult when validation fails
    settings_save_failed = pyqtSignal(str)  # Error message
    
    # Must be defined by subclasses
    NAMESPACE: str = ""  # e.g., "timeline", "app", "block.editor"
    SETTINGS_CLASS: Type[BaseSettings] = BaseSettings
    
    # Configuration
    SAVE_DEBOUNCE_MS: int = 300  # Debounce delay for saves
    
    def __init__(self, preferences_repo: Optional['PreferencesRepository'] = None, parent=None):
        """
        Initialize the settings manager.
        
        Args:
            preferences_repo: Repository for persistence (if None, settings are in-memory only)
            parent: Parent QObject
        """
        super().__init__(parent)
        
        if not self.NAMESPACE:
            raise ValueError(f"{self.__class__.__name__} must define NAMESPACE")
        
        self._preferences_repo = preferences_repo
        self._settings: BaseSettings = self.SETTINGS_CLASS()
        self._loaded = False
        
        # Debounce timer for saves
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(self.SAVE_DEBOUNCE_MS)
        self._save_timer.timeout.connect(self._do_save)
        self._pending_save = False
        
        # Load settings from storage
        self._load_from_storage()
    
    # =========================================================================
    # Storage Key
    # =========================================================================
    
    @property
    def _storage_key(self) -> str:
        """Get the storage key for this settings namespace."""
        return f"{self.NAMESPACE}.settings"
    
    # =========================================================================
    # Generic Access
    # =========================================================================
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key."""
        return getattr(self._settings, key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a setting value by key.
        
        Returns True if the setting was changed.
        """
        if hasattr(self._settings, key):
            old_value = getattr(self._settings, key)
            if old_value != value:
                setattr(self._settings, key, value)
                self._save_setting(key)
                return True
        return False
    
    def get_all(self) -> Dict[str, Any]:
        """Get all settings as a dictionary."""
        return self._settings.to_dict()
    
    def reset_to_defaults(self):
        """Reset all settings to their default values."""
        self._settings = self.SETTINGS_CLASS()
        self._do_save()
        self.settings_loaded.emit()
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def _load_from_storage(self):
        """Load settings from the preferences repository."""
        if not self._preferences_repo:
            self._loaded = True
            return
        
        try:
            stored_data = self._preferences_repo.get(self._storage_key, {})
            
            if stored_data and isinstance(stored_data, dict):
                self._settings = self.SETTINGS_CLASS.from_dict(stored_data)
            
            self._loaded = True
            self.settings_loaded.emit()
            
        except Exception as e:
            print(f"{self.__class__.__name__}: Failed to load settings: {e}")
            self._loaded = True
    
    def _save_setting(self, key: str):
        """Queue a save operation (debounced)."""
        self._pending_save = True
        self._save_timer.start()
        self.settings_changed.emit(key)
    
    def _do_save(self):
        """Actually persist settings to storage."""
        if not self._preferences_repo:
            self._pending_save = False
            return
        
        try:
            self._preferences_repo.set(self._storage_key, self._settings.to_dict())
            self._pending_save = False
        except Exception as e:
            print(f"{self.__class__.__name__}: Failed to save settings: {e}")
            self.settings_save_failed.emit(str(e))
    
    def force_save(self):
        """Force an immediate save (bypasses debounce)."""
        self._save_timer.stop()
        self._do_save()
    
    # =========================================================================
    # Status
    # =========================================================================
    
    def is_loaded(self) -> bool:
        """Check if settings have been loaded from storage."""
        return self._loaded
    
    def has_pending_save(self) -> bool:
        """Check if there's a pending save operation."""
        return self._pending_save

    # =========================================================================
    # Validation
    # =========================================================================
    
    def validate(self) -> ValidationResult:
        """
        Validate current settings.
        
        Returns:
            ValidationResult with any errors or warnings.
        """
        return self._settings.validate()
    
    def validate_field(self, field_name: str) -> ValidationResult:
        """
        Validate a single field.
        
        Args:
            field_name: Name of the field to validate
            
        Returns:
            ValidationResult for that field.
        """
        return self._settings.validate_field(field_name)
    
    def is_valid(self) -> bool:
        """
        Quick check if current settings are valid.
        
        Returns:
            True if all validations pass.
        """
        return self._settings.is_valid()
    
    def set_validated(self, key: str, value: Any) -> ValidationResult:
        """
        Set a setting value with validation.
        
        Unlike set(), this validates the new value before saving.
        If validation fails, the value is NOT saved.
        
        Args:
            key: Setting name
            value: New value
            
        Returns:
            ValidationResult - check result.valid to see if it was saved
        """
        if not hasattr(self._settings, key):
            result = ValidationResult()
            result.add_error(f"Unknown setting: {key}")
            return result
        
        # Temporarily set the value to validate
        old_value = getattr(self._settings, key)
        setattr(self._settings, key, value)
        
        # Validate just this field
        result = self._settings.validate_field(key)
        
        if result.valid:
            # Value is valid - save it
            if old_value != value:
                self._save_setting(key)
        else:
            # Validation failed - revert
            setattr(self._settings, key, old_value)
            self.validation_failed.emit(result)
        
        return result
    
    def get_validation_errors(self) -> List[str]:
        """
        Get list of current validation errors.
        
        Convenience method that returns just the error messages.
        
        Returns:
            List of error message strings (empty if valid)
        """
        return self.validate().errors









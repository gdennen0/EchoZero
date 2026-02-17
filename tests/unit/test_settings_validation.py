"""
Tests for the settings validation framework.

Tests the ValidationResult, FieldValidator, and BaseSettings.validate() functionality.
"""
import pytest
from dataclasses import dataclass

from src.application.settings.base_settings import (
    BaseSettings,
    ValidationResult,
    FieldValidator,
    validated_field,
)


class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_initial_state_is_valid(self):
        """New ValidationResult should be valid."""
        result = ValidationResult()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
    
    def test_add_error_marks_invalid(self):
        """Adding an error should mark result as invalid."""
        result = ValidationResult()
        result.add_error("Test error")
        assert result.valid is False
        assert "Test error" in result.errors
    
    def test_add_warning_keeps_valid(self):
        """Adding a warning should not affect validity."""
        result = ValidationResult()
        result.add_warning("Test warning")
        assert result.valid is True
        assert "Test warning" in result.warnings
    
    def test_merge_combines_results(self):
        """Merging results should combine errors and warnings."""
        result1 = ValidationResult()
        result1.add_error("Error 1")
        
        result2 = ValidationResult()
        result2.add_warning("Warning 1")
        
        result1.merge(result2)
        assert result1.valid is False  # Still invalid from error
        assert "Error 1" in result1.errors
        assert "Warning 1" in result1.warnings
    
    def test_merge_invalid_makes_target_invalid(self):
        """Merging an invalid result into a valid one should make it invalid."""
        result1 = ValidationResult()  # Valid
        result2 = ValidationResult()
        result2.add_error("Error")  # Invalid
        
        result1.merge(result2)
        assert result1.valid is False
    
    def test_bool_conversion(self):
        """ValidationResult should be usable in boolean context."""
        valid_result = ValidationResult()
        invalid_result = ValidationResult()
        invalid_result.add_error("Error")
        
        assert bool(valid_result) is True
        assert bool(invalid_result) is False
        
        # Can use in if statements
        if valid_result:
            pass  # Should reach here
        else:
            pytest.fail("Valid result should be truthy")


class TestFieldValidator:
    """Tests for FieldValidator class."""
    
    def test_min_value_validation(self):
        """Test minimum value validation."""
        validator = FieldValidator(min_value=0)
        
        result = validator.validate(10, "test_field")
        assert result.valid is True
        
        result = validator.validate(-1, "test_field")
        assert result.valid is False
        assert "below minimum" in result.errors[0]
    
    def test_max_value_validation(self):
        """Test maximum value validation."""
        validator = FieldValidator(max_value=100)
        
        result = validator.validate(50, "test_field")
        assert result.valid is True
        
        result = validator.validate(101, "test_field")
        assert result.valid is False
        assert "above maximum" in result.errors[0]
    
    def test_range_validation(self):
        """Test combined min/max validation."""
        validator = FieldValidator(min_value=0, max_value=100)
        
        result = validator.validate(50, "test_field")
        assert result.valid is True
        
        result = validator.validate(-1, "test_field")
        assert result.valid is False
        
        result = validator.validate(101, "test_field")
        assert result.valid is False
    
    def test_choices_validation(self):
        """Test choices validation."""
        validator = FieldValidator(choices=["dark", "light", "system"])
        
        result = validator.validate("dark", "theme")
        assert result.valid is True
        
        result = validator.validate("invalid", "theme")
        assert result.valid is False
        assert "not in allowed choices" in result.errors[0]
    
    def test_pattern_validation(self):
        """Test regex pattern validation."""
        validator = FieldValidator(pattern=r"^\d{3}-\d{4}$")
        
        result = validator.validate("123-4567", "phone")
        assert result.valid is True
        
        result = validator.validate("invalid", "phone")
        assert result.valid is False
    
    def test_pattern_custom_message(self):
        """Test custom pattern error message."""
        validator = FieldValidator(
            pattern=r"^\d{3}-\d{4}$",
            pattern_message="Must be in format XXX-XXXX"
        )
        
        result = validator.validate("invalid", "phone")
        assert result.valid is False
        assert "Must be in format XXX-XXXX" in result.errors[0]
    
    def test_min_length_validation(self):
        """Test minimum length validation."""
        validator = FieldValidator(min_length=3)
        
        result = validator.validate("abc", "name")
        assert result.valid is True
        
        result = validator.validate("ab", "name")
        assert result.valid is False
        assert "below minimum" in result.errors[0]
    
    def test_max_length_validation(self):
        """Test maximum length validation."""
        validator = FieldValidator(max_length=10)
        
        result = validator.validate("short", "name")
        assert result.valid is True
        
        result = validator.validate("this is too long", "name")
        assert result.valid is False
        assert "above maximum" in result.errors[0]
    
    def test_required_validation(self):
        """Test required field validation."""
        validator = FieldValidator(required=True)
        
        result = validator.validate("value", "name")
        assert result.valid is True
        
        result = validator.validate("", "name")
        assert result.valid is False
        assert "Required field" in result.errors[0]
        
        result = validator.validate(None, "name")
        assert result.valid is False
    
    def test_allow_none_validation(self):
        """Test allow_none validation."""
        validator = FieldValidator(allow_none=False)
        
        result = validator.validate("value", "name")
        assert result.valid is True
        
        result = validator.validate(None, "name")
        assert result.valid is False
        assert "Cannot be None" in result.errors[0]
    
    def test_custom_validation(self):
        """Test custom validation function."""
        def even_only(value, field_name):
            if value % 2 != 0:
                return f"{field_name}: Must be an even number"
            return None
        
        validator = FieldValidator(custom=even_only)
        
        result = validator.validate(4, "count")
        assert result.valid is True
        
        result = validator.validate(3, "count")
        assert result.valid is False
        assert "Must be an even number" in result.errors[0]


class TestValidatedField:
    """Tests for validated_field() helper function."""
    
    def test_creates_field_with_validator(self):
        """validated_field should create a field with FieldValidator in metadata."""
        @dataclass
        class TestSettings(BaseSettings):
            volume: int = validated_field(50, min_value=0, max_value=100)
        
        settings = TestSettings()
        validators = settings.get_field_validators()
        
        assert "volume" in validators
        assert validators["volume"].min_value == 0
        assert validators["volume"].max_value == 100
    
    def test_default_value_works(self):
        """validated_field should set the default value correctly."""
        @dataclass
        class TestSettings(BaseSettings):
            theme: str = validated_field("dark", choices=["dark", "light"])
        
        settings = TestSettings()
        assert settings.theme == "dark"


class TestBaseSettingsValidation:
    """Tests for BaseSettings.validate() method."""
    
    def test_validate_with_no_validators(self):
        """Settings without validators should always be valid."""
        @dataclass
        class SimpleSettings(BaseSettings):
            name: str = "default"
            count: int = 0
        
        settings = SimpleSettings(name="test", count=10)
        result = settings.validate()
        assert result.valid is True
    
    def test_validate_with_valid_values(self):
        """Valid values should pass validation."""
        @dataclass
        class ValidatedSettings(BaseSettings):
            volume: int = validated_field(50, min_value=0, max_value=100)
            theme: str = validated_field("dark", choices=["dark", "light"])
        
        settings = ValidatedSettings(volume=75, theme="light")
        result = settings.validate()
        assert result.valid is True
    
    def test_validate_with_invalid_values(self):
        """Invalid values should fail validation."""
        @dataclass
        class ValidatedSettings(BaseSettings):
            volume: int = validated_field(50, min_value=0, max_value=100)
        
        settings = ValidatedSettings(volume=150)  # Above max
        result = settings.validate()
        assert result.valid is False
        assert len(result.errors) == 1
        assert "volume" in result.errors[0]
    
    def test_validate_multiple_errors(self):
        """Multiple validation errors should all be reported."""
        @dataclass
        class ValidatedSettings(BaseSettings):
            volume: int = validated_field(50, min_value=0, max_value=100)
            name: str = validated_field("", required=True, min_length=1)
        
        settings = ValidatedSettings(volume=150, name="")  # Both invalid
        result = settings.validate()
        assert result.valid is False
        assert len(result.errors) == 2
    
    def test_validate_field_single(self):
        """validate_field should validate just one field."""
        @dataclass
        class ValidatedSettings(BaseSettings):
            volume: int = validated_field(50, min_value=0, max_value=100)
            name: str = validated_field("", required=True)
        
        settings = ValidatedSettings(volume=150, name="")  # Both invalid
        
        # Only validate volume
        result = settings.validate_field("volume")
        assert result.valid is False
        assert len(result.errors) == 1
        assert "volume" in result.errors[0]
    
    def test_is_valid_shorthand(self):
        """is_valid() should return boolean result."""
        @dataclass
        class ValidatedSettings(BaseSettings):
            volume: int = validated_field(50, min_value=0, max_value=100)
        
        valid_settings = ValidatedSettings(volume=50)
        invalid_settings = ValidatedSettings(volume=150)
        
        assert valid_settings.is_valid() is True
        assert invalid_settings.is_valid() is False
    
    def test_get_field_validators(self):
        """get_field_validators should return all validators."""
        @dataclass
        class ValidatedSettings(BaseSettings):
            volume: int = validated_field(50, min_value=0, max_value=100)
            theme: str = validated_field("dark", choices=["dark", "light"])
            name: str = "default"  # No validator
        
        validators = ValidatedSettings.get_field_validators()
        
        assert "volume" in validators
        assert "theme" in validators
        assert "name" not in validators  # No validator defined
    
    def test_from_dict_preserves_validation(self):
        """Settings created from dict should still be validated."""
        @dataclass
        class ValidatedSettings(BaseSettings):
            volume: int = validated_field(50, min_value=0, max_value=100)
        
        # Create from dict with invalid value
        settings = ValidatedSettings.from_dict({"volume": 150})
        
        # Value is loaded (no validation on load)
        assert settings.volume == 150
        
        # But validation catches it
        result = settings.validate()
        assert result.valid is False


class TestIntegration:
    """Integration tests for real-world usage patterns."""
    
    def test_settings_with_mixed_validation(self):
        """Test settings with various validation types."""
        @dataclass
        class AppSettings(BaseSettings):
            # Validated fields
            volume: int = validated_field(50, min_value=0, max_value=100)
            theme: str = validated_field("dark", choices=["dark", "light", "system"])
            username: str = validated_field("", min_length=3, max_length=20)
            
            # Unvalidated fields (backwards compatible)
            last_opened: str = ""
            window_x: int = 0
            window_y: int = 0
        
        # Valid settings
        settings = AppSettings(
            volume=75,
            theme="light",
            username="john",
            last_opened="/path/to/file",
            window_x=100,
            window_y=200,
        )
        assert settings.is_valid() is True
        
        # Invalid volume
        settings.volume = 150
        result = settings.validate()
        assert result.valid is False
        assert len(result.errors) == 1
        
        # Fix volume, break username
        settings.volume = 50
        settings.username = "ab"  # Too short
        result = settings.validate()
        assert result.valid is False
        assert "username" in result.errors[0]

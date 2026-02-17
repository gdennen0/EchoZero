"""
Tests for the Validation Framework.

Tests validators, composition, and convenience functions.
"""
import pytest

from src.shared.application.validation.validation_framework import (
    ValidationResult,
    ValidationError,
    Validator,
    RequiredValidator,
    RangeValidator,
    PatternValidator,
    LengthValidator,
    ChoicesValidator,
    TypeValidator,
    CustomValidator,
    All,
    Any,
    validate,
    validate_field,
)


# =============================================================================
# ValidationResult Tests
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_default_is_valid(self):
        """Test that default result is valid."""
        result = ValidationResult()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
    
    def test_add_error(self):
        """Test adding an error."""
        result = ValidationResult()
        result.add_error("something went wrong")
        assert result.valid is False
        assert "something went wrong" in result.errors
    
    def test_add_error_with_field_name(self):
        """Test adding error includes field name."""
        result = ValidationResult(field_name="username")
        result.add_error("is required")
        assert "username: is required" in result.errors
    
    def test_add_warning(self):
        """Test adding a warning doesn't affect validity."""
        result = ValidationResult()
        result.add_warning("this might be an issue")
        assert result.valid is True
        assert "this might be an issue" in result.warnings
    
    def test_merge(self):
        """Test merging results."""
        result1 = ValidationResult()
        result1.add_error("error 1")
        
        result2 = ValidationResult()
        result2.add_error("error 2")
        result2.add_warning("warning 1")
        
        result1.merge(result2)
        assert result1.valid is False
        assert "error 1" in result1.errors
        assert "error 2" in result1.errors
        assert "warning 1" in result1.warnings
    
    def test_bool_conversion(self):
        """Test boolean conversion."""
        valid_result = ValidationResult()
        assert bool(valid_result) is True
        
        invalid_result = ValidationResult()
        invalid_result.add_error("error")
        assert bool(invalid_result) is False
    
    def test_success_factory(self):
        """Test success factory method."""
        result = ValidationResult.success()
        assert result.valid is True
    
    def test_failure_factory(self):
        """Test failure factory method."""
        result = ValidationResult.failure("error message", "field")
        assert result.valid is False
        assert "field: error message" in result.errors
    
    def test_raise_if_invalid(self):
        """Test raise_if_invalid raises on invalid result."""
        result = ValidationResult.failure("error", "field")
        with pytest.raises(ValidationError):
            result.raise_if_invalid()
    
    def test_raise_if_invalid_on_valid(self):
        """Test raise_if_invalid doesn't raise on valid result."""
        result = ValidationResult.success()
        result.raise_if_invalid()  # Should not raise


# =============================================================================
# RequiredValidator Tests
# =============================================================================

class TestRequiredValidator:
    """Tests for RequiredValidator."""
    
    def test_none_is_invalid(self):
        """Test None fails validation."""
        validator = RequiredValidator()
        result = validator.validate(None, "field")
        assert not result.valid
    
    def test_empty_string_is_invalid(self):
        """Test empty string fails validation."""
        validator = RequiredValidator()
        result = validator.validate("", "field")
        assert not result.valid
        
        result = validator.validate("   ", "field")
        assert not result.valid
    
    def test_empty_list_is_invalid(self):
        """Test empty list fails validation."""
        validator = RequiredValidator()
        result = validator.validate([], "field")
        assert not result.valid
    
    def test_empty_dict_is_invalid(self):
        """Test empty dict fails validation."""
        validator = RequiredValidator()
        result = validator.validate({}, "field")
        assert not result.valid
    
    def test_value_is_valid(self):
        """Test non-empty value passes validation."""
        validator = RequiredValidator()
        
        assert validator.validate("hello", "field").valid
        assert validator.validate(0, "field").valid
        assert validator.validate([1, 2], "field").valid
        assert validator.validate({"a": 1}, "field").valid
    
    def test_custom_message(self):
        """Test custom error message."""
        validator = RequiredValidator(message="cannot be blank")
        result = validator.validate(None, "field")
        assert "cannot be blank" in result.errors[0]


# =============================================================================
# RangeValidator Tests
# =============================================================================

class TestRangeValidator:
    """Tests for RangeValidator."""
    
    def test_within_range(self):
        """Test value within range is valid."""
        validator = RangeValidator(min_value=0, max_value=100)
        assert validator.validate(50, "field").valid
        assert validator.validate(0, "field").valid
        assert validator.validate(100, "field").valid
    
    def test_below_min(self):
        """Test value below min is invalid."""
        validator = RangeValidator(min_value=0)
        result = validator.validate(-1, "field")
        assert not result.valid
        assert "at least 0" in result.errors[0]
    
    def test_above_max(self):
        """Test value above max is invalid."""
        validator = RangeValidator(max_value=100)
        result = validator.validate(101, "field")
        assert not result.valid
        assert "at most 100" in result.errors[0]
    
    def test_none_is_valid(self):
        """Test None passes (handled by RequiredValidator)."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate(None, "field")
        assert result.valid
    
    def test_non_numeric(self):
        """Test non-numeric value fails."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate("not a number", "field")
        assert not result.valid
    
    def test_float_values(self):
        """Test float values work."""
        validator = RangeValidator(min_value=0.0, max_value=1.0)
        assert validator.validate(0.5, "field").valid
        assert not validator.validate(1.5, "field").valid


# =============================================================================
# PatternValidator Tests
# =============================================================================

class TestPatternValidator:
    """Tests for PatternValidator."""
    
    def test_matching_pattern(self):
        """Test matching pattern is valid."""
        validator = PatternValidator(r'^[a-z]+$')
        assert validator.validate("hello", "field").valid
    
    def test_non_matching_pattern(self):
        """Test non-matching pattern is invalid."""
        validator = PatternValidator(r'^[a-z]+$')
        result = validator.validate("Hello123", "field")
        assert not result.valid
    
    def test_custom_message(self):
        """Test custom error message."""
        validator = PatternValidator(r'^[a-z]+$', "must be lowercase letters")
        result = validator.validate("ABC", "field")
        assert "must be lowercase letters" in result.errors[0]
    
    def test_none_is_valid(self):
        """Test None passes (handled by RequiredValidator)."""
        validator = PatternValidator(r'^[a-z]+$')
        result = validator.validate(None, "field")
        assert result.valid
    
    def test_non_string(self):
        """Test non-string value fails."""
        validator = PatternValidator(r'^[a-z]+$')
        result = validator.validate(123, "field")
        assert not result.valid


# =============================================================================
# LengthValidator Tests
# =============================================================================

class TestLengthValidator:
    """Tests for LengthValidator."""
    
    def test_within_length(self):
        """Test value within length is valid."""
        validator = LengthValidator(min_length=3, max_length=10)
        assert validator.validate("hello", "field").valid
    
    def test_below_min_length(self):
        """Test value below min length is invalid."""
        validator = LengthValidator(min_length=3)
        result = validator.validate("ab", "field")
        assert not result.valid
        assert "at least 3" in result.errors[0]
    
    def test_above_max_length(self):
        """Test value above max length is invalid."""
        validator = LengthValidator(max_length=5)
        result = validator.validate("toolong", "field")
        assert not result.valid
        assert "at most 5" in result.errors[0]
    
    def test_list_length(self):
        """Test list length validation."""
        validator = LengthValidator(min_length=2, max_length=5)
        assert validator.validate([1, 2, 3], "field").valid
        assert not validator.validate([1], "field").valid
    
    def test_none_is_valid(self):
        """Test None passes (handled by RequiredValidator)."""
        validator = LengthValidator(min_length=3)
        result = validator.validate(None, "field")
        assert result.valid


# =============================================================================
# ChoicesValidator Tests
# =============================================================================

class TestChoicesValidator:
    """Tests for ChoicesValidator."""
    
    def test_valid_choice(self):
        """Test valid choice is valid."""
        validator = ChoicesValidator(["dark", "light", "system"])
        assert validator.validate("dark", "field").valid
    
    def test_invalid_choice(self):
        """Test invalid choice is invalid."""
        validator = ChoicesValidator(["dark", "light", "system"])
        result = validator.validate("blue", "field")
        assert not result.valid
        assert "must be one of" in result.errors[0]
    
    def test_none_is_valid(self):
        """Test None passes (handled by RequiredValidator)."""
        validator = ChoicesValidator(["dark", "light"])
        result = validator.validate(None, "field")
        assert result.valid
    
    def test_integer_choices(self):
        """Test integer choices work."""
        validator = ChoicesValidator([1, 2, 3])
        assert validator.validate(2, "field").valid
        assert not validator.validate(4, "field").valid


# =============================================================================
# TypeValidator Tests
# =============================================================================

class TestTypeValidator:
    """Tests for TypeValidator."""
    
    def test_correct_type(self):
        """Test correct type is valid."""
        validator = TypeValidator(int)
        assert validator.validate(42, "field").valid
    
    def test_wrong_type(self):
        """Test wrong type is invalid."""
        validator = TypeValidator(int)
        result = validator.validate("not an int", "field")
        assert not result.valid
        assert "must be int" in result.errors[0]
    
    def test_multiple_types(self):
        """Test multiple types work."""
        validator = TypeValidator((int, float))
        assert validator.validate(42, "field").valid
        assert validator.validate(3.14, "field").valid
        assert not validator.validate("string", "field").valid
    
    def test_none_is_valid(self):
        """Test None passes (handled by RequiredValidator)."""
        validator = TypeValidator(int)
        result = validator.validate(None, "field")
        assert result.valid


# =============================================================================
# CustomValidator Tests
# =============================================================================

class TestCustomValidator:
    """Tests for CustomValidator."""
    
    def test_passing_custom(self):
        """Test passing custom validation."""
        validator = CustomValidator(lambda v: v % 2 == 0, "must be even")
        assert validator.validate(4, "field").valid
    
    def test_failing_custom(self):
        """Test failing custom validation."""
        validator = CustomValidator(lambda v: v % 2 == 0, "must be even")
        result = validator.validate(3, "field")
        assert not result.valid
        assert "must be even" in result.errors[0]
    
    def test_exception_in_custom(self):
        """Test exception in custom function is caught."""
        def bad_validator(v):
            raise ValueError("oops")
        
        validator = CustomValidator(bad_validator, "custom failed")
        result = validator.validate("anything", "field")
        assert not result.valid
        assert "validation error" in result.errors[0]


# =============================================================================
# Composable Validators Tests
# =============================================================================

class TestAllValidator:
    """Tests for All validator."""
    
    def test_all_pass(self):
        """Test all validators pass."""
        validator = All(
            RequiredValidator(),
            LengthValidator(min_length=3),
        )
        result = validator.validate("hello", "field")
        assert result.valid
    
    def test_one_fails(self):
        """Test one validator fails."""
        validator = All(
            RequiredValidator(),
            LengthValidator(min_length=10),
        )
        result = validator.validate("hello", "field")
        assert not result.valid
    
    def test_multiple_fail(self):
        """Test multiple validators fail and all errors collected."""
        validator = All(
            LengthValidator(min_length=10),
            PatternValidator(r'^[0-9]+$', "must be digits"),
        )
        result = validator.validate("hi", "field")
        assert not result.valid
        assert len(result.errors) == 2
    
    def test_stop_on_first_error(self):
        """Test stop on first error mode."""
        validator = All(
            LengthValidator(min_length=10),
            PatternValidator(r'^[0-9]+$', "must be digits"),
            stop_on_first_error=True,
        )
        result = validator.validate("hi", "field")
        assert not result.valid
        assert len(result.errors) == 1


class TestAnyValidator:
    """Tests for Any validator."""
    
    def test_one_passes(self):
        """Test one validator passes is enough."""
        validator = Any(
            PatternValidator(r'^[0-9]+$', "must be digits"),
            PatternValidator(r'^[a-z]+$', "must be letters"),
        )
        result = validator.validate("123", "field")
        assert result.valid
    
    def test_all_fail(self):
        """Test all validators fail."""
        validator = Any(
            PatternValidator(r'^[0-9]+$', "must be digits"),
            PatternValidator(r'^[a-z]+$', "must be letters"),
        )
        result = validator.validate("ABC123", "field")
        assert not result.valid
    
    def test_custom_message(self):
        """Test custom error message."""
        validator = Any(
            PatternValidator(r'^[0-9]+$'),
            PatternValidator(r'^[a-z]+$'),
            message="must be digits or lowercase letters",
        )
        result = validator.validate("ABC", "field")
        assert "must be digits or lowercase letters" in result.errors[0]


# =============================================================================
# Convenience Functions Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_validate_with_single(self):
        """Test validate with single validator."""
        result = validate("hello", RequiredValidator(), "field")
        assert result.valid
    
    def test_validate_with_list(self):
        """Test validate with list of validators."""
        result = validate("hello", [
            RequiredValidator(),
            LengthValidator(min_length=3),
        ], "field")
        assert result.valid
    
    def test_validate_field(self):
        """Test validate_field convenience."""
        result = validate_field("username", "hello", [
            RequiredValidator(),
            LengthValidator(min_length=3),
        ])
        assert result.valid
        
        result = validate_field("username", "", RequiredValidator())
        assert not result.valid
        assert "username" in result.errors[0]


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for real-world usage patterns."""
    
    def test_form_validation(self):
        """Test typical form validation pattern."""
        # Simulate form data
        form_data = {
            "username": "johndoe",
            "email": "john@example.com",
            "age": 25,
            "theme": "dark",
        }
        
        # Define validators
        validators = {
            "username": [
                RequiredValidator(),
                LengthValidator(min_length=3, max_length=20),
                PatternValidator(r'^[a-z0-9_]+$', "must be lowercase alphanumeric"),
            ],
            "email": [
                RequiredValidator(),
                PatternValidator(r'^[\w.-]+@[\w.-]+\.\w+$', "must be valid email"),
            ],
            "age": [
                RangeValidator(min_value=18, max_value=120),
            ],
            "theme": [
                ChoicesValidator(["dark", "light", "system"]),
            ],
        }
        
        # Validate all fields
        all_valid = True
        for field_name, field_validators in validators.items():
            result = validate_field(field_name, form_data.get(field_name), field_validators)
            if not result.valid:
                all_valid = False
        
        assert all_valid
    
    def test_api_input_validation(self):
        """Test API input validation pattern."""
        api_data = {
            "limit": 50,
            "offset": 0,
            "sort_by": "name",
        }
        
        # Validate limit
        result = validate_field("limit", api_data["limit"], [
            TypeValidator(int),
            RangeValidator(min_value=1, max_value=100),
        ])
        assert result.valid
        
        # Validate sort_by
        result = validate_field("sort_by", api_data["sort_by"], [
            ChoicesValidator(["name", "date", "size"]),
        ])
        assert result.valid
    
    def test_nested_composition(self):
        """Test nested validator composition."""
        # Either a valid email OR "anonymous"
        validator = All(
            RequiredValidator(),
            Any(
                PatternValidator(r'^[\w.-]+@[\w.-]+\.\w+$'),
                CustomValidator(lambda v: v == "anonymous", "must be anonymous"),
            ),
        )
        
        assert validator.validate("test@example.com", "email").valid
        assert validator.validate("anonymous", "email").valid
        assert not validator.validate("invalid", "email").valid

"""
Validation Framework

Provides a unified, composable validation pattern for the application.
Use for forms, inputs, settings, API data, and any validation needs.

Usage:
    # Simple validation
    result = validate(user_input, [
        RequiredValidator(),
        LengthValidator(min_length=3, max_length=50),
        PatternValidator(r'^[a-zA-Z0-9_]+$', "alphanumeric only"),
    ])
    if not result.valid:
        show_errors(result.errors)
    
    # Field validation with context
    result = validate_field("username", username_value, [
        RequiredValidator(),
        LengthValidator(min_length=3),
    ])
    
    # Composable validators
    result = validate(data, All(
        RequiredValidator(),
        RangeValidator(0, 100),
    ))
    
    # Custom validators
    result = validate(data, CustomValidator(lambda v: v % 2 == 0, "must be even"))

Features:
- Composable validator pattern
- Common validators for typical use cases
- ValidationResult with errors and warnings
- Field context for meaningful error messages
- Short-circuit or collect-all modes
- Type-safe and extensible
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List, Callable, Union, Type, Set
import re


# =============================================================================
# Exceptions
# =============================================================================

class ValidationError(Exception):
    """
    Exception for validation failures.
    
    Can be raised when validation must fail immediately.
    """
    
    def __init__(self, message: str, field_name: Optional[str] = None):
        self.message = message
        self.field_name = field_name
        full_message = f"{field_name}: {message}" if field_name else message
        super().__init__(full_message)


# =============================================================================
# Validation Result
# =============================================================================

@dataclass
class ValidationResult:
    """
    Result of validation operations.
    
    Attributes:
        valid: True if all validations passed
        errors: List of error messages (validation failures)
        warnings: List of warning messages (non-blocking issues)
        field_name: Optional field name for context
    
    Usage:
        result = validator.validate(value)
        if not result.valid:
            for error in result.errors:
                print(f"Error: {error}")
        
        # Can also use in boolean context
        if result:
            print("Valid!")
    """
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_name: Optional[str] = None
    
    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        if self.field_name and not message.startswith(self.field_name):
            message = f"{self.field_name}: {message}"
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        if self.field_name and not message.startswith(self.field_name):
            message = f"{self.field_name}: {message}"
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another ValidationResult into this one."""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
    
    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.valid
    
    def raise_if_invalid(self) -> None:
        """Raise ValidationError if result is invalid."""
        if not self.valid:
            raise ValidationError("; ".join(self.errors), self.field_name)
    
    @classmethod
    def success(cls) -> 'ValidationResult':
        """Create a successful validation result."""
        return cls(valid=True)
    
    @classmethod
    def failure(cls, message: str, field_name: Optional[str] = None) -> 'ValidationResult':
        """Create a failed validation result."""
        result = cls(valid=False, field_name=field_name)
        result.errors.append(f"{field_name}: {message}" if field_name else message)
        return result


# =============================================================================
# Base Validator
# =============================================================================

class Validator(ABC):
    """
    Abstract base class for validators.
    
    Validators check values against rules and return ValidationResult.
    
    Subclass and implement validate() to create custom validators:
        
        class EvenNumberValidator(Validator):
            def validate(self, value: Any, field_name: str = "") -> ValidationResult:
                result = ValidationResult(field_name=field_name)
                if value % 2 != 0:
                    result.add_error("must be an even number")
                return result
    """
    
    @abstractmethod
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        """
        Validate a value.
        
        Args:
            value: The value to validate
            field_name: Optional field name for error messages
            
        Returns:
            ValidationResult with any errors/warnings
        """
        pass
    
    def __call__(self, value: Any, field_name: str = "") -> ValidationResult:
        """Allow validators to be called directly."""
        return self.validate(value, field_name)


# =============================================================================
# Common Validators
# =============================================================================

class RequiredValidator(Validator):
    """
    Validates that a value is not None/empty.
    
    Usage:
        validator = RequiredValidator()
        result = validator.validate(value, "username")
    """
    
    def __init__(self, message: str = "is required"):
        self.message = message
    
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        result = ValidationResult(field_name=field_name)
        
        if value is None:
            result.add_error(self.message)
        elif isinstance(value, str) and not value.strip():
            result.add_error(self.message)
        elif isinstance(value, (list, dict, set)) and len(value) == 0:
            result.add_error(self.message)
        
        return result


class RangeValidator(Validator):
    """
    Validates that a numeric value is within a range.
    
    Usage:
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate(50, "volume")
    """
    
    def __init__(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        message: Optional[str] = None,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.message = message
    
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        result = ValidationResult(field_name=field_name)
        
        if value is None:
            return result  # None is handled by RequiredValidator
        
        try:
            num_value = float(value)
        except (TypeError, ValueError):
            result.add_error(f"must be a number, got {type(value).__name__}")
            return result
        
        if self.min_value is not None and num_value < self.min_value:
            msg = self.message or f"must be at least {self.min_value}"
            result.add_error(msg)
        
        if self.max_value is not None and num_value > self.max_value:
            msg = self.message or f"must be at most {self.max_value}"
            result.add_error(msg)
        
        return result


class PatternValidator(Validator):
    """
    Validates that a string matches a regex pattern.
    
    Usage:
        validator = PatternValidator(r'^[a-z]+$', "must be lowercase letters")
        result = validator.validate("hello", "name")
    """
    
    def __init__(self, pattern: str, message: Optional[str] = None):
        self.pattern = pattern
        self.compiled = re.compile(pattern)
        self.message = message or f"does not match pattern: {pattern}"
    
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        result = ValidationResult(field_name=field_name)
        
        if value is None:
            return result  # None is handled by RequiredValidator
        
        if not isinstance(value, str):
            result.add_error(f"must be a string, got {type(value).__name__}")
            return result
        
        if not self.compiled.match(value):
            result.add_error(self.message)
        
        return result


class LengthValidator(Validator):
    """
    Validates the length of a string, list, or other sized object.
    
    Usage:
        validator = LengthValidator(min_length=3, max_length=50)
        result = validator.validate("hello", "username")
    """
    
    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        message: Optional[str] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.message = message
    
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        result = ValidationResult(field_name=field_name)
        
        if value is None:
            return result  # None is handled by RequiredValidator
        
        try:
            length = len(value)
        except TypeError:
            result.add_error(f"cannot determine length of {type(value).__name__}")
            return result
        
        if self.min_length is not None and length < self.min_length:
            msg = self.message or f"must be at least {self.min_length} characters"
            result.add_error(msg)
        
        if self.max_length is not None and length > self.max_length:
            msg = self.message or f"must be at most {self.max_length} characters"
            result.add_error(msg)
        
        return result


class ChoicesValidator(Validator):
    """
    Validates that a value is one of the allowed choices.
    
    Usage:
        validator = ChoicesValidator(["dark", "light", "system"])
        result = validator.validate("dark", "theme")
    """
    
    def __init__(
        self,
        choices: Union[List[Any], Set[Any]],
        message: Optional[str] = None,
    ):
        self.choices = set(choices) if not isinstance(choices, set) else choices
        self.message = message
    
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        result = ValidationResult(field_name=field_name)
        
        if value is None:
            return result  # None is handled by RequiredValidator
        
        if value not in self.choices:
            choices_str = ", ".join(repr(c) for c in sorted(str(c) for c in self.choices))
            msg = self.message or f"must be one of: {choices_str}"
            result.add_error(msg)
        
        return result


class TypeValidator(Validator):
    """
    Validates that a value is of the expected type(s).
    
    Usage:
        validator = TypeValidator(int)
        result = validator.validate(42, "count")
        
        # Multiple types
        validator = TypeValidator((int, float))
        result = validator.validate(3.14, "number")
    """
    
    def __init__(
        self,
        expected_type: Union[Type, tuple],
        message: Optional[str] = None,
    ):
        self.expected_type = expected_type
        self.message = message
    
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        result = ValidationResult(field_name=field_name)
        
        if value is None:
            return result  # None is handled by RequiredValidator
        
        if not isinstance(value, self.expected_type):
            if isinstance(self.expected_type, tuple):
                type_names = " or ".join(t.__name__ for t in self.expected_type)
            else:
                type_names = self.expected_type.__name__
            msg = self.message or f"must be {type_names}, got {type(value).__name__}"
            result.add_error(msg)
        
        return result


class CustomValidator(Validator):
    """
    Validator with a custom validation function.
    
    Usage:
        # Lambda style
        validator = CustomValidator(lambda v: v % 2 == 0, "must be even")
        
        # Function style
        def is_valid_email(value):
            return "@" in value and "." in value
        validator = CustomValidator(is_valid_email, "must be a valid email")
    """
    
    def __init__(
        self,
        func: Callable[[Any], bool],
        message: str = "validation failed",
    ):
        self.func = func
        self.message = message
    
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        result = ValidationResult(field_name=field_name)
        
        if value is None:
            return result  # None is handled by RequiredValidator
        
        try:
            if not self.func(value):
                result.add_error(self.message)
        except Exception as e:
            result.add_error(f"validation error: {e}")
        
        return result


# =============================================================================
# Composable Validators
# =============================================================================

class All(Validator):
    """
    Composes multiple validators with AND logic.
    
    All validators must pass for the result to be valid.
    
    Usage:
        validator = All(
            RequiredValidator(),
            LengthValidator(min_length=3),
            PatternValidator(r'^[a-z]+$'),
        )
        result = validator.validate("hello", "name")
    """
    
    def __init__(self, *validators: Validator, stop_on_first_error: bool = False):
        self.validators = validators
        self.stop_on_first_error = stop_on_first_error
    
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        result = ValidationResult(field_name=field_name)
        
        for validator in self.validators:
            sub_result = validator.validate(value, field_name)
            result.merge(sub_result)
            
            if self.stop_on_first_error and not sub_result.valid:
                break
        
        return result


class Any(Validator):
    """
    Composes multiple validators with OR logic.
    
    At least one validator must pass for the result to be valid.
    
    Usage:
        validator = Any(
            PatternValidator(r'^\\d+$', "must be digits"),
            PatternValidator(r'^[a-z]+$', "must be letters"),
        )
        result = validator.validate("123", "code")  # Passes (digits)
    """
    
    def __init__(self, *validators: Validator, message: Optional[str] = None):
        self.validators = validators
        self.message = message
    
    def validate(self, value: Any, field_name: str = "") -> ValidationResult:
        if not self.validators:
            return ValidationResult(field_name=field_name)
        
        all_errors = []
        
        for validator in self.validators:
            sub_result = validator.validate(value, field_name)
            if sub_result.valid:
                return ValidationResult(field_name=field_name)  # Success!
            all_errors.extend(sub_result.errors)
        
        # All validators failed
        result = ValidationResult(field_name=field_name)
        if self.message:
            result.add_error(self.message)
        else:
            result.add_error("none of the validation options passed")
        
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def validate(
    value: Any,
    validators: Union[Validator, List[Validator]],
    field_name: str = "",
) -> ValidationResult:
    """
    Validate a value against one or more validators.
    
    Args:
        value: The value to validate
        validators: Single validator or list of validators
        field_name: Optional field name for error messages
        
    Returns:
        ValidationResult with any errors/warnings
    
    Usage:
        result = validate(user_input, [
            RequiredValidator(),
            LengthValidator(min_length=3),
        ])
    """
    if isinstance(validators, Validator):
        return validators.validate(value, field_name)
    
    # Wrap in All for list of validators
    combined = All(*validators)
    return combined.validate(value, field_name)


def validate_field(
    field_name: str,
    value: Any,
    validators: Union[Validator, List[Validator]],
) -> ValidationResult:
    """
    Validate a field value (field_name first for readability).
    
    Args:
        field_name: Name of the field
        value: The value to validate
        validators: Single validator or list of validators
        
    Returns:
        ValidationResult with any errors/warnings
    
    Usage:
        result = validate_field("username", username, [
            RequiredValidator(),
            LengthValidator(min_length=3),
        ])
    """
    return validate(value, validators, field_name)

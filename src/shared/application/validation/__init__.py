"""
Shared validation module.

Provides a unified validation framework for:
- Form validation
- Input validation
- Data validation
- Settings validation

Key Components:
- ValidationResult: Result container with errors/warnings
- Validator: Base class for validators
- Common validators: Required, Range, Pattern, Length, Choices, etc.
- validate(): Convenience function for validation chains
"""
from .validation_framework import (
    ValidationResult,
    Validator,
    ValidationError,
    # Common validators
    RequiredValidator,
    RangeValidator,
    PatternValidator,
    LengthValidator,
    ChoicesValidator,
    TypeValidator,
    CustomValidator,
    # Convenience functions
    validate,
    validate_field,
    # Composable validators
    All,
    Any as AnyOf,
)

__all__ = [
    'ValidationResult',
    'Validator',
    'ValidationError',
    'RequiredValidator',
    'RangeValidator',
    'PatternValidator',
    'LengthValidator',
    'ChoicesValidator',
    'TypeValidator',
    'CustomValidator',
    'validate',
    'validate_field',
    'All',
    'AnyOf',
]

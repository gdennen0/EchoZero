"""
Smoke tests: Basic sanity checks that the package loads and core types exist.
Exists to catch broken imports, missing modules, and version mismatches early.
Runs first in CI — if these fail, nothing else matters.
"""

import echozero
from echozero.errors import (
    DomainError,
    EchoZeroError,
    ExecutionError,
    InfrastructureError,
    OperationCancelledError,
    ValidationError,
)
from echozero.result import Err, Ok, err, is_err, is_ok, map_result, ok, unwrap


def test_version_exists() -> None:
    """Package version is set and follows semver format."""
    assert hasattr(echozero, "__version__")
    assert echozero.__version__ == "2.0.0-dev"


def test_error_hierarchy() -> None:
    """Error classes form the correct inheritance chain."""
    assert issubclass(DomainError, EchoZeroError)
    assert issubclass(ValidationError, DomainError)
    assert issubclass(InfrastructureError, EchoZeroError)
    assert issubclass(ExecutionError, EchoZeroError)
    assert issubclass(OperationCancelledError, EchoZeroError)


def test_error_is_catchable() -> None:
    """Domain errors can be caught by their parent type."""
    try:
        raise ValidationError("test")
    except DomainError as e:
        assert str(e) == "test"
    except Exception:
        raise AssertionError("ValidationError should be catchable as DomainError")


def test_result_ok() -> None:
    """Ok result contains a value and reports as ok."""
    result = ok(42)
    assert is_ok(result)
    assert not is_err(result)
    assert unwrap(result) == 42


def test_result_err() -> None:
    """Err result contains an error and reports as err."""
    error = ValueError("something broke")
    result = err(error)
    assert is_err(result)
    assert not is_ok(result)


def test_result_unwrap_err_raises() -> None:
    """Unwrapping an Err raises the contained error."""
    error = ValueError("boom")
    result = err(error)
    try:
        unwrap(result)
        raise AssertionError("Should have raised")
    except ValueError as e:
        assert str(e) == "boom"


def test_result_map_ok() -> None:
    """Mapping over Ok applies the function."""
    result = ok(10)
    mapped = map_result(result, lambda x: x * 2)
    assert is_ok(mapped)
    assert unwrap(mapped) == 20


def test_result_map_err_passthrough() -> None:
    """Mapping over Err passes the error through unchanged."""
    error = ValueError("original")
    result = err(error)
    mapped = map_result(result, lambda x: x * 2)
    assert is_err(mapped)

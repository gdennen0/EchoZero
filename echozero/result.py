"""
Result: Typed success/failure container for operations that can fail.
Exists because pipeline boundary operations return Result[T] instead of raising (STYLE.md).
Used by pipeline dispatcher and command handlers to communicate success or failure without exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Successful result containing a value."""

    value: T


@dataclass(frozen=True)
class Err:
    """Failed result containing an error."""

    error: Exception


Result = Union[Ok[T], Err]


def ok(value: T) -> Ok[T]:
    """Create a successful result."""
    return Ok(value=value)


def err(error: Exception) -> Err:
    """Create a failed result."""
    return Err(error=error)


def is_ok(result: Result[T]) -> bool:
    """Check if a result is successful."""
    return isinstance(result, Ok)


def is_err(result: Result[T]) -> bool:
    """Check if a result is a failure."""
    return isinstance(result, Err)


def unwrap(result: Result[T]) -> T:
    """Extract the value from a successful result, or raise the error."""
    if isinstance(result, Ok):
        return result.value
    raise result.error


def map_result(result: Result[T], fn: Callable[[T], U]) -> Result[U]:
    """Apply a function to a successful result's value, passing errors through."""
    if isinstance(result, Ok):
        try:
            return ok(fn(result.value))
        except Exception as e:
            return err(e)
    return result

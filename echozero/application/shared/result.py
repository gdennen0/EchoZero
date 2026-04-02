"""Minimal shared result type for application-layer operations."""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Result(Generic[T]):
    ok: bool
    value: T | None = None
    errors: list[str] = field(default_factory=list)

    @classmethod
    def success(cls, value: T | None = None) -> "Result[T]":
        return cls(ok=True, value=value, errors=[])

    @classmethod
    def failure(cls, *errors: str) -> "Result[T]":
        return cls(ok=False, value=None, errors=list(errors))

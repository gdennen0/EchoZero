"""
Remote HTTP helpers: Shared request and payload utilities for the private wrapper.
Exists because server routes should stay small while auth, JSON, and range parsing remain consistent.
Connects the remote HTTP surface to typed payload handling and browser-friendly byte serving.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def coerce_params(value: Any) -> dict[str, Any] | None:
    """Return one JSON object payload or reject non-object params."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("params must be a JSON object when provided.")
    return value


def first_query_value(params: dict[str, list[str]], key: str) -> str | None:
    """Return the first query value for one key when present."""
    values = params.get(key)
    if not values:
        return None
    return values[0]


def normalize_token(value: str | None) -> str | None:
    """Return one stripped token string when it is non-empty."""
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def parse_bearer_token(value: str | None) -> str | None:
    """Extract one bearer token from an Authorization header."""
    if value is None:
        return None
    prefix = "Bearer "
    if not value.startswith(prefix):
        return None
    token = value[len(prefix) :].strip()
    return token or None


def parse_byte_range(value: str | None, *, total_size: int) -> tuple[int, int] | None:
    """Parse one HTTP byte range header into an inclusive start/end pair."""
    if value is None or total_size <= 0:
        return None
    prefix = "bytes="
    if not value.startswith(prefix):
        raise ValueError("Malformed Range header.")
    first_range = value[len(prefix) :].split(",", 1)[0].strip()
    if "-" not in first_range:
        raise ValueError("Malformed Range header.")
    start_text, end_text = first_range.split("-", 1)
    if not start_text and not end_text:
        raise ValueError("Malformed Range header.")
    if not start_text:
        length = int(end_text)
        if length <= 0:
            raise ValueError("Malformed Range header.")
        start = max(total_size - length, 0)
        end = total_size - 1
        return start, end
    start = int(start_text)
    end = total_size - 1 if not end_text else int(end_text)
    if start < 0 or start >= total_size or end < start:
        raise ValueError("Malformed Range header.")
    return start, min(end, total_size - 1)


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and tuples into JSON-safe Python values."""
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value

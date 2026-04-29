"""Shared cue-number parsing and formatting helpers."""

from __future__ import annotations

import math
import re
from typing import TypeAlias

CueNumber: TypeAlias = int | float

_CUE_REF_NUMBER_PATTERN = re.compile(r"^[A-Za-z]*\s*(\d+(?:\.\d+)?)")
_INTEGER_TEXT_PATTERN = re.compile(r"^[+-]?\d+$")


def parse_positive_cue_number(value: object) -> CueNumber | None:
    """Parse one positive cue number, preserving decimals when present."""

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed: CueNumber = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            return None
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = int(text) if _INTEGER_TEXT_PATTERN.fullmatch(text) else float(text)
        except (TypeError, ValueError):
            return None
    if float(parsed) < 1.0:
        return None
    if isinstance(parsed, float) and parsed.is_integer():
        return int(parsed)
    return parsed


def coerce_positive_cue_number(value: object) -> CueNumber:
    """Coerce one cue number or raise when the value is not positive numeric data."""

    parsed = parse_positive_cue_number(value)
    if parsed is None:
        raise ValueError(f"cue_number must be positive numeric data, got {value!r}")
    return parsed


def cue_number_text(value: CueNumber | None) -> str | None:
    """Render one cue number as stable text for labels, refs, and MA commands."""

    parsed = parse_positive_cue_number(value)
    if parsed is None:
        return None
    if isinstance(parsed, int):
        return str(parsed)
    return format(parsed, "f").rstrip("0").rstrip(".")


def cue_number_from_ref_text(cue_ref: str | None) -> CueNumber | None:
    """Resolve one best-effort cue number from a cue-ref string."""

    if cue_ref in {None, ""}:
        return None
    match = _CUE_REF_NUMBER_PATTERN.match(str(cue_ref).strip())
    if match is None:
        return None
    return parse_positive_cue_number(match.group(1))

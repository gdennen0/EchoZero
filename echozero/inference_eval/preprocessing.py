from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .constants import REQUIRED_PREPROCESSING_KEYS

_PREPROCESSING_ALIASES = {
    "sample_rate": "sampleRate",
    "max_length": "maxLength",
    "n_fft": "nFft",
    "hop_length": "hopLength",
    "n_mels": "nMels",
    "f_max": "fmax",
}


def canonicalize_preprocessing_keys(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize preprocessing metadata to the shared camelCase contract keys."""

    normalized: dict[str, Any] = {}
    for raw_key, value in payload.items():
        key = str(raw_key)
        normalized[_PREPROCESSING_ALIASES.get(key, key)] = value
    return normalized


def runtime_preprocessing_from_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract runtime preprocessing fields using the shared canonical key set."""

    if not isinstance(payload, Mapping):
        return {}
    preprocessing = canonicalize_preprocessing_keys(payload)
    return {
        key: preprocessing[key]
        for key in sorted(REQUIRED_PREPROCESSING_KEYS)
        if key in preprocessing
    }

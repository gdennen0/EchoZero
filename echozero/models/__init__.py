"""
Model registry: Track, resolve, and manage ML models for EchoZero processors.

Local-first. Cloud sync is a future extension — the registry contract supports it
but ships with filesystem-only storage.
"""

from echozero.models.registry import (
    ModelCard,
    ModelRegistry,
    ModelType,
)

__all__ = [
    "ModelCard",
    "ModelRegistry",
    "ModelType",
]

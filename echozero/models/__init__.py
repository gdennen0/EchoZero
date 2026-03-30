"""
Model registry: Track, resolve, and manage ML models for EchoZero processors.

Local-first. Cloud sync via HuggingFace Hub — the provider handles downloads,
the registry handles the catalog.
"""

from echozero.models.registry import (
    ModelCard,
    ModelRegistry,
    ModelSource,
    ModelStatus,
    ModelType,
)
from echozero.models.provider import (
    DownloadProgress,
    HuggingFaceSource,
    LocalFileSource,
    ModelProvider,
    ModelUpdate,
    ProgressCallback,
)

__all__ = [
    "DownloadProgress",
    "HuggingFaceSource",
    "LocalFileSource",
    "ModelCard",
    "ModelProvider",
    "ModelRegistry",
    "ModelSource",
    "ModelStatus",
    "ModelType",
    "ModelUpdate",
    "ProgressCallback",
]

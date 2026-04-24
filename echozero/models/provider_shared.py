"""Shared model-provider types and helpers.
Exists to keep provider source adapters and the public provider root aligned on the same contracts.
Connects model registry safety checks and progress payloads to provider workflows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from echozero.models.registry import ModelType


@dataclass(frozen=True)
class ModelUpdate:
    """Describes an available update for a model."""

    model_type: ModelType
    current_version: str | None
    available_version: str
    model_id: str
    size_bytes: int | None = None
    description: str = ""
    changelog: str = ""


@dataclass(frozen=True)
class DownloadProgress:
    """Progress report for a model download."""

    model_id: str
    bytes_downloaded: int
    bytes_total: int | None
    status: str = "downloading"

    @property
    def fraction(self) -> float | None:
        if self.bytes_total and self.bytes_total > 0:
            return self.bytes_downloaded / self.bytes_total
        return None


ProgressCallback = Callable[[DownloadProgress], None]


class RemoteSource(Protocol):
    """Protocol for remote model sources."""

    def check_available(self, org: str, model_type: ModelType) -> list[ModelUpdate]:
        ...

    def download(
        self,
        model_id: str,
        target_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        ...

    def get_latest_version(self, org: str, model_type: ModelType) -> str | None:
        ...


_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-.]*$")


def validate_model_id(model_id: str) -> str:
    """Sanitize model_id for safe filesystem use."""

    safe_id = model_id.replace("/", "_")
    if not _SAFE_ID_RE.match(safe_id):
        raise ValueError(f"Invalid model ID after sanitization: {model_id!r}")
    if ".." in safe_id:
        raise ValueError(f"Model ID contains path traversal: {model_id!r}")
    return safe_id


def parse_version(version: str) -> tuple[int, ...]:
    """Parse version string to a comparable tuple."""

    try:
        return tuple(int(part) for part in version.split("."))
    except (ValueError, AttributeError):
        return (0,)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_version_from_tags(tags: list[str]) -> str | None:
    """Extract a semantic version string from source tags."""

    for tag in tags:
        if tag.startswith("v") and "." in tag:
            return tag.lstrip("v")
    return None


__all__ = [
    "DownloadProgress",
    "ModelUpdate",
    "ProgressCallback",
    "RemoteSource",
    "extract_version_from_tags",
    "parse_version",
    "sha256_file",
    "validate_model_id",
]

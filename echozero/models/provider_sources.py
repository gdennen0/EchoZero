"""Remote source adapters for the model provider.
Exists to isolate HuggingFace and local-copy download behavior from provider registry orchestration.
Connects provider contracts to concrete remote and local model sources.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, cast

from echozero.models.registry import ModelType

from .provider_shared import (
    DownloadProgress,
    ModelUpdate,
    ProgressCallback,
    extract_version_from_tags,
    parse_version,
)

logger = logging.getLogger(__name__)


class HuggingFaceSource:
    """Remote source backed by HuggingFace Hub."""

    def __init__(self) -> None:
        self._hub: Any | None = None
        try:
            import huggingface_hub  # type: ignore[import-not-found]

            self._hub = huggingface_hub
        except ImportError:
            logger.warning(
                "huggingface_hub not installed. Remote model management unavailable. "
                "Install with: pip install huggingface_hub"
            )

    @property
    def available(self) -> bool:
        return self._hub is not None

    def check_available(self, org: str, model_type: ModelType) -> list[ModelUpdate]:
        if not self.available:
            return []

        updates: list[ModelUpdate] = []
        hub = cast(Any, self._hub)
        try:
            api = hub.HfApi()
            models = api.list_models(author=org, tags=[f"echozero-{model_type.value}"])
            for model in models:
                updates.append(
                    ModelUpdate(
                        model_type=model_type,
                        current_version=None,
                        available_version=extract_version_from_tags(model.tags or []) or "unknown",
                        model_id=model.id,
                        size_bytes=getattr(model, "size", None),
                        description=getattr(model, "description", ""),
                    )
                )
        except Exception as exc:
            logger.error(
                "Failed to check HuggingFace for %s models (org=%r): %s",
                model_type.value,
                org,
                exc,
            )

        return updates

    def download(
        self,
        model_id: str,
        target_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        if not self.available:
            raise RuntimeError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        if on_progress:
            on_progress(
                DownloadProgress(
                    model_id=model_id,
                    bytes_downloaded=0,
                    bytes_total=None,
                    status="downloading",
                )
            )

        try:
            hub = cast(Any, self._hub)
            path = hub.snapshot_download(
                repo_id=model_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
            if on_progress:
                on_progress(
                    DownloadProgress(
                        model_id=model_id,
                        bytes_downloaded=0,
                        bytes_total=0,
                        status="complete",
                    )
                )
            return Path(path)
        except Exception as exc:
            if on_progress:
                on_progress(
                    DownloadProgress(
                        model_id=model_id,
                        bytes_downloaded=0,
                        bytes_total=None,
                        status="failed",
                    )
                )
            raise RuntimeError(f"Failed to download model '{model_id}': {exc}") from exc

    def get_latest_version(self, org: str, model_type: ModelType) -> str | None:
        updates = self.check_available(org, model_type)
        if not updates:
            return None
        versions = sorted(updates, key=lambda update: parse_version(update.available_version), reverse=True)
        return versions[0].available_version


class LocalFileSource:
    """Remote source that copies a model from a local directory."""

    def __init__(self, source_dir: Path | None = None) -> None:
        self._source_dir = source_dir

    def check_available(self, org: str, model_type: ModelType) -> list[ModelUpdate]:
        return []

    def download(
        self,
        model_id: str,
        target_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        del on_progress
        if self._source_dir is None:
            raise RuntimeError("No source directory configured")

        source_path = (self._source_dir / model_id).resolve()
        base = self._source_dir.resolve()
        try:
            if not source_path.is_relative_to(base):
                raise ValueError(f"Model ID escapes source directory: {model_id!r}")
        except AttributeError:
            base_str = str(base)
            source_path_str = str(source_path)
            if not (
                source_path_str == base_str or source_path_str.startswith(base_str + _os_sep())
            ):
                raise ValueError(f"Model ID escapes source directory: {model_id!r}")

        if not source_path.exists():
            raise FileNotFoundError(f"Model not found at {source_path}")

        target_dir.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            target = target_dir / source_path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source_path, target)
            return target

        target = target_dir / source_path.name
        shutil.copy2(source_path, target)
        return target

    def get_latest_version(self, org: str, model_type: ModelType) -> str | None:
        return None


def _os_sep() -> str:
    import os

    return os.sep


__all__ = ["HuggingFaceSource", "LocalFileSource"]

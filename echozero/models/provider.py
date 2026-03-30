"""
ModelProvider: Download, update, and manage ML model files.

Exists because the ModelRegistry is a catalog (metadata only) — it doesn't move files.
The ModelProvider handles the I/O: downloading from HuggingFace Hub, checking for
updates, and managing the local model cache.

Architecture:
    HuggingFace Hub (remote)
         │
         ▼  download / check_updates
    ModelProvider (this file)
         │
         ▼  register / update status
    ModelRegistry (catalog)
         │
         ▼  resolve → file path
    Processor (consumer)

The provider is the bridge between cloud and local. It uses huggingface_hub
when available, but degrades gracefully to manual model management if not installed.

Design decisions:
- HuggingFace Hub is the distribution channel (industry standard for ML models)
- Downloads go to the models directory managed by ModelRegistry
- Progress callbacks for UI integration (download bars, notifications)
- Update checks are explicit (not automatic) — user controls when to download
- Provider is injectable for testing (no real HTTP in tests)
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from echozero.models.registry import (
    ModelCard,
    ModelRegistry,
    ModelSource,
    ModelStatus,
    ModelType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelUpdate:
    """Describes an available update for a model."""
    model_type: ModelType
    current_version: str | None   # None if not installed
    available_version: str
    model_id: str                 # remote model ID (e.g., "echozero/onset-v2")
    size_bytes: int | None = None
    description: str = ""
    changelog: str = ""


@dataclass(frozen=True)
class DownloadProgress:
    """Progress report for a model download."""
    model_id: str
    bytes_downloaded: int
    bytes_total: int | None
    status: str = "downloading"   # downloading, extracting, registering, complete, failed

    @property
    def fraction(self) -> float | None:
        if self.bytes_total and self.bytes_total > 0:
            return self.bytes_downloaded / self.bytes_total
        return None


# Callback type for progress reporting
ProgressCallback = Callable[[DownloadProgress], None]


class RemoteSource(Protocol):
    """Protocol for remote model sources. HuggingFace is the default implementation.

    Injectable for testing — swap in a mock that returns local files.
    """

    def check_available(self, org: str, model_type: ModelType) -> list[ModelUpdate]:
        """Check what models are available remotely for this type."""
        ...

    def download(
        self,
        model_id: str,
        target_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        """Download a model to the target directory. Returns the model file path."""
        ...

    def get_latest_version(self, org: str, model_type: ModelType) -> str | None:
        """Get the latest version string for a model type."""
        ...


# ---------------------------------------------------------------------------
# HuggingFace implementation
# ---------------------------------------------------------------------------


class HuggingFaceSource:
    """Remote source backed by HuggingFace Hub.

    Requires `huggingface_hub` pip package. Handles download, caching, and
    version resolution through HF's infrastructure.

    Usage:
        source = HuggingFaceSource()
        updates = source.check_available("echozero", ModelType.ONSET_DETECTION)
        path = source.download("echozero/onset-v2", target_dir)
    """

    def __init__(self) -> None:
        self._hub = None
        try:
            import huggingface_hub
            self._hub = huggingface_hub
        except ImportError:
            logger.warning(
                "huggingface_hub not installed. Remote model management unavailable. "
                "Install with: pip install huggingface_hub"
            )

    @property
    def available(self) -> bool:
        """Whether HuggingFace Hub is installed and usable."""
        return self._hub is not None

    def check_available(self, org: str, model_type: ModelType) -> list[ModelUpdate]:
        """List available models on HuggingFace for this org and type."""
        if not self.available:
            return []

        updates: list[ModelUpdate] = []
        try:
            api = self._hub.HfApi()
            # Search for models in the org with a tag matching the type
            models = api.list_models(
                author=org,
                tags=[f"echozero-{model_type.value}"],
            )
            for model in models:
                # Parse version from model card or tags
                version = _extract_version_from_tags(model.tags or [])
                size = getattr(model, "size", None)
                updates.append(ModelUpdate(
                    model_type=model_type,
                    current_version=None,  # caller fills this in
                    available_version=version or "unknown",
                    model_id=model.id,
                    size_bytes=size,
                    description=getattr(model, "description", ""),
                ))
        except Exception as e:
            logger.error("Failed to check HuggingFace for models: %s", e)

        return updates

    def download(
        self,
        model_id: str,
        target_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        """Download a model from HuggingFace Hub to the target directory."""
        if not self.available:
            raise RuntimeError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )

        target_dir.mkdir(parents=True, exist_ok=True)

        if on_progress:
            on_progress(DownloadProgress(
                model_id=model_id,
                bytes_downloaded=0,
                bytes_total=None,
                status="downloading",
            ))

        try:
            # Download entire repo snapshot to target directory
            path = self._hub.snapshot_download(
                repo_id=model_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

            if on_progress:
                on_progress(DownloadProgress(
                    model_id=model_id,
                    bytes_downloaded=0,
                    bytes_total=0,
                    status="complete",
                ))

            return Path(path)
        except Exception as e:
            if on_progress:
                on_progress(DownloadProgress(
                    model_id=model_id,
                    bytes_downloaded=0,
                    bytes_total=None,
                    status="failed",
                ))
            raise RuntimeError(f"Failed to download model '{model_id}': {e}") from e

    def get_latest_version(self, org: str, model_type: ModelType) -> str | None:
        """Get latest version for a model type from HuggingFace."""
        updates = self.check_available(org, model_type)
        if not updates:
            return None
        # Sort by version, return highest
        versions = sorted(updates, key=lambda u: u.available_version, reverse=True)
        return versions[0].available_version


# ---------------------------------------------------------------------------
# Local file source (for testing and manual installs)
# ---------------------------------------------------------------------------


class LocalFileSource:
    """Remote source that copies from a local directory. For testing and manual model import.

    Usage:
        source = LocalFileSource(source_dir=Path("/path/to/model/files"))
        path = source.download("my-model", target_dir)
    """

    def __init__(self, source_dir: Path | None = None) -> None:
        self._source_dir = source_dir

    def check_available(self, org: str, model_type: ModelType) -> list[ModelUpdate]:
        """Local source doesn't support remote checking."""
        return []

    def download(
        self,
        model_id: str,
        target_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        """Copy a model from the local source directory."""
        if self._source_dir is None:
            raise RuntimeError("No source directory configured")

        source_path = self._source_dir / model_id
        if not source_path.exists():
            raise FileNotFoundError(f"Model not found at {source_path}")

        target_dir.mkdir(parents=True, exist_ok=True)

        if source_path.is_dir():
            target = target_dir / source_path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source_path, target)
            return target
        else:
            target = target_dir / source_path.name
            shutil.copy2(source_path, target)
            return target

    def get_latest_version(self, org: str, model_type: ModelType) -> str | None:
        return None


# ---------------------------------------------------------------------------
# ModelProvider — the main API
# ---------------------------------------------------------------------------


class ModelProvider:
    """High-level API for model lifecycle management.

    Bridges RemoteSource (HuggingFace/local) with ModelRegistry (catalog).
    Handles: download → extract → register → set default.

    Usage:
        provider = ModelProvider(registry, source=HuggingFaceSource())

        # Check for updates
        updates = provider.check_updates(org="echozero")

        # Download and register a model
        card = provider.install(
            model_id="echozero/onset-v2",
            model_type=ModelType.ONSET_DETECTION,
            version="2.0.0",
            on_progress=my_progress_callback,
        )

        # Import a local model file
        card = provider.import_local(
            path=Path("/downloads/my-model.pt"),
            model_type=ModelType.CLASSIFICATION,
            name="My Custom Model",
            version="1.0.0",
        )
    """

    def __init__(
        self,
        registry: ModelRegistry,
        source: RemoteSource | None = None,
        org: str = "echozero",
    ) -> None:
        self._registry = registry
        self._source = source or HuggingFaceSource()
        self._org = org

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def check_updates(
        self,
        model_type: ModelType | None = None,
    ) -> list[ModelUpdate]:
        """Check for available model updates.

        Compares local registry versions against remote. Returns updates
        where the remote version is newer (or not installed locally).

        Args:
            model_type: Check a specific type, or all types if None.
        """
        types_to_check = [model_type] if model_type else list(ModelType)
        all_updates: list[ModelUpdate] = []

        for mt in types_to_check:
            remote_models = self._source.check_available(self._org, mt)
            for update in remote_models:
                # Check if we have this type locally
                local = self._registry.resolve(mt)
                current_ver = local.version if local else None

                # Only report if newer or not installed
                if current_ver is None or update.available_version > current_ver:
                    all_updates.append(ModelUpdate(
                        model_type=mt,
                        current_version=current_ver,
                        available_version=update.available_version,
                        model_id=update.model_id,
                        size_bytes=update.size_bytes,
                        description=update.description,
                    ))

        return all_updates

    def install(
        self,
        model_id: str,
        model_type: ModelType,
        version: str,
        name: str | None = None,
        set_default: bool = True,
        on_progress: ProgressCallback | None = None,
    ) -> ModelCard:
        """Download and register a model from the remote source.

        Args:
            model_id: Remote model identifier (e.g., "echozero/onset-v2").
            model_type: What kind of model this is.
            version: Version string.
            name: Human-readable name (defaults to model_id).
            set_default: Whether to make this the default for its type.
            on_progress: Progress callback for UI.

        Returns:
            The registered ModelCard.
        """
        # Determine target directory
        safe_id = model_id.replace("/", "_")
        target_dir = self._registry.models_dir / safe_id

        if on_progress:
            on_progress(DownloadProgress(
                model_id=model_id,
                bytes_downloaded=0,
                bytes_total=None,
                status="downloading",
            ))

        # Download
        downloaded_path = self._source.download(model_id, target_dir, on_progress)

        # Register
        relative_path = str(downloaded_path.relative_to(self._registry.models_dir))
        card = ModelCard(
            id=safe_id,
            model_type=model_type,
            name=name or model_id,
            version=version,
            source=ModelSource.CLOUD,
            relative_path=relative_path,
        )
        self._registry.register(card)

        if set_default:
            self._registry.set_default(model_type, card.id)

        # Persist
        self._registry.save()

        if on_progress:
            on_progress(DownloadProgress(
                model_id=model_id,
                bytes_downloaded=0,
                bytes_total=0,
                status="complete",
            ))

        logger.info("Installed model %s (v%s) as %s", model_id, version, card.id)
        return card

    def import_local(
        self,
        path: Path,
        model_type: ModelType,
        name: str,
        version: str = "1.0.0",
        model_id: str | None = None,
        set_default: bool = True,
    ) -> ModelCard:
        """Import a model from a local file/directory into the registry.

        Copies the file into the models directory and registers it.

        Args:
            path: Path to the model file or directory.
            model_type: What kind of model.
            name: Human-readable name.
            version: Version string.
            model_id: Registry ID (auto-generated from name if not given).
            set_default: Whether to set as default.

        Returns:
            The registered ModelCard.
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Generate ID from name
        card_id = model_id or name.lower().replace(" ", "-").replace("/", "-")

        # Copy to models directory
        target = self._registry.models_dir / card_id
        target.mkdir(parents=True, exist_ok=True)

        if path.is_dir():
            dest = target / path.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(path, dest)
            relative_path = str(dest.relative_to(self._registry.models_dir))
        else:
            dest = target / path.name
            shutil.copy2(path, dest)
            relative_path = str(dest.relative_to(self._registry.models_dir))

        card = ModelCard(
            id=card_id,
            model_type=model_type,
            name=name,
            version=version,
            source=ModelSource.IMPORTED,
            relative_path=relative_path,
        )
        self._registry.register(card)

        if set_default:
            self._registry.set_default(model_type, card_id)

        self._registry.save()

        logger.info("Imported local model '%s' as %s", name, card_id)
        return card

    def uninstall(self, model_id: str, delete_files: bool = True) -> bool:
        """Remove a model from the registry and optionally delete its files.

        Args:
            model_id: The model ID to remove.
            delete_files: Whether to delete the model files from disk.

        Returns:
            True if the model was found and removed.
        """
        card = self._registry.unregister(model_id)
        if card is None:
            return False

        if delete_files:
            model_path = self._registry.model_path(card)
            if model_path.exists():
                if model_path.is_dir():
                    shutil.rmtree(model_path)
                else:
                    model_path.unlink()

        self._registry.save()
        logger.info("Uninstalled model %s", model_id)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_version_from_tags(tags: list[str]) -> str | None:
    """Extract a version string from HuggingFace model tags."""
    for tag in tags:
        if tag.startswith("v") and "." in tag:
            return tag.lstrip("v")
    return None

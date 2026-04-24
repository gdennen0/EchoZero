"""Model artifact provider for EchoZero processor models.
Exists to download, update, and manage installed model files outside the registry catalog.
Connects remote model distribution and local cache state to processor-facing resolution.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from echozero.models.paths import ensure_installed_models_dir
from echozero.models.registry import (
    ModelCard,
    ModelRegistry,
    ModelSource,
    ModelType,
)
from echozero.models.provider_shared import (
    DownloadProgress,
    ModelUpdate,
    ProgressCallback,
    RemoteSource,
    parse_version,
    sha256_file,
    validate_model_id,
)
from echozero.models.provider_sources import HuggingFaceSource, LocalFileSource

logger = logging.getLogger(__name__)

# Canonical user-local runtime model install root.
DEFAULT_MODELS_DIR = ensure_installed_models_dir()


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

                # Only report if newer or not installed (semantic version comparison)
                if current_ver is None or parse_version(update.available_version) > parse_version(current_ver):
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
        # Validate and sanitize model_id for safe filesystem use
        safe_id = validate_model_id(model_id)
        target_dir = self._registry.models_dir / safe_id

        if on_progress:
            on_progress(DownloadProgress(
                model_id=model_id,
                bytes_downloaded=0,
                bytes_total=None,
                status="downloading",
            ))

        # Download — clean up on failure
        try:
            downloaded_path = self._source.download(model_id, target_dir, on_progress)
        except Exception:
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            raise

        # Compute SHA-256 for file integrity
        metadata: dict[str, Any] = {}
        if downloaded_path.is_file():
            metadata["sha256"] = sha256_file(downloaded_path)

        # Register
        relative_path = str(downloaded_path.relative_to(self._registry.models_dir))
        card = ModelCard(
            id=safe_id,
            model_type=model_type,
            name=name or model_id,
            version=version,
            source=ModelSource.CLOUD,
            relative_path=relative_path,
            metadata=metadata,
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
        force: bool = False,
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
            force: If True, overwrite an existing model at the same destination.

        Returns:
            The registered ModelCard.
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Generate ID from name and validate
        card_id = model_id or name.lower().replace(" ", "-").replace("/", "-")
        validate_model_id(card_id)  # raise on unsafe IDs

        # Copy to models directory
        target = self._registry.models_dir / card_id
        target.mkdir(parents=True, exist_ok=True)

        metadata: dict[str, Any] = {}

        if path.is_dir():
            dest = target / path.name
            if dest.exists():
                if not force:
                    raise ValueError(
                        f"Model already exists at {dest}. Use force=True to overwrite."
                    )
                shutil.rmtree(dest)
            shutil.copytree(path, dest)
            relative_path = str(dest.relative_to(self._registry.models_dir))
        else:
            dest = target / path.name
            if dest.exists() and not force:
                raise ValueError(
                    f"Model already exists at {dest}. Use force=True to overwrite."
                )
            shutil.copy2(path, dest)
            relative_path = str(dest.relative_to(self._registry.models_dir))
            metadata["sha256"] = sha256_file(dest)

        card = ModelCard(
            id=card_id,
            model_type=model_type,
            name=name,
            version=version,
            source=ModelSource.IMPORTED,
            relative_path=relative_path,
            metadata=metadata,
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

        Raises:
            RuntimeError: If the model is currently in use (acquired).
        """
        if self._registry.is_in_use(model_id):
            raise RuntimeError(
                f"Cannot uninstall model '{model_id}': it is currently in use. "
                "Call registry.release() first."
            )

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
                # Clean up empty parent directory
                parent = model_path.parent
                try:
                    if (
                        parent != self._registry.models_dir
                        and parent.exists()
                        and not any(parent.iterdir())
                    ):
                        parent.rmdir()
                except OSError:
                    pass

        self._registry.save()
        logger.info("Uninstalled model %s", model_id)
        return True


def _parse_version(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except (ValueError, AttributeError):
        return (0,)


__all__ = [
    "DEFAULT_MODELS_DIR",
    "DownloadProgress",
    "HuggingFaceSource",
    "LocalFileSource",
    "ModelProvider",
    "ModelUpdate",
    "ProgressCallback",
    "RemoteSource",
    "_parse_version",
]

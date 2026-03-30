"""
ModelRegistry: Catalog and resolution of ML models.

Exists because processors need models, and the mapping of "what model to use"
should be decoupled from the processor code. A processor asks for
"onset_detection, best available" and the registry resolves it to a file path.

Design principles:
- Local-first: models are files on disk, registry is a JSON manifest.
- Versioned: multiple versions of the same model type can coexist.
- Resolvable: ask for a type → get the best (or a specific) version.
- Extensible: ModelSource enum has LOCAL and CLOUD — cloud is a future plug-in.
- No downloads at import time: registry is metadata only, ModelProvider handles I/O.

The registry is a catalog. It doesn't load models, train them, or run inference.
It answers: "where is the model file, and what do I know about it?"
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelType(Enum):
    """What kind of task this model performs."""
    ONSET_DETECTION = "onset_detection"
    CLASSIFICATION = "classification"
    SEPARATION = "separation"
    NOTE_TRANSCRIPTION = "note_transcription"
    CUSTOM = "custom"


class ModelSource(Enum):
    """Where the model came from."""
    BUILTIN = "builtin"     # shipped with EchoZero
    USER_TRAINED = "user"   # trained by user in-app
    IMPORTED = "imported"   # manually added by user
    CLOUD = "cloud"         # downloaded from EchoZero cloud (future)


class ModelStatus(Enum):
    """Availability state."""
    AVAILABLE = "available"   # file exists, ready to use
    MISSING = "missing"       # registered but file not found
    DOWNLOADING = "downloading"  # cloud download in progress (future)
    CORRUPT = "corrupt"       # file exists but integrity check failed


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCard:
    """Metadata for a single model version. Immutable once registered.

    This is the "business card" for a model — everything you need to know
    without loading the actual weights.
    """
    id: str                          # unique ID, e.g. "onset-v2.1"
    model_type: ModelType            # what it does
    name: str                        # human-readable, e.g. "Onset Detection v2.1"
    version: str                     # semver-ish, e.g. "2.1.0"
    source: ModelSource              # where it came from
    relative_path: str               # path relative to models root dir
    description: str = ""            # optional description
    framework: str = "pytorch"       # pytorch, onnx, etc.
    metadata: dict[str, Any] = field(default_factory=dict)  # arbitrary extra info

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model_type"] = self.model_type.value
        d["source"] = self.source.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelCard:
        d = dict(d)  # don't mutate input
        d["model_type"] = ModelType(d["model_type"])
        d["source"] = ModelSource(d["source"])
        return cls(**d)


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


MANIFEST_FILENAME = "models.json"


class ModelRegistry:
    """Catalog of available ML models with resolution.

    Backed by a JSON manifest file in the models directory.
    Thread-safe: writes protected by RLock.

    Usage:
        registry = ModelRegistry(models_dir)
        registry.load()

        # Register a new model
        card = ModelCard(id="onset-v1", model_type=ModelType.ONSET_DETECTION, ...)
        registry.register(card)

        # Resolve: "give me the best onset detection model"
        card = registry.resolve(ModelType.ONSET_DETECTION)
        model_path = registry.model_path(card)

        # Or resolve a specific version
        card = registry.resolve(ModelType.ONSET_DETECTION, version="2.1.0")
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._cards: dict[str, ModelCard] = {}  # id → card
        self._defaults: dict[ModelType, str] = {}  # type → default model id
        self._lock = threading.RLock()
        self._in_use: set[str] = set()

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def load(self) -> None:
        """Load the manifest from disk. Creates directory + empty manifest if missing."""
        with self._lock:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = self._models_dir / MANIFEST_FILENAME

            if not manifest_path.exists():
                self._cards = {}
                self._defaults = {}
                self.save()
                return

            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(
                    "Corrupt manifest at %s, falling back to empty registry: %s",
                    manifest_path, e,
                )
                self._cards = {}
                self._defaults = {}
                return

            cards: dict[str, ModelCard] = {}
            for card_id, card_data in data.get("models", {}).items():
                try:
                    cards[card_id] = ModelCard.from_dict(card_data)
                except Exception as e:
                    logger.warning("Skipping bad model entry %r: %s", card_id, e)
            self._cards = cards

            defaults: dict[ModelType, str] = {}
            for k, v in data.get("defaults", {}).items():
                try:
                    defaults[ModelType(k)] = v
                except (ValueError, KeyError) as e:
                    logger.warning("Skipping unknown default type %r: %s", k, e)
            self._defaults = defaults

            logger.info("Loaded %d models from registry", len(self._cards))

    def save(self) -> None:
        """Persist the manifest to disk (atomic write)."""
        with self._lock:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = self._models_dir / MANIFEST_FILENAME
            data = {
                "version": 1,
                "models": {card_id: card.to_dict() for card_id, card in self._cards.items()},
                "defaults": {k.value: v for k, v in self._defaults.items()},
            }
            tmp = manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(str(tmp), str(manifest_path))

    # -- Registration -------------------------------------------------------

    def register(self, card: ModelCard) -> None:
        """Register a model. Overwrites if same ID exists."""
        with self._lock:
            self._cards[card.id] = card
            # Auto-set default if first of this type
            if card.model_type not in self._defaults:
                self._defaults[card.model_type] = card.id

    def unregister(self, model_id: str) -> ModelCard | None:
        """Remove a model from the registry. Returns the card or None."""
        with self._lock:
            card = self._cards.pop(model_id, None)
            if card is not None:
                # Clear default if it pointed to this model
                if self._defaults.get(card.model_type) == model_id:
                    # Find next available of same type, or clear
                    others = [c for c in self._cards.values() if c.model_type == card.model_type]
                    if others:
                        self._defaults[card.model_type] = others[0].id
                    else:
                        del self._defaults[card.model_type]
            return card

    def set_default(self, model_type: ModelType, model_id: str) -> None:
        """Set the default model for a type. The model must already be registered."""
        with self._lock:
            if model_id not in self._cards:
                raise KeyError(f"Model '{model_id}' not registered")
            card = self._cards[model_id]
            if card.model_type != model_type:
                raise ValueError(
                    f"Model '{model_id}' is type {card.model_type.value}, "
                    f"not {model_type.value}"
                )
            self._defaults[model_type] = model_id

    # -- Resolution ---------------------------------------------------------

    def resolve(
        self,
        model_type: ModelType,
        version: str | None = None,
        model_id: str | None = None,
    ) -> ModelCard | None:
        """Resolve a model by type, optionally filtered by version or ID.

        Resolution order:
        1. If model_id given → return that exact model (or None)
        2. If version given → find matching type + version
        3. Otherwise → return the default for this type

        Returns None if no matching model found.
        """
        if model_id is not None:
            return self._cards.get(model_id)

        if version is not None:
            for card in self._cards.values():
                if card.model_type == model_type and card.version == version:
                    return card
            return None

        # Default resolution
        default_id = self._defaults.get(model_type)
        if default_id is not None:
            return self._cards.get(default_id)
        return None

    def list_models(self, model_type: ModelType | None = None) -> list[ModelCard]:
        """List all models, optionally filtered by type."""
        if model_type is None:
            return list(self._cards.values())
        return [c for c in self._cards.values() if c.model_type == model_type]

    def get(self, model_id: str) -> ModelCard | None:
        """Get a specific model by ID."""
        return self._cards.get(model_id)

    # -- Path resolution ----------------------------------------------------

    def model_path(self, card: ModelCard) -> Path:
        """Resolve a ModelCard to its absolute file path.

        Raises ValueError if the path escapes the models directory (path traversal).
        Recreates models_dir if it has been removed at runtime.
        """
        self._models_dir.mkdir(parents=True, exist_ok=True)
        path = (self._models_dir / card.relative_path).resolve()
        base = self._models_dir.resolve()
        try:
            path.is_relative_to(base)  # probe — raises AttributeError on < 3.9
            if not path.is_relative_to(base):
                raise ValueError(
                    f"Model path escapes models directory: {card.relative_path!r}"
                )
        except AttributeError:
            # Python < 3.9 fallback
            base_str = str(base)
            if not (str(path) == base_str or str(path).startswith(base_str + os.sep)):
                raise ValueError(
                    f"Model path escapes models directory: {card.relative_path!r}"
                )
        return path

    def check_status(self, card: ModelCard) -> ModelStatus:
        """Check whether the model file actually exists on disk and is not corrupt."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_path(card)
        if not path.exists():
            return ModelStatus.MISSING
        if path.is_file() and path.stat().st_size == 0:
            return ModelStatus.CORRUPT
        expected_hash = card.metadata.get("sha256")
        if expected_hash and path.is_file():
            import hashlib
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected_hash:
                return ModelStatus.CORRUPT
        return ModelStatus.AVAILABLE

    # -- Reference counting -------------------------------------------------

    def acquire(self, model_id: str) -> None:
        """Mark model as in-use. Prevents uninstall."""
        self._in_use.add(model_id)

    def release(self, model_id: str) -> None:
        """Release model from in-use."""
        self._in_use.discard(model_id)

    def is_in_use(self, model_id: str) -> bool:
        """Return True if the model is currently acquired."""
        return model_id in self._in_use

    # -- Introspection ------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._cards)

    @property
    def defaults(self) -> dict[ModelType, str]:
        """Current default model IDs by type. Read-only copy."""
        return dict(self._defaults)

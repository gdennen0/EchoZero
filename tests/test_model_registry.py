"""
Model registry tests: Catalog, resolution, persistence, and status checking.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from echozero.models.registry import (
    MANIFEST_FILENAME,
    ModelCard,
    ModelRegistry,
    ModelSource,
    ModelStatus,
    ModelType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card(
    model_id: str = "onset-v1",
    model_type: ModelType = ModelType.ONSET_DETECTION,
    version: str = "1.0.0",
    name: str = "Onset v1",
    relative_path: str = "onset/v1/model.pt",
) -> ModelCard:
    return ModelCard(
        id=model_id,
        model_type=model_type,
        name=name,
        version=version,
        source=ModelSource.BUILTIN,
        relative_path=relative_path,
    )


# ---------------------------------------------------------------------------
# ModelCard tests
# ---------------------------------------------------------------------------


class TestModelCard:
    def test_to_dict_roundtrip(self) -> None:
        card = _card()
        d = card.to_dict()
        restored = ModelCard.from_dict(d)
        assert restored == card

    def test_to_dict_serializes_enums(self) -> None:
        card = _card()
        d = card.to_dict()
        assert d["model_type"] == "onset_detection"
        assert d["source"] == "builtin"

    def test_frozen(self) -> None:
        card = _card()
        with pytest.raises(AttributeError):
            card.name = "changed"  # type: ignore


# ---------------------------------------------------------------------------
# Registry: registration and listing
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_and_get(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        card = _card()
        reg.register(card)
        assert reg.get("onset-v1") == card
        assert reg.count == 1

    def test_register_overwrites(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card(version="1.0.0"))
        reg.register(_card(version="2.0.0"))
        assert reg.get("onset-v1").version == "2.0.0"
        assert reg.count == 1

    def test_unregister(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card())
        removed = reg.unregister("onset-v1")
        assert removed is not None
        assert reg.count == 0

    def test_unregister_nonexistent(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        assert reg.unregister("ghost") is None

    def test_list_all(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card("onset-v1", ModelType.ONSET_DETECTION))
        reg.register(_card("class-v1", ModelType.CLASSIFICATION, name="Class v1", relative_path="class/v1/model.pt"))
        assert len(reg.list_models()) == 2

    def test_list_by_type(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card("onset-v1", ModelType.ONSET_DETECTION))
        reg.register(_card("class-v1", ModelType.CLASSIFICATION, name="Class v1", relative_path="class/v1/model.pt"))
        onset_models = reg.list_models(ModelType.ONSET_DETECTION)
        assert len(onset_models) == 1
        assert onset_models[0].id == "onset-v1"


# ---------------------------------------------------------------------------
# Registry: resolution
# ---------------------------------------------------------------------------


class TestResolution:
    def test_resolve_default(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card())
        resolved = reg.resolve(ModelType.ONSET_DETECTION)
        assert resolved is not None
        assert resolved.id == "onset-v1"

    def test_resolve_by_version(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card("onset-v1", version="1.0.0"))
        reg.register(_card("onset-v2", version="2.0.0", relative_path="onset/v2/model.pt"))
        resolved = reg.resolve(ModelType.ONSET_DETECTION, version="2.0.0")
        assert resolved is not None
        assert resolved.id == "onset-v2"

    def test_resolve_by_id(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card())
        resolved = reg.resolve(ModelType.ONSET_DETECTION, model_id="onset-v1")
        assert resolved is not None

    def test_resolve_missing_returns_none(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        assert reg.resolve(ModelType.CLASSIFICATION) is None

    def test_resolve_wrong_version_returns_none(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card(version="1.0.0"))
        assert reg.resolve(ModelType.ONSET_DETECTION, version="9.9.9") is None

    def test_auto_default_first_registered(self, tmp_path: Path) -> None:
        """First model of a type becomes the default automatically."""
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card("onset-v1"))
        reg.register(_card("onset-v2", relative_path="onset/v2/model.pt"))
        # First one is still default
        assert reg.defaults[ModelType.ONSET_DETECTION] == "onset-v1"

    def test_set_default(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card("onset-v1"))
        reg.register(_card("onset-v2", relative_path="onset/v2/model.pt"))
        reg.set_default(ModelType.ONSET_DETECTION, "onset-v2")
        resolved = reg.resolve(ModelType.ONSET_DETECTION)
        assert resolved.id == "onset-v2"

    def test_set_default_wrong_type_raises(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card("onset-v1", ModelType.ONSET_DETECTION))
        with pytest.raises(ValueError):
            reg.set_default(ModelType.CLASSIFICATION, "onset-v1")

    def test_set_default_unregistered_raises(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        with pytest.raises(KeyError):
            reg.set_default(ModelType.ONSET_DETECTION, "ghost")

    def test_unregister_default_promotes_next(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card("onset-v1"))
        reg.register(_card("onset-v2", relative_path="onset/v2/model.pt"))
        reg.unregister("onset-v1")
        # Default should shift to onset-v2
        assert reg.defaults[ModelType.ONSET_DETECTION] == "onset-v2"

    def test_unregister_last_clears_default(self, tmp_path: Path) -> None:
        reg = ModelRegistry(tmp_path / "models")
        reg.load()
        reg.register(_card("onset-v1"))
        reg.unregister("onset-v1")
        assert ModelType.ONSET_DETECTION not in reg.defaults


# ---------------------------------------------------------------------------
# Registry: persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        reg = ModelRegistry(models_dir)
        reg.load()
        reg.register(_card("onset-v1"))
        reg.register(_card("class-v1", ModelType.CLASSIFICATION, name="Class", relative_path="class/model.pt"))
        reg.save()

        # New instance, same dir
        reg2 = ModelRegistry(models_dir)
        reg2.load()
        assert reg2.count == 2
        assert reg2.get("onset-v1") is not None
        assert reg2.defaults[ModelType.ONSET_DETECTION] == "onset-v1"

    def test_load_creates_empty_manifest(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        reg = ModelRegistry(models_dir)
        reg.load()
        assert models_dir.exists()
        assert (models_dir / MANIFEST_FILENAME).exists()
        assert reg.count == 0

    def test_manifest_is_valid_json(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        reg = ModelRegistry(models_dir)
        reg.load()
        reg.register(_card())
        reg.save()

        data = json.loads((models_dir / MANIFEST_FILENAME).read_text())
        assert "version" in data
        assert "models" in data
        assert "defaults" in data


# ---------------------------------------------------------------------------
# Registry: status checking
# ---------------------------------------------------------------------------


class TestStatus:
    def test_available_when_file_exists(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        reg = ModelRegistry(models_dir)
        reg.load()
        card = _card(relative_path="onset/model.pt")
        reg.register(card)
        # Create the file
        (models_dir / "onset").mkdir(parents=True)
        (models_dir / "onset" / "model.pt").write_bytes(b"fake weights")

        assert reg.check_status(card) == ModelStatus.AVAILABLE

    def test_missing_when_file_absent(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        reg = ModelRegistry(models_dir)
        reg.load()
        card = _card(relative_path="onset/model.pt")
        reg.register(card)
        # Don't create the file
        assert reg.check_status(card) == ModelStatus.MISSING

    def test_model_path_resolution(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        reg = ModelRegistry(models_dir)
        card = _card(relative_path="onset/v1/model.pt")
        path = reg.model_path(card)
        assert path == (models_dir / "onset" / "v1" / "model.pt").resolve()


# ---------------------------------------------------------------------------
# Security: path traversal
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_path_traversal_in_model_path(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        reg = ModelRegistry(models_dir)
        card = _card(relative_path="../../etc/passwd")
        with pytest.raises(ValueError, match="escapes"):
            reg.model_path(card)

    def test_path_traversal_dotdot(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        reg = ModelRegistry(models_dir)
        card = _card(relative_path="..")
        with pytest.raises(ValueError, match="escapes"):
            reg.model_path(card)

    def test_safe_relative_path_works(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        reg = ModelRegistry(models_dir)
        card = _card(relative_path="onset/model.pt")
        path = reg.model_path(card)
        assert str(models_dir.resolve()) in str(path)


# ---------------------------------------------------------------------------
# Corrupt manifest recovery
# ---------------------------------------------------------------------------


class TestCorruptManifest:
    def test_load_corrupt_json_recovers(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        manifest = models_dir / MANIFEST_FILENAME
        manifest.write_text("THIS IS NOT JSON }{{{", encoding="utf-8")

        reg = ModelRegistry(models_dir)
        reg.load()  # should not raise
        assert reg.count == 0

    def test_load_bad_entry_skips_it(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        manifest = models_dir / MANIFEST_FILENAME
        data = {
            "version": 1,
            "models": {
                "good-model": {
                    "id": "good-model",
                    "model_type": "onset_detection",
                    "name": "Good",
                    "version": "1.0.0",
                    "source": "builtin",
                    "relative_path": "good/model.pt",
                    "description": "",
                    "framework": "pytorch",
                    "metadata": {},
                },
                "bad-model": {
                    "id": "bad-model",
                    "model_type": "onset_detection",
                    # missing required fields to trigger error
                },
            },
            "defaults": {},
        }
        manifest.write_text(json.dumps(data), encoding="utf-8")

        reg = ModelRegistry(models_dir)
        reg.load()
        # good entry loaded, bad entry skipped
        assert reg.get("good-model") is not None
        assert reg.get("bad-model") is None

    def test_load_unknown_enum_skips(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        manifest = models_dir / MANIFEST_FILENAME
        data = {
            "version": 1,
            "models": {
                "future-model": {
                    "id": "future-model",
                    "model_type": "future_type",  # unknown enum value
                    "name": "Future",
                    "version": "1.0.0",
                    "source": "builtin",
                    "relative_path": "future/model.pt",
                    "description": "",
                    "framework": "pytorch",
                    "metadata": {},
                },
            },
            "defaults": {},
        }
        manifest.write_text(json.dumps(data), encoding="utf-8")

        reg = ModelRegistry(models_dir)
        reg.load()  # should not raise, just skip
        assert reg.count == 0


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------


class TestAtomicSave:
    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        reg = ModelRegistry(models_dir)
        reg.load()
        reg.register(_card())
        reg.save()

        manifest = models_dir / MANIFEST_FILENAME
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert "models" in data
        assert "onset-v1" in data["models"]

    def test_save_no_tmp_file_left(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        reg = ModelRegistry(models_dir)
        reg.load()
        reg.register(_card())
        reg.save()

        tmp_file = models_dir / "models.tmp"
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# Locking / concurrency
# ---------------------------------------------------------------------------


class TestLocking:
    def test_concurrent_registers(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        reg = ModelRegistry(models_dir)
        reg.load()

        errors: list[Exception] = []

        def register_many(start: int) -> None:
            for i in range(start, start + 20):
                try:
                    reg.register(_card(
                        model_id=f"model-{i}",
                        relative_path=f"models/model-{i}.pt",
                    ))
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=register_many, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert reg.count == 100

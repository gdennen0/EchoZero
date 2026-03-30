"""
ModelProvider tests: Download, install, import, uninstall, and update checking.
All tests use LocalFileSource — no real HTTP, no HuggingFace dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from echozero.models.registry import (
    ModelCard,
    ModelRegistry,
    ModelSource,
    ModelStatus,
    ModelType,
)
from echozero.models.provider import (
    DownloadProgress,
    LocalFileSource,
    ModelProvider,
    ModelUpdate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingProgress:
    """Captures progress callbacks for assertions."""
    def __init__(self):
        self.events: list[DownloadProgress] = []

    def __call__(self, progress: DownloadProgress) -> None:
        self.events.append(progress)


class FakeRemoteSource:
    """Mock remote source that serves from a local directory and reports fake updates."""

    def __init__(self, source_dir: Path, updates: list[ModelUpdate] | None = None):
        self._local = LocalFileSource(source_dir)
        self._updates = updates or []

    def check_available(self, org: str, model_type: ModelType) -> list[ModelUpdate]:
        return [u for u in self._updates if u.model_type == model_type]

    def download(self, model_id: str, target_dir: Path, on_progress=None) -> Path:
        return self._local.download(model_id, target_dir, on_progress)

    def get_latest_version(self, org: str, model_type: ModelType) -> str | None:
        matching = [u for u in self._updates if u.model_type == model_type]
        if matching:
            return max(u.available_version for u in matching)
        return None


def _setup_registry(tmp_path: Path) -> tuple[ModelRegistry, Path]:
    models_dir = tmp_path / "models"
    reg = ModelRegistry(models_dir)
    reg.load()
    return reg, models_dir


# ---------------------------------------------------------------------------
# Install tests
# ---------------------------------------------------------------------------


class TestInstall:
    def test_install_from_remote(self, tmp_path: Path) -> None:
        reg, models_dir = _setup_registry(tmp_path)

        # Create a fake remote model
        remote_dir = tmp_path / "remote"
        model_dir = remote_dir / "onset-v2"
        model_dir.mkdir(parents=True)
        (model_dir / "model.pt").write_bytes(b"fake weights")
        (model_dir / "config.json").write_text('{"type": "onset"}')

        source = FakeRemoteSource(remote_dir)
        provider = ModelProvider(reg, source=source)

        card = provider.install(
            model_id="onset-v2",
            model_type=ModelType.ONSET_DETECTION,
            version="2.0.0",
            name="Onset Detection v2",
        )

        assert card.id == "onset-v2"
        assert card.version == "2.0.0"
        assert card.source == ModelSource.CLOUD
        assert reg.get("onset-v2") is not None
        assert reg.check_status(card) == ModelStatus.AVAILABLE

    def test_install_reports_progress(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)

        remote_dir = tmp_path / "remote"
        model_dir = remote_dir / "my-model"
        model_dir.mkdir(parents=True)
        (model_dir / "weights.pt").write_bytes(b"data")

        source = FakeRemoteSource(remote_dir)
        provider = ModelProvider(reg, source=source)
        progress = RecordingProgress()

        provider.install(
            model_id="my-model",
            model_type=ModelType.CLASSIFICATION,
            version="1.0.0",
            on_progress=progress,
        )

        statuses = [e.status for e in progress.events]
        assert "downloading" in statuses
        assert "complete" in statuses

    def test_install_sets_default(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)

        remote_dir = tmp_path / "remote"
        (remote_dir / "m1").mkdir(parents=True)
        (remote_dir / "m1" / "model.pt").write_bytes(b"v1")

        source = FakeRemoteSource(remote_dir)
        provider = ModelProvider(reg, source=source)

        provider.install("m1", ModelType.ONSET_DETECTION, "1.0.0")
        resolved = reg.resolve(ModelType.ONSET_DETECTION)
        assert resolved is not None
        assert resolved.id == "m1"

    def test_install_without_setting_default(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)

        remote_dir = tmp_path / "remote"
        (remote_dir / "m1").mkdir(parents=True)
        (remote_dir / "m1" / "model.pt").write_bytes(b"v1")
        (remote_dir / "m2").mkdir(parents=True)
        (remote_dir / "m2" / "model.pt").write_bytes(b"v2")

        source = FakeRemoteSource(remote_dir)
        provider = ModelProvider(reg, source=source)

        provider.install("m1", ModelType.ONSET_DETECTION, "1.0.0", set_default=True)
        provider.install("m2", ModelType.ONSET_DETECTION, "2.0.0", set_default=False)

        # Default should still be m1
        assert reg.defaults[ModelType.ONSET_DETECTION] == "m1"

    def test_install_persists_registry(self, tmp_path: Path) -> None:
        reg, models_dir = _setup_registry(tmp_path)

        remote_dir = tmp_path / "remote"
        (remote_dir / "m1").mkdir(parents=True)
        (remote_dir / "m1" / "model.pt").write_bytes(b"data")

        source = FakeRemoteSource(remote_dir)
        provider = ModelProvider(reg, source=source)
        provider.install("m1", ModelType.ONSET_DETECTION, "1.0.0")

        # Load fresh registry from disk
        reg2 = ModelRegistry(models_dir)
        reg2.load()
        assert reg2.get("m1") is not None


# ---------------------------------------------------------------------------
# Import local tests
# ---------------------------------------------------------------------------


class TestImportLocal:
    def test_import_file(self, tmp_path: Path) -> None:
        reg, models_dir = _setup_registry(tmp_path)
        provider = ModelProvider(reg)

        model_file = tmp_path / "my_model.pt"
        model_file.write_bytes(b"trained weights")

        card = provider.import_local(
            path=model_file,
            model_type=ModelType.CLASSIFICATION,
            name="My Classifier",
            version="1.0.0",
        )

        assert card.id == "my-classifier"
        assert card.source == ModelSource.IMPORTED
        assert reg.check_status(card) == ModelStatus.AVAILABLE

    def test_import_directory(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)
        provider = ModelProvider(reg)

        model_dir = tmp_path / "my_model_dir"
        model_dir.mkdir()
        (model_dir / "weights.pt").write_bytes(b"data")
        (model_dir / "config.json").write_text("{}")

        card = provider.import_local(
            path=model_dir,
            model_type=ModelType.SEPARATION,
            name="Custom Separator",
        )

        assert card.id == "custom-separator"
        assert reg.check_status(card) == ModelStatus.AVAILABLE

    def test_import_nonexistent_raises(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)
        provider = ModelProvider(reg)

        with pytest.raises(FileNotFoundError):
            provider.import_local(
                path=tmp_path / "ghost.pt",
                model_type=ModelType.CLASSIFICATION,
                name="Ghost",
            )

    def test_import_with_custom_id(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)
        provider = ModelProvider(reg)

        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"data")

        card = provider.import_local(
            path=model_file,
            model_type=ModelType.ONSET_DETECTION,
            name="Onset v3",
            model_id="custom-onset-v3",
        )

        assert card.id == "custom-onset-v3"


# ---------------------------------------------------------------------------
# Uninstall tests
# ---------------------------------------------------------------------------


class TestUninstall:
    def test_uninstall_removes_from_registry(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)
        provider = ModelProvider(reg)

        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"data")
        provider.import_local(model_file, ModelType.ONSET_DETECTION, "Test")

        result = provider.uninstall("test")
        assert result is True
        assert reg.get("test") is None

    def test_uninstall_deletes_files(self, tmp_path: Path) -> None:
        reg, models_dir = _setup_registry(tmp_path)
        provider = ModelProvider(reg)

        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"data")
        card = provider.import_local(model_file, ModelType.ONSET_DETECTION, "Test")

        model_path = reg.model_path(card)
        assert model_path.exists()

        provider.uninstall("test", delete_files=True)
        assert not model_path.exists()

    def test_uninstall_keeps_files_if_requested(self, tmp_path: Path) -> None:
        reg, models_dir = _setup_registry(tmp_path)
        provider = ModelProvider(reg)

        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"data")
        card = provider.import_local(model_file, ModelType.ONSET_DETECTION, "Test")

        model_path = reg.model_path(card)
        provider.uninstall("test", delete_files=False)
        assert model_path.exists()  # file preserved

    def test_uninstall_nonexistent_returns_false(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)
        provider = ModelProvider(reg)
        assert provider.uninstall("ghost") is False


# ---------------------------------------------------------------------------
# Update check tests
# ---------------------------------------------------------------------------


class TestCheckUpdates:
    def test_detects_new_model(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)

        updates = [ModelUpdate(
            model_type=ModelType.ONSET_DETECTION,
            current_version=None,
            available_version="2.0.0",
            model_id="echozero/onset-v2",
            size_bytes=50_000_000,
        )]
        source = FakeRemoteSource(tmp_path, updates=updates)
        provider = ModelProvider(reg, source=source)

        result = provider.check_updates(ModelType.ONSET_DETECTION)
        assert len(result) == 1
        assert result[0].available_version == "2.0.0"
        assert result[0].current_version is None

    def test_detects_newer_version(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)

        # Install v1 locally
        reg.register(ModelCard(
            id="onset-v1",
            model_type=ModelType.ONSET_DETECTION,
            name="Onset v1",
            version="1.0.0",
            source=ModelSource.BUILTIN,
            relative_path="onset/model.pt",
        ))

        updates = [ModelUpdate(
            model_type=ModelType.ONSET_DETECTION,
            current_version=None,
            available_version="2.0.0",
            model_id="echozero/onset-v2",
        )]
        source = FakeRemoteSource(tmp_path, updates=updates)
        provider = ModelProvider(reg, source=source)

        result = provider.check_updates(ModelType.ONSET_DETECTION)
        assert len(result) == 1
        assert result[0].current_version == "1.0.0"
        assert result[0].available_version == "2.0.0"

    def test_no_update_when_current(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)

        reg.register(ModelCard(
            id="onset-v2",
            model_type=ModelType.ONSET_DETECTION,
            name="Onset v2",
            version="2.0.0",
            source=ModelSource.BUILTIN,
            relative_path="onset/model.pt",
        ))

        updates = [ModelUpdate(
            model_type=ModelType.ONSET_DETECTION,
            current_version=None,
            available_version="2.0.0",  # same as local
            model_id="echozero/onset-v2",
        )]
        source = FakeRemoteSource(tmp_path, updates=updates)
        provider = ModelProvider(reg, source=source)

        result = provider.check_updates(ModelType.ONSET_DETECTION)
        assert len(result) == 0  # no update needed

    def test_check_all_types(self, tmp_path: Path) -> None:
        reg, _ = _setup_registry(tmp_path)

        updates = [
            ModelUpdate(ModelType.ONSET_DETECTION, None, "1.0.0", "echozero/onset"),
            ModelUpdate(ModelType.CLASSIFICATION, None, "1.0.0", "echozero/classify"),
        ]
        source = FakeRemoteSource(tmp_path, updates=updates)
        provider = ModelProvider(reg, source=source)

        result = provider.check_updates()  # all types
        assert len(result) == 2


# ---------------------------------------------------------------------------
# DownloadProgress tests
# ---------------------------------------------------------------------------


class TestDownloadProgress:
    def test_fraction_calculation(self) -> None:
        p = DownloadProgress("m1", bytes_downloaded=50, bytes_total=100)
        assert p.fraction == 0.5

    def test_fraction_none_when_unknown(self) -> None:
        p = DownloadProgress("m1", bytes_downloaded=50, bytes_total=None)
        assert p.fraction is None

    def test_fraction_none_when_zero_total(self) -> None:
        p = DownloadProgress("m1", bytes_downloaded=0, bytes_total=0)
        assert p.fraction is None

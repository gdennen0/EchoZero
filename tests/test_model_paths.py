from __future__ import annotations

from pathlib import Path

from echozero.models.paths import ensure_installed_models_dir, installed_models_dir


def test_installed_models_dir_uses_echozero_user_root() -> None:
    path = installed_models_dir()

    assert path == Path.home() / ".echozero" / "models"


def test_ensure_installed_models_dir_creates_directory() -> None:
    path = ensure_installed_models_dir()

    assert path.exists()
    assert path.is_dir()

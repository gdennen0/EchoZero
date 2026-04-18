from __future__ import annotations

from pathlib import Path


def installed_models_dir() -> Path:
    """Canonical app-managed runtime model install directory."""
    return Path.home() / ".echozero" / "models"


def ensure_installed_models_dir() -> Path:
    path = installed_models_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path

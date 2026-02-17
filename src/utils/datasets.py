"""
Dataset location and migration helpers.

Centralizes where training datasets live and provides a safe one-time migration
path from legacy repo-local datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Optional

from src.utils.paths import get_app_install_dir, get_user_data_dir
from src.utils.message import Log


@dataclass
class DatasetMigrationResult:
    """Result metadata for dataset migration attempts."""

    migrated: bool
    source: Optional[Path]
    destination: Path
    message: str


def get_managed_datasets_dir() -> Path:
    """
    Return the canonical managed datasets directory in user data storage.

    Example on macOS: ~/Library/Application Support/EchoZero/datasets
    """
    base = get_user_data_dir() / "datasets"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_managed_external_datasets_dir() -> Path:
    """Return canonical directory for extracted external datasets."""
    path = get_managed_datasets_dir() / "external"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_managed_raw_datasets_dir() -> Path:
    """Return canonical directory for raw dataset archives."""
    path = get_managed_datasets_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_default_idmt_dataset_root() -> Path:
    """Return canonical IDMT-SMT-DRUMS extracted dataset root."""
    return get_managed_external_datasets_dir() / "idmt_smt_drums"


def resolve_dataset_path(path_or_name: Optional[str]) -> Optional[str]:
    """
    Resolve a dataset path to the managed datasets location when possible.

    Rules:
    - If input is empty, return unchanged.
    - If input points inside legacy repo-local data/datasets, map to managed path.
    - If input is a dataset key like "idmt_smt_drums", map to managed external path.
    - Otherwise, return unchanged.
    """
    if not path_or_name:
        return path_or_name

    raw = str(path_or_name).strip()
    if not raw:
        return raw

    # Dataset key support: "idmt_smt_drums"
    if raw == "idmt_smt_drums":
        return str(get_default_idmt_dataset_root())

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        return raw

    install_dir = get_app_install_dir()
    if install_dir is None:
        return str(candidate)

    legacy_root = install_dir / "data" / "datasets"
    try:
        rel = candidate.resolve().relative_to(legacy_root.resolve())
    except Exception:
        return str(candidate)

    return str(get_managed_datasets_dir() / rel)


def migrate_legacy_repo_datasets() -> DatasetMigrationResult:
    """
    One-time safe migration from repo-local data/datasets to managed user storage.

    This migration copies only missing files/directories (non-destructive).
    """
    destination = get_managed_datasets_dir()
    install_dir = get_app_install_dir()
    if install_dir is None:
        return DatasetMigrationResult(
            migrated=False,
            source=None,
            destination=destination,
            message="App install directory unavailable; skipped dataset migration.",
        )

    source = install_dir / "data" / "datasets"
    if not source.exists() or not source.is_dir():
        return DatasetMigrationResult(
            migrated=False,
            source=source,
            destination=destination,
            message="No legacy repo-local datasets found.",
        )

    copied_any = False
    for entry in source.iterdir():
        src = source / entry.name
        dst = destination / entry.name
        if src.is_dir():
            if not dst.exists():
                shutil.copytree(src, dst)
                copied_any = True
        else:
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied_any = True

    if copied_any:
        msg = f"Migrated legacy datasets from {source} to {destination}"
        Log.info(msg)
        return DatasetMigrationResult(
            migrated=True,
            source=source,
            destination=destination,
            message=msg,
        )

    return DatasetMigrationResult(
        migrated=False,
        source=source,
        destination=destination,
        message="Managed dataset directory already populated; no migration needed.",
    )

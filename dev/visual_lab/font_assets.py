"""Visual Lab font assets.
Exists to keep imported editor fonts in a support-only lab-managed directory.
Copies and optionally registers font files without touching production runtime paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from PyQt6.QtGui import QFontDatabase

FONT_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
SUPPORTED_FONT_EXTENSIONS = frozenset({".ttf", ".otf", ".ttc"})


@dataclass(frozen=True, slots=True)
class ImportedFont:
    """One font copied into the Visual Lab font asset directory."""

    source_path: Path
    asset_path: Path
    application_font_id: int | None
    families: tuple[str, ...]


def import_lab_fonts(
    font_paths: list[str | Path],
    *,
    asset_dir: str | Path | None = None,
    register: bool = True,
) -> tuple[ImportedFont, ...]:
    """Copy supported font files into the lab asset directory and register them."""
    target_dir = Path(asset_dir) if asset_dir is not None else FONT_ASSET_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    imported: list[ImportedFont] = []
    for font_path in font_paths:
        source_path = _validate_font_path(Path(font_path))
        asset_path = _copy_font_asset(source_path, target_dir)
        application_font_id = None
        families: tuple[str, ...] = ()
        if register:
            application_font_id = QFontDatabase.addApplicationFont(str(asset_path))
            if application_font_id >= 0:
                families = tuple(QFontDatabase.applicationFontFamilies(application_font_id))
        imported.append(
            ImportedFont(
                source_path=source_path,
                asset_path=asset_path,
                application_font_id=application_font_id,
                families=families,
            )
        )
    return tuple(imported)


def list_lab_font_assets(asset_dir: str | Path | None = None) -> tuple[Path, ...]:
    """Return copied Visual Lab font assets in stable name order."""
    target_dir = Path(asset_dir) if asset_dir is not None else FONT_ASSET_DIR
    if not target_dir.exists():
        return ()
    return tuple(
        sorted(
            path
            for path in target_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_FONT_EXTENSIONS
        )
    )


def _validate_font_path(path: Path) -> Path:
    if path.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_FONT_EXTENSIONS))
        raise ValueError(f"font import only supports {allowed}: {path}")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"font file does not exist: {path}")
    return path


def _copy_font_asset(source_path: Path, target_dir: Path) -> Path:
    target_path = target_dir / source_path.name
    if target_path.resolve() == source_path.resolve():
        return target_path
    if target_path.exists():
        target_path = _deduplicated_target_path(target_dir, source_path)
    shutil.copy2(source_path, target_path)
    return target_path


def _deduplicated_target_path(target_dir: Path, source_path: Path) -> Path:
    index = 2
    while True:
        target_path = target_dir / f"{source_path.stem}-{index}{source_path.suffix}"
        if not target_path.exists():
            return target_path
        index += 1

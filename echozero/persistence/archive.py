"""
Archive: .ez file format — ZIP-based project archive with atomic write.
Exists because projects need a portable, single-file format for save/share/backup.
Pack on explicit save, unpack on open. Audio files use STORED compression (already compressed).
"""

from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1
APP_VERSION = "2.0.0"


def pack_ez(working_dir: Path, dest_path: Path) -> None:
    """Pack a working directory into a .ez archive.

    Uses atomic write: writes to .tmp first, then renames.
    Audio files use STORED compression (already compressed).
    DB and manifest use DEFLATED compression.

    Structure:
        project.ez (ZIP)
        +-- manifest.json
        +-- project.db
        +-- audio/
            +-- a3f2c8d1....wav
            +-- 91bd4e7a....wav
    """
    tmp_path = dest_path.with_suffix(".ez.tmp")

    manifest = {
        "format_version": MANIFEST_VERSION,
        "app_version": APP_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with zipfile.ZipFile(tmp_path, "w") as zf:
            # Manifest
            zf.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )

            # Database
            db_path = working_dir / "project.db"
            if db_path.exists():
                zf.write(db_path, "project.db", compress_type=zipfile.ZIP_DEFLATED)

            # Audio files (STORED — already compressed)
            audio_dir = working_dir / "audio"
            if audio_dir.exists():
                for audio_file in sorted(audio_dir.rglob('*')):
                    if audio_file.is_file():
                        rel_path = audio_file.relative_to(working_dir)
                        zf.write(
                            audio_file,
                            str(rel_path).replace('\\', '/'),
                            compress_type=zipfile.ZIP_STORED,
                        )

        # Atomic rename
        tmp_path.replace(dest_path)
    except Exception:
        # Clean up tmp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def unpack_ez(ez_path: Path, working_dir: Path) -> dict:
    """Unpack a .ez archive to a working directory.

    Extracts to a temp directory first for atomicity — if extraction fails,
    the working_dir is NOT created or left in partial state.

    Args:
        ez_path: Path to the .ez archive
        working_dir: Target working directory (created if needed)

    Returns:
        The manifest dict from manifest.json

    Raises:
        ValueError if archive is invalid, manifest is missing, or zip-slip detected
        FileNotFoundError if ez_path doesn't exist
    """
    if not ez_path.exists():
        raise FileNotFoundError(f"Archive not found: {ez_path}")

    # Extract to temp dir first for atomicity (P2)
    parent = working_dir.parent
    parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = parent / f"tmp_unpack_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(ez_path, "r") as zf:
            # Validate manifest exists
            if "manifest.json" not in zf.namelist():
                raise ValueError(f"Invalid .ez archive: no manifest.json in {ez_path}")

            # Zip-slip validation (P1/A1): check every member stays within tmp_dir
            tmp_dir_resolved = tmp_dir.resolve()
            for member in zf.namelist():
                member_path = (tmp_dir / member).resolve()
                if not member_path.is_relative_to(tmp_dir_resolved):
                    raise ValueError(
                        f"Archive contains path traversal: {member!r}"
                    )

            # Read manifest
            manifest_data = zf.read("manifest.json")
            manifest = json.loads(manifest_data)

            # Check format version
            fmt_version = manifest.get("format_version", 0)
            if fmt_version > MANIFEST_VERSION:
                raise ValueError(
                    f"Archive format version {fmt_version} is newer than supported "
                    f"({MANIFEST_VERSION}). Please update EchoZero."
                )

            # Extract to temp dir
            zf.extractall(tmp_dir)

        # Atomic move from temp to final location
        if working_dir.exists():
            shutil.rmtree(working_dir)
        shutil.move(str(tmp_dir), str(working_dir))

    except Exception:
        # Clean up temp dir on failure
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return manifest


def is_valid_ez(ez_path: Path) -> bool:
    """Check if a file is a valid .ez archive (has manifest.json and project.db)."""
    try:
        if not ez_path.exists():
            return False
        with zipfile.ZipFile(ez_path, "r") as zf:
            names = zf.namelist()
            return "manifest.json" in names and "project.db" in names
    except (zipfile.BadZipFile, OSError):
        return False

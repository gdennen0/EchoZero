#!/usr/bin/env python3
"""Build a portable EchoZero MA3 harness transfer package.

Exists to make the MA3-side Lua plugins, harness tools, docs, and validation
steps easy to move to a fresh workstation without relying on tribal setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

PACKAGE_ROOT_NAME = "echozero-ma3-harness"
SCHEMA = "echozero.ma3-harness-transfer.v1"
DEFAULT_OUTPUT = Path("artifacts") / "ma3-harness-transfer" / "echozero-ma3-harness-transfer.zip"
FIXED_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)

EZ_CANONICAL_NAME_MAP = {
    "echozero.lua": "ez_core.lua",
    "echozero_debug.lua": "ez_debug.lua",
    "echozero_init.lua": "ez_init.lua",
    "echozero_osc.lua": "ez_osc.lua",
    "echozero_print.lua": "ez_print.lua",
    "presets.lua": "ez_presets.lua",
    "Sequence.lua": "ez_sequence.lua",
    "timecode.lua": "ez_timecode.lua",
}

EZ_PLUGIN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<GMA3 DataVersion="2.3.2.0">
    <UserPlugin Name="Ez#2" Guid="46 39 8C 92 02 3E FD 10 1E BE E8 09 E8 1D 15 BA" Version="0.0.0.0" UserRights="Admin">
        <ComponentLua Name="Init" Guid="46 39 8C 92 E4 CE 1F 5E D3 44 D1 0E E9 1D 15 BA" FileName="ez_init.lua" FilePath="EZ" Installed="Yes"/>
        <ComponentLua Name="OSC" Guid="46 39 8C 92 42 D7 A9 7E 88 CB B9 13 C1 1D 15 BA" FileName="ez_osc.lua" FilePath="EZ" Installed="Yes" UserRights="Admin"/>
        <ComponentLua Name="Core" Guid="46 39 8C 92 E5 13 1E 3C 3D 52 A2 18 C8 1D 15 BA" FileName="ez_core.lua" FilePath="EZ" Installed="Yes"/>
        <ComponentLua Name="Debug" Guid="46 39 8C 92 8C DA 55 56 F2 D8 8A 1D CD 1D 15 BA" FileName="ez_debug.lua" FilePath="EZ" Installed="Yes"/>
        <ComponentLua Name="Print" Guid="46 39 8C 92 98 1B A8 1A A7 5F 73 22 36 1E 15 BA" FileName="ez_print.lua" FilePath="EZ" Installed="Yes"/>
        <ComponentLua Name="Sequence" Guid="46 39 8C 92 08 E0 4A 0E 5C E6 5B 27 3E 1E 15 BA" FileName="ez_sequence.lua" FilePath="EZ" Installed="Yes"/>
        <ComponentLua Name="Presets" Guid="46 39 8C 92 74 51 9E 14 DA 29 80 2A 4C 1E 15 BA" FileName="ez_presets.lua" FilePath="EZ" Installed="Yes"/>
        <ComponentLua Name="Timecode" Guid="46 39 8C 92 B0 F3 19 5A 11 6D 44 2C 23 1E 15 BA" FileName="ez_timecode.lua" FilePath="EZ" Installed="Yes"/>
    </UserPlugin>
</GMA3>
"""

TC22_PLUGIN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<GMA3 DataVersion="2.3.2.0">
    <UserPlugin Name="TC22" Guid="54 43 32 32 20 26 05 15 10 21 00 00 00 00 00 01" Version="0.1.0.0" UserRights="Admin">
        <ComponentLua Name="AutosaveShowfile" Guid="54 43 32 32 20 26 05 15 10 21 00 00 00 00 00 02" FileName="autosave_showfile.lua" FilePath="TC22" Installed="Yes" UserRights="Admin"/>
    </UserPlugin>
</GMA3>
"""

INSTALL_README = """# EchoZero MA3 Harness Transfer Package

This package contains the MA3-side EchoZero harness/plugin payload plus the
repo harness scripts and docs needed to validate it on a new machine.

## Contents

- `grandMA3/datapools/plugins/` - copy-ready MA3 plugin folders.
- `grandMA3/datapools/plugins/*.xml` - plugin wrappers MA3 uses to register
  the copied Lua payloads after `RP`/reload.
  - `EZ/` contains the EchoZero Lua modules using the canonical live bundle
    filenames (`ez_core.lua`, `ez_osc.lua`, etc.).
  - `HitMaker/` and `TC22/` are copied from the repo plugin sources.
- `source/MA3/plugins/` - original repo plugin sources, preserved for editing
  and diffing.
- `source/MA3/dev/` - Python harness/validation utilities.
- `source/MA3/docs/` plus `source/MA3/README.md` - reference docs.
- `manifest.json` - file list, SHA-256 hashes, install paths, and validation
  commands for this package.

## Install on the new machine

1. Install or clone EchoZero on the target machine and set up its Python env.
   The MA3 harness CLI imports EchoZero Python modules, so validation should be
   run from an EchoZero checkout or editable install.
2. Copy the contents of `grandMA3/datapools/plugins/` into the target MA3 plugin
   root, usually:

   ```text
   ~/MALightingTechnology/gma3_library/datapools/plugins/
   ```

   After copying, the target should contain folders such as:

   ```text
   ~/MALightingTechnology/gma3_library/datapools/plugins/Ez#2.xml
   ~/MALightingTechnology/gma3_library/datapools/plugins/TC22.xml
   ~/MALightingTechnology/gma3_library/datapools/plugins/EZ/
   ~/MALightingTechnology/gma3_library/datapools/plugins/HitMaker/
   ~/MALightingTechnology/gma3_library/datapools/plugins/TC22/
   ```

3. In grandMA3, reload plugins:

   ```text
   RP
   ```

4. From the EchoZero repo on the new machine, run the smoke validation:

   ```bash
   python MA3/dev/ma3_harness_cli.py --json smoke
   ```

   If the MA3 target is not already configured in EchoZero settings, pass it
   explicitly, for example:

   ```bash
   python MA3/dev/ma3_harness_cli.py --json --ma3-host 127.0.0.1 --ma3-port 8000 smoke
   ```

5. For a fuller evidence bundle, run:

   ```bash
   python MA3/dev/ma3_harness_cli.py --json validation-report
   ```

## Notes

- This package intentionally excludes macOS `.DS_Store`, Python bytecode,
  caches, and generated artifact folders.
- The `EZ/` install payload is derived from the current repo Lua files and
  renamed to match the live MA3 bundle naming convention documented in
  `source/MA3/README.md`.
- The original source filenames are preserved under `source/MA3/plugins/`.
"""


@dataclass(frozen=True)
class PackageResult:
    """Summary returned after building one transfer package."""

    output_path: Path
    manifest_path: Path
    file_count: int
    archive_root: str


def repo_root_from_script() -> Path:
    """Resolve the EchoZero repository root from this script location."""

    return Path(__file__).resolve().parents[2]


def should_skip(path: Path) -> bool:
    """Return True for files/directories that should never enter the package."""

    ignored_names = {".DS_Store", "__pycache__", ".pytest_cache", ".mypy_cache"}
    if any(part in ignored_names for part in path.parts):
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return False


def copy_tree_filtered(source: Path, destination: Path) -> list[Path]:
    """Copy source into destination while excluding cache/noise files."""

    copied: list[Path] = []
    if not source.exists():
        return copied
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if should_skip(relative):
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def copy_file(source: Path, destination: Path) -> None:
    """Copy one file, creating parents first."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_grandma3_plugin_payload(ma3_root: Path, package_root: Path) -> None:
    """Build a copy-ready grandMA3 datapools/plugins payload."""

    plugins_root = ma3_root / "plugins"
    if not plugins_root.is_dir():
        raise FileNotFoundError(f"MA3 plugins directory was not found: {plugins_root}")

    install_root = package_root / "grandMA3" / "datapools" / "plugins"
    ez_install_root = install_root / "EZ"
    ez_install_root.mkdir(parents=True, exist_ok=True)
    (install_root / "Ez#2.xml").write_text(EZ_PLUGIN_XML, encoding="utf-8")
    (install_root / "TC22.xml").write_text(TC22_PLUGIN_XML, encoding="utf-8")

    for source_name, install_name in sorted(EZ_CANONICAL_NAME_MAP.items()):
        source = plugins_root / source_name
        if not source.is_file():
            raise FileNotFoundError(f"Required EZ plugin source is missing: {source}")
        copy_file(source, ez_install_root / install_name)

    for plugin_dir_name in ("HitMaker", "TC22"):
        source_dir = plugins_root / plugin_dir_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Required MA3 plugin directory is missing: {source_dir}")
        copy_tree_filtered(source_dir, install_root / plugin_dir_name)


def build_source_payload(ma3_root: Path, package_root: Path) -> None:
    """Copy editable MA3 source docs, plugins, and harness scripts."""

    source_root = package_root / "source" / "MA3"
    for file_name in ("README.md", "MA3_INTEGRATION_PITFALLS.md"):
        source = ma3_root / file_name
        if source.is_file():
            copy_file(source, source_root / file_name)

    for dir_name in ("plugins", "dev", "docs"):
        source_dir = ma3_root / dir_name
        if source_dir.is_dir():
            copy_tree_filtered(source_dir, source_root / dir_name)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_role(relative_path: str) -> str:
    """Classify one package file for the manifest."""

    if relative_path == "INSTALL.md":
        return "package_metadata"
    if relative_path.startswith("grandMA3/datapools/plugins/"):
        return "grandma3_install_payload"
    if relative_path.startswith("source/MA3/plugins/"):
        return "source_plugin"
    if relative_path.startswith("source/MA3/dev/"):
        return "harness_tool"
    if relative_path.startswith("source/MA3/docs/") or relative_path.startswith("source/MA3/README"):
        return "documentation"
    return "source"


def build_manifest(package_root: Path, *, repo_root: Path) -> dict[str, object]:
    """Build the transfer manifest for all files currently staged."""

    files: list[dict[str, object]] = []
    for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
        relative = path.relative_to(package_root).as_posix()
        if relative == "manifest.json":
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": classify_role(relative),
            }
        )

    return {
        "schema": SCHEMA,
        "package_name": PACKAGE_ROOT_NAME,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "repo_root": str(repo_root),
            "ma3_root": str(repo_root / "MA3"),
        },
        "install": {
            "copy_payload_from": "grandMA3/datapools/plugins/",
            "copy_payload_to": "~/MALightingTechnology/gma3_library/datapools/plugins/",
            "reload_command": "RP",
            "validation_command": "python MA3/dev/ma3_harness_cli.py --json smoke",
            "validation_report_command": "python MA3/dev/ma3_harness_cli.py --json validation-report",
        },
        "contents": {
            "archive_root": PACKAGE_ROOT_NAME,
            "file_count": len(files),
            "files": files,
        },
    }


def write_zip_from_directory(source_dir: Path, output_path: Path) -> None:
    """Write a stable zip archive from a staged package directory."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(item for item in source_dir.rglob("*") if item.is_file()):
            arcname = path.relative_to(source_dir.parent).as_posix()
            info = ZipInfo(arcname, date_time=FIXED_ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def build_transfer_package(
    output_path: Path,
    *,
    repo_root: Path | None = None,
    force: bool = False,
    keep_staging_dir: Path | None = None,
) -> PackageResult:
    """Build one MA3 harness transfer zip and return its summary."""

    resolved_repo_root = (repo_root or repo_root_from_script()).expanduser().resolve()
    ma3_root = resolved_repo_root / "MA3"
    if not ma3_root.is_dir():
        raise FileNotFoundError(f"MA3 directory was not found: {ma3_root}")

    resolved_output = output_path.expanduser().resolve()
    if resolved_output.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing package: {resolved_output}")

    def stage_into(staging_parent: Path) -> PackageResult:
        package_root = staging_parent / PACKAGE_ROOT_NAME
        if package_root.exists():
            shutil.rmtree(package_root)
        package_root.mkdir(parents=True)
        (package_root / "INSTALL.md").write_text(INSTALL_README, encoding="utf-8")
        build_grandma3_plugin_payload(ma3_root, package_root)
        build_source_payload(ma3_root, package_root)
        manifest = build_manifest(package_root, repo_root=resolved_repo_root)
        manifest_path = package_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_zip_from_directory(package_root, resolved_output)
        return PackageResult(
            output_path=resolved_output,
            manifest_path=manifest_path,
            file_count=int(manifest["contents"]["file_count"]),
            archive_root=PACKAGE_ROOT_NAME,
        )

    if keep_staging_dir is not None:
        staging_parent = keep_staging_dir.expanduser().resolve()
        staging_parent.mkdir(parents=True, exist_ok=True)
        return stage_into(staging_parent)

    with tempfile.TemporaryDirectory(prefix="echozero_ma3_harness_package_") as temp_dir_text:
        return stage_into(Path(temp_dir_text))


def build_parser() -> argparse.ArgumentParser:
    """Build the package CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build a portable EchoZero MA3 harness/plugin transfer package."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination .zip path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override the EchoZero repo root. Defaults to this script's repo.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Optional directory to keep the unpacked package tree for inspection.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output archive.",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the transfer package CLI."""

    parser = build_parser()
    parsed = parser.parse_args(argv)
    result = build_transfer_package(
        parsed.output,
        repo_root=parsed.repo_root,
        force=parsed.force,
        keep_staging_dir=parsed.staging_dir,
    )
    manifest_location = (
        str(result.manifest_path)
        if parsed.staging_dir is not None
        else f"{result.archive_root}/manifest.json"
    )
    payload = {
        "status": "ok",
        "archive_path": str(result.output_path),
        "archive_root": result.archive_root,
        "manifest_path": manifest_location,
        "manifest_location": "staging_dir" if parsed.staging_dir is not None else "inside_archive",
        "file_count": result.file_count,
        "validation_command": "python MA3/dev/ma3_harness_cli.py --json smoke",
    }
    if parsed.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"archive={result.output_path}")
        print(f"archive_root={result.archive_root}")
        print(f"file_count={result.file_count}")
        print(f"validation_command={payload['validation_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

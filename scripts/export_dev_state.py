"""
Export EchoZero machine-local dev state for cross-machine handoff.
Exists to make Stage Zero runtime state portable without relying on tribal setup.
Connects canonical models/settings paths to one explicit archive artifact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from echozero.infrastructure.settings.json_store import JsonAppSettingsStore
from echozero.models.paths import installed_models_dir


@dataclass(frozen=True, slots=True)
class ExportSelection:
    """Requested source paths for one dev-state export."""

    settings_path: Path | None
    models_dir: Path | None


def resolve_default_settings_export_path() -> Path:
    """Resolve the active app-settings file path to export by default."""

    store = JsonAppSettingsStore()
    store.load()
    return store.path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for dev-state export."""

    parser = argparse.ArgumentParser(
        description="Export EchoZero dev state (settings + installed models) to a zip archive."
    )
    parser.add_argument("output", type=Path, help="Destination .zip archive path.")
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=None,
        help="Override the app settings JSON path to export.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Override the installed models directory to export.",
    )
    parser.add_argument(
        "--skip-settings",
        action="store_true",
        help="Do not export the machine-local app settings JSON.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Do not export the installed runtime models directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output archive.",
    )
    return parser


def export_dev_state(
    output_path: Path,
    *,
    selection: ExportSelection,
    force: bool = False,
) -> dict[str, object]:
    """Write one explicit EchoZero dev-state archive."""

    archive_path = output_path.expanduser().resolve()
    if archive_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    settings_path = (
        None if selection.settings_path is None else selection.settings_path.expanduser().resolve()
    )
    models_dir = (
        None if selection.models_dir is None else selection.models_dir.expanduser().resolve()
    )

    exported_settings = settings_path is not None and settings_path.is_file()
    exported_models = models_dir is not None and models_dir.is_dir()
    if not exported_settings and not exported_models:
        raise FileNotFoundError("No selected EchoZero dev-state sources were found to export.")

    manifest = {
        "schema": "echozero.dev-state.v1",
        "components": {
            "settings": {
                "selected": settings_path is not None,
                "exported": exported_settings,
                "source_path": None if settings_path is None else str(settings_path),
                "archive_path": "settings/app-settings.json" if exported_settings else None,
            },
            "models": {
                "selected": models_dir is not None,
                "exported": exported_models,
                "source_path": None if models_dir is None else str(models_dir),
                "archive_prefix": "models/" if exported_models else None,
            },
        },
    }

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        if exported_settings and settings_path is not None:
            archive.write(settings_path, arcname="settings/app-settings.json")
        if exported_models and models_dir is not None:
            for file_path in sorted(models_dir.rglob("*")):
                if not file_path.is_file():
                    continue
                archive.write(
                    file_path, arcname=str(Path("models") / file_path.relative_to(models_dir))
                )

    return {
        "archive_path": archive_path,
        "exported_settings": exported_settings,
        "exported_models": exported_models,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the dev-state export CLI."""

    parser = build_parser()
    parsed = parser.parse_args(argv)

    settings_path = (
        None
        if parsed.skip_settings
        else (parsed.settings_path or resolve_default_settings_export_path())
    )
    models_dir = None if parsed.skip_models else (parsed.models_dir or installed_models_dir())
    result = export_dev_state(
        parsed.output,
        selection=ExportSelection(
            settings_path=settings_path,
            models_dir=models_dir,
        ),
        force=parsed.force,
    )
    print(f"archive={result['archive_path']}")
    print(f"exported_settings={result['exported_settings']}")
    print(f"exported_models={result['exported_models']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

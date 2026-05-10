"""
Import EchoZero machine-local dev state from an explicit archive artifact.
Exists to restore Stage Zero runtime state on another laptop without hidden setup.
Connects the portable dev-state archive to canonical settings and models paths.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

from echozero.infrastructure.settings.json_store import default_app_settings_path
from echozero.models.paths import installed_models_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for dev-state import."""

    parser = argparse.ArgumentParser(
        description="Import EchoZero dev state (settings + installed models) from a zip archive."
    )
    parser.add_argument("archive", type=Path, help="Source dev-state .zip archive path.")
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=None,
        help="Override the destination app settings JSON path.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Override the destination installed models directory.",
    )
    parser.add_argument(
        "--skip-settings",
        action="store_true",
        help="Do not import the machine-local app settings JSON.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Do not import the installed runtime models directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination settings/models content.",
    )
    return parser


def import_dev_state(
    archive_path: Path,
    *,
    settings_path: Path | None,
    models_dir: Path | None,
    import_settings: bool,
    import_models: bool,
    force: bool = False,
) -> dict[str, object]:
    """Restore one explicit EchoZero dev-state archive."""

    source_archive = archive_path.expanduser().resolve()
    if not source_archive.is_file():
        raise FileNotFoundError(f"Dev-state archive was not found: {source_archive}")

    destination_settings = None if settings_path is None else settings_path.expanduser().resolve()
    destination_models = None if models_dir is None else models_dir.expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="echozero_dev_state_") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        with ZipFile(source_archive) as archive:
            archive.extractall(temp_dir)

        manifest_path = temp_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Dev-state archive is missing manifest.json.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "echozero.dev-state.v1":
            raise ValueError("Unsupported dev-state archive schema.")

        imported_settings = False
        imported_models = False

        if import_settings and destination_settings is not None:
            settings_source = temp_dir / "settings" / "app-settings.json"
            if settings_source.is_file():
                if destination_settings.exists() and not force:
                    raise FileExistsError(
                        f"Refusing to overwrite existing settings file: {destination_settings}"
                    )
                destination_settings.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(settings_source, destination_settings)
                imported_settings = True

        if import_models and destination_models is not None:
            models_source = temp_dir / "models"
            if models_source.is_dir():
                if destination_models.exists():
                    has_existing_content = any(destination_models.iterdir())
                    if has_existing_content and not force:
                        raise FileExistsError(
                            f"Refusing to overwrite existing models directory: {destination_models}"
                        )
                    if force:
                        shutil.rmtree(destination_models)
                destination_models.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(models_source, destination_models, dirs_exist_ok=False)
                imported_models = True

    return {
        "archive_path": source_archive,
        "imported_settings": imported_settings,
        "imported_models": imported_models,
        "settings_path": destination_settings,
        "models_dir": destination_models,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the dev-state import CLI."""

    parser = build_parser()
    parsed = parser.parse_args(argv)

    result = import_dev_state(
        parsed.archive,
        settings_path=(
            None if parsed.skip_settings else (parsed.settings_path or default_app_settings_path())
        ),
        models_dir=None if parsed.skip_models else (parsed.models_dir or installed_models_dir()),
        import_settings=not parsed.skip_settings,
        import_models=not parsed.skip_models,
        force=parsed.force,
    )
    print(f"archive={result['archive_path']}")
    print(f"imported_settings={result['imported_settings']}")
    print(f"imported_models={result['imported_models']}")
    if result["settings_path"] is not None:
        print(f"settings_path={result['settings_path']}")
    if result["models_dir"] is not None:
        print(f"models_dir={result['models_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

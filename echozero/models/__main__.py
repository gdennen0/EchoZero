"""Command line model manager for EchoZero app-installed models.
Exists to support v1-alpha model install/list/validate without packaging cloud SDKs.
Connects central model manifests and local imports to ~/.echozero/models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from echozero.models.distribution import (
    default_registry_manifest_source,
    discover_registry_models,
    import_local_model_bundle,
    install_model_from_registry,
    list_installed_models,
    save_registry_manifest_source,
    validate_installed_model,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage EchoZero app-installed models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    source_parser = subparsers.add_parser(
        "set-registry",
        help="Persist a registry manifest source.",
    )
    source_parser.add_argument("manifest")
    source_parser.add_argument("--models-dir", type=Path, default=None)

    available_parser = subparsers.add_parser("available", help="List models from the registry.")
    available_parser.add_argument("--manifest", default=None)
    available_parser.add_argument("--models-dir", type=Path, default=None)

    install_parser = subparsers.add_parser(
        "install",
        help="Install a model from a registry manifest.",
    )
    install_parser.add_argument("model_id")
    install_parser.add_argument(
        "--manifest",
        default=None,
        help="Local path, file URL, or HTTPS manifest URL.",
    )
    install_parser.add_argument("--models-dir", type=Path, default=None)

    import_parser = subparsers.add_parser("import", help="Import a local model bundle.")
    import_parser.add_argument("path", type=Path)
    import_parser.add_argument("--model-id", required=True)
    import_parser.add_argument("--type", required=True)
    import_parser.add_argument("--label", required=True)
    import_parser.add_argument("--version", required=True)
    import_parser.add_argument("--class", dest="classes", action="append", default=[])
    import_parser.add_argument("--runtime-consumer", default=None)
    import_parser.add_argument("--models-dir", type=Path, default=None)

    list_parser = subparsers.add_parser("list", help="List installed models.")
    list_parser.add_argument("--models-dir", type=Path, default=None)

    validate_parser = subparsers.add_parser("validate", help="Validate installed model records.")
    validate_parser.add_argument("--models-dir", type=Path, default=None)

    parsed = parser.parse_args(argv)

    if parsed.command == "set-registry":
        path = save_registry_manifest_source(parsed.manifest, models_dir=parsed.models_dir)
        print(f"registry_source={path}")
        return 0

    if parsed.command == "available":
        source = parsed.manifest or default_registry_manifest_source(parsed.models_dir)
        if not source:
            print("No registry manifest source configured.")
            return 1
        for listing in discover_registry_models(
            manifest_source=source,
            models_dir=parsed.models_dir,
        ):
            entry = listing.entry
            print(
                f"{entry.model_id}\t{entry.version}\t{entry.label}\t"
                f"{entry.model_type}\t{listing.state.value}"
            )
        return 0

    if parsed.command == "install":
        source = parsed.manifest or default_registry_manifest_source(parsed.models_dir)
        if not source:
            print("No registry manifest source configured.")
            return 1
        record = install_model_from_registry(
            model_id=parsed.model_id,
            manifest_source=source,
            models_dir=parsed.models_dir,
        )
        print(f"installed {record.model_id} {record.version} -> {record.bundle_dir}")
        return 0

    if parsed.command == "import":
        record = import_local_model_bundle(
            bundle_path=parsed.path,
            model_id=parsed.model_id,
            model_type=parsed.type,
            label=parsed.label,
            version=parsed.version,
            classes=tuple(parsed.classes),
            runtime_consumer=parsed.runtime_consumer,
            models_dir=parsed.models_dir,
        )
        print(f"imported {record.model_id} {record.version} -> {record.bundle_dir}")
        return 0

    if parsed.command == "list":
        for record in list_installed_models(parsed.models_dir):
            print(f"{record.model_id}\t{record.version}\t{record.label}\t{record.bundle_dir}")
        return 0

    if parsed.command == "validate":
        failed = False
        for record in list_installed_models(parsed.models_dir):
            ok = validate_installed_model(record, models_dir=parsed.models_dir)
            print(f"{record.model_id}\t{record.version}\t{'ok' if ok else 'invalid'}")
            failed = failed or not ok
        return 1 if failed else 0

    parser.error(f"Unsupported command: {parsed.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

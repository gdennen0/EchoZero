"""
Runtime bundle selection for app-installed classification models.
Exists because Stage Zero needs stable product-level rules for resolving installed runtime bundles.
Used by app-shell pipeline actions to find the correct local Foundry exports without manual file picking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from echozero.errors import ValidationError

from .paths import ensure_installed_models_dir
from .runtime_bundle_index import IndexedBinaryDrumBundle, load_binary_drum_bundle_index


@dataclass(frozen=True, slots=True)
class InstalledRuntimeBundle:
    """Resolved installed runtime bundle metadata for one classifier label."""

    label: str
    manifest_path: Path
    weights_path: Path
    bundle_dir: Path


def resolve_installed_binary_drum_bundles(
    *,
    labels: tuple[str, ...] = ("kick", "snare"),
    models_dir: Path | None = None,
) -> dict[str, InstalledRuntimeBundle]:
    """Resolve one installed one-vs-rest runtime bundle per requested drum label."""
    root = models_dir or ensure_installed_models_dir()
    manifests = sorted(root.glob("*/*.manifest.json"))
    indexed_bundles = load_binary_drum_bundle_index(root)
    bundles: dict[str, InstalledRuntimeBundle] = {}

    for label in labels:
        indexed_bundle = _resolve_indexed_bundle(root, indexed_bundles.get(label), label=label)
        if indexed_bundle is not None:
            bundles[label] = indexed_bundle
            continue
        matches: list[InstalledRuntimeBundle] = []
        for manifest_path in manifests:
            manifest = _load_manifest(manifest_path)
            if not _manifest_matches_label(manifest, label=label):
                continue
            weights_path = _resolve_weights_path(manifest_path, manifest.get("weightsPath"))
            if weights_path is None or not weights_path.exists():
                continue
            matches.append(
                InstalledRuntimeBundle(
                    label=label,
                    manifest_path=manifest_path.resolve(),
                    weights_path=weights_path.resolve(),
                    bundle_dir=manifest_path.parent.resolve(),
                )
            )

        if not matches:
            raise FileNotFoundError(
                f"No installed runtime bundle for '{label}' was found in {root}. "
                f"Expected a Foundry artifact manifest with classes ['{label}', 'other']."
            )
        if len(matches) > 1:
            match_paths = ", ".join(str(match.manifest_path) for match in matches)
            raise ValidationError(
                f"Multiple installed runtime bundles matched '{label}'. Keep one installed bundle per class: "
                f"{match_paths}"
            )
        bundles[label] = matches[0]

    return bundles


def _resolve_indexed_bundle(
    root: Path,
    record: IndexedBinaryDrumBundle | None,
    *,
    label: str,
) -> InstalledRuntimeBundle | None:
    if record is None:
        return None
    bundle_dir = (root / record.bundle_dir).resolve()
    manifest_path = (bundle_dir / record.manifest_file).resolve()
    weights_path = (bundle_dir / record.weights_file).resolve()
    if not manifest_path.exists() or not weights_path.exists():
        return None
    manifest = _load_manifest(manifest_path)
    if not _manifest_matches_label(manifest, label=label):
        return None
    return InstalledRuntimeBundle(
        label=label,
        manifest_path=manifest_path,
        weights_path=weights_path,
        bundle_dir=bundle_dir,
    )


def _load_manifest(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _manifest_matches_label(manifest: dict[str, object] | None, *, label: str) -> bool:
    if manifest is None:
        return False
    classes = manifest.get("classes")
    if not isinstance(classes, list):
        return False
    normalized_classes = tuple(str(value).strip().lower() for value in classes)
    return (
        len(normalized_classes) == 2
        and label in normalized_classes
        and "other" in normalized_classes
    )


def _resolve_weights_path(manifest_path: Path, raw_weights_path: object) -> Path | None:
    if not isinstance(raw_weights_path, str) or not raw_weights_path.strip():
        return None
    weights_path = Path(raw_weights_path)
    if weights_path.is_absolute():
        return weights_path
    return manifest_path.parent / weights_path

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
    bundles: dict[str, InstalledRuntimeBundle] = {}

    for label in labels:
        matches: list[InstalledRuntimeBundle] = []
        for manifest_path in manifests:
            manifest = _load_manifest(manifest_path)
            if manifest is None:
                continue
            classes = manifest.get("classes")
            if not isinstance(classes, list):
                continue
            normalized_classes = tuple(str(value).strip().lower() for value in classes)
            if label not in normalized_classes or "other" not in normalized_classes:
                continue
            if len(normalized_classes) != 2:
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


def _load_manifest(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _resolve_weights_path(manifest_path: Path, raw_weights_path: object) -> Path | None:
    if not isinstance(raw_weights_path, str) or not raw_weights_path.strip():
        return None
    weights_path = Path(raw_weights_path)
    if weights_path.is_absolute():
        return weights_path
    return manifest_path.parent / weights_path

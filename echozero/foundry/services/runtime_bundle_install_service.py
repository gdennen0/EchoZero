"""
Foundry runtime bundle installer for Stage Zero binary drum classifiers.
Exists because chosen Foundry artifacts need a stable promoted location outside per-run export folders.
Used by Foundry CLI to make the current kick/snare runtime bundle explicit for EZ.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from echozero.foundry.persistence import ModelArtifactRepository
from echozero.models.paths import ensure_installed_models_dir
from echozero.models.runtime_bundle_index import (
    IndexedBinaryDrumBundle,
    load_binary_drum_bundle_index,
    save_binary_drum_bundle_index,
)
from echozero.runtime_models.bundle_compat import sync_manifest_fingerprint


@dataclass(frozen=True, slots=True)
class InstalledFoundryRuntimeBundle:
    """Installed runtime bundle metadata for a promoted Foundry artifact."""

    label: str
    bundle_name: str
    bundle_dir: Path
    manifest_path: Path
    weights_path: Path
    artifact_id: str | None = None
    run_id: str | None = None
    source_manifest_path: Path | None = None


class RuntimeBundleInstallService:
    """Promote a Foundry artifact into the app-managed runtime bundle directory."""

    def __init__(
        self,
        root: Path,
        *,
        artifact_repository: ModelArtifactRepository | None = None,
    ) -> None:
        self._root = root
        self._artifact_repo = artifact_repository or ModelArtifactRepository(root)

    def install_binary_drum_artifact(
        self,
        artifact_ref: str,
        *,
        bundle_label: str | None = None,
        bundle_name: str | None = None,
        models_dir: Path | None = None,
    ) -> InstalledFoundryRuntimeBundle:
        """Install one Foundry one-vs-rest artifact as the current runtime bundle for its label."""
        source_manifest_path, artifact_id, run_id = self._resolve_artifact_ref(artifact_ref)
        manifest = _load_manifest(source_manifest_path)
        label = _resolve_binary_label(manifest, requested_label=bundle_label)
        raw_weights_path = manifest.get("weightsPath")
        if not isinstance(raw_weights_path, str) or not raw_weights_path.strip():
            raise ValueError(f"Artifact manifest is missing weightsPath: {source_manifest_path}")
        source_weights_path = _resolve_weights_path(source_manifest_path, raw_weights_path)
        if source_weights_path is None or not source_weights_path.exists():
            raise FileNotFoundError(f"Artifact weights file not found for {source_manifest_path}")

        target_root = models_dir or ensure_installed_models_dir()
        target_root.mkdir(parents=True, exist_ok=True)
        resolved_bundle_name = bundle_name or f"binary-drum-{_slug(label)}"
        target_dir = target_root / resolved_bundle_name
        staging_dir = target_root / f".{resolved_bundle_name}.staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        target_manifest_path = staging_dir / source_manifest_path.name
        target_weights_path = staging_dir / source_weights_path.name
        try:
            shutil.copy2(source_manifest_path, target_manifest_path)
            shutil.copy2(source_weights_path, target_weights_path)
            for optional_name in ("metrics.json", "run_summary.json"):
                optional_path = source_manifest_path.parent / optional_name
                if optional_path.exists():
                    shutil.copy2(optional_path, staging_dir / optional_name)
            sync_manifest_fingerprint(target_manifest_path, target_weights_path)
            if target_dir.exists():
                shutil.rmtree(target_dir)
            staging_dir.rename(target_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        records = load_binary_drum_bundle_index(target_root)
        records[label] = IndexedBinaryDrumBundle(
            label=label,
            bundle_dir=resolved_bundle_name,
            manifest_file=target_manifest_path.name,
            weights_file=target_weights_path.name,
            artifact_id=artifact_id or _optional_string(manifest.get("artifactId")),
            run_id=run_id or _optional_string(manifest.get("runId")),
            source_manifest_path=str(source_manifest_path),
        )
        save_binary_drum_bundle_index(target_root, records)

        return InstalledFoundryRuntimeBundle(
            label=label,
            bundle_name=resolved_bundle_name,
            bundle_dir=target_dir.resolve(),
            manifest_path=(target_dir / target_manifest_path.name).resolve(),
            weights_path=(target_dir / target_weights_path.name).resolve(),
            artifact_id=artifact_id or _optional_string(manifest.get("artifactId")),
            run_id=run_id or _optional_string(manifest.get("runId")),
            source_manifest_path=source_manifest_path.resolve(),
        )

    def _resolve_artifact_ref(self, artifact_ref: str) -> tuple[Path, str | None, str | None]:
        artifact = self._artifact_repo.get(artifact_ref)
        if artifact is not None:
            return artifact.path.resolve(), artifact.id, artifact.run_id
        path = Path(artifact_ref).expanduser()
        if path.exists():
            return path.resolve(), None, None
        manifest_name = f"{artifact_ref}.manifest.json"
        matches = sorted(self._root.rglob(manifest_name))
        if len(matches) == 1:
            return matches[0].resolve(), artifact_ref, None
        if len(matches) > 1:
            joined = ", ".join(str(match) for match in matches)
            raise ValueError(f"Artifact ref '{artifact_ref}' matched multiple manifests: {joined}")
        raise FileNotFoundError(f"Artifact manifest or artifact id not found: {artifact_ref}")


def _load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact manifest must be a JSON object: {path}")
    return payload


def _resolve_binary_label(manifest: dict[str, object], *, requested_label: str | None) -> str:
    classes = manifest.get("classes")
    if not isinstance(classes, list):
        raise ValueError("Artifact manifest is missing classes")
    normalized = [str(value).strip().lower() for value in classes]
    if len(normalized) != 2 or "other" not in normalized:
        raise ValueError("Runtime bundle install requires a binary one-vs-rest artifact with classes [label, other]")
    labels = [value for value in normalized if value != "other"]
    if len(labels) != 1:
        raise ValueError("Runtime bundle install could not infer the positive label")
    label = labels[0]
    if requested_label is not None and requested_label.strip().lower() != label:
        raise ValueError(
            f"Requested label '{requested_label}' does not match artifact classes {normalized}"
        )
    return label


def _resolve_weights_path(manifest_path: Path, raw_weights_path: str) -> Path | None:
    weights_path = Path(raw_weights_path)
    if weights_path.is_absolute():
        return weights_path
    return manifest_path.parent / weights_path


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _slug(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
    )

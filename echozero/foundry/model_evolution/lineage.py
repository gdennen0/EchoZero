"""Model lineage resolution for Foundry model evolution.
Exists so evolving models can clearly state which installed model they continued from.
Connects app-installed runtime bundles to candidate training run specs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from echozero.models.runtime_bundle_index import load_binary_drum_bundle_index
from echozero.models.runtime_bundle_selection import resolve_installed_binary_drum_bundles


@dataclass(frozen=True, slots=True)
class ModelLineage:
    """Resolved seed model metadata for one candidate model."""

    label: str
    kind: str
    initial_model_path: Path | None = None
    source_bundle_dir: Path | None = None
    source_artifact_id: str | None = None
    source_run_id: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Serialize lineage metadata into run and manifest payloads."""
        return {
            "schema": "foundry.model_lineage.v1",
            "label": self.label,
            "kind": self.kind,
            "initialModelPath": (
                None if self.initial_model_path is None else str(self.initial_model_path)
            ),
            "sourceBundleDir": None if self.source_bundle_dir is None else str(self.source_bundle_dir),
            "sourceArtifactId": self.source_artifact_id,
            "sourceRunId": self.source_run_id,
        }


class ModelLineageResolver:
    """Resolves seed models from the installed runtime bundle index."""

    def __init__(self, *, models_dir: Path) -> None:
        self._models_dir = Path(models_dir)

    def resolve_installed_binary_drum_lineage(
        self,
        labels: tuple[str, ...],
        *,
        enabled: bool = True,
        explicit_initial_model_paths: dict[str, Path] | None = None,
    ) -> dict[str, ModelLineage]:
        """Resolve lineage for each label from explicit paths or installed bundles."""
        explicit = {
            str(label).strip().lower(): Path(path).expanduser().resolve()
            for label, path in (explicit_initial_model_paths or {}).items()
            if str(label).strip()
        }
        normalized_labels = tuple(
            dict.fromkeys(str(label).strip().lower() for label in labels if str(label).strip())
        )
        resolved: dict[str, ModelLineage] = {}
        for label in normalized_labels:
            if label in explicit:
                resolved[label] = ModelLineage(
                    label=label,
                    kind="explicit_seed",
                    initial_model_path=explicit[label],
                )
        remaining = tuple(label for label in normalized_labels if label not in resolved)
        bundles = (
            resolve_installed_binary_drum_bundles(labels=remaining, models_dir=self._models_dir)
            if enabled and remaining
            else {}
        )
        index = load_binary_drum_bundle_index(self._models_dir) if enabled and remaining else {}
        for label in remaining:
            bundle = bundles.get(label)
            if bundle is None:
                resolved[label] = ModelLineage(label=label, kind="from_scratch")
                continue
            index_record = index.get(label)
            resolved[label] = ModelLineage(
                label=label,
                kind="installed_runtime_bundle",
                initial_model_path=bundle.manifest_path,
                source_bundle_dir=bundle.bundle_dir,
                source_artifact_id=None if index_record is None else index_record.artifact_id,
                source_run_id=None if index_record is None else index_record.run_id,
            )
        return resolved

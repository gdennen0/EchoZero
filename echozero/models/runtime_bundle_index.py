"""
Runtime bundle index for installed binary drum classifiers.
Exists because folder scans alone cannot express the intentional current bundle per label.
Used by Foundry promotion flows and Stage Zero runtime bundle resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_INDEX_SCHEMA = "echozero.binary_drum_bundle_index.v1"
_INDEX_FILENAME = "binary_drum_bundles.json"


@dataclass(frozen=True, slots=True)
class IndexedBinaryDrumBundle:
    """Saved pointer to the current installed runtime bundle for one drum label."""

    label: str
    bundle_dir: str
    manifest_file: str
    weights_file: str
    artifact_id: str | None = None
    run_id: str | None = None
    source_manifest_path: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "bundleDir": self.bundle_dir,
            "manifestFile": self.manifest_file,
            "weightsFile": self.weights_file,
            "artifactId": self.artifact_id,
            "runId": self.run_id,
            "sourceManifestPath": self.source_manifest_path,
        }

    @classmethod
    def from_payload(cls, label: str, payload: object) -> IndexedBinaryDrumBundle | None:
        if not isinstance(payload, dict):
            return None
        bundle_dir = payload.get("bundleDir")
        manifest_file = payload.get("manifestFile")
        weights_file = payload.get("weightsFile")
        if not all(isinstance(value, str) and value.strip() for value in (bundle_dir, manifest_file, weights_file)):
            return None
        return cls(
            label=str(label).strip().lower(),
            bundle_dir=bundle_dir,
            manifest_file=manifest_file,
            weights_file=weights_file,
            artifact_id=_optional_string(payload.get("artifactId")),
            run_id=_optional_string(payload.get("runId")),
            source_manifest_path=_optional_string(payload.get("sourceManifestPath")),
        )


def binary_drum_bundle_index_path(models_dir: Path) -> Path:
    """Return the canonical installed-bundle index path."""
    return models_dir / _INDEX_FILENAME


def load_binary_drum_bundle_index(models_dir: Path) -> dict[str, IndexedBinaryDrumBundle]:
    """Load the current installed binary drum bundle pointers."""
    path = binary_drum_bundle_index_path(models_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    raw_bundles = payload.get("bundles")
    if not isinstance(raw_bundles, dict):
        return {}
    records: dict[str, IndexedBinaryDrumBundle] = {}
    for raw_label, raw_record in raw_bundles.items():
        record = IndexedBinaryDrumBundle.from_payload(str(raw_label), raw_record)
        if record is not None:
            records[record.label] = record
    return records


def save_binary_drum_bundle_index(
    models_dir: Path,
    records: dict[str, IndexedBinaryDrumBundle],
) -> Path:
    """Persist the current installed binary drum bundle pointers."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = binary_drum_bundle_index_path(models_dir)
    payload = {
        "schema": _INDEX_SCHEMA,
        "bundles": {
            label: records[label].to_payload()
            for label in sorted(records)
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value

"""
Compatibility helpers for installed runtime bundles.
Exists because older local Foundry exports may need explicit upgrade steps before runtime use.
Used by app-level model management flows, not by pure resolution helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

from echozero.inference_eval.runtime_preflight import checkpoint_contract_fingerprint


def upgrade_installed_runtime_bundles(models_dir: Path) -> int:
    """Explicitly upgrade installed manifests in-place when a repair is needed."""
    upgraded = 0
    for manifest_path in sorted(models_dir.glob("*/*.manifest.json")):
        weights_path = _resolve_weights_path(manifest_path)
        if weights_path is None or not weights_path.exists():
            continue
        if sync_manifest_fingerprint(manifest_path, weights_path):
            upgraded += 1
    return upgraded


def backfill_manifest_fingerprint(manifest_path: Path, weights_path: Path) -> bool:
    """Backfill a missing shared contract fingerprint into one manifest."""
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        return False
    fingerprint = manifest.get("sharedContractFingerprint")
    if isinstance(fingerprint, str) and fingerprint.strip():
        return False
    return sync_manifest_fingerprint(manifest_path, weights_path)


def sync_manifest_fingerprint(manifest_path: Path, weights_path: Path) -> bool:
    """Repair one manifest so it matches the checkpoint-derived shared contract fingerprint."""
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        return False
    try:
        import torch
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    if not isinstance(checkpoint, dict):
        return False
    expected_fingerprint = checkpoint_contract_fingerprint(checkpoint)
    fingerprint = manifest.get("sharedContractFingerprint")
    if isinstance(fingerprint, str) and fingerprint.strip() == expected_fingerprint:
        return False
    updated_manifest = dict(manifest)
    updated_manifest["sharedContractFingerprint"] = expected_fingerprint
    manifest_path.write_text(json.dumps(updated_manifest, indent=2), encoding="utf-8")
    return True


def _load_manifest(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _resolve_weights_path(manifest_path: Path) -> Path | None:
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        return None
    raw_weights_path = manifest.get("weightsPath")
    if not isinstance(raw_weights_path, str) or not raw_weights_path.strip():
        return None
    weights_path = Path(raw_weights_path)
    if weights_path.is_absolute():
        return weights_path
    return manifest_path.parent / weights_path

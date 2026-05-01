"""
Compatibility helpers for installed runtime bundles.
Exists because older local Foundry exports may need explicit upgrade steps before runtime use.
Used by app-level model management flows, not by pure resolution helpers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from echozero.inference_eval.runtime_preflight import checkpoint_contract_fingerprint


def load_runtime_checkpoint(weights_path: Path, *, map_location: str = "cpu") -> Mapping[str, Any]:
    """Load one local runtime checkpoint, retrying trusted legacy exports when needed."""
    import torch

    try:
        checkpoint = torch.load(weights_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if not _should_retry_without_weights_only(exc):
            raise
        checkpoint = torch.load(weights_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unexpected checkpoint format from {weights_path}")
    return checkpoint


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
        checkpoint = load_runtime_checkpoint(weights_path, map_location="cpu")
    except Exception:
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


def _should_retry_without_weights_only(exc: Exception) -> bool:
    message = str(exc)
    return "Weights only load failed" in message or "Unsupported global" in message

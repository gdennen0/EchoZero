from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from echozero.errors import ValidationError

from .core import EvalContract, InferenceContract, contract_fingerprint
from .validation import validate_manifest_inference_section, validate_runtime_consumer


def _manifest_matches_model(manifest: Mapping[str, Any], model_path: Path) -> bool:
    weights_path = manifest.get("weightsPath")
    if not isinstance(weights_path, str) or not weights_path:
        return False
    return Path(weights_path).name == model_path.name


def _load_manifest(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if isinstance(payload, Mapping):
        return payload
    return None


def _discover_manifest(model_path: Path) -> Mapping[str, Any] | None:
    manifest_candidates = sorted(model_path.parent.glob("*.manifest.json"))
    for candidate in manifest_candidates:
        manifest = _load_manifest(candidate)
        if manifest is not None and _manifest_matches_model(manifest, model_path):
            return manifest
    return None


def _checkpoint_preprocessing(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    preprocessing = checkpoint.get("inference_preprocessing")
    if isinstance(preprocessing, Mapping):
        return dict(preprocessing)

    preprocessing = checkpoint.get("preprocessing")
    if isinstance(preprocessing, Mapping):
        return dict(preprocessing)

    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        nested = config.get("preprocessing")
        if isinstance(nested, Mapping):
            return dict(nested)

    return {}


def _checkpoint_classes(checkpoint: Mapping[str, Any]) -> tuple[str, ...]:
    classes = checkpoint.get("classes")
    if isinstance(classes, list):
        return tuple(str(label) for label in classes)
    if isinstance(classes, tuple):
        return tuple(str(label) for label in classes)
    return ()


def _checkpoint_model_signature(checkpoint: Mapping[str, Any]) -> str | None:
    for key in ("model_type", "trainer", "schema"):
        value = checkpoint.get(key)
        if value:
            return str(value)

    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        value = config.get("model_class")
        if value:
            return str(value)

    return None


def _checkpoint_classification_mode(checkpoint: Mapping[str, Any]) -> str:
    for key in ("classification_mode", "classificationMode"):
        value = checkpoint.get(key)
        if value:
            return str(value)
    return "multiclass"


def checkpoint_contract_fingerprint(checkpoint: Mapping[str, Any]) -> str:
    inference_contract = InferenceContract(
        preprocessing=_checkpoint_preprocessing(checkpoint),
        class_map=_checkpoint_classes(checkpoint),
        model_signature=_checkpoint_model_signature(checkpoint),
    )
    eval_contract = EvalContract(
        classification_mode=_checkpoint_classification_mode(checkpoint),
        split_name="test",
    )
    return contract_fingerprint(inference_contract, eval_contract)


def run_runtime_preflight(
    model_path: str | Path,
    checkpoint: Mapping[str, Any],
    *,
    consumer: str = "PyTorchAudioClassify",
) -> None:
    resolved_model_path = Path(model_path)
    manifest = _discover_manifest(resolved_model_path)
    if manifest is None:
        return

    errors: list[str] = []
    manifest_report = validate_manifest_inference_section(manifest)
    runtime_report = validate_runtime_consumer(manifest, consumer=consumer)

    errors.extend(issue.message for issue in manifest_report.errors)
    errors.extend(issue.message for issue in runtime_report.errors)

    manifest_fingerprint = manifest.get("sharedContractFingerprint")
    if manifest_fingerprint is not None:
        expected_fingerprint = checkpoint_contract_fingerprint(checkpoint)
        if manifest_fingerprint != expected_fingerprint:
            errors.append(
                "manifest.sharedContractFingerprint must match the checkpoint-derived shared contract fingerprint"
            )

    if errors:
        joined = "; ".join(errors)
        raise ValidationError(
            f"Runtime bundle preflight failed for {resolved_model_path.name}: {joined}"
        )

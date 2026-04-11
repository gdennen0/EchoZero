from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from echozero.errors import ValidationError

from .constants import REQUIRED_PREPROCESSING_KEYS
from .core import EvalContract, InferenceContract, contract_fingerprint
from .validation import validate_manifest_inference_section, validate_runtime_consumer

_PREPROCESSING_ALIASES = {
    "sample_rate": "sampleRate",
    "max_length": "maxLength",
    "n_fft": "nFft",
    "hop_length": "hopLength",
    "n_mels": "nMels",
    "f_max": "fmax",
}


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


def _canonicalize_preprocessing_keys(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in payload.items():
        key = str(raw_key)
        normalized[_PREPROCESSING_ALIASES.get(key, key)] = value
    return normalized


def _checkpoint_preprocessing(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    preprocessing = checkpoint.get("inference_preprocessing")
    if isinstance(preprocessing, Mapping):
        return _canonicalize_preprocessing_keys(preprocessing)

    preprocessing = checkpoint.get("preprocessing")
    if isinstance(preprocessing, Mapping):
        return _canonicalize_preprocessing_keys(preprocessing)

    config = checkpoint.get("config")
    if isinstance(config, Mapping):
        nested = config.get("preprocessing")
        if isinstance(nested, Mapping):
            return _canonicalize_preprocessing_keys(nested)

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

    manifest_preprocessing = manifest.get("inferencePreprocessing")
    if isinstance(manifest_preprocessing, Mapping):
        checkpoint_preprocessing = _checkpoint_preprocessing(checkpoint)
        for key in sorted(REQUIRED_PREPROCESSING_KEYS):
            expected = checkpoint_preprocessing.get(key)
            actual = manifest_preprocessing.get(key)
            if expected is None or actual is None:
                continue
            if actual != expected:
                errors.append(f"manifest.inferencePreprocessing.{key} must match checkpoint preprocessing")

    manifest_classes = manifest.get("classes")
    checkpoint_classes = _checkpoint_classes(checkpoint)
    if checkpoint_classes and isinstance(manifest_classes, list):
        if manifest_classes != list(checkpoint_classes):
            errors.append("manifest.classes must match checkpoint class map order")

    manifest_classification_mode = manifest.get("classificationMode")
    checkpoint_classification_mode = _checkpoint_classification_mode(checkpoint)
    if manifest_classification_mode is not None and str(manifest_classification_mode) != checkpoint_classification_mode:
        errors.append("manifest.classificationMode must match checkpoint classification mode")

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

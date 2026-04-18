from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from echozero.errors import ValidationError

from .constants import REQUIRED_PREPROCESSING_KEYS
from .core import EvalContract, InferenceContract, contract_fingerprint
from .diagnostics import ValidationReport, attach_validation_report
from .validation import validate_manifest_inference_section, validate_runtime_consumer

_PREPROCESSING_ALIASES = {
    "sample_rate": "sampleRate",
    "max_length": "maxLength",
    "n_fft": "nFft",
    "hop_length": "hopLength",
    "n_mels": "nMels",
    "f_max": "fmax",
}


@dataclass(frozen=True, slots=True)
class _ManifestCandidate:
    path: Path
    manifest: Mapping[str, Any]


def _load_manifest(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if isinstance(payload, Mapping):
        return payload
    return None


def _iter_manifest_candidates(model_path: Path) -> list[_ManifestCandidate]:
    candidates: list[_ManifestCandidate] = []
    for candidate_path in sorted(model_path.parent.glob("*.manifest.json")):
        manifest = _load_manifest(candidate_path)
        if manifest is None:
            continue
        candidates.append(_ManifestCandidate(path=candidate_path, manifest=manifest))
    return candidates


def _manifest_target_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path | None:
    raw_weights_path = manifest.get("weightsPath")
    if not isinstance(raw_weights_path, str) or not raw_weights_path.strip():
        return None

    weights_path = Path(raw_weights_path)
    if weights_path.is_absolute():
        return weights_path.resolve()
    return (manifest_path.parent / weights_path).resolve()


def resolve_runtime_model_path(model_path: str | Path) -> Path:
    requested_path = Path(model_path)
    if not requested_path.exists():
        raise FileNotFoundError(f"Runtime model path does not exist: {requested_path}")

    if requested_path.is_dir():
        manifests = _iter_manifest_candidates(requested_path / "model.pth")
        if not manifests:
            fallback = requested_path / "model.pth"
            if fallback.exists():
                return fallback.resolve()
            raise ValidationError(
                "Runtime model directory must contain either model.pth or a matching *.manifest.json bundle: "
                f"{requested_path}"
            )
        if len(manifests) > 1:
            raise ValidationError(
                "Runtime model directory is ambiguous; keep exactly one *.manifest.json artifact in the bundle: "
                f"{requested_path}"
            )
        target_path = _manifest_target_path(manifests[0].path, manifests[0].manifest)
        if target_path is None:
            raise ValidationError(
                f"Artifact manifest is missing a usable weightsPath: {manifests[0].path}"
            )
        if not target_path.exists():
            raise FileNotFoundError(f"Resolved artifact weights do not exist: {target_path}")
        return target_path

    if requested_path.name.endswith(".manifest.json"):
        manifest = _load_manifest(requested_path)
        if manifest is None:
            raise ValidationError(f"Artifact manifest is not valid JSON: {requested_path}")
        target_path = _manifest_target_path(requested_path, manifest)
        if target_path is None:
            raise ValidationError(f"Artifact manifest is missing a usable weightsPath: {requested_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Resolved artifact weights do not exist: {target_path}")
        return target_path

    return requested_path.resolve()


def _resolve_manifest_for_model(model_path: Path, report: ValidationReport) -> Mapping[str, Any] | None:
    candidates = _iter_manifest_candidates(model_path)
    if not candidates:
        report.add_error(
            "missing_model_manifest",
            "manifest",
            "artifact manifest is required for runtime preflight; no *.manifest.json files were found"
            f" alongside requested model path (requested model path: {model_path.resolve()})",
        )
        return None

    resolved_model_path = model_path.resolve()
    matches: list[_ManifestCandidate] = []
    candidate_descriptions: list[str] = []
    for candidate in candidates:
        target_path = _manifest_target_path(candidate.path, candidate.manifest)
        if target_path is None:
            candidate_descriptions.append(candidate.path.name)
            continue
        candidate_descriptions.append(f"{candidate.path.name} -> {target_path}")
        if target_path == resolved_model_path:
            matches.append(candidate)

    candidate_summary = ", ".join(candidate_descriptions)

    if not matches:
        report.add_error(
            "manifest_not_found_for_model",
            "manifest.weightsPath",
            "no manifest in model directory resolves manifest.weightsPath to the requested model path"
            f" (requested model path: {resolved_model_path}; candidates: {candidate_summary})",
        )
        return None

    if len(matches) > 1:
        report.add_error(
            "ambiguous_manifest_match",
            "manifest.weightsPath",
            "multiple manifests resolve to the requested model path; keep exactly one matching artifact manifest"
            f" (requested model path: {resolved_model_path}; candidates: {candidate_summary})",
        )
        return None

    return matches[0].manifest


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
    return ""


def _validate_checkpoint_completeness_for_manifest(checkpoint: Mapping[str, Any], report: ValidationReport) -> None:
    checkpoint_classes = _checkpoint_classes(checkpoint)
    if not checkpoint_classes:
        report.add_error(
            "missing_checkpoint_classes",
            "checkpoint.classes",
            "checkpoint.classes must be present when validating against an artifact manifest",
        )

    checkpoint_preprocessing = _checkpoint_preprocessing(checkpoint)
    missing_preprocessing = sorted(REQUIRED_PREPROCESSING_KEYS - set(checkpoint_preprocessing.keys()))
    if missing_preprocessing:
        report.add_error(
            "missing_checkpoint_preprocessing_keys",
            "checkpoint.preprocessing",
            "checkpoint preprocessing missing keys required for manifest verification: "
            f"{', '.join(missing_preprocessing)}",
        )

    checkpoint_classification_mode = _checkpoint_classification_mode(checkpoint)
    if not checkpoint_classification_mode:
        report.add_error(
            "missing_checkpoint_classification_mode",
            "checkpoint.classification_mode",
            "checkpoint classification mode must be present when validating against an artifact manifest",
        )


def checkpoint_contract_fingerprint(checkpoint: Mapping[str, Any]) -> str:
    inference_contract = InferenceContract(
        preprocessing=_checkpoint_preprocessing(checkpoint),
        class_map=_checkpoint_classes(checkpoint),
        model_signature=_checkpoint_model_signature(checkpoint),
    )
    eval_contract = EvalContract(
        classification_mode=_checkpoint_classification_mode(checkpoint) or "multiclass",
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
    report = ValidationReport()
    manifest = _resolve_manifest_for_model(resolved_model_path, report)

    if manifest is not None:
        manifest_report = validate_manifest_inference_section(manifest)
        runtime_report = validate_runtime_consumer(manifest, consumer=consumer)
        report.merge(manifest_report).merge(runtime_report)
        _validate_checkpoint_completeness_for_manifest(checkpoint, report)

        manifest_preprocessing = manifest.get("inferencePreprocessing")
        if isinstance(manifest_preprocessing, Mapping):
            checkpoint_preprocessing = _checkpoint_preprocessing(checkpoint)
            for key in sorted(REQUIRED_PREPROCESSING_KEYS):
                expected = checkpoint_preprocessing.get(key)
                actual = manifest_preprocessing.get(key)
                if expected is None or actual is None:
                    continue
                if actual != expected:
                    report.add_error(
                        "preprocessing_checkpoint_mismatch",
                        f"manifest.inferencePreprocessing.{key}",
                        f"manifest.inferencePreprocessing.{key} must match checkpoint preprocessing",
                    )

        manifest_classes = manifest.get("classes")
        checkpoint_classes = _checkpoint_classes(checkpoint)
        if checkpoint_classes and isinstance(manifest_classes, list):
            if manifest_classes != list(checkpoint_classes):
                report.add_error(
                    "checkpoint_class_order_mismatch",
                    "manifest.classes",
                    "manifest.classes must match checkpoint class map order",
                )

        manifest_classification_mode = manifest.get("classificationMode")
        checkpoint_classification_mode = _checkpoint_classification_mode(checkpoint)
        if (
            manifest_classification_mode is not None
            and checkpoint_classification_mode
            and str(manifest_classification_mode) != checkpoint_classification_mode
        ):
            report.add_error(
                "checkpoint_classification_mode_mismatch",
                "manifest.classificationMode",
                "manifest.classificationMode must match checkpoint classification mode",
            )

        manifest_fingerprint = manifest.get("sharedContractFingerprint")
        if isinstance(manifest_fingerprint, str) and manifest_fingerprint.strip():
            expected_fingerprint = checkpoint_contract_fingerprint(checkpoint)
            if manifest_fingerprint != expected_fingerprint:
                report.add_error(
                    "shared_fingerprint_mismatch",
                    "manifest.sharedContractFingerprint",
                    "manifest.sharedContractFingerprint must match the checkpoint-derived shared contract fingerprint",
                )

    if report.errors:
        joined = "; ".join(issue.message for issue in report.errors)
        error = ValidationError(
            f"Runtime bundle preflight failed for {resolved_model_path.name}: {joined}"
        )
        attach_validation_report(error, report)
        raise error

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .constants import REQUIRED_PREPROCESSING_KEYS, SUPPORTED_CLASSIFICATION_MODES
from .core import EvalContract, InferenceContract
from .diagnostics import ValidationReport


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _has_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _contract_value(contract: object, field: str) -> Any:
    if isinstance(contract, Mapping):
        return contract.get(field)
    return getattr(contract, field, None)


def validate_inference_contract(contract: InferenceContract | Mapping[str, Any]) -> ValidationReport:
    report = ValidationReport()
    schema = _contract_value(contract, "schema")
    if schema != "echozero.shared.inference_contract.v1":
        report.add_error(
            "invalid_schema",
            "schema",
            "inference.schema must be echozero.shared.inference_contract.v1",
        )

    preprocessing = _as_mapping(_contract_value(contract, "preprocessing"))
    missing = sorted(REQUIRED_PREPROCESSING_KEYS - set(preprocessing.keys()))
    if missing:
        report.add_error(
            "missing_preprocessing_keys",
            "preprocessing",
            f"inference.preprocessing missing keys: {', '.join(missing)}",
        )

    class_map = _contract_value(contract, "class_map")
    if not _has_sequence(class_map) or not tuple(class_map):
        report.add_error(
            "invalid_class_map",
            "class_map",
            "inference.class_map must be a non-empty sequence",
        )

    return report


def validate_eval_contract(contract: EvalContract | Mapping[str, Any]) -> ValidationReport:
    report = ValidationReport()
    schema = _contract_value(contract, "schema")
    if schema != "echozero.shared.eval_contract.v1":
        report.add_error(
            "invalid_schema",
            "schema",
            "eval.schema must be echozero.shared.eval_contract.v1",
        )

    classification_mode = _contract_value(contract, "classification_mode")
    if classification_mode not in SUPPORTED_CLASSIFICATION_MODES:
        supported = ", ".join(sorted(SUPPORTED_CLASSIFICATION_MODES))
        report.add_error(
            "unsupported_classification_mode",
            "classification_mode",
            f"eval.classification_mode must be one of: {supported}",
        )

    metric_keys = _contract_value(contract, "metric_keys")
    if not _has_sequence(metric_keys) or not tuple(metric_keys):
        report.add_error(
            "invalid_metric_keys",
            "metric_keys",
            "eval.metric_keys must be a non-empty sequence",
        )

    split_name = _contract_value(contract, "split_name")
    if not isinstance(split_name, str) or not split_name.strip():
        report.add_error(
            "invalid_split_name",
            "split_name",
            "eval.split_name must be a non-empty string",
        )

    return report


def validate_manifest_inference_section(
    manifest: Mapping[str, Any],
    expected_run_data: Mapping[str, Any] | None = None,
    expected_classes: Sequence[str] | None = None,
    expected_taxonomy: Mapping[str, Any] | None = None,
    expected_label_policy: Mapping[str, Any] | None = None,
) -> ValidationReport:
    report = ValidationReport()
    if manifest.get("schema") != "foundry.artifact_manifest.v1":
        report.add_error(
            "invalid_manifest_schema",
            "manifest.schema",
            "manifest.schema must be foundry.artifact_manifest.v1",
        )

    classes = manifest.get("classes")
    if not isinstance(classes, list) or not classes:
        report.add_error(
            "invalid_classes",
            "manifest.classes",
            "manifest.classes must be a non-empty list",
        )

    preprocessing = _as_mapping(manifest.get("inferencePreprocessing"))
    missing = sorted(REQUIRED_PREPROCESSING_KEYS - set(preprocessing.keys()))
    if missing:
        report.add_error(
            "missing_preprocessing_keys",
            "manifest.inferencePreprocessing",
            f"manifest.inferencePreprocessing missing keys: {', '.join(missing)}",
        )

    fingerprint = manifest.get("sharedContractFingerprint")
    if fingerprint is None:
        report.add_error(
            "missing_shared_contract_fingerprint",
            "manifest.sharedContractFingerprint",
            "manifest.sharedContractFingerprint is required",
        )
    elif not isinstance(fingerprint, str) or not fingerprint.strip():
        report.add_error(
            "invalid_shared_contract_fingerprint",
            "manifest.sharedContractFingerprint",
            "manifest.sharedContractFingerprint must be a non-empty string",
        )

    run_data = _as_mapping(expected_run_data)
    expected_classification_mode = run_data.get("classificationMode")
    if expected_classification_mode is not None and manifest.get("classificationMode") != expected_classification_mode:
        report.add_error(
            "classification_mode_mismatch",
            "manifest.classificationMode",
            "manifest.classificationMode must match run spec classificationMode",
        )

    for key in sorted(REQUIRED_PREPROCESSING_KEYS):
        expected = run_data.get(key)
        actual = preprocessing.get(key)
        if expected is None or actual is None:
            continue
        if actual != expected:
            report.add_error(
                "preprocessing_mismatch",
                f"manifest.inferencePreprocessing.{key}",
                f"manifest.inferencePreprocessing.{key} must match run spec",
            )

    if expected_classes is not None and isinstance(classes, list):
        if classes != list(expected_classes):
            report.add_error(
                "class_order_mismatch",
                "manifest.classes",
                "manifest.classes must match dataset version class_map order",
            )

    if expected_taxonomy is not None and manifest.get("taxonomy") != dict(expected_taxonomy):
        report.add_error(
            "taxonomy_mismatch",
            "manifest.taxonomy",
            "manifest.taxonomy must match dataset version taxonomy",
        )

    if expected_label_policy is not None and manifest.get("labelPolicy") != dict(expected_label_policy):
        report.add_error(
            "label_policy_mismatch",
            "manifest.labelPolicy",
            "manifest.labelPolicy must match dataset version label_policy",
        )

    if manifest.get("classificationMode") == "binary" and "thresholdPolicy" not in manifest:
        report.add_warning(
            "missing_threshold_policy",
            "manifest.thresholdPolicy",
            "binary classifier missing thresholdPolicy",
        )

    return report


def validate_runtime_consumer(
    manifest: Mapping[str, Any],
    consumer: str = "PyTorchAudioClassify",
) -> ValidationReport:
    report = ValidationReport()
    if consumer == "PyTorchAudioClassify":
        weights_path = manifest.get("weightsPath")
        if not weights_path:
            report.add_error(
                "missing_weights_path",
                "manifest.weightsPath",
                "manifest.weightsPath is required for PyTorchAudioClassify",
            )
        elif not str(weights_path).endswith(".pth"):
            report.add_error(
                "invalid_weights_path",
                "manifest.weightsPath",
                "manifest.weightsPath must point to a .pth file for PyTorchAudioClassify",
            )

        if weights_path and str(weights_path).startswith(("/", "\\")):
            report.add_error(
                "absolute_weights_path",
                "manifest.weightsPath",
                "manifest.weightsPath must be relative for portable runtime use",
            )
        elif isinstance(weights_path, str) and len(weights_path) > 1 and weights_path[1] == ":":
            report.add_error(
                "absolute_weights_path",
                "manifest.weightsPath",
                "manifest.weightsPath must be relative for portable runtime use",
            )

        runtime = _as_mapping(manifest.get("runtime"))
        if runtime.get("consumer") != consumer:
            report.add_error(
                "runtime_consumer_mismatch",
                "manifest.runtime.consumer",
                "manifest.runtime.consumer must match the validated consumer",
            )

    return report

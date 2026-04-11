from __future__ import annotations

from echozero.inference_eval import (
    EvalContract,
    InferenceContract,
    REQUIRED_PREPROCESSING_KEYS,
    contract_fingerprint,
    validate_inference_contract,
    validate_manifest_inference_section,
)


def _required_preprocessing(**overrides: int) -> dict[str, int]:
    base = {
        "sampleRate": 22050,
        "maxLength": 22050,
        "nFft": 2048,
        "hopLength": 512,
        "nMels": 128,
        "fmax": 8000,
    }
    base.update(overrides)
    return {key: base[key] for key in REQUIRED_PREPROCESSING_KEYS}


def test_contract_fingerprint_is_stable_for_equivalent_payload_ordering() -> None:
    preprocessing = _required_preprocessing()
    inference_a = InferenceContract(
        preprocessing=preprocessing,
        class_map=("kick", "snare"),
        model_signature="cnn",
    )
    inference_b = InferenceContract(
        preprocessing={key: preprocessing[key] for key in reversed(tuple(preprocessing))},
        class_map=("kick", "snare"),
        model_signature="cnn",
    )
    eval_contract = EvalContract(classification_mode="multiclass", split_name="test")

    assert contract_fingerprint(inference_a, eval_contract) == contract_fingerprint(inference_b, eval_contract)


def test_validate_inference_contract_enforces_required_preprocessing_keys_from_shared_constants() -> None:
    missing_key = sorted(REQUIRED_PREPROCESSING_KEYS)[0]
    preprocessing = _required_preprocessing()
    preprocessing.pop(missing_key)

    report = validate_inference_contract(
        InferenceContract(
            preprocessing=preprocessing,
            class_map=("kick", "snare"),
            model_signature="cnn",
        )
    )

    assert report.ok is False
    assert len(report.errors) == 1
    issue = report.errors[0]
    assert issue.code == "missing_preprocessing_keys"
    assert issue.path == "preprocessing"
    assert issue.message == f"inference.preprocessing missing keys: {missing_key}"


def test_validate_manifest_inference_section_enforces_required_preprocessing_keys_from_shared_constants() -> None:
    missing_key = sorted(REQUIRED_PREPROCESSING_KEYS)[-1]
    preprocessing = _required_preprocessing()
    preprocessing.pop(missing_key)

    report = validate_manifest_inference_section(
        {
            "schema": "foundry.artifact_manifest.v1",
            "weightsPath": "model.pth",
            "classes": ["kick", "snare"],
            "classificationMode": "multiclass",
            "runtime": {"consumer": "PyTorchAudioClassify"},
            "inferencePreprocessing": preprocessing,
        }
    )

    assert report.ok is False
    issue = next(error for error in report.errors if error.code == "missing_preprocessing_keys")
    assert issue.path == "manifest.inferencePreprocessing"
    assert issue.message == f"manifest.inferencePreprocessing missing keys: {missing_key}"

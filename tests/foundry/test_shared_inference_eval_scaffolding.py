from __future__ import annotations

from echozero.inference_eval import EvalContract, EvalRequest, EvalResult, InferenceContract, contract_fingerprint
from echozero.inference_eval.echozero_adapter import create_echozero_adapter
from echozero.inference_eval.foundry_adapter import create_foundry_adapter


def test_contract_fingerprint_is_stable_for_equivalent_payload_ordering() -> None:
    inference_a = InferenceContract(
        preprocessing={"sampleRate": 22050, "nMels": 128, "hopLength": 512},
        class_map=("kick", "snare"),
        model_signature="cnn",
    )
    inference_b = InferenceContract(
        preprocessing={"hopLength": 512, "nMels": 128, "sampleRate": 22050},
        class_map=("kick", "snare"),
        model_signature="cnn",
    )
    eval_contract = EvalContract(classification_mode="multiclass", split_name="test")

    assert contract_fingerprint(inference_a, eval_contract) == contract_fingerprint(inference_b, eval_contract)


def test_foundry_adapter_to_eval_payload_matches_expected_contract_fields() -> None:
    adapter = create_foundry_adapter()
    request = EvalRequest(
        contract=EvalContract(classification_mode="multiclass", split_name="val"),
        run_id="run_123",
        dataset_version_id="dsv_456",
    )
    result = EvalResult(
        metrics={"accuracy": 0.91, "macro_f1": 0.88},
        aggregate_metrics={"sample_count": 200},
        per_class_metrics={"kick": {"recall": 0.9}},
        baseline={"macro_f1": 0.81},
        summary={"primary_metric_value": 0.88},
    )

    payload = adapter.to_eval_payload(request, result)

    assert payload["run_id"] == "run_123"
    assert payload["classification_mode"] == "multiclass"
    assert payload["dataset_version_id"] == "dsv_456"
    assert payload["split_name"] == "val"
    assert payload["metrics"]["macro_f1"] == 0.88
    assert payload["aggregate_metrics"]["sample_count"] == 200


def test_echozero_adapter_builds_inference_contract_from_checkpoint_metadata() -> None:
    adapter = create_echozero_adapter()
    checkpoint = {
        "model_type": "crnn",
        "classes": ["kick", "snare", "hat"],
        "inference_preprocessing": {
            "sample_rate": 22050,
            "n_mels": 128,
            "hop_length": 512,
        },
    }

    contract = adapter.inference_contract_from_checkpoint(checkpoint)

    assert contract.model_signature == "crnn"
    assert contract.class_map == ("kick", "snare", "hat")
    assert contract.preprocessing["sample_rate"] == 22050

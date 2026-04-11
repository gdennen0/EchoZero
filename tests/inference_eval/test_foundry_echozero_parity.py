from __future__ import annotations

from echozero.inference_eval import create_echozero_adapter, create_foundry_adapter


def test_foundry_and_echozero_adapters_produce_matching_fingerprints_for_equivalent_metadata() -> None:
    class_map = ("kick", "snare", "hat")
    run_spec = {
        "classificationMode": "multiclass",
        "data": {
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "model": {"type": "cnn"},
    }
    checkpoint = {
        "model_type": "cnn",
        "classes": ["kick", "snare", "hat"],
        "inference_preprocessing": {
            "fmax": 8000,
            "nMels": 128,
            "hopLength": 512,
            "nFft": 2048,
            "maxLength": 22050,
            "sampleRate": 22050,
        },
    }

    foundry_fingerprint = create_foundry_adapter().contract_fingerprint_from_run_spec(run_spec, class_map=class_map)
    echozero_fingerprint = create_echozero_adapter().contract_fingerprint_from_checkpoint(checkpoint)

    assert foundry_fingerprint == echozero_fingerprint


def test_foundry_and_echozero_adapters_ignore_source_mapping_order_for_equivalent_metadata() -> None:
    class_map = ("kick", "snare")
    run_spec = {
        "model": {"type": "crnn"},
        "data": {
            "fmax": 16000,
            "nMels": 64,
            "hopLength": 256,
            "nFft": 1024,
            "maxLength": 44100,
            "sampleRate": 44100,
        },
        "classificationMode": "multiclass",
    }
    checkpoint = {
        "inference_preprocessing": {
            "sampleRate": 44100,
            "maxLength": 44100,
            "nFft": 1024,
            "hopLength": 256,
            "nMels": 64,
            "fmax": 16000,
        },
        "classes": ["kick", "snare"],
        "model_type": "crnn",
    }

    foundry_fingerprint = create_foundry_adapter().contract_fingerprint_from_run_spec(run_spec, class_map=class_map)
    echozero_fingerprint = create_echozero_adapter().contract_fingerprint_from_checkpoint(checkpoint)

    assert foundry_fingerprint == echozero_fingerprint

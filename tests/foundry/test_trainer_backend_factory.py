from __future__ import annotations

import pytest

from echozero.foundry.services import BaselineTrainer, TrainerBackendFactory


def test_resolve_defaults_to_legacy_backend(tmp_path):
    factory = TrainerBackendFactory()
    legacy = BaselineTrainer(tmp_path)

    resolved = factory.resolve(
        {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "multiclass",
            "data": {"datasetVersionId": "dsv_x", "sampleRate": 22050, "maxLength": 22050, "nFft": 2048, "hopLength": 512, "nMels": 128, "fmax": 8000},
            "training": {"epochs": 1, "batchSize": 1, "learningRate": 0.01},
        },
        legacy_backend=legacy,
    )

    assert resolved is legacy


def test_resolve_rejects_unknown_model_type(tmp_path):
    factory = TrainerBackendFactory()
    legacy = BaselineTrainer(tmp_path)

    with pytest.raises(ValueError, match="run_spec.model.type"):
        factory.resolve({"model": {"type": "transformer"}}, legacy_backend=legacy)


def test_resolve_marks_cnn_not_implemented_yet(tmp_path):
    factory = TrainerBackendFactory()
    legacy = BaselineTrainer(tmp_path)

    with pytest.raises(NotImplementedError, match="backend is not implemented yet"):
        factory.resolve({"model": {"type": "cnn"}}, legacy_backend=legacy)

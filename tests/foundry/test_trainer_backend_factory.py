from __future__ import annotations

import pytest

from echozero.foundry.services import BaselineTrainer, CnnTrainer, CrnnTrainer, TrainerBackendFactory


class _FakeBackend:
    def __init__(self, marker: str):
        self.marker = marker

    def train(self, run, dataset_version, cancel_event=None, progress_callback=None):
        del run, dataset_version, cancel_event, progress_callback
        raise NotImplementedError


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


def test_resolve_cnn_backend(tmp_path):
    factory = TrainerBackendFactory()
    legacy = BaselineTrainer(tmp_path)

    resolved = factory.resolve({"model": {"type": "cnn"}}, legacy_backend=legacy)
    assert isinstance(resolved, CnnTrainer)


def test_resolve_crnn_backend(tmp_path):
    factory = TrainerBackendFactory()
    legacy = BaselineTrainer(tmp_path)

    resolved = factory.resolve({"model": {"type": "crnn"}}, legacy_backend=legacy)
    assert isinstance(resolved, CrnnTrainer)


def test_resolve_supports_registered_custom_backend(tmp_path):
    factory = TrainerBackendFactory()
    legacy = BaselineTrainer(tmp_path)

    factory.register("custom_runtime", lambda run_spec, legacy_backend: _FakeBackend(str(run_spec.get("schema"))))

    resolved = factory.resolve(
        {
            "schema": "foundry.train_run_spec.v1",
            "training": {"backend": "custom_runtime"},
        },
        legacy_backend=legacy,
    )

    assert isinstance(resolved, _FakeBackend)
    assert resolved.marker == "foundry.train_run_spec.v1"


def test_resolve_rejects_unknown_registered_backend(tmp_path):
    factory = TrainerBackendFactory()
    legacy = BaselineTrainer(tmp_path)

    with pytest.raises(ValueError, match="run_spec.training.backend"):
        factory.resolve({"training": {"backend": "missing_backend"}}, legacy_backend=legacy)

from __future__ import annotations

import pytest

pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from echozero.foundry.services import (
    BaselineTrainer,
    CnnTrainer,
    CrnnTrainer,
    TrainerBackendFactory,
)


class _FakeBackend:
    def __init__(self, marker: str):
        self.marker = marker

    def train(self, run, dataset_version, cancel_event=None, progress_callback=None):
        del run, dataset_version, cancel_event, progress_callback
        raise NotImplementedError


def test_resolve_defaults_to_baseline_backend(tmp_path):
    factory = TrainerBackendFactory()
    baseline = BaselineTrainer(tmp_path)

    resolved = factory.resolve(
        {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "multiclass",
            "data": {
                "datasetVersionId": "dsv_x",
                "sampleRate": 22050,
                "maxLength": 22050,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "training": {"epochs": 1, "batchSize": 1, "learningRate": 0.01},
        },
        baseline_backend=baseline,
    )

    assert resolved is baseline


def test_resolve_rejects_unknown_model_type(tmp_path):
    factory = TrainerBackendFactory()
    baseline = BaselineTrainer(tmp_path)

    with pytest.raises(ValueError, match="run_spec.model.type"):
        factory.resolve({"model": {"type": "transformer"}}, baseline_backend=baseline)


def test_resolve_cnn_backend(tmp_path):
    factory = TrainerBackendFactory()
    baseline = BaselineTrainer(tmp_path)

    resolved = factory.resolve({"model": {"type": "cnn"}}, baseline_backend=baseline)
    assert isinstance(resolved, CnnTrainer)


def test_resolve_crnn_backend(tmp_path):
    factory = TrainerBackendFactory()
    baseline = BaselineTrainer(tmp_path)

    resolved = factory.resolve({"model": {"type": "crnn"}}, baseline_backend=baseline)
    assert isinstance(resolved, CrnnTrainer)


def test_crnn_trainer_resolves_explicit_and_balanced_class_weights(tmp_path):
    trainer = CrnnTrainer(tmp_path)

    explicit = trainer._resolve_class_weights(
        {"classWeights": {"kick": 1.0, "other": 3.0}},
        class_names=["kick", "other"],
        train_labels=np.asarray([0, 1, 1], dtype=np.int64),
    )
    assert explicit.tolist() == [1.0, 3.0]

    balanced = trainer._resolve_class_weights(
        {"classWeighting": "balanced"},
        class_names=["kick", "other"],
        train_labels=np.asarray([0, 1, 1, 1], dtype=np.int64),
    )
    assert balanced[0] == pytest.approx(2.0)
    assert balanced[1] == pytest.approx(2.0 / 3.0)


def test_resolve_supports_registered_custom_backend(tmp_path):
    factory = TrainerBackendFactory()
    baseline = BaselineTrainer(tmp_path)

    factory.register(
        "custom_runtime",
        lambda run_spec, baseline_backend: _FakeBackend(str(run_spec.get("schema"))),
    )

    resolved = factory.resolve(
        {
            "schema": "foundry.train_run_spec.v1",
            "training": {"backend": "custom_runtime"},
        },
        baseline_backend=baseline,
    )

    assert isinstance(resolved, _FakeBackend)
    assert resolved.marker == "foundry.train_run_spec.v1"


def test_resolve_rejects_unknown_registered_backend(tmp_path):
    factory = TrainerBackendFactory()
    baseline = BaselineTrainer(tmp_path)

    with pytest.raises(ValueError, match="run_spec.training.backend"):
        factory.resolve({"training": {"backend": "missing_backend"}}, baseline_backend=baseline)

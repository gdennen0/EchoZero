from __future__ import annotations

from typing import Protocol

from echozero.foundry.domain import DatasetVersion, TrainRun
from echozero.foundry.services.cnn_trainer import CnnTrainer
from echozero.foundry.services.crnn_trainer import CrnnTrainer


class TrainerBackend(Protocol):
    def train(
        self,
        run: TrainRun,
        dataset_version: DatasetVersion,
        cancel_event=None,
        progress_callback=None,
    ): ...


_SUPPORTED_MODEL_TYPES = {"baseline_sgd", "cnn", "crnn"}


class TrainerBackendFactory:
    """Resolves run specs to trainer backends without changing run-service contracts."""

    def resolve(self, run_spec: dict, *, legacy_backend: TrainerBackend) -> TrainerBackend:
        model = run_spec.get("model")
        model_type = "baseline_sgd"

        if model is not None:
            if not isinstance(model, dict):
                raise ValueError("run_spec.model must be an object when provided")
            model_type = str(model.get("type", "baseline_sgd")).lower()

        if model_type not in _SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"run_spec.model.type must be one of: {', '.join(sorted(_SUPPORTED_MODEL_TYPES))}"
            )

        if model_type == "baseline_sgd":
            return legacy_backend

        if model_type == "cnn":
            root = getattr(legacy_backend, "_root", None)
            if root is None:
                raise ValueError("legacy backend must expose a root path for cnn backend resolution")
            return CnnTrainer(root)

        if model_type == "crnn":
            root = getattr(legacy_backend, "_root", None)
            if root is None:
                raise ValueError("legacy backend must expose a root path for crnn backend resolution")
            return CrnnTrainer(root)

        raise ValueError(
            f"run_spec.model.type must be one of: {', '.join(sorted(_SUPPORTED_MODEL_TYPES))}"
        )

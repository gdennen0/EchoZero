from __future__ import annotations

from typing import Protocol

from echozero.foundry.domain import DatasetVersion, TrainRun


class TrainerBackend(Protocol):
    def train(self, run: TrainRun, dataset_version: DatasetVersion, cancel_event=None): ...


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

        # Phase 1 architecture wiring is complete; concrete backends land next.
        raise NotImplementedError(
            f"model.type={model_type} is wired in resolver but backend is not implemented yet"
        )

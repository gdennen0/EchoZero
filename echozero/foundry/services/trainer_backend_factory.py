from __future__ import annotations

from collections.abc import Callable
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
BackendBuilder = Callable[[dict, TrainerBackend], TrainerBackend]


class TrainerBackendFactory:
    """Resolves run specs to trainer backends via a registry for swappable runtimes."""

    def __init__(self):
        self._registry: dict[str, BackendBuilder] = {
            "baseline_sgd": self._build_baseline,
            "cnn": self._build_cnn,
            "crnn": self._build_crnn,
        }

    def register(self, backend_name: str, builder: BackendBuilder) -> None:
        key = str(backend_name).strip().lower()
        if not key:
            raise ValueError("backend_name must be a non-empty string")
        self._registry[key] = builder

    def resolve(self, run_spec: dict, *, legacy_backend: TrainerBackend) -> TrainerBackend:
        backend_key = self._resolve_backend_key(run_spec)
        builder = self._registry.get(backend_key)
        if builder is None:
            available = ", ".join(sorted(self._registry))
            raise ValueError(f"run_spec.training.backend must resolve to a known backend: {available}")
        return builder(run_spec, legacy_backend)

    def _resolve_backend_key(self, run_spec: dict) -> str:
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

        training = run_spec.get("training")
        if isinstance(training, dict) and training.get("backend") is not None:
            backend_key = str(training.get("backend")).strip().lower()
            if not backend_key:
                raise ValueError("run_spec.training.backend must be a non-empty string when provided")
            return backend_key

        return model_type

    @staticmethod
    def _build_baseline(run_spec: dict, legacy_backend: TrainerBackend) -> TrainerBackend:
        del run_spec
        return legacy_backend

    @staticmethod
    def _build_cnn(run_spec: dict, legacy_backend: TrainerBackend) -> TrainerBackend:
        del run_spec
        root = getattr(legacy_backend, "_root", None)
        if root is None:
            raise ValueError("legacy backend must expose a root path for cnn backend resolution")
        return CnnTrainer(root)

    @staticmethod
    def _build_crnn(run_spec: dict, legacy_backend: TrainerBackend) -> TrainerBackend:
        del run_spec
        root = getattr(legacy_backend, "_root", None)
        if root is None:
            raise ValueError("legacy backend must expose a root path for crnn backend resolution")
        return CrnnTrainer(root)

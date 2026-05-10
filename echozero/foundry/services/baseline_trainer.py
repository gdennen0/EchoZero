"""Foundry baseline trainer runtime.
Exists to train and evaluate the baseline classifier from prepared dataset features.
Connects Foundry training runs and artifact output to the baseline model pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import ModuleType
from typing import Callable, cast

import numpy as np

from echozero.foundry.domain import DatasetSample, DatasetVersion, TrainRun


class MissingFoundryMlDependencyError(RuntimeError):
    """Raised when a Foundry training runtime is used without ML extras installed."""


def _load_baseline_runtime() -> ModuleType:
    try:
        from echozero.foundry.services import baseline_trainer_runtime
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "sklearn"}:
            raise MissingFoundryMlDependencyError(
                "Foundry training requires optional ML dependencies. "
                'Install them with `pip install -e ".[ml]"` before starting a training run.'
            ) from exc
        raise
    return baseline_trainer_runtime


class RunCanceledError(RuntimeError):
    pass


@dataclass(slots=True)
class BaselineTrainingResult:
    checkpoint_metrics: list[dict[str, float | int | None]]
    final_metrics: dict[str, float | int | str]
    aggregate_metrics: dict[str, float | int]
    per_class_metrics: dict[str, dict[str, float | int]]
    confusion: dict[str, list[object]]
    summary: dict[str, str | float | bool]
    baseline: dict[str, object]
    artifact_manifest: dict[str, object]
    model_path: Path
    metrics_path: Path
    run_summary_path: Path
    eval_split_name: str
    synthetic_eval: dict[str, object] | None = None


class BaselineTrainer:
    def __init__(self, root: Path):
        self._root = root

    def train(
        self,
        run: TrainRun,
        dataset_version: DatasetVersion,
        cancel_event: Event | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> BaselineTrainingResult:
        payload = _load_baseline_runtime().run_baseline_training(
            host=self,
            run=run,
            dataset_version=dataset_version,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        return BaselineTrainingResult(**payload)

    @staticmethod
    def _resolve_training_options(training_spec: dict) -> dict[str, object]:
        return cast(
            dict[str, object], _load_baseline_runtime().resolve_training_options(training_spec)
        )

    @staticmethod
    def _resolve_train_samples(
        train_samples: list[DatasetSample],
        synthetic_mix_spec: dict[str, object],
        *,
        rng: np.random.Generator,
    ) -> tuple[list[DatasetSample], dict[str, int | float | bool | None]]:
        return cast(
            tuple[list[DatasetSample], dict[str, int | float | bool | None]],
            _load_baseline_runtime().resolve_train_samples(
                train_samples,
                synthetic_mix_spec,
                rng=rng,
            ),
        )

    @staticmethod
    def _compute_class_weight(
        y: np.ndarray,
        *,
        classes: np.ndarray,
        mode: str,
    ) -> dict[int, float] | None:
        return cast(
            dict[int, float] | None,
            _load_baseline_runtime().compute_class_weight(y, classes=classes, mode=mode),
        )

    @staticmethod
    def _rebalance_training_set(
        x: np.ndarray,
        y: np.ndarray,
        *,
        class_count: int,
        strategy: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        return cast(
            tuple[np.ndarray, np.ndarray],
            _load_baseline_runtime().rebalance_training_set(
                x,
                y,
                class_count=class_count,
                strategy=strategy,
                rng=rng,
            ),
        )

    @staticmethod
    def _augment_features(
        x: np.ndarray,
        y: np.ndarray,
        *,
        copies: int,
        noise_std: float,
        gain_jitter: float,
        enabled: bool,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        return cast(
            tuple[np.ndarray, np.ndarray],
            _load_baseline_runtime().augment_features(
                x,
                y,
                copies=copies,
                noise_std=noise_std,
                gain_jitter=gain_jitter,
                enabled=enabled,
                rng=rng,
            ),
        )

    @staticmethod
    def _evaluate_split(
        classifier: object,
        x: np.ndarray,
        y: np.ndarray,
        class_names: list[str],
    ) -> dict[str, object]:
        return cast(
            dict[str, object],
            _load_baseline_runtime().evaluate_split(classifier, x, y, class_names),
        )

    def _build_features(
        self,
        samples: list[DatasetSample],
        *,
        sample_rate: int,
        max_length: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        fmax: int,
        label_to_index: dict[str, int],
        cancel_event: Event | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return cast(
            tuple[np.ndarray, np.ndarray],
            _load_baseline_runtime().build_features(
                samples,
                sample_rate=sample_rate,
                max_length=max_length,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmax=fmax,
                label_to_index=label_to_index,
                cancel_event=cancel_event,
                ensure_not_canceled_fn=self._ensure_not_canceled,
                load_audio_fn=lambda path: self._load_audio(
                    path,
                    sample_rate=sample_rate,
                    max_length=max_length,
                ),
            ),
        )

    @staticmethod
    def _ensure_not_canceled(cancel_event: Event | None) -> None:
        _load_baseline_runtime().ensure_not_canceled(
            cancel_event,
            canceled_error_cls=RunCanceledError,
        )

    @staticmethod
    def _load_audio(path: Path, *, sample_rate: int, max_length: int) -> np.ndarray:
        return cast(
            np.ndarray,
            _load_baseline_runtime().load_audio(
                path,
                sample_rate=sample_rate,
                max_length=max_length,
            ),
        )


__all__ = [
    "BaselineTrainer",
    "BaselineTrainingResult",
    "MissingFoundryMlDependencyError",
    "RunCanceledError",
]

"""Foundry baseline trainer runtime.
Exists to train and evaluate the baseline classifier from prepared dataset features.
Connects Foundry training runs and artifact output to the baseline model pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, cast

import numpy as np

from echozero.foundry.domain import DatasetSample, DatasetVersion, TrainRun
from echozero.foundry.services.baseline_trainer_runtime import (
    augment_features,
    build_features,
    compute_class_weight,
    ensure_not_canceled,
    evaluate_split,
    load_audio,
    rebalance_training_set,
    resolve_train_samples,
    resolve_training_options,
    run_baseline_training,
)


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
        payload = run_baseline_training(
            host=self,
            run=run,
            dataset_version=dataset_version,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        return BaselineTrainingResult(**payload)

    _resolve_training_options = staticmethod(resolve_training_options)
    _resolve_train_samples = staticmethod(resolve_train_samples)
    _compute_class_weight = staticmethod(compute_class_weight)
    _rebalance_training_set = staticmethod(rebalance_training_set)
    _augment_features = staticmethod(augment_features)
    _evaluate_split = staticmethod(evaluate_split)

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
            build_features(
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
        ensure_not_canceled(cancel_event, canceled_error_cls=RunCanceledError)

    @staticmethod
    def _load_audio(path: Path, *, sample_rate: int, max_length: int) -> np.ndarray:
        return cast(np.ndarray, load_audio(path, sample_rate=sample_rate, max_length=max_length))


__all__ = ["BaselineTrainer", "BaselineTrainingResult", "RunCanceledError"]

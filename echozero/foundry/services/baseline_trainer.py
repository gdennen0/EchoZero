from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from echozero.foundry.domain import DatasetSample, DatasetVersion, TrainRun
from echozero.foundry.services.training_runtime import (
    compute_config_fingerprint,
    configure_reproducibility,
    ensure_finite_array,
)


class RunCanceledError(RuntimeError):
    pass


@dataclass(slots=True)
class BaselineTrainingResult:
    checkpoint_metrics: list[dict[str, float | int | None]]
    final_metrics: dict[str, float | int | str]
    aggregate_metrics: dict[str, float | int]
    per_class_metrics: dict[str, dict[str, float | int]]
    confusion: dict[str, list]
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
        data_spec = run.spec["data"]
        training_spec = run.spec["training"]
        sample_rate = int(data_spec["sampleRate"])
        max_length = int(data_spec["maxLength"])
        n_fft = int(data_spec["nFft"])
        hop_length = int(data_spec["hopLength"])
        n_mels = int(data_spec["nMels"])
        fmax = int(data_spec["fmax"])
        epochs = int(training_spec["epochs"])
        learning_rate = float(training_spec["learningRate"])
        random_seed = int(training_spec.get("seed", 17))
        deterministic = bool(training_spec.get("deterministic", True))
        batch_size = int(training_spec.get("batchSize", 16))

        reproducibility = configure_reproducibility(random_seed, deterministic=deterministic)
        config_fingerprint = compute_config_fingerprint(
            {
                "schema": "foundry.training_fingerprint.v1",
                "runSpec": run.spec,
                "datasetVersionId": dataset_version.id,
                "datasetManifestHash": dataset_version.manifest_hash,
                "classMap": list(dataset_version.class_map),
            }
        )

        options = self._resolve_training_options(training_spec)
        rng = np.random.default_rng(random_seed)

        sample_by_id = {sample.sample_id: sample for sample in dataset_version.samples}
        split_plan = dataset_version.split_plan or {}
        train_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("train_ids", [])]
        val_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("val_ids", []) if sample_id in sample_by_id]
        test_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("test_ids", []) if sample_id in sample_by_id]
        train_samples, synthetic_mix = self._resolve_train_samples(train_samples, options["synthetic_mix"], rng=rng)
        val_samples = [sample for sample in val_samples if not sample.is_synthetic]
        test_samples = [sample for sample in test_samples if not sample.is_synthetic]
        synthetic_eval_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("val_ids", []) if sample_id in sample_by_id]
        synthetic_eval_samples.extend(
            [sample_by_id[sample_id] for sample_id in split_plan.get("test_ids", []) if sample_id in sample_by_id]
        )
        synthetic_eval_samples = [sample for sample in synthetic_eval_samples if sample.is_synthetic]

        if len(train_samples) < 2:
            raise ValueError("baseline trainer requires at least two training samples")

        class_names = list(dataset_version.class_map)
        label_to_index = {label: index for index, label in enumerate(class_names)}
        train_label_ids = {sample.label for sample in train_samples}
        if train_label_ids != set(class_names):
            missing = sorted(set(class_names) - train_label_ids)
            raise ValueError(f"training split is missing classes required for baseline training: {', '.join(missing)}")

        train_x_raw, train_y_raw = self._build_features(
            train_samples,
            sample_rate=sample_rate,
            max_length=max_length,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax,
            label_to_index=label_to_index,
            cancel_event=cancel_event,
        )
        val_x, val_y = self._build_features(
            val_samples,
            sample_rate=sample_rate,
            max_length=max_length,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax,
            label_to_index=label_to_index,
            cancel_event=cancel_event,
        )
        eval_samples = test_samples or val_samples
        eval_split_name = "test" if test_samples else "val"
        eval_x, eval_y = self._build_features(
            eval_samples,
            sample_rate=sample_rate,
            max_length=max_length,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax,
            label_to_index=label_to_index,
            cancel_event=cancel_event,
        )
        synthetic_eval_x, synthetic_eval_y = self._build_features(
            synthetic_eval_samples,
            sample_rate=sample_rate,
            max_length=max_length,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax,
            label_to_index=label_to_index,
            cancel_event=cancel_event,
        )

        train_x, train_y = self._rebalance_training_set(
            train_x_raw,
            train_y_raw,
            class_count=len(class_names),
            strategy=options["rebalance_strategy"],
            rng=rng,
        )
        ensure_finite_array("baseline/train_x", train_x, context="post_rebalance")

        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        val_x_scaled = scaler.transform(val_x) if len(val_x) else np.empty((0, train_x_scaled.shape[1]), dtype=np.float32)
        eval_x_scaled = scaler.transform(eval_x) if len(eval_x) else np.empty((0, train_x_scaled.shape[1]), dtype=np.float32)
        ensure_finite_array("baseline/train_x_scaled", train_x_scaled, context="scaler.fit_transform")
        ensure_finite_array("baseline/val_x_scaled", val_x_scaled, context="scaler.transform")
        ensure_finite_array("baseline/eval_x_scaled", eval_x_scaled, context="scaler.transform")

        classes = np.arange(len(class_names), dtype=np.int64)
        class_weight = self._compute_class_weight(train_y_raw, classes=classes, mode=options["class_weighting"])
        classifier = SGDClassifier(
            loss="log_loss",
            learning_rate="optimal" if options["optimizer"] == "sgd_optimal" else "constant",
            eta0=learning_rate,
            alpha=float(options["regularization_alpha"]),
            random_state=random_seed,
            shuffle=False,
            class_weight=class_weight,
            average=bool(options["average_weights"]),
        )

        checkpoint_metrics: list[dict[str, float | int | None]] = []
        best_classifier: SGDClassifier | None = None
        best_epoch = 0
        best_primary_metric = float("-inf")
        best_split_name = "val" if len(val_y) else "train"
        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            self._ensure_not_canceled(cancel_event)
            epoch_x, epoch_y = self._augment_features(
                train_x_scaled,
                train_y,
                copies=options["augment_copies"],
                noise_std=options["augment_noise_std"],
                gain_jitter=options["augment_gain_jitter"],
                enabled=options["augment_train"],
                rng=rng,
            )
            ensure_finite_array("baseline/epoch_x", epoch_x, context=f"epoch={epoch}")
            classifier.partial_fit(epoch_x, epoch_y, classes=classes)
            train_epoch = self._evaluate_split(classifier, train_x_scaled, train_y, class_names)
            val_epoch = self._evaluate_split(classifier, val_x_scaled, val_y, class_names) if len(val_y) else {}
            train_epoch_metrics = train_epoch.get("metrics", {})
            val_epoch_metrics = val_epoch.get("metrics", {})
            checkpoint = {
                "epoch": epoch,
                "train_loss": train_epoch_metrics.get("loss"),
                "train_accuracy": train_epoch_metrics.get("accuracy"),
                "train_macro_f1": train_epoch_metrics.get("macro_f1"),
                "val_loss": val_epoch_metrics.get("loss"),
                "val_accuracy": val_epoch_metrics.get("accuracy"),
                "val_macro_f1": val_epoch_metrics.get("macro_f1"),
            }
            checkpoint_metrics.append(checkpoint)
            if progress_callback is not None:
                progress_callback({"epoch": epoch, "total_epochs": epochs, "checkpoint": checkpoint})

            monitor_metrics = val_epoch_metrics if val_epoch_metrics else train_epoch_metrics
            metric_name = "val_macro_f1" if val_epoch else "train_macro_f1"
            current_primary_metric = float(monitor_metrics.get("macro_f1", float("-inf")))
            if current_primary_metric > best_primary_metric:
                best_primary_metric = current_primary_metric
                best_epoch = epoch
                best_split_name = "val" if val_epoch else "train"
                best_classifier = copy.deepcopy(classifier)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            patience = options["early_stopping_patience"]
            min_epochs = options["min_epochs"]
            if (
                patience is not None
                and epoch >= min_epochs
                and epochs_without_improvement >= patience
            ):
                checkpoint_metrics[-1]["stopped_early"] = 1
                checkpoint_metrics[-1]["best_metric_name"] = metric_name
                checkpoint_metrics[-1]["best_metric_value"] = best_primary_metric
                break

        if best_classifier is not None:
            classifier = best_classifier

        self._ensure_not_canceled(cancel_event)
        final_eval = self._evaluate_split(classifier, eval_x_scaled, eval_y, class_names)
        if not final_eval:
            raise ValueError("baseline trainer requires validation or test samples for evaluation")
        synthetic_eval_x_scaled = (
            scaler.transform(synthetic_eval_x)
            if len(synthetic_eval_y)
            else np.empty((0, train_x_scaled.shape[1]), dtype=np.float32)
        )
        synthetic_eval = (
            self._evaluate_split(classifier, synthetic_eval_x_scaled, synthetic_eval_y, class_names)
            if len(synthetic_eval_y)
            else None
        )

        run.exports_dir(self._root).mkdir(parents=True, exist_ok=True)
        model_path = run.exports_dir(self._root) / "model.pth"
        torch.save(
            {
                "schema": "foundry.baseline_model.v1",
                "trainer": "baseline_sgd_melspec_v1_5",
                "classes": class_names,
                "classification_mode": run.spec["classificationMode"],
                "coef": classifier.coef_.astype(np.float32),
                "intercept": classifier.intercept_.astype(np.float32),
                "scaler_mean": scaler.mean_.astype(np.float32),
                "scaler_scale": scaler.scale_.astype(np.float32),
                "preprocessing": {
                    "sampleRate": sample_rate,
                    "maxLength": max_length,
                    "nFft": n_fft,
                    "hopLength": hop_length,
                    "nMels": n_mels,
                    "fmax": fmax,
                    "featurePooling": ["mean", "std", "max"],
                },
                "training": {
                    "epochs": epochs,
                    "completedEpochs": len(checkpoint_metrics),
                    "learningRate": learning_rate,
                    "batchSize": batch_size,
                    "seed": random_seed,
                    "deterministic": deterministic,
                    "configFingerprint": config_fingerprint,
                    "trainerProfile": options["trainer_profile"],
                    "optimizer": options["optimizer"],
                    "regularizationAlpha": options["regularization_alpha"],
                    "averageWeights": options["average_weights"],
                    "earlyStoppingPatience": options["early_stopping_patience"],
                    "minEpochs": options["min_epochs"],
                    "classWeighting": options["class_weighting"],
                    "rebalanceStrategy": options["rebalance_strategy"],
                    "augmentTrain": options["augment_train"],
                    "augmentCopies": options["augment_copies"],
                    "augmentNoiseStd": options["augment_noise_std"],
                    "augmentGainJitter": options["augment_gain_jitter"],
                    "syntheticMix": synthetic_mix,
                },
                "metrics": final_eval["metrics"],
            },
            model_path,
        )

        metrics_payload = {
            "schema": "foundry.training_metrics.v1",
            "runId": run.id,
            "datasetVersionId": dataset_version.id,
            "classificationMode": run.spec["classificationMode"],
            "checkpoints": checkpoint_metrics,
            "finalEval": final_eval,
            "trainerOptions": {
                "trainerProfile": options["trainer_profile"],
                "optimizer": options["optimizer"],
                "regularizationAlpha": options["regularization_alpha"],
                "averageWeights": options["average_weights"],
                "earlyStoppingPatience": options["early_stopping_patience"],
                "minEpochs": options["min_epochs"],
                "classWeighting": options["class_weighting"],
                "rebalanceStrategy": options["rebalance_strategy"],
                "augmentTrain": options["augment_train"],
                "augmentCopies": options["augment_copies"],
                "augmentNoiseStd": options["augment_noise_std"],
                "augmentGainJitter": options["augment_gain_jitter"],
                "syntheticMix": synthetic_mix,
            },
            "reproducibility": {
                **reproducibility,
                "configFingerprint": config_fingerprint,
            },
        }
        if synthetic_eval is not None:
            metrics_payload["syntheticEval"] = synthetic_eval
        metrics_path = run.exports_dir(self._root) / "metrics.json"
        self._ensure_not_canceled(cancel_event)
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

        run_summary_payload = {
            "runId": run.id,
            "status": "completed",
            "evalSplit": eval_split_name,
            "modelPath": model_path.name,
            "metricsPath": metrics_path.name,
            "primaryMetric": final_eval["metrics"]["macro_f1"],
            "accuracy": final_eval["metrics"]["accuracy"],
            "trainerProfile": options["trainer_profile"],
            "dataProfile": "next_level_v1_5" if any(
                [
                    options["augment_train"],
                    options["rebalance_strategy"] != "none",
                    options["class_weighting"] != "none",
                    synthetic_mix["actualSyntheticCount"] > 0,
                ]
            ) else "baseline_v1",
            "completedEpochs": len(checkpoint_metrics),
            "bestCheckpointEpoch": best_epoch,
            "bestCheckpointMetric": best_primary_metric if best_epoch else None,
            "bestCheckpointSplit": best_split_name if best_epoch else None,
            "syntheticMix": synthetic_mix,
            "reproducibility": {
                **reproducibility,
                "configFingerprint": config_fingerprint,
            },
        }
        if synthetic_eval is not None:
            run_summary_payload["syntheticEval"] = {
                "sampleCount": synthetic_eval["metrics"]["sample_count"],
                "accuracy": synthetic_eval["metrics"]["accuracy"],
                "macroF1": synthetic_eval["metrics"]["macro_f1"],
            }
        run_summary_path = run.exports_dir(self._root) / "run_summary.json"
        self._ensure_not_canceled(cancel_event)
        run_summary_path.write_text(json.dumps(run_summary_payload, indent=2, sort_keys=True), encoding="utf-8")

        return BaselineTrainingResult(
            checkpoint_metrics=checkpoint_metrics,
            final_metrics=final_eval["metrics"],
            aggregate_metrics=final_eval["aggregate_metrics"],
            per_class_metrics=final_eval["per_class_metrics"],
            confusion=final_eval["confusion"],
            summary={
                "primary_metric": "macro_f1",
                "primary_metric_value": float(final_eval["metrics"]["macro_f1"]),
                "split_name": eval_split_name,
                "supports_threshold_tuning": False,
            },
            baseline={
                "family": "baseline_sgd_melspec_v1_5",
                "profile": options["trainer_profile"],
                "optimizer": options["optimizer"],
                "epochs": epochs,
                "completed_epochs": len(checkpoint_metrics),
                "checkpoint_epoch": best_epoch or len(checkpoint_metrics),
                "early_stopping_patience": options["early_stopping_patience"],
                "min_epochs": options["min_epochs"],
                "regularization_alpha": options["regularization_alpha"],
                "average_weights": options["average_weights"],
                "feature_pooling": ["mean", "std", "max"],
                "class_weighting": options["class_weighting"],
                "rebalance_strategy": options["rebalance_strategy"],
                "augment_train": options["augment_train"],
                "augment_copies": options["augment_copies"],
                "augment_noise_std": options["augment_noise_std"],
                "augment_gain_jitter": options["augment_gain_jitter"],
                "synthetic_mix": synthetic_mix,
                "reproducibility": {
                    **reproducibility,
                    "config_fingerprint": config_fingerprint,
                },
            },
            artifact_manifest={
                "weightsPath": model_path.name,
                "classes": class_names,
                "classificationMode": run.spec["classificationMode"],
                "inferencePreprocessing": {
                    "sampleRate": sample_rate,
                    "maxLength": max_length,
                    "nFft": n_fft,
                    "hopLength": hop_length,
                    "nMels": n_mels,
                    "fmax": fmax,
                },
                "evalSummary": {
                    "splitName": eval_split_name,
                    "accuracy": float(final_eval["metrics"]["accuracy"]),
                    "macroF1": float(final_eval["metrics"]["macro_f1"]),
                },
                "trainingSummary": {
                    "trainer": "baseline_sgd_melspec_v1_5",
                    "trainerProfile": options["trainer_profile"],
                    "optimizer": options["optimizer"],
                    "bestCheckpointEpoch": best_epoch,
                    "metricsPath": metrics_path.name,
                    "runSummaryPath": run_summary_path.name,
                    "classWeighting": options["class_weighting"],
                    "rebalanceStrategy": options["rebalance_strategy"],
                    "augmentTrain": options["augment_train"],
                    "syntheticMix": synthetic_mix,
                    "reproducibility": {
                        **reproducibility,
                        "configFingerprint": config_fingerprint,
                    },
                },
            },
            model_path=model_path,
            metrics_path=metrics_path,
            run_summary_path=run_summary_path,
            eval_split_name=eval_split_name,
            synthetic_eval=synthetic_eval,
        )

    @staticmethod
    def _resolve_training_options(training_spec: dict) -> dict[str, object]:
        trainer_profile = str(training_spec.get("trainerProfile", "baseline_v1")).lower()
        if trainer_profile not in {"baseline_v1", "stronger_v1"}:
            raise ValueError("run_spec.training.trainerProfile must be one of: baseline_v1, stronger_v1")

        optimizer = str(training_spec.get("optimizer", "sgd_constant")).lower()
        if optimizer not in {"sgd_constant", "sgd_optimal"}:
            raise ValueError("run_spec.training.optimizer must be one of: sgd_constant, sgd_optimal")

        regularization_alpha = float(training_spec.get("regularizationAlpha", 0.0001))
        if regularization_alpha <= 0:
            raise ValueError("run_spec.training.regularizationAlpha must be > 0")

        average_weights = bool(training_spec.get("averageWeights", False))
        early_stopping_patience = training_spec.get("earlyStoppingPatience")
        if early_stopping_patience is not None and int(early_stopping_patience) < 1:
            raise ValueError("run_spec.training.earlyStoppingPatience must be >= 1")
        min_epochs = training_spec.get("minEpochs")
        if min_epochs is not None and int(min_epochs) < 1:
            raise ValueError("run_spec.training.minEpochs must be >= 1")

        if trainer_profile == "stronger_v1":
            if optimizer == "sgd_constant" and "optimizer" not in training_spec:
                optimizer = "sgd_optimal"
            average_weights = True if "averageWeights" not in training_spec else average_weights
            if early_stopping_patience is None:
                early_stopping_patience = 3
            if min_epochs is None:
                min_epochs = min(int(training_spec.get("epochs", 1)), 3)

        class_weighting = str(training_spec.get("classWeighting", "none")).lower()
        if class_weighting not in {"none", "balanced"}:
            raise ValueError("run_spec.training.classWeighting must be one of: none, balanced")

        rebalance_strategy = str(training_spec.get("rebalanceStrategy", "none")).lower()
        if rebalance_strategy not in {"none", "oversample"}:
            raise ValueError("run_spec.training.rebalanceStrategy must be one of: none, oversample")

        augment_train = bool(training_spec.get("augmentTrain", False))
        augment_noise_std = float(training_spec.get("augmentNoiseStd", 0.02))
        augment_gain_jitter = float(training_spec.get("augmentGainJitter", 0.10))
        augment_copies = int(training_spec.get("augmentCopies", 1))
        if augment_noise_std < 0:
            raise ValueError("run_spec.training.augmentNoiseStd must be >= 0")
        if augment_gain_jitter < 0:
            raise ValueError("run_spec.training.augmentGainJitter must be >= 0")
        if augment_copies < 0:
            raise ValueError("run_spec.training.augmentCopies must be >= 0")

        synthetic_mix_spec = training_spec.get("syntheticMix") or {}
        if not isinstance(synthetic_mix_spec, dict):
            raise ValueError("run_spec.training.syntheticMix must be an object")
        synthetic_enabled = bool(synthetic_mix_spec.get("enabled", False))
        synthetic_ratio = float(synthetic_mix_spec.get("ratio", 0.0))
        synthetic_cap = synthetic_mix_spec.get("cap")
        if synthetic_ratio < 0 or synthetic_ratio > 1:
            raise ValueError("run_spec.training.syntheticMix.ratio must be between 0 and 1")
        if synthetic_cap is not None and int(synthetic_cap) < 0:
            raise ValueError("run_spec.training.syntheticMix.cap must be >= 0")

        return {
            "trainer_profile": trainer_profile,
            "optimizer": optimizer,
            "regularization_alpha": regularization_alpha,
            "average_weights": average_weights,
            "early_stopping_patience": None if early_stopping_patience is None else int(early_stopping_patience),
            "min_epochs": 1 if min_epochs is None else int(min_epochs),
            "class_weighting": class_weighting,
            "rebalance_strategy": rebalance_strategy,
            "augment_train": augment_train,
            "augment_noise_std": augment_noise_std,
            "augment_gain_jitter": augment_gain_jitter,
            "augment_copies": augment_copies,
            "synthetic_mix": {
                "enabled": synthetic_enabled,
                "ratio": synthetic_ratio,
                "cap": None if synthetic_cap is None else int(synthetic_cap),
            },
        }

    @staticmethod
    def _resolve_train_samples(
        train_samples: list[DatasetSample],
        synthetic_mix_spec: dict[str, object],
        *,
        rng: np.random.Generator,
    ) -> tuple[list[DatasetSample], dict[str, int | float | bool | None]]:
        real_samples = [sample for sample in train_samples if not sample.is_synthetic]
        synthetic_samples = [sample for sample in train_samples if sample.is_synthetic]
        enabled = bool(synthetic_mix_spec.get("enabled", False))
        ratio = float(synthetic_mix_spec.get("ratio", 0.0))
        cap = synthetic_mix_spec.get("cap")
        cap_value = None if cap is None else int(cap)

        selected_synthetic: list[DatasetSample] = []
        if enabled and synthetic_samples:
            ratio_limit = int(len(real_samples) * ratio)
            max_synthetic = ratio_limit
            if cap_value is not None:
                max_synthetic = min(max_synthetic, cap_value) if max_synthetic > 0 else cap_value
            max_synthetic = max(0, min(max_synthetic, len(synthetic_samples)))
            if max_synthetic >= len(synthetic_samples):
                selected_synthetic = list(synthetic_samples)
            elif max_synthetic > 0:
                selected_indices = np.sort(rng.choice(len(synthetic_samples), size=max_synthetic, replace=False))
                selected_synthetic = [synthetic_samples[index] for index in selected_indices.tolist()]

        resolved = list(real_samples) + selected_synthetic
        return resolved, {
            "enabled": enabled,
            "ratio": ratio,
            "cap": cap_value,
            "availableSyntheticCount": len(synthetic_samples),
            "actualSyntheticCount": len(selected_synthetic),
            "realTrainCount": len(real_samples),
            "totalTrainCount": len(resolved),
        }

    @staticmethod
    def _compute_class_weight(y: np.ndarray, *, classes: np.ndarray, mode: str) -> dict[int, float] | None:
        if mode != "balanced" or len(y) == 0:
            return None
        total = float(len(y))
        num_classes = float(len(classes))
        weights: dict[int, float] = {}
        for cls in classes:
            count = float(np.sum(y == cls))
            if count <= 0:
                continue
            weights[int(cls)] = total / (num_classes * count)
        return weights or None

    @staticmethod
    def _rebalance_training_set(
        x: np.ndarray,
        y: np.ndarray,
        *,
        class_count: int,
        strategy: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if strategy != "oversample" or len(y) == 0:
            return x, y

        class_indices: list[np.ndarray] = [np.where(y == cls)[0] for cls in range(class_count)]
        non_empty = [idx for idx in class_indices if len(idx) > 0]
        if not non_empty:
            return x, y

        target = max(len(idx) for idx in non_empty)
        selected: list[int] = []
        for idx in class_indices:
            if len(idx) == 0:
                continue
            if len(idx) < target:
                extra = rng.choice(idx, size=target - len(idx), replace=True)
                merged = np.concatenate([idx, extra])
            else:
                merged = idx
            selected.extend(merged.tolist())

        selected_arr = np.asarray(selected, dtype=np.int64)
        rng.shuffle(selected_arr)
        return x[selected_arr], y[selected_arr]

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
        if not enabled or copies <= 0 or len(y) == 0:
            return x, y

        feature_batches = [x]
        label_batches = [y]
        for _ in range(copies):
            gain = rng.uniform(1.0 - gain_jitter, 1.0 + gain_jitter, size=(x.shape[0], 1)).astype(np.float32)
            noise = rng.normal(loc=0.0, scale=noise_std, size=x.shape).astype(np.float32)
            aug = (x.astype(np.float32) * gain) + noise
            feature_batches.append(aug)
            label_batches.append(y)

        return np.vstack(feature_batches), np.concatenate(label_batches)

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
        if not samples:
            feature_size = n_mels * 3
            return np.empty((0, feature_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

        features: list[np.ndarray] = []
        labels: list[int] = []
        for sample in samples:
            self._ensure_not_canceled(cancel_event)
            audio = self._load_audio(Path(sample.audio_ref), sample_rate=sample_rate, max_length=max_length)
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmax=fmax,
                power=2.0,
            )
            mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max)
            ensure_finite_array("baseline/mel_db", mel_db, context=f"sample_id={sample.sample_id}")
            pooled = np.concatenate(
                [
                    mel_db.mean(axis=1),
                    mel_db.std(axis=1),
                    mel_db.max(axis=1),
                ]
            ).astype(np.float32)
            ensure_finite_array("baseline/pooled", pooled, context=f"sample_id={sample.sample_id}")
            features.append(pooled)
            labels.append(label_to_index[sample.label])

        return np.vstack(features), np.asarray(labels, dtype=np.int64)

    @staticmethod
    def _ensure_not_canceled(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCanceledError("run canceled")

    @staticmethod
    def _load_audio(path: Path, *, sample_rate: int, max_length: int) -> np.ndarray:
        audio, file_sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sample_rate != sample_rate:
            audio = librosa.resample(audio, orig_sr=file_sample_rate, target_sr=sample_rate)
        if len(audio) > max_length:
            audio = audio[:max_length]
        elif len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)))
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0:
            audio = audio / peak
        audio = audio.astype(np.float32)
        ensure_finite_array("baseline/audio", audio, context=str(path))
        return audio

    @staticmethod
    def _evaluate_split(
        classifier: SGDClassifier,
        x: np.ndarray,
        y: np.ndarray,
        class_names: list[str],
    ) -> dict[str, object]:
        if len(y) == 0:
            return {}

        probabilities = classifier.predict_proba(x)
        ensure_finite_array("baseline/probabilities", probabilities, context="eval")
        predictions = probabilities.argmax(axis=1)
        labels = np.arange(len(class_names), dtype=np.int64)
        precision, recall, f1, support = precision_recall_fscore_support(
            y,
            predictions,
            labels=labels,
            zero_division=0,
        )
        matrix = confusion_matrix(y, predictions, labels=labels)
        metrics = {
            "loss": float(log_loss(y, probabilities, labels=labels)),
            "accuracy": float(accuracy_score(y, predictions)),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "sample_count": int(len(y)),
        }
        per_class_metrics = {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(class_names)
        }
        return {
            "metrics": metrics,
            "aggregate_metrics": metrics,
            "per_class_metrics": per_class_metrics,
            "confusion": {"labels": class_names, "matrix": matrix.astype(int).tolist()},
        }

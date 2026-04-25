from __future__ import annotations

import copy
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support

from echozero.foundry.domain import DatasetSample, DatasetVersion, TrainRun
from echozero.runtime_models.architectures import CrnnRuntimeModel
from echozero.foundry.services.baseline_trainer import BaselineTrainer, BaselineTrainingResult, RunCanceledError
from echozero.foundry.services.audio_source_validation import InvalidAudioSourceError


_Crnn = CrnnRuntimeModel


@dataclass(slots=True)
class _TensorDataset:
    x: np.ndarray
    y: np.ndarray


class CrnnTrainer:
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
        batch_size = int(training_spec.get("batchSize", 16))
        learning_rate = float(training_spec["learningRate"])
        seed = int(training_spec.get("seed", 17))
        synthetic_mix_spec = training_spec.get("syntheticMix") or {}
        gradient_clip_norm = float(training_spec.get("gradientClipNorm", 1.0))
        weight_decay = float(training_spec.get("weightDecay", 0.0001))
        early_stopping_patience = training_spec.get("earlyStoppingPatience")
        min_epochs = int(training_spec.get("minEpochs", 1))

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        sample_by_id = {sample.sample_id: sample for sample in dataset_version.samples}
        split_plan = dataset_version.split_plan or {}
        train_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("train_ids", [])]
        val_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("val_ids", []) if sample_id in sample_by_id]
        test_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("test_ids", []) if sample_id in sample_by_id]
        train_samples, synthetic_mix = BaselineTrainer._resolve_train_samples(train_samples, synthetic_mix_spec, rng=rng)
        val_samples = [sample for sample in val_samples if not sample.is_synthetic]
        test_samples = [sample for sample in test_samples if not sample.is_synthetic]
        synthetic_eval_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("val_ids", []) if sample_id in sample_by_id]
        synthetic_eval_samples.extend(
            [sample_by_id[sample_id] for sample_id in split_plan.get("test_ids", []) if sample_id in sample_by_id]
        )
        synthetic_eval_samples = [sample for sample in synthetic_eval_samples if sample.is_synthetic]

        if len(train_samples) < 2:
            raise ValueError("crnn trainer requires at least two training samples")

        class_names = list(dataset_version.class_map)
        label_to_index = {label: index for index, label in enumerate(class_names)}
        train_label_ids = {sample.label for sample in train_samples}
        if train_label_ids != set(class_names):
            missing = sorted(set(class_names) - train_label_ids)
            raise ValueError(f"training split is missing classes required for crnn training: {', '.join(missing)}")

        train_ds = self._build_dataset(
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
        if len(train_ds.y) < 2:
            raise ValueError("crnn trainer has fewer than two usable training samples after skipping invalid source audio")
        observed_train_labels = {class_names[index] for index in np.unique(train_ds.y)}
        if observed_train_labels != set(class_names):
            missing = sorted(set(class_names) - observed_train_labels)
            raise ValueError(
                "crnn trainer skipped invalid source audio and training split lost required classes: "
                + ", ".join(missing)
            )
        val_ds = self._build_dataset(
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
        eval_ds = self._build_dataset(
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
        if eval_samples and len(eval_ds.y) == 0:
            raise ValueError("crnn trainer has no usable evaluation samples after skipping invalid source audio")
        synthetic_eval_ds = self._build_dataset(
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

        model = _Crnn(num_classes=len(class_names), mel_bins=n_mels)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        checkpoint_metrics: list[dict[str, float | int | None]] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_primary_metric = float("-inf")
        best_split_name = "val" if len(val_ds.y) else "train"
        epochs_without_improvement = 0
        start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            self._ensure_not_canceled(cancel_event)
            model.train()
            permutation = rng.permutation(len(train_ds.y))
            total_loss = 0.0
            sample_count = 0
            for start_idx in range(0, len(permutation), max(1, batch_size)):
                idx = permutation[start_idx : start_idx + max(1, batch_size)]
                xb = torch.from_numpy(train_ds.x[idx])
                yb = torch.from_numpy(train_ds.y[idx])
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                if not torch.isfinite(loss):
                    raise ValueError(f"crnn trainer encountered non-finite loss at epoch {epoch}")
                loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                total_loss += float(loss.item()) * len(idx)
                sample_count += len(idx)

            train_epoch = self._evaluate_split(model, train_ds, class_names)
            val_epoch = self._evaluate_split(model, val_ds, class_names) if len(val_ds.y) else {}
            train_metrics = train_epoch.get("metrics", {})
            val_metrics = val_epoch.get("metrics", {})
            avg_loss = total_loss / max(1, sample_count) if sample_count else train_metrics.get("loss")
            elapsed = max(1e-6, time.perf_counter() - start)
            eta_seconds = max(0.0, (epochs - epoch) * (elapsed / epoch))
            checkpoint = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_accuracy": train_metrics.get("accuracy"),
                "train_macro_f1": train_metrics.get("macro_f1"),
                "val_loss": val_metrics.get("loss"),
                "val_accuracy": val_metrics.get("accuracy"),
                "val_macro_f1": val_metrics.get("macro_f1"),
                "eta_seconds": eta_seconds,
            }
            checkpoint_metrics.append(checkpoint)
            if progress_callback is not None:
                progress_callback({"epoch": epoch, "total_epochs": epochs, "checkpoint": checkpoint})

            monitor_metrics = val_metrics if val_metrics else train_metrics
            current_primary_metric = float(monitor_metrics.get("macro_f1", float("-inf")))
            if current_primary_metric > best_primary_metric:
                best_primary_metric = current_primary_metric
                best_epoch = epoch
                best_split_name = "val" if val_metrics else "train"
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                early_stopping_patience is not None
                and epoch >= max(1, min_epochs)
                and epochs_without_improvement >= int(early_stopping_patience)
            ):
                checkpoint_metrics[-1]["stopped_early"] = 1
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        self._ensure_not_canceled(cancel_event)
        final_eval = self._evaluate_split(model, eval_ds, class_names)
        if not final_eval:
            raise ValueError("crnn trainer requires validation or test samples for evaluation")

        synthetic_eval = (
            self._evaluate_split(model, synthetic_eval_ds, class_names)
            if len(synthetic_eval_ds.y)
            else None
        )

        run.exports_dir(self._root).mkdir(parents=True, exist_ok=True)
        model_path = run.exports_dir(self._root) / "model.pth"
        inference_preprocessing = {
            "datasetVersionId": str(data_spec.get("datasetVersionId") or dataset_version.id),
            "sampleRate": sample_rate,
            "maxLength": max_length,
            "nFft": n_fft,
            "hopLength": hop_length,
            "nMels": n_mels,
            "fmax": fmax,
        }
        torch.save(
            {
                "schema": "foundry.crnn_model.v1",
                "trainer": "crnn_melspec_v1",
                "classes": class_names,
                "classification_mode": run.spec["classificationMode"],
                "model_type": str((run.spec.get("model") or {}).get("type") or "crnn"),
                "model_state_dict": model.state_dict(),
                "inference_preprocessing": dict(inference_preprocessing),
                "preprocessing": dict(inference_preprocessing),
                "training": {
                    "epochs": epochs,
                    "completedEpochs": len(checkpoint_metrics),
                    "learningRate": learning_rate,
                    "batchSize": batch_size,
                    "seed": seed,
                    "syntheticMix": synthetic_mix,
                    "gradientClipNorm": gradient_clip_norm,
                    "weightDecay": weight_decay,
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
                "trainerProfile": "crnn_v1",
                "optimizer": "adamw",
                "earlyStoppingPatience": None if early_stopping_patience is None else int(early_stopping_patience),
                "minEpochs": max(1, min_epochs),
                "syntheticMix": synthetic_mix,
                "gradientClipNorm": gradient_clip_norm,
                "weightDecay": weight_decay,
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
            "trainerProfile": "crnn_v1",
            "dataProfile": "crnn_v1",
            "completedEpochs": len(checkpoint_metrics),
            "bestCheckpointEpoch": best_epoch,
            "bestCheckpointMetric": best_primary_metric if best_epoch else None,
            "bestCheckpointSplit": best_split_name if best_epoch else None,
            "syntheticMix": synthetic_mix,
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
                "family": "crnn_melspec_v1",
                "profile": "crnn_v1",
                "optimizer": "adamw",
                "epochs": epochs,
                "completed_epochs": len(checkpoint_metrics),
                "checkpoint_epoch": best_epoch or len(checkpoint_metrics),
                "synthetic_mix": synthetic_mix,
                "gradient_clip_norm": gradient_clip_norm,
                "weight_decay": weight_decay,
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
                    "trainer": "crnn_melspec_v1",
                    "trainerProfile": "crnn_v1",
                    "bestCheckpointEpoch": best_epoch,
                    "metricsPath": metrics_path.name,
                    "runSummaryPath": run_summary_path.name,
                    "syntheticMix": synthetic_mix,
                },
            },
            model_path=model_path,
            metrics_path=metrics_path,
            run_summary_path=run_summary_path,
            eval_split_name=eval_split_name,
            synthetic_eval=synthetic_eval,
        )

    def _build_dataset(
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
    ) -> _TensorDataset:
        if not samples:
            return _TensorDataset(
                x=np.empty((0, 1, n_mels, 1), dtype=np.float32),
                y=np.empty((0,), dtype=np.int64),
            )

        features: list[np.ndarray] = []
        labels: list[int] = []
        target_frames: int | None = None
        for sample in samples:
            self._ensure_not_canceled(cancel_event)
            try:
                audio = BaselineTrainer._load_audio(Path(sample.audio_ref), sample_rate=sample_rate, max_length=max_length)
            except InvalidAudioSourceError as exc:
                warnings.warn(
                    f"Skipping invalid dataset sample {sample.sample_id} ({sample.audio_ref}): {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmax=fmax,
                power=2.0,
            )
            mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max).astype(np.float32)
            mel_mean = float(np.mean(mel_db))
            mel_std = float(np.std(mel_db))
            if mel_std > 0:
                mel_db = (mel_db - mel_mean) / mel_std

            if target_frames is None:
                target_frames = int(mel_db.shape[1])
            if mel_db.shape[1] > target_frames:
                mel_db = mel_db[:, :target_frames]
            elif mel_db.shape[1] < target_frames:
                mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - mel_db.shape[1])))

            features.append(mel_db[np.newaxis, :, :])
            labels.append(label_to_index[sample.label])

        if not features:
            return _TensorDataset(
                x=np.empty((0, 1, n_mels, 1), dtype=np.float32),
                y=np.empty((0,), dtype=np.int64),
            )
        return _TensorDataset(
            x=np.stack(features).astype(np.float32),
            y=np.asarray(labels, dtype=np.int64),
        )

    @staticmethod
    def _ensure_not_canceled(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCanceledError("run canceled")

    @staticmethod
    def _evaluate_split(model: _Crnn, dataset: _TensorDataset, class_names: list[str]) -> dict[str, object]:
        if len(dataset.y) == 0:
            return {}

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(dataset.x))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        predictions = probabilities.argmax(axis=1)
        labels = np.arange(len(class_names), dtype=np.int64)
        precision, recall, f1, support = precision_recall_fscore_support(
            dataset.y,
            predictions,
            labels=labels,
            zero_division=0,
        )
        matrix = confusion_matrix(dataset.y, predictions, labels=labels)
        metrics = {
            "loss": float(log_loss(dataset.y, probabilities, labels=labels)),
            "accuracy": float(accuracy_score(dataset.y, predictions)),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "sample_count": int(len(dataset.y)),
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

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support

from echozero.foundry.domain import DatasetSample, DatasetVersion, TrainRun
from echozero.runtime_models.architectures import SimpleCnnRuntimeModel
from echozero.foundry.services.baseline_trainer import BaselineTrainer, BaselineTrainingResult, RunCanceledError
from echozero.foundry.services.training_runtime import (
    compute_config_fingerprint,
    configure_reproducibility,
    ensure_finite_array,
    ensure_finite_tensor,
)


_SimpleCnn = SimpleCnnRuntimeModel


@dataclass(slots=True)
class _TensorDataset:
    x: np.ndarray
    y: np.ndarray


class CnnTrainer:
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
        deterministic = bool(training_spec.get("deterministic", True))
        synthetic_mix_spec = training_spec.get("syntheticMix") or {}
        gradient_clip_norm = float(training_spec.get("gradientClipNorm", 1.0))
        weight_decay = float(training_spec.get("weightDecay", 0.0001))
        early_stopping_patience = training_spec.get("earlyStoppingPatience")
        min_epochs = int(training_spec.get("minEpochs", 1))

        reproducibility = configure_reproducibility(seed, deterministic=deterministic)
        config_fingerprint = compute_config_fingerprint(
            {
                "schema": "foundry.training_fingerprint.v1",
                "runSpec": run.spec,
                "datasetVersionId": dataset_version.id,
                "datasetManifestHash": dataset_version.manifest_hash,
                "classMap": list(dataset_version.class_map),
            }
        )

        rng = np.random.default_rng(seed)

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
            raise ValueError("cnn trainer requires at least two training samples")

        class_names = list(dataset_version.class_map)
        label_to_index = {label: index for index, label in enumerate(class_names)}
        train_label_ids = {sample.label for sample in train_samples}
        if train_label_ids != set(class_names):
            missing = sorted(set(class_names) - train_label_ids)
            raise ValueError(f"training split is missing classes required for cnn training: {', '.join(missing)}")

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

        ensure_finite_array("cnn/train_x", train_ds.x, context="dataset_build")
        ensure_finite_array("cnn/val_x", val_ds.x, context="dataset_build")
        ensure_finite_array("cnn/eval_x", eval_ds.x, context="dataset_build")
        ensure_finite_array("cnn/synth_eval_x", synthetic_eval_ds.x, context="dataset_build")

        model = _SimpleCnn(num_classes=len(class_names))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        checkpoint_metrics: list[dict[str, float | int | None]] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_primary_metric = float("-inf")
        best_split_name = "val" if len(val_ds.y) else "train"
        epochs_without_improvement = 0
        started = time.perf_counter()

        for epoch in range(1, epochs + 1):
            self._ensure_not_canceled(cancel_event)
            model.train()
            permutation = rng.permutation(len(train_ds.y))
            total_loss = 0.0
            sample_count = 0
            for batch_index, start in enumerate(range(0, len(permutation), max(1, batch_size)), start=1):
                idx = permutation[start : start + max(1, batch_size)]
                xb = torch.from_numpy(train_ds.x[idx])
                yb = torch.from_numpy(train_ds.y[idx])
                ensure_finite_tensor("cnn/train_batch", xb, context=f"epoch={epoch},batch={batch_index}")
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                ensure_finite_tensor("cnn/logits", logits, context=f"epoch={epoch},batch={batch_index}")
                loss = criterion(logits, yb)
                ensure_finite_tensor("cnn/loss", loss, context=f"epoch={epoch},batch={batch_index}")
                loss.backward()
                self._ensure_gradients_finite(model, epoch=epoch, batch_index=batch_index)
                if gradient_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    ensure_finite_tensor("cnn/grad_norm", grad_norm, context=f"epoch={epoch},batch={batch_index}")
                optimizer.step()
                self._ensure_parameters_finite(model, epoch=epoch, batch_index=batch_index)
                total_loss += float(loss.item()) * len(idx)
                sample_count += len(idx)

            train_epoch = self._evaluate_split(model, train_ds, class_names)
            val_epoch = self._evaluate_split(model, val_ds, class_names) if len(val_ds.y) else {}
            train_metrics = train_epoch.get("metrics", {})
            val_metrics = val_epoch.get("metrics", {})
            elapsed = max(1e-6, time.perf_counter() - started)
            checkpoint = {
                "epoch": epoch,
                "train_loss": total_loss / max(1, sample_count) if sample_count else train_metrics.get("loss"),
                "train_accuracy": train_metrics.get("accuracy"),
                "train_macro_f1": train_metrics.get("macro_f1"),
                "val_loss": val_metrics.get("loss"),
                "val_accuracy": val_metrics.get("accuracy"),
                "val_macro_f1": val_metrics.get("macro_f1"),
                "eta_seconds": max(0.0, (epochs - epoch) * (elapsed / epoch)),
            }
            checkpoint_metrics.append(checkpoint)

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

            if progress_callback is not None:
                progress_callback({"epoch": epoch, "total_epochs": epochs, "checkpoint": checkpoint})

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
            raise ValueError("cnn trainer requires validation or test samples for evaluation")

        synthetic_eval = (
            self._evaluate_split(model, synthetic_eval_ds, class_names)
            if len(synthetic_eval_ds.y)
            else None
        )

        run.exports_dir(self._root).mkdir(parents=True, exist_ok=True)
        model_path = run.exports_dir(self._root) / "model.pth"
        torch.save(
            {
                "schema": "foundry.cnn_model.v1",
                "trainer": "cnn_melspec_v1",
                "classes": class_names,
                "classification_mode": run.spec["classificationMode"],
                "model_state_dict": model.state_dict(),
                "preprocessing": {
                    "sampleRate": sample_rate,
                    "maxLength": max_length,
                    "nFft": n_fft,
                    "hopLength": hop_length,
                    "nMels": n_mels,
                    "fmax": fmax,
                },
                "training": {
                    "epochs": epochs,
                    "completedEpochs": len(checkpoint_metrics),
                    "learningRate": learning_rate,
                    "batchSize": batch_size,
                    "seed": seed,
                    "deterministic": deterministic,
                    "configFingerprint": config_fingerprint,
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
                "trainerProfile": "cnn_v1",
                "optimizer": "adamw",
                "earlyStoppingPatience": None if early_stopping_patience is None else int(early_stopping_patience),
                "minEpochs": max(1, min_epochs),
                "syntheticMix": synthetic_mix,
                "gradientClipNorm": gradient_clip_norm,
                "weightDecay": weight_decay,
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
            "trainerProfile": "cnn_v1",
            "dataProfile": "cnn_v1",
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
                "family": "cnn_melspec_v1",
                "profile": "cnn_v1",
                "optimizer": "adamw",
                "epochs": epochs,
                "completed_epochs": len(checkpoint_metrics),
                "checkpoint_epoch": best_epoch or len(checkpoint_metrics),
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
                    "trainer": "cnn_melspec_v1",
                    "trainerProfile": "cnn_v1",
                    "bestCheckpointEpoch": best_epoch,
                    "metricsPath": metrics_path.name,
                    "runSummaryPath": run_summary_path.name,
                    "syntheticMix": synthetic_mix,
                    "gradientClipNorm": gradient_clip_norm,
                    "weightDecay": weight_decay,
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
    def _ensure_gradients_finite(model: _SimpleCnn, *, epoch: int, batch_index: int) -> None:
        for name, parameter in model.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            ensure_finite_tensor(
                f"cnn/gradient:{name}",
                grad,
                context=f"epoch={epoch},batch={batch_index}",
            )

    @staticmethod
    def _ensure_parameters_finite(model: _SimpleCnn, *, epoch: int, batch_index: int) -> None:
        for name, parameter in model.named_parameters():
            ensure_finite_tensor(
                f"cnn/parameter:{name}",
                parameter,
                context=f"epoch={epoch},batch={batch_index}",
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
            audio = BaselineTrainer._load_audio(Path(sample.audio_ref), sample_rate=sample_rate, max_length=max_length)
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
            ensure_finite_array("cnn/mel_db", mel_db, context=f"sample_id={sample.sample_id}")
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

            feature = mel_db[np.newaxis, :, :]
            ensure_finite_array("cnn/feature", feature, context=f"sample_id={sample.sample_id}")
            features.append(feature)
            labels.append(label_to_index[sample.label])

        return _TensorDataset(
            x=np.stack(features).astype(np.float32),
            y=np.asarray(labels, dtype=np.int64),
        )

    @staticmethod
    def _ensure_not_canceled(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCanceledError("run canceled")

    @staticmethod
    def _evaluate_split(model: _SimpleCnn, dataset: _TensorDataset, class_names: list[str]) -> dict[str, object]:
        if len(dataset.y) == 0:
            return {}

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(dataset.x))
            ensure_finite_tensor("cnn/eval_logits", logits, context="eval")
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        ensure_finite_array("cnn/probabilities", probabilities, context="eval")

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

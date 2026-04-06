from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from echozero.foundry.domain import DatasetSample, DatasetVersion, TrainRun


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


class BaselineTrainer:
    def __init__(self, root: Path):
        self._root = root

    def train(self, run: TrainRun, dataset_version: DatasetVersion) -> BaselineTrainingResult:
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

        sample_by_id = {sample.sample_id: sample for sample in dataset_version.samples}
        split_plan = dataset_version.split_plan or {}
        train_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("train_ids", [])]
        val_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("val_ids", [])]
        test_samples = [sample_by_id[sample_id] for sample_id in split_plan.get("test_ids", [])]

        if len(train_samples) < 2:
            raise ValueError("baseline trainer requires at least two training samples")

        class_names = list(dataset_version.class_map)
        label_to_index = {label: index for index, label in enumerate(class_names)}
        train_label_ids = {sample.label for sample in train_samples}
        if train_label_ids != set(class_names):
            missing = sorted(set(class_names) - train_label_ids)
            raise ValueError(f"training split is missing classes required for baseline training: {', '.join(missing)}")

        train_x, train_y = self._build_features(
            train_samples,
            sample_rate=sample_rate,
            max_length=max_length,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax,
            label_to_index=label_to_index,
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
        )

        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        val_x_scaled = scaler.transform(val_x) if len(val_x) else np.empty((0, train_x_scaled.shape[1]), dtype=np.float32)
        eval_x_scaled = scaler.transform(eval_x) if len(eval_x) else np.empty((0, train_x_scaled.shape[1]), dtype=np.float32)

        classifier = SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=learning_rate,
            alpha=0.0001,
            random_state=random_seed,
            shuffle=False,
        )
        classes = np.arange(len(class_names), dtype=np.int64)

        checkpoint_metrics: list[dict[str, float | int | None]] = []
        for epoch in range(1, epochs + 1):
            classifier.partial_fit(train_x_scaled, train_y, classes=classes)
            train_epoch = self._evaluate_split(classifier, train_x_scaled, train_y, class_names)
            val_epoch = self._evaluate_split(classifier, val_x_scaled, val_y, class_names) if len(val_y) else {}
            checkpoint_metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": train_epoch.get("loss"),
                    "train_accuracy": train_epoch.get("accuracy"),
                    "val_loss": val_epoch.get("loss"),
                    "val_accuracy": val_epoch.get("accuracy"),
                    "val_macro_f1": val_epoch.get("macro_f1"),
                }
            )

        final_eval = self._evaluate_split(classifier, eval_x_scaled, eval_y, class_names)
        if not final_eval:
            raise ValueError("baseline trainer requires validation or test samples for evaluation")

        run.exports_dir(self._root).mkdir(parents=True, exist_ok=True)
        model_path = run.exports_dir(self._root) / "model.pth"
        torch.save(
            {
                "schema": "foundry.baseline_model.v1",
                "trainer": "baseline_sgd_melspec_v1",
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
                    "learningRate": learning_rate,
                    "batchSize": int(training_spec["batchSize"]),
                    "seed": random_seed,
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
        }
        metrics_path = run.exports_dir(self._root) / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

        run_summary_payload = {
            "runId": run.id,
            "status": "completed",
            "evalSplit": eval_split_name,
            "modelPath": model_path.name,
            "metricsPath": metrics_path.name,
            "primaryMetric": final_eval["metrics"]["macro_f1"],
            "accuracy": final_eval["metrics"]["accuracy"],
        }
        run_summary_path = run.exports_dir(self._root) / "run_summary.json"
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
                "family": "baseline_sgd_melspec_v1",
                "optimizer": "sgd_log_loss",
                "epochs": epochs,
                "checkpoint_epoch": epochs,
                "feature_pooling": ["mean", "std", "max"],
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
                    "trainer": "baseline_sgd_melspec_v1",
                    "metricsPath": metrics_path.name,
                    "runSummaryPath": run_summary_path.name,
                },
            },
            model_path=model_path,
            metrics_path=metrics_path,
            run_summary_path=run_summary_path,
            eval_split_name=eval_split_name,
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
    ) -> tuple[np.ndarray, np.ndarray]:
        if not samples:
            feature_size = n_mels * 3
            return np.empty((0, feature_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

        features: list[np.ndarray] = []
        labels: list[int] = []
        for sample in samples:
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
            pooled = np.concatenate(
                [
                    mel_db.mean(axis=1),
                    mel_db.std(axis=1),
                    mel_db.max(axis=1),
                ]
            ).astype(np.float32)
            features.append(pooled)
            labels.append(label_to_index[sample.label])

        return np.vstack(features), np.asarray(labels, dtype=np.int64)

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
        return audio.astype(np.float32)

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

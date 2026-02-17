"""
Model Evaluation and Diagnostics

Comprehensive evaluation metrics for both multi-class and binary classification:
- Confusion matrix
- Per-class precision, recall, F1
- Classification report
- ROC curve + AUC (binary)
- Precision-Recall curve + Average Precision (binary)
- Threshold tuning (binary)
- Hard negative mining (binary)
- Test-Time Augmentation (TTA)
- Temperature scaling for calibration
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.message import Log

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        precision_recall_fscore_support,
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Core Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: "nn.Module",
    data_loader: "torch.utils.data.DataLoader",
    config: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run full evaluation on a dataset.

    Collects predictions and ground truth, then computes all relevant metrics
    based on classification mode (binary or multiclass).

    Args:
        model: Trained model in eval mode
        data_loader: DataLoader for evaluation data
        config: Training configuration
        device: Device to run evaluation on

    Returns:
        Dictionary containing all evaluation metrics
    """
    if not HAS_PYTORCH:
        return {"error": "PyTorch not available"}

    model.eval()
    is_binary = config.get("classification_mode", "multiclass") == "binary"

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_data in data_loader:
            if len(batch_data) == 3:
                inputs, labels, _ = batch_data  # onset strengths
            else:
                inputs, labels = batch_data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    if is_binary:
        return _evaluate_binary(logits, labels, config)
    else:
        return _evaluate_multiclass(logits, labels, config)


def _evaluate_multiclass(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute multiclass metrics."""
    probs = F.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)

    y_true = labels.numpy()
    y_pred = predictions.numpy()

    # Basic accuracy
    accuracy = (y_pred == y_true).mean() * 100

    results: Dict[str, Any] = {
        "classification_mode": "multiclass",
        "accuracy": float(accuracy),
        "total_samples": len(y_true),
    }

    if HAS_SKLEARN:
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0,
        )
        results["per_class"] = {}
        classes = config.get("_classes", [])
        for i in range(len(precision)):
            cls_name = classes[i] if i < len(classes) else f"class_{i}"
            results["per_class"][cls_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        # Classification report (text)
        target_names = classes if classes else None
        results["classification_report"] = classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0,
        )

        # Macro averages
        results["macro_precision"] = float(precision.mean())
        results["macro_recall"] = float(recall.mean())
        results["macro_f1"] = float(f1.mean())

    return results


def _evaluate_binary(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute binary classification metrics."""
    # Sigmoid probabilities
    logits_squeezed = logits.squeeze(-1)
    probs = torch.sigmoid(logits_squeezed).numpy()
    y_true = labels.numpy().astype(int)

    threshold = config.get("confidence_threshold", 0.5)
    y_pred = (probs >= threshold).astype(int)

    # Basic metrics at current threshold
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / len(y_true) * 100
    specificity = tn / max(tn + fp, 1)

    results: Dict[str, Any] = {
        "classification_mode": "binary",
        "target_class": config.get("target_class", "positive"),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": len(y_true),
    }

    if HAS_SKLEARN:
        # ROC curve + AUC
        fpr, tpr, roc_thresholds = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        results["roc_auc"] = float(roc_auc)
        results["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        }

        # Precision-Recall curve + AP
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        results["pr_auc"] = float(ap)
        results["pr_curve"] = {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()

    return results


# ---------------------------------------------------------------------------
# Threshold Tuning (Binary)
# ---------------------------------------------------------------------------

def tune_threshold(
    model: "nn.Module",
    data_loader: "torch.utils.data.DataLoader",
    metric: str = "f1",
    device: str = "cpu",
    num_thresholds: int = 200,
) -> Tuple[float, Dict[str, Any]]:
    """
    Find the optimal decision threshold for binary classification.

    Sweeps thresholds from 0 to 1 and selects the one that maximizes
    the specified metric.

    Args:
        model: Trained binary model
        data_loader: Validation DataLoader
        metric: Metric to optimize ("f1", "precision", "recall", "youden")
        device: Compute device
        num_thresholds: Number of thresholds to evaluate

    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal)
    """
    if not HAS_PYTORCH:
        return 0.5, {}

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_data in data_loader:
            inputs = batch_data[0].to(device)
            labels = batch_data[1]

            outputs = model(inputs).squeeze(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels, dtype=int)

    best_threshold = 0.5
    best_score = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.01, 0.99, num_thresholds)

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        spec = tn / max(tn + fp, 1)

        if metric == "f1":
            score = f1
        elif metric == "precision":
            score = prec
        elif metric == "recall":
            score = rec
        elif metric == "youden":
            score = rec + spec - 1  # Youden's J statistic
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_threshold = float(t)
            best_metrics = {
                "threshold": float(t),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "specificity": float(spec),
                "youden_j": float(rec + spec - 1),
            }

    Log.info(
        f"Threshold tuning ({metric}): optimal={best_threshold:.3f}, "
        f"score={best_score:.4f}"
    )
    return best_threshold, best_metrics


# ---------------------------------------------------------------------------
# Hard Negative Mining (Binary)
# ---------------------------------------------------------------------------

def find_hard_negatives(
    model: "nn.Module",
    data_loader: "torch.utils.data.DataLoader",
    threshold: float = 0.5,
    top_k: int = 100,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Find hard negatives: negative samples that the model incorrectly
    classifies as positive (false positives).

    These samples should be added to the training set for the next
    training iteration to improve model accuracy.

    Args:
        model: Trained binary model
        data_loader: DataLoader containing negative samples
        threshold: Decision threshold
        top_k: Maximum number of hard negatives to return
        device: Compute device

    Returns:
        List of dicts with 'index', 'probability', and 'file_path' (if available)
    """
    if not HAS_PYTORCH:
        return []

    model.eval()
    hard_negatives = []

    sample_idx = 0
    with torch.no_grad():
        for batch_data in data_loader:
            inputs = batch_data[0].to(device)
            labels = batch_data[1]

            outputs = model(inputs).squeeze(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            batch_labels = labels.numpy()

            for i in range(len(probs)):
                if batch_labels[i] == 0 and probs[i] >= threshold:
                    hard_negatives.append({
                        "index": sample_idx + i,
                        "probability": float(probs[i]),
                    })

            sample_idx += len(probs)

    # Sort by probability (highest = hardest)
    hard_negatives.sort(key=lambda x: x["probability"], reverse=True)

    if top_k > 0:
        hard_negatives = hard_negatives[:top_k]

    Log.info(f"Found {len(hard_negatives)} hard negatives above threshold {threshold}")
    return hard_negatives


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

def predict_with_tta(
    model: "nn.Module",
    inputs: "torch.Tensor",
    num_augmentations: int = 5,
    device: str = "cpu",
) -> "torch.Tensor":
    """
    Test-Time Augmentation: create multiple augmented versions of each
    input, predict all of them, and average the predictions.

    Can improve accuracy by 1-3% at the cost of N times more inference.

    Args:
        model: Trained model in eval mode
        inputs: Input tensor (batch, channels, height, width)
        num_augmentations: Number of augmented versions to create
        device: Compute device

    Returns:
        Averaged prediction logits
    """
    if not HAS_PYTORCH:
        return inputs

    model.eval()
    all_outputs = []

    with torch.no_grad():
        # Original prediction
        all_outputs.append(model(inputs.to(device)))

        for _ in range(num_augmentations - 1):
            # Apply random augmentations
            augmented = inputs.clone()

            # Random horizontal flip (time-reversal for spectrograms)
            if torch.rand(1).item() > 0.5:
                augmented = torch.flip(augmented, dims=[3])

            # Small random noise
            noise = torch.randn_like(augmented) * 0.01
            augmented = augmented + noise

            # Slight random crop and resize (frequency axis shift)
            if augmented.size(2) > 4:
                shift = torch.randint(-2, 3, (1,)).item()
                if shift != 0:
                    augmented = torch.roll(augmented, shifts=shift, dims=2)

            all_outputs.append(model(augmented.to(device)))

    # Average all predictions
    stacked = torch.stack(all_outputs, dim=0)
    return stacked.mean(dim=0)


# ---------------------------------------------------------------------------
# Temperature Scaling (Calibration)
# ---------------------------------------------------------------------------

class TemperatureScaling(nn.Module):
    """
    Post-training temperature scaling for probability calibration.

    Learns a single temperature parameter T on the validation set such
    that softmax(logits / T) produces well-calibrated probabilities.

    A well-calibrated model that says "90% kick" should be correct 90%
    of the time.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        model: nn.Module,
        val_loader: "torch.utils.data.DataLoader",
        device: str = "cpu",
        max_iter: int = 50,
    ) -> float:
        """
        Learn optimal temperature on validation set.

        Args:
            model: Trained model (frozen)
            val_loader: Validation DataLoader
            device: Compute device
            max_iter: Maximum optimization iterations

        Returns:
            Learned temperature value
        """
        model.eval()
        self.to(device)

        # Collect all logits and labels
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data[0].to(device)
                labels = batch_data[1].to(device)
                logits = model(inputs)
                all_logits.append(logits)
                all_labels.append(labels)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_fn():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)

        optimal_temp = self.temperature.item()
        Log.info(f"Temperature scaling: T={optimal_temp:.4f}")
        return optimal_temp

"""
Loss Functions for Audio Classification Training

Provides all loss functions used during training, including:
- Standard CrossEntropy with label smoothing
- Class-weighted CrossEntropy for imbalanced datasets
- Focal Loss for hard-example mining
- Binary Cross Entropy variants for binary classification
- Onset-aware loss (DOSE-inspired) for drum classification
- Combined loss wrapper

The create_loss_function() factory auto-selects the appropriate loss
based on TrainingConfig.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.message import Log

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


# ---------------------------------------------------------------------------
# Label Smoothing Cross Entropy
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Instead of hard labels [1, 0, 0], uses soft labels [0.9, 0.05, 0.05].
    Prevents overconfident predictions and significantly improves
    generalization and calibration.

    Standard practice at Google, Meta, and most production ML systems.
    """

    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        """
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = typical)
            weight: Optional per-class weights for imbalanced datasets
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size, num_classes) logits
            targets: (batch_size,) class indices
        """
        num_classes = predictions.size(1)
        log_probs = F.log_softmax(predictions, dim=1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(predictions.device)
            sample_weights = weight[targets]
            loss = -(smooth_targets * log_probs).sum(dim=1)
            loss = (loss * sample_weights).mean()
        else:
            loss = -(smooth_targets * log_probs).sum(dim=1).mean()

        return loss


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for hard-example mining.

    Down-weights easy examples and focuses training on hard ones.
    Originally from "Focal Loss for Dense Object Detection" (Lin et al., 2017).
    Effective for both imbalanced datasets and general training.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        """
        Args:
            alpha: Per-class weights (tensor of shape [num_classes])
            gamma: Focusing parameter (0 = standard CE, 2 = typical focal)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(predictions.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Binary Loss Functions
# ---------------------------------------------------------------------------

class BinaryCrossEntropyWithLogits(nn.Module):
    """
    Binary cross entropy with logits for one-vs-all classification.

    Supports pos_weight for class imbalance handling. If you have 3x more
    negatives than positives, set pos_weight=3.0 to compensate.
    """

    def __init__(self, pos_weight: Optional[float] = None, label_smoothing: float = 0.0):
        """
        Args:
            pos_weight: Weight for positive class (compensates imbalance)
            label_smoothing: Smooth labels toward 0.5 (0.0 = disabled)
        """
        super().__init__()
        self.label_smoothing = label_smoothing

        pw = None
        if pos_weight is not None:
            pw = torch.tensor([pos_weight])
        self.register_buffer("pos_weight", pw)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size, 1) or (batch_size,) logits
            targets: (batch_size,) binary labels (0 or 1), or float for mixup
        """
        predictions = predictions.squeeze(-1)
        targets = targets.float()

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        pw = self.pos_weight
        if pw is not None:
            pw = pw.to(predictions.device)

        return F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=pw)


class BinaryFocalLoss(nn.Module):
    """
    Focal loss variant for binary classification.

    Focuses on hard-to-classify boundary cases, which is especially
    useful for one-vs-all detectors where the model needs to clearly
    distinguish the target class from similar sounds.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weight for positive class
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.squeeze(-1)
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Onset-Aware Loss (DOSE-inspired)
# ---------------------------------------------------------------------------

class OnsetAwareLoss(nn.Module):
    """
    Onset-aware loss function inspired by DOSE paper.

    Combines standard loss with an onset-weighted component that emphasizes
    accurate prediction of samples with strong initial transients, which are
    crucial for drum sound classification.

    Reference: DOSE - Drum One-Shot Extraction (https://arxiv.org/pdf/2504.18157)
    """

    def __init__(self, onset_weight: float = 0.3, base_criterion: Optional[nn.Module] = None):
        """
        Args:
            onset_weight: Weight for onset component (0.0-1.0)
            base_criterion: Base loss function (defaults to CrossEntropyLoss)
        """
        super().__init__()
        self.onset_weight = onset_weight
        self.base_criterion = base_criterion or nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        onset_strengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_loss = self.base_criterion(predictions, targets)

        if onset_strengths is not None and self.onset_weight > 0:
            onset_norm = (onset_strengths - onset_strengths.min()) / (
                onset_strengths.max() - onset_strengths.min() + 1e-8
            )
            sample_weights = 1.0 + self.onset_weight * onset_norm

            if base_loss.dim() > 0:
                loss = (base_loss * sample_weights).mean()
            else:
                loss = base_loss
        else:
            loss = base_loss.mean() if base_loss.dim() > 0 else base_loss

        return loss


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Wrapper that combines multiple loss functions with configurable weights.

    Example:
        combined = CombinedLoss([
            (CrossEntropyLoss(), 1.0),
            (FocalLoss(gamma=2.0), 0.5),
        ])
    """

    def __init__(self, losses_and_weights: List[tuple]):
        """
        Args:
            losses_and_weights: List of (loss_fn, weight) tuples
        """
        super().__init__()
        self.losses = nn.ModuleList([l for l, _ in losses_and_weights])
        self.weights = [w for _, w in losses_and_weights]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        total = torch.tensor(0.0, device=predictions.device)
        for loss_fn, weight in zip(self.losses, self.weights):
            try:
                loss = loss_fn(predictions, targets, **kwargs)
            except TypeError:
                # Some losses don't accept kwargs
                loss = loss_fn(predictions, targets)
            total = total + weight * loss
        return total


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def compute_class_weights(class_counts: Dict[str, int]) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    Args:
        class_counts: Dictionary mapping class name to sample count

    Returns:
        Tensor of weights (one per class, ordered by class index)
    """
    counts = np.array(list(class_counts.values()), dtype=np.float64)
    total = counts.sum()
    weights = total / (len(counts) * counts)
    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_binary_pos_weight(num_positive: int, num_negative: int) -> float:
    """
    Compute pos_weight for binary BCE loss to handle class imbalance.

    Args:
        num_positive: Number of positive samples
        num_negative: Number of negative samples

    Returns:
        pos_weight value (ratio of negative to positive)
    """
    if num_positive == 0:
        return 1.0
    return num_negative / num_positive


def create_loss_function(
    config: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None,
    pos_weight: Optional[float] = None,
) -> nn.Module:
    """
    Create the appropriate loss function based on training configuration.

    Automatically selects binary or multiclass loss, applies label smoothing,
    class weights, focal loss, and onset-aware components as configured.

    Args:
        config: Training configuration dictionary (or TrainingConfig.to_dict())
        class_weights: Pre-computed class weights tensor (for multiclass)
        pos_weight: Pre-computed positive class weight (for binary)

    Returns:
        Loss function module
    """
    is_binary = config.get("classification_mode", "multiclass") == "binary"
    label_smoothing = config.get("label_smoothing", 0.0)
    use_onset = config.get("use_onset_weighting", False)

    if is_binary:
        # Binary classification loss
        if config.get("use_focal_loss", False):
            criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        else:
            criterion = BinaryCrossEntropyWithLogits(
                pos_weight=pos_weight,
                label_smoothing=label_smoothing,
            )
    else:
        # Multiclass classification loss
        use_class_weights = config.get("use_class_weights", True) and class_weights is not None

        if label_smoothing > 0:
            criterion = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                weight=class_weights if use_class_weights else None,
            )
        elif config.get("use_focal_loss", False):
            criterion = FocalLoss(
                alpha=class_weights if use_class_weights else None,
                gamma=2.0,
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights if use_class_weights else None,
            )

    # Wrap with onset-aware loss if enabled
    if use_onset:
        onset_weight = config.get("onset_loss_weight", 0.3)
        criterion = OnsetAwareLoss(
            onset_weight=onset_weight,
            base_criterion=criterion if hasattr(criterion, "forward") else None,
        )

    return criterion

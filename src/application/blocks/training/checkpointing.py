"""
Checkpoint Management

Saves and restores full training state for seamless resume capability.
Includes model weights, optimizer state, scheduler state, EMA model,
training progress, configuration, dataset statistics, and random states.

Standard checkpoint layout (production .pth from save_final_model):
  - model_state_dict: Weights for inference/resume.
  - config: Full training parameters (canonical source for "how was this trained").
    Use this for tracking, reproducibility, and loading preprocessing/architecture.
  - classes: Class names (order matches model output).
  - training_date: ISO datetime when the model was saved.
  - training_history: Per-epoch loss/accuracy (for curves and best_epoch).
  - dataset_stats: Sample counts, class distribution, balance strategy, etc.
  - test_metrics / normalization / optimal_threshold: Evaluation and inference hints.
  - architecture_name, classification_mode, target_class: Convenience copies from config.

To track training params: read config (and optionally training_date, test_metrics,
dataset_stats) from the saved file. Use load_training_metadata() to load only
these fields without loading weights.
"""
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import random
import re

import numpy as np

from src.utils.message import Log
from src.utils.paths import get_models_dir

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False


def _unique_model_path(candidate: Path) -> Path:
    """
    Return a path that does not exist.     Uses versioned naming for the same prefix:
    stem_v1.pth, stem_v2.pth, ... so saves with the same base name get explicit versions.
    """
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    parent = candidate.parent
    # Find existing stem_vN.suffix to choose next version
    pattern = re.compile(re.escape(stem) + r"_v(\d+)$")
    existing = []
    for p in parent.iterdir():
        if p.suffix != suffix or not p.is_file():
            continue
        match = pattern.match(p.stem)
        if match:
            existing.append(int(match.group(1)))
    next_v = max(existing, default=0) + 1
    return parent / f"{stem}_v{next_v}{suffix}"


def _unique_folder_path(candidate: Path) -> Path:
    """Return a directory path that does not exist; use stem_v1, stem_v2 if needed."""
    if not candidate.exists():
        return candidate
    stem = candidate.name
    parent = candidate.parent
    pattern = re.compile(re.escape(stem) + r"_v(\d+)$")
    existing = []
    for p in parent.iterdir():
        if not p.is_dir():
            continue
        match = pattern.match(p.name)
        if match:
            existing.append(int(match.group(1)))
        elif p.name == stem:
            existing.append(0)
    next_v = max(existing, default=0) + 1
    return parent / f"{stem}_v{next_v}"


# Keys used for inference preprocessing; must match between trainer and classifier.
INFERENCE_PREPROCESSING_KEYS = (
    "sample_rate",
    "max_length",
    "n_fft",
    "hop_length",
    "n_mels",
    "fmax",
)


def build_inference_preprocessing(
    config: Dict[str, Any],
    normalization: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the canonical preprocessing dict for inference (single source of truth).

    PyTorch Audio Classify uses this to align mel spectrogram and normalization
    with the trainer. All values come from training config; do not override at inference.
    """
    out = {
        "sample_rate": config.get("sample_rate", 22050),
        "max_length": config.get("max_length", 22050),
        "n_fft": config.get("n_fft", 2048),
        "hop_length": config.get("hop_length", 512),
        "n_mels": config.get("n_mels", 128),
        "fmax": config.get("fmax", 8000),
    }
    if normalization and normalization.get("mean") is not None and normalization.get("std") is not None:
        out["normalization_mean"] = normalization["mean"]
        out["normalization_std"] = normalization["std"]
    return out


def write_model_summary_txt(
    folder: Path,
    model_path: Path,
    classes: List[str],
    config: Dict[str, Any],
    inference_preprocessing: Dict[str, Any],
    dataset_stats: Optional[Dict[str, Any]] = None,
    training_date: Optional[str] = None,
    optimal_threshold: Optional[float] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write MODEL_SUMMARY.txt into the model folder for LLM/agent troubleshooting.

    Plain text, sectioned, so an LLM or support agent can quickly understand
    how the model was trained and how to use it.
    """
    mode = config.get("classification_mode", "multiclass")
    lines = [
        "# PyTorch Audio Trainer – Model Summary",
        "",
        "This file describes the saved model for troubleshooting and integration.",
        "Use it with the PyTorch Audio Classify block; preprocessing is fixed to match training.",
        "",
        "## Model path",
        f"  {model_path}",
        "",
        "## Classes (output order)",
    ]
    for i, c in enumerate(classes):
        lines.append(f"  {i}: {c}")
    lines.extend([
        "",
        "## Classification mode",
        f"  {mode}",
        "",
        "## Inference preprocessing (must match at inference)",
        "  These values are baked into the checkpoint. The classifier uses them automatically.",
        "",
    ])
    for k, v in inference_preprocessing.items():
        lines.append(f"  {k}: {v}")
    lines.extend([
        "",
        "## Dataset (training-time)",
        "",
    ])
    if dataset_stats:
        dist = dataset_stats.get("class_distribution") or {}
        total = dataset_stats.get("total_samples") or 0
        for name, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = (100.0 * count / total) if total else 0
            lines.append(f"  {name}: {count} ({pct:.1f}%)")
        lines.append(f"  total_samples: {total}")
        if dataset_stats.get("balance_strategy"):
            lines.append(f"  balance_strategy: {dataset_stats['balance_strategy']}")
    else:
        lines.append("  (no dataset_stats)")
    lines.extend([
        "",
        "## Key training config",
        f"  model_type: {config.get('model_type', 'cnn')}",
        f"  positive_classes: {config.get('positive_classes')}",
        f"  use_class_weights: {config.get('use_class_weights')}",
        f"  balance_strategy: {config.get('balance_strategy', 'none')}",
        "",
    ])
    if mode == "binary" and optimal_threshold is not None:
        lines.extend([
            "## Binary threshold",
            f"  optimal_threshold: {optimal_threshold}",
            "",
        ])
    if training_date:
        lines.append(f"## Training date\n  {training_date}\n")
    if test_metrics:
        lines.append("## Test metrics")
        for k, v in test_metrics.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    lines.extend([
        "## How to use",
        "  1. In EchoZero, add a PyTorch Audio Classify block.",
        "  2. Set model_path to this model's .pth path (or connect the trainer's model output).",
        "  3. Connect events (e.g. from DetectOnsets or Editor) and optional audio.",
        "  4. Preprocessing (sample rate, mel, normalization) is read from the model; do not override.",
        "",
        "## Troubleshooting",
        "  - If all events are classified as one class (e.g. 'other'): check class balance in dataset,",
        "    enable balance_strategy or class weights, and ensure inference audio matches training",
        "    (same sample rate and source). Preprocessing is now fixed from the saved model.",
        "  - If accuracy is poor: verify training data quality and that validation accuracy was reasonable.",
        "  - Load this summary for context: scripts/inspect_pth_model.py <path_to_model.pth>",
        "",
    ])
    summary_path = folder / "MODEL_SUMMARY.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    Log.info(f"Model summary written to {summary_path}")


def save_checkpoint(
    model: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: Any,
    epoch: int,
    best_val_metric: float,
    config: Dict[str, Any],
    classes: List[str],
    training_stats: Dict[str, Any],
    dataset_stats: Optional[Dict[str, Any]] = None,
    normalization: Optional[Dict[str, Any]] = None,
    ema_model: Optional["nn.Module"] = None,
    checkpoint_dir: Optional[str] = None,
    is_best: bool = False,
) -> str:
    """
    Save a full training checkpoint.

    Args:
        model: Current model state
        optimizer: Current optimizer state
        scheduler: Current LR scheduler state (or None)
        epoch: Current epoch number
        best_val_metric: Best validation metric so far
        config: Training configuration
        classes: List of class names
        training_stats: Training history (losses, accuracies per epoch)
        dataset_stats: Dataset statistics dict
        normalization: Normalization parameters (mean, std)
        ema_model: EMA model state (optional)
        checkpoint_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far

    Returns:
        Path to saved checkpoint
    """
    if not HAS_PYTORCH:
        raise RuntimeError("PyTorch required for checkpoint saving")

    # Determine save directory
    if checkpoint_dir:
        save_dir = Path(checkpoint_dir)
    else:
        save_dir = get_models_dir() / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        # Model
        "model_state_dict": model.state_dict(),
        "model_architecture": config.get("model_type", "cnn"),

        # Optimizer + Scheduler
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler and hasattr(scheduler, "state_dict") else None,

        # Training progress
        "epoch": epoch,
        "best_val_metric": best_val_metric,

        # Configuration
        "config": config,
        "classes": classes,
        "classification_mode": config.get("classification_mode", "multiclass"),
        "target_class": config.get("target_class"),

        # Statistics
        "training_stats": training_stats,
        "dataset_stats": dataset_stats,
        "normalization": normalization,
        "training_date": datetime.now().isoformat(),

        # EMA
        "ema_state_dict": ema_model.state_dict() if ema_model else None,

        # Random states for reproducibility
        "random_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    # Save periodic checkpoint
    checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)

    # Save best model separately
    if is_best:
        best_path = save_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        Log.info(f"Saved best checkpoint at epoch {epoch} to {best_path}")

    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: "nn.Module",
    optimizer: Optional["torch.optim.Optimizer"] = None,
    scheduler: Any = None,
    ema_model: Optional["nn.Module"] = None,
    device: str = "cpu",
    restore_random_states: bool = True,
) -> Dict[str, Any]:
    """
    Load a training checkpoint and restore all state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to restore state (optional)
        scheduler: LR scheduler to restore state (optional)
        ema_model: EMA model to restore (optional)
        device: Device to load to
        restore_random_states: Whether to restore RNG states for reproducibility

    Returns:
        Dictionary with checkpoint metadata (epoch, best_val_metric, config, etc.)
    """
    if not HAS_PYTORCH:
        raise RuntimeError("PyTorch required for checkpoint loading")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore model
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer
    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            Log.warning(f"Could not restore optimizer state: {e}")

    # Restore scheduler
    if scheduler and checkpoint.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            Log.warning(f"Could not restore scheduler state: {e}")

    # Restore EMA model
    if ema_model and checkpoint.get("ema_state_dict"):
        try:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
        except Exception as e:
            Log.warning(f"Could not restore EMA state: {e}")

    # Restore random states
    if restore_random_states and checkpoint.get("random_states"):
        states = checkpoint["random_states"]
        try:
            if states.get("python"):
                random.setstate(states["python"])
            if states.get("numpy"):
                np.random.set_state(states["numpy"])
            if states.get("torch") is not None:
                torch.random.set_rng_state(states["torch"])
            if states.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(states["cuda"])
        except Exception as e:
            Log.warning(f"Could not restore random states: {e}")

    Log.info(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
        f"(best_val_metric={checkpoint.get('best_val_metric', '?')})"
    )

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_val_metric": checkpoint.get("best_val_metric", 0.0),
        "config": checkpoint.get("config", {}),
        "classes": checkpoint.get("classes", []),
        "training_stats": checkpoint.get("training_stats", {}),
        "dataset_stats": checkpoint.get("dataset_stats"),
        "normalization": checkpoint.get("normalization"),
    }


def save_final_model(
    model: "nn.Module",
    classes: List[str],
    config: Dict[str, Any],
    training_stats: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]] = None,
    dataset_stats: Optional[Dict[str, Any]] = None,
    normalization: Optional[Dict[str, Any]] = None,
    optimal_threshold: Optional[float] = None,
    ema_model: Optional["nn.Module"] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Save the final trained model with all metadata.

    This is the production model format that the classify block loads.
    Backward-compatible with existing checkpoint format.

    Args:
        model: Final trained model
        classes: List of class names
        config: Training configuration
        training_stats: Full training history
        test_metrics: Test set evaluation results
        dataset_stats: Dataset statistics
        normalization: Normalization parameters
        optimal_threshold: Auto-tuned threshold (binary mode)
        ema_model: EMA model (saved as separate key if provided)
        output_path: Custom output path

    Returns:
        Path to saved model file
    """
    if not HAS_PYTORCH:
        raise RuntimeError("PyTorch required for model saving")

    # Determine output path; never overwrite existing files
    if output_path:
        candidate = Path(output_path)
    else:
        models_dir = get_models_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_name = (config.get("model_name") or "").strip()
        if custom_name:
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in custom_name)[:80]
            if not safe:
                safe = "model"
            filename = f"{safe}_{timestamp}.pth"
        else:
            mode = config.get("classification_mode", "multiclass")
            arch = config.get("model_type", "cnn")
            if mode == "binary":
                pos = config.get("positive_classes") or []
                tag = "_".join(pos)[:60] if pos else (config.get("target_class") or "positive")
                filename = f"binary_{tag}_{arch}_{timestamp}.pth"
            else:
                filename = f"multiclass_{arch}_{timestamp}.pth"
        candidate = models_dir / filename

    model_path = _unique_model_path(candidate)
    if model_path != candidate:
        Log.info(f"Path {candidate} already exists; saving as {model_path} (no overwrite)")

    # Model folder: put .pth and MODEL_SUMMARY.txt in a dedicated folder
    folder_candidate = model_path.parent / model_path.stem
    model_folder = _unique_folder_path(folder_candidate)
    if model_folder != folder_candidate:
        Log.info(f"Model folder {folder_candidate} already exists; using {model_folder}")
    model_folder.mkdir(parents=True, exist_ok=True)
    pth_in_folder = model_folder / model_path.name

    # Canonical preprocessing for inference (single source of truth for classifier)
    inference_preprocessing = build_inference_preprocessing(config, normalization)
    training_date_iso = datetime.now().isoformat()

    # Build checkpoint -- backward compatible with existing format
    checkpoint = {
        # Existing format (unchanged for backward compat)
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "config": config,
        "training_date": training_date_iso,

        # New fields
        "architecture_name": config.get("model_type", "cnn"),
        "classification_mode": config.get("classification_mode", "multiclass"),
        "target_class": config.get("target_class"),
        "optimal_threshold": optimal_threshold,
        "normalization": normalization,
        "test_metrics": test_metrics,
        "training_history": training_stats,
        "dataset_stats": dataset_stats,
        "inference_preprocessing": inference_preprocessing,
    }

    # EMA model
    if ema_model:
        checkpoint["ema_state_dict"] = ema_model.state_dict()

    torch.save(checkpoint, pth_in_folder)
    Log.info(f"Model saved to {pth_in_folder}")

    write_model_summary_txt(
        folder=model_folder,
        model_path=pth_in_folder,
        classes=classes,
        config=config,
        inference_preprocessing=inference_preprocessing,
        dataset_stats=dataset_stats,
        training_date=training_date_iso,
        optimal_threshold=optimal_threshold,
        test_metrics=test_metrics,
    )

    return str(pth_in_folder)


def load_training_metadata(model_path: str) -> Dict[str, Any]:
    """
    Load only training metadata from a saved model (no weights).

    Use this to track or display how a model was trained without loading
    the full state dict. Returns the same keys as the checkpoint except
    model_state_dict and ema_state_dict are omitted.

    Args:
        model_path: Path to a .pth file from save_final_model or save_checkpoint.

    Returns:
        Dict with config, classes, training_date, training_history,
        dataset_stats, test_metrics, normalization, optimal_threshold, etc.
        Empty dict if the file cannot be read or is not a valid checkpoint.
    """
    if not HAS_PYTORCH:
        return {}
    path = Path(model_path)
    if not path.exists():
        return {}
    if path.is_dir():
        pth_files = list(path.glob("*.pth"))
        if len(pth_files) != 1:
            return {}
        path = pth_files[0]
    if not path.is_file():
        return {}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    # Omit large tensors so callers get only tracking/reproducibility data
    omit = {"model_state_dict", "ema_state_dict", "optimizer_state_dict", "scheduler_state_dict", "random_states"}
    return {k: v for k, v in checkpoint.items() if k not in omit}

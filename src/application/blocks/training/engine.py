"""
Training Engine

Core training loop with all professional techniques:
- Mixed precision training (AMP)
- Gradient accumulation
- Exponential Moving Average (EMA)
- Stochastic Weight Averaging (SWA)
- Warmup + cosine annealing with restarts
- Mixup / CutMix batch augmentation
- TensorBoard logging
- Checkpoint saving
- Seed reproducibility
- Early stopping

The TrainingEngine is the heart of the training infrastructure, orchestrating
all components (model, optimizer, scheduler, loss, augmentation) into a
cohesive training loop.
"""
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random
import copy
import time

import numpy as np

from src.utils.message import Log

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler, autocast
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except (ImportError, Exception):
    # TensorBoard may fail to import due to version incompatibilities
    HAS_TENSORBOARD = False

try:
    from torch.optim.swa_utils import AveragedModel, SWALR
    HAS_SWA = True
except ImportError:
    HAS_SWA = False


# ---------------------------------------------------------------------------
# Training Result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Container for training results."""
    model: Any  # nn.Module
    ema_model: Any = None  # Optional EMA nn.Module
    classes: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    best_val_metric: float = 0.0
    best_epoch: int = 0
    normalization: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# EMA Model
# ---------------------------------------------------------------------------

class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of weights that is a smoothed version of the
    training weights. Often produces better generalization than the
    final training weights.
    """

    def __init__(self, model: "nn.Module", decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

        # Disable gradient computation for shadow model
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: "nn.Module") -> None:
        """Update shadow weights with exponential moving average."""
        for shadow_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            shadow_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1.0 - self.decay
            )

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)

    def eval(self):
        return self.shadow.eval()

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


# ---------------------------------------------------------------------------
# Seed Everything
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set all random seeds for reproducibility.

    Sets seeds for Python random, NumPy, PyTorch CPU/CUDA, and
    configures CuDNN for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)

    if HAS_PYTORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic kernels improve reproducibility but can significantly
        # reduce throughput. Keep fast kernels as the default training mode.
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)

    Log.debug(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Training Engine
# ---------------------------------------------------------------------------

class TrainingEngine:
    """
    Professional-grade training engine for audio classification.

    Orchestrates the full training loop with AMP, gradient accumulation,
    EMA, SWA, warmup scheduling, batch augmentation, checkpointing,
    and comprehensive logging.
    """

    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        """
        Args:
            config: Training configuration dict (from TrainingConfig.to_dict())
            device: Compute device ("cpu", "cuda", "mps")
        """
        self.config = config
        self.device = torch.device(device) if HAS_PYTORCH else None

        # AMP
        self.use_amp = config.get("use_amp", False) and device != "cpu"
        self.scaler = GradScaler() if self.use_amp and HAS_PYTORCH else None

        # Gradient accumulation
        self.accum_steps = config.get("gradient_accumulation_steps", 1)

        # Classification mode
        self.is_binary = config.get("classification_mode", "multiclass") == "binary"

        # TensorBoard
        self.writer = None
        if config.get("use_tensorboard", False) and HAS_TENSORBOARD:
            tb_dir = config.get("tensorboard_dir")
            if tb_dir:
                self.writer = SummaryWriter(log_dir=tb_dir)
                Log.info(f"TensorBoard logging to {tb_dir}")

    def train(
        self,
        model: "nn.Module",
        criterion: "nn.Module",
        train_loader: "DataLoader",
        val_loader: "DataLoader",
        classes: List[str],
        progress_tracker=None,
    ) -> TrainingResult:
        """
        Run the full training loop.

        Args:
            model: Model to train
            criterion: Loss function
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            classes: List of class names
            progress_tracker: Optional EchoZero progress tracker

        Returns:
            TrainingResult with trained model and statistics
        """
        model.to(self.device)
        criterion.to(self.device)

        # Create optimizer
        optimizer = self._create_optimizer(model)

        # Create LR scheduler with warmup
        scheduler = self._create_scheduler(optimizer, len(train_loader))

        # EMA
        ema = None
        if self.config.get("use_ema", False):
            ema = EMAModel(model, decay=self.config.get("ema_decay", 0.999))
            Log.info(f"EMA enabled with decay={self.config.get('ema_decay', 0.999)}")

        # SWA
        swa_model = None
        swa_scheduler = None
        swa_start = self.config.get("swa_start_epoch", 50)
        if self.config.get("use_swa", False) and HAS_SWA:
            swa_model = AveragedModel(model)
            swa_lr = self.config.get("swa_lr", 0.0001)
            swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
            Log.info(f"SWA enabled starting at epoch {swa_start}")

        # Checkpoint config
        checkpoint_every = self.config.get("checkpoint_every_n_epochs", 10)
        checkpoint_dir = self.config.get("checkpoint_dir")

        # Resume from checkpoint
        start_epoch = 0
        best_val_metric = 0.0
        resume_path = self.config.get("resume_from")
        if resume_path:
            from .checkpointing import load_checkpoint
            ckpt_info = load_checkpoint(
                resume_path, model, optimizer, scheduler,
                ema_model=ema.shadow if ema else None,
                device=str(self.device),
            )
            start_epoch = ckpt_info.get("epoch", 0) + 1
            best_val_metric = ckpt_info.get("best_val_metric", 0.0)
            Log.info(f"Resuming from epoch {start_epoch}")

        # Training loop state
        epochs = self.config.get("epochs", 100)
        patience = self.config.get("early_stopping_patience", 15)
        min_delta = self.config.get("early_stopping_min_delta", 0.001)
        patience_counter = 0
        best_model_state = None
        best_epoch = 0

        stats = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rates": [],
        }

        # Batch augmentation config
        use_mixup = self.config.get("use_mixup", False)
        mixup_alpha = self.config.get("mixup_alpha", 0.2)
        use_cutmix = self.config.get("use_cutmix", False)
        cutmix_alpha = self.config.get("cutmix_alpha", 1.0)
        num_classes = 1 if self.is_binary else len(classes)

        # Progress
        from src.features.execution.application.progress_helpers import IncrementalProgress
        progress = IncrementalProgress(progress_tracker, "Training model", total=epochs)

        Log.info(f"Starting training for up to {epochs} epochs on {self.device}")
        training_start = time.time()

        for epoch in range(start_epoch, epochs):
            # --- Training Phase ---
            train_loss, train_acc = self._train_epoch(
                model, criterion, optimizer, train_loader,
                use_mixup, mixup_alpha, use_cutmix, cutmix_alpha, num_classes,
            )

            # Update EMA
            if ema:
                ema.update(model)

            # --- Validation Phase ---
            eval_model = ema.shadow if ema else model
            val_loss, val_acc = self._validate_epoch(eval_model, criterion, val_loader)

            # --- LR Scheduling ---
            in_swa = swa_model is not None and epoch >= swa_start
            if in_swa:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            elif scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            # Record stats
            current_lr = optimizer.param_groups[0]["lr"]
            stats["epochs"].append(epoch + 1)
            stats["train_loss"].append(train_loss)
            stats["val_loss"].append(val_loss)
            stats["train_accuracy"].append(train_acc)
            stats["val_accuracy"].append(val_acc)
            stats["learning_rates"].append(current_lr)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

            # Early stopping
            if val_acc > best_val_metric + min_delta:
                best_val_metric = val_acc
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            # Periodic logging
            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                Log.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                    f"LR={current_lr:.6f}, Patience={patience_counter}/{patience}"
                )

            # Progress
            progress.step(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train: {train_acc:.1f}%, Val: {val_acc:.1f}%"
            )

            # Periodic checkpointing
            if checkpoint_dir and (epoch + 1) % checkpoint_every == 0:
                from .checkpointing import save_checkpoint
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, best_val_metric,
                    self.config, classes, stats,
                    ema_model=ema.shadow if ema else None,
                    checkpoint_dir=checkpoint_dir,
                    is_best=(epoch + 1 == best_epoch),
                )

            # Early stopping
            if patience_counter >= patience:
                Log.info(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best val accuracy: {best_val_metric:.2f}% (epoch {best_epoch})"
                )
                break

        # Training complete
        elapsed = time.time() - training_start
        Log.info(
            f"Training complete in {elapsed:.1f}s. "
            f"Best accuracy: {best_val_metric:.2f}% (epoch {best_epoch})"
        )
        progress.complete(f"Training complete - Best accuracy: {best_val_metric:.1f}%")

        # Restore best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Update SWA batch norm if used
        if swa_model is not None:
            try:
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device=self.device)
            except Exception as e:
                Log.warning(f"SWA BN update failed: {e}")

        # Close TensorBoard
        if self.writer:
            self.writer.close()

        return TrainingResult(
            model=model,
            ema_model=ema.shadow if ema else None,
            classes=classes,
            stats=stats,
            best_val_metric=best_val_metric,
            best_epoch=best_epoch,
        )

    def _train_epoch(
        self,
        model: "nn.Module",
        criterion: "nn.Module",
        optimizer: "optim.Optimizer",
        train_loader: "DataLoader",
        use_mixup: bool,
        mixup_alpha: float,
        use_cutmix: bool,
        cutmix_alpha: float,
        num_classes: int,
    ) -> Tuple[float, float]:
        """Run one training epoch. Returns (avg_loss, accuracy_percent)."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad()
        data_wait_total = 0.0
        max_data_wait = 0.0
        next_batch_wait_start = time.perf_counter()

        for batch_idx, batch_data in enumerate(train_loader):
            data_wait = time.perf_counter() - next_batch_wait_start
            data_wait_total += data_wait
            if data_wait > max_data_wait:
                max_data_wait = data_wait

            # Unpack batch (may include onset strengths)
            if len(batch_data) == 3:
                inputs, labels, onset_strengths = batch_data
                onset_strengths = onset_strengths.to(self.device)
            else:
                inputs, labels = batch_data
                onset_strengths = None

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup or CutMix (mutually exclusive per batch); skip when batch size is 1
            mixed_targets = None
            if inputs.size(0) > 1:
                if use_mixup and not use_cutmix:
                    from .augmentation import apply_mixup
                    inputs, mixed_targets = apply_mixup(inputs, labels, mixup_alpha, num_classes)
                elif use_cutmix:
                    from .augmentation import apply_cutmix
                    inputs, mixed_targets = apply_cutmix(inputs, labels, cutmix_alpha, num_classes)

            # Forward pass with optional AMP
            if self.use_amp and self.scaler:
                with autocast():
                    outputs = model(inputs)
                    loss = self._compute_loss(
                        criterion, outputs, labels, mixed_targets, onset_strengths
                    )
                    loss = loss / self.accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.accum_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(optimizer)
                    clip_norm = self.config.get("gradient_clip_norm", 1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(inputs)
                loss = self._compute_loss(
                    criterion, outputs, labels, mixed_targets, onset_strengths
                )
                loss = loss / self.accum_steps
                loss.backward()

                if (batch_idx + 1) % self.accum_steps == 0:
                    clip_norm = self.config.get("gradient_clip_norm", 1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.accum_steps

            if self.is_binary:
                preds = (torch.sigmoid(outputs.squeeze(-1)) >= 0.5).long()
            else:
                _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            next_batch_wait_start = time.perf_counter()

        if max_data_wait > 2.0:
            Log.warning(
                f"Data loader was the bottleneck in this epoch "
                f"(max batch wait {max_data_wait:.2f}s, total wait {data_wait_total:.2f}s). "
                "If augmentation is enabled, try increasing Data Workers."
            )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / max(total, 1)
        return avg_loss, accuracy

    def _validate_epoch(
        self,
        model: "nn.Module",
        criterion: "nn.Module",
        val_loader: "DataLoader",
    ) -> Tuple[float, float]:
        """Run one validation epoch. Returns (avg_loss, accuracy_percent)."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    inputs, labels, onset_strengths = batch_data
                    onset_strengths = onset_strengths.to(self.device)
                else:
                    inputs, labels = batch_data
                    onset_strengths = None

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = model(inputs)
                        loss = self._compute_loss(
                            criterion, outputs, labels, None, onset_strengths
                        )
                else:
                    outputs = model(inputs)
                    loss = self._compute_loss(
                        criterion, outputs, labels, None, onset_strengths
                    )

                total_loss += loss.item()

                if self.is_binary:
                    preds = (torch.sigmoid(outputs.squeeze(-1)) >= 0.5).long()
                else:
                    _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / max(total, 1)
        return avg_loss, accuracy

    def _compute_loss(
        self,
        criterion: "nn.Module",
        outputs: "torch.Tensor",
        labels: "torch.Tensor",
        mixed_targets: Optional["torch.Tensor"],
        onset_strengths: Optional["torch.Tensor"],
    ) -> "torch.Tensor":
        """Compute loss handling mixed targets (Mixup/CutMix) and onset weighting."""
        if mixed_targets is not None:
            # Mixup/CutMix: use soft targets
            if self.is_binary:
                outputs_squeezed = outputs.squeeze(-1)
                return nn.functional.binary_cross_entropy_with_logits(
                    outputs_squeezed, mixed_targets.float()
                )
            else:
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                return -(mixed_targets * log_probs).sum(dim=1).mean()

        # Standard loss
        try:
            if onset_strengths is not None:
                return criterion(outputs, labels, onset_strengths)
            return criterion(outputs, labels)
        except TypeError:
            return criterion(outputs, labels)

    def _create_optimizer(self, model: "nn.Module") -> "optim.Optimizer":
        """Create optimizer from configuration."""
        opt_type = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 0.001)
        wd = self.config.get("weight_decay", 1e-4)
        momentum = self.config.get("momentum", 0.9)

        if opt_type == "adamw":
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        else:
            return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def _create_scheduler(
        self, optimizer: "optim.Optimizer", steps_per_epoch: int
    ) -> Optional[Any]:
        """Create LR scheduler with warmup."""
        sched_type = self.config.get("lr_scheduler", "cosine_restarts").lower()
        warmup_epochs = self.config.get("warmup_epochs", 5)
        epochs = self.config.get("epochs", 100)

        if sched_type == "none":
            base_scheduler = None
        elif sched_type == "step":
            base_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.get("lr_step_size", 30),
                gamma=self.config.get("lr_gamma", 0.1),
            )
        elif sched_type == "cosine":
            base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs
            )
        elif sched_type == "cosine_restarts":
            base_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.get("cosine_t_0", 10),
                T_mult=self.config.get("cosine_t_mult", 2),
            )
        elif sched_type == "plateau":
            base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
            )
        else:
            Log.warning(f"Unknown scheduler '{sched_type}', using cosine_restarts")
            base_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2,
            )

        # Wrap with warmup if needed
        if warmup_epochs > 0 and base_scheduler is not None:
            # ReduceLROnPlateau doesn't support SequentialLR well, skip warmup wrapping
            if isinstance(base_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                return base_scheduler

            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, base_scheduler],
                milestones=[warmup_epochs],
            )
            return scheduler

        return base_scheduler

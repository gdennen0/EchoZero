"""
Training Configuration

Single source of truth for all training parameters. Provides a validated
TrainingConfig dataclass that replaces the dict-based configuration used
in the original trainer blocks.

Supports both multi-class and binary (one-vs-all) classification modes.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.utils.message import Log


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration for audio classification models.

    Supports three classification modes:
    - "multiclass": Standard N-class classification (kick vs snare vs clap vs ...)
    - "binary": One-vs-all detection (is this a kick? yes/no)
    - "positive_vs_other": Multiple selected classes as positives; all others become "other"

    All parameters have sensible defaults. Create from block metadata via
    TrainingConfig.from_block_metadata(block.metadata).
    """

    # === Classification Mode ===
    classification_mode: str = "multiclass"  # "binary", "multiclass", or "positive_vs_other"
    target_class: Optional[str] = None  # For binary mode: primary class (backward compat; use positive_classes)
    positive_classes: List[str] = field(default_factory=list)  # For binary: class names to classify as "positive"; all others = negative
    negative_ratio: float = 1.0  # Ratio of negative to positive samples in binary mode
    hard_negative_dir: Optional[str] = None  # Directory of known hard negatives
    # Optional filter applied to all positive-class samples (binary mode): lowpass, highpass, or bandpass
    positive_filter_type: Optional[str] = None  # "lowpass", "highpass", "bandpass", or None
    positive_filter_cutoff_hz: float = 1000.0  # Cutoff for lowpass/highpass; low edge for bandpass
    positive_filter_cutoff_high_hz: float = 4000.0  # High edge for bandpass only
    positive_filter_order: int = 4  # Butterworth filter order (1-8)
    confidence_threshold: float = 0.5  # Decision threshold for binary mode
    auto_tune_threshold: bool = True  # Auto-find optimal threshold on validation set
    threshold_metric: str = "f1"  # Metric to optimize: "f1", "precision", "recall", "youden"

    # === Data Configuration ===
    data_dir: Optional[str] = None  # Directory containing class folders (required)
    sample_rate: int = 22050
    max_length: int = 22050  # Maximum audio length in samples
    audio_formats: List[str] = field(
        default_factory=lambda: ["wav", "flac", "ogg", "aiff"]
    )

    # === Data Splitting ===
    validation_split: float = 0.15
    test_split: float = 0.10
    stratified_split: bool = True  # Maintain class proportions in splits

    # === Model Architecture ===
    model_type: str = "cnn"  # "cnn", "resnet18", "resnet34", "efficientnet_b0", "rnn", "transformer", "wav2vec2", "ensemble"
    pretrained_backbone: bool = True  # Use pretrained ImageNet weights for resnet/efficientnet

    # CNN-specific
    num_conv_layers: int = 4
    base_channels: int = 32
    fc_hidden_size: int = 512
    use_se_blocks: bool = False  # Squeeze-and-Excitation attention

    # RNN-specific
    rnn_type: str = "LSTM"  # "LSTM", "GRU", "RNN"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 2
    rnn_bidirectional: bool = True
    use_attention: bool = False

    # Transformer-specific
    transformer_d_model: int = 256
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_input_size: int = 128

    # Wav2Vec2-specific
    wav2vec2_model: str = "facebook/wav2vec2-base"
    freeze_wav2vec2: bool = True

    # Ensemble-specific
    ensemble_models: List[Dict[str, Any]] = field(default_factory=list)
    ensemble_weights: List[float] = field(default_factory=list)
    use_meta_classifier: bool = False

    # === Training Parameters ===
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    dropout_rate: float = 0.4
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    momentum: float = 0.9  # SGD momentum

    # === Learning Rate Scheduling ===
    lr_scheduler: str = "cosine_restarts"  # "none", "step", "cosine", "plateau", "cosine_restarts"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    warmup_epochs: int = 5  # Linear warmup before main schedule
    cosine_t_0: int = 10  # Period for cosine restarts
    cosine_t_mult: int = 2  # Period multiplier for cosine restarts

    # === Early Stopping ===
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    # === Regularization ===
    label_smoothing: float = 0.0  # 0.0 = disabled, 0.1 = typical
    gradient_clip_norm: float = 1.0

    # === Class Imbalance / Dataset Balancing ===
    use_class_weights: bool = True  # Weight loss by inverse class frequency
    use_weighted_sampling: bool = False  # WeightedRandomSampler in DataLoader
    balance_strategy: str = "none"  # "none", "undersample_min", "undersample_median", "undersample_target", "oversample_max", "oversample_target", "smart_undersample", "hybrid"
    balance_target_count: Optional[int] = None  # Target count per class (for target-based strategies)

    # === Data Augmentation ===
    use_augmentation: bool = False
    pitch_shift_range: float = 2.0  # Max semitones
    time_stretch_range: float = 0.2  # Max stretch factor deviation
    noise_factor: float = 0.01  # Gaussian noise stddev
    volume_factor: float = 0.1  # Volume change range
    time_shift_max: float = 0.1  # Max time shift as fraction of length
    frequency_mask: int = 0  # SpecAugment frequency mask width
    time_mask: int = 0  # SpecAugment time mask width
    polarity_inversion_prob: float = 0.0  # Probability of polarity flip

    # === Advanced Augmentation ===
    use_mixup: bool = False
    mixup_alpha: float = 0.2  # Beta distribution parameter
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0  # Beta distribution parameter
    use_random_eq: bool = False  # Random EQ filtering

    # === Mixed Precision ===
    use_amp: bool = True  # Automatic mixed precision (auto-disabled on CPU)

    # === Gradient Accumulation ===
    gradient_accumulation_steps: int = 1  # Effective batch = batch_size * this

    # === EMA (Exponential Moving Average) ===
    use_ema: bool = False
    ema_decay: float = 0.999

    # === SWA (Stochastic Weight Averaging) ===
    use_swa: bool = False
    swa_start_epoch: int = 50  # Start SWA after this many epochs
    swa_lr: float = 0.0001

    # === DOSE-Inspired Features ===
    use_onset_weighting: bool = False
    onset_loss_weight: float = 0.3
    use_transient_emphasis: bool = False
    use_multi_scale_features: bool = False

    # === Cross-Validation ===
    use_cross_validation: bool = False
    cv_folds: int = 5

    # === Hyperparameter Optimization ===
    use_hyperopt: bool = False
    hyperopt_trials: int = 50
    hyperopt_timeout: int = 3600

    # === DataLoader ===
    num_workers: int = 0  # DataLoader workers. 0 = load in training thread. With augmentation OFF, feature cache makes 0 fast. With augmentation ON, use 2-4 workers so load+augment runs in parallel with training (otherwise the GPU stalls waiting for CPU).

    # === Feature Caching ===
    use_feature_cache: bool = True  # Cache spectrograms to disk

    # === Normalization ===
    normalize_per_dataset: bool = True  # Use dataset-wide mean/std
    normalization_mean: Optional[List[float]] = None  # Precomputed (set during training)
    normalization_std: Optional[List[float]] = None  # Precomputed (set during training)

    # === Spectrogram Parameters ===
    n_mels: int = 128
    hop_length: int = 512
    fmax: int = 8000
    n_fft: int = 2048

    # === Logging ===
    use_tensorboard: bool = False
    tensorboard_dir: Optional[str] = None  # Auto-generated if None

    # === Checkpointing ===
    checkpoint_dir: Optional[str] = None  # Auto-generated if None
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    checkpoint_every_n_epochs: int = 10  # Save checkpoint every N epochs
    save_best_only: bool = True

    # === Export ===
    export_onnx: bool = False
    export_quantized: bool = False
    output_model_path: Optional[str] = None  # Custom save path
    model_name: Optional[str] = None  # User-defined name for the trained model (used in filename and output item)

    # === Reproducibility ===
    seed: int = 42
    deterministic_training: bool = False  # True = reproducible but slower backend kernels; False = faster training kernels where available

    # === Device ===
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        errors = []

        # Classification mode
        if self.classification_mode not in ("binary", "multiclass", "positive_vs_other"):
            errors.append(
                f"classification_mode must be 'binary', 'multiclass', or 'positive_vs_other', "
                f"got '{self.classification_mode}'"
            )

        # Binary mode: require at least one positive class (normalize from target_class if needed)
        if self.classification_mode == "binary":
            if not self.positive_classes and self.target_class:
                self.positive_classes = [self.target_class]
            if not self.positive_classes:
                errors.append(
                    "When classification_mode='binary', select at least one class to classify (positive_classes). "
                    "All other classes will be treated as negative."
                )
            elif any(not (c and str(c).strip()) for c in self.positive_classes):
                errors.append("positive_classes must contain non-empty class names.")
            elif len(self.positive_classes) != len(set(self.positive_classes)):
                errors.append("positive_classes must not contain duplicates.")

        # positive_vs_other mode: require at least one positive class
        if self.classification_mode == "positive_vs_other":
            if not self.positive_classes and self.target_class:
                self.positive_classes = [self.target_class]
            if not self.positive_classes:
                errors.append(
                    "When classification_mode='positive_vs_other', select at least one positive class. "
                    "All other classes will be grouped as 'other'."
                )
            elif any(not (c and str(c).strip()) for c in self.positive_classes):
                errors.append("positive_classes must contain non-empty class names.")
            elif len(self.positive_classes) != len(set(self.positive_classes)):
                errors.append("positive_classes must not contain duplicates.")

        if self.negative_ratio <= 0:
            errors.append("negative_ratio must be positive")

        if self.threshold_metric not in ("f1", "precision", "recall", "youden"):
            errors.append(
                f"threshold_metric must be 'f1', 'precision', 'recall', or 'youden', "
                f"got '{self.threshold_metric}'"
            )

        # Positive-class filter (binary mode only; applied to all positive samples)
        if self.positive_filter_type is not None:
            if self.positive_filter_type not in ("lowpass", "highpass", "bandpass"):
                errors.append(
                    f"positive_filter_type must be 'lowpass', 'highpass', or 'bandpass', "
                    f"got '{self.positive_filter_type}'"
                )
            if self.positive_filter_cutoff_hz <= 0:
                errors.append("positive_filter_cutoff_hz must be positive")
            if self.positive_filter_type == "bandpass":
                if self.positive_filter_cutoff_high_hz <= self.positive_filter_cutoff_hz:
                    errors.append(
                        "positive_filter_cutoff_high_hz must be greater than "
                        "positive_filter_cutoff_hz for bandpass"
                    )
            if not 1 <= self.positive_filter_order <= 8:
                errors.append("positive_filter_order must be between 1 and 8")

        # Splits
        total_split = self.validation_split + self.test_split
        if total_split >= 1.0:
            errors.append(
                f"validation_split ({self.validation_split}) + test_split ({self.test_split}) "
                f"must be less than 1.0"
            )

        # Training params
        if self.epochs < 1:
            errors.append("epochs must be >= 1")
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.num_workers < 0:
            errors.append("num_workers must be >= 0")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if not 0.0 <= self.dropout_rate <= 1.0:
            errors.append("dropout_rate must be between 0.0 and 1.0")
        if not 0.0 <= self.label_smoothing < 1.0:
            errors.append("label_smoothing must be between 0.0 and 1.0")

        # Optimizer
        if self.optimizer not in ("adam", "adamw", "sgd"):
            errors.append(f"optimizer must be 'adam', 'adamw', or 'sgd', got '{self.optimizer}'")

        # LR scheduler
        valid_schedulers = ("none", "step", "cosine", "plateau", "cosine_restarts")
        if self.lr_scheduler not in valid_schedulers:
            errors.append(f"lr_scheduler must be one of {valid_schedulers}, got '{self.lr_scheduler}'")

        # Model type
        valid_models = (
            "cnn", "resnet18", "resnet34", "resnet50",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
            "rnn", "lstm", "gru", "transformer",
            "wav2vec2", "ensemble",
        )
        if self.model_type.lower() not in valid_models:
            errors.append(f"model_type must be one of {valid_models}, got '{self.model_type}'")

        # Balance strategy
        valid_balance = (
            "none", "undersample_min", "undersample_median", "undersample_target",
            "oversample_max", "oversample_target", "smart_undersample", "hybrid",
        )
        if self.balance_strategy not in valid_balance:
            errors.append(
                f"balance_strategy must be one of {valid_balance}, got '{self.balance_strategy}'"
            )
        if self.balance_strategy in ("undersample_target", "oversample_target") and not self.balance_target_count:
            errors.append(
                f"balance_target_count is required when balance_strategy='{self.balance_strategy}'"
            )

        # Gradient accumulation
        if self.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps must be >= 1")

        # EMA
        if self.use_ema and not 0.0 < self.ema_decay < 1.0:
            errors.append("ema_decay must be between 0.0 and 1.0")

        # SWA
        if self.use_swa and self.swa_start_epoch >= self.epochs:
            errors.append(
                f"swa_start_epoch ({self.swa_start_epoch}) must be less than epochs ({self.epochs})"
            )

        if errors:
            error_str = "\n".join(f"  - {e}" for e in errors)
            raise ValueError(f"Invalid TrainingConfig:\n{error_str}")

    @classmethod
    def from_block_metadata(cls, metadata: Optional[Dict[str, Any]]) -> "TrainingConfig":
        """
        Create TrainingConfig from block metadata dictionary.

        Handles type coercion and ignores unknown keys gracefully.

        Args:
            metadata: Block metadata dictionary (may contain non-config keys)

        Returns:
            Validated TrainingConfig instance
        """
        if not metadata:
            return cls()

        # Get only keys that are valid TrainingConfig fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}

        # Filter and coerce types
        config_dict = {}
        for key, value in metadata.items():
            if key in valid_fields and value is not None:
                config_dict[key] = value

        # Handle device auto-detection
        if config_dict.get("device", "auto") == "auto":
            config_dict["device"] = cls._detect_device()

        # Auto-disable AMP on CPU
        if config_dict.get("device", "cpu") == "cpu":
            config_dict["use_amp"] = False

        # Binary / positive_vs_other: normalize positive_classes from target_class for backward compat
        if config_dict.get("classification_mode") in ("binary", "positive_vs_other"):
            pos = config_dict.get("positive_classes") or []
            if not pos and config_dict.get("target_class"):
                config_dict["positive_classes"] = [config_dict["target_class"]]
            pos = config_dict.get("positive_classes") or []
            if pos and not config_dict.get("target_class"):
                config_dict["target_class"] = pos[0]

        try:
            return cls(**config_dict)
        except (TypeError, ValueError) as e:
            Log.warning(f"TrainingConfig creation error: {e}. Using defaults for invalid fields.")
            # Fallback: try field by field
            safe_dict = {}
            for key, value in config_dict.items():
                try:
                    test_dict = {key: value}
                    # Validate individually by checking type
                    safe_dict[key] = value
                except Exception:
                    Log.warning(f"Skipping invalid config field: {key}={value}")
            return cls(**safe_dict)

    @staticmethod
    def _detect_device() -> str:
        """Detect the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size accounting for gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def is_binary(self) -> bool:
        """Whether this is a binary classification task."""
        return self.classification_mode == "binary"

    @property
    def is_positive_vs_other(self) -> bool:
        """Whether this is positive-vs-other (multi-positive, rest = other)."""
        return self.classification_mode == "positive_vs_other"

    @property
    def num_output_classes(self) -> int:
        """
        Number of output classes for the model head.

        For binary mode, this is 1 (sigmoid output).
        For multiclass, this is determined by the dataset at runtime.
        Returns 0 as a sentinel for multiclass (must be set by dataset).
        """
        return 1 if self.is_binary else 0

    def get_model_save_name(self) -> str:
        """Generate a descriptive model filename."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.is_binary:
            names = self.positive_classes or ([self.target_class] if self.target_class else [])
            tag = "_".join(names)[:60] if names else "positive"
            return f"binary_{tag}_{self.model_type}_{timestamp}.pth"
        if self.is_positive_vs_other:
            names = self.positive_classes or ([self.target_class] if self.target_class else [])
            tag = "_".join(names)[:60] if names else "positive"
            return f"positive_vs_other_{tag}_{self.model_type}_{timestamp}.pth"
        return f"multiclass_{self.model_type}_{timestamp}.pth"

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            f"Classification Mode: {self.classification_mode}",
        ]
        if self.is_binary:
            names = self.positive_classes or ([self.target_class] if self.target_class else [])
            if len(names) == 1:
                lines.append(f"  Target Class: {names[0]}")
            else:
                lines.append(f"  Positive Classes: {', '.join(names)}")
            lines.append(f"  Negative Ratio: {self.negative_ratio}")
            lines.append(f"  Auto-tune Threshold: {self.auto_tune_threshold}")
        elif self.is_positive_vs_other:
            names = self.positive_classes or ([self.target_class] if self.target_class else [])
            lines.append(f"  Positive Classes: {', '.join(names)}")
            lines.append("  Other: all unselected classes")

        balance_desc = self.balance_strategy
        if self.balance_strategy != "none" and self.balance_target_count:
            balance_desc += f" (target={self.balance_target_count})"

        lines.extend([
            f"Model: {self.model_type} (pretrained={self.pretrained_backbone})",
            f"Epochs: {self.epochs} (patience={self.early_stopping_patience})",
            f"Batch Size: {self.batch_size} (effective={self.effective_batch_size})",
            f"Learning Rate: {self.learning_rate} (scheduler={self.lr_scheduler})",
            f"Warmup: {self.warmup_epochs} epochs",
            f"AMP: {self.use_amp}, EMA: {self.use_ema}, SWA: {self.use_swa}",
            f"Augmentation: {self.use_augmentation} (mixup={self.use_mixup}, cutmix={self.use_cutmix})",
            f"Label Smoothing: {self.label_smoothing}",
            f"Class Weights: {self.use_class_weights}",
            f"Dataset Balance: {balance_desc}",
            f"DataLoader num_workers: {self.num_workers}",
            f"Feature Cache: {self.use_feature_cache}",
            f"Device: {self.device}",
            f"Seed: {self.seed}",
            f"Deterministic Training: {self.deterministic_training}",
        ])
        return "\n".join(lines)

"""
Training Infrastructure Package

Professional-grade PyTorch training for audio classification with support
for multiclass, binary (one-vs-all), and positive-vs-other classification modes.

Public API:
    - TrainingConfig: Configuration dataclass
    - AudioClassificationDataset: Dataset with caching and augmentation
    - create_data_loaders: Create train/val/test DataLoaders
    - create_classifier: Model factory
    - create_loss_function: Loss factory
    - TrainingEngine: Core training loop
    - evaluate_model: Full model evaluation
    - tune_threshold: Binary threshold optimization
    - save_final_model: Production model saving
    - export_onnx: ONNX model export
    - seed_everything: Reproducibility

Backward Compatibility:
    Architecture classes are re-exported for model checkpoint loading.
"""
# Configuration
from .config import TrainingConfig

# Architectures (re-exported for backward compat with existing checkpoints)
from .architectures import (
    AudioClassifierBase,
    AudioClassifierFactory,
    CNNClassifier,
    RNNClassifier,
    TransformerClassifier,
    Wav2Vec2Classifier,
    EnsembleClassifier,
    ResNetClassifier,
    EfficientNetClassifier,
    SEBlock,
    create_classifier,
    ARCHITECTURE_REGISTRY,
)

# Datasets
from .datasets import (
    AudioClassificationDataset,
    DatasetStats,
    create_data_loaders,
    create_data_splits,
)

# Augmentation
from .augmentation import (
    AudioAugmentationPipeline,
    SpectrogramAugmentationPipeline,
    apply_mixup,
    apply_cutmix,
)

# Losses
from .losses import (
    LabelSmoothingCrossEntropy,
    FocalLoss,
    BinaryCrossEntropyWithLogits,
    BinaryFocalLoss,
    OnsetAwareLoss,
    CombinedLoss,
    create_loss_function,
    compute_class_weights,
    compute_binary_pos_weight,
)

# Training Engine
from .engine import (
    TrainingEngine,
    TrainingResult,
    EMAModel,
    seed_everything,
)

# Evaluation
from .evaluation import (
    evaluate_model,
    tune_threshold,
    find_hard_negatives,
    predict_with_tta,
    TemperatureScaling,
)

# Checkpointing
from .checkpointing import (
    save_checkpoint,
    load_checkpoint,
    save_final_model,
    load_training_metadata,
)

# Export
from .export import (
    export_onnx,
    quantize_model,
)

# Balancing
from .balancing import (
    balance_dataset,
    preview_balance,
    BalanceResult,
    BALANCE_STRATEGIES,
)

__all__ = [
    # Config
    "TrainingConfig",
    # Architectures
    "AudioClassifierBase",
    "AudioClassifierFactory",
    "CNNClassifier",
    "RNNClassifier",
    "TransformerClassifier",
    "Wav2Vec2Classifier",
    "EnsembleClassifier",
    "ResNetClassifier",
    "EfficientNetClassifier",
    "SEBlock",
    "create_classifier",
    "ARCHITECTURE_REGISTRY",
    # Datasets
    "AudioClassificationDataset",
    "DatasetStats",
    "create_data_loaders",
    "create_data_splits",
    # Augmentation
    "AudioAugmentationPipeline",
    "SpectrogramAugmentationPipeline",
    "apply_mixup",
    "apply_cutmix",
    # Losses
    "LabelSmoothingCrossEntropy",
    "FocalLoss",
    "BinaryCrossEntropyWithLogits",
    "BinaryFocalLoss",
    "OnsetAwareLoss",
    "CombinedLoss",
    "create_loss_function",
    "compute_class_weights",
    "compute_binary_pos_weight",
    # Engine
    "TrainingEngine",
    "TrainingResult",
    "EMAModel",
    "seed_everything",
    # Evaluation
    "evaluate_model",
    "tune_threshold",
    "find_hard_negatives",
    "predict_with_tta",
    "TemperatureScaling",
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "save_final_model",
    "load_training_metadata",
    # Export
    "export_onnx",
    "quantize_model",
    # Balancing
    "balance_dataset",
    "preview_balance",
    "BalanceResult",
    "BALANCE_STRATEGIES",
]

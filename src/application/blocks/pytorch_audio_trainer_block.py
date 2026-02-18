"""
PyTorch Audio Trainer Block Processor

Thin orchestration layer that delegates to the training/ package for all
heavy lifting. Supports both multi-class and binary (one-vs-all) classification.

Configuration is provided via block.metadata and converted to TrainingConfig.
See training/config.py for all available parameters.
"""
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import os

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.entities.data_item import AudioDataItem as ModelDataItem
from src.application.blocks import register_processor_class
from src.features.execution.application.progress_helpers import get_progress_tracker
from src.utils.datasets import resolve_dataset_path
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade

# Lazy import training infrastructure to avoid loading PyTorch at module level
_TRAINING_AVAILABLE = None


def _check_training_available():
    """Check if training dependencies are available (cached)."""
    global _TRAINING_AVAILABLE
    if _TRAINING_AVAILABLE is None:
        try:
            import torch
            import librosa
            _TRAINING_AVAILABLE = True
        except ImportError as e:
            _TRAINING_AVAILABLE = False
            Log.warning(f"Training dependencies not available: {e}")
        
    return _TRAINING_AVAILABLE


class PyTorchAudioTrainerBlockProcessor(BlockProcessor):
    """
    Advanced PyTorch Audio Trainer with professional ML practices.

    Supports multiclass, binary (one-vs-all), and positive-vs-other classification modes.
    Delegates to the training/ package for all model training, evaluation,
    and export functionality.

    Key features:
    - Multiple architectures: CNN, ResNet, EfficientNet, RNN, Transformer
    - Binary mode: independent per-class detectors with threshold tuning
    - Mixed precision training (AMP)
    - Data augmentation with Mixup/CutMix
    - EMA and SWA for better generalization
    - Warmup + cosine annealing with restarts
    - Cross-validation and hyperparameter optimization
    - Comprehensive evaluation metrics
    - ONNX export and model quantization

    See TrainingConfig for all configuration parameters.
    """

    def __init__(self):
        """Initialize processor."""
        if not _check_training_available():
            Log.error(
                "PyTorchAudioTrainer requires PyTorch and librosa. "
                "Install with: pip install torch torchaudio librosa"
            )
            return

        self.device = None

    def cleanup(self, block: Block) -> None:
        """
        Release all PyTorch resources after training completes or when the
        block is removed / project unloaded.
        """
        import gc

        self.device = None

        gc.collect()

        if _check_training_available():
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        Log.debug("PyTorchAudioTrainerBlockProcessor: Resources cleaned up")

    def can_process(self, block: Block) -> bool:
        return block.type == "PyTorchAudioTrainer"

    def get_block_type(self) -> str:
        return "PyTorchAudioTrainer"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        from src.features.blocks.domain import BlockStatusLevel
        
        def check_data_dir(blk: Block, f: "ApplicationFacade") -> bool:
            data_dir = resolve_dataset_path(blk.metadata.get("data_dir"))
            return bool(data_dir and os.path.isdir(data_dir))
        
        def check_dependencies(blk: Block, f: "ApplicationFacade") -> bool:
            return _check_training_available()
        
        return [
            BlockStatusLevel(
                priority=0, name="error", display_name="Error",
                color="#ff6b6b", conditions=[check_dependencies, check_data_dir],
            ),
            BlockStatusLevel(
                priority=1, name="ready", display_name="Ready",
                color="#51cf66", conditions=[],
            ),
        ]

    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataItem]:
        """
        Train an audio classification model.

        Orchestrates the full training pipeline:
        1. Parse and validate configuration
        2. Create dataset and data loaders
        3. Build model architecture
        4. Run training with all professional techniques
        5. Evaluate on test set
        6. Tune threshold (binary mode)
        7. Save model and export

        Args:
            block: Block entity with training configuration in metadata
            inputs: Input data items (unused -- training data comes from data_dir)
            metadata: Execution metadata (progress tracker, etc.)

        Returns:
            Dict with "model" key containing training results

        Raises:
            ProcessingError: If training fails
        """
        if not _check_training_available():
            raise ProcessingError(
                "PyTorch and librosa are required for training. "
                "Install with: pip install torch torchaudio librosa",
                block_id=block.id,
                block_name=block.name,
            )

        # Import training infrastructure
        from .training import (
            TrainingConfig,
            AudioClassificationDataset,
            create_data_loaders,
            create_classifier,
            create_loss_function,
            TrainingEngine,
            evaluate_model,
            tune_threshold,
            save_final_model,
            export_onnx,
            quantize_model,
            seed_everything,
        )
        from .training.model_coach import build_training_feedback

        # --- 1. Configuration ---
        try:
            config = TrainingConfig.from_block_metadata(block.metadata)
        except ValueError as e:
            raise ProcessingError(
                f"Invalid training configuration:\n{e}",
                block_id=block.id,
                block_name=block.name,
            )

        # Normalize legacy repo-local paths to managed datasets location.
        config.data_dir = resolve_dataset_path(config.data_dir)
        runtime_device = self._resolve_runtime_device(config, block)
        self.device = runtime_device

        # Validate data directory
        self._validate_data_dir(config, block)

        Log.info(f"Training configuration:\n{config.summary()}")

        # Seed for reproducibility
        seed_everything(config.seed, deterministic=config.deterministic_training)

        progress_tracker = get_progress_tracker(metadata)
        coach_feedback = None

        def _progress(message: str, current: int = 0, total: int = 100) -> None:
            if progress_tracker:
                if not getattr(progress_tracker, "_trainer_started", False):
                    progress_tracker.start(message, total=total, current=current)
                    progress_tracker._trainer_started = True
                else:
                    progress_tracker.update(current=current, total=total, message=message)

        try:
            # --- 2. Dataset ---
            _progress("Loading dataset...", 0, 100)
            config_dict = config.to_dict()
            dataset = AudioClassificationDataset(config_dict)
            # Persist the actual preprocessing values used by the dataset so
            # saved model inference preprocessing matches training exactly.
            config_dict["sample_rate"] = int(dataset.sample_rate)
            config_dict["max_length"] = int(dataset.max_length)
            config_dict["n_fft"] = int(dataset.n_fft)
            config_dict["n_mels"] = int(dataset.n_mels)
            config_dict["hop_length"] = int(dataset.hop_length)
            config_dict["fmax"] = int(dataset.fmax)
            config_dict["audio_input_standard"] = {
                "encoding": "wav_pcm16",
                "channels": 1,
                "sample_rate": int(dataset.sample_rate),
            }
            if config.normalize_per_dataset:
                _progress("Dataset loaded, computing normalization...", 0, 100)
            else:
                _progress("Dataset loaded, per-sample normalization selected.", 0, 100)

            # Compute per-dataset normalization
            if config.normalize_per_dataset:
                norm_mean, norm_std = dataset.compute_normalization_stats()
            else:
                norm_mean, norm_std = None, None
                # Ensure no stale normalization data is carried into checkpoint metadata.
                config_dict["normalization_mean"] = None
                config_dict["normalization_std"] = None
            _progress("Creating data loaders...", 0, 100)

            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(dataset, config_dict)
            _progress("Building model...", 0, 100)

            # --- 3. Model ---
            if config.is_binary:
                num_classes = 1
            else:
                # multiclass and positive_vs_other both use N classes (positive_vs_other: K positives + "other")
                num_classes = len(dataset.classes)

            model = create_classifier(num_classes, config_dict)
            model.to(runtime_device)

            Log.info(
                f"Model: {config.model_type} with {sum(p.numel() for p in model.parameters()):,} parameters"
            )

            # --- 4. Loss Function ---
            class_weights = None
            pos_weight = None

            if config.is_binary:
                pos_weight = dataset.get_binary_pos_weight()
                Log.info(f"Binary pos_weight: {pos_weight:.2f}")
            elif (config.is_positive_vs_other or not config.is_binary) and config.use_class_weights:
                class_weights = dataset.get_class_weights().to(runtime_device)
                Log.info(f"Class weights: {class_weights.tolist()}")

            criterion = create_loss_function(config_dict, class_weights, pos_weight)
            _progress("Starting training...", 0, 100)

            # --- 5. Training ---
            engine = TrainingEngine(config_dict, device=str(runtime_device))
            result = engine.train(
                model=model,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                classes=dataset.classes,
                progress_tracker=progress_tracker,
            )

            # --- 6. Evaluation ---
            if progress_tracker:
                progress_tracker.update(current=100, total=100, message="Training complete, evaluating...")
            eval_model = result.ema_model if result.ema_model else result.model
            eval_model.eval()

            # Evaluate on validation set
            eval_config = config_dict.copy()
            eval_config["_classes"] = dataset.classes
            val_metrics = evaluate_model(eval_model, val_loader, eval_config, str(runtime_device))

            # Evaluate on test set if available
            test_metrics = None
            if test_loader:
                test_metrics = evaluate_model(eval_model, test_loader, eval_config, str(runtime_device))
                Log.info(f"Test set accuracy: {test_metrics.get('accuracy', 0):.2f}%")

            # --- 7. Threshold Tuning (binary mode) ---
            if progress_tracker:
                progress_tracker.update(current=100, total=100, message="Evaluation done, tuning threshold...")
            optimal_threshold = None
            threshold_metrics = None
            if config.is_binary and config.auto_tune_threshold:
                optimal_threshold, threshold_metrics = tune_threshold(
                    eval_model, val_loader,
                    metric=config.threshold_metric,
                    device=str(runtime_device),
                )
                Log.info(f"Optimal threshold: {optimal_threshold:.3f}")

            # --- 8. Save Model ---
            if progress_tracker:
                progress_tracker.update(current=100, total=100, message="Saving model...")
            normalization = None
            if norm_mean is not None:
                normalization = {"mean": norm_mean.tolist(), "std": norm_std.tolist()}

            model_path = save_final_model(
                model=eval_model,
                classes=dataset.classes,
                config=config_dict,
                training_stats=result.stats,
                test_metrics=test_metrics or val_metrics,
                dataset_stats=dataset.stats.to_dict(),
                normalization=normalization,
                optimal_threshold=optimal_threshold,
                ema_model=result.ema_model,
                output_path=config.output_model_path,
            )
            if progress_tracker:
                progress_tracker.update(current=100, total=100, message=f"Model saved: {model_path}")

            # --- 9. Export ---
            onnx_path = None
            quantized_path = None

            if config.export_onnx:
                onnx_path = export_onnx(eval_model, config_dict, model_path)

            if config.export_quantized:
                quantized_path = quantize_model(eval_model, model_path)

            try:
                coach_feedback = build_training_feedback(
                    block_id=block.id,
                    model_path=model_path,
                    config=config_dict,
                    training_stats=result.stats,
                    dataset_stats=dataset.stats.to_dict(),
                    validation_metrics=val_metrics,
                    test_metrics=test_metrics,
                    threshold_metrics=threshold_metrics,
                    excluded_bad_file_count=int(getattr(dataset, "_excluded_bad_file_count", 0)),
                )
            except Exception as coach_error:
                Log.warning(f"Model coach feedback generation failed: {coach_error}")
                coach_feedback = None

        except ProcessingError:
            raise
        except Exception as e:
            error_msg = str(e)
            if "CUDA" in error_msg and "out of memory" in error_msg:
                raise ProcessingError(
                    "GPU Memory Error\n"
                    "Not enough GPU memory for training.\n"
                    "Try: reduce batch_size, reduce max_length, or use CPU.",
                    block_id=block.id,
                    block_name=block.name,
                ) from e
            elif "No audio files found" in error_msg:
                raise ProcessingError(
                    "No Audio Files Found\n"
                    f"No audio files found in the data directory.\n"
                    "Each class folder must contain audio files (.wav, .mp3, etc.).",
                    block_id=block.id,
                    block_name=block.name,
                ) from e
            elif "Failed to prepare training sample" in error_msg:
                raise ProcessingError(
                    "Training Data Pipeline Error\n"
                    f"{error_msg}\n"
                    "A sample failed during load/augment/spectrogram processing. "
                    "Check logs for the exact file path and failing operation.",
                    block_id=block.id,
                    block_name=block.name,
                ) from e
            elif "Failed to standardize input audio files to canonical WAV PCM_16" in error_msg:
                raise ProcessingError(
                    "Audio Standardization Failed\n"
                    f"{error_msg}\n"
                    "At least one input file could not be decoded/converted to canonical WAV PCM_16. "
                    "Remove or replace the failing file(s), then retry.",
                    block_id=block.id,
                    block_name=block.name,
                ) from e
            else:
                raise ProcessingError(
                    f"Training Failed\nError: {error_msg}\n"
                    "Check logs for details.",
                    block_id=block.id,
                    block_name=block.name,
                ) from e

        # --- Build Output ---
        output_metadata = {
            "model_path": model_path,
            "classes": dataset.classes,
            "training_stats": result.stats,
            "config": config_dict,
            "classification_mode": config.classification_mode,
        }
        if hasattr(dataset, "_excluded_bad_file_count"):
            output_metadata["excluded_bad_file_count"] = int(dataset._excluded_bad_file_count)
        if hasattr(dataset, "_excluded_bad_files") and dataset._excluded_bad_files:
            output_metadata["excluded_bad_files"] = list(dataset._excluded_bad_files)

        if test_metrics:
            output_metadata["test_metrics"] = test_metrics
        if optimal_threshold is not None:
            output_metadata["optimal_threshold"] = optimal_threshold
            output_metadata["threshold_metrics"] = threshold_metrics
        if onnx_path:
            output_metadata["onnx_path"] = onnx_path
        if quantized_path:
            output_metadata["quantized_path"] = quantized_path
        if coach_feedback:
            output_metadata["coach_feedback"] = coach_feedback

        # Store results in block metadata
        block.metadata = block.metadata or {}
        block.metadata["last_training"] = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "classes": dataset.classes,
            "classification_mode": config.classification_mode,
            "target_class": config.target_class,
            "best_accuracy": result.best_val_metric,
            "optimal_threshold": optimal_threshold,
            "stats": result.stats,
            "excluded_bad_file_count": int(getattr(dataset, "_excluded_bad_file_count", 0)),
            "excluded_bad_files": list(getattr(dataset, "_excluded_bad_files", [])),
            "coach_feedback": coach_feedback,
        }

        # Wrap output in a DataItem so the execution engine can save it
        display_name = (config.model_name or "").strip()
        if not display_name:
            display_name = f"{config.classification_mode}_{config.model_type}"
            if config.is_binary:
                names = config.positive_classes or ([config.target_class] if config.target_class else [])
                tag = "_".join(names) if names else "positive"
                display_name = f"binary_{tag}_{config.model_type}"

        output_item = ModelDataItem(
            id="",
            block_id=block.id,
            name=display_name,
            type="Model",
            created_at=datetime.now(),
            file_path=model_path,
            metadata=output_metadata,
        )

        return {"model": output_item}

    def _resolve_runtime_device(self, config, block: Block):
        """Resolve and validate runtime device from strict config selection."""
        import torch

        requested = (config.device or "auto").strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():
                resolved = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved = "mps"
            else:
                resolved = "cpu"
            Log.info(f"PyTorchAudioTrainer device auto-resolved to {resolved}")
            return torch.device(resolved)

        if requested == "cuda":
            if not torch.cuda.is_available():
                raise ProcessingError(
                    "Invalid Device Configuration\n"
                    "Device is set to 'cuda' but CUDA is not available on this machine.",
                    block_id=block.id,
                    block_name=block.name,
                )
            return torch.device("cuda")

        if requested == "mps":
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not mps_ok:
                raise ProcessingError(
                    "Invalid Device Configuration\n"
                    "Device is set to 'mps' but Apple Metal (MPS) is not available.",
                    block_id=block.id,
                    block_name=block.name,
                )
            return torch.device("mps")

        if requested == "cpu":
            return torch.device("cpu")

        raise ProcessingError(
            f"Invalid Device Configuration\nUnsupported device '{requested}'. "
            "Allowed values: auto, cpu, cuda, mps.",
            block_id=block.id,
            block_name=block.name,
        )

    def _validate_data_dir(self, config, block: Block) -> None:
        """Validate the training data directory exists and has content."""
        data_dir = resolve_dataset_path(config.data_dir)
        if not data_dir:
            raise ProcessingError(
                "No Training Data Directory\n"
                "Set 'data_dir' in block settings to point to your training data.\n"
                "Example: /path/to/drum_samples\n"
                "The directory should contain subfolders for each class.",
                block_id=block.id,
                block_name=block.name,
            )

        data_path = Path(data_dir)
        if not data_path.exists():
            raise ProcessingError(
                f"Training Directory Not Found\n"
                f"Directory '{data_dir}' does not exist.",
                block_id=block.id,
                block_name=block.name,
            )

        if not data_path.is_dir():
            raise ProcessingError(
                f"Invalid Training Directory\n"
                f"'{data_dir}' is not a directory.",
                block_id=block.id,
                block_name=block.name,
            )

        # Check for class subdirectories
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        if not class_dirs:
            raise ProcessingError(
                "No Class Folders Found\n"
                f"No subdirectories found in '{data_dir}'.\n"
                "Create folders for each class (e.g., kick/, snare/, hihat/).",
                block_id=block.id,
                block_name=block.name,
            )

        # Binary mode: check at least one positive class folder exists
        if config.is_binary:
            folder_names = [d.name for d in class_dirs]
            positive_classes = config.positive_classes or ([config.target_class] if config.target_class else [])
            has_any_positive_class = any(p in folder_names for p in positive_classes) if positive_classes else False

            if not has_any_positive_class:
                names = ", ".join(positive_classes) if positive_classes else "target_class"
                raise ProcessingError(
                    f"Positive Class Folders Not Found\n"
                    f"No folder for selected positive class(es) '{names}' in '{data_dir}'.\n"
                    f"Available folders: {', '.join(folder_names)}\n"
                    "Create folder(s) matching the selected positive class names.",
                    block_id=block.id,
                    block_name=block.name,
                )

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None,
    ) -> List[str]:
        """Validate block configuration and return error messages."""
        errors = []

        if not _check_training_available():
            errors.append(
                "Missing Dependencies\n"
                "PyTorch and librosa are required for training.\n"
                "Install with: pip install torch torchaudio librosa"
            )
            return errors

        data_dir = block.metadata.get("data_dir") if block.metadata else None
        data_dir = resolve_dataset_path(data_dir)
        if not data_dir:
            errors.append(
                "No Training Data Directory Set\n"
                "Set 'data_dir' in block settings."
            )
            return errors

        data_path = Path(data_dir)
        if not data_path.exists():
            errors.append(f"Training directory not found: {data_dir}")
            return errors

        if not data_path.is_dir():
            errors.append(f"'{data_dir}' is not a directory")
            return errors

        device = str((block.metadata or {}).get("device", "auto")).strip().lower()
        if device not in {"auto", "cpu", "cuda", "mps"}:
            errors.append(f"Unsupported device '{device}'. Allowed: auto, cpu, cuda, mps")
            return errors

        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                errors.append("Device set to cuda but CUDA is not available")
                return errors
            if device == "mps":
                mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                if not mps_ok:
                    errors.append("Device set to mps but MPS is not available")
                    return errors
        except Exception as e:
            errors.append(f"Failed to validate device '{device}': {e}")
            return errors

        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        if not class_dirs:
            errors.append(f"No class subdirectories found in '{data_dir}'")
            return errors

        # Check audio files
        total_files = 0
        for d in class_dirs:
            count = sum(
                1 for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in {
                    ".wav", ".flac", ".ogg", ".aiff", ".aif",
                    ".mp3", ".m4a", ".aac", ".wma", ".alac", ".opus", ".mp4",
                }
            )
            total_files += count
            if count == 0:
                errors.append(f"No audio files in {d.name}/")

        if total_files == 0:
            errors.append("No audio files found in any class folder")
        elif total_files < 10:
            errors.append(f"Very few samples ({total_files}). Recommend 50+ per class.")

        return errors


# Auto-register
register_processor_class(PyTorchAudioTrainerBlockProcessor)

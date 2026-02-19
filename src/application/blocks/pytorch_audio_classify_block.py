"""
PyTorchAudioClassify Block Processor

Processes PyTorchAudioClassify blocks - classifies events using PyTorch models
created by the PyTorch Audio Trainer block. Automatically loads model architecture,
classes, and preprocessing config from saved models.
"""
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from datetime import datetime
from collections import Counter
import time
import os
from pathlib import Path
import numpy as np
import json

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.entities import EventDataItem, Event, EventLayer
from src.shared.domain.entities import AudioDataItem
from src.application.blocks import register_processor_class
from src.utils.message import Log
from src.application.blocks.training.model_coach import build_inference_feedback

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade

# Try to import required libraries
try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    Log.warning("PyTorch not available - PyTorchAudioClassify will not work")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    Log.warning("librosa not available - audio preprocessing may be limited")


@register_processor_class
class PyTorchAudioClassifyBlockProcessor(BlockProcessor):
    """
    Processor for PyTorchAudioClassify block type.
    
    Classifies events using PyTorch models created by PyTorch Audio Trainer.
    Automatically loads model architecture, classes, and config from saved models.
    
    Configuration parameters (via block.metadata):
    - model_path: Path to PyTorch model file (.pth) created by PyTorch Audio Trainer (required)
    - sample_rate: Audio sample rate in Hz (default: from model config or 22050)
    - batch_size: Batch size for prediction (optional, None = auto)
    - device: Device to use ("cpu", "cuda", or "mps", default: "cpu")
    """
    
    def __init__(self):
        """Initialize processor"""
        if not HAS_PYTORCH:
            raise ImportError(
                "PyTorch is required for PyTorchAudioClassify. "
                "Install with: pip install torch"
            )
        self._model_cache = None
        self._cached_model_path = None
        self._cached_model_mtime = None
        self._model_config = None
        self._classes = None
        self._preprocessing_config = None
        self._is_binary = False
        self._target_class = None
        self._optimal_threshold = 0.5
        self._normalization = None
        self._classification_mode = "multiclass"  # "binary", "multiclass", or "positive_vs_other"

    # ------------------------------------------------------------------
    # Model lifecycle: flush / cleanup
    # ------------------------------------------------------------------

    def _flush_model(self) -> None:
        """
        Completely release the current PyTorch model and all related state.

        This performs a thorough cleanup of all GPU/MPS/CPU memory held by the
        classification model so that a new model can be loaded into a clean
        environment.  Safe to call even when no model is loaded.
        """
        import gc

        if self._model_cache is not None:
            # Move model to CPU before deletion to release GPU/MPS memory
            try:
                self._model_cache.cpu()
            except Exception:
                pass
            del self._model_cache
            self._model_cache = None

        # Clear all cached state tied to the old model
        self._cached_model_path = None
        self._cached_model_mtime = None
        self._model_config = None
        self._classes = None
        self._preprocessing_config = None
        self._is_binary = False
        self._target_class = None
        self._optimal_threshold = 0.5
        self._normalization = None
        self._classification_mode = "multiclass"

        # Force Python garbage collection so reference cycles are broken
        gc.collect()

        # Release GPU/MPS memory back to the driver
        if HAS_PYTORCH:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        Log.debug("PyTorchAudioClassifyBlockProcessor: Model flushed and memory released")

    def cleanup(self, block: Block) -> None:
        """
        Clean up all resources when the block is removed or the project is
        unloaded.  Delegates to ``_flush_model`` for the heavy lifting.
        """
        self._flush_model()
    
    def can_process(self, block: Block) -> bool:
        """Check if this processor can handle the block"""
        return block.type == "PyTorchAudioClassify"
    
    def get_block_type(self) -> str:
        """Get the block type this processor handles"""
        return "PyTorchAudioClassify"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for PyTorchAudioClassify block.
        
        Status levels:
        - Error (0): No model source (no model input connected AND no model_path set)
        - Warning (1): Events input not connected
        - Ready (2): All requirements met
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order
        """
        from src.features.blocks.domain import BlockStatusLevel
        import os
        
        def check_model_source(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if a model source is available (input port OR settings path)."""
            # Check if model input port is connected
            if hasattr(f, 'connection_service'):
                connections = f.connection_service.list_connections_by_block(blk.id)
                model_connected = any(
                    c.target_block_id == blk.id and c.target_input_name == "model"
                    for c in connections
                )
                if model_connected:
                    return True
            
            # Fall back to model_path in settings
            model_path = blk.metadata.get("model_path") if blk.metadata else None
            if model_path and os.path.exists(model_path):
                return True
            
            return False
        
        def check_events_input(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if events input is connected."""
            if not hasattr(f, 'connection_service'):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [c for c in connections if c.target_block_id == blk.id and c.target_input_name == "events"]
            return len(incoming) > 0
        
        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[check_model_source]
            ),
            BlockStatusLevel(
                priority=1,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[check_events_input]
            ),
            BlockStatusLevel(
                priority=2,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[]
            )
        ]
    
    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        Process PyTorchAudioClassify block.
        
        Supports both single EventDataItem and list of EventDataItems.
        For multiple event items, creates separate EventDataItem for each.
        
        Args:
            block: Block entity to process
            inputs: Input data items (should contain "events" EventDataItem or list)
            metadata: Optional metadata (not used currently)
            
        Returns:
            Dictionary with "events" port containing EventDataItem or list of EventDataItems
            
        Raises:
            ProcessingError: If events input missing or processing fails
        """
        # Get events input
        events_input = inputs.get("events")
        if not events_input:
            raise ProcessingError(
                "Events input required for PyTorchAudioClassify block. "
                "Connect an event source (e.g., DetectOnsets or Editor block) to the 'events' input port.",
                block_id=block.id,
                block_name=block.name
            )
        
        # Get audio input (optional - for extracting clips from source audio)
        # Audio port is NOT required; if not connected, we look up source audio
        # from event metadata (audio_id) via the data item repository.
        audio_input = inputs.get("audio")
        audio_items = []
        if audio_input:
            if isinstance(audio_input, list):
                audio_items = [item for item in audio_input if isinstance(item, AudioDataItem)]
            elif isinstance(audio_input, AudioDataItem):
                audio_items = [audio_input]
        
        # Fallback: look up source audio from event metadata references
        if not audio_items and events_input:
            audio_items = self._lookup_audio_from_events(events_input, metadata)
        
        # Handle both single event item and list of event items
        if isinstance(events_input, list):
            event_items = events_input
        elif isinstance(events_input, EventDataItem):
            event_items = [events_input]
        else:
            raise ProcessingError(
                f"Events input must be EventDataItem or list of EventDataItems, got {type(events_input)}",
                block_id=block.id,
                block_name=block.name
            )
        
        Log.info(f"PyTorchAudioClassifyBlockProcessor: Processing {len(event_items)} event item(s)")
        
        # Resolve model path: prefer model input port, fall back to block settings
        model_path = self._get_model_path(inputs, block)
        
        # Load model (cached if same path AND file hasn't been modified)
        current_mtime = os.path.getmtime(model_path)
        cache_valid = (
            self._cached_model_path == model_path
            and self._cached_model_mtime == current_mtime
        )
        if not cache_valid:
            # Determine reason for reload for logging
            if self._cached_model_path is None:
                Log.info(f"PyTorchAudioClassifyBlockProcessor: Loading model from {model_path}")
            elif self._cached_model_path != model_path:
                Log.info(
                    f"PyTorchAudioClassifyBlockProcessor: Model path changed "
                    f"({os.path.basename(self._cached_model_path)} -> {os.path.basename(model_path)}), "
                    f"flushing old model and loading new one"
                )
            else:
                Log.info(f"PyTorchAudioClassifyBlockProcessor: Model file changed on disk, reloading")

            # Flush old model completely before loading new one
            self._flush_model()

            self._load_model(model_path, block)
            self._cached_model_path = model_path
            self._cached_model_mtime = current_mtime
        else:
            Log.debug("PyTorchAudioClassifyBlockProcessor: Using cached model")
        
        # Use saved-model preprocessing only (aligned with trainer); do not override with block settings
        sample_rate = (
            self._preprocessing_config.get("sample_rate", 22050)
            if self._preprocessing_config
            else (block.metadata.get("sample_rate") or 22050)
        )
        
        batch_size = block.metadata.get("batch_size")
        device = block.metadata.get("device", "cpu")
        
        # Track execution summaries for all event items
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        
        all_execution_summaries = []
        all_output_items: List[EventDataItem] = []
        
        # Process each event item
        for item_idx, event_item in enumerate(event_items, 1):
            if not isinstance(event_item, EventDataItem):
                Log.warning(
                    f"PyTorchAudioClassifyBlockProcessor: Skipping non-EventDataItem input: {type(event_item)}"
                )
                continue
            
            input_events = event_item.get_events()
            event_count = len(input_events)
            
            # Initialize progress for this item
            item_message = f"Classifying {event_count} events from '{event_item.name}'"
            if len(event_items) > 1:
                item_message = f"Item {item_idx}/{len(event_items)}: {item_message}"
            
            if progress_tracker and event_count > 10:
                progress_tracker.start(item_message, total=event_count)
            elif len(event_items) > 1:
                Log.info(
                    f"PyTorchAudioClassifyBlockProcessor: Processing item {item_idx}/{len(event_items)}: "
                    f"{event_count} events from '{event_item.name}'"
                )
            else:
                Log.info(
                    f"PyTorchAudioClassifyBlockProcessor: Processing {event_count} events "
                    f"from '{event_item.name}'"
                )
            
            # Track processing time
            processing_start_time = time.time()
            
            try:
                # Classify events (pass tracker for incremental progress)
                meta = block.metadata or {}
                confidence_threshold = meta.get("confidence_threshold")
                multiclass_multi_label = meta.get("multiclass_multi_label", False)
                multiclass_confidence_threshold = meta.get("multiclass_confidence_threshold", 0.4)
                create_other_layer = meta.get("create_other_layer", True)
                output_events = self._classify_events(
                    event_item,
                    self._model_cache,
                    sample_rate,
                    batch_size,
                    device,
                    audio_items,
                    confidence_threshold=confidence_threshold,
                    multiclass_multi_label=multiclass_multi_label,
                    multiclass_confidence_threshold=multiclass_confidence_threshold,
                    create_other_layer=create_other_layer,
                    progress_tracker=progress_tracker if event_count > 10 else None,
                    progress_total=event_count,
                )
                
                processing_time = time.time() - processing_start_time
                
                # Complete progress for this item
                if progress_tracker and event_count > 10:
                    progress_tracker.complete(f"Classified {event_count} events")
                
                # Collect statistics from output events
                output_event_list = output_events.get_events()
                
                # Count classifications and collect confidence scores
                classification_counts = Counter()
                confidence_scores = []
                classified_count = 0
                skipped_count = 0
                
                for event in output_event_list:
                    classification = event.classification
                    if classification and classification != "unknown":
                        classification_counts[classification] += 1
                        classified_count += 1
                        
                        # Extract confidence if available
                        conf = event.metadata.get("classification_confidence")
                        if conf is not None:
                            try:
                                confidence_scores.append(float(conf))
                            except (ValueError, TypeError):
                                pass
                    else:
                        skipped_count += 1
                
                # Calculate confidence statistics
                confidence_stats = {}
                if confidence_scores:
                    confidence_stats = {
                        "min": float(min(confidence_scores)),
                        "max": float(max(confidence_scores)),
                        "avg": float(sum(confidence_scores) / len(confidence_scores))
                    }
                
                # Store execution summary statistics for this event item
                execution_summary = {
                    "event_item_name": event_item.name,
                    "event_item_id": event_item.id if hasattr(event_item, 'id') else None,
                    "events_input": event_count,
                    "events_output": len(output_event_list),
                    "events_classified": classified_count,
                    "events_skipped": skipped_count,
                    "processing_time_seconds": processing_time,
                    "events_per_second": event_count / processing_time if processing_time > 0 else 0,
                    "classification_counts": dict(classification_counts),
                    "confidence_stats": confidence_stats
                }
                
                all_execution_summaries.append(execution_summary)
                all_output_items.append(output_events)
                
                Log.info(
                    f"PyTorchAudioClassifyBlockProcessor: Classified {classified_count}/{event_count} events "
                    f"from '{event_item.name}' in {processing_time:.2f}s "
                    f"({event_count / processing_time:.1f} events/sec)"
                )
                
            except ProcessingError as e:
                Log.error(
                    f"PyTorchAudioClassifyBlockProcessor: Processing error for event item '{event_item.name}': {e.message}"
                )
                continue
            except Exception as e:
                error_msg = (
                    f"Failed to process event item '{event_item.name}': {str(e)}\n"
                    f"  - Event count: {event_count}\n"
                    f"  - Model path: {model_path}"
                )
                Log.error(f"PyTorchAudioClassifyBlockProcessor: {error_msg}")
                continue
        
        if not all_output_items:
            error_details = []
            if not event_items:
                error_details.append("No event items found in input")
            else:
                error_details.append(f"{len(event_items)} event item(s) provided but none processed successfully")
                error_details.append("Check that:")
                error_details.append("  - Events have associated audio data (from DetectOnsets with audio slicing)")
                error_details.append("  - Model path is valid and was created by PyTorch Audio Trainer")
                error_details.append("  - Model file contains required metadata (classes, config)")
            
            raise ProcessingError(
                "No event items processed successfully.\n" + "\n".join(error_details),
                block_id=block.id,
                block_name=block.name
            )
        
        # Store execution summary in block metadata for UI display
        if all_execution_summaries:
            total_events_input = sum(s["events_input"] for s in all_execution_summaries)
            total_events_output = sum(s["events_output"] for s in all_execution_summaries)
            total_classified = sum(s["events_classified"] for s in all_execution_summaries)
            total_skipped = sum(s["events_skipped"] for s in all_execution_summaries)
            total_processing_time = sum(s["processing_time_seconds"] for s in all_execution_summaries)
            
            # Aggregate classification counts
            all_classification_counts = Counter()
            for summary in all_execution_summaries:
                all_classification_counts.update(summary["classification_counts"])
            
            # Aggregate confidence stats
            all_confidence_scores = []
            for output_item in all_output_items:
                for event in output_item.get_events():
                    conf = event.metadata.get("classification_confidence")
                    if conf is not None:
                        try:
                            all_confidence_scores.append(float(conf))
                        except (ValueError, TypeError):
                            pass
            
            overall_confidence_stats = {}
            if all_confidence_scores:
                overall_confidence_stats = {
                    "min": float(min(all_confidence_scores)),
                    "max": float(max(all_confidence_scores)),
                    "avg": float(sum(all_confidence_scores) / len(all_confidence_scores))
                }

            overall_events_per_second = (
                total_events_input / total_processing_time
                if total_processing_time > 0 else 0.0
            )

            confidence_threshold = block.metadata.get("confidence_threshold") if block.metadata else None
            coach_feedback = build_inference_feedback(
                execution_summary={
                    "total_events_input": total_events_input,
                    "total_classified": total_classified,
                    "total_skipped": total_skipped,
                    "events_per_second": overall_events_per_second,
                    "confidence_stats": overall_confidence_stats,
                },
                confidence_threshold=confidence_threshold,
            )
            
            summary_data = {
                "last_execution": {
                    "timestamp": datetime.now().isoformat(),
                    "event_items_processed": len(all_execution_summaries),
                    "total_events_input": total_events_input,
                    "total_events_output": total_events_output,
                    "total_classified": total_classified,
                    "total_skipped": total_skipped,
                    "processing_time_seconds": total_processing_time,
                    "events_per_second": overall_events_per_second,
                    "model_path": model_path,
                    "classification_counts": dict(all_classification_counts),
                    "confidence_stats": overall_confidence_stats,
                    "coach_feedback": coach_feedback,
                    "details": all_execution_summaries
                }
            }
            # Merge with existing metadata
            if block.metadata:
                block.metadata.update(summary_data)
            else:
                block.metadata = summary_data
        
        # Return single item if only one, otherwise return list
        if len(all_output_items) == 1:
            return {"events": all_output_items[0]}
        else:
            Log.info(
                f"PyTorchAudioClassifyBlockProcessor: Created {len(all_output_items)} event items "
                f"from {len(event_items)} input items"
            )
            return {"events": all_output_items}
    
    def _get_model_path(self, inputs: Dict[str, DataItem], block: Block) -> str:
        """
        Resolve model path from input port or block settings.

        Priority:
        1. Model input port (from connected PyTorch Audio Trainer)
        2. model_path in block metadata/settings

        Returns a validated path to a .pth file (resolves model folder to .pth if needed).
        """
        model_path = None

        # Check model input port first (from connected Trainer block)
        model_input = inputs.get("model")
        if model_input:
            # Handle single item or list
            model_item = model_input[0] if isinstance(model_input, list) else model_input

            # Extract model_path from the DataItem
            if hasattr(model_item, 'file_path') and model_item.file_path:
                model_path = model_item.file_path
                Log.info(
                    f"PyTorchAudioClassifyBlockProcessor: Using model from connected trainer: "
                    f"{os.path.basename(model_path)}"
                )
            elif hasattr(model_item, 'metadata') and model_item.metadata:
                model_path = model_item.metadata.get("model_path")
                if model_path:
                    Log.info(
                        f"PyTorchAudioClassifyBlockProcessor: Using model path from input metadata: "
                        f"{os.path.basename(model_path)}"
                    )

        # Fall back to block settings
        if not model_path:
            model_path = block.metadata.get("model_path") if block.metadata else None

        # Validate we have a model path
        if not model_path:
            raise ProcessingError(
                "No model available. Either:\n"
                "  - Connect a PyTorch Audio Trainer's model output to this block's model input, or\n"
                "  - Set model_path in block settings.",
                block_id=block.id,
                block_name=block.name,
            )

        # model_path must be a string or path-like (e.g. from metadata or trainer output)
        if not isinstance(model_path, (str, bytes, os.PathLike)):
            raise ProcessingError(
                f"model_path must be a file or folder path, got {type(model_path).__name__}. "
                "Check block settings or the connected trainer output.",
                block_id=block.id,
                block_name=block.name,
            )
        model_path = os.fspath(model_path) if not isinstance(model_path, str) else model_path

        # If path is a model folder (contains .pth), use the .pth inside
        model_path = self._resolve_model_path(model_path, block)

        # Recover from stale/nonexistent configured path when we can determine
        # a single unambiguous candidate model file.
        model_path = self._recover_missing_model_path(model_path, block)

        # Validate model path exists
        if not os.path.exists(model_path):
            raise ProcessingError(
                f"Model path not found: {model_path}.\n"
                "Please check the model path.",
                block_id=block.id,
                block_name=block.name,
            )
        
        return model_path

    def _resolve_model_path(self, model_path: str, block: Block) -> str:
        """
        Resolve to a .pth file path. If model_path is a directory (model folder
        from the trainer), find the single .pth inside and return it.
        """
        path = Path(model_path)
        if path.is_dir():
            pth_files = list(path.glob("*.pth"))
            if len(pth_files) == 1:
                return str(pth_files[0])
            if len(pth_files) > 1:
                raise ProcessingError(
                    f"Model folder contains multiple .pth files: {path}. "
                    "Set model_path to the specific .pth file.",
                    block_id=block.id,
                    block_name=block.name,
                )
            raise ProcessingError(
                f"Model folder contains no .pth file: {path}.",
                block_id=block.id,
                block_name=block.name,
            )
        return model_path

    def _recover_missing_model_path(self, model_path: str, block: Block) -> str:
        """
        Best-effort recovery for stale manual paths.

        Handles common case where a configured *.pth path no longer exists but
        the parent folder (or configured folder) contains exactly one .pth file.
        """
        path = Path(model_path)
        if path.exists():
            return model_path

        candidate_dirs = []
        if path.suffix.lower() == ".pth":
            candidate_dirs.append(path.parent)
            if path.parent.parent.exists():
                candidate_dirs.append(path.parent.parent)
        else:
            candidate_dirs.append(path)
            if path.parent.exists():
                candidate_dirs.append(path.parent)

        for d in candidate_dirs:
            if not d or not d.exists() or not d.is_dir():
                continue
            direct = list(d.glob("*.pth"))
            if len(direct) == 1:
                recovered = str(direct[0])
                Log.warning(
                    "PyTorchAudioClassifyBlockProcessor: recovered missing model path "
                    f"'{model_path}' -> '{recovered}'"
                )
                return recovered
            if len(direct) == 0:
                nested = list(d.glob("*/*.pth"))
                if len(nested) == 1:
                    recovered = str(nested[0])
                    Log.warning(
                        "PyTorchAudioClassifyBlockProcessor: recovered missing model path "
                        f"'{model_path}' -> '{recovered}'"
                    )
                    return recovered

        return model_path

    def _load_model(self, model_path: str, block: Block):
        """
        Load PyTorch model created by PyTorch Audio Trainer.
        
        Models saved by PyTorch Audio Trainer contain:
        - model_state_dict: Model weights
        - classes: List of class names
        - config: Training configuration (architecture, preprocessing, etc.)
        - training_date: When model was trained
        """
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
        except Exception as e:
            raise ProcessingError(
                f"Failed to load model from {model_path}: {e}",
                block_id=block.id,
                block_name=block.name
            )
        
        # Check if this is a PyTorch Audio Trainer model
        if 'classes' not in checkpoint or 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
            raise ProcessingError(
                f"Model file {model_path} does not appear to be from PyTorch Audio Trainer. "
                "It must contain 'classes', 'config', and 'model_state_dict' keys.",
                block_id=block.id,
                block_name=block.name
            )
        
        # Extract model info
        self._classes = checkpoint['classes']
        self._model_config = checkpoint['config']
        model_state_dict = checkpoint['model_state_dict']
        
        Log.info(
            f"PyTorchAudioClassifyBlockProcessor: Loaded model with {len(self._classes)} classes: {self._classes}"
        )
        
        # Determine num_classes: binary models output 1 logit, multiclass outputs N
        is_binary = checkpoint.get("classification_mode", "multiclass") == "binary"
        num_classes = 1 if is_binary else len(self._classes)
        
        # Reconstruct model architecture from config
        # First, check if the state dict uses the legacy CNN format (conv1/conv2/conv3/fc1/fc2)
        # which predates the current nn.Sequential-based CNNClassifier
        legacy_keys = {"conv1.weight", "conv2.weight", "conv3.weight", "fc1.weight", "fc2.weight"}
        is_legacy_cnn = legacy_keys.issubset(set(model_state_dict.keys()))
        
        if is_legacy_cnn:
            Log.info("PyTorchAudioClassifyBlockProcessor: Detected legacy CNN model format, using compatibility loader")
            model = self._create_legacy_cnn_from_state_dict(model_state_dict, num_classes)
        else:
            model = self._create_model_from_config(self._model_config, num_classes)
        
        # Load state dict
        try:
            model.load_state_dict(model_state_dict)
        except Exception as e:
            raise ProcessingError(
                f"Failed to load model weights: {e}. "
                "Model architecture may not match saved weights.",
                block_id=block.id,
                block_name=block.name
            )
        
        # Set to evaluation mode
        device = block.metadata.get("device", "cpu")
        model = model.to(device)
        model.eval()
        
        self._model_cache = model
        
        # Preprocessing: use canonical inference_preprocessing from checkpoint so we stay aligned with trainer
        inference_prep = checkpoint.get("inference_preprocessing")
        if not isinstance(inference_prep, dict):
            raise ProcessingError(
                "Model is missing canonical inference_preprocessing metadata. "
                "Re-train the model with the current trainer so classifier preprocessing is fully aligned.",
                block_id=block.id,
                block_name=block.name,
            )

        self._preprocessing_config = {
            "sample_rate": inference_prep.get("sample_rate", 22050),
            "max_length": inference_prep.get("max_length", 22050),
            "n_fft": inference_prep.get("n_fft", 2048),
            "n_mels": inference_prep.get("n_mels", 128),
            "hop_length": inference_prep.get("hop_length", 512),
            "fmax": inference_prep.get("fmax", 8000),
            "audio_input_standard": inference_prep.get("audio_input_standard", {}),
        }
        if inference_prep.get("normalization_mean") is not None and inference_prep.get("normalization_std") is not None:
            self._normalization = {
                "mean": inference_prep["normalization_mean"],
                "std": inference_prep["normalization_std"],
            }
        else:
            self._normalization = checkpoint.get("normalization")
        Log.info(
            "PyTorchAudioClassifyBlockProcessor: Using saved-model preprocessing (aligned with trainer)"
        )
        
        # Classification mode: binary, multiclass, or positive_vs_other
        self._classification_mode = checkpoint.get("classification_mode", "multiclass")
        self._is_binary = self._classification_mode == "binary"
        self._target_class = checkpoint.get("target_class")
        self._optimal_threshold = checkpoint.get("optimal_threshold", 0.5)

        if self._is_binary:
            Log.info(
                f"PyTorchAudioClassifyBlockProcessor: Binary model for '{self._target_class}' "
                f"(threshold={self._optimal_threshold:.3f})"
            )
        elif self._classification_mode == "positive_vs_other":
            Log.info(
                f"PyTorchAudioClassifyBlockProcessor: Positive-vs-other model "
                f"({len(self._classes)} classes including 'other')"
            )
    
    def _create_model_from_config(self, config: Dict[str, Any], num_classes: int) -> nn.Module:
        """
        Recreate model architecture from training config.
        
        Uses the training.architectures module which supports all architecture
        types including ResNet, EfficientNet, and other transfer learning models.
        """
        from src.application.blocks.training.architectures import create_classifier
        
        # Support both old "model_architecture" key and new "model_type" key
        model_type = config.get("model_type", config.get("model_architecture", "cnn")).lower()
        config_with_type = {**config, "model_type": model_type}
        
        try:
            return create_classifier(num_classes, config_with_type)
        except ValueError as e:
            raise ProcessingError(
                f"Unsupported model architecture: {model_type}. Error: {e}"
            )
    
    def _create_legacy_cnn_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        num_classes: int
    ) -> nn.Module:
        """
        Reconstruct a legacy CNN model from its state dict keys/shapes.
        
        Legacy models used named layers (conv1, conv2, conv3, fc1, fc2) instead of
        the current nn.Sequential-based CNNClassifier. This method infers the
        architecture from the weight tensor shapes so old models can still be loaded.
        """
        # Infer conv layer shapes from weight tensors: shape is (out_ch, in_ch, kH, kW)
        conv1_out = state_dict["conv1.weight"].shape[0]
        conv2_out = state_dict["conv2.weight"].shape[0]
        conv3_out = state_dict["conv3.weight"].shape[0]
        kernel_size = state_dict["conv1.weight"].shape[2]
        
        fc1_in = state_dict["fc1.weight"].shape[1]
        fc1_out = state_dict["fc1.weight"].shape[0]
        
        class LegacyCNN(nn.Module):
            """Backward-compatible CNN matching the old conv1/conv2/conv3/fc1/fc2 layout."""
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=kernel_size, padding=kernel_size // 2)
                self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=kernel_size, padding=kernel_size // 2)
                self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size=kernel_size, padding=kernel_size // 2)
                self.fc1 = nn.Linear(fc1_in, fc1_out)
                self.fc2 = nn.Linear(fc1_out, num_classes)
            
            def forward(self, x):
                import torch.nn.functional as F
                x = F.max_pool2d(F.relu(self.conv1(x)), 2)
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = F.max_pool2d(F.relu(self.conv3(x)), 2)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, p=0.4, training=self.training)
                x = self.fc2(x)
                return x
        
        return LegacyCNN()
    
    def _lookup_audio_from_events(
        self,
        events_input: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> List[AudioDataItem]:
        """
        Look up source AudioDataItems from event metadata references.
        
        Events created by DetectOnsets store an 'audio_id' in their metadata
        that points to the source AudioDataItem. This method collects those IDs
        and fetches the corresponding AudioDataItems from the data item repository,
        so audio clips can be extracted for classification without requiring a
        direct audio input connection.
        
        Args:
            events_input: EventDataItem or list of EventDataItems
            metadata: Execution metadata (contains 'data_item_repo')
            
        Returns:
            List of AudioDataItems found, or empty list
        """
        if not metadata:
            return []
        
        data_item_repo = metadata.get("data_item_repo")
        if not data_item_repo:
            return []
        
        # Collect unique audio_id references from event metadata
        audio_ids = set()
        items = events_input if isinstance(events_input, list) else [events_input]
        for item in items:
            if not isinstance(item, EventDataItem):
                continue
            for event in item.get_events():
                audio_id = event.metadata.get("audio_id") if event.metadata else None
                if audio_id:
                    audio_ids.add(audio_id)
        
        if not audio_ids:
            return []
        
        # Fetch AudioDataItems from data store
        audio_items = []
        for audio_id in audio_ids:
            try:
                audio_item = data_item_repo.get(audio_id)
                if audio_item and isinstance(audio_item, AudioDataItem):
                    # Ensure audio data is loaded (lazy-loads from file if needed)
                    if audio_item.get_audio_data() is not None:
                        audio_items.append(audio_item)
                        Log.info(
                            f"PyTorchAudioClassifyBlockProcessor: Resolved source audio "
                            f"'{audio_item.name}' from event metadata"
                        )
                    else:
                        Log.warning(
                            f"PyTorchAudioClassifyBlockProcessor: Source audio '{audio_id}' "
                            f"found but has no audio data (file may be missing)"
                        )
            except Exception as e:
                Log.warning(
                    f"PyTorchAudioClassifyBlockProcessor: Failed to look up audio '{audio_id}': {e}"
                )
        
        return audio_items
    
    def _classify_events(
        self,
        events: EventDataItem,
        model: nn.Module,
        sample_rate: int,
        batch_size: Optional[int],
        device: str,
        audio_items: List[AudioDataItem],
        confidence_threshold: Optional[float] = None,
        multiclass_multi_label: bool = False,
        multiclass_confidence_threshold: float = 0.4,
        create_other_layer: bool = True,
        progress_tracker: Optional[Any] = None,
        progress_total: Optional[int] = None,
    ) -> EventDataItem:
        """
        Classify events using the PyTorch model.

        Args:
            events: Input EventDataItem with events to classify
            model: Loaded PyTorch model
            sample_rate: Audio sample rate
            batch_size: Batch size for prediction
            device: Device to use
            audio_items: Optional audio items for extracting event clips
            confidence_threshold: Optional confidence threshold override (binary mode)
            multiclass_multi_label: When True (multiclass only), create events for all
                classes above threshold (one input event can produce multiple output events)
            multiclass_confidence_threshold: Min probability to include a class when multi_label is True
            create_other_layer: When False, do not create an EventLayer for "other"; those events are dropped
            progress_tracker: Optional ProgressTracker for incremental progress
            progress_total: Total events (for progress); used when progress_tracker is set

        Returns:
            EventDataItem with classified events
        """
        if not HAS_LIBROSA:
            raise ProcessingError("librosa is required for audio preprocessing")
        
        input_events = events.get_events()
        if not input_events:
            return events
        
        # Determine batch size
        if batch_size is None:
            batch_size = min(32, len(input_events))
        
        # Preprocess events into model input format (report progress every N events)
        processed_features = []
        valid_event_indices = []
        progress_interval = max(1, (progress_total or len(input_events)) // 20)  # ~20 updates during preprocessing
        
        for idx, event in enumerate(input_events):
            try:
                # Extract audio clip for this event
                audio_clip = self._extract_event_audio(event, audio_items, sample_rate)
                if audio_clip is None:
                    continue
                
                # Preprocess audio to match training format
                feature = self._preprocess_audio(audio_clip, sample_rate)
                processed_features.append(feature)
                valid_event_indices.append(idx)
                
                # Incremental progress during preprocessing
                if progress_tracker and progress_total and len(processed_features) % progress_interval == 0:
                    progress_tracker.update(
                        current=len(processed_features),
                        total=progress_total,
                        message=f"Preparing {len(processed_features)}/{progress_total} events",
                    )
                
            except Exception as e:
                Log.warning(f"PyTorchAudioClassifyBlockProcessor: Failed to preprocess event {idx}: {e}")
                continue
        
        if not processed_features:
            Log.warning("PyTorchAudioClassifyBlockProcessor: No valid features extracted from events")
            return events
        
        # Mark preprocessing complete so progress bar reflects preparation phase done
        if progress_tracker and progress_total is not None:
            progress_tracker.update(
                current=progress_total,
                total=progress_total,
                message=f"Prepared {len(processed_features)} events",
            )
        
        # Convert to tensor
        features_tensor = torch.tensor(np.array(processed_features), dtype=torch.float32)
        features_tensor = features_tensor.to(device)
        
        # Add channel dimension if needed (for CNN)
        if len(features_tensor.shape) == 3:
            features_tensor = features_tensor.unsqueeze(1)  # (batch, 1, freq, time)
        
        # Extend progress total to include inference batches so bar moves during classification
        num_batches = (len(processed_features) + batch_size - 1) // batch_size
        if progress_tracker and progress_total is not None:
            progress_tracker.update(
                total=progress_total + num_batches,
                current=progress_total,
                message=f"Classifying {num_batches} batch(es)",
            )
        
        # Run inference in batches (report progress after each batch)
        all_predictions = []
        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, len(features_tensor), batch_size)):
                batch = features_tensor[i:i + batch_size]
                outputs = model(batch)
                
                if self._is_binary:
                    # Binary mode: sigmoid probability for target class
                    probs = torch.sigmoid(outputs.squeeze(-1)).cpu().numpy()
                    all_predictions.append(probs)
                else:
                    # Multiclass mode: softmax probabilities
                    probabilities = torch.softmax(outputs, dim=1)
                    all_predictions.append(probabilities.cpu().numpy())
                
                if progress_tracker and progress_total is not None:
                    progress_tracker.update(
                        current=progress_total + batch_idx + 1,
                        message=f"Classifying batch {batch_idx + 1}/{num_batches}",
                    )
        
        predictions = np.concatenate(all_predictions, axis=0)
        
        # Collect events by classification for EventLayers
        # Structure: EventDataItem -> EventLayers -> Events
        from collections import defaultdict
        events_by_classification = defaultdict(list)
        
        # Use explicit threshold override if provided, otherwise use model's optimal threshold
        threshold = self._optimal_threshold
        if confidence_threshold is not None:
            threshold = confidence_threshold
        
        for pred_idx, event_idx in enumerate(valid_event_indices):
            event = input_events[event_idx]

            if self._is_binary:
                # Binary classification: target_class vs "other"
                prob = float(predictions[pred_idx])
                if prob >= threshold:
                    class_name = self._target_class or "positive"
                    confidence = prob
                else:
                    class_name = "other"
                    confidence = 1.0 - prob
                prob_dict = {
                    self._target_class or "positive": prob,
                    "other": 1.0 - prob,
                }
                new_event = Event(
                    time=event.time,
                    classification=class_name,
                    duration=event.duration,
                    metadata={
                        **event.metadata,
                        "classified_by": "pytorch_audio_classify",
                        "classification_confidence": confidence,
                        "classification_mode": self._classification_mode,
                        "all_class_probabilities": prob_dict,
                    }
                )
                events_by_classification[class_name].append(new_event)
            else:
                # Multiclass classification
                prob_dict = {
                    self._classes[i]: float(predictions[pred_idx][i])
                    for i in range(len(self._classes))
                }
                if multiclass_multi_label:
                    # Create an event for each class that exceeds the threshold
                    for class_idx, class_name in enumerate(self._classes):
                        if class_idx >= len(predictions[pred_idx]):
                            continue
                        conf = float(predictions[pred_idx][class_idx])
                        if conf >= multiclass_confidence_threshold:
                            new_event = Event(
                                time=event.time,
                                classification=class_name,
                                duration=event.duration,
                                metadata={
                                    **event.metadata,
                                    "classified_by": "pytorch_audio_classify",
                                    "classification_confidence": conf,
                                    "classification_mode": self._classification_mode,
                                    "all_class_probabilities": prob_dict,
                                }
                            )
                            events_by_classification[class_name].append(new_event)
                else:
                    # Single-label: argmax, one event per input
                    class_idx = np.argmax(predictions[pred_idx])
                    confidence = float(predictions[pred_idx][class_idx])
                    class_name = self._classes[class_idx] if class_idx < len(self._classes) else "unknown"
                    new_event = Event(
                        time=event.time,
                        classification=class_name,
                        duration=event.duration,
                        metadata={
                            **event.metadata,
                            "classified_by": "pytorch_audio_classify",
                            "classification_confidence": confidence,
                            "classification_mode": self._classification_mode,
                            "all_class_probabilities": prob_dict,
                        }
                    )
                    events_by_classification[class_name].append(new_event)
        
        # Create EventLayers from grouped events
        layers = []
        for classification, layer_events in events_by_classification.items():
            if layer_events and (create_other_layer or classification != "other"):
                layer = EventLayer(
                    name=classification,
                    events=layer_events,
                    metadata={
                        "source": "pytorch_audio_classify",
                        "event_count": len(layer_events)
                    }
                )
                layers.append(layer)
        
        # Create output EventDataItem with EventLayers
        output_item = EventDataItem(
            id="",
            block_id=events.block_id if hasattr(events, 'block_id') else None,
            name=f"{events.name}_classified" if hasattr(events, 'name') else "classified_events",
            type="Event",
            created_at=datetime.now(),
            metadata={
                "classified_by": "pytorch_audio_classify",
                "classification_summary": {
                    "total": len(input_events),
                    "classified": len(valid_event_indices),
                    "skipped": len(input_events) - len(valid_event_indices)
                }
            },
            layers=layers  # SINGLE SOURCE OF TRUTH: EventLayers
        )
        
        Log.info(
            f"PyTorchAudioClassifyBlockProcessor: Classification complete - "
            f"Classified: {len(valid_event_indices)}/{len(input_events)} events, "
            f"Layers: {len(layers)}"
        )
        
        return output_item
    
    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Preprocess audio to match training format (mel spectrogram).

        Uses the same preprocessing as PyTorch Audio Trainer (from inference_preprocessing or config).
        """
        max_length = self._preprocessing_config.get("max_length", 22050)
        n_fft = self._preprocessing_config.get("n_fft", 2048)
        n_mels = self._preprocessing_config.get("n_mels", 128)
        hop_length = self._preprocessing_config.get("hop_length", 512)
        fmax = self._preprocessing_config.get("fmax", 8000)

        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)

        # Pad or truncate to max_length
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)), "constant")
        else:
            audio = audio[:max_length]

        # Compute mel spectrogram (n_fft must match trainer to align time/freq dimensions)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            fmax=fmax,
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize: use per-dataset stats if available, otherwise per-sample
        if self._normalization and self._normalization.get("mean") and self._normalization.get("std"):
            mean = np.array(self._normalization["mean"])
            std = np.array(self._normalization["std"])
            mel_spec_norm = (mel_spec_db - mean) / (std + 1e-10)
        else:
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
        
        return mel_spec_norm
    
    def _extract_event_audio(
        self,
        event: Event,
        audio_items: List[AudioDataItem],
        sample_rate: int
    ) -> Optional[np.ndarray]:
        """
        Extract audio clip for an event using clip_start_time and clip_end_time.
        
        Events MUST have clip_start_time and clip_end_time in their metadata
        (set by DetectOnsets). Returns None with a warning if missing.
        """
        # Require clip timing from event metadata
        if not event.metadata:
            return None
        
        start_time = event.metadata.get("clip_start_time")
        end_time = event.metadata.get("clip_end_time")
        
        if start_time is None or end_time is None:
            Log.warning(
                f"PyTorchAudioClassifyBlockProcessor: Event at {event.time:.3f}s missing "
                f"clip_start_time/clip_end_time. Events need clip end times for classification "
                f"(use DetectOnsets with clip end detection enabled)."
            )
            return None
        
        if not audio_items:
            return None
        
        # Match event to its source audio via audio_id
        event_audio_id = event.metadata.get("audio_id")
        
        for audio_item in audio_items:
            # If event specifies audio_id, only use the matching audio item
            if event_audio_id and hasattr(audio_item, 'id') and audio_item.id != event_audio_id:
                continue
            
            audio_array = audio_item.get_audio_data()
            if audio_array is None:
                continue
            
            audio_sr = getattr(audio_item, 'sample_rate', sample_rate) or sample_rate
            
            # Resample if needed
            if audio_sr != sample_rate:
                audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=sample_rate)
            
            # Ensure 1D for sample indexing (handle multi-channel)
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=0)
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            if start_sample < len(audio_array) and end_sample <= len(audio_array) and start_sample < end_sample:
                return audio_array[start_sample:end_sample]
        
        return None


"""
TensorFlowClassify Block Processor

Processes TensorFlowClassify blocks - classifies events using TensorFlow/Keras models.
Supports .h5, .keras, and SavedModel formats.

Uses simplified model loading approach matching ez_speedy DrumClassify implementation.
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
from src.shared.domain.entities import EventDataItem, Event
from src.shared.domain.entities import EventLayer
from src.shared.domain.entities import AudioDataItem
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade

# Try to import TensorFlow and librosa
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    Log.warning("TensorFlow not available - TensorFlowClassify will not work")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    Log.warning("librosa not available - audio preprocessing may be limited")

class TensorFlowClassifyBlockProcessor(BlockProcessor):
    """
    Processor for TensorFlowClassify block type.
    
    Classifies events using TensorFlow/Keras models.
    Takes EventDataItem as input and outputs EventDataItem with classifications.
    
    Configuration parameters (via block.metadata):
    - model_path: Path to TensorFlow model file (.h5, .keras) or SavedModel directory (required)
    - preprocessing_config: Optional JSON string with preprocessing parameters
    - sample_rate: Audio sample rate in Hz (default: 22050)
    - batch_size: Batch size for prediction (optional, None = auto)
    - min_confidence_percentage: Optional minimum confidence threshold (0-100).
        Events with confidence below this percentage will be filtered out (deleted).
        Example: 50.0 means only events with >= 50% confidence are kept.
    """
    
    def __init__(self):
        """Initialize processor"""
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required for TensorFlowClassify. "
                "Install with: pip install tensorflow"
            )
        self._model_cache = None
        self._cached_model_path = None
        self._model_type = None  # 'savedmodel' or 'keras'
        self._serving_function = None  # For SavedModel format
        self._detected_class_names = None  # Class names detected from model
    
    def can_process(self, block: Block) -> bool:
        """Check if this processor can handle the block"""
        return block.type == "TensorFlowClassify"
    
    def get_block_type(self) -> str:
        """Get the block type this processor handles"""
        return "TensorFlowClassify"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for TensorFlowClassify block.
        
        Status levels:
        - Error (0): Model path missing or invalid
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
        
        def check_model_path(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if model path is configured and exists."""
            model_path = blk.metadata.get("model_path")
            if not model_path:
                return False
            return os.path.exists(model_path)
        
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
                conditions=[check_model_path]
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
        Process TensorFlowClassify block.
        
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
                "Events input required for TensorFlowClassify block. "
                "Connect an event source (e.g., DetectOnsets or Editor block) to the 'events' input port.",
                block_id=block.id,
                block_name=block.name
            )
        
        # Get audio input (optional - for extracting clips from source audio)
        audio_input = inputs.get("audio")
        audio_items = []
        if audio_input:
            if isinstance(audio_input, list):
                audio_items = [item for item in audio_input if isinstance(item, AudioDataItem)]
            elif isinstance(audio_input, AudioDataItem):
                audio_items = [audio_input]
        
        # Fallback: If no audio input but we have events, try to look up audio from source blocks
        if not audio_items and events_input:
            audio_items = self._lookup_audio_from_events(events_input, metadata)
        
        # Handle both single event item and list of event items
        # NOTE: Inputs are already filtered by execution engine using filter_selections
        # No additional filtering should be applied here - rely on port-level filter only
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
        
        # Preview input EventDataItems for logging
        preview_info = self._preview_event_items(event_items)
        Log.info(
            f"TensorFlowClassifyBlockProcessor: Input preview - {preview_info['summary']}\n"
            f"  EventDataItem names: {preview_info['names']}\n"
            f"  Common metadata keys: {preview_info['common_metadata_keys'][:10]}{'...' if len(preview_info['common_metadata_keys']) > 10 else ''}\n"
            f"  Sample metadata values: {dict(list(preview_info['sample_metadata'].items())[:5])}"
        )
        
        if not event_items:
            Log.warning(
                "TensorFlowClassifyBlockProcessor: No event items provided. "
                "Check filter settings in block metadata (filter_selections)."
            )
            return {"events": []}
        
        Log.info(f"TensorFlowClassifyBlockProcessor: Processing {len(event_items)} event item(s)")
        if audio_items:
            Log.info(f"TensorFlowClassifyBlockProcessor: Found {len(audio_items)} audio source(s) for clip extraction")
        
        # Get model path from block metadata
        model_path = block.metadata.get("model_path")
        if not model_path:
            raise ProcessingError(
                "model_path required in block metadata for TensorFlowClassify block. "
                "Set model_path in block settings.",
                block_id=block.id,
                block_name=block.name
            )
        
        # Validate model path
        if not os.path.exists(model_path):
            raise ProcessingError(
                f"Model path not found: {model_path}. "
                "Please check the model_path in block settings.",
                block_id=block.id,
                block_name=block.name
            )
        
        # Load model (cached if same path)
        if self._cached_model_path != model_path:
            Log.info(f"TensorFlowClassifyBlockProcessor: Loading model from {model_path}")
            self._load_model(model_path)
            self._cached_model_path = model_path
            # Reset detected class names when loading new model
            self._detected_class_names = None
        else:
            Log.debug("TensorFlowClassifyBlockProcessor: Using cached model")
        
        # Get preprocessing config
        preprocessing_config = block.metadata.get("preprocessing_config")
        if preprocessing_config:
            try:
                import json
                preprocessing_config = json.loads(preprocessing_config) if isinstance(preprocessing_config, str) else preprocessing_config
            except Exception as e:
                Log.warning(f"TensorFlowClassifyBlockProcessor: Invalid preprocessing_config: {e}")
                preprocessing_config = None
        
        # Get other settings
        sample_rate = block.metadata.get("sample_rate", 22050)
        batch_size = block.metadata.get("batch_size")
        
        # Get confidence threshold (as percentage, e.g., 50 for 50%)
        min_confidence_percentage = block.metadata.get("min_confidence_percentage")
        if min_confidence_percentage is not None:
            try:
                min_confidence_percentage = float(min_confidence_percentage)
                if min_confidence_percentage < 0 or min_confidence_percentage > 100:
                    Log.warning(
                        f"TensorFlowClassifyBlockProcessor: min_confidence_percentage must be between 0 and 100, "
                        f"got {min_confidence_percentage}. Ignoring threshold."
                    )
                    min_confidence_percentage = None
            except (ValueError, TypeError):
                Log.warning(
                    f"TensorFlowClassifyBlockProcessor: Invalid min_confidence_percentage value: {min_confidence_percentage}. "
                    f"Ignoring threshold."
                )
                min_confidence_percentage = None
        
        # Get class names from block metadata (optional - overrides defaults)
        class_names = block.metadata.get("class_names")
        if class_names and isinstance(class_names, str):
            try:
                import json
                class_names = json.loads(class_names) if isinstance(class_names, str) else class_names
            except Exception:
                # If not JSON, try comma-separated
                class_names = [name.strip() for name in class_names.split(",")]
        
        # Add class_names to preprocessing_config if provided
        if class_names and isinstance(class_names, (list, tuple)):
            if preprocessing_config is None:
                preprocessing_config = {}
            preprocessing_config['class_names'] = class_names
        
        # Get progress tracker from metadata
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        
        # Track execution summaries for all event items
        all_execution_summaries = []
        all_output_items: List[EventDataItem] = []
        
        # Process each event item
        for item_idx, event_item in enumerate(event_items, 1):
            if not isinstance(event_item, EventDataItem):
                Log.warning(
                    f"TensorFlowClassifyBlockProcessor: Skipping non-EventDataItem input: {type(event_item)}"
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
                    f"TensorFlowClassifyBlockProcessor: Processing item {item_idx}/{len(event_items)}: "
                    f"{event_count} events from '{event_item.name}'"
                )
            else:
                Log.info(
                    f"TensorFlowClassifyBlockProcessor: Processing {event_count} events "
                    f"from '{event_item.name}'"
                )
            
            # Track processing time
            processing_start_time = time.time()
            
            try:
                # Classify events
                output_events = self._classify_events(
                    event_item,
                    self._model_cache,
                    preprocessing_config,
                    sample_rate,
                    batch_size,
                    audio_items,  # Pass audio items for clip extraction
                    metadata,  # Pass metadata for fallback audio lookup
                    min_confidence_percentage  # Pass confidence threshold
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
                
                # Calculate filtered count (events that were classified but filtered by confidence threshold)
                # Filtered events are not in output, so calculate as: input - output
                # All non-filtered events (classified, failed, or skipped) are in output
                filtered_count = max(0, event_count - len(output_event_list))
                
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
                    "events_filtered": filtered_count,
                    "processing_time_seconds": processing_time,
                    "events_per_second": event_count / processing_time if processing_time > 0 else 0,
                    "classification_counts": dict(classification_counts),
                    "confidence_stats": confidence_stats
                }
                
                all_execution_summaries.append(execution_summary)
                all_output_items.append(output_events)
                
                Log.info(
                    f"TensorFlowClassifyBlockProcessor: Classified {classified_count}/{event_count} events "
                    f"from '{event_item.name}' in {processing_time:.2f}s "
                    f"({event_count / processing_time:.1f} events/sec)"
                )
                
            except ProcessingError as e:
                Log.error(
                    f"TensorFlowClassifyBlockProcessor: Processing error for event item '{event_item.name}': {e.message}"
                )
                continue
            except Exception as e:
                error_msg = (
                    f"Failed to process event item '{event_item.name}': {str(e)}\n"
                    f"  - Event count: {event_count}\n"
                    f"  - Model path: {model_path}"
                )
                Log.error(f"TensorFlowClassifyBlockProcessor: {error_msg}")
                continue
        
        if not all_output_items:
            error_details = []
            if not event_items:
                error_details.append("No event items found in input")
            else:
                error_details.append(f"{len(event_items)} event item(s) provided but none processed successfully")
                error_details.append("Check that:")
                error_details.append("  - Events have associated audio data (from DetectOnsets with audio slicing)")
                error_details.append("  - Model path is valid and accessible")
                error_details.append("  - Model format is supported (.h5, .keras, or SavedModel)")
            
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
            total_filtered = sum(s.get("events_filtered", 0) for s in all_execution_summaries)
            
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
            
            summary_data = {
                "last_execution": {
                    "timestamp": datetime.now().isoformat(),
                    "event_items_processed": len(all_execution_summaries),
                    "total_events_input": total_events_input,
                    "total_events_output": total_events_output,
                    "total_classified": total_classified,
                    "total_skipped": total_skipped,
                    "total_filtered": total_filtered,
                    "model_path": model_path,
                    "classification_counts": dict(all_classification_counts),
                    "confidence_stats": overall_confidence_stats,
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
                f"TensorFlowClassifyBlockProcessor: Created {len(all_output_items)} event items "
                f"from {len(event_items)} input items"
            )
            return {"events": all_output_items}
    
    def _load_model(self, model_path: str):
        """
        Load TensorFlow model using simplified approach matching ez_speedy.
        
        For SavedModel format: uses tf.saved_model.load() directly and accesses
        model.signatures["serving_default"].
        
        For .h5/.keras files: uses tf.keras.models.load_model().
        
        Args:
            model_path: Path to model file or directory
            
        Raises:
            ProcessingError: If model loading fails
        """
        if not HAS_TENSORFLOW:
            raise ProcessingError(
                "TensorFlow is required. Install with: pip install tensorflow",
                block_id="",
                block_name=""
            )
        
        model_path_obj = Path(model_path)
        
        # Check if it's a SavedModel directory
        is_saved_model = model_path_obj.is_dir() and (model_path_obj / "saved_model.pb").exists()
        
        if is_saved_model:
            # Load SavedModel directly (matching ez_speedy approach)
            Log.info("TensorFlowClassifyBlockProcessor: Loading SavedModel format")
            try:
                self._model_cache = tf.saved_model.load(model_path)
                
                # Get serving function from signatures (matching ez_speedy)
                signature_keys = list(self._model_cache.signatures.keys())
                Log.info(f"TensorFlowClassifyBlockProcessor: Model signature keys: {signature_keys}")
                
                if "serving_default" not in signature_keys:
                    raise ProcessingError(
                        f"Model does not have a 'serving_default' signature. "
                        f"Available signatures: {signature_keys}",
                        block_id="",
                        block_name=""
                    )
                
                # Get the serving function (matching ez_speedy)
                self._serving_function = self._model_cache.signatures["serving_default"]
                self._model_type = "savedmodel"
                
                # Try to extract class names from SavedModel metadata if available
                # Some models store class names in the model's metadata or as attributes
                try:
                    # Check if model has class_names attribute
                    if hasattr(self._model_cache, 'class_names'):
                        self._detected_class_names = self._model_cache.class_names
                        Log.info(f"TensorFlowClassifyBlockProcessor: Found class_names in SavedModel: {self._detected_class_names}")
                    # Check if there's metadata with class names
                    elif hasattr(self._model_cache, 'metadata') and self._model_cache.metadata:
                        if 'class_names' in self._model_cache.metadata:
                            self._detected_class_names = self._model_cache.metadata['class_names']
                            Log.info(f"TensorFlowClassifyBlockProcessor: Found class_names in SavedModel metadata: {self._detected_class_names}")
                    
                    # Try to read from assets if available (some models store class names as text files)
                    if self._detected_class_names is None:
                        try:
                            assets_path = model_path_obj / "assets"
                            if assets_path.exists():
                                # Look for common class name file patterns
                                for pattern in ["class_names.txt", "classes.txt", "labels.txt"]:
                                    class_file = assets_path / pattern
                                    if class_file.exists():
                                        with open(class_file, 'r', encoding='utf-8') as f:
                                            self._detected_class_names = [line.strip() for line in f if line.strip()]
                                        Log.info(f"TensorFlowClassifyBlockProcessor: Found class_names in {pattern}: {self._detected_class_names}")
                                        break
                        except Exception as e:
                            Log.debug(f"TensorFlowClassifyBlockProcessor: Could not read class names from assets: {e}")
                    
                    # Try to read from variables (some models store class names as variables)
                    if self._detected_class_names is None:
                        try:
                            # Check if there's a variable with class names
                            if hasattr(self._model_cache, 'variables'):
                                for var in self._model_cache.variables:
                                    if 'class' in var.name.lower() and 'name' in var.name.lower():
                                        try:
                                            var_value = var.numpy()
                                            if isinstance(var_value, (list, tuple)) or (hasattr(var_value, 'shape') and len(var_value.shape) == 1):
                                                self._detected_class_names = [str(v) for v in var_value]
                                                Log.info(f"TensorFlowClassifyBlockProcessor: Found class_names in variable {var.name}: {self._detected_class_names}")
                                                break
                                        except Exception:
                                            pass
                        except Exception as e:
                            Log.debug(f"TensorFlowClassifyBlockProcessor: Could not read class names from variables: {e}")
                    
                    # Try to read from model directory (some models store class names as files in the model dir)
                    if self._detected_class_names is None:
                        try:
                            # Look for common class name file patterns in model directory
                            for pattern in ["class_names.txt", "classes.txt", "labels.txt", "class_names.json"]:
                                class_file = model_path_obj / pattern
                                if class_file.exists():
                                    if pattern.endswith('.json'):
                                        import json
                                        with open(class_file, 'r', encoding='utf-8') as f:
                                            self._detected_class_names = json.load(f)
                                    else:
                                        with open(class_file, 'r', encoding='utf-8') as f:
                                            self._detected_class_names = [line.strip() for line in f if line.strip()]
                                    Log.info(f"TensorFlowClassifyBlockProcessor: Found class_names in {pattern}: {self._detected_class_names}")
                                    break
                        except Exception as e:
                            Log.debug(f"TensorFlowClassifyBlockProcessor: Could not read class names from model directory: {e}")
                    
                    # If still no class names found, log a warning
                    if self._detected_class_names is None:
                        Log.warning(
                            f"TensorFlowClassifyBlockProcessor: Could not detect class names from SavedModel. "
                            f"If classifications are incorrect, please set 'class_names' in block metadata. "
                            f"Expected format: JSON array or comma-separated list, e.g., "
                            f'["Kick Drum", "Snare Drum", "Closed Hat Cymbal", "Open Hat Cymbal", "Clap Drum"]'
                        )
                except Exception as e:
                    Log.debug(f"TensorFlowClassifyBlockProcessor: Could not extract class names from SavedModel: {e}")
                
                Log.info("TensorFlowClassifyBlockProcessor: SavedModel loaded successfully")
            except Exception as e:
                raise ProcessingError(
                    f"Failed to load SavedModel: {str(e)}",
                    block_id="",
                    block_name=""
                ) from e
        else:
            # Load .h5 or .keras file
            Log.info("TensorFlowClassifyBlockProcessor: Loading Keras model file")
            try:
                self._model_cache = tf.keras.models.load_model(model_path, compile=False)
                self._serving_function = None
                self._model_type = "keras"
                
                # Try to extract class names from Keras model
                try:
                    if hasattr(self._model_cache, 'class_names'):
                        self._detected_class_names = self._model_cache.class_names
                        Log.info(f"TensorFlowClassifyBlockProcessor: Found class_names in Keras model: {self._detected_class_names}")
                    elif hasattr(self._model_cache, 'config') and isinstance(self._model_cache.config, dict):
                        if 'class_names' in self._model_cache.config:
                            self._detected_class_names = self._model_cache.config['class_names']
                            Log.info(f"TensorFlowClassifyBlockProcessor: Found class_names in Keras model config: {self._detected_class_names}")
                except Exception as e:
                    Log.debug(f"TensorFlowClassifyBlockProcessor: Could not extract class names from Keras model: {e}")
                
                Log.info("TensorFlowClassifyBlockProcessor: Keras model loaded successfully")
            except Exception as e:
                raise ProcessingError(
                    f"Failed to load Keras model: {str(e)}",
                    block_id="",
                    block_name=""
                ) from e
    
    def _classify_events(
        self,
        events: EventDataItem,
        model: Any,
        preprocessing_config: Optional[Dict[str, Any]],
        sample_rate: int,
        batch_size: Optional[int],
        audio_items: Optional[List[AudioDataItem]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        min_confidence_percentage: Optional[float] = None
    ) -> EventDataItem:
        """
        Classify events using the TensorFlow model.
        
        Args:
            events: Input EventDataItem with events to classify
            model: Loaded TensorFlow model
            preprocessing_config: Optional preprocessing configuration
            sample_rate: Audio sample rate
            batch_size: Batch size for prediction
            audio_items: Optional list of audio items for clip extraction
            metadata: Optional processing metadata
            min_confidence_percentage: Optional minimum confidence threshold (0-100).
                Events with confidence below this percentage will be filtered out.
            
        Returns:
            EventDataItem with classifications added
        """
        input_events = events.get_events()
        if not input_events:
            Log.warning("TensorFlowClassifyBlockProcessor: No events to classify")
            return events
        
        total_events = len(input_events)  # Define early for use in metadata
        Log.info(f"TensorFlowClassifyBlockProcessor: Classifying {total_events} events")
        
        # Get data_item_repo from metadata for fallback audio lookup
        data_item_repo = metadata.get("data_item_repo") if metadata else None
        
        # Create output EventDataItem with EventLayers (single source of truth)
        # Group events by classification into EventLayers
        from collections import defaultdict
        
        # Collect events by classification for grouping into layers
        events_by_classification = defaultdict(list)
        
        # Set output_name with "_classified" suffix to indicate classified events
        input_output_name = events.metadata.get('output_name')
        if input_output_name:
            # Parse and add "_classified" suffix
            from src.application.processing.output_name_helpers import parse_output_name, make_output_name
            try:
                port_name, item_name = parse_output_name(input_output_name)
                classified_item_name = f"{item_name}_classified"
                output_name = make_output_name(port_name, classified_item_name)
            except ValueError:
                # Invalid format, use default with classified suffix
                output_name = "events:main_classified"
        else:
            # Use default with classified suffix
            output_name = "events:main_classified"
        
        # Process each event
        classified_count = 0
        failed_count = 0
        skipped_count = 0
        filtered_count = 0  # Count events filtered by confidence threshold
        for idx, event in enumerate(input_events):
            audio_data = None
            sr = sample_rate
            
            
            
            # Strategy 1: Check if event has audio_path metadata (saved clip file)
            audio_path = (
                event.metadata.get("audio_path") or
                event.metadata.get("file_path") or
                event.metadata.get("audio_file")
            )
            
            
            
            if audio_path and os.path.exists(audio_path):
                # Load from saved clip file
                try:
                    if HAS_LIBROSA:
                        audio_data, sr = librosa.load(audio_path, sr=sample_rate)
                    else:
                        raise ProcessingError(
                            "librosa is required for audio preprocessing. "
                            "Install with: pip install librosa",
                            block_id="",
                            block_name=""
                        )
                except Exception as e:
                    Log.warning(
                        f"TensorFlowClassifyBlockProcessor: Failed to load audio from {audio_path}: {e}"
                    )
            
            # Strategy 2: Extract clip from source audio using event timing
            source_audio = None  # Initialize for use in Strategy 4
            if audio_data is None and audio_items:
                
                
                source_audio = self._find_source_audio(event, audio_items, data_item_repo)
                
                
                
                if source_audio and source_audio.file_path and os.path.exists(source_audio.file_path):
                    try:
                        # Extract audio clip using event timing (matching ez_speedy approach)
                        clip_start = event.time
                        clip_duration = event.duration if event.duration > 0 else 0.1  # Default 100ms if no duration
                        clip_end = clip_start + clip_duration
                        
                        if HAS_LIBROSA:
                            
                            
                            # Use librosa.load with offset and duration (efficient - doesn't load entire file)
                            audio_data, sr = librosa.load(
                                source_audio.file_path,
                                sr=sample_rate,
                                offset=clip_start,
                                duration=clip_duration,
                                mono=True
                            )
                            
                            
                            
                            Log.debug(
                                f"TensorFlowClassifyBlockProcessor: Extracted clip from {source_audio.file_path} "
                                f"({clip_start:.3f}s - {clip_end:.3f}s)"
                            )
                        else:
                            raise ProcessingError(
                                "librosa is required for audio preprocessing. "
                                "Install with: pip install librosa",
                                block_id="",
                                block_name=""
                            )
                    except Exception as e:
                        
                        Log.warning(
                            f"TensorFlowClassifyBlockProcessor: Failed to extract clip from source audio: {e}"
                        )
            
            # Strategy 3: If audio_items is empty or source_audio not found, use data_item_repo to lookup audio by audio_id or audio_name
            if audio_data is None and data_item_repo and event.metadata and source_audio is None:
                audio_id = event.metadata.get("audio_id")
                audio_name = event.metadata.get("audio_name")
                source_block_id = event.metadata.get("_original_source_block_id") or event.metadata.get("source_block_id")
                
                # Try lookup by audio_id first (most direct)
                if audio_id:
                    try:
                        audio_item = data_item_repo.get(audio_id)
                        if audio_item and isinstance(audio_item, AudioDataItem):
                            source_audio = audio_item
                            Log.debug(
                                f"TensorFlowClassifyBlockProcessor: Found audio by ID {audio_id} for event at {event.time}s"
                            )
                    except Exception as e:
                        Log.debug(
                            f"TensorFlowClassifyBlockProcessor: Failed to lookup audio by ID {audio_id}: {e}"
                        )
                
                # Try lookup by audio_name in source block
                if source_audio is None and audio_name and source_block_id:
                    try:
                        audio_item = data_item_repo.find_by_name(source_block_id, audio_name)
                        if audio_item and isinstance(audio_item, AudioDataItem):
                            source_audio = audio_item
                            Log.debug(
                                f"TensorFlowClassifyBlockProcessor: Found audio by name '{audio_name}' in block {source_block_id} for event at {event.time}s"
                            )
                    except Exception as e:
                        Log.debug(
                            f"TensorFlowClassifyBlockProcessor: Failed to lookup audio by name '{audio_name}' in block {source_block_id}: {e}"
                        )
                
                # If found, extract clip
                if source_audio and source_audio.file_path and os.path.exists(source_audio.file_path):
                    try:
                        clip_start = event.time
                        clip_duration = event.duration if event.duration > 0 else 0.1
                        
                        if HAS_LIBROSA:
                            audio_data, sr = librosa.load(
                                source_audio.file_path,
                                sr=sample_rate,
                                offset=clip_start,
                                duration=clip_duration,
                                mono=True
                            )
                            Log.debug(
                                f"TensorFlowClassifyBlockProcessor: Extracted clip via Strategy 3 from {source_audio.file_path} "
                                f"({clip_start:.3f}s - {clip_start + clip_duration:.3f}s)"
                            )
                    except Exception as e:
                        Log.warning(
                            f"TensorFlowClassifyBlockProcessor: Strategy 3 audio extraction failed: {e}"
                        )
            
            # Strategy 4: If all else fails and we have source audio, try loading entire file and extracting clip
            # This is a fallback if offset/duration loading failed but we know the source file
            if audio_data is None and source_audio and source_audio.file_path and os.path.exists(source_audio.file_path):
                try:
                    clip_start = event.time
                    clip_duration = event.duration if event.duration > 0 else 0.1
                    
                    if HAS_LIBROSA:
                        # Load entire file (less efficient but more robust)
                        full_audio, sr = librosa.load(
                            source_audio.file_path,
                            sr=sample_rate,
                            mono=True
                        )
                        
                        # Extract clip manually
                        start_sample = int(clip_start * sr)
                        end_sample = int((clip_start + clip_duration) * sr)
                        end_sample = min(end_sample, len(full_audio))
                        
                        if start_sample < len(full_audio):
                            audio_data = full_audio[start_sample:end_sample]
                            Log.debug(
                                f"TensorFlowClassifyBlockProcessor: Extracted clip via Strategy 4 (full file load) from {source_audio.file_path} "
                                f"({clip_start:.3f}s - {clip_start + clip_duration:.3f}s)"
                            )
                except Exception as e:
                    Log.warning(
                        f"TensorFlowClassifyBlockProcessor: Strategy 4 audio extraction failed: {e}"
                    )
            
            
            
            # Classify if we have audio data
            if audio_data is not None and len(audio_data) > 0:
                try:
                    # Preprocess audio
                    preprocessed = self._preprocess_audio(
                        audio_data,
                        sr,
                        preprocessing_config
                    )
                    
                    # Run prediction using simplified approach
                    predictions = self._predict(model, preprocessed)
                    
                    # Ensure predictions is 2D: (batch, classes)
                    if len(predictions.shape) == 1:
                        # Single prediction, add batch dimension
                        predictions = predictions.reshape(1, -1)
                    
                    # Get predicted class and confidence from first batch item
                    predicted_idx = int(np.argmax(predictions[0]))
                    confidence = float(predictions[0][predicted_idx])
                    
                    # Log prediction details for debugging class order issues (debug level to reduce console noise)
                    Log.debug(
                        f"TensorFlowClassifyBlockProcessor: Event at {event.time}s - "
                        f"predicted_idx={predicted_idx}, confidence={confidence:.3f}, "
                        f"num_classes={len(predictions[0])}"
                    )
                    
                    # Log all prediction probabilities with class names for debugging (debug level to reduce console noise)
                    if len(predictions[0]) <= 10:  # Only log if reasonable number of classes
                        # Try to get class names to show in log
                        class_names_for_log = []
                        for i in range(len(predictions[0])):
                            class_name = self._get_class_name(model, i, preprocessing_config)
                            if class_name:
                                class_names_for_log.append(f"{i}:{class_name}={predictions[0][i]:.3f}")
                            else:
                                class_names_for_log.append(f"{i}:class_{i}={predictions[0][i]:.3f}")
                        Log.debug(
                            f"TensorFlowClassifyBlockProcessor: All class probabilities: {', '.join(class_names_for_log)}"
                        )
                    
                    # Get class name - try multiple sources
                    classification = self._get_class_name(model, predicted_idx, preprocessing_config)
                    
                    # If still no class name, use generic
                    if not classification or classification.startswith("class_"):
                        classification = f"class_{predicted_idx}"
                        Log.warning(
                            f"TensorFlowClassifyBlockProcessor: No class name found for index {predicted_idx}, "
                            f"using generic '{classification}'. "
                            f"Consider setting class_names in block metadata or preprocessing_config."
                        )
                    
                    # Convert confidence to percentage
                    confidence_percentage = confidence * 100.0
                    
                    # Log individual classification result at debug level to reduce console noise
                    Log.debug(
                        f"TensorFlowClassifyBlockProcessor: Event at {event.time}s classified as '{classification}' "
                        f"(predicted_idx={predicted_idx}, confidence={confidence:.3f}, {confidence_percentage:.1f}%)"
                    )
                    
                    # Filter by confidence threshold if specified
                    if min_confidence_percentage is not None and confidence_percentage < min_confidence_percentage:
                        Log.debug(
                            f"TensorFlowClassifyBlockProcessor: Filtering event at {event.time}s - "
                            f"confidence {confidence_percentage:.1f}% below threshold {min_confidence_percentage:.1f}%"
                        )
                        filtered_count += 1
                        continue  # Skip adding this event
                    
                    # Group events by classification for EventLayers
                    classified_event = Event(
                        time=event.time,
                        classification=classification,
                        duration=event.duration,
                        metadata={
                            **event.metadata,
                            "classified_by": "tensorflow_classify",
                            "classification_confidence": confidence,
                            "classification_confidence_percentage": confidence_percentage,
                            "classification_timestamp": datetime.now().isoformat(),
                        }
                    )
                    events_by_classification[classification].append(classified_event)
                    classified_count += 1
                    
                except Exception as e:
                    Log.warning(
                        f"TensorFlowClassifyBlockProcessor: Failed to classify event at {event.time}s: {e}"
                    )
                    # Add event with error metadata - ensure it's still exported
                    error_classification = event.classification or "unknown"
                    error_event = Event(
                        time=event.time,
                        classification=error_classification,
                        duration=event.duration,
                        metadata={
                            **event.metadata,
                            "classified_by": "tensorflow_classify",
                            "classification_confidence": None,
                            "classification_timestamp": datetime.now().isoformat(),
                            "classification_error": str(e),
                            "classification_error_type": type(e).__name__
                        }
                    )
                    events_by_classification[error_classification].append(error_event)
                    failed_count += 1
            else:
                # No audio data - still export event with error metadata
                error_reason = "No audio data available for classification"
                Log.debug(
                    f"TensorFlowClassifyBlockProcessor: Event at {event.time}s has no audio data - "
                    f"exporting with error metadata"
                )
                skipped_classification = event.classification or "unknown"
                skipped_event = Event(
                    time=event.time,
                    classification=skipped_classification,
                    duration=event.duration,
                    metadata={
                        **event.metadata,
                        "classified_by": "tensorflow_classify",
                        "classification_confidence": None,
                        "classification_timestamp": datetime.now().isoformat(),
                        "classification_error": error_reason,
                        "classification_attempted": False
                    }
                )
                events_by_classification[skipped_classification].append(skipped_event)
                skipped_count += 1
        
        # Create EventLayers from grouped events (single source of truth)
        # Each classification becomes a layer (matches timeline layer names)
        layers = []
        for classification, layer_events in events_by_classification.items():
            if layer_events:  # Only create layers with events
                layer = EventLayer(
                    name=classification,  # Layer name = classification name (matches timeline)
                    events=layer_events,
                    metadata={
                        "source": "tensorflow_classify",
                        "event_count": len(layer_events)
                    }
                )
                layers.append(layer)
        
        # Create EventDataItem with EventLayers (single source of truth)
        output_events = EventDataItem(
            id="",
            block_id=events.block_id,
            name=f"{events.name}_classified",
            type="Event",
            metadata={
                "output_name": output_name,
                "classified_by": "tensorflow_classify",
                "classification_summary": {
                    "total": total_events,
                    "classified": classified_count,
                    "failed": failed_count,
                    "skipped": skipped_count,
                    "filtered": filtered_count
                }
            },
            layers=layers  # SINGLE SOURCE OF TRUTH: EventLayers
        )
        
        # Log comprehensive statistics
        log_message = (
            f"TensorFlowClassifyBlockProcessor: Classification complete - "
            f"Total: {total_events}, Classified: {classified_count}, "
            f"Failed: {failed_count}, Skipped: {skipped_count}, "
            f"Layers: {len(layers)}"
        )
        if filtered_count > 0:
            log_message += f", Filtered (low confidence): {filtered_count}"
        Log.info(log_message)
        
        return output_events
    
    def _get_class_name(self, model: Any, predicted_idx: int, preprocessing_config: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Get class name for predicted index.
        
        Tries multiple sources:
        1. Preprocessing config class_names (user-provided, highest priority)
        2. Model class_names attribute
        3. Model config class_names
        4. Default drum class mapping (if model appears to be a drum classifier)
        
        Normalizes class names to canonical forms (kick, snare, hh, ohh, clap)
        for consistent layer assignment.
        
        Args:
            model: Loaded TensorFlow model
            predicted_idx: Predicted class index
            preprocessing_config: Optional preprocessing config that may contain class names
            
        Returns:
            Class name string in canonical form (kick, snare, hh, ohh, clap) or None
        """
        # Strategy 1: Check preprocessing_config for class_names (user-provided, highest priority)
        if preprocessing_config and isinstance(preprocessing_config, dict):
            class_names = preprocessing_config.get('class_names')
            if isinstance(class_names, (list, tuple)) and predicted_idx < len(class_names):
                raw_name = str(class_names[predicted_idx])
                normalized = self._normalize_class_name(raw_name)
                if normalized:
                    Log.debug(f"TensorFlowClassifyBlockProcessor: Using class name from preprocessing_config: '{raw_name}' -> '{normalized}' (idx={predicted_idx})")
                    return normalized
        
        # Strategy 2: Check detected class names from model (SavedModel assets or Keras model)
        if self._detected_class_names and isinstance(self._detected_class_names, (list, tuple)) and predicted_idx < len(self._detected_class_names):
            raw_name = str(self._detected_class_names[predicted_idx])
            normalized = self._normalize_class_name(raw_name)
            if normalized:
                Log.debug(
                    f"TensorFlowClassifyBlockProcessor: Using class name from detected model class_names: "
                    f"'{raw_name}' -> '{normalized}' (idx={predicted_idx}). "
                    f"All class names in order: {list(self._detected_class_names)}"
                )
                return normalized
        
        # Strategy 2b: Check model.class_names attribute (fallback)
        if hasattr(model, 'class_names'):
            class_names = model.class_names
            if isinstance(class_names, (list, tuple)) and predicted_idx < len(class_names):
                raw_name = str(class_names[predicted_idx])
                normalized = self._normalize_class_name(raw_name)
                if normalized:
                    # Log all class names for debugging order issues
                    all_class_names = [str(n) for n in class_names] if isinstance(class_names, (list, tuple)) else []
                    Log.debug(
                        f"TensorFlowClassifyBlockProcessor: Using class name from model.class_names: "
                        f"'{raw_name}' -> '{normalized}' (idx={predicted_idx}). "
                        f"All class names in order: {all_class_names}"
                    )
                    return normalized
        
        # Strategy 3: Check model.config['class_names']
        if hasattr(model, 'config') and isinstance(model.config, dict):
            class_names = model.config.get('class_names')
            if isinstance(class_names, (list, tuple)) and predicted_idx < len(class_names):
                raw_name = str(class_names[predicted_idx])
                normalized = self._normalize_class_name(raw_name)
                if normalized:
                    # Log all class names for debugging order issues
                    all_class_names = [str(n) for n in class_names] if isinstance(class_names, (list, tuple)) else []
                    Log.debug(
                        f"TensorFlowClassifyBlockProcessor: Using class name from model.config: "
                        f"'{raw_name}' -> '{normalized}' (idx={predicted_idx}). "
                        f"All class names in order: {all_class_names}"
                    )
                    return normalized
        
        # Strategy 4: Default class mapping for 5-class drum models
        # Use the EXACT same class order as the old DrumClassify implementation (ez_speedy branch)
        # Reference: https://github.com/gdennen0/EchoZero/tree/ez_speedy/src/Project/Block/BlockTypes/Analyze/DrumClassify
        # The old implementation used: {0: "clap", 1: "hhc", 2: "kick", 3: "hho", 4: "snare"}
        # This is the class order the model was trained with
        drum_audio_classifier_classes = [
            "Clap Drum",           # Index 0 - was "clap" in old implementation
            "Closed Hat Cymbal",   # Index 1 - was "hhc" in old implementation
            "Kick Drum",           # Index 2 - was "kick" in old implementation
            "Open Hat Cymbal",     # Index 3 - was "hho" in old implementation
            "Snare Drum"           # Index 4 - was "snare" in old implementation
        ]
        
        # For any 5-class model, use this class order (matches drum-audio-classifier)
        if predicted_idx < len(drum_audio_classifier_classes):
            raw_name = drum_audio_classifier_classes[predicted_idx]
            normalized = self._normalize_class_name(raw_name)
            if normalized:
                Log.debug(
                    f"TensorFlowClassifyBlockProcessor: Using class '{raw_name}' -> '{normalized}' (idx={predicted_idx})"
                )
                return normalized
            # If normalization failed, return raw name
            return raw_name
        
        Log.warning(f"TensorFlowClassifyBlockProcessor: No class name found for predicted index {predicted_idx}")
        return None
    
    def _normalize_class_name(self, class_name: str) -> Optional[str]:
        """
        Normalize class name to canonical form for consistent layer assignment.
        
        Maps various class name formats to canonical names:
        - "Kick Drum", "Kick", "kick" -> "kick"
        - "Snare Drum", "Snare", "snare" -> "snare"
        - "Closed Hat Cymbal", "Closed Hat", "HiHat", "hihat", "hh" -> "hh"
        - "Open Hat Cymbal", "Open Hat", "Open HiHat", "ohh" -> "ohh"
        - "Clap Drum", "Clap", "clap" -> "clap"
        
        Args:
            class_name: Raw class name from model or config
            
        Returns:
            Normalized canonical class name or None if no match
        """
        if not class_name:
            return None
        
        class_name_lower = class_name.lower().strip()
        
        # Mapping of various formats to canonical names that normalize correctly in timeline widget
        # These names will be normalized by timeline widget's _normalize_layer_name:
        # - "kick" -> "Kick"
        # - "snare" -> "Snare"
        # - "hihat" -> "HiHat"
        # - "openhat" -> may normalize to "HiHat" (needs special handling for open hat)
        # - "clap" -> "Clap"
        # Order matters - check more specific patterns first
        normalization_map = {
            # Kick variations -> "kick" (normalizes to "Kick" in timeline)
            "kick drum": "kick",
            "kick": "kick",
            
            # Snare variations -> "snare" (normalizes to "Snare" in timeline)
            "snare drum": "snare",
            "snare": "snare",
            
            # Closed Hat variations -> "hihat" (normalizes to "HiHat" in timeline)
            "closed hat cymbal": "hihat",
            "closed hat": "hihat",
            "closed hi-hat": "hihat",
            "closed hihat": "hihat",
            "hi-hat": "hihat",
            "hihat": "hihat",
            "hi hat": "hihat",
            "hh": "hihat",  # Short form -> hihat for proper normalization
            "hat": "hihat",  # Generic hat -> hihat (assume closed)
            
            # Open Hat variations -> "openhat" (may normalize to "HiHat", needs separate layer)
            "open hat cymbal": "openhat",
            "open hat": "openhat",
            "open hi-hat": "openhat",
            "open hihat": "openhat",
            "open hi hat": "openhat",
            "ohh": "openhat",  # Short form -> openhat for proper normalization
            "openhat": "openhat",
            
            # Clap variations -> "clap" (normalizes to "Clap" in timeline)
            "clap drum": "clap",
            "clap": "clap",
        }
        
        # Check exact matches first
        if class_name_lower in normalization_map:
            return normalization_map[class_name_lower]
        
        # Check if any key is contained in the class name (for partial matches)
        for pattern, canonical in normalization_map.items():
            if pattern in class_name_lower:
                return canonical
        
        # If no match found, return None (will fall back to default mapping or generic)
        Log.debug(f"TensorFlowClassifyBlockProcessor: Could not normalize class name '{class_name}' - using as-is or falling back to default")
        return None
    
    def _lookup_audio_from_events(
        self,
        events_input: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> List[AudioDataItem]:
        """
        Look up audio items from source blocks using event metadata.
        
        When no audio input is connected, this tries to find audio items by:
        1. Looking up audio_name from event metadata in source blocks
        2. Looking up _original_source_item_name in source blocks
        
        Args:
            events_input: EventDataItem or list of EventDataItems
            metadata: Processing metadata (may contain data_item_repo)
            
        Returns:
            List of found AudioDataItem objects
        """
        audio_items = []
        data_item_repo = metadata.get("data_item_repo") if metadata else None
        
        
        
        if not data_item_repo:
            
            return audio_items
        
        # Get events to check metadata
        if isinstance(events_input, list):
            event_items = events_input
        elif isinstance(events_input, EventDataItem):
            event_items = [events_input]
        else:
            
            return audio_items
        
        # Collect unique audio names, audio IDs, and source block IDs from events
        audio_names_to_find = set()
        audio_ids_to_find = set()
        source_block_ids = set()
        
        for event_item in event_items:
            if not isinstance(event_item, EventDataItem):
                continue
            
            # Check EventDataItem metadata for source block ID (Editor sets this)
            if event_item.metadata:
                source_block_id = event_item.metadata.get("_source_block_id")
                if source_block_id:
                    source_block_ids.add(source_block_id)
            
            events = event_item.get_events()
            if not events:
                continue
            
            # Check first event for metadata (assuming all events from same source)
            first_event = events[0]
            if first_event.metadata:
                
                
                # Get audio_name
                audio_name = first_event.metadata.get("audio_name")
                if audio_name:
                    audio_names_to_find.add(audio_name)
                
                # Get audio_id (preferred - direct lookup)
                audio_id = first_event.metadata.get("audio_id")
                if audio_id:
                    audio_ids_to_find.add(audio_id)
                
                # Get source block ID from event metadata (if available)
                source_block_id = first_event.metadata.get("_original_source_block_id")
                if source_block_id:
                    source_block_ids.add(source_block_id)
                
                # Also check for source_block_id (DetectOnsets sets this)
                source_block_id_alt = first_event.metadata.get("source_block_id")
                if source_block_id_alt:
                    source_block_ids.add(source_block_id_alt)
            
            # Fallback: use the event item's block_id if no source block found
            if not source_block_ids and event_item.block_id:
                source_block_ids.add(event_item.block_id)
        
        
        
        # Try to find audio items by ID first (most direct)
        found_audio_items = {}
        for audio_id in audio_ids_to_find:
            try:
                
                audio_item = data_item_repo.get(audio_id)
                if audio_item and isinstance(audio_item, AudioDataItem):
                    found_audio_items[audio_item.name] = audio_item
            except Exception as e:
                
                continue
        
        # Try to find audio items by name in source blocks
        for block_id in source_block_ids:
            for audio_name in audio_names_to_find:
                if audio_name in found_audio_items:
                    continue  # Already found
                try:
                    
                    audio_item = data_item_repo.find_by_name(block_id, audio_name)
                    if audio_item and isinstance(audio_item, AudioDataItem):
                        found_audio_items[audio_item.name] = audio_item
                except Exception as e:
                    
                    continue
        
        # Fallback 1: search in event item's block_id if we have audio_name but no source_block_id
        if audio_names_to_find and not found_audio_items:
            for event_item in event_items:
                if isinstance(event_item, EventDataItem) and event_item.block_id:
                    for audio_name in audio_names_to_find:
                        try:
                            
                            audio_item = data_item_repo.find_by_name(event_item.block_id, audio_name)
                            if audio_item and isinstance(audio_item, AudioDataItem):
                                found_audio_items[audio_item.name] = audio_item
                                
                        except Exception as e:
                            
                            continue
        
        # Fallback 2: If we still haven't found audio and have audio_name, try searching all blocks in project
        if audio_names_to_find and not found_audio_items and metadata:
            project_id = metadata.get("project_id")
            block_repo = metadata.get("block_repo")
            if project_id and block_repo:
                try:
                    
                    all_blocks = block_repo.list_by_project(project_id)
                    for block in all_blocks:
                        for audio_name in audio_names_to_find:
                            if audio_name in found_audio_items:
                                continue
                            try:
                                audio_item = data_item_repo.find_by_name(block.id, audio_name)
                                if audio_item and isinstance(audio_item, AudioDataItem):
                                    found_audio_items[audio_item.name] = audio_item
                                    
                                    break  # Found in this block, move to next audio_name
                            except Exception as e:
                                continue
                except Exception as e:
                    
                    pass
        
        
        
        return list(found_audio_items.values())
    
    def _find_source_audio(
        self, 
        event: Any, 
        audio_items: List[AudioDataItem],
        data_item_repo: Optional[Any] = None
    ) -> Optional[AudioDataItem]:
        """
        Find source audio for an event by matching audio_name metadata.
        
        Uses multiple strategies:
        1. Match by audio_name in provided audio_items
        2. Match by audio_id using data_item_repo
        3. Match by audio_name in source block using data_item_repo
        4. Fallback to single audio item if only one available
        
        Args:
            event: Event to find source audio for
            audio_items: List of available AudioDataItem objects
            data_item_repo: Optional DataItemRepository for fallback lookups
            
        Returns:
            Matching AudioDataItem or None
        """
        if not event.metadata:
            return None
        
        
        
        # Strategy 1: Try to match by audio_name from event metadata in provided audio_items
        audio_name = event.metadata.get("audio_name")
        if audio_name and audio_items:
            for audio_item in audio_items:
                if audio_item.name == audio_name:
                    
                    return audio_item
        
        # Strategy 2: Try lookup by audio_id using data_item_repo (most direct)
        audio_id = event.metadata.get("audio_id")
        if audio_id and data_item_repo:
            try:
                audio_item = data_item_repo.get(audio_id)
                if audio_item and isinstance(audio_item, AudioDataItem):
                    
                    return audio_item
            except Exception as e:
                Log.debug(f"TensorFlowClassifyBlockProcessor: Failed to lookup audio by ID {audio_id}: {e}")
        
        # Strategy 3: Try lookup by audio_name in source block using data_item_repo
        source_block_id = event.metadata.get("_original_source_block_id") or event.metadata.get("source_block_id")
        if audio_name and source_block_id and data_item_repo:
            try:
                audio_item = data_item_repo.find_by_name(source_block_id, audio_name)
                if audio_item and isinstance(audio_item, AudioDataItem):
                    
                    return audio_item
            except Exception as e:
                Log.debug(f"TensorFlowClassifyBlockProcessor: Failed to lookup audio by name '{audio_name}' in block {source_block_id}: {e}")
        
        if not audio_items:
            return None
        
        # Try to match by audio_name from event metadata
        audio_name = event.metadata.get("audio_name")
        if audio_name:
            for audio_item in audio_items:
                if audio_item.name == audio_name:
                    
                    return audio_item
        
        # Fallback 4: use first audio item if only one
        if len(audio_items) == 1:
            
            return audio_items[0]
        
        # Fallback 5: try matching by source item name
        source_item_name = event.metadata.get("_original_source_item_name")
        if source_item_name:
            for audio_item in audio_items:
                if source_item_name in audio_item.name or audio_item.name in source_item_name:
                    
                    return audio_item
        
        
        return None
    
    def _preview_event_items(self, event_items: List[EventDataItem]) -> Dict[str, Any]:
        """
        Preview and analyze input EventDataItems to help with filtering configuration.
        
        Returns a summary of EventDataItem properties including:
        - Names
        - Common metadata keys
        - Sample metadata values
        
        Args:
            event_items: List of EventDataItems to analyze
            
        Returns:
            Dictionary with preview information
        """
        if not event_items:
            return {
                "summary": "No EventDataItems to preview",
                "names": [],
                "common_metadata_keys": [],
                "sample_metadata": {}
            }
        
        # Collect names
        names = [item.name for item in event_items if item.name]
        
        # Collect all metadata keys from EventDataItems
        all_metadata_keys = set()
        sample_metadata = {}
        
        for item in event_items:
            if item.metadata:
                all_metadata_keys.update(item.metadata.keys())
                # Store sample values for common keys (first non-empty value)
                for key, value in item.metadata.items():
                    if key not in sample_metadata and value:
                        # Convert value to string for preview, truncate if too long
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                        sample_metadata[key] = value_str
        
        # Also check event-level metadata (from first event of first item)
        event_metadata_keys = set()
        if event_items and event_items[0].get_events():
            first_event = event_items[0].get_events()[0]
            if first_event.metadata:
                event_metadata_keys.update(first_event.metadata.keys())
                # Add sample values from event metadata
                for key, value in first_event.metadata.items():
                    if key not in sample_metadata and value:
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                        sample_metadata[key] = value_str
        
        # Combine metadata keys
        all_metadata_keys.update(event_metadata_keys)
        
        # Count events per item
        event_counts = [len(item.get_events()) if hasattr(item, 'get_events') else 0 for item in event_items]
        total_events = sum(event_counts)
        
        summary = (
            f"{len(event_items)} EventDataItem(s), {total_events} total events "
            f"(avg {total_events / len(event_items):.1f} per item)"
        )
        
        return {
            "summary": summary,
            "names": names,
            "common_metadata_keys": sorted(list(all_metadata_keys)),
            "sample_metadata": sample_metadata,
            "event_counts": event_counts
        }
    
    def _predict(self, model: Any, input_data: np.ndarray) -> np.ndarray:
        """
        Run prediction on model using simplified approach matching ez_speedy.
        
        For SavedModel: uses serving function directly.
        For Keras model: uses model.predict().
        
        Args:
            model: Loaded TensorFlow model
            input_data: Preprocessed input data
            
        Returns:
            Prediction results as numpy array
        """
        # Ensure input is float32 (matching ez_speedy)
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        if self._model_type == "savedmodel" and self._serving_function is not None:
            # Use serving function directly (matching ez_speedy)
            try:
                # Convert to TensorFlow tensor
                input_tensor = tf.convert_to_tensor(input_data)
                
                # Call serving function
                prediction = self._serving_function(input_tensor)
                
                # Extract prediction array from output dict
                # The output is typically a dict with keys like 'dense_3' or similar
                prediction_key = list(prediction.keys())[0]
                prediction_array = prediction[prediction_key].numpy()
                
                return prediction_array
            except Exception as e:
                raise ProcessingError(
                    f"Error during SavedModel prediction: {str(e)}",
                    block_id="",
                    block_name=""
                ) from e
        else:
            # Use Keras model.predict()
            try:
                predictions = model.predict(
                    input_data,
                    verbose=0,
                    batch_size=1  # Use batch_size=1 for simplicity
                )
                
                # Ensure output is numpy array
                if hasattr(predictions, 'numpy'):
                    predictions = predictions.numpy()
                elif isinstance(predictions, tf.Tensor):
                    predictions = predictions.numpy()
                
                return predictions
            except Exception as e:
                raise ProcessingError(
                    f"Error during Keras model prediction: {str(e)}",
                    block_id="",
                    block_name=""
                ) from e
    
    def _preprocess_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        preprocessing_config: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Preprocess audio for model input.
        
        Matches ez_speedy DrumClassify approach for compatibility:
        - Resamples to 22050Hz
        - Trims silence
        - Generates mel spectrogram
        - Creates fixed 128x100x3 array (matching drum classifier model)
        
        Args:
            audio_data: Audio waveform
            sample_rate: Sample rate
            preprocessing_config: Optional preprocessing configuration
            
        Returns:
            Preprocessed array ready for model input (shape: 1, 128, 100, 3)
        """
        if not HAS_LIBROSA:
            raise ProcessingError(
                "librosa is required for audio preprocessing. "
                "Install with: pip install librosa",
                block_id="",
                block_name=""
            )
        
        # Use preprocessing config if provided, otherwise use defaults
        if preprocessing_config:
            preprocess_type = preprocessing_config.get("type", "mel_spectrogram")
            target_length = preprocessing_config.get("target_length", 100)  # Time steps
            n_mels = preprocessing_config.get("n_mels", 128)
            target_sr = preprocessing_config.get("target_sr", 22050)
        else:
            # Default preprocessing: mel spectrogram matching ez_speedy
            preprocess_type = "mel_spectrogram"
            target_length = 100  # Fixed length matching ez_speedy
            n_mels = 128
            target_sr = 22050
        
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=0)
        else:
            audio_mono = audio_data
        
        if preprocess_type == "mel_spectrogram":
            # Resample to target sample rate (matching ez_speedy)
            if sample_rate != target_sr:
                audio_mono = librosa.resample(audio_mono, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr
            
            # Trim silence (matching ez_speedy)
            audio_mono, _ = librosa.effects.trim(audio_mono, top_db=50)
            
            # Calculate appropriate n_fft based on audio length to avoid warnings
            # Use largest power of 2 <= audio length, with minimum of 512
            audio_len = len(audio_mono)
            if audio_len < 512:
                # Pad very short audio to minimum length
                audio_mono = np.pad(audio_mono, (0, 512 - audio_len), mode='constant')
                audio_len = 512
            
            # Find largest power of 2 <= audio_len
            n_fft = 512
            while n_fft * 2 <= audio_len:
                n_fft *= 2
            
            # Generate mel spectrogram with appropriate n_fft to avoid warnings
            mel_spec = librosa.feature.melspectrogram(
                y=audio_mono,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=n_fft // 4,  # Standard hop_length is n_fft/4
                n_mels=n_mels
            )
            
            # Create fixed-size array matching ez_speedy approach (128x100x3)
            # This matches the drum classifier model's expected input shape
            sample = np.zeros((n_mels, target_length, 3), dtype=np.float32)
            
            # Fill array with mel spectrogram values (matching ez_speedy)
            # Copy the same value to all 3 channels (RGB)
            for i in range(min(n_mels, mel_spec.shape[0])):
                for j in range(min(target_length, mel_spec.shape[1])):
                    value = float(mel_spec[i][j])  # Ensure it's a float
                    sample[i][j][0] = value  # R channel
                    sample[i][j][1] = value  # G channel
                    sample[i][j][2] = value  # B channel
            
            # Add batch dimension: (1, 128, 100, 3)
            sample_batch = np.expand_dims(sample, axis=0).astype(np.float32)
            
            return sample_batch
        else:
            # For other preprocessing types, return raw audio with batch dimension
            return np.expand_dims(audio_mono, axis=0)

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate TensorFlowClassify block configuration before execution.

        Validates filter selections to ensure upstream data types match expected types.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Validate model path
        model_path = block.metadata.get("model_path")
        if not model_path:
            errors.append(
                f"Block '{block.name}' (type: {block.type}): "
                "model_path required in block metadata. Set model_path in block settings."
            )

        # Validate model path exists
        elif not os.path.exists(model_path):
            errors.append(
                f"Block '{block.name}' (type: {block.type}): "
                f"Model path not found: {model_path}. Please check the model_path in block settings."
            )

        # Validate filter selections if they exist
        if data_item_repo and connection_repo and block_registry:
            filter_selections = block.metadata.get("filter_selections", {})
            if filter_selections:
                # Get block type metadata to check expected input types
                block_metadata = block_registry.get(block.type)
                if block_metadata:
                    expected_inputs = block_metadata.inputs

                    # Build a map of connections by target (target_block_id, target_input_name) -> list of connections
                    connections_by_target = {}
                    all_connections = connection_repo.list_by_block(block.id)
                    for conn in all_connections:
                        if conn.target_block_id == block.id:
                            key = (conn.target_block_id, conn.target_input_name)
                            if key not in connections_by_target:
                                connections_by_target[key] = []
                            connections_by_target[key].append(conn)

                    # For each port with a filter selection
                    for port_name, selected_ids in filter_selections.items():
                        if not selected_ids or not isinstance(selected_ids, list):
                            continue

                        # Find connections to this input port
                        key = (block.id, port_name)
                        port_connections = connections_by_target.get(key, [])

                        if not port_connections:
                            # No connection for this port - skip validation
                            continue

                        # Get expected input type for this port
                        expected_type = expected_inputs.get(port_name)
                        if not expected_type:
                            # Port not declared in block registry - skip validation
                            continue

                        # Check each connection to this port
                        for conn in port_connections:
                            # Get current data items from source block's output port
                            source_items = data_item_repo.list_by_block(conn.source_block_id)
                            matching_items = [
                                item for item in source_items
                                if item.metadata.get('output_port') == conn.source_output_name
                            ]

                            if not matching_items:
                                # No data items available - can't validate types
                                # This is okay - it just means source hasn't executed yet
                                continue

                            # Validate that data item types match expected type
                            # We check the first item's type (all items on same port should have same type)
                            sample_item = matching_items[0]
                            actual_type_name = sample_item.type if hasattr(sample_item, 'type') else None

                            if not actual_type_name:
                                errors.append(
                                    f"Block '{block.name}' (type: {block.type}): "
                                    f"Cannot determine type of upstream data item from "
                                    f"'{conn.source_block_id}.{conn.source_output_name}' for input '{port_name}'"
                                )
                                continue

                            from src.shared.domain.value_objects.port_type import get_port_type
                            try:
                                actual_type = get_port_type(actual_type_name)
                            except Exception as e:
                                errors.append(
                                    f"Block '{block.name}' (type: {block.type}): "
                                    f"Invalid type '{actual_type_name}' from "
                                    f"'{conn.source_block_id}.{conn.source_output_name}' for input '{port_name}': {e}"
                                )
                                continue

                            from src.application.processing.type_validation import types_compatible
                            if not types_compatible(actual_type, expected_type):
                                errors.append(
                                    f"Block '{block.name}' (type: {block.type}): "
                                    f"Filter selection for input '{port_name}' references data of type '{actual_type_name}' "
                                    f"from '{conn.source_block_id}.{conn.source_output_name}', "
                                    f"but expected type '{expected_type.name}'. "
                                    f"Filter selections are invalid for this type mismatch."
                                )

        return errors
    
    def cleanup(self, block: Block) -> None:
        """
        Clean up TensorFlow model and session.
        
        Clears model cache and frees GPU memory if using TensorFlow.
        """
        # Clear model cache
        if self._model_cache is not None:
            self._model_cache = None
            self._cached_model_path = None
            self._model_type = None
            self._serving_function = None
            Log.debug(f"TensorFlowClassifyBlockProcessor: Cleared model cache for {block.name}")
        
        # Clear TensorFlow/Keras session if available
        try:
            import tensorflow as tf
            if hasattr(tf.keras.backend, 'clear_session'):
                tf.keras.backend.clear_session()
                Log.debug(f"TensorFlowClassifyBlockProcessor: Cleared Keras session for {block.name}")
        except ImportError:
            pass  # TensorFlow not available
        except Exception as e:
            Log.warning(f"TensorFlowClassifyBlockProcessor: Failed to clear TensorFlow session: {e}")


# Auto-register this processor class
register_processor_class(TensorFlowClassifyBlockProcessor)


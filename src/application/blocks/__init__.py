"""
Block Processors Package

Auto-registration of all block processors.
Import this module to register all available processors with the execution engine.
"""
from typing import Optional

from src.application.processing.block_processor import BlockProcessor
from src.features.execution.application import BlockExecutionEngine
from src.utils.message import Log
from src.application.bootstrap_loading_progress import LoadingProgressTracker


# Registry of processor classes - populated by individual processor modules
_PROCESSOR_CLASSES: list[type[BlockProcessor]] = []


def register_processor_class(processor_class: type[BlockProcessor]) -> None:
    """
    Register a processor class for auto-registration.
    
    Each processor module should call this at module level to register itself.
    
    Args:
        processor_class: BlockProcessor subclass to register
    """
    if processor_class not in _PROCESSOR_CLASSES:
        _PROCESSOR_CLASSES.append(processor_class)
        Log.debug(f"Registered processor class: {processor_class.__name__}")


def register_all_processors(
    execution_engine: BlockExecutionEngine,
    progress_tracker: Optional[LoadingProgressTracker] = None
) -> None:
    """
    Register all available block processors with the execution engine.
    
    This function instantiates and registers all processor classes that have
    been registered via register_processor_class().
    
    Args:
        execution_engine: BlockExecutionEngine instance to register processors with
        progress_tracker: Optional progress tracker for reporting registration progress
    """
    if not execution_engine:
        Log.warning("No execution engine provided - cannot register processors")
        return
    
    # Import all processor modules to trigger their registration
    # This ensures all processors are registered before we instantiate them
    try:
        from . import load_audio_block
        from . import setlist_audio_input_block
        from . import detect_onsets_block
        from . import learned_onset_detector_block
        from . import tensorflow_classify_block
        from . import pytorch_audio_trainer_block
        from . import learned_onset_trainer_block
        from . import pytorch_audio_classify_block
        from . import separator_block
        # Import cloud blocks separately so one failure doesn't prevent the other
        try:
            from . import separator_cloud_block
        except ImportError as e:
            Log.debug(f"separator_cloud_block not available: {e}")
        from . import editor_block
        from . import export_audio_block
        from . import export_clips_by_class_block
        from . import note_extractor_basicpitch_block
        from . import note_extractor_librosa_block
        from . import plot_events_block
        from . import audio_filter_block
        from . import audio_negate_block
        from . import show_manager_block
        from . import audio_player_block
        from . import eq_bands_block
        from . import export_audio_dataset_block
        from . import dataset_viewer_block
    except ImportError as e:
        Log.debug(f"Some processor modules not available: {e}")
        # Continue - not all processors may be implemented yet
    
    # Register all processor instances
    registered_count = 0
    failed_processors = []
    total_processors = len(_PROCESSOR_CLASSES)
    
    for idx, processor_class in enumerate(_PROCESSOR_CLASSES, 1):
        try:
            # Get block type name for progress reporting
            processor_instance = processor_class()
            block_type = processor_instance.get_block_type()
            
            # Report progress if tracker available
            if progress_tracker:
                progress_tracker.update_step(f"Registering {block_type} processor", step_number=idx)
                # Process Qt events to update splash screen
                try:
                    from PyQt6.QtWidgets import QApplication
                    app = QApplication.instance()
                    if app:
                        app.processEvents()
                except ImportError:
                    pass
            
            execution_engine.register_processor(processor_instance)
            registered_count += 1
        except TypeError as e:
            # Handle abstract method errors more clearly
            if "abstract" in str(e).lower():
                error_msg = (
                    f"Processor '{processor_class.__name__}' is missing required methods.\n"
                    f"  All BlockProcessor subclasses must implement:\n"
                    f"    - get_block_type() -> str\n"
                    f"    - can_process(block: Block) -> bool\n"
                    f"    - process(block: Block, inputs: Dict, metadata: Dict) -> Dict\n"
                    f"  Error: {e}"
                )
                Log.error(error_msg)
                failed_processors.append(processor_class.__name__)
            else:
                Log.error(f"Failed to instantiate processor {processor_class.__name__}: {e}")
                failed_processors.append(processor_class.__name__)
        except Exception as e:
            Log.error(f"Failed to instantiate processor {processor_class.__name__}: {e}")
            failed_processors.append(processor_class.__name__)
    
    if failed_processors:
        Log.warning(f"Failed to register {len(failed_processors)} processor(s): {', '.join(failed_processors)}")
    
    Log.info(f"Auto-registered {registered_count}/{total_processors} block processors")


def get_registered_processors() -> list[type[BlockProcessor]]:
    """
    Get list of registered processor classes.
    
    Returns:
        List of BlockProcessor subclasses that have been registered
    """
    return list(_PROCESSOR_CLASSES)


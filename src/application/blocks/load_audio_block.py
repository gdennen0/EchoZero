"""
LoadAudio Block Processor

Processes LoadAudio blocks - loads audio files into the project.
"""
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.entities import AudioDataItem
from src.application.blocks import register_processor_class
from src.features.execution.application.progress_helpers import (
    progress_scope, yield_progress, get_progress_tracker
)
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


class LoadAudioBlockProcessor(BlockProcessor):
    """
    Processor for LoadAudio block type.
    
    Loads audio files from disk and creates AudioDataItem outputs.
    """
    
    def can_process(self, block: Block) -> bool:
        """Check if this processor can handle the block"""
        return block.type == "LoadAudio"
    
    def get_block_type(self) -> str:
        """Get the block type this processor handles"""
        return "LoadAudio"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for LoadAudio block.
        
        Status levels:
        - Error (0): File path missing or file doesn't exist
        - Warning (1): Data is stale (needs re-execution)
        - Ready (2): File path configured, file exists, and data is fresh
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order
        """
        from src.features.blocks.domain import BlockStatusLevel
        from src.shared.domain.data_state import DataState
        import os
        
        def check_file_path(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if audio file path is configured and exists."""
            audio_path = blk.metadata.get("audio_path")
            if not audio_path:
                return False
            return os.path.exists(audio_path)
        
        def check_data_fresh(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if block data is fresh (not stale)."""
            if not hasattr(f, 'data_state_service') or not f.data_state_service:
                return True  # If no data state service, assume fresh
            try:
                project_id = getattr(f, 'current_project_id', None) if hasattr(f, 'current_project_id') else None
                data_state = f.data_state_service.get_block_data_state(blk.id, project_id)
                return data_state != DataState.STALE
            except Exception:
                return True  # On error, assume fresh
        
        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[check_file_path]
            ),
            BlockStatusLevel(
                priority=1,
                name="stale",
                display_name="Stale",
                color="#ffa94d",
                conditions=[check_data_fresh]
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
        Process LoadAudio block.
        
        Args:
            block: Block entity to process
            inputs: Input data items (should be empty for LoadAudio)
            metadata: Optional metadata (not used currently)
            
        Returns:
            Dictionary with "audio" port containing loaded AudioDataItem
            
        Raises:
            ProcessingError: If file path not specified or loading fails
        """
        
        # Get file path from block metadata
        file_path = block.metadata.get("audio_path")
        if not file_path:
            raise ProcessingError(
                "No file path specified in block metadata",
                block_id=block.id,
                block_name=block.name
            )
        
        # Get progress tracker from metadata
        progress_tracker = get_progress_tracker(metadata)
        
        Log.info(f"LoadAudioBlockProcessor: Loading audio from {file_path}")
        
        try:
            # Use progress scope for automatic start/complete
            with progress_scope(progress_tracker, "Loading audio", total=3):
                # Step 1: Create audio data item
                yield_progress(progress_tracker, 1, "Initializing audio item...")
                audio_item = AudioDataItem(
                    id="",  # Will be generated
                    block_id=block.id,
                    name=f"{block.name}_audio_output",
                    type="Audio"
                )
                
                # Step 2: Load audio file (this is the long operation)
                yield_progress(progress_tracker, 2, "Reading and decoding audio file...")
                audio_item.load_audio(file_path)
                
                Log.info(
                    f"LoadAudioBlockProcessor: Loaded audio - "
                    f"{audio_item.sample_rate}Hz, {audio_item.length_ms:.2f}ms"
                )
                
                # Step 3: Generate and store waveform
                try:
                    yield_progress(progress_tracker, 3, "Generating waveform...")
                    from src.shared.application.services.waveform_service import get_waveform_service
                    waveform_service = get_waveform_service()
                    waveform_service.compute_and_store(audio_item)
                except Exception as e:
                    Log.warning(f"LoadAudioBlockProcessor: Failed to generate waveform: {e}")
                    # Continue without waveform (backward compatible)
            
            # Return output
            return {
                "audio": audio_item
            }
            
        except Exception as e:
            raise ProcessingError(
                f"Failed to load audio file: {str(e)}",
                block_id=block.id,
                block_name=block.name
            ) from e


    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate LoadAudio block configuration before execution.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        # LoadAudioBlock doesn't have specific validation requirements
        return []


# Auto-register this processor class
register_processor_class(LoadAudioBlockProcessor)


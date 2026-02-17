"""
SetlistAudioInput Block Processor

Dedicated block for setlist audio entry point.
Simplifies setlist processing by providing a clear audio input block.
"""
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.entities import AudioDataItem
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


class SetlistAudioInputBlockProcessor(BlockProcessor):
    """
    Processor for SetlistAudioInput block type.
    
    Dedicated block for setlist processing that loads audio files.
    Similar to LoadAudio but simpler and setlist-specific.
    """
    
    def can_process(self, block: Block) -> bool:
        """Check if this processor can handle the block"""
        return block.type == "SetlistAudioInput"
    
    def get_block_type(self) -> str:
        """Get the block type this processor handles"""
        return "SetlistAudioInput"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for SetlistAudioInput block.
        
        Status levels:
        - Error (0): Audio path missing or file doesn't exist
        - Ready (1): Audio path configured and file exists
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order
        """
        from src.features.blocks.domain import BlockStatusLevel
        import os
        
        def check_audio_path(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if audio file path is configured and exists."""
            audio_path = blk.metadata.get("audio_path")
            if not audio_path:
                return False
            return os.path.exists(audio_path)
        
        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[check_audio_path]
            ),
            BlockStatusLevel(
                priority=1,
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
        Process SetlistAudioInput block.
        
        Args:
            block: Block entity to process
            inputs: Input data items (should be empty for SetlistAudioInput)
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
                "No audio path specified in block metadata. "
                "Set the audio file path for this setlist song.",
                block_id=block.id,
                block_name=block.name
            )
        
        # Get progress tracker from metadata
        progress_tracker = metadata.get("progress_tracker") if metadata else None
        
        Log.info(f"SetlistAudioInputBlockProcessor: Loading audio from {file_path}")
        
        try:
            # Start progress tracking
            if progress_tracker:
                progress_tracker.start(f"Loading audio from {file_path}", total=None)
            
            # Create audio data item
            audio_item = AudioDataItem(
                id="",  # Will be generated
                block_id=block.id,
                name=f"{block.name}_audio_output",
                type="Audio"
            )
            
            # Load audio file
            if progress_tracker:
                progress_tracker.update(message="Reading audio file...")
            audio_item.load_audio(file_path)
            
            Log.info(
                f"SetlistAudioInputBlockProcessor: Loaded audio - "
                f"{audio_item.sample_rate}Hz, {audio_item.length_ms:.2f}ms"
            )
            
            # Generate and store waveform
            try:
                if progress_tracker:
                    progress_tracker.update(message="Generating waveform...")
                from src.shared.application.services.waveform_service import get_waveform_service
                waveform_service = get_waveform_service()
                waveform_service.compute_and_store(audio_item)
            except Exception as e:
                Log.warning(f"SetlistAudioInputBlockProcessor: Failed to generate waveform: {e}")
                # Continue without waveform (backward compatible)
            
            # Complete progress tracking
            if progress_tracker:
                progress_tracker.complete("Audio loaded successfully")
            
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
        Validate SetlistAudioInput block configuration before execution.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Validate audio path is set
        audio_path = block.metadata.get("audio_path")
        if not audio_path:
            errors.append(
                f"Block '{block.name}' (type: {block.type}): "
                "audio_path required in block metadata. Set the audio file path for this setlist song."
            )
        elif not isinstance(audio_path, str):
            errors.append(
                f"Block '{block.name}' (type: {block.type}): "
                "audio_path must be a string."
            )
        
        return errors


# Auto-register this processor class
register_processor_class(SetlistAudioInputBlockProcessor)


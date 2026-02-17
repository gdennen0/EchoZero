"""
ExportAudio Block Processor

Copies incoming audio DataItems into a configured directory using the requested format.
"""
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from pathlib import Path
import shutil

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.application.blocks import register_processor_class
from src.features.execution.application.progress_helpers import (
    track_progress, get_progress_tracker
)
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


class ExportAudioBlockProcessor(BlockProcessor):
    """Processor for ExportAudio block type."""

    _SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg", "m4a", "aiff", "aif"}

    def can_process(self, block: Block) -> bool:
        return block.type == "ExportAudio"

    def get_block_type(self) -> str:
        return "ExportAudio"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for ExportAudio block.
        
        Status levels:
        - Error (0): Output directory not configured or invalid
        - Warning (1): Audio input not connected
        - Ready (2): All requirements met
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order
        """
        from src.features.blocks.domain import BlockStatusLevel
        import os
        
        def check_output_dir(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if output directory is configured and writable."""
            output_dir = blk.metadata.get("output_dir")
            if not output_dir:
                return False
            # Check if directory exists and is writable
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception:
                    return False
            return os.access(output_dir, os.W_OK)
        
        def check_audio_input(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if audio input is connected."""
            if not hasattr(f, 'connection_service'):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [c for c in connections if c.target_block_id == blk.id and c.target_input_name == "audio"]
            return len(incoming) > 0
        
        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[check_output_dir]
            ),
            BlockStatusLevel(
                priority=1,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[check_audio_input]
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
        # Settings are stored at block.metadata top level by ExportAudioSettingsManager
        output_dir = block.metadata.get("output_dir")
        fmt = (block.metadata.get("audio_format") or "wav").lower()

        if not output_dir:
            raise ProcessingError(
                "ExportAudio block requires 'set_output_dir' before executing",
                block_id=block.id,
                block_name=block.name
            )

        if fmt not in self._SUPPORTED_FORMATS:
            raise ProcessingError(
                f"Unsupported export format '{fmt}'",
                block_id=block.id,
                block_name=block.name
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        Log.info(f"ExportAudioBlockProcessor: Exporting to directory: {output_path.absolute()}")

        if not inputs:
            Log.info(f"ExportAudioBlockProcessor: No input data for block {block.name}")
            return {}

        # Flatten inputs - handle both single items and lists
        data_items_to_export = []
        for port_name, port_data in inputs.items():
            if isinstance(port_data, list):
                Log.info(f"ExportAudioBlockProcessor: Port '{port_name}' has {len(port_data)} item(s)")
                data_items_to_export.extend(port_data)
            else:
                Log.info(f"ExportAudioBlockProcessor: Port '{port_name}' has 1 item")
                data_items_to_export.append(port_data)
        
        Log.info(f"ExportAudioBlockProcessor: Exporting {len(data_items_to_export)} total item(s)")

        # Get progress tracker from metadata
        progress_tracker = get_progress_tracker(metadata)
        
        # Use track_progress for automatic progress updates
        exported = 0
        for data_item in track_progress(data_items_to_export, progress_tracker, "Exporting audio files"):
            if not data_item.file_path:
                Log.warning(f"ExportAudioBlockProcessor: DataItem has no file_path (block {block.name})")
                continue

            src = Path(data_item.file_path)
            if not src.is_file():
                Log.warning(f"ExportAudioBlockProcessor: Source file missing: {src}")
                continue

            dest = output_path / f"{data_item.name}_{data_item.id}.{fmt}"
            shutil.copy2(src, dest)
            Log.info(f"ExportAudioBlockProcessor: Exported {src} -> {dest}")
            exported += 1

        if exported == 0:
            raise ProcessingError(
                "ExportAudio block did not find any valid audio files to export",
                block_id=block.id,
                block_name=block.name
            )

        return {}


    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate ExportAudio block configuration before execution.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        # ExportAudioBlock doesn't have specific validation requirements
        return []


register_processor_class(ExportAudioBlockProcessor)


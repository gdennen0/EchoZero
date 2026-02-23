"""
ExportMA2 Block Processor

Exports event data to GrandMA2 timecode format.
"""
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


class ExportMA2BlockProcessor(BlockProcessor):
    """Processor for ExportMA2 block type.

    Exports event timing data to GrandMA2 timecode format files.
    Requires an output path to be configured via block metadata.
    """

    def can_process(self, block: Block) -> bool:
        return block.type == "ExportMA2"

    def get_block_type(self) -> str:
        return "ExportMA2"

    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        from src.features.blocks.domain import BlockStatusLevel

        def check_output_path(blk: Block, f: "ApplicationFacade") -> bool:
            output_path = blk.metadata.get("output_path")
            return bool(output_path)

        def check_events_input(blk: Block, f: "ApplicationFacade") -> bool:
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
                conditions=[check_output_path]
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
        output_path = block.metadata.get("output_path")
        if not output_path:
            raise ProcessingError(
                "ExportMA2 block requires 'output_path' to be configured before executing",
                block_id=block.id,
                block_name=block.name
            )

        if not inputs or "events" not in inputs:
            raise ProcessingError(
                "ExportMA2 block requires event data on the 'events' input",
                block_id=block.id,
                block_name=block.name
            )

        Log.warning(f"ExportMA2BlockProcessor: MA2 timecode export not yet implemented for block {block.name}")
        raise ProcessingError(
            "MA2 timecode export is not yet implemented",
            block_id=block.id,
            block_name=block.name
        )


register_processor_class(ExportMA2BlockProcessor)

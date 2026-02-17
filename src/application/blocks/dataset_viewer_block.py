"""
Dataset Viewer Block Processor

Block for manually auditing directories of audio clips. No execution output;
the block provides a UI to select a source directory, step through samples,
play audio, and move rejected samples into a "removed" subdirectory.
"""
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.application.blocks import register_processor_class

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


class DatasetViewerBlockProcessor(BlockProcessor):
    """
    Processor for DatasetViewer block type.

    No inputs or outputs. Execution is a no-op; the block is used for
    interactive auditing via its panel (source dir, sample list, waveform,
    play/remove/prev/next).
    """

    def can_process(self, block: Block) -> bool:
        return block.type == "DatasetViewer"

    def get_block_type(self) -> str:
        return "DatasetViewer"

    def get_status_levels(
        self, block: Block, facade: "ApplicationFacade"
    ) -> List["BlockStatusLevel"]:
        """
        Status levels for DatasetViewer block.

        - Warning (0): Source directory not configured
        - Ready (1): Source directory set (or not required for UI-only use)
        """
        from src.features.blocks.domain import BlockStatusLevel

        def has_source_dir(blk: Block, _f: "ApplicationFacade") -> bool:
            return bool(blk.metadata.get("source_dir"))

        return [
            BlockStatusLevel(
                priority=0,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[lambda b, f: not has_source_dir(b, f)],
            ),
            BlockStatusLevel(
                priority=1,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[],
            ),
        ]

    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, DataItem]:
        """
        No-op: Dataset Viewer is an interactive auditing tool; no pipeline output.
        """
        return {}

    def cleanup(self, block: Block) -> None:
        """No persistent resources to release."""
        pass


register_processor_class(DatasetViewerBlockProcessor)

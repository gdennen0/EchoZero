"""
Cloud Block Processor Wrapper

Wraps existing BlockProcessor implementations to execute on cloud providers.
Maintains the BlockProcessor interface while delegating compute to cloud.

Security: Users connect their own cloud accounts. We never access their data
beyond what's needed for job execution.
"""
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import tempfile
import shutil

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.application.services.cloud_compute_service import (
    CloudComputeService,
    CloudProvider
)
from src.utils.message import Log


class CloudBlockProcessor(BlockProcessor):
    """
    Wrapper that executes block processing on cloud providers.
    
    Flow:
    1. Upload input files to cloud storage
    2. Submit job to cloud compute service
    3. Poll for completion with progress updates
    4. Download results from cloud storage
    5. Return DataItems as normal
    
    Maintains full BlockProcessor interface - transparent to execution engine.
    """
    
    def __init__(
        self,
        wrapped_processor: BlockProcessor,
        cloud_service: CloudComputeService,
        provider: Optional[CloudProvider] = None
    ):
        """
        Initialize cloud block processor.
        
        Args:
            wrapped_processor: Original BlockProcessor to wrap
            cloud_service: CloudComputeService instance
            provider: Cloud provider to use (defaults to first connected)
        """
        self._wrapped = wrapped_processor
        self._cloud = cloud_service
        self._provider = provider
    
    def can_process(self, block: Block) -> bool:
        """Check if this processor can handle the block"""
        return self._wrapped.can_process(block)
    
    def get_block_type(self) -> str:
        """Get the block type this processor handles"""
        return self._wrapped.get_block_type()
    
    def get_status_levels(self, block: Block, facade: Any) -> list:
        """Get status levels (same as wrapped processor)"""
        return self._wrapped.get_status_levels(block, facade)
    
    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Dict[str, DataItem]:
        """
        Process block on cloud provider.
        
        Args:
            block: Block entity to process
            inputs: Dictionary mapping input port names to DataItem instances
            metadata: Optional metadata for processing
            progress_callback: Optional progress callback (0-100)
            
        Returns:
            Dictionary mapping output port names to DataItem instances
            
        Raises:
            ProcessingError: If processing fails
        """
        # Check if cloud is enabled for this block
        use_cloud = block.metadata.get("use_cloud", False)
        if not use_cloud:
            # Fall back to local processing
            return self._wrapped.process(block, inputs, metadata)
        
        # Check if provider is connected
        provider = self._provider or self._cloud._get_default_provider()
        if not self._cloud.is_provider_connected(provider):
            raise ProcessingError(
                f"Cloud provider {provider.value} not connected. "
                "Please connect your cloud account in settings.",
                block_id=block.id,
                block_name=block.name
            )
        
        Log.info(f"CloudBlockProcessor: Processing {block.name} on {provider.value}")
        
        try:
            # Step 1: Upload inputs to cloud storage
            if progress_callback:
                progress_callback(5)
            
            cloud_inputs = self._upload_inputs(inputs, block)
            Log.debug(f"CloudBlockProcessor: Uploaded {len(cloud_inputs)} input files")
            
            # Step 2: Submit job to cloud
            if progress_callback:
                progress_callback(10)
            
            job_id = self._cloud.submit_job(
                block_type=block.type,
                inputs=cloud_inputs,
                settings=block.metadata,
                provider=provider
            )
            Log.info(f"CloudBlockProcessor: Submitted job {job_id}")
            
            # Step 3: Wait for completion with progress updates
            def cloud_progress(p: int):
                # Map cloud progress (0-90) to overall progress (10-90)
                overall = 10 + int(p * 0.8)
                if progress_callback:
                    progress_callback(overall)
            
            completed_job = self._cloud.wait_for_completion(
                job_id,
                progress_callback=cloud_progress,
                timeout_seconds=3600  # 1 hour max
            )
            
            if completed_job.status != "completed":
                raise ProcessingError(
                    f"Cloud job {job_id} failed: {completed_job.error_message}",
                    block_id=block.id,
                    block_name=block.name
                )
            
            # Step 4: Download results
            if progress_callback:
                progress_callback(90)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_paths = self._cloud.download_results(
                    job_id,
                    Path(temp_dir)
                )
                
                # Step 5: Convert downloaded files to DataItems
                if progress_callback:
                    progress_callback(95)
                
                outputs = self._create_output_data_items(
                    block,
                    output_paths,
                    metadata
                )
                
                if progress_callback:
                    progress_callback(100)
                
                Log.info(f"CloudBlockProcessor: Completed {block.name} (job {job_id})")
                return outputs
        
        except Exception as e:
            error_msg = str(e)
            Log.error(f"CloudBlockProcessor: Error processing {block.name}: {error_msg}")
            raise ProcessingError(
                f"Cloud processing failed: {error_msg}",
                block_id=block.id,
                block_name=block.name
            )
    
    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> list:
        """Validate block configuration (delegate to wrapped processor)"""
        return self._wrapped.validate_configuration(
            block, data_item_repo, connection_repo, block_registry
        )
    
    def cleanup(self, block: Block) -> None:
        """Cleanup resources (delegate to wrapped processor)"""
        self._wrapped.cleanup(block)
    
    def get_expected_outputs(self, block: Block) -> Dict[str, list]:
        """Get expected outputs (same as wrapped processor)"""
        return self._wrapped.get_expected_outputs(block)
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _upload_inputs(
        self,
        inputs: Dict[str, DataItem],
        block: Block
    ) -> Dict[str, Any]:
        """
        Upload input files to cloud storage.
        
        Args:
            inputs: Input DataItems
            block: Block entity
            
        Returns:
            Dictionary with cloud storage paths for each input
        """
        cloud_inputs = {}
        
        for port_name, data_item in inputs.items():
            if not data_item.file_path:
                raise ProcessingError(
                    f"Input '{port_name}' has no file path for cloud upload",
                    block_id=block.id,
                    block_name=block.name
                )
            
            file_path = Path(data_item.file_path)
            if not file_path.exists():
                raise ProcessingError(
                    f"Input file not found: {file_path}",
                    block_id=block.id,
                    block_name=block.name
                )
            
            # In real implementation, this would:
            # 1. Upload file to S3/GCS/Azure Blob
            # 2. Return cloud storage path
            
            # Placeholder: Just return local path (would be cloud path in real impl)
            cloud_inputs[port_name] = {
                "cloud_path": f"s3://echozero-inputs/{block.id}/{port_name}/{file_path.name}",
                "local_path": str(file_path),
                "size_bytes": file_path.stat().st_size
            }
        
        return cloud_inputs
    
    def _create_output_data_items(
        self,
        block: Block,
        output_paths: Dict[str, Path],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, DataItem]:
        """
        Create DataItems from downloaded output files.
        
        Args:
            block: Block entity
            output_paths: Dictionary mapping output names to file paths
            metadata: Optional processing metadata
            
        Returns:
            Dictionary mapping output port names to DataItems
        """
        # In real implementation, this would:
        # 1. Determine output port names from block configuration
        # 2. Create appropriate DataItem types (AudioDataItem, EventDataItem, etc.)
        # 3. Load file metadata (sample rate, length, etc.)
        # 4. Return DataItems matching expected outputs
        
        # Placeholder: Would create actual DataItems based on block type
        outputs = {}
        
        # For SeparatorBlock, outputs would be AudioDataItems
        # For other blocks, would create appropriate types
        
        return outputs

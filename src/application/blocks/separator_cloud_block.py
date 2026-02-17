"""
Separator Cloud Block Processor

Copy of SeparatorBlock that runs on AWS/Google Cloud.
Allows credential input directly in block settings.

Usage:
1. Create "SeparatorCloud" block
2. Enter AWS credentials in block settings
3. Execute - runs on cloud automatically
"""
import subprocess
import os
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, TYPE_CHECKING
import tempfile
import time

import boto3
from botocore.exceptions import ClientError

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import AudioDataItem
from src.shared.domain.entities import DataItem
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade

# Import Demucs models from separator_block
from src.application.blocks.separator_block import DEMUCS_MODELS


class SeparatorCloudBlockProcessor(BlockProcessor):
    """
    Cloud-powered Separator block processor.
    
    Same as SeparatorBlock but runs on AWS Batch.
    Credentials configured in block settings.
    """
    
    def can_process(self, block: Block) -> bool:
        return block.type == "SeparatorCloud"
    
    def get_block_type(self) -> str:
        return "SeparatorCloud"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """Status levels for cloud separator"""
        from src.features.blocks.domain import BlockStatusLevel
        
        def check_audio_input(blk: Block, f: "ApplicationFacade") -> bool:
            if not hasattr(f, 'connection_service'):
                return False
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [c for c in connections if c.target_block_id == blk.id and c.target_input_name == "audio"]
            return len(incoming) > 0
        
        def check_credentials(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if AWS credentials are configured"""
            aws_key = blk.metadata.get("aws_access_key_id", "").strip()
            aws_secret = blk.metadata.get("aws_secret_access_key", "").strip()
            return bool(aws_key and aws_secret)
        
        def check_aws_config(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if AWS config (bucket, queue, etc.) is set"""
            bucket = blk.metadata.get("aws_s3_bucket", "").strip()
            queue = blk.metadata.get("aws_batch_queue", "").strip()
            job_def = blk.metadata.get("aws_batch_job_def", "").strip()
            return bool(bucket and queue and job_def)
        
        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[check_audio_input]
            ),
            BlockStatusLevel(
                priority=1,
                name="warning",
                display_name="Missing Credentials",
                color="#ffd43b",
                conditions=[check_credentials]
            ),
            BlockStatusLevel(
                priority=2,
                name="warning",
                display_name="Missing AWS Config",
                color="#ffd43b",
                conditions=[check_aws_config]
            ),
            BlockStatusLevel(
                priority=3,
                name="ready",
                display_name="Ready (Cloud)",
                color="#51cf66",
                conditions=[]
            )
        ]
    
    def get_expected_outputs(self, block: Block) -> Dict[str, List[str]]:
        """Same as regular SeparatorBlock"""
        from src.application.processing.output_name_helpers import make_output_name
        
        two_stems = block.metadata.get("two_stems")
        if two_stems:
            return {
                "audio": [
                    make_output_name("audio", two_stems),
                    make_output_name("audio", "other")
                ]
            }
        else:
            return {
                "audio": [
                    make_output_name("audio", "vocals"),
                    make_output_name("audio", "drums"),
                    make_output_name("audio", "bass"),
                    make_output_name("audio", "other")
                ]
            }
    
    def _get_aws_client(self, service: str, block: Block):
        """Get AWS client using credentials from block metadata"""
        aws_key = block.metadata.get("aws_access_key_id", "").strip()
        aws_secret = block.metadata.get("aws_secret_access_key", "").strip()
        aws_region = block.metadata.get("aws_region", "us-east-1").strip()
        
        if not aws_key or not aws_secret:
            raise ProcessingError(
                "AWS credentials not configured. Please set AWS Access Key ID and Secret Access Key in block settings.",
                block_id=block.id,
                block_name=block.name
            )
        
        if service == "s3":
            return boto3.client(
                's3',
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                region_name=aws_region
            )
        elif service == "batch":
            return boto3.client(
                'batch',
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                region_name=aws_region
            )
        else:
            raise ValueError(f"Unknown AWS service: {service}")
    
    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        Process Separator block on AWS Batch.
        
        Flow:
        1. Upload input audio to S3
        2. Submit AWS Batch job
        3. Wait for completion
        4. Download results from S3
        5. Create AudioDataItems
        """
        audio_item = inputs.get("audio")
        if not audio_item or not audio_item.file_path:
            raise ProcessingError(
                "SeparatorCloud block requires audio data from upstream block",
                block_id=block.id,
                block_name=block.name
            )
        
        input_file = Path(audio_item.file_path)
        if not input_file.exists():
            raise ProcessingError(
                f"Audio file not found: {input_file}",
                block_id=block.id,
                block_name=block.name
            )
        
        # Get AWS configuration from block metadata
        s3_bucket = block.metadata.get("aws_s3_bucket", "").strip()
        batch_queue = block.metadata.get("aws_batch_queue", "").strip()
        batch_job_def = block.metadata.get("aws_batch_job_def", "").strip()
        model = block.metadata.get("model", "htdemucs")
        two_stems = block.metadata.get("two_stems")
        
        if not s3_bucket or not batch_queue or not batch_job_def:
            raise ProcessingError(
                "AWS configuration incomplete. Please set S3 Bucket, Batch Queue, and Job Definition in block settings.",
                block_id=block.id,
                block_name=block.name
            )
        
        Log.info(f"SeparatorCloudBlockProcessor: Processing {input_file.name} on AWS")
        
        # Create unique job ID
        job_id = f"{block.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Get AWS clients
            s3 = self._get_aws_client("s3", block)
            batch = self._get_aws_client("batch", block)
            
            # Step 1: Upload input to S3
            Log.info("Uploading input to S3...")
            input_s3_key = f"inputs/{job_id}/{input_file.name}"
            s3.upload_file(str(input_file), s3_bucket, input_s3_key)
            input_s3_path = f"s3://{s3_bucket}/{input_s3_key}"
            Log.info(f"Uploaded to {input_s3_path}")
            
            # Step 2: Submit batch job
            Log.info("Submitting AWS Batch job...")
            output_s3_prefix = f"outputs/{job_id}/"
            job_name = f"echozero-separator-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            response = batch.submit_job(
                jobName=job_name,
                jobQueue=batch_queue,
                jobDefinition=batch_job_def,
                parameters={
                    "INPUT_S3_PATH": input_s3_path,
                    "OUTPUT_S3_PREFIX": f"s3://{s3_bucket}/{output_s3_prefix}",
                    "MODEL": model
                }
            )
            batch_job_id = response['jobId']
            Log.info(f"Job submitted: {batch_job_id}")
            
            # Step 3: Wait for completion
            Log.info("Waiting for job to complete...")
            start_time = time.time()
            timeout_seconds = 1800  # 30 minutes
            
            while True:
                response = batch.describe_jobs(jobs=[batch_job_id])
                job = response['jobs'][0]
                status = job['status']
                
                if status == 'SUCCEEDED':
                    Log.info("Job completed!")
                    break
                elif status == 'FAILED':
                    reason = job.get('statusReason', 'Unknown error')
                    raise ProcessingError(
                        f"AWS Batch job failed: {reason}",
                        block_id=block.id,
                        block_name=block.name
                    )
                
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    raise ProcessingError(
                        f"Job {batch_job_id} timed out after {timeout_seconds} seconds",
                        block_id=block.id,
                        block_name=block.name
                    )
                
                # Poll every 5 seconds
                time.sleep(5)
            
            # Step 4: Download results
            Log.info("Downloading results from S3...")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Determine expected outputs
                if two_stems:
                    expected_stems = [two_stems, "other"]
                else:
                    expected_stems = ["vocals", "drums", "bass", "other"]
                
                # Download each stem
                downloaded_files = {}
                for stem in expected_stems:
                    s3_key = f"{output_s3_prefix}{stem}.wav"
                    local_file = temp_path / f"{stem}.wav"
                    
                    try:
                        s3.download_file(s3_bucket, s3_key, str(local_file))
                        downloaded_files[stem] = local_file
                        Log.info(f"Downloaded {stem}.wav")
                    except ClientError as e:
                        if e.response['Error']['Code'] == '404':
                            Log.warning(f"Stem {stem} not found in S3 (may not have been generated)")
                        else:
                            raise
            
            # Step 5: Create AudioDataItems and copy to permanent location
            output_dir = Path(audio_item.file_path).parent / f"{block.name}_stems"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            audio_items = []
            for stem, file_path in downloaded_files.items():
                if file_path.exists():
                    # Copy to permanent location
                    final_path = output_dir / f"{stem}.wav"
                    import shutil
                    shutil.copy2(file_path, final_path)
                    
                    # Create AudioDataItem
                    audio_item = AudioDataItem(
                        id=f"{block.id}_{stem}",
                        block_id=block.id,
                        name=f"{block.name}_{stem}",
                        type="Audio",
                        created_at=datetime.now(),
                        file_path=str(final_path),
                        sample_rate=audio_item.sample_rate if hasattr(audio_item, 'sample_rate') else 44100,
                        length_ms=audio_item.length_ms if hasattr(audio_item, 'length_ms') else None,
                        metadata={
                            "stem": stem,
                            "source": "aws_batch",
                            "job_id": batch_job_id
                        }
                    )
                    audio_items.append(audio_item)
            
            if not audio_items:
                raise ProcessingError(
                    "No output files downloaded from S3",
                    block_id=block.id,
                    block_name=block.name
                )
            
            outputs = {"audio": audio_items}
            Log.info(f"SeparatorCloudBlockProcessor: Completed {len(audio_items)} stems")
            return outputs
        
        except ClientError as e:
            error_msg = f"AWS error: {e.response.get('Error', {}).get('Message', str(e))}"
            Log.error(f"SeparatorCloudBlockProcessor: {error_msg}")
            raise ProcessingError(
                error_msg,
                block_id=block.id,
                block_name=block.name
            )
        except Exception as e:
            error_msg = str(e)
            Log.error(f"SeparatorCloudBlockProcessor: Error: {error_msg}")
            raise ProcessingError(
                f"Cloud processing failed: {error_msg}",
                block_id=block.id,
                block_name=block.name
            )


# Register processor
register_processor_class(SeparatorCloudBlockProcessor)

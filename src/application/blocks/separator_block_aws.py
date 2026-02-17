"""
Quick & Dirty AWS Separator Block

Drop-in replacement for SeparatorBlockProcessor that runs on AWS Batch.
Just configure AWS credentials and it works.

Usage:
1. Set AWS credentials: export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
2. Replace SeparatorBlockProcessor with SeparatorBlockProcessorAWS in registration
3. That's it!
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


# AWS Configuration (Quick Setup)
AWS_S3_BUCKET = os.getenv("ECHOZERO_AWS_BUCKET", "echozero-cloud-storage")
AWS_BATCH_QUEUE = os.getenv("ECHOZERO_AWS_BATCH_QUEUE", "echozero-batch-queue")
AWS_BATCH_JOB_DEF = os.getenv("ECHOZERO_AWS_BATCH_JOB_DEF", "echozero-demucs")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def upload_to_s3(file_path: Path, s3_key: str, bucket: str = AWS_S3_BUCKET) -> str:
    """Upload file to S3"""
    s3 = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3.upload_file(str(file_path), bucket, s3_key)
        return f"s3://{bucket}/{s3_key}"
    except ClientError as e:
        raise ProcessingError(f"Failed to upload to S3: {e}")


def download_from_s3(s3_key: str, local_path: Path, bucket: str = AWS_S3_BUCKET):
    """Download file from S3"""
    s3 = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3.download_file(bucket, s3_key, str(local_path))
    except ClientError as e:
        raise ProcessingError(f"Failed to download from S3: {e}")


def submit_batch_job(
    input_s3_path: str,
    output_s3_prefix: str,
    model: str = "htdemucs",
    queue: str = AWS_BATCH_QUEUE,
    job_def: str = AWS_BATCH_JOB_DEF
) -> str:
    """Submit job to AWS Batch"""
    batch = boto3.client('batch', region_name=AWS_REGION)
    
    job_name = f"echozero-separator-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    try:
        response = batch.submit_job(
            jobName=job_name,
            jobQueue=queue,
            jobDefinition=job_def,
            parameters={
                "INPUT_S3_PATH": input_s3_path,
                "OUTPUT_S3_PREFIX": output_s3_prefix,
                "MODEL": model
            }
        )
        return response['jobId']
    except ClientError as e:
        raise ProcessingError(f"Failed to submit batch job: {e}")


def wait_for_job(job_id: str, timeout_seconds: int = 1800) -> dict:
    """Wait for AWS Batch job to complete"""
    batch = boto3.client('batch', region_name=AWS_REGION)
    
    start_time = time.time()
    while True:
        try:
            response = batch.describe_jobs(jobs=[job_id])
            job = response['jobs'][0]
            status = job['status']
            
            if status == 'SUCCEEDED':
                return {"status": "completed", "job": job}
            elif status == 'FAILED':
                reason = job.get('statusReason', 'Unknown error')
                raise ProcessingError(f"AWS Batch job failed: {reason}")
            
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                raise ProcessingError(f"Job {job_id} timed out after {timeout_seconds} seconds")
            
            # Poll every 5 seconds
            time.sleep(5)
        
        except ClientError as e:
            raise ProcessingError(f"Failed to check job status: {e}")


class SeparatorBlockProcessorAWS(BlockProcessor):
    """
    AWS-powered Separator block processor.
    
    Quick setup:
    1. Set environment variables:
       export AWS_ACCESS_KEY_ID=your_key
       export AWS_SECRET_ACCESS_KEY=your_secret
       export ECHOZERO_AWS_BUCKET=your-bucket-name
       export ECHOZERO_AWS_BATCH_QUEUE=your-queue-name
       export ECHOZERO_AWS_BATCH_JOB_DEF=your-job-def-name
    
    2. Replace SeparatorBlockProcessor with this in registration
    
    3. That's it!
    """
    
    def can_process(self, block: Block) -> bool:
        return block.type == "Separator"
    
    def get_block_type(self) -> str:
        return "Separator"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """Same status levels as regular SeparatorBlock"""
        from src.features.blocks.domain import BlockStatusLevel
        
        def check_audio_input(blk: Block, f: "ApplicationFacade") -> bool:
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
                conditions=[check_audio_input]
            ),
            BlockStatusLevel(
                priority=1,
                name="ready",
                display_name="Ready (AWS)",
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
                "Separator block requires audio data from upstream block",
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
        
        # Get settings
        model = block.metadata.get("model", "htdemucs")
        two_stems = block.metadata.get("two_stems")
        
        Log.info(f"SeparatorBlockProcessorAWS: Processing {input_file.name} on AWS")
        
        # Create unique job ID
        job_id = f"{block.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Step 1: Upload input to S3
            Log.info("Uploading input to S3...")
            input_s3_key = f"inputs/{job_id}/{input_file.name}"
            input_s3_path = upload_to_s3(input_file, input_s3_key)
            Log.info(f"Uploaded to {input_s3_path}")
            
            # Step 2: Submit batch job
            Log.info("Submitting AWS Batch job...")
            output_s3_prefix = f"outputs/{job_id}/"
            batch_job_id = submit_batch_job(
                input_s3_path=input_s3_path,
                output_s3_prefix=output_s3_prefix,
                model=model
            )
            Log.info(f"Job submitted: {batch_job_id}")
            
            # Step 3: Wait for completion
            Log.info("Waiting for job to complete...")
            result = wait_for_job(batch_job_id, timeout_seconds=1800)  # 30 min max
            Log.info("Job completed!")
            
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
                    # S3 key pattern: outputs/{job_id}/{stem}.wav
                    s3_key = f"{output_s3_prefix}{stem}.wav"
                    local_file = temp_path / f"{stem}.wav"
                    
                    try:
                        download_from_s3(s3_key, local_file)
                        downloaded_files[stem] = local_file
                        Log.info(f"Downloaded {stem}.wav")
                    except Exception as e:
                        Log.warning(f"Failed to download {stem}: {e}")
                
                # Step 5: Create AudioDataItems
                outputs = {}
                audio_items = []
                
                for stem, file_path in downloaded_files.items():
                    if file_path.exists():
                        # Create AudioDataItem
                        audio_item = AudioDataItem(
                            id=f"{block.id}_{stem}",
                            block_id=block.id,
                            name=f"{block.name}_{stem}",
                            type="Audio",
                            created_at=datetime.now(),
                            file_path=str(file_path),
                            sample_rate=audio_item.sample_rate if hasattr(audio_item, 'sample_rate') else 44100,
                            length_ms=audio_item.length_ms if hasattr(audio_item, 'length_ms') else None,
                            metadata={
                                "stem": stem,
                                "source": "aws_batch",
                                "job_id": batch_job_id
                            }
                        )
                        audio_items.append(audio_item)
                
                # Copy files to permanent location
                output_dir = Path(audio_item.file_path).parent / f"{block.name}_stems"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                final_items = []
                for audio_item in audio_items:
                    stem = audio_item.metadata.get("stem", "unknown")
                    final_path = output_dir / f"{stem}.wav"
                    
                    # Copy to permanent location
                    import shutil
                    shutil.copy2(audio_item.file_path, final_path)
                    
                    # Update file_path
                    audio_item.file_path = str(final_path)
                    final_items.append(audio_item)
                
                # Return as list (SeparatorBlock outputs multiple items on same port)
                outputs["audio"] = final_items
                
                Log.info(f"SeparatorBlockProcessorAWS: Completed {len(final_items)} stems")
                return outputs
        
        except Exception as e:
            error_msg = str(e)
            Log.error(f"SeparatorBlockProcessorAWS: Error: {error_msg}")
            raise ProcessingError(
                f"AWS processing failed: {error_msg}",
                block_id=block.id,
                block_name=block.name
            )


# Register processor
register_processor_class(SeparatorBlockProcessorAWS)

"""
Cloud Compute Service

Handles cloud computing integration for compute-intensive blocks.
Manages OAuth authentication, job submission, and usage tracking.

Security: Users connect their own cloud accounts via OAuth.
We never store cloud credentials - only OAuth tokens (encrypted).
"""
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

from src.utils.message import Log


class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"


@dataclass
class CloudJob:
    """Represents a cloud computing job"""
    job_id: str
    block_type: str
    provider: CloudProvider
    status: str  # "submitted", "running", "completed", "failed"
    created_at: datetime
    completed_at: Optional[datetime] = None
    cost_estimate: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class UsageSummary:
    """Summary of cloud computing usage"""
    total_jobs: int
    total_cost: float
    service_fee: float
    total_charged: float
    period_start: datetime
    period_end: datetime


class CloudComputeService:
    """
    Service for managing cloud computing jobs.
    
    Handles:
    - OAuth authentication with cloud providers
    - Job submission and monitoring
    - Usage tracking and billing
    - File upload/download
    
    Security Model:
    - Users authenticate directly with cloud providers (OAuth)
    - We never store cloud credentials
    - OAuth tokens stored encrypted, user-controlled
    """
    
    def __init__(self, backend_url: Optional[str] = None):
        """
        Initialize cloud compute service.
        
        Args:
            backend_url: URL of EchoZero backend service (for OAuth, usage tracking)
                        If None, uses environment variable or default
        """
        self._backend_url = backend_url or os.getenv("ECHOZERO_BACKEND_URL", "https://api.echozero.com")
        self._connected_providers: Dict[CloudProvider, bool] = {}
        self._oauth_tokens: Dict[CloudProvider, str] = {}  # Encrypted tokens
        self._active_jobs: Dict[str, CloudJob] = {}
        
        # Load saved OAuth tokens (encrypted)
        self._load_tokens()
        
        Log.info(f"CloudComputeService: Initialized (backend: {self._backend_url})")
    
    def is_provider_connected(self, provider: CloudProvider) -> bool:
        """Check if a cloud provider is connected"""
        return self._connected_providers.get(provider, False)
    
    def connect_provider(self, provider: CloudProvider) -> str:
        """
        Start OAuth flow for connecting a cloud provider.
        
        Args:
            provider: Cloud provider to connect
            
        Returns:
            OAuth authorization URL (user opens in browser)
        """
        # In real implementation, this would:
        # 1. Request OAuth URL from backend
        # 2. Open browser for user to authorize
        # 3. Handle callback with authorization code
        # 4. Exchange code for access token
        # 5. Store token encrypted
        
        Log.info(f"CloudComputeService: Starting OAuth flow for {provider.value}")
        
        # Placeholder: Would call backend to get OAuth URL
        oauth_url = f"{self._backend_url}/auth/oauth/start/{provider.value}"
        
        return oauth_url
    
    def handle_oauth_callback(self, provider: CloudProvider, code: str) -> bool:
        """
        Handle OAuth callback after user authorizes.
        
        Args:
            provider: Cloud provider
            code: Authorization code from OAuth callback
            
        Returns:
            True if connection successful
        """
        # In real implementation, this would:
        # 1. Exchange code for access token (via backend)
        # 2. Encrypt and store token
        # 3. Mark provider as connected
        
        Log.info(f"CloudComputeService: Handling OAuth callback for {provider.value}")
        
        # Placeholder: Would exchange code for token
        # For now, just mark as connected
        self._connected_providers[provider] = True
        self._save_tokens()
        
        return True
    
    def submit_job(
        self,
        block_type: str,
        inputs: Dict[str, Any],
        settings: Dict[str, Any],
        provider: Optional[CloudProvider] = None
    ) -> str:
        """
        Submit a cloud computing job.
        
        Args:
            block_type: Type of block to execute (e.g., "Separator")
            inputs: Input data items (file paths, etc.)
            settings: Block settings/metadata
            provider: Cloud provider to use (defaults to first connected)
            
        Returns:
            Job ID for tracking
            
        Raises:
            ValueError: If no provider connected
        """
        # Determine provider
        if provider is None:
            provider = self._get_default_provider()
        
        if not self.is_provider_connected(provider):
            raise ValueError(f"Cloud provider {provider.value} not connected. Please connect your account first.")
        
        Log.info(f"CloudComputeService: Submitting {block_type} job to {provider.value}")
        
        # In real implementation, this would:
        # 1. Upload input files to cloud storage (S3, GCS, etc.)
        # 2. Submit job to cloud compute (AWS Batch, GCP Cloud Run, etc.)
        # 3. Return job ID
        
        job_id = f"{block_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = CloudJob(
            job_id=job_id,
            block_type=block_type,
            provider=provider,
            status="submitted",
            created_at=datetime.now()
        )
        
        self._active_jobs[job_id] = job
        
        # Track usage (send to backend)
        self._track_usage(job_id, block_type, provider)
        
        return job_id
    
    def get_job_status(self, job_id: str) -> CloudJob:
        """
        Get status of a cloud computing job.
        
        Args:
            job_id: Job ID returned from submit_job()
            
        Returns:
            CloudJob with current status
        """
        if job_id not in self._active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self._active_jobs[job_id]
        
        # In real implementation, this would poll cloud provider API
        # For now, return current status
        
        return job
    
    def wait_for_completion(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[int], None]] = None,
        timeout_seconds: int = 3600
    ) -> CloudJob:
        """
        Wait for job to complete, polling status.
        
        Args:
            job_id: Job ID to wait for
            progress_callback: Optional callback for progress updates (0-100)
            timeout_seconds: Maximum time to wait
            
        Returns:
            Completed CloudJob
            
        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        import time
        
        start_time = time.time()
        last_progress = 0
        
        while True:
            job = self.get_job_status(job_id)
            
            if job.status == "completed":
                if progress_callback:
                    progress_callback(100)
                return job
            
            if job.status == "failed":
                raise RuntimeError(f"Job {job_id} failed: {job.error_message}")
            
            # Update progress (simplified - real implementation would get actual progress)
            if progress_callback:
                elapsed = time.time() - start_time
                progress = min(int((elapsed / timeout_seconds) * 90), 90)  # Cap at 90% until done
                if progress > last_progress:
                    progress_callback(progress)
                    last_progress = progress
            
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Job {job_id} timed out after {timeout_seconds} seconds")
            
            time.sleep(2)  # Poll every 2 seconds
    
    def download_results(self, job_id: str, output_dir: Path) -> Dict[str, Path]:
        """
        Download job results from cloud storage.
        
        Args:
            job_id: Job ID
            output_dir: Local directory to save results
            
        Returns:
            Dictionary mapping output names to file paths
        """
        job = self.get_job_status(job_id)
        
        if job.status != "completed":
            raise ValueError(f"Job {job_id} not completed (status: {job.status})")
        
        Log.info(f"CloudComputeService: Downloading results for job {job_id}")
        
        # In real implementation, this would:
        # 1. Get result file locations from cloud provider
        # 2. Download files from cloud storage (S3, GCS, etc.)
        # 3. Save to output_dir
        # 4. Return file paths
        
        # Placeholder
        results = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return results
    
    def get_usage_summary(self, days: int = 30) -> UsageSummary:
        """
        Get usage summary for billing.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            UsageSummary with costs and fees
        """
        # In real implementation, this would query backend for usage data
        # Backend calculates: cloud costs + markup
        
        Log.info(f"CloudComputeService: Getting usage summary for last {days} days")
        
        # Placeholder
        return UsageSummary(
            total_jobs=0,
            total_cost=0.0,
            service_fee=0.0,
            total_charged=0.0,
            period_start=datetime.now(),
            period_end=datetime.now()
        )
    
    def disconnect_provider(self, provider: CloudProvider) -> None:
        """Disconnect a cloud provider (revoke OAuth token)"""
        Log.info(f"CloudComputeService: Disconnecting {provider.value}")
        
        self._connected_providers[provider] = False
        if provider in self._oauth_tokens:
            del self._oauth_tokens[provider]
        
        self._save_tokens()
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _get_default_provider(self) -> CloudProvider:
        """Get default cloud provider (first connected)"""
        for provider in CloudProvider:
            if self.is_provider_connected(provider):
                return provider
        
        raise ValueError("No cloud provider connected")
    
    def _track_usage(self, job_id: str, block_type: str, provider: CloudProvider) -> None:
        """Track usage event (send to backend for billing)"""
        # In real implementation, this would POST to backend:
        # POST /usage/track
        # {
        #   "job_id": job_id,
        #   "block_type": block_type,
        #   "provider": provider.value,
        #   "timestamp": datetime.now().isoformat()
        # }
        
        Log.debug(f"CloudComputeService: Tracking usage for job {job_id}")
    
    def _load_tokens(self) -> None:
        """Load encrypted OAuth tokens from local storage"""
        # In real implementation, this would:
        # 1. Load encrypted tokens from secure storage (keyring, encrypted file)
        # 2. Decrypt tokens
        # 3. Verify tokens are still valid
        # 4. Mark providers as connected
        
        token_file = Path.home() / ".echozero" / "cloud_tokens.json"
        if token_file.exists():
            try:
                with open(token_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # In real implementation, decrypt tokens
                    for provider_str, connected in data.get("connected", {}).items():
                        provider = CloudProvider(provider_str)
                        self._connected_providers[provider] = connected
                Log.debug("CloudComputeService: Loaded saved tokens")
            except Exception as e:
                Log.warning(f"CloudComputeService: Failed to load tokens: {e}")
    
    def _save_tokens(self) -> None:
        """Save encrypted OAuth tokens to local storage"""
        # In real implementation, this would:
        # 1. Encrypt tokens
        # 2. Save to secure storage (keyring, encrypted file)
        
        token_file = Path.home() / ".echozero" / "cloud_tokens.json"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "connected": {
                provider.value: connected
                for provider, connected in self._connected_providers.items()
            }
        }
        
        try:
            with open(token_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            Log.debug("CloudComputeService: Saved tokens")
        except Exception as e:
            Log.warning(f"CloudComputeService: Failed to save tokens: {e}")

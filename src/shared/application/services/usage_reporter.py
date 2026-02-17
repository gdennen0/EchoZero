"""
Usage Reporting Service

Reports cloud computing usage to Stripe Billing.
Stripe handles accumulation and bi-weekly invoicing automatically.
"""
from typing import Optional
import httpx
from datetime import datetime

from src.utils.message import Log


class UsageReporter:
    """
    Reports usage to Stripe Billing for automatic bi-weekly invoicing.
    
    Responsibilities:
    - Report completed cloud jobs
    - Calculate costs (cloud + markup)
    - Send to backend → Stripe Billing
    
    Does NOT handle:
    - Invoice creation (Stripe does this)
    - Payment processing (Stripe does this)
    - Billing cycles (Stripe manages this)
    """
    
    def __init__(self, backend_url: str, auth_service):
        """
        Initialize usage reporter.
        
        Args:
            backend_url: Backend service URL
            auth_service: AuthService instance (for user session)
        """
        self._backend_url = backend_url
        self._auth_service = auth_service
        
        Log.info("UsageReporter: Initialized")
    
    def report_job(
        self,
        block_type: str,
        duration_seconds: float,
        cloud_cost: float,
        provider: str,
        job_id: Optional[str] = None
    ) -> bool:
        """
        Report a completed cloud computing job.
        
        Backend creates Stripe UsageRecord.
        Stripe accumulates usage and invoices bi-weekly automatically.
        
        Args:
            block_type: Type of block (e.g., "Separator")
            duration_seconds: Job duration in seconds
            cloud_cost: Cloud provider cost (e.g., $0.01)
            provider: Cloud provider ("aws", "gcp", "azure")
            job_id: Optional job identifier
            
        Returns:
            True if reported successfully, False otherwise
        """
        if not self._auth_service.is_logged_in():
            Log.warning("UsageReporter: Cannot report usage - user not logged in")
            return False
        
        stripe_customer_id = self._auth_service.get_stripe_customer_id()
        if not stripe_customer_id:
            Log.warning("UsageReporter: Cannot report usage - no Stripe Customer ID")
            return False
        
        # Calculate total cost (cloud cost + markup)
        markup_percent = 0.10  # 10% markup
        service_fee = cloud_cost * markup_percent
        total_cost = cloud_cost + service_fee
        
        try:
            access_token = self._auth_service.get_access_token()
            if not access_token:
                Log.warning("UsageReporter: No access token available")
                return False
            
            with httpx.Client() as client:
                response = client.post(
                    f"{self._backend_url}/usage/report",
                    json={
                        "stripe_customer_id": stripe_customer_id,
                        "block_type": block_type,
                        "duration_seconds": duration_seconds,
                        "cloud_cost": cloud_cost,
                        "service_fee": service_fee,
                        "total_cost": total_cost,
                        "provider": provider,
                        "job_id": job_id,
                        "timestamp": datetime.now().isoformat()
                    },
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10.0
                )
                response.raise_for_status()
            
            Log.info(
                f"UsageReporter: Reported {block_type} job "
                f"(cost: ${total_cost:.4f}, provider: {provider})"
            )
            return True
        
        except httpx.HTTPStatusError as e:
            Log.error(f"UsageReporter: Failed to report usage: {e.response.text}")
            return False
        except Exception as e:
            Log.error(f"UsageReporter: Error reporting usage: {e}")
            return False
    
    def get_billing_portal_url(self) -> Optional[str]:
        """
        Get Stripe Customer Portal URL for user to manage billing.
        
        Users can:
        - View invoices
        - Update payment method
        - View usage
        - Cancel subscription
        
        Returns:
            URL to Stripe Customer Portal, or None if not available
        """
        if not self._auth_service.is_logged_in():
            return None
        
        stripe_customer_id = self._auth_service.get_stripe_customer_id()
        if not stripe_customer_id:
            return None
        
        try:
            access_token = self._auth_service.get_access_token()
            if not access_token:
                return None
            
            with httpx.Client() as client:
                response = client.post(
                    f"{self._backend_url}/billing/portal",
                    json={"stripe_customer_id": stripe_customer_id},
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("url")
        
        except Exception as e:
            Log.error(f"UsageReporter: Failed to get billing portal URL: {e}")
            return None

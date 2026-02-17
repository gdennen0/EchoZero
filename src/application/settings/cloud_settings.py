"""
Cloud Computing Settings Manager

Manages cloud computing preferences and connection state.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

from .base_settings import BaseSettings, BaseSettingsManager

if TYPE_CHECKING:
    from src.infrastructure.persistence.sqlite.preferences_repository_impl import PreferencesRepository


@dataclass
class CloudSettings(BaseSettings):
    """
    Cloud computing settings schema.
    """
    # Cloud provider connections
    aws_connected: bool = False
    aws_access_token: Optional[str] = None  # Encrypted in real implementation
    azure_connected: bool = False
    azure_access_token: Optional[str] = None
    
    # Billing
    stripe_customer_id: Optional[str] = None
    credit_balance: float = 0.0  # Prepaid credits
    
    # Preferences
    default_provider: str = "aws"  # "aws", "gcp", "azure"
    auto_use_cloud: bool = False  # Automatically use cloud for supported blocks
    
    # Usage tracking
    total_jobs: int = 0
    total_spent: float = 0.0


class CloudSettingsManager(BaseSettingsManager):
    """
    Manager for cloud computing settings.
    """
    
    NAMESPACE = "cloud"
    SETTINGS_CLASS = CloudSettings
    
    def __init__(self, preferences_repo: Optional['PreferencesRepository'] = None, parent=None):
        super().__init__(preferences_repo, parent)
    
    # =========================================================================
    # Provider Connection Properties
    # =========================================================================
    
    @property
    def aws_connected(self) -> bool:
        return self._settings.aws_connected
    
    @aws_connected.setter
    def aws_connected(self, value: bool):
        if value != self._settings.aws_connected:
            self._settings.aws_connected = value
            self._save_setting('aws_connected')
    
    @property
    def azure_connected(self) -> bool:
        return self._settings.azure_connected
    
    @azure_connected.setter
    def azure_connected(self, value: bool):
        if value != self._settings.azure_connected:
            self._settings.azure_connected = value
            self._save_setting('azure_connected')
    
    def is_any_provider_connected(self) -> bool:
        """Check if any cloud provider is connected"""
        return self.aws_connected or self.azure_connected

    def get_connected_providers(self) -> list[str]:
        """Get list of connected provider names"""
        providers = []
        if self.aws_connected:
            providers.append("aws")
        if self.azure_connected:
            providers.append("azure")
        return providers
    
    # =========================================================================
    # Billing Properties
    # =========================================================================
    
    @property
    def stripe_customer_id(self) -> Optional[str]:
        return self._settings.stripe_customer_id
    
    @stripe_customer_id.setter
    def stripe_customer_id(self, value: Optional[str]):
        if value != self._settings.stripe_customer_id:
            self._settings.stripe_customer_id = value
            self._save_setting('stripe_customer_id')
    
    @property
    def credit_balance(self) -> float:
        return self._settings.credit_balance
    
    @credit_balance.setter
    def credit_balance(self, value: float):
        if value != self._settings.credit_balance:
            self._settings.credit_balance = max(0.0, value)
            self._save_setting('credit_balance')
    
    def add_credits(self, amount: float):
        """Add credits to balance"""
        self.credit_balance = self.credit_balance + amount
    
    def deduct_credits(self, amount: float) -> bool:
        """
        Deduct credits from balance.
        
        Returns:
            True if deduction successful, False if insufficient credits
        """
        if self.credit_balance >= amount:
            self.credit_balance = self.credit_balance - amount
            return True
        return False
    
    # =========================================================================
    # Preferences Properties
    # =========================================================================
    
    @property
    def default_provider(self) -> str:
        return self._settings.default_provider
    
    @default_provider.setter
    def default_provider(self, value: str):
        valid_providers = {"aws", "azure"}
        if value in valid_providers and value != self._settings.default_provider:
            self._settings.default_provider = value
            self._save_setting('default_provider')
    
    @property
    def auto_use_cloud(self) -> bool:
        return self._settings.auto_use_cloud
    
    @auto_use_cloud.setter
    def auto_use_cloud(self, value: bool):
        if value != self._settings.auto_use_cloud:
            self._settings.auto_use_cloud = value
            self._save_setting('auto_use_cloud')
    
    # =========================================================================
    # Usage Tracking Properties
    # =========================================================================
    
    @property
    def total_jobs(self) -> int:
        return self._settings.total_jobs
    
    @total_jobs.setter
    def total_jobs(self, value: int):
        if value != self._settings.total_jobs:
            self._settings.total_jobs = max(0, value)
            self._save_setting('total_jobs')
    
    @property
    def total_spent(self) -> float:
        return self._settings.total_spent
    
    @total_spent.setter
    def total_spent(self, value: float):
        if value != self._settings.total_spent:
            self._settings.total_spent = max(0.0, value)
            self._save_setting('total_spent')
    
    def record_job(self, cost: float):
        """Record a completed cloud job"""
        self.total_jobs = self.total_jobs + 1
        self.total_spent = self.total_spent + cost

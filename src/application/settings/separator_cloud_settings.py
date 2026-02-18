"""
SeparatorCloud block settings.
"""

from dataclasses import dataclass
from typing import Optional

from src.application.blocks.separator_block import DEMUCS_MODELS
from src.shared.application.settings import register_block_settings

from .base_settings import BaseSettings
from .block_settings import BlockSettingsManager


@register_block_settings(
    "SeparatorCloud",
    description="Cloud separator settings",
    tags=["separator", "cloud", "aws"],
)
@dataclass
class SeparatorCloudBlockSettings(BaseSettings):
    """Settings schema for SeparatorCloud blocks."""

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    aws_s3_bucket: str = ""
    aws_batch_queue: str = ""
    aws_batch_job_def: str = ""
    model: str = "htdemucs"
    two_stems: Optional[str] = None


class SeparatorCloudSettingsManager(BlockSettingsManager):
    """Type-safe settings manager for SeparatorCloud block metadata."""

    SETTINGS_CLASS = SeparatorCloudBlockSettings

    @property
    def aws_access_key_id(self) -> str:
        return self._settings.aws_access_key_id

    @aws_access_key_id.setter
    def aws_access_key_id(self, value: str) -> None:
        value = value.strip()
        if value != self._settings.aws_access_key_id:
            self._settings.aws_access_key_id = value
            self._save_setting("aws_access_key_id")

    @property
    def aws_secret_access_key(self) -> str:
        return self._settings.aws_secret_access_key

    @aws_secret_access_key.setter
    def aws_secret_access_key(self, value: str) -> None:
        value = value.strip()
        if value != self._settings.aws_secret_access_key:
            self._settings.aws_secret_access_key = value
            self._save_setting("aws_secret_access_key")

    @property
    def aws_region(self) -> str:
        return self._settings.aws_region

    @aws_region.setter
    def aws_region(self, value: str) -> None:
        value = value.strip() or "us-east-1"
        if value != self._settings.aws_region:
            self._settings.aws_region = value
            self._save_setting("aws_region")

    @property
    def aws_s3_bucket(self) -> str:
        return self._settings.aws_s3_bucket

    @aws_s3_bucket.setter
    def aws_s3_bucket(self, value: str) -> None:
        value = value.strip()
        if value != self._settings.aws_s3_bucket:
            self._settings.aws_s3_bucket = value
            self._save_setting("aws_s3_bucket")

    @property
    def aws_batch_queue(self) -> str:
        return self._settings.aws_batch_queue

    @aws_batch_queue.setter
    def aws_batch_queue(self, value: str) -> None:
        value = value.strip()
        if value != self._settings.aws_batch_queue:
            self._settings.aws_batch_queue = value
            self._save_setting("aws_batch_queue")

    @property
    def aws_batch_job_def(self) -> str:
        return self._settings.aws_batch_job_def

    @aws_batch_job_def.setter
    def aws_batch_job_def(self, value: str) -> None:
        value = value.strip()
        if value != self._settings.aws_batch_job_def:
            self._settings.aws_batch_job_def = value
            self._save_setting("aws_batch_job_def")

    @property
    def model(self) -> str:
        return self._settings.model

    @model.setter
    def model(self, value: str) -> None:
        if value not in DEMUCS_MODELS:
            raise ValueError(f"Invalid model: {value}")
        if value != self._settings.model:
            self._settings.model = value
            self._save_setting("model")

    @property
    def two_stems(self) -> Optional[str]:
        return self._settings.two_stems

    @two_stems.setter
    def two_stems(self, value: Optional[str]) -> None:
        if value is not None and value not in {"vocals", "drums", "bass", "other"}:
            raise ValueError(f"Invalid two_stems: {value}")
        if value != self._settings.two_stems:
            self._settings.two_stems = value
            self._save_setting("two_stems")

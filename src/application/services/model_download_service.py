"""
Model Download Service

Fetches the model manifest from the EchoZero models worker and downloads
model files to the local models directory with progress tracking and
SHA256 verification.

Auth: validates the local license lease (proves active subscription)
and sends only the APP_SECRET to the worker. No per-request Memberstack
token exchange -- the lease is the single source of truth, valid until
the billing cycle ends.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from src.infrastructure.auth.license_lease import LeaseStatus
from src.infrastructure.auth.token_storage import TokenStorage
from src.utils.message import Log
from src.utils.paths import get_models_dir


@dataclass
class RemoteModel:
    """A model available for download from the model store."""

    id: str
    name: str
    description: str
    filename: str
    size_bytes: int
    sha256: str
    version: str
    classification_mode: str = "multiclass"
    classes: List[str] = field(default_factory=list)
    architecture: str = "cnn"
    created_at: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RemoteModel":
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            filename=d["filename"],
            size_bytes=d.get("size_bytes", 0),
            sha256=d.get("sha256", ""),
            version=d.get("version", "1.0.0"),
            classification_mode=d.get("classification_mode", "multiclass"),
            classes=d.get("classes", []),
            architecture=d.get("architecture", "cnn"),
            created_at=d.get("created_at", ""),
        )

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024) if self.size_bytes else 0.0


class ModelDownloadError(Exception):
    """Raised when a model download or manifest fetch fails."""
    pass


class ModelDownloadService:
    """
    Manages fetching the remote model manifest and downloading model files.

    Auth: the local license lease is checked before each request (proves
    active subscription without a server round-trip).  Only the APP_SECRET
    is sent to the worker -- no per-user tokens leave the machine.
    """

    def __init__(
        self,
        models_worker_url: str,
        app_secret: str = "",
        token_storage: Optional[TokenStorage] = None,
    ):
        self._base_url = models_worker_url.rstrip("/")
        self._app_secret = app_secret
        self._token_storage = token_storage or TokenStorage()

    def _check_lease(self) -> None:
        """Verify the local license lease is valid before making requests."""
        from src.infrastructure.auth.license_lease import LicenseLeaseManager

        manager = LicenseLeaseManager(self._token_storage, app_secret=self._app_secret)
        status = manager.validate_lease()

        if status == LeaseStatus.VALID:
            return

        if status == LeaseStatus.EXPIRED:
            raise ModelDownloadError(
                "Your subscription lease has expired. "
                "Please connect to the internet to renew."
            )
        if status == LeaseStatus.MISSING:
            raise ModelDownloadError(
                "Not logged in. Please log in to download models."
            )
        raise ModelDownloadError(
            "License validation failed. Please log in again."
        )

    def _auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._app_secret:
            headers["X-App-Token"] = self._app_secret
        return headers

    def fetch_available_models(self) -> List[RemoteModel]:
        """
        Fetch the model manifest from the remote worker.

        Returns:
            List of RemoteModel entries available for download.

        Raises:
            ModelDownloadError: On network, auth, or parse failure.
        """
        try:
            self._check_lease()
            headers = self._auth_headers()

            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    f"{self._base_url}/manifest",
                    headers=headers,
                )

            if resp.status_code == 401:
                raise ModelDownloadError("Authentication failed. Please log in again.")
            if resp.status_code == 403:
                raise ModelDownloadError("Active subscription required to access models.")
            if resp.status_code != 200:
                raise ModelDownloadError(f"Server returned HTTP {resp.status_code}")

            data = resp.json()
            models = [RemoteModel.from_dict(m) for m in data.get("models", [])]
            Log.info(f"ModelDownloadService: Fetched manifest with {len(models)} model(s)")
            return models

        except ModelDownloadError:
            raise
        except httpx.RequestError as exc:
            raise ModelDownloadError(f"Network error: {exc}") from exc
        except Exception as exc:
            raise ModelDownloadError(f"Failed to fetch manifest: {exc}") from exc

    def get_download_status(self, models: List[RemoteModel]) -> Dict[str, bool]:
        """
        Check which remote models are already downloaded locally.

        Returns:
            Dict mapping model id to True (installed) / False (not installed).
        """
        models_dir = get_models_dir()
        status: Dict[str, bool] = {}
        for model in models:
            local_path = self._find_local_model(model, models_dir)
            status[model.id] = local_path is not None
        return status

    def download_model(
        self,
        model: RemoteModel,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Download a model file from the worker and save to the models directory.

        Args:
            model: The RemoteModel to download.
            progress_callback: Called with (bytes_downloaded, total_bytes).

        Returns:
            Path to the downloaded file.

        Raises:
            ModelDownloadError: On network, auth, hash mismatch, or write failure.
        """
        models_dir = get_models_dir()
        dest_path = models_dir / model.filename

        if dest_path.exists():
            if model.sha256 and self._verify_hash(dest_path, model.sha256):
                Log.info(f"ModelDownloadService: {model.filename} already exists and hash matches")
                return str(dest_path)
            Log.warning(f"ModelDownloadService: {model.filename} exists but hash mismatch, re-downloading")

        tmp_path = dest_path.with_suffix(".pth.download")

        try:
            self._check_lease()
            headers = self._auth_headers()

            with httpx.Client(timeout=httpx.Timeout(15.0, read=300.0)) as client:
                with client.stream(
                    "GET",
                    f"{self._base_url}/download/{model.filename}",
                    headers=headers,
                ) as resp:
                    if resp.status_code == 401:
                        raise ModelDownloadError("Authentication failed.")
                    if resp.status_code == 403:
                        raise ModelDownloadError("Active subscription required.")
                    if resp.status_code == 404:
                        raise ModelDownloadError(f"Model file not found on server: {model.filename}")
                    if resp.status_code != 200:
                        raise ModelDownloadError(f"Server returned HTTP {resp.status_code}")

                    total = int(resp.headers.get("Content-Length", 0)) or model.size_bytes
                    downloaded = 0

                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=256 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback:
                                progress_callback(downloaded, total)

            if model.sha256:
                if not self._verify_hash(tmp_path, model.sha256):
                    tmp_path.unlink(missing_ok=True)
                    raise ModelDownloadError(
                        f"SHA256 mismatch for {model.filename}. "
                        "The download may be corrupted. Please try again."
                    )

            tmp_path.rename(dest_path)
            Log.info(f"ModelDownloadService: Downloaded {model.filename} ({dest_path})")
            return str(dest_path)

        except ModelDownloadError:
            raise
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise ModelDownloadError(f"Download failed: {exc}") from exc

    def delete_model(self, model: RemoteModel) -> bool:
        """Remove a downloaded model file."""
        models_dir = get_models_dir()
        local_path = self._find_local_model(model, models_dir)
        if local_path and local_path.exists():
            local_path.unlink()
            Log.info(f"ModelDownloadService: Deleted {local_path}")
            return True
        return False

    def _find_local_model(self, model: RemoteModel, models_dir: Path) -> Optional[Path]:
        """Find the local path for a remote model if it exists."""
        candidate = models_dir / model.filename
        if candidate.exists():
            return candidate
        for pth in models_dir.rglob("*.pth"):
            if pth.name == model.filename:
                return pth
        return None

    @staticmethod
    def _verify_hash(path: Path, expected_sha256: str) -> bool:
        """Verify file SHA256 matches expected value."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(256 * 1024)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest() == expected_sha256


def create_model_download_service() -> ModelDownloadService:
    """
    Factory that builds a ModelDownloadService from environment variables.

    Reads MODELS_WORKER_URL and MEMBERSTACK_APP_SECRET the same way
    the auth system reads its config in main_qt.py.
    """
    models_url = (
        (os.getenv("MODELS_WORKER_URL") or "").strip()
        or "https://echozero-models.speeoflight.workers.dev"
    )
    app_secret = (os.getenv("MEMBERSTACK_APP_SECRET") or "").strip()
    return ModelDownloadService(
        models_worker_url=models_url,
        app_secret=app_secret,
    )

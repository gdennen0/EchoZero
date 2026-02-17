"""
Secure Token Storage

Stores and retrieves Memberstack authentication credentials and license lease.
Uses a single file in the user data directory with mode 0o600 (user read/write only).
No keychain is used, so users are never prompted for their macOS password.

- Path: Application Support/EchoZero/auth_data.json (or platform equivalent)
- Permissions: 0o600 so only the current user can read the file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.message import Log
from src.utils.paths import get_user_data_dir


# Keys in the auth data file
_KEY_MEMBER_ID = "member_id"
_KEY_TOKEN = "token"
_KEY_MEMBER_DATA = "member_data"
_KEY_LICENSE_LEASE = "license_lease"


def _auth_file_path() -> Path:
    """Path to the auth data file (user data dir, not world-readable)."""
    return get_user_data_dir() / "auth_data.json"


def _read_auth_file() -> Dict[str, Any]:
    """Read auth file; return empty dict if missing or invalid."""
    path = _auth_file_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        Log.warning(f"TokenStorage: Could not read auth file: {e}")
        return {}


def _write_auth_file(data: Dict[str, Any]) -> bool:
    """Write auth file with mode 0o600 (user read/write only)."""
    path = _auth_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=0), encoding="utf-8")
        path.chmod(0o600)
        return True
    except OSError as e:
        Log.error(f"TokenStorage: Could not write auth file: {e}")
        return False


class TokenStorage:
    """
    File-based credential and lease storage (no keychain).

    Stores Memberstack member_id, token, and optional member_data, plus the
    license lease JSON. All in one file under Application Support with mode 0o600.
    """

    def store_credentials(
        self,
        member_id: str,
        token: str,
        member_data: Optional[Dict] = None,
    ) -> bool:
        """Store authentication credentials."""
        try:
            data = _read_auth_file()
            data[_KEY_MEMBER_ID] = member_id
            data[_KEY_TOKEN] = token
            data[_KEY_MEMBER_DATA] = member_data if member_data is not None else data.get(_KEY_MEMBER_DATA)
            if not _write_auth_file(data):
                return False
            Log.info("TokenStorage: Credentials stored.")
            return True
        except Exception as e:
            Log.error(f"TokenStorage: Failed to store credentials: {e}")
            return False

    def get_credentials(self) -> Optional[Dict[str, Any]]:
        """Retrieve stored credentials."""
        try:
            data = _read_auth_file()
            member_id = data.get(_KEY_MEMBER_ID)
            if not member_id:
                return None
            result = {
                "member_id": member_id,
                "token": data.get(_KEY_TOKEN, ""),
            }
            if _KEY_MEMBER_DATA in data and data[_KEY_MEMBER_DATA]:
                result["member_data"] = data[_KEY_MEMBER_DATA]
            return result
        except Exception as e:
            Log.error(f"TokenStorage: Failed to retrieve credentials: {e}")
            return None

    def clear_credentials(self) -> bool:
        """Remove stored credentials (leave lease if present)."""
        try:
            data = _read_auth_file()
            for key in (_KEY_MEMBER_ID, _KEY_TOKEN, _KEY_MEMBER_DATA):
                data.pop(key, None)
            _write_auth_file(data)
            Log.info("TokenStorage: Credentials cleared.")
            return True
        except Exception as e:
            Log.error(f"TokenStorage: Failed to clear credentials: {e}")
            return False

    def store_lease(self, lease_json: str) -> bool:
        """Store license lease JSON."""
        try:
            data = _read_auth_file()
            data[_KEY_LICENSE_LEASE] = lease_json
            if not _write_auth_file(data):
                return False
            return True
        except Exception as e:
            Log.error(f"TokenStorage: Failed to store lease: {e}")
            return False

    def get_lease(self) -> Optional[str]:
        """Retrieve stored license lease JSON."""
        try:
            data = _read_auth_file()
            return data.get(_KEY_LICENSE_LEASE)
        except Exception as e:
            Log.error(f"TokenStorage: Failed to retrieve lease: {e}")
            return None

    def clear_lease(self) -> bool:
        """Remove stored lease."""
        try:
            data = _read_auth_file()
            data.pop(_KEY_LICENSE_LEASE, None)
            _write_auth_file(data)
            return True
        except Exception as e:
            Log.error(f"TokenStorage: Failed to clear lease: {e}")
            return False

    def has_credentials(self) -> bool:
        """Check if credentials are stored."""
        try:
            data = _read_auth_file()
            return bool(data.get(_KEY_MEMBER_ID))
        except Exception:
            return False

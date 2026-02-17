"""
Memberstack Authentication Service

Verifies member identity via a server-side Cloudflare Worker proxy.
The Memberstack Admin API key stays on the server -- this client never sees it.

Production flow:
    Desktop App  --->  Cloudflare Worker  --->  Memberstack Admin API
                POST /verify {member_id, token}

The worker validates that:
  1) token is valid,
  2) token member id matches member_id, and
  3) member has an active allowed plan.
"""
from typing import Optional, Dict, Any

import httpx

from src.utils.message import Log


class MemberstackAuthError(Exception):
    """Raised when Memberstack authentication or verification fails."""
    pass


class MemberstackAuth:
    """
    Verifies EchoZero members via a server-side verification proxy.
    
    The proxy (Cloudflare Worker) holds the Memberstack Admin API key.
    This client only needs the proxy URL and an optional app secret.
    
    Usage:
        auth = MemberstackAuth(
            verify_url="https://echozero-auth.your-subdomain.workers.dev",
            app_secret="your_app_secret",
        )
        member = auth.verify_member("mem_abc123", token="jwt_here")
        if member:
            print(f"Verified: {member['email']}")
    """
    
    def __init__(self, verify_url: str, app_secret: str = ""):
        """
        Initialize with the verification proxy URL.
        
        Args:
            verify_url: URL of the Cloudflare Worker verification endpoint
                        (e.g., "https://echozero-auth.your-subdomain.workers.dev").
            app_secret: Optional shared secret sent as X-App-Token header
                        to authenticate requests to the worker.
        """
        self._verify_url = verify_url.rstrip("/") if verify_url else ""
        self._app_secret = app_secret
        self._last_error_kind = ""
        self._last_error_message = ""
        
        if not self._verify_url:
            Log.warning("MemberstackAuth: No verify URL provided. Verification will fail.")
    
    def verify_member(self, member_id: str, token: str = "") -> Optional[Dict[str, Any]]:
        """
        Verify a member exists and is active via the server-side proxy.
        
        Args:
            member_id: Memberstack member ID (e.g., "mem_abc123").
            token: Memberstack access token (JWT) from browser login.
            
        Returns:
            Member data dict with keys: id, email, plan_connections, etc.
            Returns None if member is invalid or verification fails.
        """
        self._last_error_kind = ""
        self._last_error_message = ""

        if not self._verify_url:
            Log.error("MemberstackAuth: Cannot verify -- no verify URL configured.")
            self._set_error("configuration", "Missing verify URL")
            return None
        
        if not member_id:
            Log.error("MemberstackAuth: Cannot verify -- no member_id provided.")
            self._set_error("invalid_request", "Missing member_id")
            return None

        if not token:
            Log.error("MemberstackAuth: Cannot verify -- no token provided.")
            self._set_error("invalid_request", "Missing member token")
            return None
        
        try:
            headers = {"Content-Type": "application/json"}
            if self._app_secret:
                headers["X-App-Token"] = self._app_secret
            
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f"{self._verify_url}/verify",
                    json={"member_id": member_id, "token": token},
                    headers=headers,
                )
            
            data = response.json()
            
            if response.status_code == 401:
                reason = data.get("error", "Unauthorized")
                # Distinguish token failures from app-secret failures for lease decisions.
                if "token" in reason.lower():
                    self._set_error("token_invalid", reason)
                    Log.warning(f"MemberstackAuth: Token verification failed: {reason}")
                else:
                    self._set_error("unauthorized", reason)
                    Log.error("MemberstackAuth: Unauthorized -- invalid app secret.")
                return None
            
            if response.status_code == 404:
                Log.warning(f"MemberstackAuth: Member {member_id} not found.")
                self._set_error("member_not_found", data.get("error", "Member not found"))
                return None
            
            if response.status_code == 403:
                # Member exists but does not have the required plan
                reason = data.get("error", "No active plan")
                Log.warning(f"MemberstackAuth: Access denied for {member_id}: {reason}")
                self._set_error("access_denied", reason)
                return None
            
            if response.status_code != 200:
                Log.error(f"MemberstackAuth: Verification proxy returned {response.status_code}.")
                self._set_error("service_unavailable", f"HTTP {response.status_code}")
                return None
            
            if not data.get("verified"):
                reason = data.get("error", "unknown")
                Log.warning(f"MemberstackAuth: Member not verified: {reason}")
                self._set_error("not_verified", reason)
                return None
            
            member_info = {
                "id": data.get("id", member_id),
                "email": data.get("email", ""),
                "plan_connections": data.get("plan_connections", []),
                "billing_period_end": data.get("billing_period_end", ""),
                "created_at": data.get("created_at", ""),
                "verified": True,
            }
            
            Log.info(f"MemberstackAuth: Member verified: {member_info['email']}")
            return member_info
            
        except httpx.RequestError as e:
            Log.error(f"MemberstackAuth: Network error reaching verification proxy: {e}")
            self._set_error("network_error", str(e))
            return None
        except Exception as e:
            Log.error(f"MemberstackAuth: Unexpected error: {e}")
            self._set_error("unexpected_error", str(e))
            return None
    
    def is_member_active(self, member_id: str, token: str = "") -> bool:
        """
        Check if a member exists and is verified.
        
        Args:
            member_id: Memberstack member ID.
            
        Returns:
            True if member is verified, False otherwise.
        """
        member = self.verify_member(member_id, token=token)
        return member is not None

    @property
    def last_error_kind(self) -> str:
        """Most recent verification failure type."""
        return self._last_error_kind

    @property
    def last_error_message(self) -> str:
        """Most recent verification failure message."""
        return self._last_error_message

    def should_allow_cached_lease(self) -> bool:
        """
        Whether a failure should still permit use of an existing valid lease.

        We permit cached lease usage for transient failures (network/service)
        and token expiration. Access-denied responses do not qualify.
        """
        return self._last_error_kind in {
            "network_error",
            "service_unavailable",
            "token_invalid",
        }

    def _set_error(self, kind: str, message: str) -> None:
        self._last_error_kind = kind
        self._last_error_message = message

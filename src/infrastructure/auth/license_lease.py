"""
License Lease Manager

Manages offline license leases for EchoZero's subscription system.

When a user successfully authenticates online, a signed lease is granted
and stored in the OS keychain. The lease is anchored to the member's
billing period end when available, with a hard fallback window when it is
not available. While the app is running, periodic server checks can renew
the lease. If the subscription lapses or the lease expires, the user must
re-authenticate.

Industry-standard pattern used by Adobe Creative Cloud, JetBrains, etc.

Security:
    - Lease data is HMAC-SHA256 signed with a device-derived key
    - Leases are non-transferable between machines
    - Clock manipulation is detected and forces a server re-check
"""
import hashlib
import hmac
import json
import platform
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from src.utils.message import Log


# -- Configuration ----------------------------------------------------------

# Hard fallback lease duration if billing period end is unavailable
LEASE_HARD_FALLBACK_DAYS = 30

# How often we attempt a silent server re-check (while the app is running)
SERVER_CHECK_INTERVAL_HOURS = 24

# Internal salt mixed into the HMAC key derivation
_HMAC_SALT = "echozero-lease-v1"


class LeaseStatus(Enum):
    """Result of validating a stored lease."""
    VALID = "valid"                      # Lease exists, signature OK, not expired
    EXPIRED = "expired"                  # Lease exists but past expiration date
    INVALID_SIGNATURE = "invalid_sig"    # Lease data was tampered with
    MISSING = "missing"                  # No lease data found
    CLOCK_TAMPERED = "clock_tampered"    # System clock appears to have been set back


@dataclass
class LicenseLeaseData:
    """
    Signed lease granting offline access for a limited period.

    All timestamps are ISO 8601 UTC strings.
    """
    member_id: str
    email: str
    lease_granted_at: str       # When the lease was created / last renewed
    lease_expires_at: str       # Effective lease expiry used for validation
    last_server_check_at: str   # Last successful server verification timestamp
    subscription_status: str    # "active", "expired", "grace"
    billing_period_end_at: str = ""   # Preferred expiry anchor from subscription data
    hard_fallback_expires_at: str = ""  # Safety fallback expiry

    # Signature is stored alongside but excluded from the signing payload
    signature: str = ""


class LicenseLeaseManager:
    """
    Creates, validates, and revokes license leases.

    Leases are signed JSON blobs stored in the OS keychain via TokenStorage.
    The HMAC key is derived from a combination of machine identity, member ID,
    and an application secret so that leases cannot be copied between machines
    or tampered with.

    Usage:
        from src.infrastructure.auth.token_storage import TokenStorage
        manager = LicenseLeaseManager(TokenStorage(), app_secret="...")
        manager.grant_lease("mem_abc", "user@example.com")
        status = manager.validate_lease()
        if status == LeaseStatus.VALID:
            ...
    """

    def __init__(self, token_storage, app_secret: str = ""):
        """
        Args:
            token_storage: TokenStorage instance for keychain read/write.
            app_secret: Shared application secret (from env / config).
        """
        self._storage = token_storage
        self._app_secret = app_secret

    # -- Public API ---------------------------------------------------------

    def grant_lease(
        self,
        member_id: str,
        email: str,
        billing_period_end_at: Optional[str] = None,
    ) -> bool:
        """
        Create a new lease after a successful server verification.

        Stores the signed lease in the OS keychain.

        Args:
            member_id: Memberstack member ID.
            email: Member email address.

        Returns:
            True if the lease was stored successfully.
        """
        now = datetime.now(timezone.utc)
        fallback_expires = now + timedelta(days=LEASE_HARD_FALLBACK_DAYS)
        effective_expires = self._resolve_effective_expiry(now, billing_period_end_at, fallback_expires)

        lease = LicenseLeaseData(
            member_id=member_id,
            email=email,
            lease_granted_at=now.isoformat(),
            lease_expires_at=effective_expires.isoformat(),
            last_server_check_at=now.isoformat(),
            subscription_status="active",
            billing_period_end_at=billing_period_end_at or "",
            hard_fallback_expires_at=fallback_expires.isoformat(),
        )

        lease.signature = self._sign(lease, member_id)

        lease_json = json.dumps(asdict(lease))
        success = self._storage.store_lease(lease_json)

        if success:
            Log.info(
                f"LicenseLeaseManager: Lease granted for {email}, "
                f"expires {effective_expires.strftime('%Y-%m-%d %H:%M UTC')}"
            )
        else:
            Log.error("LicenseLeaseManager: Failed to store lease in keychain.")

        return success

    def renew_lease(self, billing_period_end_at: Optional[str] = None) -> bool:
        """
        Renew the existing lease after a successful periodic server check.

        Updates last_server_check_at, extends expiration, and re-signs.

        Returns:
            True if renewal succeeded, False if no valid lease exists.
        """
        lease = self._load_lease()
        if lease is None:
            return False

        now = datetime.now(timezone.utc)
        fallback_expires = now + timedelta(days=LEASE_HARD_FALLBACK_DAYS)
        # Keep the prior billing anchor unless a fresh one is supplied.
        billing_anchor = (
            billing_period_end_at
            if billing_period_end_at is not None
            else (lease.billing_period_end_at or None)
        )
        effective_expires = self._resolve_effective_expiry(now, billing_anchor, fallback_expires)

        lease.last_server_check_at = now.isoformat()
        lease.lease_expires_at = effective_expires.isoformat()
        lease.subscription_status = "active"
        lease.billing_period_end_at = billing_anchor or ""
        lease.hard_fallback_expires_at = fallback_expires.isoformat()
        lease.signature = self._sign(lease, lease.member_id)

        lease_json = json.dumps(asdict(lease))
        success = self._storage.store_lease(lease_json)

        if success:
            Log.info("LicenseLeaseManager: Lease renewed silently.")
        return success

    def validate_lease(self) -> LeaseStatus:
        """
        Check whether the stored lease is valid for continued offline use.

        Validates:
            1. Lease exists in keychain
            2. HMAC signature is intact (no tampering)
            3. Expiration date has not passed
            4. System clock has not been set backward

        Returns:
            LeaseStatus indicating the result.
        """
        lease = self._load_lease_raw()
        if lease is None:
            return LeaseStatus.MISSING

        # Verify HMAC signature
        expected_sig = self._sign(lease, lease.member_id)
        if not hmac.compare_digest(lease.signature, expected_sig):
            Log.warning("LicenseLeaseManager: Lease signature mismatch -- possible tampering.")
            return LeaseStatus.INVALID_SIGNATURE

        now = datetime.now(timezone.utc)

        # Clock manipulation check: if current time is before last_server_check,
        # the clock was likely set backward.
        try:
            last_check = datetime.fromisoformat(lease.last_server_check_at)
            if now < last_check - timedelta(minutes=5):
                Log.warning(
                    "LicenseLeaseManager: System clock appears to be set backward. "
                    "Forcing server re-check."
                )
                return LeaseStatus.CLOCK_TAMPERED
        except (ValueError, TypeError):
            return LeaseStatus.INVALID_SIGNATURE

        # Expiration check
        try:
            expires = datetime.fromisoformat(self._get_effective_expiry(lease))
            if now > expires:
                Log.info("LicenseLeaseManager: Lease has expired.")
                return LeaseStatus.EXPIRED
        except (ValueError, TypeError):
            return LeaseStatus.INVALID_SIGNATURE

        return LeaseStatus.VALID

    def needs_server_check(self) -> bool:
        """
        Determine if a server re-check is due.

        Returns True if more than SERVER_CHECK_INTERVAL_HOURS have passed
        since the last successful server verification.
        """
        lease = self._load_lease()
        if lease is None:
            return True

        try:
            last_check = datetime.fromisoformat(lease.last_server_check_at)
            now = datetime.now(timezone.utc)
            elapsed = now - last_check
            return elapsed > timedelta(hours=SERVER_CHECK_INTERVAL_HOURS)
        except (ValueError, TypeError):
            return True

    def revoke_lease(self) -> bool:
        """
        Remove the stored lease (logout or subscription ended).

        Returns:
            True if cleared successfully.
        """
        success = self._storage.clear_lease()
        if success:
            Log.info("LicenseLeaseManager: Lease revoked.")
        return success

    def get_lease_info(self) -> Optional[LicenseLeaseData]:
        """
        Load and return the current lease data (if valid).

        Returns:
            LicenseLeaseData or None if no valid lease exists.
        """
        return self._load_lease()

    def get_days_remaining(self) -> Optional[int]:
        """
        How many days remain on the current lease.

        Returns:
            Number of days remaining, or None if no valid lease.
        """
        lease = self._load_lease()
        if lease is None:
            return None

        try:
            expires = datetime.fromisoformat(self._get_effective_expiry(lease))
            now = datetime.now(timezone.utc)
            remaining = (expires - now).days
            return max(0, remaining)
        except (ValueError, TypeError):
            return None

    # -- Internal -----------------------------------------------------------

    def _load_lease(self) -> Optional[LicenseLeaseData]:
        """Load and validate the lease from keychain. Returns None on any failure."""
        lease = self._load_lease_raw()
        if lease is None:
            return None

        # Quick signature check
        expected_sig = self._sign(lease, lease.member_id)
        if not hmac.compare_digest(lease.signature, expected_sig):
            return None

        return lease

    def _load_lease_raw(self) -> Optional[LicenseLeaseData]:
        """Load the lease from keychain without signature validation."""
        lease_json = self._storage.get_lease()
        if not lease_json:
            return None

        try:
            data = json.loads(lease_json)
            return LicenseLeaseData(**data)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            Log.warning(f"LicenseLeaseManager: Failed to parse stored lease: {e}")
            return None

    def _sign(self, lease: LicenseLeaseData, member_id: str) -> str:
        """
        Compute HMAC-SHA256 signature for the lease payload.

        The signing key is derived from machine identity + member_id + app_secret
        so the lease cannot be moved between machines or users.
        """
        key = self._derive_key(member_id)

        # Build the payload from all fields except signature
        payload = (
            f"{lease.member_id}|"
            f"{lease.email}|"
            f"{lease.lease_granted_at}|"
            f"{lease.lease_expires_at}|"
            f"{lease.last_server_check_at}|"
            f"{lease.subscription_status}|"
            f"{lease.billing_period_end_at}|"
            f"{lease.hard_fallback_expires_at}"
        )

        return hmac.new(
            key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _derive_key(self, member_id: str) -> str:
        """
        Derive an HMAC key from machine identity, member ID, and app secret.

        Uses uuid.getnode() (MAC address) and platform.node() (hostname)
        as machine fingerprints. This makes the lease non-transferable.
        """
        machine_id = f"{uuid.getnode()}-{platform.node()}"
        raw = f"{_HMAC_SALT}:{machine_id}:{member_id}:{self._app_secret}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _resolve_effective_expiry(
        self,
        now: datetime,
        billing_period_end_at: Optional[str],
        fallback_expires: datetime,
    ) -> datetime:
        """
        Resolve lease expiry:
          - Use billing period end when valid and in the future.
          - Otherwise, use hard fallback expiry.
        """
        if billing_period_end_at:
            try:
                billing_end = datetime.fromisoformat(billing_period_end_at)
                if billing_end > now:
                    return billing_end
            except (ValueError, TypeError):
                pass
        return fallback_expires

    def _get_effective_expiry(self, lease: LicenseLeaseData) -> str:
        """
        Get effective lease expiry for validation.

        Uses billing-period-based expiry when present, otherwise falls back to
        hard fallback expiry or legacy lease_expires_at.
        """
        if lease.billing_period_end_at:
            return lease.billing_period_end_at
        if lease.hard_fallback_expires_at:
            return lease.hard_fallback_expires_at
        # Backward compatibility with legacy leases.
        return lease.lease_expires_at

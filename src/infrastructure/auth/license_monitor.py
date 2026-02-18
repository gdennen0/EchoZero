"""
License Monitor

Background service that periodically re-validates the license lease
while EchoZero is running. Uses a QTimer to check every hour whether
a server re-check is due (every 24 hours). If the subscription has
lapsed or the offline grace period has expired, it emits Qt signals
so the UI can prompt the user.

Industry-standard pattern: silent lease renewal while online, graceful
lock-out when the subscription ends or the offline period expires.
"""
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from src.infrastructure.auth.license_lease import (
    LicenseLeaseManager,
    LeaseStatus,
)
from src.infrastructure.auth.memberstack_auth import MemberstackAuth
from src.infrastructure.auth.token_storage import TokenStorage
from src.utils.message import Log


# How often the timer fires to check if a server re-check is needed (ms)
_CHECK_INTERVAL_MS = 60 * 60 * 1000  # 1 hour

# Warning threshold: emit license_warning when this many days remain
_WARNING_THRESHOLD_DAYS = 7


class LicenseMonitor(QObject):
    """
    Background license lease monitor.

    Runs on the Qt main thread via QTimer. Periodically checks whether
    a server re-validation is due and silently renews the lease. Emits
    signals when the license is about to expire or has expired.

    Signals:
        license_expired: Emitted when the lease has expired and the user
            must re-authenticate. Carries a reason string.
        license_warning: Emitted when the lease is close to expiring.
            Carries the number of days remaining.

    Usage:
        monitor = LicenseMonitor(ms_auth, token_storage, lease_manager)
        monitor.license_expired.connect(on_license_expired)
        monitor.license_warning.connect(on_license_warning)
        monitor.start()
    """

    license_expired = pyqtSignal(str)   # reason message
    license_warning = pyqtSignal(int)   # days remaining

    def __init__(
        self,
        ms_auth: MemberstackAuth,
        token_storage: TokenStorage,
        lease_manager: LicenseLeaseManager,
        parent: QObject = None,
    ):
        super().__init__(parent)
        self._ms_auth = ms_auth
        self._token_storage = token_storage
        self._lease_manager = lease_manager

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)

    # -- Public API ---------------------------------------------------------

    def start(self):
        """Start the periodic license check timer."""
        self._timer.start(_CHECK_INTERVAL_MS)
        Log.info("LicenseMonitor: Started (checking every 1 hour).")

    def stop(self):
        """Stop the periodic license check timer."""
        self._timer.stop()
        Log.info("LicenseMonitor: Stopped.")

    def force_check(self):
        """Run an immediate license check (e.g., after waking from sleep)."""
        self._on_timer()

    # -- Internal -----------------------------------------------------------

    def _on_timer(self):
        """
        Timer callback. Checks lease validity and attempts server
        re-check if due.
        """
        # First, validate the lease locally
        status = self._lease_manager.validate_lease()

        if status == LeaseStatus.EXPIRED:
            Log.info("LicenseMonitor: Lease expired.")
            self.license_expired.emit(
                "Your offline license period has expired. "
                "Please connect to the internet and re-activate."
            )
            return

        if status == LeaseStatus.INVALID_SIGNATURE:
            Log.warning("LicenseMonitor: Lease signature invalid.")
            self.license_expired.emit(
                "License validation failed. Please log in again."
            )
            return

        if status == LeaseStatus.MISSING:
            Log.warning("LicenseMonitor: No lease found during monitor check.")
            self.license_expired.emit(
                "No active license found. Please log in."
            )
            return

        if status == LeaseStatus.CLOCK_TAMPERED:
            Log.warning("LicenseMonitor: Clock tampering detected, forcing re-check.")
            self._attempt_server_recheck(force=True)
            return

        # Lease is valid -- check if a server re-check is due
        if self._lease_manager.needs_server_check():
            self._attempt_server_recheck(force=False)
        else:
            # Check if we should emit a warning
            days = self._lease_manager.get_days_remaining()
            if days is not None and days <= _WARNING_THRESHOLD_DAYS:
                Log.info(
                    f"LicenseMonitor: Lease expires in {days} days, "
                    f"emitting warning."
                )
                self.license_warning.emit(days)

    def _attempt_server_recheck(self, force: bool = False):
        """
        Try to silently re-validate the subscription with the server.

        Args:
            force: If True, treat failure as expiration (clock tamper case).
        """
        credentials = self._token_storage.get_credentials()
        member_id = credentials.get("member_id", "") if credentials else ""
        token = credentials.get("token", "") if credentials else ""

        if not member_id:
            Log.warning("LicenseMonitor: No member_id for server re-check.")
            if force:
                self.license_expired.emit(
                    "License verification required. Please log in."
                )
            return

        Log.info("LicenseMonitor: Attempting silent server re-check...")
        member_info = self._ms_auth.verify_member(member_id, token=token)

        if member_info:
            # Server confirmed subscription is active -- renew silently
            self._lease_manager.renew_lease(
                billing_period_end_at=member_info.get("billing_period_end", ""),
            )
            Log.info("LicenseMonitor: Lease renewed silently.")
        elif member_info is None:
            # Hard failure mode: any failed re-check requires re-activation.
            kind = self._ms_auth.last_error_kind
            Log.info(f"LicenseMonitor: Re-check failed ({kind}). Requiring re-activation.")
            if kind in ("invalid_app_token", "unauthorized"):
                self.license_expired.emit(
                    "MEMBERSTACK_APP_SECRET does not match the verification server. "
                    "Update .env with the correct secret (see .env.example)."
                )
            else:
                self.license_expired.emit(
                    "Your subscription could not be verified. "
                    "Please log in again to continue."
                )

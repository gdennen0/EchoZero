"""
License Expired Dialog

Modal dialog shown when the license lease expires during an active session.
Gives the user two options:

1. "Save Work and Exit" -- saves the current project, then closes the app.
2. "Reactivate Now"     -- opens the browser-based login flow to renew.

This dialog blocks interaction with the main window until the user takes
action, preventing continued use with an expired license.
"""
import webbrowser
from typing import Optional
from urllib.parse import quote

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from src.infrastructure.auth.memberstack_auth import MemberstackAuth, generate_link_code
from src.infrastructure.auth.token_storage import TokenStorage
from src.infrastructure.auth.license_lease import LicenseLeaseManager
from src.infrastructure.auth.local_auth_server import LocalAuthServer
from src.infrastructure.auth.auth_url_handler import (
    register_pending_auth_server,
    unregister_pending_auth_server,
)
from src.utils.message import Log
from ui.qt_gui.design_system import Colors, Spacing, border_radius

LINK_POLL_INTERVAL_MS = 2000
LINK_TIMEOUT_SEC = 300


class LicenseExpiredDialog(QDialog):
    """
    Modal dialog for handling mid-session license expiration.

    Blocks the application until the user either re-activates their
    subscription or saves their work and exits.

    Returns:
        QDialog.Accepted if reactivation succeeded (lease renewed).
        QDialog.Rejected if user chose to save and exit.
    """

    # URL of the Webflow desktop login page
    LOGIN_PAGE_URL = "https://www.speedoflight.dev/desktop-login"

    def __init__(
        self,
        reason: str,
        ms_auth: MemberstackAuth,
        token_storage: TokenStorage,
        lease_manager: LicenseLeaseManager,
        parent: Optional[QWidget] = None,
    ):
        """
        Args:
            reason: Human-readable explanation of why the license expired.
            ms_auth: MemberstackAuth instance for server verification.
            token_storage: TokenStorage instance for credential persistence.
            lease_manager: LicenseLeaseManager for granting a new lease.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._reason = reason
        self._ms_auth = ms_auth
        self._token_storage = token_storage
        self._lease_manager = lease_manager
        self._auth_server: Optional[LocalAuthServer] = None
        self._poll_timer: Optional[QTimer] = None
        self._link_code: Optional[str] = None
        self._link_elapsed_sec: float = 0.0

        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog UI."""
        self.setWindowTitle("License Expired")
        self.setFixedSize(480, 380)
        self.setModal(True)

        # Prevent closing via window controls -- user must choose an action
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)

        # -- Warning icon (text-based, no emoji) --
        icon_label = QLabel("!")
        icon_font = QFont()
        icon_font.setPixelSize(40)
        icon_font.setWeight(QFont.Weight.Bold)
        icon_label.setFont(icon_font)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setFixedHeight(56)
        icon_label.setStyleSheet(f"""
            color: {Colors.ACCENT_YELLOW.name()};
            background-color: {Colors.BG_LIGHT.name()};
            border-radius: {border_radius(28)};
        """)
        icon_label.setFixedWidth(56)
        # Center the icon
        icon_row = QHBoxLayout()
        icon_row.addStretch()
        icon_row.addWidget(icon_label)
        icon_row.addStretch()
        layout.addLayout(icon_row)

        layout.addSpacing(Spacing.SM)

        # -- Title --
        title_label = QLabel("License Expired")
        title_font = QFont()
        title_font.setFamily("SF Pro Display, Segoe UI, -apple-system, system-ui")
        title_font.setPixelSize(22)
        title_font.setWeight(QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
        layout.addWidget(title_label)

        # -- Reason / description --
        reason_label = QLabel(self._reason)
        reason_label.setWordWrap(True)
        reason_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reason_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY.name()};
            font-size: 13px;
            line-height: 1.4;
        """)
        layout.addWidget(reason_label)

        layout.addSpacing(Spacing.LG)

        # -- Reactivate Button (primary action) --
        self._reactivate_btn = QPushButton("Reactivate Now")
        self._reactivate_btn.setFixedHeight(44)
        self._reactivate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reactivate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: none;
                border-radius: {border_radius(6)};
                font-size: 15px;
                font-weight: 600;
                padding: 0 24px;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.lighter(115).name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.ACCENT_BLUE.darker(110).name()};
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_DISABLED.name()};
            }}
        """)
        self._reactivate_btn.clicked.connect(self._on_reactivate_clicked)
        layout.addWidget(self._reactivate_btn)

        # -- Status Label --
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY.name()};
            font-size: 13px;
        """)
        layout.addWidget(self._status_label)

        layout.addStretch()

        # -- Bottom: Save and Exit Button --
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        save_exit_btn = QPushButton("Save Work and Exit")
        save_exit_btn.setFixedWidth(160)
        save_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_exit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_SECONDARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                font-size: 13px;
                padding: 6px 16px;
            }}
            QPushButton:hover {{
                background-color: {Colors.HOVER.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
        """)
        save_exit_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(save_exit_btn)

        layout.addLayout(bottom_layout)

    # -- Reactivation Flow --------------------------------------------------

    def _on_reactivate_clicked(self):
        """Start the browser-based reactivation flow."""
        self._reactivate_btn.setEnabled(False)
        self._status_label.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 13px;"
        )
        self._status_label.setText("Opening browser...")

        if self._ms_auth.verify_url:
            self._start_server_mediated_flow()
        else:
            self._start_localhost_flow()

    def _start_server_mediated_flow(self):
        try:
            self._link_code = generate_link_code()
            self._link_elapsed_sec = 0.0
            worker_param = quote(self._ms_auth.verify_url, safe="")
            login_url = f"{self.LOGIN_PAGE_URL}?code={self._link_code}&worker={worker_param}"
            webbrowser.open(login_url)
            self._status_label.setText(
                "Waiting for login...\n"
                "Complete the login in your browser, then return here."
            )
            self._poll_timer = QTimer(self)
            self._poll_timer.timeout.connect(self._check_link_poll)
            self._poll_timer.start(LINK_POLL_INTERVAL_MS)
        except Exception as e:
            Log.error(f"LicenseExpiredDialog: Failed to start reactivation: {e}")
            self._show_error(f"Failed to open browser: {e}")
            self._reactivate_btn.setEnabled(True)

    def _start_localhost_flow(self):
        try:
            self._auth_server = LocalAuthServer()
            port, nonce = self._auth_server.start()
            register_pending_auth_server(self._auth_server)
            login_url = f"{self.LOGIN_PAGE_URL}?port={port}&nonce={nonce}"
            webbrowser.open(login_url)
            self._status_label.setText(
                "Waiting for login...\n"
                "Complete the login in your browser, then return here."
            )
            self._poll_timer = QTimer(self)
            self._poll_timer.timeout.connect(self._check_callback)
            self._poll_timer.start(500)
        except Exception as e:
            Log.error(f"LicenseExpiredDialog: Failed to start reactivation: {e}")
            self._show_error(f"Failed to open browser: {e}")
            self._reactivate_btn.setEnabled(True)

    def _check_link_poll(self):
        if not self._link_code:
            return
        self._link_elapsed_sec += LINK_POLL_INTERVAL_MS / 1000.0
        if self._link_elapsed_sec >= LINK_TIMEOUT_SEC:
            self._poll_timer.stop()
            self._show_error("Login timed out. Please try again.")
            self._reactivate_btn.setEnabled(True)
            self._link_code = None
            return
        credentials = self._ms_auth.poll_link(self._link_code)
        if credentials is None:
            return
        self._poll_timer.stop()
        self._link_code = None
        self._complete_reactivation(
            credentials.get("member_id", ""),
            credentials.get("token", ""),
            credentials.get("member_info") or {},
        )

    def _check_callback(self):
        """Poll the auth server for localhost/URL-scheme callback."""
        if not self._auth_server:
            return
        credentials = self._auth_server.get_credentials()
        if credentials is None:
            if self._auth_server._thread and not self._auth_server._thread.is_alive():
                self._poll_timer.stop()
                self._show_error("Login timed out. Please try again.")
                self._reactivate_btn.setEnabled(True)
            return
        self._poll_timer.stop()
        self._status_label.setText("Verifying your account...")
        member_id = credentials.get("member_id", "")
        token = credentials.get("token", "")
        member_info = self._ms_auth.verify_member(member_id, token=token)
        if member_info:
            self._complete_reactivation(member_id, token, member_info)
        else:
            reason = self._ms_auth.last_error_kind
            detail = (self._ms_auth.last_error_message or "").strip()
            if reason == "invalid_request":
                message = (
                    "Login callback missing token.\n"
                    "Please refresh /desktop-login and try again."
                )
            elif reason == "token_invalid":
                message = "Your session may have expired. Please log in again in the browser."
            elif reason in ("unauthorized", "invalid_app_token"):
                message = (
                    "Your app's MEMBERSTACK_APP_SECRET does not match the verification server.\n\n"
                    "Fix: Set the correct secret in .env (project root or Application Support). "
                    "It must match the auth worker's APP_SECRET. See .env.example."
                )
            elif reason == "access_denied":
                message = (
                    "Your account does not have an active EchoZero plan.\n"
                    "Subscribe at speedoflight.dev to continue."
                )
            elif reason == "member_not_found":
                message = "Account not found. Please sign up or log in with the correct account."
            elif reason == "network_error":
                message = "Could not reach the verification server. Check your internet connection."
            else:
                message = (
                    "Account verification failed.\n"
                    "Please ensure you have an active EchoZero membership."
                )
            if detail and reason not in ("invalid_request", "token_invalid"):
                message = message.rstrip() + "\n\n(" + detail + ")"
            self._show_error(message)
            self._reactivate_btn.setEnabled(True)

    def _complete_reactivation(self, member_id: str, token: str, member_info: dict):
        """Store credentials, grant lease, and close on success."""
        self._token_storage.store_credentials(
            member_id=member_id,
            token=token,
            member_data=member_info,
        )
        email = member_info.get("email", "")
        self._lease_manager.grant_lease(
            member_id=member_id,
            email=email,
            billing_period_end_at=member_info.get("billing_period_end", ""),
        )
        self._status_label.setStyleSheet(
            f"color: {Colors.ACCENT_GREEN.name()}; font-size: 13px;"
        )
        self._status_label.setText(f"Reactivated! Welcome back, {email}.")
        Log.info(f"LicenseExpiredDialog: Reactivation successful for {email}")
        QTimer.singleShot(800, self.accept)

    def _show_error(self, message: str):
        """Display an error message in the status label."""
        self._status_label.setStyleSheet(
            f"color: {Colors.ACCENT_RED.name()}; font-size: 13px;"
        )
        self._status_label.setText(message)

    def closeEvent(self, event):
        """Clean up resources when dialog is closed."""
        if self._poll_timer:
            self._poll_timer.stop()
        unregister_pending_auth_server()
        if self._auth_server:
            self._auth_server.stop()
        super().closeEvent(event)

"""
Login Dialog

Modal dialog that gates access to EchoZero until the user authenticates
via Memberstack. Uses a browser-based login flow:

1. User clicks "Login" button
2. System browser opens the Webflow /desktop-login page
3. User logs in via Memberstack in the browser
4. Browser securely POSTs credentials to localhost callback
5. Dialog verifies credentials and closes on success
"""
import webbrowser
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QWidget,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from src.infrastructure.auth.memberstack_auth import MemberstackAuth
from src.infrastructure.auth.token_storage import TokenStorage
from src.infrastructure.auth.local_auth_server import LocalAuthServer
from src.utils.message import Log
from ui.qt_gui.design_system import Colors, Spacing, border_radius


class LoginDialog(QDialog):
    """
    Modal login dialog for EchoZero authentication.
    
    Blocks application access until the user successfully logs in
    via Memberstack through the browser-based flow.
    
    Returns QDialog.Accepted on successful login, QDialog.Rejected if
    the user closes the dialog without logging in.
    """
    
    # URL of the Webflow desktop login page
    LOGIN_PAGE_URL = "https://www.speedoflight.dev/desktop-login"
    
    def __init__(
        self,
        ms_auth: MemberstackAuth,
        token_storage: TokenStorage,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._ms_auth = ms_auth
        self._token_storage = token_storage
        self._auth_server: Optional[LocalAuthServer] = None
        self._poll_timer: Optional[QTimer] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the dialog UI."""
        self.setWindowTitle("EchoZero Login")
        self.setFixedSize(420, 340)
        self.setModal(True)
        
        # Remove the close button from the title bar on macOS/Windows
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        
        # Dark theme styling
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(Spacing.XL, Spacing.XL, Spacing.XL, Spacing.XL)
        
        # -- Title --
        title_label = QLabel("EchoZero")
        title_font = QFont()
        title_font.setFamily("SF Pro Display, Segoe UI, -apple-system, system-ui")
        title_font.setPixelSize(28)
        title_font.setWeight(QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
        layout.addWidget(title_label)
        
        # -- Subtitle --
        subtitle_label = QLabel("Sign in to continue")
        subtitle_font = QFont()
        subtitle_font.setFamily("SF Pro Text, Segoe UI, -apple-system, system-ui")
        subtitle_font.setPixelSize(14)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(subtitle_label)
        
        layout.addSpacing(Spacing.LG)
        
        # -- Login Button --
        self._login_btn = QPushButton("Login")
        self._login_btn.setFixedHeight(44)
        self._login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._login_btn.setStyleSheet(f"""
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
        self._login_btn.clicked.connect(self._on_login_clicked)
        layout.addWidget(self._login_btn)
        
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
        
        # -- Bottom: Quit Button --
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        quit_btn = QPushButton("Quit")
        quit_btn.setFixedWidth(80)
        quit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        quit_btn.setStyleSheet(f"""
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
        quit_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(quit_btn)
        
        layout.addLayout(bottom_layout)
    
    def _on_login_clicked(self):
        """Start the browser-based login flow."""
        self._login_btn.setEnabled(False)
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 13px;")
        self._status_label.setText("Opening browser...")
        
        try:
            # Start the local callback server
            self._auth_server = LocalAuthServer()
            port, nonce = self._auth_server.start()
            
            # Open the login page in the system browser
            login_url = f"{self.LOGIN_PAGE_URL}?port={port}&nonce={nonce}"
            webbrowser.open(login_url)
            
            self._status_label.setText(
                "Waiting for login...\n"
                "Complete the login in your browser, then return here."
            )
            
            # Poll for callback completion (non-blocking, keeps Qt event loop alive)
            self._poll_timer = QTimer(self)
            self._poll_timer.timeout.connect(self._check_callback)
            self._poll_timer.start(500)  # Check every 500ms
            
        except Exception as e:
            Log.error(f"LoginDialog: Failed to start login flow: {e}")
            self._show_error(f"Failed to open browser: {e}")
            self._login_btn.setEnabled(True)
    
    def _check_callback(self):
        """Poll the auth server to see if the callback was received."""
        if not self._auth_server:
            return
        
        credentials = self._auth_server.get_credentials()
        if credentials is None:
            # Still waiting -- check if server thread is still alive
            if self._auth_server._thread and not self._auth_server._thread.is_alive():
                # Server died without credentials (timeout or error)
                self._poll_timer.stop()
                self._show_error("Login timed out. Please try again.")
                self._login_btn.setEnabled(True)
            return
        
        # Callback received -- stop polling
        self._poll_timer.stop()
        self._status_label.setText("Verifying your account...")
        
        # Verify the member with Memberstack Admin API
        member_id = credentials.get("member_id", "")
        token = credentials.get("token", "")
        
        member_info = self._ms_auth.verify_member(member_id, token=token)
        
        if member_info:
            # Store credentials securely
            self._token_storage.store_credentials(
                member_id=member_id,
                token=token,
                member_data=member_info,
            )
            
            email = member_info.get("email", "")
            self._status_label.setStyleSheet(
                f"color: {Colors.ACCENT_GREEN.name()}; font-size: 13px;"
            )
            self._status_label.setText(f"Welcome, {email}!")
            Log.info(f"LoginDialog: Login successful for {email}")
            
            # Close dialog with success after a brief pause
            QTimer.singleShot(800, self.accept)
        else:
            reason = self._ms_auth.last_error_kind
            detail = (self._ms_auth.last_error_message or "").strip()
            if reason == "invalid_request":
                message = (
                    "Login callback missing token.\n"
                    "Please republish /desktop-login and hard refresh the browser tab."
                )
            elif reason == "token_invalid":
                message = (
                    "Your session may have expired.\n"
                    "Please log in again in the browser and complete the flow."
                )
            elif reason == "unauthorized":
                message = (
                    "Auth service rejected this app request.\n"
                    "Please check MEMBERSTACK_APP_SECRET configuration."
                )
            elif reason == "access_denied":
                message = (
                    "Your account does not have an active EchoZero plan.\n"
                    "Subscribe at speedoflight.dev to continue."
                )
            elif reason == "member_not_found":
                message = (
                    "Account not found.\n"
                    "Please sign up or log in with the correct account."
                )
            elif reason == "network_error":
                message = (
                    "Could not reach the verification server.\n"
                    "Check your internet connection and try again."
                )
            else:
                message = (
                    "Account verification failed.\n"
                    "Please ensure you have an active EchoZero membership and try again."
                )
            if detail and reason not in ("invalid_request", "token_invalid"):
                message = message.rstrip() + "\n\n(" + detail + ")"
            self._show_error(message)
            self._login_btn.setEnabled(True)
    
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
        if self._auth_server:
            self._auth_server.stop()
        super().closeEvent(event)

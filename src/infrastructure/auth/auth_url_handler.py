"""
Auth URL Scheme Handler

Handles echozero-auth:// callbacks when the browser redirects to our custom URL
scheme instead of POSTing to localhost. Used because Safari blocks fetch() from
HTTPS pages to HTTP localhost (mixed content).

The login flow registers a LocalAuthServer as the pending handler. When the app
receives a QFileOpenEvent with echozero-auth://callback?... the handler parses
the URL and delivers credentials to the registered server.
"""
from typing import Optional
from urllib.parse import parse_qs, urlparse

from src.utils.message import Log


_pending_server = None


def register_pending_auth_server(server) -> None:
    """Register the LocalAuthServer that is waiting for a callback."""
    global _pending_server
    _pending_server = server
    Log.debug("AuthUrlHandler: Registered pending auth server.")


def unregister_pending_auth_server() -> None:
    """Clear the pending auth server (call when login flow ends)."""
    global _pending_server
    _pending_server = None
    Log.debug("AuthUrlHandler: Unregistered pending auth server.")


def handle_auth_url(url_str: str) -> bool:
    """
    Parse an echozero-auth://callback URL and deliver credentials to the
    pending auth server.

    Returns:
        True if the URL was recognized and handled.
    """
    global _pending_server
    if not _pending_server:
        return False

    try:
        parsed = urlparse(url_str)
        if parsed.scheme != "echozero-auth":
            return False
        # echozero-auth://callback?...) -> netloc=callback; echozero-auth:///callback?...) -> path=/callback
        if parsed.netloc != "callback" and parsed.path not in ("/callback", "callback"):
            return False

        params = parse_qs(parsed.query)
        token = (params.get("token") or [""])[0]
        member_id = (params.get("member_id") or [""])[0]
        nonce = (params.get("nonce") or [""])[0]

        if _pending_server.set_credentials_from_url_scheme(token, member_id, nonce):
            unregister_pending_auth_server()
            return True
    except Exception as e:
        Log.warning(f"AuthUrlHandler: Failed to parse auth URL: {e}")

    return False

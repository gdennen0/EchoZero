"""
Local Authentication Callback Server

Temporary HTTP server on localhost that receives the OAuth-style callback
from the Memberstack login page in the browser.

Flow:
1. Desktop app starts this server on a random available port
2. Browser POSTs JSON to http://127.0.0.1:PORT/callback
   { token, member_id, nonce }
3. Server validates the nonce, extracts credentials, returns a success response
4. Server shuts down immediately after receiving the callback

Security:
- Random port prevents hardcoded targeting
- One-time nonce prevents CSRF / replay attacks
- Server accepts exactly one request then terminates
"""
import json
import secrets
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict
from urllib.parse import urlparse

from src.utils.message import Log


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the auth callback."""
    
    def do_GET(self):
        """Reject legacy GET callbacks. Only POST /callback is accepted."""
        parsed = urlparse(self.path)
        if parsed.path != "/callback":
            self._send_error("Invalid endpoint.")
            return

        Log.warning("LocalAuthServer: Rejected legacy GET callback.")
        self._send_error(
            "Legacy callback rejected. Please use the latest /desktop-login script."
        )

    def do_OPTIONS(self):
        """Handle CORS preflight requests for POST /callback."""
        parsed = urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_POST(self):
        """Handle secure callback POST request from the browser."""
        parsed = urlparse(self.path)
        if parsed.path != "/callback":
            self._send_json_error("Invalid endpoint.", status_code=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json_error("Invalid JSON payload.", status_code=400)
            return

        received_nonce = payload.get("nonce")
        token = payload.get("token")
        member_id = payload.get("member_id")

        if not received_nonce or received_nonce != self.server.expected_nonce:
            Log.warning("LocalAuthServer: Invalid nonce received (POST).")
            self._send_json_error("Authentication failed: invalid security token.", status_code=400)
            return

        if not member_id:
            Log.warning("LocalAuthServer: No member_id in POST callback.")
            self._send_json_error("Authentication failed: no member ID received.", status_code=400)
            return

        if not token:
            Log.warning("LocalAuthServer: No token in POST callback.")
            self._send_json_error("Authentication failed: no token received.", status_code=400)
            return

        self.server.received_credentials = {
            "token": token,
            "member_id": member_id,
        }

        Log.info(f"LocalAuthServer: Secure callback received for member {member_id[:12]}...")
        self._send_json_success("Login callback received. You can return to EchoZero.")

        # Schedule server shutdown (can't call shutdown from request handler thread)
        threading.Thread(target=self.server.shutdown, daemon=True).start()
    
    def _send_success(self):
        """Send a success HTML page to the browser."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>EchoZero - Login Successful</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #1c1c20;
            color: #f0f0f5;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }
        .container {
            text-align: center;
            padding: 48px;
        }
        .check {
            font-size: 64px;
            margin-bottom: 24px;
        }
        h1 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        p {
            color: #b4b4b9;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="check">&#10003;</div>
        <h1>Login Successful</h1>
        <p>You can close this tab and return to EchoZero.</p>
    </div>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))
    
    def _send_error(self, message: str):
        """Send an error HTML page to the browser."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>EchoZero - Login Error</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #1c1c20;
            color: #f0f0f5;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }}
        .container {{
            text-align: center;
            padding: 48px;
        }}
        h1 {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #dc5050;
        }}
        p {{
            color: #b4b4b9;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Login Error</h1>
        <p>{message}</p>
        <p>Please close this tab and try again from EchoZero.</p>
    </div>
</body>
</html>"""
        self.send_response(400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _send_json_success(self, message: str):
        """Send a JSON success response for POST callback flows."""
        body = json.dumps({"ok": True, "message": message}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_json_error(self, message: str, status_code: int = 400):
        """Send a JSON error response for POST callback flows."""
        body = json.dumps({"ok": False, "error": message}).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
    
    def log_message(self, format, *args):
        """Suppress default HTTP server logging (we use our own)."""
        pass


class LocalAuthServer:
    """
    Temporary localhost HTTP server for receiving Memberstack auth callbacks.
    
    Starts on a random available port, generates a one-time nonce,
    and waits for a single callback request before shutting down.
    
    Usage:
        server = LocalAuthServer()
        port, nonce = server.start()
        # ... open browser to login page with ?port=PORT&nonce=NONCE ...
        server.wait_for_callback(timeout=120)
        credentials = server.get_credentials()
        # credentials = {"token": "...", "member_id": "mem_..."}
    """
    
    def __init__(self):
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._nonce: str = ""
        self._port: int = 0
    
    def start(self) -> tuple:
        """
        Start the callback server on a random available port.
        
        Returns:
            Tuple of (port: int, nonce: str) to pass to the login page URL.
        """
        # Generate a cryptographically secure one-time nonce
        self._nonce = secrets.token_urlsafe(32)
        
        # Bind to port 0 to get a random available port
        self._server = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
        self._server.expected_nonce = self._nonce
        self._server.received_credentials = None
        
        self._port = self._server.server_address[1]
        
        # Run server in a background thread
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="echozero-auth-callback",
        )
        self._thread.start()
        
        Log.info(f"LocalAuthServer: Listening on port {self._port}")
        return self._port, self._nonce
    
    def wait_for_callback(self, timeout: float = 120.0) -> bool:
        """
        Block until the callback is received or timeout expires.
        
        Args:
            timeout: Maximum seconds to wait for the callback.
            
        Returns:
            True if callback was received, False if timed out.
        """
        if not self._thread:
            return False
        
        self._thread.join(timeout=timeout)
        
        if self._thread.is_alive():
            # Timed out -- force shutdown
            Log.warning("LocalAuthServer: Timed out waiting for callback.")
            self.stop()
            return False
        
        return self._server.received_credentials is not None
    
    def get_credentials(self) -> Optional[Dict[str, str]]:
        """
        Get the credentials received from the callback.
        
        Returns:
            Dict with "token" and "member_id", or None if no callback received.
        """
        if self._server and self._server.received_credentials:
            return self._server.received_credentials
        return None

    def set_credentials_from_url_scheme(self, token: str, member_id: str, nonce: str) -> bool:
        """
        Accept credentials delivered via custom URL scheme (echozero-auth://).

        Used when Safari blocks fetch() to localhost from HTTPS pages.
        Validates the nonce and completes the auth flow same as a POST callback.

        Returns:
            True if nonce matched and credentials were accepted, False otherwise.
        """
        if not self._server:
            return False
        if not nonce or nonce != self._nonce:
            Log.warning("LocalAuthServer: Invalid nonce in URL scheme callback.")
            return False
        if not member_id:
            Log.warning("LocalAuthServer: No member_id in URL scheme callback.")
            return False
        if not token:
            Log.warning("LocalAuthServer: No token in URL scheme callback.")
            return False
        self._server.received_credentials = {"token": token, "member_id": member_id}
        Log.info(f"LocalAuthServer: URL scheme callback received for member {member_id[:12]}...")
        threading.Thread(target=self._server.shutdown, daemon=True).start()
        return True
    
    @property
    def port(self) -> int:
        """The port the server is listening on."""
        return self._port
    
    @property
    def nonce(self) -> str:
        """The one-time nonce for CSRF protection."""
        return self._nonce
    
    def stop(self):
        """Force-stop the server if still running."""
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        Log.info("LocalAuthServer: Stopped.")

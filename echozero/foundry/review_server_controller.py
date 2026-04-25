"""ReviewServerController keeps one reusable background review server alive per root.
Exists because desktop surfaces should open click-through review without blocking the UI thread.
Connects Foundry and EZ launcher actions to the shared phone-first review web app.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .review_server import create_review_http_server


DEFAULT_REVIEW_SERVER_HOST = "0.0.0.0"
DEFAULT_REVIEW_SERVER_PORT = 8421


@dataclass(slots=True, frozen=True)
class ReviewServerLaunch:
    """Resolved browser and phone-facing URLs for one review session."""

    url: str
    desktop_url: str
    phone_url: str | None
    bind_host: str
    port: int


class ReviewServerController:
    """Serve Foundry review sessions in one reusable background thread."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_REVIEW_SERVER_HOST,
        port: int = DEFAULT_REVIEW_SERVER_PORT,
    ) -> None:
        self._host = host
        self._port = port
        self._is_enabled = False
        self._root: Path | None = None
        self._server = None
        self._thread: threading.Thread | None = None

    @property
    def is_enabled(self) -> bool:
        """Return whether the phone review server may be started on demand."""

        return self._is_enabled

    def set_enabled(self, enabled: bool) -> None:
        """Toggle the phone review server control path for this controller."""

        self._is_enabled = bool(enabled)
        if not self._is_enabled:
            self.stop()

    def enable(self) -> None:
        """Allow the controller to start phone review sessions."""

        self.set_enabled(True)

    def disable(self) -> None:
        """Disable the phone review control path and stop any active server."""

        self.set_enabled(False)

    def build_session_url(self, root: str | Path, session_id: str) -> str:
        """Return one browser URL for the requested root/session pair."""

        return self.build_session_launch(root, session_id).url

    def build_session_launch(self, root: str | Path, session_id: str) -> ReviewServerLaunch:
        """Return browser and phone-facing URLs for the requested root/session pair."""

        if not self._is_enabled:
            raise ValueError("Phone review service is disabled. Enable it before opening review.")
        normalized_root = Path(root).expanduser().resolve()
        normalized_session_id = str(session_id).strip()
        if not normalized_session_id:
            raise ValueError("session_id is required")

        self._ensure_server(normalized_root, normalized_session_id)
        assert self._server is not None
        bind_host, port = self._server.server_address[:2]
        bind_host = str(bind_host)
        port = int(port)
        encoded_session_id = quote(normalized_session_id, safe="")
        desktop_url = _build_review_url(
            host=_desktop_review_host(bind_host),
            port=port,
            session_id=encoded_session_id,
        )
        phone_host = _phone_review_host(bind_host)
        phone_url = None
        if phone_host is not None:
            phone_url = _build_review_url(
                host=phone_host,
                port=port,
                session_id=encoded_session_id,
            )
        return ReviewServerLaunch(
            url=desktop_url,
            desktop_url=desktop_url,
            phone_url=phone_url,
            bind_host=bind_host,
            port=port,
        )

    def stop(self) -> None:
        """Stop the background review server when it is running."""

        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._root = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _ensure_server(self, root: Path, session_id: str) -> None:
        if self._server is not None and self._root == root:
            self._server.default_session_id = session_id
            return

        self.stop()
        server = create_review_http_server(root, session_id, host=self._host, port=self._port)
        thread = threading.Thread(
            target=server.serve_forever,
            name="echozero-review-server",
            daemon=True,
        )
        thread.start()
        self._root = root
        self._server = server
        self._thread = thread


__all__ = [
    "DEFAULT_REVIEW_SERVER_HOST",
    "DEFAULT_REVIEW_SERVER_PORT",
    "ReviewServerController",
    "ReviewServerLaunch",
]


def _build_review_url(*, host: str, port: int, session_id: str) -> str:
    return f"http://{host}:{port}/?sessionId={session_id}"


def _desktop_review_host(bind_host: str) -> str:
    normalized = bind_host.strip().lower()
    if normalized in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    return bind_host


def _phone_review_host(bind_host: str) -> str | None:
    normalized = bind_host.strip().lower()
    if normalized in {"", "0.0.0.0", "::"}:
        return _detect_lan_ipv4_address()
    if normalized.startswith("127.") or normalized == "localhost":
        return None
    return bind_host


def _detect_lan_ipv4_address() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as handle:
            handle.connect(("8.8.8.8", 80))
            candidate = str(handle.getsockname()[0]).strip()
            if _is_phone_reachable_ipv4(candidate):
                return candidate
    except OSError:
        pass
    try:
        _, _, candidates = socket.gethostbyname_ex(socket.gethostname())
    except OSError:
        return None
    for candidate in candidates:
        if _is_phone_reachable_ipv4(candidate):
            return candidate
    return None


def _is_phone_reachable_ipv4(candidate: str) -> bool:
    normalized = candidate.strip()
    return bool(normalized) and not normalized.startswith("127.")

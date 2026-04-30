"""ReviewServerController keeps one reusable background review server alive per runtime root.
Exists because desktop surfaces should open click-through review without blocking the UI thread.
Connects Foundry and EZ launcher actions to the shared phone-first review web app.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from pathlib import Path

from .domain.review import ReviewPolarity
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
        self._runtime_root: Path | None = None
        self._runtime_application_session: dict[str, object] | None = None
        self._last_root: Path | None = None
        self._last_session_id: str | None = None
        self._last_live_scope_key: tuple[str | None, str | None, str | None] | None = None
        self._state_revision = 0
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
            return
        if self._runtime_root is None:
            return
        self._ensure_server(
            self._runtime_root,
            self._last_session_id if self._last_root == self._runtime_root else None,
        )
        assert self._server is not None
        self._server.current_application_session = _copy_application_session(
            self._runtime_application_session
        )
        self._ensure_live_runtime_session()

    def enable(self) -> None:
        """Allow the controller to start phone review sessions."""

        self.set_enabled(True)

    def disable(self) -> None:
        """Disable the phone review control path and stop any active server."""

        self.set_enabled(False)

    def build_session_url(self, root: str | Path, session_id: str) -> str:
        """Return one stable browser URL for the requested root/session pair."""

        return self.build_session_launch(root, session_id).url

    @property
    def last_session_id(self) -> str | None:
        """Return the most recently launched review session id, if any."""

        return self._last_session_id

    def bind_root(
        self,
        root: str | Path,
        *,
        default_session_id: str | None = None,
    ) -> ReviewServerLaunch | None:
        """Bind the reusable server to one project root, even without an active session."""

        normalized_root = Path(root).expanduser().resolve()
        self._runtime_root = normalized_root
        normalized_session_id = (
            str(default_session_id).strip() if default_session_id is not None else ""
        )
        if not normalized_session_id and self._last_root == normalized_root:
            normalized_session_id = self._last_session_id or ""

        self._last_root = normalized_root
        self._last_session_id = normalized_session_id or None
        if normalized_session_id and default_session_id is not None and str(default_session_id).strip():
            self._last_live_scope_key = _live_scope_key(self._runtime_application_session)
        if not self._is_enabled:
            return None

        self._ensure_server(normalized_root, self._last_session_id)
        assert self._server is not None
        self._server.current_application_session = _copy_application_session(
            self._runtime_application_session
        )
        self._ensure_live_runtime_session()
        self._server.state_revision = self._state_revision
        return self._current_launch()

    def set_runtime_context(
        self,
        root: str | Path,
        *,
        application_session: dict[str, object] | None = None,
        clear_active_session: bool = False,
    ) -> ReviewServerLaunch | None:
        """Retarget the enabled review server to the current EZ runtime root."""

        normalized_root = Path(root).expanduser().resolve()
        self._runtime_root = normalized_root
        self._runtime_application_session = _copy_application_session(application_session)
        if clear_active_session:
            self._last_root = normalized_root
            self._last_session_id = None
            self._last_live_scope_key = None
        if not self._is_enabled:
            return None
        if not clear_active_session and self._last_root == normalized_root:
            return self.bind_root(normalized_root)
        return self.bind_root(normalized_root)

    def build_session_launch(self, root: str | Path, session_id: str) -> ReviewServerLaunch:
        """Return browser and phone-facing URLs for the requested root/session pair."""

        if not self._is_enabled:
            raise ValueError("Phone review service is disabled. Enable it before opening review.")
        normalized_root = Path(root).expanduser().resolve()
        normalized_session_id = str(session_id).strip()
        if not normalized_session_id:
            raise ValueError("session_id is required")

        launch = self.bind_root(
            normalized_root,
            default_session_id=normalized_session_id,
        )
        assert launch is not None
        return launch

    def reload_status(
        self,
        root: str | Path | None = None,
        session_id: str | None = None,
    ) -> ReviewServerLaunch:
        """Bump the review-state revision and return the active review launch."""

        normalized_root = Path(root).expanduser().resolve() if root is not None else self._last_root
        normalized_session_id = (
            str(session_id).strip()
            if session_id is not None
            else self._last_session_id
        )
        if normalized_root is None or normalized_session_id is None or not normalized_session_id:
            raise ValueError("Open a phone review session before reloading its status.")
        self._state_revision += 1
        if self._server is not None:
            self._server.state_revision = self._state_revision
        return self.build_session_launch(normalized_root, normalized_session_id)

    def stop(self) -> None:
        """Stop the background review server when it is running."""

        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._root = None
        self._last_live_scope_key = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _ensure_server(self, root: Path, session_id: str | None) -> None:
        if self._server is not None and self._root == root:
            self._server.default_session_id = str(session_id or "").strip()
            self._server.current_application_session = _copy_application_session(
                self._runtime_application_session
            )
            return

        self.stop()
        server = create_review_http_server(
            root,
            session_id,
            host=self._host,
            port=self._port,
            application_session=self._runtime_application_session,
        )
        thread = threading.Thread(
            target=server.serve_forever,
            name="echozero-review-server",
            daemon=True,
        )
        thread.start()
        self._root = root
        self._server = server
        self._thread = thread

    def _ensure_live_runtime_session(self) -> None:
        if self._server is None or self._runtime_root is None:
            return
        scope_key = _live_scope_key(self._runtime_application_session)
        if (
            self._last_live_scope_key == scope_key
            and self._last_root == self._runtime_root
            and isinstance(self._last_session_id, str)
            and self._last_session_id.strip()
        ):
            self._server.default_session_id = self._last_session_id
            return
        try:
            session = self._server.service.create_project_session(
                self._runtime_root,
                song_id=scope_key[1],
                song_version_id=scope_key[2],
                polarity=ReviewPolarity.POSITIVE,
                review_mode="all_events",
                item_limit=None,
                application_session=self._runtime_application_session,
            )
        except ValueError:
            return
        self._last_root = self._runtime_root
        self._last_session_id = session.id
        self._last_live_scope_key = scope_key
        self._server.default_session_id = session.id
        self._state_revision += 1
        self._server.state_revision = self._state_revision

    def _current_launch(self) -> ReviewServerLaunch:
        assert self._server is not None
        bind_host, port = self._server.server_address[:2]
        bind_host = str(bind_host)
        port = int(port)
        desktop_url = _build_review_url(
            host=_desktop_review_host(bind_host),
            port=port,
        )
        phone_host = _phone_review_host(bind_host)
        phone_url = None
        if phone_host is not None:
            phone_url = _build_review_url(
                host=phone_host,
                port=port,
            )
        return ReviewServerLaunch(
            url=desktop_url,
            desktop_url=desktop_url,
            phone_url=phone_url,
            bind_host=bind_host,
            port=port,
        )


__all__ = [
    "DEFAULT_REVIEW_SERVER_HOST",
    "DEFAULT_REVIEW_SERVER_PORT",
    "ReviewServerController",
    "ReviewServerLaunch",
]


def _build_review_url(*, host: str, port: int) -> str:
    return f"http://{host}:{port}/"


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


def _copy_application_session(
    application_session: dict[str, object] | None,
) -> dict[str, object] | None:
    if application_session is None:
        return None
    return dict(application_session)


def _live_scope_key(
    application_session: dict[str, object] | None,
) -> tuple[str | None, str | None, str | None]:
    if application_session is None:
        return (None, None, None)
    project_ref = _normalized_optional_text(application_session.get("projectRef"))
    song_id = _normalized_optional_text(application_session.get("activeSongId"))
    song_version_id = _normalized_optional_text(application_session.get("activeSongVersionId"))
    return (project_ref, song_id, song_version_id)


def _normalized_optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None

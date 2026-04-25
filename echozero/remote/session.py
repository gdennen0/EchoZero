"""
RemoteSessionStore: Short-lived token issuance for EchoZero remote control.
Exists because phone access should stay session-scoped even when the wrapper is only shared privately.
Connects wrapper requests to expiring capability-limited sessions for one browser/device.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Callable

from echozero.errors import InfrastructureError

from .capabilities import RemoteCapabilities


class RemoteSessionError(InfrastructureError):
    """Base error for remote session validation and lifecycle failures."""


class SessionNotFoundError(RemoteSessionError):
    """Raised when a caller references an unknown remote session token."""


class SessionExpiredError(RemoteSessionError):
    """Raised when a caller references a session token that has expired."""


@dataclass(frozen=True, slots=True)
class RemoteSession:
    """One expiring remote-control session bound to a token and capability set."""

    token: str
    issued_at: float
    expires_at: float
    capabilities: RemoteCapabilities

    def is_expired(self, now: float) -> bool:
        """Return whether this session is expired at one point in time."""
        return now >= self.expires_at

    def as_payload(self) -> dict[str, object]:
        """Return a JSON-safe session payload for API responses."""
        return {
            "token": self.token,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "capabilities": self.capabilities.as_payload(),
        }


class RemoteSessionStore:
    """Issue, validate, and revoke expiring remote-control sessions."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 900.0,
        time_source: Callable[[], float] = time.time,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self._ttl_seconds = float(ttl_seconds)
        self._time_source = time_source
        self._token_factory = token_factory or self._build_token
        self._sessions: dict[str, RemoteSession] = {}

    def create_session(
        self,
        *,
        capabilities: RemoteCapabilities | None = None,
    ) -> RemoteSession:
        """Issue one new session token with expiry and a capability set."""
        now = self._time_source()
        session = RemoteSession(
            token=self._token_factory(),
            issued_at=now,
            expires_at=now + self._ttl_seconds,
            capabilities=capabilities or RemoteCapabilities(),
        )
        self._sessions[session.token] = session
        self.purge_expired()
        return session

    def require_session(self, token: str) -> RemoteSession:
        """Return one live session or raise if the token is missing or expired."""
        session = self._sessions.get(token)
        if session is None:
            raise SessionNotFoundError("Remote session token was not found.")
        if session.is_expired(self._time_source()):
            self._sessions.pop(token, None)
            raise SessionExpiredError("Remote session token has expired.")
        return session

    def revoke_session(self, token: str) -> None:
        """Remove one session token if it is present."""
        self._sessions.pop(token, None)

    def purge_expired(self) -> None:
        """Delete expired sessions from the in-memory session store."""
        now = self._time_source()
        expired_tokens = [
            token for token, session in self._sessions.items() if session.is_expired(now)
        ]
        for token in expired_tokens:
            self._sessions.pop(token, None)

    @staticmethod
    def _build_token() -> str:
        return secrets.token_urlsafe(24)

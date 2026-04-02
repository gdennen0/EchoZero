"""In-memory session repository implementation for the new application architecture."""

from echozero.application.session.models import Session
from echozero.application.session.repository import SessionRepository
from echozero.application.shared.ids import SessionId


class InMemorySessionRepository(SessionRepository):
    def __init__(self) -> None:
        self._sessions: dict[SessionId, Session] = {}

    def get(self, session_id: SessionId) -> Session:
        return self._sessions[session_id]

    def save(self, session: Session) -> None:
        self._sessions[session.id] = session

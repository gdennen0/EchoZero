"""Session persistence boundary for the new EchoZero application layer."""

from abc import ABC, abstractmethod

from echozero.application.session.models import Session
from echozero.application.shared.ids import SessionId


class SessionRepository(ABC):
    @abstractmethod
    def get(self, session_id: SessionId) -> Session:
        raise NotImplementedError

    @abstractmethod
    def save(self, session: Session) -> None:
        raise NotImplementedError

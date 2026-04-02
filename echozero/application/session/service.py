"""Session service contract for current working EchoZero context."""

from abc import ABC, abstractmethod

from echozero.application.shared.ids import SongId, SongVersionId, TimelineId
from echozero.application.session.models import Session


class SessionService(ABC):
    """Owns active project/song/version/timeline runtime context."""

    @abstractmethod
    def get_session(self) -> Session:
        """Return the current session state snapshot."""
        raise NotImplementedError

    @abstractmethod
    def set_active_song(self, song_id: SongId | None) -> Session:
        """Set the active song in the current session."""
        raise NotImplementedError

    @abstractmethod
    def set_active_song_version(self, song_version_id: SongVersionId | None) -> Session:
        """Set the active song version in the current session."""
        raise NotImplementedError

    @abstractmethod
    def set_active_timeline(self, timeline_id: TimelineId | None) -> Session:
        """Set the active timeline in the current session."""
        raise NotImplementedError

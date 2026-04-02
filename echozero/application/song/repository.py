"""Song persistence boundary for the new EchoZero application layer."""

from abc import ABC, abstractmethod

from echozero.application.song.models import Song, SongVersion
from echozero.application.shared.ids import SongId, SongVersionId


class SongRepository(ABC):
    @abstractmethod
    def get_song(self, song_id: SongId) -> Song:
        raise NotImplementedError

    @abstractmethod
    def get_song_version(self, song_version_id: SongVersionId) -> SongVersion:
        raise NotImplementedError

    @abstractmethod
    def save_song(self, song: Song) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_song_version(self, song_version: SongVersion) -> None:
        raise NotImplementedError

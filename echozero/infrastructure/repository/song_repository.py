"""In-memory song repository implementation for the new application architecture."""

from echozero.application.shared.ids import SongId, SongVersionId
from echozero.application.song.models import Song, SongVersion
from echozero.application.song.repository import SongRepository


class InMemorySongRepository(SongRepository):
    def __init__(self) -> None:
        self._songs: dict[SongId, Song] = {}
        self._versions: dict[SongVersionId, SongVersion] = {}

    def get_song(self, song_id: SongId) -> Song:
        return self._songs[song_id]

    def get_song_version(self, song_version_id: SongVersionId) -> SongVersion:
        return self._versions[song_version_id]

    def save_song(self, song: Song) -> None:
        self._songs[song.id] = song

    def save_song_version(self, song_version: SongVersion) -> None:
        self._versions[song_version.id] = song_version

"""
Domain layer for setlists feature.

Contains:
- Setlist entity
- SetlistSong entity
- Repository interfaces
- Processing result types
"""
from src.features.setlists.domain.setlist import Setlist
from src.features.setlists.domain.setlist_song import SetlistSong
from src.features.setlists.domain.setlist_repository import SetlistRepository
from src.features.setlists.domain.setlist_song_repository import SetlistSongRepository
from src.features.setlists.domain.processing_result import SongProcessingResult, SetlistProcessingResult

__all__ = [
    'Setlist',
    'SetlistSong',
    'SetlistRepository',
    'SetlistSongRepository',
    'SongProcessingResult',
    'SetlistProcessingResult',
]

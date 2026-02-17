"""
Setlists feature module.

Usage:
    from src.features.setlists.domain import Setlist, SetlistSong
    from src.features.setlists.application import SetlistService
    from src.features.setlists.infrastructure import SQLiteSetlistRepository
"""
# Only export domain by default - application and infrastructure via submodules
from src.features.setlists.domain import (
    Setlist,
    SetlistSong,
    SetlistRepository,
    SetlistSongRepository,
)

__all__ = [
    'Setlist',
    'SetlistSong',
    'SetlistRepository',
    'SetlistSongRepository',
]

"""
Setlist Song Repository Interface

Defines the interface for setlist song persistence.
"""
from abc import ABC, abstractmethod
from typing import Optional, List

from src.features.setlists.domain import SetlistSong


class SetlistSongRepository(ABC):
    """
    Repository interface for setlist song persistence.
    
    Handles CRUD operations for SetlistSong entities.
    """
    
    @abstractmethod
    def create(self, song: SetlistSong) -> SetlistSong:
        """
        Create a new setlist song.
        
        Args:
            song: SetlistSong entity to create
            
        Returns:
            Created song (with generated ID if needed)
        """
        pass
    
    @abstractmethod
    def get(self, song_id: str) -> Optional[SetlistSong]:
        """
        Get song by ID.
        
        Args:
            song_id: Song identifier
            
        Returns:
            SetlistSong entity or None if not found
        """
        pass
    
    @abstractmethod
    def list_by_setlist(self, setlist_id: str) -> List[SetlistSong]:
        """
        List all songs in a setlist.
        
        Args:
            setlist_id: Setlist identifier
            
        Returns:
            List of songs, ordered by order_index
        """
        pass
    
    @abstractmethod
    def update(self, song: SetlistSong) -> None:
        """
        Update song.
        
        Args:
            song: SetlistSong entity to update
            
        Raises:
            ValueError: If song not found
        """
        pass
    
    @abstractmethod
    def delete(self, song_id: str) -> None:
        """
        Delete song.
        
        Args:
            song_id: Song identifier to delete
        """
        pass
    
    @abstractmethod
    def delete_by_setlist(self, setlist_id: str) -> int:
        """
        Delete all songs in a setlist.
        
        Args:
            setlist_id: Setlist identifier
            
        Returns:
            Number of songs deleted
        """
        pass


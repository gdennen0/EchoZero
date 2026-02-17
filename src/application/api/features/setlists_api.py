"""
Setlists API - Feature-specific facade for setlist operations.

Provides a focused API for managing setlists and songs.
"""
from typing import TYPE_CHECKING, Optional, Dict, Any

from src.application.api.result_types import CommandResult

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class SetlistsAPI:
    """
    Setlists feature API.
    
    Provides setlist management operations:
    - Create/delete setlists
    - Add/remove songs
    - Manage song order
    
    Usage:
        setlists = SetlistsAPI(facade)
        result = setlists.create_setlist("My Show")
    """
    
    def __init__(self, facade: "ApplicationFacade"):
        """Initialize with reference to main facade."""
        self._facade = facade
    
    def create_setlist(self, name: str, description: str = "") -> CommandResult:
        """Create a new setlist."""
        return self._facade.create_setlist(name, description)
    
    def delete_setlist(self, setlist_id: str) -> CommandResult:
        """Delete a setlist."""
        return self._facade.delete_setlist(setlist_id)
    
    def list_setlists(self) -> CommandResult:
        """List all setlists in current project."""
        return self._facade.list_setlists()
    
    def add_song(
        self,
        setlist_id: str,
        name: str,
        audio_path: Optional[str] = None,
        settings_overrides: Optional[Dict[str, Any]] = None
    ) -> CommandResult:
        """Add a song to a setlist."""
        return self._facade.add_setlist_song(setlist_id, name, audio_path, settings_overrides)
    
    def remove_song(self, song_id: str) -> CommandResult:
        """Remove a song from its setlist."""
        return self._facade.remove_setlist_song(song_id)
    
    def reorder_songs(self, setlist_id: str, song_ids: list) -> CommandResult:
        """Reorder songs in a setlist."""
        return self._facade.reorder_setlist_songs(setlist_id, song_ids)

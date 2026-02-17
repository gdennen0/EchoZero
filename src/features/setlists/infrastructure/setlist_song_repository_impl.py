"""
SQLite implementation of SetlistSongRepository

Handles persistence of SetlistSong entities in SQLite database.
"""
from typing import Optional, List
from datetime import datetime

from src.features.setlists.domain import SetlistSong
from src.features.setlists.domain import SetlistSongRepository
from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


class SQLiteSetlistSongRepository(SetlistSongRepository):
    """SQLite implementation of SetlistSongRepository"""
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    def create(self, song: SetlistSong) -> SetlistSong:
        """
        Create a new setlist song.
        
        Args:
            song: SetlistSong entity to create
            
        Returns:
            Created song (with generated ID if needed)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Insert song
            cursor.execute("""
                INSERT INTO setlist_songs (
                    id, setlist_id, audio_path, order_index, status, processed_at,
                    action_overrides, error_message, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                song.id,
                song.setlist_id,
                song.audio_path,
                song.order_index,
                song.status,
                song.processed_at.isoformat() if song.processed_at else None,
                Database.json_encode(song.action_overrides),
                song.error_message,
                Database.json_encode(song.metadata)
            ))
            
            Log.info(f"Created setlist song: {song.audio_path} (id: {song.id})")
            return song
    
    def get(self, song_id: str) -> Optional[SetlistSong]:
        """
        Get song by ID.
        
        Args:
            song_id: Song identifier
            
        Returns:
            SetlistSong entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, setlist_id, audio_path, order_index, status, processed_at,
                       action_overrides, error_message, metadata
                FROM setlist_songs
                WHERE id = ?
            """, (song_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return self._row_to_song(row)
    
    def list_by_setlist(self, setlist_id: str) -> List[SetlistSong]:
        """
        List all songs in a setlist.
        
        Args:
            setlist_id: Setlist identifier
            
        Returns:
            List of songs, ordered by order_index
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, setlist_id, audio_path, order_index, status, processed_at,
                       action_overrides, error_message, metadata
                FROM setlist_songs
                WHERE setlist_id = ?
                ORDER BY order_index ASC
            """, (setlist_id,))
            
            return [self._row_to_song(row) for row in cursor.fetchall()]
    
    def update(self, song: SetlistSong) -> None:
        """
        Update song.
        
        Args:
            song: SetlistSong entity to update
            
        Raises:
            ValueError: If song not found
        """
        # Verify song exists
        existing = self.get(song.id)
        if existing is None:
            raise ValueError(f"Song with id '{song.id}' not found")
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE setlist_songs
                SET audio_path = ?, order_index = ?, status = ?, processed_at = ?,
                    action_overrides = ?, error_message = ?, metadata = ?
                WHERE id = ?
            """, (
                song.audio_path,
                song.order_index,
                song.status,
                song.processed_at.isoformat() if song.processed_at else None,
                Database.json_encode(song.action_overrides),
                song.error_message,
                Database.json_encode(song.metadata),
                song.id
            ))
            
            Log.debug(f"Updated setlist song: {song.audio_path} (id: {song.id})")
    
    def delete(self, song_id: str) -> None:
        """
        Delete song.
        
        Args:
            song_id: Song identifier to delete
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM setlist_songs WHERE id = ?", (song_id,))
            
            Log.debug(f"Deleted setlist song (id: {song_id})")
    
    def delete_by_setlist(self, setlist_id: str) -> int:
        """
        Delete all songs in a setlist.
        
        Args:
            setlist_id: Setlist identifier
            
        Returns:
            Number of songs deleted
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM setlist_songs WHERE setlist_id = ?", (setlist_id,))
            deleted_count = cursor.rowcount
            
            Log.info(f"Deleted {deleted_count} song(s) from setlist {setlist_id}")
            return deleted_count
    
    def _row_to_song(self, row) -> SetlistSong:
        """
        Convert database row to SetlistSong entity.
        
        Args:
            row: Database row (sqlite3.Row)
            
        Returns:
            SetlistSong entity
        """
        # sqlite3.Row uses dictionary-style access
        processed_at = None
        if "processed_at" in row.keys() and row["processed_at"]:
            processed_at = datetime.fromisoformat(row["processed_at"])
        
        # Handle migration from old schema (block_settings_overrides -> action_overrides)
        action_overrides = {}
        if "block_settings_overrides" in row.keys() and "action_overrides" not in row.keys():
            if row["block_settings_overrides"]:
                action_overrides = Database.json_decode(row["block_settings_overrides"]) or {}
        elif "action_overrides" in row.keys() and row["action_overrides"]:
            action_overrides = Database.json_decode(row["action_overrides"]) or {}
        
        error_message = None
        if "error_message" in row.keys() and row["error_message"]:
            error_message = row["error_message"]
        
        metadata = {}
        if "metadata" in row.keys() and row["metadata"]:
            metadata = Database.json_decode(row["metadata"])
        
        return SetlistSong(
            id=row["id"],
            setlist_id=row["setlist_id"],
            audio_path=row["audio_path"],
            order_index=row["order_index"],
            status=row["status"],
            processed_at=processed_at,
            action_overrides=action_overrides,
            error_message=error_message,
            metadata=metadata
        )


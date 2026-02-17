"""
SQLite implementation of SessionStateRepository

Handles persistence of session-specific state that should be restored
when the application is restarted (but not project-specific).
"""
from typing import Optional, Dict, Any
from datetime import datetime

from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


class SessionStateRepository:
    """
    Repository for session state persistence.
    
    Session state is application-wide data like:
    - Currently open panels
    - Selected block
    - Last opened project
    - Window positions and sizes
    
    This data persists across application restarts to restore user's session,
    but is NOT project-specific (unlike ui_state).
    """
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a session state value.
        
        Args:
            key: State key
            value: Value to store (will be JSON serialized)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Check if state exists
            cursor.execute("""
                SELECT key FROM session_state WHERE key = ?
            """, (key,))
            
            exists = cursor.fetchone() is not None
            
            # Serialize value using Database.json_encode for consistency
            value_json = Database.json_encode(value) or str(value)
            
            if exists:
                # Update existing
                cursor.execute("""
                    UPDATE session_state
                    SET value = ?, updated_at = ?
                    WHERE key = ?
                """, (
                    value_json,
                    datetime.now().isoformat(),
                    key
                ))
                Log.debug(f"Updated session state: {key}")
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO session_state (key, value, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    key,
                    value_json,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                Log.debug(f"Created session state: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a session state value.
        
        Args:
            key: State key
            default: Default value if state not found
            
        Returns:
            State value or default
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM session_state WHERE key = ?
            """, (key,))
            
            row = cursor.fetchone()
            if row is None:
                return default
            
            # Deserialize using Database.json_decode for consistency
            value_str = row[0]
            if value_str is None:
                return default
            decoded = Database.json_decode(value_str)
            return decoded if decoded is not None else value_str
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all session state.
        
        Returns:
            Dictionary of all state key-value pairs
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT key, value FROM session_state
            """)
            
            state = {}
            for row in cursor.fetchall():
                key = row[0]
                value_str = row[1]
                
                # Deserialize using Database.json_decode for consistency
                decoded = Database.json_decode(value_str)
                state[key] = decoded if decoded is not None else value_str
            
            return state
    
    def delete(self, key: str) -> None:
        """
        Delete a session state entry.
        
        Args:
            key: State key to delete
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM session_state WHERE key = ?
            """, (key,))
            
            Log.debug(f"Deleted session state: {key}")
    
    def clear(self) -> None:
        """
        Clear all session state.
        
        Called when user explicitly wants to reset their session.
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM session_state")
            Log.debug("Cleared all session state")


